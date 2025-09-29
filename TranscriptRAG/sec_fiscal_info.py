"""
Alternative SEC-based fiscal info fetcher for TranscriptRAG

Provides an alternative to Alpha Vantage's OVERVIEW API using SEC data.
Uses SEC company facts API to determine fiscal year end and latest quarter.
"""

import requests
import json
import time
from typing import Dict, Optional, Any, List
from datetime import datetime
import logging
from urllib.parse import quote
import os

logger = logging.getLogger(__name__)


class SECFiscalInfoClient:
    """Client for fetching fiscal information from SEC API"""
    
    def __init__(self, user_agent: str = None):
        self.base_url = "https://data.sec.gov/api"
        
        # Use provided user agent or look for environment variable
        if user_agent:
            self.user_agent = user_agent
        else:
            self.user_agent = os.getenv('SEC_USER_AGENT', 'transcript-rag@example.com')
        
        if '@' not in self.user_agent or '.' not in self.user_agent:
            raise ValueError("SEC_USER_AGENT must be a valid email address")
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'application/json',
            'Host': 'data.sec.gov'
        })
        
        # Rate limiting - SEC allows 10 requests per second
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Ensure we don't exceed SEC rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """
        Fetches the CIK (Central Index Key) for a given stock ticker
        using the SEC's public mapping file.
        
        Args:
            ticker (str): Stock ticker symbol (case-insensitive)
        
        Returns:
            str: 10-digit zero-padded CIK or None if not found
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": self.user_agent}

        try:
            self._rate_limit()
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            ticker = ticker.upper().strip()

            for item in data.values():
                if item["ticker"] == ticker:
                    cik_int = int(item["cik_str"])
                    formatted_cik = str(cik_int).zfill(10)  # zero-pad to 10 digits
                    logger.info(f"Found CIK {formatted_cik} for ticker {ticker}")
                    return formatted_cik

            logger.warning(f"No CIK found for ticker {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching CIK for ticker {ticker}: {e}")
            return None
    
    def _format_cik(self, cik: Any) -> str:
        """Format CIK with leading zeros to 10 digits"""
        if isinstance(cik, int):
            return f"{cik:010d}"
        elif isinstance(cik, str):
            # Remove any existing leading zeros and reformat
            return f"{int(cik):010d}"
        else:
            raise ValueError(f"Invalid CIK format: {cik}")
    
    def get_company_facts(self, cik: str) -> Optional[Dict[str, Any]]:
        """
        Fetch company facts JSON from SEC API
        
        Args:
            cik: CIK with leading zeros (e.g., '0000320193')
            
        Returns:
            Company facts dictionary or None if error
        """
        self._rate_limit()
        
        try:
            # Ensure CIK is properly formatted (zero-padded to 10 digits)
            formatted_cik = self._format_cik(cik)
                
            url = f"{self.base_url}/xbrl/companyfacts/CIK{formatted_cik}.json"
            
            logger.info(f"Fetching company facts from: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            company_facts = response.json()
            
            # Validate response structure
            if 'facts' not in company_facts:
                logger.error(f"Invalid company facts response for CIK {cik}")
                return None
            
            logger.info(f"Successfully fetched company facts for CIK {cik}")
            return company_facts
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Company facts not found for CIK {cik}")
            else:
                logger.error(f"HTTP error fetching company facts for CIK {cik}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching company facts for CIK {cik}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for CIK {cik}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching company facts for CIK {cik}: {e}")
            return None


def get_fiscal_info_from_sec(ticker: str, user_agent: str = None) -> Optional[Dict[str, str]]:
    """
    Simplified SEC-based fiscal info fetcher.
    
    Logic:
    1. Take ticker as input
    2. Get CIK from ticker using SEC company tickers mapping
    3. Fetch company facts using CIK
    4. From Assets data, find the most recent 'end' date entry
    5. Extract 'end', 'fy', and 'fp' fields from that entry
    
    Args:
        ticker: Stock ticker symbol
        user_agent: Email address for SEC API (optional)
        
    Returns:
        Dictionary with end, fy, and fp info, or None if error
        Format: {
            'end': '2025-06-30',    # Most recent period end date
            'fy': 2025,             # Fiscal year
            'fp': 'Q2'              # Fiscal period (Q1, Q2, Q3, Q4, FY)
        }
    """
    
    try:
        # Initialize SEC client
        client = SECFiscalInfoClient(user_agent)
        
        # Step 1: Get CIK from ticker
        cik = client.get_cik_from_ticker(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker: {ticker}")
            return None
        
        logger.info(f"Found CIK {cik} for ticker {ticker}")
        
        # Step 2: Get company facts using CIK
        company_facts = client.get_company_facts(cik)
        if not company_facts:
            logger.error(f"Could not fetch company facts for CIK: {cik}")
            return None
        
        # Step 3: Extract Assets data from us-gaap facts
        us_gaap_facts = company_facts.get('facts', {}).get('us-gaap', {})
        if not us_gaap_facts:
            logger.error(f"No us-gaap facts found for ticker {ticker}")
            return None
        
        # Look for Assets field (total assets)
        assets_data = us_gaap_facts.get('Assets')
        if not assets_data or 'units' not in assets_data:
            logger.error(f"No Assets data found for ticker {ticker}")
            return None
        
        # Get USD units data
        usd_data = assets_data.get('units', {}).get('USD', [])
        if not usd_data:
            logger.error(f"No USD Assets data found for ticker {ticker}")
            return None
        
        logger.info(f"Found {len(usd_data)} Assets entries for {ticker}")
        
        # Step 4: Find entry with most recent 'end' date
        most_recent_entry = None
        most_recent_end_date = None
        
        for entry in usd_data:
            end_date_str = entry.get('end')
            if end_date_str:
                try:
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                    if most_recent_end_date is None or end_date > most_recent_end_date:
                        most_recent_end_date = end_date
                        most_recent_entry = entry
                except ValueError:
                    continue
        
        if not most_recent_entry:
            logger.error(f"No valid end dates found in Assets data for {ticker}")
            return None
        
        # Step 5: Extract end, fy, and fp fields
        end_date = most_recent_entry.get('end')
        fiscal_year = most_recent_entry.get('fy')
        fiscal_period = most_recent_entry.get('fp')
        
        logger.info(f"Most recent Assets entry for {ticker}: end={end_date}, fy={fiscal_year}, fp={fiscal_period}")
        
        # Return the simplified result
        result = {
            'end': end_date,
            'fy': fiscal_year,
            'fp': fiscal_period
        }
        
        logger.info(f"SEC fiscal info for {ticker}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error getting fiscal info for {ticker}: {e}")
        return None


def test_get_fiscal_info():
    """Test function to verify the SEC fiscal info logic"""
    
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    test_tickers = ['TGT', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    print("Testing SEC-based fiscal info fetcher...")
    print("=" * 60)
    
    for ticker in test_tickers:
        print(f"\nTesting {ticker}:")
        print("-" * 30)
        
        try:
            fiscal_info = get_fiscal_info_from_sec(ticker)
            
            if fiscal_info:
                print(f"✓ Success!")
                print(f"  End Date: {fiscal_info['end']}")
                print(f"  Fiscal Year: {fiscal_info['fy']}")
                print(f"  Fiscal Period: {fiscal_info['fp']}")
            else:
                print(f"✗ Failed to get fiscal info for {ticker}")
                
        except Exception as e:
            print(f"✗ Error testing {ticker}: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_get_fiscal_info()