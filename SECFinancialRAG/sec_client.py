"""
SEC API Client for fetching company financial data
Handles CIK lookup and company facts retrieval
"""

import requests
import json
import time
from typing import Dict, Optional, Any
import logging
from urllib.parse import quote
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class SECClient:
    """Client for interacting with SEC EDGAR API"""
    
    def __init__(self):
        self.base_url = "https://data.sec.gov/api"
        self.user_agent = os.getenv('SEC_USER_AGENT', 'example@email.com')
        
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
        Get CIK (Central Index Key) from ticker symbol
        
        Args:
            ticker: Company ticker symbol (e.g., 'AAPL')
            
        Returns:
            CIK string with leading zeros (e.g., '0000320193') or None if not found
        """
        # Known major companies mapping for demo purposes
        # In production, you'd want to maintain a more comprehensive mapping
        known_ciks = {
            'AAPL': '0000320193',
            'MSFT': '0000789019', 
            'GOOGL': '0001652044',
            'GOOG': '0001652044',
            'AMZN': '0001018724',
            'TSLA': '0001318605',
            'META': '0001326801',
            'FB': '0001326801',
            'NVDA': '0001045810',
            'BRK.A': '0001067983',
            'BRK.B': '0001067983',
            'JNJ': '0000200406',
            'V': '0001403161',
            'PG': '0000080424',
            'JPM': '0000019617',
            'UNH': '0000731766',
            'MA': '0001141391',
            'HD': '0000354950',
            'CVX': '0000093410',
            'LLY': '0000059478',
            'ABBV': '0001551152',
            'PFE': '0000078003',
            'KO': '0000021344',
            'AVGO': '0001730168',
            'PEP': '0000077476',
            'TMO': '0000097745',
            'COST': '0000909832',
            'DIS': '0001744489',
            'ABT': '0000001800',
            'ACN': '0001467373',
            'WMT': '0000104169'
        }
        
        ticker_upper = ticker.upper()
        
        # Check known mapping first
        if ticker_upper in known_ciks:
            logger.info(f"Found CIK for {ticker} in known mappings: {known_ciks[ticker_upper]}")
            return known_ciks[ticker_upper]
        
        # If not in known mappings, try SEC API lookup
        self._rate_limit()
        
        try:
            # Try the submissions endpoint which is more reliable
            url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                cik = data.get('cik')
                if cik:
                    formatted_cik = self._format_cik(cik)
                    logger.info(f"Found CIK for {ticker} via SEC API: {formatted_cik}")
                    return formatted_cik
            
            # If that fails, try a few common CIK patterns
            for potential_cik in [ticker_upper, f"000{ticker_upper}", f"0000{ticker_upper}"]:
                test_url = f"{self.base_url}/xbrl/companyfacts/CIK{potential_cik}.json"
                self._rate_limit()
                
                response = self.session.get(test_url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('entityName', '').upper().startswith(ticker_upper):
                        cik = data.get('cik')
                        if cik:
                            formatted_cik = self._format_cik(cik)
                            logger.info(f"Found CIK for {ticker} via pattern matching: {formatted_cik}")
                            return formatted_cik
            
            logger.warning(f"CIK not found for ticker: {ticker}")
            logger.info(f"You may need to add {ticker} to the known_ciks mapping")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting CIK for {ticker}: {e}")
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
            # Ensure CIK is properly formatted
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
    
    def get_company_financials_by_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get company financial data by ticker (combines CIK lookup and facts retrieval)
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Company facts dictionary or None if error
        """
        logger.info(f"Getting financials for ticker: {ticker}")
        
        # Step 1: Get CIK from ticker
        cik = self.get_cik_from_ticker(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker: {ticker}")
            return None
        
        logger.info(f"Found CIK {cik} for ticker {ticker}")
        
        # Step 2: Get company facts using CIK
        company_facts = self.get_company_facts(cik)
        if not company_facts:
            logger.error(f"Could not fetch company facts for CIK: {cik}")
            return None
        
        # Add ticker and CIK to the response for convenience
        company_facts['ticker'] = ticker.upper()
        company_facts['cik'] = cik
        
        ##Save company facts to a file for debugging
        try:
            with open(f"{ticker}_company_facts.json", 'w') as f:
                json.dump(company_facts, f, indent=2)
            logger.info(f"Saved company facts for {ticker} to {ticker}_company_facts.json")
        except Exception as e:
            logger.error(f"Error saving company facts for {ticker}: {e}")

        return company_facts
    
    def validate_connection(self) -> bool:
        """
        Test SEC API connection
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            self._rate_limit()
            
            # Test with the main API endpoint
            test_url = f"{self.base_url}/xbrl/companyfacts/CIK0000320193.json"  # Apple's CIK for testing
            response = self.session.get(test_url, timeout=10)
            
            if response.status_code == 200:
                logger.info("SEC API connection validated successfully")
                return True
            else:
                logger.warning(f"SEC API returned status code: {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"SEC API connection validation failed: {e}")
            return False
    
    def get_company_info(self, cik: str) -> Optional[Dict[str, Any]]:
        """
        Get basic company information
        
        Args:
            cik: Company CIK
            
        Returns:
            Company info dictionary or None if error
        """
        try:
            company_facts = self.get_company_facts(cik)
            if not company_facts:
                return None
            
            # Extract basic company info
            return {
                'cik': cik,
                'name': company_facts.get('entityName', 'Unknown'),
                'ticker': company_facts.get('ticker', 'Unknown'),
                'sic': company_facts.get('sic', 'Unknown'),
                'sicDescription': company_facts.get('sicDescription', 'Unknown'),
                'ein': company_facts.get('ein', 'Unknown'),
                'description': company_facts.get('description', ''),
                'website': company_facts.get('website', ''),
                'investorWebsite': company_facts.get('investorWebsite', ''),
                'category': company_facts.get('category', 'Unknown'),
                'fiscalYearEnd': company_facts.get('fiscalYearEnd', 'Unknown'),
                'stateOfIncorporation': company_facts.get('stateOfIncorporation', 'Unknown'),
                'stateOfIncorporationDescription': company_facts.get('stateOfIncorporationDescription', 'Unknown')
            }
            
        except Exception as e:
            logger.error(f"Error getting company info for CIK {cik}: {e}")
            return None