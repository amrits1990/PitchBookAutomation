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
        Fetches the CIK (Central Index Key) for a given stock ticker
        using the SEC's public mapping file.
        
        Args:
            ticker (str): Stock ticker symbol (case-insensitive)
        
        Returns:
            str: 10-digit zero-padded CIK or None if not found
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {"User-Agent": "my-app (your_email@example.com)"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        ticker = ticker.upper().strip()

        for item in data.values():
            if item["ticker"] == ticker:
                cik_int = int(item["cik_str"])
                return str(cik_int).zfill(10)  # zero-pad to 10 digits

        return None
    
    def _lookup_cik_from_company_tickers(self, ticker: str) -> Optional[str]:
        """
        Look up CIK using SEC's company tickers API endpoints
        This method tries multiple SEC endpoints to find the CIK
        """
        # Try multiple possible endpoints for company tickers
        endpoints_to_try = [
            "https://www.sec.gov/files/company_tickers.json",
            "https://data.sec.gov/api/xbrl/companyfacts/company_tickers.json",
            "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt",
        ]
        
        for endpoint in endpoints_to_try:
            try:
                self._rate_limit()
                logger.info(f"Trying endpoint: {endpoint}")
                
                if endpoint.endswith('.json'):
                    cik = self._search_json_endpoint(endpoint, ticker)
                    if cik:
                        return cik
                elif endpoint.endswith('.txt'):
                    cik = self._search_text_endpoint(endpoint, ticker)
                    if cik:
                        return cik
                        
            except Exception as e:
                logger.debug(f"Endpoint {endpoint} failed: {e}")
                continue
        
        # If all endpoints fail, try a manual search approach
        return self._manual_cik_search(ticker)
    
    def _search_json_endpoint(self, url: str, ticker: str) -> Optional[str]:
        """Search JSON endpoint for ticker to CIK mapping"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return None
                
            company_data = response.json()
            
            # Handle different JSON structures
            if isinstance(company_data, dict):
                # Structure: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
                for key, company_info in company_data.items():
                    if isinstance(company_info, dict) and company_info.get('ticker', '').upper() == ticker.upper():
                        cik_str = company_info.get('cik_str')
                        if cik_str:
                            formatted_cik = self._format_cik(cik_str)
                            logger.info(f"Found {ticker} in {url}: CIK {formatted_cik}, Company: {company_info.get('title', 'Unknown')}")
                            return formatted_cik
            elif isinstance(company_data, list):
                # Handle list structure
                for company_info in company_data:
                    if isinstance(company_info, dict) and company_info.get('ticker', '').upper() == ticker.upper():
                        cik_str = company_info.get('cik_str') or company_info.get('cik')
                        if cik_str:
                            formatted_cik = self._format_cik(cik_str)
                            logger.info(f"Found {ticker} in {url}: CIK {formatted_cik}")
                            return formatted_cik
            
            return None
            
        except Exception as e:
            logger.debug(f"Error searching JSON endpoint {url}: {e}")
            return None
    
    def _search_text_endpoint(self, url: str, ticker: str) -> Optional[str]:
        """Search text-based endpoint for ticker to CIK mapping"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return None
                
            text_data = response.text
            lines = text_data.split('\n')
            
            for line in lines:
                # Common formats: "TICKER|CIK|NAME" or "TICKER:CIK:NAME" or tab-separated
                parts = line.replace('\t', '|').replace(':', '|').split('|')
                if len(parts) >= 2:
                    line_ticker = parts[0].strip().upper()
                    if line_ticker == ticker.upper():
                        potential_cik = parts[1].strip()
                        if potential_cik.isdigit():
                            formatted_cik = self._format_cik(potential_cik)
                            logger.info(f"Found {ticker} in {url}: CIK {formatted_cik}")
                            return formatted_cik
            
            return None
            
        except Exception as e:
            logger.debug(f"Error searching text endpoint {url}: {e}")
            return None
    
    def _manual_cik_search(self, ticker: str) -> Optional[str]:
        """
        Manual search approach using SEC search functionality
        This is a last resort method
        """
        try:
            # Try SEC's EDGAR search interface programmatically
            search_endpoints = [
                f"https://efts.sec.gov/LATEST/search-index?ticker={ticker}",
                f"https://www.sec.gov/cgi-bin/browse-edgar?company={ticker}&CIK=&type=&dateb=&owner=include&action=getcompany"
            ]
            
            for search_url in search_endpoints:
                try:
                    self._rate_limit()
                    response = self.session.get(search_url, timeout=30)
                    
                    if response.status_code == 200:
                        # Look for CIK patterns in the response
                        import re
                        text = response.text
                        
                        # Look for CIK patterns like "CIK: 0001234567" or "CIK=0001234567"
                        cik_patterns = [
                            r'CIK[:\s=]+(\d{10})',
                            r'CIK[:\s=]+(\d{1,10})',
                            r'cik[:\s=]+(\d{10})',
                            r'cik[:\s=]+(\d{1,10})',
                            r'"cik"[:\s]*"?(\d{1,10})"?',
                            r'Central Index Key[:\s]+(\d{1,10})'
                        ]
                        
                        for pattern in cik_patterns:
                            matches = re.findall(pattern, text, re.IGNORECASE)
                            if matches:
                                cik = matches[0]
                                formatted_cik = self._format_cik(cik)
                                logger.info(f"Found {ticker} via manual search: CIK {formatted_cik}")
                                
                                # Verify this CIK actually belongs to the ticker by testing it
                                if self._verify_cik_matches_ticker(formatted_cik, ticker):
                                    return formatted_cik
                
                except Exception as e:
                    logger.debug(f"Manual search URL {search_url} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Manual CIK search failed: {e}")
            return None
    
    def _verify_cik_matches_ticker(self, cik: str, ticker: str) -> bool:
        """Verify that a CIK actually matches the expected ticker"""
        try:
            self._rate_limit()
            test_url = f"{self.base_url}/xbrl/companyfacts/CIK{cik}.json"
            response = self.session.get(test_url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                entity_name = data.get('entityName', '').upper()
                found_ticker = data.get('ticker', '').upper()
                
                # Check if the ticker matches or if the company name contains the ticker
                if (found_ticker == ticker.upper() or 
                    ticker.upper() in entity_name or 
                    entity_name.startswith(ticker.upper())):
                    logger.info(f"Verified CIK {cik} matches {ticker} (Company: {data.get('entityName', 'Unknown')})")
                    return True
                else:
                    logger.debug(f"CIK {cik} does not match {ticker} (Found: {found_ticker}, Entity: {entity_name})")
                    return False
            
            return False
            
        except Exception as e:
            logger.debug(f"Error verifying CIK {cik} for ticker {ticker}: {e}")
            return False
    
    def _lookup_cik_from_edgar_search(self, ticker: str) -> Optional[str]:
        """
        Look up CIK using EDGAR company search
        """
        try:
            # Try the EDGAR company search endpoint
            search_url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{ticker}/us-gaap/Assets.json"
            self._rate_limit()
            
            response = self.session.get(search_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                cik = data.get('cik')
                if cik:
                    formatted_cik = self._format_cik(cik)
                    logger.info(f"Found CIK for {ticker} via EDGAR search: {formatted_cik}")
                    return formatted_cik
            
            # Try alternative EDGAR endpoints
            alternative_endpoints = [
                f"https://data.sec.gov/submissions/CIK{ticker}.json",
                f"https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/CY2023Q4.json"
            ]
            
            for endpoint in alternative_endpoints:
                try:
                    self._rate_limit()
                    response = self.session.get(endpoint, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Search for our ticker in the response
                        if 'data' in data:
                            for entry in data['data']:
                                if isinstance(entry, dict) and entry.get('ticker', '').upper() == ticker.upper():
                                    cik = entry.get('cik')
                                    if cik:
                                        formatted_cik = self._format_cik(cik)
                                        logger.info(f"Found CIK for {ticker} in EDGAR data: {formatted_cik}")
                                        return formatted_cik
                        
                        # Also check if this is a direct CIK response
                        if data.get('cik') and data.get('ticker', '').upper() == ticker.upper():
                            cik = data.get('cik')
                            formatted_cik = self._format_cik(cik)
                            logger.info(f"Found CIK for {ticker} in direct response: {formatted_cik}")
                            return formatted_cik
                            
                except Exception as e:
                    logger.debug(f"Alternative endpoint {endpoint} failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"EDGAR search failed for {ticker}: {e}")
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
            if isinstance(cik, int):
                formatted_cik = f"{cik:010d}"
            elif isinstance(cik, str):
                formatted_cik = f"{int(cik):010d}"
            else:
                raise ValueError(f"Invalid CIK format: {cik}")
                
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