"""
SEC Company Facts API Handler - Provides authoritative fiscal year data and time filters

This module provides shared functionality for:
1. Fiscal year correction during ingestion (fixes period-end-date vs actual FY mismatch)
2. Time period filters for search (gets latest 10-K/10-Q for search targeting)

Both use SEC Company Facts API as the authoritative source for fiscal year information.
"""

import requests
import time
import os
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Configurable Company Facts field - can be changed if Assets data is not available
SEC_COMPANY_FACTS_FIELD = "Assets"  # Primary field: us-gaap Assets
SEC_COMPANY_FACTS_FALLBACK_FIELDS = [  # Fallback fields if primary is missing
    "Revenues", 
    "CashAndCashEquivalentsAtCarryingValue",
    "AssetsCurrent",
    "PropertyPlantAndEquipmentNet"
]


class CompanyFactsHandler:
    """
    Shared handler for SEC Company Facts API operations.
    
    Provides shared functionality for:
    1. Fiscal year correction (ingestion)
    2. Time period filters (search)
    3. Cross-validation logic
    """
    
    def __init__(self, user_agent: str = None):
        self.user_agent = user_agent or os.getenv('SEC_USER_AGENT', 'company-facts-handler@example.com')
        self.rate_limit_delay = 0.1  # SEC allows 10 requests per second
        self.last_request_time = 0
        
        # Cache for company facts data to avoid repeated API calls
        self._company_facts_cache = {}
        
        if '@' not in self.user_agent or '.' not in self.user_agent:
            raise ValueError("SEC_USER_AGENT must be a valid email address")
    
    def _rate_limit(self):
        """Ensure we don't exceed SEC rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK from ticker using SEC company tickers mapping"""
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {"User-Agent": self.user_agent}
            
            self._rate_limit()
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            ticker = ticker.upper().strip()
            
            for item in data.values():
                if item["ticker"] == ticker:
                    cik_int = int(item["cik_str"])
                    formatted_cik = str(cik_int).zfill(10)
                    logger.info(f"Found CIK {formatted_cik} for ticker {ticker}")
                    return formatted_cik
            
            logger.warning(f"No CIK found for ticker {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching CIK for ticker {ticker}: {e}")
            return None
    
    def _get_company_facts(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get company facts for ticker (with caching)"""
        if ticker in self._company_facts_cache:
            logger.info(f"Using cached company facts for {ticker}")
            return self._company_facts_cache[ticker]
        
        try:
            # Get CIK first
            cik = self._get_cik_from_ticker(ticker)
            if not cik:
                return None
            
            # Fetch company facts
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "application/json",
                "Host": "data.sec.gov"
            }
            
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            logger.info(f"Fetching company facts from: {url}")
            
            self._rate_limit()
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            company_facts = response.json()
            
            # Cache the result
            self._company_facts_cache[ticker] = company_facts
            logger.info(f"Successfully fetched and cached company facts for {ticker}")
            return company_facts
            
        except Exception as e:
            logger.error(f"Error fetching company facts for {ticker}: {e}")
            return None
    
    def _get_usd_data_from_company_facts(self, company_facts: Dict[str, Any], ticker: str) -> List[Dict[str, Any]]:
        """
        Extract USD data from Company Facts using configurable field with fallbacks.
        
        Args:
            company_facts: Company facts data from SEC API
            ticker: Ticker symbol (for logging)
            
        Returns:
            List of USD data entries, or empty list if none found
        """
        us_gaap_facts = company_facts.get('facts', {}).get('us-gaap', {})
        if not us_gaap_facts:
            logger.warning(f"No us-gaap facts found for {ticker}")
            return []
        
        # Try primary field first
        primary_field = SEC_COMPANY_FACTS_FIELD
        field_data = us_gaap_facts.get(primary_field, {})
        usd_data = field_data.get('units', {}).get('USD', [])
        
        if usd_data:
            logger.info(f"Found {len(usd_data)} {primary_field} USD entries for {ticker}")
            return usd_data
        
        # Try fallback fields
        for fallback_field in SEC_COMPANY_FACTS_FALLBACK_FIELDS:
            logger.info(f"Primary field {primary_field} not available, trying {fallback_field}")
            field_data = us_gaap_facts.get(fallback_field, {})
            usd_data = field_data.get('units', {}).get('USD', [])
            
            if usd_data:
                logger.info(f"Found {len(usd_data)} {fallback_field} USD entries for {ticker}")
                return usd_data
        
        logger.warning(f"No USD data found in any Company Facts fields for {ticker}")
        return []
    
    def get_corrected_fiscal_year(self, ticker: str, filing_date: str, form_type: str) -> Optional[str]:
        """
        Get the correct fiscal year by matching filing_date and form_type with Company Facts data.
        Includes cross-validation logic to fix inconsistent 10-Q fiscal years in Company Facts.
        
        Args:
            ticker: Company ticker symbol
            filing_date: Filing date from metadata (format: YYYY-MM-DD or similar)
            form_type: Form type ('10-K' or '10-Q')
            
        Returns:
            Corrected fiscal year as string, or None if not found
        """
        try:
            logger.info(f"Correcting fiscal year for {ticker} {form_type} filed {filing_date}")
            
            # Get company facts
            company_facts = self._get_company_facts(ticker)
            if not company_facts:
                logger.warning(f"Could not get company facts for {ticker}")
                return None
            
            # Extract USD data using configurable field with fallbacks
            usd_data = self._get_usd_data_from_company_facts(company_facts, ticker)
            
            if not usd_data:
                logger.warning(f"No USD data found for {ticker}")
                return None
            
            # Normalize filing_date for matching
            try:
                if isinstance(filing_date, str):
                    # Handle different date formats
                    if len(filing_date) == 10 and '-' in filing_date:  # YYYY-MM-DD
                        normalized_filing_date = filing_date
                    elif len(filing_date) == 8:  # YYYYMMDD
                        normalized_filing_date = f"{filing_date[:4]}-{filing_date[4:6]}-{filing_date[6:8]}"
                    else:
                        # Try to parse as datetime
                        dt = datetime.fromisoformat(str(filing_date).split('T')[0])
                        normalized_filing_date = dt.strftime('%Y-%m-%d')
                else:
                    # Assume it's a datetime object
                    normalized_filing_date = filing_date.strftime('%Y-%m-%d')
                    
            except Exception as e:
                logger.error(f"Could not normalize filing_date {filing_date}: {e}")
                return None
            
            # Look for matching entries
            matches = []
            for entry in usd_data:
                entry_filed = entry.get('filed')  # Filed date in Company Facts
                entry_form = entry.get('form')    # Form type in Company Facts
                entry_fy = entry.get('fy')        # Fiscal year in Company Facts
                entry_fp = entry.get('fp')        # Fiscal period in Company Facts
                
                if not all([entry_filed, entry_form, entry_fy]):
                    continue
                
                # Match form type
                if entry_form != form_type:
                    continue
                
                # Match filing date
                if entry_filed == normalized_filing_date:
                    matches.append({
                        'fy': entry_fy,
                        'fp': entry_fp,
                        'filed': entry_filed,
                        'form': entry_form,
                        'end': entry.get('end')
                    })
            
            if matches:
                # If multiple matches, take the most appropriate one
                if len(matches) == 1:
                    result_fy = str(matches[0]['fy'])
                    logger.info(f"Found exact match: {matches[0]}")
                    
                    # For 10-Q, cross-validate against later 10-K filings
                    if form_type == '10-Q':
                        result_fy = self._cross_validate_10q_fiscal_year(ticker, normalized_filing_date, result_fy, company_facts)
                    
                    logger.info(f"Final corrected fiscal year for {ticker} {form_type} filed {filing_date}: {result_fy}")
                    return result_fy
                else:
                    # Multiple matches - take the one with most appropriate fiscal period
                    logger.info(f"Found {len(matches)} matches, selecting best one")
                    if form_type == '10-K':
                        # For 10-K, prefer FY period
                        fy_matches = [m for m in matches if m.get('fp') == 'FY']
                        if fy_matches:
                            result_fy = str(fy_matches[0]['fy'])
                            logger.info(f"Selected FY match: {fy_matches[0]}")
                            return result_fy
                    
                    # Default to first match
                    result_fy = str(matches[0]['fy'])
                    logger.info(f"Selected first match: {matches[0]}")
                    
                    # For 10-Q, cross-validate against later 10-K filings
                    if form_type == '10-Q':
                        result_fy = self._cross_validate_10q_fiscal_year(ticker, normalized_filing_date, result_fy, company_facts)
                    
                    return result_fy
            else:
                logger.warning(f"No exact filing date match found for {ticker} {form_type} filed {filing_date}")
                
                # Try fuzzy matching (within a few days)
                import datetime as dt
                try:
                    target_date = dt.datetime.strptime(normalized_filing_date, '%Y-%m-%d').date()
                    
                    fuzzy_matches = []
                    for entry in usd_data:
                        entry_filed = entry.get('filed')
                        entry_form = entry.get('form')
                        entry_fy = entry.get('fy')
                        
                        if not all([entry_filed, entry_form, entry_fy]):
                            continue
                        
                        if entry_form != form_type:
                            continue
                        
                        try:
                            entry_date = dt.datetime.strptime(entry_filed, '%Y-%m-%d').date()
                            days_diff = abs((target_date - entry_date).days)
                            
                            if days_diff <= 5:  # Within 5 days
                                fuzzy_matches.append({
                                    'fy': entry_fy,
                                    'fp': entry.get('fp'),
                                    'filed': entry_filed,
                                    'form': entry_form,
                                    'days_diff': days_diff
                                })
                        except:
                            continue
                    
                    if fuzzy_matches:
                        # Sort by days difference and take closest
                        fuzzy_matches.sort(key=lambda x: x['days_diff'])
                        best_match = fuzzy_matches[0]
                        result_fy = str(best_match['fy'])
                        logger.info(f"Found fuzzy match (±{best_match['days_diff']} days): {best_match}")
                        
                        # For 10-Q, cross-validate against later 10-K filings
                        if form_type == '10-Q':
                            result_fy = self._cross_validate_10q_fiscal_year(ticker, normalized_filing_date, result_fy, company_facts)
                        
                        return result_fy
                        
                except Exception as e:
                    logger.error(f"Error in fuzzy matching: {e}")
                
                return None
        
        except Exception as e:
            logger.error(f"Error correcting fiscal year for {ticker}: {e}")
            return None
    
    def _cross_validate_10q_fiscal_year(self, ticker: str, filing_date: str, initial_fy: str, company_facts: Dict[str, Any]) -> str:
        """
        Cross-validate 10-Q fiscal year against 10-K fiscal years to fix Company Facts inconsistencies.
        
        Logic: A 10-Q fiscal year should be ≤ the fiscal year of any 10-K filed after it.
        If not, use the 10-K's fiscal year instead.
        
        Args:
            ticker: Company ticker
            filing_date: 10-Q filing date (YYYY-MM-DD)
            initial_fy: Initial fiscal year from Company Facts
            company_facts: Company facts data
            
        Returns:
            Validated/corrected fiscal year
        """
        try:
            logger.info(f"Cross-validating 10-Q FY {initial_fy} for {ticker} filed {filing_date}")
            
            # Extract USD data using configurable field with fallbacks
            usd_data = self._get_usd_data_from_company_facts(company_facts, ticker)
            
            if not usd_data:
                logger.warning(f"No USD data for cross-validation")
                return initial_fy
            
            # Parse filing date
            import datetime as dt
            try:
                filing_date_obj = dt.datetime.strptime(filing_date, '%Y-%m-%d').date()
            except:
                logger.warning(f"Could not parse filing date {filing_date}")
                return initial_fy
            
            # Find all 10-K filings with later filing dates
            later_10k_filings = []
            for entry in usd_data:
                entry_form = entry.get('form')
                entry_filed = entry.get('filed')
                entry_fy = entry.get('fy')
                entry_fp = entry.get('fp')
                
                # Only consider 10-K annual filings
                if entry_form != '10-K' or entry_fp != 'FY':
                    continue
                
                if not all([entry_filed, entry_fy]):
                    continue
                
                try:
                    entry_date_obj = dt.datetime.strptime(entry_filed, '%Y-%m-%d').date()
                    
                    # Check if this 10-K was filed after our 10-Q
                    if entry_date_obj > filing_date_obj:
                        later_10k_filings.append({
                            'filed': entry_filed,
                            'fy': entry_fy,
                            'days_after': (entry_date_obj - filing_date_obj).days
                        })
                except:
                    continue
            
            if not later_10k_filings:
                logger.info(f"No later 10-K filings found, keeping FY {initial_fy}")
                return initial_fy
            
            # Sort by filing date and find the nearest later 10-K
            later_10k_filings.sort(key=lambda x: x['days_after'])
            nearest_10k = later_10k_filings[0]
            
            logger.info(f"Found nearest later 10-K: filed {nearest_10k['filed']} ({nearest_10k['days_after']} days later), FY {nearest_10k['fy']}")
            
            # Cross-validation rule: 10-Q FY should be ≤ 10-K FY
            initial_fy_int = int(initial_fy)
            nearest_10k_fy_int = int(nearest_10k['fy'])
            
            if initial_fy_int > nearest_10k_fy_int:
                logger.warning(f"INCONSISTENCY DETECTED: 10-Q FY {initial_fy} > later 10-K FY {nearest_10k['fy']}")
                logger.info(f"Correcting 10-Q fiscal year: {initial_fy} → {nearest_10k['fy']}")
                return str(nearest_10k['fy'])
            else:
                logger.info(f"Cross-validation passed: 10-Q FY {initial_fy} ≤ later 10-K FY {nearest_10k['fy']}")
                return initial_fy
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return initial_fy
    
    def get_latest_filings_for_search(self, ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get latest 10-K and 10-Q fiscal year information for search filtering.
        
        This method is used by search functionality to determine time period filters.
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Tuple of (latest_10k_info, latest_10q_info) with form_type, fiscal_year, fiscal_quarter, end_date
        """
        try:
            logger.info(f"Getting latest filings for search: {ticker}")
            
            # Get company facts
            company_facts = self._get_company_facts(ticker)
            if not company_facts:
                logger.warning(f"Could not get company facts for search filters: {ticker}")
                return {}, {}
            
            # Extract USD data using configurable field with fallbacks
            usd_data = self._get_usd_data_from_company_facts(company_facts, ticker)
            
            if not usd_data:
                logger.warning(f"No USD data found for search filters: {ticker}")
                return {}, {}
            
            # Group entries by fiscal year and period
            fy_entries = []
            q_entries = []
            quarter_order = {"Q1": 1, "Q2": 2, "Q3": 3}
            
            for entry in usd_data:
                fiscal_year = entry.get('fy')
                fiscal_period = entry.get('fp')
                end_date = entry.get('end')
                
                if not all([fiscal_year, fiscal_period, end_date]):
                    continue
                
                if fiscal_period == 'FY':
                    fy_entries.append((fiscal_year, end_date, entry))
                elif fiscal_period in ['Q1', 'Q2', 'Q3']:
                    q_entries.append((fiscal_year, fiscal_period, end_date, entry))
            
            # Find latest 10-K: highest fiscal year, and for ties, latest end date
            latest_10k = {}
            if fy_entries:
                latest_fy = max(fy_entry[0] for fy_entry in fy_entries)
                latest_fy_entries = [entry for entry in fy_entries if entry[0] == latest_fy]
                latest_fy_entries.sort(key=lambda x: x[1], reverse=True)
                
                latest_10k_year, latest_10k_end, _ = latest_fy_entries[0]
                latest_10k = {
                    "form_type": "10-K",
                    "fiscal_year": str(latest_10k_year),
                    "end_date": latest_10k_end
                }
                logger.info(f"Latest 10-K for search: FY {latest_10k_year} ending {latest_10k_end}")
            
            # Find latest 10-Q: highest fiscal year+quarter, and for ties, latest end date
            latest_10q = {}
            if q_entries:
                latest_fyq = max((entry[0], quarter_order.get(entry[1], 0)) for entry in q_entries)
                latest_fyq_entries = [entry for entry in q_entries if entry[0] == latest_fyq[0] and quarter_order.get(entry[1], 0) == latest_fyq[1]]
                latest_fyq_entries.sort(key=lambda x: x[2], reverse=True)
                
                latest_10q_year, latest_10q_quarter, latest_10q_end, _ = latest_fyq_entries[0]
                latest_10q = {
                    "form_type": "10-Q",
                    "fiscal_year": str(latest_10q_year),
                    "fiscal_quarter": latest_10q_quarter,
                    "end_date": latest_10q_end
                }
                logger.info(f"Latest 10-Q for search: FY {latest_10q_year} {latest_10q_quarter} ending {latest_10q_end}")
            
            return latest_10k, latest_10q
            
        except Exception as e:
            logger.error(f"Error getting latest filings for search: {e}")
            return {}, {}
    
    def get_time_filters_for_search(self, ticker: str, time_period: str) -> Dict[str, Any]:
        """
        Generate time-based filters for search operations.
        
        This replaces the _determine_time_filters function from agent_interface.py
        
        Args:
            ticker: Company ticker symbol
            time_period: One of 'latest', 'latest_10k_and_10q', 'latest_10k', 'latest_10q'
            
        Returns:
            Dict with time filters for search
        """
        try:
            logger.info(f"Getting time filters for {ticker}, period: {time_period}")
            
            if time_period == "latest":
                # Get the single most recent report (10-K or 10-Q, whichever is newer by end_date)
                latest_10k, latest_10q = self.get_latest_filings_for_search(ticker)
                
                if latest_10k and latest_10q:
                    # Compare end dates to find the most recent
                    from datetime import datetime
                    try:
                        k_end_date = datetime.strptime(latest_10k['end_date'], '%Y-%m-%d')
                        q_end_date = datetime.strptime(latest_10q['end_date'], '%Y-%m-%d')
                        
                        if q_end_date > k_end_date:
                            logger.info(f"Latest single report for {ticker}: 10-Q (end date {latest_10q['end_date']}) is newer than 10-K (end date {latest_10k['end_date']})")
                            return {k: v for k, v in latest_10q.items() if k != 'end_date'}  # Remove end_date from filter
                        else:
                            logger.info(f"Latest single report for {ticker}: 10-K (end date {latest_10k['end_date']}) is newer than or equal to 10-Q (end date {latest_10q['end_date']})")
                            return {k: v for k, v in latest_10k.items() if k != 'end_date'}  # Remove end_date from filter
                    except Exception as e:
                        logger.warning(f"Could not compare end dates for {ticker}: {e}. Defaulting to 10-K.")
                        return {k: v for k, v in latest_10k.items() if k != 'end_date'}
                        
                elif latest_10k:
                    logger.info(f"Only 10-K available for {ticker}")
                    return {k: v for k, v in latest_10k.items() if k != 'end_date'}
                elif latest_10q:
                    logger.info(f"Only 10-Q available for {ticker}")
                    return {k: v for k, v in latest_10q.items() if k != 'end_date'}
                else:
                    logger.warning(f"Could not determine latest filings for {ticker}")
                    return {}
                    
            elif time_period == "latest_10k_and_10q":
                # Get both latest 10-K AND latest 10-Q (old 'latest' behavior)
                latest_10k, latest_10q = self.get_latest_filings_for_search(ticker)
                
                if latest_10k and latest_10q:
                    # Remove end_date from filters before returning
                    latest_10k_filter = {k: v for k, v in latest_10k.items() if k != 'end_date'}
                    latest_10q_filter = {k: v for k, v in latest_10q.items() if k != 'end_date'}
                    return {
                        "multi_report_filter": True,
                        "reports": [latest_10k_filter, latest_10q_filter]
                    }
                elif latest_10k:
                    return {k: v for k, v in latest_10k.items() if k != 'end_date'}
                elif latest_10q:
                    return {k: v for k, v in latest_10q.items() if k != 'end_date'}
                else:
                    logger.warning(f"Could not determine latest filings for {ticker}")
                    return {}
                    
            elif time_period == "latest_10k":
                # Get latest 10-K only
                latest_10k, _ = self.get_latest_filings_for_search(ticker)
                return {k: v for k, v in latest_10k.items() if k != 'end_date'}
                
            elif time_period == "latest_10q":
                # Get latest 10-Q only
                _, latest_10q = self.get_latest_filings_for_search(ticker)
                return {k: v for k, v in latest_10q.items() if k != 'end_date'}
            
            else:
                # For other time periods, return no restrictions
                logger.info(f"No Company Facts filters for time_period: {time_period}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting time filters for search: {e}")
            return {}
    
    def correct_metadata_fiscal_year(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correct the fiscal_year field in metadata using Company Facts API.
        
        Args:
            metadata: Metadata dictionary containing ticker, filing_date, form_type
            
        Returns:
            Updated metadata with corrected fiscal_year, or original if correction fails
        """
        try:
            ticker = metadata.get('ticker')
            filing_date = metadata.get('filing_date')
            form_type = metadata.get('form_type')
            
            if not all([ticker, filing_date, form_type]):
                logger.warning(f"Missing required fields for fiscal year correction: ticker={ticker}, filing_date={filing_date}, form_type={form_type}")
                return metadata
            
            # Get corrected fiscal year
            corrected_fy = self.get_corrected_fiscal_year(ticker, filing_date, form_type)
            
            if corrected_fy:
                original_fy = metadata.get('fiscal_year')
                metadata = metadata.copy()  # Don't modify original
                metadata['fiscal_year'] = corrected_fy
                metadata['fiscal_year_corrected'] = True
                metadata['original_fiscal_year'] = original_fy
                
                logger.info(f"Corrected fiscal year: {original_fy} → {corrected_fy}")
                return metadata
            else:
                logger.warning(f"Could not correct fiscal year for {ticker} {form_type} filed {filing_date}")
                return metadata
                
        except Exception as e:
            logger.error(f"Error in correct_metadata_fiscal_year: {e}")
            return metadata


class FiscalYearCorrector:
    """
    Backward-compatible wrapper for CompanyFactsHandler focused on fiscal year correction.
    
    This class maintains the existing API for fiscal year correction while leveraging
    the shared CompanyFactsHandler functionality.
    """
    
    def __init__(self, user_agent: str = None):
        self._handler = CompanyFactsHandler(user_agent=user_agent)
    
    def get_corrected_fiscal_year(self, ticker: str, filing_date: str, form_type: str) -> Optional[str]:
        """Get corrected fiscal year - delegates to CompanyFactsHandler"""
        return self._handler.get_corrected_fiscal_year(ticker, filing_date, form_type)
    
    def correct_metadata_fiscal_year(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Correct metadata fiscal year - delegates to CompanyFactsHandler"""
        return self._handler.correct_metadata_fiscal_year(metadata)


def correct_fiscal_year_for_metadata(metadata: Dict[str, Any], user_agent: str = None) -> Dict[str, Any]:
    """
    Convenience function to correct fiscal year for a single metadata entry.
    
    Args:
        metadata: Metadata dictionary
        user_agent: SEC user agent (email address)
        
    Returns:
        Metadata with corrected fiscal_year if possible
    """
    corrector = FiscalYearCorrector(user_agent=user_agent)
    return corrector.correct_metadata_fiscal_year(metadata)


def get_time_filters_for_search(ticker: str, time_period: str, user_agent: str = None) -> Dict[str, Any]:
    """
    Convenience function to get time filters for search operations.
    
    Args:
        ticker: Company ticker symbol
        time_period: One of 'latest', 'latest_10k', 'latest_10q'
        user_agent: SEC user agent (email address)
        
    Returns:
        Dict with time filters for search
    """
    handler = CompanyFactsHandler(user_agent=user_agent)
    return handler.get_time_filters_for_search(ticker, time_period)


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Testing Fiscal Year Corrector with Cross-Validation...")
    print("=" * 60)
    
    # Test 1: TGT 10-K fiscal year correction
    test_10k_metadata = {
        'ticker': 'TGT',
        'filing_date': '2025-03-12',
        'form_type': '10-K',
        'fiscal_year': '2025',  # Current (incorrect) value
        'period_end_date': '2025-02-01'
    }
    
    print("\nTest 1: TGT 10-K Correction")
    print(f"Original metadata: {test_10k_metadata}")
    corrected_10k = correct_fiscal_year_for_metadata(test_10k_metadata)
    print(f"Corrected metadata: {corrected_10k}")
    
    # Test 2: TGT 10-Q fiscal year correction with cross-validation
    test_10q_metadata = {
        'ticker': 'TGT', 
        'filing_date': '2022-11-23',  # This is the problematic 10-Q
        'form_type': '10-Q',
        'fiscal_year': '2023',  # Company Facts says 2023, but should be 2022
        'period_end_date': '2022-10-29'
    }
    
    print("\nTest 2: TGT 10-Q Cross-Validation (should correct 2023 → 2022)")
    print(f"Original metadata: {test_10q_metadata}")
    corrected_10q = correct_fiscal_year_for_metadata(test_10q_metadata)
    print(f"Corrected metadata: {corrected_10q}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"10-K: {test_10k_metadata['fiscal_year']} → {corrected_10k.get('fiscal_year', 'FAILED')}")
    print(f"10-Q: {test_10q_metadata['fiscal_year']} → {corrected_10q.get('fiscal_year', 'FAILED')}")
    if corrected_10q.get('fiscal_year') == '2022':
        print("✅ Cross-validation logic working correctly!")
    else:
        print("❌ Cross-validation may need adjustment")