"""
Alpha Vantage transcript data source implementation
Implements the TranscriptDataSource interface for Alpha Vantage API
"""

import os
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re
import json
from dataclasses import asdict

from .data_source_interface import (
    TranscriptDataSource, TranscriptData, TranscriptQuery, 
    DataSourceError, transcript_registry
)


class AlphaVantageTranscriptSource(TranscriptDataSource):
    """Alpha Vantage implementation of transcript data source"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Alpha Vantage transcript source
        
        Args:
            api_key: Alpha Vantage API key (if not provided, uses ALPHA_VANTAGE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise DataSourceError(
                "Alpha Vantage API key required. Set ALPHA_VANTAGE_API_KEY environment variable or pass api_key parameter.",
                error_code="MISSING_API_KEY",
                source="alpha_vantage"
            )
        
        self.base_url = "https://www.alphavantage.co/query"
        
        # Get configurable delay from environment variable
        try:
            self.rate_limit_delay = int(os.getenv('ALPHA_VANTAGE_DELAY', '3'))
        except (ValueError, TypeError):
            self.rate_limit_delay = 3  # Default fallback
            
        self.last_request_time = 0
        
        # Set up logging
        self.logger = logging.getLogger(f'alpha_vantage_{id(self)}')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Supported date range (Alpha Vantage earnings data typically goes back ~10 years)
        self.earliest_date = datetime(2014, 1, 1)
        self.latest_date = datetime.now()
    
    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        Make rate-limited request to Alpha Vantage API with detailed logging
        
        Args:
            params: Request parameters
            
        Returns:
            JSON response as dictionary
            
        Raises:
            DataSourceError: If request fails
        """
        # Log the request being made
        function_name = params.get('function', 'UNKNOWN')
        symbol = params.get('symbol', 'UNKNOWN')
        quarter = params.get('quarter', 'N/A')
        
        if quarter != 'N/A':
            self.logger.info(f"Making API request: {function_name} for {symbol} {quarter}")
            print(f"    📡 API Request: {function_name} for {symbol} {quarter}")
        else:
            self.logger.info(f"Making API request: {function_name} for {symbol}")
            print(f"    📡 API Request: {function_name} for {symbol}")
        
        # Rate limiting with detailed logging
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            self.logger.info(f"Rate limiting: Waiting {sleep_time:.1f} seconds (configured delay: {self.rate_limit_delay}s)")
            print(f"    ⏳ Rate limit delay: {sleep_time:.1f} seconds (configured: {self.rate_limit_delay}s)")
            time.sleep(sleep_time)
        
        # Add API key to parameters
        params['apikey'] = self.api_key
        
        try:
            request_start_time = time.time()
            response = requests.get(self.base_url, params=params, timeout=30)
            request_end_time = time.time()
            
            response.raise_for_status()
            
            self.last_request_time = time.time()
            
            # Log response time
            response_time = request_end_time - request_start_time
            self.logger.info(f"API response received in {response_time:.2f} seconds")
            print(f"    ✅ Response received in {response_time:.2f}s")
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                error_msg = data['Error Message']
                self.logger.error(f"Alpha Vantage API error: {error_msg}")
                print(f"    ❌ API Error: {error_msg}")
                raise DataSourceError(
                    f"Alpha Vantage API error: {error_msg}",
                    error_code="API_ERROR",
                    source="alpha_vantage"
                )
            
            if "Note" in data:
                rate_limit_msg = data['Note']
                # Hide API key from logs for security
                sanitized_msg = rate_limit_msg.replace(self.api_key, "***API_KEY_HIDDEN***")
                self.logger.warning(f"Alpha Vantage rate limit: {sanitized_msg}")
                print(f"    ⚠️ Rate Limit: {sanitized_msg}")
                raise DataSourceError(
                    f"Alpha Vantage rate limit exceeded (API key hidden for security)",
                    error_code="RATE_LIMIT",
                    source="alpha_vantage"
                )
            
            # Check for rate limit via "Information" field (daily limit exhausted)
            if "Information" in data and "rate limit" in data["Information"].lower():
                rate_limit_msg = data['Information']
                # Hide API key from logs for security
                sanitized_msg = rate_limit_msg.replace(self.api_key, "***API_KEY_HIDDEN***")
                self.logger.warning(f"Alpha Vantage daily rate limit exhausted: {sanitized_msg}")
                print(f"    🚫 Daily Rate Limit Exhausted: {sanitized_msg}")
                raise DataSourceError(
                    f"Alpha Vantage daily rate limit exhausted (API key hidden for security)",
                    error_code="DAILY_RATE_LIMIT",
                    source="alpha_vantage"
                )
            
            # Log successful data retrieval
            if 'transcript' in data and isinstance(data['transcript'], list):
                transcript_count = len(data['transcript'])
                self.logger.info(f"Successfully retrieved transcript with {transcript_count} entries")
                print(f"    📄 Transcript data: {transcript_count} entries")
            elif 'symbol' in data and function_name == 'OVERVIEW':
                self.logger.info(f"Successfully retrieved company overview for {symbol}")
                print(f"    🏢 Company overview retrieved for {symbol}")
            else:
                self.logger.info("API request successful - response structure unknown")
                print(f"    ✅ API request successful")
            
            return data
            
        except requests.RequestException as e:
            self.logger.error(f"Network error during API request: {str(e)}")
            print(f"    🌐 Network Error: {str(e)}")
            raise DataSourceError(
                f"Failed to fetch data from Alpha Vantage: {str(e)}",
                error_code="NETWORK_ERROR",
                source="alpha_vantage"
            )
    
    def _parse_transcript_data(self, data: Dict[str, Any], ticker: str, quarter: str) -> Optional[TranscriptData]:
        """
        Parse Alpha Vantage transcript data into TranscriptData object
        
        Args:
            data: Raw API response
            ticker: Company ticker symbol
            quarter: Quarter string (e.g., '2024Q1')
            
        Returns:
            TranscriptData object or None if invalid
        """
        # Validate response structure
        if not data.get('transcript') or not isinstance(data['transcript'], list):
            return None
        
        # Get company name from symbol
        company_name = data.get('symbol', ticker)
        quarter_str = data.get('quarter', quarter)
        
        # Parse quarter and year
        try:
            year_str = quarter_str[:4]
            quarter_num = quarter_str[5:]  # Extract Q1, Q2, etc.
            fiscal_year = year_str
            
            # Estimate transcript date based on quarter
            quarter_month_map = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}
            month = quarter_month_map.get(quarter_num, 1)
            transcript_date = datetime(int(year_str), month, 15)  # Mid-quarter estimate
        except (ValueError, KeyError):
            return None
        
        # Build structured transcript content from API response
        content_parts = []
        participants = []
        speakers_seen = set()
        
        # Add header
        content_parts.append(f"EARNINGS CALL TRANSCRIPT - {company_name}")
        content_parts.append(f"Quarter: {quarter_str}")
        content_parts.append("=" * 80)
        content_parts.append("")
        
        # Process transcript entries
        for entry in data['transcript']:
            speaker = entry.get('speaker', 'Unknown')
            title = entry.get('title', '')
            content = entry.get('content', '')
            sentiment = entry.get('sentiment', '0.0')
            
            if speaker not in speakers_seen:
                speakers_seen.add(speaker)
                if title:
                    participants.append(f"{speaker} ({title})")
                else:
                    participants.append(speaker)
            
            # Format content with speaker
            if title and title != speaker:
                speaker_line = f"{speaker} ({title}): {content}"
            else:
                speaker_line = f"{speaker}: {content}"
            
            content_parts.append(speaker_line)
            content_parts.append("")
        
        full_content = "\n".join(content_parts)
        
        # Create metadata
        metadata = {
            'source': 'alpha_vantage',
            'data_type': 'earnings_call_transcript',
            'quarter': quarter_str,
            'has_actual_transcript': True,
            'speaker_count': len(speakers_seen),
            'entry_count': len(data['transcript'])
        }
        
        transcript_data = TranscriptData(
            ticker=ticker.upper(),
            company_name=company_name,
            transcript_date=transcript_date,
            quarter=quarter_num,
            fiscal_year=fiscal_year,
            transcript_type="earnings_call",
            content=full_content,
            participants=participants,
            raw_data=data,
            source="alpha_vantage",
            metadata=metadata
        )
        
        return transcript_data
    
    def get_transcripts(self, query: TranscriptQuery) -> List[TranscriptData]:
        """
        Retrieve transcripts based on query parameters
        
        Args:
            query: TranscriptQuery with search parameters
            
        Returns:
            List of TranscriptData objects
        """
        transcripts = []
        
        # Calculate date range for quarter generation
        end_date = query.end_date or datetime.now()
        if query.years_back:
            start_date = end_date - timedelta(days=365 * query.years_back)
        else:
            start_date = query.start_date or self.earliest_date
        
        # Generate quarters to fetch within date range
        self.logger.info(f"Generating quarters for date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"    📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        quarters_to_fetch = self._generate_quarters_in_range(start_date, end_date, query.ticker)
        
        self.logger.info(f"Generated {len(quarters_to_fetch)} quarters to check: {quarters_to_fetch}")
        print(f"    🔍 Checking {len(quarters_to_fetch)} quarters: {quarters_to_fetch}")
        
        successful_transcripts = 0
        for i, quarter in enumerate(quarters_to_fetch):
            self.logger.info(f"Processing quarter {i+1}/{len(quarters_to_fetch)}: {quarter}")
            print(f"    [{i+1}/{len(quarters_to_fetch)}] Processing {quarter}...")
            try:
                params = {
                    'function': 'EARNINGS_CALL_TRANSCRIPT',
                    'symbol': query.ticker.upper(),
                    'quarter': quarter
                }
                
                data = self._make_request(params)
                transcript = self._parse_transcript_data(data, query.ticker, quarter)
                
                if transcript:
                    transcripts.append(transcript)
                    successful_transcripts += 1
                    self.logger.info(f"✓ Successfully retrieved transcript for {quarter}")
                    print(f"      ✅ Success: Found transcript for {quarter}")
                else:
                    self.logger.info(f"✗ No transcript data found for {quarter}")
                    print(f"      ❌ No data: No transcript found for {quarter}")
                
                # Apply limit during fetching to avoid unnecessary API calls
                if query.limit and successful_transcripts >= query.limit:
                    self.logger.info(f"Limit reached ({query.limit}), stopping quarter search")
                    print(f"      🎯 Limit reached ({query.limit}), stopping search")
                    break
                    
            except DataSourceError as e:
                # Handle different types of errors
                if e.error_code == "API_ERROR":
                    self.logger.warning(f"✗ API error for {quarter}: {e}")
                    print(f"      ⚠️ API Error for {quarter}: {e}")
                    continue
                elif e.error_code == "DAILY_RATE_LIMIT":
                    self.logger.error(f"Daily rate limit exhausted on {quarter}: {e}")
                    print(f"      🚫 Daily Rate Limit Exhausted on {quarter}")
                    print(f"      💡 Please wait 24 hours or upgrade your Alpha Vantage plan")
                    # Stop processing when daily limit is hit
                    break
                elif e.error_code == "RATE_LIMIT":
                    self.logger.warning(f"Rate limit hit for {quarter}: {e}")
                    print(f"      ⚠️ Rate Limit for {quarter}: {e}")
                    continue
                else:
                    self.logger.error(f"Fatal error for {quarter}: {e}")
                    print(f"      💥 Fatal Error for {quarter}: {e}")
                    raise
        
        # Sort by date (newest first)
        transcripts.sort(key=lambda x: x.transcript_date, reverse=True)
        
        # Apply final limit
        if query.limit:
            transcripts = transcripts[:query.limit]
        
        # Final summary logging
        self.logger.info(f"Transcript fetching complete: {len(transcripts)} transcripts found out of {len(quarters_to_fetch)} quarters checked")
        print(f"    🎉 Fetching complete: {len(transcripts)} transcripts found")
        
        return transcripts
    
    def _get_fiscal_year_end(self, ticker: str) -> str:
        """
        Get the fiscal year end month for a company using Alpha Vantage OVERVIEW API
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Fiscal year end month name (e.g., 'December', 'September')
            Default to 'December' if unable to retrieve
        """
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker.upper()
            }
            
            data = self._make_request(params)
            fiscal_year_end = data.get('FiscalYearEnd', 'December')
            
            # Validate the month name
            valid_months = [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]
            
            if fiscal_year_end in valid_months:
                return fiscal_year_end
            else:
                return 'December'  # Default fallback
                
        except DataSourceError:
            # If we can't get fiscal year end, default to calendar year (December)
            return 'December'
    
    def _get_fiscal_quarter_start_months(self, fiscal_year_end_month: str) -> List[int]:
        """
        Get the starting months for fiscal quarters based on fiscal year end - FIXED VERSION
        
        Args:
            fiscal_year_end_month: Name of the fiscal year end month
            
        Returns:
            List of 4 months representing fiscal quarter start months [Q1, Q2, Q3, Q4]
        """
        month_name_to_number = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        fiscal_year_end_month_num = month_name_to_number.get(fiscal_year_end_month, 12)
        
        self.logger.info(f"Fiscal year ends in {fiscal_year_end_month} (month {fiscal_year_end_month_num})")
        
        # Calculate fiscal quarter start months
        # Fiscal quarters work backwards from the fiscal year end
        # For September fiscal year end:
        # Q4 ends in Sep (Jul-Sep), so Q4 starts in Jul (month 7)
        # Q3 ends in Jun (Apr-Jun), so Q3 starts in Apr (month 4)  
        # Q2 ends in Mar (Jan-Mar), so Q2 starts in Jan (month 1)
        # Q1 ends in Dec (Oct-Dec), so Q1 starts in Oct (month 10)
        
        # Q4 starts 3 months before fiscal year end
        q4_start = fiscal_year_end_month_num - 2  # 3 months: end-2, end-1, end
        if q4_start <= 0:
            q4_start += 12
            
        # Q3 starts 6 months before fiscal year end  
        q3_start = fiscal_year_end_month_num - 5  # 6 months before
        if q3_start <= 0:
            q3_start += 12
            
        # Q2 starts 9 months before fiscal year end
        q2_start = fiscal_year_end_month_num - 8  # 9 months before
        if q2_start <= 0:
            q2_start += 12
            
        # Q1 starts 12 months before fiscal year end (or immediately after previous fiscal year)
        q1_start = fiscal_year_end_month_num + 1  # Month after fiscal year end
        if q1_start > 12:
            q1_start -= 12
        
        quarters = [q1_start, q2_start, q3_start, q4_start]
        
        self.logger.info(f"Calculated quarter start months: Q1={q1_start}, Q2={q2_start}, Q3={q3_start}, Q4={q4_start}")
        
        return quarters
    
    def _generate_quarters_in_range(self, start_date: datetime, end_date: datetime, ticker: str) -> List[str]:
        """
        Generate quarter strings (e.g., '2024Q1') within the date range based on company's fiscal year
        
        Args:
            start_date: Start date
            end_date: End date
            ticker: Company ticker symbol
            
        Returns:
            List of quarter strings in reverse chronological order
        """
        quarters = []
        
        self.logger.info(f"Starting quarter generation for {ticker}")
        print(f"    🔧 Starting quarter generation for {ticker}")
        
        # Get fiscal year end information for the company
        fiscal_year_end_month = self._get_fiscal_year_end(ticker)
        self.logger.info(f"Fiscal year end month for {ticker}: {fiscal_year_end_month}")
        print(f"    📊 Fiscal year end: {fiscal_year_end_month}")
        
        quarter_start_months = self._get_fiscal_quarter_start_months(fiscal_year_end_month)
        self.logger.info(f"Fiscal quarter start months: {quarter_start_months}")
        print(f"    📅 Quarter start months: {quarter_start_months}")
        
        # Start from the end date and work backwards
        current_date = end_date
        iteration_count = 0
        max_iterations = 50  # Safety limit to prevent infinite loops
        
        self.logger.info(f"Starting quarter iteration from {current_date}")
        print(f"    🔄 Starting from {current_date.strftime('%Y-%m-%d')}")
        
        while current_date >= start_date and iteration_count < max_iterations:
            iteration_count += 1
            
            # Determine fiscal year and quarter for current date
            fiscal_year, fiscal_quarter = self._get_fiscal_year_and_quarter(current_date, quarter_start_months)
            quarter_str = f"{fiscal_year}Q{fiscal_quarter}"
            
            self.logger.info(f"Iteration {iteration_count}: {current_date.strftime('%Y-%m-%d')} -> {quarter_str}")
            print(f"      [{iteration_count}] {current_date.strftime('%Y-%m-%d')} -> {quarter_str}")
            
            # Add quarter if not already added (avoid duplicates)
            if quarter_str not in quarters:
                quarters.append(quarter_str)
                self.logger.info(f"Added quarter: {quarter_str}")
                print(f"        ✓ Added: {quarter_str}")
            else:
                self.logger.info(f"Duplicate quarter skipped: {quarter_str}")
                print(f"        ⚠️ Duplicate: {quarter_str}")
            
            # Move to previous quarter
            prev_date = current_date
            current_date = self._get_previous_quarter_date(current_date, quarter_start_months)
            
            self.logger.info(f"Moving from {prev_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
            print(f"        ⬅️ Next: {current_date.strftime('%Y-%m-%d')}")
            
            # Safety check to avoid infinite loops
            if fiscal_year < 2011:  # Alpha Vantage data availability
                self.logger.info(f"Reached data availability limit (fiscal year {fiscal_year})")
                print(f"        🛑 Reached limit: fiscal year {fiscal_year}")
                break
                
            # Additional safety check for date progression
            if current_date >= prev_date:
                self.logger.error(f"Date not progressing backwards! {prev_date} -> {current_date}")
                print(f"        🚨 ERROR: Date not progressing! {prev_date} -> {current_date}")
                break
        
        if iteration_count >= max_iterations:
            self.logger.warning(f"Hit maximum iteration limit ({max_iterations})")
            print(f"    ⚠️ Hit iteration limit: {max_iterations}")
        
        self.logger.info(f"Quarter generation complete: {len(quarters)} quarters generated in {iteration_count} iterations")
        print(f"    ✅ Generated {len(quarters)} quarters in {iteration_count} iterations")
        
        return quarters
    
    def _get_fiscal_year_and_quarter(self, date: datetime, quarter_start_months: List[int]) -> tuple[int, int]:
        """
        Determine fiscal year and quarter for a given date - SYSTEMATIC APPROACH
        
        Args:
            date: Date to analyze
            quarter_start_months: List of fiscal quarter start months [Q1, Q2, Q3, Q4]
            
        Returns:
            Tuple of (fiscal_year, fiscal_quarter)
        """
        month = date.month
        year = date.year
        
        self.logger.info(f"Determining fiscal year/quarter for {date.strftime('%Y-%m-%d')} (month {month})")
        
        q1_start, q2_start, q3_start, q4_start = quarter_start_months
        
        # Determine fiscal year end month
        fiscal_year_end_month = q1_start - 1
        if fiscal_year_end_month <= 0:
            fiscal_year_end_month += 12
        
        # Determine which quarter the month belongs to
        # Create quarter ranges and check which one the month falls into
        
        quarter = None
        
        # Simple approach: check each quarter's range
        # Need to handle wrap-around for fiscal years that span calendar years
        
        def month_in_range(target_month, start_month, quarters_length=3):
            """Check if target month is within quarters_length months starting from start_month"""
            for i in range(quarters_length):
                check_month = start_month + i
                if check_month > 12:
                    check_month -= 12
                if target_month == check_month:
                    return True
            return False
        
        # Check each quarter (Q1, Q2, Q3, Q4)
        if month_in_range(month, q1_start):
            quarter = 1
        elif month_in_range(month, q2_start):
            quarter = 2
        elif month_in_range(month, q3_start):
            quarter = 3
        elif month_in_range(month, q4_start):
            quarter = 4
        else:
            # Fallback: shouldn't happen with proper quarter start months
            quarter = 1
        
        # Now determine the fiscal year
        # The fiscal year is the calendar year when the fiscal year ENDS
        
        # Key insight: if current month > fiscal_year_end_month, then we're in NEXT fiscal year
        # Exception: when fiscal year spans calendar years and we're in Q1
        
        if fiscal_year_end_month < q1_start:
            # Fiscal year spans calendar years
            # Examples: Sept FY end (fiscal_year_end_month=9, q1_start=10)
            #          June FY end (fiscal_year_end_month=6, q1_start=7)  
            #          Jan FY end (fiscal_year_end_month=1, q1_start=2)
            
            if quarter == 1:
                # Q1 behavior depends on whether we're in the late part (before new year) or early part (after new year)
                if month >= q1_start:
                    # Late part of Q1 (e.g., Oct, Nov, Dec for Sept FY end)
                    # Belongs to fiscal year ending in NEXT calendar year
                    fiscal_year = year + 1
                else:
                    # Early part of Q1 (e.g., Jan for June FY end, when Q1 spans Jul-Sep but we're looking at Jan of following year - shouldn't happen)
                    # Actually, this case shouldn't occur with proper quarter setup. Let's use the fiscal year ending in current year.
                    fiscal_year = year
            else:
                # Q2, Q3, Q4: check if we're before or after fiscal year end
                if month <= fiscal_year_end_month:
                    # Before fiscal year end, so we're in the fiscal year ending in current calendar year
                    fiscal_year = year
                else:
                    # After fiscal year end, so we're in the fiscal year ending in NEXT calendar year
                    fiscal_year = year + 1
        else:
            # Fiscal year aligns with calendar year (e.g., Jan-Dec)
            # fiscal_year_end_month >= q1_start
            fiscal_year = year
        
        self.logger.info(f"Quarter starts: Q1={q1_start}, Q2={q2_start}, Q3={q3_start}, Q4={q4_start}")
        self.logger.info(f"Fiscal year end month: {fiscal_year_end_month}")
        self.logger.info(f"Detected quarter: {quarter}")
        self.logger.info(f"Result: {date.strftime('%Y-%m-%d')} -> fiscal {fiscal_year}Q{quarter}")
        
        return fiscal_year, quarter
    
    def _get_previous_quarter_date(self, date: datetime, quarter_start_months: List[int]) -> datetime:
        """
        Get a date in the previous fiscal quarter - FIXED VERSION
        
        Args:
            date: Current date
            quarter_start_months: List of fiscal quarter start months
            
        Returns:
            Date in the previous fiscal quarter (going backwards in time)
        """
        current_fiscal_year, current_fiscal_quarter = self._get_fiscal_year_and_quarter(date, quarter_start_months)
        
        self.logger.info(f"Current position: {current_fiscal_year}Q{current_fiscal_quarter}")
        
        # Calculate previous quarter
        if current_fiscal_quarter == 1:
            # Previous quarter is Q4 of previous fiscal year
            prev_quarter = 4
            prev_fiscal_year = current_fiscal_year - 1
        else:
            # Previous quarter in same fiscal year
            prev_quarter = current_fiscal_quarter - 1
            prev_fiscal_year = current_fiscal_year
        
        self.logger.info(f"Previous quarter: {prev_fiscal_year}Q{prev_quarter}")
        
        # Get the start month of the previous quarter
        prev_quarter_start_month = quarter_start_months[prev_quarter - 1]
        
        # Calculate the calendar year for the previous quarter
        # For Apple with Sept fiscal year end:
        # Q4 (Jul-Sep): calendar year = fiscal year
        # Q3 (Apr-Jun): calendar year = fiscal year  
        # Q2 (Jan-Mar): calendar year = fiscal year
        # Q1 (Oct-Dec): calendar year = fiscal year - 1
        
        if prev_quarter == 1:  # Q1 is Oct-Dec of previous calendar year
            calendar_year = prev_fiscal_year - 1
        else:  # Q2, Q3, Q4 are in the same calendar year as fiscal year
            calendar_year = prev_fiscal_year
            
        # Handle edge case where start month calculation might be wrong
        if prev_quarter_start_month > 12 or prev_quarter_start_month < 1:
            self.logger.error(f"Invalid month calculated: {prev_quarter_start_month}")
            # Fallback: just subtract 3 months from current date
            if date.month <= 3:
                return datetime(date.year - 1, date.month + 9, 15)
            else:
                return datetime(date.year, date.month - 3, 15)
        
        # Return date in the middle of the previous quarter
        result_date = datetime(calendar_year, prev_quarter_start_month, 15)
        
        self.logger.info(f"Previous quarter date: {result_date.strftime('%Y-%m-%d')}")
        
        # Validate that we're actually going backwards
        if result_date >= date:
            self.logger.error(f"ERROR: Previous date {result_date} is not before current date {date}")
            # Emergency fallback: subtract 90 days
            result_date = date - timedelta(days=90)
            self.logger.info(f"Fallback: Using {result_date.strftime('%Y-%m-%d')}")
        
        return result_date
    
    def get_latest_transcript(self, ticker: str) -> Optional[TranscriptData]:
        """
        Get the most recent transcript for a ticker
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            TranscriptData object or None if not found
        """
        query = TranscriptQuery(ticker=ticker, limit=1)
        transcripts = self.get_transcripts(query)
        return transcripts[0] if transcripts else None
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if ticker is supported by Alpha Vantage
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            True if ticker is valid and supported
        """
        if not ticker or not isinstance(ticker, str):
            return False
        
        # Basic ticker format validation
        ticker = ticker.strip().upper()
        if not re.match(r'^[A-Z]{1,5}$', ticker):
            return False
        
        try:
            # Try to fetch data for validation
            query = TranscriptQuery(ticker=ticker, limit=1)
            transcripts = self.get_transcripts(query)
            return len(transcripts) > 0
        except DataSourceError:
            return False
    
    def get_supported_date_range(self) -> tuple[datetime, datetime]:
        """
        Get the date range supported by Alpha Vantage
        
        Returns:
            Tuple of (earliest_date, latest_date)
        """
        return (self.earliest_date, self.latest_date)
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get rate limiting information for Alpha Vantage
        
        Returns:
            Dictionary with rate limit details
        """
        return {
            'requests_per_minute': 5,
            'requests_per_day': 500,  # Free tier limit
            'delay_between_requests': self.rate_limit_delay,
            'delay_configured_from_env': os.getenv('ALPHA_VANTAGE_DELAY', '3'),
            'api_tier': 'free' if 'demo' in self.api_key.lower() else 'premium'
        }


def register_alpha_vantage_source(api_key: str = None, is_default: bool = True):
    """
    Register Alpha Vantage source with the global registry
    
    Args:
        api_key: Alpha Vantage API key
        is_default: Whether to set as default source
    """
    source = AlphaVantageTranscriptSource(api_key)
    transcript_registry.register_source('alpha_vantage', source, is_default)
    return source


# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv('../.env')  # Load from parent directory
except ImportError:
    pass

# Auto-register if API key is available
if os.getenv('ALPHA_VANTAGE_API_KEY'):
    try:
        register_alpha_vantage_source(is_default=True)
    except DataSourceError:
        pass  # Skip auto-registration if API key is invalid