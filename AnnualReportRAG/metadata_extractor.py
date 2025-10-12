"""
Metadata extraction module for SEC filings
Separated from monolithic SECFilingCleanerChunker for better modularity
"""

import re
import os
from typing import Dict, Optional
from datetime import datetime, date
from pathlib import Path


class MetadataExtractor:
    """Handles metadata extraction from SEC filing headers"""
    
    def __init__(self):
        self.ticker_patterns = [
            r'(?i)nasdaq\s*global\s*select\s*market\s*under\s*the\s*symbol\s*["\']?([A-Z]{1,5})["\']?',
            r'(?i)nasdaq\s*under\s*the\s*symbol\s*["\']?([A-Z]{1,5})["\']?',
            r'(?i)trading\s*symbol[:\s]*["\']?([A-Z]{1,5})["\']?',
            r'(?i)ticker\s*symbol[:\s]*["\']?([A-Z]{1,5})["\']?',
            r'(?i)stock\s*symbol[:\s]*["\']?([A-Z]{1,5})["\']?'
        ]
        
        self._compiled_ticker_patterns = [
            re.compile(pattern) for pattern in self.ticker_patterns
        ]
        
        # Pre-compiled metadata patterns for performance
        self._metadata_patterns = {
            'company_name': re.compile(r'COMPANY CONFORMED NAME:\s*([^\n\r]+)', re.IGNORECASE),
            'form_type': re.compile(r'FORM TYPE:\s*([^\n\r]+)', re.IGNORECASE),
            'filing_date': re.compile(r'FILED AS OF DATE:\s*(\d{8})', re.IGNORECASE),
            'period_end_date': re.compile(r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', re.IGNORECASE),
            'fiscal_year_end': re.compile(r'FISCAL YEAR END:\s*(\d{4})', re.IGNORECASE),
        }
    
    def extract_filing_metadata(self, file_path: str, content: str) -> Dict:
        """Extract comprehensive metadata from SEC filing header"""
        metadata = {
            'company_name': None,
            'ticker': None,
            'form_type': None,
            'filing_date': None,
            'period_end_date': None,
            'fiscal_year_end': None,
            'fiscal_year': None,
            'fiscal_quarter': None,
        }
        
        try:
            
            header_content = content[:100000]
            for key, pattern in self._metadata_patterns.items():
                match = pattern.search(header_content)
                if match:
                    value = match.group(1).strip()
                    if key in ['filing_date', 'period_end_date']:
                        try:
                            metadata[key] = datetime.strptime(value, '%Y%m%d').date()
                        except:
                            metadata[key] = value
                    else:
                        metadata[key] = value
            
            # Extract ticker symbol
            ticker = self._extract_ticker_symbol(file_path, header_content)
            if ticker:
                metadata['ticker'] = ticker
            
            # Calculate fiscal info
            if metadata['period_end_date'] and metadata['fiscal_year_end']:
                self._calculate_fiscal_info(metadata, content)
            
            # Correct fiscal year using SEC Company Facts API if we have required fields
            if metadata.get('ticker') and metadata.get('filing_date') and metadata.get('form_type'):
                try:
                    from fiscal_year_corrector import FiscalYearCorrector
                    corrector = FiscalYearCorrector()
                    metadata = corrector.correct_metadata_fiscal_year(metadata)
                except Exception as e:
                    print(f"Warning: Could not correct fiscal year using Company Facts API: {e}")
            
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            
        return metadata
    
    def _extract_ticker_symbol(self, file_path: str, content: str) -> Optional[str]:
        """Extract ticker symbol using multiple methods"""
        # Method 1: From filename pattern
        filename = os.path.basename(file_path).lower()
        filename_match = re.match(r'([a-z]{2,5})-', filename)
        if filename_match:
            return filename_match.group(1).upper()
        
        # Method 2: From directory structure
        path_parts = Path(file_path).parts
        for part in reversed(path_parts):
            if 2 <= len(part) <= 5 and part.isalpha():
                return part.upper()
        
        # Method 3: Search in document content
        for pattern in self._compiled_ticker_patterns:
            match = pattern.search(content)
            if match:
                return match.group(1).upper()
        
        return None
    
    def _calculate_fiscal_info(self, metadata: Dict, full_content: str) -> None:
        """Calculate fiscal year and quarter information.
        For 10-Q filings, infer quarter by counting phrases in the document:
        - "nine months ended" => Q3
        - "six months ended" => Q2
        - otherwise => Q1
        """
        try:
            period_date = metadata['period_end_date']
            if isinstance(period_date, str):
                period_date = datetime.strptime(period_date, '%Y%m%d').date()
            
            fiscal_year_end_str = metadata['fiscal_year_end']
            if len(fiscal_year_end_str) == 4:
                fiscal_month = int(fiscal_year_end_str[:2])
                fiscal_day = int(fiscal_year_end_str[2:])
                
                # Calculate fiscal year
                if (period_date.month > fiscal_month or 
                    (period_date.month == fiscal_month and period_date.day > fiscal_day)):
                    fiscal_year = period_date.year + 1
                else:
                    fiscal_year = period_date.year
                
                metadata['fiscal_year'] = fiscal_year
                
                # Calculate quarter for 10-Q using content-based heuristics
                if metadata.get('form_type') and '10-Q' in metadata['form_type']:
                    try:
                        text = (full_content or '').lower()
                        # Count occurrences (simple heuristic)
                        nine_count = len(re.findall(r"\bnine\s+months\s+ended\b", text))
                        six_count = len(re.findall(r"\bsix\s+months\s+ended\b", text))
                        # Threshold for "a lot"; tune as needed
                        threshold = 2
                        if (nine_count >= threshold and nine_count > six_count) or (nine_count == 1 and six_count == 0):
                            metadata['fiscal_quarter'] = 'Q3'
                        elif (six_count >= threshold and six_count > nine_count) or six_count >= 1:
                            metadata['fiscal_quarter'] = 'Q2'
                        else:
                            metadata['fiscal_quarter'] = 'Q1'
                    except Exception:
                        # Fallback default for 10-Q
                        metadata['fiscal_quarter'] = 'Q1'
        except Exception as e:
            print(f"Error calculating fiscal year/quarter: {e}")