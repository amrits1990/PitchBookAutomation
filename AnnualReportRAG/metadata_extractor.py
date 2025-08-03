"""
Metadata extraction module for SEC filings
Separated from monolithic SECFilingCleanerChunker for better modularity
"""

import re
import os
from typing import Dict, List, Optional
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
            'central_index_key': re.compile(r'CENTRAL INDEX KEY:\s*([^\n\r]+)', re.IGNORECASE),
            'form_type': re.compile(r'FORM TYPE:\s*([^\n\r]+)', re.IGNORECASE),
            'filing_date': re.compile(r'FILED AS OF DATE:\s*(\d{8})', re.IGNORECASE),
            'period_end_date': re.compile(r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', re.IGNORECASE),
            'fiscal_year_end': re.compile(r'FISCAL YEAR END:\s*(\d{4})', re.IGNORECASE),
            'sic_code': re.compile(r'STANDARD INDUSTRIAL CLASSIFICATION:[^\[]*\[(\d{4})\]', re.IGNORECASE),
            'state_incorporation': re.compile(r'STATE OF INCORPORATION:\s*([^\n\r]+)', re.IGNORECASE),
            'irs_number': re.compile(r'IRS NUMBER:\s*([^\n\r]+)', re.IGNORECASE),
        }
    
    def extract_filing_metadata(self, file_path: str, content: str) -> Dict:
        """Extract comprehensive metadata from SEC filing header"""
        metadata = {
            'company_name': None,
            'ticker': None,
            'central_index_key': None,
            'form_type': None,
            'filing_date': None,
            'period_end_date': None,
            'fiscal_year_end': None,
            'fiscal_year': None,
            'fiscal_quarter': None,
            'sic_code': None,
            'state_incorporation': None,
            'irs_number': None
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
                self._calculate_fiscal_info(metadata)
            
            # Search body content for missing fields if needed
            missing_fields = []
            if not metadata['sic_code']:
                missing_fields.append('sic_code')
            if not metadata['state_incorporation']:
                missing_fields.append('state_incorporation')
            
            if missing_fields:
                self._search_body_content(file_path, metadata, missing_fields)
                
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
    
    def _calculate_fiscal_info(self, metadata: Dict) -> None:
        """Calculate fiscal year and quarter information"""
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
                
                # Calculate quarter for 10-Q
                if metadata.get('form_type') and '10-Q' in metadata['form_type']:
                    if fiscal_month == 12 and fiscal_day == 31:
                        # Calendar year quarters
                        quarter_map = {1: 'Q1', 2: 'Q1', 3: 'Q1', 
                                    4: 'Q2', 5: 'Q2', 6: 'Q2',
                                    7: 'Q3', 8: 'Q3', 9: 'Q3', 
                                    10: 'Q4', 11: 'Q4', 12: 'Q4'}
                        metadata['fiscal_quarter'] = quarter_map.get(period_date.month, 'Q1')
                    else:
                        # Fiscal year quarters
                        try:
                            if fiscal_month < 12:
                                fy_start_month = fiscal_month + 1
                                fy_start_year = fiscal_year - 1
                            else:
                                fy_start_month = 1
                                fy_start_year = fiscal_year - 1
                            
                            fy_start = date(fy_start_year, fy_start_month, 1)
                            days_into_fy = (period_date - fy_start).days
                            
                            if days_into_fy <= 90:
                                metadata['fiscal_quarter'] = 'Q1'
                            elif days_into_fy <= 180:
                                metadata['fiscal_quarter'] = 'Q2'
                            elif days_into_fy <= 270:
                                metadata['fiscal_quarter'] = 'Q3'
                            else:
                                metadata['fiscal_quarter'] = 'Q4'
                        except:
                            metadata['fiscal_quarter'] = 'Q1'
                                
        except Exception as e:
            print(f"Error calculating fiscal year/quarter: {e}")
    
    def _search_body_content(self, file_path: str, metadata: Dict, missing_fields: List[str]) -> None:
        """Search body content for missing metadata fields"""
        if not missing_fields:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore', buffering=8192) as f:
                f.seek(100000)  # Skip header
                body_content = f.read(400000)  # Read next 400KB
            
            # Pre-compiled body search patterns
            body_search_patterns = {
                'sic_code': [
                    re.compile(r'SIC(?:\s+Code)?[:\s]+(\d{4})', re.IGNORECASE),
                    re.compile(r'Standard Industrial Classification[:\s]+[^(\d]*(\d{4})', re.IGNORECASE),
                    re.compile(r'Industry[:\s]+[^\[]*\[(\d{4})\]', re.IGNORECASE)
                ],
                'state_incorporation': [
                    re.compile(r'(Delaware)\s+corporation', re.IGNORECASE),
                    re.compile(r'(California)\s+corporation', re.IGNORECASE),
                    re.compile(r'(Nevada)\s+corporation', re.IGNORECASE),
                    re.compile(r'incorporated?\s+(?:in\s+)?(?:the\s+)?(?:state\s+of\s+)?([A-Z][a-z]+)', re.IGNORECASE),
                ]
            }
            
            state_mapping = {
                'Delaware': 'DE', 'California': 'CA', 'Nevada': 'NV',
                'New York': 'NY', 'Texas': 'TX', 'Florida': 'FL',
                'Illinois': 'IL', 'Pennsylvania': 'PA', 'Ohio': 'OH',
                'Georgia': 'GA', 'North Carolina': 'NC', 'Michigan': 'MI'
            }
            
            # Search for missing fields
            if 'sic_code' in missing_fields and not metadata['sic_code']:
                for pattern in body_search_patterns['sic_code']:
                    match = pattern.search(body_content)
                    if match:
                        metadata['sic_code'] = match.group(1)
                        break
            
            if 'state_incorporation' in missing_fields and not metadata['state_incorporation']:
                for pattern in body_search_patterns['state_incorporation']:
                    match = pattern.search(body_content)
                    if match:
                        state_name = match.group(1).title()
                        metadata['state_incorporation'] = state_mapping.get(state_name, state_name[:2].upper())
                        break
                        
        except Exception as e:
            print(f"Error searching body content: {e}")