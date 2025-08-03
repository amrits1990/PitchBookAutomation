"""
Content processing module for SEC filings
Separated from monolithic SECFilingCleanerChunker for better modularity
"""

import re
import os
import io
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
import concurrent.futures
from datetime import datetime


class ContentProcessor:
    """Handles content extraction and cleaning from SEC filings"""
    
    def __init__(self):
        self.section_patterns = {
            'k10': [
                {"pattern": "ITEM 1. BUSINESS", "name": "Business", "number": "1"},
            {"pattern": "ITEM 1A. RISK FACTORS", "name": "Risk Factors", "number": "1a"},
            {"pattern": "ITEM 1B. UNRESOLVED STAFF COMMENTS", "name": "Unresolved Staff Comments", "number": "1b"},
            {"pattern": "ITEM 1C. CYBERSECURITY", "name": "Cybersecurity", "number": "1c"},
            {"pattern": "ITEM 2. PROPERTIES", "name": "Properties", "number": "2"},
            {"pattern": "ITEM 3. LEGAL PROCEEDINGS", "name": "Legal Proceedings", "number": "3"},
            {"pattern": "ITEM 4. MINE SAFETY DISCLOSURES", "name": "Mine Safety Disclosures", "number": "4"},
            {"pattern": "ITEM 5. MARKET FOR REGISTRANT'S COMMON EQUITY", "name": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities", "number": "5"},
            {"pattern": "ITEM 6. RESERVED", "name": "Reserved", "number": "6"},
            {"pattern": "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS", "name": "Management's Discussion and Analysis of Financial Condition and Results of Operations", "number": "7"},
            {"pattern": "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES", "name": "Market Risk", "number": "7a"},
            {"pattern": "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA", "name": "Financial Statements and Supplementary Data", "number": "8"},
            {"pattern": "ITEM 9. CHANGES IN AND DISAGREEMENTS", "name": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure", "number": "9"},
            {"pattern": "ITEM 9A. CONTROLS AND PROCEDURES", "name": "Controls and Procedures", "number": "9a"},
            {"pattern": "ITEM 9B. OTHER INFORMATION", "name": "Other Information", "number": "9b"},
            {"pattern": "ITEM 9C. DISCLOSURE REGARDING FOREIGN JURISDICTIONS", "name": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections", "number": "9c"},
            {"pattern": "ITEM 10. DIRECTORS, EXECUTIVE OFFICERS", "name": "Directors, Executive Officers and Corporate Governance", "number": "10"},
            {"pattern": "ITEM 11. EXECUTIVE COMPENSATION", "name": "Executive Compensation", "number": "11"},
            {"pattern": "ITEM 12. SECURITY OWNERSHIP", "name": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters", "number": "12"},
            {"pattern": "ITEM 13. CERTAIN RELATIONSHIPS", "name": "Certain Relationships and Related Transactions, and Director Independence", "number": "13"},
            {"pattern": "ITEM 14. PRINCIPAL ACCOUNTANT FEES", "name": "Principal Accountant Fees and Services", "number": "14"},
            {"pattern": "ITEM 15. EXHIBIT AND FINANCIAL STATEMENT", "name": "Exhibit and Financial Statement Schedules", "number": "15"},
            {"pattern": "ITEM 16. FORM 10-K SUMMARY", "name": "Form 10-K Summary", "number": "16"},
            ],
            'q10': [
                {"pattern": "Item 1. Financial Statements", "name": "Financial Statements", "number": "1"},
            {"pattern": "Item 2. Management’s Discussion and Analysis of Financial Condition and Results of Operations", "name": "Management's Discussion and Analysis", "number": "2"},
            {"pattern": "Item 3. Quantitative and Qualitative Disclosures About Market Risk", "name": "Market Risk", "number": "3"},
            {"pattern": "Item 4. CONTROLS AND PROCEDURES", "name": "Controls and Procedures", "number": "4"},
            {"pattern": "Item 1. Legal Proceedings", "name": "Legal Proceedings", "number": "5"},
            {"pattern": "Item 1A. Risk Factors", "name": "Risk Factors", "number": "6"},
            {"pattern": "Item 2. Unregistered Sales of Equity Securities and Use of Proceeds", "name": "Unregistered Sales of Equity Securities and Use of Proceeds", "number": "7"},
            {"pattern": "Item 3. Defaults Upon Senior Securities", "name": "Defaults Upon Senior Securities", "number": "8"},
            {"pattern": "Item 4. Mine Safety Disclosures", "name": "Mine Safety Disclosures", "number": "9"},
            {"pattern": "Item 5. Other Information", "name": "Other Information", "number": "10"},
            {"pattern": "Item 6. Exhibits", "name": "Exhibits", "number": "11"},
            ]
        }
        
        # Expected headers for cleaning extracted content
        self.k10_expected_headers = {
            "1": "ITEM 1. BUSINESS",
            "1a": "ITEM 1A. RISK FACTORS", 
            "1b": "ITEM 1B. UNRESOLVED STAFF COMMENTS",
            "1c": "ITEM 1C. CYBERSECURITY",
            "2": "ITEM 2. PROPERTIES",
            "3": "ITEM 3. LEGAL PROCEEDINGS",
            "4": "ITEM 4. MINE SAFETY DISCLOSURES",
            "5": "ITEM 5. MARKET FOR REGISTRANT'S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES",
            "6": "ITEM 6. RESERVED",
            "7": "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
            "7a": "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
            "8": "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            "9": "ITEM 9. CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURE",
            "9a": "ITEM 9A. CONTROLS AND PROCEDURES",
            "9b": "ITEM 9B. OTHER INFORMATION",
            "9c": "ITEM 9C. DISCLOSURE REGARDING FOREIGN JURISDICTIONS THAT PREVENT INSPECTIONS",
            "10": "ITEM 10. DIRECTORS, EXECUTIVE OFFICERS AND CORPORATE GOVERNANCE",
            "11": "ITEM 11. EXECUTIVE COMPENSATION",
            "12": "ITEM 12. SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT AND RELATED STOCKHOLDER MATTERS",
            "13": "ITEM 13. CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS, AND DIRECTOR INDEPENDENCE",
            "14": "ITEM 14. PRINCIPAL ACCOUNTANT FEES AND SERVICES",
            "15": "ITEM 15. EXHIBIT AND FINANCIAL STATEMENT SCHEDULES",
            "16": "ITEM 16. FORM 10-K SUMMARY"
        }
        
        self.q10_expected_headers = {
            "1": "Item 1. Financial Statements",
            "2": "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations", 
            "3": "Item 3. Quantitative and Qualitative Disclosures About Market Risk",
            "4": "Item 4. CONTROLS AND PROCEDURES",
            "5": "Item 1. Legal Proceedings",
            "6": "Item 1A. Risk Factors",
            "7": "Item 2. Unregistered Sales of Equity Securities and Use of Proceeds",
            "8": "Item 3. Defaults Upon Senior Securities",
            "9": "Item 4. Mine Safety Disclosures",
            "10": "Item 5. Other Information",
            "11": "Item 6. Exhibits",
        }
        
        self._compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        for form_type in ['k10', 'q10']:
            patterns = self.section_patterns[form_type]
            pattern_dict = {item['name']: item['pattern'] for item in patterns}
            self._compiled_patterns[form_type] = {
                name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for name, pattern in pattern_dict.items()
            }
    
    def get_section_patterns(self, form_type: str):
        """Get section patterns based on form type"""
        if form_type and '10-K' in form_type.upper():
            return self.section_patterns['k10']
        elif form_type and '10-Q' in form_type.upper():
            return self.section_patterns['q10']
        else:
            return self.section_patterns['k10']
    
    def extract_document_content(self, content: str) -> str:
        """Extract main document content from SEC filing"""
        # Handle XBRL inline documents
        if 'ix:' in content or 'xbrli:' in content:
            # Find body content
            body_start = content.find('<body')
            if body_start != -1:
                content = content[body_start:]
            
            # Remove hidden XBRL metadata sections
            content = re.sub(r'<div[^>]*style="display:\s*none"[^>]*>.*?</div>', '', content, flags=re.DOTALL | re.IGNORECASE)
            return content
        
        # Handle traditional SEC filing format
        doc_start = content.find('<DOCUMENT>')
        if doc_start != -1:
            doc_end = content.find('</DOCUMENT>')
            if doc_end != -1:
                document_content = content[doc_start:doc_end]
            else:
                document_content = content[doc_start:]
        else:
            document_content = content
        
        # Find HTML content
        html_start = document_content.find('<html')
        if html_start == -1:
            html_start = document_content.find('<HTML')
        
        if html_start != -1:
            return document_content[html_start:]
        
        return document_content
    
    def clean_html_content(self, html_content: str) -> BeautifulSoup:
        """Clean HTML content and return BeautifulSoup object"""
        if not html_content:
            return None
        
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except:
            soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        unwanted_tags = ['script', 'style', 'meta', 'link', 'head', 'title', 'noscript']
        for tag_name in unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # Remove XBRL namespaced tags
        xbrl_tags = soup.find_all(lambda tag: tag.name and ':' in tag.name)
        for tag in xbrl_tags:
            tag.unwrap()
        
        return soup
    
    def extract_clean_text(self, soup: BeautifulSoup) -> str:
        """Extract clean, readable text from BeautifulSoup object"""
        if not soup:
            return ""
        
        text = soup.get_text(separator='\n')
        
        # Clean up the text
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove lines that are mostly non-alphanumeric
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            if len(line) < 3:
                continue
            
            # Keep lines with substantial content
            alpha_num_count = sum(c.isalnum() for c in line)
            if len(line) > 0 and alpha_num_count / len(line) > 0.4:
                if not re.search(r'(.)\1{10,}', line):
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_tables_as_structured_data(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract tables as structured data using parallel processing"""
        tables_data = []
        tables = soup.find_all('table')
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_table = {
                executor.submit(self._process_single_table, i, table): i 
                for i, table in enumerate(tables)
            }
            
            for future in concurrent.futures.as_completed(future_to_table):
                table_index = future_to_table[future]
                try:
                    table_data = future.result()
                    if table_data:
                        tables_data.append(table_data)
                        
                        # Replace table with placeholder
                        placeholder = soup.new_tag('div')
                        placeholder.string = f"[TABLE_{table_index+1}_PLACEHOLDER]"
                        tables[table_index].replace_with(placeholder)
                        
                except Exception as e:
                    print(f"Error processing table {table_index}: {e}")
        
        return tables_data
    
    def _process_single_table(self, table_index: int, table) -> Optional[Dict]:
        """Process a single table for parallel execution"""
        try:
            # Quick size check
            table_str = str(table)
            if len(table_str) > 100000:  # Skip very large tables
                return None
            
            # Try pandas first
            df = pd.read_html(io.StringIO(table_str))[0]
            
            # Limit size
            if len(df) > 1000:
                df = df.head(1000)
            
            return {
                'table_id': f'table_{table_index+1}',
                'type': 'table',
                'data': df.to_dict('records'),
                'columns': df.columns.tolist(),
                'raw_text': df.to_string(index=False)
            }
            
        except Exception:
            # Fallback to manual extraction
            rows = table.find_all('tr')
            if len(rows) > 1000:
                return None
                
            table_data = []
            for row in rows[:100]:
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if any(cell.strip() for cell in row_data):
                    table_data.append(row_data)
            
            if table_data:
                return {
                    'table_id': f'table_{table_index+1}',
                    'type': 'table',
                    'data': table_data,
                    'raw_text': '\n'.join([' | '.join(row) for row in table_data])
                }
        
        return None
    
    def find_all_pattern_positions(self, text: str, pattern: str) -> List[int]:
        """Find all positions where pattern appears"""
        positions = []
        start = 0
        pattern_lower = pattern.lower()
        text_lower = text.lower()
        
        while True:
            pos = text_lower.find(pattern_lower, start)
            if pos == -1:
                pos = text_lower.find(pattern_lower + '.', start)
                if pos == -1:
                    break
            positions.append(pos)
            start = pos + 1
        
        return positions
    
    def find_next_item_boundary(self, content: str, start_pos: int, all_patterns: List[str]) -> int:
        """Find the position of the next item boundary"""
        content_lower = content.lower()
        min_pos = len(content)
        
        for pattern in all_patterns:
            pattern_lower = pattern.lower()
            pos = content_lower.find(pattern_lower, start_pos + 1)
            if pos != -1 and pos < min_pos:
                min_pos = pos
        
        return min_pos if min_pos < len(content) else len(content)
    
    def extract_clean_content(self, full_text: str, item_pattern: str, expected_headers: Dict) -> Tuple[str, str]:
        """Extract content and create clean header"""
        lines = full_text.split('\n')
        
        if lines:
            first_line = lines[0]
            
            # Create regex pattern for case-insensitive matching
            pattern_escaped = re.escape(item_pattern)
            item_regex = r'^(' + pattern_escaped + r'.*?)(\s+[A-Z][a-z].*)'
            
            match = re.match(item_regex, first_line, re.IGNORECASE)
            
            if match:
                header_line = match.group(1).strip()
                content_start = match.group(2).strip()
                remaining_lines = lines[1:]
                item_content = content_start + '\n' + '\n'.join(remaining_lines) if remaining_lines else content_start
                item_content = item_content.strip()
            else:
                # Find matching header number
                item_num = None
                for num, expected_header in expected_headers.items():
                    if expected_header.lower() in item_pattern.lower():
                        item_num = num
                        break
                
                header_line = expected_headers.get(item_num, item_pattern)
                
                # Remove the pattern from content (case-insensitive)
                pattern_lower = item_pattern.lower()
                full_text_lower = full_text.lower()
                pattern_pos = full_text_lower.find(pattern_lower)
                
                if pattern_pos != -1:
                    item_content = full_text[pattern_pos + len(item_pattern):].strip()
                else:
                    item_content = full_text.strip()
        else:
            header_line = item_pattern
            item_content = ""
        
        return header_line, item_content
    
    def extract_section_content(self, file_path: str, section_name: str, filing_metadata: Dict = None) -> Dict:
        """Extract specific section with both text and tables, including comprehensive metadata"""
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract document content
        html_content = self.extract_document_content(content)
        
        # Clean HTML
        soup = self.clean_html_content(html_content)
        
        if not soup:
            return {'text': '', 'metadata': filing_metadata or {}}
        
        # Get clean text
        clean_text = self.extract_clean_text(soup)

        # Clean up the text and prepare for section extraction
        clean_text = clean_text.replace("'", "'")
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'(?i)(?<!\n)(item\s*\d+[a-z]?\s*[\.\:])', r'\n\1', clean_text)
        
        # Get appropriate section patterns
        items_config = self.get_section_patterns(filing_metadata.get('form_type') if filing_metadata else None)
        all_patterns = [item["pattern"] for item in items_config]
        
        # Find the current item configuration based on section_name
        current_item = None
        for item in items_config:
            if item['name'] == section_name:
                current_item = item
                break
        
        if current_item is None:
            # Section not found
            section_metadata = (filing_metadata or {}).copy()
            section_metadata.update({
                'section_found': False,
                'section_name': section_name,
                'text_length': 0,
                'extraction_timestamp': datetime.now().isoformat()
            })
            return {
                'text': '',
                'metadata': section_metadata
            }
        
        # Find ALL positions where this pattern appears (case-insensitive)
        pattern_positions = self.find_all_pattern_positions(clean_text, current_item["pattern"])
        
        if pattern_positions:
            all_content_parts = []
            header_line = None
            
            for i, start_pos in enumerate(pattern_positions):
                # Find the end position (next item boundary)
                end_pos = self.find_next_item_boundary(clean_text, start_pos, all_patterns)
                
                # Extract content for this occurrence
                full_content = clean_text[start_pos:end_pos].strip()
                
                if full_content:
                    # Extract clean content and header
                    expected_headers = (self.k10_expected_headers if filing_metadata and filing_metadata.get('form_type') and '10-K' in filing_metadata['form_type'].upper() 
                                      else self.q10_expected_headers)
                    extracted_header, item_content = self.extract_clean_content(full_content, current_item["pattern"], expected_headers)
                    
                    if header_line is None:  # Use header from first occurrence
                        header_line = extracted_header
                    
                    if item_content.strip():  # Only add non-empty content
                        all_content_parts.append(item_content.strip())
            
            # Concatenate all content parts
            final_content = '\n\n--- SECTION BREAK ---\n\n'.join(all_content_parts)
            
            section_metadata = (filing_metadata or {}).copy()
            section_metadata.update({
                'section_found': True,
                'section_name': current_item["name"],
                'text_length': len(final_content),
                'extraction_timestamp': datetime.now().isoformat(),
                'occurrences_found': len(pattern_positions)
            })
            section_text = final_content
            
            print(f"✓ Extracted {current_item['name']}: {len(final_content)} characters ({len(pattern_positions)} occurrences)")
        else:
            print(f"✗ Could not find {current_item['name']}")
            section_metadata = (filing_metadata or {}).copy()
            section_metadata.update({
                'section_found': False,
                'section_name': current_item["name"],
                'text_length': 0,
                'extraction_timestamp': datetime.now().isoformat()
            })
            section_text = ""
    
        return {
            'text': section_text,
            'metadata': section_metadata
        }
    
    def extract_unclassified_content(self, file_path: str, filing_metadata: Dict = None) -> Dict:
        """Extract content that doesn't match any section patterns"""
        # Read the file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract document content
        html_content = self.extract_document_content(content)
        
        # Clean HTML
        soup = self.clean_html_content(html_content)
        
        if not soup:
            return {'text': '', 'metadata': filing_metadata or {}}
        
        # Get clean text
        clean_text = self.extract_clean_text(soup)
        
        # Clean up the text and prepare for section extraction
        clean_text = clean_text.replace("'", "'")
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = re.sub(r'(?i)(?<!\n)(item\s*\d+[a-z]?\s*[\.\:])', r'\n\1', clean_text)
        
        items_config = self.get_section_patterns(filing_metadata.get('form_type') if filing_metadata else None)
        all_patterns = [item["pattern"] for item in items_config]
        
        # Find all section boundaries in the document
        section_boundaries = []
        
        for item in items_config:
            pattern_positions = self.find_all_pattern_positions(clean_text, item["pattern"])
            for pos in pattern_positions:
                end_pos = self.find_next_item_boundary(clean_text, pos, all_patterns)
                section_boundaries.append((pos, end_pos))
        
        # Sort boundaries by start position
        section_boundaries.sort(key=lambda x: x[0])
        
        # Extract unclassified text segments
        unclassified_segments = []
        current_pos = 0
        
        for start_pos, end_pos in section_boundaries:
            # Add text before this section if it exists
            if current_pos < start_pos:
                unclassified_text = clean_text[current_pos:start_pos].strip()
                if len(unclassified_text) > 100:  # Only include substantial content
                    unclassified_segments.append(unclassified_text)
            
            # Move current position to end of this section
            current_pos = end_pos
        
        # Add any remaining text after the last section
        if current_pos < len(clean_text):
            remaining_text = clean_text[current_pos:].strip()
            if len(remaining_text) > 100:  # Only include substantial content
                unclassified_segments.append(remaining_text)
        
        # Combine all unclassified segments
        final_unclassified_content = '\n\n--- UNCLASSIFIED SECTION BREAK ---\n\n'.join(unclassified_segments)
        
        # Create metadata for unclassified content
        section_metadata = (filing_metadata or {}).copy()
        section_metadata.update({
            'section_found': len(unclassified_segments) > 0,
            'section_name': 'Unclassified Content',
            'text_length': len(final_unclassified_content),
            'unclassified_segments_count': len(unclassified_segments),
            'extraction_timestamp': datetime.now().isoformat()
        })
        
        print(f"✓ Extracted Unclassified Content: {len(final_unclassified_content)} characters ({len(unclassified_segments)} segments)")
        
        return {
            'text': final_unclassified_content,
            'metadata': section_metadata
        }