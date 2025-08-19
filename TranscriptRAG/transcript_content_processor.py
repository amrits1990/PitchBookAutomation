"""
Transcript content processing module
Handles cleaning, formatting, and structuring of transcript content for optimal chunking
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .data_source_interface import TranscriptData


class TranscriptContentProcessor:
    """Processes and cleans transcript content for optimal chunking"""
    
    def __init__(self):
        # Speaker patterns for identifying speakers in transcripts
        self.speaker_patterns = [
            r'^([A-Za-z][A-Za-z\s\.]+)\s*\([^)]+\):\s*(.+)$',  # "John Smith (Title): content"
            r'^([A-Z][a-zA-Z\s]+):\s*(.+)$',  # "John Smith: content"
            r'^([A-Z]+\s+[A-Z]+):\s*(.+)$',   # "JOHN SMITH: content"
            r'^\[([^\]]+)\]:\s*(.+)$',        # "[Speaker Name]: content"
            r'^(\w+\s+\w+),\s*([^:]+):\s*(.+)$'  # "John Smith, CEO: content"
        ]
        
        # Question and answer patterns
        self.qa_patterns = [
            r'(?i)^(question|q)[:.]?\s*(.+)$',
            r'(?i)^(answer|a)[:.]?\s*(.+)$',
            r'(?i)^(operator)[:.]?\s*(.+)$'
        ]
        
        # Section headers for earnings calls
        self.section_headers = [
            'Management Remarks',
            'Opening Remarks', 
            'Financial Results',
            'Business Update',
            'Q&A Session',
            'Question and Answer',
            'Closing Remarks',
            'Forward Looking Statements',
            'Safe Harbor'
        ]
    
    def process_transcript_content(self, transcript_data: TranscriptData) -> Dict[str, any]:
        """
        Process and structure transcript content
        
        Args:
            transcript_data: TranscriptData object
            
        Returns:
            Dictionary with processed content and structure
        """
        content = transcript_data.content
        
        # Check if this is Alpha Vantage data with structured raw_data
        if (transcript_data.source == "alpha_vantage" and 
            hasattr(transcript_data, 'raw_data') and 
            isinstance(transcript_data.raw_data, dict) and 
            'transcript' in transcript_data.raw_data):
            
            # Use structured Alpha Vantage data directly
            segments = self._parse_alpha_vantage_transcript(transcript_data.raw_data['transcript'])
            cleaned_content = content
            
        else:
            # Clean the content
            cleaned_content = self._clean_content(content)
            
            # Parse speakers and segments
            segments = self._parse_speakers_and_segments(cleaned_content)
        
        # Identify sections
        sections = self._identify_sections(segments)
        
        # Create structured content
        structured_content = self._create_structured_content(sections, transcript_data)
        
        return {
            'original_content': content,
            'cleaned_content': cleaned_content,
            'segments': segments,
            'sections': sections,
            'structured_content': structured_content,
            'processing_metadata': {
                'segment_count': len(segments),
                'section_count': len(sections),
                'processed_at': datetime.now().isoformat(),
                'content_length': len(cleaned_content),
                'original_length': len(content)
            }
        }
    
    def _parse_alpha_vantage_transcript(self, transcript_entries: List[Dict]) -> List[Dict[str, str]]:
        """
        Parse Alpha Vantage structured transcript data into segments
        
        Args:
            transcript_entries: List of transcript entry dictionaries
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        for entry in transcript_entries:
            speaker = entry.get('speaker', 'Unknown')
            title = entry.get('title', '')
            content = entry.get('content', '')
            sentiment = entry.get('sentiment', '0.0')
            
            # Determine segment type
            segment_type = 'statement'
            speaker_lower = speaker.lower()
            content_lower = content.lower()
            
            # Check for Q&A patterns
            if ('question' in content_lower or 'answer' in content_lower or 
                'analyst' in title.lower() or 'q&a' in content_lower):
                segment_type = 'qa'
            
            # Create segment
            segment = {
                'speaker': speaker,
                'content': content,
                'type': segment_type,
                'title': title,
                'sentiment': sentiment
            }
            
            segments.append(segment)
        
        return segments
    
    def _clean_content(self, content: str) -> str:
        """
        Clean and normalize transcript content
        
        Args:
            content: Raw transcript content
            
        Returns:
            Cleaned content string
        """
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common transcript artifacts
        content = re.sub(r'\[.*?\](?=\s|$)', '', content)  # Remove stage directions like [AUDIO CUT]
        content = re.sub(r'\(.*?\)(?=\s|$)', '', content)  # Remove parenthetical notes
        
        # Fix common formatting issues
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)  # Ensure space after sentences
        content = re.sub(r'\s*:\s*', ': ', content)  # Normalize colons
        
        # Remove excessive punctuation
        content = re.sub(r'[.]{3,}', '...', content)
        content = re.sub(r'[-]{2,}', '--', content)
        
        # Normalize quotes
        content = re.sub(r'[\u201c\u201d]', '"', content)
        content = re.sub(r'[\u2018\u2019]', "'", content)
        
        # Clean up line breaks and spacing
        content = re.sub(r'\n\s*\n+', '\n\n', content)
        content = content.strip()
        
        return content
    
    def _parse_speakers_and_segments(self, content: str) -> List[Dict[str, str]]:
        """
        Parse content to identify speakers and their statements
        
        Args:
            content: Cleaned transcript content
            
        Returns:
            List of dictionaries with speaker and content segments
        """
        segments = []
        lines = content.split('\n')
        current_speaker = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to identify speaker
            speaker_found = False
            for pattern in self.speaker_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous segment if exists
                    if current_speaker and current_content:
                        segments.append({
                            'speaker': current_speaker,
                            'content': ' '.join(current_content).strip(),
                            'type': 'statement'
                        })
                    
                    # Start new segment
                    current_speaker = match.group(1).strip()
                    current_content = [match.group(2).strip()] if len(match.groups()) > 1 else []
                    speaker_found = True
                    break
            
            # Check for Q&A patterns
            if not speaker_found:
                for pattern in self.qa_patterns:
                    match = re.match(pattern, line)
                    if match:
                        # Save previous segment
                        if current_speaker and current_content:
                            segments.append({
                                'speaker': current_speaker,
                                'content': ' '.join(current_content).strip(),
                                'type': 'statement'
                            })
                        
                        # Create Q&A segment
                        qa_type = match.group(1).lower()
                        qa_content = match.group(2).strip() if len(match.groups()) > 1 else line
                        segments.append({
                            'speaker': qa_type.upper(),
                            'content': qa_content,
                            'type': 'qa'
                        })
                        current_speaker = None
                        current_content = []
                        speaker_found = True
                        break
            
            # If no speaker identified, add to current content
            if not speaker_found:
                if current_speaker:
                    current_content.append(line)
                else:
                    # Standalone content without speaker
                    segments.append({
                        'speaker': 'UNKNOWN',
                        'content': line,
                        'type': 'statement'
                    })
        
        # Add final segment
        if current_speaker and current_content:
            segments.append({
                'speaker': current_speaker,
                'content': ' '.join(current_content).strip(),
                'type': 'statement'
            })
        
        return segments
    
    def _identify_sections(self, segments: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """
        Identify logical sections within the transcript
        
        Args:
            segments: List of speaker segments
            
        Returns:
            List of section dictionaries
        """
        sections = []
        current_section = None
        current_segments = []
        
        for i, segment in enumerate(segments):
            content = segment['content']
            
            # Check if this segment starts a new section
            section_header = self._detect_section_header(content)
            
            if section_header:
                # Save previous section
                if current_section and current_segments:
                    sections.append({
                        'title': current_section,
                        'segments': current_segments.copy(),
                        'start_index': len(sections),
                        'segment_count': len(current_segments)
                    })
                
                # Start new section
                current_section = section_header
                current_segments = [segment]
            else:
                # Add to current section
                if current_section is None:
                    current_section = "Opening Remarks"
                current_segments.append(segment)
        
        # Add final section
        if current_section and current_segments:
            sections.append({
                'title': current_section,
                'segments': current_segments.copy(),
                'start_index': len(sections),
                'segment_count': len(current_segments)
            })
        
        # If no sections identified, create a default section
        if not sections and segments:
            sections.append({
                'title': "Full Transcript",
                'segments': segments,
                'start_index': 0,
                'segment_count': len(segments)
            })
        
        return sections
    
    def _detect_section_header(self, content: str) -> Optional[str]:
        """
        Detect if content contains a section header
        
        Args:
            content: Segment content to check
            
        Returns:
            Section title if found, None otherwise
        """
        content_lower = content.lower()
        
        for header in self.section_headers:
            if header.lower() in content_lower:
                return header
        
        # Check for Q&A patterns
        if any(pattern in content_lower for pattern in ['question', 'q&a', 'questions and answers']):
            return "Q&A Session"
        
        # Check for financial results patterns
        if any(pattern in content_lower for pattern in ['financial results', 'earnings', 'revenue']):
            return "Financial Results"
        
        return None
    
    def _create_structured_content(self, sections: List[Dict], transcript_data: TranscriptData) -> str:
        """
        Create a structured, clean version of the transcript content
        
        Args:
            sections: List of identified sections
            transcript_data: Original transcript data
            
        Returns:
            Structured content string ready for chunking
        """
        structured_parts = []
        
        # Add header information
        header = f"EARNINGS CALL TRANSCRIPT - {transcript_data.company_name} ({transcript_data.ticker})"
        header += f"\nDate: {transcript_data.transcript_date.strftime('%Y-%m-%d')}"
        header += f"\nQuarter: {transcript_data.quarter} {transcript_data.fiscal_year}"
        header += f"\nType: {transcript_data.transcript_type.replace('_', ' ').title()}"
        
        if transcript_data.participants:
            header += f"\nParticipants: {', '.join(transcript_data.participants)}"
        
        structured_parts.append(header)
        structured_parts.append("=" * 80)
        
        # Add sections
        for section in sections:
            # Section header
            section_header = f"\n{section['title'].upper()}"
            structured_parts.append(section_header)
            structured_parts.append("-" * len(section_header))
            
            # Section content
            for segment in section['segments']:
                speaker = segment['speaker']
                content = segment['content']
                
                if segment['type'] == 'qa':
                    if speaker.upper() in ['Q', 'QUESTION']:
                        formatted_segment = f"\nQ: {content}"
                    elif speaker.upper() in ['A', 'ANSWER']:
                        formatted_segment = f"\nA: {content}"
                    else:
                        formatted_segment = f"\n{speaker}: {content}"
                else:
                    formatted_segment = f"\n{speaker}: {content}"
                
                structured_parts.append(formatted_segment)
        
        return '\n'.join(structured_parts)
    
    def get_section_content(self, processed_content: Dict, section_title: str) -> Optional[str]:
        """
        Extract content for a specific section
        
        Args:
            processed_content: Result from process_transcript_content
            section_title: Title of section to extract
            
        Returns:
            Section content string or None if not found
        """
        for section in processed_content.get('sections', []):
            if section['title'].lower() == section_title.lower():
                section_parts = []
                
                for segment in section['segments']:
                    speaker = segment['speaker']
                    content = segment['content']
                    section_parts.append(f"{speaker}: {content}")
                
                return '\n'.join(section_parts)
        
        return None
    
    def create_section_chunks_metadata(self, processed_content: Dict, transcript_data: TranscriptData) -> List[Dict]:
        """
        Create metadata for section-based chunking similar to SEC filings
        
        Args:
            processed_content: Result from process_transcript_content
            transcript_data: Original transcript data
            
        Returns:
            List of section metadata dictionaries
        """
        section_metadata_list = []
        
        for section in processed_content.get('sections', []):
            # Get section content
            section_content = self.get_section_content(processed_content, section['title'])
            
            if section_content:
                # Extract speaker information with titles from section segments
                speakers_info = []
                speakers_with_titles = {}
                
                for segment in section.get('segments', []):
                    speaker = segment.get('speaker', 'Unknown')
                    title = segment.get('title', '')
                    
                    if speaker not in speakers_with_titles:
                        speakers_with_titles[speaker] = title
                        speakers_info.append({
                            'speaker': speaker,
                            'title': title,
                            'full_name': f"{speaker} ({title})" if title else speaker
                        })
                
                section_metadata = {
                    'section_found': True,
                    'section_name': section['title'],
                    'text_length': len(section_content),
                    'segment_count': section['segment_count'],
                    'extraction_timestamp': datetime.now().isoformat(),
                    'ticker': transcript_data.ticker,
                    'company_name': transcript_data.company_name,
                    'transcript_date': transcript_data.transcript_date,
                    'quarter': transcript_data.quarter,
                    'fiscal_year': transcript_data.fiscal_year,
                    'transcript_type': transcript_data.transcript_type,
                    'source': transcript_data.source,
                    'speakers_info': speakers_info,
                    'speakers_with_titles': speakers_with_titles
                }
                
                section_metadata_list.append({
                    'text': section_content,
                    'metadata': section_metadata
                })
        
        return section_metadata_list