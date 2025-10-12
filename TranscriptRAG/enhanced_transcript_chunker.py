"""
Enhanced Transcript Chunker - Leverages Alpha Vantage Natural Structure
Replaces the old fixed-size chunking with speaker-aware semantic chunking
while preserving all metadata required for search filtering and caching.
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class EnhancedTranscriptChunker:
    """Enhanced chunking that leverages Alpha Vantage's natural structure"""
    
    def __init__(self, max_words_per_chunk: int = 500):
        """
        Initialize enhanced chunker.
        
        Args:
            max_words_per_chunk: Maximum words per chunk for speaker segments
        """
        self.max_words_per_chunk = max_words_per_chunk
        
        # Q&A patterns for detecting question-answer pairs
        self.qa_patterns = [
            r'(?i)^(question|q|query|asking)[:.]?\s*(.+)$',
            r'(?i)^(answer|a|response)[:.]?\s*(.+)$',
            r'(?i)^(operator)[:.]?\s*(.+)$',
            r'(?i)thanks?\s*.*\s*(question|ask)',
            r'(?i)(question|ask).*about',
        ]
        
        # Section identification patterns
        self.section_patterns = {
            'opening_remarks': [
                r'(?i)(opening|management)\s*remark',
                r'(?i)prepared\s*statement',
                r'(?i)financial\s*result',
                r'(?i)business\s*update',
            ],
            'qa_session': [
                r'(?i)q\s*&?\s*a\s*session',
                r'(?i)question.*answer',
                r'(?i)analyst.*question',
                r'(?i)question.*time',
            ],
            'closing_remarks': [
                r'(?i)closing\s*remark',
                r'(?i)concluding\s*statement',
                r'(?i)thank.*you.*joining',
            ]
        }
    
    def create_enhanced_chunks(self, alpha_vantage_transcript: List[Dict], 
                             base_metadata: Dict) -> List[Dict]:
        """
        Create enhanced chunks from Alpha Vantage transcript data.
        
        Args:
            alpha_vantage_transcript: List of Alpha Vantage transcript entries
            base_metadata: Base metadata to include in all chunks
            
        Returns:
            List of enhanced chunks with preserved metadata
        """
        chunks = []
        chunk_sequence = 0
        
        # Group entries by section type
        grouped_entries = self._group_by_sections(alpha_vantage_transcript)
        
        for section_type, entries in grouped_entries.items():
            if section_type == 'qa_session':
                # Handle Q&A pairs
                qa_chunks = self._create_qa_chunks(entries, base_metadata, chunk_sequence)
                chunks.extend(qa_chunks)
                chunk_sequence += len(qa_chunks)
            else:
                # Handle opening/closing remarks by speaker
                speaker_chunks = self._create_speaker_chunks(
                    entries, section_type, base_metadata, chunk_sequence
                )
                chunks.extend(speaker_chunks)
                chunk_sequence += len(speaker_chunks)
        
        return chunks
    
    def _group_by_sections(self, transcript_entries: List[Dict]) -> Dict[str, List[Dict]]:
        """Group transcript entries by section type."""
        grouped = {
            'opening_remarks': [],
            'qa_session': [], 
            'closing_remarks': [],
            'other': []
        }
        
        current_section = 'opening_remarks'  # Start with opening remarks
        
        for entry in transcript_entries:
            content = entry.get('content', '').lower()
            speaker = entry.get('speaker', '').lower()
            
            # Detect Q&A session start
            if (self._contains_pattern(content, self.section_patterns['qa_session']) or
                'operator' in speaker or
                self._looks_like_question(content)):
                current_section = 'qa_session'
            
            # Detect closing remarks
            elif self._contains_pattern(content, self.section_patterns['closing_remarks']):
                current_section = 'closing_remarks'
            
            grouped[current_section].append(entry)
        
        return grouped
    
    def _create_speaker_chunks(self, entries: List[Dict], section_type: str,
                             base_metadata: Dict, start_sequence: int) -> List[Dict]:
        """Create chunks for speaker segments (opening/closing remarks)."""
        chunks = []
        current_speaker_content = []
        current_speaker = None
        current_speaker_title = None
        
        for entry in entries:
            speaker = entry.get('speaker', 'Unknown')
            title = entry.get('title', '')
            content = entry.get('content', '')
            
            # If speaker changed, create chunk for previous speaker
            if current_speaker and speaker != current_speaker:
                if current_speaker_content:
                    chunk = self._create_speaker_chunk(
                        current_speaker_content,
                        current_speaker,
                        current_speaker_title,
                        section_type,
                        base_metadata,
                        start_sequence + len(chunks)
                    )
                    if chunk:
                        chunks.append(chunk)
                    current_speaker_content = []
            
            # Update current speaker
            current_speaker = speaker
            current_speaker_title = title
            current_speaker_content.append(content)
        
        # Create chunk for last speaker
        if current_speaker_content and current_speaker:
            chunk = self._create_speaker_chunk(
                current_speaker_content,
                current_speaker,
                current_speaker_title,
                section_type,
                base_metadata,
                start_sequence + len(chunks)
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_speaker_chunk(self, content_list: List[str], speaker: str,
                            speaker_title: str, section_type: str,
                            base_metadata: Dict, sequence: int) -> Optional[Dict]:
        """Create a single speaker chunk."""
        full_content = ' '.join(content_list).strip()
        if not full_content or len(full_content) < 20:
            return None
        
        word_count = len(full_content.split())
        
        # If content is too long, split by complete sentences
        if word_count > self.max_words_per_chunk:
            sub_chunks = self._split_by_sentences(full_content, self.max_words_per_chunk)
            # For now, just take the first chunk (can be enhanced later)
            full_content = sub_chunks[0] if sub_chunks else full_content
            word_count = len(full_content.split())
        
        # Create preserved metadata structure
        chunk_metadata = self._create_preserved_metadata(
            base_metadata, 
            speaker, 
            speaker_title, 
            section_type,
            'speaker_segment',
            sequence,
            word_count
        )
        
        # Keep quarter in numeric format for vector store compatibility
        quarter_value = base_metadata.get('quarter')
        # Convert "Q3" to "3" if needed, keep "3" as "3"
        if quarter_value and str(quarter_value).upper().startswith('Q'):
            quarter_search_format = str(quarter_value)[1:]  # Remove 'Q' prefix
        else:
            quarter_search_format = str(quarter_value) if quarter_value else quarter_value

        return {
            'chunk_id': sequence,
            'text': full_content,
            'content': full_content,  # Keep both for compatibility
            'length': len(full_content),
            'metadata': chunk_metadata,
            # Flatten key fields for vector store compatibility
            'ticker': base_metadata.get('ticker'),
            'quarter': quarter_search_format,
            'fiscal_year': base_metadata.get('fiscal_year'),
            'transcript_date': base_metadata.get('transcript_date'),
            'transcript_type': base_metadata.get('transcript_type'),
            'section_name': section_type.replace('_', ' ').title(),
            'speaker': speaker
        }
    
    def _create_qa_chunks(self, entries: List[Dict], base_metadata: Dict,
                         start_sequence: int) -> List[Dict]:
        """Create Q&A pair chunks."""
        chunks = []
        qa_pairs = self._pair_questions_answers(entries)
        
        for i, (question_entry, answer_entries) in enumerate(qa_pairs):
            qa_content = self._format_qa_pair(question_entry, answer_entries)
            word_count = len(qa_content.split())
            
            # Get primary answering speaker
            primary_speaker = answer_entries[0].get('speaker', 'Unknown') if answer_entries else 'Unknown'
            
            # Create preserved metadata structure
            chunk_metadata = self._create_preserved_metadata(
                base_metadata,
                primary_speaker,
                answer_entries[0].get('title', '') if answer_entries else '',
                'qa_session',
                'qa_pair', 
                start_sequence + i,
                word_count
            )
            
            # Keep quarter in numeric format for vector store compatibility
            quarter_value = base_metadata.get('quarter')
            # Convert "Q3" to "3" if needed, keep "3" as "3"
            if quarter_value and str(quarter_value).upper().startswith('Q'):
                quarter_search_format = str(quarter_value)[1:]  # Remove 'Q' prefix
            else:
                quarter_search_format = str(quarter_value) if quarter_value else quarter_value

            chunk = {
                'chunk_id': start_sequence + i,
                'text': qa_content,
                'content': qa_content,
                'length': len(qa_content),
                'metadata': chunk_metadata,
                # Flatten key fields for vector store compatibility
                'ticker': base_metadata.get('ticker'),
                'quarter': quarter_search_format,
                'fiscal_year': base_metadata.get('fiscal_year'),
                'transcript_date': base_metadata.get('transcript_date'),
                'transcript_type': base_metadata.get('transcript_type'),
                'section_name': 'Q&A Session',
                'speaker': primary_speaker
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_preserved_metadata(self, base_metadata: Dict, speaker: str,
                                 speaker_title: str, section_type: str,
                                 chunk_type: str, sequence: int,
                                 word_count: int) -> Dict:
        """Create metadata that preserves all required fields for compatibility."""
        
        # Keep quarter in numeric format for vector store compatibility
        quarter_value = base_metadata.get('quarter')
        # Convert "Q3" to "3" if needed, keep "3" as "3"
        if quarter_value and str(quarter_value).upper().startswith('Q'):
            quarter_search_format = str(quarter_value)[1:]  # Remove 'Q' prefix
        else:
            quarter_search_format = str(quarter_value) if quarter_value else quarter_value
        
        # Core required fields for search/cache compatibility
        preserved_metadata = {
            # Search filter fields (MUST preserve exact format)
            'ticker': base_metadata.get('ticker'),
            'quarter': quarter_search_format,  # Store in numeric format for vector store compatibility
            'fiscal_year': base_metadata.get('fiscal_year'),  # Keep as "2025"
            'transcript_date': base_metadata.get('transcript_date'),
            'transcript_type': base_metadata.get('transcript_type', 'earnings_call'),
            'section_name': section_type.replace('_', ' ').title(),
            'speaker': speaker,
            
            # Cache compatibility fields
            'content_type': 'transcript',
            'source': base_metadata.get('source', 'alpha_vantage'),
            
            # Enhanced chunking fields (new)
            'chunk_type': chunk_type,
            'chunk_sequence': sequence,
            'section_type': section_type,
            'speaker_title': speaker_title,
            'word_count': word_count,
            'enhanced_chunking': True,
            'chunk_created_at': datetime.now().isoformat(),
            
            # Minimal processing info (reduced from old version)
            'extraction_timestamp': base_metadata.get('extraction_timestamp', datetime.now().isoformat()),
        }
        
        # Add any other required base metadata fields
        for key in ['company_name', 'speakers_info', 'speakers_with_titles']:
            if key in base_metadata:
                preserved_metadata[key] = base_metadata[key]
        
        return preserved_metadata
    
    def _pair_questions_answers(self, qa_entries: List[Dict]) -> List[Tuple[Dict, List[Dict]]]:
        """Pair questions with their corresponding answers."""
        pairs = []
        current_question = None
        current_answers = []
        
        for entry in qa_entries:
            content = entry.get('content', '')
            speaker = entry.get('speaker', '').lower()
            
            if self._looks_like_question(content) or 'operator' in speaker:
                # If we have a previous question, save the pair
                if current_question:
                    pairs.append((current_question, current_answers))
                
                # Start new question
                current_question = entry
                current_answers = []
            else:
                # This is likely an answer
                if current_question:
                    current_answers.append(entry)
        
        # Add final pair
        if current_question:
            pairs.append((current_question, current_answers))
        
        return pairs
    
    def _format_qa_pair(self, question_entry: Dict, answer_entries: List[Dict]) -> str:
        """Format a Q&A pair into readable content."""
        content_parts = []
        
        # Add question
        q_speaker = question_entry.get('speaker', 'Unknown')
        q_content = question_entry.get('content', '')
        content_parts.append(f"Q ({q_speaker}): {q_content}")
        
        # Add answers
        for answer_entry in answer_entries:
            a_speaker = answer_entry.get('speaker', 'Unknown')
            a_title = answer_entry.get('title', '')
            a_content = answer_entry.get('content', '')
            
            if a_title and a_title != a_speaker:
                speaker_label = f"{a_speaker} ({a_title})"
            else:
                speaker_label = a_speaker
            
            content_parts.append(f"A ({speaker_label}): {a_content}")
        
        return '\n\n'.join(content_parts)
    
    def _looks_like_question(self, content: str) -> bool:
        """Check if content looks like a question."""
        content_lower = content.lower().strip()
        
        # Check for question patterns
        for pattern in self.qa_patterns:
            if re.search(pattern, content_lower):
                return True
        
        # Check for question marks
        if '?' in content:
            return True
        
        # Check for common question starters
        question_starters = [
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'would', 'could',
            'should', 'can you', 'are you', 'do you', 'will you', 'have you'
        ]
        
        for starter in question_starters:
            if content_lower.startswith(starter + ' '):
                return True
        
        return False
    
    def _contains_pattern(self, text: str, patterns: List[str]) -> bool:
        """Check if text contains any of the given patterns."""
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _split_by_sentences(self, content: str, max_words: int) -> List[str]:
        """Split content by complete sentences respecting word limit."""
        sentences = re.split(r'[.!?]+', content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = len(sentence.split())
            current_words = len(current_chunk.split())
            
            if current_words + sentence_words <= max_words:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content]