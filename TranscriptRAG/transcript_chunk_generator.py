"""
Transcript chunk generation module
Adapted from AnnualReportRAG chunk generator for transcript-specific chunking needs
"""

import re
from typing import Dict, List
from datetime import datetime


class TranscriptChunkGenerator:
    """Handles text chunking for transcript RAG pipeline"""
    
    def create_transcript_chunks(self, section_text: str, section_metadata: Dict, 
                               chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
        """
        Create chunks suitable for transcript RAG pipeline with improved overlap handling
        
        Args:
            section_text: Text content to chunk
            section_metadata: Metadata about the section
            chunk_size: Maximum size of each chunk
            overlap: Overlap between consecutive chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not section_text:
            return []
        
        # Clean and normalize text for transcript-specific processing
        section_text = self._preprocess_transcript_text(section_text)
        
        # Better sentence splitting for transcripts (handles speaker changes)
        sentences = self._split_transcript_sentences(section_text)
        
        # Filter out very short fragments
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        base_metadata = section_metadata.copy()
        timestamp = datetime.now().isoformat()
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()
            if not sentence:
                i += 1
                continue
            
            sentence_length = len(sentence)
            
            # Check if adding sentence would exceed chunk size
            if current_length + sentence_length + 1 > chunk_size and current_chunk:
                # Create chunk
                chunk_metadata = base_metadata.copy()
                
                # Extract speaker information from chunk text
                chunk_speakers_info = self._extract_speakers_from_chunk(current_chunk, base_metadata)
                
                chunk_metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_length': current_length,
                    'chunk_size_setting': chunk_size,
                    'overlap_setting': overlap,
                    'chunk_created_at': timestamp,
                    'section_name': base_metadata.get('section_name', 'Unknown Section'),
                    'content_type': 'transcript',
                    'speaker': chunk_speakers_info['primary_speaker'],  # Add speaker field for vector store
                    'chunk_speakers_info': chunk_speakers_info['speakers_info'],
                    'chunk_speakers_with_titles': chunk_speakers_info['speakers_with_titles'],
                    'primary_speaker': chunk_speakers_info['primary_speaker']
                })
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': current_length,
                    'metadata': chunk_metadata
                })
                
                # Handle overlap for next chunk
                if overlap > 0 and len(chunks) > 0:
                    overlap_text = current_chunk[-overlap:].strip()
                    
                    # Find good breaking point (prefer speaker boundaries)
                    overlap_text = self._find_speaker_boundary(overlap_text)
                    
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = sentence_length
                
                chunk_id += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                    current_length += sentence_length + 1
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            
            i += 1
        
        # Handle final chunk
        if current_chunk.strip():
            chunk_metadata = base_metadata.copy()
            
            # Extract speaker information from chunk text
            chunk_speakers_info = self._extract_speakers_from_chunk(current_chunk, base_metadata)
            
            chunk_metadata.update({
                'chunk_id': chunk_id,
                'chunk_length': len(current_chunk),
                'chunk_size_setting': chunk_size,
                'overlap_setting': overlap,
                'chunk_created_at': timestamp,
                'section_name': base_metadata.get('section_name', 'Unknown Section'),
                'content_type': 'transcript',
                'speaker': chunk_speakers_info['primary_speaker'],  # Add speaker field for vector store
                'chunk_speakers_info': chunk_speakers_info['speakers_info'],
                'chunk_speakers_with_titles': chunk_speakers_info['speakers_with_titles'],
                'primary_speaker': chunk_speakers_info['primary_speaker']
            })
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk),
                'metadata': chunk_metadata
            })
        
        # Post-process to remove near-duplicates
        cleaned_chunks = self._deduplicate_chunks(chunks, chunk_size)
        
        # Update chunk IDs
        for i, chunk in enumerate(cleaned_chunks):
            chunk['chunk_id'] = i
            chunk['metadata']['chunk_id'] = i
        
        return cleaned_chunks
    
    def _preprocess_transcript_text(self, text: str) -> str:
        """
        Preprocess transcript text for optimal chunking
        
        Args:
            text: Raw transcript text
            
        Returns:
            Preprocessed text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Ensure proper spacing after colons (speaker indicators)
        text = re.sub(r':(\S)', r': \1', text)
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Normalize speaker patterns for better chunking
        text = re.sub(r'\n\s*([A-Z][A-Za-z\s]+):\s*', r' \1: ', text)
        
        return text
    
    def _split_transcript_sentences(self, text: str) -> List[str]:
        """
        Split transcript text into sentences, respecting speaker boundaries
        
        Args:
            text: Preprocessed transcript text
            
        Returns:
            List of sentence strings
        """
        # First, split on speaker boundaries
        speaker_pattern = r'([A-Z][A-Za-z\s]+:)'
        segments = re.split(speaker_pattern, text)
        
        sentences = []
        current_speaker = None
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            # Check if this is a speaker indicator
            if re.match(r'^[A-Z][A-Za-z\s]+:$', segment):
                current_speaker = segment
            else:
                # This is content - split into sentences but keep speaker context
                content_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', segment)
                
                for i, sentence in enumerate(content_sentences):
                    sentence = sentence.strip()
                    if sentence:
                        # Add speaker to first sentence of their segment
                        if current_speaker and i == 0:
                            sentences.append(f"{current_speaker} {sentence}")
                            current_speaker = None  # Only add speaker to first sentence
                        else:
                            sentences.append(sentence)
        
        return sentences
    
    def _find_speaker_boundary(self, overlap_text: str) -> str:
        """
        Find a good breaking point at a speaker boundary within overlap text
        
        Args:
            overlap_text: Text to find boundary in
            
        Returns:
            Adjusted overlap text ending at a speaker boundary if possible
        """
        # Look for speaker patterns in reverse
        speaker_matches = list(re.finditer(r'([A-Z][A-Za-z\s]+:)', overlap_text))
        
        if speaker_matches:
            # Use the last speaker boundary found
            last_match = speaker_matches[-1]
            return overlap_text[last_match.start():]
        
        # If no speaker boundary, fall back to sentence boundary
        sentence_matches = list(re.finditer(r'[.!?]\s+', overlap_text))
        if sentence_matches:
            last_sentence = sentence_matches[-1]
            return overlap_text[last_sentence.end():]
        
        # If no good boundary found, return original
        return overlap_text
    
    def _deduplicate_chunks(self, chunks: List[Dict], chunk_size: int) -> List[Dict]:
        """
        Remove near-duplicate chunks and validate sizes
        
        Args:
            chunks: List of chunk dictionaries
            chunk_size: Maximum chunk size
            
        Returns:
            Cleaned list of chunks
        """
        cleaned_chunks = []
        seen_content = set()
        
        for chunk in chunks:
            # Create signature for deduplication (use first 100 chars)
            content_signature = re.sub(r'\s+', ' ', chunk['text'][:100]).strip().lower()
            
            # Skip if very similar content seen
            if content_signature not in seen_content:
                seen_content.add(content_signature)
                
                # Ensure chunk isn't too small or too large
                if 20 <= chunk['length'] <= chunk_size * 1.2:
                    cleaned_chunks.append(chunk)
        
        return cleaned_chunks
    
    def create_speaker_aware_chunks(self, section_text: str, section_metadata: Dict,
                                  chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
        """
        Create chunks that respect speaker boundaries as much as possible
        
        Args:
            section_text: Text content to chunk
            section_metadata: Metadata about the section
            chunk_size: Maximum size of each chunk
            overlap: Overlap between consecutive chunks
            
        Returns:
            List of chunk dictionaries optimized for speaker coherence
        """
        if not section_text:
            return []
        
        # Split by speakers first
        speaker_segments = self._extract_speaker_segments(section_text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        current_speakers = set()
        
        base_metadata = section_metadata.copy()
        timestamp = datetime.now().isoformat()
        
        for segment in speaker_segments:
            speaker = segment['speaker']
            content = segment['content']
            segment_length = len(f"{speaker}: {content}")
            
            # Check if adding this segment would exceed chunk size
            if current_length + segment_length + 1 > chunk_size and current_chunk:
                # Create chunk
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_length': current_length,
                    'chunk_size_setting': chunk_size,
                    'overlap_setting': overlap,
                    'chunk_created_at': timestamp,
                    'section_name': base_metadata.get('section_name', 'Unknown Section'),
                    'content_type': 'transcript',
                    'speaker': list(current_speakers)[0] if current_speakers else None,  # Primary speaker for vector store
                    'speakers': list(current_speakers)
                })
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': current_length,
                    'metadata': chunk_metadata
                })
                
                # Start new chunk
                current_chunk = f"{speaker}: {content}"
                current_length = segment_length
                current_speakers = {speaker}
                chunk_id += 1
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += f" {speaker}: {content}"
                    current_length += segment_length + 1
                else:
                    current_chunk = f"{speaker}: {content}"
                    current_length = segment_length
                
                current_speakers.add(speaker)
        
        # Handle final chunk
        if current_chunk.strip():
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': chunk_id,
                'chunk_length': len(current_chunk),
                'chunk_size_setting': chunk_size,
                'overlap_setting': overlap,
                'chunk_created_at': timestamp,
                'section_name': base_metadata.get('section_name', 'Unknown Section'),
                'content_type': 'transcript',
                'speaker': list(current_speakers)[0] if current_speakers else None,  # Primary speaker for vector store
                'speakers': list(current_speakers)
            })
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk),
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def create_qa_grouped_chunks(self, section_text: str, section_metadata: Dict, 
                                chunk_size: int = 1200, overlap: int = 0) -> List[Dict]:
        """
        Create Q&A chunks that group analyst questions with their answers
        
        Args:
            section_text: Q&A section text content
            section_metadata: Metadata about the Q&A section
            chunk_size: Maximum size of each chunk (larger for Q&A to fit Q+A)
            overlap: Overlap between consecutive chunks (0 for Q&A to avoid mixing)
            
        Returns:
            List of chunk dictionaries with grouped Q&A pairs
        """
        if not section_text:
            return []
        
        # Parse the section text to extract individual speaker segments
        segments = self._extract_speaker_segments(section_text)
        if not segments:
            return []
        
        # Group segments into Q&A pairs
        qa_groups = self._group_qa_segments(segments)
        
        chunks = []
        base_metadata = section_metadata.copy()
        timestamp = datetime.now().isoformat()
        
        for i, qa_group in enumerate(qa_groups):
            # Combine all segments in this Q&A group
            group_text_parts = []
            group_speakers = set()
            primary_speaker = None  # Will be the person answering (management)
            analyst_speaker = None  # Track the analyst who asked
            
            for j, segment in enumerate(qa_group):
                speaker = segment['speaker']
                content = segment['content']
                title = segment.get('title', '')
                group_speakers.add(speaker)
                group_text_parts.append(f"{speaker}: {content}")
                
                if j == 0:
                    # First segment is always the analyst asking
                    analyst_speaker = speaker
                elif primary_speaker is None and 'analyst' not in title.lower():
                    # First non-analyst speaker becomes primary (the answerer)
                    primary_speaker = speaker
            
            # If no management response found, fall back to analyst
            if primary_speaker is None:
                primary_speaker = analyst_speaker
            
            group_text = '\n'.join(group_text_parts)
            
            # Create chunk metadata
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_id': i,
                'chunk_length': len(group_text),
                'chunk_size_setting': chunk_size,
                'overlap_setting': overlap,
                'chunk_created_at': timestamp,
                'section_name': base_metadata.get('section_name', 'Q&A Session'),
                'content_type': 'qa_transcript',
                'speaker': primary_speaker,  # Person answering the question (for search filtering)
                'primary_speaker': primary_speaker,  # Management person answering
                'analyst_speaker': analyst_speaker,  # Analyst who asked the question
                'speakers': list(group_speakers),
                'qa_group': True,
                'qa_pair_count': len(qa_group)
            })
            
            chunks.append({
                'chunk_id': i,
                'text': group_text,
                'length': len(group_text),
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def _group_qa_segments(self, segments: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """
        Group segments into Q&A pairs: analyst question + all responses until next analyst
        
        Args:
            segments: List of speaker segments from Q&A section
            
        Returns:
            List of groups, where each group is [analyst_question, response1, response2, ...]
        """
        if not segments:
            return []
        
        qa_groups = []
        current_group = []
        
        for segment in segments:
            speaker = segment['speaker']
            title = segment.get('title', '')
            
            # Check if this is an analyst (new question starts)
            if 'analyst' in title.lower():
                # Save previous group if it exists
                if current_group:
                    qa_groups.append(current_group)
                
                # Start new group with analyst question
                current_group = [segment]
            else:
                # Add response to current group
                if current_group:
                    current_group.append(segment)
                else:
                    # Edge case: response without preceding analyst question
                    current_group = [segment]
        
        # Add final group
        if current_group:
            qa_groups.append(current_group)
        
        return qa_groups
    
    def create_opening_remarks_chunks(self, section_text: str, section_metadata: Dict, 
                                    chunk_size: int = 800) -> List[Dict]:
        """
        Create opening remarks chunks with no overlap between JSON entries
        Each original JSON entry becomes a separate chunk (or multiple if too large)
        
        Args:
            section_text: Opening remarks section text content
            section_metadata: Metadata about the opening remarks section
            chunk_size: Maximum size of each chunk
            
        Returns:
            List of chunk dictionaries without overlap
        """
        if not section_text:
            return []
        
        # Extract individual speaker segments (each represents an original JSON entry)
        segments = self._extract_speaker_segments(section_text)
        if not segments:
            return []
        
        chunks = []
        base_metadata = section_metadata.copy()
        timestamp = datetime.now().isoformat()
        chunk_id = 0
        
        for segment in segments:
            speaker = segment['speaker']
            content = segment['content']
            segment_text = f"{speaker}: {content}"
            
            # If segment is small enough, create single chunk
            if len(segment_text) <= chunk_size:
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_length': len(segment_text),
                    'chunk_size_setting': chunk_size,
                    'overlap_setting': 0,  # No overlap for opening remarks
                    'chunk_created_at': timestamp,
                    'section_name': base_metadata.get('section_name', 'Opening Remarks'),
                    'content_type': 'opening_remarks_transcript',
                    'speaker': speaker,
                    'speakers': [speaker],
                    'json_entry_chunk': True  # Indicates this came from single JSON entry
                })
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': segment_text,
                    'length': len(segment_text),
                    'metadata': chunk_metadata
                })
                
                chunk_id += 1
            
            else:
                # Segment is too large - split it but maintain no overlap
                # Split content by sentences
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current_chunk_content = f"{speaker}: "
                current_length = len(current_chunk_content)
                
                for sentence in sentences:
                    sentence_length = len(sentence) + 1  # +1 for space
                    
                    if current_length + sentence_length <= chunk_size:
                        current_chunk_content += sentence + " "
                        current_length += sentence_length
                    else:
                        # Create chunk with current content
                        if current_chunk_content.strip() != f"{speaker}:":
                            chunk_metadata = base_metadata.copy()
                            chunk_metadata.update({
                                'chunk_id': chunk_id,
                                'chunk_length': len(current_chunk_content.strip()),
                                'chunk_size_setting': chunk_size,
                                'overlap_setting': 0,
                                'chunk_created_at': timestamp,
                                'section_name': base_metadata.get('section_name', 'Opening Remarks'),
                                'content_type': 'opening_remarks_transcript',
                                'speaker': speaker,
                                'speakers': [speaker],
                                'json_entry_chunk': True,
                                'split_from_large_entry': True
                            })
                            
                            chunks.append({
                                'chunk_id': chunk_id,
                                'text': current_chunk_content.strip(),
                                'length': len(current_chunk_content.strip()),
                                'metadata': chunk_metadata
                            })
                            
                            chunk_id += 1
                        
                        # Start new chunk
                        current_chunk_content = f"{speaker}: {sentence} "
                        current_length = len(current_chunk_content)
                
                # Add final chunk if there's remaining content
                if current_chunk_content.strip() != f"{speaker}:":
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        'chunk_id': chunk_id,
                        'chunk_length': len(current_chunk_content.strip()),
                        'chunk_size_setting': chunk_size,
                        'overlap_setting': 0,
                        'chunk_created_at': timestamp,
                        'section_name': base_metadata.get('section_name', 'Opening Remarks'),
                        'content_type': 'opening_remarks_transcript',
                        'speaker': speaker,
                        'speakers': [speaker],
                        'json_entry_chunk': True,
                        'split_from_large_entry': True
                    })
                    
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk_content.strip(),
                        'length': len(current_chunk_content.strip()),
                        'metadata': chunk_metadata
                    })
                    
                    chunk_id += 1
        
        return chunks
    
    def _extract_speaker_segments(self, text: str) -> List[Dict[str, str]]:
        """
        Extract individual speaker segments from transcript text
        
        Args:
            text: Transcript text
            
        Returns:
            List of dictionaries with speaker and content
        """
        segments = []
        
        # Split on speaker patterns
        pattern = r'([A-Z][A-Za-z\s]+):\s*([^:]+?)(?=\s+[A-Z][A-Za-z\s]+:|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for speaker, content in matches:
            segments.append({
                'speaker': speaker.strip(),
                'content': content.strip()
            })
        
        return segments
    
    def _extract_speakers_from_chunk(self, chunk_text: str, base_metadata: Dict) -> Dict:
        """
        Extract speaker information from chunk text using patterns and base metadata
        
        Args:
            chunk_text: Text content of the chunk
            base_metadata: Base metadata including speakers_with_titles if available
            
        Returns:
            Dictionary with speaker information for the chunk
        """
        speakers_info = []
        speakers_with_titles = {}
        primary_speaker = None
        
        # Get available speaker titles from base metadata
        available_speakers_titles = base_metadata.get('speakers_with_titles', {})
        
        # Find speakers mentioned in this chunk using patterns
        speaker_patterns = [
            r'^([A-Za-z][A-Za-z\s\.]+)\s*\([^)]+\):\s*',  # "John Smith (Title): content"
            r'^([A-Z][a-zA-Z\s]+):\s*',                    # "John Smith: content"
            r'^([A-Z]+\s+[A-Z]+):\s*',                     # "JOHN SMITH: content"
            r'([A-Za-z][A-Za-z\s\.]+)\s*\([^)]+\)',       # "John Smith (Title)" anywhere in text
        ]
        
        speakers_found = set()
        
        # Search for speakers in chunk text
        for pattern in speaker_patterns:
            matches = re.findall(pattern, chunk_text, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    speaker = match[0].strip()
                else:
                    speaker = match.strip()
                
                # Clean up speaker name
                speaker = re.sub(r'\s+', ' ', speaker)
                if len(speaker) > 2 and speaker.replace(' ', '').isalpha():
                    speakers_found.add(speaker)
        
        # Build speaker information
        for speaker in speakers_found:
            title = available_speakers_titles.get(speaker, '')
            
            speaker_info = {
                'speaker': speaker,
                'title': title,
                'full_name': f"{speaker} ({title})" if title else speaker
            }
            
            speakers_info.append(speaker_info)
            speakers_with_titles[speaker] = title
            
            # Set primary speaker (first one found, or most prominent)
            if primary_speaker is None:
                primary_speaker = speaker
        
        # If no speakers found through patterns, try to infer from chunk start
        if not speakers_found:
            # Look for speaker at the very beginning of chunk
            first_line = chunk_text.split('\n')[0] if chunk_text else ''
            speaker_match = re.match(r'^([A-Za-z][A-Za-z\s\.]+?)(?:\s*\([^)]+\))?:\s*', first_line)
            if speaker_match:
                speaker = speaker_match.group(1).strip()
                # Skip generic terms that don't represent real speakers
                if speaker.upper() not in ['UNKNOWN', 'UNIDENTIFIED', 'QUESTION', 'ANSWER', 'OPERATOR']:
                    title = available_speakers_titles.get(speaker, '')
                    
                    speakers_info.append({
                        'speaker': speaker,
                        'title': title,
                        'full_name': f"{speaker} ({title})" if title else speaker
                    })
                    speakers_with_titles[speaker] = title
                    primary_speaker = speaker
            else:
                # Try additional patterns for common names
                # Look for "Tim Cook:" or "Cook:" or similar patterns in chunk
                name_patterns = [
                    r'\b([A-Z][a-z]+\s+[A-Z][a-z]+):\s+',  # "Tim Cook: "
                    r'\b([A-Z][a-z]+):\s+(?:Thank|I|We|Our)',  # "Cook: Thank"
                ]
                for pattern in name_patterns:
                    matches = re.findall(pattern, chunk_text)
                    for match in matches:
                        speaker = match.strip()
                        if len(speaker) > 1 and speaker.upper() not in ['UNKNOWN', 'UNIDENTIFIED']:
                            title = available_speakers_titles.get(speaker, '')
                            speakers_info.append({
                                'speaker': speaker,
                                'title': title,
                                'full_name': f"{speaker} ({title})" if title else speaker
                            })
                            speakers_with_titles[speaker] = title
                            if primary_speaker is None:
                                primary_speaker = speaker
                            break
                    if primary_speaker is not None:
                        break
        
        return {
            'speakers_info': speakers_info,
            'speakers_with_titles': speakers_with_titles,
            'primary_speaker': primary_speaker
        }