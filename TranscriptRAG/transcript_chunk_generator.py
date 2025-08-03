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
                'speakers': list(current_speakers)
            })
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk),
                'metadata': chunk_metadata
            })
        
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
                title = available_speakers_titles.get(speaker, '')
                
                speakers_info.append({
                    'speaker': speaker,
                    'title': title,
                    'full_name': f"{speaker} ({title})" if title else speaker
                })
                speakers_with_titles[speaker] = title
                primary_speaker = speaker
        
        return {
            'speakers_info': speakers_info,
            'speakers_with_titles': speakers_with_titles,
            'primary_speaker': primary_speaker
        }