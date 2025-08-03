"""
Chunk generation module for SEC filings
Separated from monolithic SECFilingCleanerChunker for better modularity
"""

import re
from typing import Dict, List
from datetime import datetime


class ChunkGenerator:
    """Handles text chunking for RAG pipeline"""
    
    def create_rag_chunks(self, section_text: str, section_metadata: Dict, 
                         chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
        """Create chunks suitable for RAG pipeline with improved overlap handling"""
        if not section_text:
            return []
        
        # Clean and normalize text
        section_text = re.sub(r'\s+', ' ', section_text.strip())
        
        # Better sentence splitting for SEC filings
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, section_text)
        
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
                chunk_metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_length': current_length,
                    'chunk_size_setting': chunk_size,
                    'overlap_setting': overlap,
                    'chunk_created_at': timestamp,
                    'section_name': base_metadata.get('section_name', 'Unknown Section')
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
                    
                    # Find good breaking point
                    space_idx = overlap_text.find(' ')
                    if space_idx > 0:
                        overlap_text = overlap_text[space_idx:].strip()
                    
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
            chunk_metadata.update({
                'chunk_id': chunk_id,
                'chunk_length': len(current_chunk),
                'chunk_size_setting': chunk_size,
                'overlap_setting': overlap,
                'chunk_created_at': timestamp,
                'section_name': base_metadata.get('section_name', 'Unknown Section')
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
    
    def _deduplicate_chunks(self, chunks: List[Dict], chunk_size: int) -> List[Dict]:
        """Remove near-duplicate chunks and validate sizes"""
        cleaned_chunks = []
        seen_content = set()
        
        for chunk in chunks:
            # Create signature for deduplication
            content_signature = re.sub(r'\s+', ' ', chunk['text'][:100]).strip().lower()
            
            # Skip if very similar content seen
            if content_signature not in seen_content:
                seen_content.add(content_signature)
                
                # Ensure chunk isn't too small or too large
                if 20 <= chunk['length'] <= chunk_size * 1.2:
                    cleaned_chunks.append(chunk)
        
        return cleaned_chunks
    
    def create_adaptive_chunks(self, text: str, metadata: Dict, 
                             min_chunk_size: int = 400, max_chunk_size: int = 1200,
                             target_chunk_size: int = 800) -> List[Dict]:
        """Create chunks with adaptive sizing based on content structure"""
        if not text:
            return []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # If paragraph alone exceeds max_chunk_size, split it
            if paragraph_length > max_chunk_size:
                # Finish current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        chunk_id, current_chunk, current_length, metadata
                    ))
                    chunk_id += 1
                    current_chunk = ""
                    current_length = 0
                
                # Split long paragraph
                sub_chunks = self._split_long_paragraph(paragraph, max_chunk_size)
                for sub_chunk in sub_chunks:
                    chunks.append(self._create_chunk_dict(
                        chunk_id, sub_chunk, len(sub_chunk), metadata
                    ))
                    chunk_id += 1
                
            # If adding paragraph would exceed target size
            elif current_length + paragraph_length > target_chunk_size:
                # If current chunk is too small, try to add part of paragraph
                if current_length < min_chunk_size:
                    # Take what we can from the paragraph
                    available_space = target_chunk_size - current_length - 1
                    if available_space > 50:  # Only if meaningful space
                        partial = paragraph[:available_space]
                        current_chunk += "\n\n" + partial
                        paragraph = paragraph[available_space:].strip()
                
                # Finish current chunk
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        chunk_id, current_chunk, len(current_chunk), metadata
                    ))
                    chunk_id += 1
                
                # Start new chunk with remaining paragraph
                current_chunk = paragraph
                current_length = len(paragraph)
            
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                    current_length += paragraph_length + 2
                else:
                    current_chunk = paragraph
                    current_length = paragraph_length
        
        # Handle final chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk_dict(
                chunk_id, current_chunk, len(current_chunk), metadata
            ))
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str, max_size: int) -> List[str]:
        """Split a long paragraph into smaller chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_chunk_dict(self, chunk_id: int, text: str, length: int, 
                          base_metadata: Dict) -> Dict:
        """Create a standardized chunk dictionary"""
        chunk_metadata = base_metadata.copy()
        chunk_metadata.update({
            'chunk_id': chunk_id,
            'chunk_length': length,
            'chunk_created_at': datetime.now().isoformat(),
            'section_name': base_metadata.get('section_name', 'Unknown Section')
        })
        
        return {
            'chunk_id': chunk_id,
            'text': text.strip(),
            'length': length,
            'metadata': chunk_metadata
        }