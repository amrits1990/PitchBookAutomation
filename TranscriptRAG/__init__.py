"""
TranscriptRAG - Transcript Processing and Chunking System
Independent system for processing earnings call transcripts and creating RAG-ready chunks
"""

from .get_transcript_chunks import get_transcript_chunks
from .data_source_interface import TranscriptData, TranscriptQuery, transcript_registry
from .alpha_vantage_source import register_alpha_vantage_source
from .transcript_config import get_transcript_config, validate_transcript_environment

__version__ = "1.0.0"
__author__ = "AI Assistant"

# Main API function
__all__ = [
    'get_transcript_chunks',
    'TranscriptData',
    'TranscriptQuery', 
    'transcript_registry',
    'register_alpha_vantage_source',
    'get_transcript_config',
    'validate_transcript_environment'
]