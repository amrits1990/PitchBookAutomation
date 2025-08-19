"""
TranscriptRAG - Transcript Processing and Chunking System
Independent system for processing earnings call transcripts and creating RAG-ready chunks
"""

from .get_transcript_chunks import get_transcript_chunks
from .data_source_interface import TranscriptData, TranscriptQuery, transcript_registry
from .alpha_vantage_source import register_alpha_vantage_source
from .transcript_config import get_transcript_config, validate_transcript_environment
from .agent_interface import (
    get_transcript_insights_for_agent,
    get_earnings_call_summary,
)

# Import vector-enhanced interfaces as primary interfaces
try:
    from .vector_enhanced_interface import (
        index_transcripts_for_agent_vector as index_transcripts_for_agent,
        search_transcripts_for_agent_vector as search_transcripts_for_agent,
    )
except ImportError:
    # Fallback to original interfaces if vector enhancement unavailable
    from .agent_interface import (
        index_transcripts_for_agent,
        search_transcripts_for_agent,
    )

__version__ = "1.0.0"
__author__ = "AI Assistant"

# Main API function
__all__ = [
    'get_transcript_chunks',
    'get_transcript_insights_for_agent',  # Agent-friendly interface
    'get_earnings_call_summary',          # Earnings call summary
    'TranscriptData',
    'TranscriptQuery', 
    'transcript_registry',
    'register_alpha_vantage_source',
    'get_transcript_config',
    'validate_transcript_environment',
    'index_transcripts_for_agent',
    'search_transcripts_for_agent',
]