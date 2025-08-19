"""
NewsRAG Package - News Retrieval and Processing for RAG Systems

A production-ready package for retrieving and processing news articles 
using the Exa API for feeding into RAG systems.

Key Features:
- Company-specific news retrieval
- Multiple news categories support
- Chunked news content with metadata
- JSON output ready for RAG ingestion
- Efficient rate limiting and error handling
"""

from .tavily_direct import TavilyDirectClient
from .config import NewsConfig
from .models import NewsChunk, NewsQuery, NewsCategory
from .main import get_company_news_chunks
from .agent_interface import (
    get_news_for_agent,
    get_news_sentiment_analysis,
)

# Import vector-enhanced interfaces as primary interfaces
try:
    from .vector_enhanced_interface import (
        index_news_for_agent_vector as index_news_for_agent,
        search_news_for_agent_vector as search_news_for_agent,
    )
except ImportError:
    # Fallback to original interfaces if vector enhancement unavailable
    from .agent_interface import (
        index_news_for_agent,
        search_news_for_agent,
    )

__version__ = "2.0.0"  # Updated version for simplified architecture
__author__ = "PitchBook Generator"

# Main interface function - simplified exports
__all__ = [
    'get_company_news_chunks',
    'get_news_for_agent',           # Agent-friendly interface
    'get_news_sentiment_analysis',  # Sentiment-focused analysis
    'index_news_for_agent',
    'search_news_for_agent',
    'TavilyDirectClient',
    'NewsConfig',
    'NewsChunk',
    'NewsQuery',
    'NewsCategory'
]