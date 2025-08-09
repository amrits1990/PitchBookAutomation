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

__version__ = "2.0.0"  # Updated version for simplified architecture
__author__ = "PitchBook Generator"

# Main interface function - simplified exports
__all__ = [
    'get_company_news_chunks',
    'TavilyDirectClient',
    'NewsConfig',
    'NewsChunk',
    'NewsQuery',
    'NewsCategory'
]