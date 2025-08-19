"""
Data models for the NewsRAG package
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class NewsCategory(Enum):
    """Supported news categories"""
    EARNINGS = "earnings"
    ACQUISITIONS = "acquisitions" 
    PARTNERSHIPS = "partnerships"
    PRODUCTS = "products"
    LEADERSHIP = "leadership"
    REGULATORY = "regulatory"
    MARKET = "market"
    FINANCIAL = "financial"
    BUSINESS = "business"
    GENERAL = "general"


@dataclass
class NewsQuery:
    """Query parameters for news retrieval"""
    companies: List[str]
    categories: List[NewsCategory] = field(default_factory=lambda: [NewsCategory.GENERAL])
    days_back: int = 30
    max_articles_per_query: int = 10
    include_similar: bool = True
    
    def __post_init__(self):
        # Ensure companies are properly formatted
        self.companies = [company.strip().upper() for company in self.companies]
        
        # Convert string categories to enum if needed
        if self.categories and isinstance(self.categories[0], str):
            self.categories = [NewsCategory(cat.lower()) for cat in self.categories]


@dataclass
class NewsArticle:
    """Represents a news article with metadata"""
    title: str
    content: str
    url: str
    published_date: datetime
    author: Optional[str] = None
    source: Optional[str] = None
    summary: Optional[str] = None
    companies_mentioned: List[str] = field(default_factory=list)
    category: Optional[NewsCategory] = None
    
    # Tavily-specific metadata
    tavily_id: Optional[str] = None
    tavily_score: Optional[float] = None
    
    # Processing metadata
    word_count: int = 0
    retrieval_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if self.content:
            self.word_count = len(self.content.split())


@dataclass
class NewsChunk:
    """Represents a chunk of news content for RAG ingestion"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    
    # Source article reference
    source_article_id: str
    source_url: str
    
    # Chunk-specific info
    chunk_index: int
    total_chunks: int
    word_count: int
    char_count: int
    
    # RAG-ready fields
    embedding_ready: bool = False
    processed_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if self.content:
            self.word_count = len(self.content.split())
            self.char_count = len(self.content)


@dataclass
class NewsProcessingResult:
    """Result of news processing operation"""
    success: bool
    query: NewsQuery
    articles_retrieved: int
    chunks_generated: int
    chunks: List[NewsChunk] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    
    # Summary statistics
    total_word_count: int = 0
    companies_covered: List[str] = field(default_factory=list)
    date_range: Optional[Dict[str, datetime]] = None
    
    def add_chunk(self, chunk: NewsChunk):
        """Add a chunk to the result"""
        self.chunks.append(chunk)
        self.chunks_generated = len(self.chunks)
        self.total_word_count += chunk.word_count
    
    def add_error(self, error: str):
        """Add an error to the result"""
        self.errors.append(error)
        if self.success and self.errors:
            self.success = False