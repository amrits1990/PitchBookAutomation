"""
Configuration settings for NewsRAG package
"""

import os
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path


@dataclass
class NewsConfig:
    """Configuration class for NewsRAG package"""
    
    # Tavily API Configuration
    tavily_api_key: str = ""
    
    # Rate limiting
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    
    # Content processing
    chunk_size: int = 1000  # characters
    chunk_overlap: int = 200  # characters
    min_chunk_size: int = 100  # minimum chunk size
    
    # News retrieval defaults
    default_days_back: int = 30
    default_max_articles: int = 10
    
    # Content filtering
    min_word_count: int = 50  # minimum words per article
    max_word_count: int = 10000  # maximum words per article
    
    # Output settings
    output_format: str = "json"  # json, csv
    include_raw_content: bool = True
    include_metadata: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "newsrag.log"
    
    # Cache settings
    enable_cache: bool = True
    cache_dir: str = "cache"
    cache_expiry_hours: int = 24
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        
        # Load .env file if available
        try:
            from dotenv import load_dotenv
            load_dotenv()  # Load from current directory
            load_dotenv("../.env")  # Load from parent directory
        except ImportError:
            pass  # dotenv not available
        
        # Load API key from environment if not provided
        if not self.tavily_api_key:
            self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
            
        if not self.tavily_api_key:
            raise ValueError("Tavily API key is required. Set TAVILY_API_KEY environment variable or pass tavily_api_key parameter.")
        
        # Validate chunk settings
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
            
        if self.min_chunk_size > self.chunk_size:
            raise ValueError("min_chunk_size cannot be larger than chunk_size")
    
    @classmethod
    def from_env(cls) -> 'NewsConfig':
        """Create configuration from environment variables"""
        return cls(
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            requests_per_minute=int(os.getenv("TAVILY_REQUESTS_PER_MINUTE", "60")),
            requests_per_hour=int(os.getenv("TAVILY_REQUESTS_PER_HOUR", "1000")),
            chunk_size=int(os.getenv("NEWS_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("NEWS_CHUNK_OVERLAP", "200")),
            log_level=os.getenv("NEWS_LOG_LEVEL", "INFO"),
            enable_cache=os.getenv("NEWS_ENABLE_CACHE", "true").lower() == "true"
        )
    
    @classmethod
    def default(cls) -> 'NewsConfig':
        """Create default configuration"""
        return cls.from_env()


# Category-specific search terms for better results
NEWS_CATEGORY_TERMS: Dict[str, List[str]] = {
    "earnings": ["earnings", "quarterly results", "financial results", "revenue", "profit"],
    "acquisitions": ["acquisition", "merger", "buyout", "takeover", "deal"],
    "partnerships": ["partnership", "collaboration", "alliance", "joint venture"],
    "products": ["product launch", "new product", "innovation", "release"],
    "leadership": ["CEO", "executive", "leadership", "management", "appointment"],
    "regulatory": ["regulation", "compliance", "SEC", "FDA", "legal"],
    "market": ["market", "stock", "shares", "trading", "analyst"],
    "financial": ["funding", "investment", "IPO", "financing", "capital"],
    "general": []  # No additional terms for general news
}

# Domain filters for higher quality news sources  
TRUSTED_NEWS_DOMAINS: List[str] = [
    "bloomberg.com",
    "cnbc.com", 
    "reuters.com",
    "forbes.com",
    "ft.com",
    "wsj.com",
    "marketwatch.com",
    "yahoo.com",
    "sec.gov",
    "businesswire.com",
    "prnewswire.com"
]