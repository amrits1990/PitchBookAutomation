"""
Main configuration for AgentSystem
Simple and clear configuration management
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Simple configuration for the Agent System"""
    
    # Vector Database Settings
    vector_db_path: str = "./data/vector_store"
    embedding_model: str = "all-mpnet-base-v2"  # Better performance for financial documents
    
    # Search Settings
    default_search_k: int = 20
    bm25_weight: float = 0.4  # Weight for BM25 in hybrid search (better for financial terms)
    semantic_weight: float = 0.6  # Weight for semantic search
    
    # Chunk Settings
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # LLM Settings
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "openai/gpt-4o-mini"
    openai_api_key: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize LLM client after dataclass creation"""
        self._llm_client = None
        self._setup_llm_client()
    
    def _setup_llm_client(self):
        """Setup LLM client based on available API keys"""
        try:
            if self.openrouter_api_key:
                # Use OpenRouter
                import openai
                self._llm_client = openai.OpenAI(
                    api_key=self.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            elif self.openai_api_key:
                # Use OpenAI directly
                import openai
                self._llm_client = openai.OpenAI(api_key=self.openai_api_key)
            else:
                self._llm_client = None
        except ImportError:
            self._llm_client = None
    
    @property
    def llm_client(self):
        """Get the LLM client"""
        return self._llm_client
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables"""
        return cls(
            vector_db_path=os.getenv("VECTOR_DB_PATH", "./data/vector_store"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2"),
            default_search_k=int(os.getenv("DEFAULT_SEARCH_K", "20")),
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.3")),
            semantic_weight=float(os.getenv("SEMANTIC_WEIGHT", "0.7")),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv("OPENROUTER_REFINEMENT_MODEL", "openai/gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    def get_vector_db_path(self) -> Path:
        """Get vector database path as Path object"""
        return Path(self.vector_db_path)
    
    def ensure_directories(self) -> None:
        """Create necessary directories"""
        self.get_vector_db_path().mkdir(parents=True, exist_ok=True)


# Default configuration instance
config = AgentConfig.from_env()