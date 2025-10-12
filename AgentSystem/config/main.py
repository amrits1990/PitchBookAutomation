"""
Configuration for AgentSystem - Vector Database and RAG Support
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from main .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

class AgentSystemConfig:
    """Configuration class for AgentSystem vector database operations"""
    
    def __init__(self):
        # Vector database configuration
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'all-mpnet-base-v2')
        self.bm25_weight = float(os.getenv('BM25_WEIGHT', '0.4'))
        self.semantic_weight = float(os.getenv('SEMANTIC_WEIGHT', '0.6'))
        self.default_search_k = int(os.getenv('DEFAULT_SEARCH_K', '20'))
        
        # Vector database path
        self.vector_db_path = self.get_vector_db_path()
        
        # Ensure vector database directory exists
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
    
    def get_vector_db_path(self) -> Path:
        """Get the path for vector database storage"""
        base_dir = Path(__file__).parent.parent.parent  # Go up to PitchBookGenerator level
        return base_dir / "data" / "vector_store"
    
    def get_embeddings_config(self) -> dict:
        """Get embeddings configuration"""
        return {
            "model_name": self.embedding_model,
            "device": "cpu",  # Use CPU for compatibility
            "normalize_embeddings": True
        }
    
    def get_search_config(self) -> dict:
        """Get search configuration"""
        return {
            "bm25_weight": self.bm25_weight,
            "semantic_weight": self.semantic_weight,
            "default_k": self.default_search_k
        }

# Global config instance
config = AgentSystemConfig()