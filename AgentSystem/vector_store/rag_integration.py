"""
RAG Integration utilities for seamless integration with TranscriptRAG and AnnualReportRAG
"""

import logging
from typing import Dict, Any, List, Optional
from .lance_store import VectorStore
from ..config import config

logger = logging.getLogger(__name__)

class RAGVectorStore(VectorStore):
    """
    Extended VectorStore class with RAG-specific convenience methods
    """
    
    def __init__(self, db_path: Optional[str] = None):
        super().__init__(db_path)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for RAG systems
        Returns status information about vector store availability
        """
        try:
            # Check if database path exists and is accessible
            db_accessible = self.db_path.exists()
            
            # Check if we can connect to database
            tables = self.list_tables()
            
            # Check if embeddings model can be loaded
            try:
                _ = self.embeddings_model
                embeddings_available = True
            except Exception:
                embeddings_available = False
            
            return {
                "success": True,
                "database_accessible": db_accessible,
                "available_tables": tables,
                "table_count": len(tables),
                "embeddings_available": embeddings_available,
                "db_path": str(self.db_path)
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "database_accessible": False,
                "available_tables": [],
                "table_count": 0,
                "embeddings_available": False
            }
    
    def get_rag_status(self) -> Dict[str, Any]:
        """Get comprehensive status for RAG system integration"""
        health = self.health_check()
        
        # Check for common RAG tables
        common_tables = []
        if health["success"]:
            for table_name in health["available_tables"]:
                table_info = self.get_table_info(table_name)
                if table_info.get("exists"):
                    common_tables.append({
                        "name": table_name,
                        "row_count": table_info.get("row_count", 0)
                    })
        
        return {
            "vector_store_available": health["success"],
            "total_tables": len(common_tables),
            "tables": common_tables,
            "configuration": {
                "embedding_model": config.embedding_model,
                "bm25_weight": config.bm25_weight,
                "semantic_weight": config.semantic_weight
            }
        }
    
    def setup_for_rag(self, rag_type: str) -> Dict[str, Any]:
        """
        Setup vector store for specific RAG type
        
        Args:
            rag_type: Either 'transcript' or 'annual_report'
        """
        try:
            table_name = f"{rag_type}_chunks"
            
            # Ensure database directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Check if table already exists
            table_info = self.get_table_info(table_name)
            
            return {
                "success": True,
                "rag_type": rag_type,
                "table_name": table_name,
                "table_exists": table_info.get("exists", False),
                "row_count": table_info.get("row_count", 0),
                "ready_for_indexing": True
            }
            
        except Exception as e:
            self.logger.error(f"Setup failed for {rag_type}: {e}")
            return {
                "success": False,
                "rag_type": rag_type,
                "error": str(e),
                "ready_for_indexing": False
            }

# Convenience function for RAG systems
def get_vector_store() -> RAGVectorStore:
    """Get a configured vector store instance for RAG systems"""
    return RAGVectorStore()

def check_vector_dependencies() -> Dict[str, Any]:
    """Check if all vector database dependencies are available"""
    try:
        import lancedb
        import sentence_transformers
        import rank_bm25
        import pandas
        import numpy
        
        return {
            "available": True,
            "lancedb": True,
            "sentence_transformers": True,
            "rank_bm25": True,
            "pandas": True,
            "numpy": True
        }
        
    except ImportError as e:
        missing_deps = []
        for dep in ["lancedb", "sentence_transformers", "rank_bm25", "pandas", "numpy"]:
            try:
                __import__(dep.replace("-", "_"))
            except ImportError:
                missing_deps.append(dep)
        
        return {
            "available": False,
            "missing_dependencies": missing_deps,
            "install_command": "pip install lancedb>=0.13.0 rank-bm25>=0.2.2 sentence-transformers pandas numpy"
        }