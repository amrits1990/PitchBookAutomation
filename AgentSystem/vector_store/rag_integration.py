"""
Integration helpers for RAG packages with vector storage
Provides simple functions to update existing agent interfaces
"""

import logging
from typing import Dict, List, Any, Optional
from .lance_store import VectorStore
from ..config import config

logger = logging.getLogger(__name__)


class RAGVectorIntegration:
    """Helper class to integrate RAG packages with vector storage"""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
    
    def enhance_index_function(
        self,
        original_index_func,
        table_name: str,
        content_field: str = "content"
    ):
        """
        Enhance an existing index function to also store in vector database
        
        Args:
            original_index_func: Original index function from RAG package
            table_name: Name for the vector table
            content_field: Field containing the text content
            
        Returns:
            Enhanced function that stores in vector DB and returns original result
        """
        def enhanced_index(*args, **kwargs):
            # Call original function first
            result = original_index_func(*args, **kwargs)
            
            # If successful, also store in vector database
            if result.get("success") and result.get("chunks"):
                try:
                    # Index in vector database
                    vector_result = self.vector_store.index_documents(
                        table_name=table_name,
                        documents=result["chunks"],
                        text_field=content_field,
                        overwrite=False  # Append by default
                    )
                    
                    # Add vector storage info to result
                    result["vector_storage"] = vector_result
                    logger.info(f"Indexed {len(result['chunks'])} chunks in vector store table '{table_name}'")
                    
                except Exception as e:
                    logger.warning(f"Failed to store in vector database: {e}")
                    result["vector_storage"] = {"success": False, "error": str(e)}
            
            return result
        
        return enhanced_index
    
    def enhance_search_function(
        self,
        original_search_func,
        table_name: str,
        use_vector_search: bool = True
    ):
        """
        Enhance an existing search function to use vector search when available
        
        Args:
            original_search_func: Original search function from RAG package
            table_name: Name of the vector table
            use_vector_search: Whether to use vector search or fall back to original
            
        Returns:
            Enhanced function that uses vector search when available
        """
        def enhanced_search(*args, **kwargs):
            # Extract common parameters
            query = kwargs.get("query", args[1] if len(args) > 1 else "")
            k = kwargs.get("k", 20)
            
            # Try vector search first if enabled
            if use_vector_search and query:
                try:
                    # Check if table exists
                    table_info = self.vector_store.get_table_info(table_name)
                    
                    if table_info.get("exists") and table_info.get("document_count", 0) > 0:
                        # Use hybrid search
                        vector_results = self.vector_store.hybrid_search(
                            table_name=table_name,
                            query=query,
                            k=k,
                            semantic_weight=config.semantic_weight,
                            bm25_weight=config.bm25_weight
                        )
                        
                        if vector_results:
                            logger.info(f"Used vector search for table '{table_name}', found {len(vector_results)} results")
                            
                            # Format results to match original function output
                            return {
                                "success": True,
                                "ticker": args[0] if args else kwargs.get("ticker", "unknown"),
                                "query": query,
                                "results": vector_results,
                                "returned": len(vector_results),
                                "total_candidates": table_info.get("document_count", 0),
                                "search_method": "vector_hybrid",
                                "metadata": {
                                    "k": k,
                                    "table_name": table_name,
                                    "semantic_weight": config.semantic_weight,
                                    "bm25_weight": config.bm25_weight
                                }
                            }
                
                except Exception as e:
                    logger.warning(f"Vector search failed, falling back to original: {e}")
            
            # Fall back to original search function
            result = original_search_func(*args, **kwargs)
            if result.get("success"):
                result["search_method"] = "original_fallback"
            
            return result
        
        return enhanced_search


# Global integration instance for easy access
rag_vector_integration = RAGVectorIntegration()


def enhance_rag_package(
    index_func,
    search_func, 
    table_name: str,
    content_field: str = "content"
) -> tuple:
    """
    Simple function to enhance both index and search functions for a RAG package
    
    Args:
        index_func: Original index function
        search_func: Original search function  
        table_name: Name for vector table
        content_field: Field containing text content
        
    Returns:
        Tuple of (enhanced_index_func, enhanced_search_func)
    """
    enhanced_index = rag_vector_integration.enhance_index_function(
        index_func, table_name, content_field
    )
    
    enhanced_search = rag_vector_integration.enhance_search_function(
        search_func, table_name, use_vector_search=True
    )
    
    return enhanced_index, enhanced_search