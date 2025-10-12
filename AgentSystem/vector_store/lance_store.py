"""
LanceDB Vector Store Implementation for RAG Systems
Compatible with TranscriptRAG and AnnualReportRAG requirements
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json

# Import vector database dependencies
try:
    import lancedb
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import numpy as np
    VECTOR_DEPS_AVAILABLE = True
except ImportError as e:
    lancedb = None
    pd = None
    SentenceTransformer = None
    BM25Okapi = None
    np = None
    VECTOR_DEPS_AVAILABLE = False
    print(f"Vector database dependencies not available: {e}")
    print("Run 'pip install lancedb>=0.13.0 rank-bm25>=0.2.2 sentence-transformers' to enable vector search")

from ..config import config

logger = logging.getLogger(__name__)

class VectorStore:
    """
    LanceDB-based vector store for document storage and retrieval
    Compatible with TranscriptRAG and AnnualReportRAG
    """
    
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """Initialize the vector store"""
        if not VECTOR_DEPS_AVAILABLE:
            raise ImportError("Vector database dependencies not available. Run: pip install lancedb>=0.13.0 rank-bm25>=0.2.2 sentence-transformers")
        
        self.db_path = Path(db_path) if db_path else config.vector_db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        self._db = None
        self._embeddings_model = None
        self._bm25_models = {}  # Cache BM25 models per table
        
        # Load configuration
        self.embedding_config = config.get_embeddings_config()
        self.search_config = config.get_search_config()
        
        logger.info(f"VectorStore initialized with path: {self.db_path}")
    
    @property
    def db(self):
        """Lazy-load database connection"""
        if self._db is None:
            self._db = lancedb.connect(self.db_path)
        return self._db
    
    @property
    def embeddings_model(self):
        """Lazy-load embeddings model"""
        if self._embeddings_model is None:
            model_name = self.embedding_config["model_name"]
            self._embeddings_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embeddings model: {model_name}")
        return self._embeddings_model
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        try:
            if table_name in self.db.table_names():
                table = self.db.open_table(table_name)
                schema = table.schema
                count = table.count_rows()
                
                return {
                    "exists": True,
                    "schema": str(schema),
                    "row_count": count,
                    "table_name": table_name
                }
            else:
                return {
                    "exists": False,
                    "table_name": table_name,
                    "available_tables": self.db.table_names()
                }
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            return {
                "exists": False,
                "error": str(e),
                "table_name": table_name
            }
    
    def get_filter_summary(self, table_name: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get summary of available filters for a table"""
        try:
            if table_name not in self.db.table_names():
                return {"error": f"Table {table_name} does not exist"}
            
            table = self.db.open_table(table_name)
            
            # Get sample data to understand available fields
            sample_df = table.limit(100).to_pandas()
            
            filter_info = {}
            for column in sample_df.columns:
                if column not in ['vector', 'text', 'content']:  # Skip vector and text columns
                    unique_values = sample_df[column].dropna().unique()
                    if len(unique_values) < 50:  # Only show if manageable number of unique values
                        filter_info[column] = {
                            "type": str(sample_df[column].dtype),
                            "unique_values": unique_values.tolist()[:20],  # Limit to 20 values
                            "total_unique": len(unique_values)
                        }
                    else:
                        filter_info[column] = {
                            "type": str(sample_df[column].dtype),
                            "total_unique": len(unique_values),
                            "sample_values": unique_values[:10].tolist()
                        }
            
            return {
                "table_name": table_name,
                "available_filters": filter_info,
                "total_rows": table.count_rows()
            }
            
        except Exception as e:
            logger.error(f"Error getting filter summary for {table_name}: {e}")
            return {"error": str(e)}
    
    def index_documents(self, documents: List[Dict[str, Any]], table_name: str, 
                       text_field: str = "content", **kwargs) -> Dict[str, Any]:
        """Index documents into the vector store"""
        try:
            if not documents:
                return {"success": False, "error": "No documents provided"}
            
            # Extract text content for embedding
            texts = [doc.get(text_field, "") for doc in documents]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.embeddings_model.encode(texts, show_progress_bar=True)
            
            # Prepare data for LanceDB
            data_rows = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                row = doc.copy()
                row['vector'] = embedding.tolist()
                row['text'] = texts[i]  # Ensure text field exists
                data_rows.append(row)
            
            # Create or update table
            df = pd.DataFrame(data_rows)
            
            if table_name in self.db.table_names():
                # Append to existing table
                table = self.db.open_table(table_name)
                table.add(df)
                logger.info(f"Added {len(documents)} documents to existing table {table_name}")
            else:
                # Create new table
                table = self.db.create_table(table_name, df)
                logger.info(f"Created new table {table_name} with {len(documents)} documents")
            
            # Create BM25 index for keyword search
            self._create_bm25_index(table_name, texts)
            
            return {
                "success": True,
                "table_name": table_name,
                "documents_indexed": len(documents),
                "total_rows": table.count_rows()
            }
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return {"success": False, "error": str(e)}
    
    def add_documents(self, documents: List[Dict[str, Any]], table_name: str,
                     text_field: str = "content", **kwargs) -> Dict[str, Any]:
        """Alias for index_documents for compatibility"""
        return self.index_documents(documents, table_name, text_field, **kwargs)
    
    def _create_bm25_index(self, table_name: str, texts: List[str]):
        """Create BM25 index for keyword search"""
        try:
            # Tokenize texts (simple word-based tokenization)
            tokenized_texts = [text.lower().split() for text in texts]
            bm25 = BM25Okapi(tokenized_texts)
            self._bm25_models[table_name] = {
                "model": bm25,
                "texts": texts
            }
            logger.info(f"Created BM25 index for table {table_name}")
        except Exception as e:
            logger.warning(f"Failed to create BM25 index for {table_name}: {e}")
    
    def semantic_search(self, table_name: str, query: str, k: int = 10,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform semantic (vector) search"""
        try:
            if table_name not in self.db.table_names():
                return []
            
            table = self.db.open_table(table_name)
            
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query])[0]
            
            # Perform vector search
            search_query = table.search(query_embedding).limit(k)
            
            # Apply filters if provided
            if filters:
                for field, value in filters.items():
                    if isinstance(value, list):
                        search_query = search_query.where(f"{field} IN {value}")
                    else:
                        search_query = search_query.where(f"{field} = '{value}'")
            
            results = search_query.to_pandas()
            
            return results.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, table_name: str, query: str, k: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform keyword (BM25) search"""
        try:
            if table_name not in self._bm25_models:
                # Fallback to semantic search if BM25 not available
                logger.warning(f"BM25 model not available for {table_name}, falling back to semantic search")
                return self.semantic_search(table_name, query, k, filters)
            
            bm25_data = self._bm25_models[table_name]
            bm25_model = bm25_data["model"]
            texts = bm25_data["texts"]
            
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = bm25_model.get_scores(query_tokens)
            
            # Get top k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            # Get full documents from vector store
            if table_name in self.db.table_names():
                table = self.db.open_table(table_name)
                df = table.to_pandas()
                
                results = []
                for idx in top_indices:
                    if idx < len(df):
                        row = df.iloc[idx].to_dict()
                        row['bm25_score'] = float(scores[idx])
                        
                        # Apply filters if provided
                        if filters:
                            matches_filter = True
                            for field, value in filters.items():
                                if field in row:
                                    if isinstance(value, list):
                                        if row[field] not in value:
                                            matches_filter = False
                                            break
                                    else:
                                        if row[field] != value:
                                            matches_filter = False
                                            break
                            
                            if matches_filter:
                                results.append(row)
                        else:
                            results.append(row)
                
                return results[:k]
            
            return []
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def hybrid_search(self, table_name: str, query: str, k: int = 10,
                     filters: Optional[Dict[str, Any]] = None,
                     bm25_weight: Optional[float] = None,
                     semantic_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search"""
        try:
            # Use configured weights if not provided
            bm25_w = bm25_weight or self.search_config["bm25_weight"]
            semantic_w = semantic_weight or self.search_config["semantic_weight"]
            
            # Get results from both methods
            semantic_results = self.semantic_search(table_name, query, k * 2, filters)
            keyword_results = self.keyword_search(table_name, query, k * 2, filters)
            
            # Combine and score results
            combined_results = {}
            
            # Add semantic results
            for i, result in enumerate(semantic_results):
                doc_id = result.get('id', str(hash(result.get('text', '')[:100])))
                semantic_score = 1.0 / (i + 1)  # Reciprocal rank
                combined_results[doc_id] = {
                    'document': result,
                    'semantic_score': semantic_score,
                    'bm25_score': 0.0
                }
            
            # Add BM25 results
            for i, result in enumerate(keyword_results):
                doc_id = result.get('id', str(hash(result.get('text', '')[:100])))
                bm25_score = result.get('bm25_score', 1.0 / (i + 1))
                
                if doc_id in combined_results:
                    combined_results[doc_id]['bm25_score'] = bm25_score
                else:
                    combined_results[doc_id] = {
                        'document': result,
                        'semantic_score': 0.0,
                        'bm25_score': bm25_score
                    }
            
            # Calculate hybrid scores and rank
            final_results = []
            for doc_id, scores in combined_results.items():
                hybrid_score = (semantic_w * scores['semantic_score'] + 
                              bm25_w * scores['bm25_score'])
                
                doc = scores['document'].copy()
                doc['hybrid_score'] = hybrid_score
                doc['semantic_score'] = scores['semantic_score']
                doc['bm25_score'] = scores['bm25_score']
                
                final_results.append(doc)
            
            # Sort by hybrid score and return top k
            final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def search(self, table_name: str, query: str, k: int = 10,
              filters: Optional[Dict[str, Any]] = None,
              method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Generic search method that dispatches to specific search types
        Compatible with various RAG system interfaces
        """
        if method == "semantic":
            return self.semantic_search(table_name, query, k, filters)
        elif method == "keyword" or method == "bm25":
            return self.keyword_search(table_name, query, k, filters)
        else:  # Default to hybrid
            return self.hybrid_search(table_name, query, k, filters)
    
    def delete_table(self, table_name: str) -> Dict[str, Any]:
        """Delete a table from the vector store"""
        try:
            if table_name in self.db.table_names():
                self.db.drop_table(table_name)
                
                # Remove BM25 model if exists
                if table_name in self._bm25_models:
                    del self._bm25_models[table_name]
                
                logger.info(f"Deleted table {table_name}")
                return {"success": True, "message": f"Table {table_name} deleted"}
            else:
                return {"success": False, "error": f"Table {table_name} does not exist"}
        except Exception as e:
            logger.error(f"Error deleting table {table_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def list_tables(self) -> List[str]:
        """List all tables in the vector store"""
        try:
            return self.db.table_names()
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []