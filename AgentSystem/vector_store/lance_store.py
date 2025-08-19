"""
Simple LanceDB vector store wrapper for the Agent System
Provides easy-to-use vector storage and hybrid search capabilities
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, date

# Core dependencies
try:
    import lancedb
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install lancedb sentence-transformers rank-bm25")
    raise

from ..config import config

logger = logging.getLogger(__name__)


def _serialize_metadata(metadata: Dict[str, Any]) -> str:
    """Serialize metadata to JSON string, handling date objects"""
    def json_serializer(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    
    try:
        return json.dumps(metadata, default=json_serializer)
    except Exception as e:
        # Fallback: convert problematic objects to strings
        safe_metadata = {}
        for key, value in metadata.items():
            try:
                json.dumps(value)  # Test if serializable
                safe_metadata[key] = value
            except (TypeError, ValueError):
                safe_metadata[key] = str(value)  # Convert to string
        return json.dumps(safe_metadata)


class VectorStore:
    """Simple vector store using LanceDB with hybrid search"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(config.get_vector_db_path())
        self.embedding_model_name = config.embedding_model
        
        # Initialize components
        self._db = None
        self._encoder = None
        self._tables = {}  # Cache for table references
        
        # Ensure database directory exists
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
    @property
    def db(self):
        """Lazy initialization of LanceDB connection"""
        if self._db is None:
            self._db = lancedb.connect(self.db_path)
        return self._db
    
    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy initialization of sentence transformer"""
        if self._encoder is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        if not texts:
            return np.array([])
        
        logger.debug(f"Creating embeddings for {len(texts)} texts")
        embeddings = self.encoder.encode(texts, show_progress_bar=len(texts) > 10)
        return embeddings
    
    def index_documents(
        self,
        table_name: str,
        documents: List[Dict[str, Any]],
        text_field: str = "content",
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Index documents into a LanceDB table
        
        Args:
            table_name: Name of the table to create/update
            documents: List of document dictionaries
            text_field: Field containing the text to embed
            overwrite: Whether to overwrite existing table
            
        Returns:
            Status dictionary with success/failure info
        """
        try:
            if not documents:
                return {"success": False, "error": "No documents provided"}
            
            logger.info(f"Indexing {len(documents)} documents into table '{table_name}'")
            
            # Extract texts and create embeddings
            texts = [doc.get(text_field, "") for doc in documents]
            texts = [t for t in texts if t]  # Remove empty texts
            
            if not texts:
                return {"success": False, "error": f"No valid text found in field '{text_field}'"}
            
            embeddings = self.create_embeddings(texts)
            
            # Prepare data for LanceDB
            records = []
            for i, doc in enumerate(documents):
                if doc.get(text_field):  # Only include docs with text
                    record = {
                        "id": doc.get("id", f"{table_name}_{i}"),
                        "content": doc.get(text_field),
                        "vector": embeddings[len(records)].tolist(),  # Match embedding index
                        "metadata": _serialize_metadata(doc.get("metadata", {})),
                        "indexed_at": datetime.now().isoformat(),
                    }
                    # Add other fields directly, handling date serialization
                    for key, value in doc.items():
                        if key not in ["id", "content", "metadata", "vector"]:
                            # Handle date objects in other fields
                            if isinstance(value, (datetime, date)):
                                record[key] = value.isoformat()
                            else:
                                record[key] = value
                    records.append(record)
            
            # Create or update table
            if table_name in self.db.table_names() and overwrite:
                self.db.drop_table(table_name)
            
            if table_name not in self.db.table_names():
                table = self.db.create_table(table_name, records)
            else:
                table = self.db.open_table(table_name)
                table.add(records)
            
            self._tables[table_name] = table
            
            return {
                "success": True,
                "table_name": table_name,
                "documents_indexed": len(records),
                "embedding_dimension": len(embeddings[0]) if len(embeddings) > 0 else 0,
                "indexed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return {"success": False, "error": str(e)}
    
    def semantic_search(
        self,
        table_name: str,
        query: str,
        k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity
        
        Args:
            table_name: Table to search in
            query: Search query text
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of matching documents with similarity scores
        """
        try:
            if table_name not in self.db.table_names():
                logger.warning(f"Table '{table_name}' not found")
                return []
            
            # Get or create table reference
            if table_name not in self._tables:
                self._tables[table_name] = self.db.open_table(table_name)
            table = self._tables[table_name]
            
            # Create query embedding
            query_embedding = self.create_embeddings([query])[0]
            
            # Perform vector search
            search_results = table.search(query_embedding).limit(k)
            
            # Apply metadata filters if provided
            if filters:
                # Note: LanceDB filtering would be done here
                # For now, we'll do post-processing filtering
                pass
            
            results = search_results.to_list()
            
            # Process results
            processed_results = []
            for result in results:
                # Parse metadata if it's JSON
                metadata = result.get("metadata", "{}")
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                processed_results.append({
                    "content": result.get("content", ""),
                    "metadata": metadata,
                    "similarity_score": float(result.get("_distance", 0.0)),  # LanceDB uses distance
                    "id": result.get("id", ""),
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(
        self,
        table_name: str,
        query: str,
        k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search
        
        Args:
            table_name: Table to search in
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of matching documents with BM25 scores
        """
        try:
            if table_name not in self.db.table_names():
                return []
            
            # Get all documents from table
            if table_name not in self._tables:
                self._tables[table_name] = self.db.open_table(table_name)
            table = self._tables[table_name]
            
            # Convert to pandas DataFrame for easier processing
            df = table.to_pandas()
            
            if df.empty:
                return []
            
            # Tokenize documents for BM25
            docs = df["content"].fillna("").tolist()
            tokenized_docs = [doc.lower().split() for doc in docs]
            
            # Create BM25 model
            bm25 = BM25Okapi(tokenized_docs)
            
            # Get scores for query
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)
            
            # Get top k results
            top_indices = np.argsort(scores)[-k:][::-1]  # Descending order
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include documents with positive scores
                    row = df.iloc[idx]
                    metadata = row.get("metadata", "{}")
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    
                    results.append({
                        "content": row.get("content", ""),
                        "metadata": metadata,
                        "bm25_score": float(scores[idx]),
                        "id": row.get("id", ""),
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def hybrid_search(
        self,
        table_name: str,
        query: str,
        k: int = 20,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search
        
        Args:
            table_name: Table to search in
            query: Search query text
            k: Number of results to return
            semantic_weight: Weight for semantic search scores
            bm25_weight: Weight for BM25 scores
            filters: Optional metadata filters
            
        Returns:
            List of documents ranked by combined scores
        """
        try:
            # Get results from both search methods
            semantic_results = self.semantic_search(table_name, query, k=k*2, filters=filters)
            keyword_results = self.keyword_search(table_name, query, k=k*2)
            
            # Normalize scores
            def normalize_scores(results: List[Dict], score_field: str) -> Dict[str, float]:
                if not results:
                    return {}
                scores = [r[score_field] for r in results]
                max_score = max(scores) if scores else 1.0
                min_score = min(scores) if scores else 0.0
                score_range = max_score - min_score if max_score != min_score else 1.0
                
                return {
                    r["id"]: (r[score_field] - min_score) / score_range
                    for r in results
                }
            
            # Normalize both sets of scores
            semantic_scores = normalize_scores(semantic_results, "similarity_score")
            bm25_scores = normalize_scores(keyword_results, "bm25_score")
            
            # Combine results and compute hybrid scores
            all_docs = {}
            
            # Add semantic results
            for result in semantic_results:
                doc_id = result["id"]
                all_docs[doc_id] = result
                all_docs[doc_id]["hybrid_score"] = semantic_weight * semantic_scores.get(doc_id, 0.0)
            
            # Add keyword results and combine scores
            for result in keyword_results:
                doc_id = result["id"]
                if doc_id in all_docs:
                    # Document found in both searches - combine scores
                    all_docs[doc_id]["hybrid_score"] += bm25_weight * bm25_scores.get(doc_id, 0.0)
                    all_docs[doc_id]["bm25_score"] = result["bm25_score"]
                else:
                    # Document only in keyword search
                    all_docs[doc_id] = result
                    all_docs[doc_id]["hybrid_score"] = bm25_weight * bm25_scores.get(doc_id, 0.0)
            
            # Sort by hybrid score and return top k
            sorted_results = sorted(
                all_docs.values(),
                key=lambda x: x["hybrid_score"],
                reverse=True
            )
            
            return sorted_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table"""
        try:
            if table_name not in self.db.table_names():
                return {"exists": False}
            
            table = self.db.open_table(table_name)
            df = table.to_pandas()
            
            return {
                "exists": True,
                "document_count": len(df),
                "schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_metadata": json.loads(df["metadata"].iloc[0]) if len(df) > 0 and "metadata" in df.columns else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {"exists": False, "error": str(e)}
    
    def list_tables(self) -> List[str]:
        """List all tables in the database"""
        try:
            return self.db.table_names()
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []