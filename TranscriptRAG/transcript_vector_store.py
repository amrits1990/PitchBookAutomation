"""
TranscriptRAG-specific LanceDB vector store
- Adds denormalized metadata columns for robust transcript filtering without impacting other RAGs
- Mirrors the shared VectorStore API for index and hybrid search
- Optimized for earnings call transcript metadata (quarters, speakers, sections)
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, date
from pathlib import Path
import json
import logging
import re
import sys
import os

# Add parent directory to import shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import query enhancer
try:
    from .transcript_query_enhancer import TranscriptQueryEnhancer
except ImportError:
    from transcript_query_enhancer import TranscriptQueryEnhancer

# Try to import optional vector database dependencies
try:
    import lancedb
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    VECTOR_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    lancedb = None
    np = None
    pd = None
    SentenceTransformer = None
    BM25Okapi = None
    VECTOR_DEPENDENCIES_AVAILABLE = False
    print(f"Warning: Vector database dependencies not available: {e}")
    print("Run 'pip install lancedb>=0.13.0 rank-bm25>=0.2.2' to enable vector search")

# Import robust embedding loader for reliable model loading
try:
    from shared_utils import create_embedding_loader
    ROBUST_LOADER_AVAILABLE = True
except ImportError as e:
    ROBUST_LOADER_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"Robust embedding loader not available: {e}. Falling back to standard loading.")

# Try to import config from AgentSystem, with fallback
try:
    from AgentSystem.config import config
except ImportError:
    # Fallback config class for standalone usage
    import os
    class FallbackConfig:
        def __init__(self):
            self.embedding_model = os.getenv('EMBEDDING_MODEL', 'all-mpnet-base-v2')
            self.bm25_weight = float(os.getenv('BM25_WEIGHT', '0.4'))
            self.semantic_weight = float(os.getenv('SEMANTIC_WEIGHT', '0.6'))
        
        def get_vector_db_path(self):
            # Use same path as AgentSystem would use, relative to current package
            base_dir = Path(__file__).parent.parent  # Go up to PitchBookGenerator level
            return base_dir / "data" / "vector_store"
    
    config = FallbackConfig()

logger = logging.getLogger(__name__)


def _serialize_metadata(metadata: Dict[str, Any]) -> str:
    def json_serializer(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    try:
        return json.dumps(metadata, default=json_serializer)
    except Exception:
        safe_metadata = {}
        for k, v in metadata.items():
            try:
                json.dumps(v)
                safe_metadata[k] = v
            except (TypeError, ValueError):
                safe_metadata[k] = str(v)
        return json.dumps(safe_metadata)


class TranscriptVectorStore:
    """Specialized LanceDB store for TranscriptRAG with denormalized transcript metadata columns."""

    def __init__(self, db_path: Optional[str] = None):
        if not VECTOR_DEPENDENCIES_AVAILABLE:
            raise ImportError("Vector database dependencies not available. Run: pip install lancedb>=0.13.0 rank-bm25>=0.2.2")

        self.db_path = db_path or str(config.get_vector_db_path())
        self.embedding_model_name = config.embedding_model
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self._db = None
        self._encoder = None
        self._embedding_loader = None  # Robust embedding loader
        self._tables: Dict[str, Any] = {}
        self.query_enhancer = TranscriptQueryEnhancer()

    @property
    def db(self):
        if self._db is None:
            self._db = lancedb.connect(self.db_path)
        return self._db

    @property
    def encoder(self) -> SentenceTransformer:
        """Load embedding model with robust error handling."""
        if self._encoder is None:
            if ROBUST_LOADER_AVAILABLE:
                # Use robust loader with retry logic and cache management
                if self._embedding_loader is None:
                    self._embedding_loader = create_embedding_loader(
                        model_name=self.embedding_model_name,
                        max_retries=3,
                        retry_delay=2.0
                    )
                logger.info(f"Loading embedding model with robust loader: {self.embedding_model_name}")
                self._encoder = self._embedding_loader.load_model()
            else:
                # Fallback to standard loading
                logger.info(f"Loading embedding model (standard): {self.embedding_model_name}")
                self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings with automatic error recovery."""
        if not texts:
            return np.array([])

        if ROBUST_LOADER_AVAILABLE and self._embedding_loader is not None:
            # Use robust loader's encode method with built-in error handling
            try:
                return self._embedding_loader.encode(
                    texts,
                    show_progress_bar=len(texts) > 10
                )
            except Exception as e:
                logger.error(f"Robust embedding creation failed: {e}")
                raise
        else:
            # Fallback to standard encoding
            return self.encoder.encode(texts, show_progress_bar=len(texts) > 10)

    def index_documents(self, table_name: str, documents: List[Dict[str, Any]], text_field: str = "content", overwrite: bool = False) -> Dict[str, Any]:
        try:
            if not documents:
                return {"success": False, "error": "No documents provided"}
            texts = [doc.get(text_field, "") for doc in documents]
            texts = [t for t in texts if t]
            if not texts:
                return {"success": False, "error": f"No valid text found in field '{text_field}'"}
            embeddings = self.create_embeddings(texts)

            records: List[Dict[str, Any]] = []
            for i, doc in enumerate(documents):
                if not doc.get(text_field):
                    continue
                md = doc.get("metadata", {}) or {}
                # Denormalized metadata columns for robust filtering
                record = {
                    "id": doc.get("id", f"{table_name}_{i}"),
                    "content": doc.get(text_field),
                    "vector": embeddings[len(records)].tolist(),
                    "metadata": _serialize_metadata(md),
                    "indexed_at": datetime.now().isoformat(),
                    # Transcript-specific fields based on metadata structure
                    "ticker": str(md.get("ticker")).upper() if md.get("ticker") is not None else None,
                    "quarter": str(md.get("quarter") or md.get("fiscal_quarter")).upper() if (md.get("quarter") or md.get("fiscal_quarter")) is not None else None,
                    "fiscal_year": str(md.get("fiscal_year")) if md.get("fiscal_year") is not None else None,
                    "transcript_date": str(md.get("transcript_date")) if md.get("transcript_date") is not None else None,
                    "transcript_type": str(md.get("transcript_type")) if md.get("transcript_type") is not None else None,
                    "section_name": str(md.get("section_name")) if md.get("section_name") is not None else None,
                    "speaker": str(md.get("speaker")) if md.get("speaker") is not None else str(md.get("primary_speaker")) if md.get("primary_speaker") is not None else None,
                    # Speaker analysis fields
                    "is_management": md.get("speaker_analysis", {}).get("is_management", None) if md.get("speaker_analysis") else None,
                    "is_analyst": md.get("speaker_analysis", {}).get("is_analyst", None) if md.get("speaker_analysis") else None,
                    # Section analysis fields  
                    "is_qa_section": md.get("section_analysis", {}).get("is_qa_section", None) if md.get("section_analysis") else None,
                    "is_prepared_remarks": md.get("section_analysis", {}).get("is_prepared_remarks", None) if md.get("section_analysis") else None,
                    # Content analytics
                    "content_length": md.get("chunk_analytics", {}).get("content_length", None) if md.get("chunk_analytics") else None,
                    "word_count": md.get("chunk_analytics", {}).get("word_count", None) if md.get("chunk_analytics") else None,
                    "contains_financial_terms": md.get("chunk_analytics", {}).get("contains_financial_terms", None) if md.get("chunk_analytics") else None,
                }
                # Attach other fields passthrough
                for key, value in doc.items():
                    if key not in record and key not in ["vector", "metadata"]:
                        record[key] = value
                records.append(record)

            if table_name in self.db.table_names() and overwrite:
                self.db.drop_table(table_name)
            if table_name not in self.db.table_names():
                table = self.db.create_table(table_name, records)
            else:
                table = self.db.open_table(table_name)
                table.add(records)
            self._tables[table_name] = table
            return {"success": True, "table_name": table_name, "documents_indexed": len(records), "indexed_at": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"TranscriptVectorStore index error: {e}")
            return {"success": False, "error": str(e)}

    def _build_where_clause(self, table, filters: Dict[str, Any]) -> Optional[str]:
        if not filters:
            return None
        try:
            cols = set(table.to_pandas().columns)
        except Exception:
            return None
        conds: List[str] = []
        def esc(val: Any) -> str:
            s = str(val)
            # Escape single quotes for SQL literal safety
            return s.replace("'", "''")
        
        # Equality filters
        def add_eq_or_in(key, value, normalize=None):
            if key not in cols or value is None:
                return
            # Accept list/tuple/set for multi-value filtering
            if isinstance(value, (list, tuple, set)):
                vals = [v for v in value if v is not None]
                if not vals:
                    return
                norm_vals = []
                for v in vals:
                    try:
                        nv = normalize(v) if normalize else v
                        norm_vals.append(str(nv))
                    except Exception:
                        continue
                norm_vals = [v for v in norm_vals if v]
                if not norm_vals:
                    return
                # Deduplicate while preserving order
                seen = set()
                uniq = []
                for v in norm_vals:
                    if v not in seen:
                        seen.add(v)
                        uniq.append(v)
                if len(uniq) == 1:
                    conds.append(f"{key} = '{esc(uniq[0])}'")
                else:
                    in_list = ", ".join(f"'{esc(x)}'" for x in uniq)
                    conds.append(f"{key} IN ({in_list})")
            else:
                v = normalize(value) if normalize else value
                conds.append(f"{key} = '{esc(v)}'")
        
        # Transcript-specific filters
        add_eq_or_in("ticker", filters.get("ticker"), normalize=lambda x: str(x).upper())
        
        # Fix quarter filtering - data is stored as numbers ('1','2','3','4') not Q-prefixed ('Q1','Q2','Q3','Q4')
        def normalize_quarter(x):
            q_str = str(x).upper()
            if q_str.startswith('Q'):
                return q_str[1:]  # Remove 'Q' prefix: 'Q3' -> '3'
            return q_str  # Already just the number
        add_eq_or_in("quarter", filters.get("quarter"), normalize=normalize_quarter)
        
        add_eq_or_in("fiscal_year", filters.get("fiscal_year"), normalize=lambda x: str(x))
        add_eq_or_in("transcript_type", filters.get("transcript_type"))
        # Section_name filtering removed - search across all sections
        add_eq_or_in("speaker", filters.get("speaker"))
        
        # Boolean filters for analysis fields
        if "is_management" in filters and filters["is_management"] is not None:
            conds.append(f"is_management = {str(filters['is_management']).lower()}")
        if "is_analyst" in filters and filters["is_analyst"] is not None:
            conds.append(f"is_analyst = {str(filters['is_analyst']).lower()}")
        if "is_qa_section" in filters and filters["is_qa_section"] is not None:
            conds.append(f"is_qa_section = {str(filters['is_qa_section']).lower()}")
        if "is_prepared_remarks" in filters and filters["is_prepared_remarks"] is not None:
            conds.append(f"is_prepared_remarks = {str(filters['is_prepared_remarks']).lower()}")
        if "contains_financial_terms" in filters and filters["contains_financial_terms"] is not None:
            conds.append(f"contains_financial_terms = {str(filters['contains_financial_terms']).lower()}")
        
        # Date range filters (ISO strings compare lexicographically)
        if "transcript_date_after" in filters or "transcript_date_before" in filters:
            if "transcript_date" in cols:
                if filters.get("transcript_date_after"):
                    conds.append(f"transcript_date >= '{esc(str(filters['transcript_date_after']))}'")
                if filters.get("transcript_date_before"):
                    conds.append(f"transcript_date <= '{esc(str(filters['transcript_date_before']))}'")
        
        return " AND ".join(conds) if conds else None

    def semantic_search(self, table_name: str, query: str, k: int = 20, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            if table_name not in self.db.table_names():
                return []
            if table_name not in self._tables:
                self._tables[table_name] = self.db.open_table(table_name)
            table = self._tables[table_name]
            qvec = self.create_embeddings([query])[0]
            where_clause = self._build_where_clause(table, filters or {})
            search = table.search(qvec)
            if where_clause:
                try:
                    search = search.where(where_clause)
                except Exception as e:
                    logger.warning(f"Filter failed ({str(e)}), using fallback approach")
                    # Retry with fallback filters if there's an issue
                    if filters:
                        retry_filters = {k: v for k, v in filters.items() if k in ["ticker", "quarter", "fiscal_year"]}
                        retry_where_clause = self._build_where_clause(table, retry_filters)
                        if retry_where_clause:
                            try:
                                search = search.where(retry_where_clause)
                            except Exception:
                                pass  # If still fails, continue without filters
            results = search.limit(k).to_list()
            
            # Add defensive type checking for search results
            if not isinstance(results, (list, tuple)):
                logger.error(f"LanceDB search returned unexpected type: {type(results)} = {results}")
                return []
            
            out = []
            for r in results:
                meta = r.get("metadata", "{}")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                
                # Debug: Check if speaker field exists
                if logger.isEnabledFor(logging.DEBUG):
                    speaker_debug = meta.get("speaker", "MISSING")
                    primary_speaker_debug = meta.get("primary_speaker", "MISSING")
                    logger.debug(f"Vector search result: speaker='{speaker_debug}', primary_speaker='{primary_speaker_debug}'")
                
                # Fix speaker field if missing but primary_speaker exists
                if not meta.get("speaker") and meta.get("primary_speaker"):
                    meta["speaker"] = meta["primary_speaker"]
                
                # Convert distance to normalized similarity score (0-1 range, higher is better)
                distance = float(r.get("_distance", 1.0))
                # Use exponential decay to convert distance to similarity
                # Cosine distance ranges 0-2, this maps it to roughly 0-1 similarity
                similarity_score = max(0.0, min(1.0, 1.0 / (1.0 + distance)))
                
                out.append({
                    "content": r.get("content", ""),
                    "metadata": meta,
                    "similarity_score": similarity_score,
                    "relevance_score": similarity_score,  # Alias for compatibility
                    "_raw_distance": distance,  # Keep original distance for debugging
                    "id": r.get("id", ""),
                })
            return out
        except Exception as e:
            logger.error(f"TranscriptVectorStore semantic_search error: {e}")
            return []

    def keyword_search(self, table_name: str, query: str, k: int = 20, filters: Optional[Dict[str, Any]] = None, enhance_query: bool = True) -> List[Dict[str, Any]]:
        try:
            # Enhance query with financial synonyms for better BM25 matching
            original_query = query
            if enhance_query:
                query = self.query_enhancer.enhance_bm25_query(query)
                logger.debug(f"Enhanced query: '{original_query}' -> '{query}'")
            
            if table_name not in self.db.table_names():
                return []
            if table_name not in self._tables:
                self._tables[table_name] = self.db.open_table(table_name)
            table = self._tables[table_name]
            df = table.to_pandas()
            if df.empty:
                return []
            
            # DataFrame-level filtering using denormalized columns
            f = filters or {}
            def eq(col, val, normalize=None):
                nonlocal df
                if col not in df.columns or val is None:
                    return
                if isinstance(val, (list, tuple, set)):
                    vals = []
                    for v in val:
                        if v is None:
                            continue
                        try:
                            nv = normalize(v) if normalize else v
                            vals.append(str(nv))
                        except Exception:
                            continue
                    if not vals:
                        return
                    df = df[df[col].astype(str).isin(vals)]
                else:
                    v = normalize(val) if normalize else val
                    df = df[df[col].astype(str) == str(v)]
            
            eq("ticker", f.get("ticker"), normalize=lambda x: str(x).upper())
            eq("quarter", f.get("quarter"), normalize=lambda x: str(x).upper())
            eq("fiscal_year", f.get("fiscal_year"), normalize=lambda x: str(x))
            # Section_name filtering removed
            eq("speaker", f.get("speaker"))
            eq("transcript_type", f.get("transcript_type"))
            
            # Boolean filters
            if f.get("is_management") is not None:
                df = df[df["is_management"] == f["is_management"]]
            if f.get("is_analyst") is not None:
                df = df[df["is_analyst"] == f["is_analyst"]]
            if f.get("is_qa_section") is not None:
                df = df[df["is_qa_section"] == f["is_qa_section"]]
            if f.get("is_prepared_remarks") is not None:
                df = df[df["is_prepared_remarks"] == f["is_prepared_remarks"]]

            # Build BM25 corpus, skipping empty documents to avoid division by zero
            docs = df["content"].fillna("").tolist()
            tokenized_all = [d.lower().split() for d in docs]
            index_map = [i for i, toks in enumerate(tokenized_all) if len(toks) > 0]
            if not index_map:
                return []
            tokenized = [tokenized_all[i] for i in index_map]
            q_tokens = query.lower().split()
            if not q_tokens:
                return []
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(q_tokens)
            top_idx = np.argsort(scores)[-k:][::-1]
            results: List[Dict[str, Any]] = []
            for idx in top_idx:
                if scores[idx] > 0:
                    row = df.iloc[index_map[idx]]
                    meta = row.get("metadata", "{}")
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {}
                    
                    # Fix speaker field if missing but primary_speaker exists
                    if not meta.get("speaker") and meta.get("primary_speaker"):
                        meta["speaker"] = meta["primary_speaker"]
                    
                    results.append({
                        "content": row.get("content", ""),
                        "metadata": meta,
                        "bm25_score": float(scores[idx]),
                        "id": row.get("id", ""),
                    })
            return results
        except Exception as e:
            logger.error(f"TranscriptVectorStore keyword_search error: {e}")
            return []

    def hybrid_search(self, table_name: str, query: str, k: int = 20, semantic_weight: float = 0.4, bm25_weight: float = 0.6, filters: Optional[Dict[str, Any]] = None, enhance_query: bool = True) -> List[Dict[str, Any]]:
        try:
            semantic_results = self.semantic_search(table_name, query, k=k*2, filters=filters)
            keyword_results = self.keyword_search(table_name, query, k=k*2, filters=filters, enhance_query=enhance_query)
            
            # Defensive type checking for search results
            if not isinstance(semantic_results, list):
                logger.error(f"Semantic search returned non-list type: {type(semantic_results)} = {semantic_results}")
                semantic_results = []
            if not isinstance(keyword_results, list):
                logger.error(f"Keyword search returned non-list type: {type(keyword_results)} = {keyword_results}")
                keyword_results = []
            
            def normalize(results: List[Dict[str, Any]], field: str) -> Dict[str, float]:
                if not results:
                    return {}
                vals = [r[field] for r in results if field in r]
                if not vals:
                    return {}
                mx, mn = max(vals), min(vals)
                rng = (mx - mn) or 1.0
                return {r["id"]: (r[field] - mn) / rng for r in results if field in r}
            sem_scores = normalize(semantic_results, "similarity_score")
            bm_scores = normalize(keyword_results, "bm25_score")
            # Content quality filter patterns
            def is_low_quality_content(content: str) -> bool:
                """Filter out operator announcements and other low-quality content"""
                content_lower = content.lower().strip()
                
                # Operator announcement patterns
                operator_patterns = [
                    r"our next question is from",
                    r"next question.*from",
                    r"operator.*:",
                    r"thank you.*next",
                    r"we'll take.*question.*from",
                    r"question.*from.*with",
                    r"^q \(operator\):",
                    r"^thank you\. *$",
                    r"thank you\. our next",
                ]
                
                for pattern in operator_patterns:
                    if re.search(pattern, content_lower):
                        return True
                
                # Very short content (likely not substantial)
                if len(content.strip()) < 50:
                    return True
                    
                return False
            
            all_docs: Dict[str, Dict[str, Any]] = {}
            for r in semantic_results:
                all_docs[r["id"]] = dict(r)
                base_score = semantic_weight * sem_scores.get(r["id"], 0.0)
                
                # Apply content quality penalty
                content = r.get("content", "")
                if is_low_quality_content(content):
                    logger.debug(f"Content quality penalty applied to: {content[:100]}...")
                    base_score *= 0.1  # Heavy penalty for low-quality content
                
                all_docs[r["id"]]["hybrid_score"] = base_score
                
            for r in keyword_results:
                if r["id"] in all_docs:
                    keyword_score = bm25_weight * bm_scores.get(r["id"], 0.0)
                    
                    # Apply content quality penalty
                    content = r.get("content", "")
                    if is_low_quality_content(content):
                        logger.debug(f"Content quality penalty applied (BM25-existing): {content[:100]}...")
                        keyword_score *= 0.1
                    
                    all_docs[r["id"]]["hybrid_score"] += keyword_score
                    all_docs[r["id"]]["bm25_score"] = r.get("bm25_score")
                else:
                    all_docs[r["id"]] = dict(r)
                    keyword_score = bm25_weight * bm_scores.get(r["id"], 0.0)
                    
                    # Apply content quality penalty
                    content = r.get("content", "")
                    if is_low_quality_content(content):
                        logger.debug(f"Content quality penalty applied (BM25-new): {content[:100]}...")
                        keyword_score *= 0.1
                    
                    all_docs[r["id"]]["hybrid_score"] = keyword_score
            
            return sorted(all_docs.values(), key=lambda x: x["hybrid_score"], reverse=True)[:k]
        except Exception as e:
            logger.error(f"TranscriptVectorStore hybrid_search error: {e}")
            return []

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        try:
            if table_name not in self.db.table_names():
                return {"exists": False}
            table = self.db.open_table(table_name)
            df = table.to_pandas()
            sample_meta = {}
            if len(df) > 0 and "metadata" in df.columns:
                try:
                    sample_meta = json.loads(df["metadata"].iloc[0])
                except Exception:
                    sample_meta = {}
            return {
                "exists": True,
                "document_count": int(len(df)),
                "schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_metadata": sample_meta,
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def list_tables(self) -> List[str]:
        try:
            return self.db.table_names()
        except Exception as e:
            logger.error(f"TranscriptVectorStore list_tables error: {e}")
            return []

    def index_chunks(self, table_name: str, chunks: List[Dict[str, Any]], 
                     chunk_column: str = "text", metadata_columns: List[str] = None,
                     overwrite: bool = False) -> Dict[str, Any]:
        """
        Compatibility method that wraps index_documents for chunk-based ingestion.
        
        Args:
            table_name: Name of the table to store chunks
            chunks: List of chunk dictionaries with text and metadata
            chunk_column: Column name containing the text (default: "text")
            metadata_columns: List of metadata columns to include (for compatibility, ignored)
            overwrite: Whether to overwrite existing table
            
        Returns:
            Dictionary with indexing results
        """
        try:
            # Convert chunks to documents format expected by index_documents
            documents = []
            for chunk in chunks:
                doc = {
                    "id": chunk.get("id", f"chunk_{len(documents)}"),
                    chunk_column: chunk.get(chunk_column, chunk.get("text", "")),
                    "metadata": chunk.get("metadata", {})
                }
                documents.append(doc)
            
            return self.index_documents(
                table_name=table_name,
                documents=documents,
                text_field=chunk_column,
                overwrite=overwrite
            )
        except Exception as e:
            logger.error(f"TranscriptVectorStore index_chunks error: {e}")
            return {"success": False, "error": str(e)}

    def get_filter_summary(self, table_name: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return unique values for common transcript filter columns and basic date ranges.
        Useful for LLMs to propose valid filters for transcript search.
        """
        try:
            if table_name not in self.db.table_names():
                return {"exists": False}
            table = self.db.open_table(table_name)
            df = table.to_pandas()
            
            # Optionally scope to a filtered subset to get context-aware uniques
            f = filters or {}
            if not df.empty and f:
                def eq(col, val, normalize=None):
                    nonlocal df
                    if col in df.columns and val is not None:
                        v = normalize(val) if normalize else val
                        df = df[df[col].astype(str) == str(v)]
                eq("ticker", f.get("ticker"), normalize=lambda x: str(x).upper())
                eq("quarter", f.get("quarter"), normalize=lambda x: str(x).upper())
                eq("fiscal_year", f.get("fiscal_year"), normalize=lambda x: str(x))
            
            if df.empty:
                return {"exists": True, "document_count": 0}
            
            def uniques(col: str, normalize=None, limit: int = 20):
                if col not in df.columns:
                    return []
                series = df[col].dropna().astype(str)
                if normalize:
                    series = series.map(normalize)
                vals = sorted(series.unique().tolist())
                return vals[:limit]
            
            def date_range(col: str):
                if col not in df.columns or df[col].dropna().empty:
                    return {"min": None, "max": None}
                s = df[col].dropna().astype(str)
                try:
                    return {"min": s.min(), "max": s.max()}
                except Exception:
                    return {"min": None, "max": None}
            
            # Combine quarter and fiscal year for proper display
            def get_combined_quarters():
                """Get quarters in proper Q1 2024 format"""
                if "quarter" not in df.columns or "fiscal_year" not in df.columns:
                    return []
                
                # Get unique combinations of quarter and fiscal year
                quarter_year_combos = df[["quarter", "fiscal_year"]].dropna()
                if quarter_year_combos.empty:
                    return []
                
                combined_quarters = []
                for _, row in quarter_year_combos.drop_duplicates().iterrows():
                    quarter = str(row["quarter"]).upper()
                    fiscal_year = str(row["fiscal_year"])
                    
                    # Normalize quarter format to Q1, Q2, Q3, Q4
                    if quarter.isdigit():
                        quarter = f"Q{quarter}"
                    elif not quarter.startswith("Q"):
                        # Handle other formats
                        if quarter in ["1", "2", "3", "4"]:
                            quarter = f"Q{quarter}"
                    
                    combined_quarters.append(f"{quarter} {fiscal_year}")
                
                # Sort by year and quarter (most recent first)
                def sort_key(q):
                    try:
                        parts = q.split()
                        if len(parts) == 2:
                            quarter_num = int(parts[0][1:]) if parts[0].startswith("Q") else 0
                            year = int(parts[1])
                            return (year, quarter_num)
                    except:
                        pass
                    return (0, 0)
                
                return sorted(list(set(combined_quarters)), key=sort_key, reverse=True)
            
            combined_quarters = get_combined_quarters()
            
            # Simple speaker list - no classification needed
            all_speakers = []
            if "speaker" in df.columns:
                unique_speakers = df["speaker"].dropna().unique()
                all_speakers = [str(speaker) for speaker in unique_speakers][:15]  # Limit to top 15
            
            return {
                "exists": True,
                "document_count": int(len(df)),
                "quarters": combined_quarters,  # Now in Q1 2024 format
                "fiscal_years": uniques("fiscal_year", normalize=lambda x: str(x)),
                "transcript_types": uniques("transcript_type"),
                # "section_names": removed - not filtering by sections
                "speakers": all_speakers,  # Simple list of all speakers, no classification
                "transcript_date_range": date_range("transcript_date"),
                "content_stats": {
                    "avg_content_length": int(df["content_length"].mean()) if "content_length" in df.columns and not df["content_length"].isna().all() else None,
                    "avg_word_count": int(df["word_count"].mean()) if "word_count" in df.columns and not df["word_count"].isna().all() else None,
                    "financial_content_ratio": float(df["contains_financial_terms"].mean()) if "contains_financial_terms" in df.columns and not df["contains_financial_terms"].isna().all() else None
                }
            }
        except Exception as e:
            logger.error(f"TranscriptVectorStore get_filter_summary error: {e}")
            return {"exists": False, "error": str(e)}