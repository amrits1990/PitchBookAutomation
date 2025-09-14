"""
AnnualReportRAG-specific LanceDB vector store
- Adds denormalized metadata columns for robust filtering without impacting other RAGs
- Mirrors the shared VectorStore API for index and hybrid search
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, date
from pathlib import Path
import json
import logging

import lancedb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

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


class AnnualVectorStore:
    """Specialized LanceDB store for AnnualReportRAG with denormalized metadata columns."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(config.get_vector_db_path())
        self.embedding_model_name = config.embedding_model
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self._db = None
        self._encoder = None
        self._tables: Dict[str, Any] = {}

    @property
    def db(self):
        if self._db is None:
            self._db = lancedb.connect(self.db_path)
        return self._db

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
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
                    # Common fields based on sample metadata
                    "company_name": str(md.get("company_name")) if md.get("company_name") is not None else None,
                    "ticker": str(md.get("ticker")).upper() if md.get("ticker") is not None else None,
                    "form_type": str(md.get("form_type")).upper() if md.get("form_type") is not None else None,
                    "filing_date": str(md.get("filing_date")) if md.get("filing_date") is not None else None,
                    "period_end_date": str(md.get("period_end_date")) if md.get("period_end_date") is not None else None,
                    "fiscal_year_end": str(md.get("fiscal_year_end")) if md.get("fiscal_year_end") is not None else None,
                    "fiscal_year": str(md.get("fiscal_year")) if md.get("fiscal_year") is not None else None,
                    "fiscal_quarter": str(md.get("fiscal_quarter")) if md.get("fiscal_quarter") is not None else None,
                    "section_name": str(md.get("section_name")) if md.get("section_name") is not None else None,
                    "granularity": str(md.get("granularity")) if md.get("granularity") is not None else None,
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
            logger.error(f"AnnualVectorStore index error: {e}")
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
        add_eq_or_in("ticker", filters.get("ticker"), normalize=lambda x: str(x).upper())
        add_eq_or_in("form_type", filters.get("form_type"), normalize=lambda x: str(x).upper())
        add_eq_or_in("section_name", filters.get("section_name"))
        add_eq_or_in("company_name", filters.get("company_name"))
        add_eq_or_in("granularity", filters.get("granularity"))
        add_eq_or_in("fiscal_year", filters.get("fiscal_year"), normalize=lambda x: str(x))
        # Normalize quarter to Q1..Q4
        def _norm_quarter(x: Any) -> str:
            s = str(x).strip().upper()
            num_map = {"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}
            return num_map.get(s, s)
        add_eq_or_in("fiscal_quarter", filters.get("fiscal_quarter"), normalize=_norm_quarter)
        # Date range filters (ISO strings compare lexicographically)
        def add_range(col, after_key, before_key):
            if col in cols:
                if filters.get(after_key):
                    conds.append(f"{col} >= '{esc(str(filters[after_key]))}'")
                if filters.get(before_key):
                    conds.append(f"{col} <= '{esc(str(filters[before_key]))}'")
        add_range("filing_date", "filing_date_after", "filing_date_before")
        add_range("period_end_date", "period_end_after", "period_end_before")
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
            print(f"Where clause: {where_clause}")
            search = table.search(qvec)
            if where_clause:
                try:
                    search = search.where(where_clause)
                except Exception:
                    pass
            print(f"Search object: {search}")
            results = search.limit(k).to_list()
            out = []
            for r in results:
                meta = r.get("metadata", "{}")
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                out.append({
                    "content": r.get("content", ""),
                    "metadata": meta,
                    # Convert distance to a monotonic similarity (higher is better)
                    "similarity_score": -float(r.get("_distance", 0.0)),
                    "id": r.get("id", ""),
                })
            # Removed noisy Semantic Search Results print
            return out
        except Exception as e:
            logger.error(f"AnnualVectorStore semantic_search error: {e}")
            return []

    def keyword_search(self, table_name: str, query: str, k: int = 20, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
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
            eq("form_type", f.get("form_type"), normalize=lambda x: str(x).upper())
            eq("section_name", f.get("section_name"))
            eq("company_name", f.get("company_name"))
            eq("granularity", f.get("granularity"))
            eq("fiscal_year", f.get("fiscal_year"), normalize=lambda x: str(x))
            eq("fiscal_quarter", f.get("fiscal_quarter"), normalize=lambda x: str(x).upper())
            # Date ranges
            def rng(col, after_key, before_key):
                nonlocal df
                if col in df.columns:
                    if f.get(after_key):
                        df = df[df[col].astype(str) >= str(f[after_key])]
                    if f.get(before_key):
                        df = df[df[col].astype(str) <= str(f[before_key])]
            rng("filing_date", "filing_date_after", "filing_date_before")
            rng("period_end_date", "period_end_after", "period_end_before")

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
                    results.append({
                        "content": row.get("content", ""),
                        "metadata": meta,
                        "bm25_score": float(scores[idx]),
                        "id": row.get("id", ""),
                    })
            # Removed noisy Keyword Search Results print
            return results
        except Exception as e:
            logger.error(f"AnnualVectorStore keyword_search error: {e}")
            return []

    def hybrid_search(self, table_name: str, query: str, k: int = 20, semantic_weight: float = 0.7, bm25_weight: float = 0.3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            semantic_results = self.semantic_search(table_name, query, k=k*2, filters=filters)
            keyword_results = self.keyword_search(table_name, query, k=k*2, filters=filters)
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
            all_docs: Dict[str, Dict[str, Any]] = {}
            for r in semantic_results:
                all_docs[r["id"]] = dict(r)
                all_docs[r["id"]]["hybrid_score"] = semantic_weight * sem_scores.get(r["id"], 0.0)
            for r in keyword_results:
                if r["id"] in all_docs:
                    all_docs[r["id"]]["hybrid_score"] += bm25_weight * bm_scores.get(r["id"], 0.0)
                    all_docs[r["id"]]["bm25_score"] = r.get("bm25_score")
                else:
                    all_docs[r["id"]] = dict(r)
                    all_docs[r["id"]]["hybrid_score"] = bm25_weight * bm_scores.get(r["id"], 0.0)
            return sorted(all_docs.values(), key=lambda x: x["hybrid_score"], reverse=True)[:k]
        except Exception as e:
            logger.error(f"AnnualVectorStore hybrid_search error: {e}")
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
            logger.error(f"AnnualVectorStore list_tables error: {e}")
            return []

    def get_filter_summary(self, table_name: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return unique values for common filter columns and basic date ranges.
        Useful for LLMs to propose valid filters.
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
                eq("form_type", f.get("form_type"), normalize=lambda x: str(x).upper())
                eq("fiscal_year", f.get("fiscal_year"), normalize=lambda x: str(x))
                # Normalize quarter to Q-notation
                def _norm_quarter(x: Any) -> str:
                    s = str(x).strip().upper()
                    num_map = {"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}
                    return num_map.get(s, s)
                eq("fiscal_quarter", f.get("fiscal_quarter"), normalize=_norm_quarter)
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
            return {
                "exists": True,
                "document_count": int(len(df)),
                "form_type": uniques("form_type", normalize=lambda x: x.upper()),
                "section_name": uniques("section_name"),
                "fiscal_year": uniques("fiscal_year", normalize=lambda x: str(x)),
                "fiscal_quarter": uniques("fiscal_quarter", normalize=lambda x: x.upper()),
                "granularity": uniques("granularity"),
                "filing_date_range": date_range("filing_date"),
                "period_end_date_range": date_range("period_end_date"),
            }
        except Exception as e:
            logger.error(f"AnnualVectorStore get_filter_summary error: {e}")
            return {"exists": False, "error": str(e)}
