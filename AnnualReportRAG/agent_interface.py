"""Agent-friendly interface for AnnualReportRAG package
Provides simplified index and search functions optimized for agent consumption
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date

try:
    from .get_filing_chunks import get_filing_chunks_api
except Exception:  # pragma: no cover
    # Fallback import path if relative import fails in ad-hoc runs
    from get_filing_chunks import get_filing_chunks_api  # type: ignore

logger = logging.getLogger(__name__)

# Try to import vector database functionality
try:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    if str(current_dir / "AgentSystem") not in sys.path:
        sys.path.append(str(current_dir / "AgentSystem"))
    from AgentSystem.vector_store import VectorStore
    from AgentSystem.config import config
    VECTOR_DB_AVAILABLE = True
except ImportError:
    VECTOR_DB_AVAILABLE = False
    logger.warning("Vector database not available, using fallback search")


def index_reports_for_agent(ticker: str, years_back: int = 2, 
                          filing_types: Optional[List[str]] = None, start_date: Optional[str] = None) -> Dict[str, Any]:
    """Index annual reports by fetching fresh data from SEC API and storing in vector database"""
    try:
        filing_types = filing_types or ["10-K", "10-Q"]
        start_date = start_date or date.today().isoformat()
        
        # Always fetch fresh data from API
        api_result = get_filing_chunks_api(
            tickers=[ticker], 
            start_date=start_date, 
            years_back=years_back, 
            filing_types=filing_types, 
            chunk_size=800, 
            overlap=150, 
            return_full_data=True
        )
        
        if api_result.get("status") != "success":
            return _error(api_result.get("message", "Processing failed"))
        
        chunks, filings_used = _extract_chunks_from_api(api_result, target_ticker=ticker)
        
        result = {
            "success": True, 
            "ticker": ticker.upper(), 
            "chunk_count": len(chunks), 
            "filings_used": filings_used, 
            "chunks": chunks, 
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "years_back": years_back, 
                "filing_types": filing_types,
                "data_source": "fresh_api_call"
            }
        }
        
        # Store in vector database if available
        if VECTOR_DB_AVAILABLE and chunks:
            try:
                vector_store = VectorStore()
                table_name = f"annual_reports_{ticker.lower()}"
                
                # Prepare documents for vector storage
                documents = []
                for chunk in chunks:
                    # Use 'text' or 'content' field
                    content = chunk.get("text") or chunk.get("content", "")
                    if content:
                        doc = {
                            "id": f"{ticker}_{chunk.get('chunk_id', len(documents))}",
                            "content": content,
                            "metadata": chunk.get("metadata", {})
                        }
                        # Add ticker to metadata
                        doc["metadata"]["ticker"] = ticker.upper()
                        documents.append(doc)
                
                vector_result = vector_store.index_documents(
                    table_name=table_name,
                    documents=documents,
                    text_field="content",
                    overwrite=False
                )
                
                result["vector_storage"] = vector_result
                if vector_result.get("success"):
                    logger.info(f"Indexed {len(documents)} chunks in vector store table '{table_name}'")
                
            except Exception as e:
                logger.warning(f"Failed to store in vector database: {e}")
                result["vector_storage"] = {"success": False, "error": str(e)}
        
        return result
        
    except Exception as e:
        logger.exception("Error in index_reports_for_agent")
        return _error(str(e))


def search_report_for_agent(ticker: str, query: str, k: int = 20, filters: Optional[Dict[str, Any]] = None,
                          years_back: int = 2, filing_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """Search annual reports using vector database with hybrid search and metadata filtering"""
    try:
        # Try vector database search first
        if VECTOR_DB_AVAILABLE and query:
            try:
                vector_store = VectorStore()
                table_name = f"annual_reports_{ticker.lower()}"
                
                # Check if table exists and has data
                table_info = vector_store.get_table_info(table_name)
                if table_info.get("exists") and table_info.get("document_count", 0) > 0:
                    # Use hybrid search with vector database
                    vector_results = vector_store.hybrid_search(
                        table_name=table_name,
                        query=query,
                        k=k,
                        semantic_weight=0.7,
                        bm25_weight=0.3,
                        filters=filters
                    )
                    
                    if vector_results:
                        # Apply metadata-based filtering
                        if filters:
                            filtered_results = []
                            for result in vector_results:
                                if _matches_filters_vector(result, filters):
                                    filtered_results.append(result)
                            vector_results = filtered_results
                        
                        logger.info(f"Used vector hybrid search, found {len(vector_results)} results")
                        return {
                            "success": True,
                            "ticker": ticker.upper(),
                            "query": query,
                            "results": vector_results,
                            "returned": len(vector_results),
                            "total_candidates": table_info.get("document_count", 0),
                            "search_method": "vector_hybrid",
                            "metadata": {
                                "k": k,
                                "filters": filters,
                                "generated_at": datetime.now().isoformat(),
                                "data_source": "vector_database",
                                "table_name": table_name
                            }
                        }
                
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to API search: {e}")
        
        # Fallback to fresh API search
        idx = index_reports_for_agent(ticker=ticker, years_back=years_back, filing_types=filing_types)
        if not idx.get("success"):
            return idx
        chunks = idx.get("chunks", [])
        
        # Apply filters
        filters = filters or {}
        filtered = [c for c in chunks if _matches_filters(c, filters)]
        
        if not query:
            top = filtered[:k]
        else:
            # Simple keyword scoring
            q, terms = query.lower(), [t for t in query.lower().replace("\n", " ").split(" ") if t]
            scored = []
            for c in filtered:
                text = (c.get("text", "") or c.get("content", "")).lower()
                tf = sum(text.count(t) for t in terms)  # Term frequency
                phrase_bonus = 3 if q in text else 0  # Exact phrase match bonus
                score = tf + phrase_bonus
                if score > 0:
                    scored.append((score, c))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [c for _, c in scored[:k]]
        
        return {
            "success": True, 
            "ticker": ticker.upper(), 
            "query": query, 
            "results": top, 
            "returned": len(top), 
            "total_candidates": len(filtered), 
            "search_method": "fallback_api",
            "metadata": {
                "k": k, 
                "filters": filters, 
                "generated_at": datetime.now().isoformat(),
                "data_source": "fresh_api_call"
            }
        }
    except Exception as e:
        logger.exception("Error in search_report_for_agent")
        return _error(str(e))


def _matches_filters(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Helper function for basic metadata filtering (API chunks)"""
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    if filters.get("section_name") and (meta.get("section_name", "") or "").lower() != str(filters["section_name"]).lower():
        return False
    if filters.get("form_type") and (meta.get("form_type", "") or "").upper() != str(filters["form_type"]).upper():
        return False
    return True


def _matches_filters_vector(result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Helper function for metadata filtering (vector search results)"""
    if not filters:
        return True
    meta = result.get("metadata", {})
    if filters.get("section_name") and (meta.get("section_name", "") or "").lower() != str(filters["section_name"]).lower():
        return False
    if filters.get("form_type") and (meta.get("form_type", "") or "").upper() != str(filters["form_type"]).upper():
        return False
    return True


def _extract_chunks_from_api(api_result: Dict[str, Any], target_ticker: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract chunks and filing metadata from API result"""
    chunks: List[Dict[str, Any]] = []
    filings_used: List[Dict[str, Any]] = []

    for filing in api_result.get("successful_filings", []):
        if filing.get("ticker", "").upper() != target_ticker.upper():
            continue
        rag = filing.get("rag_dataset") or {}
        chunks.extend(rag.get("chunks", []))

        meta = rag.get("filing_metadata", {})
        filings_used.append({
            "form_type": filing.get("filing_type") or meta.get("form_type"),
            "filing_date": str(filing.get("filing_date") or meta.get("filing_date")),
            "fiscal_year": filing.get("fiscal_year") or meta.get("fiscal_year"),
            "fiscal_quarter": meta.get("fiscal_quarter"),
            "company_name": filing.get("company_name") or meta.get("company_name"),
            "chunk_count": rag.get("chunk_count")
        })

    return chunks, filings_used


def _error(message: str) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "success": False,
        "error": message,
        "chunks": [],
        "filings_used": [],
    }