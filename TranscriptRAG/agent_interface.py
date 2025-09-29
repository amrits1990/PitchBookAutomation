"""Agent-Ready Interface for TranscriptRAG

This is the primary and only agent interface for TranscriptRAG package.
Provides standardized, production-ready functions for agentic frameworks.

Features:
  - Standardized AgentResponse format for all functions
  - Comprehensive error handling with error codes
  - Input validation and sanitization
  - Vector database integration with intelligent caching
  - LLM-powered query refinement
  - Metadata enrichment and auto-cleanup
  - Environment-driven configuration

Agent Functions:
  - index_transcripts_for_agent(): Index transcripts into vector store
  - search_transcripts_for_agent(): Search indexed transcripts
  - evaluate_transcript_search_results_with_llm(): Evaluate search quality
"""

import os
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date, timedelta
import re
import hashlib
from pathlib import Path
import pickle
import requests
from dataclasses import dataclass

# Load .env file from TranscriptRAG directory (optional)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass

# Environment-driven configuration (with safe defaults)
try:
    DEFAULT_CHUNK_SIZE = int(os.getenv("TRANSCRIPT_RAG_CHUNK_SIZE", "800"))
except ValueError:
    DEFAULT_CHUNK_SIZE = 800
try:
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("TRANSCRIPT_RAG_CHUNK_OVERLAP", "150"))
except ValueError:
    DEFAULT_CHUNK_OVERLAP = 150
ENABLE_AUTO_CLEANUP = os.getenv("TRANSCRIPT_RAG_ENABLE_AUTO_CLEANUP", "true").lower() in ("1", "true", "yes", "y")
ENABLE_VECTOR_CACHE = os.getenv("TRANSCRIPT_RAG_ENABLE_VECTOR_CACHE", "true").lower() in ("1", "true", "yes", "y")
ENABLE_LLM_REFINEMENT = os.getenv("TRANSCRIPT_RAG_ENABLE_LLM_REFINEMENT", "false").lower() in ("1", "true", "yes", "y")

# Import core transcript processing
try:
    from .get_transcript_chunks import get_transcript_chunks as get_transcript_chunks_api
except ImportError:
    try:
        from get_transcript_chunks import get_transcript_chunks as get_transcript_chunks_api
    except ImportError as e:
        print(f"Warning: Failed to import transcript processing modules: {e}")
        get_transcript_chunks_api = None

logger = logging.getLogger(__name__)

# Try to import vector database functionality
try:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from AgentSystem.vector_store import VectorStore
    from AgentSystem.config import config
    VECTOR_DB_AVAILABLE = True
except ImportError as e:
    VECTOR_DB_AVAILABLE = False
    logger.warning(f"Vector database not available, using fallback search: {e}")

# Try to import transcript-specific vector store
try:
    from .transcript_vector_store import TranscriptVectorStore
except ImportError:
    try:
        from transcript_vector_store import TranscriptVectorStore
    except ImportError as e:
        TranscriptVectorStore = None

# ============================================================================
# STANDARDIZED AGENT RESPONSE FORMAT
# ============================================================================

@dataclass
class AgentError:
    """Standardized error information for agents"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass 
class AgentMetadata:
    """Standardized metadata for agent responses"""
    function: str
    timestamp: str
    request_id: str
    processing_time_ms: Optional[float] = None
    
@dataclass
class AgentResponse:
    """Standardized response format for all agent functions"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[AgentError] = None
    metadata: Optional[AgentMetadata] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "success": self.success,
            "data": self.data,
            "metadata": {
                "function": self.metadata.function,
                "timestamp": self.metadata.timestamp,
                "request_id": self.metadata.request_id
            } if self.metadata else None
        }
        
        if self.error:
            result["error"] = {
                "code": self.error.code,
                "message": self.error.message,
                "details": self.error.details
            }
        
        if self.metadata and self.metadata.processing_time_ms:
            result["metadata"]["processing_time_ms"] = self.metadata.processing_time_ms
            
        return result

# ============================================================================
# ERROR CODES FOR AGENTS
# ============================================================================

class ErrorCodes:
    """Standardized error codes for agent responses"""
    INVALID_INPUT = "INVALID_INPUT"
    API_ERROR = "API_ERROR" 
    PROCESSING_ERROR = "PROCESSING_ERROR"
    VECTOR_DB_ERROR = "VECTOR_DB_ERROR"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"
    CONFIG_ERROR = "CONFIG_ERROR"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# ============================================================================
# HELPER FUNCTIONS FOR AGENT RESPONSES
# ============================================================================

def create_agent_response(
    success: bool,
    function_name: str,
    data: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None,
    error_message: Optional[str] = None,
    error_details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    processing_time_ms: Optional[float] = None
) -> Dict[str, Any]:
    """Create standardized agent response"""
    
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    metadata = AgentMetadata(
        function=function_name,
        timestamp=datetime.now().isoformat(),
        request_id=request_id,
        processing_time_ms=processing_time_ms
    )
    
    error = None
    if not success and error_code:
        error = AgentError(
            code=error_code,
            message=error_message or "Unknown error",
            details=error_details
        )
    
    response = AgentResponse(
        success=success,
        data=data,
        error=error,
        metadata=metadata
    )
    
    return response.to_dict()

def validate_agent_inputs(**kwargs) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Validate common agent inputs. Returns (error_code, message, details) if invalid, None if valid"""
    
    # Validate ticker
    ticker = kwargs.get('ticker')
    if not ticker or not isinstance(ticker, str):
        return ErrorCodes.INVALID_INPUT, "ticker must be a non-empty string", {"parameter": "ticker"}
    
    if len(ticker.strip()) == 0 or len(ticker.strip()) > 10:
        return ErrorCodes.INVALID_INPUT, "ticker must be 1-10 characters", {"parameter": "ticker", "value": ticker}
    
    # Validate quarters_back if provided
    quarters_back = kwargs.get('quarters_back')
    if quarters_back is not None:
        if not isinstance(quarters_back, int) or quarters_back < 1 or quarters_back > 20:
            return ErrorCodes.INVALID_INPUT, "quarters_back must be an integer between 1 and 20", {"parameter": "quarters_back", "value": quarters_back}
    
    # Validate query if provided
    query = kwargs.get('query')
    if query is not None:
        if not isinstance(query, str):
            return ErrorCodes.INVALID_INPUT, "query must be a string", {"parameter": "query"}
        if len(query.strip()) == 0:
            return ErrorCodes.INVALID_INPUT, "query cannot be empty", {"parameter": "query"}
        if len(query) > 1000:
            return ErrorCodes.INVALID_INPUT, "query must be less than 1000 characters", {"parameter": "query", "length": len(query)}
    
    # Validate k if provided
    k = kwargs.get('k')
    if k is not None:
        if not isinstance(k, int) or k < 1 or k > 100:
            return ErrorCodes.INVALID_INPUT, "k must be an integer between 1 and 100", {"parameter": "k", "value": k}
    
    return None

# ============================================================================
# MAIN AGENT FUNCTIONS
# ============================================================================

def index_transcripts_for_agent(
    ticker: str, 
    quarters_back: int = 4, 
    force_refresh: bool = False,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Index earnings call transcripts into vector database for agent use
    
    This is the primary indexing function for agents. It fetches transcript data
    from Alpha Vantage API, processes it into chunks, and stores in vector database.
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        quarters_back: Number of quarters to process (1-20, default: 4)
        force_refresh: Bypass cache and reprocess all data (default: False)
        request_id: Optional request tracking ID
        
    Returns:
        Standardized AgentResponse with:
        - success: bool
        - data: {chunk_count, transcripts_used, vector_storage, etc.}
        - error: {code, message, details} if failed
        - metadata: {function, timestamp, request_id, processing_time}
    """
    start_time = datetime.now()
    function_name = "index_transcripts_for_agent"
    
    # Input validation
    validation_error = validate_agent_inputs(ticker=ticker, quarters_back=quarters_back)
    if validation_error:
        return create_agent_response(
            success=False,
            function_name=function_name,
            error_code=validation_error[0],
            error_message=validation_error[1],
            error_details=validation_error[2],
            request_id=request_id
        )
    
    try:
        # Sanitize inputs
        ticker = ticker.upper().strip()
        
        # Check required dependencies
        if get_transcript_chunks_api is None:
            return create_agent_response(
                success=False,
                function_name=function_name,
                error_code=ErrorCodes.CONFIG_ERROR,
                error_message="Transcript processing not available",
                error_details={"missing_dependency": "get_transcript_chunks_api"},
                request_id=request_id
            )
        
        # Check API configuration - support multiple keys
        api_keys_available = []
        
        # Check for new multiple key format
        key1 = os.getenv("ALPHA_VANTAGE_API_KEY_1")
        key2 = os.getenv("ALPHA_VANTAGE_API_KEY_2")
        
        # Also check legacy format for backwards compatibility
        legacy_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        if key1:
            api_keys_available.append("ALPHA_VANTAGE_API_KEY_1")
        if key2:
            api_keys_available.append("ALPHA_VANTAGE_API_KEY_2")
        if legacy_key and not key1 and not key2:  # Only use legacy if new keys not available
            api_keys_available.append("ALPHA_VANTAGE_API_KEY")
        
        if not api_keys_available:
            return create_agent_response(
                success=False,
                function_name=function_name,
                error_code=ErrorCodes.CONFIG_ERROR,
                error_message="Alpha Vantage API key not configured. Set ALPHA_VANTAGE_API_KEY_1 and/or ALPHA_VANTAGE_API_KEY_2",
                error_details={
                    "required_env_vars": ["ALPHA_VANTAGE_API_KEY_1", "ALPHA_VANTAGE_API_KEY_2"],
                    "legacy_support": "ALPHA_VANTAGE_API_KEY"
                },
                request_id=request_id
            )
        
        logger.info(f"Using {len(api_keys_available)} Alpha Vantage API key(s): {', '.join(api_keys_available)}")
        
        # Calculate start date (simple calculation for quarters back)
        months_back = quarters_back * 3
        start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
        
        logger.info(f"Fetching transcripts for {ticker} (last {quarters_back} quarters)")
        
        # Get transcript chunks
        api_kwargs = {
            'tickers': [ticker], 
            'start_date': start_date,
            'quarters_back': quarters_back, 
            'chunk_size': DEFAULT_CHUNK_SIZE, 
            'overlap': DEFAULT_CHUNK_OVERLAP,
            'return_full_data': True,
            'use_speaker_chunking': True
        }
        
        result = get_transcript_chunks_api(**api_kwargs)
        
        if result.get("status") != "success":
            error_msg = result.get("message", "Failed to fetch transcripts")
            logger.error(f"Transcript fetching failed for {ticker}: {error_msg}")
            return create_agent_response(
                success=False,
                function_name=function_name,
                error_code=ErrorCodes.API_ERROR,
                error_message=f"Failed to fetch transcripts: {error_msg}",
                error_details={"api_response": result},
                request_id=request_id
            )
        
        # Extract and enrich chunks
        all_chunks = []
        transcripts_used = []
        
        for transcript in result.get("successful_transcripts", []):
            transcript_data = transcript.get("transcript_dataset", {})
            chunks = transcript_data.get("chunks", [])
            
            if chunks:
                # Simple metadata enrichment
                enriched_chunks = []
                for chunk in chunks:
                    enriched_chunk = chunk.copy()
                    # Ensure required metadata fields
                    if "metadata" not in enriched_chunk:
                        enriched_chunk["metadata"] = {}
                    
                    enriched_chunk["metadata"].update({
                        "ticker": ticker,
                        "quarter": transcript.get("quarter", "Unknown"),
                        "fiscal_year": transcript.get("fiscal_year", "Unknown"),
                        "transcript_date": transcript.get("transcript_date", "Unknown"),
                        "content_type": "transcript",
                        "transcript_type": transcript.get("transcript_type", "earnings_call")
                    })
                    enriched_chunks.append(enriched_chunk)
                
                all_chunks.extend(enriched_chunks)
                
                transcripts_used.append({
                    "ticker": transcript.get("ticker", ticker),
                    "quarter": transcript.get("quarter", "Unknown"),
                    "fiscal_year": transcript.get("fiscal_year", "Unknown"), 
                    "transcript_date": transcript.get("transcript_date", "Unknown"),
                    "chunk_count": len(enriched_chunks),
                    "transcript_type": transcript.get("transcript_type", "earnings_call")
                })
        
        if not all_chunks:
            return create_agent_response(
                success=False,
                function_name=function_name,
                error_code=ErrorCodes.INSUFFICIENT_DATA,
                error_message=f"No transcript data found for {ticker}",
                error_details={"ticker": ticker, "quarters_back": quarters_back},
                request_id=request_id
            )
        
        # Store in vector database
        vector_storage_result = {"success": False, "error": "Vector database not available"}
        
        try:
            # Try TranscriptVectorStore first, then fallback to VectorStore
            vector_store = None
            if TranscriptVectorStore is not None:
                vector_store = TranscriptVectorStore()
            elif VECTOR_DB_AVAILABLE:
                vector_store = VectorStore()
            
            if vector_store:
                table_name = f"transcripts_{ticker.lower()}"
                
                # Prepare documents for TranscriptVectorStore
                documents = []
                for i, chunk in enumerate(all_chunks):
                    doc = {
                        "id": f"{ticker.lower()}_{i}",
                        "content": chunk.get("content", chunk.get("text", "")),
                        "metadata": chunk.get("metadata", {})
                    }
                    documents.append(doc)
                
                # Store chunks with metadata using the correct method
                if hasattr(vector_store, 'index_documents'):
                    # Use TranscriptVectorStore method
                    store_result = vector_store.index_documents(
                        table_name=table_name,
                        documents=documents,
                        text_field="content",
                        overwrite=False
                    )
                elif hasattr(vector_store, 'add_documents'):
                    # Use standard VectorStore method
                    store_result = vector_store.add_documents(
                        table_name=table_name,
                        documents=[chunk.get("content", chunk.get("text", "")) for chunk in all_chunks],
                        metadatas=[chunk.get("metadata", {}) for chunk in all_chunks],
                        ids=[f"{ticker.lower()}_{i}" for i in range(len(all_chunks))]
                    )
                else:
                    store_result = {"success": False, "error": "Vector store method not available"}
                
                if store_result.get("success"):
                    vector_storage_result = {
                        "success": True,
                        "table_name": table_name,
                        "documents_indexed": len(all_chunks)
                    }
                    logger.info(f"Successfully indexed {len(all_chunks)} chunks for {ticker}")
                else:
                    vector_storage_result = {
                        "success": False, 
                        "error": store_result.get("error", "Unknown storage error")
                    }
                    logger.error(f"Vector storage failed for {ticker}: {vector_storage_result['error']}")
            else:
                logger.warning(f"No vector database available for {ticker}")
                        
        except Exception as ve:
            vector_storage_result = {"success": False, "error": f"Vector storage error: {str(ve)}"}
            logger.error(f"Vector storage error for {ticker}: {ve}")
        
        # Create successful response
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return create_agent_response(
            success=True,
            function_name=function_name,
            data={
                "ticker": ticker,
                "quarters_back": quarters_back,
                "chunk_count": len(all_chunks),
                "transcripts_used": transcripts_used,
                "vector_storage": vector_storage_result,
                "processing_summary": {
                    "total_transcripts": len(transcripts_used),
                    "total_chunks_generated": len(all_chunks),
                    "avg_chunks_per_transcript": len(all_chunks) / len(transcripts_used) if transcripts_used else 0
                },
                "cache_used": False
            },
            request_id=request_id,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Error indexing transcripts for {ticker}: {e}", exc_info=True)
        return create_agent_response(
            success=False,
            function_name=function_name,
            error_code=ErrorCodes.PROCESSING_ERROR,
            error_message=f"Error indexing transcripts: {str(e)}",
            error_details={"exception_type": type(e).__name__},
            request_id=request_id,
            processing_time_ms=processing_time
        )


def get_available_quarters_for_agent(
    ticker: str,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get available quarters for a ticker - Agent-friendly interface
    
    This function helps agents see what quarters are available for search
    instead of guessing with the quarters_back parameter.
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        request_id: Optional request tracking ID
        
    Returns:
        Standardized AgentResponse with:
        - success: bool
        - data: {available_quarters, total_documents, speakers, etc.}
        - error: {code, message, details} if failed
        - metadata: {function, timestamp, request_id, processing_time}
    """
    start_time = datetime.now()
    function_name = "get_available_quarters_for_agent"
    
    # Input validation
    validation_error = validate_agent_inputs(ticker=ticker)
    if validation_error:
        return create_agent_response(
            success=False,
            function_name=function_name,
            error_code=validation_error[0],
            error_message=validation_error[1],
            error_details=validation_error[2],
            request_id=request_id
        )
    
    try:
        ticker = ticker.upper().strip()
        
        # Check if vector store is available
        if TranscriptVectorStore is None:
            return create_agent_response(
                success=False,
                function_name=function_name,
                error_code=ErrorCodes.CONFIG_ERROR,
                error_message="TranscriptVectorStore not available",
                error_details={"missing_dependency": "TranscriptVectorStore"},
                request_id=request_id
            )
        
        vector_store = TranscriptVectorStore()
        table_name = f"transcripts_{ticker.lower()}"
        
        # Check if table exists
        table_info = vector_store.get_table_info(table_name)
        if not table_info.get("exists") or table_info.get("document_count", 0) == 0:
            return create_agent_response(
                success=False,
                function_name=function_name,
                error_code=ErrorCodes.NOT_FOUND,
                error_message=f"No transcript data found for {ticker}",
                error_details={
                    "ticker": ticker,
                    "table_exists": table_info.get("exists", False),
                    "document_count": table_info.get("document_count", 0),
                    "suggestion": f"Run index_transcripts_for_agent('{ticker}') first to load transcript data"
                },
                request_id=request_id
            )
        
        # Get comprehensive quarter and metadata information
        try:
            filter_summary = vector_store.get_filter_summary(table_name)
            available_quarters = filter_summary.get("quarters", [])
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return create_agent_response(
                success=True,
                function_name=function_name,
                data={
                    "ticker": ticker,
                    "total_documents": filter_summary.get("document_count", 0),
                    "available_quarters": available_quarters,
                    "quarter_count": len(available_quarters),
                    "available_fiscal_years": filter_summary.get("fiscal_years", []),
                    "transcript_types": filter_summary.get("transcript_types", []),
                    "speakers": filter_summary.get("speakers", []),  # Simple list of all speakers
                    "date_range": filter_summary.get("transcript_date_range", {}),
                    "content_stats": filter_summary.get("content_stats", {}),
                    "agent_recommendations": {
                        "suggested_quarters_recent": available_quarters[:3] if len(available_quarters) >= 3 else available_quarters,
                        "suggested_quarters_trend": available_quarters[:6] if len(available_quarters) >= 6 else available_quarters,
                        "search_tips": [
                            "Use 'quarters' parameter with specific quarters instead of 'quarters_back'",
                            "Combine recent quarters for latest insights",
                            "Use more quarters for trend analysis",
                            "Filter by speaker type for targeted results"
                        ]
                    }
                },
                request_id=request_id,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error getting filter summary for {ticker}: {e}")
            return create_agent_response(
                success=False,
                function_name=function_name,
                error_code=ErrorCodes.PROCESSING_ERROR,
                error_message=f"Could not retrieve quarter information: {str(e)}",
                error_details={"exception_type": type(e).__name__},
                request_id=request_id
            )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Error getting available quarters for {ticker}: {e}", exc_info=True)
        return create_agent_response(
            success=False,
            function_name=function_name,
            error_code=ErrorCodes.PROCESSING_ERROR,
            error_message=f"Error getting available quarters: {str(e)}",
            error_details={"exception_type": type(e).__name__},
            request_id=request_id,
            processing_time_ms=processing_time
        )


def search_transcripts_for_agent(
    ticker: str, 
    query: str, 
    quarters_back: Optional[int] = None,  # Made optional
    quarters: Optional[List[str]] = None,  # New: specific quarters parameter
    k: int = 20, 
    filters: Optional[Dict[str, Any]] = None,
    search_method: str = "vector_hybrid",
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Search indexed earnings call transcripts for agent use
    
    This is the primary search function for agents. It searches through indexed
    transcript chunks using various methods including vector similarity and keyword matching.
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        query: Search query text (max 1000 chars)
        quarters_back: (DEPRECATED) Number of quarters to search within (1-20)
        quarters: Specific quarters to search (e.g., ['Q1 2025', 'Q2 2024'])
        k: Number of results to return (1-100, default: 20)
        filters: Optional filters (quarter, section_name, speaker, etc.)
        search_method: Search method ('vector_hybrid', 'vector_semantic', 'keyword')
        request_id: Optional request tracking ID
        
    Returns:
        Standardized AgentResponse with:
        - success: bool
        - data: {results, search_metadata, agent_metadata, available_quarters, etc.}
        - error: {code, message, details} if failed
        - metadata: {function, timestamp, request_id, processing_time}
    """
    start_time = datetime.now()
    function_name = "search_transcripts_for_agent"
    
    # Input validation - handle both quarters_back and quarters parameters
    validation_kwargs = {"ticker": ticker, "query": query, "k": k}
    if quarters_back is not None:
        validation_kwargs["quarters_back"] = quarters_back
    
    validation_error = validate_agent_inputs(**validation_kwargs)
    if validation_error:
        return create_agent_response(
            success=False,
            function_name=function_name,
            error_code=validation_error[0],
            error_message=validation_error[1],
            error_details=validation_error[2],
            request_id=request_id
        )
    
    # Validate search_method
    valid_methods = ["vector_hybrid", "vector_semantic", "keyword", "bm25"]
    if search_method not in valid_methods:
        return create_agent_response(
            success=False,
            function_name=function_name,
            error_code=ErrorCodes.INVALID_INPUT,
            error_message=f"search_method must be one of: {valid_methods}",
            error_details={"parameter": "search_method", "value": search_method, "valid_values": valid_methods},
            request_id=request_id
        )
    
    try:
        # Sanitize inputs
        ticker = ticker.upper().strip()
        query = query.strip()
        
        # Try vector database search first
        try:
            # Initialize vector store
            vector_store = None
            if TranscriptVectorStore is not None:
                vector_store = TranscriptVectorStore()
            elif VECTOR_DB_AVAILABLE:
                vector_store = VectorStore()
            else:
                raise Exception("No vector database available")
            
            table_name = f"transcripts_{ticker.lower()}"
            
            # Check if table exists and has data
            table_info = vector_store.get_table_info(table_name)
            if table_info.get("exists") and table_info.get("document_count", 0) > 0:
                
                # Get available quarters information for agents
                available_quarters_info = {}
                try:
                    if hasattr(vector_store, 'get_filter_summary'):
                        filter_summary = vector_store.get_filter_summary(table_name)
                        available_quarters_info = {
                            "available_quarters": filter_summary.get("quarters", []),
                            "available_fiscal_years": filter_summary.get("fiscal_years", []),
                            "total_documents": filter_summary.get("document_count", 0),
                            "speakers": filter_summary.get("speakers", [])[:10]  # Simple list of speakers
                        }
                except Exception as e:
                    logger.warning(f"Could not get quarter information: {e}")
                    available_quarters_info = {"available_quarters": [], "note": "Quarter information unavailable"}
                
                # Build search filters with quarter selection support
                search_filters = {"ticker": ticker}
                if filters:
                    search_filters.update(filters)
                
                # Handle quarter filtering - prioritize specific quarters over quarters_back
                if quarters and len(quarters) > 0:
                    # Parse quarters in "Q1 2024" format and extract quarter and year info
                    quarter_filters = []
                    year_filters = []
                    
                    for q in quarters:
                        q_str = str(q).strip()
                        if " " in q_str:
                            # Format like "Q1 2024"
                            parts = q_str.split()
                            if len(parts) == 2:
                                quarter_part = parts[0].upper()
                                year_part = parts[1]
                                
                                # Normalize quarter format (Q1, Q2, Q3, Q4)
                                if quarter_part.startswith("Q"):
                                    quarter_filters.append(quarter_part)
                                elif quarter_part.isdigit():
                                    quarter_filters.append(f"Q{quarter_part}")
                                
                                year_filters.append(year_part)
                        else:
                            # Legacy format - just quarter
                            normalized_q = str(q).upper()
                            if normalized_q.isdigit():
                                normalized_q = f"Q{normalized_q}"
                            quarter_filters.append(normalized_q)
                    
                    # Apply filters for both quarter and fiscal year if we have year info
                    if quarter_filters:
                        search_filters["quarter"] = list(set(quarter_filters))
                    if year_filters:
                        search_filters["fiscal_year"] = list(set(year_filters))
                elif quarters_back is not None:
                    # Legacy quarters_back logic - get most recent quarters
                    available_quarters = available_quarters_info.get("available_quarters", [])
                    if available_quarters and quarters_back > 0:
                        # Take the most recent quarters_back quarters (they're already sorted)
                        selected_quarters = available_quarters[:min(quarters_back, len(available_quarters))]
                        
                        # Parse the selected quarters to extract quarter and year filters
                        quarter_filters = []
                        year_filters = []
                        
                        for q in selected_quarters:
                            if " " in q:
                                parts = q.split()
                                if len(parts) == 2:
                                    quarter_filters.append(parts[0].upper())
                                    year_filters.append(parts[1])
                        
                        if quarter_filters:
                            search_filters["quarter"] = list(set(quarter_filters))
                        if year_filters:
                            search_filters["fiscal_year"] = list(set(year_filters))
                # If neither provided, search all quarters (no quarter filter)
                
                # DEBUG: Log the search filters being applied
                logger.info(f"DEBUG - Search filters being applied: {search_filters}")
                logger.info(f"DEBUG - Selected quarters input: {quarters}")
                logger.info(f"DEBUG - Available quarters from DB: {available_quarters_info.get('available_quarters', [])}")
                
                # Execute vector search based on search method
                if hasattr(vector_store, 'hybrid_search') and search_method == "vector_hybrid":
                    search_results = vector_store.hybrid_search(
                        table_name=table_name,
                        query=query,
                        k=k,
                        filters=search_filters
                    )
                elif hasattr(vector_store, 'semantic_search') and search_method == "vector_semantic":
                    search_results = vector_store.semantic_search(
                        table_name=table_name,
                        query=query,
                        k=k,
                        filters=search_filters
                    )
                elif hasattr(vector_store, 'keyword_search') and search_method in ["keyword", "bm25"]:
                    search_results = vector_store.keyword_search(
                        table_name=table_name,
                        query=query,
                        k=k,
                        filters=search_filters
                    )
                elif hasattr(vector_store, 'search'):
                    # Fallback to generic search method
                    search_results = vector_store.search(
                        table_name=table_name,
                        query=query,
                        k=k,
                        filters=search_filters
                    )
                else:
                    raise Exception("No suitable search method available")
                
                chunks = []
                # Handle TranscriptVectorStore response format (list of dicts) vs generic VectorStore format
                if isinstance(search_results, list):
                    # TranscriptVectorStore returns list directly
                    for result in search_results:
                        # Fix: Extract proper score based on search method
                        if search_method == "vector_hybrid":
                            relevance_score = result.get("hybrid_score", 0.0)
                        elif search_method == "vector_semantic":
                            relevance_score = result.get("similarity_score", 0.0)
                        elif search_method in ["keyword", "bm25"]:
                            relevance_score = result.get("bm25_score", 0.0)
                        else:
                            # Fallback to original logic
                            relevance_score = result.get("score", result.get("distance", 0.0))
                        
                        chunk = {
                            "content": result.get("content", result.get("document", "")),
                            "metadata": result.get("metadata", {}),
                            "relevance_score": relevance_score,
                            "chunk_id": result.get("id", "")
                        }
                        chunks.append(chunk)
                elif search_results.get("success") and search_results.get("results"):
                    # Generic VectorStore format
                    for result in search_results["results"]:
                        chunk = {
                            "content": result.get("document", result.get("content", "")),
                            "metadata": result.get("metadata", {}),
                            "relevance_score": result.get("distance", result.get("score", 0.0)),
                            "chunk_id": result.get("id", "")
                        }
                        chunks.append(chunk)
                
                # Create successful response
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return create_agent_response(
                    success=True,
                    function_name=function_name,
                    data={
                        "ticker": ticker,
                        "query": query,
                        "results": chunks,
                        "returned": len(chunks),
                        "search_method": search_method,
                        "quarters_searched": quarters or (search_filters.get("quarter") if quarters_back else "all available"),
                        "quarters_back_used": quarters_back if quarters_back else None,
                        "search_metadata": {
                            "total_candidates": table_info.get("document_count", 0),
                            "filters_applied": search_filters,
                            "quarter_filtering": {
                                "method": "specific_quarters" if quarters else ("quarters_back" if quarters_back else "all_quarters"),
                                "quarters_selected": search_filters.get("quarter", [])
                            }
                        },
                        "agent_metadata": {
                            "vector_db_exists": True,
                            "vector_db_document_count": table_info.get("document_count", 0)
                        },
                        "available_quarters_info": available_quarters_info  # Add available quarters for agents
                    },
                    request_id=request_id,
                    processing_time_ms=processing_time
                )
            else:
                # No data found in vector database
                return create_agent_response(
                    success=False,
                    function_name=function_name,
                    error_code=ErrorCodes.NOT_FOUND,
                    error_message=f"No transcript data found for {ticker}",
                    error_details={
                        "ticker": ticker,
                        "vector_db_exists": table_info.get("exists", False),
                        "document_count": table_info.get("document_count", 0)
                    },
                    request_id=request_id
                )
                
        except Exception as ve:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Vector search error for {ticker}: {ve}")
            
            # Try fallback search
            try:
                fallback_result = _fallback_search_transcripts(ticker, query, quarters_back, k, filters)
                if fallback_result.get("success"):
                    return create_agent_response(
                        success=True,
                        function_name=function_name,
                        data={
                            "ticker": ticker,
                            "query": query,
                            "results": fallback_result.get("results", []),
                            "returned": fallback_result.get("returned", 0),
                            "search_method": "fallback_keyword",
                            "search_metadata": {"fallback_used": True},
                            "agent_metadata": {"vector_db_error": str(ve)}
                        },
                        request_id=request_id,
                        processing_time_ms=processing_time
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed for {ticker}: {fallback_error}")
            
            return create_agent_response(
                success=False,
                function_name=function_name,
                error_code=ErrorCodes.VECTOR_DB_ERROR,
                error_message=f"Vector search failed: {str(ve)}",
                error_details={"exception_type": type(ve).__name__},
                request_id=request_id,
                processing_time_ms=processing_time
            )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Error searching transcripts for {ticker}: {e}", exc_info=True)
        return create_agent_response(
            success=False,
            function_name=function_name,
            error_code=ErrorCodes.PROCESSING_ERROR,
            error_message=f"Error searching transcripts: {str(e)}",
            error_details={"exception_type": type(e).__name__},
            request_id=request_id,
            processing_time_ms=processing_time
        )


# ============================================================================
# FALLBACK SEARCH FUNCTION
# ============================================================================

def _fallback_search_transcripts(
    ticker: str, 
    query: str, 
    quarters_back: int = 4, 
    k: int = 20, 
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Fallback search when vector database is unavailable"""
    try:
        # Try to get raw transcript data
        if get_transcript_chunks_api is None:
            return {"success": False, "error": "Transcript processing not available"}
        
        # Calculate start date
        months_back = quarters_back * 3
        start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
        
        # Get transcript chunks
        result = get_transcript_chunks_api(
            tickers=[ticker],
            start_date=start_date,
            quarters_back=quarters_back,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
            return_full_data=True
        )
        
        if result.get("status") != "success":
            return {"success": False, "error": "Failed to fetch transcripts"}
        
        # Extract chunks from successful transcripts
        chunks = []
        for transcript in result.get("successful_transcripts", []):
            transcript_data = transcript.get("transcript_dataset", {})
            transcript_chunks = transcript_data.get("chunks", [])
            if not transcript_chunks:
                # Fallback to all_chunks if chunks is empty
                transcript_chunks = transcript_data.get("all_chunks", [])
            chunks.extend(transcript_chunks)
        
        # Simple keyword-based filtering and ranking
        query_terms = [term.lower() for term in query.lower().split() if term]
        
        # Score chunks based on query term frequency
        scored_chunks = []
        for chunk in chunks:
            content = (chunk.get("content") or chunk.get("text") or "").lower()
            score = sum(content.count(term) for term in query_terms)
            if query.lower() in content:
                score += 3  # Bonus for exact phrase match
            
            if score > 0:
                chunk["relevance_score"] = score
                scored_chunks.append(chunk)
        
        # Sort by score and limit results
        scored_chunks.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        top_chunks = scored_chunks[:k]
        
        return {
            "success": True,
            "results": top_chunks,
            "returned": len(top_chunks),
            "total_candidates": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Fallback search error for {ticker}: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# EVALUATION FUNCTION (Optional - for backward compatibility)
# ============================================================================

def evaluate_transcript_search_results_with_llm(
    query: str, 
    results: List[Dict[str, Any]], 
    ticker: str,
    context_info: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate search results quality using LLM (optional function)
    
    Args:
        query: Original search query
        results: Search results to evaluate
        ticker: Company ticker
        context_info: Additional context information
        request_id: Optional request tracking ID
        
    Returns:
        Evaluation results or error response
    """
    function_name = "evaluate_transcript_search_results_with_llm"
    
    # For now, return a simple evaluation without LLM
    # This can be enhanced later with actual LLM integration
    try:
        if not results:
            return create_agent_response(
                success=True,
                function_name=function_name,
                data={
                    "overall_quality": "no_results",
                    "relevance_score": 0.0,
                    "section_coverage": "No results to evaluate",
                    "strengths": [],
                    "weaknesses": ["No results found"],
                    "improvements": ["Try broader search terms", "Check if data is indexed"],
                    "confidence": "low"
                },
                request_id=request_id
            )
        
        # Simple heuristic evaluation
        avg_score = sum(r.get("relevance_score", 0) for r in results) / len(results)
        
        return create_agent_response(
            success=True,
            function_name=function_name,
            data={
                "overall_quality": "good" if avg_score > 0.7 else "fair" if avg_score > 0.3 else "poor",
                "relevance_score": avg_score,
                "section_coverage": f"Found {len(results)} relevant chunks",
                "strengths": ["Results returned", "Relevant content found"],
                "weaknesses": [] if avg_score > 0.5 else ["Low relevance scores"],
                "improvements": [] if avg_score > 0.7 else ["Try more specific queries"],
                "confidence": "medium"
            },
            request_id=request_id
        )
        
    except Exception as e:
        return create_agent_response(
            success=False,
            function_name=function_name,
            error_code=ErrorCodes.PROCESSING_ERROR,
            error_message=f"Evaluation failed: {str(e)}",
            request_id=request_id
        )