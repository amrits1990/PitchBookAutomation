"""Agent-friendly interface for AnnualReportRAG package
Provides simplified index and search functions optimized for agent consumption
Enhanced with:
  - Dual granularity chunks (base + micro) via sentence-aware splitting
  - Intelligent vector database caching to prevent reprocessing
  - Auto-cleanup of processed files after successful vector DB indexing
  - Metadata enrichment (hash, numeric_density, tags, granularity, inferred_section)
  - LLM-powered query refinement with smart filter mapping
  - Environment-driven configuration for all features"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
import re
import unicodedata
from difflib import SequenceMatcher
import hashlib
from math import ceil
from dotenv import load_dotenv

load_dotenv()

# Environment-driven configuration (with safe defaults)
try:
    BASE_CHUNK_SIZE = int(os.getenv("ANNUAL_RAG_BASE_CHUNK_SIZE", "800"))
except ValueError:
    BASE_CHUNK_SIZE = 800
try:
    BASE_CHUNK_OVERLAP = int(os.getenv("ANNUAL_RAG_BASE_CHUNK_OVERLAP", "150"))
except ValueError:
    BASE_CHUNK_OVERLAP = 150
try:
    MICRO_TARGET_SIZE = int(os.getenv("ANNUAL_RAG_MICRO_CHUNK_SIZE", "300"))
except ValueError:
    MICRO_TARGET_SIZE = 300
try:
    MICRO_OVERLAP = int(os.getenv("ANNUAL_RAG_MICRO_CHUNK_OVERLAP", "60"))
except ValueError:
    MICRO_OVERLAP = 60
ENABLE_DUAL_GRANULARITY = os.getenv("ANNUAL_RAG_ENABLE_DUAL_GRANULARITY", "true").lower() in ("1", "true", "yes", "y")
ENABLE_AUTO_CLEANUP = os.getenv("ANNUAL_RAG_ENABLE_AUTO_CLEANUP", "true").lower() in ("1", "true", "yes", "y")
ENABLE_VECTOR_CACHE = os.getenv("ANNUAL_RAG_ENABLE_VECTOR_CACHE", "true").lower() in ("1", "true", "yes", "y")

try:
    from .enhanced_get_filing_chunks import get_filing_chunks_with_cleanup as get_filing_chunks_api
    from .filing_manager import FilingManager
    from .fiscal_year_corrector import get_time_filters_for_search, CompanyFactsHandler
except Exception:  # pragma: no cover
    # Fallback import path if relative import fails in ad-hoc runs
    try:
        from enhanced_get_filing_chunks import get_filing_chunks_with_cleanup as get_filing_chunks_api
        from filing_manager import FilingManager
        from fiscal_year_corrector import get_time_filters_for_search, CompanyFactsHandler
    except ImportError:
        # Final fallback to original function
        from get_filing_chunks import get_filing_chunks_api  # type: ignore
        FilingManager = None
        get_time_filters_for_search = None
        CompanyFactsHandler = None

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
                          filing_types: Optional[List[str]] = None, start_date: Optional[str] = None, 
                          force_refresh: bool = False) -> Dict[str, Any]:
    """Index annual reports by fetching fresh data from SEC API and storing in vector database
    Enhanced: expands base chunks into micro chunks & enriches metadata.
    Includes intelligent caching - checks vector DB for existing reports before processing.
    
    Args:
        ticker: Company ticker symbol
        years_back: Number of years to look back
        filing_types: Types of filings to process (default: ['10-K', '10-Q'])
        start_date: Start date for filing search
        force_refresh: If True, bypass cache and reprocess all reports
        
    Configuration via environment variables:
      ANNUAL_RAG_BASE_CHUNK_SIZE, ANNUAL_RAG_BASE_CHUNK_OVERLAP,
      ANNUAL_RAG_MICRO_CHUNK_SIZE, ANNUAL_RAG_MICRO_CHUNK_OVERLAP,
      ANNUAL_RAG_ENABLE_DUAL_GRANULARITY, ANNUAL_RAG_ENABLE_AUTO_CLEANUP.
    """
    try:
        # Input validation for security
        if not ticker or not isinstance(ticker, str):
            return _error("ticker must be a non-empty string")
        
        ticker = ticker.strip().upper()
        if not ticker.isalpha() or not (1 <= len(ticker) <= 5):
            return _error("ticker must be 1-5 alphabetic characters")
        
        if not isinstance(years_back, int) or not (1 <= years_back <= 10):
            return _error("years_back must be between 1 and 10")
            
        filing_types = filing_types or ["10-K", "10-Q"]
        start_date = start_date or date.today().isoformat()
        
        # Calculate and display date range for transparency
        end_dt = datetime.strptime(start_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=365 * years_back)
        print(f"📅 Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} ({years_back} years back)")
        print(f"📋 Looking for {', '.join(filing_types)} reports for {ticker.upper()} in this timeframe...")
        
        # Check cache first (unless force_refresh=True)
        print(f"🔧 Cache check conditions: force_refresh={force_refresh}, VECTOR_DB_AVAILABLE={VECTOR_DB_AVAILABLE}, ENABLE_VECTOR_CACHE={ENABLE_VECTOR_CACHE}")
        if not force_refresh and VECTOR_DB_AVAILABLE and ENABLE_VECTOR_CACHE:
            print(f"🔍 Checking vector database for existing {ticker} reports...")
            cache_result = _check_existing_reports_in_vector_db(
                ticker=ticker,
                filing_types=filing_types,
                years_back=years_back,
                start_date=start_date
            )
            
            if cache_result.get("cache_hit"):
                cache_info = cache_result.get("metadata", {}).get("cache_info", {})
                existing_reports = cache_info.get("existing_reports", {})
                print(f"📋 CACHE HIT: All requested reports found in vector DB")
                for form_type, years in existing_reports.items():
                    print(f"   ✅ {form_type}: {', '.join(years)} (cached)")
                print(f"   📊 Total cached chunks: {cache_info.get('total_cached_chunks', 0)}")
                logger.info(f"Cache hit: Found existing reports for {ticker} in vector DB")
                
                # Cleanup temporary files even during cache hits
                if ENABLE_AUTO_CLEANUP:
                    try:
                        # Note: We pass empty filings_used since no new processing occurred
                        cleanup_result = _cleanup_processed_files_after_indexing(
                            ticker=ticker,
                            filings_used=[],
                            years_back=years_back
                        )
                        cache_result["auto_cleanup"] = cleanup_result
                        if cleanup_result.get("files_removed", 0) > 0:
                            files_removed = cleanup_result['files_removed']
                            size_freed = cleanup_result.get('size_freed_mb', 0)
                            print(f"🧹 Auto-cleanup (cache hit): Removed {files_removed} temp files, freed {size_freed:.1f}MB")
                            logger.info(f"Auto-cleanup (cache hit): Removed {files_removed} temporary files")
                    except Exception as e:
                        logger.warning(f"Auto-cleanup failed during cache hit: {e}")
                        cache_result["auto_cleanup"] = {"success": False, "error": str(e)}
                
                # Add vector_storage status for cache hits
                cache_result["vector_storage"] = {
                    "success": True,
                    "source": "cache_hit",
                    "message": "Data already exists in vector database"
                }
                return cache_result
            elif cache_result.get("partial_cache_hit"):
                print(f"📋 PARTIAL CACHE HIT: Some reports found, will download missing ones")
                existing_reports = cache_result.get("existing_reports", {})
                missing_types = cache_result.get("missing_filing_types", [])
                
                # Show what we have cached
                for form_type, report_data in existing_reports.items():
                    years = report_data.get("years_covered", [])
                    chunk_count = report_data.get("chunk_count", 0)
                    print(f"   ✅ {form_type}: {', '.join(years)} ({chunk_count} chunks cached)")
                
                # Show what we need to download
                cache_info = cache_result.get("metadata", {}).get("cache_info", {})
                cache_type = cache_info.get("cache_type", "unknown")
                
                if cache_type == "partial_hit_incomplete_years":
                    coverage_analysis = cache_result.get("coverage_analysis", {})
                    missing_years_summary = []
                    for filing_type, analysis in coverage_analysis.items():
                        if analysis.get("missing_years"):
                            missing_years_summary.append(f"{filing_type}: {', '.join(analysis['missing_years'])}")
                    
                    print(f"   ⬇️  Will download: {', '.join(missing_types)} (to get missing years)")
                    if missing_years_summary:
                        print(f"       Missing years: {' | '.join(missing_years_summary)}")
                else:
                    print(f"   ⬇️  Will download: {', '.join(missing_types)}")
                
                logger.info(f"Partial cache hit: Some reports exist for {ticker}, will process missing ones")
                # Update filing_types to only process missing reports
                filing_types = cache_result.get("missing_filing_types", filing_types)
                if not filing_types:
                    # All reports are cached
                    return cache_result
            else:
                cache_reason = cache_result.get("reason", "unknown")
                if cache_reason == "no_existing_data":
                    print(f"📋 CACHE MISS: No existing reports found for {ticker}")
                    print(f"   ⬇️  Will download: {', '.join(filing_types)}")
                else:
                    print(f"📋 CACHE CHECK: {cache_reason}")
                    print(f"   ⬇️  Will download: {', '.join(filing_types)}")
        
        # Show what we're about to process
        if filing_types:
            print(f"⬇️  Downloading and processing: {', '.join(filing_types)} for {ticker} ({years_back} years back)")
        
        # Fetch data with enhanced cleanup
        api_result = get_filing_chunks_api(
            tickers=[ticker], 
            start_date=start_date, 
            years_back=years_back, 
            filing_types=filing_types, 
            chunk_size=BASE_CHUNK_SIZE, 
            overlap=BASE_CHUNK_OVERLAP, 
            return_full_data=True,
            auto_cleanup=True,  # Enable automatic cleanup
            cleanup_older_than_years=years_back + 1  # Clean files older than requested range
        )
        
        if api_result.get("status") != "success":
            return _error(api_result.get("message", "Processing failed"))
        
        chunks, filings_used = _extract_chunks_from_api(api_result, target_ticker=ticker)
        
        # Smart filing-level deduplication: Skip entire filings that are already in vector DB
        original_chunk_count = len(chunks)
        original_filing_count = len(filings_used)
        print(f"📥 Downloaded {original_chunk_count} chunks from {original_filing_count} filings, checking for duplicates...")
        
        if not force_refresh and ENABLE_VECTOR_CACHE:
            chunks, filings_used, dedup_stats = _deduplicate_filings_by_metadata(
                ticker=ticker,
                chunks=chunks, 
                filings_used=filings_used
            )
            
            filings_skipped = dedup_stats.get("filings_skipped", 0)
            chunks_skipped = dedup_stats.get("chunks_skipped", 0)
            new_chunks = len(chunks)
            new_filings = len(filings_used)
            
            if filings_skipped > 0:
                print(f"🔄 Smart deduplication: Skipped {filings_skipped}/{original_filing_count} already-cached filings "
                      f"({chunks_skipped} chunks), will process {new_filings} new filings ({new_chunks} chunks)")
                print(f"   💰 Saved embedding costs for {chunks_skipped} chunks ({chunks_skipped/original_chunk_count*100:.1f}% savings)")
            else:
                print(f"✅ All {new_filings} filings are new (no duplicates found)")
        else:
            print(f"⚠️  Deduplication disabled (force_refresh={force_refresh}, vector_cache={ENABLE_VECTOR_CACHE})")
        
        # ---- Enhancements: metadata enrichment & dual granularity ----
        enriched_chunks = _enrich_and_expand_chunks(chunks, ticker)
        
        # Check if all filings were deduplicated (no new content to process)
        if len(enriched_chunks) == 0 and original_filing_count > 0:
            print(f"📋 ALL FILINGS DEDUPLICATED: All {original_filing_count} filings already exist in vector DB")
            
            # Cleanup temporary files since no new processing is needed
            if ENABLE_AUTO_CLEANUP:
                try:
                    cleanup_result = _cleanup_processed_files_after_indexing(
                        ticker=ticker,
                        filings_used=[],  # No new filings processed
                        years_back=years_back
                    )
                    if cleanup_result.get("files_removed", 0) > 0:
                        files_removed = cleanup_result['files_removed']
                        size_freed = cleanup_result.get('size_freed_mb', 0)
                        print(f"🧹 Auto-cleanup (deduplication): Removed {files_removed} temp files, freed {size_freed:.1f}MB")
                        logger.info(f"Auto-cleanup (deduplication): Removed {files_removed} temporary files")
                except Exception as e:
                    logger.warning(f"Auto-cleanup failed during deduplication: {e}")
            
            # Return result indicating successful deduplication
            return {
                "success": True,
                "ticker": ticker.upper(),
                "chunk_count": 0,
                "filings_used": [],
                "chunks": [],
                "vector_storage": {
                    "success": True,
                    "source": "deduplication_skip",
                    "message": "All filings already exist in vector database"
                },
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "years_back": years_back,
                    "filing_types": filing_types,
                    "data_source": "deduplication_skip",
                    "cache_info": {
                        "all_filings_deduplicated": True,
                        "original_filing_count": original_filing_count,
                        "chunks_skipped": dedup_stats.get("chunks_skipped", 0) if 'dedup_stats' in locals() else 0
                    }
                }
            }
        
        result = {
            "success": True, 
            "ticker": ticker.upper(), 
            "chunk_count": len(enriched_chunks), 
            "filings_used": filings_used, 
            "chunks": enriched_chunks, 
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "years_back": years_back, 
                "filing_types": filing_types,
                "data_source": "fresh_api_call",
                "cache_info": {
                    "force_refresh": force_refresh,
                    "cache_checked": not force_refresh and VECTOR_DB_AVAILABLE,
                    "processed_filing_types": filing_types
                },
                "enhancements": {
                    "dual_granularity": ENABLE_DUAL_GRANULARITY,
                    "auto_cleanup": ENABLE_AUTO_CLEANUP,
                    "vector_cache": ENABLE_VECTOR_CACHE,
                    "embedding_model_version": getattr(globals().get('config'), 'embedding_model', 'default') if VECTOR_DB_AVAILABLE else 'default',
                    "metadata_enriched": True,
                    "base_chunk_size": BASE_CHUNK_SIZE,
                    "base_chunk_overlap": BASE_CHUNK_OVERLAP,
                    "micro_chunk_size": MICRO_TARGET_SIZE if ENABLE_DUAL_GRANULARITY else None,
                    "micro_chunk_overlap": MICRO_OVERLAP if ENABLE_DUAL_GRANULARITY else None
                }
            }
        }
        
        # Store in vector database if available
        if enriched_chunks:
            try:
                # Use AnnualReportRAG-specific store for denormalized metadata
                try:
                    try:
                        from .annual_vector_store import AnnualVectorStore
                    except ImportError:
                        # Fallback for when running from examples directory
                        from annual_vector_store import AnnualVectorStore
                    vector_store = AnnualVectorStore()
                    vector_available = True
                except Exception as e:
                    print(f"AnnualVectorStore failed: {e}")
                    try:
                        if VECTOR_DB_AVAILABLE:
                            vector_store = VectorStore()
                            vector_available = True
                        else:
                            print("AgentSystem VectorStore also unavailable")
                            vector_available = False
                    except Exception as e2:
                        print(f"VectorStore also failed: {e2}")
                        vector_available = False
                
                if not vector_available:
                    print("⚠️  No vector database available - saving chunks to JSON only")
                    result["vector_storage"] = {"success": False, "error": "No vector database available", "chunks_saved_to_json": True}
                    return result
                table_name = f"annual_reports_{ticker.lower()}"
                documents = []
                for chunk in enriched_chunks:
                    content = chunk.get("text") or chunk.get("content", "")
                    if not content:
                        continue
                    meta = chunk.get("metadata", {}).copy()
                    meta["granularity"] = chunk.get("granularity", meta.get("granularity", "base"))
                    doc = {
                        "id": f"{ticker}_{chunk.get('granularity','b')}_{chunk.get('chunk_id', len(documents))}",
                        "content": content,
                        "metadata": meta
                    }
                    documents.append(doc)
                vector_result = vector_store.index_documents(
                    table_name=table_name,
                    documents=documents,
                    text_field="content",
                    overwrite=False
                )
                result["vector_storage"] = vector_result
                if vector_result.get("success"):
                    # Merge with cached chunks if we had a partial cache hit
                    final_chunk_count = len(documents)
                    total_chunks_message = f"{len(documents)} new chunks"
                    
                    # Check if we need to merge with cached data
                    if not force_refresh and VECTOR_DB_AVAILABLE and ENABLE_VECTOR_CACHE:
                        cached_result = _check_existing_reports_in_vector_db(
                            ticker=ticker,
                            filing_types=filing_types or ["10-K", "10-Q"],
                            years_back=years_back,
                            start_date=start_date
                        )
                        if cached_result.get("partial_cache_hit") or cached_result.get("cache_hit"):
                            cached_chunks = cached_result.get("total_existing_chunks", 0)
                            final_chunk_count = len(documents) + cached_chunks
                            total_chunks_message = f"{len(documents)} new + {cached_chunks} cached = {final_chunk_count} total chunks"
                    
                    print(f"📊 Successfully indexed {total_chunks_message} in vector database")
                    
                    # Show what was processed
                    form_type_counts = {}
                    for filing in filings_used:
                        form_type = filing.get("form_type", "Unknown")
                        fiscal_year = filing.get("fiscal_year", "Unknown")
                        chunk_count = filing.get("chunk_count", 0)
                        
                        if form_type not in form_type_counts:
                            form_type_counts[form_type] = []
                        form_type_counts[form_type].append(f"{fiscal_year} ({chunk_count} chunks)")
                    
                    for form_type, details in form_type_counts.items():
                        print(f"   ✅ {form_type}: {', '.join(details)}")
                    
                    logger.info(f"Indexed {len(documents)} (enriched) chunks in vector store table '{table_name}'")
                    
                    # Auto-cleanup: Remove processed files after successful vector DB indexing
                    if ENABLE_AUTO_CLEANUP:
                        try:
                            cleanup_result = _cleanup_processed_files_after_indexing(
                                ticker=ticker,
                                filings_used=filings_used,
                                years_back=years_back
                            )
                            result["auto_cleanup"] = cleanup_result
                            if cleanup_result.get("files_removed", 0) > 0:
                                files_removed = cleanup_result['files_removed']
                                size_freed = cleanup_result.get('size_freed_mb', 0)
                                print(f"🧹 Auto-cleanup: Removed {files_removed} processed files, freed {size_freed:.1f}MB")
                                logger.info(f"Auto-cleanup: Removed {files_removed} processed files, "
                                           f"freed {size_freed:.1f}MB")
                            else:
                                print(f"🧹 Auto-cleanup: No files to remove")
                        except Exception as cleanup_error:
                            logger.warning(f"Auto-cleanup failed: {cleanup_error}")
                            result["auto_cleanup"] = {"success": False, "error": str(cleanup_error)}
                    else:
                        result["auto_cleanup"] = {"success": True, "skipped": True, "reason": "ENABLE_AUTO_CLEANUP=false"}
            except Exception as e:
                logger.warning(f"Failed to store in vector database: {e}")
                result["vector_storage"] = {"success": False, "error": str(e)}
        
        return result
        
    except Exception as e:
        logger.exception("Error in index_reports_for_agent")
        return _error(str(e))


def search_report_for_agent(ticker: str, query: str, k: int = 20, filters: Optional[Dict[str, Any]] = None,
                          time_period: str = "latest", years_back: int = 2, filing_types: Optional[List[str]] = None,
                          enable_llm_refinement: bool = True, refinement_model: Optional[str] = None,
                          fallback_to_api_on_empty: bool = False) -> Dict[str, Any]:
    """Search annual reports using vector database with hybrid search and metadata filtering
    
    Args:
        ticker: Company ticker symbol (1-5 alphabetic characters)
        query: Search query string
        k: Number of results to return (default 20)
        filters: Additional metadata filters
        time_period: Time period constraint. Options:
            - "latest": Most recent single report (10-K or 10-Q, whichever is newer by end date)
            - "latest_10k_and_10q": Most recent 10-K and 10-Q reports (both)
            - "latest_10k": Most recent 10-K only
            - "latest_10q": Most recent 10-Q only  
            - "last_n_reports": Last N reports (default N=4)
            - "last_N_reports": Last N reports (e.g., "last_3_reports", "last_5_reports")
        years_back: Years to look back (for fallback API calls)
        filing_types: Filing types to include
        enable_llm_refinement: Whether to use LLM for query refinement
        refinement_model: Model to use for refinement
        fallback_to_api_on_empty: Whether to download new reports if none found
        
    Returns:
        Dict containing search results and metadata
    """
    try:
        # Input validation for security
        if not ticker or not isinstance(ticker, str):
            return _error("ticker must be a non-empty string")
        
        ticker = ticker.strip().upper()
        if not ticker.isalpha() or not (1 <= len(ticker) <= 5):
            return _error("ticker must be 1-5 alphabetic characters")
            
        if not query or not isinstance(query, str):
            return _error("query must be a non-empty string")
        
        query = query.strip()
        if len(query) > 1000:  # Reasonable query length limit
            return _error("query must be less than 1000 characters")
        
        # Validate time_period parameter
        import re
        valid_time_periods = ["latest", "latest_10k_and_10q", "latest_10k", "latest_10q", "last_n_reports"]
        # Also accept last_N_reports pattern (e.g., last_3_reports, last_5_reports)
        last_n_pattern = re.compile(r'^last_\d+_reports$')
        
        if time_period not in valid_time_periods and not last_n_pattern.match(time_period):
            return _error(f"time_period must be one of: {valid_time_periods} or 'last_N_reports' (e.g., last_3_reports)")
            
        # Try vector database search first
        if query:
            try:
                # First try to use AnnualVectorStore
                try:
                    from .annual_vector_store import AnnualVectorStore
                    vector_store = AnnualVectorStore()
                except ImportError:
                    try:
                        # Fallback for when running from examples directory
                        from annual_vector_store import AnnualVectorStore
                        vector_store = AnnualVectorStore()
                    except ImportError:
                        # Final fallback to shared VectorStore if available
                        if VECTOR_DB_AVAILABLE:
                            vector_store = VectorStore()
                        else:
                            raise Exception("No vector database available")
                table_name = f"annual_reports_{ticker.lower()}"
                
                # Check if table exists and has data
                table_info = vector_store.get_table_info(table_name)
                if table_info.get("exists") and table_info.get("document_count", 0) > 0:
                    # Optional: LLM refinement before search
                    refined_query, refined_filters, refinement_meta = query, (filters or {}), None
                    # Always include ticker in store-level filters
                    # Initialize agent tracking flags
                    q4_conversion_applied = False
                    filters_relaxed = False
                    llm_refinement_used = False
                    
                    # Check if original filters contain Q4 before refinement
                    original_fq = (filters or {}).get("fiscal_quarter")
                    if original_fq and str(original_fq).upper() in ["Q4", "4"]:
                        q4_conversion_applied = True
                    
                    # Also check if the query text contains Q4 indicators
                    query_lower = query.lower()
                    if any(q4_term in query_lower for q4_term in ["q4", "fourth quarter", "4th quarter"]):
                        q4_conversion_applied = True
                    
                    refined_filters = {**(refined_filters or {}), "ticker": ticker.upper()}
                    
                    # Apply time-based filters based on time_period parameter
                    time_filters = _determine_time_filters(ticker, time_period, vector_store)
                    multi_report_specs = None
                    
                    if time_filters:
                        if time_filters.get("multi_report_filter"):
                            # Special handling for latest - we have specific reports to target
                            multi_report_specs = time_filters["reports"]
                            report_descriptions = []
                            for r in multi_report_specs:
                                form_type = r.get("form_type", "")
                                fiscal_year = r.get("fiscal_year", "")
                                fiscal_quarter = r.get("fiscal_quarter", "")
                                quarter_part = f"/{fiscal_quarter}" if fiscal_quarter else ""
                                report_descriptions.append(f"{form_type} {fiscal_year}{quarter_part}")
                            print(f"Applying multi-report time filters for time_period='{time_period}': {report_descriptions}")
                        else:
                            # Standard single filter
                            print(f"Applying time filters for time_period='{time_period}': {time_filters}")
                            refined_filters.update(time_filters)
                    else:
                        print(f"No time filters applied for time_period='{time_period}'")
                    
                    # Check if API key is available for refinement
                    try:
                        has_api = bool(os.getenv('OPENROUTER_API_KEY'))
                    except Exception:
                        has_api = False
                    print(f"LLM refinement enabled: {enable_llm_refinement}, api_key: {'present' if has_api else 'missing'}")
                    if enable_llm_refinement and has_api:
                        # Prepare optional samples/summary, but don't block refinement if these fail
                        sample_texts: List[str] = []
                        filter_summary: Dict[str, Any] = {}
                        try:
                            sample_results = vector_store.keyword_search(table_name=table_name, query=query, k=3, filters=refined_filters or {}) or []
                            sample_texts = [(r.get('content') or '')[:600] for r in sample_results if r.get('content')]
                        except Exception as e:
                            print(f"Refinement prep: keyword sample failed: {e}")
                        try:
                            # Scope summary to current context for better allowed section suggestions
                            ctx_filters = {
                                "ticker": ticker.upper(),
                                **({"form_type": refined_filters.get("form_type")} if refined_filters.get("form_type") else {}),
                                **({"fiscal_year": refined_filters.get("fiscal_year")} if refined_filters.get("fiscal_year") else {}),
                                **({"fiscal_quarter": refined_filters.get("fiscal_quarter")} if refined_filters.get("fiscal_quarter") else {}),
                            }
                            filter_summary = vector_store.get_filter_summary(table_name, filters=ctx_filters)  # type: ignore[attr-defined]
                        except Exception as e:
                            print(f"Refinement prep: filter summary failed: {e}")
                            filter_summary = {}
                        if filter_summary:
                            try:
                                import json
                                sample_texts = [
                                    "AVAILABLE FILTER VALUES:\n" + json.dumps(filter_summary)[:1200]
                                ] + sample_texts
                            except Exception as e:
                                print(f"Refinement prep: failed to serialize filter summary: {e}")
                        allowed_sections: List[str] = []
                        try:
                            allowed_sections = filter_summary.get("section_name", []) if filter_summary else []
                        except Exception as e:
                            print(f"Refinement prep: extract allowed sections failed: {e}")
                            allowed_sections = []
                        print(f"Calling LLM refinement with {len(sample_texts)} samples; allowed_sections={len(allowed_sections)}")
                        refined = _refine_query_with_llm(
                            ticker=ticker,
                            original_query=query,
                            current_filters=refined_filters or {},
                            sample_texts=sample_texts,
                            model=refinement_model,
                            allowed_sections=allowed_sections,
                        )
                        if not refined:
                            print("LLM refinement returned None (no update).")
                        elif refined.get('query'):
                            refined_query = refined['query']
                            if isinstance(refined.get('filters'), dict):
                                rf = refined['filters']
                                
                                # BLOCK time-related fields from LLM refinement
                                blocked_fields = ['fiscal_year', 'fiscal_quarter', 'form_type']
                                original_rf = rf.copy()
                                for field in blocked_fields:
                                    if field in rf:
                                        print(f"BLOCKED: LLM refinement attempted to modify {field}='{rf[field]}' - this is controlled by time period logic")
                                        rf.pop(field)
                                
                                # Only merge non-blocked fields
                                refined_filters = {**(refined_filters or {}), **rf}
                                # Enforce section_name to one of allowed values (with robust normalization)
                                try:
                                    if refined_filters.get("section_name"):
                                        sec_val = refined_filters["section_name"]
                                        if isinstance(sec_val, (list, tuple, set)):
                                            # Handle list of section names
                                            mapped_sections = []
                                            for s in sec_val:
                                                s_str = str(s).strip()
                                                mapped = _map_to_allowed_section(s_str, allowed_sections or [])
                                                if mapped:
                                                    if mapped != s_str:
                                                        print(f"Mapped section_name '{s_str}' -> '{mapped}'")
                                                    mapped_sections.append(mapped)
                                                else:
                                                    print(f"Could not map section_name '{s_str}' (no confident match)")
                                            if mapped_sections:
                                                # Deduplicate while preserving order
                                                seen = set(); dedup = []
                                                for m in mapped_sections:
                                                    if m not in seen:
                                                        seen.add(m); dedup.append(m)
                                                refined_filters["section_name"] = dedup
                                                print(f"Mapped section_name list -> {dedup}")
                                            else:
                                                print("Dropping section_name list (no valid mappings)")
                                                refined_filters.pop("section_name", None)
                                        else:
                                            # Handle single section name
                                            sec = str(sec_val).strip()
                                            mapped = _map_to_allowed_section(sec, allowed_sections or [])
                                            if mapped:
                                                if mapped != sec:
                                                    print(f"Mapped section_name '{sec}' -> '{mapped}'")
                                                refined_filters["section_name"] = mapped
                                            else:
                                                # If we cannot confidently map, drop it per instructions
                                                print(f"Dropping non-canonical section_name '{sec}' (no confident match)")
                                                refined_filters.pop("section_name", None)
                                    # Normalize fiscal_quarter to Q1..Q4 if numeric provided
                                    # Also fix LLM errors where form types are placed in fiscal_quarter
                                    fq = refined_filters.get("fiscal_quarter")
                                    if fq is not None:
                                        fq_str = str(fq).strip().upper()
                                        
                                        # Fix common LLM errors: form types in fiscal_quarter
                                        if fq_str in ["10-Q", "10-K"]:
                                            print(f"LLM Error: Found form type '{fq_str}' in fiscal_quarter, moving to form_type")
                                            if "form_type" not in refined_filters:
                                                refined_filters["form_type"] = fq_str
                                            refined_filters.pop("fiscal_quarter", None)
                                        else:
                                            # Normal numeric normalization
                                            num_map = {"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}
                                            if fq_str in num_map:
                                                refined_filters["fiscal_quarter"] = num_map[fq_str]
                                    
                                    # Validate section_name against allowed values and drop invalid ones
                                    section_name = refined_filters.get("section_name")
                                    if section_name:
                                        invalid_sections = ["revenue", "sales", "total revenue", "income", "profit", "cash flow", "balance sheet"]
                                        if isinstance(section_name, str) and section_name.lower() in invalid_sections:
                                            print(f"LLM Error: Removing invalid section_name '{section_name}' - not in allowed list")
                                            refined_filters.pop("section_name", None)
                                        elif isinstance(section_name, list):
                                            valid_sections = [s for s in section_name if s.lower() not in invalid_sections]
                                            if len(valid_sections) != len(section_name):
                                                removed = [s for s in section_name if s.lower() in invalid_sections]
                                                print(f"LLM Error: Removing invalid section_names {removed} - not in allowed list")
                                                if valid_sections:
                                                    refined_filters["section_name"] = valid_sections[0] if len(valid_sections) == 1 else valid_sections
                                                else:
                                                    refined_filters.pop("section_name", None)
                                    
                                    # Similar fix for form_type errors (quarters in form_type)
                                    ft = refined_filters.get("form_type")
                                    if ft is not None and isinstance(ft, (str, list)):
                                        if isinstance(ft, str) and ft.upper() in ["Q1", "Q2", "Q3", "Q4"]:
                                            print(f"LLM Error: Found quarter '{ft}' in form_type, moving to fiscal_quarter")
                                            refined_filters["fiscal_quarter"] = ft.upper()
                                            refined_filters.pop("form_type", None)
                                        elif isinstance(ft, list):
                                            quarters_found = [f for f in ft if str(f).upper() in ["Q1", "Q2", "Q3", "Q4"]]
                                            if quarters_found:
                                                print(f"LLM Error: Found quarters {quarters_found} in form_type, moving to fiscal_quarter")
                                                refined_filters["fiscal_quarter"] = quarters_found[0] if len(quarters_found) == 1 else quarters_found
                                                refined_filters["form_type"] = [f for f in ft if str(f).upper() not in ["Q1", "Q2", "Q3", "Q4"]]
                                                if not refined_filters["form_type"]:
                                                    refined_filters.pop("form_type", None)
                                except Exception as e:
                                    print(f"Refinement postprocess: section_name enforcement failed: {e}")
                            refinement_meta = {k: v for k, v in refined.items() if k not in ('query', 'filters')}
                            llm_refinement_used = True

                    # After refinement, re-scope allowed sections to the refined context and remap section_name if needed
                    try:
                        if refined_filters.get("section_name"):
                            # Use original allowed_sections for mapping, not scoped ones that might be empty
                            # due to conflicting filters (e.g., fiscal_quarter + 10-K)
                            sections_for_mapping = allowed_sections or []
                            
                            sec_val = refined_filters["section_name"]
                            if isinstance(sec_val, (list, tuple, set)):
                                mapped = []
                                for s in sec_val:
                                    s_str = str(s).strip()
                                    m = _map_to_allowed_section(s_str, sections_for_mapping)
                                    if m:
                                        mapped.append(m)
                                # Deduplicate preserving order
                                seen = set(); dedup = []
                                for m in mapped:
                                    if m not in seen:
                                        seen.add(m); dedup.append(m)
                                if dedup:
                                    refined_filters["section_name"] = dedup
                                    print(f"Post-refine mapped section_name list -> {dedup}")
                                else:
                                    print("Post-refine dropping section_name list (no valid mappings)")
                                    refined_filters.pop("section_name", None)
                            else:
                                sec0 = str(sec_val).strip()
                                sec_mapped = _map_to_allowed_section(sec0, sections_for_mapping)
                                if sec_mapped and sec_mapped != sec0:
                                    print(f"Post-refine mapped section_name '{sec0}' -> '{sec_mapped}'")
                                    refined_filters["section_name"] = sec_mapped
                                elif not sec_mapped and sections_for_mapping:
                                    print(f"Post-refine dropping section_name '{sec0}' (no valid mapping)")
                                    refined_filters.pop("section_name", None)
                    except Exception as e:
                        print(f"Post-refine section_name remap failed: {e}")

                    # Fix LLM annual query detection: if query mentions multiple years without quarters, it's annual
                    try:
                        fiscal_years = refined_filters.get("fiscal_year", [])
                        form_types = refined_filters.get("form_type", [])
                        fiscal_quarter = refined_filters.get("fiscal_quarter")
                        
                        # Extract years from the original query if LLM missed them
                        import re
                        year_matches = re.findall(r'\b20[12][0-9]\b', query)
                        if year_matches and not fiscal_years:
                            print(f"LLM missed years in query: extracting {year_matches} from query text")
                            refined_filters["fiscal_year"] = year_matches
                            fiscal_years = year_matches
                        
                        # Check if this is clearly an annual query
                        is_annual_query = False
                        if isinstance(fiscal_years, list) and len(fiscal_years) > 1:
                            # Multiple years mentioned
                            is_annual_query = True
                        elif any(term in query.lower() for term in ['annual', 'year-over-year', 'yearly', 'full year']):
                            # Annual terms in query
                            is_annual_query = True
                        
                        if is_annual_query and fiscal_quarter is None:
                            # This should be 10-K only
                            if isinstance(form_types, list) and '10-Q' in form_types and '10-K' in form_types:
                                print(f"Annual query detected: removing 10-Q from form_type (keeping only 10-K)")
                                refined_filters["form_type"] = ['10-K']
                            elif not form_types:
                                print(f"Annual query detected: setting form_type to 10-K")
                                refined_filters["form_type"] = ['10-K']
                    except Exception as e:
                        print(f"Annual query detection failed: {e}")

                    # Ensure growth/performance queries include MD&A section
                    try:
                        section_names = refined_filters.get("section_name", [])
                        if not section_names:
                            # Check if this is a growth/performance/business analysis query
                            growth_terms = ['growth', 'driver', 'performance', 'revenue', 'analysis', 'cause', 'factor', 'strategy', 'trend']
                            if any(term in query.lower() for term in growth_terms):
                                mda_section = "Management's Discussion and Analysis of Financial Condition and Results of Operations"
                                print(f"Growth/performance query detected: adding MD&A section for business insights")
                                refined_filters["section_name"] = [mda_section]
                                # Also add financial statements for supporting data
                                refined_filters["section_name"].append("Financial Statements and Supplementary Data")
                        
                        # Ensure granularity is set for growth/analysis queries
                        if not refined_filters.get("granularity"):
                            growth_terms = ['growth', 'driver', 'performance', 'analysis', 'cause', 'factor', 'strategy', 'trend', 'insight']
                            if any(term in query.lower() for term in growth_terms):
                                print(f"Analysis query detected: setting granularity to 'micro' for better business content")
                                refined_filters["granularity"] = "micro"
                    except Exception as e:
                        print(f"Section enforcement for growth queries failed: {e}")

                    # Heuristic augmentation (quarters, sections, full-year comparisons)
                    try:
                        refined_filters = _augment_multi_filters(refined_query, refined_filters)
                    except Exception as e:
                        print(f"Heuristic augmentation failed: {e}")

                    # Normalize single-element lists to scalars for cleaner equality checks
                    try:
                        refined_filters = _normalize_filter_values(refined_filters)
                    except Exception as e:
                        print(f"Filter normalization failed: {e}")

                    # Handle Q4 queries: Convert to 10-K search (Q4 data is in annual reports)
                    try:
                        fq = refined_filters.get("fiscal_quarter")
                        if fq and str(fq).upper() in ["Q4", "4"]:
                            print(f"Converting Q4 query to 10-K search (Q4 data is in annual reports)")
                            refined_filters["form_type"] = "10-K"
                            refined_filters.pop("fiscal_quarter", None)  # Remove quarter filter for annual report
                            print(f"Updated filters: form_type=10-K, removed fiscal_quarter")
                            q4_conversion_applied = True
                        
                        # Also detect if LLM already correctly set form_type to 10-K for Q4 queries
                        elif refined_filters.get("form_type") == "10-K" and not q4_conversion_applied:
                            # Check if this looks like a Q4 context
                            if any(q4_term in query.lower() for q4_term in ["q4", "fourth quarter", "4th quarter"]):
                                print(f"LLM correctly identified Q4 query and set form_type to 10-K")
                                q4_conversion_applied = True
                    except Exception as e:
                        print(f"Q4 to 10-K conversion failed: {e}")

                    # Log the final query and filters used
                    try:
                        print(f"Refined query (final): {refined_query}")
                        print(f"Filters used (final): {refined_filters}")
                    except Exception:
                        pass

                    # Enhance query for better business content prioritization
                    enhanced_query = refined_query
                    if any(term in refined_query.lower() for term in ['revenue', 'growth', 'performance', 'sales', 'income']):
                        # Add business context terms to prioritize narrative over technical accounting
                        business_terms = " management discussion analysis results operations growth drivers performance factors"
                        enhanced_query = refined_query + business_terms

                    # Multi-report balanced search: if user asks for data from multiple reports
                    # Split k chunks equally between different report types to ensure balanced coverage
                    balanced_results = []
                    form_types = refined_filters.get('form_type', [])
                    fiscal_quarters = refined_filters.get('fiscal_quarter', [])
                    
                    # Determine if we need balanced multi-report search
                    needs_balanced_search = False
                    search_configs = []
                    
                    # NEW: Check if we have specific multi-report specifications from time filters
                    if multi_report_specs:
                        needs_balanced_search = True
                        for report_spec in multi_report_specs:
                            # Create config for each specific report (e.g., 10-K 2024, 10-Q 2025/Q3)
                            config = {k: v for k, v in refined_filters.items() if k not in ('form_type', 'fiscal_year', 'fiscal_quarter')}
                            config.update(report_spec)  # Add the specific report filters
                            search_configs.append(config)
                    
                    # Case 1: Multiple form types (10-Q + 10-K) - only if not using multi_report_specs
                    elif isinstance(form_types, list) and len(form_types) > 1:
                        needs_balanced_search = True
                        for form_type in form_types:
                            config = {k: v for k, v in refined_filters.items() if k not in ('_full_year_compare', '_full_year_form_type')}
                            config['form_type'] = form_type
                            if form_type == '10-K':
                                config.pop('fiscal_quarter', None)  # Remove quarter for annual reports
                            search_configs.append(config)
                    
                    # Case 2: Full year expansion (quarterly + annual)
                    elif refined_filters.get('_full_year_compare') and refined_filters.get('fiscal_quarter'):
                        needs_balanced_search = True
                        
                        # Config 1: Quarterly data (10-Q)
                        quarterly_config = {k: v for k, v in refined_filters.items() if k not in ('_full_year_compare', '_full_year_form_type')}
                        quarterly_config['form_type'] = '10-Q'
                        search_configs.append(quarterly_config)
                        
                        # Config 2: Annual data (10-K)
                        annual_config = {k: v for k, v in refined_filters.items() if k not in ('fiscal_quarter', '_full_year_compare', '_full_year_form_type')}
                        annual_config['form_type'] = refined_filters.get('_full_year_form_type', '10-K')
                        search_configs.append(annual_config)
                    
                    # Case 3: Multiple quarters (multiple 10-Qs)
                    elif isinstance(fiscal_quarters, list) and len(fiscal_quarters) > 1:
                        needs_balanced_search = True
                        for quarter in fiscal_quarters:
                            config = {k: v for k, v in refined_filters.items() if k not in ('_full_year_compare', '_full_year_form_type')}
                            config['fiscal_quarter'] = quarter
                            search_configs.append(config)
                    
                    if needs_balanced_search and search_configs:
                        print(f"Multi-report balanced search: {len(search_configs)} report types, {k} chunks total")
                        chunks_per_search = max(1, k // len(search_configs))
                        remaining_chunks = k % len(search_configs)
                        
                        for i, config in enumerate(search_configs):
                            # Distribute extra chunks to first few searches
                            current_k = chunks_per_search + (1 if i < remaining_chunks else 0)
                            config_type = config.get('form_type', 'Unknown')
                            config_quarter = config.get('fiscal_quarter', 'Annual')
                            
                            print(f"  Search {i+1}: {config_type} {config_quarter} - requesting {current_k} chunks")
                            
                            try:
                                search_results = vector_store.hybrid_search(
                                    table_name=table_name,
                                    query=enhanced_query,
                                    k=current_k,
                                    semantic_weight=0.7,
                                    bm25_weight=0.3,
                                    filters=config
                                ) or []
                                
                                if search_results:
                                    # Add metadata to track which search this came from
                                    for result in search_results:
                                        result['metadata'] = result.get('metadata', {}) or {}
                                        result['metadata']['_search_config'] = f"{config_type}_{config_quarter}"
                                    
                                    balanced_results.extend(search_results)
                                    print(f"    ✅ Found {len(search_results)} chunks from {config_type} {config_quarter}")
                                else:
                                    print(f"    ❌ No results from {config_type} {config_quarter}")
                                    
                            except Exception as e:
                                print(f"    ❌ Search failed for {config_type} {config_quarter}: {e}")
                        
                        vector_results = balanced_results
                        print(f"Balanced search complete: {len(vector_results)} total chunks from {len(search_configs)} report types")
                        full_year_expansion = needs_balanced_search  # Track that we did multi-report search
                    
                    else:
                        # Single report type - use original logic
                        vector_results = vector_store.hybrid_search(
                            table_name=table_name,
                            query=enhanced_query,
                            k=k,
                            semantic_weight=0.7,
                            bm25_weight=0.3,
                            filters=refined_filters
                        ) or []
                        full_year_expansion = False
                    relaxation_steps: List[Dict[str, Any]] = []
                    # If strict filters yield zero, progressively relax to avoid empty results
                    if not vector_results:
                        relaxed_filters = dict(refined_filters)
                        if relaxed_filters.pop("section_name", None) is not None:
                            section_names = relaxed_filters.get("section_name", [])
                            print(f"No results with section_name: {section_names}; retrying without section_name")
                            print("  → This may indicate: section names don't match data, or combination of filters too restrictive")
                            vr = vector_store.hybrid_search(
                                table_name=table_name,
                                query=refined_query,
                                k=k,
                                semantic_weight=0.7,
                                bm25_weight=0.3,
                                filters=relaxed_filters,
                            )
                            relaxation_steps.append({"dropped": ["section_name"], "returned": len(vr or [])})
                            if vr:
                                vector_results = vr
                                refined_filters = relaxed_filters
                                print(f"  ✅ Found {len(vr)} results after dropping section_name")
                        if not vector_results and relaxed_filters.get("fiscal_quarter") is not None:
                            quarter = relaxed_filters.get("fiscal_quarter")
                            rf2 = dict(relaxed_filters)
                            rf2.pop("fiscal_quarter", None)
                            print(f"No results after dropping section_name; retrying without fiscal_quarter ({quarter}) as well")
                            if str(quarter).upper() in ['Q4', '4']:
                                print("  → Q4 data is in annual 10-K reports, not quarterly 10-Q reports")
                                print("  → For Q4 data, use form_type=10-K without fiscal_quarter filter")
                            else:
                                print("  → This may indicate: quarter data not available, or stored with different format")
                            vr2 = vector_store.hybrid_search(
                                table_name=table_name,
                                query=refined_query,
                                k=k,
                                semantic_weight=0.7,
                                bm25_weight=0.3,
                                filters=rf2,
                            )
                            relaxation_steps.append({"dropped": ["section_name", "fiscal_quarter"], "returned": len(vr2 or [])})
                            if vr2:
                                vector_results = vr2
                                refined_filters = rf2
                                print(f"  ✅ Found {len(vr2)} results after dropping section_name + fiscal_quarter")
                    
                    # Apply optional post-filter (against original filters) and verify filter logic
                    if filters:
                        filtered_results = []
                        for result in vector_results:
                            if _matches_filters_vector(result, filters):
                                filtered_results.append(result)
                        vector_results = filtered_results

                    # Debug verification: ensure results match refined_filters used in store-level where
                    try:
                        verify_keys = [
                            ("section_name", lambda x: str(x or "").lower()),
                            ("form_type", lambda x: str(x or "").upper()),
                            ("fiscal_year", lambda x: str(x or "")),
                            ("fiscal_quarter", lambda x: str(x or "").upper()),
                            ("granularity", lambda x: str(x or "")),
                        ]
                        mismatches = []
                        total = len(vector_results)
                        matched = 0
                        for r in vector_results:
                            meta = r.get("metadata", {}) or {}
                            ok = True
                            details = {}
                            for key, norm in verify_keys:
                                if key in refined_filters and refined_filters.get(key) is not None:
                                    exp_val = refined_filters.get(key)
                                    actual_val = norm(meta.get(key))
                                    if isinstance(exp_val, (list, tuple, set)):
                                        exp_norm = [norm(v) for v in exp_val]
                                        if actual_val not in exp_norm:
                                            ok = False
                                            details[key] = {"expected_any_of": exp_norm, "actual": actual_val}
                                    else:
                                        expected = norm(exp_val)
                                        if expected != actual_val:
                                            ok = False
                                            details[key] = {"expected": expected, "actual": actual_val}
                            if ok:
                                matched += 1
                            else:
                                mismatches.append({"id": r.get("id"), "mismatch": details})
                        print(f"Filter verification: {matched}/{total} results match refined filters; mismatches (sample): {mismatches[:3]}")
                    except Exception:
                        pass
                    
                    # Enhance results with accessible metadata for agents
                    enhanced_results = []
                    for result in vector_results:
                        enhanced_result = dict(result)
                        
                        # Ensure metadata is accessible at top level
                        metadata_dict = {}
                        if 'metadata' in result:
                            if isinstance(result['metadata'], str):
                                try:
                                    import json
                                    metadata_dict = json.loads(result['metadata'])
                                except:
                                    pass
                            elif isinstance(result['metadata'], dict):
                                metadata_dict = result['metadata']
                        
                        # Add key metadata fields to top level for easy agent access
                        for key in ['form_type', 'fiscal_quarter', 'fiscal_year', 'section_name', 'granularity', 'filing_date']:
                            if key in metadata_dict:
                                enhanced_result[key] = metadata_dict[key]
                        
                        enhanced_results.append(enhanced_result)
                    
                    logger.info(f"Used vector hybrid search, found {len(enhanced_results)} results")
                    return {
                        "success": True,
                        "ticker": ticker.upper(),
                        "query": refined_query,
                        "results": enhanced_results,
                        "returned": len(enhanced_results),
                        "total_candidates": table_info.get("document_count", 0),
                        "search_method": "vector_hybrid",
                        "agent_info": {
                            "q4_conversion_applied": q4_conversion_applied,
                            "filters_relaxed": len(relaxation_steps) > 0,
                            "llm_refinement_used": llm_refinement_used,
                            "form_types_searched": refined_filters.get("form_type", "all"),
                            "quarters_searched": refined_filters.get("fiscal_quarter", "all"),
                            "sections_searched": refined_filters.get("section_name", "all"),
                        },
                        "metadata": {
                            "k": k,
                            "filters": refined_filters,
                            "original_filters": filters,
                            "generated_at": datetime.now().isoformat(),
                            "data_source": "vector_database",
                            "table_name": table_name,
                            "refinement": refinement_meta,
                            "relaxation": relaxation_steps,
                            **({"full_year_expansion": True} if full_year_expansion else {}),
                        }
                    }
                else:
                    # Table is missing or empty
                    if not fallback_to_api_on_empty:
                        return {
                            "success": True,
                            "ticker": ticker.upper(),
                            "query": query,
                            "results": [],
                            "returned": 0,
                            "total_candidates": 0,
                            "search_method": "vector_hybrid",
                            "metadata": {
                                "k": k,
                                "filters": filters or {},
                                "generated_at": datetime.now().isoformat(),
                                "data_source": "vector_database",
                                "table_name": table_name,
                                "note": "table_missing_or_empty",
                            }
                        }
                
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to API search: {e}")
                # Return agent-friendly error information
                error_info = {
                    "success": False,
                    "ticker": ticker.upper(),
                    "query": query,
                    "error": str(e),
                    "error_type": "vector_search_failure",
                    "agent_info": {
                        "recommended_action": "retry with different filters or check data availability",
                        "fallback_available": fallback_to_api_on_empty,
                        "error_category": "technical_error"
                    },
                    "metadata": {
                        "k": k,
                        "filters": filters,
                        "generated_at": datetime.now().isoformat(),
                        "data_source": "vector_database",
                        "search_method": "vector_hybrid",
                        "note": "vector_search_failed"
                    }
                }
                if not fallback_to_api_on_empty:
                    return error_info
        
        # If not falling back and vector search didn't return, return empty result set cleanly
        if not fallback_to_api_on_empty:
            return {
                "success": True,
                "ticker": ticker.upper(),
                "query": query,
                "results": [],
                "returned": 0,
                "total_candidates": 0,
                "search_method": "vector_hybrid",
                "agent_info": {
                    "data_available": False,
                    "recommended_action": "try broader filters or check if data exists for this ticker/timeframe",
                    "q4_conversion_applied": False,
                    "filters_relaxed": False,
                    "llm_refinement_used": False,
                },
                "metadata": {
                    "k": k,
                    "filters": filters,
                    "generated_at": datetime.now().isoformat(),
                    "data_source": "vector_database",
                    "search_method": "vector_hybrid",
                    "note": "no_results_found"
                }
            }

        # Fallback to fresh API search only if explicitly requested (or vector store unavailable)
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
        # Should not reach here without returning; safeguard empty result
        return {
            "success": True,
            "ticker": ticker.upper(),
            "query": query,
            "results": [],
            "returned": 0,
            "total_candidates": 0,
            "search_method": "vector_hybrid",
            "metadata": {
                "k": k,
                "filters": filters or {},
                "generated_at": datetime.now().isoformat(),
                "data_source": "vector_database",
                "note": "safeguard_return",
            }
        }
    except Exception as e:
        logger.exception("Error in search_report_for_agent")
        return _error(str(e))


def _determine_time_filters(ticker: str, time_period: str, vector_store) -> Dict[str, Any]:
    """Determine time-based filters using shared Company Facts functionality
    
    Args:
        ticker: Company ticker symbol
        time_period: One of 'latest', 'latest_10k_and_10q', 'latest_10k', 'latest_10q', 'last_n_reports', or 'last_N_reports' (e.g., 'last_3_reports')
        vector_store: Vector store instance (used for last_n_reports fallback)
        
    Returns:
        Dict with specific filters targeting the actual latest reports from SEC Company Facts
    """
    try:
        # For Company Facts-based time periods, use shared functionality
        if time_period in ["latest", "latest_10k_and_10q", "latest_10k", "latest_10q"]:
            if get_time_filters_for_search is None:
                logger.warning("fiscal_year_corrector module not available, falling back to default filters")
                return {"ticker": ticker}
            return get_time_filters_for_search(ticker, time_period)
        
        elif time_period == "last_n_reports" or time_period.startswith("last_") and time_period.endswith("_reports"):
            # Extract N from last_N_reports pattern or use default
            if time_period == "last_n_reports":
                n = 4  # Default
            else:
                # Extract number from last_N_reports (e.g., last_3_reports -> 3)
                import re
                match = re.match(r'last_(\d+)_reports', time_period)
                n = int(match.group(1)) if match else 4
            
            # Use vector database approach for getting last N reports
            return _get_last_n_reports_from_vector_db(ticker, vector_store, n)
        
        else:
            # For any other time_period, return no restrictions
            return {}
            
    except Exception as e:
        print(f"Warning: Error determining time filters: {e}")
        return {}


def _get_last_n_reports_from_vector_db(ticker: str, vector_store, n: int = 4) -> Dict[str, Any]:
    """Get last N reports using vector database metadata (fallback for last_n_reports)"""
    try:
        table_name = f"annual_reports_{ticker.lower()}"
        
        # Query for all available reports
        all_reports = vector_store.hybrid_search(
            table_name=table_name,
            query="filing",
            k=100,
            filters={"ticker": ticker.upper()}
        )
        
        if not all_reports:
            print(f"No reports found in vector database for {ticker}")
            return {}
        
        # Extract metadata and sort by fiscal year and quarter
        reports_metadata = []
        for result in all_reports:
            metadata = result.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    import json
                    metadata = json.loads(metadata)
                except:
                    continue
            
            form_type = metadata.get('form_type')
            fiscal_year = metadata.get('fiscal_year')
            fiscal_quarter = metadata.get('fiscal_quarter')
            
            if form_type and fiscal_year:
                # Create sortable key: year * 10 + quarter_order for Q, year * 10 + 5 for annual
                if form_type == '10-K':
                    sort_key = int(fiscal_year) * 10 + 5  # Annual goes after Q3 but before next year Q1
                elif form_type == '10-Q' and fiscal_quarter:
                    quarter_num = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}.get(fiscal_quarter, 0)
                    sort_key = int(fiscal_year) * 10 + quarter_num
                else:
                    continue
                
                reports_metadata.append({
                    "form_type": form_type,
                    "fiscal_year": str(fiscal_year),
                    "fiscal_quarter": fiscal_quarter,
                    "sort_key": sort_key
                })
        
        # Sort by sort_key descending (most recent first) and take first N unique reports
        reports_metadata.sort(key=lambda x: x['sort_key'], reverse=True)
        
        # Remove duplicates by (form_type, fiscal_year, fiscal_quarter)
        seen = set()
        unique_reports = []
        for report in reports_metadata:
            key = (report['form_type'], report['fiscal_year'], report.get('fiscal_quarter'))
            if key not in seen:
                seen.add(key)
                unique_reports.append(report)
                if len(unique_reports) >= n:
                    break
        
        if unique_reports:
            # Create report list description separately to avoid f-string syntax issues
            report_descriptions = [f"{r['form_type']} {r['fiscal_year']} {r.get('fiscal_quarter', '')}" for r in unique_reports]
            print(f"Found last {len(unique_reports)} reports: {report_descriptions}")
            return {
                "multi_report_filter": True,
                "reports": [{"form_type": r["form_type"], "fiscal_year": r["fiscal_year"], 
                           **({"fiscal_quarter": r["fiscal_quarter"]} if r.get("fiscal_quarter") else {})} 
                           for r in unique_reports]
            }
        else:
            return {}
            
    except Exception as e:
        print(f"Error querying vector database for last N reports: {e}")
        return {}


def _refine_query_with_llm(
    ticker: str,
    original_query: str,
    current_filters: Dict[str, Any],
    sample_texts: Optional[List[str]] = None,
    model: Optional[str] = None,
    allowed_sections: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Call OpenRouter to refine query & filters. Returns dict or None on failure.
    Ensures robust JSON parsing and keeps prompt compact. Supports multi-value fields.
    """
    try:
        import os
        import requests
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            return None
        model = model or os.getenv('OPENROUTER_REFINEMENT_MODEL', 'moonshotai/kimi-k2:free')
        # Trim prompt inputs aggressively to avoid max token issues
        trimmed_samples: List[str] = []
        try:
            if sample_texts:
                for s in sample_texts[:2]:  # limit to 2 examples
                    trimmed_samples.append(str(s)[:400])
        except Exception:
            trimmed_samples = []
        sample_block = "\n\nSAMPLE CHUNKS:\n" + "\n---\n".join(trimmed_samples) if trimmed_samples else ""
        allowed_sections_block = ""
        if allowed_sections:
            try:
                limited_allowed = list(allowed_sections)[:25]
                allowed_sections_block = (
                    "\n⚠️ ALLOWED SECTION NAMES - USE ONLY THESE (or omit section_name):\n"
                    + json.dumps(limited_allowed, ensure_ascii=False)
                    + "\n⚠️ DO NOT use invented names like 'revenue', 'sales' - stick to the list above!\n"
                )
            except Exception:
                allowed_sections_block = ""
        prompt = (
            f"TICKER: {ticker}\n"
            f"QUERY: {original_query}\n"
            f"CURRENT FILTERS: {json.dumps(current_filters)}\n"
            "Return ONLY valid JSON for the schema. Use arrays for multi-value filters (comparisons, multiple quarters/sections).\n"
            "\n"
            "CRITICAL FIELD DISTINCTION:\n"
            "- form_type: '10-Q' or '10-K' (filing document type)\n"
            "- fiscal_quarter: 'Q1', 'Q2', 'Q3', or 'Q4' (quarter designation)\n"
            "NEVER put '10-Q' or '10-K' in fiscal_quarter - they belong in form_type!\n"
            "\n"
            "SECTION NAME RULES:\n"
            "- ONLY use section_name values from the allowed list below\n"
            "- Do NOT invent section names like 'revenue', 'sales', etc.\n"
            "- Use COMPLETE section names from the list, not abbreviated versions\n"
            "- For revenue/performance analysis, prefer granularity: 'micro' for better business content\n"
            "\n"
            "SECTION SELECTION GUIDE:\n"
            "- For growth drivers, performance analysis, business insights: ALWAYS include \"Management's Discussion and Analysis of Financial Condition and Results of Operations\"\n"
            "- For financial data, revenue figures, statements: include \"Financial Statements and Supplementary Data\"\n"
            "- For risk analysis: include \"Risk Factors\"\n"
            "- For business overview: include \"Business\"\n"
            "- If query doesn't match above categories AND no exact section match exists, then omit section_name entirely\n"
            "\n"
            "GRANULARITY RULES:\n"
            "- For business analysis, growth drivers, performance insights: ALWAYS use granularity: 'micro'\n"
            "- For basic financial lookups: use granularity: 'base'\n"
            "- ALWAYS include granularity field when sections are specified\n"
            "\n"
            "Form type rules: Use '10-Q' for Q1, Q2, Q3 quarterly data. For Q4 data, use '10-K' in form_type and omit fiscal_quarter.\n"
            "For granularity, use 'micro' for revenue/performance analysis (more detailed business content) or 'base' for standard chunks.\n"
            "\n"
            "QUERY REFINEMENT FOCUS:\n"
            "- Focus ONLY on refining the query text and selecting appropriate sections\n"
            "- NEVER modify fiscal_year, fiscal_quarter, or form_type - these are controlled by time period logic\n"
            "- Do NOT add or change any time-related filters (fiscal_year, fiscal_quarter, form_type)\n"
            "- Your job is ONLY to improve query text and select relevant section_name and granularity\n"
            "- Preserve all existing filters - only add section_name and granularity if appropriate\n"
            "- Time filtering is handled separately - do not make any temporal assumptions\n"
            "\n"
            "\nSchema: { query: string, filters: { section_name?: string | string[], form_type?: string | string[], fiscal_year?: string | string[], fiscal_quarter?: string | string[], granularity?: 'base' | 'micro', filing_date_after?: string, filing_date_before?: string } }\n"
            + allowed_sections_block
            + sample_block
        )
        print("Refinement prompt:", prompt[:1000] + ("..." if len(prompt) > 1000 else ""))
        sys_msg = (
            "You are a JSON generator for a financial RAG system. Output strict JSON only. No prose, no markdown."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "top_p": 0,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        try:
            print(f"Refinement response Result: {resp.json()}")
        except Exception:
            pass
        if resp.status_code != 200:
            return None
        content = resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        def _try_parse_json(s: str) -> Optional[dict]:
            try:
                return json.loads(s)
            except Exception:
                return None
        data = _try_parse_json(content)
        if data is None:
            fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, re.IGNORECASE)
            if fenced:
                data = _try_parse_json(fenced.group(1).strip())
        if data is None:
            text = content
            start = text.find('{'); end = -1
            if start != -1:
                depth = 0
                for i, ch in enumerate(text[start:], start):
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1; break
            if end != -1:
                data = _try_parse_json(text[start:end])
        if not isinstance(data, dict):
            return None
        rq = data.get('query') or original_query
        rf = data.get('filters') or {}
        if not isinstance(rf, dict):
            rf = {}
        print(f"Refined query: {rq}, filters: {rf}")
        return {"query": rq, "filters": rf, "model": model}
    except Exception:
        return None


def _matches_filters(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Enhanced metadata filtering with comprehensive criteria"""
    if not filters:
        return True
    
    metadata = chunk.get("metadata", {})
    
    # Original filters (backward compatibility)
    if filters.get("section_name"):
        section_name = metadata.get("section_name", "").lower()
        filter_section = str(filters["section_name"]).lower()
        if section_name != filter_section:
            return False
    
    if filters.get("form_type"):
        form_type = metadata.get("form_type", "").upper()
        filter_form = str(filters["form_type"]).upper()
        if form_type != filter_form:
            return False
    
    # Enhanced date filtering
    if filters.get("filing_date_after"):
        filing_date_str = metadata.get("filing_date", "")
        if filing_date_str:
            try:
                filing_date = datetime.fromisoformat(filing_date_str.replace('Z', '+00:00'))
                cutoff_date = datetime.fromisoformat(filters["filing_date_after"])
                if filing_date < cutoff_date:
                    return False
            except (ValueError, TypeError):
                return False
    
    # Fiscal year filtering
    if filters.get("fiscal_year"):
        fiscal_year = str(metadata.get("fiscal_year", ""))
        filter_year = str(filters["fiscal_year"])
        if fiscal_year != filter_year:
            return False
    
    # Fiscal quarter filtering
    if filters.get("fiscal_quarter"):
        fiscal_quarter = str(metadata.get("fiscal_quarter", "")).upper()
        filter_quarter = str(filters["fiscal_quarter"]).upper()
        if fiscal_quarter != filter_quarter:
            return False
    
    return True


def _matches_filters_vector(result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Vector search metadata filtering with support for list-valued filters."""
    if not filters:
        return True
    meta = result.get("metadata", {}) or {}
    def _matches(key, norm=lambda x: x, val_transform=lambda x: x):
        if key not in filters or filters.get(key) is None:
            return True
        expected = filters.get(key)
        actual = meta.get(key)
        if isinstance(expected, (list, tuple, set)):
            return norm(val_transform(actual)) in [norm(val_transform(e)) for e in expected]
        return norm(val_transform(actual)) == norm(val_transform(expected))
    if not _matches("section_name", lambda x: (x or "").lower()):
        return False
    if not _matches("form_type", lambda x: (x or "").upper()):
        return False
    if not _matches("fiscal_year", str):
        return False
    if not _matches("fiscal_quarter", lambda x: str(x or "").upper()):
        return False
    return True

# --- Heuristic & normalization helpers (added) ---
def _augment_multi_filters(query_text: str, filters: Dict[str, Any]) -> Dict[str, Any]:
    q = query_text.lower()
    out = dict(filters)
    
    # Detect multiple quarters (Q2 2024 vs Q3 2024 etc)
    quarter_pattern = re.findall(r"q([1-4])\s+20(\d{2})", q, re.I)
    if quarter_pattern:
        quarters: List[str] = []
        years: set[str] = set()
        for num, yr_tail in quarter_pattern:
            quarters.append(f"Q{num}")
            years.add(f"20{yr_tail}")
        seen = set(); qlist: List[str] = []
        for qu in quarters:
            if qu not in seen:
                seen.add(qu); qlist.append(qu)
        if qlist:
            if len(qlist) > 1:
                out['fiscal_quarter'] = qlist
            else:
                out.setdefault('fiscal_quarter', qlist[0])
        if years:
            if len(years) == 1:
                out.setdefault('fiscal_year', list(years)[0])
            else:
                out['fiscal_year'] = sorted(years)
    
    # Full year logic: if user mentions "full year" with a specific quarter, 
    # prepare for dual search (quarter-specific + annual 10-K)
    if ('full year' in q or 'annual' in q) and out.get('fiscal_quarter'):
        out['_full_year_compare'] = True
        # For full year, we want 10-K data (annual report), not quarterly 10-Q
        out['_full_year_form_type'] = '10-K'
    
    # Multi-section detection for comprehensive coverage
    if (('md&a' in q or 'mda' in q or "management's discussion" in q) and
        any(term in q for term in ['financial statements','balance sheet','income statement','cash flow'])):
        existing = out.get('section_name')
        sec_list: List[str] = []
        if isinstance(existing, str):
            sec_list.append(existing)
        elif isinstance(existing, (list, tuple, set)):
            sec_list.extend([str(s) for s in existing])
        
        # Add both MD&A and Financial Statements for comprehensive coverage
        md_candidates = ["Management's Discussion and Analysis", "Management's Discussion and Analysis of Financial Condition and Results of Operations"]
        fs_candidates = ["Financial Statements", "Financial Statements and Supplementary Data"]
        
        for candidate in md_candidates + fs_candidates:
            if candidate not in sec_list:
                sec_list.append(candidate)
        
        if sec_list:
            out['section_name'] = sec_list
    
    return out

def _normalize_filter_values(filters: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for k, v in filters.items():
        if k.startswith('_'):
            normalized[k] = v
            continue
        if isinstance(v, (list, tuple)) and len(v) == 1:
            normalized[k] = v[0]
        else:
            normalized[k] = v
    return normalized


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

# ---- New helper functions for enhancements ----
_MICRO_TARGET_SIZE = 300
_MICRO_OVERLAP = 60
_SECTION_INFERENCE_PATTERNS = [
    ("Management's Discussion and Analysis", re.compile(r"management'?s\s+discussion|results of operations|liquidity|capital resources", re.I)),
    ("Risk Factors", re.compile(r"risk factors", re.I)),
    ("Business", re.compile(r"\b(the )?company\b|products?\b|services?\b|market\b", re.I)),
]
_TAG_KEYWORDS = {
    'revenue': 'revenue',
    'sales': 'revenue',
    'liquidity': 'liquidity',
    'cash': 'liquidity',
    'cash flow': 'liquidity',
    'operating expenses': 'expenses',
    'expense': 'expenses',
    'competition': 'competition',
    'competitor': 'competition',
    'risk': 'risk',
    'strategy': 'strategy',
    'growth': 'strategy',
    'capital expenditure': 'capex',
    'capex': 'capex'
}

def _sentence_split(text: str) -> List[str]:
    # Simple sentence splitter; keeps punctuation
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text.strip())
    return [p for p in parts if p]

def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()

def _numeric_density(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(c.isdigit() for c in text)
    return round(digits / max(1, len(text)), 4)

def _extract_tags(text: str) -> List[str]:
    lower = text.lower()
    tags = set()
    for kw, tag in _TAG_KEYWORDS.items():
        if kw in lower:
            tags.add(tag)
    return sorted(tags)

def _infer_section(existing_section: str, text: str) -> Tuple[str, bool]:
    if existing_section and existing_section.lower() not in {"unclassified content", "", "other", "unknown"}:
        return existing_section, False
    for name, pattern in _SECTION_INFERENCE_PATTERNS:
        if pattern.search(text):
            return name, True
    return existing_section or "Unclassified Content", False

def _create_micro_chunks(base_chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not ENABLE_DUAL_GRANULARITY:
        return []
    text = base_chunk.get('text') or base_chunk.get('content') or ''
    if len(text) <= MICRO_TARGET_SIZE + 50:  # Too small to split
        return []
    sentences = _sentence_split(text)
    micro_chunks = []
    current = []
    current_len = 0
    chunk_id_base = base_chunk.get('chunk_id', 0)
    seq = 0
    for sent in sentences:
        s_len = len(sent)
        if current_len + s_len > MICRO_TARGET_SIZE and current:
            micro_text = ' '.join(current)
            micro_chunks.append(_build_micro_chunk(base_chunk, micro_text, chunk_id_base, seq))
            seq += 1
            # overlap: retain tail part of previous text (~_MICRO_OVERLAP chars)
            tail = micro_text[-MICRO_OVERLAP:]
            current = [tail, sent]
            current_len = len(tail) + s_len
        else:
            current.append(sent)
            current_len += s_len
    if current:
        micro_text = ' '.join(current)
        micro_chunks.append(_build_micro_chunk(base_chunk, micro_text, chunk_id_base, seq))
    return micro_chunks

def _build_micro_chunk(base_chunk: Dict[str, Any], text: str, parent_id: Any, seq: int) -> Dict[str, Any]:
    meta = base_chunk.get('metadata', {}).copy()
    meta['parent_chunk_id'] = parent_id
    meta['granularity'] = 'micro'
    meta['numeric_density'] = _numeric_density(text)
    meta['text_hash'] = _hash_text(text)
    meta['extracted_tags'] = _extract_tags(text)
    return {
        'chunk_id': f"{parent_id}_m{seq}",
        'text': text,
        'metadata': meta,
        'granularity': 'micro'
    }

def _enrich_and_expand_chunks(chunks: List[Dict[str, Any]], ticker: str) -> List[Dict[str, Any]]:
    seen_hashes = set()
    enriched: List[Dict[str, Any]] = []
    for c in chunks:
        text = c.get('text') or c.get('content') or ''
        meta = c.get('metadata', {})
        # Section inference
        section_before = meta.get('section_name', '')
        inferred_section, inferred_flag = _infer_section(section_before, text)
        meta['section_name'] = inferred_section
        if inferred_flag:
            meta['inferred_section'] = True
        # Add granularity
        c['granularity'] = 'base'
        meta['granularity'] = 'base'
        # Hash & numeric density
        meta['text_hash'] = _hash_text(text)
        meta['numeric_density'] = _numeric_density(text)
        meta['extracted_tags'] = _extract_tags(text)
        meta['ticker'] = ticker.upper()
        c['metadata'] = meta
        # Dedup base
        if meta['text_hash'] in seen_hashes:
            continue
        seen_hashes.add(meta['text_hash'])
        enriched.append(c)
        # Micro chunks
        micro_chunks = _create_micro_chunks(c)
        for m in micro_chunks:
            if m['metadata']['text_hash'] in seen_hashes:
                continue
            seen_hashes.add(m['metadata']['text_hash'])
            enriched.append(m)
    return enriched

# ---- section_name canonicalization helpers ----
def _normalize_section_string(s: str) -> str:
    if s is None:
        return ""
    try:
        s2 = unicodedata.normalize("NFKD", str(s))
    except Exception:
        s2 = str(s)
    # unify quotes and symbols
    s2 = s2.replace("\u2019", "'").replace("\u2018", "'")
    s2 = s2.replace("\u201c", '"').replace("\u201d", '"')
    s2 = s2.replace("&", " and ")
    s2 = s2.lower()
    # drop leading "item x." prefix
    s2 = re.sub(r"^item\s+[0-9a-z]+\.?\s*", "", s2)
    # common abbreviations
    s2 = s2.replace("mda", "management s discussion and analysis")
    # normalize whitespace
    s2 = re.sub(r"\s+", " ", s2).strip()
    # remove punctuation except spaces
    s2 = re.sub(r"[^a-z0-9 ]+", "", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def _section_category(norm: str) -> str:
    n = norm
    if any(tok in n for tok in ["management s discussion", "results of operations", "liquidity", "capital resources"]):
        return "mda"
    if "risk factors" in n:
        return "risk"
    if any(tok in n for tok in ["financial statements", "notes to consolidated", "balance sheets", "income statements", "cash flows"]):
        return "financials"
    if any(tok in n for tok in ["controls and procedures", "disclosure controls", "internal control over financial reporting"]):
        return "controls"
    if any(tok in n for tok in ["market risk", "quantitative and qualitative disclosures about market risk"]):
        return "market_risk"
    if "business" in n:
        return "business"
    return "other"

def _map_to_allowed_section(sec: str, allowed_sections: List[str]) -> Optional[str]:
    if not sec:
        return None
    if not allowed_sections:
        return None
    norm_sec = _normalize_section_string(sec)
    # build normalized map
    cand_norm = [(cand, _normalize_section_string(cand)) for cand in allowed_sections]
    # 1) exact normalized match
    for orig, n in cand_norm:
        if n == norm_sec:
            return orig
    # 2) containment with improved matching
    if len(norm_sec) >= 8:  # Lowered threshold for better matching
        # First try: exact containment
        for orig, n in cand_norm:
            if norm_sec in n or n in norm_sec:
                return orig
        
        # Second try: partial matching for common abbreviations
        # Handle "Financial Statements" -> "Financial Statements and Supplementary Data"
        if "financial statements" in norm_sec and not "supplementary" in norm_sec:
            for orig, n in cand_norm:
                if "financial statements" in n and "supplementary" in n:
                    return orig
        
        # Handle "Management's Discussion" -> "Management's Discussion and Analysis of Financial Condition..."
        if "management s discussion" in norm_sec and not "financial condition" in norm_sec:
            for orig, n in cand_norm:
                if "management s discussion" in n and ("financial condition" in n or "results of operations" in n):
                    return orig
    # 3) category-based filter then similarity
    cat = _section_category(norm_sec)
    prioritized = []
    if cat == "mda":
        prioritized = [orig for orig, n in cand_norm if ("management s discussion" in n or "results of operations" in n)]
    elif cat == "risk":
        prioritized = [orig for orig, n in cand_norm if "risk factors" in n]
    elif cat == "financials":
        prioritized = [orig for orig, n in cand_norm if ("financial statements" in n or "notes to consolidated" in n or "balance sheets" in n)]
    elif cat == "controls":
        prioritized = [orig for orig, n in cand_norm if ("controls and procedures" in n or "internal control" in n)]
    elif cat == "market_risk":
        prioritized = [orig for orig, n in cand_norm if ("market risk" in n)]
    elif cat == "business":
        prioritized = [orig for orig, n in cand_norm if "business" in n]
    pool = prioritized if prioritized else [orig for orig, _ in cand_norm]
    # 4) similarity threshold
    best = None
    best_score = 0.0
    for orig in pool:
        n = _normalize_section_string(orig)
        score = SequenceMatcher(None, n, norm_sec).ratio()
        if score > best_score:
            best_score = score
            best = orig
    if best and best_score >= 0.86:
        return best
    # 5) last-chance: if MD&A-like short hint
    if cat == "mda":
        for orig, n in cand_norm:
            if "management s discussion" in n:
                if SequenceMatcher(None, n, norm_sec).ratio() >= 0.75:
                    return orig
    return None


def _cleanup_processed_files_after_indexing(
    ticker: str,
    filings_used: List[Dict[str, Any]],
    years_back: int
) -> Dict[str, Any]:
    """
    Auto-cleanup processed files after successful vector DB indexing
    Removes raw SEC filings and intermediate JSON files to save space
    """
    import os
    import glob
    from pathlib import Path
    
    cleanup_stats = {
        "success": True,
        "files_removed": 0,
        "size_freed_mb": 0,
        "errors": []
    }
    
    try:
        # Get the AnnualReportRAG directory
        script_dir = Path(__file__).parent
        
        # 1. Clean raw SEC filings for this ticker
        filings_dir = script_dir / "filings" / ticker.lower()
        if filings_dir.exists():
            total_size = 0
            files_count = 0
            
            for file_path in filings_dir.rglob("*.txt"):
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    total_size += file_size
                    files_count += 1
                except Exception as e:
                    cleanup_stats["errors"].append(f"Failed to remove {file_path.name}: {e}")
            
            # Remove empty directories
            try:
                if filings_dir.exists() and not any(filings_dir.iterdir()):
                    filings_dir.rmdir()
                # Also clean parent directories if empty
                for parent in filings_dir.parents:
                    if parent.name == "filings":
                        break
                    try:
                        if parent.exists() and not any(parent.iterdir()):
                            parent.rmdir()
                        else:
                            break
                    except:
                        break
            except Exception as e:
                cleanup_stats["errors"].append(f"Failed to clean directories: {e}")
            
            cleanup_stats["files_removed"] += files_count
            cleanup_stats["size_freed_mb"] += total_size / (1024 * 1024)
        
        # 2. Clean processed JSON files in rag_ready_data for this ticker
        rag_dir = script_dir / "rag_ready_data"
        if rag_dir.exists():
            json_pattern = f"{ticker.upper()}_*.json"
            json_files = list(rag_dir.glob(json_pattern))
            
            for json_file in json_files:
                try:
                    file_size = json_file.stat().st_size
                    json_file.unlink()
                    cleanup_stats["files_removed"] += 1
                    cleanup_stats["size_freed_mb"] += file_size / (1024 * 1024)
                except Exception as e:
                    cleanup_stats["errors"].append(f"Failed to remove {json_file.name}: {e}")
        
        # 3. Clean debug files if any exist
        debug_files = list(rag_dir.glob("*_debug.json")) if rag_dir.exists() else []
        for debug_file in debug_files:
            try:
                file_size = debug_file.stat().st_size
                debug_file.unlink()
                cleanup_stats["files_removed"] += 1
                cleanup_stats["size_freed_mb"] += file_size / (1024 * 1024)
            except Exception as e:
                cleanup_stats["errors"].append(f"Failed to remove debug file {debug_file.name}: {e}")
        
        # 4. Clean batch processing results file if it exists (optional)
        batch_results_file = script_dir / "batch_processing_results.json"
        if batch_results_file.exists():
            try:
                file_size = batch_results_file.stat().st_size
                batch_results_file.unlink()
                cleanup_stats["files_removed"] += 1
                cleanup_stats["size_freed_mb"] += file_size / (1024 * 1024)
            except Exception as e:
                cleanup_stats["errors"].append(f"Failed to remove batch results: {e}")
        
        logger.info(f"Auto-cleanup completed for {ticker}: {cleanup_stats['files_removed']} files, "
                   f"{cleanup_stats['size_freed_mb']:.1f}MB freed")
        
        return cleanup_stats
        
    except Exception as e:
        cleanup_stats["success"] = False
        cleanup_stats["errors"].append(f"Cleanup failed: {e}")
        return cleanup_stats


def _deduplicate_filings_by_metadata(
    ticker: str,
    chunks: List[Dict[str, Any]], 
    filings_used: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Skip entire filings that are already in the vector database based on metadata matching.
    Much more efficient than chunk-by-chunk comparison.
    Returns: (filtered_chunks, filtered_filings_used, dedup_stats)
    """
    try:
        # Initialize vector store
        try:
            try:
                from .annual_vector_store import AnnualVectorStore
            except ImportError:
                # Fallback for when running from examples directory
                from annual_vector_store import AnnualVectorStore
            vector_store = AnnualVectorStore()
        except Exception:
            if VECTOR_DB_AVAILABLE:
                vector_store = VectorStore()
            else:
                raise Exception("No vector database available")
        
        table_name = f"annual_reports_{ticker.lower()}"
        
        # Check if table exists
        try:
            table_info = vector_store.get_table_info(table_name)
        except Exception:
            # If we can't check, assume no duplicates and process all
            return chunks, filings_used, {
                "filings_skipped": 0,
                "chunks_skipped": 0,
                "error": "could_not_check_cache"
            }
        
        if not table_info.get("exists") or table_info.get("document_count", 0) == 0:
            # No existing data, all filings are new
            return chunks, filings_used, {
                "filings_skipped": 0,
                "chunks_skipped": 0,
                "reason": "no_existing_data"
            }
        
        # Get existing filing metadata from vector DB
        existing_filings = set()
        
        try:
            # Query for existing filings metadata - get a representative sample
            existing_results = vector_store.hybrid_search(
                table_name=table_name,
                query=f"ticker {ticker}",
                k=1000,  # Large enough to cover typical filing counts
                filters={"ticker": ticker.upper()}
            ) or []
            
            # Extract unique filing identifiers from existing chunks
            for result in existing_results:
                meta = result.get("metadata", {}) or {}
                form_type = meta.get("form_type", "")
                fiscal_year = meta.get("fiscal_year", "")
                fiscal_quarter = meta.get("fiscal_quarter", "")
                
                # Create filing identifier: form_type_year_quarter (e.g., "10-K_2024_None", "10-Q_2024_Q3")
                filing_id = f"{form_type}_{fiscal_year}_{fiscal_quarter or 'None'}"
                existing_filings.add(filing_id)
            
            print(f"   🔍 Found {len(existing_filings)} existing filings in vector DB for {ticker}")
            if existing_filings:
                sample_existing = sorted(list(existing_filings))[:5]
                print(f"   📋 Sample existing: {sample_existing}")
            
        except Exception as e:
            logger.warning(f"Failed to get existing filings metadata: {e}")
            # If we can't get existing data, process all to be safe
            return chunks, filings_used, {
                "filings_skipped": 0,
                "chunks_skipped": 0,
                "error": "could_not_get_existing_filings"
            }
        
        # Filter filings_used and corresponding chunks
        new_filings = []
        new_chunks = []
        skipped_filings = []
        chunks_skipped_count = 0
        
        # Group chunks by filing for easier processing
        chunks_by_filing = {}
        for chunk in chunks:
            meta = chunk.get("metadata", {}) or {}
            filing_key = f"{meta.get('form_type', '')}_{meta.get('fiscal_year', '')}_{meta.get('fiscal_quarter', '') or 'None'}"
            if filing_key not in chunks_by_filing:
                chunks_by_filing[filing_key] = []
            chunks_by_filing[filing_key].append(chunk)
        
        print(f"   🔍 Checking {len(filings_used)} downloaded filings against existing cache...")
        
        for filing in filings_used:
            # Create filing identifier from filing metadata
            form_type = filing.get("form_type", "")
            fiscal_year = filing.get("fiscal_year", "")
            fiscal_quarter = filing.get("fiscal_quarter", "")
            
            filing_id = f"{form_type}_{fiscal_year}_{fiscal_quarter or 'None'}"
            
            # Check if this filing already exists in vector DB
            if filing_id in existing_filings:
                # Skip this entire filing
                skipped_filings.append({
                    "filing_id": filing_id,
                    "form_type": form_type,
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": fiscal_quarter,
                    "chunk_count": filing.get("chunk_count", 0)
                })
                chunks_skipped_count += filing.get("chunk_count", 0)
                print(f"      ⏭️  Skipping {filing_id} (already in vector DB)")
            else:
                # Include this filing
                new_filings.append(filing)
                # Add all chunks from this filing
                if filing_id in chunks_by_filing:
                    new_chunks.extend(chunks_by_filing[filing_id])
                print(f"      ✅ Including {filing_id} (new filing)")
        
        filings_skipped_count = len(skipped_filings)
        
        if skipped_filings:
            print(f"   📊 Skipped filings:")
            for skipped in skipped_filings[:3]:  # Show first 3
                print(f"      - {skipped['form_type']} {skipped['fiscal_year']} Q{skipped['fiscal_quarter'] or 'A'}: {skipped['chunk_count']} chunks")
            if len(skipped_filings) > 3:
                print(f"      ... and {len(skipped_filings) - 3} more")
        
        print(f"   📊 Filing-level deduplication: {len(filings_used)} downloaded → {len(new_filings)} new "
              f"(skipped {filings_skipped_count} filings with {chunks_skipped_count} chunks)")
        
        return new_chunks, new_filings, {
            "filings_skipped": filings_skipped_count,
            "chunks_skipped": chunks_skipped_count,
            "new_filings": len(new_filings),
            "new_chunks": len(new_chunks),
            "success": True
        }
        
    except Exception as e:
        logger.warning(f"Filing-level deduplication failed: {e}")
        # If deduplication fails, process all to be safe
        return chunks, filings_used, {
            "filings_skipped": 0,
            "chunks_skipped": 0,
            "error": str(e)
        }


def _check_existing_reports_in_vector_db(
    ticker: str,
    filing_types: List[str],
    years_back: int,
    start_date: str
) -> Dict[str, Any]:
    """
    Check vector database for existing reports to avoid reprocessing
    Returns cache hit status and existing chunks if found
    """
    try:
        # Initialize vector store
        try:
            try:
                from .annual_vector_store import AnnualVectorStore
            except ImportError:
                # Fallback for when running from examples directory
                from annual_vector_store import AnnualVectorStore
            vector_store = AnnualVectorStore()
        except Exception:
            if VECTOR_DB_AVAILABLE:
                vector_store = VectorStore()
            else:
                raise Exception("No vector database available")
        
        table_name = f"annual_reports_{ticker.lower()}"
        
        # Check if table exists
        try:
            table_info = vector_store.get_table_info(table_name)
        except Exception as e:
            logger.warning(f"Failed to get table info for {table_name}: {e}")
            return {
                "cache_hit": False,
                "partial_cache_hit": False,
                "reason": "table_info_failed",
                "error": str(e)
            }
        
        if not table_info.get("exists") or table_info.get("document_count", 0) == 0:
            return {
                "cache_hit": False,
                "partial_cache_hit": False,
                "reason": "no_existing_data",
                "table_exists": table_info.get("exists", False),
                "document_count": 0
            }
        
        # Calculate date range for comparison
        end_dt = datetime.strptime(start_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=365 * years_back)
        
        # Calculate expected years in the range
        expected_years = set()
        for year in range(start_dt.year, end_dt.year + 1):
            expected_years.add(str(year))
        
        print(f"   📅 Expected years based on date range: {', '.join(sorted(expected_years))}")
        print(f"   🔍 Searching vector DB for existing reports...")
        
        # Query existing reports by filing type and date range
        existing_reports = {}
        total_existing_chunks = 0
        
        for filing_type in filing_types:
            try:
                # Search for existing chunks of this filing type within date range
                existing_chunks = vector_store.hybrid_search(
                    table_name=table_name,
                    query=f"filing {filing_type}",  # Simple query to find any chunks
                    k=1000,  # Large number to get all chunks
                    filters={
                        "ticker": ticker.upper(),
                        "form_type": filing_type
                    }
                ) or []
                
                if existing_chunks:
                    # Group by fiscal year and quarter to understand coverage
                    years_covered = set()
                    quarters_info = {}
                    
                    for chunk in existing_chunks:
                        meta = chunk.get("metadata", {})
                        fiscal_year = meta.get("fiscal_year")
                        fiscal_quarter = meta.get("fiscal_quarter")
                        filing_date = meta.get("filing_date")
                        
                        if fiscal_year:
                            years_covered.add(str(fiscal_year))
                            
                            # Track quarterly details for better logging
                            year_key = str(fiscal_year)
                            if year_key not in quarters_info:
                                quarters_info[year_key] = {
                                    "quarters": set(),
                                    "filing_date": filing_date
                                }
                            if fiscal_quarter:
                                quarters_info[year_key]["quarters"].add(fiscal_quarter)
                    
                    # Create human-readable description
                    coverage_details = []
                    for year in sorted(years_covered):
                        if year in quarters_info:
                            quarters = sorted(list(quarters_info[year]["quarters"]))
                            if filing_type == "10-K":
                                coverage_details.append(f"{year} (Annual)")
                            elif quarters:
                                coverage_details.append(f"{year} ({', '.join(quarters)})")
                            else:
                                coverage_details.append(year)
                        else:
                            coverage_details.append(year)
                    
                    existing_reports[filing_type] = {
                        "chunk_count": len(existing_chunks),
                        "years_covered": sorted(list(years_covered)),
                        "coverage_details": coverage_details,
                        "chunks": existing_chunks
                    }
                    total_existing_chunks += len(existing_chunks)
                    
                    # Debug print for detailed cache info
                    print(f"   🔍 Found {filing_type}: {', '.join(coverage_details)} ({len(existing_chunks)} chunks)")
                
            except Exception as e:
                logger.warning(f"Error checking existing {filing_type} reports: {e}")
                continue
        
        # Analyze what's missing
        missing_filing_types = []
        coverage_analysis = {}
        
        for filing_type in filing_types:
            if filing_type not in existing_reports:
                missing_filing_types.append(filing_type)
                coverage_analysis[filing_type] = {
                    "found_years": [],
                    "missing_years": sorted(expected_years),
                    "status": "completely_missing"
                }
            else:
                found_years = set(existing_reports[filing_type]["years_covered"])
                missing_years = expected_years - found_years
                coverage_analysis[filing_type] = {
                    "found_years": sorted(found_years),
                    "missing_years": sorted(missing_years),
                    "status": "partial" if missing_years else "complete"
                }
                
                # If we have missing years for this filing type, consider it partially missing
                if missing_years:
                    print(f"   ⚠️  {filing_type}: Missing years {', '.join(sorted(missing_years))}")
        
        # Show coverage summary
        print(f"   📊 Coverage analysis:")
        for filing_type, analysis in coverage_analysis.items():
            status = analysis["status"]
            found = analysis["found_years"]
            missing = analysis["missing_years"]
            
            if status == "completely_missing":
                print(f"      {filing_type}: ❌ No data (need all {len(expected_years)} years)")
            elif status == "complete":
                print(f"      {filing_type}: ✅ Complete coverage ({len(found)} years)")
            else:
                print(f"      {filing_type}: ⚠️  Partial coverage ({len(found)}/{len(expected_years)} years, missing: {', '.join(missing)})")
        
        # Full cache hit: all requested filing types have complete year coverage
        has_complete_coverage = True
        for filing_type, analysis in coverage_analysis.items():
            if analysis["status"] != "complete":
                has_complete_coverage = False
                break
        
        if has_complete_coverage and not missing_filing_types and total_existing_chunks > 0:
            # Combine all existing chunks
            all_existing_chunks = []
            filings_used = []
            
            for filing_type, report_data in existing_reports.items():
                all_existing_chunks.extend(report_data["chunks"])
                filings_used.append({
                    "form_type": filing_type,
                    "years_covered": report_data["years_covered"],
                    "chunk_count": report_data["chunk_count"],
                    "source": "vector_db_cache"
                })
            
            return {
                "success": True,
                "ticker": ticker.upper(),
                "chunk_count": len(all_existing_chunks),
                "chunks": all_existing_chunks,
                "filings_used": filings_used,
                "cache_hit": True,
                "partial_cache_hit": False,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "years_back": years_back,
                    "filing_types": filing_types,
                    "data_source": "vector_db_cache",
                    "cache_info": {
                        "cache_type": "full_hit",
                        "existing_reports": {k: v["years_covered"] for k, v in existing_reports.items()},
                        "total_cached_chunks": total_existing_chunks
                    }
                }
            }
        
        # Partial cache hit: some filing types have data OR incomplete year coverage
        elif (len(missing_filing_types) < len(filing_types) and existing_reports) or (existing_reports and not has_complete_coverage):
            # If we have incomplete year coverage, we need to download all filing types to get missing years
            types_to_download = missing_filing_types if has_complete_coverage else filing_types
            
            return {
                "cache_hit": False,
                "partial_cache_hit": True,
                "missing_filing_types": types_to_download,
                "existing_reports": existing_reports,
                "total_existing_chunks": total_existing_chunks,
                "coverage_analysis": coverage_analysis,
                "metadata": {
                    "cache_info": {
                        "cache_type": "partial_hit_incomplete_years" if not has_complete_coverage else "partial_hit_missing_types",
                        "existing_filing_types": list(existing_reports.keys()),
                        "missing_filing_types": types_to_download,
                        "complete_coverage": has_complete_coverage
                    }
                }
            }
        
        # No cache hit
        else:
            return {
                "cache_hit": False,
                "partial_cache_hit": False,
                "reason": "insufficient_existing_data",
                "existing_reports": existing_reports,
                "total_existing_chunks": total_existing_chunks
            }
            
    except Exception as e:
        logger.warning(f"Cache check failed for {ticker}: {e}")
        return {
            "cache_hit": False,
            "partial_cache_hit": False,
            "reason": "cache_check_failed",
            "error": str(e)
        }