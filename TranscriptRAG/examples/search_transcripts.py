"""
Interactive Transcript Search Example for TranscriptRAG
Demonstrates the agent-ready search interface with standardized responses.
Searches through indexed earnings call transcripts with filtering and evaluation.

Features:
- Agent-ready standardized response format
- Comprehensive error handling with error codes
- Request tracking and performance metrics
- Interactive search with result evaluation

Usage:
    cd TranscriptRAG/examples/
    python search_transcripts.py
"""

import json
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import TranscriptRAG
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agent_interface import (
        search_transcripts_for_agent, 
        evaluate_transcript_search_results_with_llm,
        get_available_quarters_for_agent
    )
except ImportError:
    # Fallback for different directory structures
    sys.path.insert(0, str(Path(__file__).parent))
    from agent_interface import (
        search_transcripts_for_agent, 
        evaluate_transcript_search_results_with_llm,
        get_available_quarters_for_agent
    )


def extract_filters_from_query(query: str) -> Dict[str, Any]:
    """Extract search filters from user query using simple keyword matching"""
    filters = {}
    query_lower = query.lower()
    
    # Quarter detection
    quarters = ["q1", "q2", "q3", "q4", "first quarter", "second quarter", "third quarter", "fourth quarter"]
    for i, quarter_term in enumerate(quarters):
        if quarter_term in query_lower:
            if i < 4:
                filters["quarter"] = f"Q{i+1}"
            else:
                filters["quarter"] = f"Q{(i-3)}"
            break
    
    # Speaker type detection
    ceo_terms = ["ceo", "chief executive", "tim cook", "cook"]
    cfo_terms = ["cfo", "chief financial", "kevan parekh", "parekh"]
    management_terms = ["management", "executives", "leadership"]
    analyst_terms = ["analyst", "analysts", "analyst questions"]
    
    if any(term in query_lower for term in ceo_terms):
        filters["ceo_content"] = True
    elif any(term in query_lower for term in cfo_terms):
        filters["cfo_content"] = True
    elif any(term in query_lower for term in management_terms):
        filters["is_management"] = True
    elif any(term in query_lower for term in analyst_terms):
        filters["is_analyst"] = True
    
    # Section detection
    if any(term in query_lower for term in ["outlook", "guidance", "forecast", "future"]):
        filters["section_name"] = "Financial Results"  # Usually guidance is in prepared remarks
    elif any(term in query_lower for term in ["question", "q&a", "analyst question"]):
        filters["is_qa_section"] = True
    
    return filters


def search_transcripts(
    ticker: str,
    query: str,
    quarters_back: Optional[int] = None,
    quarters: Optional[List[str]] = None,
    k: int = 20,
    search_method: str = "vector_hybrid",
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Search transcripts for a ticker with optional filters and evaluation.

    Returns the full search result with metadata and optional LLM evaluation.
    """
    print(f"\n=== Agent Search: {ticker.upper()} | k={k} ===")
    print(f"Query: {query}")
    print(f"Method: {search_method}")
    
    print("\n📡 Calling search_transcripts_for_agent...")
    result = search_transcripts_for_agent(
        ticker=ticker,
        query=query,
        quarters_back=quarters_back,
        quarters=quarters,
        k=k,
        filters=filters,
        search_method=search_method
    )
    
    # Show metadata information
    metadata = result.get("metadata", {})
    if metadata:
        print(f"🆔 Request ID: {metadata.get('request_id', 'unknown')}")
        processing_time = metadata.get('processing_time_ms')
        if processing_time:
            print(f"⏱️  Processing time: {processing_time:.0f}ms")
    
    if not result.get("success"):
        error_info = result.get("error", {})
        if isinstance(error_info, dict):
            error_code = error_info.get("code", "UNKNOWN")
            error_msg = error_info.get("message", "Unknown error")
            error_details = error_info.get("details", {})
            
            print(f"❌ Search failed [{error_code}]: {error_msg}")
            if error_details:
                print(f"📋 Details: {error_details}")
        else:
            print(f"❌ Search failed: {str(error_info)}")
        return result

    # Extract data from standardized response format
    data = result.get("data", {})
    results = data.get("results", [])
    returned = data.get("returned", 0)
    method = data.get("search_method", "unknown")
    
    print(f"✅ Returned {returned} chunks via {method}")
    
    # Show agent information
    agent_meta = data.get("agent_metadata", {})
    search_meta = data.get("search_metadata", {})
    
    if agent_meta or search_meta:
        print(f"\n📊 Search Information:")
        print(f"  Search Method: {method}")
        
        if search_meta.get("fallback_used"):
            print(f"  Fallback Used: ✅ (Vector DB unavailable)")
        
        if agent_meta.get('vector_db_exists'):
            print(f"  Vector DB Documents: {agent_meta.get('vector_db_document_count', 0)}")
        
        if search_meta.get("filters_applied"):
            print(f"  Filters Applied: {search_meta['filters_applied']}")
            
        if search_meta.get("total_candidates"):
            print(f"  Total Candidates: {search_meta['total_candidates']}")
    
    # Show sample results
    print(f"\n📄 Sample Chunks:")
    if not results:
        print("  (No results found)")
    else:
        for i, chunk in enumerate(results[:3]):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            score = chunk.get("relevance_score", 0.0)
            
            speaker = metadata.get("speaker") or metadata.get("primary_speaker", "Unknown")
            section = metadata.get("section_name", "Unknown")
            quarter = metadata.get("quarter", "Q?")
            fiscal_year = metadata.get("fiscal_year", "????")
            
            print(f"\n  [{i+1}] Score: {score:.3f} | {speaker} | {section}")
            print(f"      Quarter: {quarter} {fiscal_year}")
            print(f"      Content: {content[:200]}{'...' if len(content) > 200 else ''}")
    
    # Show quarter distribution in results to verify filtering worked
    if results:
        quarter_counts = {}
        for chunk in results:
            metadata = chunk.get("metadata", {})
            quarter = metadata.get("quarter", "Q?")
            fiscal_year = metadata.get("fiscal_year", "????")
            quarter_key = f"{quarter} {fiscal_year}"
            quarter_counts[quarter_key] = quarter_counts.get(quarter_key, 0) + 1
        
        print(f"\n📊 Quarter Distribution in Results:")
        for quarter, count in sorted(quarter_counts.items()):
            print(f"    {quarter}: {count} chunks")
    
    return result


def interactive_search():
    """Interactive search with multiple rounds and options"""
    print("🎯 TranscriptRAG Agent Interface - Interactive Search")
    print("=" * 65)
    print("This script demonstrates the agent-ready interface for searching transcripts")
    print("Standardized response format with error codes and request tracking\n")
    
    ticker = input("Enter ticker (e.g., AAPL): ").strip().upper()
    if not ticker:
        print("❌ Ticker required")
        return
        
    print(f"\n🎯 Configuration:")
    print(f"   Ticker: {ticker}")
    print(f"   Interface: Agent-Ready (Standardized Response)")
    
    # Get available quarters first
    print(f"\n📅 Checking available quarters for {ticker}...")
    quarters_result = get_available_quarters_for_agent(ticker)
    available_quarters = []
    
    if quarters_result.get("success"):
        data = quarters_result.get("data", {})
        available_quarters = data.get("available_quarters", [])
        total_docs = data.get("total_documents", 0)
        
        print(f"✅ Found transcript data: {total_docs} chunks")
        print(f"📅 Available quarters: {', '.join(available_quarters[:8])}{'...' if len(available_quarters) > 8 else ''}")
        
        # Show available speakers if needed
        speakers = data.get("speakers", [])
        if speakers:
            print(f"👥 Available speakers: {', '.join(speakers[:5])}{'...' if len(speakers) > 5 else ''}")
    else:
        error = quarters_result.get("error", {})
        print(f"⚠️  Could not get quarters: {error.get('message', 'Unknown error')}")
        error_details = error.get("details", {})
        if error_details.get("suggestion"):
            print(f"💡 {error_details['suggestion']}")
        print("Will proceed with quarters_back parameter instead...")
    
    while True:
        print(f"\n--- Searching {ticker} ---")
        query = input("Enter search query: ").strip()
        if not query:
            print("❌ Query required")
            continue
            
        try:
            k = int(input("Top-K results (1-100, default 20): ") or "20")
            if k < 1 or k > 100:
                print("⚠️  K should be between 1 and 100, using default of 20")
                k = 20
        except ValueError:
            print("⚠️  Invalid input, using default of 20")
            k = 20
        
        # Ask for search method
        method_input = input("Search method (vector_hybrid/vector_semantic/keyword, default hybrid): ").strip().lower()
        valid_methods = ["vector_hybrid", "vector_semantic", "keyword", "bm25"]
        search_method = "vector_hybrid"
        if method_input in valid_methods:
            search_method = method_input
        elif method_input and method_input not in valid_methods:
            print(f"⚠️  Invalid search method, using default: vector_hybrid")
        
        # Quarter selection - use improved interface
        selected_quarters = None
        quarters_back = None
        
        if available_quarters:
            print(f"\n📅 Quarter Selection (Available: {', '.join(available_quarters[:6])}{'...' if len(available_quarters) > 6 else ''}):")
            print("Choose option:")
            print("  1. Select specific quarters")
            print("  2. Use quarters back (legacy)")
            print("  3. Search all quarters")
            
            choice = input("Selection (1-3, default 1): ").strip() or "1"
            
            if choice == "1":
                # Let user select specific quarters
                print(f"\nAvailable quarters:")
                for i, quarter in enumerate(available_quarters[:10], 1):  # Show up to 10
                    print(f"  {i}. {quarter}")
                
                quarter_input = input("Select quarters (comma-separated numbers, e.g., '1,2,3', or direct quarters like 'Q3 2025', or 'all'): ").strip()
                
                if quarter_input.lower() == 'all':
                    selected_quarters = available_quarters
                elif quarter_input:
                    try:
                        # Check if input is direct quarter format like "Q3 2025" or "Q1 2024, Q2 2024"
                        if any(q.strip() in available_quarters for q in quarter_input.split(',')):
                            # Direct quarter input
                            direct_quarters = [q.strip() for q in quarter_input.split(',')]
                            selected_quarters = [q for q in direct_quarters if q in available_quarters]
                            if not selected_quarters:
                                print("⚠️  None of the specified quarters found, using recent 2 quarters")
                                selected_quarters = available_quarters[:2]
                        else:
                            # Numeric indices input
                            indices = [int(x.strip()) - 1 for x in quarter_input.split(',')]
                            selected_quarters = [available_quarters[i] for i in indices if 0 <= i < len(available_quarters)]
                            if not selected_quarters:
                                print("⚠️  Invalid selection, using recent 2 quarters")
                                selected_quarters = available_quarters[:2]
                    except (ValueError, IndexError):
                        print("⚠️  Invalid selection, using recent 2 quarters")
                        selected_quarters = available_quarters[:2]
                else:
                    # Default to recent 2 quarters
                    selected_quarters = available_quarters[:2]
                    
            elif choice == "2":
                # Legacy quarters_back
                try:
                    quarters_input = input("Quarters back (1-20, default 4): ").strip()
                    quarters_back = int(quarters_input) if quarters_input else 4
                    if quarters_back < 1 or quarters_back > 20:
                        print("⚠️  Quarters back should be between 1 and 20, using default of 4")
                        quarters_back = 4
                except ValueError:
                    print("⚠️  Invalid input, using default of 4 quarters")
                    quarters_back = 4
            else:
                # Search all quarters (no filtering)
                selected_quarters = None
                quarters_back = None
        else:
            # Fallback to quarters_back if no quarter info available
            try:
                quarters_input = input("Quarters back (1-20, default 4): ").strip()
                quarters_back = int(quarters_input) if quarters_input else 4
                if quarters_back < 1 or quarters_back > 20:
                    print("⚠️  Quarters back should be between 1 and 20, using default of 4")
                    quarters_back = 4
            except ValueError:
                print("⚠️  Invalid input, using default of 4 quarters")
                quarters_back = 4
            
        # Show search configuration
        print(f"\n🔍 Search Configuration:")
        print(f"   Query: {query}")
        print(f"   Top-K: {k}")
        print(f"   Method: {search_method}")
        if selected_quarters:
            print(f"   Quarters: {', '.join(selected_quarters)}")
        elif quarters_back:
            print(f"   Quarters back: {quarters_back}")
        else:
            print(f"   Quarters: all available")
        
        # Execute search
        try:
            result = search_transcripts(
                ticker=ticker,
                query=query,
                quarters_back=quarters_back,
                quarters=selected_quarters,
                k=k,
                search_method=search_method,
                filters=None  # Simple search without pre-filters
            )
        except Exception as e:
            print(f"❌ Unexpected search error: {e}")
            continue
        
        if result.get("success") and result.get("data", {}).get("results"):
            # Ask if user wants LLM evaluation
            eval_input = input("\nRun LLM evaluation of results? [Y/n]: ").strip().lower()
            if not eval_input.startswith('n'):
                print("\n🤖 Running LLM Evaluation...")
                try:
                    eval_result = evaluate_transcript_search_results_with_llm(
                        query=query,
                        results=result["data"]["results"],
                        ticker=ticker
                    )
                except Exception as e:
                    print(f"❌ Evaluation error: {e}")
                    eval_result = None
            
                if eval_result and eval_result.get("success"):
                    eval_data = eval_result.get("data", {})
                    print(f"\n🎯 LLM Evaluation Results:")
                    print(f"  Overall Quality: {eval_data.get('overall_quality', 'unknown')}")
                    print(f"  Relevance Score: {eval_data.get('relevance_score', 0.0):.2f}")
                    print(f"  Section Coverage: {eval_data.get('section_coverage', 'N/A')}")
                    
                    strengths = eval_data.get("strengths", [])
                    if strengths:
                        print(f"  ✅ Strengths: {', '.join(strengths[:2])}")
                    
                    improvements = eval_data.get("improvements", [])
                    if improvements:
                        print(f"  💡 Improvements: {', '.join(improvements[:2])}")
                        
                    # Show evaluation metadata
                    eval_metadata = eval_result.get("metadata", {})
                    if eval_metadata.get("request_id"):
                        print(f"  🆔 Eval Request ID: {eval_metadata['request_id']}")
                        
                elif eval_result:
                    error_info = eval_result.get("error", {})
                    if isinstance(error_info, dict):
                        error_code = error_info.get("code", "UNKNOWN")
                        error_msg = error_info.get("message", "Unknown error")
                        print(f"❌ LLM evaluation failed [{error_code}]: {error_msg}")
                    else:
                        print(f"❌ LLM evaluation failed: {str(error_info)}")
            
            # Save results
            save_choice = input("\nSave results to file? [y/N]: ").lower().startswith('y')
            if save_choice:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Create search_results directory within TranscriptRAG
                search_results_dir = Path(__file__).parent.parent / "data" / "search_results"
                search_results_dir.mkdir(parents=True, exist_ok=True)
                
                out_path = search_results_dir / f"agent_search_result_{ticker}_{ts}.json"
                
                # Add search configuration to the saved data
                save_data = {
                    "search_config": {
                        "ticker": ticker,
                        "query": query,
                        "k": k,
                        "search_method": search_method,
                        "quarters_back": quarters_back,
                        "quarters": selected_quarters,
                        "quarter_selection_method": "specific_quarters" if selected_quarters else ("quarters_back" if quarters_back else "all_quarters")
                    },
                    "agent_response": result,
                    "saved_at": datetime.now().isoformat()
                }
                
                def _json_default(o):
                    try:
                        if isinstance(o, (datetime, date)):
                            return o.isoformat()
                    except Exception:
                        pass
                    return str(o)
                
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, default=_json_default)
                print(f"💾 Results saved to: {out_path}")
        
        else:
            print("\n💥 Search failed or returned no results. Check error details above.")
        
        # Continue searching?
        continue_input = input("\nSearch again? [Y/n]: ").strip().lower()
        if continue_input.startswith('n'):
            print("\n👋 Goodbye! Thanks for testing the TranscriptRAG agent interface.")
            break


if __name__ == "__main__":
    interactive_search()