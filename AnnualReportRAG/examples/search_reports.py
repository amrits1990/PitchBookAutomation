"""
Interactive Search Example for AnnualReportRAG
Performs hybrid search over previously ingested filings in the vector store.
Optional: call LLM for evaluation.

Usage:
    cd AnnualReportRAG/examples/
    python search_reports.py
"""

import json
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path to import AnnualReportRAG
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agent_interface import search_report_for_agent
except ImportError:
    # Fallback for different directory structures
    sys.path.insert(0, str(Path(__file__).parent))
    from agent_interface import search_report_for_agent

import requests


def evaluate_chunks_with_llm(query: str, chunks: list, filing_info: Dict[str, Any]) -> Dict[str, Any]:
    """Optional LLM evaluation of search results"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        return {"status": "skipped", "message": "No OPENROUTER_API_KEY configured"}

    chunks_text = ""
    for i, chunk in enumerate(chunks):
        text = chunk.get('content') or chunk.get('text') or ''
        chunks_text += f"\n--- Chunk {i+1} ---\n{text[:500]}...\n"

    prompt = f"""
You are evaluating the relevance and quality of document chunks retrieved for a financial question.

QUERY: "{query}"

DOCUMENT INFO:
- Company: {filing_info.get('ticker', 'N/A')}
- Total Returned: {len(chunks)}

RETRIEVED CHUNKS:
{chunks_text}

Evaluate on a 1-5 scale where:
- 5 = Excellent: Directly answers query with specific data/metrics
- 4 = Good: Relevant with useful context, minor gaps
- 3 = Moderate: Somewhat relevant but lacks specificity
- 2 = Poor: Limited relevance, mostly generic content
- 1 = Very Poor: Irrelevant or misleading

Provide JSON with fields: relevance_score, quality_score, coverage_score, key_insights, missing_elements, overall_assessment
"""

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "moonshotai/kimi-k2:free",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.1,
            },
            timeout=60,
        )
        if response.status_code != 200:
            return {"status": "error", "message": f"OpenRouter error: {response.status_code}"}
        content = response.json()['choices'][0]['message']['content']
        try:
            data = json.loads(content)
            data['status'] = 'success'
            return data
        except json.JSONDecodeError:
            return {"status": "success", "raw_response": content}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def extract_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from chunk, handling both top-level and nested metadata"""
    metadata = {}
    
    # Try to get metadata from top-level fields first
    for key in ['form_type', 'fiscal_quarter', 'fiscal_year', 'section_name', 'granularity', 'filing_date']:
        if key in chunk:
            metadata[key] = chunk[key]
    
    # If not found, try nested metadata
    nested_metadata = chunk.get('metadata', {})
    if isinstance(nested_metadata, str):
        try:
            nested_metadata = json.loads(nested_metadata)
        except:
            nested_metadata = {}
    
    # Fill in missing metadata from nested structure
    for key in ['form_type', 'fiscal_quarter', 'fiscal_year', 'section_name', 'granularity', 'filing_date']:
        if key not in metadata and key in nested_metadata:
            metadata[key] = nested_metadata[key]
    
    return metadata


def display_agent_info(result):
    """Display agent-friendly information about the search"""
    agent_info = result.get('agent_info', {})
    metadata = result.get('metadata', {})
    
    print(f"\n📊 Agent Information:")
    print(f"  Search Method: {result.get('search_method', 'unknown')}")
    print(f"  Q4→10-K Conversion: {'✅ Applied' if agent_info.get('q4_conversion_applied') else '❌ Not needed'}")
    print(f"  LLM Refinement: {'✅ Used' if agent_info.get('llm_refinement_used') else '❌ Disabled'}")
    print(f"  Filter Relaxation: {'✅ Applied' if agent_info.get('filters_relaxed') else '❌ Not needed'}")
    
    # Show final filters used
    final_filters = metadata.get('filters', {})
    if final_filters:
        print(f"  Final Filters Used:")
        for key, value in final_filters.items():
            if key != 'ticker':  # Ticker is obvious
                print(f"    {key}: {value}")
    
    # Show what was searched
    print(f"  Form Types: {agent_info.get('form_types_searched', 'all')}")
    print(f"  Quarters: {agent_info.get('quarters_searched', 'all')}")
    if agent_info.get('sections_searched') != 'all':
        print(f"  Sections: {agent_info.get('sections_searched', 'all')}")


def display_chunk_metadata(results):
    """Display metadata summary of retrieved chunks"""
    if not results:
        return
    
    # Collect metadata statistics
    form_types = set()
    quarters = set()
    years = set()
    sections = set()
    
    for chunk in results:
        metadata = extract_chunk_metadata(chunk)
        if metadata.get('form_type'):
            form_types.add(metadata['form_type'])
        if metadata.get('fiscal_quarter'):
            quarters.add(metadata['fiscal_quarter'])
        if metadata.get('fiscal_year'):
            years.add(str(metadata['fiscal_year']))
        if metadata.get('section_name'):
            sections.add(metadata['section_name'])
    
    print(f"\n📋 Retrieved Chunks Metadata:")
    print(f"  Form Types: {sorted(form_types) if form_types else ['Not available']}")
    print(f"  Fiscal Years: {sorted(years) if years else ['Not available']}")
    print(f"  Fiscal Quarters: {sorted(quarters) if quarters else ['Annual/Not specified']}")
    print(f"  Sections: {len(sections)} unique sections")
    if sections and len(sections) <= 5:
        print(f"    {list(sorted(sections))}")
    elif sections:
        print(f"    Top 3: {list(sorted(sections))[:3]}...")


def run_retrieval(ticker: str, query: str, k: int = 20, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run search query against indexed reports"""
    filters = filters or {}
    res = search_report_for_agent(
        ticker=ticker,
        query=query,
        k=k,
        filters=filters,
        # Do not auto-download/index when retrieval finds no matches
        fallback_to_api_on_empty=False,
        # Leave refinement default-enabled in the agent
    )
    return res


def search_reports(ticker: str, query: str, k: int = 20) -> Dict[str, Any]:
    """Search reports for a ticker and return results with analysis.
    
    Returns the full search result, including agent information and chunk metadata.
    """
    print(f"\n=== Search: {ticker.upper()} | k={k} ===")
    print(f"Query: {query}")
    
    result = run_retrieval(ticker, query, k)
    
    if not result.get('success'):
        print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
        
        # Show agent-friendly error information
        agent_info = result.get('agent_info', {})
        if agent_info:
            print(f"💡 Recommended Action: {agent_info.get('recommended_action', 'Retry with different parameters')}")
            print(f"🔧 Error Category: {agent_info.get('error_category', 'unknown')}")
        return result
    
    results = result.get('results', [])
    print(f"✅ Returned {len(results)} chunks via {result.get('search_method', 'unknown')}")
    
    # Display agent information
    display_agent_info(result)
    
    # Display chunk metadata summary  
    display_chunk_metadata(results)
    
    # Show sample chunks with metadata
    print(f"\n📄 Sample Chunks:")
    for i, r in enumerate(results[:5]):
        text = r.get('content') or r.get('text') or ''
        chunk_metadata = extract_chunk_metadata(r)
        
        # Build metadata summary
        meta_parts = []
        if chunk_metadata.get('form_type'):
            meta_parts.append(f"Form: {chunk_metadata['form_type']}")
        if chunk_metadata.get('fiscal_quarter'):
            meta_parts.append(f"Q: {chunk_metadata['fiscal_quarter']}")
        if chunk_metadata.get('fiscal_year'):
            meta_parts.append(f"FY: {chunk_metadata['fiscal_year']}")
        if chunk_metadata.get('section_name'):
            section = chunk_metadata['section_name']
            if len(section) > 30:
                section = section[:27] + "..."
            meta_parts.append(f"Section: {section}")
        
        meta_str = " | ".join(meta_parts) if meta_parts else "No metadata"
        print(f"  {i+1}. [{meta_str}]")
        print(f"     {text[:120]}...")
    
    return result


if __name__ == "__main__":
    ticker = (input("Enter ticker (e.g., AAPL): ") or "AAPL").strip().upper()
    query = input("Enter query: ").strip() or "What were the main revenue drivers for the company?"
    try:
        k = int(input("Top-K (default 20): ") or "20")
    except ValueError:
        k = 20

    result = search_reports(ticker, query, k)
    
    if result.get('success'):
        # Optional evaluation
        results = result.get('results', [])
        do_eval = (input("\nRun LLM evaluation? (y/N): ") or "n").strip().lower().startswith('y')
        if do_eval:
            filing_info = {"ticker": ticker}
            eval_res = evaluate_chunks_with_llm(query, results, filing_info)
            print("\n🤖 LLM Evaluation:")
            print(json.dumps(eval_res, indent=2))
    
    # Save result for analysis in AnnualReportRAG/data/search_results/
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create search_results directory within AnnualReportRAG
    search_results_dir = Path(__file__).parent.parent / "data" / "search_results"
    search_results_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = search_results_dir / f"search_result_{ticker}_{ts}.json"
    
    def _json_default(o):
        try:
            if isinstance(o, (datetime, date)):
                return o.isoformat()
        except Exception:
            pass
        return str(o)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"\n💾 Search result saved to: {out_path}")