"""
Interactive Ingestion Example for AnnualReportRAG
Downloads filings, chunks them, enriches metadata, and indexes into the vector store.

Usage:
    cd AnnualReportRAG/examples/
    python ingest_reports.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import AnnualReportRAG
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agent_interface import index_reports_for_agent
except ImportError:
    # Fallback for different directory structures
    sys.path.insert(0, str(Path(__file__).parent))
    from agent_interface import index_reports_for_agent


def ingest_reports(
    ticker: str,
    years_back: int = 1,
    filing_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Ingest filings for a ticker and index into the vector store.

    Returns the full indexing result, including vector storage summary.
    """
    filing_types = filing_types or ["10-K", "10-Q"]
    print(f"\n=== Ingestion: {ticker.upper()} | Years back: {years_back} | Types: {', '.join(filing_types)} ===")
    result = index_reports_for_agent(ticker=ticker, years_back=years_back, filing_types=filing_types)
    if not result.get("success"):
        print(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
        return result

    chunk_count = result.get("chunk_count", 0)
    filings_used = result.get("filings_used", [])
    print(f"✅ Ingested {chunk_count} chunks from {len(filings_used)} filings")

    vec = result.get("vector_storage", {}) or {}
    if vec.get("success"):
        source = vec.get("source", "new_indexing")
        if source == "cache_hit":
            print("🗄️  Vector storage: ✅ Data retrieved from cache (already in database)")
        else:
            print("🗄️  Vector storage: ✅ Successfully indexed new data")
    else:
        err = vec.get("error")
        if not vec:  # No vector_storage key at all
            print("⚠️  Vector storage: Status unknown (no vector storage info returned)")
        else:
            print(f"⚠️  Vector storage: ❌ Failed{': ' + err if err else ''}")

    # Persist a small summary artifact in AnnualReportRAG/data/summaries/
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = {
        "ticker": ticker.upper(),
        "years_back": years_back,
        "filing_types": filing_types,
        "chunk_count": chunk_count,
        "filings_used": filings_used,
        "vector_storage": vec,
        "generated_at": datetime.now().isoformat(),
    }
    
    # Create summaries directory within AnnualReportRAG
    summaries_dir = Path(__file__).parent.parent / "data" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = summaries_dir / f"ingestion_summary_{ticker.upper()}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to: {out_path}")

    return result


if __name__ == "__main__":
    ticker = (input("Enter ticker (e.g., AAPL): ") or "AAPL").strip().upper()
    try:
        years_back = int(input("Years back (default 1): ") or "1")
    except ValueError:
        years_back = 1
    types_raw = input("Filing types comma-separated (default 10-K,10-Q): ")
    if types_raw.strip():
        # Parse and normalize filing types
        filing_types = []
        for t in types_raw.split(','):
            t_clean = t.strip().upper()
            # Handle common variations: 10K -> 10-K, 10Q -> 10-Q, 8K -> 8-K
            if t_clean == "10K":
                filing_types.append("10-K")
            elif t_clean == "10Q":
                filing_types.append("10-Q")
            elif t_clean == "8K":
                filing_types.append("8-K")
            else:
                # Already properly formatted or other type
                filing_types.append(t_clean)
    else:
        filing_types = ["10-K", "10-Q"]

    ingest_reports(ticker, years_back, filing_types)
