"""
Interactive Transcript Ingestion Example for TranscriptRAG
Downloads earnings call transcripts from Alpha Vantage API, processes them, and indexes into the vector store.

Features:
- Interactive user input for ticker and quarters selection
- Intelligent caching - checks for existing data before API calls
- Automatic vector database indexing with metadata enrichment
- Progress tracking and result summaries
- Error handling and validation

Requirements:
- ALPHA_VANTAGE_API_KEY environment variable must be set
- TranscriptRAG package properly configured

Usage:
    cd TranscriptRAG/examples/
    python ingest_transcripts.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path to import TranscriptRAG
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agent_interface import index_transcripts_for_agent
except ImportError:
    # Fallback for different directory structures
    sys.path.insert(0, str(Path(__file__).parent))
    from agent_interface import index_transcripts_for_agent




def ingest_transcripts(
    ticker: str,
    quarters_back: int = 4,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Ingest transcripts for a ticker and index into the vector store.

    Returns the full indexing result, including vector storage summary.
    """
    print(f"\n=== Transcript Ingestion: {ticker.upper()} | Quarters back: {quarters_back} ==")
    if force_refresh:
        print("🔄 Force refresh enabled - bypassing cache")
    
    # Use Alpha Vantage API through agent interface
    print("📡 Calling index_transcripts_for_agent...")
    result = index_transcripts_for_agent(
        ticker=ticker, 
        quarters_back=quarters_back, 
        force_refresh=force_refresh
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
            
            print(f"❌ Ingestion failed [{error_code}]: {error_msg}")
            if error_details:
                print(f"📋 Details: {error_details}")
        else:
            print(f"❌ Ingestion failed: {str(error_info)}")
        return result

    # Extract data from standardized response format
    data = result.get("data", {})
    chunk_count = data.get("chunk_count", 0)
    transcripts_used = data.get("transcripts_used", [])
    print(f"✅ Ingested {chunk_count} chunks from {len(transcripts_used)} transcripts")

    vec = data.get("vector_storage", {}) or {}
    if vec.get("success"):
        source = vec.get("source", "new_indexing")
        if source == "cache_hit":
            print("🗄️  Vector storage: ✅ Data retrieved from cache (already in database)")
        else:
            print("🗄️  Vector storage: ✅ Successfully indexed new transcript data")
    else:
        err = vec.get("error")
        if not vec:  # No vector_storage key at all
            print("⚠️  Vector storage: Status unknown (no vector storage info returned)")
        else:
            print(f"⚠️  Vector storage: ❌ Failed{': ' + err if err else ''}")

    # Show transcript breakdown
    if transcripts_used:
        print(f"\n📊 Transcript Breakdown:")
        for i, transcript in enumerate(transcripts_used, 1):
            quarter = transcript.get("quarter", "Q?")
            fiscal_year = transcript.get("fiscal_year", "????")
            chunk_count = transcript.get("chunk_count", 0)
            transcript_type = transcript.get("transcript_type", "earnings_call")
            transcript_date = transcript.get("transcript_date", "Unknown")
            print(f"   [{i}] {quarter} {fiscal_year} ({transcript_type}): {chunk_count} chunks")
            print(f"       Date: {transcript_date}")

    # Show processing summary
    processing_summary = data.get("processing_summary", {})
    if processing_summary:
        print(f"\n📈 Processing Summary:")
        print(f"   Total transcripts: {processing_summary.get('total_transcripts', 0)}")
        print(f"   Total chunks: {processing_summary.get('total_chunks_generated', 0)}")
        avg_chunks = processing_summary.get('avg_chunks_per_transcript', 0)
        print(f"   Avg chunks per transcript: {avg_chunks:.1f}")
    
    # Persist a small summary artifact in TranscriptRAG/data/summaries/
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = {
        "success": result.get("success"),
        "ticker": ticker.upper(),
        "quarters_back": quarters_back,
        "force_refresh": force_refresh,
        "chunk_count": chunk_count,
        "transcripts_used": transcripts_used,
        "vector_storage": vec,
        "processing_summary": processing_summary,
        "generated_at": datetime.now().isoformat(),
        "request_id": result.get("metadata", {}).get("request_id", "unknown"),
        "processing_time_ms": result.get("metadata", {}).get("processing_time_ms")
    }
    
    # Create summaries directory within TranscriptRAG
    summaries_dir = Path(__file__).parent.parent / "data" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = summaries_dir / f"transcript_ingestion_summary_{ticker.upper()}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to: {out_path}")

    return result


if __name__ == "__main__":
    # Interactive input for ticker selection
    print("🎯 TranscriptRAG Agent Interface - Transcript Ingestion")
    print("=" * 60)
    print("This script demonstrates the agent-ready interface for indexing transcripts")
    print("Standardized response format with error codes and request tracking\n")
    
    ticker = (input("Enter ticker (e.g., AAPL): ") or "AAPL").strip().upper()
    
    try:
        quarters_back = int(input("Quarters back (1-20, default 4): ") or "4")
        if quarters_back < 1 or quarters_back > 20:
            print("⚠️  Quarters back should be between 1 and 20, using default of 4")
            quarters_back = 4
    except ValueError:
        print("⚠️  Invalid input, using default of 4 quarters")
        quarters_back = 4
    
    # Ask about force refresh
    refresh_input = input("Force refresh (bypass cache)? [y/N]: ").strip().lower()
    force_refresh = refresh_input.startswith('y')
    
    print(f"\n🎯 Configuration:")
    print(f"   Ticker: {ticker}")
    print(f"   Quarters back: {quarters_back}")
    print(f"   Force refresh: {force_refresh}")
    print(f"   Data source: Alpha Vantage API")
    print(f"   Interface: Agent-Ready (Standardized Response)")

    try:
        result = ingest_transcripts(ticker, quarters_back, force_refresh)
        
        # Show final status
        if result.get("success"):
            print("\n🎉 Ingestion completed successfully!")
        else:
            print("\n💥 Ingestion failed. Check error details above.")
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()