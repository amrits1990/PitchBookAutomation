"""
Simple ingestion script that avoids the import issues
"""
import sys
import os
from pathlib import Path

# Add the AnnualReportRAG directory to Python path
annualreportrag_dir = Path(__file__).parent.parent
sys.path.insert(0, str(annualreportrag_dir))

# Change to the AnnualReportRAG directory so relative imports work
original_dir = os.getcwd()
os.chdir(annualreportrag_dir)

try:
    # Now import should work
    from agent_interface import index_reports_for_agent
    
    def ingest_reports(ticker="MSFT", years_back=1, filing_types=None):
        filing_types = filing_types or ["10-K", "10-Q"]
        print(f"\n=== Simple Ingestion: {ticker.upper()} ===")
        
        result = index_reports_for_agent(
            ticker=ticker, 
            years_back=years_back, 
            filing_types=filing_types
        )
        
        if result.get("success"):
            print(f"✅ Success! Processed {result.get('chunk_count', 0)} chunks")
            vector_storage = result.get("vector_storage", {})
            if vector_storage.get("success"):
                print("✅ Vector storage successful")
            else:
                print(f"⚠️ Vector storage failed: {vector_storage.get('error', 'Unknown error')}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    if __name__ == "__main__":
        # Test with MSFT
        ingest_reports("MSFT", years_back=1, filing_types=["10-K", "10-Q"])

finally:
    # Restore original directory
    os.chdir(original_dir)