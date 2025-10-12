#!/usr/bin/env python3
"""
Debug script to investigate TGT vector database metadata
Check what fiscal years and quarters are actually stored vs what Company Facts reports
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_tgt_vector_db():
    """Debug TGT vector database to see what fiscal years/quarters are stored"""
    
    print("=" * 80)
    print("TGT Vector Database Metadata Debug")
    print("=" * 80)
    
    try:
        # Import vector store
        from annual_vector_store import AnnualVectorStore
        vector_store = AnnualVectorStore()
        print("✅ AnnualVectorStore loaded")
        
        table_name = "annual_reports_tgt"
        print(f"📋 Table name: {table_name}")
        
        # Check if table exists
        try:
            table_info = vector_store.get_table_info(table_name)
            print(f"✅ Table exists with {table_info.get('count', 0)} records")
        except Exception as e:
            print(f"❌ Table does not exist or error: {e}")
            return
        
        print(f"\n1. Getting ALL available chunks for TGT...")
        
        # Get all chunks for TGT (no filters)
        all_chunks = vector_store.hybrid_search(
            table_name=table_name,
            query="filing",
            k=200,  # Get many chunks
            filters={"ticker": "TGT"}
        )
        
        print(f"📊 Found {len(all_chunks)} total chunks for TGT")
        
        if not all_chunks:
            print("❌ No chunks found for TGT")
            return
        
        # Analyze metadata patterns
        form_types = {}
        fiscal_years = {}
        fiscal_quarters = {}
        sections = {}
        granularities = {}
        
        sample_10k = None
        sample_10q = None
        
        print(f"\n2. Analyzing metadata from all {len(all_chunks)} chunks...")
        
        for i, chunk in enumerate(all_chunks):
            metadata = chunk.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    continue
            
            # Track form types
            form_type = metadata.get('form_type')
            if form_type:
                form_types[form_type] = form_types.get(form_type, 0) + 1
            
            # Track fiscal years
            fiscal_year = metadata.get('fiscal_year')
            if fiscal_year:
                fiscal_years[str(fiscal_year)] = fiscal_years.get(str(fiscal_year), 0) + 1
            
            # Track fiscal quarters
            fiscal_quarter = metadata.get('fiscal_quarter')
            if fiscal_quarter:
                fiscal_quarters[fiscal_quarter] = fiscal_quarters.get(fiscal_quarter, 0) + 1
            
            # Track sections
            section_name = metadata.get('section_name')
            if section_name:
                sections[section_name] = sections.get(section_name, 0) + 1
            
            # Track granularities
            granularity = metadata.get('granularity')
            if granularity:
                granularities[granularity] = granularities.get(granularity, 0) + 1
            
            # Collect sample metadata
            if form_type == '10-K' and not sample_10k:
                sample_10k = metadata
            elif form_type == '10-Q' and not sample_10q:
                sample_10q = metadata
        
        # Print analysis
        print(f"\n3. Metadata Analysis:")
        print(f"   📋 Form Types: {dict(sorted(form_types.items()))}")
        print(f"   📅 Fiscal Years: {dict(sorted(fiscal_years.items()))}")
        print(f"   📆 Fiscal Quarters: {dict(sorted(fiscal_quarters.items()))}")
        print(f"   🔍 Granularities: {dict(sorted(granularities.items()))}")
        print(f"   📑 Sections ({len(sections)} unique): {list(sections.keys())[:5]}...")
        
        # Check specifically for FY 2024 10-K
        print(f"\n4. Specific Search for 10-K FY 2024:")
        fy2024_10k_chunks = vector_store.hybrid_search(
            table_name=table_name,
            query="filing",
            k=50,
            filters={"ticker": "TGT", "form_type": "10-K", "fiscal_year": "2024"}
        )
        print(f"   📊 Found {len(fy2024_10k_chunks)} chunks for 10-K FY 2024")
        
        # Check for FY 2025 10-Q Q2  
        print(f"\n5. Specific Search for 10-Q FY 2025 Q2:")
        fy2025_q2_chunks = vector_store.hybrid_search(
            table_name=table_name,
            query="filing",
            k=50,
            filters={"ticker": "TGT", "form_type": "10-Q", "fiscal_year": "2025", "fiscal_quarter": "Q2"}
        )
        print(f"   📊 Found {len(fy2025_q2_chunks)} chunks for 10-Q FY 2025 Q2")
        
        # Show sample complete metadata
        print(f"\n6. Sample Complete Metadata:")
        if sample_10k:
            print(f"\n   📄 Sample 10-K Metadata:")
            for key, value in sorted(sample_10k.items()):
                print(f"      {key}: {value}")
        
        if sample_10q:
            print(f"\n   📄 Sample 10-Q Metadata:")
            for key, value in sorted(sample_10q.items()):
                print(f"      {key}: {value}")
        
        # Look for any 2024 data
        print(f"\n7. Search for ANY 2024 data:")
        any_2024_chunks = vector_store.hybrid_search(
            table_name=table_name,
            query="filing",
            k=50,
            filters={"ticker": "TGT", "fiscal_year": "2024"}
        )
        print(f"   📊 Found {len(any_2024_chunks)} chunks with fiscal_year=2024")
        
        if any_2024_chunks:
            print(f"   📋 First 5 chunks with fiscal_year=2024:")
            for i, chunk in enumerate(any_2024_chunks[:5]):
                metadata = chunk.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        continue
                form_type = metadata.get('form_type', 'Unknown')
                fiscal_quarter = metadata.get('fiscal_quarter', 'N/A')
                section = metadata.get('section_name', 'Unknown')[:50]
                print(f"      {i+1}. {form_type} FY2024 {fiscal_quarter} - {section}...")
        
        # Look for recent annual reports
        print(f"\n8. Search for recent 10-K reports (any fiscal year):")
        recent_10k_chunks = vector_store.hybrid_search(
            table_name=table_name,
            query="filing",
            k=50,
            filters={"ticker": "TGT", "form_type": "10-K"}
        )
        print(f"   📊 Found {len(recent_10k_chunks)} total 10-K chunks")
        
        if recent_10k_chunks:
            # Extract unique fiscal years for 10-K
            tenk_years = set()
            for chunk in recent_10k_chunks:
                metadata = chunk.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        continue
                fiscal_year = metadata.get('fiscal_year')
                if fiscal_year:
                    tenk_years.add(str(fiscal_year))
            
            print(f"   📅 Available 10-K fiscal years: {sorted(tenk_years)}")
            
            # Show sample from latest available 10-K
            if tenk_years:
                latest_available_year = max(tenk_years)
                print(f"   📄 Sample metadata from latest available 10-K (FY {latest_available_year}):")
                for chunk in recent_10k_chunks:
                    metadata = chunk.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            continue
                    if str(metadata.get('fiscal_year')) == latest_available_year:
                        for key, value in sorted(metadata.items()):
                            print(f"      {key}: {value}")
                        break
        
        return True
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_tgt_vector_db()
    
    print(f"\n{'='*80}")
    if success:
        print("✅ Debug completed - check results above")
    else:
        print("❌ Debug failed - check errors above")
    print(f"{'='*80}")