"""
Example usage of SECFinancialRAG as a standalone package
Demonstrates the simple interface for external projects
"""

import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Demonstrate standalone package usage"""
    
    print("=== SECFinancialRAG Standalone Package Demo ===\n")
    
    # Import the main function - this is all external packages need
    try:
        # Try importing as installed package first
        from SECFinancialRAG import get_financial_data, get_multiple_companies_data
        print("✓ Successfully imported SECFinancialRAG")
    except ImportError as e:
        # If not installed, try importing from current directory
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from standalone_interface import get_financial_data, get_multiple_companies_data
            print("✓ Successfully imported from local directory")
        except ImportError as e2:
            print(f"✗ Import error: {e}")
            print(f"✗ Local import error: {e2}")
            print("Either install the package with 'pip install -e .' or run from the package directory")
            return
    
    # Example 1: Get comprehensive data for a single company
    print("\n1. Getting comprehensive financial data for AAPL...")
    
    try:
        df_aapl = get_financial_data('AAPL')
        
        if df_aapl is not None:
            print(f"   ✓ Retrieved data: {len(df_aapl)} records, {len(df_aapl.columns)} columns")
            
            # Check if expected columns exist
            if 'period_end_date' in df_aapl.columns:
                print(f"   ✓ Date range: {df_aapl['period_end_date'].min()} to {df_aapl['period_end_date'].max()}")
            else:
                print("   ⚠ period_end_date column not found")
            
            # Show data sources
            if 'data_source' in df_aapl.columns:
                data_sources = df_aapl['data_source'].value_counts()
                print(f"   ✓ Data sources: {dict(data_sources)}")
            else:
                print("   ⚠ data_source column not found")
            
            # Show statement types
            if 'statement_type' in df_aapl.columns:
                stmt_types = df_aapl['statement_type'].value_counts()
                print(f"   ✓ Statement types: {dict(stmt_types)}")
            else:
                print("   ⚠ statement_type column not found")
            
            # Show sample of financial metrics
            print("\n   Sample Financial Metrics:")
            sample_cols = ['period_end_date', 'data_source', 'total_revenue', 'net_income', 'total_assets']
            available_cols = [col for col in sample_cols if col in df_aapl.columns]
            if available_cols:
                print(df_aapl[available_cols].head(3).to_string(index=False))
            else:
                print("   Note: Expected financial columns not found. Available columns:")
                print(f"   {list(df_aapl.columns)[:10]}...")  # Show first 10 columns
            
            # Show sample ratios
            if 'statement_type' in df_aapl.columns:
                ratio_data = df_aapl[df_aapl['statement_type'] == 'ratio']
                if not ratio_data.empty:
                    print(f"\n   Sample Ratios ({len(ratio_data)} total):")
                    ratio_cols = ['period_end_date', 'name', 'ratio_value', 'description', 'category']
                    available_ratio_cols = [col for col in ratio_cols if col in ratio_data.columns]
                    if available_ratio_cols:
                        print(ratio_data[available_ratio_cols].head(3).to_string(index=False))
                    else:
                        print("   Available ratio columns:", list(ratio_data.columns))
        else:
            print("   ✗ No data retrieved for AAPL")
            
    except Exception as e:
        print(f"   ✗ Error getting AAPL data: {e}")
    
    # # Example 2: Get data for multiple companies
    # print("\n2. Getting data for multiple companies...")
    
    # try:
    #     tickers = ['AAPL', 'MSFT']
    #     df_multi = get_multiple_companies_data(tickers)
        
    #     if df_multi is not None:
    #         print(f"   ✓ Retrieved data for {len(df_multi['ticker'].unique())} companies")
    #         print(f"   ✓ Total records: {len(df_multi)}")
            
    #         # Show data by company
    #         company_counts = df_multi['ticker'].value_counts()
    #         print(f"   ✓ Records per company: {dict(company_counts)}")
            
    #     else:
    #         print("   ✗ No data retrieved for multiple companies")
            
    # except Exception as e:
    #     print(f"   ✗ Error getting multi-company data: {e}")
    
    # # Example 3: DataFrame operations
    # print("\n3. DataFrame operations example...")
    
    # if 'df_aapl' in locals() and df_aapl is not None:
    #     try:
    #         # Filter to LTM data only
    #         if 'data_source' in df_aapl.columns:
    #             ltm_data = df_aapl[df_aapl['data_source'] == 'LTM']
    #             print(f"   ✓ LTM records: {len(ltm_data)}")
    #         else:
    #             print(f"   ⚠ Cannot filter LTM data - data_source column missing")
    #             ltm_data = pd.DataFrame()
            
    #         # Filter to ratios only  
    #         if 'statement_type' in df_aapl.columns:
    #             ratio_data = df_aapl[df_aapl['statement_type'] == 'ratio']
    #             print(f"   ✓ Ratio records: {len(ratio_data)}")
                
    #             # Filter to balance sheet only
    #             bs_data = df_aapl[df_aapl['statement_type'] == 'balance_sheet']
    #             print(f"   ✓ Balance sheet records: {len(bs_data)}")
    #         else:
    #             print(f"   ⚠ Cannot filter by statement type - statement_type column missing")
    #             ratio_data = pd.DataFrame()
    #             bs_data = pd.DataFrame()
            
    #         # Example analysis - calculate some metrics
    #         if not ltm_data.empty and 'total_revenue' in ltm_data.columns:
    #             # Sort by date to get most recent first
    #             if 'period_end_date' in ltm_data.columns:
    #                 ltm_data_sorted = ltm_data.sort_values('period_end_date', ascending=False)
    #                 latest_revenue = ltm_data_sorted['total_revenue'].iloc[0]
    #             else:
    #                 latest_revenue = ltm_data['total_revenue'].iloc[0]
                
    #             if latest_revenue and pd.notna(latest_revenue):
    #                 print(f"   ✓ Latest LTM Revenue: ${latest_revenue:,.0f}")
    #             else:
    #                 print(f"   ⚠ Latest LTM Revenue: No valid data")
    #         elif not ltm_data.empty:
    #             print(f"   ⚠ LTM data found but no total_revenue column")
    #         else:
    #             print(f"   ⚠ No LTM data available for analysis")
            
    #     except Exception as e:
    #         print(f"   ✗ Error with DataFrame operations: {e}")
    #         # Add debug info
    #         print(f"       DataFrame shape: {df_aapl.shape}")
    #         print(f"       DataFrame columns: {list(df_aapl.columns)[:5]}...")  # First 5 columns
    
    print("\n=== Demo completed successfully! ===")
    print("\nFor external projects, simply use:")
    print("```python")
    print("import SECFinancialRAG as sfr")
    print("df = sfr.get_financial_data('AAPL')")
    print("# df now contains comprehensive financial data")
    print("```")


if __name__ == "__main__":
    main()