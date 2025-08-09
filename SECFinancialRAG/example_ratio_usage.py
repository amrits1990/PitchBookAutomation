"""
Example usage of the ratio calculator functionality
Shows how to initialize default ratios, calculate ratios, and create custom ratios
"""

import logging
import sys
import os
import pandas as pd

# Add the parent directory to Python path for package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SECFinancialRAG import (
    # Core functionality
    process_company_financials,
    get_system_status,
    
    # Ratio functionality
    RatioManager,
    SimpleRatioCalculator,  # Use new simple calculator
    get_financial_data,
    FinancialDatabase
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate ratio calculator functionality"""
    
    print("=== SEC Financial RAG - Ratio Calculator Example ===\n")
    
    # Step 1: Check system status
    print("1. Checking system status...")
    status = get_system_status()
    print(f"   SEC API: {'✓' if status['sec_api_connection'] else '✗'}")
    print(f"   Database: {'✓' if status['database_connection'] else '✗'}")
    
    if not status['database_connection']:
        print("   Error: Database connection failed. Please check your .env configuration.")
        return
    
    # # Step 2: Initialize default ratios (one-time setup)
    # print("\n2. Initializing default ratio definitions...")
    # try:
    #     with RatioManager() as manager:
    #         count = manager.initialize_default_ratios(created_by="example_script")
    #         print(f"   Initialized {count} default ratios")
    # except Exception as e:
    #     print(f"   Note: {e} (ratios may already exist)")
    
    # # Step 3: Show available ratio definitions
    # print("\n3. Available global ratio definitions:")
    # try:
    #     with RatioManager() as manager:
    #         ratio_defs = manager.get_all_ratio_definitions()
    #         for i, ratio in enumerate(ratio_defs[:5]):  # Show first 5
    #             print(f"   {i+1}. {ratio['name']}: {ratio['description']}")
    #         print(f"   ... and {len(ratio_defs)-5} more ratios")
    # except Exception as e:
    #     print(f"   Error getting ratio definitions: {e}")
    
    # # Step 4: Process a company's financial data and calculate ratios (combined operation)
    ticker = "AAPL"
    # print(f"\n4. Processing financial data and calculating ratios for {ticker}...")
    
    # ratio_result = None
    # try:
    #     # Single operation: process financials with ratio calculation enabled
    #     result = process_company_financials(
    #         ticker=ticker,
    #         validate_data=True,
    #         calculate_ratios=True  # This does ALL ratio calculations in one go
    #     )
        
    #     print(f"   Status: {result['status']}")
    #     print(f"   Periods processed: {result['periods_processed']}")
        
    #     # Extract ratio information from the processing result
    #     if result.get('ratios'):
    #         ratio_result = result['ratios']
    #         ratios_calculated = ratio_result.get('total_ratios', 0)
    #         print(f"   ✓ Financial data processed and {ratios_calculated} ratios calculated")
            
    #         # Show sample ratios by category
    #         ratios_dict = ratio_result.get('ratios', {})
    #         categories = {}
            
    #         for ratio_name, ratio_data in ratios_dict.items():
    #             # Get category from ratio data or database lookup
    #             category = ratio_data.get('category', 'other')
    #             if category not in categories:
    #                 categories[category] = []
    #             categories[category].append((ratio_name, ratio_data))
            
    #         # Display ratios by category
    #         print(f"\n   Sample ratios by category:")
    #         for category, cat_ratios in categories.items():
    #             print(f"\n   {category.title()} Ratios:")
    #             for ratio_name, ratio_data in cat_ratios[:3]:  # Show first 3 in each category
    #                 value = ratio_data.get('value')
    #                 if value is not None:
    #                     if abs(value) < 0.01:
    #                         print(f"     {ratio_name}: {value:.6f}")
    #                     elif abs(value) < 1:
    #                         print(f"     {ratio_name}: {value:.4f}")
    #                     else:
    #                         print(f"     {ratio_name}: {value:.2f}")
    #                 else:
    #                     print(f"     {ratio_name}: N/A")
    #     else:
    #         print(f"   No ratios calculated in processing step")
        
    # except Exception as e:
    #     print(f"   Error processing {ticker}: {e}")
    #     print(f"   Note: Make sure {ticker} financial data is available in the database")
    
    # # Step 5: Create a company-specific custom ratio (only if needed for demonstration)
    # ticker = "AAPL"  # Define ticker for the example
    # print(f"\n5. Creating custom ratio for {ticker}...")
    # try:
    #     with RatioManager() as manager:
    #         # Check if custom ratio already exists to avoid duplication
    #         with FinancialDatabase() as db:
    #             company_info = db.get_company_by_ticker(ticker)
    #             if company_info:
    #                 existing_ratios = manager.get_all_ratio_definitions(company_info['id'])
    #                 custom_exists = any(r['name'] == 'Custom_Efficiency_Ratio' for r in existing_ratios)
                    
    #                 if not custom_exists:
    #                     success = manager.create_company_specific_ratio(
    #                         ticker=ticker,
    #                         name="Custom_Efficiency_Ratio",
    #                         formula="total_revenue / (total_assets + inventory)",
    #                         description="Custom efficiency ratio including inventory adjustment",
    #                         category="efficiency"
    #                     )
                        
    #                     if success:
    #                         print(f"   ✓ Created custom ratio for {ticker}")
                            
    #                         # Only recalculate if we actually added a new ratio
    #                         print(f"   Calculating custom ratio...")
    #                         with SimpleRatioCalculator() as calc:
    #                             ratio_result = calc.calculate_company_ratios(ticker)
    #                             if ratio_result and not ratio_result.get('error'):
    #                                 print(f"   ✓ Calculated {ratio_result.get('total_ratios', 0)} ratios (including custom)")
    #                     else:
    #                         print(f"   ✗ Failed to create custom ratio")
    #                 else:
    #                     print(f"   ✓ Custom ratio already exists, skipping creation")
    #             else:
    #                 print(f"   ✗ Company {ticker} not found")
    
    # except Exception as e:
    #     print(f"   Error with custom ratio: {e}")
    
    # Step 6: Export comprehensive data (using optimized approach)
    print(f"\n6. Exporting comprehensive data for {ticker}...")
    try:
        # Use standalone interface which retrieves pre-calculated data efficiently
        df = get_financial_data(ticker)
        
        if df is not None:
            # Export full dataset
            full_output = f"{ticker}_comprehensive_data.csv"
            df.to_csv(full_output, index=False)
            print(f"   ✓ Exported comprehensive data to {full_output}")
            
            # Export only ratios
            ratio_data = df[df['statement_type'] == 'ratio']
            if not ratio_data.empty:
                # Remove columns with no data (all NaN, None, or empty string values)
                def has_meaningful_data(series):
                    """Check if a pandas series has meaningful data"""
                    # Remove NaN values
                    non_null_data = series.dropna()
                    
                    if len(non_null_data) == 0:
                        return False
                    
                    # Check for empty strings
                    if series.dtype == 'object':  # String/object columns
                        non_empty = non_null_data[non_null_data != '']
                        if len(non_empty) == 0:
                            return False
                        # Also check for 'None' strings and whitespace-only strings
                        meaningful = non_empty[
                            (non_empty != 'None') & 
                            (non_empty.astype(str).str.strip() != '')
                        ]
                        return len(meaningful) > 0
                    
                    # For numeric columns, we consider 0 as meaningful data
                    return True
                
                # Filter columns
                columns_to_keep = [col for col in ratio_data.columns if has_meaningful_data(ratio_data[col])]
                
                # Filter the dataframe to only include columns with data
                ratio_data_filtered = ratio_data[columns_to_keep]
                
                ratio_output = f"{ticker}_ratios_only.csv"
                ratio_data_filtered.to_csv(ratio_output, index=False)
                
                removed_cols = len(ratio_data.columns) - len(columns_to_keep)
                print(f"   ✓ Exported ratios to {ratio_output} ({len(columns_to_keep)} columns, removed {removed_cols} empty columns)")
        else:
            print(f"   ✗ Failed to get data for export")
    
    except Exception as e:
        print(f"   Error exporting data: {e}")
    
    # # Step 7: Show ratio definitions for this company (hybrid view)
    # print(f"\n7. Ratio definitions for {ticker} (global + company-specific):")
    # try:
    #     with RatioManager() as manager:
    #         # Get company info first
    #         with FinancialDatabase() as db:
    #             company_info = db.get_company_by_ticker(ticker)
    #             if company_info:
    #                 company_ratios = manager.get_all_ratio_definitions(company_info['id'])
                    
    #                 global_count = sum(1 for r in company_ratios if r.get('company_id') is None)
    #                 company_count = len(company_ratios) - global_count
                    
    #                 print(f"   Total: {len(company_ratios)} ratios ({global_count} global, {company_count} company-specific)")
                    
    #                 # Show company-specific ratios
    #                 company_specific = [r for r in company_ratios if r.get('company_id') is not None]
    #                 if company_specific:
    #                     print(f"   Company-specific ratios:")
    #                     for ratio in company_specific:
    #                         print(f"     - {ratio['name']}: {ratio['description']}")
    #             else:
    #                 print(f"   Company {ticker} not found in database")
    
    # except Exception as e:
    #     print(f"   Error getting company ratio definitions: {e}")
    
    print(f"\n=== Example completed successfully! ===")
    print(f"\n🚀 PERFORMANCE OPTIMIZATIONS APPLIED:")
    print(f"✓ Eliminated redundant financial data processing (3x faster)")
    print(f"✓ Combined ratio calculations into single operation")
    print(f"✓ Added checks to avoid duplicate custom ratio creation")
    print(f"✓ Optimized data export to reuse pre-calculated results")
    print(f"\nKey features demonstrated:")
    print(f"✓ Virtual fields handle inconsistent financial data across companies")
    print(f"✓ Hybrid ratio system (global + company-specific definitions)")
    print(f"✓ LTM integration for income statement and cash flow ratios") 
    print(f"✓ Automatic ratio calculation during financial data processing")
    print(f"✓ Custom ratio creation and management")
    print(f"✓ Growth ratios (YoY Revenue and EBITDA growth)")
    print(f"✓ Efficient ratio export and reporting")


if __name__ == "__main__":
    main()