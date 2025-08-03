"""
Example usage of the ratio calculator functionality
Shows how to initialize default ratios, calculate ratios, and create custom ratios
"""

import logging
from SECFinancialRAG import (
    # Core functionality
    process_company_financials,
    
    # Ratio functionality
    initialize_default_ratios,
    calculate_company_ratios,
    get_company_ratios,
    create_company_specific_ratio,
    get_ratio_definitions,
    export_ratio_data,
    
    # System status
    get_system_status
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
    
    # Step 2: Initialize default ratios (one-time setup)
    print("\n2. Initializing default ratio definitions...")
    try:
        count = initialize_default_ratios(created_by="example_script")
        print(f"   Initialized {count} default ratios")
    except Exception as e:
        print(f"   Note: {e} (ratios may already exist)")
    
    # Step 3: Show available ratio definitions
    print("\n3. Available global ratio definitions:")
    ratio_defs = get_ratio_definitions()
    for i, ratio in enumerate(ratio_defs[:5]):  # Show first 5
        print(f"   {i+1}. {ratio['name']}: {ratio['description']}")
    print(f"   ... and {len(ratio_defs)-5} more ratios")
    
    # Step 4: Process a company's financial data (required before calculating ratios)
    ticker = "AAPL"
    print(f"\n4. Processing financial data for {ticker}...")
    
    try:
        # Process with ratio calculation enabled
        result = process_company_financials(
            ticker=ticker,
            validate_data=True,
            calculate_ratios=True  # This is the key parameter
        )
        
        print(f"   Status: {result['status']}")
        print(f"   Periods processed: {result['periods_processed']}")
        
        if result.get('ratios'):
            ratios_calculated = result['ratios'].get('total_ratios', 0)
            print(f"   Ratios calculated: {ratios_calculated}")
        
    except Exception as e:
        print(f"   Error processing {ticker}: {e}")
        print(f"   Note: Make sure {ticker} financial data is available in the database")
    
    # Step 5: Get calculated ratios
    print(f"\n5. Retrieved calculated ratios for {ticker}:")
    try:
        ratios = get_company_ratios(ticker)
        
        if ratios:
            # Group by category
            categories = {}
            for ratio in ratios:
                cat = ratio.get('category', 'other')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(ratio)
            
            for category, cat_ratios in categories.items():
                print(f"\n   {category.title()} Ratios:")
                for ratio in cat_ratios[:3]:  # Show first 3 in each category
                    value = ratio.get('ratio_value')
                    if value is not None:
                        if abs(value) < 0.01:
                            print(f"     {ratio['name']}: {value:.6f}")
                        elif abs(value) < 1:
                            print(f"     {ratio['name']}: {value:.4f}")
                        else:
                            print(f"     {ratio['name']}: {value:.2f}")
                    else:
                        print(f"     {ratio['name']}: N/A")
        else:
            print(f"   No ratios found for {ticker}")
            print(f"   Make sure to process {ticker} with calculate_ratios=True first")
    
    except Exception as e:
        print(f"   Error getting ratios: {e}")
    
    # Step 6: Create a company-specific custom ratio
    print(f"\n6. Creating custom ratio for {ticker}...")
    try:
        success = create_company_specific_ratio(
            ticker=ticker,
            name="Custom_Efficiency_Ratio",
            formula="total_revenue / (total_assets + inventory)",
            description="Custom efficiency ratio including inventory adjustment",
            category="efficiency"
        )
        
        if success:
            print(f"   ✓ Created custom ratio for {ticker}")
            
            # Recalculate ratios to include the new custom ratio
            print(f"   Recalculating ratios to include custom ratio...")
            ratio_result = calculate_company_ratios(ticker)
            if ratio_result and not ratio_result.get('error'):
                print(f"   ✓ Recalculated {ratio_result.get('total_ratios', 0)} ratios")
        else:
            print(f"   ✗ Failed to create custom ratio")
    
    except Exception as e:
        print(f"   Error creating custom ratio: {e}")
    
    # Step 7: Export ratio data
    print(f"\n7. Exporting ratio data for {ticker}...")
    try:
        output_file = f"{ticker}_ratios.csv"
        success = export_ratio_data(ticker, output_file, format='csv')
        
        if success:
            print(f"   ✓ Exported ratios to {output_file}")
        else:
            print(f"   ✗ Failed to export ratios")
    
    except Exception as e:
        print(f"   Error exporting ratios: {e}")
    
    # Step 8: Show ratio definitions for this company (hybrid view)
    print(f"\n8. Ratio definitions for {ticker} (global + company-specific):")
    try:
        company_ratios = get_ratio_definitions(ticker)
        
        global_count = sum(1 for r in company_ratios if r.get('company_id') is None)
        company_count = len(company_ratios) - global_count
        
        print(f"   Total: {len(company_ratios)} ratios ({global_count} global, {company_count} company-specific)")
        
        # Show company-specific ratios
        company_specific = [r for r in company_ratios if r.get('company_id') is not None]
        if company_specific:
            print(f"   Company-specific ratios:")
            for ratio in company_specific:
                print(f"     - {ratio['name']}: {ratio['description']}")
    
    except Exception as e:
        print(f"   Error getting company ratio definitions: {e}")
    
    print(f"\n=== Example completed successfully! ===")
    print(f"\nKey features demonstrated:")
    print(f"✓ Virtual fields handle inconsistent financial data across companies")
    print(f"✓ Hybrid ratio system (global + company-specific definitions)")
    print(f"✓ LTM integration for income statement and cash flow ratios") 
    print(f"✓ Automatic ratio calculation during financial data processing")
    print(f"✓ Custom ratio creation and management")
    print(f"✓ Ratio export and reporting")


if __name__ == "__main__":
    main()