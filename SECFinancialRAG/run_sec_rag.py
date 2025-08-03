#!/usr/bin/env python
"""
CLI entry point for SEC Financial RAG
Use this script to run the package directly from command line
"""

import sys
import os

# Add the parent directory to Python path so we can import from SECFinancialRAG
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: python SECFinancialRAG/run_sec_rag.py <ticker> [ticker2 ticker3 ...] [--validate] [--ltm]")
        print("Example: python SECFinancialRAG/run_sec_rag.py AAPL MSFT GOOGL")
        sys.exit(1)
    
    # Handle help flag
    if sys.argv[1] in ['--help', '-h', 'help']:
        print("SEC Financial RAG - Processing Financial Statements")
        print("=" * 60)
        print("Usage: python SECFinancialRAG/run_sec_rag.py <ticker> [ticker2 ticker3 ...] [--validate] [--ltm]")
        print("Example: python SECFinancialRAG/run_sec_rag.py AAPL MSFT GOOGL")
        print("Example: python SECFinancialRAG/run_sec_rag.py AAPL --validate --ltm")
        print("\nThis script processes SEC financial data for the specified company tickers.")
        print("It fetches 10-K and 10-Q filings, extracts financial statements,")
        print("and stores them in a PostgreSQL database for analysis.")
        print("\nOptions:")
        print("  --validate  Run data validation after processing")
        print("  --ltm       Generate LTM (Last Twelve Months) CSV files in ltm_exports/ directory")
        sys.exit(0)
    
    # Parse command line arguments
    args = sys.argv[1:]
    tickers = []
    generate_ltm = False
    validate_data = False
    
    for arg in args:
        if arg == '--ltm':
            generate_ltm = True
        elif arg == '--validate':
            validate_data = True
        elif not arg.startswith('--'):
            tickers.append(arg.upper())
        else:
            print(f"Unknown flag: {arg}")
            sys.exit(1)
    
    if not tickers:
        print("Error: No ticker symbols provided")
        print("Usage: python SECFinancialRAG/run_sec_rag.py <ticker> [ticker2 ticker3 ...] [--validate] [--ltm]")
        sys.exit(1)
    
    print("SEC Financial RAG - Processing Financial Statements")
    print("=" * 60)
    
    # Import after path setup
    from main import process_company_financials, process_multiple_companies, get_system_status
    
    # System status check
    print("Checking system status...")
    status = get_system_status()
    print(f"SEC API: {'[OK]' if status['sec_api_connection'] else '[FAIL]'}")
    print(f"Database: {'[OK]' if status['database_connection'] else '[FAIL]'}")
    
    if not status['sec_api_connection'] or not status['database_connection']:
        print("System checks failed. Please check configuration.")
        for error in status['errors']:
            print(f"Error: {error}")
        sys.exit(1)
    
    print(f"\nProcessing {len(tickers)} companies{'with validation and LTM generation' if validate_data and generate_ltm else ' with LTM generation' if generate_ltm else ' with validation' if validate_data else ''}...")
    
    if len(tickers) == 1:
        # Single company processing
        result = process_company_financials(tickers[0], validate_data=validate_data, generate_ltm=generate_ltm)
        print(f"\nResults for {tickers[0]}:")
        print(f"Status: {result['status']}")
        
        # Handle both success and error cases
        if result['status'] == 'success':
            print(f"Periods processed: {result.get('periods_processed', 0)}")
            print(f"Periods skipped: {result.get('periods_skipped', 0)}")
            
            if result.get('summary'):
                summary = result['summary']
                print(f"Total periods in DB: {summary['total_periods']}")
                print(f"Latest period: {summary['latest_period']}")
                print(f"Fiscal years: {summary['fiscal_years']}")
            
            if result.get('validation'):
                validation = result['validation']
                issues = validation.get('issues', [])
                print(f"Validation issues: {len(issues)}")
                for issue in issues:
                    print(f"  - {issue}")
            
            if result.get('ltm_export'):
                ltm_export = result['ltm_export']
                if 'error' in ltm_export:
                    print(f"LTM export error: {ltm_export['error']}")
                else:
                    successful_exports = sum(1 for success in ltm_export.values() if success)
                    print(f"LTM files generated: {successful_exports}/{len(ltm_export)} successful")
        else:
            # Error case
            print(f"Error: {result.get('error_message', 'Unknown error occurred')}")
            if result.get('metadata'):
                print(f"Metadata: {result['metadata']}")
    
    else:
        # Multiple companies processing
        results = process_multiple_companies(tickers, validate_data=validate_data)
        
        print(f"\nBatch Processing Results:")
        print("-" * 40)
        
        for ticker, result in results.items():
            status_icon = "[OK]" if result['status'] == 'success' else "[FAIL]"
            periods_processed = result.get('periods_processed', 0)
            print(f"{status_icon} {ticker}: {periods_processed} periods processed")
            
            if result.get('error_message'):
                print(f"    Error: {result['error_message']}")
        
        # Summary
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        total_periods = sum(r.get('periods_processed', 0) for r in results.values())
        
        print(f"\nSummary: {successful}/{len(tickers)} successful, {total_periods} total periods processed")
    
    print("\nProcessing completed!")

if __name__ == "__main__":
    main()
