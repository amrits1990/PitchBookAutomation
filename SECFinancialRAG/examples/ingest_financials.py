"""
Financial Data Ingestion Example for SECFinancialRAG

This example demonstrates the complete financial data ingestion workflow:
1. Fetch raw financial data from SEC EDGAR API
2. Process and store financial statements in PostgreSQL database
3. Calculate and store all financial ratios (quarterly and annual)
4. Generate and store Last Twelve Months (LTM) data and ratios

This is an EXAMPLE file - the core functionality should be built into agent interface functions
for ultimate use by AI agents.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add the parent directory to Python path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from main import (
        process_company_financials, 
        get_system_status,
        initialize_default_ratios
    )
    from standalone_interface import get_financial_data
    from database import FinancialDatabase
    from ratio_manager import RatioManager
    print("OK - Successfully imported from package")
except ImportError as e:
    print(f"ERROR - Import error: {e}")
    print("Please run from the SECFinancialRAG directory or install the package")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_system_health() -> Dict[str, Any]:
    """Check if all system components are ready for ingestion"""
    print("[Checking system health...]")
    
    status = get_system_status()
    
    print(f"   SEC API Connection: {'OK' if status['sec_api_connection'] else 'FAILED'}")
    print(f"   Database Connection: {'OK' if status['database_connection'] else 'FAILED'}")
    
    if not status['database_connection']:
        print("   💡 Tip: Check your .env file for database configuration")
        return status
    
    # Check if ratio definitions are initialized
    try:
        with FinancialDatabase() as db:
            # Use a simple connection test to check if ratio_definitions table exists
            db._ensure_connection()
            with db.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM ratio_definitions WHERE company_id IS NULL")
                result = cursor.fetchone()
                ratio_count = result[0] if result else 0
                status['ratio_definitions_count'] = ratio_count
                if ratio_count > 0:
                    print(f"   Ratio Definitions: {ratio_count} global ratios available OK")
                else:
                    print("   Ratio Definitions: FAILED - No global ratios found")
    except Exception as e:
        print(f"   Ratio Definitions: ERROR - {e}")
        status['ratio_definitions_count'] = 0
    
    return status


def initialize_ratio_system() -> bool:
    """Initialize the ratio system with default definitions"""
    print("\n📊 Initializing ratio system...")
    
    try:
        with RatioManager() as manager:
            count = manager.initialize_default_ratios(created_by="ingest_example")
            if count > 0:
                print(f"   OK - Initialized {count} default ratio definitions")
            else:
                print("   OK - Default ratios already initialized")
            return True
    except Exception as e:
        print(f"   FAILED - Initialize ratios error: {e}")
        return False


def ingest_company_financials(ticker: str) -> Dict[str, Any]:
    """
    Ingest complete financial data for a company
    
    This is an EXAMPLE function - in the final agent system, this logic should be
    built into agent interface functions like:
    - ingest_financials_for_agent(ticker) -> AgentResponse
    - get_ingestion_status_for_agent(ticker) -> AgentResponse
    """
    print(f"\n💰 Ingesting financial data for {ticker.upper()}...")
    print(f"   📅 Using automatic 24-hour cache check")
    
    result = {
        'ticker': ticker.upper(),
        'success': False,
        'steps_completed': [],
        'errors': [],
        'data_summary': {}
    }
    
    try:
        # Step 1: Check if data is fresh and decide whether to process
        print(f"   🔄 Step 1: Checking data freshness...")
        
        should_process = True
        # Check if data is fresh (within last 24 hours)
        with FinancialDatabase() as db:
            if db.is_company_data_fresh(ticker.upper(), hours=24):
                print(f"   ✅ Using cached data for {ticker.upper()} (fresh within 24 hours)")
                should_process = False
                # Create a success result for cached data
                processing_result = {
                    'status': 'success',
                    'ticker': ticker.upper(),
                    'periods_processed': 0,
                    'periods_skipped': 0,
                    'summary': {'cached_data_used': True}
                }
            else:
                print(f"   🔄 Data for {ticker.upper()} is stale, fetching from SEC...")
        
        if should_process:
            print(f"   🔄 Step 1b: Fetching and processing SEC financial data...")
            processing_result = process_company_financials(
                ticker=ticker.upper(),
                calculate_ratios=True,       # Calculate ratios during processing
                generate_ltm=True,           # Generate LTM data
                validate_data=True
                # force_refresh defaults to False, so 24-hour cache check will be used
            )
            
            # Update the company timestamp after successful processing
            if processing_result.get('status') == 'success':
                with FinancialDatabase() as db:
                    db.update_company_timestamp(ticker.upper())
        
        if processing_result.get('status') == 'success':
            result['steps_completed'].append('sec_data_processing')
            print(f"   OK - SEC data processed successfully")
            
            # Extract summary information
            summary = processing_result.get('summary', {})
            result['data_summary'].update({
                'periods_processed': processing_result.get('periods_processed', 0),
                'periods_skipped': processing_result.get('periods_skipped', 0),
                'total_statements': summary.get('total_periods', 0)
            })
            
        else:
            error_msg = processing_result.get('error_message', 'Unknown error in SEC data processing')
            result['errors'].append(f"SEC processing failed: {error_msg}")
            print(f"   FAILED - SEC data processing: {error_msg}")
            return result
        
        # Step 2: Verify LTM data in database
        print(f"   🔄 Step 2: Verifying LTM data in database...")
        
        try:
            with FinancialDatabase() as db:
                # Check for LTM data in database
                ltm_income = db.get_ltm_income_statements(ticker.upper(), limit=1)
                ltm_cash_flow = db.get_ltm_cash_flow_statements(ticker.upper(), limit=1)
                
                ltm_count = len(ltm_income) + len(ltm_cash_flow)
                
                if ltm_count > 0:
                    result['steps_completed'].append('ltm_generation')
                    result['data_summary']['ltm_records_in_db'] = ltm_count
                    print(f"   OK - LTM data found in database: {ltm_count} records")
                else:
                    result['errors'].append("LTM data generation failed - no database records found")
                    print(f"   ⚠️  LTM data not found in database")
        except Exception as e:
            result['errors'].append(f"Error checking LTM data: {str(e)}")
            print(f"   ERROR - Checking LTM data: {e}")
        
        # Step 3: Verify ratio calculations
        print(f"   🔄 Step 3: Verifying ratio calculations...")
        
        try:
            with FinancialDatabase() as db:
                # Count calculated ratios for this company using direct SQL
                db._ensure_connection()
                with db.connection.cursor() as cursor:
                    ratios_query = """
                        SELECT COUNT(*) 
                        FROM calculated_ratios cr
                        JOIN companies c ON cr.company_id = c.id
                        WHERE c.ticker = %s
                    """
                    cursor.execute(ratios_query, (ticker.upper(),))
                    result_row = cursor.fetchone()
                    ratio_count = result_row[0] if result_row else 0
                    
                    if ratio_count > 0:
                        result['steps_completed'].append('ratio_calculation')
                        result['data_summary']['calculated_ratios'] = ratio_count
                        print(f"   OK - Calculated ratios: {ratio_count} ratios stored")
                    else:
                        result['errors'].append("No calculated ratios found in database")
                        print(f"   FAILED - No calculated ratios found")
                    
        except Exception as e:
            result['errors'].append(f"Error verifying ratios: {str(e)}")
            print(f"   ERROR - Verifying ratios: {e}")
        
        # Step 4: Test data retrieval
        print(f"   🔄 Step 4: Testing comprehensive data retrieval...")
        
        try:
            df = get_financial_data(ticker.upper())
            if df is not None and not df.empty:
                result['steps_completed'].append('data_retrieval')
                result['data_summary'].update({
                    'total_records': len(df),
                    'columns_available': len(df.columns),
                    'data_types': df['data_source'].value_counts().to_dict() if 'data_source' in df.columns else {}
                })
                print(f"   OK - Data retrieval successful: {len(df)} records, {len(df.columns)} columns")
            else:
                result['errors'].append("Data retrieval returned empty DataFrame")
                print(f"   FAILED - Data retrieval failed - empty result")
        except Exception as e:
            result['errors'].append(f"Data retrieval error: {str(e)}")
            print(f"   ERROR - Data retrieval: {e}")
        
        # Mark as successful if we completed core steps
        if 'sec_data_processing' in result['steps_completed'] and 'data_retrieval' in result['steps_completed']:
            result['success'] = True
            print(f"   🎉 Ingestion completed successfully for {ticker.upper()}")
        else:
            print(f"   ⚠️  Ingestion completed with issues for {ticker.upper()}")
            
    except Exception as e:
        result['errors'].append(f"Unexpected error: {str(e)}")
        print(f"   ERROR - Unexpected error during ingestion: {e}")
        logger.exception(f"Error ingesting data for {ticker}")
    
    return result


def ingest_multiple_companies(tickers: List[str]) -> Dict[str, Any]:
    """
    Ingest financial data for multiple companies
    
    This is an EXAMPLE function - in the final agent system, this should be:
    - ingest_multiple_financials_for_agent(tickers) -> AgentResponse
    """
    print(f"\n📊 Batch ingestion for {len(tickers)} companies...")
    
    results = {
        'total_companies': len(tickers),
        'successful': [],
        'failed': [],
        'summary': {}
    }
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker.upper()}...")
        
        company_result = ingest_company_financials(ticker)
        
        if company_result['success']:
            results['successful'].append(ticker.upper())
        else:
            results['failed'].append({
                'ticker': ticker.upper(),
                'errors': company_result['errors']
            })
    
    results['summary'] = {
        'success_rate': len(results['successful']) / len(tickers) * 100,
        'companies_successful': len(results['successful']),
        'companies_failed': len(results['failed'])
    }
    
    print(f"\n📈 Batch Ingestion Summary:")
    print(f"   OK - Successful: {len(results['successful'])}/{len(tickers)} companies")
    print(f"   FAILED - {len(results['failed'])}/{len(tickers)} companies")
    print(f"   📊 Success Rate: {results['summary']['success_rate']:.1f}%")
    
    return results


def interactive_ingestion():
    """Interactive mode for testing financial data ingestion"""
    print("🏦 SECFinancialRAG - Financial Data Ingestion Example")
    print("=" * 60)
    print("This example demonstrates complete financial data ingestion workflow")
    print("NOTE: This is demonstration code - core functionality should be in agent interfaces\n")
    
    # System health check
    status = check_system_health()
    
    if not status['database_connection']:
        print("\nERROR - Cannot proceed without database connection")
        return
    
    # Initialize ratio system if needed
    if status.get('ratio_definitions_count', 0) == 0:
        if not initialize_ratio_system():
            print("\nERROR - Cannot proceed without ratio system initialization")
            return
    
    # Get ticker input
    print(f"\n📝 Enter ticker symbols to ingest financial data")
    print(f"   Examples: AAPL, MSFT, GOOGL, TSLA")
    
    ticker_input = input("   Ticker(s) [comma-separated]: ").strip()
    
    if not ticker_input:
        print("ERROR - No ticker provided")
        return
    
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    
    print(f"\n🚀 Starting ingestion for: {', '.join(tickers)}")
    print("   📅 Using automatic 24-hour cache check")
    
    # Execute ingestion
    start_time = datetime.now()
    
    if len(tickers) == 1:
        result = ingest_company_financials(tickers[0])
        
        print(f"\n📋 Ingestion Results for {tickers[0]}:")
        print(f"   Status: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"   Steps Completed: {', '.join(result['steps_completed'])}")
        
        if result['errors']:
            print(f"   Errors: {len(result['errors'])}")
            for error in result['errors']:
                print(f"     - {error}")
        
        if result['data_summary']:
            print(f"   Data Summary:")
            for key, value in result['data_summary'].items():
                print(f"     - {key}: {value}")
                
    else:
        results = ingest_multiple_companies(tickers)
        
        print(f"\n📋 Batch Ingestion Results:")
        if results['successful']:
            print(f"   SUCCESS - {', '.join(results['successful'])}")
        if results['failed']:
            print(f"   FAILED - {', '.join([f['ticker'] for f in results['failed']])}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n⏱️  Total processing time: {duration:.1f} seconds")
    print(f"🎯 Ingestion workflow complete!")
    
    # Next steps guidance
    print(f"\n💡 Next Steps:")
    print(f"   1. Use examples/query_financials.py to test data retrieval")
    print(f"   2. Check the database for stored financial statements and ratios")
    print(f"   3. Review LTM data in database tables (ltm_income_statements, ltm_cash_flow_statements)")
    print(f"   4. NOTE: In the final agent system, this will be agent interface functions")


if __name__ == "__main__":
    try:
        interactive_ingestion()
    except KeyboardInterrupt:
        print("\n\n👋 Ingestion cancelled by user")
    except Exception as e:
        print(f"\nERROR - Unexpected error: {e}")
        logger.exception("Unexpected error in ingestion example")