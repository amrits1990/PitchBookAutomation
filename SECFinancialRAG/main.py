"""
Main interface for SEC Financial RAG
Provides high-level functions for processing company financials
"""

import logging
from typing import List, Dict, Any, Optional
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .simplified_processor import SimplifiedSECFinancialProcessor
    from .models import ProcessingMetadata
    from .sec_client import SECClient
    from .database import FinancialDatabase
    from .ltm_calculator import LTMCalculator
    from .calculated_fields_processor import CalculatedFieldsProcessor
except ImportError:
    # If relative imports fail (when run directly), use absolute imports
    from simplified_processor import SimplifiedSECFinancialProcessor
    from models import ProcessingMetadata
    from sec_client import SECClient
    from database import FinancialDatabase
    from ltm_calculator import LTMCalculator
    from calculated_fields_processor import CalculatedFieldsProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sec_financial_rag.log') if os.path.exists('.') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


def process_company_financials(ticker: str, validate_data: bool = True, filing_preference: str = 'latest', 
                              generate_ltm: bool = False, calculate_ratios: bool = False, 
                              force_refresh: bool = False) -> Dict[str, Any]:
    """
    Process financial statements for a company by ticker
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        validate_data: Whether to run data validation after processing
        filing_preference: Which filing to use when multiple exist for same period ('original' or 'latest')
        generate_ltm: Whether to automatically generate LTM files after processing
        calculate_ratios: Whether to calculate financial ratios using LTM data
        force_refresh: If True, overwrite existing database records with fresh SEC data
        
    Returns:
        Dictionary with processing results and summary
    """
    logger.info(f"Starting financial data processing for {ticker} with filing preference: {filing_preference}")
    
    try:
        # Initialize components
        processor = SimplifiedSECFinancialProcessor(filing_preference=filing_preference)
        sec_client = SECClient()
        database = FinancialDatabase()
        
        # Get company CIK
        cik = sec_client.get_cik_from_ticker(ticker)
        if not cik:
            logger.error(f"Could not find CIK for ticker {ticker}")
            return {
                'status': 'error',
                'ticker': ticker,
                'error_message': f'Could not find CIK for ticker {ticker}',
                'metadata': None
            }
        
        # Fetch company facts
        company_facts = sec_client.get_company_facts(cik)
        if not company_facts:
            logger.error(f"Could not fetch company facts for {ticker}")
            return {
                'status': 'error',
                'ticker': ticker,
                'error_message': f'Could not fetch company facts for {ticker}',
                'metadata': None
            }
        
        # Process the company facts into statements
        statements = processor.process_company_facts(company_facts, ticker)
        if not statements:
            logger.error(f"No financial statements generated for {ticker}")
            return {
                'status': 'error',
                'ticker': ticker,
                'error_message': f'No financial statements generated for {ticker}',
                'metadata': None
            }
        
        # Check if data needs refresh (automatic 24-hour cache expiry)
        auto_refresh_triggered = False
        if not force_refresh:
            data_is_fresh = database.is_company_data_fresh(ticker, hours=24)
            if not data_is_fresh:
                force_refresh = True
                auto_refresh_triggered = True
                logger.info(f"Data for {ticker} is older than 24 hours - enabling force_refresh")
            else:
                logger.info(f"Using cached data for {ticker} (fresh within 24 hours)")
        
        # When auto-refresh is triggered, also enable LTM and ratio regeneration
        if auto_refresh_triggered:
            generate_ltm = True
            calculate_ratios = True
            logger.info(f"Auto-refresh triggered for {ticker} - enabling LTM generation and ratio calculation")
            
            # Clear all existing data (including LTM and ratio tables) before refresh
            logger.info(f"Clearing existing data for {ticker} before refresh")
            clear_success = database.force_refresh_company_data(ticker)
            if clear_success:
                logger.info(f"Successfully cleared existing data for {ticker}")
            else:
                logger.warning(f"Failed to clear existing data for {ticker} - continuing with refresh")
        
        # Store statements in database
        periods_processed = 0
        periods_skipped = 0
        company_id = None
        
        with database:
            # Ensure company exists in database
            try:
                from .models import Company
            except ImportError:
                from models import Company
            company_data = Company(
                cik=cik,
                ticker=ticker.upper(),
                name=company_facts.get('entityName', ticker),
                sic=company_facts.get('sic', ''),
                sic_description=company_facts.get('sicDescription', ''),
                ein=company_facts.get('ein', ''),
                description='',
                website='',
                investor_website='',
                category='',
                fiscal_year_end='',
                state_of_incorporation='',
                state_of_incorporation_description=''
            )
            company_id = database.insert_company(company_data)
            
            # Insert statements
            for statement_dict in statements:
                try:
                    # Add company_id to each statement
                    statement_dict['company_id'] = company_id
                    
                    # Fix period_type/period_length_months consistency issues
                    period_type = statement_dict.get('period_type')
                    period_length = statement_dict.get('period_length_months')
                    
                    # Ensure consistency: if period_type is FY, period_length must be 12
                    if period_type == 'FY' and period_length != 12:
                        logger.warning(f"Fixing period_type inconsistency: FY period with {period_length} months, changing to Q4")
                        statement_dict['period_type'] = 'Q4'
                    # Ensure consistency: if period_length is 12, period_type should be FY
                    elif period_length == 12 and period_type != 'FY':
                        logger.warning(f"Fixing period_length inconsistency: {period_type} period with 12 months, changing to FY")
                        statement_dict['period_type'] = 'FY'
                    
                    # Convert to appropriate model
                    if statement_dict.get('statement_type') == 'income_statement':
                        try:
                            from .models import IncomeStatement
                        except ImportError:
                            from models import IncomeStatement
                        statement_model = IncomeStatement(**statement_dict)
                    elif statement_dict.get('statement_type') == 'balance_sheet':
                        try:
                            from .models import BalanceSheet
                        except ImportError:
                            from models import BalanceSheet
                        statement_model = BalanceSheet(**statement_dict)
                    elif statement_dict.get('statement_type') == 'cash_flow':
                        try:
                            from .models import CashFlowStatement
                        except ImportError:
                            from models import CashFlowStatement
                        statement_model = CashFlowStatement(**statement_dict)
                    else:
                        logger.warning(f"Unknown statement type: {statement_dict.get('statement_type')}")
                        periods_skipped += 1
                        continue
                    
                    if force_refresh:
                        # Use upsert operation to update existing records or insert new ones
                        result = database.upsert_statement(statement_model)
                        if result == 'inserted' or result == 'updated':
                            periods_processed += 1
                        else:
                            periods_skipped += 1
                    else:
                        # Normal behavior: insert new records, skip duplicates
                        result_id = database.insert_statement(statement_model)
                        if result_id:
                            periods_processed += 1
                        else:
                            # Statement already exists (duplicate) - count as skipped but not an error
                            periods_skipped += 1
                        
                except Exception as e:
                    logger.error(f"Error inserting statement: {e}")
                    periods_skipped += 1
            
            # Determine processing status - if we have statements (processed or already existing), it's success
            total_statements_found = periods_processed + periods_skipped
            if total_statements_found > 0:
                processing_status = "success"
                error_message = None
                if periods_processed == 0:
                    logger.info(f"All {periods_skipped} statements for {ticker} already exist in database")
            else:
                processing_status = "error"
                error_message = "No financial statements could be processed"
            
            # Create and update metadata within the same connection
            metadata = ProcessingMetadata(
                ticker=ticker,
                cik=cik,
                processing_status=processing_status,
                periods_processed=periods_processed,
                periods_skipped=periods_skipped,
                error_message=error_message,
                last_processed=datetime.utcnow()
            )
            
            # Update metadata in database
            database.update_processing_metadata(metadata)
        
        # Run data validation if requested (with a new connection)
        validation_results = None
        if validate_data and metadata.processing_status == "success":
            try:
                with FinancialDatabase() as db:
                    # Get data from all statement types for comprehensive validation
                    income_statements = db.get_income_statements(ticker, limit=10)
                    balance_sheets = db.get_balance_sheets(ticker, limit=10) 
                    cash_flows = db.get_cash_flow_statements(ticker, limit=10)
                    
                    total_statements = len(income_statements) + len(balance_sheets) + len(cash_flows)
                    statements_with_revenue = len([s for s in income_statements if s.get('total_revenue')])
                    
                    # Find latest period and fiscal years from income statements
                    latest_period = income_statements[0].get('period_end_date') if income_statements else None
                    fiscal_years = list(set(s.get('fiscal_year') for s in income_statements if s.get('fiscal_year')))
                    fiscal_years.sort() if fiscal_years else None
                    
                    validation_results = {
                        'total_statements': total_statements,
                        'latest_period': latest_period,
                        'fiscal_years': fiscal_years,
                        'issues': [],
                        'summary': {
                            'income_statements': len(income_statements),
                            'balance_sheets': len(balance_sheets),
                            'cash_flow_statements': len(cash_flows),
                            'latest_revenue': income_statements[0].get('total_revenue') if income_statements else None,
                            'revenue_coverage': f"{statements_with_revenue}/{len(income_statements)}" if income_statements else "0/0"
                        }
                    }
                    
                    # Add validation issues
                    if total_statements == 0:
                        validation_results['issues'].append("No financial statements found in database")
                    elif statements_with_revenue / len(income_statements) < 0.5 if income_statements else False:
                        validation_results['issues'].append("Less than 50% of income statements have revenue data")
                        
            except Exception as e:
                logger.warning(f"Validation failed for {ticker}: {e}")
                validation_results = {'error': str(e)}
        
        # Clean duplicate period_end_date entries if processing was successful
        cleanup_results = None
        if metadata.processing_status == "success":
            try:
                logger.info(f"Cleaning duplicate period entries for {ticker}")
                cleanup_results = cleanup_duplicate_period_entries(ticker)
                if cleanup_results:
                    logger.info(f"Cleanup completed for {ticker}: {cleanup_results}")
                else:
                    logger.debug(f"No cleanup needed for {ticker}")
            except Exception as e:
                logger.error(f"Error during cleanup for {ticker}: {e}")
                cleanup_results = {'error': str(e)}
        
        # Process calculated fields if processing was successful
        calculated_fields_results = None
        if metadata.processing_status == "success":
            try:
                logger.info(f"Processing calculated fields for {ticker}")
                calc_processor = CalculatedFieldsProcessor()
                calculated_fields_results = calc_processor.process_calculated_fields_for_company(ticker)
                calc_processor.close()
                
                if calculated_fields_results and calculated_fields_results.get('total_fields_calculated', 0) > 0:
                    logger.info(f"Calculated fields completed for {ticker}: {calculated_fields_results['total_fields_calculated']} fields calculated")
                else:
                    logger.debug(f"No calculated fields needed for {ticker}")
            except Exception as e:
                logger.error(f"Error during calculated fields processing for {ticker}: {e}")
                calculated_fields_results = {'error': str(e)}
        
        # Process EBITDA post-calculations if processing was successful
        ebitda_results = None
        if metadata.processing_status == "success":
            try:
                logger.info(f"Processing EBITDA post-calculations for {ticker}")
                try:
                    from .ebitda_post_processor import run_ebitda_post_processing
                except ImportError:
                    from ebitda_post_processor import run_ebitda_post_processing
                
                ebitda_success = run_ebitda_post_processing(ticker, force_recalculate=force_refresh)
                ebitda_results = {'success': ebitda_success}
                
                if ebitda_success:
                    logger.info(f"EBITDA post-processing completed successfully for {ticker}")
                else:
                    logger.warning(f"EBITDA post-processing had issues for {ticker}")
            except Exception as e:
                logger.error(f"Error during EBITDA post-processing for {ticker}: {e}")
                ebitda_results = {'error': str(e)}
        
        # Store LTM data in database if requested - works regardless of periods_processed count
        ltm_storage_results = None
        if generate_ltm and metadata.processing_status == "success":
            try:
                logger.info(f"Storing LTM data in database for {ticker} (periods_processed={metadata.periods_processed}, periods_skipped={metadata.periods_skipped})")
                ltm_storage_results = store_ltm_data_in_database(ticker)
                if any(ltm_storage_results.values()):
                    logger.info(f"Successfully stored LTM data for {ticker}: {ltm_storage_results}")
                else:
                    logger.warning(f"Failed to store LTM data for {ticker}: {ltm_storage_results}")
            except Exception as e:
                logger.error(f"Error storing LTM data for {ticker}: {e}")
                ltm_storage_results = {'error': str(e)}
        elif generate_ltm:
            logger.warning(f"LTM generation skipped for {ticker} - processing_status: {metadata.processing_status}")
        else:
            logger.debug(f"LTM generation not requested for {ticker}")
        
        # Calculate financial ratios if requested
        ratio_results = None
        if calculate_ratios and metadata.processing_status == "success":
            try:
                logger.info(f"Calculating financial ratios for {ticker}")
                try:
                    from .simple_ratio_calculator import calculate_ratios_simple as calculate_ratios_for_company
                except ImportError:
                    # Fallback for when run directly
                    from simple_ratio_calculator import calculate_ratios_simple as calculate_ratios_for_company
                ratio_results = calculate_ratios_for_company(ticker)
                if ratio_results and 'ratios' in ratio_results:
                    logger.info(f"Successfully calculated {ratio_results.get('total_ratios', 0)} ratios for {ticker}")
                else:
                    logger.warning(f"No ratios calculated for {ticker}: {ratio_results}")
            except Exception as e:
                logger.error(f"Error calculating ratios for {ticker}: {e}")
                ratio_results = {'error': str(e)}
        elif calculate_ratios:
            logger.warning(f"Ratio calculation skipped for {ticker} - processing_status: {metadata.processing_status}")
        else:
            logger.debug(f"Ratio calculation not requested for {ticker}")
        
        result = {
            'ticker': ticker,
            'status': metadata.processing_status,  # Use 'status' consistently
            'processing_status': metadata.processing_status,  # Keep for backwards compatibility
            'periods_processed': metadata.periods_processed,
            'periods_skipped': metadata.periods_skipped, 
            'error_message': metadata.error_message,
            'summary': {
                'total_periods': periods_processed + periods_skipped,
                'company_name': company_facts.get('entityName', ticker),
                'cik': cik,
                'latest_period': None,  # Will be populated from validation if available
                'fiscal_years': None   # Will be populated from validation if available
            },
            'validation': validation_results,
            'cleanup': cleanup_results,
            'calculated_fields': calculated_fields_results,
            'ebitda_post_processing': ebitda_results,
            'ltm_storage': ltm_storage_results,
            'ratios': ratio_results
        }
        
        # Update summary with validation data if available
        if validation_results and 'summary' in validation_results:
            if 'latest_period' in validation_results:
                result['summary']['latest_period'] = validation_results.get('latest_period')
            if 'fiscal_years' in validation_results:
                result['summary']['fiscal_years'] = validation_results.get('fiscal_years')
        
        logger.info(f"Processing completed for {ticker}: {metadata.processing_status}")
        return result
            
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return {
            'ticker': ticker,
            'status': 'error',  # Use 'status' consistently
            'processing_status': 'error',  # Keep for backwards compatibility
            'error_message': str(e),
            'periods_processed': 0,
            'periods_skipped': 0,
            'summary': None,
            'validation': None,
            'calculated_fields': None
        }


def process_multiple_companies(tickers: List[str], validate_data: bool = True, filing_preference: str = 'latest', 
                              calculate_ratios: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Process financial statements for multiple companies
    
    Args:
        tickers: List of company ticker symbols
        validate_data: Whether to run data validation after processing
        filing_preference: Which filing to use when multiple exist for same period ('original' or 'latest')  
        calculate_ratios: Whether to calculate financial ratios using LTM data
        
    Returns:
        Dictionary mapping ticker to processing results
    """
    logger.info(f"Starting batch processing for {len(tickers)} companies with filing preference: {filing_preference}")
    
    results = {}
    
    for ticker in tickers:
        try:
            result = process_company_financials(ticker, validate_data, filing_preference, calculate_ratios=calculate_ratios)
            results[ticker] = result
            
            # Log progress
            if result['status'] in ['success', 'partial']:
                status_label = 'SUCCESS' if result['status'] == 'success' else 'PARTIAL'
                logger.info(f"[{status_label}] {ticker}: {result['periods_processed']} periods processed")
            else:
                logger.warning(f"[FAILED] {ticker}: {result['error_message']}")
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            results[ticker] = {
                'ticker': ticker,
                'status': 'error',
                'processing_status': 'error',  # Keep for backwards compatibility
                'error_message': str(e),
                'periods_processed': 0,
                'periods_skipped': 0
            }
    
    # Summary statistics
    successful = sum(1 for r in results.values() if r['status'] in ['success', 'partial'])
    total_periods = sum(r['periods_processed'] for r in results.values())
    
    logger.info(f"Batch processing completed: {successful}/{len(tickers)} successful, {total_periods} total periods")
    
    return results


def calculate_company_ltm(ticker: str, statement_types: List[str] = ['income_statement', 'cash_flow']) -> Dict[str, Any]:
    """
    Calculate Last Twelve Months (LTM) data for a company
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        statement_types: List of statement types to calculate LTM for
        
    Returns:
        Dictionary with LTM calculation results
    """
    logger.info(f"Calculating LTM data for {ticker}")
    
    try:
        with LTMCalculator() as calculator:
            results = {
                'ticker': ticker,
                'calculation_date': datetime.now().isoformat(),
                'ltm_data': {}
            }
            
            for statement_type in statement_types:
                logger.info(f"Calculating LTM for {ticker} {statement_type}")
                
                ltm_records = calculator.calculate_ltm_for_all_quarters(ticker, statement_type)
                ltm_summary = calculator.get_ltm_summary(ticker, statement_type)
                
                results['ltm_data'][statement_type] = {
                    'records': ltm_records,
                    'summary': ltm_summary,
                    'count': len(ltm_records)
                }
                
                logger.info(f"Calculated {len(ltm_records)} LTM periods for {ticker} {statement_type}")
            
            return results
            
    except Exception as e:
        logger.error(f"Error calculating LTM for {ticker}: {e}")
        return {
            'ticker': ticker,
            'error': str(e),
            'ltm_data': {}
        }


def store_ltm_data_in_database(ticker: str, 
                              statement_types: List[str] = ['income_statement', 'cash_flow']) -> Dict[str, bool]:
    """
    Store calculated LTM data in database tables
    
    Args:
        ticker: Company ticker symbol
        statement_types: List of statement types to store LTM data for
        
    Returns:
        Dictionary with storage success status for each statement type
    """
    logger.info(f"Storing LTM data in database for {ticker}")
    
    results = {}
    
    try:
        with LTMCalculator() as calculator:
            for statement_type in statement_types:
                
                # 1. Store LTM data in database (new future state)
                db_result = calculator.store_ltm_data_in_database(ticker, statement_type)
                db_success = db_result.get('success', False)
                
                
                if db_success:
                    logger.info(f"LTM DB SUCCESS: {ticker} {statement_type} - {db_result['inserted']} inserted, {db_result['skipped']} skipped")
                else:
                    logger.error(f"LTM DB FAILED: {ticker} {statement_type} - {db_result.get('error', 'Unknown error')}")
                
                # Success is based on database storage only
                results[statement_type] = db_success
        
        return results
        
    except Exception as e:
        logger.error(f"Error exporting LTM data for {ticker}: {e}")
        return {stmt_type: False for stmt_type in statement_types}


def get_company_summary(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get summary information for a company
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        Company summary dictionary or None if not found
    """
    try:
        with FinancialDatabase() as db:
            # Get basic summary info
            statements = db.get_income_statements(ticker, limit=10)
            if not statements:
                return None
            
            return {
                'company_name': statements[0].get('company_name', ticker),
                'ticker': ticker,
                'cik': statements[0].get('cik', ''),
                'total_periods': len(statements),
                'latest_period': statements[0].get('period_end_date') if statements else None,
                'latest_revenue': statements[0].get('total_revenue') if statements else None
            }
    except Exception as e:
        logger.error(f"Error getting summary for {ticker}: {e}")
        return None


def validate_company_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Validate data integrity for a company
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        Validation results dictionary or None if error
    """
    try:
        with FinancialDatabase() as db:
            # Simple validation
            income_statements = db.get_income_statements(ticker)
            balance_sheets = db.get_balance_sheets(ticker) 
            cash_flows = db.get_cash_flow_statements(ticker)
            
            total_statements = len(income_statements) + len(balance_sheets) + len(cash_flows)
            issues = []
            
            if total_statements == 0:
                issues.append("No financial statements found")
                
            statements_with_revenue = len([s for s in income_statements if s.get('total_revenue')])
            if income_statements and statements_with_revenue / len(income_statements) < 0.5:
                issues.append("Less than 50% of income statements have revenue data")
            
            return {
                'total_statements': total_statements,
                'issues': issues,
                'summary': {
                    'income_statements': len(income_statements),
                    'balance_sheets': len(balance_sheets),
                    'cash_flow_statements': len(cash_flows),
                    'revenue_coverage': f"{statements_with_revenue}/{len(income_statements)}" if income_statements else "0/0"
                }
            }
    except Exception as e:
        logger.error(f"Error validating {ticker}: {e}")
        return None


def test_sec_connection() -> bool:
    """
    Test connection to SEC API
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = SECClient()
        return client.validate_connection()
    except Exception as e:
        logger.error(f"SEC connection test failed: {e}")
        return False


def test_database_connection() -> bool:
    """
    Test connection to PostgreSQL database
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with FinancialDatabase() as db:
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def get_financial_statements_df(ticker: str) -> Optional[Any]:
    """
    Get financial statements as pandas DataFrame from all three tables
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        pandas DataFrame with financial statements or None if error
    """
    try:
        import pandas as pd
        
        with FinancialDatabase() as db:
            # Get statements from all three tables
            income_statements = db.get_income_statements(ticker)
            balance_sheets = db.get_balance_sheets(ticker)
            cash_flow_statements = db.get_cash_flow_statements(ticker)
            
            # Add statement type identifier to each record
            for stmt in income_statements:
                stmt['statement_type'] = 'income_statement'
            for stmt in balance_sheets:
                stmt['statement_type'] = 'balance_sheet'
            for stmt in cash_flow_statements:
                stmt['statement_type'] = 'cash_flow_statement'
            
            # Combine all statements
            all_statements = income_statements + balance_sheets + cash_flow_statements
            
            if all_statements:
                return pd.DataFrame(all_statements)
            else:
                logger.warning(f"No financial statements found for {ticker}")
                return None
                
    except ImportError:
        logger.error("pandas not installed - cannot return DataFrame")
        return None
    except Exception as e:
        logger.error(f"Error getting statements for {ticker}: {e}")
        return None


def export_financial_data(ticker: str, output_file: str, format: str = 'csv') -> bool:
    """
    Export financial data to file
    
    Args:
        ticker: Company ticker symbol
        output_file: Output file path
        format: Export format ('csv', 'json', 'excel')
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        df = get_financial_statements_df(ticker)
        if df is None or df.empty:
            logger.warning(f"No data to export for {ticker}")
            return False
        
        if format.lower() == 'csv':
            df.to_csv(output_file, index=False)
        elif format.lower() == 'json':
            df.to_json(output_file, orient='records', date_format='iso', indent=2)
        elif format.lower() == 'excel':
            df.to_excel(output_file, index=False)
        else:
            logger.error(f"Unsupported export format: {format}")
            return False
        
        logger.info(f"Exported {ticker} data to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting data for {ticker}: {e}")
        return False


def get_system_status() -> Dict[str, Any]:
    """
    Get system status and health check
    
    Returns:
        System status dictionary
    """
    status = {
        'timestamp': datetime.now().isoformat(),
        'sec_api_connection': False,
        'database_connection': False,
        'environment_variables': {},
        'errors': []
    }
    
    # Test SEC API connection
    try:
        status['sec_api_connection'] = test_sec_connection()
    except Exception as e:
        status['errors'].append(f"SEC API test failed: {e}")
    
    # Test database connection
    try:
        status['database_connection'] = test_database_connection()
    except Exception as e:
        status['errors'].append(f"Database test failed: {e}")
    
    # Check environment variables
    env_vars = ['SEC_USER_AGENT', 'DB_HOST', 'DB_NAME', 'DB_USER']
    for var in env_vars:
        value = os.getenv(var)
        if var == 'SEC_USER_AGENT':
            status['environment_variables'][var] = 'SET' if value else 'NOT SET'
        else:
            status['environment_variables'][var] = value or 'NOT SET'
    
    return status


# Ratio calculation functions
def calculate_company_ratios(ticker: str) -> Dict[str, Any]:
    """
    Calculate financial ratios for a company using LTM data
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        Dictionary with calculated ratios and metadata
    """
    try:
        from .simple_ratio_calculator import calculate_ratios_simple
        return calculate_ratios_simple(ticker)
    except ImportError:
        from simple_ratio_calculator import calculate_ratios_simple
        return calculate_ratios_simple(ticker)


def get_company_ratios(ticker: str, category: Optional[str] = None) -> List[Dict]:
    """
    Get stored financial ratios for a company
    
    Args:
        ticker: Company ticker symbol
        category: Filter by ratio category (optional)
        
    Returns:
        List of calculated ratio dictionaries
    """
    try:
        from .simple_ratio_calculator import get_stored_ratios_simple
        return get_stored_ratios_simple(ticker, category)
    except ImportError:
        from simple_ratio_calculator import get_stored_ratios_simple
        return get_stored_ratios_simple(ticker, category)


def initialize_default_ratios(created_by: str = "system") -> int:
    """
    Initialize default ratio definitions in the database
    
    Args:
        created_by: User who is initializing the ratios
        
    Returns:
        Number of ratios created
    """
    try:
        from .ratio_manager import initialize_default_ratios as init_ratios
        return init_ratios(created_by)
    except ImportError:
        from ratio_manager import initialize_default_ratios as init_ratios
        return init_ratios(created_by)


def create_company_specific_ratio(ticker: str, name: str, formula: str, 
                                description: str = None, category: str = None) -> bool:
    """
    Create a company-specific ratio definition
    
    Args:
        ticker: Company ticker
        name: Ratio name
        formula: Mathematical formula using field names
        description: Ratio description
        category: Ratio category
        
    Returns:
        True if created successfully, False otherwise
    """
    try:
        from .ratio_manager import create_company_ratio
        return create_company_ratio(ticker, name, formula, description, category)
    except ImportError:
        from ratio_manager import create_company_ratio
        return create_company_ratio(ticker, name, formula, description, category)


def get_ratio_definitions(ticker: str = None) -> List[Dict]:
    """
    Get ratio definitions for a company (or global if no ticker provided)
    
    Args:
        ticker: Company ticker (optional, returns global ratios if None)
        
    Returns:
        List of ratio definition dictionaries
    """
    try:
        if ticker:
            from .ratio_manager import get_company_ratio_definitions
            return get_company_ratio_definitions(ticker)
        else:
            from .ratio_manager import RatioManager
            with RatioManager() as manager:
                return manager.get_all_ratio_definitions()
    except ImportError:
        if ticker:
            from ratio_manager import get_company_ratio_definitions
            return get_company_ratio_definitions(ticker)
        else:
            from ratio_manager import RatioManager
            with RatioManager() as manager:
                return manager.get_all_ratio_definitions()


def export_ratio_data(ticker: str, output_file: str, format: str = 'csv') -> bool:
    """
    Export calculated ratios to file
    
    Args:
        ticker: Company ticker symbol
        output_file: Output file path
        format: Export format ('csv', 'json', 'excel')
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        ratios = get_company_ratios(ticker)
        if not ratios:
            logger.warning(f"No ratio data to export for {ticker}")
            return False
        
        import pandas as pd
        df = pd.DataFrame(ratios)
        
        if format.lower() == 'csv':
            df.to_csv(output_file, index=False)
        elif format.lower() == 'json':
            df.to_json(output_file, orient='records', date_format='iso', indent=2)
        elif format.lower() == 'excel':
            df.to_excel(output_file, index=False)
        else:
            logger.error(f"Unsupported export format: {format}")
            return False
        
        logger.info(f"Exported {ticker} ratios to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting ratio data for {ticker}: {e}")
        return False


# Agent Interface Functions (New)
def get_financial_metrics_for_agent(ticker: str, metrics: List[str], period: str = 'LTM'):
    """
    Agent-friendly function to get specific financial metrics
    
    Args:
        ticker: Company ticker symbol
        metrics: List of metric names
        period: Period type ('LTM', 'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'latest')
        
    Returns:
        FinancialAgentResponse with JSON-serializable data
    """
    try:
        from .agent_interface import get_financial_metrics_for_agent as agent_func
        return agent_func(ticker, metrics, period)
    except ImportError:
        from agent_interface import get_financial_metrics_for_agent as agent_func
        return agent_func(ticker, metrics, period)

def get_ratios_for_agent(ticker: str, categories: List[str] = None, period: str = 'LTM'):
    """
    Agent-friendly function to get calculated ratios with category filtering
    
    Args:
        ticker: Company ticker symbol
        categories: List of ratio categories (['profitability', 'liquidity', etc.]) or None for all
        period: Period type ('LTM', 'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'latest')
        
    Returns:
        FinancialAgentResponse with JSON-serializable ratio data organized by category
    """
    try:
        from .agent_interface import get_ratios_for_agent as agent_func
        return agent_func(ticker, categories, period)
    except ImportError:
        from agent_interface import get_ratios_for_agent as agent_func
        return agent_func(ticker, categories, period)

def get_ratio_definition_for_agent(ratio_name: str, ticker: str = None):
    """
    Agent-friendly function to get ratio definition with formula and interpretation
    
    Args:
        ratio_name: Name of the ratio (e.g., 'ROE', 'Current_Ratio')
        ticker: Optional company ticker for company-specific ratios
        
    Returns:
        FinancialAgentResponse with ratio definition, formula, interpretation, and guidance
    """
    try:
        from .agent_interface import get_ratio_definition_for_agent as agent_func
        return agent_func(ratio_name, ticker)
    except ImportError:
        from agent_interface import get_ratio_definition_for_agent as agent_func
        return agent_func(ratio_name, ticker)

def compare_companies_for_agent(tickers: List[str], metrics: List[str], period: str = 'LTM'):
    """
    Agent-friendly function to compare financial metrics across multiple companies
    
    Args:
        tickers: List of company ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        metrics: List of financial metrics to compare (e.g., ['total_revenue', 'net_income'])
        period: Period type ('LTM', 'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'latest')
        
    Returns:
        FinancialAgentResponse with comparative analysis including rankings and statistics
    """
    try:
        from .agent_interface import compare_companies_for_agent as agent_func
        return agent_func(tickers, metrics, period)
    except ImportError:
        from agent_interface import compare_companies_for_agent as agent_func
        return agent_func(tickers, metrics, period)


def cleanup_duplicate_period_entries(ticker: str) -> Dict[str, Any]:
    """
    Clean duplicate period_end_date entries in income_statement and cash_flow_statement tables.
    For each period_end_date, keep only the row with the highest number of non-null values.
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        Dictionary with cleanup results and statistics
    """
    logger.info(f"Starting cleanup of duplicate period entries for {ticker}")
    
    try:
        with FinancialDatabase() as db:
            # Get company info
            company_info = db.get_company_by_ticker(ticker)
            if not company_info:
                logger.warning(f"Company {ticker} not found for cleanup")
                return {'status': 'skipped', 'reason': 'Company not found'}
            
            cik = company_info['cik']
            
            cleanup_results = {
                'status': 'success',
                'income_statement_cleanup': {},
                'cash_flow_cleanup': {},
                'total_deleted': 0
            }
            
            # Clean income_statements table
            income_cleanup = _cleanup_table_duplicates(db, 'income_statements', cik, ticker)
            cleanup_results['income_statement_cleanup'] = income_cleanup
            cleanup_results['total_deleted'] += income_cleanup.get('deleted_count', 0)
            
            # Clean cash_flow_statements table
            cash_flow_cleanup = _cleanup_table_duplicates(db, 'cash_flow_statements', cik, ticker)
            cleanup_results['cash_flow_cleanup'] = cash_flow_cleanup
            cleanup_results['total_deleted'] += cash_flow_cleanup.get('deleted_count', 0)
            
            logger.info(f"Cleanup completed for {ticker}: {cleanup_results['total_deleted']} total rows deleted")
            return cleanup_results
            
    except Exception as e:
        logger.error(f"Error during cleanup for {ticker}: {e}")
        return {'status': 'error', 'error': str(e), 'total_deleted': 0}


def _cleanup_table_duplicates(db: FinancialDatabase, table_name: str, cik: str, ticker: str) -> Dict[str, Any]:
    """
    Clean duplicate period_end_date entries in a specific table.
    
    Args:
        db: Database connection
        table_name: Name of the table to clean ('income_statements' or 'cash_flow_statements')
        cik: Company CIK
        ticker: Company ticker
        
    Returns:
        Dictionary with cleanup results for this table
    """
    logger.info(f"Cleaning duplicates in {table_name} for {ticker}")
    
    try:
        with db.connection.cursor() as cursor:
            # Step 1: Find period_end_dates with multiple entries
            cursor.execute(f"""
                SELECT period_end_date, COUNT(*) as entry_count
                FROM {table_name}
                WHERE cik = %s
                GROUP BY period_end_date
                HAVING COUNT(*) > 1
                ORDER BY period_end_date DESC
            """, (cik,))
            
            duplicate_periods = cursor.fetchall()
            
            if not duplicate_periods:
                logger.debug(f"No duplicates found in {table_name} for {ticker}")
                return {
                    'duplicate_periods_found': 0,
                    'deleted_count': 0,
                    'kept_count': 0,
                    'details': []
                }
            
            logger.info(f"Found {len(duplicate_periods)} period_end_dates with duplicates in {table_name} for {ticker}")
            
            total_deleted = 0
            total_kept = 0
            cleanup_details = []
            
            # Step 2: For each duplicate period, find the row with most non-null values
            for period_end_date, entry_count in duplicate_periods:
                logger.debug(f"Processing duplicates for {ticker} {table_name} period {period_end_date} ({entry_count} entries)")
                
                # Get all rows for this period_end_date
                cursor.execute(f"""
                    SELECT id, *
                    FROM {table_name}
                    WHERE cik = %s AND period_end_date = %s
                    ORDER BY created_at DESC
                """, (cik, period_end_date))
                
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                if len(rows) <= 1:
                    continue
                
                # Step 3: Calculate non-null count for each row
                row_scores = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    
                    # Count non-null financial values (exclude metadata columns)
                    financial_columns = [col for col in columns if col not in [
                        'id', 'company_id', 'cik', 'ticker', 'company_name',
                        'created_at', 'updated_at', 'period_end_date', 'period_start_date',
                        'filing_date', 'period_type', 'form_type', 'fiscal_year', 'period_length_months'
                    ]]
                    
                    non_null_count = sum(1 for col in financial_columns 
                                       if row_dict.get(col) is not None and row_dict.get(col) != 0)
                    
                    row_scores.append({
                        'id': row_dict['id'],
                        'non_null_count': non_null_count,
                        'period_type': row_dict.get('period_type'),
                        'period_length_months': row_dict.get('period_length_months'),
                        'filing_date': row_dict.get('filing_date'),
                        'created_at': row_dict.get('created_at'),
                        'has_start_date': 1 if row_dict.get('period_start_date') is not None else 0
                    })
                
                # Step 4: Sort with start_date prioritization
                # Priority: 1) Has start_date (proper quarterly data), 2) Non-null count, 3) Creation date
                def sort_key(x):
                    return (x['has_start_date'], x['non_null_count'], x['created_at'])
                
                row_scores.sort(key=sort_key, reverse=True)
                
                # Keep the best row, delete the rest
                best_row = row_scores[0]
                rows_to_delete = row_scores[1:]
                
                logger.debug(f"For {ticker} {table_name} {period_end_date}: keeping row with {best_row['non_null_count']} non-null values (has_start_date: {best_row['has_start_date']}), deleting {len(rows_to_delete)} rows")
                
                # Step 5: Delete the inferior rows
                for row_to_delete in rows_to_delete:
                    cursor.execute(f"""
                        DELETE FROM {table_name}
                        WHERE id = %s
                    """, (row_to_delete['id'],))
                    total_deleted += 1
                
                total_kept += 1
                
                cleanup_details.append({
                    'period_end_date': str(period_end_date),
                    'entries_found': entry_count,
                    'entries_deleted': len(rows_to_delete),
                    'kept_entry_non_null_count': best_row['non_null_count'],
                    'kept_entry_period_type': best_row['period_type']
                })
            
            # Commit the changes
            db.connection.commit()
            
            result = {
                'duplicate_periods_found': len(duplicate_periods),
                'deleted_count': total_deleted,
                'kept_count': total_kept,
                'details': cleanup_details
            }
            
            logger.info(f"Completed cleanup of {table_name} for {ticker}: {total_deleted} deleted, {total_kept} kept")
            return result
            
    except Exception as e:
        db.connection.rollback()
        logger.error(f"Error cleaning {table_name} for {ticker}: {e}")
        raise