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
except ImportError:
    # If relative imports fail (when run directly), use absolute imports
    from simplified_processor import SimplifiedSECFinancialProcessor
    from models import ProcessingMetadata
    from sec_client import SECClient
    from database import FinancialDatabase
    from ltm_calculator import LTMCalculator

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
                              generate_ltm: bool = False, calculate_ratios: bool = False) -> Dict[str, Any]:
    """
    Process financial statements for a company by ticker
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        validate_data: Whether to run data validation after processing
        filing_preference: Which filing to use when multiple exist for same period ('original' or 'latest')
        generate_ltm: Whether to automatically generate LTM files after processing
        calculate_ratios: Whether to calculate financial ratios using LTM data
        
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
        
        # Generate LTM files if requested - works regardless of periods_processed count
        ltm_export_results = None
        if generate_ltm and metadata.processing_status == "success":
            try:
                logger.info(f"Generating LTM files for {ticker} (periods_processed={metadata.periods_processed}, periods_skipped={metadata.periods_skipped})")
                ltm_export_results = export_ltm_data(ticker, output_dir='ltm_exports')
                if any(ltm_export_results.values()):
                    logger.info(f"Successfully generated LTM files for {ticker}: {ltm_export_results}")
                else:
                    logger.warning(f"Failed to generate LTM files for {ticker}: {ltm_export_results}")
            except Exception as e:
                logger.error(f"Error generating LTM files for {ticker}: {e}")
                ltm_export_results = {'error': str(e)}
        elif generate_ltm:
            logger.warning(f"LTM generation skipped for {ticker} - processing_status: {metadata.processing_status}")
        else:
            logger.debug(f"LTM generation not requested for {ticker}")
        
        # Calculate financial ratios if requested
        ratio_results = None
        if calculate_ratios and metadata.processing_status == "success":
            try:
                logger.info(f"Calculating financial ratios for {ticker}")
                from .simple_ratio_calculator import calculate_ratios_simple as calculate_ratios_for_company
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
            'ltm_export': ltm_export_results,
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
            'validation': None
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


def export_ltm_data(ticker: str, output_prefix: str = None, 
                   statement_types: List[str] = ['income_statement', 'cash_flow'],
                   output_dir: str = None) -> Dict[str, bool]:
    """
    Export LTM and FY data to CSV files
    
    Args:
        ticker: Company ticker symbol
        output_prefix: Prefix for output files (default: ticker_ltm_)
        statement_types: List of statement types to export
        output_dir: Output directory (optional, will be created if specified)
        
    Returns:
        Dictionary with export success status for each statement type
    """
    logger.info(f"Exporting LTM data for {ticker}")
    
    # Handle output directory
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        if output_prefix is None:
            output_prefix = f"{output_dir}/{ticker}_ltm_"
        elif not output_prefix.startswith(output_dir):
            output_prefix = f"{output_dir}/{output_prefix}"
    elif output_prefix is None:
        output_prefix = f"{ticker}_ltm_"
    
    results = {}
    
    try:
        with LTMCalculator() as calculator:
            for statement_type in statement_types:
                output_file = f"{output_prefix}{statement_type}.csv"  # Back to CSV
                success = calculator.export_ltm_data(ticker, output_file, statement_type)
                results[statement_type] = success
                
                if success:
                    logger.info(f"Successfully exported {ticker} {statement_type} LTM and FY data to {output_file}")
                else:
                    logger.warning(f"Failed to export {ticker} {statement_type} LTM and FY data")
        
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