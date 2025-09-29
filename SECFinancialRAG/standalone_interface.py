"""
Standalone Interface for SEC Financial RAG Package
Returns comprehensive financial data as a single pandas DataFrame
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, date
import warnings

try:
    from .main import process_company_financials, initialize_default_ratios
    from .ltm_calculator import LTMCalculator
    from .simple_ratio_calculator import SimpleRatioCalculator
    from .database import FinancialDatabase
    from .models import Company
except ImportError:
    from main import process_company_financials, initialize_default_ratios
    from ltm_calculator import LTMCalculator
    from simple_ratio_calculator import SimpleRatioCalculator
    from database import FinancialDatabase
    from models import Company

logger = logging.getLogger(__name__)


def get_company_financial_data(ticker: str, auto_process: bool = True, 
                             include_ratios: bool = True) -> Optional[pd.DataFrame]:
    """
    Standalone interface to get comprehensive financial data for a company
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        auto_process: Whether to automatically process financial data if not found
        include_ratios: Whether to include calculated ratios in the DataFrame
        
    Returns:
        pandas DataFrame with comprehensive financial data or None if error
        
    DataFrame Structure:
        - period_end_date: Date index
        - period_type: Q1, Q2, Q3, Q4, FY, LTM
        - data_source: LTM, quarterly, annual, point_in_time
        - All LTM income statement fields (when data_source='LTM')
        - All LTM cash flow fields (when data_source='LTM') 
        - All point-in-time balance sheet fields (when data_source='point_in_time')
        - All calculated ratios (when include_ratios=True)
    """
    logger.info(f"Getting comprehensive financial data for {ticker}")
    
    try:
        # Step 1: Check cache and ensure company data exists
        should_process = True
        if auto_process:
            # Check if data is fresh (within last 24 hours)
            with FinancialDatabase() as db:
                if db.is_company_data_fresh(ticker, hours=24):
                    logger.info(f"Using cached data for {ticker} (fresh within 24 hours)")
                    should_process = False
                else:
                    logger.info(f"Data for {ticker} is stale, will refresh from SEC")
            
            if should_process:
                logger.info(f"Auto-processing financial data for {ticker}")
                process_result = process_company_financials(
                    ticker=ticker,
                    validate_data=True,
                    generate_ltm=True,        # ← ADD THIS: Always generate LTM data for caching
                    calculate_ratios=include_ratios
                )
                
                if process_result.get('status') != 'success':
                    logger.error(f"Failed to process {ticker}: {process_result.get('error_message')}")
                    return None
                
                # Update the company timestamp after successful processing
                with FinancialDatabase() as db:
                    db.update_company_timestamp(ticker)
        
        # Step 2: Initialize default ratios if needed
        if include_ratios:
            try:
                initialize_default_ratios()
            except Exception as e:
                logger.debug(f"Default ratios already initialized or error: {e}")
        
        # Step 3: Collect all financial data
        all_data_records = []
        
        with FinancialDatabase() as db:
            # Get company info
            company_info = db.get_company_by_ticker(ticker)
            if not company_info:
                logger.error(f"Company {ticker} not found in database")
                return None
            
            # Get all LTM data
            ltm_records = _get_all_ltm_data(ticker)
            all_data_records.extend(ltm_records)
            
            # Get all balance sheet data (point-in-time)
            balance_sheet_records = _get_balance_sheet_data(ticker, db)
            all_data_records.extend(balance_sheet_records)
            
            # Get all calculated ratios if requested
            if include_ratios:
                ratio_records = _get_ratio_data(ticker, db)
                all_data_records.extend(ratio_records)
        
        # Step 4: Convert to DataFrame
        if not all_data_records:
            logger.warning(f"No financial data found for {ticker}")
            return None
        
        df = pd.DataFrame(all_data_records)
        
        # Step 5: Clean and structure DataFrame
        df = _structure_dataframe(df, ticker)
        
        logger.info(f"Retrieved comprehensive data for {ticker}: {len(df)} records, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error getting financial data for {ticker}: {e}")
        return None


def _get_all_ltm_data(ticker: str) -> List[Dict[str, Any]]:
    """Get all LTM data for income statement and cash flow (24-hour cache aware)"""
    ltm_records = []
    
    try:
        with FinancialDatabase() as db:
            # Check if company data is fresh (within 24 hours) AND LTM data exists
            if db.is_company_data_fresh(ticker, hours=24):
                # Get LTM income statement data from database
                ltm_income_statements = db.get_ltm_income_statements(ticker)
                ltm_cash_flows = db.get_ltm_cash_flow_statements(ticker)
                
                if ltm_income_statements or ltm_cash_flows:
                    logger.info(f"Using cached LTM data for {ticker} (fresh within 24 hours)")
                    
                    # Process income statements
                    for record in ltm_income_statements:
                        # Convert database record to standard format
                        record_dict = dict(record)
                        record_dict['data_source'] = 'LTM'
                        record_dict['statement_type'] = 'income_statement'
                        record_dict['period_type'] = 'LTM'
                        record_dict['period_end_date'] = record_dict.get('period_end_date')  # Use standardized field name
                        record_dict['ticker'] = ticker
                        ltm_records.append(record_dict)
                    
                    # Process cash flows
                    for record in ltm_cash_flows:
                        # Convert database record to standard format
                        record_dict = dict(record)
                        record_dict['data_source'] = 'LTM'
                        record_dict['statement_type'] = 'cash_flow'
                        record_dict['period_type'] = 'LTM'
                        record_dict['period_end_date'] = record_dict.get('period_end_date')  # Use standardized field name
                        record_dict['ticker'] = ticker
                        ltm_records.append(record_dict)
                    
                    logger.debug(f"Retrieved {len(ltm_records)} cached LTM records for {ticker}")
                    return ltm_records
            
            # Data is stale or no LTM data exists, calculate fresh LTM data
            logger.info(f"Company data for {ticker} is stale or no LTM data exists, calculating fresh LTM data")
            return _get_all_ltm_data_calculated(ticker)
        
    except Exception as e:
        logger.error(f"Error getting LTM data from database for {ticker}: {e}")
        # Fallback to calculation on database error
        return _get_all_ltm_data_calculated(ticker)
    
    return ltm_records


def _get_all_ltm_data_calculated(ticker: str) -> List[Dict[str, Any]]:
    """Get all LTM data by calculation (fallback method)"""
    ltm_records = []
    
    try:
        with LTMCalculator() as ltm_calc:
            # Get LTM income statement data
            income_ltm = ltm_calc.calculate_ltm_for_all_quarters(ticker, 'income_statement')
            for record in income_ltm:
                record['data_source'] = 'LTM'
                record['statement_type'] = 'income_statement'
                record['period_type'] = 'LTM'
                record['ticker'] = ticker
                ltm_records.append(record)
            
            # Get LTM cash flow data  
            cashflow_ltm = ltm_calc.calculate_ltm_for_all_quarters(ticker, 'cash_flow')
            for record in cashflow_ltm:
                record['data_source'] = 'LTM'
                record['statement_type'] = 'cash_flow'
                record['period_type'] = 'LTM' 
                record['ticker'] = ticker
                ltm_records.append(record)
                
        logger.debug(f"Calculated {len(ltm_records)} LTM records for {ticker}")
        
    except Exception as e:
        logger.error(f"Error calculating LTM data for {ticker}: {e}")
    
    return ltm_records


def _get_balance_sheet_data(ticker: str, db: FinancialDatabase) -> List[Dict[str, Any]]:
    """Get all balance sheet data (point-in-time)"""
    balance_records = []
    
    try:
        balance_sheets = db.get_balance_sheets(ticker)
        
        for bs in balance_sheets:
            # Convert to dictionary and add metadata
            bs_dict = dict(bs)
            bs_dict['data_source'] = 'point_in_time'
            bs_dict['statement_type'] = 'balance_sheet'
            bs_dict['ticker'] = ticker
            balance_records.append(bs_dict)
            
        logger.debug(f"Retrieved {len(balance_records)} balance sheet records for {ticker}")
        
    except Exception as e:
        logger.error(f"Error getting balance sheet data for {ticker}: {e}")
    
    return balance_records


def _get_ratio_data(ticker: str, db: FinancialDatabase) -> List[Dict[str, Any]]:
    """Get all calculated ratios (24-hour cache aware)"""
    ratio_records = []
    
    try:
        # Check if company data is fresh (within 24 hours) AND ratios exist
        if db.is_company_data_fresh(ticker, hours=24):
            existing_ratios = db.get_calculated_ratios(ticker)
            if existing_ratios:
                logger.info(f"Using cached ratio data for {ticker} (fresh within 24 hours)")
                # Use existing cached data
                for ratio in existing_ratios:
                    # Convert to dictionary and add metadata
                    ratio_dict = dict(ratio)
                    ratio_dict['data_source'] = 'calculated'
                    ratio_dict['statement_type'] = 'ratio'
                    ratio_dict['ticker'] = ticker
                    
                    # Parse calculation inputs if JSON string
                    if isinstance(ratio_dict.get('calculation_inputs'), str):
                        try:
                            import json
                            ratio_dict['calculation_inputs'] = json.loads(ratio_dict['calculation_inputs'])
                        except:
                            pass
                    
                    ratio_records.append(ratio_dict)
                
                logger.debug(f"Retrieved {len(ratio_records)} cached ratio records for {ticker}")
                return ratio_records
        
        # Data is stale or no ratios exist, calculate fresh ratios
        logger.info(f"Company data for {ticker} is stale or no ratios exist, calculating fresh ratios")
        with SimpleRatioCalculator() as calc:
            ratio_result = calc.calculate_company_ratios(ticker)
            
            if ratio_result and not ratio_result.get('error'):
                # Get newly calculated ratios from database
                ratios = db.get_calculated_ratios(ticker)
                
                for ratio in ratios:
                    # Convert to dictionary and add metadata
                    ratio_dict = dict(ratio)
                    ratio_dict['data_source'] = 'calculated'
                    ratio_dict['statement_type'] = 'ratio'
                    ratio_dict['ticker'] = ticker
                    
                    # Parse calculation inputs if JSON string
                    if isinstance(ratio_dict.get('calculation_inputs'), str):
                        try:
                            import json
                            ratio_dict['calculation_inputs'] = json.loads(ratio_dict['calculation_inputs'])
                        except:
                            pass
                    
                    ratio_records.append(ratio_dict)
        
        logger.debug(f"Retrieved {len(ratio_records)} ratio records for {ticker}")
        
    except Exception as e:
        logger.error(f"Error getting ratio data for {ticker}: {e}")
    
    return ratio_records


def _structure_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Structure and clean the DataFrame"""
    try:
        # Ensure period_end_date is datetime
        if 'period_end_date' in df.columns:
            df['period_end_date'] = pd.to_datetime(df['period_end_date'])
        elif 'ltm_period_end' in df.columns:
            df['period_end_date'] = pd.to_datetime(df['ltm_period_end'])
            
        # Sort by period end date (most recent first)
        if 'period_end_date' in df.columns:
            df = df.sort_values('period_end_date', ascending=False)
        
        # Add consistent ticker column
        df['ticker'] = ticker
        
        # Reorder columns logically
        priority_columns = [
            'ticker', 'period_end_date', 'period_type', 'data_source', 
            'statement_type', 'fiscal_year'
        ]
        
        existing_priority = [col for col in priority_columns if col in df.columns]
        other_columns = [col for col in df.columns if col not in priority_columns]
        
        # Reorder DataFrame columns
        df = df[existing_priority + sorted(other_columns)]
        
        # Remove unnecessary columns
        columns_to_remove = ['id', 'company_id', 'created_at', 'updated_at', 'calculation_date']
        df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
        
        logger.debug(f"Structured DataFrame for {ticker}: {len(df)} rows, {len(df.columns)} columns")
        
        return df
        
    except Exception as e:
        logger.error(f"Error structuring DataFrame for {ticker}: {e}")
        return df


def get_company_ratios_only(ticker: str) -> Optional[pd.DataFrame]:
    """
    Get only calculated ratios for a company as DataFrame
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        DataFrame with only ratio data
    """
    full_df = get_company_financial_data(ticker, include_ratios=True)
    
    if full_df is None:
        return None
    
    # Filter to only ratio data
    ratio_df = full_df[full_df['statement_type'] == 'ratio'].copy()
    
    if ratio_df.empty:
        logger.warning(f"No ratio data found for {ticker}")
        return None
    
    return ratio_df


def get_company_ltm_only(ticker: str) -> Optional[pd.DataFrame]:
    """
    Get only LTM data for a company as DataFrame
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        DataFrame with only LTM data
    """
    full_df = get_company_financial_data(ticker, include_ratios=False)
    
    if full_df is None:
        return None
    
    # Filter to only LTM data
    ltm_df = full_df[full_df['data_source'] == 'LTM'].copy()
    
    if ltm_df.empty:
        logger.warning(f"No LTM data found for {ticker}")
        return None
    
    return ltm_df


# Standalone package interface - main entry point
def get_financial_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Main standalone interface - get all financial data for a company
    
    This is the primary function for external packages to use.
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        
    Returns:
        pandas DataFrame with comprehensive financial data:
        - LTM income statement and cash flow data
        - Point-in-time balance sheet data  
        - Calculated financial ratios
        - All data indexed by period_end_date
        
    Example:
        >>> import SECFinancialRAG as sfr
        >>> df = sfr.get_financial_data('AAPL')
        >>> print(df.head())
    """
    return get_company_financial_data(ticker, auto_process=True, include_ratios=True)


# Convenience function for multiple companies
def get_multiple_companies_data(tickers: List[str]) -> Optional[pd.DataFrame]:
    """
    Get financial data for multiple companies as single DataFrame
    
    Args:
        tickers: List of company ticker symbols
        
    Returns:
        Combined DataFrame with data for all companies
    """
    all_dfs = []
    
    for ticker in tickers:
        df = get_financial_data(ticker)
        if df is not None:
            all_dfs.append(df)
        else:
            logger.warning(f"No data retrieved for {ticker}")
    
    if not all_dfs:
        return None
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by ticker and period_end_date
    combined_df = combined_df.sort_values(['ticker', 'period_end_date'], ascending=[True, False])
    
    logger.info(f"Retrieved data for {len(all_dfs)} companies: {combined_df['ticker'].unique()}")
    
    return combined_df