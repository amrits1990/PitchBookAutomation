"""
EBITDA Post-Processor
Calculates and populates EBITDA values after income statement and cash flow processing
but before LTM calculations, using data already stored in the database.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
import uuid

try:
    from .database import FinancialDatabase
except ImportError:
    from database import FinancialDatabase

logger = logging.getLogger(__name__)


class EBITDAPostProcessor:
    """
    Post-processes EBITDA calculations using database-stored values
    
    This handles cases where:
    1. EBITDA couldn't be computed from raw GAAP fields during initial processing
    2. We have operating_income (from income_statements) and depreciation_and_amortization (from cash_flow_statements)
    3. We want to populate EBITDA before LTM calculations run
    """
    
    def __init__(self):
        self.database = FinancialDatabase()
    
    def calculate_ebitda_for_company(self, ticker: str, force_recalculate: bool = False) -> Dict[str, int]:
        """
        Calculate EBITDA for all periods of a company using database values
        
        Args:
            ticker: Company ticker symbol
            force_recalculate: If True, recalculate even if EBITDA already exists
            
        Returns:
            Dictionary with statistics: {
                'periods_processed': int,
                'ebitda_calculated': int,
                'ebitda_skipped': int,
                'errors': int
            }
        """
        logger.info(f"Starting EBITDA post-processing for {ticker}")
        
        stats = {
            'periods_processed': 0,
            'ebitda_calculated': 0,
            'ebitda_skipped': 0,
            'errors': 0
        }
        
        try:
            # Get all income statement records for the company
            income_statements = self.database.get_income_statements(ticker)
            
            if not income_statements:
                logger.warning(f"No income statements found for {ticker}")
                return stats
            
            logger.info(f"Found {len(income_statements)} income statement periods for {ticker}")
            
            for income_record in income_statements:
                stats['periods_processed'] += 1
                
                try:
                    result = self._calculate_ebitda_for_period(
                        ticker, 
                        income_record,
                        force_recalculate
                    )
                    
                    if result == 'calculated':
                        stats['ebitda_calculated'] += 1
                    elif result == 'skipped':
                        stats['ebitda_skipped'] += 1
                    else:
                        stats['errors'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing period {income_record['period_end_date']} for {ticker}: {e}")
                    stats['errors'] += 1
            
            logger.info(f"EBITDA post-processing completed for {ticker}: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in EBITDA post-processing for {ticker}: {e}")
            stats['errors'] += 1
            return stats
    
    def _calculate_ebitda_for_period(self, ticker: str, income_record: Dict[str, Any], 
                                   force_recalculate: bool = False) -> str:
        """
        Calculate EBITDA for a specific period
        
        Args:
            ticker: Company ticker symbol
            income_record: Income statement record from database
            force_recalculate: If True, recalculate even if EBITDA already exists
            
        Returns:
            'calculated', 'skipped', or 'error'
        """
        period_end_date = income_record['period_end_date']
        period_type = income_record['period_type']
        period_length = income_record['period_length_months']
        
        # Check if EBITDA already exists and we're not forcing recalculation
        existing_ebitda = income_record.get('ebitda')
        if existing_ebitda is not None and not force_recalculate:
            logger.debug(f"EBITDA already exists for {ticker} {period_end_date}, skipping")
            return 'skipped'
        
        # Get operating income from the current record
        operating_income = income_record.get('operating_income')
        if operating_income is None:
            logger.debug(f"No operating income for {ticker} {period_end_date}, cannot calculate EBITDA")
            return 'error'
        
        # Find matching cash flow statement for depreciation data
        depreciation_value = self._get_matching_depreciation(
            ticker, period_end_date, period_type, period_length
        )
        
        if depreciation_value is None:
            logger.debug(f"No depreciation data for {ticker} {period_end_date}, cannot calculate EBITDA")
            return 'error'
        
        # Calculate EBITDA = Operating Income + Depreciation & Amortization
        try:
            ebitda_value = Decimal(str(operating_income)) + Decimal(str(depreciation_value))
            
            # Update the income statement record with calculated EBITDA
            success = self._update_ebitda_in_database(income_record['id'], ebitda_value)
            
            if success:
                logger.info(f"Calculated EBITDA for {ticker} {period_end_date}: {operating_income:,.0f} + {depreciation_value:,.0f} = {ebitda_value:,.0f}")
                return 'calculated'
            else:
                logger.error(f"Failed to update EBITDA in database for {ticker} {period_end_date}")
                return 'error'
                
        except Exception as e:
            logger.error(f"Error calculating EBITDA for {ticker} {period_end_date}: {e}")
            return 'error'
    
    def _get_matching_depreciation(self, ticker: str, period_end_date, period_type: str, 
                                 period_length: int) -> Optional[Decimal]:
        """
        Get depreciation & amortization value from matching cash flow statement
        
        Args:
            ticker: Company ticker symbol
            period_end_date: Period end date to match
            period_type: Period type (Q1, Q2, Q3, Q4, FY)
            period_length: Period length in months
            
        Returns:
            Depreciation & amortization value or None if not found
        """
        try:
            # Get cash flow statements for the company
            cash_flow_statements = self.database.get_cash_flow_statements(ticker)
            
            # Find exact match by period_end_date, period_type, and period_length
            for cf_record in cash_flow_statements:
                if (cf_record['period_end_date'] == period_end_date and
                    cf_record['period_type'] == period_type and
                    cf_record['period_length_months'] == period_length):
                    
                    # Try different depreciation fields in order of preference
                    depreciation_fields = [
                        'depreciation_and_amortization',
                        'depreciation',
                        'amortization'
                    ]
                    
                    total_depreciation = Decimal('0')
                    found_any = False
                    
                    for field in depreciation_fields:
                        value = cf_record.get(field)
                        if value is not None:
                            total_depreciation += Decimal(str(value))
                            found_any = True
                            if field == 'depreciation_and_amortization':
                                # If we have the combined field, use it directly
                                return total_depreciation
                    
                    if found_any:
                        return total_depreciation
            
            logger.debug(f"No matching cash flow statement found for {ticker} {period_end_date} {period_type}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting depreciation for {ticker} {period_end_date}: {e}")
            return None
    
    def _update_ebitda_in_database(self, income_statement_id: str, ebitda_value: Decimal) -> bool:
        """
        Update EBITDA value in the income_statements table
        
        Args:
            income_statement_id: UUID of the income statement record
            ebitda_value: Calculated EBITDA value
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            with self.database.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE income_statements 
                    SET ebitda = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING id
                """, (float(ebitda_value), income_statement_id))
                
                result = cursor.fetchone()
                if result:
                    self.database.connection.commit()
                    return True
                else:
                    logger.error(f"No income statement found with id {income_statement_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating EBITDA in database: {e}")
            self.database.connection.rollback()
            return False
    
    def calculate_ebitda_for_all_companies(self, force_recalculate: bool = False) -> Dict[str, Any]:
        """
        Calculate EBITDA for all companies in the database
        
        Args:
            force_recalculate: If True, recalculate even if EBITDA already exists
            
        Returns:
            Dictionary with overall statistics
        """
        logger.info("Starting EBITDA post-processing for all companies")
        
        overall_stats = {
            'companies_processed': 0,
            'total_periods': 0,
            'total_calculated': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'company_results': {}
        }
        
        try:
            # Get all companies from database
            with self.database.connection.cursor() as cursor:
                cursor.execute("SELECT DISTINCT ticker FROM companies ORDER BY ticker")
                companies = cursor.fetchall()
            
            logger.info(f"Found {len(companies)} companies to process")
            
            for (ticker,) in companies:
                overall_stats['companies_processed'] += 1
                
                company_stats = self.calculate_ebitda_for_company(ticker, force_recalculate)
                overall_stats['company_results'][ticker] = company_stats
                
                # Aggregate stats
                overall_stats['total_periods'] += company_stats['periods_processed']
                overall_stats['total_calculated'] += company_stats['ebitda_calculated']
                overall_stats['total_skipped'] += company_stats['ebitda_skipped']
                overall_stats['total_errors'] += company_stats['errors']
            
            logger.info(f"EBITDA post-processing completed for all companies: {overall_stats}")
            return overall_stats
            
        except Exception as e:
            logger.error(f"Error in EBITDA post-processing for all companies: {e}")
            return overall_stats


def run_ebitda_post_processing(ticker: str = None, force_recalculate: bool = False) -> bool:
    """
    Standalone function to run EBITDA post-processing
    
    Args:
        ticker: Specific ticker to process (None for all companies)
        force_recalculate: If True, recalculate even if EBITDA already exists
        
    Returns:
        True if successful, False otherwise
    """
    try:
        processor = EBITDAPostProcessor()
        
        if ticker:
            stats = processor.calculate_ebitda_for_company(ticker, force_recalculate)
            success = stats['errors'] == 0
        else:
            stats = processor.calculate_ebitda_for_all_companies(force_recalculate)
            success = stats['total_errors'] == 0
        
        return success
        
    except Exception as e:
        logger.error(f"Error in EBITDA post-processing: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    # Simple command line interface
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    force = '--force' in sys.argv
    
    if ticker:
        logger.info(f"Running EBITDA post-processing for {ticker}")
    else:
        logger.info("Running EBITDA post-processing for all companies")
    
    success = run_ebitda_post_processing(ticker, force)
    
    if success:
        print("✅ EBITDA post-processing completed successfully!")
    else:
        print("❌ EBITDA post-processing encountered errors")
    
    sys.exit(0 if success else 1)