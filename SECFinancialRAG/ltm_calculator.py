"""
Last Twelve Months (LTM) Calculator for Financial Statements
Calculates trailing twelve-month values for income statement and cash flow data
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from decimal import Decimal
import pandas as pd

try:
    from .database import FinancialDatabase
    from .models import IncomeStatement, CashFlowStatement
except ImportError:
    from database import FinancialDatabase
    from models import IncomeStatement, CashFlowStatement

logger = logging.getLogger(__name__)


class LTMCalculator:
    """
    Calculates Last Twelve Months (LTM) financial data
    
    Formula for LTM calculation:
    - For 9-month period: LTM = FY(previous year) + 9M(current) - 9M(previous year same period)
    - For 6-month period: LTM = FY(previous year) + 6M(current) - 6M(previous year same period)  
    - For 3-month period: LTM = FY(previous year) + 3M(current) - 3M(previous year same period)
    """
    
    def __init__(self):
        self.database = FinancialDatabase()
        
        # Define which fields are additive for LTM calculations
        self.income_statement_fields = {
            'total_revenue', 'cost_of_revenue', 'gross_profit',
            'research_and_development', 'sales_and_marketing', 'sales_general_and_admin', 'general_and_administrative',
            'total_operating_expenses', 'operating_income',
            'interest_income', 'interest_expense', 'other_income', 
            'income_before_taxes', 'income_tax_expense', 'net_income',
            'earnings_per_share_basic', 'earnings_per_share_diluted'
        }
        
        self.cash_flow_fields = {
            'net_cash_from_operating_activities', 'depreciation_and_amortization', 'depreciation', 'amortization',
            'stock_based_compensation', 'changes_in_accounts_receivable', 'changes_in_other_receivable',
            'changes_in_inventory', 'changes_in_accounts_payable', 'changes_in_other_operating_assets',
            'changes_in_other_operating_liabilities', 'net_cash_from_investing_activities', 'capital_expenditures',
            'acquisitions', 'purchases_of_intangible_assets', 'investments_purchased', 'investments_sold', 'divestitures',
            'net_cash_from_financing_activities', 'dividends_paid',
            'share_repurchases', 'proceeds_from_stock_issuance', 'debt_issued', 'debt_repaid', 'net_change_in_cash'
        }
    
    def calculate_ltm_for_period(self, ticker: str, period_end_date: date, 
                                statement_type: str = 'income_statement') -> Optional[Dict[str, Any]]:
        """
        Calculate LTM values for a specific period
        
        Args:
            ticker: Company ticker symbol
            period_end_date: The period end date to calculate LTM for
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            Dictionary with LTM values or None if calculation not possible
        """
        try:
            logger.info(f"Calculating LTM for {ticker} {statement_type} ending {period_end_date}")
            
            # First check if FY data already exists for this period_end_date
            # If it does, skip this period to avoid duplicates (let existing FY logic handle it)
            existing_fy_data = self._get_fy_data_for_period_end_date(ticker, period_end_date, statement_type)
            if existing_fy_data:
                logger.info(f"Skipping {ticker} {period_end_date} - FY data already exists, will be handled by existing FY logic")
                return None
            
            # Get the specific period data
            current_period = self._get_period_data(ticker, period_end_date, statement_type)
            if not current_period:
                logger.warning(f"No data found for {ticker} {period_end_date}")
                return None
            
            period_length = current_period.get('period_length_months', 3)
            period_type = current_period.get('period_type', 'Q4')
            fiscal_year = current_period.get('fiscal_year')
            
            if period_length == 12:
                logger.info(f"Period is already full year, skipping - will be handled by existing FY logic")
                return None
            
            # Special handling for Q2 and Q3 periods that might need quarterly sum approach
            if period_type in ['Q2', 'Q3']:
                return self._calculate_ltm_for_quarterly_period(ticker, current_period, statement_type)
            
            # Standard calculation for Q1 and Q4
            return self._calculate_ltm_standard(ticker, current_period, statement_type)
            
        except Exception as e:
            logger.error(f"Error calculating LTM for {ticker} {period_end_date}: {e}")
            return None
    
    def _calculate_ltm_for_quarterly_period(self, ticker: str, current_period: Dict[str, Any], 
                                          statement_type: str) -> Optional[Dict[str, Any]]:
        """
        Handle Q2 and Q3 LTM calculation with both standard and quarterly sum approaches
        
        For Q2:
        - If 6M data available for current and previous year: use standard logic
        - If only 3M data available: use FY + Q1 + Q2 - Prev(Q1 + Q2)
        
        For Q3:
        - If 9M data available for current and previous year: use standard logic
        - If only 3M data available: use FY + Q1 + Q2 + Q3 - Prev(Q1 + Q2 + Q3)
        """
        try:
            period_length = current_period.get('period_length_months', 3)
            period_type = current_period.get('period_type')
            fiscal_year = current_period.get('fiscal_year')
            period_end_date = current_period.get('period_end_date')
            
            if isinstance(period_end_date, str):
                period_end_date = datetime.strptime(period_end_date, '%Y-%m-%d').date()
            elif isinstance(period_end_date, datetime):
                period_end_date = period_end_date.date()
            
            logger.info(f"{period_type} LTM calculation: period_length={period_length}M")
            
            # Determine expected period length for standard approach
            expected_length = 6 if period_type == 'Q2' else 9  # Q2 = 6M, Q3 = 9M
            
            # Check if we have expected period length data for current and previous year
            if period_length == expected_length:
                previous_same_period = self._get_same_period_previous_year(
                    ticker, period_end_date, expected_length, statement_type
                )
                if previous_same_period:
                    logger.info(f"Using {expected_length}M {period_type} data with standard LTM calculation")
                    return self._calculate_ltm_standard(ticker, current_period, statement_type)
            
            # If we reach here, we need to use 3M quarterly data approach
            # But first validate that current period is actually 3M
            if period_length != 3:
                logger.warning(f"Cannot use quarterly sum approach with {period_length}M data for {period_type}")
                return None
            
            logger.info(f"Using 3M quarterly data for {period_type} LTM calculation")
            
            # Get previous full year data (FY)
            previous_fy = self._get_full_year_data(ticker, fiscal_year - 1, statement_type)
            if not previous_fy:
                logger.warning(f"No previous full year data found for {ticker} FY{fiscal_year - 1}")
                return None
            
            # Collect required quarters based on period type
            if period_type == 'Q2':
                # For Q2: need Q1, Q2 (current and previous year)
                current_quarters = [
                    self._get_quarterly_period_3m_only(ticker, fiscal_year, 'Q1', statement_type),
                    current_period  # This is Q2
                ]
                previous_quarters = [
                    self._get_quarterly_period_3m_only(ticker, fiscal_year - 1, 'Q1', statement_type),
                    self._get_quarterly_period_3m_only(ticker, fiscal_year - 1, 'Q2', statement_type)
                ]
                required_period_names = ['Current Q1', 'Current Q2', 'Previous Q1', 'Previous Q2']
                
            elif period_type == 'Q3':
                # For Q3: need Q1, Q2, Q3 (current and previous year)
                current_quarters = [
                    self._get_quarterly_period_3m_only(ticker, fiscal_year, 'Q1', statement_type),
                    self._get_quarterly_period_3m_only(ticker, fiscal_year, 'Q2', statement_type),
                    current_period  # This is Q3
                ]
                previous_quarters = [
                    self._get_quarterly_period_3m_only(ticker, fiscal_year - 1, 'Q1', statement_type),
                    self._get_quarterly_period_3m_only(ticker, fiscal_year - 1, 'Q2', statement_type),
                    self._get_quarterly_period_3m_only(ticker, fiscal_year - 1, 'Q3', statement_type)
                ]
                required_period_names = ['Current Q1', 'Current Q2', 'Current Q3', 
                                       'Previous Q1', 'Previous Q2', 'Previous Q3']
            
            # Validate all required periods are available and are 3M periods
            all_periods = current_quarters + previous_quarters
            if not all(all_periods):
                missing_periods = []
                for i, period in enumerate(all_periods):
                    if not period:
                        missing_periods.append(required_period_names[i])
                
                logger.warning(f"Missing required 3M quarterly periods for {period_type} LTM: {', '.join(missing_periods)}")
                return None
            
            # Double-check that all periods are actually 3M
            for i, period in enumerate(all_periods):
                if period and period.get('period_length_months') != 3:
                    logger.warning(f"Period {required_period_names[i]} has {period.get('period_length_months')}M data, expected 3M")
                    return None
            
            # Calculate LTM using quarterly sum approach
            ltm_values = self._calculate_ltm_quarterly_sum(
                previous_fy, current_quarters, previous_quarters, statement_type
            )
            
            ltm_record = self._create_ltm_record(current_period, ltm_values)
            
            logger.info(f"Successfully calculated {period_type} LTM using quarterly sum method")
            return ltm_record
            
        except Exception as e:
            logger.error(f"Error in {current_period.get('period_type')} LTM calculation: {e}")
            return None
    
    def _calculate_ltm_standard(self, ticker: str, current_period: Dict[str, Any], 
                               statement_type: str) -> Optional[Dict[str, Any]]:
        """
        Standard LTM calculation: FY(prev year) + Current Period - Same Period(prev year)
        """
        try:
            fiscal_year = current_period.get('fiscal_year')
            period_length = current_period.get('period_length_months', 3)
            period_end_date = current_period.get('period_end_date')
            
            if isinstance(period_end_date, str):
                period_end_date = datetime.strptime(period_end_date, '%Y-%m-%d').date()
            elif isinstance(period_end_date, datetime):
                period_end_date = period_end_date.date()
            
            # Get previous full year data (FY)
            previous_fy = self._get_full_year_data(ticker, fiscal_year - 1, statement_type)
            if not previous_fy:
                logger.warning(f"No previous full year data found for {ticker} FY{fiscal_year - 1}")
                return None
            
            # Get same period from previous year for subtraction
            previous_same_period = self._get_same_period_previous_year(
                ticker, period_end_date, period_length, statement_type
            )
            if not previous_same_period:
                logger.warning(f"No same period data from previous year found for {ticker}")
                return None
            
            # Calculate LTM: FY(prev year) + Current Period - Same Period(prev year)
            ltm_values = self._calculate_ltm_values_standard(
                previous_fy, current_period, previous_same_period, statement_type
            )
            
            ltm_record = self._create_ltm_record(current_period, ltm_values)
            
            logger.info(f"Successfully calculated standard LTM")
            return ltm_record
            
        except Exception as e:
            logger.error(f"Error in standard LTM calculation: {e}")
            return None
    
    def calculate_ltm_for_all_quarters(self, ticker: str, 
                                      statement_type: str = 'income_statement') -> List[Dict[str, Any]]:
        """
        Calculate LTM values for all quarterly periods of a company with smart deduplication
        For periods with same period_end_date but different period lengths, keeps the LTM 
        record with more non-null values.
        
        Args:
            ticker: Company ticker symbol
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            List of LTM records (deduplicated by period_end_date based on data quality)
        """
        try:
            logger.info(f"Calculating LTM for all {ticker} {statement_type} quarters")
            
            # Get all quarterly periods (exclude full year periods)
            if statement_type == 'income_statement':
                periods = self.database.get_income_statements(ticker)
            else:
                periods = self.database.get_cash_flow_statements(ticker)
            
            # Filter to quarterly periods only - DON'T deduplicate yet
            quarterly_periods = [
                p for p in periods 
                if p.get('period_type') in ['Q1', 'Q2', 'Q3', 'Q4'] and 
                   p.get('period_length_months', 12) < 12
            ]
            
            logger.info(f"Found {len(quarterly_periods)} quarterly periods (before deduplication)")
            
            # Track LTM records by period_end_date for smart deduplication
            ltm_records_by_date = {}
            
            for period in quarterly_periods:
                period_end_date = period.get('period_end_date')
                if isinstance(period_end_date, str):
                    period_end_date = datetime.strptime(period_end_date, '%Y-%m-%d').date()
                
                date_str = str(period_end_date)
                period_length = period.get('period_length_months', 3)
                
                logger.debug(f"Processing {ticker} {date_str} with {period_length}M period length")
                
                # Calculate LTM for this period
                ltm_record = self.calculate_ltm_for_period(ticker, period_end_date, statement_type)
                
                if ltm_record:
                    # Check if we already have an LTM record for this period_end_date
                    if date_str in ltm_records_by_date:
                        existing_record = ltm_records_by_date[date_str]
                        
                        # Compare data quality (non-null values)
                        existing_non_null_count = self._count_non_null_values(existing_record)
                        new_non_null_count = self._count_non_null_values(ltm_record)
                        
                        logger.debug(f"Duplicate period_end_date {date_str}: "
                                   f"existing={existing_non_null_count} non-nulls, "
                                   f"new={new_non_null_count} non-nulls")
                        
                        # Keep the record with more non-null values
                        if new_non_null_count > existing_non_null_count:
                            logger.info(f"Replacing LTM record for {date_str} (better data quality: "
                                      f"{new_non_null_count} vs {existing_non_null_count} non-null values)")
                            ltm_records_by_date[date_str] = ltm_record
                        else:
                            logger.debug(f"Keeping existing LTM record for {date_str} (better or equal data quality)")
                    else:
                        # First LTM record for this period_end_date
                        ltm_records_by_date[date_str] = ltm_record
            
            # Convert to list
            ltm_records = list(ltm_records_by_date.values())
            
            logger.info(f"Calculated {len(ltm_records)} unique LTM records for {ticker} {statement_type} "
                       f"(after smart deduplication)")
            return ltm_records
            
        except Exception as e:
            logger.error(f"Error calculating LTM for all {ticker} quarters: {e}")
            return []
    
    def _get_period_data(self, ticker: str, period_end_date: date, 
                        statement_type: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific period"""
        try:
            if statement_type == 'income_statement':
                periods = self.database.get_income_statements(ticker)
            else:
                periods = self.database.get_cash_flow_statements(ticker)
            
            for period in periods:
                p_date = period.get('period_end_date')
                if isinstance(p_date, str):
                    p_date = datetime.strptime(p_date, '%Y-%m-%d').date()
                elif isinstance(p_date, datetime):
                    p_date = p_date.date()
                
                if p_date == period_end_date:
                    return period
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting period data: {e}")
            return None
    
    def _get_fy_data_for_period_end_date(self, ticker: str, period_end_date: date, 
                                        statement_type: str) -> Optional[Dict[str, Any]]:
        """
        Check if FY data already exists for a specific period_end_date
        
        Args:
            ticker: Company ticker symbol
            period_end_date: The period end date to check for
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            FY period data if found, None otherwise
        """
        try:
            if statement_type == 'income_statement':
                periods = self.database.get_income_statements(ticker)
            else:
                periods = self.database.get_cash_flow_statements(ticker)
            
            for period in periods:
                p_date = period.get('period_end_date')
                if isinstance(p_date, str):
                    p_date = datetime.strptime(p_date, '%Y-%m-%d').date()
                elif isinstance(p_date, datetime):
                    p_date = p_date.date()
                
                # Check if it's the same period_end_date and is FY data (12 months)
                if (p_date == period_end_date and 
                    (period.get('period_type') == 'FY' or 
                     period.get('period_length_months') == 12)):
                    return period
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting FY data for period end date: {e}")
            return None
    
    def _get_full_year_data(self, ticker: str, fiscal_year: int, 
                           statement_type: str) -> Optional[Dict[str, Any]]:
        """Get full year data for a specific fiscal year"""
        try:
            if statement_type == 'income_statement':
                periods = self.database.get_income_statements(ticker)
            else:
                periods = self.database.get_cash_flow_statements(ticker)
            
            # Look for full year period (period_type = 'FY' or period_length_months = 12)
            for period in periods:
                if (period.get('fiscal_year') == fiscal_year and 
                    (period.get('period_type') == 'FY' or 
                     period.get('period_length_months') == 12)):
                    return period
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting full year data: {e}")
            return None
    
    def _get_same_period_previous_year(self, ticker: str, current_end_date: date, 
                                      period_length: int, statement_type: str) -> Optional[Dict[str, Any]]:
        """Get the same period from the previous year"""
        try:
            if statement_type == 'income_statement':
                periods = self.database.get_income_statements(ticker)
            else:
                periods = self.database.get_cash_flow_statements(ticker)
            
            # Look for same period length from previous year
            target_year = current_end_date.year - 1
            
            for period in periods:
                p_date = period.get('period_end_date')
                if isinstance(p_date, str):
                    p_date = datetime.strptime(p_date, '%Y-%m-%d').date()
                elif isinstance(p_date, datetime):
                    p_date = p_date.date()
                
                # Check if it's from the target year and has same period length
                if (p_date.year == target_year and 
                    period.get('period_length_months') == period_length):
                    
                    # For more precision, check if it's roughly the same period
                    # (within 2 months of the same month)
                    if abs(p_date.month - current_end_date.month) <= 2:
                        return period
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting same period previous year: {e}")
            return None
    
    def _get_quarterly_period_3m_only(self, ticker: str, fiscal_year: int, quarter: str, 
                                     statement_type: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific quarterly period, ensuring it's 3-month data only
        
        Args:
            ticker: Company ticker
            fiscal_year: Fiscal year
            quarter: Quarter ('Q1', 'Q2', 'Q3', 'Q4')
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            Period data if found and is 3-month period, None otherwise
        """
        try:
            if statement_type == 'income_statement':
                periods = self.database.get_income_statements(ticker)
            else:
                periods = self.database.get_cash_flow_statements(ticker)
            
            # Look for the specific quarter with EXACTLY 3-month period length
            for period in periods:
                if (period.get('fiscal_year') == fiscal_year and 
                    period.get('period_type') == quarter and
                    period.get('period_length_months') == 3):  # Must be exactly 3 months
                    return period
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting 3M quarterly period {quarter} FY{fiscal_year}: {e}")
            return None
    
    def _calculate_ltm_values_standard(self, previous_fy: Dict[str, Any], current_period: Dict[str, Any],
                             previous_same_period: Dict[str, Any], statement_type: str) -> Dict[str, Any]:
        """
        Calculate LTM values using: FY(prev year) + Current Period - Same Period(prev year)
        """
        ltm_values = {}
        
        # Determine which fields to calculate based on statement type
        if statement_type == 'income_statement':
            fields_to_calculate = self.income_statement_fields
        else:
            fields_to_calculate = self.cash_flow_fields
        
        for field in fields_to_calculate:
            try:
                fy_value = self._get_decimal_value(previous_fy.get(field))
                current_value = self._get_decimal_value(current_period.get(field))
                previous_value = self._get_decimal_value(previous_same_period.get(field))
                
                # LTM = FY(prev) + Current - Previous Same Period
                if all(v is not None for v in [fy_value, current_value, previous_value]):
                    ltm_value = fy_value + current_value - previous_value
                    ltm_values[field] = ltm_value
                elif fy_value is not None and current_value is not None:
                    # If we don't have previous same period, use just FY + current
                    ltm_value = fy_value + current_value
                    ltm_values[field] = ltm_value
                else:
                    ltm_values[field] = None
                    
            except Exception as e:
                logger.debug(f"Error calculating LTM for field {field}: {e}")
                ltm_values[field] = None
        
        return ltm_values
    
    def _calculate_ltm_quarterly_sum(self, previous_fy: Dict[str, Any], 
                                   current_quarters: List[Dict[str, Any]],
                                   previous_quarters: List[Dict[str, Any]], 
                                   statement_type: str) -> Dict[str, Any]:
        """
        Calculate LTM values using: FY + sum(current quarters) - sum(previous quarters)
        For Q3 with 3M data: FY + Q1 + Q2 + Q3 - Prev(Q1 + Q2 + Q3)
        """
        ltm_values = {}
        
        # Determine which fields to calculate based on statement type
        if statement_type == 'income_statement':
            fields_to_calculate = self.income_statement_fields
        else:
            fields_to_calculate = self.cash_flow_fields
        
        for field in fields_to_calculate:
            try:
                # Get FY value
                fy_value = self._get_decimal_value(previous_fy.get(field))
                
                # Sum current quarters
                current_sum = Decimal('0')
                current_count = 0
                for quarter in current_quarters:
                    quarter_value = self._get_decimal_value(quarter.get(field))
                    if quarter_value is not None:
                        current_sum += quarter_value
                        current_count += 1
                
                # Sum previous quarters
                previous_sum = Decimal('0')
                previous_count = 0
                for quarter in previous_quarters:
                    quarter_value = self._get_decimal_value(quarter.get(field))
                    if quarter_value is not None:
                        previous_sum += quarter_value
                        previous_count += 1
                
                # Calculate LTM: FY + Current Quarters Sum - Previous Quarters Sum
                if fy_value is not None and current_count > 0 and previous_count > 0:
                    ltm_value = fy_value + current_sum - previous_sum
                    ltm_values[field] = ltm_value
                elif fy_value is not None and current_count > 0:
                    # If we don't have all previous quarters, use FY + current sum
                    ltm_value = fy_value + current_sum
                    ltm_values[field] = ltm_value
                else:
                    ltm_values[field] = None
                    
            except Exception as e:
                logger.debug(f"Error calculating quarterly sum LTM for field {field}: {e}")
                ltm_values[field] = None
        
        return ltm_values
    
    def _get_decimal_value(self, value: Any) -> Optional[Decimal]:
        """Convert value to Decimal, handling various input types"""
        if value is None:
            return None
        
        try:
            if isinstance(value, Decimal):
                return value
            elif isinstance(value, (int, float)):
                return Decimal(str(value))
            elif isinstance(value, str):
                return Decimal(value)
            else:
                return None
        except Exception:
            return None
    
    def _count_non_null_values(self, record: Dict[str, Any]) -> int:
        """
        Count the number of non-null financial values in a record
        Excludes metadata fields like dates, IDs, etc.
        
        Args:
            record: Dictionary containing financial data
            
        Returns:
            Count of non-null financial values
        """
        # Financial fields to count (exclude metadata fields)
        financial_fields = (
            # Common fields
            self.income_statement_fields | self.cash_flow_fields
        )
        
        # Additional common financial fields that might not be in the sets
        additional_fields = {
            'total_assets', 'total_liabilities', 'total_stockholders_equity',
            'cash_and_cash_equivalents', 'total_current_assets', 'total_current_liabilities'
        }
        
        all_financial_fields = financial_fields | additional_fields
        
        non_null_count = 0
        for field in all_financial_fields:
            value = record.get(field)
            if value is not None and value != 0:  # Count non-null and non-zero values
                non_null_count += 1
        
        return non_null_count
    
    def _create_ltm_record(self, base_period: Dict[str, Any], ltm_values: Dict[str, Any]) -> Dict[str, Any]:
        """Create an LTM record based on the base period structure"""
        ltm_record = base_period.copy()
        
        # Update with LTM values
        ltm_record.update(ltm_values)
        
        # Mark as calculated LTM data
        ltm_record['period_type'] = 'LTM'
        ltm_record['period_length_months'] = 12
        ltm_record['is_ltm'] = True
        ltm_record['data_source'] = 'Calculated_LTM'
        ltm_record['ltm_calculation_date'] = datetime.utcnow()
        
        return ltm_record
    
    def export_ltm_data(self, ticker: str, output_file: str, 
                       statement_type: str = 'income_statement',
                       include_both_fy_and_ltm: bool = True) -> bool:
        """
        Export LTM data and existing FY data to CSV file with deduplication
        
        Args:
            ticker: Company ticker symbol
            output_file: Output CSV file path
            statement_type: 'income_statement' or 'cash_flow'
            include_both_fy_and_ltm: If True, includes both FY and LTM for year-end dates
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            logger.info(f"Exporting deduplicated LTM and FY data for {ticker} {statement_type} to {output_file}")
            
            # Get calculated LTM records (quarterly data converted to LTM)
            ltm_records = self.calculate_ltm_for_all_quarters(ticker, statement_type)
            
            # Get existing FY data directly from database
            fy_records = self._get_fy_data(ticker, statement_type)
            
            if not ltm_records and not fy_records:
                logger.warning(f"No LTM or FY records to export for {ticker}")
                return False
            
            # Combine all records into one list with deduplication
            all_records = []
            
            # Add calculated LTM data with deduplication
            if ltm_records:
                # Convert to DataFrame for easier deduplication
                ltm_df = pd.DataFrame(ltm_records)
                ltm_df['data_type'] = 'LTM_Calculated'
                
                # Ensure filing_date is available for sorting
                if 'filing_date' not in ltm_df.columns:
                    ltm_df['filing_date'] = pd.NaT
                
                # Sort by filing_date (most recent first) and drop duplicates by period_end_date
                ltm_df = ltm_df.sort_values('filing_date', ascending=False, na_position='last')
                ltm_df = ltm_df.drop_duplicates(subset=['period_end_date'], keep='first')
                
                all_records.extend(ltm_df.to_dict('records'))
                logger.info(f"Added {len(ltm_df)} deduplicated LTM records (from {len(ltm_records)} original)")
            
            # Add existing FY data with deduplication
            if fy_records:
                # Convert to DataFrame for easier deduplication
                fy_df = pd.DataFrame(fy_records) 
                fy_df['data_type'] = 'FY_Original'
                
                # Sort by filing_date (most recent first) and drop duplicates by period_end_date
                fy_df = fy_df.sort_values('filing_date', ascending=False, na_position='last')
                fy_df = fy_df.drop_duplicates(subset=['period_end_date'], keep='first')
                
                # If include_both_fy_and_ltm is False, remove FY records that have LTM equivalents
                if not include_both_fy_and_ltm and ltm_records:
                    ltm_dates = set(str(r.get('period_end_date', '')) for r in all_records 
                                  if r.get('data_type') == 'LTM_Calculated')
                    fy_df['period_end_date_str'] = fy_df['period_end_date'].astype(str)
                    fy_df = fy_df[~fy_df['period_end_date_str'].isin(ltm_dates)]
                    fy_df = fy_df.drop('period_end_date_str', axis=1)
                
                all_records.extend(fy_df.to_dict('records'))
                logger.info(f"Added {len(fy_df)} deduplicated FY records (from {len(fy_records)} original)")
            
            # Final deduplication: for same period_end_date and data_type, keep most recent
            if all_records:
                final_df = pd.DataFrame(all_records)
                
                # Ensure proper sorting and deduplication
                final_df = final_df.sort_values('filing_date', ascending=False, na_position='last')
                final_df = final_df.drop_duplicates(subset=['period_end_date', 'data_type'], keep='first')
                
                # Sort by period_end_date for chronological order (newest first)
                final_df = final_df.sort_values('period_end_date', ascending=False)
                
                # Export to CSV
                final_df.to_csv(output_file, index=False)
                
                logger.info(f"Successfully exported {len(final_df)} deduplicated records to {output_file}")
                return True
            else:
                logger.warning(f"No records to export after deduplication for {ticker}")
                return False
            
        except Exception as e:
            logger.error(f"Error exporting LTM data: {e}")
            return False
    
    def _get_fy_data(self, ticker: str, statement_type: str = 'income_statement') -> List[Dict[str, Any]]:
        """
        Get existing FY (12-month) data directly from database
        
        Args:
            ticker: Company ticker symbol
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            List of FY records as dictionaries
        """
        try:
            if statement_type == 'income_statement':
                # Get all FY (12-month) income statements
                cursor = self.database.connection.cursor()
                cursor.execute("""
                    SELECT * FROM income_statements 
                    WHERE ticker = %s AND period_length_months = 12
                    ORDER BY period_end_date DESC
                """, (ticker.upper(),))
                
                columns = [desc[0] for desc in cursor.description]
                fy_data = []
                
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    # Clean up the record - convert Decimal to float, handle None values
                    cleaned_record = {}
                    for key, value in record.items():
                        if value is None:
                            cleaned_record[key] = None
                        elif isinstance(value, Decimal):
                            cleaned_record[key] = float(value)
                        else:
                            cleaned_record[key] = value
                    fy_data.append(cleaned_record)
                
                return fy_data
                
            elif statement_type == 'cash_flow':
                # Get all FY (12-month) cash flow statements
                cursor = self.database.connection.cursor()
                cursor.execute("""
                    SELECT * FROM cash_flow_statements 
                    WHERE ticker = %s AND period_length_months = 12
                    ORDER BY period_end_date DESC
                """, (ticker.upper(),))
                
                columns = [desc[0] for desc in cursor.description]
                fy_data = []
                
                for row in cursor.fetchall():
                    record = dict(zip(columns, row))
                    # Clean up the record - convert Decimal to float, handle None values
                    cleaned_record = {}
                    for key, value in record.items():
                        if value is None:
                            cleaned_record[key] = None
                        elif isinstance(value, Decimal):
                            cleaned_record[key] = float(value)
                        else:
                            cleaned_record[key] = value
                    fy_data.append(cleaned_record)
                
                return fy_data
                
            else:
                logger.error(f"Unsupported statement type: {statement_type}")
                return []
                    
        except Exception as e:
            logger.error(f"Error getting FY data for {ticker}: {e}")
            return []
    
    def get_latest_ltm_data(self, ticker: str, statement_type: str = 'income_statement') -> Optional[Dict]:
        """
        Get the most recent LTM data for a company
        
        Args:
            ticker: Company ticker symbol
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            Dictionary with latest LTM financial data or None
        """
        try:
            ltm_records = self.calculate_ltm_for_all_quarters(ticker, statement_type)
            
            if not ltm_records:
                logger.warning(f"No LTM records found for {ticker} {statement_type}")
                return None
            
            # Sort by period end date and get the most recent
            # Use 'period_end_date' field, not 'ltm_period_end'
            ltm_records.sort(key=lambda x: x.get('period_end_date', ''), reverse=True)
            latest_record = ltm_records[0]
            
            # Add ltm_period_end field for compatibility
            latest_record['ltm_period_end'] = latest_record.get('period_end_date')
            
            logger.debug(f"Retrieved latest LTM data for {ticker} {statement_type}: {latest_record.get('period_end_date')}")
            return latest_record
            
        except Exception as e:
            logger.error(f"Error getting latest LTM data for {ticker}: {e}")
            return None
    
    def get_ltm_summary(self, ticker: str, statement_type: str = 'income_statement') -> Dict[str, Any]:
        """
        Get summary of LTM calculations for a company
        
        Args:
            ticker: Company ticker symbol
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            Summary dictionary with LTM statistics
        """
        try:
            ltm_records = self.calculate_ltm_for_all_quarters(ticker, statement_type)
            
            if not ltm_records:
                return {
                    'ticker': ticker,
                    'statement_type': statement_type,
                    'ltm_periods': 0,
                    'date_range': None,
                    'latest_ltm_revenue': None,
                    'latest_ltm_net_income': None
                }
            
            # Sort by period end date
            ltm_records.sort(key=lambda x: x.get('period_end_date', ''), reverse=True)
            
            latest_record = ltm_records[0]
            earliest_record = ltm_records[-1]
            
            return {
                'ticker': ticker,
                'statement_type': statement_type,
                'ltm_periods': len(ltm_records),
                'date_range': f"{earliest_record.get('period_end_date')} to {latest_record.get('period_end_date')}",
                'latest_ltm_revenue': float(latest_record.get('total_revenue', 0)) if latest_record.get('total_revenue') else None,
                'latest_ltm_net_income': float(latest_record.get('net_income', 0)) if latest_record.get('net_income') else None,
                'latest_period_end': latest_record.get('period_end_date')
            }
            
        except Exception as e:
            logger.error(f"Error getting LTM summary: {e}")
            return {
                'ticker': ticker,
                'statement_type': statement_type,
                'error': str(e)
            }
    
    def close(self):
        """Close database connection"""
        if self.database:
            self.database.close()
    
    def __enter__(self):
        # Establish database connection when entering context
        self.database.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close database connection when exiting context
        self.database.__exit__(exc_type, exc_val, exc_tb)


# Convenience functions for easy access
def calculate_ltm_income_statement(ticker: str, period_end_date: date) -> Optional[Dict[str, Any]]:
    """Calculate LTM for a specific income statement period"""
    with LTMCalculator() as calculator:
        return calculator.calculate_ltm_for_period(ticker, period_end_date, 'income_statement')


def calculate_ltm_cash_flow(ticker: str, period_end_date: date) -> Optional[Dict[str, Any]]:
    """Calculate LTM for a specific cash flow period"""
    with LTMCalculator() as calculator:
        return calculator.calculate_ltm_for_period(ticker, period_end_date, 'cash_flow')


def get_all_ltm_data(ticker: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get all LTM data for both income statement and cash flow"""
    with LTMCalculator() as calculator:
        return {
            'income_statement_ltm': calculator.calculate_ltm_for_all_quarters(ticker, 'income_statement'),
            'cash_flow_ltm': calculator.calculate_ltm_for_all_quarters(ticker, 'cash_flow')
        }
