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
            'total_operating_expenses', 'operating_income', 'ebitda',
            'interest_income', 'interest_expense', 'other_income', 
            'income_before_taxes', 'income_tax_expense', 'net_income',
            'earnings_per_share_basic', 'earnings_per_share_diluted',
            'weighted_average_shares_basic', 'weighted_average_shares_diluted'
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
        Enhanced Q2 and Q3 LTM calculation with data completeness prioritization
        
        Logic:
        1. Find all available period data for the same period_end_date
        2. Evaluate data completeness for each approach (standard vs quarterly sum)
        3. Use the approach with best data completeness
        4. Skip calculation if insufficient data
        """
        try:
            period_type = current_period.get('period_type')
            fiscal_year = current_period.get('fiscal_year')
            period_end_date = current_period.get('period_end_date')
            
            if isinstance(period_end_date, str):
                period_end_date = datetime.strptime(period_end_date, '%Y-%m-%d').date()
            elif isinstance(period_end_date, datetime):
                period_end_date = period_end_date.date()
            
            logger.info(f"Enhanced {period_type} LTM calculation for {ticker} ending {period_end_date}")
            
            # Step 1: Find ALL available period data for this period_end_date
            all_period_data = self._get_all_period_data_for_date(ticker, period_end_date, statement_type)
            
            if not all_period_data:
                logger.warning(f"No period data found for {ticker} {period_end_date}")
                return None
            
            logger.info(f"Found {len(all_period_data)} period records for {period_end_date}")
            for period_data in all_period_data:
                period_length = period_data.get('period_length_months', 0)
                logger.debug(f"  - Period length: {period_length}M")
            
            # Step 2: Determine expected cumulative period length
            expected_length = 6 if period_type == 'Q2' else 9  # Q2 = 6M, Q3 = 9M
            
            # Step 3: Try standard calculation with cumulative data first
            cumulative_period = self._find_best_cumulative_period(all_period_data, expected_length)
            
            if cumulative_period:
                logger.info(f"Found {expected_length}M cumulative data, checking completeness...")
                
                # Check if standard calculation is viable with this data
                standard_viability = self._assess_standard_calculation_viability(
                    ticker, cumulative_period, statement_type, fiscal_year
                )
                
                if standard_viability['viable']:
                    logger.info(f"Using standard calculation with {expected_length}M data")
                    logger.info(f"Data completeness: {standard_viability['completeness_score']:.1f}%")
                    return self._calculate_ltm_standard(ticker, cumulative_period, statement_type)
                else:
                    logger.warning(f"Standard calculation not viable: {standard_viability['reason']}")
                    logger.info(f"Data completeness: {standard_viability['completeness_score']:.1f}%")
            
            # Step 4: Try quarterly sum approach with 3M data
            quarterly_period = self._find_best_quarterly_period(all_period_data)
            
            if quarterly_period:
                logger.info(f"Found 3M quarterly data, checking completeness...")
                
                # Check if quarterly sum calculation is viable
                quarterly_viability = self._assess_quarterly_sum_viability(
                    ticker, quarterly_period, statement_type, fiscal_year, period_type
                )
                
                if quarterly_viability['viable']:
                    logger.info(f"Using quarterly sum calculation with 3M data")
                    logger.info(f"Data completeness: {quarterly_viability['completeness_score']:.1f}%")
                    return self._calculate_ltm_quarterly_sum_enhanced(ticker, quarterly_period, statement_type)
                else:
                    logger.warning(f"Quarterly sum calculation not viable: {quarterly_viability['reason']}")
                    logger.info(f"Data completeness: {quarterly_viability['completeness_score']:.1f}%")
            
            # Step 5: No viable calculation method found
            logger.error(f"No viable LTM calculation method for {ticker} {period_end_date}")
            logger.info(f"Available data insufficient for reliable LTM calculation")
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced quarterly LTM calculation: {e}")
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
            
            # Get same period from previous year for subtraction
            previous_same_period = self._get_same_period_previous_year(
                ticker, period_end_date, period_length, statement_type
            )
            
            # Flexible calculation based on available data
            if previous_fy and previous_same_period:
                # Ideal case: Standard calculation with full data
                logger.info(f"Standard LTM calculation with full data")
                ltm_values = self._calculate_ltm_values_standard(
                    previous_fy, current_period, previous_same_period, statement_type
                )
                calculation_method = "standard_full"
            elif previous_fy and not previous_same_period:
                # Fallback: FY + Current (missing same period)
                logger.info(f"Standard LTM calculation with FY data only (missing same period)")
                ltm_values = self._calculate_ltm_values_fy_plus_current(
                    previous_fy, current_period, statement_type
                )
                calculation_method = "standard_fy_only"
            elif previous_same_period and not previous_fy:
                # Less common: Current - Previous same period (missing FY)
                logger.warning(f"Standard LTM calculation without FY data (using same period only)")
                logger.warning(f"This may not provide accurate LTM values for {ticker} FY{fiscal_year - 1}")
                return None  # Skip this case as it's not reliable for LTM
            else:
                # No viable data for any standard calculation
                logger.warning(f"No previous data available for standard LTM calculation for {ticker} FY{fiscal_year - 1}")
                return None
            
            ltm_record = self._create_ltm_record(current_period, ltm_values)
            
            # Add metadata about calculation method
            ltm_record['calculation_method'] = calculation_method
            ltm_record['calculation_inputs'] = {
                'has_previous_fy': previous_fy is not None,
                'has_previous_same_period': previous_same_period is not None,
                'method_used': calculation_method
            }
            
            logger.info(f"Successfully calculated standard LTM using {calculation_method}")
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
    
    def _get_all_period_data_for_date(self, ticker: str, period_end_date: date, 
                                     statement_type: str) -> List[Dict[str, Any]]:
        """Get all period data (3M, 6M, 9M, 12M) for a specific period_end_date"""
        try:
            if statement_type == 'income_statement':
                all_periods = self.database.get_income_statements(ticker)
            else:
                all_periods = self.database.get_cash_flow_statements(ticker)
            
            matching_periods = []
            for period in all_periods:
                p_date = period.get('period_end_date')
                if isinstance(p_date, str):
                    p_date = datetime.strptime(p_date, '%Y-%m-%d').date()
                elif isinstance(p_date, datetime):
                    p_date = p_date.date()
                
                if p_date == period_end_date:
                    matching_periods.append(period)
            
            return matching_periods
            
        except Exception as e:
            logger.error(f"Error getting all period data for date: {e}")
            return []

    def _find_best_cumulative_period(self, all_periods: List[Dict], expected_length: int) -> Optional[Dict]:
        """Find the best cumulative period (6M for Q2, 9M for Q3)"""
        cumulative_candidates = [
            period for period in all_periods 
            if period.get('period_length_months') == expected_length
        ]
        
        if not cumulative_candidates:
            return None
        
        # If multiple candidates, choose the one with most recent filing_date
        cumulative_candidates.sort(
            key=lambda x: x.get('filing_date', '1900-01-01'), 
            reverse=True
        )
        
        return cumulative_candidates[0]

    def _find_best_quarterly_period(self, all_periods: List[Dict]) -> Optional[Dict]:
        """Find the best 3M quarterly period"""
        quarterly_candidates = [
            period for period in all_periods 
            if period.get('period_length_months') == 3
        ]
        
        if not quarterly_candidates:
            return None
        
        # If multiple candidates, choose the one with most recent filing_date
        quarterly_candidates.sort(
            key=lambda x: x.get('filing_date', '1900-01-01'), 
            reverse=True
        )
        
        return quarterly_candidates[0]

    def _assess_standard_calculation_viability(self, ticker: str, current_period: Dict, 
                                             statement_type: str, fiscal_year: int) -> Dict[str, Any]:
        """
        Assess whether standard calculation is viable based on data completeness
        
        Returns:
            Dictionary with 'viable' boolean and 'completeness_score' percentage
        """
        try:
            # Get required data for standard calculation
            previous_fy = self._get_full_year_data(ticker, fiscal_year - 1, statement_type)
            previous_same_period = self._get_same_period_previous_year(
                ticker, current_period.get('period_end_date'), 
                current_period.get('period_length_months'), statement_type
            )
            
            # Calculate completeness score
            required_fields = self._get_required_fields(statement_type)
            
            current_completeness = self._calculate_field_completeness(current_period, required_fields)
            fy_completeness = self._calculate_field_completeness(previous_fy, required_fields) if previous_fy else 0
            prev_completeness = self._calculate_field_completeness(previous_same_period, required_fields) if previous_same_period else 0
            
            # Overall completeness score with fallback strategies
            if previous_fy and previous_same_period:
                # Ideal case: have both FY and same period data
                completeness_score = (current_completeness + fy_completeness + prev_completeness) / 3
                viable = completeness_score > 50  # At least 50% of fields must be non-null
                reason = "Sufficient data completeness" if viable else f"Low data completeness ({completeness_score:.1f}%)"
            elif previous_fy and not previous_same_period:
                # Have FY but missing same period - still viable but lower threshold
                completeness_score = (current_completeness + fy_completeness) / 2
                viable = completeness_score > 40  # Lower threshold when missing same period
                reason = "Viable with FY data (missing same period)" if viable else f"Low data completeness without same period ({completeness_score:.1f}%)"
            elif previous_same_period and not previous_fy:
                # Have same period but missing FY - viable for newer data patterns
                completeness_score = (current_completeness + prev_completeness) / 2
                viable = completeness_score > 40  # Lower threshold when missing FY
                reason = "Viable with same period data (missing FY)" if viable else f"Low data completeness without FY ({completeness_score:.1f}%)"
            else:
                # Missing both - not viable for standard calculation
                completeness_score = current_completeness
                viable = False
                missing_components = []
                if not previous_fy:
                    missing_components.append("previous FY data")
                if not previous_same_period:
                    missing_components.append("previous same period data")
                reason = f"Missing required components: {', '.join(missing_components)}"
            
            return {
                'viable': viable,
                'completeness_score': completeness_score,
                'reason': reason,
                'components': {
                    'current': current_completeness,
                    'previous_fy': fy_completeness,
                    'previous_same': prev_completeness
                }
            }
            
        except Exception as e:
            return {
                'viable': False,
                'completeness_score': 0,
                'reason': f"Error assessing viability: {e}",
                'components': {}
            }

    def _assess_quarterly_sum_viability(self, ticker: str, current_period: Dict, 
                                      statement_type: str, fiscal_year: int, 
                                      period_type: str) -> Dict[str, Any]:
        """
        Assess whether quarterly sum calculation is viable based on data completeness
        """
        try:
            # Get required data for quarterly sum calculation
            previous_fy = self._get_full_year_data(ticker, fiscal_year - 1, statement_type)
            
            if not previous_fy:
                return {
                    'viable': False,
                    'completeness_score': 0,
                    'reason': "Missing previous FY data",
                    'components': {}
                }
            
            # Determine required quarters
            if period_type == 'Q2':
                required_quarters = ['Q1', 'Q2']
            else:  # Q3
                required_quarters = ['Q1', 'Q2', 'Q3']
            
            # Get current and previous year quarters
            current_quarters = []
            previous_quarters = []
            
            for quarter in required_quarters:
                if quarter == period_type:
                    current_quarters.append(current_period)
                else:
                    current_q = self._get_quarterly_period_3m_only(ticker, fiscal_year, quarter, statement_type)
                    current_quarters.append(current_q)
                
                previous_q = self._get_quarterly_period_3m_only(ticker, fiscal_year - 1, quarter, statement_type)
                previous_quarters.append(previous_q)
            
            # Calculate completeness scores
            required_fields = self._get_required_fields(statement_type)
            
            fy_completeness = self._calculate_field_completeness(previous_fy, required_fields)
            
            current_scores = []
            previous_scores = []
            
            for quarter in current_quarters:
                score = self._calculate_field_completeness(quarter, required_fields) if quarter else 0
                current_scores.append(score)
            
            for quarter in previous_quarters:
                score = self._calculate_field_completeness(quarter, required_fields) if quarter else 0
                previous_scores.append(score)
            
            # Overall assessment
            avg_current = sum(current_scores) / len(current_scores) if current_scores else 0
            avg_previous = sum(previous_scores) / len(previous_scores) if previous_scores else 0
            
            # Check if we have at least some data for each required component
            current_available = sum(1 for q in current_quarters if q is not None)
            previous_available = sum(1 for q in previous_quarters if q is not None)
            
            completeness_score = (fy_completeness + avg_current + avg_previous) / 3
            
            # Viability criteria:
            # 1. Must have FY data
            # 2. Must have at least 50% of required quarters
            # 3. Must have reasonable data completeness
            min_quarters_needed = len(required_quarters) * 0.5  # At least 50% of quarters
            
            viable = (
                fy_completeness > 20 and  # FY must have some data
                current_available >= min_quarters_needed and
                completeness_score > 30  # Lower threshold for quarterly sum
            )
            
            if not viable:
                reasons = []
                if fy_completeness <= 20:
                    reasons.append("insufficient FY data")
                if current_available < min_quarters_needed:
                    reasons.append(f"only {current_available}/{len(required_quarters)} current quarters available")
                if completeness_score <= 30:
                    reasons.append(f"low overall completeness ({completeness_score:.1f}%)")
                reason = "; ".join(reasons)
            else:
                reason = "Sufficient quarterly data available"
            
            return {
                'viable': viable,
                'completeness_score': completeness_score,
                'reason': reason,
                'components': {
                    'previous_fy': fy_completeness,
                    'current_quarters': current_scores,
                    'previous_quarters': previous_scores,
                    'quarters_available': f"{current_available + previous_available}/{len(required_quarters) * 2}"
                }
            }
            
        except Exception as e:
            return {
                'viable': False,
                'completeness_score': 0,
                'reason': f"Error assessing quarterly viability: {e}",
                'components': {}
            }

    def _calculate_field_completeness(self, period: Optional[Dict], required_fields: set) -> float:
        """
        Calculate what percentage of required fields are non-null in a period
        
        Returns:
            Percentage (0-100) of required fields that have non-null values
        """
        if not period:
            return 0
        
        non_null_count = 0
        for field in required_fields:
            value = period.get(field)
            if value is not None and value != 0:  # Count non-null and non-zero as valid
                non_null_count += 1
        
        return (non_null_count / len(required_fields)) * 100 if required_fields else 0

    def _get_required_fields(self, statement_type: str) -> set:
        """Get the set of most important fields for calculation viability"""
        if statement_type == 'income_statement':
            # Most critical income statement fields for LTM calculation
            return {
                'total_revenue',
                'net_income', 
                'operating_income',
                'cost_of_revenue',
                'gross_profit'
            }
        else:  # cash_flow
            # Most critical cash flow fields for LTM calculation
            return {
                'net_cash_from_operating_activities',
                'capital_expenditures',
                'depreciation_and_amortization',
                'net_cash_from_investing_activities',
                'net_cash_from_financing_activities'
            }

    def _calculate_ltm_quarterly_sum_enhanced(self, ticker: str, current_period: Dict, 
                                            statement_type: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced quarterly sum calculation with better error handling
        """
        try:
            period_type = current_period.get('period_type')
            fiscal_year = current_period.get('fiscal_year')
            
            # Get previous FY
            previous_fy = self._get_full_year_data(ticker, fiscal_year - 1, statement_type)
            if not previous_fy:
                logger.error(f"No previous FY data found for enhanced quarterly sum")
                return None
            
            # Determine required quarters
            if period_type == 'Q2':
                required_quarters = ['Q1', 'Q2']
            else:  # Q3
                required_quarters = ['Q1', 'Q2', 'Q3']
            
            # Collect quarters with detailed logging
            current_quarters = []
            previous_quarters = []
            
            logger.info(f"Collecting quarters for enhanced calculation:")
            
            for quarter in required_quarters:
                if quarter == period_type:
                    current_quarters.append(current_period)
                    logger.info(f"  Current {quarter}: Using provided period data")
                else:
                    current_q = self._get_quarterly_period_3m_only(ticker, fiscal_year, quarter, statement_type)
                    current_quarters.append(current_q)
                    status = "Found" if current_q else "Missing"
                    logger.info(f"  Current {quarter}: {status}")
                
                previous_q = self._get_quarterly_period_3m_only(ticker, fiscal_year - 1, quarter, statement_type)
                previous_quarters.append(previous_q)
                status = "Found" if previous_q else "Missing"
                logger.info(f"  Previous {quarter}: {status}")
            
            # Calculate with enhanced error handling
            ltm_values = self._calculate_ltm_values_quarterly_sum_enhanced(
                previous_fy, current_quarters, previous_quarters, statement_type
            )
            
            # Create LTM record
            ltm_record = self._create_ltm_record(current_period, ltm_values)
            
            # Add enhanced metadata
            ltm_record['calculation_method'] = 'enhanced_quarterly_sum'
            ltm_record['calculation_inputs'] = {
                'method': 'quarterly_sum_enhanced',
                'quarters_used': len([q for q in current_quarters + previous_quarters if q is not None]),
                'total_quarters_needed': len(required_quarters) * 2,
                'data_quality_check': 'passed'
            }
            
            logger.info(f"Successfully calculated enhanced quarterly sum LTM")
            return ltm_record
            
        except Exception as e:
            logger.error(f"Error in enhanced quarterly sum calculation: {e}")
            return None

    def _calculate_ltm_values_quarterly_sum_enhanced(self, previous_fy: Dict[str, Any], 
                                                   current_quarters: List[Optional[Dict[str, Any]]],
                                                   previous_quarters: List[Optional[Dict[str, Any]]], 
                                                   statement_type: str) -> Dict[str, Any]:
        """
        Enhanced quarterly sum calculation with field-by-field null handling
        """
        ltm_values = {}
        
        # Determine which fields to calculate based on statement type
        if statement_type == 'income_statement':
            # Exclude share count fields from additive calculations - they're handled separately as point-in-time
            fields_to_calculate = self.income_statement_fields - {'weighted_average_shares_basic', 'weighted_average_shares_diluted'}
        else:
            fields_to_calculate = self.cash_flow_fields
        
        for field in fields_to_calculate:
            try:
                # Get FY value
                fy_value = self._get_decimal_value(previous_fy.get(field))
                
                if fy_value is None:
                    ltm_values[field] = None
                    continue
                
                # Sum current quarters (skip None values)
                current_sum = Decimal('0')
                current_values_found = []
                
                for quarter in current_quarters:
                    if quarter:
                        quarter_value = self._get_decimal_value(quarter.get(field))
                        if quarter_value is not None:
                            current_sum += quarter_value
                            current_values_found.append(quarter_value)
                
                # Sum previous quarters (skip None values)
                previous_sum = Decimal('0')
                previous_values_found = []
                
                for quarter in previous_quarters:
                    if quarter:
                        quarter_value = self._get_decimal_value(quarter.get(field))
                        if quarter_value is not None:
                            previous_sum += quarter_value
                            previous_values_found.append(quarter_value)
                
                # Enhanced calculation logic:
                # Only calculate if we have reasonable data coverage
                current_coverage = len(current_values_found)
                previous_coverage = len(previous_values_found)
                total_quarters_expected = len(current_quarters)
                
                # Require at least 50% coverage for current quarters
                min_coverage = max(1, total_quarters_expected // 2)
                
                if current_coverage >= min_coverage:
                    if previous_coverage >= min_coverage:
                        # Full calculation with both current and previous
                        ltm_value = fy_value + current_sum - previous_sum
                        logger.debug(f"Field {field}: Full calculation = {fy_value} + {current_sum} - {previous_sum} = {ltm_value}")
                    else:
                        # Only current quarters available - use FY + current (with warning)
                        ltm_value = fy_value + current_sum
                        logger.warning(f"Field {field}: Partial calculation (missing previous data) = {fy_value} + {current_sum} = {ltm_value}")
                    
                    ltm_values[field] = ltm_value
                else:
                    # Insufficient current quarter data
                    logger.warning(f"Field {field}: Insufficient current quarter data ({current_coverage}/{total_quarters_expected} quarters)")
                    ltm_values[field] = None
                    
            except Exception as e:
                logger.debug(f"Error calculating enhanced quarterly sum for field {field}: {e}")
                ltm_values[field] = None
        
        # Handle share counts as point-in-time values (not additive)
        if statement_type == 'income_statement' and current_quarters:
            # For share counts, use the most recent quarter as point-in-time values
            # Filter out None values first
            valid_quarters = [q for q in current_quarters if q is not None]
            if valid_quarters:
                most_recent_quarter = max(valid_quarters, key=lambda q: q.get('period_end_date', date.min))
                ltm_values['weighted_average_shares_basic'] = self._get_decimal_value(most_recent_quarter.get('weighted_average_shares_basic'))
                ltm_values['weighted_average_shares_diluted'] = self._get_decimal_value(most_recent_quarter.get('weighted_average_shares_diluted'))
                logger.debug(f"Added point-in-time share counts from most recent quarter: basic={ltm_values['weighted_average_shares_basic']}, diluted={ltm_values['weighted_average_shares_diluted']}")
        
        return ltm_values
    
    def _calculate_ltm_values_standard(self, previous_fy: Dict[str, Any], current_period: Dict[str, Any],
                             previous_same_period: Dict[str, Any], statement_type: str) -> Dict[str, Any]:
        """
        Calculate LTM values using: FY(prev year) + Current Period - Same Period(prev year)
        """
        ltm_values = {}
        
        # Determine which fields to calculate based on statement type
        if statement_type == 'income_statement':
            # Exclude share count fields from additive calculations - they're handled separately as point-in-time
            fields_to_calculate = self.income_statement_fields - {'weighted_average_shares_basic', 'weighted_average_shares_diluted'}
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
                    logger.debug(f"Field {field}: Complete standard calculation = {fy_value} + {current_value} - {previous_value} = {ltm_value}")
                else:
                    # FIXED: Don't use incorrect fallback calculation that double-counts periods
                    # If we don't have all required data, skip this field to avoid mathematical errors
                    ltm_values[field] = None
                    missing_components = []
                    if fy_value is None:
                        missing_components.append("FY data")
                    if current_value is None:
                        missing_components.append("current period data") 
                    if previous_value is None:
                        missing_components.append("previous same period data")
                    logger.warning(f"Field {field}: Skipping calculation due to missing {', '.join(missing_components)}")
                    
            except Exception as e:
                logger.debug(f"Error calculating LTM for field {field}: {e}")
                ltm_values[field] = None
        
        # Handle share counts as point-in-time values (not additive)
        if statement_type == 'income_statement':
            # For share counts, use the most recent period (current_period) as point-in-time values
            ltm_values['weighted_average_shares_basic'] = self._get_decimal_value(current_period.get('weighted_average_shares_basic'))
            ltm_values['weighted_average_shares_diluted'] = self._get_decimal_value(current_period.get('weighted_average_shares_diluted'))
            logger.debug(f"Added point-in-time share counts from current period: basic={ltm_values['weighted_average_shares_basic']}, diluted={ltm_values['weighted_average_shares_diluted']}")
        
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
            # Exclude share count fields from additive calculations - they're handled separately as point-in-time
            fields_to_calculate = self.income_statement_fields - {'weighted_average_shares_basic', 'weighted_average_shares_diluted'}
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
        
        # Handle share counts as point-in-time values (not additive)
        if statement_type == 'income_statement' and current_quarters:
            # For share counts, use the most recent quarter as point-in-time values
            # Filter out None values first
            valid_quarters = [q for q in current_quarters if q is not None]
            if valid_quarters:
                most_recent_quarter = max(valid_quarters, key=lambda q: q.get('period_end_date', date.min))
                ltm_values['weighted_average_shares_basic'] = self._get_decimal_value(most_recent_quarter.get('weighted_average_shares_basic'))
                ltm_values['weighted_average_shares_diluted'] = self._get_decimal_value(most_recent_quarter.get('weighted_average_shares_diluted'))
                logger.debug(f"Added point-in-time share counts from most recent quarter: basic={ltm_values['weighted_average_shares_basic']}, diluted={ltm_values['weighted_average_shares_diluted']}")
        
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
            'cash_and_cash_equivalents', 'cash_and_short_term_investments', 'total_current_assets', 'total_current_liabilities'
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
        
        # Calculate derived metrics from LTM components (fix for negative gross profit issue)
        self._calculate_derived_ltm_metrics(ltm_record)
        
        # Mark as calculated LTM data
        ltm_record['period_type'] = 'LTM'
        ltm_record['period_length_months'] = 12
        ltm_record['is_ltm'] = True
        ltm_record['data_source'] = 'Calculated_LTM'
        ltm_record['ltm_calculation_date'] = datetime.utcnow()
        
        return ltm_record
    
    def _calculate_derived_ltm_metrics(self, ltm_record: Dict[str, Any]) -> None:
        """
        Calculate derived metrics from LTM component values to fix calculation errors
        
        This fixes issues like negative gross profit by computing derived metrics
        from their LTM components instead of relying on direct summation.
        """
        try:
            # Income statement derived metrics
            
            # 1. Gross Profit = Total Revenue - Cost of Revenue
            total_revenue = self._get_decimal_value(ltm_record.get('total_revenue'))
            cost_of_revenue = self._get_decimal_value(ltm_record.get('cost_of_revenue'))
            
            if total_revenue is not None and cost_of_revenue is not None:
                calculated_gross_profit = total_revenue - cost_of_revenue
                
                # Only update if the calculated value differs significantly from existing
                existing_gross_profit = self._get_decimal_value(ltm_record.get('gross_profit'))
                
                if existing_gross_profit is None or abs(calculated_gross_profit - existing_gross_profit) > abs(calculated_gross_profit * Decimal('0.1')):
                    logger.info(f"Correcting LTM gross profit: {existing_gross_profit} -> {calculated_gross_profit}")
                    ltm_record['gross_profit'] = calculated_gross_profit
            
            # 2. Operating Income = Gross Profit - Total Operating Expenses
            gross_profit = self._get_decimal_value(ltm_record.get('gross_profit'))
            total_operating_expenses = self._get_decimal_value(ltm_record.get('total_operating_expenses'))
            
            if gross_profit is not None and total_operating_expenses is not None:
                calculated_operating_income = gross_profit - total_operating_expenses
                existing_operating_income = self._get_decimal_value(ltm_record.get('operating_income'))
                
                if existing_operating_income is None or abs(calculated_operating_income - existing_operating_income) > abs(calculated_operating_income * Decimal('0.1')):
                    logger.info(f"Correcting LTM operating income: {existing_operating_income} -> {calculated_operating_income}")
                    ltm_record['operating_income'] = calculated_operating_income
            
            # 3. Income Before Taxes = Operating Income + Other Income - Interest Expense
            operating_income = self._get_decimal_value(ltm_record.get('operating_income'))
            other_income = self._get_decimal_value(ltm_record.get('other_income')) or Decimal('0')
            interest_expense = self._get_decimal_value(ltm_record.get('interest_expense')) or Decimal('0')
            interest_income = self._get_decimal_value(ltm_record.get('interest_income')) or Decimal('0')
            
            if operating_income is not None:
                calculated_income_before_taxes = operating_income + other_income + interest_income - interest_expense
                existing_income_before_taxes = self._get_decimal_value(ltm_record.get('income_before_taxes'))
                
                if existing_income_before_taxes is None or abs(calculated_income_before_taxes - existing_income_before_taxes) > abs(calculated_income_before_taxes * Decimal('0.1')):
                    logger.info(f"Correcting LTM income before taxes: {existing_income_before_taxes} -> {calculated_income_before_taxes}")
                    ltm_record['income_before_taxes'] = calculated_income_before_taxes
            
            # 4. Net Income = Income Before Taxes - Income Tax Expense
            income_before_taxes = self._get_decimal_value(ltm_record.get('income_before_taxes'))
            income_tax_expense = self._get_decimal_value(ltm_record.get('income_tax_expense')) or Decimal('0')
            
            if income_before_taxes is not None:
                calculated_net_income = income_before_taxes - income_tax_expense
                existing_net_income = self._get_decimal_value(ltm_record.get('net_income'))
                
                if existing_net_income is None or abs(calculated_net_income - existing_net_income) > abs(calculated_net_income * Decimal('0.1')):
                    logger.info(f"Correcting LTM net income: {existing_net_income} -> {calculated_net_income}")
                    ltm_record['net_income'] = calculated_net_income
            
            # 5. EPS Basic = Net Income / Weighted Average Shares Basic (for share-count based metrics, use latest quarter values not LTM sum)
            net_income = self._get_decimal_value(ltm_record.get('net_income'))
            shares_basic = self._get_decimal_value(ltm_record.get('weighted_average_shares_basic'))
            
            if net_income is not None and shares_basic is not None and shares_basic > 0:
                calculated_eps_basic = net_income / shares_basic
                existing_eps_basic = self._get_decimal_value(ltm_record.get('earnings_per_share_basic'))
                
                if existing_eps_basic is None or abs(calculated_eps_basic - existing_eps_basic) > abs(calculated_eps_basic * Decimal('0.1')):
                    logger.info(f"Correcting LTM EPS basic: {existing_eps_basic} -> {calculated_eps_basic}")
                    ltm_record['earnings_per_share_basic'] = calculated_eps_basic
            
            # 6. EPS Diluted = Net Income / Weighted Average Shares Diluted
            shares_diluted = self._get_decimal_value(ltm_record.get('weighted_average_shares_diluted'))
            
            if net_income is not None and shares_diluted is not None and shares_diluted > 0:
                calculated_eps_diluted = net_income / shares_diluted
                existing_eps_diluted = self._get_decimal_value(ltm_record.get('earnings_per_share_diluted'))
                
                if existing_eps_diluted is None or abs(calculated_eps_diluted - existing_eps_diluted) > abs(calculated_eps_diluted * Decimal('0.1')):
                    logger.info(f"Correcting LTM EPS diluted: {existing_eps_diluted} -> {calculated_eps_diluted}")
                    ltm_record['earnings_per_share_diluted'] = calculated_eps_diluted
            
            # Cash flow derived metrics (if this is a cash flow record)
            
            # 7. Free Cash Flow = Net Cash from Operating Activities - Capital Expenditures
            operating_cash_flow = self._get_decimal_value(ltm_record.get('net_cash_from_operating_activities'))
            capex = self._get_decimal_value(ltm_record.get('capital_expenditures')) or Decimal('0')
            
            if operating_cash_flow is not None:
                # Note: capex is typically negative, so we add it (subtract absolute value)
                calculated_free_cash_flow = operating_cash_flow + capex  # capex is already negative
                existing_free_cash_flow = self._get_decimal_value(ltm_record.get('free_cash_flow'))
                
                if existing_free_cash_flow is None or abs(calculated_free_cash_flow - existing_free_cash_flow) > abs(calculated_free_cash_flow * Decimal('0.1')):
                    logger.info(f"Correcting LTM free cash flow: {existing_free_cash_flow} -> {calculated_free_cash_flow}")
                    ltm_record['free_cash_flow'] = calculated_free_cash_flow
            
            # 8. Net Change in Cash = Operating + Investing + Financing Cash Flows
            investing_cash_flow = self._get_decimal_value(ltm_record.get('net_cash_from_investing_activities')) or Decimal('0')
            financing_cash_flow = self._get_decimal_value(ltm_record.get('net_cash_from_financing_activities')) or Decimal('0')
            
            if operating_cash_flow is not None:
                calculated_net_change_cash = operating_cash_flow + investing_cash_flow + financing_cash_flow
                existing_net_change_cash = self._get_decimal_value(ltm_record.get('net_change_in_cash'))
                
                if existing_net_change_cash is None or abs(calculated_net_change_cash - existing_net_change_cash) > abs(calculated_net_change_cash * Decimal('0.1')):
                    logger.info(f"Correcting LTM net change in cash: {existing_net_change_cash} -> {calculated_net_change_cash}")
                    ltm_record['net_change_in_cash'] = calculated_net_change_cash
            
            logger.debug(f"Completed derived LTM metrics calculation with 8 derived metrics")
            
        except Exception as e:
            logger.warning(f"Error calculating derived LTM metrics: {e}")
    
    
    def store_ltm_data_in_database(self, ticker: str, statement_type: str = 'income_statement') -> Dict[str, Any]:
        """
        Store LTM data directly in PostgreSQL database tables
        
        Args:
            ticker: Company ticker symbol
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            Dictionary with storage results
        """
        try:
            print(f"[LTM STORAGE] Starting database storage for {ticker} {statement_type}")
            
            # Get calculated LTM records
            print(f"[LTM STORAGE] Calling calculate_ltm_for_all_quarters...")
            ltm_records = self.calculate_ltm_for_all_quarters(ticker, statement_type)
            print(f"[LTM STORAGE] Got {len(ltm_records) if ltm_records else 0} LTM records")
            
            if not ltm_records:
                print(f"[LTM STORAGE] No LTM records found - returning failure")
                return {
                    'success': False,
                    'ticker': ticker,
                    'statement_type': statement_type,
                    'error': 'No LTM records calculated',
                    'inserted': 0,
                    'skipped': 0
                }
            
            # Get company info for company_id
            print(f"[LTM STORAGE] Getting company info for {ticker}...")
            company_info = self.database.get_company_by_ticker(ticker)
            if not company_info:
                print(f"[LTM STORAGE] Company {ticker} not found in database")
                return {
                    'success': False,
                    'ticker': ticker,
                    'statement_type': statement_type,
                    'error': f'Company {ticker} not found in database',
                    'inserted': 0,
                    'skipped': 0
                }
            
            company_id = company_info['id']
            print(f"[LTM STORAGE] Found company {ticker} with ID: {company_id}")
            
            # Transform LTM records to database format
            print(f"[LTM STORAGE] Transforming {len(ltm_records)} LTM records to database format...")
            db_records = []
            for i, record in enumerate(ltm_records):
                print(f"[LTM STORAGE] Transforming record {i+1}: period_end_date={record.get('period_end_date')}")
                # Map the LTM record to database schema
                db_record = self._transform_ltm_record_for_database(
                    record, company_id, company_info, statement_type
                )
                if db_record:
                    db_records.append(db_record)
                    print(f"[LTM STORAGE] Record {i+1} transformed successfully")
                else:
                    print(f"[LTM STORAGE] Record {i+1} transformation failed")
            
            print(f"[LTM STORAGE] {len(db_records)} records ready for database insertion")
            
            if not db_records:
                print(f"[LTM STORAGE] No valid records after transformation")
                return {
                    'success': False,
                    'ticker': ticker,
                    'statement_type': statement_type,
                    'error': 'No valid LTM records to store',
                    'inserted': 0,
                    'skipped': 0
                }
            
            # Store calculated LTM records in database
            print(f"[LTM STORAGE] Calling bulk_insert_ltm_statements for calculated LTM data...")
            inserted, skipped = self.database.bulk_insert_ltm_statements(db_records, statement_type)
            print(f"[LTM STORAGE] Calculated LTM bulk insert result: {inserted} inserted, {skipped} skipped")
            
            # Also store FY data (existing 12-month statements) as LTM records
            print(f"[LTM STORAGE] Getting existing 12-month FY data for LTM storage...")
            fy_ltm_records = self._get_fy_data_as_ltm_records(ticker, company_id, company_info, statement_type)
            
            if fy_ltm_records:
                print(f"[LTM STORAGE] Found {len(fy_ltm_records)} FY records to store as LTM...")
                fy_inserted, fy_skipped = self.database.bulk_insert_ltm_statements(fy_ltm_records, statement_type)
                print(f"[LTM STORAGE] FY-as-LTM bulk insert result: {fy_inserted} inserted, {fy_skipped} skipped")
                inserted += fy_inserted
                skipped += fy_skipped
            else:
                print(f"[LTM STORAGE] No FY data found to convert to LTM records")
            
            result = {
                'success': True,
                'ticker': ticker,
                'statement_type': statement_type,
                'inserted': inserted,
                'skipped': skipped,
                'total_records': len(db_records)
            }
            
            logger.info(f"Successfully stored LTM {statement_type} for {ticker}: "
                       f"{inserted} inserted, {skipped} skipped")
            
            return result
            
        except Exception as e:
            logger.error(f"Error storing LTM {statement_type} data for {ticker}: {e}")
            return {
                'success': False,
                'ticker': ticker,
                'statement_type': statement_type,
                'error': str(e),
                'inserted': 0,
                'skipped': 0
            }
    
    def _transform_ltm_record_for_database(self, ltm_record: Dict[str, Any], company_id: str, 
                                         company_info: Dict[str, Any], statement_type: str) -> Optional[Dict[str, Any]]:
        """
        Transform an LTM record to match the database schema
        
        Args:
            ltm_record: LTM calculation result
            company_id: Company UUID
            company_info: Company information dictionary
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            Database-ready record dictionary or None if invalid
        """
        try:
            # Extract base information
            base_quarter_end_date = ltm_record.get('period_end_date')
            if not base_quarter_end_date:
                logger.warning("LTM record missing period_end_date")
                return None
            
            # Convert string dates to date objects if needed
            if isinstance(base_quarter_end_date, str):
                base_quarter_end_date = datetime.strptime(base_quarter_end_date, '%Y-%m-%d').date()
            elif isinstance(base_quarter_end_date, datetime):
                base_quarter_end_date = base_quarter_end_date.date()
            
            # Calculate LTM period dates (approximately 365 days back)
            from datetime import timedelta
            ltm_period_start_date = base_quarter_end_date - timedelta(days=364)  # 365 days back, plus 1 day
            
            # Determine fiscal quarter
            period_type = ltm_record.get('period_type', 'Q4')  # Default to Q4
            if period_type not in ['Q1', 'Q2', 'Q3', 'Q4']:
                # Extract quarter from period_type if it's in different format
                if 'Q' in str(period_type):
                    period_type = str(period_type)
                else:
                    period_type = 'Q4'  # Default
            
            # Create the database record with all common fields
            db_record = {
                'company_id': company_id,
                'cik': company_info.get('cik', ''),
                'ticker': company_info.get('ticker', '').upper(),
                'company_name': company_info.get('name', ''),
                
                # LTM period information (using standardized field names)
                'period_end_date': base_quarter_end_date,
                'ltm_period_start_date': ltm_period_start_date,
                'base_quarter_end_date': base_quarter_end_date,
                'fiscal_year': ltm_record.get('fiscal_year'),
                'period_type': period_type,
                'form_type': ltm_record.get('form_type'),
                
                # Currency and units
                'currency': ltm_record.get('currency', 'USD'),
                'units': ltm_record.get('units', 'USD'),
                
                # LTM-specific metadata
                'calculation_method': 'standard',
                'calculation_inputs': ltm_record.get('calculation_inputs', {}),
                'ltm_calculation_date': datetime.now()
            }
            
            # Add statement-specific fields
            if statement_type == 'income_statement':
                # Copy income statement fields (including share counts)
                for field in self.income_statement_fields:
                    db_record[field] = ltm_record.get(field)
                
            elif statement_type == 'cash_flow':
                # Copy cash flow fields  
                for field in self.cash_flow_fields:
                    db_record[field] = ltm_record.get(field)
            
            else:
                logger.error(f"Unsupported statement type for database storage: {statement_type}")
                return None
            
            return db_record
            
        except Exception as e:
            logger.error(f"Error transforming LTM record for database: {e}")
            return None
    
    def _get_fy_data_as_ltm_records(self, ticker: str, company_id: str, company_info: Dict[str, Any], 
                                   statement_type: str) -> List[Dict[str, Any]]:
        """
        Get existing 12-month FY data and convert to LTM record format
        
        Args:
            ticker: Company ticker symbol
            company_id: Company UUID
            company_info: Company information dictionary
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            List of FY records converted to LTM format
        """
        fy_ltm_records = []
        
        try:
            print(f"[LTM STORAGE FY] Getting 12-month FY data for {ticker} {statement_type}...")
            
            # Get 12-month records from regular statement tables
            if statement_type == 'income_statement':
                table_name = 'income_statements'
            elif statement_type == 'cash_flow':
                table_name = 'cash_flow_statements'
            else:
                print(f"[LTM STORAGE FY] Unsupported statement type: {statement_type}")
                return []
            
            with self.database.connection.cursor() as cursor:
                # Get all 12-month records (FY data)
                cursor.execute(f"""
                    SELECT * FROM {table_name} 
                    WHERE ticker = %s AND period_length_months = 12
                    ORDER BY period_end_date DESC
                """, (ticker.upper(),))
                
                columns = [desc[0] for desc in cursor.description]
                fy_statements = cursor.fetchall()
                
                print(f"[LTM STORAGE FY] Found {len(fy_statements)} 12-month records")
                
                for i, row in enumerate(fy_statements):
                    statement_dict = dict(zip(columns, row))
                    print(f"[LTM STORAGE FY] Processing FY record {i+1}: period_end={statement_dict.get('period_end_date')}")
                    
                    # Convert FY statement to LTM record format
                    ltm_record = self._convert_fy_statement_to_ltm_record(
                        statement_dict, company_id, company_info, statement_type
                    )
                    
                    if ltm_record:
                        fy_ltm_records.append(ltm_record)
                        print(f"[LTM STORAGE FY] FY record {i+1} converted to LTM format successfully")
                    else:
                        print(f"[LTM STORAGE FY] FY record {i+1} conversion failed")
                        
        except Exception as e:
            print(f"[LTM STORAGE FY] Error getting FY data as LTM records: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"[LTM STORAGE FY] Converted {len(fy_ltm_records)} FY records to LTM format")
        return fy_ltm_records
    
    def _convert_fy_statement_to_ltm_record(self, statement_dict: Dict[str, Any], company_id: str,
                                          company_info: Dict[str, Any], statement_type: str) -> Optional[Dict[str, Any]]:
        """
        Convert a 12-month FY statement record to LTM table format
        
        Args:
            statement_dict: FY statement record from regular table
            company_id: Company UUID
            company_info: Company information
            statement_type: 'income_statement' or 'cash_flow'
            
        Returns:
            LTM-formatted record or None if conversion fails
        """
        try:
            period_end_date = statement_dict.get('period_end_date')
            if not period_end_date:
                return None
            
            # Convert string dates to date objects if needed
            if isinstance(period_end_date, str):
                period_end_date = datetime.strptime(period_end_date, '%Y-%m-%d').date()
            elif isinstance(period_end_date, datetime):
                period_end_date = period_end_date.date()
            
            # Calculate LTM period dates (for FY data, it's the same as the FY period)
            from datetime import timedelta
            ltm_period_start_date = period_end_date - timedelta(days=364)  # Approximately 1 year back
            
            # Determine period type (for FY data, it's usually Q4, but check period_type)
            source_period_type = statement_dict.get('period_type', 'Q4')
            if source_period_type == 'FY':
                period_type = 'Q4'  # FY typically ends at Q4
            else:
                period_type = source_period_type
            
            # Create the LTM record with all fields from the FY statement
            ltm_record = {
                'company_id': company_id,
                'cik': statement_dict.get('cik', company_info.get('cik', '')),
                'ticker': company_info.get('ticker', '').upper(),
                'company_name': statement_dict.get('company_name', company_info.get('name', '')),
                
                # LTM period information (using standardized field names)
                'period_end_date': period_end_date,
                'ltm_period_start_date': ltm_period_start_date,
                'base_quarter_end_date': period_end_date,
                'fiscal_year': statement_dict.get('fiscal_year'),
                'period_type': period_type,
                'form_type': statement_dict.get('form_type'),
                
                # Currency and units
                'currency': statement_dict.get('currency', 'USD'),
                'units': statement_dict.get('units', 'USD'),
                
                # LTM-specific metadata
                'calculation_method': 'fy_direct',  # Mark as direct FY data
                'calculation_inputs': {'source': 'fy_12m_statement', 'original_period_type': statement_dict.get('period_type')},
                'ltm_calculation_date': datetime.now()
            }
            
            # Copy all financial fields from the FY statement
            financial_fields = self.income_statement_fields if statement_type == 'income_statement' else self.cash_flow_fields
            
            for field in financial_fields:
                ltm_record[field] = statement_dict.get(field)
            
            # Add additional fields for income statements
            if statement_type == 'income_statement':
                ltm_record['weighted_average_shares_basic'] = statement_dict.get('weighted_average_shares_basic')
                ltm_record['weighted_average_shares_diluted'] = statement_dict.get('weighted_average_shares_diluted')
            
            return ltm_record
            
        except Exception as e:
            print(f"[LTM STORAGE FY] Error converting FY statement to LTM: {e}")
            return None
    
    def _calculate_ltm_values_standard(self, previous_fy: Dict[str, Any], 
                                     current_period: Dict[str, Any], 
                                     previous_same_period: Dict[str, Any], 
                                     statement_type: str) -> Dict[str, Any]:
        """
        Standard LTM calculation: FY(prev year) + Current Period - Same Period(prev year)
        """
        ltm_values = {}
        
        # Determine which fields to calculate based on statement type
        if statement_type == 'income_statement':
            fields_to_calculate = self.income_statement_fields
        else:
            fields_to_calculate = self.cash_flow_fields
        
        for field in fields_to_calculate:
            try:
                # Get values from each period
                fy_value = self._get_decimal_value(previous_fy.get(field))
                current_value = self._get_decimal_value(current_period.get(field))
                previous_value = self._get_decimal_value(previous_same_period.get(field))
                
                # Standard LTM formula: FY + Current - Previous Same Period
                if all(v is not None for v in [fy_value, current_value, previous_value]):
                    ltm_value = fy_value + current_value - previous_value
                    ltm_values[field] = ltm_value
                else:
                    # If any required value is missing, skip this field
                    ltm_values[field] = None
                    
            except Exception as e:
                logger.debug(f"Error calculating standard LTM for field {field}: {e}")
                ltm_values[field] = None
        
        # Handle share counts as point-in-time values (not additive)
        if statement_type == 'income_statement':
            # For share counts, use the most recent period (current_period) as point-in-time values
            ltm_values['weighted_average_shares_basic'] = self._get_decimal_value(current_period.get('weighted_average_shares_basic'))
            ltm_values['weighted_average_shares_diluted'] = self._get_decimal_value(current_period.get('weighted_average_shares_diluted'))
            logger.debug(f"Added point-in-time share counts from current period: basic={ltm_values['weighted_average_shares_basic']}, diluted={ltm_values['weighted_average_shares_diluted']}")
        
        return ltm_values
    
    def _calculate_ltm_values_fy_plus_current(self, previous_fy: Dict[str, Any], 
                                            current_period: Dict[str, Any], 
                                            statement_type: str) -> Dict[str, Any]:
        """
        Fallback LTM calculation when same period data is missing: FY(prev year) + Current Period
        Note: This is an approximation and may not be as accurate as the standard calculation
        """
        ltm_values = {}
        
        # Determine which fields to calculate based on statement type
        if statement_type == 'income_statement':
            fields_to_calculate = self.income_statement_fields
        else:
            fields_to_calculate = self.cash_flow_fields
        
        for field in fields_to_calculate:
            try:
                # Get values from each period
                fy_value = self._get_decimal_value(previous_fy.get(field))
                current_value = self._get_decimal_value(current_period.get(field))
                
                # Fallback formula: FY + Current (no subtraction)
                # This assumes the current period doesn't overlap with FY period
                if fy_value is not None and current_value is not None:
                    ltm_value = fy_value + current_value
                    ltm_values[field] = ltm_value
                    logger.debug(f"Field {field}: Fallback calculation = {fy_value} + {current_value} = {ltm_value}")
                else:
                    # If either value is missing, skip this field
                    ltm_values[field] = None
                    
            except Exception as e:
                logger.debug(f"Error calculating fallback LTM for field {field}: {e}")
                ltm_values[field] = None
        
        # Handle share counts as point-in-time values (not additive)
        if statement_type == 'income_statement':
            # For share counts, use the most recent period (current_period) as point-in-time values
            ltm_values['weighted_average_shares_basic'] = self._get_decimal_value(current_period.get('weighted_average_shares_basic'))
            ltm_values['weighted_average_shares_diluted'] = self._get_decimal_value(current_period.get('weighted_average_shares_diluted'))
            logger.debug(f"Added point-in-time share counts from current period: basic={ltm_values['weighted_average_shares_basic']}, diluted={ltm_values['weighted_average_shares_diluted']}")
        
        return ltm_values
    
    def _get_decimal_value(self, value) -> Optional[Decimal]:
        """Convert a value to Decimal, handling None and string values"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            try:
                return Decimal(value)
            except (ValueError, TypeError):
                return None
        if isinstance(value, Decimal):
            return value
        return None
    
    def _create_ltm_record(self, base_period: Dict[str, Any], ltm_values: Dict[str, Any]) -> Dict[str, Any]:
        """Create an LTM record with metadata"""
        ltm_record = {}
        
        # Copy base period metadata
        for key in ['ticker', 'company_id', 'period_end_date', 'fiscal_year', 
                   'period_type', 'currency', 'units', 'form_type']:
            if key in base_period:
                ltm_record[key] = base_period[key]
        
        # Add LTM values
        ltm_record.update(ltm_values)
        
        # Calculate derived metrics from LTM components (fix for negative gross profit issue)
        self._calculate_derived_ltm_metrics(ltm_record)
        
        # Add LTM-specific metadata
        ltm_record['period_length_months'] = 12  # LTM is always 12 months
        ltm_record['ltm_calculation_date'] = datetime.now()
        
        return ltm_record
    
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
