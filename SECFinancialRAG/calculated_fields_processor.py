"""
Calculated Fields Processor for SEC Financial RAG
Populates missing financial statement fields using accounting relationships
Executes after IS, BS, CF tables are populated but before LTM and ratio calculations
"""

import logging
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal
from datetime import datetime

try:
    from .database import FinancialDatabase
    from .models import IncomeStatement, BalanceSheet, CashFlowStatement
except ImportError:
    from database import FinancialDatabase
    from models import IncomeStatement, BalanceSheet, CashFlowStatement

logger = logging.getLogger(__name__)


class CalculatedFieldsProcessor:
    """
    Processor for calculating missing financial statement fields using accounting relationships
    """
    
    def __init__(self):
        self.database = FinancialDatabase()
        self.calculated_fields_log = []  # Track what was calculated for audit trail
    
    def process_calculated_fields_for_company(self, ticker: str) -> Dict[str, Any]:
        """
        Process calculated fields for all financial statements of a company
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Starting calculated fields processing for {ticker}")
        
        try:
            results = {
                'ticker': ticker,
                'income_statements_updated': 0,
                'balance_sheets_updated': 0,
                'cash_flow_statements_updated': 0,
                'total_fields_calculated': 0,
                'errors': [],
                'calculated_fields': []
            }
            
            # Process each statement type
            income_results = self._process_income_statements(ticker)
            balance_results = self._process_balance_sheets(ticker)
            cash_flow_results = self._process_cash_flow_statements(ticker)
            
            # Aggregate results
            results['income_statements_updated'] = income_results['updated_count']
            results['balance_sheets_updated'] = balance_results['updated_count']
            results['cash_flow_statements_updated'] = cash_flow_results['updated_count']
            results['total_fields_calculated'] = (
                income_results['fields_calculated'] + 
                balance_results['fields_calculated'] + 
                cash_flow_results['fields_calculated']
            )
            results['calculated_fields'].extend(income_results['calculated_fields'])
            results['calculated_fields'].extend(balance_results['calculated_fields'])
            results['calculated_fields'].extend(cash_flow_results['calculated_fields'])
            
            logger.info(f"Completed calculated fields processing for {ticker}: "
                       f"{results['total_fields_calculated']} fields calculated across "
                       f"{results['income_statements_updated'] + results['balance_sheets_updated'] + results['cash_flow_statements_updated']} statements")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing calculated fields for {ticker}: {e}")
            return {
                'ticker': ticker,
                'status': 'error',
                'error_message': str(e),
                'total_fields_calculated': 0
            }
    
    def _process_income_statements(self, ticker: str) -> Dict[str, Any]:
        """Process calculated fields for income statements"""
        try:
            statements = self.database.get_income_statements(ticker)
            updated_count = 0
            fields_calculated = 0
            calculated_fields = []
            
            for statement_dict in statements:
                # Convert to mutable dict for calculations
                updated_data = dict(statement_dict)
                original_data = dict(statement_dict)
                
                # Apply income statement calculations
                calc_count = self._calculate_income_statement_fields(updated_data, ticker)
                
                if calc_count > 0:
                    # Update database if any fields were calculated
                    success = self._update_income_statement_in_db(updated_data, original_data)
                    if success:
                        updated_count += 1
                        fields_calculated += calc_count
                        
                        # Log calculated fields
                        for field, value in updated_data.items():
                            if (field in original_data and original_data[field] is None and 
                                value is not None and field not in ['updated_at']):
                                calculated_fields.append({
                                    'statement_type': 'income_statement',
                                    'period_end_date': str(updated_data.get('period_end_date')),
                                    'field_name': field,
                                    'calculated_value': float(value) if isinstance(value, Decimal) else value,
                                    'ticker': ticker
                                })
            
            return {
                'updated_count': updated_count,
                'fields_calculated': fields_calculated,
                'calculated_fields': calculated_fields
            }
            
        except Exception as e:
            logger.error(f"Error processing income statements for {ticker}: {e}")
            return {'updated_count': 0, 'fields_calculated': 0, 'calculated_fields': []}
    
    def _process_balance_sheets(self, ticker: str) -> Dict[str, Any]:
        """Process calculated fields for balance sheets"""
        try:
            statements = self.database.get_balance_sheets(ticker)
            updated_count = 0
            fields_calculated = 0
            calculated_fields = []
            
            for statement_dict in statements:
                # Convert to mutable dict for calculations
                updated_data = dict(statement_dict)
                original_data = dict(statement_dict)
                
                # Apply balance sheet calculations
                calc_count = self._calculate_balance_sheet_fields(updated_data, ticker)
                
                if calc_count > 0:
                    # Update database if any fields were calculated
                    success = self._update_balance_sheet_in_db(updated_data, original_data)
                    if success:
                        updated_count += 1
                        fields_calculated += calc_count
                        
                        # Log calculated fields
                        for field, value in updated_data.items():
                            if (field in original_data and original_data[field] is None and 
                                value is not None and field not in ['updated_at']):
                                calculated_fields.append({
                                    'statement_type': 'balance_sheet',
                                    'period_end_date': str(updated_data.get('period_end_date')),
                                    'field_name': field,
                                    'calculated_value': float(value) if isinstance(value, Decimal) else value,
                                    'ticker': ticker
                                })
            
            return {
                'updated_count': updated_count,
                'fields_calculated': fields_calculated,
                'calculated_fields': calculated_fields
            }
            
        except Exception as e:
            logger.error(f"Error processing balance sheets for {ticker}: {e}")
            return {'updated_count': 0, 'fields_calculated': 0, 'calculated_fields': []}
    
    def _process_cash_flow_statements(self, ticker: str) -> Dict[str, Any]:
        """Process calculated fields for cash flow statements"""
        try:
            statements = self.database.get_cash_flow_statements(ticker)
            updated_count = 0
            fields_calculated = 0
            calculated_fields = []
            
            for statement_dict in statements:
                # Convert to mutable dict for calculations
                updated_data = dict(statement_dict)
                original_data = dict(statement_dict)
                
                # Apply cash flow calculations
                calc_count = self._calculate_cash_flow_fields(updated_data, ticker)
                
                if calc_count > 0:
                    # Update database if any fields were calculated
                    success = self._update_cash_flow_statement_in_db(updated_data, original_data)
                    if success:
                        updated_count += 1
                        fields_calculated += calc_count
                        
                        # Log calculated fields
                        for field, value in updated_data.items():
                            if (field in original_data and original_data[field] is None and 
                                value is not None and field not in ['updated_at']):
                                calculated_fields.append({
                                    'statement_type': 'cash_flow',
                                    'period_end_date': str(updated_data.get('period_end_date')),
                                    'field_name': field,
                                    'calculated_value': float(value) if isinstance(value, Decimal) else value,
                                    'ticker': ticker
                                })
            
            return {
                'updated_count': updated_count,
                'fields_calculated': fields_calculated,
                'calculated_fields': calculated_fields
            }
            
        except Exception as e:
            logger.error(f"Error processing cash flow statements for {ticker}: {e}")
            return {'updated_count': 0, 'fields_calculated': 0, 'calculated_fields': []}
    
    def _calculate_income_statement_fields(self, data: Dict[str, Any], ticker: str) -> int:
        """
        Calculate missing income statement fields using accounting relationships
        Returns number of fields calculated
        """
        calc_count = 0
        
        # Helper function to safely get decimal value
        def get_decimal(value) -> Optional[Decimal]:
            if value is None:
                return None
            try:
                return Decimal(str(value)) if value != 0 else Decimal('0')
            except:
                return None
        
        # Helper function to check if field is missing
        def is_missing(field_name: str) -> bool:
            return data.get(field_name) is None
        
        # 1. Revenue & Cost Relationships
        total_revenue = get_decimal(data.get('total_revenue'))
        cost_of_revenue = get_decimal(data.get('cost_of_revenue'))
        gross_profit = get_decimal(data.get('gross_profit'))
        
        if is_missing('cost_of_revenue') and total_revenue and gross_profit:
            data['cost_of_revenue'] = total_revenue - gross_profit
            calc_count += 1
            logger.debug(f"{ticker}: Calculated cost_of_revenue = {data['cost_of_revenue']}")
        
        if is_missing('gross_profit') and total_revenue and cost_of_revenue:
            data['gross_profit'] = total_revenue - cost_of_revenue
            calc_count += 1
            logger.debug(f"{ticker}: Calculated gross_profit = {data['gross_profit']}")
        
        # 2. Operating Expense Relationships
        sales_and_marketing = get_decimal(data.get('sales_and_marketing'))
        general_and_administrative = get_decimal(data.get('general_and_administrative'))
        
        if (is_missing('sales_general_and_admin') and 
            sales_and_marketing is not None and general_and_administrative is not None):
            data['sales_general_and_admin'] = sales_and_marketing + general_and_administrative
            calc_count += 1
            logger.debug(f"{ticker}: Calculated sales_general_and_admin = {data['sales_general_and_admin']}")
        
        # 3. Operating Income Relationships
        total_operating_expenses = get_decimal(data.get('total_operating_expenses'))
        operating_income = get_decimal(data.get('operating_income'))
        gross_profit = get_decimal(data.get('gross_profit'))  # Refresh after potential calculation
        
        if is_missing('operating_income') and gross_profit and total_operating_expenses:
            data['operating_income'] = gross_profit - total_operating_expenses
            calc_count += 1
            logger.debug(f"{ticker}: Calculated operating_income = {data['operating_income']}")
        
        if is_missing('total_operating_expenses') and gross_profit and operating_income:
            data['total_operating_expenses'] = gross_profit - operating_income
            calc_count += 1
            logger.debug(f"{ticker}: Calculated total_operating_expenses = {data['total_operating_expenses']}")
        
        # 4. Pre-tax Income Relationships
        net_income = get_decimal(data.get('net_income'))
        income_tax_expense = get_decimal(data.get('income_tax_expense'))
        income_before_taxes = get_decimal(data.get('income_before_taxes'))
        
        if is_missing('income_before_taxes') and net_income is not None and income_tax_expense is not None:
            data['income_before_taxes'] = net_income + income_tax_expense
            calc_count += 1
            logger.debug(f"{ticker}: Calculated income_before_taxes = {data['income_before_taxes']}")
        
        # 5. Net Income Relationships
        income_before_taxes = get_decimal(data.get('income_before_taxes'))  # Refresh after potential calculation
        
        if is_missing('net_income') and income_before_taxes and income_tax_expense is not None:
            data['net_income'] = income_before_taxes - income_tax_expense
            calc_count += 1
            logger.debug(f"{ticker}: Calculated net_income = {data['net_income']}")
        
        if is_missing('income_tax_expense') and income_before_taxes and net_income:
            data['income_tax_expense'] = income_before_taxes - net_income
            calc_count += 1
            logger.debug(f"{ticker}: Calculated income_tax_expense = {data['income_tax_expense']}")
        
        # 6. EPS Calculations
        net_income = get_decimal(data.get('net_income'))  # Refresh after potential calculation
        weighted_average_shares_basic = get_decimal(data.get('weighted_average_shares_basic'))
        weighted_average_shares_diluted = get_decimal(data.get('weighted_average_shares_diluted'))
        
        if (is_missing('earnings_per_share_basic') and net_income and 
            weighted_average_shares_basic and weighted_average_shares_basic != 0):
            data['earnings_per_share_basic'] = net_income / weighted_average_shares_basic
            calc_count += 1
            logger.debug(f"{ticker}: Calculated earnings_per_share_basic = {data['earnings_per_share_basic']}")
        
        if (is_missing('earnings_per_share_diluted') and net_income and 
            weighted_average_shares_diluted and weighted_average_shares_diluted != 0):
            data['earnings_per_share_diluted'] = net_income / weighted_average_shares_diluted
            calc_count += 1
            logger.debug(f"{ticker}: Calculated earnings_per_share_diluted = {data['earnings_per_share_diluted']}")
        
        # 7. Reverse EPS Calculations (when EPS is available but share count is missing)
        earnings_per_share_basic = get_decimal(data.get('earnings_per_share_basic'))
        earnings_per_share_diluted = get_decimal(data.get('earnings_per_share_diluted'))
        net_income = get_decimal(data.get('net_income'))  # Refresh after potential calculation
        
        if (is_missing('weighted_average_shares_basic') and net_income and 
            earnings_per_share_basic and earnings_per_share_basic != 0):
            data['weighted_average_shares_basic'] = net_income / earnings_per_share_basic
            calc_count += 1
            logger.debug(f"{ticker}: Calculated weighted_average_shares_basic = {data['weighted_average_shares_basic']}")
        
        if (is_missing('weighted_average_shares_diluted') and net_income and 
            earnings_per_share_diluted and earnings_per_share_diluted != 0):
            data['weighted_average_shares_diluted'] = net_income / earnings_per_share_diluted
            calc_count += 1
            logger.debug(f"{ticker}: Calculated weighted_average_shares_diluted = {data['weighted_average_shares_diluted']}")
        
        return calc_count
    
    def _calculate_balance_sheet_fields(self, data: Dict[str, Any], ticker: str) -> int:
        """
        Calculate missing balance sheet fields using accounting relationships
        Returns number of fields calculated
        """
        calc_count = 0
        
        # Helper function to safely get decimal value
        def get_decimal(value) -> Optional[Decimal]:
            if value is None:
                return None
            try:
                return Decimal(str(value)) if value != 0 else Decimal('0')
            except:
                return None
        
        # Helper function to check if field is missing
        def is_missing(field_name: str) -> bool:
            return data.get(field_name) is None
        
        # Get key totals
        total_assets = get_decimal(data.get('total_assets'))
        total_current_assets = get_decimal(data.get('total_current_assets'))
        total_non_current_assets = get_decimal(data.get('total_non_current_assets'))
        total_liabilities = get_decimal(data.get('total_liabilities'))
        total_current_liabilities = get_decimal(data.get('total_current_liabilities'))
        total_non_current_liabilities = get_decimal(data.get('total_non_current_liabilities'))
        
        # 7-8. Asset Totals (Preferred Method - using totals)
        if is_missing('total_current_assets') and total_assets and total_non_current_assets:
            data['total_current_assets'] = total_assets - total_non_current_assets
            calc_count += 1
            logger.debug(f"{ticker}: Calculated total_current_assets = {data['total_current_assets']}")
        
        if is_missing('total_non_current_assets') and total_assets and total_current_assets:
            data['total_non_current_assets'] = total_assets - total_current_assets
            calc_count += 1
            logger.debug(f"{ticker}: Calculated total_non_current_assets = {data['total_non_current_assets']}")
        
        if is_missing('total_assets') and total_current_assets and total_non_current_assets:
            data['total_assets'] = total_current_assets + total_non_current_assets
            calc_count += 1
            logger.debug(f"{ticker}: Calculated total_assets = {data['total_assets']}")
        
        # Refresh values after potential calculations
        total_assets = get_decimal(data.get('total_assets'))
        total_current_assets = get_decimal(data.get('total_current_assets'))
        total_non_current_assets = get_decimal(data.get('total_non_current_assets'))
        
        # Asset Totals (Fallback Method - component sum) - only if totals method didn't work
        if is_missing('total_current_assets') and not total_assets:
            components = [
                get_decimal(data.get('cash_and_cash_equivalents')),
                get_decimal(data.get('short_term_investments')),
                get_decimal(data.get('accounts_receivable')),
                get_decimal(data.get('inventory')),
                get_decimal(data.get('prepaid_expenses'))
            ]
            valid_components = [c for c in components if c is not None]
            if valid_components and len(valid_components) >= 2:  # Need at least 2 components
                data['total_current_assets'] = sum(valid_components)
                calc_count += 1
                logger.debug(f"{ticker}: Calculated total_current_assets from components = {data['total_current_assets']}")
        
        if is_missing('total_non_current_assets') and not total_assets:
            components = [
                get_decimal(data.get('property_plant_equipment')),
                get_decimal(data.get('goodwill')),
                get_decimal(data.get('intangible_assets')),
                get_decimal(data.get('long_term_investments')),
                get_decimal(data.get('other_assets'))
            ]
            valid_components = [c for c in components if c is not None]
            if valid_components and len(valid_components) >= 2:  # Need at least 2 components
                data['total_non_current_assets'] = sum(valid_components)
                calc_count += 1
                logger.debug(f"{ticker}: Calculated total_non_current_assets from components = {data['total_non_current_assets']}")
        
        # 9-10. Liability Totals (Preferred Method - using totals)
        if is_missing('total_current_liabilities') and total_liabilities and total_non_current_liabilities:
            data['total_current_liabilities'] = total_liabilities - total_non_current_liabilities
            calc_count += 1
            logger.debug(f"{ticker}: Calculated total_current_liabilities = {data['total_current_liabilities']}")
        
        if is_missing('total_non_current_liabilities') and total_liabilities and total_current_liabilities:
            data['total_non_current_liabilities'] = total_liabilities - total_current_liabilities
            calc_count += 1
            logger.debug(f"{ticker}: Calculated total_non_current_liabilities = {data['total_non_current_liabilities']}")
        
        if is_missing('total_liabilities') and total_current_liabilities and total_non_current_liabilities:
            data['total_liabilities'] = total_current_liabilities + total_non_current_liabilities
            calc_count += 1
            logger.debug(f"{ticker}: Calculated total_liabilities = {data['total_liabilities']}")
        
        # Refresh values after potential calculations
        total_liabilities = get_decimal(data.get('total_liabilities'))
        total_current_liabilities = get_decimal(data.get('total_current_liabilities'))
        total_non_current_liabilities = get_decimal(data.get('total_non_current_liabilities'))
        
        # Liability Totals (Fallback Method - component sum) - only if totals method didn't work
        if is_missing('total_current_liabilities') and not total_liabilities:
            components = [
                get_decimal(data.get('accounts_payable')),
                get_decimal(data.get('accrued_liabilities')),
                get_decimal(data.get('commercial_paper')),
                get_decimal(data.get('other_short_term_borrowings')),
                get_decimal(data.get('current_portion_long_term_debt')),
                get_decimal(data.get('finance_lease_liability_current')),
                get_decimal(data.get('operating_lease_liability_current'))
            ]
            valid_components = [c for c in components if c is not None]
            if valid_components and len(valid_components) >= 2:  # Need at least 2 components
                data['total_current_liabilities'] = sum(valid_components)
                calc_count += 1
                logger.debug(f"{ticker}: Calculated total_current_liabilities from components = {data['total_current_liabilities']}")
        
        if is_missing('total_non_current_liabilities') and not total_liabilities:
            components = [
                get_decimal(data.get('long_term_debt')),
                get_decimal(data.get('non_current_long_term_debt')),
                get_decimal(data.get('finance_lease_liability_noncurrent')),
                get_decimal(data.get('operating_lease_liability_noncurrent')),
                get_decimal(data.get('other_long_term_liabilities'))
            ]
            valid_components = [c for c in components if c is not None]
            if valid_components and len(valid_components) >= 2:  # Need at least 2 components
                data['total_non_current_liabilities'] = sum(valid_components)
                calc_count += 1
                logger.debug(f"{ticker}: Calculated total_non_current_liabilities from components = {data['total_non_current_liabilities']}")
        
        # 11. Equity Relationships
        total_assets = get_decimal(data.get('total_assets'))  # Refresh
        total_liabilities = get_decimal(data.get('total_liabilities'))  # Refresh
        total_stockholders_equity = get_decimal(data.get('total_stockholders_equity'))
        
        if is_missing('total_stockholders_equity') and total_assets and total_liabilities:
            data['total_stockholders_equity'] = total_assets - total_liabilities
            calc_count += 1
            logger.debug(f"{ticker}: Calculated total_stockholders_equity = {data['total_stockholders_equity']}")
        
        total_stockholders_equity = get_decimal(data.get('total_stockholders_equity'))  # Refresh
        
        if is_missing('total_liabilities_and_equity') and total_liabilities and total_stockholders_equity:
            data['total_liabilities_and_equity'] = total_liabilities + total_stockholders_equity
            calc_count += 1
            logger.debug(f"{ticker}: Calculated total_liabilities_and_equity = {data['total_liabilities_and_equity']}")
        
        # Alternative verification - total_liabilities_and_equity = total_assets
        if is_missing('total_liabilities_and_equity') and total_assets:
            data['total_liabilities_and_equity'] = total_assets
            calc_count += 1
            logger.debug(f"{ticker}: Calculated total_liabilities_and_equity from total_assets = {data['total_liabilities_and_equity']}")
        
        # 12. Debt Relationships
        long_term_debt = get_decimal(data.get('long_term_debt'))
        non_current_long_term_debt = get_decimal(data.get('non_current_long_term_debt'))
        current_portion_long_term_debt = get_decimal(data.get('current_portion_long_term_debt'))
        
        if is_missing('long_term_debt') and (non_current_long_term_debt or current_portion_long_term_debt):
            # Calculate total long-term debt as sum of non-current and current portions
            non_current_amount = non_current_long_term_debt or Decimal('0')
            current_amount = current_portion_long_term_debt or Decimal('0')
            data['long_term_debt'] = non_current_amount + current_amount
            calc_count += 1
            logger.debug(f"{ticker}: Calculated long_term_debt = {data['long_term_debt']} (non_current: {non_current_amount} + current: {current_amount})")
        
        if is_missing('non_current_long_term_debt') and long_term_debt:
            data['non_current_long_term_debt'] = long_term_debt
            calc_count += 1
            logger.debug(f"{ticker}: Calculated non_current_long_term_debt = {data['non_current_long_term_debt']}")
        
        return calc_count
    
    def _calculate_cash_flow_fields(self, data: Dict[str, Any], ticker: str) -> int:
        """
        Calculate missing cash flow fields using accounting relationships
        Returns number of fields calculated
        """
        calc_count = 0
        
        # Helper function to safely get decimal value
        def get_decimal(value) -> Optional[Decimal]:
            if value is None:
                return None
            try:
                return Decimal(str(value)) if value != 0 else Decimal('0')
            except:
                return None
        
        # Helper function to check if field is missing
        def is_missing(field_name: str) -> bool:
            return data.get(field_name) is None
        
        # 13. Depreciation & Amortization
        depreciation_and_amortization = get_decimal(data.get('depreciation_and_amortization'))
        depreciation = get_decimal(data.get('depreciation'))
        amortization = get_decimal(data.get('amortization'))
        
        if (is_missing('depreciation_and_amortization') and 
            depreciation is not None and amortization is not None):
            data['depreciation_and_amortization'] = depreciation + amortization
            calc_count += 1
            logger.debug(f"{ticker}: Calculated depreciation_and_amortization = {data['depreciation_and_amortization']}")
        
        if is_missing('depreciation') and depreciation_and_amortization and amortization is not None:
            data['depreciation'] = depreciation_and_amortization - amortization
            calc_count += 1
            logger.debug(f"{ticker}: Calculated depreciation = {data['depreciation']}")
        
        if is_missing('amortization') and depreciation_and_amortization and depreciation is not None:
            data['amortization'] = depreciation_and_amortization - depreciation
            calc_count += 1
            logger.debug(f"{ticker}: Calculated amortization = {data['amortization']}")
        
        # 14. Net Cash Change Verification
        net_cash_from_operating_activities = get_decimal(data.get('net_cash_from_operating_activities'))
        net_cash_from_investing_activities = get_decimal(data.get('net_cash_from_investing_activities'))
        net_cash_from_financing_activities = get_decimal(data.get('net_cash_from_financing_activities'))
        net_change_in_cash = get_decimal(data.get('net_change_in_cash'))
        cash_beginning_of_period = get_decimal(data.get('cash_beginning_of_period'))
        cash_end_of_period = get_decimal(data.get('cash_end_of_period'))
        
        if (is_missing('net_change_in_cash') and 
            net_cash_from_operating_activities is not None and 
            net_cash_from_investing_activities is not None and 
            net_cash_from_financing_activities is not None):
            data['net_change_in_cash'] = (net_cash_from_operating_activities + 
                                        net_cash_from_investing_activities + 
                                        net_cash_from_financing_activities)
            calc_count += 1
            logger.debug(f"{ticker}: Calculated net_change_in_cash = {data['net_change_in_cash']}")
        
        net_change_in_cash = get_decimal(data.get('net_change_in_cash'))  # Refresh
        
        if is_missing('cash_end_of_period') and cash_beginning_of_period and net_change_in_cash is not None:
            data['cash_end_of_period'] = cash_beginning_of_period + net_change_in_cash
            calc_count += 1
            logger.debug(f"{ticker}: Calculated cash_end_of_period = {data['cash_end_of_period']}")
        
        if is_missing('cash_beginning_of_period') and cash_end_of_period and net_change_in_cash is not None:
            data['cash_beginning_of_period'] = cash_end_of_period - net_change_in_cash
            calc_count += 1
            logger.debug(f"{ticker}: Calculated cash_beginning_of_period = {data['cash_beginning_of_period']}")
        
        return calc_count
    
    def _update_income_statement_in_db(self, updated_data: Dict[str, Any], original_data: Dict[str, Any]) -> bool:
        """Update income statement in database with calculated fields"""
        try:
            # Update timestamp
            updated_data['updated_at'] = datetime.utcnow()
            
            # Create IncomeStatement model instance
            statement = IncomeStatement(**updated_data)
            
            # Use upsert to update the record
            result = self.database.upsert_statement(statement)
            return result == 'updated'
            
        except Exception as e:
            logger.error(f"Error updating income statement in database: {e}")
            return False
    
    def _update_balance_sheet_in_db(self, updated_data: Dict[str, Any], original_data: Dict[str, Any]) -> bool:
        """Update balance sheet in database with calculated fields"""
        try:
            # Update timestamp
            updated_data['updated_at'] = datetime.utcnow()
            
            # Create BalanceSheet model instance
            statement = BalanceSheet(**updated_data)
            
            # Use upsert to update the record
            result = self.database.upsert_statement(statement)
            return result == 'updated'
            
        except Exception as e:
            logger.error(f"Error updating balance sheet in database: {e}")
            return False
    
    def _update_cash_flow_statement_in_db(self, updated_data: Dict[str, Any], original_data: Dict[str, Any]) -> bool:
        """Update cash flow statement in database with calculated fields"""
        try:
            # Update timestamp
            updated_data['updated_at'] = datetime.utcnow()
            
            # Create CashFlowStatement model instance
            statement = CashFlowStatement(**updated_data)
            
            # Use upsert to update the record
            result = self.database.upsert_statement(statement)
            return result == 'updated'
            
        except Exception as e:
            logger.error(f"Error updating cash flow statement in database: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.database:
            self.database.close()