"""
Simple Ratio Calculator - Efficient approach
Matches period_end_date between LTM income/cash flow and balance sheet data
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from decimal import Decimal
import json

try:
    from .virtual_fields import VirtualFieldResolver, DEFAULT_RATIOS
    from .database import FinancialDatabase
    from .ltm_calculator import LTMCalculator
    from .growth_ratio_calculator import GrowthRatioCalculator
except ImportError:
    from virtual_fields import VirtualFieldResolver, DEFAULT_RATIOS
    from database import FinancialDatabase
    from ltm_calculator import LTMCalculator
    from growth_ratio_calculator import GrowthRatioCalculator

logger = logging.getLogger(__name__)


class SimpleRatioCalculator:
    """
    Simple ratio calculator that matches periods and calculates ratios efficiently
    """
    
    def __init__(self):
        self.virtual_resolver = VirtualFieldResolver()
        self.database = None
        self.ltm_calculator = None
    
    def __enter__(self):
        self.database = FinancialDatabase()
        self.ltm_calculator = LTMCalculator()
        self.ltm_calculator.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ltm_calculator:
            self.ltm_calculator.__exit__(exc_type, exc_val, exc_tb)
        if self.database:
            self.database.close()
    
    def calculate_company_ratios(self, ticker: str) -> Dict[str, Any]:
        """
        Simple approach: Match periods and calculate ratios
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Dictionary with calculation results
        """
        logger.info(f"Calculating ratios for {ticker} using simple approach")
        
        try:
            # Step 1: Get company info
            company_info = self.database.get_company_by_ticker(ticker)
            if not company_info:
                return {'error': f'Company {ticker} not found'}
            
            company_id = company_info['id']
            
            # Step 2: Get ratio definitions (ensure they exist)
            ratio_definitions = self._get_ratio_definitions(company_id)
            if not ratio_definitions:
                return {'error': f'No ratio definitions found. Initialize default ratios first.'}
            
            # Step 3: Get all financial data with period matching
            financial_data_by_period = self._get_matched_financial_data(ticker)
            if not financial_data_by_period:
                return {'error': f'No financial data found for {ticker}'}
            
            # Step 4: Calculate ratios for each period
            all_calculated_ratios = []
            calculation_summary = {}
            
            for period_end_date, combined_data in financial_data_by_period.items():
                logger.info(f"Processing period {period_end_date} for {ticker}")
                
                # Get fiscal info from the combined data
                fiscal_year = combined_data.get('fiscal_year')
                
                # Get fiscal_quarter from balance sheet data for this period_end_date
                fiscal_quarter = self._get_fiscal_quarter_from_balance_sheet(ticker, period_end_date)
                
                if not fiscal_year:
                    logger.warning(f"Missing fiscal_year for period {period_end_date}, skipping")
                    continue
                
                period_ratios = []
                
                for ratio_def in ratio_definitions:
                    ratio_name = ratio_def['name']
                    formula = ratio_def['formula']
                    
                    # Skip growth ratios (they are handled separately)
                    if 'YOY_GROWTH(' in formula:
                        logger.debug(f"Skipping growth ratio {ratio_name} - handled by GrowthRatioCalculator")
                        continue
                    
                    try:
                        # Calculate single ratio
                        ratio_value = self._calculate_ratio(formula, combined_data)
                        
                        if ratio_value is not None:
                            # Extract inputs used in calculation
                            inputs = self._extract_formula_inputs(formula, combined_data)
                            
                            ratio_record = {
                                'company_id': company_id,
                                'ratio_definition_id': ratio_def['id'],
                                'ticker': ticker,
                                'ratio_name': ratio_name,
                                'ratio_category': ratio_def.get('category'),
                                'period_end_date': period_end_date,
                                'period_type': combined_data.get('period_type', 'LTM'),
                                'fiscal_year': fiscal_year,
                                'fiscal_quarter': fiscal_quarter,
                                'ratio_value': round(float(ratio_value), 6),
                                'calculation_inputs': inputs,
                                'data_source': combined_data.get('data_source', 'LTM')
                            }
                            
                            period_ratios.append(ratio_record)
                            
                            # Track for summary (latest period only)
                            if len(calculation_summary) == 0:
                                calculation_summary[ratio_name] = {
                                    'value': ratio_record['ratio_value'],
                                    'period_end_date': str(period_end_date),
                                    'inputs': inputs
                                }
                        else:
                            # Track failed calculations for summary
                            if len(calculation_summary) == 0:
                                calculation_summary[ratio_name] = {
                                    'error': 'Could not calculate - missing data',
                                    'period_end_date': str(period_end_date)
                                }
                    
                    except Exception as e:
                        logger.warning(f"Error calculating {ratio_name} for {ticker} period {period_end_date}: {e}")
                        if len(calculation_summary) == 0:
                            calculation_summary[ratio_name] = {
                                'error': str(e),
                                'period_end_date': str(period_end_date)
                            }
                
                all_calculated_ratios.extend(period_ratios)
                logger.info(f"Calculated {len(period_ratios)} ratios for {ticker} period {period_end_date}")
            
            # Step 5: Store in database
            if all_calculated_ratios:
                self._store_calculated_ratios(all_calculated_ratios)
                logger.info(f"Stored {len(all_calculated_ratios)} ratios across {len(financial_data_by_period)} periods")
            
            # Step 6: Calculate growth ratios (YoY)
            growth_result = None
            try:
                with GrowthRatioCalculator() as growth_calc:
                    growth_result = growth_calc.calculate_yoy_growth_ratios(ticker)
                    if growth_result.get('status') == 'success':
                        logger.info(f"Calculated {growth_result.get('ratios_calculated', 0)} growth ratios for {ticker}")
                    else:
                        logger.warning(f"Growth ratio calculation issue: {growth_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"Error calculating growth ratios for {ticker}: {e}")
                growth_result = {'error': str(e)}
            
            return {
                'ticker': ticker,
                'company_id': str(company_id),
                'calculation_date': datetime.now().isoformat(),
                'periods_processed': len(financial_data_by_period),
                'ratios': calculation_summary,
                'total_ratios': len(calculation_summary),
                'total_calculations': len(all_calculated_ratios),
                'growth_ratios': growth_result
            }
            
        except Exception as e:
            logger.error(f"Error calculating ratios for {ticker}: {e}")
            return {'error': str(e)}
    
    def _get_matched_financial_data(self, ticker: str) -> Dict[date, Dict]:
        """
        Match period_end_date between LTM/FY income/cash flow and balance sheet
        
        Returns:
            Dictionary mapping period_end_date to combined financial data
        """
        try:
            # Get LTM data
            ltm_income = self.ltm_calculator.calculate_ltm_for_all_quarters(ticker, 'income_statement')
            ltm_cash_flow = self.ltm_calculator.calculate_ltm_for_all_quarters(ticker, 'cash_flow')
            
            # Get FY data directly from database
            fy_income = self.database.get_income_statements(ticker)
            fy_cash_flow = self.database.get_cash_flow_statements(ticker)
            
            # Filter FY data to only full year periods (12 months)
            fy_income = [record for record in fy_income 
                        if record.get('period_length_months') == 12 or record.get('period_type') == 'FY']
            fy_cash_flow = [record for record in fy_cash_flow 
                           if record.get('period_length_months') == 12 or record.get('period_type') == 'FY']
            
            # Get balance sheet data
            balance_sheets = self.database.get_balance_sheets(ticker)
            
            # Create mappings by period_end_date for LTM data
            ltm_income_by_date = {}
            for record in ltm_income:
                period_date = self._parse_date(record.get('period_end_date'))
                if period_date:
                    record['data_source'] = 'LTM'
                    record['period_type'] = 'LTM'
                    ltm_income_by_date[period_date] = record
            
            ltm_cf_by_date = {}
            for record in ltm_cash_flow:
                period_date = self._parse_date(record.get('period_end_date'))
                if period_date:
                    record['data_source'] = 'LTM'
                    ltm_cf_by_date[period_date] = record
            
            # Create mappings by period_end_date for FY data
            fy_income_by_date = {}
            for record in fy_income:
                period_date = self._parse_date(record.get('period_end_date'))
                if period_date:
                    record['data_source'] = 'FY'
                    fy_income_by_date[period_date] = record
            
            fy_cf_by_date = {}
            for record in fy_cash_flow:
                period_date = self._parse_date(record.get('period_end_date'))
                if period_date:
                    record['data_source'] = 'FY'
                    fy_cf_by_date[period_date] = record
            
            # Create balance sheet mapping
            bs_by_date = {}
            for record in balance_sheets:
                period_date = self._parse_date(record.get('period_end_date'))
                if period_date:
                    bs_by_date[period_date] = record
            
            # Match periods and combine data
            matched_data = {}
            
            # Process LTM periods
            for period_date, income_data in ltm_income_by_date.items():
                combined_data = self._combine_financial_data(
                    income_data, ltm_cf_by_date.get(period_date), bs_by_date, period_date, 'LTM'
                )
                if combined_data:
                    matched_data[period_date] = combined_data
            
            # Process FY periods (avoid duplicates with LTM)
            for period_date, income_data in fy_income_by_date.items():
                # Only add FY data if we don't already have LTM data for this date
                if period_date not in matched_data:
                    combined_data = self._combine_financial_data(
                        income_data, fy_cf_by_date.get(period_date), bs_by_date, period_date, 'FY'
                    )
                    if combined_data:
                        matched_data[period_date] = combined_data
            
            logger.info(f"Matched {len(matched_data)} periods for {ticker} (LTM + FY)")
            return matched_data
            
        except Exception as e:
            logger.error(f"Error matching financial data for {ticker}: {e}")
            return {}
    
    def _combine_financial_data(self, income_data: Dict, cf_data: Optional[Dict], 
                               bs_by_date: Dict, period_date: date, data_source: str) -> Optional[Dict]:
        """
        Combine income statement, cash flow, and balance sheet data for a period
        
        Args:
            income_data: Income statement data
            cf_data: Cash flow data (optional)
            bs_by_date: Balance sheet data mapping
            period_date: Period end date
            data_source: 'LTM' or 'FY'
            
        Returns:
            Combined financial data dictionary
        """
        try:
            combined_data = income_data.copy()
            combined_data['data_source'] = data_source
            
            # Add cash flow data for same period
            if cf_data:
                # Add cash flow fields (avoid duplicating metadata)
                for key, value in cf_data.items():
                    if key not in ['id', 'company_id', 'ticker', 'cik', 'company_name', 
                                  'created_at', 'updated_at', 'period_end_date', 
                                  'fiscal_year', 'period_type', 'data_source']:
                        combined_data[key] = value
            
            # Find closest balance sheet (within 90 days)
            closest_bs = self._find_closest_balance_sheet(period_date, bs_by_date)
            if closest_bs:
                # Add balance sheet fields (avoid duplicating metadata)
                for key, value in closest_bs.items():
                    if key not in ['id', 'company_id', 'ticker', 'cik', 'company_name', 
                                  'created_at', 'updated_at']:
                        # Use balance sheet date fields if income statement doesn't have them
                        if key in ['period_end_date', 'fiscal_year', 'period_type'] and combined_data.get(key) is None:
                            combined_data[key] = value
                        elif key not in ['period_end_date', 'fiscal_year', 'period_type']:
                            combined_data[key] = value
            
            # Resolve virtual fields
            combined_data = self.virtual_resolver.resolve_virtual_fields(combined_data)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error combining financial data for {period_date}: {e}")
            return None
    
    def _parse_date(self, date_value) -> Optional[date]:
        """Parse date from various formats"""
        if isinstance(date_value, date):
            return date_value
        elif isinstance(date_value, datetime):
            return date_value.date()
        elif isinstance(date_value, str):
            try:
                return datetime.strptime(date_value, '%Y-%m-%d').date()
            except:
                return None
        return None
    
    def _find_closest_balance_sheet(self, target_date: date, bs_by_date: Dict[date, Dict]) -> Optional[Dict]:
        """Find balance sheet closest to target date (within 90 days)"""
        closest_bs = None
        min_diff = float('inf')
        
        for bs_date, bs_data in bs_by_date.items():
            diff = abs((bs_date - target_date).days)
            if diff < min_diff and diff <= 90:
                min_diff = diff
                closest_bs = bs_data
        
        return closest_bs
    
    def _get_fiscal_quarter_from_balance_sheet(self, ticker: str, period_end_date: date) -> Optional[str]:
        """Get fiscal quarter from balance sheet table for the given period_end_date"""
        try:
            with self.database.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT period_type 
                    FROM balance_sheets 
                    WHERE ticker = %s AND period_end_date = %s
                    LIMIT 1
                """, (ticker.upper(), period_end_date))
                
                result = cursor.fetchone()
                if result and result[0] in ['Q1', 'Q2', 'Q3', 'Q4']:
                    return result[0]
                return None
                
        except Exception as e:
            logger.debug(f"Error getting fiscal quarter from balance sheet: {e}")
            return None
    
    def _calculate_ratio(self, formula: str, data: Dict) -> Optional[float]:
        """Calculate a single ratio using virtual field resolver"""
        try:
            return self.virtual_resolver._evaluate_expression(formula, data)
        except Exception as e:
            logger.debug(f"Error calculating ratio with formula '{formula}': {e}")
            return None
    
    def _extract_formula_inputs(self, formula: str, data: Dict) -> Dict:
        """Extract input values used in formula"""
        import re
        
        # Find field names in formula
        field_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        fields = re.findall(field_pattern, formula)
        
        # Filter out operators and functions
        operators = {'and', 'or', 'not', 'if', 'else', 'COALESCE', 'NULL'}
        actual_fields = [f for f in fields if f not in operators and not f.isdigit()]
        
        # Extract values
        inputs = {}
        for field in actual_fields:
            if field in data:
                value = data[field]
                if isinstance(value, Decimal):
                    inputs[field] = float(value)
                else:
                    inputs[field] = value
        
        return inputs
    
    def _get_ratio_definitions(self, company_id) -> List[Dict]:
        """Get ratio definitions for company"""
        try:
            with self.database.connection.cursor() as cursor:
                cursor.execute("""
                    WITH company_ratios AS (
                        SELECT *, 1 as priority FROM ratio_definitions 
                        WHERE company_id = %s AND is_active = true
                    ),
                    global_ratios AS (
                        SELECT *, 2 as priority FROM ratio_definitions 
                        WHERE company_id IS NULL AND is_active = true
                        AND name NOT IN (SELECT name FROM company_ratios)
                    )
                    SELECT * FROM company_ratios
                    UNION ALL
                    SELECT * FROM global_ratios
                    ORDER BY priority, category, name
                """, (company_id,))
                
                ratios = cursor.fetchall()
                if ratios:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, ratio)) for ratio in ratios]
                return []
                    
        except Exception as e:
            logger.error(f"Error getting ratio definitions: {e}")
            return []
    
    def _store_calculated_ratios(self, calculated_ratios: List[Dict]):
        """Store calculated ratios in database with new schema"""
        try:
            with self.database.connection.cursor() as cursor:
                for ratio in calculated_ratios:
                    cursor.execute("""
                        INSERT INTO calculated_ratios (
                            company_id, ratio_definition_id, ticker, ratio_name, ratio_category,
                            period_end_date, period_type, fiscal_year, fiscal_quarter,
                            ratio_value, calculation_inputs, data_source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (company_id, ratio_definition_id, period_end_date, period_type, data_source)
                        DO UPDATE SET
                            ratio_name = EXCLUDED.ratio_name,
                            ratio_category = EXCLUDED.ratio_category,
                            fiscal_year = EXCLUDED.fiscal_year,
                            fiscal_quarter = EXCLUDED.fiscal_quarter,
                            ratio_value = EXCLUDED.ratio_value,
                            calculation_inputs = EXCLUDED.calculation_inputs,
                            calculation_date = CURRENT_TIMESTAMP
                    """, (
                        ratio['company_id'], ratio['ratio_definition_id'], ratio['ticker'],
                        ratio['ratio_name'], ratio.get('ratio_category'), ratio['period_end_date'], 
                        ratio['period_type'], ratio['fiscal_year'], ratio['fiscal_quarter'], 
                        ratio['ratio_value'], json.dumps(ratio['calculation_inputs']) if ratio['calculation_inputs'] else None,
                        ratio['data_source']
                    ))
                
                self.database.connection.commit()
                logger.info(f"Stored {len(calculated_ratios)} calculated ratios")
                
        except Exception as e:
            self.database.connection.rollback()
            logger.error(f"Error storing calculated ratios: {e}")
            raise


# Convenience functions
def calculate_ratios_simple(ticker: str) -> Dict[str, Any]:
    """Calculate ratios using simple approach"""
    with SimpleRatioCalculator() as calculator:
        return calculator.calculate_company_ratios(ticker)

def get_stored_ratios_simple(ticker: str, category: Optional[str] = None) -> List[Dict]:
    """Get stored ratios for a company using simple approach"""
    try:
        with FinancialDatabase() as db:
            return db.get_calculated_ratios(ticker, ratio_name=None, category=category)
    except Exception as e:
        logger.error(f"Error getting stored ratios for {ticker}: {e}")
        return []