"""
Ratio Calculator Engine
Calculates financial ratios using LTM data with virtual field resolution and hybrid definitions
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from decimal import Decimal
import uuid

try:
    from .virtual_fields import VirtualFieldResolver, DEFAULT_RATIOS
    from .database import FinancialDatabase
    from .ltm_calculator import LTMCalculator
except ImportError:
    from virtual_fields import VirtualFieldResolver, DEFAULT_RATIOS
    from database import FinancialDatabase
    from ltm_calculator import LTMCalculator

logger = logging.getLogger(__name__)


class RatioCalculator:
    """
    Financial ratio calculator with LTM integration and virtual field support
    """
    
    def __init__(self):
        self.virtual_resolver = VirtualFieldResolver()
        self.ltm_calculator = None
        self.database = None
    
    def __enter__(self):
        self.ltm_calculator = LTMCalculator()
        self.ltm_calculator.__enter__()
        self.database = FinancialDatabase()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ltm_calculator:
            self.ltm_calculator.__exit__(exc_type, exc_val, exc_tb)
        if self.database:
            self.database.close()
    
    def calculate_company_ratios(self, ticker: str, as_of_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Calculate all ratios for a company using most recent LTM data
        
        Args:
            ticker: Company ticker symbol
            as_of_date: Calculate ratios as of this date (uses latest if None)
            
        Returns:
            Dictionary with calculated ratios and metadata
        """
        logger.info(f"Calculating ratios for {ticker}")
        
        try:
            # Get company information
            company_info = self.database.get_company_by_ticker(ticker)
            if not company_info:
                logger.error(f"Company {ticker} not found in database")
                return {'error': f'Company {ticker} not found'}
            
            company_id = company_info['id']
            
            # Get LTM financial data
            ltm_data = self._get_ltm_financial_data(ticker, as_of_date)
            if not ltm_data:
                logger.warning(f"No LTM data available for {ticker}")
                return {'error': f'No LTM data available for {ticker}'}
            
            # Get ratio definitions for this company (hybrid: company-specific + global)
            ratio_definitions = self._get_ratio_definitions(company_id)
            
            # Calculate ratios
            calculated_ratios = []
            calculation_results = {}
            
            for ratio_def in ratio_definitions:
                try:
                    ratio_result = self._calculate_single_ratio(
                        ratio_def, ltm_data, company_id, ticker
                    )
                    if ratio_result:
                        calculated_ratios.append(ratio_result)
                        calculation_results[ratio_def['name']] = {
                            'value': ratio_result['ratio_value'],
                            'description': ratio_def.get('description'),
                            'category': ratio_def.get('category'),
                            'inputs': ratio_result.get('calculation_inputs', {})
                        }
                except Exception as e:
                    logger.error(f"Error calculating ratio {ratio_def['name']} for {ticker}: {e}")
                    calculation_results[ratio_def['name']] = {'error': str(e)}
            
            # Store calculated ratios in database
            if calculated_ratios:
                self._store_calculated_ratios(calculated_ratios)
            
            return {
                'ticker': ticker,
                'company_id': str(company_id),
                'calculation_date': datetime.now().isoformat(),
                'ltm_period': ltm_data.get('period_info', {}),
                'ratios': calculation_results,
                'total_ratios': len(calculation_results)
            }
            
        except Exception as e:
            logger.error(f"Error calculating ratios for {ticker}: {e}")
            return {'error': str(e)}
    
    def _get_ltm_financial_data(self, ticker: str, as_of_date: Optional[date] = None) -> Optional[Dict]:
        """
        Get LTM financial data combining income statement, balance sheet, and cash flow
        
        Args:
            ticker: Company ticker
            as_of_date: As of date for LTM calculation
            
        Returns:
            Combined LTM financial data dictionary
        """
        try:
            # Get LTM data for income statement and cash flow (these use LTM)
            ltm_income = self.ltm_calculator.get_latest_ltm_data(ticker, 'income_statement')
            ltm_cash_flow = self.ltm_calculator.get_latest_ltm_data(ticker, 'cash_flow')
            
            # Get latest balance sheet data (balance sheet is point-in-time, not LTM)
            balance_sheets = self.database.get_balance_sheets(ticker, limit=1)
            latest_balance_sheet = balance_sheets[0] if balance_sheets else {}
            
            if not ltm_income and not ltm_cash_flow and not latest_balance_sheet:
                return None
            
            # Combine all data sources
            combined_data = {}
            period_info = {}
            
            # Add LTM income statement data
            if ltm_income:
                combined_data.update(ltm_income)
                period_info['income_statement_ltm_period'] = ltm_income.get('ltm_period_end')
                logger.debug(f"Added LTM income statement data for {ticker}")
            
            # Add LTM cash flow data
            if ltm_cash_flow:
                combined_data.update(ltm_cash_flow)
                period_info['cash_flow_ltm_period'] = ltm_cash_flow.get('ltm_period_end')
                logger.debug(f"Added LTM cash flow data for {ticker}")
            
            # Add latest balance sheet data (point-in-time)
            if latest_balance_sheet:
                # Remove metadata fields that might conflict
                bs_data = {k: v for k, v in latest_balance_sheet.items() 
                          if k not in ['id', 'company_id', 'created_at', 'updated_at']}
                combined_data.update(bs_data)
                period_info['balance_sheet_date'] = latest_balance_sheet.get('period_end_date')
                logger.debug(f"Added balance sheet data for {ticker}")
            
            # Resolve virtual fields
            combined_data = self.virtual_resolver.resolve_virtual_fields(combined_data)
            combined_data['period_info'] = period_info
            
            logger.info(f"Retrieved combined financial data for {ticker}: "
                       f"{len(combined_data)} fields including virtual fields")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error getting LTM financial data for {ticker}: {e}")
            return None
    
    def _get_ratio_definitions(self, company_id: uuid.UUID) -> List[Dict]:
        """
        Get ratio definitions for a company (hybrid: company-specific + global)
        
        Args:
            company_id: Company UUID
            
        Returns:
            List of ratio definition dictionaries
        """
        try:
            with self.database.connection.cursor() as cursor:
                # Get both company-specific and global ratios
                # Company-specific ratios override global ones with same name
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
                
                # Convert to list of dictionaries
                if ratios:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, ratio)) for ratio in ratios]
                else:
                    # If no ratios defined, return empty list
                    # User can initialize default ratios separately
                    logger.info(f"No ratio definitions found for company {company_id}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting ratio definitions for company {company_id}: {e}")
            return []
    
    def _calculate_single_ratio(self, ratio_def: Dict, financial_data: Dict, 
                              company_id: uuid.UUID, ticker: str) -> Optional[Dict]:
        """
        Calculate a single ratio
        
        Args:
            ratio_def: Ratio definition dictionary
            financial_data: Financial data with virtual fields resolved
            company_id: Company UUID
            ticker: Company ticker
            
        Returns:
            Calculated ratio result dictionary
        """
        try:
            formula = ratio_def['formula']
            logger.debug(f"Calculating {ratio_def['name']} for {ticker}: {formula}")
            
            # Extract the inputs that will be used in calculation
            calculation_inputs = self._extract_formula_inputs(formula, financial_data)
            
            # Evaluate the formula
            ratio_value = self._evaluate_formula(formula, financial_data)
            
            if ratio_value is None:
                logger.warning(f"Could not calculate {ratio_def['name']} for {ticker} - missing data")
                return None
            
            # Get period information
            period_info = financial_data.get('period_info', {})
            
            return {
                'company_id': company_id,
                'ratio_definition_id': ratio_def['id'],
                'ticker': ticker,
                'period_end_date': period_info.get('income_statement_ltm_period') or 
                                 period_info.get('balance_sheet_date') or date.today(),
                'period_type': 'LTM',
                'fiscal_year': None,  # LTM spans multiple years
                'ratio_value': round(float(ratio_value), 6),
                'calculation_inputs': calculation_inputs,
                'data_source': 'LTM'
            }
            
        except Exception as e:
            logger.error(f"Error calculating ratio {ratio_def['name']}: {e}")
            return None
    
    def _extract_formula_inputs(self, formula: str, data: Dict) -> Dict:
        """Extract the actual values used in formula calculation"""
        import re
        
        # Find all field names in the formula
        field_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        fields = re.findall(field_pattern, formula)
        
        # Filter out mathematical operators and functions
        operators = {'and', 'or', 'not', 'if', 'else', 'COALESCE', 'NULL'}
        actual_fields = [f for f in fields if f not in operators and not f.isdigit()]
        
        # Extract values for these fields
        inputs = {}
        for field in actual_fields:
            if field in data:
                inputs[field] = data[field]
        
        return inputs
    
    def _evaluate_formula(self, formula: str, data: Dict) -> Optional[float]:
        """
        Safely evaluate a ratio formula
        
        Args:
            formula: Mathematical formula string
            data: Financial data dictionary
            
        Returns:
            Calculated ratio value or None if cannot calculate
        """
        try:
            # Use the virtual field resolver's evaluation method
            return self.virtual_resolver._evaluate_expression(formula, data)
        except Exception as e:
            logger.debug(f"Error evaluating formula '{formula}': {e}")
            return None
    
    def _store_calculated_ratios(self, calculated_ratios: List[Dict]):
        """Store calculated ratios in database"""
        try:
            with self.database.connection.cursor() as cursor:
                for ratio in calculated_ratios:
                    cursor.execute("""
                        INSERT INTO calculated_ratios (
                            company_id, ratio_definition_id, ticker, period_end_date,
                            period_type, fiscal_year, ratio_value, calculation_inputs, data_source
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (company_id, ratio_definition_id, period_end_date, period_type, data_source)
                        DO UPDATE SET
                            ratio_value = EXCLUDED.ratio_value,
                            calculation_inputs = EXCLUDED.calculation_inputs,
                            calculation_date = CURRENT_TIMESTAMP
                    """, (
                        ratio['company_id'], ratio['ratio_definition_id'], ratio['ticker'],
                        ratio['period_end_date'], ratio['period_type'], ratio['fiscal_year'],
                        ratio['ratio_value'], 
                        __import__('json').dumps(ratio['calculation_inputs']) if ratio['calculation_inputs'] else None,
                        ratio['data_source']
                    ))
                
                self.database.connection.commit()
                logger.info(f"Stored {len(calculated_ratios)} calculated ratios in database")
                
        except Exception as e:
            self.database.connection.rollback()
            logger.error(f"Error storing calculated ratios: {e}")
            raise
    
    def get_company_ratios(self, ticker: str, category: Optional[str] = None) -> List[Dict]:
        """
        Get stored ratios for a company
        
        Args:
            ticker: Company ticker
            category: Filter by ratio category (optional)
            
        Returns:
            List of stored ratio dictionaries
        """
        try:
            with self.database.connection.cursor() as cursor:
                query = """
                    SELECT 
                        cr.ticker, cr.period_end_date, cr.period_type, cr.ratio_value,
                        cr.calculation_inputs, cr.data_source, cr.calculation_date,
                        rd.name, rd.description, rd.category, rd.formula
                    FROM calculated_ratios cr
                    JOIN ratio_definitions rd ON cr.ratio_definition_id = rd.id
                    WHERE cr.ticker = %s
                """
                params = [ticker.upper()]
                
                if category:
                    query += " AND rd.category = %s"
                    params.append(category)
                
                query += " ORDER BY rd.category, rd.name, cr.period_end_date DESC"
                
                cursor.execute(query, params)
                ratios = cursor.fetchall()
                
                if ratios:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, ratio)) for ratio in ratios]
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting ratios for {ticker}: {e}")
            return []


# Convenience functions for easy access
def calculate_ratios_for_company(ticker: str, as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """Calculate ratios for a company - convenience function"""
    with RatioCalculator() as calculator:
        return calculator.calculate_company_ratios(ticker, as_of_date)

def get_stored_ratios(ticker: str, category: Optional[str] = None) -> List[Dict]:
    """Get stored ratios for a company - convenience function"""
    with RatioCalculator() as calculator:
        return calculator.get_company_ratios(ticker, category)