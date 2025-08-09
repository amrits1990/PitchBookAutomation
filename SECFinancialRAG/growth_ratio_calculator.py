"""
Growth Ratio Calculator
Calculates Year-over-Year growth ratios by accessing previous year data
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from decimal import Decimal
import json

try:
    from .database import FinancialDatabase
    from .ltm_calculator import LTMCalculator
    from .models import CalculatedRatio
except ImportError:
    from database import FinancialDatabase
    from ltm_calculator import LTMCalculator
    from models import CalculatedRatio

logger = logging.getLogger(__name__)


class GrowthRatioCalculator:
    """
    Calculator for Year-over-Year growth ratios
    """
    
    def __init__(self):
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
    
    def calculate_yoy_growth_ratios(self, ticker: str) -> Dict[str, Any]:
        """
        Calculate YoY growth ratios for a company using simple period_end_date comparison
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            Dictionary with calculation results
        """
        logger.info(f"Calculating YoY growth ratios for {ticker} using simplified approach")
        
        try:
            # Get company info
            company_info = self.database.get_company_by_ticker(ticker)
            if not company_info:
                return {'error': f'Company {ticker} not found'}
            
            company_id = company_info['id']
            
            # Get all financial periods (both LTM and FY) with revenue/EBIT data
            all_periods = self._get_all_financial_periods(ticker)
            if len(all_periods) < 2:
                return {'error': f'Need at least 2 periods to calculate growth ratios for {ticker}'}
            
            logger.info(f"Found {len(all_periods)} financial periods for {ticker}")
            
            # Debug: Show breakdown by data source
            ltm_count = len([p for p in all_periods if p.get('data_source') == 'LTM'])
            fy_count = len([p for p in all_periods if p.get('data_source') == 'FY'])
            logger.info(f"  - LTM periods: {ltm_count}")
            logger.info(f"  - FY periods: {fy_count}")
            
            # Calculate growth ratios for all periods
            all_growth_ratios = []
            
            for current_period in all_periods:
                current_date = current_period['period_end_date']
                current_source = current_period.get('data_source', 'Unknown')
                
                # Find previous period ~12 months earlier
                previous_period = self._find_previous_year_period(current_period, all_periods)
                if not previous_period:
                    logger.debug(f"No previous period found for {current_date} ({current_source})")
                    continue
                
                previous_date = previous_period['period_end_date']
                previous_source = previous_period.get('data_source', 'Unknown')
                days_apart = (current_date - previous_date).days
                
                logger.debug(f"Processing growth for {current_date} ({current_source}) vs {previous_date} ({previous_source}) - {days_apart} days apart")
                
                # Calculate growth ratios for this period pair
                growth_ratios = self._calculate_simple_growth_ratios(
                    ticker, str(company_id), current_period, previous_period
                )
                
                logger.debug(f"Generated {len(growth_ratios)} growth ratios for {current_date} ({current_source})")
                all_growth_ratios.extend(growth_ratios)
            
            # Store growth ratios in database
            stored_count = 0
            ltm_ratios = 0
            fy_ratios = 0
            
            for ratio in all_growth_ratios:
                if ratio.get('data_source') == 'LTM':
                    ltm_ratios += 1
                elif ratio.get('data_source') == 'FY':
                    fy_ratios += 1
                    
                if self._store_growth_ratio(ratio):
                    stored_count += 1
                    logger.debug(f"Stored {ratio['name']} for {ratio['period_end_date']} ({ratio.get('data_source')})")
                else:
                    logger.warning(f"Failed to store {ratio['name']} for {ratio['period_end_date']} ({ratio.get('data_source')})")
            
            logger.info(f"Growth ratio breakdown: {ltm_ratios} LTM, {fy_ratios} FY ratios calculated")
            
            logger.info(f"Calculated and stored {stored_count} growth ratios for {ticker}")
            
            return {
                'status': 'success',
                'ticker': ticker,
                'ratios_calculated': len(all_growth_ratios),
                'ratios_stored': stored_count,
                'periods_processed': len(all_periods)
            }
            
        except Exception as e:
            logger.error(f"Error calculating growth ratios for {ticker}: {e}")
            return {'error': str(e)}
    
    def _calculate_growth_ratios_simple(self, ticker: str, company_id: str) -> Dict[str, Any]:
        """
        Calculate growth ratios using simple annual data approach
        
        Args:
            ticker: Company ticker symbol
            company_id: Company ID
            
        Returns:
            Dictionary with calculation results
        """
        logger.info(f"Trying simple growth ratio calculation for {ticker}")
        
        try:
            # Get annual income statements (FY periods only)
            income_statements = self.database.get_income_statements(ticker)
            annual_statements = [stmt for stmt in income_statements 
                               if stmt.get('period_type') == 'FY' and stmt.get('fiscal_year')]
            
            if len(annual_statements) < 2:
                logger.info(f"Not enough annual data for {ticker} (need at least 2 years)")
                return {'ratios_calculated': 0}
            
            # Sort by fiscal year
            annual_statements.sort(key=lambda x: x.get('fiscal_year'))
            
            all_growth_ratios = []
            
            # Calculate year-over-year growth for consecutive years
            for i in range(1, len(annual_statements)):
                current_stmt = annual_statements[i]
                previous_stmt = annual_statements[i-1]
                
                current_fy = current_stmt.get('fiscal_year')
                previous_fy = previous_stmt.get('fiscal_year')
                
                # Only calculate if consecutive years
                if current_fy != previous_fy + 1:
                    continue
                
                period_end_date = current_stmt.get('period_end_date')
                
                # Calculate Revenue Growth
                current_revenue = current_stmt.get('total_revenue')
                previous_revenue = previous_stmt.get('total_revenue')
                
                if current_revenue and previous_revenue and previous_revenue != 0:
                    revenue_growth = (current_revenue - previous_revenue) / previous_revenue
                    
                    revenue_ratio = {
                        'company_id': company_id,
                        'ticker': ticker,
                        'name': 'Revenue_Growth_YoY',
                        'ratio_value': revenue_growth,
                        'description': 'Revenue Growth YoY - Year-over-year growth in total revenue',
                        'category': 'growth',
                        'period_end_date': period_end_date,
                        'fiscal_year': current_fy,
                        'fiscal_quarter': 'FY',
                        'calculation_inputs': {
                            'current_revenue': float(current_revenue) if current_revenue else None,
                            'previous_revenue': float(previous_revenue) if previous_revenue else None,
                            'current_fy': current_fy,
                            'previous_fy': previous_fy
                        },
                        'calculation_date': datetime.utcnow()
                    }
                    
                    all_growth_ratios.append(revenue_ratio)
                    logger.debug(f"Calculated Revenue Growth for FY{current_fy}: {revenue_growth:.4f}")
                
                # Calculate EBIT Growth (as proxy for EBITDA)
                current_ebit = current_stmt.get('ebit') or current_stmt.get('operating_income')
                previous_ebit = previous_stmt.get('ebit') or previous_stmt.get('operating_income')
                
                if current_ebit and previous_ebit and previous_ebit != 0:
                    ebit_growth = (current_ebit - previous_ebit) / previous_ebit
                    
                    ebit_ratio = {
                        'company_id': company_id,
                        'ticker': ticker,
                        'name': 'EBIT_Growth_YoY',  # Using EBIT as proxy for EBITDA
                        'ratio_value': ebit_growth,
                        'description': 'EBIT Growth YoY - Year-over-year growth in EBIT (Operating Income)',
                        'category': 'growth',
                        'period_end_date': period_end_date,
                        'fiscal_year': current_fy,
                        'fiscal_quarter': 'FY',
                        'calculation_inputs': {
                            'current_ebit': float(current_ebit) if current_ebit else None,
                            'previous_ebit': float(previous_ebit) if previous_ebit else None,
                            'current_fy': current_fy,
                            'previous_fy': previous_fy
                        },
                        'calculation_date': datetime.utcnow()
                    }
                    
                    all_growth_ratios.append(ebit_ratio)
                    logger.debug(f"Calculated EBIT Growth for FY{current_fy}: {ebit_growth:.4f}")
            
            # Store growth ratios
            stored_count = 0
            for ratio in all_growth_ratios:
                if self._store_growth_ratio(ratio):
                    stored_count += 1
            
            logger.info(f"Simple approach: calculated {len(all_growth_ratios)} growth ratios, stored {stored_count}")
            
            return {
                'status': 'success',
                'ticker': ticker,
                'ratios_calculated': len(all_growth_ratios),
                'ratios_stored': stored_count,
                'approach': 'simple_annual'
            }
            
        except Exception as e:
            logger.error(f"Error in simple growth calculation for {ticker}: {e}")
            return {'ratios_calculated': 0, 'error': str(e)}
    
    def _get_all_financial_periods(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get all financial periods (LTM + FY) with revenue and EBIT data
        
        Returns:
            List of financial periods sorted by period_end_date (most recent first)
        """
        try:
            all_periods = []
            
            # Get LTM income statement data
            if self.ltm_calculator:
                ltm_income_data = self.ltm_calculator.calculate_ltm_for_all_quarters(ticker, 'income_statement')
                for ltm_data in ltm_income_data:
                    period_end = ltm_data.get('period_end_date')  # Use period_end_date, not ltm_period_end
                    revenue = ltm_data.get('total_revenue')
                    ebit = ltm_data.get('ebit') or ltm_data.get('operating_income')
                    
                    if period_end and (revenue is not None or ebit is not None):
                        # Convert string date to date object if needed
                        if isinstance(period_end, str):
                            period_end = datetime.strptime(period_end, '%Y-%m-%d').date()
                        
                        all_periods.append({
                            'period_end_date': period_end,
                            'data_source': 'LTM',
                            'period_type': ltm_data.get('fiscal_quarter', 'LTM'),
                            'fiscal_year': ltm_data.get('fiscal_year'),
                            'total_revenue': revenue,
                            'ebit': ebit,
                            'original_data': ltm_data
                        })
            
            # Get FY income statement data
            income_statements = self.database.get_income_statements(ticker)
            for stmt in income_statements:
                if stmt.get('period_type') == 'FY':
                    period_end = stmt.get('period_end_date')
                    revenue = stmt.get('total_revenue')
                    ebit = stmt.get('ebit') or stmt.get('operating_income')
                    
                    if period_end and (revenue is not None or ebit is not None):
                        all_periods.append({
                            'period_end_date': period_end,
                            'data_source': 'FY',
                            'period_type': 'FY',
                            'fiscal_year': stmt.get('fiscal_year'),
                            'total_revenue': revenue,
                            'ebit': ebit,
                            'original_data': stmt
                        })
            
            # Keep both LTM and FY periods - use composite key (date + data_source)
            unique_periods = {}
            for period in all_periods:
                key = (period['period_end_date'], period['data_source'])
                unique_periods[key] = period
            
            # Sort by period_end_date (most recent first)
            sorted_periods = sorted(unique_periods.values(), 
                                  key=lambda x: x['period_end_date'], reverse=True)
            
            logger.debug(f"Found {len(sorted_periods)} unique financial periods for {ticker}")
            return sorted_periods
            
        except Exception as e:
            logger.error(f"Error getting financial periods for {ticker}: {e}")
            return []
    
    def _find_previous_year_period(self, current_period: Dict, all_periods: List[Dict]) -> Optional[Dict]:
        """
        Find the period approximately 12 months before the current period
        
        Args:
            current_period: Current period data
            all_periods: List of all available periods
            
        Returns:
            Previous year period or None if not found
        """
        current_date = current_period['period_end_date']
        
        # Look for period between 11-15 months earlier (flexible range)
        best_match = None
        best_diff = float('inf')
        
        for period in all_periods:
            if period['period_end_date'] >= current_date:
                continue  # Skip current and future periods
            
            # Calculate days difference
            days_diff = (current_date - period['period_end_date']).days
            
            # Target is ~365 days (12 months), accept 330-450 days range
            if 330 <= days_diff <= 450:
                # Find the closest to 365 days
                target_diff = abs(days_diff - 365)
                if target_diff < best_diff:
                    best_diff = target_diff
                    best_match = period
        
        if best_match:
            logger.debug(f"Found previous year period: {current_date} -> {best_match['period_end_date']} "
                        f"({(current_date - best_match['period_end_date']).days} days apart)")
        
        return best_match
    
    def _calculate_simple_growth_ratios(self, ticker: str, company_id: str, 
                                      current_period: Dict, previous_period: Dict) -> List[Dict[str, Any]]:
        """
        Calculate growth ratios between two periods
        
        Args:
            ticker: Company ticker
            company_id: Company ID
            current_period: Current period data
            previous_period: Previous period data
            
        Returns:
            List of calculated growth ratios
        """
        growth_ratios = []
        
        current_date = current_period['period_end_date']
        previous_date = previous_period['period_end_date']
        days_diff = (current_date - previous_date).days
        
        # Growth ratio definitions to calculate
        growth_definitions = {
            'Revenue_Growth_YoY': {
                'field': 'total_revenue',
                'description': f'Revenue Growth YoY - Year-over-year growth in total revenue ({days_diff} days)'
            },
            'EBIT_Growth_YoY': {
                'field': 'ebit',
                'description': f'EBIT Growth YoY - Year-over-year growth in EBIT ({days_diff} days)'
            }
        }
        
        for ratio_name, definition in growth_definitions.items():
            field_name = definition['field']
            description = definition['description']
            
            current_value = current_period.get(field_name)
            previous_value = previous_period.get(field_name)
            
            if current_value is None or previous_value is None or previous_value == 0:
                logger.debug(f"Skipping {ratio_name}: current={current_value}, previous={previous_value}")
                continue
            
            # Calculate growth rate: (Current - Previous) / Previous
            growth_rate = (float(current_value) - float(previous_value)) / float(previous_value)
            
            # Determine period type and fiscal info
            period_type = current_period.get('period_type', 'LTM')
            data_source = current_period.get('data_source', 'LTM')
            fiscal_year = current_period.get('fiscal_year')
            
            growth_ratio = {
                'company_id': company_id,
                'ticker': ticker,
                'name': ratio_name,
                'ratio_value': growth_rate,
                'description': description,
                'category': 'growth',
                'period_end_date': current_date,
                'fiscal_year': fiscal_year,
                'fiscal_quarter': period_type,
                'data_source': data_source,
                'calculation_inputs': {
                    'current_value': float(current_value),
                    'previous_value': float(previous_value),
                    'current_date': str(current_date),
                    'previous_date': str(previous_date),
                    'days_apart': days_diff,
                    'field_name': field_name,
                    'current_source': current_period.get('data_source'),
                    'previous_source': previous_period.get('data_source')
                },
                'calculation_date': datetime.utcnow()
            }
            
            growth_ratios.append(growth_ratio)
            logger.debug(f"Calculated {ratio_name} for {current_date}: {growth_rate:.4f} "
                        f"(vs {previous_date}, {days_diff} days apart)")
        
        return growth_ratios
    
    # Old methods removed - now using simplified approach
    
    def _store_growth_ratio(self, ratio_data: Dict[str, Any]) -> bool:
        """Store a growth ratio in the database using direct SQL like SimpleRatioCalculator"""
        try:
            # Get or create ratio definition ID for this growth ratio
            ratio_definition_id = self._get_or_create_growth_ratio_definition(
                ratio_data['name'], 
                ratio_data['description'], 
                ratio_data['category']
            )
            
            if not ratio_definition_id:
                logger.error(f"Failed to get/create ratio definition for {ratio_data['name']}")
                return False
            
            # Store using direct SQL approach like SimpleRatioCalculator
            with self.database.connection.cursor() as cursor:
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
                        data_source = EXCLUDED.data_source
                """, (
                    ratio_data['company_id'],
                    ratio_definition_id,
                    ratio_data['ticker'],
                    ratio_data['name'],
                    ratio_data.get('category', 'growth'),  # Add category
                    ratio_data['period_end_date'],
                    ratio_data['fiscal_quarter'],  # period_type
                    ratio_data['fiscal_year'],
                    ratio_data['fiscal_quarter'],
                    ratio_data['ratio_value'],
                    json.dumps(ratio_data['calculation_inputs']),  # JSON string
                    ratio_data.get('data_source', 'annual')  # data_source
                ))
                
                # Explicit commit to ensure data is saved
                self.database.connection.commit()
                logger.debug(f"Successfully stored growth ratio: {ratio_data['name']} = {ratio_data['ratio_value']:.2f}%")
                
            return True
            
        except Exception as e:
            logger.error(f"Error storing growth ratio: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _get_or_create_growth_ratio_definition(self, name: str, description: str, category: str) -> Optional[str]:
        """Get or create ratio definition for growth ratios"""
        try:
            with self.database.connection.cursor() as cursor:
                # Check if ratio definition already exists
                cursor.execute("""
                    SELECT id FROM ratio_definitions 
                    WHERE name = %s AND company_id IS NULL
                """, (name,))
                
                result = cursor.fetchone()
                if result:
                    return str(result[0])
                
                # Create new ratio definition
                import uuid
                ratio_def_id = uuid.uuid4()
                
                cursor.execute("""
                    INSERT INTO ratio_definitions (
                        id, name, company_id, formula, description, category, is_active, created_by
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(ratio_def_id),  # Convert UUID to string
                    name,
                    None,  # Global ratio
                    f"YOY_GROWTH({category.lower()}_field)",  # Dummy formula
                    description,
                    category,
                    True,
                    "growth_calculator"
                ))
                
                # Commit the ratio definition creation
                self.database.connection.commit()
                logger.debug(f"Created growth ratio definition: {name}")
                
                return str(ratio_def_id)
                
        except Exception as e:
            logger.error(f"Error creating growth ratio definition: {e}")
            return None


def calculate_company_growth_ratios(ticker: str) -> Dict[str, Any]:
    """
    Convenience function to calculate growth ratios for a company
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        Dictionary with calculation results
    """
    with GrowthRatioCalculator() as calculator:
        return calculator.calculate_yoy_growth_ratios(ticker)


def get_company_growth_ratios(ticker: str) -> List[Dict[str, Any]]:
    """
    Get stored growth ratios for a company
    
    Args:
        ticker: Company ticker symbol
        
    Returns:
        List of growth ratio records
    """
    try:
        with FinancialDatabase() as db:
            # Get company info
            company_info = db.get_company_by_ticker(ticker)
            if not company_info:
                return []
            
            # Get growth ratios
            ratios = db.get_calculated_ratios(ticker)
            growth_ratios = [r for r in ratios if r.get('category') == 'growth']
            
            return growth_ratios
            
    except Exception as e:
        logger.error(f"Error getting growth ratios for {ticker}: {e}")
        return []