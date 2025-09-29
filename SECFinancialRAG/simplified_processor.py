"""
Simplified SEC Financial Processor
Separates processing into distinct statement processors to avoid duplicates
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict
from datetime import datetime
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

try:
    from .mapping import FinancialDataMapper
except ImportError:
    from mapping import FinancialDataMapper

logger = logging.getLogger(__name__)
# Ensure INFO level for debugging
logger.setLevel(logging.INFO)


class BaseStatementProcessor:
    """Base class for statement processors"""
    
    def __init__(self, filing_preference: str = 'latest'):
        self.filing_preference = filing_preference
        self.mapper = FinancialDataMapper()
    
    def get_statement_fields(self) -> Set[str]:
        """Get the database fields for this statement type"""
        raise NotImplementedError
    
    def get_statement_type(self) -> str:
        """Get the statement type name"""
        raise NotImplementedError
    
    def create_statement_record(self, period_info: Dict[str, Any], facts: Dict[str, Any]) -> Dict[str, Any]:
        """Create a statement record from period info and facts"""
        raise NotImplementedError


class BalanceSheetProcessor(BaseStatementProcessor):
    """Processor for balance sheet statements"""
    
    def get_statement_fields(self) -> Set[str]:
        """Get balance sheet database fields"""
        return {
            'cash_and_cash_equivalents', 'short_term_investments', 'cash_and_short_term_investments', 'accounts_receivable', 
            'inventory', 'prepaid_expenses', 'total_current_assets', 'property_plant_equipment',
            'goodwill', 'intangible_assets', 'long_term_investments', 'other_assets',
            'total_non_current_assets', 'total_assets', 'accounts_payable', 'accrued_liabilities',
            'commercial_paper', 'other_short_term_borrowings', 'current_portion_long_term_debt',
            'finance_lease_liability_current', 'operating_lease_liability_current',
            'total_current_liabilities', 'long_term_debt', 'non_current_long_term_debt', 'finance_lease_liability_noncurrent',
            'operating_lease_liability_noncurrent', 'other_long_term_liabilities',
            'total_non_current_liabilities', 'total_liabilities', 'common_stock',
            'retained_earnings', 'accumulated_oci', 'total_stockholders_equity',
            'total_liabilities_and_equity'
        }
    
    def get_statement_type(self) -> str:
        return 'balance_sheet'
    
    def create_statement_record(self, period_info: Dict[str, Any], facts: Dict[str, Any], 
                               ticker: str = None, cik: str = None, company_name: str = None) -> Dict[str, Any]:
        """Create balance sheet record"""
        record = {
            'statement_type': 'balance_sheet',
            'period_end_date': period_info['end_date'],
            'period_start_date': period_info.get('start_date'),
            'filing_date': period_info['filed_date'],
            'period_type': period_info.get('period_type', 'Q4'),
            'form_type': period_info.get('form_type'),
            'fiscal_year': period_info.get('fiscal_year'),
            'period_length_months': period_info.get('period_length_months', 3),
            'ticker': ticker or '',
            'cik': cik or '',
            'company_name': company_name or ''
        }
        
        # Add all balance sheet facts
        record.update(facts)
        
        return record


class IncomeStatementProcessor(BaseStatementProcessor):
    """Processor for income statements"""
    
    def get_statement_fields(self) -> Set[str]:
        """Get income statement database fields"""
        return {
            'total_revenue', 'cost_of_revenue', 'gross_profit', 'research_and_development',
            'sales_and_marketing', 'sales_general_and_admin', 'general_and_administrative', 'total_operating_expenses',
            'operating_income', 'ebitda', 'interest_income', 'interest_expense', 'other_income',
            'income_before_taxes', 'income_tax_expense', 'net_income',
            'earnings_per_share_basic', 'earnings_per_share_diluted',
            'weighted_average_shares_basic', 'weighted_average_shares_diluted'
        }
    
    def get_statement_type(self) -> str:
        return 'income_statement'
    
    def create_statement_record(self, period_info: Dict[str, Any], facts: Dict[str, Any], 
                               ticker: str = None, cik: str = None, company_name: str = None) -> Dict[str, Any]:
        """Create income statement record"""
        record = {
            'statement_type': 'income_statement',
            'period_end_date': period_info['end_date'],
            'period_start_date': period_info.get('start_date'),
            'filing_date': period_info['filed_date'],
            'period_type': period_info.get('period_type', 'Q4'),
            'form_type': period_info.get('form_type'),
            'fiscal_year': period_info.get('fiscal_year'),
            'period_length_months': period_info.get('period_length_months', 3),
            'ticker': ticker or '',
            'cik': cik or '',
            'company_name': company_name or ''
        }
        
        # Add all income statement facts
        record.update(facts)
        
        return record


class CashFlowProcessor(BaseStatementProcessor):
    """Processor for cash flow statements"""
    
    def get_statement_fields(self) -> Set[str]:
        """Get cash flow statement database fields"""
        return {
            'net_cash_from_operating_activities', 'depreciation_and_amortization', 'depreciation', 'amortization',
            'stock_based_compensation', 'changes_in_accounts_receivable', 'changes_in_other_receivable',
            'changes_in_inventory', 'changes_in_accounts_payable', 'changes_in_other_operating_assets',
            'changes_in_other_operating_liabilities', 'net_cash_from_investing_activities', 'capital_expenditures', 'acquisitions',
            'purchases_of_intangible_assets', 'investments_purchased', 'investments_sold', 'divestitures', 'net_cash_from_financing_activities',
            'dividends_paid', 'share_repurchases', 'proceeds_from_stock_issuance', 'debt_issued', 'debt_repaid',
            'net_change_in_cash', 'cash_beginning_of_period', 'cash_end_of_period'
        }
    
    def get_statement_type(self) -> str:
        return 'cash_flow'
    
    def create_statement_record(self, period_info: Dict[str, Any], facts: Dict[str, Any], 
                               ticker: str = None, cik: str = None, company_name: str = None) -> Dict[str, Any]:
        """Create cash flow statement record"""
        record = {
            'statement_type': 'cash_flow',
            'period_end_date': period_info['end_date'],
            'period_start_date': period_info.get('start_date'),
            'filing_date': period_info['filed_date'],
            'period_type': period_info.get('period_type', 'Q4'),
            'form_type': period_info.get('form_type'),
            'fiscal_year': period_info.get('fiscal_year'),
            'period_length_months': period_info.get('period_length_months', 3),
            'ticker': ticker or '',
            'cik': cik or '',
            'company_name': company_name or ''
        }
        
        # Add all cash flow facts
        record.update(facts)
        
        return record


class SimplifiedSECFinancialProcessor:
    """
    Simplified SEC Financial Processor
    
    Processes each statement type independently to ensure:
    - One balance sheet record per period_end_date
    - Each data point sourced from latest/oldest filing based on preference
    - Period type always sourced from oldest 'fp' data for balance sheets
    """
    
    def __init__(self, filing_preference: str = 'latest'):
        """
        Initialize processor
        
        Args:
            filing_preference: 'latest' or 'original' - determines which filing to use
        """
        self.filing_preference = filing_preference
        self.mapper = FinancialDataMapper()
        
        # Initialize statement processors
        self.balance_sheet_processor = BalanceSheetProcessor(filing_preference)
        self.income_statement_processor = IncomeStatementProcessor(filing_preference)
        self.cash_flow_processor = CashFlowProcessor(filing_preference)
        
        logger.info(f"Initialized SimplifiedSECFinancialProcessor with preference: {filing_preference}")
    
    def process_company_facts(self, company_facts: Dict[str, Any], ticker: str) -> List[Dict[str, Any]]:
        """
        Process company facts into financial statements
        
        Args:
            company_facts: Raw SEC company facts data
            ticker: Company ticker symbol
            
        Returns:
            List of financial statement records
        """
        logger.info(f"Processing company facts for {ticker}")
        
        try:
            # Extract basic company info
            entity_info = company_facts.get('entityName', ticker)
            cik = str(company_facts.get('cik', '')).zfill(10)
            
            # Process US GAAP facts
            us_gaap_facts = company_facts.get('facts', {}).get('us-gaap', {})
            
            if not us_gaap_facts:
                logger.warning(f"No US GAAP facts found for {ticker}")
                return []
            
            # Group all facts by period
            all_periods = self._group_facts_by_period(us_gaap_facts, ticker, cik, entity_info)
            
            # Process each statement type independently
            results = []
            
            # Process balance sheets
            balance_sheet_results = self._process_balance_sheets(all_periods, ticker, cik, entity_info)
            results.extend(balance_sheet_results)
            
            # Process income statements  
            income_statement_results = self._process_income_statements(all_periods, ticker, cik, entity_info)
            results.extend(income_statement_results)
            
            # Process cash flow statements
            cash_flow_results = self._process_cash_flows(all_periods, ticker, cik, entity_info)
            results.extend(cash_flow_results)
            
            logger.info(f"Processed {ticker}: {len(balance_sheet_results)} balance sheets, "
                       f"{len(income_statement_results)} income statements, "
                       f"{len(cash_flow_results)} cash flow statements")
            
            # Sort all results by period_end_date in descending order (newest first)
            results.sort(key=lambda x: x.get('period_end_date', ''), reverse=True)
            logger.debug(f"Sorted {len(results)} records by period_end_date (newest first)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing company facts for {ticker}: {e}")
            raise
    
    def _group_facts_by_period(self, us_gaap_facts: Dict[str, Any], ticker: str, cik: str, entity_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Group all facts by period information
        
        Returns:
            Dict mapping period keys to period info with all facts
        """
        periods = defaultdict(lambda: {
            'facts': {},
            'end_date': None,
            'start_date': None,
            'filed_date': None,
            'form_type': None,
            'period_type': None,
            'fiscal_year': None,
            'period_length_months': None,
            'ticker': ticker,
            'cik': cik,
            'company_name': entity_name
        })
        
        logger.debug(f"Processing {len(us_gaap_facts)} GAAP facts for {ticker}")
        
        for fact_name, fact_data in us_gaap_facts.items():
            # Map to database column
            db_column = self.mapper.map_item(fact_name)
            if not db_column:
                continue
            
            # Process each unit type in the fact
            units = fact_data.get('units', {})
            for unit_type, entries in units.items():
                if unit_type not in ['USD', 'shares', 'pure', 'USD/shares']:
                    continue
                
                for entry in entries:
                    try:
                        # Extract period information
                        end_date = entry.get('end')
                        start_date = entry.get('start')
                        filed_date = entry.get('filed')
                        form_type = entry.get('form')
                        fp_value = entry.get('fp')  # Fiscal period indicator
                        fy_value = entry.get('fy')  # Fiscal year
                        
                        if not end_date or not filed_date:
                            continue
                        
                        # Parse dates
                        end_date_obj = parse_date(end_date).date()
                        filed_date_obj = parse_date(filed_date).date()
                        start_date_obj = parse_date(start_date).date() if start_date else None
                        
                        # Determine period length and type
                        period_length_months = 3  # Default
                        period_type = fp_value or 'Q4'  # Default to Q4 if no fp
                        
                        if start_date_obj and end_date_obj:
                            delta = end_date_obj - start_date_obj
                            if delta.days > 330:  # Approximately 11 months
                                period_length_months = 12
                                period_type = 'FY'
                            elif delta.days > 240:  # Approximately 8 months
                                period_length_months = 9
                                # Keep the original period_type (likely Q3 or similar)
                            elif delta.days > 150:  # Approximately 5 months
                                period_length_months = 6
                                # Keep the original period_type (likely Q2 or similar)
                            else:
                                period_length_months = 3
                                # If period_type was incorrectly set to FY but length is 3 months, fix it
                                if period_type == 'FY':
                                    period_type = 'Q4'  # Default to Q4 for quarterly periods
                        
                        # Create period key
                        period_key = f"{end_date}_{filed_date}_{period_type}_{period_length_months}"
                        
                        # Update period info
                        period_info = periods[period_key]
                        period_info['end_date'] = end_date_obj
                        # IMPROVED: Only update start_date if we have a valid value and current value is None
                        # This prevents overwriting valid start dates with None from subsequent fact entries
                        if start_date_obj is not None and period_info['start_date'] is None:
                            period_info['start_date'] = start_date_obj
                        period_info['filed_date'] = filed_date_obj
                        period_info['form_type'] = form_type
                        period_info['period_type'] = period_type
                        period_info['fiscal_year'] = int(fy_value) if fy_value else end_date_obj.year
                        period_info['period_length_months'] = period_length_months
                        
                        # Store the fact value with conflict detection
                        value = entry.get('val')
                        if value is not None:
                            # Instead of directly overwriting, collect all values for conflict resolution
                            if 'fact_candidates' not in period_info:
                                period_info['fact_candidates'] = defaultdict(list)
                            
                            
                            # Store both the GAAP code and value for later conflict resolution
                            period_info['fact_candidates'][db_column].append({
                                'gaap_code': fact_name,
                                'value': value,
                                'unit_type': unit_type,
                                'frame': entry.get('frame'),  # Include frame for better conflict resolution
                                'filed_date': filed_date_obj
                            })
                        
                    except Exception as e:
                        logger.debug(f"Error processing fact {fact_name} entry: {e}")
                        continue
        
        logger.info(f"Grouped facts into {len(periods)} periods for {ticker}")
        
        # CONFLICT RESOLUTION: Resolve multiple GAAP codes mapping to same database column
        for period_key, period_info in periods.items():
            if 'fact_candidates' in period_info:
                # Initialize facts dictionary if not exists
                if 'facts' not in period_info:
                    period_info['facts'] = {}
                
                # Resolve conflicts by taking the best value for each database column
                for db_column, candidates in period_info['fact_candidates'].items():
                    if len(candidates) > 1:
                        # Multiple GAAP codes map to this column
                        # First filter out None values, then take the highest value
                        non_none_candidates = [c for c in candidates if c['value'] is not None]
                        
                        
                        if non_none_candidates:
                            # Use the highest non-None value
                            highest_candidate = max(non_none_candidates, key=lambda x: x['value'])
                            logger.debug(f"Conflict resolved for {ticker} {period_info['end_date']} {db_column}: "
                                       f"selected {highest_candidate['gaap_code']} = {highest_candidate['value']} "
                                       f"from {len(non_none_candidates)} non-None candidates (total: {len(candidates)})")
                            period_info['facts'][db_column] = highest_candidate['value']
                        else:
                            # All candidates are None, log this case
                            logger.warning(f"All candidates for {ticker} {period_info['end_date']} {db_column} are None: "
                                         f"{[c['gaap_code'] for c in candidates]}")
                            period_info['facts'][db_column] = None
                    else:
                        # Only one candidate, use it directly
                        candidate_value = candidates[0]['value']
                        if candidate_value is None:
                            logger.debug(f"Single candidate for {ticker} {period_info['end_date']} {db_column} is None: "
                                       f"{candidates[0]['gaap_code']}")
                        period_info['facts'][db_column] = candidate_value
                
                # Clean up temporary data
                del period_info['fact_candidates']
        
        # COMPUTED GAAP FIELDS: Apply arithmetic computations ONLY for missing fields
        # VERIFIED: Basic consolidation works correctly, computed fields will only fill gaps
        logger.debug(f"Applying computed GAAP fields for {len(periods)} periods (AFTER conflict resolution)")
        for period_key, period_info in periods.items():
            try:
                # Apply computed field logic using raw GAAP facts for this specific period
                # This will only fill in missing/null fields, never overwrite direct mappings
                period_info['facts'] = self.mapper.compute_missing_fields_from_raw_gaap(
                    period_info['facts'], 
                    us_gaap_facts,  # Raw GAAP facts
                    period_key
                )
            except Exception as e:
                logger.debug(f"Error computing missing fields for period {period_key}: {e}")
        
        # FISCAL YEAR CONSISTENCY FIX: Ensure fiscal_year comes from oldest filing for each period
        # Group periods by end_date and period_length to consolidate fiscal_year
        periods_by_end_date = defaultdict(list)
        for period_key, period_info in periods.items():
            end_date = period_info['end_date']
            period_length = period_info['period_length_months']
            key = f"{end_date}_{period_length}"
            periods_by_end_date[key].append((period_key, period_info))
        
        # For each group of periods with same end_date and length, use fiscal_year from oldest filing
        for group_key, period_group in periods_by_end_date.items():
            if len(period_group) > 1:
                # Sort by filing date to get oldest first
                period_group.sort(key=lambda x: x[1]['filed_date'])
                oldest_period_info = period_group[0][1]
                oldest_fiscal_year = oldest_period_info['fiscal_year']
                
                # Apply the oldest fiscal_year to all periods in this group
                for period_key, period_info in period_group:
                    if period_info['fiscal_year'] != oldest_fiscal_year:
                        logger.debug(f"Fiscal year consistency fix for {ticker} {period_info['end_date']}: "
                                   f"changing from FY{period_info['fiscal_year']} to FY{oldest_fiscal_year} "
                                   f"(using oldest filing date)")
                        period_info['fiscal_year'] = oldest_fiscal_year
        
        return dict(periods)
    
    def _get_preferred_fact_value(self, fact_name: str, periods_list: List[Tuple[str, Dict[str, Any]]], filing_preference: str) -> Optional[any]:
        """
        Get preferred value for a fact, prioritizing direct mappings over computed values
        
        Args:
            fact_name: Database column name
            periods_list: List of (period_key, period_info) tuples sorted by filing date
            filing_preference: 'original' or 'latest'
            
        Returns:
            The preferred value or None if not found
        """
        # Separate direct mappings from computed values
        direct_candidates = []
        computed_candidates = []
        
        for period_key, period_info in periods_list:
            if fact_name in period_info['facts'] and period_info['facts'][fact_name] is not None:
                is_computed = period_info['facts'].get(f"_computed_{fact_name}", False)
                candidate = (period_key, period_info)
                
                if is_computed:
                    computed_candidates.append(candidate)
                else:
                    direct_candidates.append(candidate)
        
        # Prioritize direct mappings over computed values
        if direct_candidates:
            candidates = direct_candidates
            value_type = "DIRECT"
        elif computed_candidates:
            candidates = computed_candidates 
            value_type = "COMPUTED"
        else:
            return None
        
        # Apply filing preference within the chosen category
        if filing_preference == 'original':
            chosen_candidate = candidates[0]  # Oldest
        else:  # 'latest'
            chosen_candidate = candidates[-1]  # Newest
        
        chosen_value = chosen_candidate[1]['facts'][fact_name]
        
        if len(direct_candidates) > 0 and len(computed_candidates) > 0:
            logger.debug(f"Consolidation for {fact_name}: chose {value_type} value {chosen_value:,.0f} "
                        f"(had {len(direct_candidates)} direct, {len(computed_candidates)} computed candidates)")
        
        return chosen_value
    
    def _process_balance_sheets(self, all_periods: Dict[str, Dict[str, Any]], ticker: str, cik: str, entity_name: str) -> List[Dict[str, Any]]:
        """Process balance sheet statements with consolidation by period_end_date"""
        
        # Get balance sheet fields
        balance_sheet_fields = self.balance_sheet_processor.get_statement_fields()
        
        # Group periods by end_date that have balance sheet facts
        balance_sheet_periods = defaultdict(list)
        
        for period_key, period_info in all_periods.items():
            # Check if this period has any balance sheet facts
            has_balance_sheet_facts = any(
                fact_name in balance_sheet_fields 
                for fact_name in period_info['facts'].keys()
            )
            
            if has_balance_sheet_facts:
                end_date = period_info['end_date']
                balance_sheet_periods[end_date].append((period_key, period_info))
        
        logger.info(f"Found {len(balance_sheet_periods)} unique balance sheet end dates for {ticker}")
        
        # Process each end_date group
        results = []
        for end_date, periods_list in balance_sheet_periods.items():
            try:
                consolidated_record = self._consolidate_balance_sheet_for_end_date(
                    periods_list, end_date, ticker, cik, entity_name
                )
                if consolidated_record:
                    results.append(consolidated_record)
            except Exception as e:
                logger.error(f"Error consolidating balance sheet for {ticker} {end_date}: {e}")
                continue
        
        return results
    
    def _consolidate_balance_sheet_for_end_date(self, periods_list: List[Tuple[str, Dict[str, Any]]], 
                                              end_date, ticker: str, cik: str, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Consolidate balance sheet data for a single end_date
        
        For balance sheets:
        - Period type always sourced from oldest 'fp' data
        - Each data point sourced from latest/oldest filing based on preference
        """
        
        if not periods_list:
            return None
        
        # Sort periods by filing date
        periods_list.sort(key=lambda x: x[1]['filed_date'])
        
        balance_sheet_fields = self.balance_sheet_processor.get_statement_fields()
        
        # Get period_type from oldest filing (for balance sheets, period_type should be consistent)
        oldest_period = periods_list[0][1]
        reference_period_type = oldest_period.get('period_type', 'Q4')
        
        # Base period info from the reference filing (oldest for period_type)
        base_period_info = {
            'end_date': end_date,
            'start_date': oldest_period.get('start_date'),
            'period_type': reference_period_type,
            'fiscal_year': oldest_period.get('fiscal_year'),
            'period_length_months': oldest_period.get('period_length_months', 3),
            'form_type': oldest_period.get('form_type'),
            'ticker': ticker,
            'cik': cik,
            'company_name': entity_name
        }
        
        # Determine filing date based on preference
        if self.filing_preference == 'original':
            reference_filing = periods_list[0]  # Oldest
        else:  # 'latest'
            reference_filing = periods_list[-1]  # Newest
        
        base_period_info['filed_date'] = reference_filing[1]['filed_date']
        
        # Consolidate facts: for each balance sheet field, get value from preferred filing
        consolidated_facts = {}
        
        # Get all balance sheet facts available across all periods
        all_balance_sheet_facts = set()
        for _, period_info in periods_list:
            all_balance_sheet_facts.update(
                fact_name for fact_name in period_info['facts'].keys() 
                if fact_name in balance_sheet_fields
            )
        
        # For each balance sheet fact, find the preferred value with computed value priority logic
        for fact_name in all_balance_sheet_facts:
            value = self._get_preferred_fact_value(fact_name, periods_list, self.filing_preference)
            if value is not None:
                consolidated_facts[fact_name] = value
        
        # Clean up computed markers from consolidated facts (don't store these in database)
        consolidated_facts = {k: v for k, v in consolidated_facts.items() if not k.startswith('_computed_')}
        
        logger.debug(f"Consolidated balance sheet for {ticker} {end_date}: "
                    f"{len(consolidated_facts)} facts from {len(periods_list)} filings, "
                    f"period_type={reference_period_type}, preference={self.filing_preference}")
        
        # Create the final record
        record = self.balance_sheet_processor.create_statement_record(
            base_period_info, consolidated_facts, ticker, cik, entity_name
        )
        
        
        return record
    
    def _process_income_statements(self, all_periods: Dict[str, Dict[str, Any]], ticker: str, cik: str, entity_name: str) -> List[Dict[str, Any]]:
        """Process income statements - each period is processed independently"""
        
        income_statement_fields = self.income_statement_processor.get_statement_fields()
        
        results = []
        skipped_count = 0
        
        for period_key, period_info in all_periods.items():
            # Check if this period has income statement facts
            income_facts = {
                fact_name: value for fact_name, value in period_info['facts'].items()
                if fact_name in income_statement_fields
            }
            
            if income_facts:
                # CRITICAL VALIDATION: Only process records with reliable period information
                # Income statements can be 3/6/9/12 month cumulative, so we need to know the exact period
                start_date = period_info.get('start_date')
                period_length = period_info.get('period_length_months')
                
                if start_date is None and period_length is None:
                    logger.warning(f"Skipping income statement for {ticker} {period_info.get('end_date')} "
                                 f"(period_key: {period_key}): No start_date or period_length to determine "
                                 f"if this is 3/6/9/12 month data. Cannot reliably classify period type.")
                    skipped_count += 1
                    continue
                
                # Smart handling: If we have period_length but no start_date, calculate indicative start_date
                if start_date is None and period_length is not None:
                    end_date = period_info.get('end_date')
                    if end_date:
                        # Calculate indicative start date based on period length
                        calculated_start_date = end_date - relativedelta(months=period_length)
                        
                        # Update the period_info with calculated start date
                        period_info['start_date'] = calculated_start_date
                        period_info['_calculated_start_date'] = True  # Mark as calculated for transparency
                        
                        logger.info(f"Calculated start_date for {ticker} {end_date}: "
                                  f"period_length={period_length} months -> start_date={calculated_start_date} "
                                  f"(calculated from end_date)")
                    else:
                        logger.warning(f"Skipping income statement for {ticker}: "
                                     f"has period_length={period_length} but no end_date to calculate start_date")
                        skipped_count += 1
                        continue
                
                try:
                    record = self.income_statement_processor.create_statement_record(
                        period_info, income_facts, ticker, cik, entity_name
                    )
                    results.append(record)
                except Exception as e:
                    logger.error(f"Error creating income statement for {ticker} {period_key}: {e}")
                    continue
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} income statement records for {ticker} due to missing period information")
        
        logger.info(f"Created {len(results)} income statements for {ticker} (skipped {skipped_count})")
        return results
    
    def _process_cash_flows(self, all_periods: Dict[str, Dict[str, Any]], ticker: str, cik: str, entity_name: str) -> List[Dict[str, Any]]:
        """Process cash flow statements - each period is processed independently"""
        
        cash_flow_fields = self.cash_flow_processor.get_statement_fields()
        
        results = []
        skipped_count = 0
        
        for period_key, period_info in all_periods.items():
            # Check if this period has cash flow facts
            cash_flow_facts = {
                fact_name: value for fact_name, value in period_info['facts'].items()
                if fact_name in cash_flow_fields
            }
            
            if cash_flow_facts:
                # CRITICAL VALIDATION: Only process records with reliable period information
                # Cash flow statements can be 3/6/9/12 month cumulative, so we need to know the exact period
                start_date = period_info.get('start_date')
                period_length = period_info.get('period_length_months')
                
                if start_date is None and period_length is None:
                    logger.warning(f"Skipping cash flow statement for {ticker} {period_info.get('end_date')} "
                                 f"(period_key: {period_key}): No start_date or period_length to determine "
                                 f"if this is 3/6/9/12 month data. Cannot reliably classify period type.")
                    skipped_count += 1
                    continue
                
                # Smart handling: If we have period_length but no start_date, calculate indicative start_date
                if start_date is None and period_length is not None:
                    end_date = period_info.get('end_date')
                    if end_date:
                        # Calculate indicative start date based on period length
                        calculated_start_date = end_date - relativedelta(months=period_length)
                        
                        # Update the period_info with calculated start date
                        period_info['start_date'] = calculated_start_date
                        period_info['_calculated_start_date'] = True  # Mark as calculated for transparency
                        
                        logger.info(f"Calculated start_date for {ticker} {end_date}: "
                                  f"period_length={period_length} months -> start_date={calculated_start_date} "
                                  f"(calculated from end_date)")
                    else:
                        logger.warning(f"Skipping cash flow statement for {ticker}: "
                                     f"has period_length={period_length} but no end_date to calculate start_date")
                        skipped_count += 1
                        continue
                
                try:
                    record = self.cash_flow_processor.create_statement_record(
                        period_info, cash_flow_facts, ticker, cik, entity_name
                    )
                    results.append(record)
                except Exception as e:
                    logger.error(f"Error creating cash flow statement for {ticker} {period_key}: {e}")
                    continue
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} cash flow records for {ticker} due to missing period information")
        
        logger.info(f"Created {len(results)} cash flow statements for {ticker} (skipped {skipped_count})")
        return results
