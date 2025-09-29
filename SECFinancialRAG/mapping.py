"""
US GAAP to Database Mapping
Maps SEC XBRL taxonomy items to local database schema
"""

from typing import Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)

# US GAAP to Database mapping as provided
US_GAAP_TO_DB_MAPPING = {
    # Income Statement
    "RevenueFromContractWithCustomerExcludingAssessedTax": "total_revenue",
    "SalesRevenueNet": "total_revenue",
    "Revenues": "total_revenue",
    
    "CostOfGoodsAndServicesSold": "cost_of_revenue",
    "CostOfRevenue": "cost_of_revenue",

    "GrossProfit": "gross_profit",

    "ResearchAndDevelopmentExpense": "research_and_development",
    "SellingGeneralAndAdministrativeExpense": "sales_general_and_admin",
    "SellingAndMarketingExpense": "sales_and_marketing",
    "GeneralAndAdministrativeExpense": "general_and_administrative",
    "OperatingExpenses": "total_operating_expenses",

    "OperatingIncomeLoss": "operating_income",

    
    "InvestmentIncomeInterestAndDividend": "interest_income",
    "InvestmentIncomeNet": "interest_income",
    "InterestExpense": "interest_expense",
    "NonoperatingIncomeExpense": "other_income",

    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "income_before_taxes",
    "IncomeTaxExpenseBenefit": "income_tax_expense",

    "NetIncomeLoss": "net_income",
    "NetIncomeLossAvailableToCommonStockholdersBasic": "net_income",

    "EarningsPerShareBasic": "earnings_per_share_basic",
    "EarningsPerShareDiluted": "earnings_per_share_diluted",
    "WeightedAverageNumberOfSharesOutstandingBasic": "weighted_average_shares_basic",
    "WeightedAverageNumberOfDilutedSharesOutstanding": "weighted_average_shares_diluted",

    # Balance Sheet
    "CashAndCashEquivalentsAtCarryingValue": "cash_and_cash_equivalents",
    "MarketableSecuritiesCurrent": "short_term_investments",
    "CashCashEquivalentsAndShortTermInvestments": "cash_and_short_term_investments",
    "AccountsReceivableNetCurrent": "accounts_receivable",
    "AccountsAndOtherReceivablesNetCurrent": "accounts_receivable",
    "ReceivablesNetCurrent": "accounts_receivable",
    "InventoryNet": "inventory",
    "AssetsCurrent": "total_current_assets",

    "PropertyPlantAndEquipmentNet": "property_plant_equipment",
    "Goodwill": "goodwill",
    "IntangibleAssetsNetExcludingGoodwill": "intangible_assets",
    # "MarketableSecuritiesNoncurrent": "long_term_investments",
    "OtherAssetsNoncurrent": "other_assets",
    "AssetsNoncurrent": "total_non_current_assets",
    "Assets": "total_assets",

    "AccountsPayableCurrent": "accounts_payable",
    "AccruedLiabilitiesCurrent": "accrued_liabilities",
    "CommercialPaper": "commercial_paper",
    "OtherShortTermBorrowings": "other_short_term_borrowings",
    "LongTermDebtCurrent": "current_portion_long_term_debt",
    "LongTermDebtAndCapitalLeaseObligationsCurrent": "current_portion_long_term_debt",
    "FinanceLeaseLiabilityCurrent": "finance_lease_liability_current",
    "OperatingLeaseLiabilityCurrent": "operating_lease_liability_current",
    "LiabilitiesCurrent": "total_current_liabilities",

    "LongTermDebtNoncurrent": "non_current_long_term_debt",
    "LongTermDebtAndCapitalLeaseObligations": "non_current_long_term_debt",
    "DebtInstrumentCarryingAmount": "long_term_debt",
    "LongTermDebt": "long_term_debt",
    "FinanceLeaseLiabilityNonCurrent": "finance_lease_liability_noncurrent",
    "OperatingLeaseLiabilityNoncurrent": "operating_lease_liability_noncurrent",
    "OtherLiabilitiesNoncurrent": "other_long_term_liabilities",
    "LiabilitiesNonCurrent": "total_non_current_liabilities",
    "Liabilities": "total_liabilities",

    "CommonStockValue": "common_stock",
    "RetainedEarningsAccumulatedDeficit": "retained_earnings",
    "AccumulatedOtherComprehensiveIncomeLossNetOfTax": "accumulated_oci",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "total_stockholders_equity",
    "StockholdersEquity": "total_stockholders_equity",
    "LiabilitiesAndStockholdersEquity": "total_liabilities_and_equity",

    # Cash Flow Statement
    "DepreciationDepletionAndAmortization": "depreciation_and_amortization",
    "DepreciationAndAmortization": "depreciation_and_amortization",
    "DepreciationAmortizationAndAccretionNet": "depreciation_and_amortization",
    "Depreciation": "depreciation",
    "AmortizationOfIntangibleAssets": "amortization",
    "ShareBasedCompensation": "stock_based_compensation",
    "IncreaseDecreaseInAccountsReceivable": "changes_in_accounts_receivable",
    "IncreaseDecreaseInAccountsAndOtherReceivables": "changes_in_accounts_receivable",
    "IncreaseDecreaseInOtherReceivables": "changes_in_other_receivable",
    "IncreaseDecreaseInInventories": "changes_in_inventory",
    "IncreaseDecreaseInAccountsPayable": "changes_in_accounts_payable",
    "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities": "changes_in_accounts_payable",
    "IncreaseDecreaseInOtherOperatingAssets": "changes_in_other_operating_assets",
    "IncreaseDecreaseInOtherOperatingLiabilities": "changes_in_other_operating_liabilities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": "net_cash_from_operating_activities",
    "NetCashProvidedByUsedInOperatingActivities": "net_cash_from_operating_activities",
    

    
    "PaymentsToAcquirePropertyPlantAndEquipment": "capital_expenditures",
    "PaymentsToAcquireIntangibleAssets": "purchases_of_intangible_assets",
    "PaymentsToAcquireInvestments": "purchases_of_investments",
    "ProceedsFromSaleMaturityAndCollectionsOfInvestments": "sales_of_investments",
    "PaymentsToAcquireBusinessesNetOfCashAcquired": "acquisitions",
    "ProceedsFromDivestitureOfBusinesses": "divestitures",
    "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations": "net_cash_from_investing_activities",
    "NetCashProvidedByUsedInInvestingActivities": "net_cash_from_investing_activities",


    "ProceedsFromIssuanceOfLongTermDebt": "proceeds_from_debt",
    "RepaymentsOfLongTermDebt": "repayment_of_long_term_debt",
    "RepaymentsOfOtherShortTermDebt": "repayment_of_short_term_debt",
    "ProceedsFromRepaymentsOfCommercialPaper": "net_repayment_of_commercial_paper",
    "PaymentsOfDividendsCommonStock": "dividends_paid",
    "PaymentsOfDividends" : "dividends_paid",
    "PaymentsForRepurchaseOfCommonStock": "share_repurchases",
    "ProceedsFromIssuanceOfCommonStock": "proceeds_from_stock_issuance",
    "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations": "net_cash_from_financing_activities",
    "NetCashProvidedByUsedInFinancingActivities": "net_cash_from_financing_activities",

    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect": "net_change_in_cash"
}

# Combine all mappings
ALL_GAAP_MAPPINGS = {**US_GAAP_TO_DB_MAPPING}


# Computed GAAP Fields - Arithmetic operations on US GAAP codes to create database fields
# These computations are applied when direct GAAP mapping results in null/missing values
COMPUTED_GAAP_FIELDS = {
    # Net PP&E from Gross PP&E - Accumulated Depreciation
    'property_plant_equipment': [
        {
            'operation': 'subtract',
            'gaap_fields': ['PropertyPlantAndEquipmentGross', 'AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment'],
            'description': 'Net PP&E = Gross PP&E - Accumulated Depreciation'
        },
        {
            'operation': 'subtract',
            'gaap_fields': ['PropertyPlantAndEquipmentGross', 'AccumulatedDepreciation'],
            'description': 'Net PP&E = Gross PP&E - Accumulated Depreciation (alt)'
        },
        {
            'operation': 'subtract',
            'gaap_fields': ['PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization', 'OperatingLeaseRightOfUseAsset'],
            'description': 'Net PP&E = PP&E and RoU Assets - RoU Assets (alt)'
        },
    ],
    
    # Gross Profit from Revenue - COGS
    'gross_profit': [
        {
            'operation': 'subtract',
            'gaap_fields': ['Revenues', 'CostOfRevenue'],
            'description': 'Gross Profit = Revenue - COGS'
        },
        {
            'operation': 'subtract',
            'gaap_fields': ['RevenueFromContractWithCustomerExcludingAssessedTax', 'CostOfGoodsAndServicesSold'],
            'description': 'Gross Profit = Revenue - COGS (alt)'
        },
    ],
    
    # Total Operating Expenses from individual expense components
    'total_operating_expenses': [
        {
            'operation': 'add',
            'gaap_fields': ['ResearchAndDevelopmentExpense', 'SellingGeneralAndAdministrativeExpense'],
            'description': 'Total OpEx = R&D + SG&A'
        }
    ],

    # Total Long Term Debt
    # 'total_long_term_debt': [
    #     {
    #         'operation': 'add',
    #         'gaap_fields': ['LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths', 'LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo', 
    #                          'LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree', 'LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour',
    #                          'LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive', 'LongTermDebtMaturitiesRepaymentsOfPrincipalAfterFiveYears'],
    #         'description': 'Total OpEx = R&D + SG&A'
    #     }
    # ],

    # Long term investments from Marketable Securities + Other Investments
    'long_term_investments': [
        {
            'operation': 'add',
            'gaap_fields': ['MarketableSecuritiesNoncurrent', 'EquityMethodInvestments', 'OtherLongTermInvestments'],
            'description': 'Total long term investments = Marketable Securities + Equity Method Investments + Other Long Term Investments'
        }
    ],
    
    # Depreciation and Amortization from separate components
    'depreciation_and_amortization': [
        {
            'operation': 'add',
            'gaap_fields': ['Depreciation', 'AmortizationOfIntangibleAssets'],
            'description': 'Total Depreciation and Amortization = Depreciation + Amortization'
        },
        {
            'operation': 'add',
            'gaap_fields': ['Depreciation'],
            'description': 'Total Depreciation and Amortization = Depreciation'
        }
    ],

    # Operating Income from Gross Profit - Operating Expenses
    'operating_income': [
        {
            'operation': 'subtract',
            'gaap_fields': ['GrossProfit', 'OperatingExpenses'],
            'description': 'Operating Income = Gross Profit - OpEx'
        },
        {
            'operation': 'subtract',
            'gaap_fields': ['Revenues', 'CostOfRevenue', 'OperatingExpenses'],
            'description': 'Operating Income = Revenue - COGS - OpEx'
        },
        {
            'operation': 'subtract',
            'gaap_fields': ['Revenues', 'CostOfGoodsAndServicesSold', 'OperatingExpenses'],
            'description': 'Operating Income = Revenue - COGS - OpEx'
        },
    ],
    
    # Net Income from Income Before Tax - Tax Expense
    'net_income': [
        {
            'operation': 'subtract',
            'gaap_fields': ['IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 'IncomeTaxExpenseBenefit'],
            'description': 'Net Income = Income Before Tax - Tax Expense'
        },
    ],

    #Cash and ST Investments
    'cash_and_short_term_investments': [
        {
            'operation': 'add',
            'gaap_fields': ['CashAndCashEquivalentsAtCarryingValue', 'MarketableSecuritiesCurrent'],
            'description': 'Cash and ST Investments = Cash + Marketable Securities'
        },
    ], 

    #Long Term Debt Total
    'long_term_debt': [
        {
            'operation': 'add',
            'gaap_fields': ['LongTermDebtAndCapitalLeaseObligations', 'LongTermDebtAndCapitalLeaseObligationsCurrent'],
            'description': 'Total Long Term Debt = Long Term Debt + Current Portion of Long Term Debt'
        }
    ]

}


# Supported arithmetic operations
SUPPORTED_OPERATIONS = {
    'add': lambda values: sum(values),
    'subtract': lambda values: values[0] - sum(values[1:]) if len(values) > 1 else values[0],
    'multiply': lambda values: values[0] * values[1] if len(values) == 2 else None,
    'divide': lambda values: values[0] / values[1] if len(values) == 2 and values[1] != 0 else None,
}


class FinancialDataMapper:
    """Maps SEC XBRL data to database schema with conflict resolution"""
    
    def __init__(self):
        self.mapping = ALL_GAAP_MAPPINGS
        self.unmapped_items: Set[str] = set()
        self._reverse_mapping = None  # Cache for reverse mapping
    
    def get_reverse_mapping(self) -> Dict[str, Set[str]]:
        """
        Get reverse mapping: database column -> set of GAAP codes
        
        Returns:
            Dictionary mapping each database column to all GAAP codes that map to it
        """
        if self._reverse_mapping is None:
            self._reverse_mapping = {}
            for gaap_code, db_column in self.mapping.items():
                if db_column not in self._reverse_mapping:
                    self._reverse_mapping[db_column] = set()
                self._reverse_mapping[db_column].add(gaap_code)
        return self._reverse_mapping
    
    def get_multiple_mapping_columns(self) -> Dict[str, Set[str]]:
        """
        Get database columns that have multiple GAAP mappings
        
        Returns:
            Dictionary of db_column -> set of GAAP codes for columns with multiple mappings
        """
        reverse_mapping = self.get_reverse_mapping()
        return {db_col: gaap_codes for db_col, gaap_codes in reverse_mapping.items() 
                if len(gaap_codes) > 1}
    
    def resolve_value_conflicts(self, facts_dict: Dict[str, any], period_key: str = "") -> Dict[str, any]:
        """
        Resolve conflicts when multiple GAAP codes map to the same database column.
        Takes the highest value among conflicting mappings.
        
        Args:
            facts_dict: Dictionary of db_column -> value pairs
            period_key: Period identifier for logging purposes
            
        Returns:
            Dictionary with conflicts resolved (highest values retained)
        """
        # Get columns with multiple GAAP mappings
        multiple_mappings = self.get_multiple_mapping_columns()
        
        # Track which columns had conflicts resolved
        resolved_conflicts = {}
        
        # For each column that could have conflicts
        for db_column, gaap_codes in multiple_mappings.items():
            if db_column in facts_dict:
                # This column already has a value, but we need to check if there were multiple
                # GAAP codes that could have contributed to it
                continue
        
        # Since facts_dict already contains resolved values, we need to modify the processor
        # to call this method before storing values. For now, return as-is.
        return facts_dict
    
    def map_item(self, gaap_item: str) -> Optional[str]:
        """
        Map a US GAAP item to database column
        
        Args:
            gaap_item: US GAAP taxonomy item name
            
        Returns:
            Database column name or None if not mapped
        """
        db_column = self.mapping.get(gaap_item)
        
        if not db_column:
            # Try case-insensitive lookup
            for gaap_key, db_col in self.mapping.items():
                if gaap_key.lower() == gaap_item.lower():
                    db_column = db_col
                    break
        
        if not db_column:
            self.unmapped_items.add(gaap_item)
            logger.debug(f"Unmapped GAAP item: {gaap_item}")
        
        return db_column
    
    def get_all_mapped_items(self) -> Dict[str, str]:
        """Get all mapped items"""
        return self.mapping.copy()
    
    def get_unmapped_items(self) -> Set[str]:
        """Get items that couldn't be mapped"""
        return self.unmapped_items.copy()
    
    def add_custom_mapping(self, gaap_item: str, db_column: str):
        """
        Add a custom mapping
        
        Args:
            gaap_item: US GAAP taxonomy item name
            db_column: Database column name
        """
        self.mapping[gaap_item] = db_column
        logger.info(f"Added custom mapping: {gaap_item} -> {db_column}")
    
    def compute_missing_fields(self, facts_dict: Dict[str, any], all_period_facts: Dict[str, any], period_key: str = "") -> Dict[str, any]:
        """
        Compute missing database fields using arithmetic operations on already-mapped GAAP fields
        
        Args:
            facts_dict: Dictionary of db_column -> value pairs (after direct mapping)
            all_period_facts: All facts for the current period that have been processed through mapping
            period_key: Period identifier for logging purposes
            
        Returns:
            Updated facts_dict with computed values for previously null fields
        """
        computed_count = 0
        
        for db_column, computations in COMPUTED_GAAP_FIELDS.items():
            # Only compute if the field is currently missing/null
            if facts_dict.get(db_column) is not None:
                continue
            
            # Try each computation until we get a valid result
            for computation in computations:
                try:
                    operation = computation['operation']
                    gaap_fields = computation['gaap_fields']
                    description = computation.get('description', f"{operation} on {gaap_fields}")
                    
                    if operation not in SUPPORTED_OPERATIONS:
                        logger.warning(f"Unsupported operation '{operation}' for {db_column}")
                        continue
                    
                    # Map GAAP fields to database columns and extract values
                    values = []
                    missing_fields = []
                    
                    for gaap_field in gaap_fields:
                        # Map GAAP field to database column
                        mapped_db_column = self.map_item(gaap_field)
                        if mapped_db_column and mapped_db_column in all_period_facts:
                            value = all_period_facts[mapped_db_column]
                            if value is not None:
                                values.append(float(value))
                            else:
                                missing_fields.append(f"{gaap_field}→{mapped_db_column}")
                        else:
                            missing_fields.append(f"{gaap_field}→unmapped")
                    
                    # Only compute if we have all required values
                    if len(values) == len(gaap_fields):
                        operation_func = SUPPORTED_OPERATIONS[operation]
                        computed_value = operation_func(values)
                        
                        if computed_value is not None:
                            facts_dict[db_column] = computed_value
                            computed_count += 1
                            logger.debug(f"Computed {db_column} = {computed_value:,.0f} using {description} for period {period_key}")
                            break  # Success - move to next db_column
                    else:
                        logger.debug(f"Cannot compute {db_column} for period {period_key}: missing {missing_fields}")
                        
                except Exception as e:
                    logger.debug(f"Error computing {db_column} using {computation.get('description', operation)}: {e}")
                    continue
        
        if computed_count > 0:
            logger.info(f"Computed {computed_count} missing fields for period {period_key}")
        
        return facts_dict
    
    def compute_missing_fields_from_raw_gaap(self, facts_dict: Dict[str, any], raw_gaap_facts: Dict[str, any], period_key: str = "") -> Dict[str, any]:
        """
        Compute missing database fields using arithmetic operations on raw GAAP facts
        SIMPLE RULE: Only compute if direct mapping is missing/null. Never overwrite direct mappings.
        
        Args:
            facts_dict: Dictionary of db_column -> value pairs (after direct mapping)
            raw_gaap_facts: Raw US-GAAP facts data from SEC
            period_key: Period identifier for logging purposes
            
        Returns:
            Updated facts_dict with computed values for previously null fields
        """
        computed_count = 0
        
        for db_column, computations in COMPUTED_GAAP_FIELDS.items():
            # STRICT RULE: Only compute if the field is missing/null - never overwrite existing direct mappings
            current_value = facts_dict.get(db_column)
            if current_value is not None:
                logger.debug(f"Skipping computation for {db_column}: already has direct mapping value {current_value}")
                continue
            
            # Try each computation until we get a valid result
            for computation in computations:
                try:
                    operation = computation['operation']
                    gaap_fields = computation['gaap_fields']
                    description = computation.get('description', f"{operation} on {gaap_fields}")
                    
                    if operation not in SUPPORTED_OPERATIONS:
                        logger.warning(f"Unsupported operation '{operation}' for {db_column}")
                        continue
                    
                    # Extract values directly from raw GAAP facts for this period
                    values = []
                    missing_fields = []
                    
                    for gaap_field in gaap_fields:
                        value = self._extract_gaap_value(raw_gaap_facts, gaap_field, period_key)
                        if value is not None:
                            values.append(float(value))
                        else:
                            missing_fields.append(gaap_field)
                    
                    # Only compute if we have all required values
                    if len(values) == len(gaap_fields):
                        operation_func = SUPPORTED_OPERATIONS[operation]
                        computed_value = operation_func(values)
                        
                        if computed_value is not None:
                            facts_dict[db_column] = computed_value
                            # Mark as computed for consolidation priority logic
                            facts_dict[f"_computed_{db_column}"] = True
                            computed_count += 1
                            logger.debug(f"Computed {db_column} = {computed_value:,.0f} using {description} for period {period_key} [MARKED AS COMPUTED]")
                            break  # Success - move to next db_column
                    else:
                        logger.debug(f"Cannot compute {db_column} for period {period_key}: missing {missing_fields}")
                        
                except Exception as e:
                    logger.debug(f"Error computing {db_column} using {computation.get('description', operation)}: {e}")
                    continue
        
        if computed_count > 0:
            logger.info(f"Computed {computed_count} missing fields for period {period_key}")
        
        return facts_dict
    
    def _extract_gaap_value(self, gaap_facts: Dict[str, any], gaap_field: str, period_key: str = "") -> Optional[float]:
        """
        Extract value for a specific GAAP field from the raw facts data
        
        Args:
            gaap_facts: Raw GAAP facts data
            gaap_field: GAAP field name to extract
            period_key: Period identifier for context (format: end_date_filed_date_period_type_length)
            
        Returns:
            Numeric value or None if not found
        """
        if gaap_field not in gaap_facts:
            return None
        
        fact_data = gaap_facts[gaap_field]
        units = fact_data.get('units', {})
        
        # Extract period info from period_key for matching
        # Period key format: end_date_filed_date_period_type_period_length_months
        period_parts = period_key.split('_') if period_key else []
        target_end_date = period_parts[0] if len(period_parts) > 0 else None
        target_filed_date = period_parts[1] if len(period_parts) > 1 else None
        target_period_type = period_parts[2] if len(period_parts) > 2 else None
        target_period_length = period_parts[3] if len(period_parts) > 3 else None
        
        # Try USD first, then other numeric units
        for unit_type in ['USD', 'shares', 'pure', 'USD/shares']:
            if unit_type in units:
                entries = units[unit_type]
                if not entries:
                    continue
                
                # If we have period context, try to find matching entry
                if target_end_date and target_filed_date:
                    # Look for exact matches with period length consideration
                    exact_matches = []
                    for entry in entries:
                        if (entry.get('end') == target_end_date and 
                            entry.get('filed') == target_filed_date):
                            exact_matches.append(entry)
                    
                    # If we have multiple matches, try to find the one with matching period length
                    if len(exact_matches) > 1 and target_period_length:
                        from dateutil.parser import parse as parse_date
                        target_length_months = int(target_period_length)
                        
                        for entry in exact_matches:
                            start_date = entry.get('start')
                            end_date = entry.get('end')
                            
                            if start_date and end_date:
                                try:
                                    start_obj = parse_date(start_date).date()
                                    end_obj = parse_date(end_date).date()
                                    delta_days = (end_obj - start_obj).days
                                    
                                    # Convert days to months (approximately)
                                    actual_months = round(delta_days / 30.44)
                                    
                                    # Allow 1 month tolerance for rounding
                                    if abs(actual_months - target_length_months) <= 1:
                                        value = entry.get('val')
                                        if value is not None:
                                            return float(value)
                                except:
                                    continue
                    
                    # If no period-length match found, use first exact match
                    if exact_matches:
                        value = exact_matches[0].get('val')
                        if value is not None:
                            return float(value)
                    
                    # Fallback: match just end_date - REMOVED to prevent period length mixing
                    # This fallback was causing computed values to use wrong period lengths
                    # For example, 3-month gross profit using 6-month components
                    # Strict period matching ensures data integrity
                
                # Don't use fallback - return None if no period-specific match found
                # This prevents incorrect historical values from being applied to wrong periods
        
        return None

    def get_statement_type_from_column(self, db_column: str) -> str:
        """
        Determine statement type from database column
        
        Args:
            db_column: Database column name
            
        Returns:
            Statement type: 'income_statement', 'balance_sheet', or 'cash_flow'
        """
        # Income Statement columns
        income_columns = {
            'total_revenue', 'cost_of_revenue', 'gross_profit',
            'research_and_development', 'sales_and_marketing', 'sales_general_and_admin', 'general_and_administrative',
            'total_operating_expenses', 'operating_income', 'ebitda',
            'interest_income', 'interest_expense', 'other_income', 'income_before_taxes',
            'income_tax_expense', 'net_income', 'earnings_per_share_basic',
            'earnings_per_share_diluted', 'weighted_average_shares_basic',
            'weighted_average_shares_diluted'
        }
        
        # Balance Sheet columns
        balance_sheet_columns = {
            'cash_and_cash_equivalents', 'short_term_investments', 'cash_and_short_term_investments', 'accounts_receivable',
            'inventory', 'prepaid_expenses', 'total_current_assets', 'total_non_current_assets',
            'property_plant_equipment', 'goodwill', 'intangible_assets',
            'long_term_investments', 'other_assets', 'total_assets',
            'accounts_payable', 'accrued_liabilities', 'commercial_paper',
            'other_short_term_borrowings', 'current_portion_long_term_debt',
            'finance_lease_liability_current', 'operating_lease_liability_current',
            'total_current_liabilities', 'long_term_debt', 'non_current_long_term_debt', 'finance_lease_liability_noncurrent',
            'operating_lease_liability_noncurrent', 'other_long_term_liabilities',
            'total_non_current_liabilities', 'total_liabilities',
            'common_stock', 'retained_earnings', 'accumulated_oci',
            'total_stockholders_equity', 'total_liabilities_and_equity'
        }
        
        # Cash Flow columns
        cash_flow_columns = {
            'net_cash_from_operating_activities', 'depreciation_and_amortization', 'depreciation', 'amortization',
            'stock_based_compensation', 'changes_in_accounts_receivable', 'changes_in_other_receivable',
            'changes_in_inventory', 'changes_in_accounts_payable', 'changes_in_other_operating_assets',
            'changes_in_other_operating_liabilities', 'net_cash_from_investing_activities', 'capital_expenditures',
            'purchases_of_intangible_assets', 'purchases_of_investments',
            'sales_of_investments', 'acquisitions', 'divestitures', 'net_cash_from_financing_activities',
            'proceeds_from_debt', 'repayment_of_long_term_debt', 'repayment_of_short_term_debt',
            'net_repayment_of_commercial_paper', 'dividends_paid', 'share_repurchases',
            'proceeds_from_stock_issuance', 'net_change_in_cash'
        }
        
        if db_column in income_columns:
            return 'income_statement'
        elif db_column in balance_sheet_columns:
            return 'balance_sheet'
        elif db_column in cash_flow_columns:
            return 'cash_flow'
        else:
            return 'unknown'
    
    def get_statement_type_from_gaap_item(self, gaap_item: str) -> str:
        """
        Determine statement type directly from GAAP item name
        
        Args:
            gaap_item: US GAAP taxonomy item name
            
        Returns:
            Statement type: 'income_statement', 'balance_sheet', or 'cash_flow'
        """
        # Map the GAAP item to database column first
        db_column = self.map_item(gaap_item)
        if db_column:
            return self.get_statement_type_from_column(db_column)
        
        # If not mapped, try to infer from GAAP item name patterns
        gaap_lower = gaap_item.lower()
        
        # Balance sheet indicators
        balance_sheet_indicators = [
            'assets', 'liabilities', 'equity', 'stockholders', 'cash', 'receivable',
            'inventory', 'payable', 'debt', 'goodwill', 'intangible', 'property',
            'plant', 'equipment', 'investment', 'retained', 'earnings'
        ]
        
        # Income statement indicators  
        income_indicators = [
            'revenue', 'income', 'expense', 'cost', 'profit', 'loss', 'earnings',
            'operating', 'tax', 'interest', 'sales', 'marketing', 'research',
            'development', 'administrative'
        ]
        
        # Cash flow indicators
        cash_flow_indicators = [
            'cashflow', 'cash', 'financing', 'investing', 'operating', 'activities',
            'depreciation', 'amortization', 'capital', 'expenditure', 'dividend',
            'repurchase', 'acquisition'
        ]
        
        # Check for balance sheet patterns
        for indicator in balance_sheet_indicators:
            if indicator in gaap_lower:
                return 'balance_sheet'
        
        # Check for income statement patterns
        for indicator in income_indicators:
            if indicator in gaap_lower:
                return 'income_statement'
        
        # Check for cash flow patterns
        for indicator in cash_flow_indicators:
            if indicator in gaap_lower:
                return 'cash_flow'
        
        return 'unknown'
    
    def validate_mapping_coverage(self) -> Dict[str, int]:
        """
        Validate mapping coverage by statement type
        
        Returns:
            Dictionary with coverage statistics
        """
        income_mapped = sum(1 for col in self.mapping.values() 
                          if self.get_statement_type_from_column(col) == 'income_statement')
        balance_mapped = sum(1 for col in self.mapping.values() 
                           if self.get_statement_type_from_column(col) == 'balance_sheet')
        cash_flow_mapped = sum(1 for col in self.mapping.values() 
                             if self.get_statement_type_from_column(col) == 'cash_flow')
        
        return {
            'total_mappings': len(self.mapping),
            'income_statement_mappings': income_mapped,
            'balance_sheet_mappings': balance_mapped,
            'cash_flow_mappings': cash_flow_mapped,
            'unmapped_items': len(self.unmapped_items)
        }


# Global mapper instance
financial_mapper = FinancialDataMapper()


def get_mapper() -> FinancialDataMapper:
    """Get the global financial data mapper instance"""
    return financial_mapper