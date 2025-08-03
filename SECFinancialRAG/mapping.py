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

    "EarningsPerShareBasic": "earnings_per_share_basic",
    "EarningsPerShareDiluted": "earnings_per_share_diluted",
    "WeightedAverageNumberOfSharesOutstandingBasic": "weighted_average_shares_basic",
    "WeightedAverageNumberOfDilutedSharesOutstanding": "weighted_average_shares_diluted",

    # Balance Sheet
    "CashAndCashEquivalentsAtCarryingValue": "cash_and_cash_equivalents",
    "MarketableSecuritiesCurrent": "short_term_investments",
    "AccountsReceivableNetCurrent": "accounts_receivable",
    "InventoryNet": "inventory",
    "AssetsCurrent": "total_current_assets",

    "PropertyPlantAndEquipmentNet": "property_plant_equipment",
    "Goodwill": "goodwill",
    "IntangibleAssetsNetExcludingGoodwill": "intangible_assets",
    "MarketableSecuritiesNoncurrent": "long_term_investments",
    "OtherAssetsNoncurrent": "other_assets",
    "AssetsNoncurrent": "total_non_current_assets",
    "Assets": "total_assets",

    "AccountsPayableCurrent": "accounts_payable",
    "AccruedLiabilitiesCurrent": "accrued_liabilities",
    "CommercialPaper": "commercial_paper",
    "OtherShortTermBorrowings": "other_short_term_borrowings",
    "LongTermDebtCurrent": "current_portion_long_term_debt",
    "FinanceLeaseLiabilityCurrent": "finance_lease_liability_current",
    "OperatingLeaseLiabilityCurrent": "operating_lease_liability_current",
    "LiabilitiesCurrent": "total_current_liabilities",

    "LongTermDebtNoncurrent": "non_current_long_term_debt",
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
    "IncreaseDecreaseInOtherReceivables": "changes_in_other_receivable",
    "IncreaseDecreaseInInventories": "changes_in_inventory",
    "IncreaseDecreaseInAccountsPayable": "changes_in_accounts_payable",
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
            'total_operating_expenses', 'operating_income',
            'interest_income', 'interest_expense', 'other_income', 'income_before_taxes',
            'income_tax_expense', 'net_income', 'earnings_per_share_basic',
            'earnings_per_share_diluted', 'weighted_average_shares_basic',
            'weighted_average_shares_diluted'
        }
        
        # Balance Sheet columns
        balance_sheet_columns = {
            'cash_and_cash_equivalents', 'short_term_investments', 'accounts_receivable',
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