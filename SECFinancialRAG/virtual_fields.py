"""
Virtual Fields Configuration
Handles inconsistent financial line items across companies by providing fallback logic
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Configuration constants
TAX_RATE = 0.26  # 26% corporate tax rate - can be modified as needed

# Virtual fields configuration - maps virtual field names to ordered list of fallback expressions
VIRTUAL_FIELDS = {
    # SG&A Expense - handles different reporting levels
    "cogs": [
        "cost_of_revenue",  # Direct mapping
        "total_revenue - gross_profit",  # Sum components 
    ],

    "gp": [
        "gross_profit",  # Direct mapping
        "total_revenue - cogs",  # Sum components 
    ],

    # SG&A Expense - handles different reporting levels
    "sga_expense": [
        "sales_general_and_admin",  # Direct mapping
        "sales_and_marketing + COALESCE(general_and_administrative, 0)",  # Sum components 
        "sales_and_marketing",  # Only S&M available
        "general_and_administrative"  # Only G&A available
    ],
    
    # Total Operating Expenses - comprehensive fallback using virtual fields
    "total_op_expense": [
        "total_operating_expenses + cogs",  # Direct mapping
        "total_revenue - operating_income",  # Sum components
        "cost_of_revenue + COALESCE(sga_expense, 0) + COALESCE(research_and_development, 0)"  # Using virtual field
    ],
    
    # Net Interest Expense - handles interest income/expense inconsistencies
    "net_interest_expense": [
        "COALESCE(interest_expense, 0) - COALESCE(interest_income, 0)"
    ],

    # Depreciation and Amortization - handles different reporting structures
    "d&a": [
        "depreciation_and_amortization",  # Direct mapping
        "COALESCE(depreciation, 0) + COALESCE(amortization, 0)",  # Sum components
    ],

    # EBITDA (since we don't have direct EBITDA in database)
    "ebitda": [
        "total_revenue - total_op_expense + d&a",
        "total_revenue - total_op_expense"  # Fallback without D&A
    ],

    "ebit": [
        "total_revenue - total_op_expense"
    ],

    "pbt": [
        "total_revenue - total_op_expense + COALESCE(other_income, 0) + COALESCE(interest_income, 0) - COALESCE(interest_expense, 0)"
    ],

    "cash_and_st_investments": [
        "cash_and_cash_equivalents + COALESCE(short_term_investments, 0)",
        "cash_and_short_term_investments" 
    ],

    "cash_and_st_investments_and_lt_investments": [
        "cash_and_cash_equivalents + COALESCE(short_term_investments, 0) + COALESCE(long_term_investments, 0)"
    ],

    "non_current_assets": [
        "total_non_current_assets",
        "total_assets - COALESCE(total_current_assets, 0)"
    ],

    # Long-term debt - handles different reporting structures
    "total_long_term_debt": [
        "long_term_debt + COALESCE(finance_lease_liability_noncurrent, 0)",  # Most common
        "COALESCE(non_current_long_term_debt, 0) + COALESCE(current_portion_long_term_debt, 0) + COALESCE(finance_lease_liability_noncurrent, 0)"  # Include current portion
    ],
    
    # Total debt - comprehensive debt calculation
    "total_debt": [
        "total_long_term_debt + COALESCE(commercial_paper, 0) + COALESCE(other_short_term_borrowings, 0) + COALESCE(finance_lease_liability_current, 0) + COALESCE(finance_lease_liability_noncurrent, 0)"
    ],
    
    "total_debt_incl_oper_lease": [
        "total_debt + COALESCE(operating_lease_liability_current, 0) + COALESCE(operating_lease_liability_noncurrent, 0)"
    ],

    # Working Capital
    "working_capital": [
        "total_current_assets - total_current_liabilities"
    ],
    
    # Net Working Capital (excluding cash)
    "net_working_capital": [
        "working_capital - cash_and_st_investments"
    ],
    
    # Free Cash Flow
    "free_cash_flow": [
        "net_cash_from_operating_activities - capital_expenditures"
    ],
    
    "net_acquisitions": [
        "COALESCE(acquisitions, 0) - COALESCE(divestitures, 0)"
    ],

    "net_repurchases": [
        "COALESCE(share_repurchases, 0) - COALESCE(proceeds_from_stock_issuance, 0)"
    ],

    # Interest-bearing debt
    "interest_bearing_debt": [
        "total_debt_incl_oper_lease"
    ],
    
    # Total invested capital
    "invested_capital": [
        "total_stockholders_equity + total_debt ",
    ],

    "net_invested_capital": [
        "total_stockholders_equity + total_debt - cash_and_st_investments",
    ],
    
    # Book value per share (needs shares outstanding)
    "book_value": [
        "total_stockholders_equity"
    ],
    
    # Tangible book value
    "tangible_book_value": [
        "total_stockholders_equity - COALESCE(goodwill, 0) - COALESCE(intangible_assets, 0)",
    ]
}

# Standard ratio definitions using virtual fields
DEFAULT_RATIOS = {
    # Profitability Ratios
    "ROE": {
        "formula": "net_income / total_stockholders_equity",
        "description": "Return on Equity - Net income as % of shareholders' equity",
        "category": "profitability"
    },
    "ROA": {
        "formula": f"(ebit * (1 - {TAX_RATE})) / total_assets", 
        "description": f"Return on Assets - NOPAT as % of total assets (using {TAX_RATE*100}% tax rate)",
        "category": "profitability"
    },
    "ROIC": {
        "formula": f"(ebit * (1 - {TAX_RATE})) / invested_capital",
        "description": f"Return on Invested Capital - NOPAT as % of invested capital (using {TAX_RATE*100}% tax rate)",
        "category": "profitability"
    },
    "Net_ROIC": {
        "formula": f"(ebit * (1 - {TAX_RATE})) / net_invested_capital",
        "description": f"Return on Invested Capital (Ex. Cash) - NOPAT as % of net invested capital (using {TAX_RATE*100}% tax rate)",
        "category": "profitability"
    },
    "Net_Margin": {
        "formula": "net_income / total_revenue",
        "description": "Net Profit Margin - Net income as % of revenue", 
        "category": "profitability"
    },
    "EBIT_Margin": {
        "formula": "ebit / total_revenue",
        "description": "Operating Margin - Operating income as % of revenue",
        "category": "profitability" 
    },
    "EBITDA_Margin": {
        "formula": "ebitda / total_revenue",
        "description": "EBITDA Margin - EBITDA as % of revenue",
        "category": "profitability" 
    },
    "Gross_Margin": {
        "formula": "gp / total_revenue",
        "description": "Gross Margin - Gross profit as % of revenue",
        "category": "profitability"
    },
    "SGA_Margin": {
        "formula": "sga_expense / total_revenue",
        "description": "SG&A Margin - SG&A expense as % of revenue",
        "category": "profitability"
    },
    "NOPAT_Margin": {
        "formula": f"(ebit * (1 - {TAX_RATE})) / total_revenue",
        "description": f"NOPAT Margin - Net Operating Profit After Tax as % of revenue (using {TAX_RATE*100}% tax rate)",
        "category": "profitability"
    },
    
    # Liquidity Ratios
    "Current_Ratio": {
        "formula": "total_current_assets / total_current_liabilities",
        "description": "Current Ratio - Current assets divided by current liabilities",
        "category": "liquidity"
    },
    "Quick_Ratio": {
        "formula": "(cash_and_st_investments + accounts_receivable) / total_current_liabilities",
        "description": "Quick Ratio - Liquid assets divided by current liabilities",
        "category": "liquidity"
    },
    "Cash_Ratio": {
        "formula": "(cash_and_st_investments) / total_current_liabilities",
        "description": "Cash Ratio - Cash and short-term investments divided by current liabilities",
        "category": "liquidity"
    },
    "Cash_OpEx_Ratio": {
        "formula": "(cash_and_st_investments) / total_op_expense",
        "description": "Cash OpEx Ratio - Cash and short-term investments divided by operating expenses",
        "category": "liquidity"
    },
    "Cash_EBITDA_Ratio": {
        "formula": "(cash_and_st_investments) / ebitda",
        "description": "Cash EBITDA Ratio - Cash and short-term investments divided by EBITDA",
        "category": "liquidity"
    },
    "Cash_Sales_Ratio": {
        "formula": "(cash_and_st_investments) / total_revenue",
        "description": "Cash Sales Ratio - Cash and short-term investments divided by total revenue",
        "category": "liquidity"
    },
    
    # Leverage Ratios  
    "Debt_to_EBITDA": {
        "formula": "total_debt / ebitda",
        "description": "Debt-to-EBITDA Ratio - Total debt (excl. operating leases) divided by EBITDA",
        "category": "leverage"
    },
    "Net_Debt_to_EBITDA": {
        "formula": "(total_debt - cash_and_st_investments) / ebitda",
        "description": "Net Debt-to-EBITDA Ratio - Net debt (excl. operating leases) divided by EBITDA",
        "category": "leverage"
    },
    "Debt_to_Equity": {
        "formula": "total_debt / total_stockholders_equity",
        "description": "Debt-to-Equity Ratio - Total debt (excl. operating leases) divided by shareholders' equity",
        "category": "leverage"
    },
    "Debt_to_Assets": {
        "formula": "total_debt / total_assets",
        "description": "Debt-to-Assets Ratio - Total debt divided by total assets", 
        "category": "leverage"
    },
    "Equity_Ratio": {
        "formula": "total_stockholders_equity / total_assets",
        "description": "Equity Ratio - Shareholders' equity as % of total assets",
        "category": "leverage"
    },
    "EBIT_Interest_Coverage": {
        "formula": "ebit / net_interest_expense",
        "description": "Interest Coverage Ratio - EBIT divided by net interest expense",
        "category": "leverage"
    },
    "EBITDA_Interest_Coverage": {
        "formula": "ebitda / net_interest_expense",
        "description": "Interest Coverage Ratio - EBITDA divided by net interest expense",
        "category": "leverage"
    },
    
    # Efficiency Ratios
    "Asset_Turnover": {
        "formula": "total_revenue / total_assets",
        "description": "Asset Turnover - Revenue divided by average total assets",
        "category": "efficiency"
    },
    "Invested_Capital_Turnover": {
        "formula": "total_revenue / invested_capital",
        "description": "Invested Capital Turnover - Revenue divided by invested capital",
        "category": "efficiency"
    },
    "Fixed_Asset_Turnover": {
        "formula": "total_revenue / property_plant_equipment",
        "description": "Fixed Asset Turnover - Revenue divided by net property, plant and equipment",
        "category": "efficiency"
    },
    "Inventory_Turnover": {
        "formula": "cogs / inventory",
        "description": "Inventory Turnover - Cost of goods sold divided by average inventory",
        "category": "efficiency"
    },
    "Receivables_Turnover": {
        "formula": "total_revenue / accounts_receivable", 
        "description": "Receivables Turnover - Revenue divided by average accounts receivable",
        "category": "efficiency"
    },
    
    # Cash Flow Ratios
    "Operating_Cash_Margin": {
        "formula": "net_cash_from_operating_activities / total_revenue",
        "description": "Operating Cash Flow Margin - Operating cash flow as % of revenue",
        "category": "cash_flow"
    },
    "Free_Cash_Flow_Margin": {
        "formula": "free_cash_flow / total_revenue", 
        "description": "Free Cash Flow Margin - Free cash flow as % of revenue",
        "category": "cash_flow"
    },
    "Cash_Return_on_Assets": {
        "formula": "net_cash_from_operating_activities / total_assets",
        "description": "Cash Return on Assets - Operating cash flow divided by total assets",
        "category": "cash_flow"
    },
    
    # Growth Ratios (Year-over-Year)
    "Revenue_Growth_YoY": {
        "formula": "YOY_GROWTH(total_revenue)",
        "description": "Revenue Growth YoY - Year-over-year growth in total revenue",
        "category": "growth"
    },
    "EBITDA_Growth_YoY": {
        "formula": "YOY_GROWTH(ebitda)",
        "description": "EBITDA Growth YoY - Year-over-year growth in EBITDA",
        "category": "growth"
    }
}


class VirtualFieldResolver:
    """Resolves virtual fields using fallback logic"""
    
    def __init__(self):
        self.virtual_fields = VIRTUAL_FIELDS
        self.resolved_cache = {}  # Cache for performance
    
    def resolve_virtual_fields(self, financial_data: Dict) -> Dict:
        """
        Resolve virtual fields in financial data with recursive resolution
        
        Args:
            financial_data: Dictionary of financial statement data
            
        Returns:
            Enhanced dictionary with virtual fields resolved
        """
        resolved_data = financial_data.copy()
        
        # Recursive resolution with dependency tracking
        max_iterations = 10  # Increase iterations for complex dependency chains
        for iteration in range(max_iterations):
            initial_count = len([k for k, v in resolved_data.items() if k in self.virtual_fields and v is not None])
            logger.debug(f"Virtual field resolution iteration {iteration + 1}: {initial_count} fields resolved")
            
            for virtual_field, source_expressions in self.virtual_fields.items():
                if virtual_field not in resolved_data or resolved_data[virtual_field] is None:
                    # Try each source expression until we get a non-null value with valid components
                    for expression in source_expressions:
                        try:
                            # Check if expression has valid components before evaluating
                            if not self._expression_has_valid_components(expression, resolved_data):
                                logger.debug(f"Skipping {expression} for {virtual_field}: missing components")
                                continue
                                
                            value = self._evaluate_expression(expression, resolved_data)
                            if value is not None:  # Accept all valid values including zero
                                resolved_data[virtual_field] = value
                                logger.debug(f"Resolved {virtual_field} = {value} using: {expression}")
                                break
                        except Exception as e:
                            logger.debug(f"Failed to evaluate {expression} for {virtual_field}: {e}")
                            continue
            
            # Check if we made progress
            final_count = len([k for k, v in resolved_data.items() if k in self.virtual_fields and v is not None])
            logger.debug(f"After iteration {iteration + 1}: {final_count} fields resolved")
            if final_count == initial_count:
                break  # No more progress possible
                
        return resolved_data
    
    def _evaluate_expression(self, expression: str, data: Dict) -> float:
        """
        Safely evaluate a mathematical expression with financial data
        
        Args:
            expression: Mathematical expression string
            data: Financial data dictionary
            
        Returns:
            Calculated value or None if evaluation fails
        """
        try:
            # Replace field names with values, handling COALESCE for NULL values
            eval_expr = expression
            
            # Handle COALESCE function
            import re
            coalesce_pattern = r'COALESCE\(([^,]+),\s*([^)]+)\)'
            while re.search(coalesce_pattern, eval_expr):
                def replace_coalesce(match):
                    field = match.group(1).strip()
                    default = match.group(2).strip()
                    value = data.get(field)
                    if value is None or value == 0:
                        return default
                    return str(value)
                eval_expr = re.sub(coalesce_pattern, replace_coalesce, eval_expr)
            
            # Replace remaining field names with actual values
            # Sort by length descending to avoid partial replacements
            field_items = sorted(data.items(), key=lambda x: len(x[0]), reverse=True)
            for field_name, value in field_items:
                # Use word boundaries to avoid partial replacements
                import re
                pattern = r'\b' + re.escape(field_name) + r'\b'
                if re.search(pattern, eval_expr):
                    if value is None:
                        eval_expr = re.sub(pattern, "0", eval_expr)
                    else:
                        eval_expr = re.sub(pattern, str(value), eval_expr)
            
            # Evaluate the expression safely
            # In production, consider using a safer expression evaluator
            result = eval(eval_expr)
            return float(result) if result is not None else None
            
        except Exception as e:
            logger.debug(f"Error evaluating expression '{expression}': {e}")
            return None
    
    def _expression_has_valid_components(self, expression: str, data: Dict) -> bool:
        """
        Check if expression has all required components available and non-null
        Only validates fields OUTSIDE of COALESCE - fields inside COALESCE can be null
        
        Args:
            expression: Mathematical expression string
            data: Financial data dictionary
            
        Returns:
            True if all non-COALESCE components are available and non-null
        """
        try:
            import re
            
            # Remove COALESCE functions and their contents from validation
            # COALESCE(field, default) -> fields inside can be null, so exclude from validation
            expr_for_validation = expression
            coalesce_pattern = r'COALESCE\([^)]+\)'
            expr_for_validation = re.sub(coalesce_pattern, '', expr_for_validation)
            
            # Extract field names from the expression (excluding COALESCE contents)
            field_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            potential_fields = re.findall(field_pattern, expr_for_validation)
            
            # Filter out mathematical operators and functions
            operators = {'COALESCE', 'and', 'or', 'not', 'if', 'else', 'def', 'return', 'import', 'from', 'as'}
            field_names = [f for f in potential_fields if f not in operators and not f.isdigit()]
            
            # Check that all non-COALESCE fields exist and are non-null
            for field_name in field_names:
                if field_name in data:
                    value = data[field_name]
                    if value is None:
                        logger.debug(f"Rejecting expression '{expression}': field '{field_name}' is null")
                        return False
                else:
                    # Field not present in data
                    logger.debug(f"Rejecting expression '{expression}': field '{field_name}' not found in data")
                    return False
            
            # Must have at least one field to validate (or be an expression without field dependencies)
            if len(field_names) == 0:
                # Check if this is just a COALESCE expression or a constant
                if 'COALESCE' in expression or expression.replace(' ', '').replace('.', '').isdigit():
                    return True
                logger.debug(f"Rejecting expression '{expression}': no fields found")
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Error checking expression components '{expression}': {e}")
            return False
    
    def get_available_virtual_fields(self) -> List[str]:
        """Get list of available virtual field names"""
        return list(self.virtual_fields.keys())
    
    def add_virtual_field(self, name: str, source_expressions: List[str]):
        """Add a custom virtual field"""
        self.virtual_fields[name] = source_expressions
        logger.info(f"Added virtual field: {name}")


# Global instance
virtual_field_resolver = VirtualFieldResolver()

def get_virtual_field_resolver() -> VirtualFieldResolver:
    """Get the global virtual field resolver instance"""
    return virtual_field_resolver