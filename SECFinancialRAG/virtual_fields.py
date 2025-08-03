"""
Virtual Fields Configuration
Handles inconsistent financial line items across companies by providing fallback logic
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Virtual fields configuration - maps virtual field names to ordered list of fallback expressions
VIRTUAL_FIELDS = {
    # SG&A Expense - handles different reporting levels
    "sga_expense": [
        "sales_general_and_admin",  # Direct mapping (AAPL style)
        "sales_and_marketing + COALESCE(general_and_administrative, 0)",  # Sum components (MSFT style)
        "sales_and_marketing",  # Only S&M available
        "general_and_administrative"  # Only G&A available
    ],
    
    # Total Operating Expenses - comprehensive fallback
    "total_operating_expense": [
        "total_operating_expenses",  # Direct mapping
        "cost_of_revenue + COALESCE(sales_general_and_admin, 0) + COALESCE(research_and_development, 0)",
        "cost_of_revenue + COALESCE(sales_and_marketing, 0) + COALESCE(general_and_administrative, 0) + COALESCE(research_and_development, 0)",
        "COALESCE(sales_general_and_admin, 0) + COALESCE(research_and_development, 0)",
        "COALESCE(sales_and_marketing, 0) + COALESCE(general_and_administrative, 0) + COALESCE(research_and_development, 0)"
    ],
    
    # Long-term debt - handles different reporting structures
    "total_long_term_debt": [
        "long_term_debt",  # Most common
        "non_current_long_term_debt",  # Alternative mapping
        "long_term_debt + COALESCE(current_portion_long_term_debt, 0)"  # Include current portion
    ],
    
    # Total debt - comprehensive debt calculation
    "total_debt": [
        "long_term_debt + COALESCE(current_portion_long_term_debt, 0) + COALESCE(commercial_paper, 0) + COALESCE(other_short_term_borrowings, 0)",
        "long_term_debt + COALESCE(current_portion_long_term_debt, 0)",
        "long_term_debt",
        "non_current_long_term_debt + COALESCE(current_portion_long_term_debt, 0)"
    ],
    
    # Working Capital
    "working_capital": [
        "total_current_assets - total_current_liabilities"
    ],
    
    # Net Working Capital (excluding cash)
    "net_working_capital": [
        "(total_current_assets - cash_and_cash_equivalents) - total_current_liabilities",
        "total_current_assets - total_current_liabilities - cash_and_cash_equivalents"
    ],
    
    # EBITDA proxy (since we don't have direct EBITDA in database)
    "ebitda_proxy": [
        "operating_income + depreciation_and_amortization",
        "operating_income + COALESCE(depreciation, 0) + COALESCE(amortization, 0)",
        "operating_income"  # Fallback without D&A
    ],
    
    # Free Cash Flow
    "free_cash_flow": [
        "net_cash_from_operating_activities - capital_expenditures",
        "net_cash_from_operating_activities + capital_expenditures"  # In case capex is negative
    ],
    
    # Interest-bearing debt
    "interest_bearing_debt": [
        "long_term_debt + COALESCE(current_portion_long_term_debt, 0) + COALESCE(other_short_term_borrowings, 0)",
        "long_term_debt + COALESCE(current_portion_long_term_debt, 0)",
        "long_term_debt"
    ],
    
    # Total invested capital
    "invested_capital": [
        "total_stockholders_equity + long_term_debt + COALESCE(current_portion_long_term_debt, 0)",
        "total_stockholders_equity + long_term_debt",
        "total_assets - total_current_liabilities + COALESCE(current_portion_long_term_debt, 0)"
    ],
    
    # Book value per share (needs shares outstanding)
    "book_value": [
        "total_stockholders_equity"
    ],
    
    # Tangible book value
    "tangible_book_value": [
        "total_stockholders_equity - COALESCE(goodwill, 0) - COALESCE(intangible_assets, 0)",
        "total_stockholders_equity - COALESCE(goodwill, 0)",
        "total_stockholders_equity"
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
        "formula": "net_income / total_assets", 
        "description": "Return on Assets - Net income as % of total assets",
        "category": "profitability"
    },
    "ROIC": {
        "formula": "net_income / invested_capital",
        "description": "Return on Invested Capital",
        "category": "profitability"
    },
    "Net_Margin": {
        "formula": "net_income / total_revenue",
        "description": "Net Profit Margin - Net income as % of revenue", 
        "category": "profitability"
    },
    "Operating_Margin": {
        "formula": "operating_income / total_revenue",
        "description": "Operating Margin - Operating income as % of revenue",
        "category": "profitability" 
    },
    "Gross_Margin": {
        "formula": "gross_profit / total_revenue",
        "description": "Gross Margin - Gross profit as % of revenue",
        "category": "profitability"
    },
    "SGA_Margin": {
        "formula": "sga_expense / total_revenue",
        "description": "SG&A Margin - SG&A expense as % of revenue",
        "category": "profitability"
    },
    
    # Liquidity Ratios
    "Current_Ratio": {
        "formula": "total_current_assets / total_current_liabilities",
        "description": "Current Ratio - Current assets divided by current liabilities",
        "category": "liquidity"
    },
    "Quick_Ratio": {
        "formula": "(cash_and_cash_equivalents + COALESCE(short_term_investments, 0) + accounts_receivable) / total_current_liabilities",
        "description": "Quick Ratio - Liquid assets divided by current liabilities",
        "category": "liquidity"
    },
    "Cash_Ratio": {
        "formula": "(cash_and_cash_equivalents + COALESCE(short_term_investments, 0)) / total_current_liabilities",
        "description": "Cash Ratio - Cash and equivalents divided by current liabilities",
        "category": "liquidity"
    },
    
    # Leverage Ratios  
    "Debt_to_Equity": {
        "formula": "total_debt / total_stockholders_equity",
        "description": "Debt-to-Equity Ratio - Total debt divided by shareholders' equity",
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
    "Interest_Coverage": {
        "formula": "operating_income / interest_expense",
        "description": "Interest Coverage Ratio - Operating income divided by interest expense",
        "category": "leverage"
    },
    
    # Efficiency Ratios
    "Asset_Turnover": {
        "formula": "total_revenue / total_assets",
        "description": "Asset Turnover - Revenue divided by average total assets",
        "category": "efficiency"
    },
    "Inventory_Turnover": {
        "formula": "cost_of_revenue / inventory",
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
    }
}


class VirtualFieldResolver:
    """Resolves virtual fields using fallback logic"""
    
    def __init__(self):
        self.virtual_fields = VIRTUAL_FIELDS
        self.resolved_cache = {}  # Cache for performance
    
    def resolve_virtual_fields(self, financial_data: Dict) -> Dict:
        """
        Resolve virtual fields in financial data
        
        Args:
            financial_data: Dictionary of financial statement data
            
        Returns:
            Enhanced dictionary with virtual fields resolved
        """
        resolved_data = financial_data.copy()
        
        for virtual_field, source_expressions in self.virtual_fields.items():
            if virtual_field not in resolved_data:
                # Try each source expression until we get a non-null value
                for expression in source_expressions:
                    try:
                        value = self._evaluate_expression(expression, financial_data)
                        if value is not None and value != 0:  # Accept non-zero values
                            resolved_data[virtual_field] = value
                            logger.debug(f"Resolved {virtual_field} = {value} using: {expression}")
                            break
                    except Exception as e:
                        logger.debug(f"Failed to evaluate {expression} for {virtual_field}: {e}")
                        continue
        
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
            for field_name, value in data.items():
                if field_name in eval_expr:
                    if value is None:
                        eval_expr = eval_expr.replace(field_name, "0")
                    else:
                        eval_expr = eval_expr.replace(field_name, str(value))
            
            # Evaluate the expression safely
            # In production, consider using a safer expression evaluator
            result = eval(eval_expr)
            return float(result) if result is not None else None
            
        except Exception as e:
            logger.debug(f"Error evaluating expression '{expression}': {e}")
            return None
    
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