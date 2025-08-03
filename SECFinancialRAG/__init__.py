"""
SEC Financial RAG Package
Fetches financial statements from SEC API and stores in PostgreSQL database
"""

from .sec_client import SECClient
from .database import FinancialDatabase
from .models import Company, IncomeStatement, BalanceSheet, CashFlowStatement, ProcessingMetadata
from .simplified_processor import SimplifiedSECFinancialProcessor
from .main import (
    process_company_financials, 
    process_multiple_companies,
    calculate_company_ltm,
    export_ltm_data,
    get_company_summary,
    validate_company_data,
    test_sec_connection,
    test_database_connection,
    get_financial_statements_df,
    export_financial_data,
    get_system_status
)
from .ltm_calculator import LTMCalculator, calculate_ltm_income_statement, calculate_ltm_cash_flow, get_all_ltm_data

__version__ = "1.0.0"

# Create backwards compatibility aliases
SECFinancialProcessor = SimplifiedSECFinancialProcessor
get_financial_metrics = get_financial_statements_df  # Alias for backwards compatibility
get_ltm_metrics = calculate_company_ltm  # Alias for backwards compatibility

__all__ = [
    "SECClient",
    "FinancialDatabase", 
    "Company",
    "IncomeStatement",
    "BalanceSheet", 
    "CashFlowStatement",
    "ProcessingMetadata",
    "SECFinancialProcessor",
    "SimplifiedSECFinancialProcessor",
    "process_company_financials",
    "process_multiple_companies",
    "calculate_company_ltm",
    "export_ltm_data",
    "get_company_summary",
    "validate_company_data",
    "test_sec_connection",
    "test_database_connection",
    "get_financial_statements_df",
    "export_financial_data",
    "get_system_status",
    "LTMCalculator",
    "calculate_ltm_income_statement",
    "calculate_ltm_cash_flow", 
    "get_all_ltm_data",
    "get_financial_metrics",  # Alias
    "get_ltm_metrics"  # Alias
]