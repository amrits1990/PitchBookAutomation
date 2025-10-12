"""
Agent Interface for SECFinancialRAG
Provides standardized JSON responses for AI agents
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime
import logging
import json
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import sys

# Add current directory to path to ensure local module imports work
_current_file_dir = Path(__file__).parent
if str(_current_file_dir) not in sys.path:
    sys.path.insert(0, str(_current_file_dir))

logger = logging.getLogger(__name__)

# Load environment variables from parent directory
_current_dir = Path(__file__).parent
_parent_env_file = _current_dir.parent / '.env'
if _parent_env_file.exists():
    load_dotenv(_parent_env_file)
    logger.debug(f"Loaded environment from {_parent_env_file}")
else:
    logger.warning(f"Parent .env file not found at {_parent_env_file}")

# Error codes for standardized responses
class ErrorCodes:
    INVALID_TICKER = "INVALID_TICKER"
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    INVALID_PERIOD = "INVALID_PERIOD"
    INVALID_METRIC = "INVALID_METRIC"
    DATABASE_ERROR = "DATABASE_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"

@dataclass
class FinancialAgentResponse:
    """Standardized response format for agent interactions"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Add correlation ID for tracing
        if 'correlation_id' not in self.metadata:
            self.metadata['correlation_id'] = str(uuid.uuid4())
        
        # Add timestamp
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.utcnow().isoformat()

def create_success_response(data: Dict[str, Any], metadata: Dict[str, Any] = None) -> FinancialAgentResponse:
    """Create a successful response"""
    return FinancialAgentResponse(
        success=True,
        data=data,
        metadata=metadata or {}
    )

def create_error_response(error_code: str, error_message: str, 
                         details: Dict[str, Any] = None, metadata: Dict[str, Any] = None) -> FinancialAgentResponse:
    """Create an error response"""
    error_dict = {
        'code': error_code,
        'message': error_message
    }
    if details:
        error_dict['details'] = details
    
    return FinancialAgentResponse(
        success=False,
        error=error_dict,
        metadata=metadata or {}
    )

def validate_ticker(ticker: str) -> Optional[str]:
    """Validate ticker format. Returns None if valid, error message if invalid."""
    if not ticker:
        return "Ticker cannot be empty"
    if not isinstance(ticker, str):
        return "Ticker must be a string"
    if len(ticker) < 1 or len(ticker) > 10:
        return "Ticker must be 1-10 characters"
    if not ticker.replace('.', '').replace('-', '').isalnum():
        return "Ticker contains invalid characters"
    return None

def validate_period(period: str) -> Optional[str]:
    """Validate period format. Returns None if valid, error message if invalid."""
    import re
    
    # Basic period types (only latest allowed)
    if period == 'latest':
        return None
    
    # Specific year patterns: FY2024, Q1-2024, Q2-2024, Q3-2024, Q4-2024
    year_pattern = r'^(FY|Q[1-4]-?)(\d{4})$'
    if re.match(year_pattern, period):
        year = int(period[-4:])
        if 2000 <= year <= 2030:  # Reasonable year range
            return None
        else:
            return f"Year must be between 2000 and 2030, got {year}"
    
    # Trend patterns: last n quarters, last n financial years (quarters: n=1-40, years: n=1-10)
    trend_pattern = r'^last (\d+) (quarters|financial years)$'
    match = re.match(trend_pattern, period)
    if match:
        count = int(match.group(1))
        unit = match.group(2)
        if unit == "quarters":
            if 1 <= count <= 40:
                return None
            else:
                return f"Count for quarters must be between 1 and 40, got {count}"
        elif unit == "financial years":
            if 1 <= count <= 10:
                return None
            else:
                return f"Count for financial years must be between 1 and 10, got {count}"
        else:
            return f"Invalid period unit: {unit}"
    
    return f"Invalid period format. Supported: latest, FY2024, Q1-2024, last n quarters (n=1-40), last n financial years (n=1-10)"

def validate_metrics(metrics: List[str], available_metrics: List[str] = None) -> Optional[str]:
    """Validate metric names. Returns None if valid, error message if invalid."""
    if not metrics:
        return "Metrics list cannot be empty"
    if not isinstance(metrics, list):
        return "Metrics must be a list"
    
    # Basic validation - specific metric validation can be added later
    for metric in metrics:
        if not isinstance(metric, str):
            return "All metrics must be strings"
        if not metric.strip():
            return "Metric names cannot be empty"
    
    return None

# Database Schema for LLM-based field mapping
DATABASE_SCHEMA = {
    "income_statements": {
        "description": "Contains income statement data (profit & loss) for companies",
        "financial_columns": {
            "total_revenue": "Total company revenue/sales",
            "cost_of_revenue": "Cost of goods sold (COGS)",
            "gross_profit": "Revenue minus cost of revenue",
            "research_and_development": "R&D expenses",
            "sales_and_marketing": "Sales and marketing expenses",
            "sales_general_and_admin": "SG&A expenses",
            "general_and_administrative": "General administrative expenses",
            "total_operating_expenses": "Total operating expenses",
            "operating_income": "Operating profit (EBIT equivalent)",
            "interest_income": "Interest income earned",
            "interest_expense": "Interest expenses paid",
            "other_income": "Other non-operating income",
            "income_before_taxes": "Pre-tax income",
            "income_tax_expense": "Tax expense",
            "net_income": "Net profit after all expenses and taxes",
            "earnings_per_share_basic": "Basic EPS",
            "earnings_per_share_diluted": "Diluted EPS",
            "weighted_average_shares_basic": "Weighted average shares outstanding (basic)",
            "weighted_average_shares_diluted": "Weighted average shares outstanding (diluted)"
        }
    },
    "balance_sheets": {
        "description": "Contains balance sheet data (assets, liabilities, equity) for companies",
        "financial_columns": {
            "cash_and_cash_equivalents": "Cash and cash equivalents",
            "short_term_investments": "Short-term marketable securities",
            "accounts_receivable": "Money owed by customers",
            "inventory": "Inventory value",
            "total_current_assets": "Total current assets",
            "property_plant_equipment": "PP&E net of depreciation",
            "goodwill": "Goodwill from acquisitions",
            "intangible_assets": "Other intangible assets",
            "long_term_investments": "Long-term investments",
            "total_non_current_assets": "Total non-current assets",
            "total_assets": "Total assets",
            "accounts_payable": "Money owed to suppliers",
            "accrued_liabilities": "Accrued expenses",
            "commercial_paper": "Short-term debt",
            "current_portion_long_term_debt": "Current portion of long-term debt",
            "total_current_liabilities": "Total current liabilities",
            "long_term_debt": "Long-term debt",
            "total_non_current_liabilities": "Total non-current liabilities",
            "total_liabilities": "Total liabilities",
            "common_stock": "Common stock value",
            "retained_earnings": "Retained earnings",
            "total_stockholders_equity": "Total shareholders' equity"
        }
    },
    "cash_flow_statements": {
        "description": "Contains cash flow statement data for companies",
        "financial_columns": {
            "net_cash_from_operating_activities": "Operating cash flow",
            "depreciation_and_amortization": "Total D&A",
            "depreciation": "Depreciation expense",
            "amortization": "Amortization expense",
            "stock_based_compensation": "Stock-based compensation",
            "net_cash_from_investing_activities": "Investing cash flow",
            "capital_expenditures": "Capital expenditures (CapEx)",
            "acquisitions": "Cash spent on acquisitions",
            "investments_purchased": "Investments purchased",
            "investments_sold": "Investments sold",
            "net_cash_from_financing_activities": "Financing cash flow",
            "dividends_paid": "Dividends paid to shareholders",
            "share_repurchases": "Share buybacks",
            "proceeds_from_stock_issuance": "Cash from stock issuance",
            "debt_issued": "New debt issued",
            "debt_repaid": "Debt repayments"
        }
    },
    "virtual_fields": {
        "description": "Calculated fields available through virtual field system",
        "financial_columns": {
            "ebitda": "Earnings before interest, taxes, depreciation & amortization",
            "ebit": "Earnings before interest and taxes (operating income)",
            "free_cash_flow": "Operating cash flow minus capital expenditures",
            "total_debt": "Total debt including all borrowings",
            "working_capital": "Current assets minus current liabilities",
            "net_working_capital": "Working capital minus cash"
        }
    }
}

class LLMFieldMapper:
    """Uses LLM to intelligently map user requests to database fields"""
    
    def __init__(self):
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY', '')
        self.openrouter_url = 'https://openrouter.ai/api/v1/chat/completions'
        self.model = os.getenv('SEC_FINANCIAL_AGENT_LLM_MODEL', 'anthropic/claude-3.5-sonnet')
        self.available = bool(self.openrouter_api_key)
        
        if not self.available:
            logger.warning("OpenRouter API key not configured. Using fallback field mapping.")
        else:
            logger.info(f"LLM Field Mapper initialized with model: {self.model}")
    
    def map_metrics_to_fields(self, requested_metrics: List[str], available_columns: set) -> Dict[str, Any]:
        """
        Use LLM to map user-requested metrics to database columns
        
        Returns:
            Dictionary with mapping results and suggested combinations
        """
        if not self.available:
            return self._fallback_mapping(requested_metrics, available_columns)
        
        try:
            system_prompt = self._create_mapping_prompt()
            user_query = self._create_user_query(requested_metrics, list(available_columns))
            
            response = self._call_llm(system_prompt, user_query)
            return self._parse_llm_response(response, requested_metrics)
            
        except Exception as e:
            logger.error(f"LLM field mapping failed: {e}")
            return self._fallback_mapping(requested_metrics, available_columns)
    
    def _create_mapping_prompt(self) -> str:
        """Create system prompt for field mapping"""
        schema_str = json.dumps(DATABASE_SCHEMA, indent=2)
        
        return f"""You are a financial data mapping expert. Your job is to map user-requested financial metrics to the correct database columns.

Database Schema:
{schema_str}

IMPORTANT INSTRUCTIONS:
1. Map each requested metric to the most appropriate database column(s)
2. If an exact field doesn't exist, suggest a combination of fields that can calculate it
3. For EBITDA: Use operating_income + depreciation_and_amortization (if available)
4. For Free Cash Flow: Use net_cash_from_operating_activities - capital_expenditures
5. Always return valid JSON in this exact format:

{{
    "mappings": {{
        "requested_metric": {{
            "direct_field": "database_column_name" or null,
            "calculated_from": ["field1", "field2"] or null,
            "calculation_method": "field1 + field2" or null,
            "confidence": 0.95,
            "explanation": "Brief explanation of mapping choice"
        }}
    }},
    "missing_metrics": ["metric1", "metric2"],
    "suggestions": ["Consider these alternative metrics: ..."]
}}

Be precise and only use columns that exist in the schema. Prioritize accuracy over completeness."""
    
    def _create_user_query(self, requested_metrics: List[str], available_columns: List[str]) -> str:
        """Create user query for LLM"""
        return f"""Please map these requested metrics to database fields:
Requested metrics: {requested_metrics}
Available columns in current data: {available_columns}

Remember to provide combinations for metrics like EBITDA that aren't stored directly."""
    
    def _call_llm(self, system_prompt: str, user_query: str) -> str:
        """Call OpenRouter API"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        response = requests.post(
            self.openrouter_url,
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.1
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenRouter API error: {response.status_code}")
    
    def _parse_llm_response(self, response: str, requested_metrics: List[str]) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            
            parsed = json.loads(json_str)
            
            # Ensure all requested metrics are covered
            result = {
                "mappings": {},
                "missing_metrics": [],
                "suggestions": parsed.get("suggestions", [])
            }
            
            for metric in requested_metrics:
                if metric in parsed.get("mappings", {}):
                    result["mappings"][metric] = parsed["mappings"][metric]
                else:
                    result["missing_metrics"].append(metric)
                    result["mappings"][metric] = {
                        "direct_field": None,
                        "calculated_from": None,
                        "calculation_method": None,
                        "confidence": 0.0,
                        "explanation": "No mapping found"
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._fallback_mapping(requested_metrics, set())
    
    def _fallback_mapping(self, requested_metrics: List[str], available_columns: set) -> Dict[str, Any]:
        """Fallback mapping when LLM is unavailable"""
        simple_mappings = {
            'revenue': 'total_revenue',
            'sales': 'total_revenue',
            'profit': 'net_income',
            'income': 'net_income',
            'earnings': 'net_income',
            'ebit': 'operating_income',
            'operating_income': 'operating_income',
            'cash': 'cash_and_cash_equivalents',
            'assets': 'total_assets',
            'debt': 'long_term_debt',
            'equity': 'total_stockholders_equity'
        }
        
        result = {
            "mappings": {},
            "missing_metrics": [],
            "suggestions": ["Configure OPENROUTER_API_KEY for intelligent field mapping"]
        }
        
        for metric in requested_metrics:
            metric_lower = metric.lower()
            
            # Check simple mappings
            if metric_lower in simple_mappings:
                mapped_field = simple_mappings[metric_lower]
                if mapped_field in available_columns:
                    result["mappings"][metric] = {
                        "direct_field": mapped_field,
                        "calculated_from": None,
                        "calculation_method": None,
                        "confidence": 0.8,
                        "explanation": f"Simple mapping to {mapped_field}"
                    }
                    continue
            
            # Check exact match
            exact_match = None
            for col in available_columns:
                if col.lower() == metric_lower:
                    exact_match = col
                    break
            
            if exact_match:
                result["mappings"][metric] = {
                    "direct_field": exact_match,
                    "calculated_from": None,
                    "calculation_method": None,
                    "confidence": 0.9,
                    "explanation": f"Exact match to {exact_match}"
                }
            else:
                result["missing_metrics"].append(metric)
                result["mappings"][metric] = {
                    "direct_field": None,
                    "calculated_from": None,
                    "calculation_method": None,
                    "confidence": 0.0,
                    "explanation": "No mapping found in fallback mode"
                }
        
        return result

# Global LLM field mapper instance
_llm_mapper = LLMFieldMapper()

def get_llm_field_mapper() -> LLMFieldMapper:
    """Get the global LLM field mapper instance"""
    return _llm_mapper

# Core Agent Functions

def get_financial_metrics_for_agent(ticker: str, metrics: List[str], period: str = 'LTM') -> FinancialAgentResponse:
    """
    Get specific financial metrics for a company (agent-friendly)
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        metrics: List of financial metric names (e.g., ['total_revenue', 'net_income'])
        period: Period type ('LTM', 'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'latest')
        
    Returns:
        FinancialAgentResponse with requested metrics data
    """
    try:
        # Input validation
        ticker_error = validate_ticker(ticker)
        if ticker_error:
            return create_error_response(ErrorCodes.INVALID_TICKER, ticker_error)
        
        period_error = validate_period(period)
        if period_error:
            return create_error_response(ErrorCodes.INVALID_PERIOD, period_error)
            
        metrics_error = validate_metrics(metrics)
        if metrics_error:
            return create_error_response(ErrorCodes.INVALID_METRIC, metrics_error)
        
        # Import here to avoid circular imports
        # Add current directory to path for imports
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        from .standalone_interface import get_company_financial_data

        # Get financial data
        df = get_company_financial_data(ticker.upper(), auto_process=True, include_ratios=False)
        if df is None or df.empty:
            return create_error_response(
                ErrorCodes.DATA_NOT_FOUND, 
                f"No financial data found for ticker {ticker}"
            )
        
        # Filter by period using enhanced period filtering logic
        if period != 'all':
            if period == 'latest':
                # For 'latest', find the most recent period with the most complete data
                # Group by period_end_date and select the one with most non-null values
                latest_periods = df.groupby('period_end_date').agg({
                    'period_end_date': 'first',
                    'total_revenue': 'first',  # Check for income statement data
                    'net_cash_from_operating_activities': 'first',  # Check for cash flow data
                    'total_assets': 'first'  # Check for balance sheet data
                }).reset_index(drop=True)
                
                # Sort by period and take the latest
                latest_periods = latest_periods.sort_values('period_end_date', ascending=False)
                latest_date = latest_periods.iloc[0]['period_end_date']
                
                
                # Get all records for this latest period (income, cash flow, balance sheet)
                df_filtered = df[df['period_end_date'] == latest_date]
                
                # If we have multiple statement types for same period, merge them into one row
                if len(df_filtered) > 1:
                    
                    # Combine all data into single row, taking first non-null value for each column
                    # This allows us to combine income statement + cash flow + balance sheet data
                    merged_data = {}
                    
                    # Get all columns
                    for col in df_filtered.columns:
                        # For each column, take the first non-null value across all statement types
                        non_null_values = df_filtered[col].dropna()
                        if len(non_null_values) > 0:
                            merged_data[col] = non_null_values.iloc[0]
                        else:
                            merged_data[col] = None
                    
                    # Create new DataFrame with merged data
                    df_filtered = pd.DataFrame([merged_data])
            else:
                # Use enhanced period filtering (same logic as get_ratios_for_agent)
                original_count = len(df)
                df_filtered = _filter_financial_data_by_period(df, period)
        else:
            df_filtered = df
            
        if df_filtered.empty:
            return create_error_response(
                ErrorCodes.DATA_NOT_FOUND,
                f"No data found for ticker {ticker} and period {period}"
            )
        
        # Use LLM-based intelligent field mapping
        available_columns = set(df_filtered.columns)
        mapper = get_llm_field_mapper()
        
        
        # Get intelligent field mappings
        mapping_result = mapper.map_metrics_to_fields(metrics, available_columns)
        
        # Extract requested metrics using LLM mappings
        result_data = {}
        mapping_details = {}
        
        for metric in metrics:
            mapping = mapping_result["mappings"].get(metric, {})
            direct_field = mapping.get("direct_field")
            calculated_from = mapping.get("calculated_from")
            calculation_method = mapping.get("calculation_method")
            
            # Store mapping details for transparency
            mapping_details[metric] = {
                "mapping_method": mapping.get("explanation", "No mapping found"),
                "confidence": mapping.get("confidence", 0.0),
                "direct_field": direct_field,
                "calculated_from": calculated_from,
                "calculation_method": calculation_method
            }
            
            if direct_field and direct_field in available_columns:
                # Direct field mapping
                # For single-statement data (like balance sheet), skip consolidation to avoid data corruption
                # Check if we need consolidation (multiple statement types per period)
                statement_types_per_period = df_filtered.groupby('period_end_date')['statement_type'].nunique()
                needs_consolidation = (statement_types_per_period > 1).any()
                
                if needs_consolidation:
                    # Consolidate data by period for mixed statement types
                    df_processed = _consolidate_financial_data_by_period(df_filtered)
                else:
                    # Use data directly for single statement type queries
                    df_processed = df_filtered.copy()
                    # Ensure period_end_date is datetime for consistency
                    df_processed['period_end_date'] = pd.to_datetime(df_processed['period_end_date'])
                    # Sort by period_end_date (most recent first)
                    df_processed = df_processed.sort_values('period_end_date', ascending=False).reset_index(drop=True)
                
                # Check the actual data in the column with robust null handling
                column_data = df_processed[direct_field]
                
                # More robust null filtering that handles different data types
                def is_valid_value(val):
                    """Check if a value is valid (not null, not empty, not zero-string)"""
                    if pd.isna(val) or val is None:
                        return False
                    # Handle string representations of numbers
                    if isinstance(val, str) and (val.strip() == '' or val.strip() == '0' or val.strip().lower() == 'none'):
                        return False
                    # Handle numeric zeros - in financial data, zero is a valid value
                    try:
                        # Try to convert to float to check if it's a valid number
                        float_val = float(val)
                        return True  # Zero is a valid financial value
                    except (ValueError, TypeError):
                        return False
                
                # Filter for valid values
                valid_mask = column_data.apply(is_valid_value)
                valid_data = column_data[valid_mask]
                
                metric_values = valid_data.tolist()
                
                # Handle period_end_date conversion more robustly
                try:
                    # Ensure period_end_date is datetime
                    if 'period_end_date' in df_processed.columns:
                        # Get rows where the field has valid data
                        valid_rows = df_processed[valid_mask]
                        
                        if not valid_rows.empty:
                            # Convert to datetime if needed
                            period_dates = pd.to_datetime(valid_rows['period_end_date'])
                            periods = period_dates.dt.strftime('%Y-%m-%d').tolist()
                        else:
                            periods = []
                    else:
                        periods = []
                except Exception as e:
                    logger.warning(f"Error formatting periods for {metric}: {e}")
                    periods = []
                
                result_data[metric] = {
                    'values': metric_values,
                    'periods': periods,
                    'count': len(metric_values),
                    'data_source': 'direct_field',
                    'source_field': direct_field
                }
                
            elif calculated_from and calculation_method:
                # Calculated field mapping
                # Check if all required fields are available
                missing_fields = [field for field in calculated_from if field not in available_columns]
                available_fields = [field for field in calculated_from if field in available_columns]
                
                try:
                    # Group by period_end_date and consolidate data first
                    # This handles cases where we have multiple statement types for same period
                    df_consolidated = _consolidate_financial_data_by_period(df_filtered)
                    
                    calculated_values = []
                    periods = []
                    
                    for idx, row in df_consolidated.iterrows():
                        # Check field values in this consolidated row
                        field_values = {}
                        for field in calculated_from:
                            if field in available_columns and field in df_consolidated.columns:
                                field_values[field] = row[field]
                        
                        # Simple calculation support (can be extended)
                        if calculation_method and all(field in df_consolidated.columns for field in calculated_from):
                            if '+' in calculation_method:
                                # Addition with proper type handling
                                fields = [f.strip() for f in calculation_method.split('+')]
                                calc_value = 0
                                all_values_valid = True
                                
                                for field in fields:
                                    if field in df_consolidated.columns:
                                        field_value = row[field]
                                        
                                        # Handle different data types and NaN values
                                        if field_value is None:
                                            numeric_value = 0
                                        elif pd.isna(field_value):
                                            numeric_value = 0
                                        else:
                                            # Convert to float to handle Decimal/float mixing
                                            try:
                                                numeric_value = float(field_value)
                                            except (ValueError, TypeError) as e:
                                                logger.error(f"Cannot convert {field} value {field_value} to float: {e}")
                                                all_values_valid = False
                                                break
                                        
                                        calc_value += numeric_value
                                
                                if not all_values_valid:
                                    calc_value = None
                                    
                            elif '-' in calculation_method:
                                # Subtraction with proper type handling
                                fields = [f.strip() for f in calculation_method.split('-')]
                                first_field = fields[0]
                                
                                if first_field in df_consolidated.columns:
                                    first_value = row[first_field]
                                    if first_value is None or pd.isna(first_value):
                                        calc_value = 0
                                    else:
                                        calc_value = float(first_value)
                                    
                                    for field in fields[1:]:
                                        if field in df_consolidated.columns:
                                            field_value = row[field]
                                            if field_value is None or pd.isna(field_value):
                                                numeric_value = 0
                                            else:
                                                numeric_value = float(field_value)
                                            calc_value -= numeric_value
                                    
                                else:
                                    calc_value = None
                            else:
                                calc_value = None
                            
                            if calc_value is not None:
                                calculated_values.append(calc_value)
                                periods.append(row['period_end_date'].strftime('%Y-%m-%d'))
                    
                    
                    result_data[metric] = {
                        'values': calculated_values,
                        'periods': periods,
                        'count': len(calculated_values),
                        'data_source': 'calculated',
                        'calculation_method': calculation_method,
                        'source_fields': calculated_from
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to calculate {metric}: {e}")
                    result_data[metric] = None
            else:
                # No valid mapping found
                result_data[metric] = None
        
        # Create metadata with mapping details
        metadata = {
            'ticker': ticker.upper(),
            'period_requested': period,
            'metrics_requested': metrics,
            'metrics_found': [m for m in metrics if result_data.get(m) is not None],
            'total_periods': len(df_filtered),
            'data_source': 'SECFinancialRAG',
            'field_mapping_used': mapper.available,
            'mapping_details': mapping_details,
            'llm_suggestions': mapping_result.get("suggestions", []),
            'missing_metrics': mapping_result.get("missing_metrics", [])
        }
        
        return create_success_response(result_data, metadata)
        
    except Exception as e:
        logger.error(f"Error getting financial metrics for {ticker}: {e}")
        return create_error_response(
            ErrorCodes.PROCESSING_ERROR,
            f"Error processing request: {str(e)}"
        )

def validate_ratio_category(category: str) -> Optional[str]:
    """Validate ratio category. Returns None if valid, error message if invalid."""
    valid_categories = ['all', 'profitability', 'liquidity', 'leverage', 'efficiency', 'growth', 'cash_flow', 'valuation']
    if category not in valid_categories:
        return f"Invalid category. Must be one of: {', '.join(valid_categories)}"
    return None

def get_available_ratio_categories_for_agent() -> FinancialAgentResponse:
    """
    Get list of available ratio categories for agents
    
    This function provides agents with the complete list of ratio categories
    they can query, along with descriptions of what each category contains.
    
    Returns:
        FinancialAgentResponse with category information
    """
    try:
        category_info = {
            'profitability': {
                'description': 'Profit margins, returns on equity/assets, earnings efficiency',
                'examples': ['Net_Margin', 'ROE', 'ROA', 'ROIC', 'Gross_Margin', 'EBITDA_Margin']
            },
            'liquidity': {
                'description': 'Company ability to meet short-term obligations',
                'examples': ['Current_Ratio', 'Quick_Ratio', 'Cash_Ratio', 'Working_Capital_Ratio']
            },
            'leverage': {
                'description': 'Debt levels and financial risk metrics',
                'examples': ['Debt_to_Equity', 'Debt_to_Assets', 'Debt_to_EBITDA', 'Interest_Coverage']
            },
            'efficiency': {
                'description': 'Asset utilization and operational efficiency',
                'examples': ['Asset_Turnover', 'Inventory_Turnover', 'Receivables_Turnover']
            },
            'growth': {
                'description': 'Revenue, earnings, and business growth metrics',
                'examples': ['Revenue_Growth_YoY', 'EBIT_Growth_YoY', 'Net_Income_Growth']
            },
            'cash_flow': {
                'description': 'Cash generation and cash-based performance metrics',
                'examples': ['Operating_Cash_Flow_Margin', 'Free_Cash_Flow_Margin', 'Cash_Return_on_Assets']
            },
            'valuation': {
                'description': 'Market valuation and trading multiples',
                'examples': ['P/E_Ratio', 'P/B_Ratio', 'EV/EBITDA', 'Price_to_Sales']
            }
        }
        
        metadata = {
            'total_categories': len(category_info),
            'usage_note': 'Use category names as filters in get_ratios_for_agent(), or use "all" for all categories',
            'period_formats': 'FY2024, Q1-2024, latest, last N quarters, last N financial years',
            'data_source': 'SECFinancialRAG',
            'correlation_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return FinancialAgentResponse(
            success=True,
            data=category_info,
            error=None,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error getting available ratio categories: {e}")
        return create_error_response(
            ErrorCodes.PROCESSING_ERROR,
            f"Failed to get available ratio categories: {str(e)}"
        )

def _filter_ratios_by_period(ratio_df, period: str):
    """Filter ratio DataFrame by enhanced period formats"""
    import re
    import pandas as pd
    
    # Data is already clean ratio data from calculated_ratios table
    # Latest period - find the most recent period_end_date
    if period == 'latest':
        if ratio_df.empty:
            return ratio_df
        # Sort by period_end_date descending (most recent first) and get all ratios from latest period
        latest_date = ratio_df['period_end_date'].max()
        return ratio_df[ratio_df['period_end_date'] == latest_date]
    
    # Specific year patterns: FY2024, Q1-2024, etc.
    year_pattern = r'^(FY|Q[1-4])-?(\d{4})$'
    match = re.match(year_pattern, period)
    if match:
        period_type = match.group(1)
        year = int(match.group(2))
        
        if period_type == 'FY':
            # For FY2024: Since calculated_ratios table contains only period_type='LTM',
            # map FY requests to Q4 ratios for that fiscal year
            if ('fiscal_quarter' in ratio_df.columns and not ratio_df['fiscal_quarter'].dropna().empty and
                'fiscal_year' in ratio_df.columns and not ratio_df['fiscal_year'].dropna().empty):
                # Map FY2024 -> Q4-2024 ratios (fiscal year-end)
                filtered = ratio_df[
                    (ratio_df['period_type'] == 'LTM') & 
                    (ratio_df['fiscal_quarter'] == 'Q4') & 
                    (ratio_df['fiscal_year'] == year)
                ]
            else:
                # Fallback: try original FY logic first
                fy_records = ratio_df[ratio_df['period_type'] == 'FY']
                if not fy_records.empty:
                    filtered = fy_records[fy_records['fiscal_year'] == year]
                else:
                    # Final fallback: use period_end_date year for any records
                    filtered = ratio_df[ratio_df['period_end_date'].dt.year == year]
        else:
            # For Q1-2023: filter by fiscal_quarter='Q1' AND fiscal_year=2023
            quarter = period_type  # Q1, Q2, Q3, Q4
            if ('fiscal_quarter' in ratio_df.columns and not ratio_df['fiscal_quarter'].dropna().empty and
                'fiscal_year' in ratio_df.columns and not ratio_df['fiscal_year'].dropna().empty):
                filtered = ratio_df[
                    (ratio_df['fiscal_quarter'] == quarter) & 
                    (ratio_df['fiscal_year'] == year)
                ]
            else:
                # Fallback: quarters not available in this dataset
                filtered = ratio_df.iloc[0:0]  # Return empty DataFrame
        
        return filtered
    
    # Trend patterns: last n quarters, last n financial years
    trend_pattern = r'^last (\d+) (quarters|financial years)$'
    match = re.match(trend_pattern, period)
    if match:
        count = int(match.group(1))
        trend_type = match.group(2)
        
        if trend_type == 'quarters':
            # For last N quarters: get all ratios from the N most recent quarterly periods
            if 'fiscal_quarter' in ratio_df.columns and not ratio_df['fiscal_quarter'].dropna().empty:
                quarterly_data = ratio_df[ratio_df['fiscal_quarter'].isin(['Q1', 'Q2', 'Q3', 'Q4'])].copy()
                
                # Get the N most recent quarter-year combinations
                quarterly_data['quarter_year'] = quarterly_data['fiscal_quarter'] + '-' + quarterly_data['fiscal_year'].astype(str)
                recent_quarters = quarterly_data.sort_values('period_end_date', ascending=False)['quarter_year'].unique()[:count]
                
                # Return all ratios from those quarter-year combinations
                filtered = quarterly_data[quarterly_data['quarter_year'].isin(recent_quarters)]
                return filtered.drop(columns=['quarter_year']).sort_values('period_end_date', ascending=False)
            else:
                # Fallback: quarters not available in this dataset
                return ratio_df.iloc[0:0]  # Return empty DataFrame
        
        elif trend_type == 'financial years':
            # For last N financial years: get year-end ratios from the N most recent fiscal years
            
            # Strategy 1: Try Q4 ratios (most common for year-end annual ratios)
            if 'fiscal_quarter' in ratio_df.columns and not ratio_df['fiscal_quarter'].dropna().empty:
                q4_data = ratio_df[ratio_df['fiscal_quarter'] == 'Q4'].copy()
                if not q4_data.empty:
                    # Get the N most recent fiscal years from Q4 data
                    recent_years = sorted(q4_data['fiscal_year'].unique(), reverse=True)[:count]
                    
                    # Return Q4 ratios from those years (year-end annual ratios)
                    filtered = q4_data[q4_data['fiscal_year'].isin(recent_years)]
                    return filtered.sort_values('period_end_date', ascending=False)
            
            # Strategy 2: Fallback to FY period_type if available
            fy_data = ratio_df[ratio_df['period_type'] == 'FY']
            if not fy_data.empty:
                recent_years = sorted(fy_data['fiscal_year'].unique(), reverse=True)[:count]
                filtered = fy_data[fy_data['fiscal_year'].isin(recent_years)]
                return filtered.sort_values('period_end_date', ascending=False)
            
            # Strategy 3: Final fallback - use LTM data but get year-end records
            ltm_data = ratio_df[ratio_df['period_type'] == 'LTM'].copy()
            if not ltm_data.empty:
                # Group by fiscal year and get the latest record for each year (year-end)
                ltm_data['period_end_date'] = pd.to_datetime(ltm_data['period_end_date'])
                year_end_ratios = ltm_data.groupby('fiscal_year')['period_end_date'].idxmax()
                
                # Get the N most recent fiscal years
                available_years = sorted(ltm_data['fiscal_year'].unique(), reverse=True)[:count]
                
                # Filter to only include records from target years
                target_indices = []
                for year in available_years:
                    if year in year_end_ratios.index:
                        target_indices.append(year_end_ratios[year])
                
                if target_indices:
                    filtered = ltm_data.loc[target_indices]
                    return filtered.sort_values('period_end_date', ascending=False)
            
            # No suitable data found
            return ratio_df.iloc[0:0]  # Return empty DataFrame
    
    # Default: return empty DataFrame if no match
    return ratio_df.iloc[0:0]

def get_ratios_for_agent(ticker: str, categories: List[str] = None, period: str = 'LTM') -> FinancialAgentResponse:
    """
    Get calculated financial ratios for a company (agent-friendly)
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        categories: List of ratio categories (e.g., ['profitability', 'liquidity']) or None for all
        period: Period specification with these formats:
            - Basic: 'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'latest', 'all'
            - Specific year: 'FY2024', 'Q1-2024', 'Q2-2024', etc.
            - Trend analysis: 'last 5 quarters', 'last 3 financial years' (n=1-10)
        
    Returns:
        FinancialAgentResponse with calculated ratios data
    """
    try:
        # Input validation
        ticker_error = validate_ticker(ticker)
        if ticker_error:
            return create_error_response(ErrorCodes.INVALID_TICKER, ticker_error)
        
        period_error = validate_period(period)
        if period_error:
            return create_error_response(ErrorCodes.INVALID_PERIOD, period_error)
        
        # Validate categories if provided
        if categories is not None:
            if not isinstance(categories, list):
                return create_error_response(ErrorCodes.VALIDATION_ERROR, "Categories must be a list")
            
            for category in categories:
                category_error = validate_ratio_category(category)
                if category_error:
                    return create_error_response(ErrorCodes.VALIDATION_ERROR, category_error)
        
        # SIMPLIFIED: Query calculated_ratios table directly
        # Add current directory to path for imports
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from .database import FinancialDatabase
        import pandas as pd

        with FinancialDatabase() as db:
            # Check cache first
            if not db.is_company_data_fresh(ticker, hours=24):
                # Data is stale, need to refresh first
                from .standalone_interface import process_company_financials
                logger.info(f"Data for {ticker} is stale, refreshing...")
                process_result = process_company_financials(
                    ticker=ticker,
                    validate_data=True, 
                    generate_ltm=True,     # ← ADD THIS: Generate LTM tables needed for ratios
                    calculate_ratios=True
                )
                if process_result.get('status') != 'success':
                    return create_error_response(
                        ErrorCodes.DATA_NOT_FOUND,
                        f"Failed to refresh data for {ticker}: {process_result.get('error_message')}"
                    )
                db.update_company_timestamp(ticker)
            
            # Get ratios directly from calculated_ratios table
            ratios = db.get_calculated_ratios(ticker)
            if not ratios:
                return create_error_response(
                    ErrorCodes.DATA_NOT_FOUND,
                    f"No ratio data found for ticker {ticker}"
                )
            
            # Convert to DataFrame for filtering
            ratio_df = pd.DataFrame([dict(ratio) for ratio in ratios])
            ratio_df['period_end_date'] = pd.to_datetime(ratio_df['period_end_date'])
        
        
        # Check fiscal data availability
        
        
        
        # Filter by period if specified
        if period != 'all':
            ratio_df = _filter_ratios_by_period(ratio_df, period)
                
        if ratio_df.empty:
            return create_error_response(
                ErrorCodes.DATA_NOT_FOUND,
                f"No ratio data found for ticker {ticker} and period {period}"
            )
        
        # Filter by categories if specified
        if categories and 'all' not in categories:
            # Use 'category' from ratio_definitions table (rd.category) for filtering, fallback to ratio_category
            filter_column = 'category' if 'category' in ratio_df.columns else 'ratio_category'
            ratio_df = ratio_df[ratio_df[filter_column].isin(categories)]
            
        if ratio_df.empty:
            return create_error_response(
                ErrorCodes.DATA_NOT_FOUND,
                f"No ratios found for ticker {ticker} in categories {categories}"
            )
        
        # Organize ratios by category
        result_data = {}
        
        # Group by category - use same column that was used for filtering
        category_column = 'category' if 'category' in ratio_df.columns else 'ratio_category'
        if category_column in ratio_df.columns:
            for category, group in ratio_df.groupby(category_column):
                if category is None:
                    category = 'uncategorized'
                    
                category_ratios = {}
                for _, row in group.iterrows():
                    ratio_name = row.get('ratio_name')
                    if ratio_name:
                        # For trend queries (multiple periods), create unique keys
                        # Format: "Ratio_Name_FY2024" or "Ratio_Name_Q1-2024"
                        fiscal_year = row.get('fiscal_year')
                        fiscal_quarter = row.get('fiscal_quarter')
                        period_type = row.get('period_type')
                        
                        # Create period-specific key for multi-period data
                        if period.startswith('last ') and 'financial years' in period:
                            # For "last N financial years", include year in key
                            unique_key = f"{ratio_name}_FY{fiscal_year}"
                        elif period.startswith('last ') and 'quarters' in period:
                            # For "last N quarters", include quarter-year in key  
                            unique_key = f"{ratio_name}_{fiscal_quarter}-{fiscal_year}"
                        else:
                            # For single period queries, use simple ratio name
                            unique_key = ratio_name
                            
                        category_ratios[unique_key] = {
                            'value': row.get('ratio_value'),
                            'period_end_date': row.get('period_end_date').strftime('%Y-%m-%d') if row.get('period_end_date') else None,
                            'period_type': row.get('period_type'),
                            'fiscal_year': row.get('fiscal_year'),
                            'formula': row.get('formula'),
                            'description': row.get('description'),
                            'calculation_inputs': row.get('calculation_inputs')
                        }
                
                if category_ratios:
                    result_data[category] = {
                        'ratios': category_ratios,
                        'category_name': category,
                        'ratio_count': len(category_ratios)
                    }
        else:
            # Fallback if no category column
            uncategorized_ratios = {}
            for _, row in ratio_df.iterrows():
                ratio_name = row.get('ratio_name')
                if ratio_name:
                    # Apply same logic for unique keys
                    fiscal_year = row.get('fiscal_year')
                    fiscal_quarter = row.get('fiscal_quarter')
                    
                    if period.startswith('last ') and 'financial years' in period:
                        unique_key = f"{ratio_name}_FY{fiscal_year}"
                    elif period.startswith('last ') and 'quarters' in period:
                        unique_key = f"{ratio_name}_{fiscal_quarter}-{fiscal_year}"
                    else:
                        unique_key = ratio_name
                        
                    uncategorized_ratios[unique_key] = {
                        'value': row.get('ratio_value'),
                        'period_end_date': row.get('period_end_date').strftime('%Y-%m-%d') if row.get('period_end_date') else None,
                        'period_type': row.get('period_type'),
                        'fiscal_year': row.get('fiscal_year')
                    }
            
            if uncategorized_ratios:
                result_data['uncategorized'] = {
                    'ratios': uncategorized_ratios,
                    'category_name': 'uncategorized',
                    'ratio_count': len(uncategorized_ratios)
                }
        
        # Create metadata
        categories_found = list(result_data.keys())
        total_ratios = sum(category_data.get('ratio_count', 0) for category_data in result_data.values())
        
        metadata = {
            'ticker': ticker.upper(),
            'period_requested': period,
            'categories_requested': categories or ['all'],
            'categories_found': categories_found,
            'total_ratios': total_ratios,
            'total_periods': len(ratio_df),
            'data_source': 'SECFinancialRAG'
        }
        
        return create_success_response(result_data, metadata)
        
    except Exception as e:
        logger.error(f"Error getting ratios for {ticker}: {e}")
        return create_error_response(
            ErrorCodes.PROCESSING_ERROR,
            f"Error processing request: {str(e)}"
        )

def get_ratio_definition_for_agent(ratio_name: str, ticker: str = None) -> FinancialAgentResponse:
    """
    Get ratio definition with formula, description, and calculation logic (agent-friendly)
    
    Args:
        ratio_name: Name of the ratio (e.g., 'ROE', 'Current_Ratio')
        ticker: Optional company ticker for company-specific ratios
        
    Returns:
        FinancialAgentResponse with ratio definition details
    """
    try:
        # Input validation
        if not ratio_name or not isinstance(ratio_name, str):
            return create_error_response(ErrorCodes.VALIDATION_ERROR, "Ratio name must be a non-empty string")
        
        if ticker:
            ticker_error = validate_ticker(ticker)
            if ticker_error:
                return create_error_response(ErrorCodes.INVALID_TICKER, ticker_error)

        # Import here to avoid circular imports
        from .main import get_ratio_definitions

        # Get ratio definitions
        ratio_definitions = get_ratio_definitions(ticker)
        if not ratio_definitions:
            return create_error_response(
                ErrorCodes.DATA_NOT_FOUND,
                f"No ratio definitions found{' for ticker ' + ticker if ticker else ''}"
            )
        
        # Find the specific ratio (case-insensitive search)
        ratio_name_lower = ratio_name.lower()
        matching_ratio = None
        
        for ratio_def in ratio_definitions:
            if ratio_def.get('name', '').lower() == ratio_name_lower:
                matching_ratio = ratio_def
                break
        
        if not matching_ratio:
            # Try partial matching if exact match not found
            for ratio_def in ratio_definitions:
                if ratio_name_lower in ratio_def.get('name', '').lower():
                    matching_ratio = ratio_def
                    break
        
        if not matching_ratio:
            available_ratios = [r.get('name') for r in ratio_definitions[:10]]  # Show first 10
            return create_error_response(
                ErrorCodes.DATA_NOT_FOUND,
                f"Ratio '{ratio_name}' not found. Available ratios include: {', '.join(available_ratios)}"
            )
        
        # Prepare ratio definition data
        definition_data = {
            'name': matching_ratio.get('name'),
            'formula': matching_ratio.get('formula'),
            'description': matching_ratio.get('description'),
            'category': matching_ratio.get('category'),
            'is_active': matching_ratio.get('is_active', True),
            'created_by': matching_ratio.get('created_by'),
            'created_at': matching_ratio.get('created_at'),
            'updated_at': matching_ratio.get('updated_at')
        }
        
        # Add interpretation and calculation guidance
        formula = matching_ratio.get('formula', '')
        interpretation = _generate_ratio_interpretation(matching_ratio.get('name'), formula, matching_ratio.get('category'))
        
        definition_data['interpretation'] = interpretation
        definition_data['calculation_guidance'] = _generate_calculation_guidance(formula)
        
        # Create metadata
        metadata = {
            'ratio_name': ratio_name,
            'search_method': 'exact_match' if matching_ratio.get('name', '').lower() == ratio_name_lower else 'partial_match',
            'ticker': ticker.upper() if ticker else None,
            'total_available_ratios': len(ratio_definitions),
            'data_source': 'SECFinancialRAG'
        }
        
        return create_success_response(definition_data, metadata)
        
    except Exception as e:
        logger.error(f"Error getting ratio definition for {ratio_name}: {e}")
        return create_error_response(
            ErrorCodes.PROCESSING_ERROR,
            f"Error processing request: {str(e)}"
        )

def _generate_ratio_interpretation(ratio_name: str, formula: str, category: str) -> str:
    """Generate human-readable interpretation of what the ratio means"""
    interpretations = {
        'roe': "Measures how efficiently a company uses shareholders' equity to generate profits. Higher values indicate better performance.",
        'roa': "Shows how efficiently a company uses its assets to generate profits. Higher values indicate better asset utilization.",
        'current_ratio': "Measures a company's ability to pay short-term debts. Values above 1.0 indicate sufficient liquidity.",
        'quick_ratio': "More conservative liquidity measure excluding inventory. Values above 1.0 indicate good short-term liquidity.",
        'debt_to_equity': "Shows the relative amount of debt vs equity financing. Lower values generally indicate less financial risk.",
        'gross_profit_margin': "Percentage of revenue remaining after cost of goods sold. Higher values indicate better pricing power.",
        'net_profit_margin': "Percentage of revenue remaining as profit after all expenses. Higher values indicate better overall efficiency.",
        'asset_turnover': "Measures how efficiently a company uses assets to generate sales. Higher values indicate better efficiency.",
        'inventory_turnover': "Shows how quickly inventory is sold and replaced. Higher values indicate efficient inventory management."
    }
    
    # Try to find interpretation by ratio name
    ratio_key = ratio_name.lower().replace(' ', '_').replace('-', '_')
    interpretation = interpretations.get(ratio_key)
    
    if interpretation:
        return interpretation
    
    # Generate basic interpretation based on category
    if category == 'profitability':
        return "Measures the company's ability to generate profit relative to revenue, assets, or equity."
    elif category == 'liquidity':
        return "Measures the company's ability to meet short-term financial obligations."
    elif category == 'leverage':
        return "Measures the company's use of debt financing and financial risk."
    elif category == 'efficiency':
        return "Measures how efficiently the company uses its assets and resources."
    else:
        return f"Financial ratio that helps analyze the company's performance. Formula: {formula}"

def _generate_calculation_guidance(formula: str) -> Dict[str, str]:
    """Generate guidance on how to calculate the ratio"""
    guidance = {
        'formula_explanation': formula,
        'data_sources': "Values are calculated using financial statement data from SEC filings",
        'calculation_method': "Automated calculation using the latest available financial data"
    }
    
    # Add specific guidance based on formula components
    if 'net_income' in formula.lower():
        guidance['net_income_note'] = "Net income is taken from the income statement"
    if 'total_assets' in formula.lower():
        guidance['total_assets_note'] = "Total assets are taken from the balance sheet"
    if 'stockholders_equity' in formula.lower() or 'shareholders_equity' in formula.lower():
        guidance['equity_note'] = "Shareholders' equity is taken from the balance sheet"
    if 'current_assets' in formula.lower() or 'current_liabilities' in formula.lower():
        guidance['current_items_note'] = "Current assets and liabilities are taken from the balance sheet"
    if 'total_revenue' in formula.lower():
        guidance['revenue_note'] = "Total revenue is taken from the income statement"
    
    return guidance

def compare_companies_for_agent(tickers: List[str], categories: List[str], period: str = 'latest') -> FinancialAgentResponse:
    """
    Compare financial ratios across multiple companies (agent-friendly)
    
    This function gets ratios for each company individually using get_ratios_for_agent(),
    then combines the results for comparative analysis.
    
    Args:
        tickers: List of company ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        categories: List of ratio categories to compare (e.g., ['profitability', 'liquidity'])
        period: Period specification (same formats as get_ratios_for_agent)
        
    Returns:
        FinancialAgentResponse with comparative ratio analysis data
    """
    try:
        # Input validation
        if not tickers or not isinstance(tickers, list):
            return create_error_response(ErrorCodes.VALIDATION_ERROR, "Tickers must be a non-empty list")
        
        if len(tickers) < 2:
            return create_error_response(ErrorCodes.VALIDATION_ERROR, "At least 2 companies required for comparison")
        
        if len(tickers) > 10:
            return create_error_response(ErrorCodes.VALIDATION_ERROR, "Maximum 10 companies allowed for comparison")
        
        # Validate each ticker
        for ticker in tickers:
            ticker_error = validate_ticker(ticker)
            if ticker_error:
                return create_error_response(ErrorCodes.INVALID_TICKER, f"Invalid ticker '{ticker}': {ticker_error}")
        
        # Validate categories and period
        if categories and 'all' not in categories:
            for category in categories:
                category_error = validate_ratio_category(category)
                if category_error:
                    return create_error_response(ErrorCodes.VALIDATION_ERROR, f"Invalid category '{category}': {category_error}")
            
        period_error = validate_period(period)
        if period_error:
            return create_error_response(ErrorCodes.INVALID_PERIOD, period_error)
        
        # Get ratio data for each company using get_ratios_for_agent
        company_ratio_data = {}
        successful_companies = []
        failed_companies = []
        
        
        for ticker in tickers:
            try:
                # Use get_ratios_for_agent for each company individually
                response = get_ratios_for_agent(ticker.upper(), categories, period)
                
                if response.success and response.data:
                    company_ratio_data[ticker.upper()] = response.data
                    successful_companies.append(ticker.upper())
                else:
                    failed_companies.append(ticker.upper())
            except Exception as e:
                failed_companies.append(ticker.upper())
                logger.error(f"Error getting ratios for {ticker}: {e}")
        
        if not successful_companies:
            return create_error_response(
                ErrorCodes.DATA_NOT_FOUND,
                f"No ratio data found for any of the companies: {', '.join(tickers)}"
            )
        
        # Structure comparison data by category and ratio
        comparison_data = {
            'categories_comparison': {},
            'company_summaries': {},
            'rankings': {},
            'statistics': {}
        }
        
        # For multi-period comparisons, we need to handle each period separately
        # Check if this is a trend analysis request
        is_trend_analysis = period.startswith('last ') and ('quarters' in period or 'financial years' in period)
        
        if is_trend_analysis:
            # For trend analysis, organize data by period and then by ratio
            comparison_data['categories_comparison'] = _organize_trend_comparison_data(company_ratio_data, categories, successful_companies, period)
        else:
            # For single period comparisons, use the existing logic
            comparison_data['categories_comparison'] = _organize_single_period_comparison_data(company_ratio_data, categories, successful_companies)
        
        # Create company summaries
        for ticker in successful_companies:
            # Fix: Access 'ratios' key from each category data structure
            total_ratios = sum(
                len(category_data.get('ratios', {})) if isinstance(category_data, dict) else 0
                for category_data in company_ratio_data[ticker].values()
            )
            comparison_data['company_summaries'][ticker] = {
                'ticker': ticker,
                'categories_available': list(company_ratio_data[ticker].keys()),
                'total_ratios': total_ratios,
                'data_completeness': 1.0  # Since we got data successfully
            }
        
        # Overall statistics
        comparison_data['statistics'] = {
            'total_companies_requested': len(tickers),
            'successful_companies': len(successful_companies),
            'failed_companies': len(failed_companies),
            'categories_analyzed': len(categories) if categories else len(comparison_data['categories_comparison']),
            'period_analyzed': period
        }
        
        # Create metadata
        metadata = {
            'tickers_requested': [t.upper() for t in tickers],
            'successful_tickers': successful_companies,
            'failed_tickers': failed_companies,
            'categories_requested': categories,
            'period': period,
            'comparison_type': 'ratio_comparison',
            'data_source': 'SECFinancialRAG',
            'correlation_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return create_success_response(comparison_data, metadata)
        
    except Exception as e:
        logger.error(f"Error comparing companies {tickers}: {e}")
        return create_error_response(
            ErrorCodes.PROCESSING_ERROR,
            f"Error processing comparison request: {str(e)}"
        )


def _organize_trend_comparison_data(company_ratio_data, categories, successful_companies, period):
    """
    Organize comparison data for trend analysis (multiple periods)
    Returns data structured by category -> ratio -> periods
    """
    comparison_data = {}
    
    # Process each category
    for category in categories if categories else ['all']:
        if category == 'all':
            # Get all categories from the data
            available_categories = set()
            for company_data in company_ratio_data.values():
                available_categories.update(company_data.keys())
            categories_to_process = list(available_categories)
        else:
            categories_to_process = [category]
            
        for cat in categories_to_process:
            category_comparison = {
                'category_name': cat,
                'ratios': {},
                'company_count': 0,
                'ratio_count': 0
            }
            
            # Get all unique ratio names for this category (with their periods)
            all_ratio_keys = set()
            for company_data in company_ratio_data.values():
                if cat in company_data and isinstance(company_data[cat], dict):
                    # Access the 'ratios' key from the category data structure
                    ratios_dict = company_data[cat].get('ratios', {})
                    all_ratio_keys.update(ratios_dict.keys())
            
            # Group ratio keys by base name but keep period information
            ratios_by_base_name = {}
            for ratio_key in all_ratio_keys:
                # Extract base name (remove period suffix)
                base_name = ratio_key.split('_FY')[0].split('_Q')[0]
                if base_name not in ratios_by_base_name:
                    ratios_by_base_name[base_name] = []
                ratios_by_base_name[base_name].append(ratio_key)
            
            
            # For trend analysis, handle financial years vs quarters differently
            for base_ratio_name, ratio_keys in ratios_by_base_name.items():
                if 'financial years' in period:
                    # For financial years, align by period_end_date instead of fiscal_year numbers
                    aligned_periods = _align_financial_years_by_period_end_date(company_ratio_data, cat, base_ratio_name, successful_companies, period)
                    
                    for period_idx, period_data in enumerate(aligned_periods):
                        trend_key = f"{base_ratio_name}_FY_Period_{period_idx + 1}"
                        
                        ratio_comparison = {
                            'ratio_name': base_ratio_name,
                            'period_info': period_data['period_info'],
                            'company_values': period_data['company_values'],
                            'statistics': {}
                        }
                        
                        # Calculate statistics (remove rankings)
                        if period_data['company_values']:
                            ratio_values = [data['value'] for data in period_data['company_values'].values()]
                            
                            # Calculate statistics
                            ratio_comparison['statistics'] = {
                                'count': len(ratio_values),
                                'max': max(ratio_values),
                                'min': min(ratio_values),
                                'average': sum(ratio_values) / len(ratio_values),
                                'median': sorted(ratio_values)[len(ratio_values) // 2] if ratio_values else 0
                            }
                            
                            category_comparison['ratios'][trend_key] = ratio_comparison
                        
                else:
                    # For quarters, align by period_end_date instead of fiscal quarter labels
                    aligned_periods = _align_quarters_by_period_end_date(company_ratio_data, cat, base_ratio_name, successful_companies, period)

                    for period_idx, period_data in enumerate(aligned_periods):
                        trend_key = f"{base_ratio_name}_Q_Period_{period_idx + 1}"

                        ratio_comparison = {
                            'ratio_name': base_ratio_name,
                            'period_info': period_data['period_info'],
                            'company_values': period_data['company_values'],
                            'statistics': {}
                        }

                        # Calculate statistics (remove rankings)
                        if period_data['company_values']:
                            ratio_values = [data['value'] for data in period_data['company_values'].values()]

                            # Calculate statistics
                            ratio_comparison['statistics'] = {
                                'count': len(ratio_values),
                                'max': max(ratio_values),
                                'min': min(ratio_values),
                                'average': sum(ratio_values) / len(ratio_values),
                                'median': sorted(ratio_values)[len(ratio_values) // 2] if ratio_values else 0
                            }

                            category_comparison['ratios'][trend_key] = ratio_comparison
            
            category_comparison['company_count'] = len(successful_companies)
            category_comparison['ratio_count'] = len(category_comparison['ratios'])
            comparison_data[cat] = category_comparison
    
    return comparison_data


def _organize_single_period_comparison_data(company_ratio_data, categories, successful_companies):
    """
    Organize comparison data for single period analysis (existing logic)
    """
    comparison_data = {}
    
    # Process each category
    for category in categories if categories else ['all']:
        if category == 'all':
            # Get all categories from the data
            available_categories = set()
            for company_data in company_ratio_data.values():
                available_categories.update(company_data.keys())
            categories_to_process = list(available_categories)
        else:
            categories_to_process = [category]
            
        for cat in categories_to_process:
            category_comparison = {
                'category_name': cat,
                'ratios': {},
                'company_count': 0,
                'ratio_count': 0
            }
            
            # Get all ratio names for this category across companies (extract base names)
            category_ratios = set()
            for company_data in company_ratio_data.values():
                if cat in company_data and isinstance(company_data[cat], dict):
                    # Access the 'ratios' key from the category data structure
                    ratios_dict = company_data[cat].get('ratios', {})
                    for ratio_key in ratios_dict.keys():
                        # Extract base name for comparison
                        base_name = ratio_key.split('_FY')[0].split('_Q')[0]
                        category_ratios.add(base_name)
            
            # Compare each ratio across companies
            for ratio_name in category_ratios:
                ratio_comparison = {
                    'ratio_name': ratio_name,
                    'company_values': {},
                    'statistics': {}
                }
                
                ratio_values = []
                companies_with_ratio = []
                
                # Collect values for this ratio from all companies
                for ticker in successful_companies:
                    if ticker in company_ratio_data and cat in company_ratio_data[ticker]:
                        category_data = company_ratio_data[ticker][cat]
                        # Access the 'ratios' key from the category data structure
                        company_ratios = category_data.get('ratios', {}) if isinstance(category_data, dict) else {}
                        
                        # Find this ratio (may have period suffix)
                        ratio_data = None
                        for ratio_key, data in company_ratios.items():
                            if ratio_key.startswith(ratio_name):
                                ratio_data = data
                                break
                        
                        if ratio_data:
                            try:
                                value = float(ratio_data.get('value', 0))
                                ratio_comparison['company_values'][ticker] = {
                                    'value': value,
                                    'period_end_date': ratio_data.get('period_end_date'),
                                    'fiscal_year': ratio_data.get('fiscal_year'),
                                    'formula': ratio_data.get('formula'),
                                    'description': ratio_data.get('description'),
                                    'calculation_inputs': ratio_data.get('calculation_inputs')  # Add underlying $ values
                                }
                                ratio_values.append(value)
                                companies_with_ratio.append(ticker)
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid ratio value for {ticker} {ratio_name}: {ratio_data.get('value')}")
                
                # Calculate statistics if we have data (remove rankings)
                if ratio_values and companies_with_ratio:
                    # Calculate statistics
                    ratio_comparison['statistics'] = {
                        'count': len(ratio_values),
                        'max': max(ratio_values),
                        'min': min(ratio_values),
                        'average': sum(ratio_values) / len(ratio_values),
                        'median': sorted(ratio_values)[len(ratio_values) // 2] if ratio_values else 0
                    }
                
                category_comparison['ratios'][ratio_name] = ratio_comparison
                
            category_comparison['company_count'] = len(successful_companies)
            category_comparison['ratio_count'] = len(category_ratios)
            comparison_data[cat] = category_comparison
    
    return comparison_data


def _consolidate_financial_data_by_period(df):
    """
    Consolidate financial data by period_end_date.
    For each period_end_date, merge data from multiple statement types (income, cash flow, balance sheet)
    into a single row with all available financial data.
    """
    import pandas as pd
    
    if df.empty:
        return df
    
    # Ensure period_end_date is in datetime format for proper sorting
    df = df.copy()
    df['period_end_date'] = pd.to_datetime(df['period_end_date'])
    
    # Group by period_end_date and consolidate data
    consolidated_data = []
    
    for period_date, group in df.groupby('period_end_date'):
        # Start with the first row as base
        consolidated_row = group.iloc[0].copy()
        
        # For each subsequent row in this period group, merge non-null values
        for _, row in group.iloc[1:].iterrows():
            for col in row.index:
                # If the consolidated row has null/NaN and this row has a value, use it
                if (pd.isna(consolidated_row[col]) or consolidated_row[col] is None) and \
                   (not pd.isna(row[col]) and row[col] is not None):
                    consolidated_row[col] = row[col]
        
        consolidated_data.append(consolidated_row)
    
    # Create new DataFrame from consolidated data
    if consolidated_data:
        consolidated_df = pd.DataFrame(consolidated_data)
        
        # Ensure period_end_date remains datetime
        consolidated_df['period_end_date'] = pd.to_datetime(consolidated_df['period_end_date'])
        
        # Sort by period_end_date (most recent first)
        consolidated_df = consolidated_df.sort_values('period_end_date', ascending=False)
        
        return consolidated_df.reset_index(drop=True)
    else:
        return df.iloc[0:0]  # Return empty DataFrame with same structure


def _filter_financial_data_by_period(df, period: str):
    """
    Filter financial data DataFrame by enhanced period formats.
    Similar to _filter_ratios_by_period but works on financial data structure.
    """
    import re
    import pandas as pd
    
    # Specific year patterns: FY2024, Q1-2024, etc.
    year_pattern = r'^(FY|Q[1-4])-?(\d{4})$'
    match = re.match(year_pattern, period)
    if match:
        period_type, year = match.groups()
        year = int(year)
        
        if 'fiscal_year' in df.columns:
            # Smart handling for different data types
            available_period_types = df['period_type'].unique()
            
            if period_type == 'FY':
                # For FY requests, collect data from ALL appropriate sources for the fiscal year
                # Different statement types use different period types for annual data
                result_data = []
                
                # 1. Direct FY data (if available)
                if 'FY' in available_period_types:
                    fy_data = df[(df['period_type'] == 'FY') & (df['fiscal_year'] == year)]
                    if not fy_data.empty:
                        result_data.append(fy_data)
                
                # 2. LTM data for income/cash flow statements
                if 'LTM' in available_period_types:
                    ltm_data = df[(df['fiscal_year'] == year) & (df['period_type'] == 'LTM')].copy()
                    if not ltm_data.empty:
                        # Get the most recent (year-end) LTM record for EACH statement type
                        ltm_data['period_end_date'] = pd.to_datetime(ltm_data['period_end_date'])
                        
                        # Group by statement type and get the latest record for each
                        ltm_grouped = ltm_data.groupby('statement_type')
                        ltm_latest_records = []
                        
                        for stmt_type, group in ltm_grouped:
                            # Get the record with the latest period_end_date for this statement type
                            latest_idx = group['period_end_date'].idxmax()
                            latest_record = group.loc[[latest_idx]]  # Keep as DataFrame
                            ltm_latest_records.append(latest_record)
                        
                        if ltm_latest_records:
                            ltm_combined = pd.concat(ltm_latest_records, ignore_index=True)
                            result_data.append(ltm_combined)
                
                # 3. Q4 data for balance sheet statements  
                if 'Q4' in available_period_types:
                    q4_data = df[(df['period_type'] == 'Q4') & (df['fiscal_year'] == year)]
                    if not q4_data.empty:
                        result_data.append(q4_data)
                
                # Combine all the fiscal year data
                if result_data:
                    return pd.concat(result_data, ignore_index=True)
                else:
                    return df.iloc[0:0]  # Empty DataFrame
            else:
                # For quarterly requests (Q1, Q2, Q3, Q4)
                # Get all data for the specified quarter, regardless of statement type
                quarterly_data = df[(df['period_type'] == period_type) & (df['fiscal_year'] == year)]
                
                # Also get LTM data for the same period end date if quarterly data exists
                if not quarterly_data.empty and 'LTM' in available_period_types:
                    # Get the period end date from quarterly data
                    quarterly_end_dates = quarterly_data['period_end_date'].unique()
                    
                    # Get LTM data for the same period end dates
                    ltm_quarterly = df[
                        (df['period_type'] == 'LTM') & 
                        (df['period_end_date'].isin(quarterly_end_dates))
                    ]
                    
                    if not ltm_quarterly.empty:
                        return pd.concat([quarterly_data, ltm_quarterly], ignore_index=True)
                
                return quarterly_data
        else:
            # Fallback to period_end_date year matching
            df_copy = df.copy()
            df_copy['period_end_date'] = pd.to_datetime(df_copy['period_end_date'])
            df_copy['end_year'] = df_copy['period_end_date'].dt.year
            return df_copy[(df_copy['period_type'] == period_type) & (df_copy['end_year'] == year)]
    
    # Trend patterns: last n quarters, last n financial years
    trend_pattern = r'^last (\d+) (quarters|financial years)$'
    match = re.match(trend_pattern, period)
    if match:
        count = int(match.group(1))
        trend_type = match.group(2)
        
        if trend_type == 'quarters':
            # For last N quarters: get all financial data from the N most recent quarterly periods
            # Need to handle both quarterly data (balance sheets) and LTM data (income/cash flow statements)
            
            # Step 1: Get quarterly balance sheet data to determine which periods we want
            quarterly_balance_data = df[df['period_type'].isin(['Q1', 'Q2', 'Q3', 'Q4'])].copy()
            
            if quarterly_balance_data.empty:
                return quarterly_balance_data
            
            quarterly_balance_data['period_end_date'] = pd.to_datetime(quarterly_balance_data['period_end_date'])
            
            # Step 2: Determine the N most recent quarterly period end dates
            if 'fiscal_year' in quarterly_balance_data.columns:
                unique_periods = quarterly_balance_data.groupby(['period_end_date', 'fiscal_year']).first().reset_index()
                target_periods = unique_periods.sort_values('period_end_date', ascending=False).head(count)
                target_dates = target_periods['period_end_date'].tolist()
            else:
                unique_periods = quarterly_balance_data.groupby('period_end_date')['period_end_date'].first()
                target_dates = unique_periods.sort_values(ascending=False).head(count).tolist()
            
            # Step 3: Get ALL data (quarterly + LTM) that matches these period end dates
            df_copy = df.copy()
            df_copy['period_end_date'] = pd.to_datetime(df_copy['period_end_date'])
            
            # Filter for data matching the target quarterly dates
            # This includes: quarterly balance sheets + LTM income/cash flow for same dates
            matching_data = df_copy[df_copy['period_end_date'].isin(target_dates)]
            
            if matching_data.empty:
                return matching_data
            
            # Step 4: For each target date, get all available statement types
            result_data = []
            for target_date in sorted(target_dates, reverse=True):  # Most recent first
                date_data = matching_data[matching_data['period_end_date'] == target_date]
                if not date_data.empty:
                    result_data.append(date_data)
            
            if result_data:
                return pd.concat(result_data, ignore_index=True)
            else:
                return pd.DataFrame()
        
        elif trend_type == 'financial years':
            # For last N financial years: get data from the N most recent fiscal years
            # Use Q4 balance sheet data to determine the target fiscal years (most reliable)
            if 'fiscal_year' in df.columns and not df['fiscal_year'].dropna().empty:
                available_period_types = df['period_type'].unique()
                
                # Step 1: Determine target fiscal years based on Q4 data (most reliable year-end marker)
                target_years = []
                if 'Q4' in available_period_types:
                    q4_data = df[df['period_type'] == 'Q4'].copy()
                    if not q4_data.empty:
                        q4_data['period_end_date'] = pd.to_datetime(q4_data['period_end_date'])
                        unique_years = q4_data.groupby('fiscal_year')['period_end_date'].max().reset_index()
                        unique_years = unique_years.sort_values('period_end_date', ascending=False).head(count)
                        target_years = unique_years['fiscal_year'].tolist()
                
                # Fallback: if no Q4 data, use any available fiscal years
                if not target_years:
                    all_years = sorted(df['fiscal_year'].unique(), reverse=True)
                    target_years = all_years[:count]
                
                # Step 2: For each target year, get appropriate data for each statement type
                result_data = []
                for fy in target_years:
                    year_data_parts = []
                    
                    # Get FY data if available
                    if 'FY' in available_period_types:
                        fy_data = df[(df['fiscal_year'] == fy) & (df['period_type'] == 'FY')]
                        if not fy_data.empty:
                            year_data_parts.append(fy_data)
                    
                    # Get Q4 data (balance sheet year-end)
                    if 'Q4' in available_period_types:
                        q4_data = df[(df['fiscal_year'] == fy) & (df['period_type'] == 'Q4')]
                        if not q4_data.empty:
                            year_data_parts.append(q4_data)
                    
                    # Get year-end LTM data (income/cash flow annual performance)
                    if 'LTM' in available_period_types:
                        ltm_fy_data = df[(df['fiscal_year'] == fy) & (df['period_type'] == 'LTM')].copy()
                        if not ltm_fy_data.empty:
                            # Get the most recent LTM data for EACH statement type in this fiscal year
                            ltm_fy_data['period_end_date'] = pd.to_datetime(ltm_fy_data['period_end_date'])
                            
                            # Group by statement type and get latest record for each
                            ltm_grouped = ltm_fy_data.groupby('statement_type')
                            for stmt_type, group in ltm_grouped:
                                latest_idx = group['period_end_date'].idxmax()
                                latest_record = group.loc[[latest_idx]]  # Keep as DataFrame
                                year_data_parts.append(latest_record)
                    
                    # Combine all data for this fiscal year
                    if year_data_parts:
                        year_combined = pd.concat(year_data_parts, ignore_index=True)
                        result_data.append(year_combined)
                
                if result_data:
                    return pd.concat(result_data, ignore_index=True)
                else:
                    # No FY records found (common for balance sheet data)
                    # Get Q4 (year-end) data from each of the N most recent fiscal years
                    # Q4 is most representative for financial year comparisons
                    df_copy = df.copy()
                    df_copy['period_end_date'] = pd.to_datetime(df_copy['period_end_date'])
                    
                    # Get all available fiscal years sorted by most recent first
                    all_fiscal_years = sorted(df_copy['fiscal_year'].unique(), reverse=True)
                    
                    result_data = []
                    years_found = 0
                    
                    # Iterate through fiscal years until we have enough data or run out of years
                    for fy in all_fiscal_years:
                        if years_found >= count:
                            break
                            
                        fy_data = df_copy[df_copy['fiscal_year'] == fy]
                        if not fy_data.empty:
                            # First try to get Q4 data for this fiscal year (year-end)
                            q4_data = fy_data[fy_data['period_type'] == 'Q4']
                            if not q4_data.empty:
                                # Use Q4 data (most representative for year-end)
                                selected_record = q4_data.head(1)
                                result_data.append(selected_record)
                                years_found += 1
                            else:
                                # Fallback: get the most recent QUARTERLY record (exclude LTM/calculated records)
                                # Filter to only quarterly data (Q1, Q2, Q3, Q4)
                                quarterly_data = fy_data[fy_data['period_type'].isin(['Q1', 'Q2', 'Q3', 'Q4'])]
                                if not quarterly_data.empty:
                                    most_recent_quarterly = quarterly_data.sort_values('period_end_date', ascending=False).head(1)
                                    result_data.append(most_recent_quarterly)
                                    years_found += 1
                                else:
                                    # Final fallback: if no quarterly data, use most recent (including LTM)
                                    most_recent = fy_data.sort_values('period_end_date', ascending=False).head(1)
                                    result_data.append(most_recent)
                                    years_found += 1
                    
                    if result_data:
                        # Combine all selected records
                        combined_result = pd.concat(result_data, ignore_index=True)
                        
                        # Ensure we have unique records (remove any potential duplicates)
                        # Sort by period_end_date (most recent first) and remove duplicates
                        combined_result = combined_result.sort_values('period_end_date', ascending=False)
                        combined_result = combined_result.drop_duplicates(subset=['period_end_date', 'fiscal_year'], keep='first')
                        
                        # Limit to requested count in case we somehow got more
                        return combined_result.head(count)
            else:
                # Fallback: use period_type and period_end_date for years
                df_copy = df.copy()
                df_copy['period_end_date'] = pd.to_datetime(df_copy['period_end_date'])
                df_copy['year'] = df_copy['period_end_date'].dt.year
                
                # Get the N most recent years
                recent_years = sorted(df_copy['year'].unique(), reverse=True)[:count]
                
                # For each year, get the most recent quarter
                result_data = []
                for year in recent_years:
                    year_data = df_copy[df_copy['year'] == year]
                    if not year_data.empty:
                        most_recent = year_data.sort_values('period_end_date', ascending=False).head(1)
                        result_data.append(most_recent)
                
                if result_data:
                    return pd.concat(result_data, ignore_index=True)
    
    # If no pattern matched, return empty DataFrame
    return df.iloc[0:0]


def _extract_period_info_from_key(ratio_key):
    """
    Extract period information from ratio key like 'ROE_Q4-2024' or 'ROE_FY2024'
    """
    if '_FY' in ratio_key:
        year = ratio_key.split('_FY')[1]
        return {'type': 'Financial Year', 'year': year, 'quarter': None}
    elif '_Q' in ratio_key and '-' in ratio_key:
        parts = ratio_key.split('_Q')[1].split('-')
        quarter = f"Q{parts[0]}"
        year = parts[1] if len(parts) > 1 else None
        return {'type': 'Quarter', 'year': year, 'quarter': quarter}
    else:
        return {'type': 'Unknown', 'year': None, 'quarter': None}


def _align_quarters_by_period_end_date(company_ratio_data, category, base_ratio_name, successful_companies, period):
    """
    Align quarters across companies by clustering period_end_dates by temporal proximity.

    Instead of forcing "1st most recent, 2nd most recent" matching, this groups quarters that
    are temporally close (within ~45 days), handling cases where some companies report later
    than others. For example, if AAPL reports 9/30 but peers only have 6/30 available, AAPL's
    9/30 will be in a separate period (possibly alone) while 6/30 quarters from all companies
    cluster together.

    Args:
        company_ratio_data: Dictionary of company ratio data
        category: Ratio category being analyzed
        base_ratio_name: Base name of ratio (e.g., 'Current_Ratio')
        successful_companies: List of company tickers
        period: Period string (e.g., 'last 12 quarters')

    Returns:
        List of aligned period data dictionaries, ordered from most recent to oldest
    """
    from datetime import datetime, timedelta
    import re

    # Extract number of quarters requested
    match = re.search(r'last (\d+) quarters?', period)
    if not match:
        logger.warning(f"Could not parse number of quarters from period: {period}")
        return []

    num_quarters = int(match.group(1))

    # Collect all quarterly ratio data from all companies with their period_end_dates
    all_quarterly_data = []

    for ticker in successful_companies:
        if ticker in company_ratio_data and category in company_ratio_data[ticker]:
            category_data = company_ratio_data[ticker][category]
            company_ratios = category_data.get('ratios', {}) if isinstance(category_data, dict) else {}

            # Find all quarterly ratios for this base ratio name (exclude FY ratios)
            for ratio_key, ratio_data in company_ratios.items():
                if (ratio_key.startswith(base_ratio_name) and '_Q' in ratio_key and '_FY' not in ratio_key):
                    try:
                        period_end_date_str = ratio_data.get('period_end_date')
                        if period_end_date_str:
                            period_end_date = datetime.strptime(period_end_date_str, '%Y-%m-%d').date()

                            all_quarterly_data.append({
                                'ticker': ticker,
                                'ratio_key': ratio_key,
                                'period_end_date': period_end_date,
                                'fiscal_year': ratio_data.get('fiscal_year'),
                                'ratio_data': ratio_data
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse period_end_date for {ticker} {ratio_key}: {e}")

    if not all_quarterly_data:
        logger.warning(f"No quarterly data found for {base_ratio_name} across companies")
        return []

    # Sort ALL data by period_end_date (most recent first)
    all_quarterly_data.sort(key=lambda x: x['period_end_date'], reverse=True)

    # TEMPORAL CLUSTERING: Group quarters by date proximity
    # Dates within THRESHOLD days are considered "same period"
    PROXIMITY_THRESHOLD_DAYS = 45

    aligned_periods = []
    used_indices = set()

    while len(aligned_periods) < num_quarters and len(used_indices) < len(all_quarterly_data):
        # Find the most recent unused date as anchor for this period
        anchor_idx = None
        anchor_date = None

        for idx, data in enumerate(all_quarterly_data):
            if idx not in used_indices:
                anchor_idx = idx
                anchor_date = data['period_end_date']
                break

        if anchor_idx is None:
            break  # No more unused data

        # For each company, find their data closest to anchor (within threshold, unused)
        period_data_by_company = {}

        for ticker in successful_companies:
            # Find all unused data for this company within threshold of anchor
            company_candidates = []

            for idx, data in enumerate(all_quarterly_data):
                if (data['ticker'] == ticker and
                    idx not in used_indices and
                    abs((data['period_end_date'] - anchor_date).days) <= PROXIMITY_THRESHOLD_DAYS):
                    company_candidates.append((idx, data))

            if company_candidates:
                # Pick the one closest to anchor date
                best_idx, best_data = min(
                    company_candidates,
                    key=lambda x: abs((x[1]['period_end_date'] - anchor_date).days)
                )

                # Extract fiscal quarter from ratio_key
                fiscal_quarter = None
                ratio_key = best_data['ratio_key']
                if '_Q' in ratio_key:
                    quarter_match = re.search(r'_Q(\d)-(\d{4})', ratio_key)
                    if quarter_match:
                        fiscal_quarter = f"Q{quarter_match.group(1)}-{quarter_match.group(2)}"

                period_data_by_company[ticker] = {
                    'value': best_data['ratio_data'].get('value'),
                    'period_end_date': best_data['period_end_date'].strftime('%Y-%m-%d'),
                    'fiscal_year': best_data['fiscal_year'],
                    'fiscal_quarter': fiscal_quarter,
                    'formula': best_data['ratio_data'].get('formula'),
                    'description': best_data['ratio_data'].get('description'),
                    'calculation_inputs': best_data['ratio_data'].get('calculation_inputs')
                }

                used_indices.add(best_idx)

        # Create period if we have data from at least one company
        if period_data_by_company:
            period_dates = [
                datetime.strptime(data['period_end_date'], '%Y-%m-%d').date()
                for data in period_data_by_company.values()
            ]

            # Calculate cluster center for reference
            avg_days = sum((d - datetime(1970, 1, 1).date()).days for d in period_dates) / len(period_dates)
            cluster_center = datetime(1970, 1, 1).date() + timedelta(days=int(avg_days))

            period_info = {
                'type': 'Quarter',
                'period_number': len(aligned_periods) + 1,
                'clustering_method': f'Temporal proximity (within {PROXIMITY_THRESHOLD_DAYS} days)',
                'cluster_center_date': cluster_center.strftime('%Y-%m-%d'),
                'companies_included': list(period_data_by_company.keys()),
                'companies_missing': [t for t in successful_companies if t not in period_data_by_company],
                'date_range': {
                    'earliest': min(period_dates).strftime('%Y-%m-%d'),
                    'latest': max(period_dates).strftime('%Y-%m-%d'),
                    'span_days': (max(period_dates) - min(period_dates)).days
                },
                'note': 'Companies may be missing if they have not yet reported for this period'
            }

            aligned_periods.append({
                'period_info': period_info,
                'company_values': period_data_by_company
            })

    return aligned_periods


def _align_financial_years_by_period_end_date(company_ratio_data, category, base_ratio_name, successful_companies, period):
    """
    Align financial years across companies based on period_end_date proximity instead of fiscal_year numbers.
    
    This ensures that when comparing financial years, we're comparing periods that are actually 
    temporally close to each other, rather than just matching fiscal year numbers which can 
    have vastly different ending dates.
    
    Args:
        company_ratio_data: Dictionary of company ratio data
        category: Ratio category being analyzed  
        base_ratio_name: Base name of ratio (e.g., 'ROE')
        successful_companies: List of company tickers
        period: Period string (e.g., 'last 3 financial years')
        
    Returns:
        List of aligned period data dictionaries, ordered from most recent to oldest
    """
    from datetime import datetime, timedelta
    import re
    
    
    # Extract number of years requested
    match = re.search(r'last (\d+) financial years', period)
    if not match:
        logger.warning(f"Could not parse number of years from period: {period}")
        return []
    
    num_years = int(match.group(1))
    
    # Collect all FY ratio data from all companies with their period_end_dates
    all_fy_data = []
    
    for ticker in successful_companies:
        if ticker in company_ratio_data and category in company_ratio_data[ticker]:
            category_data = company_ratio_data[ticker][category]
            # Access the 'ratios' key from the category data structure
            company_ratios = category_data.get('ratios', {}) if isinstance(category_data, dict) else {}
            
            # Find all FY ratios for this base ratio name
            for ratio_key, ratio_data in company_ratios.items():
                if (ratio_key.startswith(base_ratio_name) and '_FY' in ratio_key):
                    try:
                        period_end_date_str = ratio_data.get('period_end_date')
                        if period_end_date_str:
                            period_end_date = datetime.strptime(period_end_date_str, '%Y-%m-%d').date()
                            
                            all_fy_data.append({
                                'ticker': ticker,
                                'ratio_key': ratio_key,
                                'period_end_date': period_end_date,
                                'fiscal_year': ratio_data.get('fiscal_year'),
                                'ratio_data': ratio_data
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse period_end_date for {ticker} {ratio_key}: {e}")
    
    if not all_fy_data:
        logger.warning(f"No FY data found for {base_ratio_name} across companies")
        return []
    
    # Sort all data by period_end_date (most recent first)
    all_fy_data.sort(key=lambda x: x['period_end_date'], reverse=True)
    
    # Group periods by finding the best temporal alignment
    aligned_periods = []
    used_data_indices = set()
    
    for period_idx in range(num_years):
        
        period_companies = {}
        current_period_dates = []
        
        # For each company, find the best available period for this alignment
        for ticker in successful_companies:
            # Get all unused data for this company
            company_data = [
                (idx, data) for idx, data in enumerate(all_fy_data) 
                if data['ticker'] == ticker and idx not in used_data_indices
            ]
            
            if company_data:
                # For the first period, take the most recent; for subsequent periods,
                # try to find data that's temporally close to what we've already selected
                if not current_period_dates:
                    # First period - take the most recent
                    best_idx, best_data = company_data[0]
                else:
                    # Find the data point that's closest to the average date of current period
                    avg_date = sum((d - datetime(1970, 1, 1).date()).days for d in current_period_dates) / len(current_period_dates)
                    avg_date = datetime(1970, 1, 1).date() + timedelta(days=int(avg_date))
                    
                    # Find closest match within reasonable range (e.g., 6 months)
                    best_idx, best_data = min(
                        company_data,
                        key=lambda x: abs((x[1]['period_end_date'] - avg_date).days)
                    )
                    
                    # Skip if the date is too far from the target (more than 6 months)
                    if abs((best_data['period_end_date'] - avg_date).days) > 180:
                        continue
                
                # Extract fiscal period from ratio_key (e.g., "ROE_FY-2025" -> "FY-2025")
                fiscal_period = None
                ratio_key = best_data['ratio_key']
                if '_FY' in ratio_key:
                    # Extract FY-pattern from the ratio key
                    import re
                    fy_match = re.search(r'_FY-?(\d{4})', ratio_key)
                    if fy_match:
                        fiscal_period = f"FY-{fy_match.group(1)}"

                period_companies[ticker] = {
                    'value': best_data['ratio_data'].get('value'),
                    'period_end_date': best_data['period_end_date'].strftime('%Y-%m-%d'),
                    'fiscal_year': best_data['fiscal_year'],
                    'fiscal_period': fiscal_period,  # Add fiscal period (FY-2025 format)
                    'formula': best_data['ratio_data'].get('formula'),
                    'description': best_data['ratio_data'].get('description'),
                    'calculation_inputs': best_data['ratio_data'].get('calculation_inputs')  # Add underlying $ values
                }

                current_period_dates.append(best_data['period_end_date'])
                used_data_indices.add(best_idx)
        
        # Only add this period if we have data from at least 2 companies (meaningful comparison)
        if len(period_companies) >= 2:
            # Calculate period info based on the data we've collected
            period_info = {
                'type': 'Financial Year Alignment',
                'period_number': period_idx + 1,
                'companies_included': list(period_companies.keys()),
                'date_range': {
                    'earliest': min(current_period_dates).strftime('%Y-%m-%d'),
                    'latest': max(current_period_dates).strftime('%Y-%m-%d')
                } if current_period_dates else None,
                'fiscal_years_represented': list(set(data['fiscal_year'] for data in period_companies.values()))
            }
            
            aligned_periods.append({
                'period_info': period_info,
                'company_values': period_companies
            })
        else:
            break
    
    return aligned_periods