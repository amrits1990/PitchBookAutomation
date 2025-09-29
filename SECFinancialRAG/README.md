# SECFinancialRAG - AI-Ready Financial Analysis Module

A production-ready financial RAG (Retrieval-Augmented Generation) system designed for AI agents. Provides comprehensive financial analysis capabilities with SEC EDGAR data integration, intelligent caching, and standardized agent interfaces.

## 🤖 **AI Agent Interface**

**Primary interface for AI agents - standardized JSON responses with comprehensive error handling:**

```python
from SECFinancialRAG.agent_interface import (
    get_financial_metrics_for_agent,
    get_ratios_for_agent,
    compare_companies_for_agent,
    get_ratio_definition_for_agent,
    get_available_ratio_categories_for_agent
)

# Get financial metrics with intelligent field mapping
response = get_financial_metrics_for_agent(
    ticker='AAPL', 
    metrics=['revenue', 'net income', 'free cash flow'],
    period='LTM'
)

# Get comprehensive ratio analysis
response = get_ratios_for_agent(
    ticker='AAPL',
    categories=['profitability', 'liquidity'],
    period='last 4 quarters'
)

# Compare multiple companies
response = compare_companies_for_agent(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    categories=['profitability'],
    period='latest'
)
```

### **Agent Response Format**

All agent functions return standardized `FinancialAgentResponse` objects:

```python
@dataclass
class FinancialAgentResponse:
    success: bool                    # Operation success status
    data: Optional[Dict[str, Any]]   # Structured financial data
    error: Optional[Dict[str, str]]  # Error details if failed
    metadata: Dict[str, Any]         # Response metadata & context
```

### **Supported Period Formats**

- **Latest**: `'latest'` - Most recent available data
- **Specific Years**: `'FY2024'`, `'Q1-2024'`, `'Q2-2024'` 
- **Trend Analysis**: `'last 3 quarters'`, `'last 2 financial years'`
- **Custom Ranges**: 1-10 periods supported

## 🚀 **Quick Start for AI Agents**

### **1. Financial Metrics Query**
```python
# Natural language to structured query
user_query = "What's Apple's revenue and profit margin for the last year?"

response = get_financial_metrics_for_agent(
    ticker='AAPL',
    metrics=['total_revenue', 'net_income', 'net_margin'],
    period='LTM'
)

if response.success:
    data = response.data
    # Access structured financial data
    revenue = data['total_revenue']['value']
    margin = data['net_margin']['value']
```

### **2. Ratio Analysis**
```python
# Comprehensive financial health analysis
response = get_ratios_for_agent(
    ticker='AAPL',
    categories=['profitability', 'liquidity', 'leverage'],
    period='latest'
)

# Response includes calculated ratios + explanations
ratios = response.data['ratios']
# {'ROE': 0.15, 'ROA': 0.12, 'Current_Ratio': 1.2, ...}
```

### **3. Company Comparison**
```python
# Multi-company competitive analysis
response = compare_companies_for_agent(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    categories=['profitability', 'efficiency'],
    period='LTM'
)

# Returns comparative data + rankings
comparison = response.data['comparison']
rankings = response.data['rankings']
```

## 📊 **Data Coverage & Capabilities**

### **Financial Statements**
- ✅ **Income Statements**: Revenue, expenses, profitability metrics
- ✅ **Balance Sheets**: Assets, liabilities, equity positions  
- ✅ **Cash Flow Statements**: Operating, investing, financing activities
- ✅ **LTM Calculations**: Rolling 12-month data for trend analysis

### **Financial Ratios (50+ Ratios)**
- **Profitability**: ROE, ROA, ROIC, Net Margin, EBITDA Margin, Gross Margin
- **Liquidity**: Current Ratio, Quick Ratio, Cash Ratio, Cash/OpEx Ratio
- **Leverage**: Debt/Equity, Debt/Assets, Interest Coverage, Debt/EBITDA  
- **Efficiency**: Asset Turnover, Inventory Turnover, Fixed Asset Turnover
- **Cash Flow**: Operating Cash Margin, Free Cash Flow Margin, Cash ROA
- **Growth**: Revenue Growth YoY, EBITDA Growth YoY (with trend analysis)

### **Virtual Fields System**
Intelligent handling of inconsistent financial reporting:

```python
# Automatically resolves company-specific reporting differences:
# - Apple: "sales_general_and_admin" 
# - Microsoft: "sales_and_marketing" + "general_and_administrative"
# - System picks best available data source automatically
```

## ⚡ **Performance & Reliability**

### **Intelligent Caching**
- ✅ **24-hour automatic cache**: Prevents unnecessary API calls
- ✅ **Freshness validation**: Auto-refresh stale data
- ✅ **Database storage**: Persistent LTM and ratio calculations

### **Rate Limiting & Security**
- ✅ **SEC API compliance**: Respects 10 requests/second limit
- ✅ **SQL injection protection**: Parameterized queries throughout
- ✅ **Input validation**: Comprehensive ticker/period validation
- ✅ **Error handling**: Graceful degradation with detailed error responses

### **LLM Integration**
- ✅ **OpenRouter integration**: Intelligent field mapping for natural language queries
- ✅ **Fallback logic**: Works without LLM for core functionality
- ✅ **Field suggestions**: Helps resolve ambiguous metric names

## 🛠 **Installation & Setup**

### **1. Database Setup**
```bash
# PostgreSQL required
createdb financial_data
```

### **2. Environment Configuration**
Create `.env` file:
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=financial_data
DB_USER=postgres
DB_PASSWORD=your_password

# SEC API Configuration  
SEC_USER_AGENT=YourName your.email@example.com

# Optional: LLM Integration
OPENROUTER_API_KEY=your_openrouter_key
```

### **3. Initialize System**
```python
from SECFinancialRAG import initialize_default_ratios

# One-time setup: Initialize ratio definitions
initialize_default_ratios()
```

## 📝 **Usage Examples**

### **Example 1: Basic Financial Analysis**
```python
from SECFinancialRAG import get_financial_data

# Get comprehensive financial dataset
df = get_financial_data('AAPL')

# DataFrame includes:
# - All LTM income statement and cash flow data
# - Point-in-time balance sheet data
# - All calculated financial ratios
# - Historical data across multiple periods
```

### **Example 2: Agent-Style Queries**
```python
# Revenue trend analysis
response = get_financial_metrics_for_agent(
    ticker='AAPL',
    metrics=['total_revenue'],
    period='last 4 quarters'
)

# Financial health assessment
response = get_ratios_for_agent(
    ticker='AAPL', 
    categories=['profitability', 'liquidity'],
    period='latest'
)

# Competitive benchmarking
response = compare_companies_for_agent(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    categories=['profitability'],
    period='LTM'
)
```

### **Example 3: Data Processing Pipeline**
```python
from SECFinancialRAG import process_company_financials

# Full processing pipeline
result = process_company_financials(
    ticker='AAPL',
    generate_ltm=True,      # Calculate LTM data
    calculate_ratios=True,  # Calculate all ratios
    validate_data=True      # Run data validation
)

# Automatic caching - subsequent calls use cached data
# unless data is >24 hours old
```

## 🔧 **Advanced Configuration**

### **Custom Ratio Definitions**
```python
from SECFinancialRAG import create_company_specific_ratio

# Add industry-specific ratios
create_company_specific_ratio(
    ticker='AAPL',
    name='R&D_Intensity',
    formula='research_and_development / total_revenue',
    description='R&D spending as % of revenue',
    category='efficiency'
)
```

### **Batch Processing**
```python
from SECFinancialRAG import get_multiple_companies_data

# Process multiple companies efficiently
df_multi = get_multiple_companies_data(['AAPL', 'MSFT', 'GOOGL'])

# Cross-company analysis
revenue_comparison = df_multi[
    (df_multi['data_source'] == 'LTM') & 
    (df_multi['statement_type'] == 'income_statement')
][['ticker', 'total_revenue', 'net_income']]
```

## 📚 **Database Schema**

### **Core Tables**
- `companies`: Company master data (CIK, ticker, name)
- `income_statements`: Income statement data (quarterly, annual)
- `balance_sheets`: Balance sheet data (point-in-time)
- `cash_flow_statements`: Cash flow data (quarterly, annual)

### **Enhanced Analytics Tables**
- `ltm_income_statements`: Last 12 months income data
- `ltm_cash_flow_statements`: Last 12 months cash flow data
- `ratio_definitions`: Ratio formulas (global + company-specific)
- `calculated_ratios`: Computed ratio values with metadata

### **Key Features**
- **Automatic LTM calculations**: Stored in database for performance
- **Hybrid ratio system**: Global + company-specific ratio definitions
- **Data lineage tracking**: Full audit trail of calculations
- **Deduplication logic**: Handles multiple filings per period

## 🎯 **Agent Interface Functions**

| **Function** | **Purpose** | **Use Case** |
|--------------|-------------|--------------|
| `get_financial_metrics_for_agent()` | Core financial data retrieval | Revenue, profit, cash flow analysis |
| `get_ratios_for_agent()` | Financial ratio analysis | Health, performance, efficiency metrics |
| `compare_companies_for_agent()` | Multi-company comparison | Competitive analysis, benchmarking |
| `get_ratio_definition_for_agent()` | Educational content | Ratio explanations, formula details |
| `get_available_ratio_categories_for_agent()` | Discovery interface | Available analysis types |

## 🚦 **Error Handling**

### **Standardized Error Codes**
- `INVALID_TICKER`: Ticker format validation failed
- `DATA_NOT_FOUND`: No financial data available  
- `INVALID_PERIOD`: Period format not supported
- `DATABASE_ERROR`: Database operation failed
- `PROCESSING_ERROR`: SEC data processing failed

### **Example Error Response**
```python
{
    "success": false,
    "error": {
        "code": "INVALID_TICKER",
        "message": "Ticker must be 1-10 characters"
    },
    "metadata": {
        "validation_errors": ["Ticker contains invalid characters"]
    }
}
```

## 🔍 **Monitoring & Debugging**

### **Logging Configuration**
```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SECFinancialRAG')
```

### **Performance Monitoring**
- ✅ **Request timing**: All agent calls logged with execution time
- ✅ **Cache hit rates**: Monitor cache effectiveness
- ✅ **Database performance**: Query execution times tracked
- ✅ **API rate limiting**: SEC API usage monitored

## 📈 **Production Readiness**

### **Security Features**
- ✅ **SQL injection protection**: 100% parameterized queries
- ✅ **Input validation**: Comprehensive sanitization
- ✅ **Environment variables**: Secure credential management
- ✅ **Rate limiting**: API usage compliance

### **Scalability Features**  
- ✅ **Database optimization**: Proper indexing strategy
- ✅ **Connection pooling**: Efficient database connections
- ✅ **Batch operations**: Bulk processing capabilities
- ✅ **Caching strategy**: Reduces API load

### **Reliability Features**
- ✅ **Comprehensive error handling**: Graceful failure modes
- ✅ **Data validation**: Multi-level validation pipeline
- ✅ **Retry logic**: Automatic SEC API retry on failures
- ✅ **Fallback mechanisms**: LLM-optional operation

## 🤝 **Contributing**

The module is production-ready and actively maintained. For enhancements or bug reports, please refer to the development guidelines in the project documentation.

## 📄 **License**

This project is part of the PitchBook Generator suite. Please refer to the main project license for usage terms.

---

**Ready for AI Agent Integration** ✅  
**Production Grade Security** ✅  
**Comprehensive Financial Coverage** ✅  
**Intelligent Caching & Performance** ✅