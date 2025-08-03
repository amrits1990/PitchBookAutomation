# SEC Financial RAG Package

A comprehensive financial statement RAG system that fetches real financial data from the SEC EDGAR API, stores it in PostgreSQL, and provides advanced analysis capabilities including LTM calculations and financial ratio analysis.

## Features

- **SEC Data Integration**: Direct integration with SEC EDGAR API for real financial data
- **PostgreSQL Storage**: Structured storage with separate tables for income statements, balance sheets, and cash flows
- **LTM Calculations**: Last Twelve Months calculations for income statement and cash flow metrics
- **Financial Ratio Calculator**: Advanced ratio calculation system with virtual field resolution
- **Hybrid Ratio Definitions**: Support for both global and company-specific ratio definitions
- **Virtual Fields**: Intelligent handling of inconsistent financial line items across companies
- **Data Validation**: Comprehensive validation and error handling

## Quick Start

### Basic Usage

To run the code from the project root:
```bash
python run_sec_rag.py MSFT --ltm
python SECFinancialRAG/run_sec_rag.py MSFT --ltm
```

To run directly from this folder:
```bash
python run_sec_rag.py MSFT --ltm
```

### Examples

- Process single company: `python run_sec_rag.py AAPL`
- Process multiple companies: `python run_sec_rag.py AAPL MSFT GOOGL`
- With validation and LTM: `python run_sec_rag.py AAPL --validate --ltm`
- With ratio calculation: `python run_sec_rag.py AAPL --ratios`
- Full processing: `python run_sec_rag.py AAPL --validate --ltm --ratios`
- Help: `python run_sec_rag.py --help`

### Ratio Calculator Demo

```bash
python example_ratio_usage.py
```

## Financial Ratio Calculator

The ratio calculator provides comprehensive financial analysis capabilities:

### Key Features

1. **Virtual Fields**: Automatically handles inconsistent financial reporting
   - AAPL reports SG&A as single line → uses `sales_general_and_admin`
   - MSFT reports S&M and G&A separately → uses `sales_and_marketing + general_and_administrative`
   - System automatically resolves to best available data

2. **Hybrid Ratio System**: 
   - Global ratios apply to all companies
   - Company-specific ratios override global ones
   - Easy to customize for industry-specific needs

3. **LTM Integration**:
   - Uses Last Twelve Months data for income statement ratios
   - Uses LTM data for cash flow ratios  
   - Uses point-in-time data for balance sheet ratios

4. **Default Ratios Included**:
   - **Profitability**: ROE, ROA, ROIC, Net Margin, Operating Margin, Gross Margin
   - **Liquidity**: Current Ratio, Quick Ratio, Cash Ratio
   - **Leverage**: Debt-to-Equity, Debt-to-Assets, Interest Coverage
   - **Efficiency**: Asset Turnover, Inventory Turnover, Receivables Turnover
   - **Cash Flow**: Operating Cash Margin, Free Cash Flow Margin

### Programmatic Usage

```python
from SECFinancialRAG import (
    process_company_financials,
    initialize_default_ratios,
    calculate_company_ratios,
    get_company_ratios,
    create_company_specific_ratio
)

# Initialize default ratios (one-time)
initialize_default_ratios()

# Process company with ratio calculation
result = process_company_financials('AAPL', calculate_ratios=True)

# Get calculated ratios
ratios = get_company_ratios('AAPL', category='profitability')

# Create custom company-specific ratio
create_company_specific_ratio(
    ticker='AAPL',
    name='Custom_Tech_Ratio',
    formula='research_and_development / total_revenue',
    description='R&D intensity for tech companies',
    category='efficiency'
)
```

## Virtual Fields System

Handles inconsistent financial data reporting across companies:

```python
# Virtual field 'sga_expense' automatically resolves to:
# 1. sales_general_and_admin (if available - AAPL style)
# 2. sales_and_marketing + general_and_administrative (MSFT style)  
# 3. sales_and_marketing (fallback)
# 4. general_and_administrative (fallback)

# Formula simply uses: "sga_expense / total_revenue"
# System handles the complexity behind the scenes
```

## Database Schema

### New Ratio Tables

- `ratio_definitions`: Stores ratio formulas (global and company-specific)
- `calculated_ratios`: Stores calculated ratio values with metadata

### Hybrid Support

- `company_id = NULL`: Global ratio (applies to all companies)
- `company_id = specific_uuid`: Company-specific override
- Company-specific ratios take precedence over global ones

## Configuration

Ensure your `.env` file includes database configuration:

```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=financial_data
DB_USER=postgres
DB_PASSWORD=your_password
SEC_USER_AGENT=your_name your_email
```
