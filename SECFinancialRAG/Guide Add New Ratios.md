To update exisiting ratio definition, update it in virtual_fields.py and run update_ratio_formuals.py

# Guide: Adding New Ratios to SECFinancialRAG System

This comprehensive guide provides step-by-step instructions for adding new financial ratios to the SECFinancialRAG system without encountering errors.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Ratio System Components](#ratio-system-components)
3. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
4. [Complete Code Examples](#complete-code-examples)
5. [Testing and Verification](#testing-and-verification)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Best Practices](#best-practices)

---

## System Architecture Overview

The SECFinancialRAG system implements a sophisticated 2-tier ratio architecture:

### Database Tables
- **`ratio_definitions`** - Stores ratio formulas and metadata
- **`calculated_ratios`** - Stores computed ratio values with calculation inputs
- **Financial statement tables** - Source data (income_statements, balance_sheets, cash_flow_statements)
- **LTM tables** - Last Twelve Months aggregated data

### Key Features
- **Hybrid Definitions**: Global ratios (available to all companies) + Company-specific ratios
- **Virtual Fields**: Handle inconsistent field names across companies using fallback logic
- **Multiple Data Sources**: LTM, FY (Full Year), and quarterly data support
- **Growth Ratios**: Separate calculation pipeline for Year-over-Year growth calculations
- **Agent Integration**: Direct integration with AI agent interfaces

### Data Flow Pipeline
```
Ratio Definition → Virtual Field Resolution → Financial Data Matching → 
Ratio Calculation → Storage in calculated_ratios → Agent Interface Access
```

---

## Ratio System Components

### 1. Ratio Definitions Schema
```sql
CREATE TABLE ratio_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    company_id UUID REFERENCES companies(id), -- NULL = global ratio
    formula TEXT NOT NULL,
    description TEXT,
    category VARCHAR(50),
    is_active BOOLEAN DEFAULT true,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, company_id)
);
```

### 2. Calculated Ratios Schema
```sql
CREATE TABLE calculated_ratios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID NOT NULL REFERENCES companies(id),
    ratio_definition_id UUID NOT NULL REFERENCES ratio_definitions(id),
    ticker VARCHAR(10) NOT NULL,
    ratio_name VARCHAR(100) NOT NULL,
    ratio_category VARCHAR(50),
    period_end_date DATE NOT NULL,
    period_type VARCHAR(3) NOT NULL CHECK (period_type IN ('Q1', 'Q2', 'Q3', 'Q4', 'FY', 'LTM')),
    fiscal_year INTEGER NOT NULL,
    fiscal_quarter VARCHAR(3),
    ratio_value DECIMAL(15,6),
    calculation_inputs JSONB, -- Store actual $ values used
    data_source VARCHAR(20) DEFAULT 'LTM',
    calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, ratio_definition_id, period_end_date, period_type, data_source)
);
```

### 3. Virtual Fields System
Located in `virtual_fields.py`, this system handles:
- Inconsistent field naming across companies
- Fallback logic for missing data
- Derived field calculations (EBITDA, total_debt, etc.)
- COALESCE functions for null value handling

### 4. Key Files and Their Roles

| File | Purpose |
|------|---------|
| `virtual_fields.py` | Define virtual fields and default ratios |
| `ratio_manager.py` | CRUD operations for ratio definitions |
| `simple_ratio_calculator.py` | Main ratio calculation engine |
| `growth_ratio_calculator.py` | YoY growth ratio calculations |
| `agent_interface.py` | AI agent access methods |
| `database.py` | Database operations and table creation |
| `models.py` | Pydantic data models |
| `mapping.py` | SEC field mappings and computed fields |

---

## Step-by-Step Implementation Guide

### Step 1: Determine Your Ratio Type

Choose the appropriate category for your new ratio:

**A) Standard Financial Ratio**
- Normal calculations like ROE, Current Ratio, Debt-to-Equity
- Uses existing or virtual fields
- Single period calculations

**B) Growth Ratio**
- Year-over-year growth calculations
- Requires historical data comparison
- Special handling in `growth_ratio_calculator.py`

**C) Virtual Field-Dependent Ratio**
- Requires new derived fields not directly in SEC filings
- Needs virtual field definitions first

### Step 2: Add Required Virtual Fields (If Needed)

If your ratio needs fields not directly available in SEC filings, add them to `virtual_fields.py`:

```python
# File: virtual_fields.py
# Add to VIRTUAL_FIELDS dictionary

VIRTUAL_FIELDS = {
    # Existing virtual fields...
    
    # Add your new virtual field
    "your_new_field": [
        "primary_calculation",           # Try this first
        "fallback_calculation",          # If primary fails  
        "last_resort_calculation"        # Final fallback
    ],
    
    # Example: Enterprise Value
    "enterprise_value": [
        "market_cap + total_debt - cash_and_st_investments",
        "market_cap + total_debt"  # Fallback without cash
    ],
    
    # Example: Net Debt
    "net_debt": [
        "total_debt - cash_and_st_investments",
        "total_debt - COALESCE(cash_and_cash_equivalents, 0)"
    ]
}
```

**Virtual Field Rules:**
- Use `COALESCE(field, 0)` for optional fields that might be null
- Order expressions from most specific to most general
- Ensure all referenced fields exist in database schema
- Test expressions with sample data

### Step 3A: Add Standard Ratio Definition

Add your ratio to the `DEFAULT_RATIOS` dictionary in `virtual_fields.py`:

```python
# File: virtual_fields.py
# Add to DEFAULT_RATIOS dictionary

DEFAULT_RATIOS = {
    # Existing ratios...
    
    # Add your new ratio
    "Your_Ratio_Name": {
        "formula": "numerator_field / denominator_field",
        "description": "Clear description of what this ratio measures and its interpretation",
        "category": "profitability|liquidity|leverage|efficiency|cash_flow|valuation|custom"
    },
    
    # Example: Price to Book Ratio
    "Price_to_Book": {
        "formula": "market_cap / total_stockholders_equity", 
        "description": "Market capitalization divided by book value - measures market premium over book value",
        "category": "valuation"
    },
    
    # Example: Enterprise Value to Revenue
    "EV_to_Revenue": {
        "formula": "enterprise_value / total_revenue",
        "description": "Enterprise Value to Revenue - valuation multiple comparing EV to annual revenue",
        "category": "valuation"
    },
    
    # Example: Working Capital Ratio
    "Working_Capital_Ratio": {
        "formula": "working_capital / total_revenue",
        "description": "Working capital as percentage of revenue - measures liquidity efficiency",
        "category": "liquidity"
    }
}
```

**Valid Categories:**
- `profitability` - ROE, ROA, Net Margin, Operating Margin
- `liquidity` - Current Ratio, Quick Ratio, Cash Ratio
- `leverage` - Debt ratios, Interest Coverage ratios
- `efficiency` - Asset Turnover, Inventory Turnover, Receivables Turnover
- `cash_flow` - Operating Cash Flow ratios, Free Cash Flow ratios
- `growth` - YoY growth ratios (handled separately)
- `valuation` - P/E, P/B, EV multiples
- `custom` - Your own custom categories

### Step 3B: Add Growth Ratio Definition (If Applicable)

For year-over-year growth ratios, modify `growth_ratio_calculator.py`:

```python
# File: growth_ratio_calculator.py
# Add to growth_definitions dictionary (around line 40)

growth_definitions = {
    # Existing growth ratios...
    
    # Add your new growth ratio
    "Your_Field_Growth_YoY": {
        "formula": "YOY_GROWTH(your_field_name)",
        "description": "Year-over-year growth in your field name",
        "category": "growth"
    },
    
    # Example: Free Cash Flow Growth
    "Free_Cash_Flow_Growth_YoY": {
        "formula": "YOY_GROWTH(free_cash_flow)",
        "description": "Year-over-year growth in free cash flow",
        "category": "growth"
    }
}

# Add field to data collection in _get_all_financial_periods method
def _get_all_financial_periods(self, ticker: str):
    financial_data_query = """
        SELECT 
            period_end_date, fiscal_year, fiscal_quarter,
            total_revenue, ebit, ebitda,
            your_field_name,  -- Add this line for your new field
            -- ... other existing fields
        FROM combined_financial_view 
        WHERE ticker = %s
        ORDER BY period_end_date
    """
    # ... rest of method
```

### Step 4: Initialize Ratio Definitions in Database

Create an initialization script to add your ratios to the database:

```python
#!/usr/bin/env python3
"""
Initialize new ratio definitions in the database
"""

try:
    from ratio_manager import initialize_default_ratios, RatioManager
    from models import RatioDefinition  
    from database import FinancialDatabase
except ImportError:
    from .ratio_manager import initialize_default_ratios, RatioManager
    from .models import RatioDefinition
    from .database import FinancialDatabase

def initialize_new_ratios():
    """Method 1: Reinitialize all default ratios (includes your new ones)"""
    print("Initializing new ratio definitions...")
    count = initialize_default_ratios(created_by="system")
    print(f"✅ Initialized {count} new ratio definitions")
    return count

def add_custom_ratio_manually():
    """Method 2: Add specific ratio manually using RatioManager"""
    with RatioManager() as manager:
        ratio_def = RatioDefinition(
            name="Custom_Ratio_Name",
            company_id=None,  # None = Global ratio
            formula="field1 / field2",
            description="Custom ratio description",
            category="custom",
            created_by="manual_init"
        )
        
        success = manager.create_ratio_definition(ratio_def)
        if success:
            print("✅ Custom ratio added successfully")
        else:
            print("❌ Failed to add custom ratio")
        return success

def add_company_specific_ratio():
    """Method 3: Add company-specific ratio"""
    with RatioManager() as manager:
        success = manager.create_company_specific_ratio(
            ticker="AAPL",
            name="Apple_Custom_Ratio",
            formula="custom_field1 / custom_field2", 
            description="Apple-specific custom ratio",
            category="custom",
            created_by="manual_init"
        )
        
        if success:
            print("✅ Company-specific ratio added successfully")
        else:
            print("❌ Failed to add company-specific ratio")
        return success

def verify_ratio_definitions():
    """Verify that ratios were added correctly"""
    with FinancialDatabase() as db:
        with db.connection.cursor() as cursor:
            # Check total count
            cursor.execute("SELECT COUNT(*) FROM ratio_definitions WHERE is_active = true")
            total_count = cursor.fetchone()[0]
            print(f"📊 Total active ratio definitions: {total_count}")
            
            # Check by category
            cursor.execute("""
                SELECT category, COUNT(*) as count 
                FROM ratio_definitions 
                WHERE is_active = true 
                GROUP BY category 
                ORDER BY count DESC
            """)
            
            print("\n📋 Ratios by category:")
            for category, count in cursor.fetchall():
                print(f"  {category}: {count} ratios")
            
            # Check your specific ratios
            cursor.execute("""
                SELECT name, category, formula 
                FROM ratio_definitions 
                WHERE name LIKE %s 
                ORDER BY name
            """, ("%Your%",))  # Replace with your ratio name pattern
            
            print("\n🔍 Your new ratios:")
            for name, category, formula in cursor.fetchall():
                print(f"  {name} ({category}): {formula}")

if __name__ == "__main__":
    print("=== Ratio Definition Initialization ===")
    
    # Method 1: Initialize all default ratios
    initialize_new_ratios()
    
    # Method 2: Add custom ratios (optional)
    # add_custom_ratio_manually()
    # add_company_specific_ratio()
    
    # Verify results
    verify_ratio_definitions()
    
    print("\n✅ Ratio initialization complete!")
```

### Step 5: Update Database Schema (If Needed)

If your ratio requires completely new fields not available in existing tables:

#### A) Add Computed Fields (`mapping.py`)
```python
# File: mapping.py
# Add to COMPUTED_FIELDS dictionary

COMPUTED_FIELDS = {
    # Existing fields...
    
    # Add your new computed field
    "your_new_field": {
        "table": "income_statements",  # or balance_sheets, cash_flow_statements
        "formula": "field1 + field2 - field3",
        "description": "Description of computed field calculation"
    },
    
    # Example: EBITDA margin
    "ebitda_margin": {
        "table": "income_statements",
        "formula": "ebitda / total_revenue * 100",
        "description": "EBITDA as percentage of revenue"
    }
}
```

#### B) Add to Data Models (`models.py`)
```python
# File: models.py
# Add to appropriate statement model

class IncomeStatement(BaseModel):
    # Existing fields...
    your_new_field: Optional[Decimal] = Field(None, description="Your new field description")
    
class BalanceSheet(BaseModel):
    # Existing fields...
    your_balance_field: Optional[Decimal] = Field(None, description="Your balance sheet field")
```

#### C) Update Database Schema (`database.py`)
```python
# File: database.py
# Add to _create_tables method

def _create_tables(self):
    # ... existing table creation code
    
    # Add new columns to existing tables
    with self.connection.cursor() as cursor:
        # Add to income statements
        cursor.execute("""
            ALTER TABLE income_statements 
            ADD COLUMN IF NOT EXISTS your_new_field DECIMAL(15,2)
        """)
        
        # Add to balance sheets  
        cursor.execute("""
            ALTER TABLE balance_sheets
            ADD COLUMN IF NOT EXISTS your_balance_field DECIMAL(15,2)
        """)
        
        self.connection.commit()
```

### Step 6: Test Virtual Field Resolution

Create a test script to verify your virtual fields work correctly:

```python
#!/usr/bin/env python3
"""
Test new virtual fields and ratio calculations
"""

from virtual_fields import VirtualFieldResolver
import json

def test_virtual_field_resolution():
    """Test that virtual fields resolve correctly"""
    print("=== Testing Virtual Field Resolution ===")
    
    resolver = VirtualFieldResolver()
    
    # Create test data mimicking real financial statement data
    test_data = {
        # Basic fields
        'total_revenue': 100000000.0,
        'net_income': 15000000.0,
        'total_stockholders_equity': 50000000.0,
        'total_current_assets': 30000000.0,
        'total_current_liabilities': 20000000.0,
        'cash_and_cash_equivalents': 10000000.0,
        'short_term_investments': 5000000.0,
        'total_debt': 25000000.0,
        
        # Some fields intentionally null to test fallback logic
        'sales_general_and_admin': None,
        'sales_and_marketing': 8000000.0,
        'general_and_administrative': 4000000.0
    }
    
    print("📊 Original test data:")
    for field, value in test_data.items():
        print(f"  {field}: {value}")
    
    # Resolve virtual fields
    resolved_data = resolver.resolve_virtual_fields(test_data.copy())
    
    print("\n🔧 Virtual fields resolved:")
    for field, value in resolved_data.items():
        if field not in test_data or test_data[field] != value:
            print(f"  {field}: {value}")
    
    return resolved_data

def test_ratio_calculation():
    """Test ratio calculation with resolved fields"""
    print("\n=== Testing Ratio Calculations ===")
    
    resolver = VirtualFieldResolver()
    
    # Get resolved test data
    test_data = test_virtual_field_resolution()
    
    # Test specific ratio formulas
    test_ratios = {
        "ROE": "net_income / total_stockholders_equity",
        "Current_Ratio": "total_current_assets / total_current_liabilities", 
        "Cash_Ratio": "cash_and_st_investments / total_current_liabilities",
        "Your_New_Ratio": "your_numerator / your_denominator"  # Replace with your ratio
    }
    
    print("\n📈 Ratio calculation results:")
    for ratio_name, formula in test_ratios.items():
        try:
            # Check if formula has valid components
            has_valid_components = resolver._expression_has_valid_components(formula, test_data)
            
            if has_valid_components:
                result = resolver._evaluate_expression(formula, test_data)
                print(f"  ✅ {ratio_name}: {result:.4f} (formula: {formula})")
            else:
                print(f"  ❌ {ratio_name}: Missing components (formula: {formula})")
                
        except Exception as e:
            print(f"  ❌ {ratio_name}: Error - {e}")

def test_virtual_field_validation():
    """Test validation logic for virtual field expressions"""
    print("\n=== Testing Virtual Field Validation ===")
    
    resolver = VirtualFieldResolver()
    
    test_data = {
        'field1': 1000.0,
        'field2': None,  # Null field
        'field3': 500.0
    }
    
    # Test various expression patterns
    test_expressions = [
        "field1 / field3",  # Should pass - both fields non-null
        "field1 / field2",  # Should fail - field2 is null
        "field1 / COALESCE(field2, 1)",  # Should pass - COALESCE handles null
        "COALESCE(field2, 0)",  # Should pass - pure COALESCE
        "field1 + COALESCE(field2, 0) + field3"  # Should pass - mixed
    ]
    
    print("🔍 Expression validation results:")
    for expr in test_expressions:
        is_valid = resolver._expression_has_valid_components(expr, test_data)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"  {status}: {expr}")

if __name__ == "__main__":
    try:
        test_virtual_field_resolution()
        test_ratio_calculation() 
        test_virtual_field_validation()
        print("\n✅ All virtual field tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
```

### Step 7: Calculate Ratios for Companies

Run ratio calculations to populate the `calculated_ratios` table:

```python
#!/usr/bin/env python3
"""
Calculate ratios for companies with new ratio definitions
"""

from simple_ratio_calculator import SimpleRatioCalculator
from growth_ratio_calculator import GrowthRatioCalculator
import json

def calculate_ratios_for_companies(tickers, include_growth=True):
    """Calculate all ratios for multiple companies"""
    
    print(f"=== Calculating Ratios for {len(tickers)} Companies ===")
    
    results = {}
    
    for ticker in tickers:
        print(f"\n📊 Processing {ticker}...")
        
        try:
            # Calculate standard ratios
            with SimpleRatioCalculator() as calc:
                result = calc.calculate_company_ratios(ticker)
                
                if 'error' in result:
                    print(f"❌ Error for {ticker}: {result['error']}")
                    results[ticker] = {'error': result['error']}
                    continue
                
                periods = result.get('periods', [])
                total_ratios = sum(len(p.get('ratios', {})) for p in periods)
                
                print(f"✅ {ticker}: {len(periods)} periods, {total_ratios} total ratios calculated")
                
                # Show summary of latest period
                if periods:
                    latest_period = periods[0]  # Most recent period first
                    period_date = latest_period.get('period_end_date')
                    ratios_by_category = latest_period.get('ratios', {})
                    
                    print(f"   📅 Latest period: {period_date}")
                    for category, data in ratios_by_category.items():
                        ratio_count = len(data.get('ratios', []))
                        print(f"   📈 {category}: {ratio_count} ratios")
                        
                        # Check for your new ratio
                        for ratio in data.get('ratios', []):
                            ratio_name = ratio.get('name', '')
                            if 'Your_Ratio_Name' in ratio_name:  # Replace with your ratio name
                                value = ratio.get('value')
                                print(f"      ✨ Found new ratio {ratio_name}: {value}")
                
                results[ticker] = {
                    'periods': len(periods),
                    'latest_period': period_date if periods else None,
                    'categories': list(ratios_by_category.keys()) if periods else []
                }
            
            # Calculate growth ratios if requested
            if include_growth:
                print(f"   🔄 Calculating growth ratios for {ticker}...")
                try:
                    with GrowthRatioCalculator() as growth_calc:
                        growth_result = growth_calc.calculate_growth_ratios(ticker)
                        
                        if 'error' not in growth_result:
                            growth_periods = growth_result.get('periods', [])
                            print(f"   📈 Growth ratios: {len(growth_periods)} periods")
                            results[ticker]['growth_periods'] = len(growth_periods)
                        else:
                            print(f"   ⚠️ Growth calculation warning: {growth_result['error']}")
                            
                except Exception as e:
                    print(f"   ❌ Growth calculation error: {e}")
                    
        except Exception as e:
            print(f"❌ Exception for {ticker}: {e}")
            results[ticker] = {'error': str(e)}
    
    return results

def verify_calculated_ratios(ticker, ratio_name=None):
    """Verify ratios were stored correctly in database"""
    print(f"\n=== Verifying Stored Ratios for {ticker} ===")
    
    from database import FinancialDatabase
    
    with FinancialDatabase() as db:
        with db.connection.cursor() as cursor:
            # Get overall count
            cursor.execute("""
                SELECT COUNT(*) FROM calculated_ratios 
                WHERE ticker = %s
            """, (ticker,))
            total_count = cursor.fetchone()[0]
            print(f"📊 Total calculated ratios for {ticker}: {total_count}")
            
            # Get by category
            cursor.execute("""
                SELECT ratio_category, COUNT(*) as count
                FROM calculated_ratios 
                WHERE ticker = %s 
                GROUP BY ratio_category 
                ORDER BY count DESC
            """, (ticker,))
            
            print("📋 Ratios by category:")
            for category, count in cursor.fetchall():
                print(f"  {category}: {count} ratios")
            
            # Get latest ratios
            cursor.execute("""
                SELECT ratio_name, ratio_value, period_end_date, calculation_inputs
                FROM calculated_ratios 
                WHERE ticker = %s 
                ORDER BY period_end_date DESC, ratio_name
                LIMIT 10
            """, (ticker,))
            
            print(f"\n📈 Latest ratios for {ticker}:")
            for name, value, date, inputs in cursor.fetchall():
                # Show calculation inputs if available
                inputs_summary = ""
                if inputs:
                    try:
                        inputs_dict = json.loads(inputs) if isinstance(inputs, str) else inputs
                        key_inputs = {k: v for k, v in inputs_dict.items() if v is not None}
                        if key_inputs:
                            inputs_summary = f" (inputs: {len(key_inputs)} fields)"
                    except:
                        pass
                
                print(f"  {name}: {value:.4f} ({date}){inputs_summary}")
                
                # Highlight your new ratio
                if ratio_name and ratio_name.lower() in name.lower():
                    print(f"    ✨ This is your new ratio!")

def test_agent_interface_integration():
    """Test that new ratios appear in agent interface"""
    print("\n=== Testing Agent Interface Integration ===")
    
    from agent_interface import (
        get_ratios_for_agent, 
        get_available_ratio_categories_for_agent,
        get_ratio_definition_for_agent
    )
    
    # Test available categories
    print("📋 Testing available categories...")
    categories_response = get_available_ratio_categories_for_agent()
    
    if categories_response.success:
        print("✅ Categories retrieved successfully:")
        for category, info in categories_response.data.items():
            examples = info.get('examples', [])
            print(f"  {category}: {len(examples)} example ratios")
    else:
        print(f"❌ Failed to get categories: {categories_response.error}")
    
    # Test ratio retrieval
    print("\n📊 Testing ratio retrieval...")
    test_ticker = 'AAPL'
    ratios_response = get_ratios_for_agent(test_ticker, ['profitability', 'liquidity'], 'latest')
    
    if ratios_response.success:
        print(f"✅ Ratios retrieved for {test_ticker}:")
        for category, data in ratios_response.data.items():
            ratios = data.get('ratios', {})
            print(f"  {category}: {len(ratios)} ratios")
            
            # Look for your new ratio
            for ratio_key, ratio_data in ratios.items():
                if 'Your_Ratio_Name' in ratio_key:  # Replace with your ratio name
                    value = ratio_data.get('value')
                    print(f"    ✨ Found new ratio: {ratio_key} = {value}")
    else:
        print(f"❌ Failed to get ratios: {ratios_response.error}")
    
    # Test ratio definition lookup
    print("\n📖 Testing ratio definition lookup...")
    definition_response = get_ratio_definition_for_agent('Your_Ratio_Name', test_ticker)
    
    if definition_response.success:
        definition = definition_response.data
        print(f"✅ Definition found:")
        print(f"  Name: {definition.get('name')}")
        print(f"  Formula: {definition.get('formula')}")
        print(f"  Description: {definition.get('description')}")
        print(f"  Category: {definition.get('category')}")
    else:
        print(f"❌ Definition not found: {definition_response.error}")

if __name__ == "__main__":
    # Test with a few companies first
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    print("🚀 Starting ratio calculation and verification...")
    
    # Calculate ratios
    results = calculate_ratios_for_companies(test_tickers, include_growth=True)
    
    # Verify one company in detail
    if 'AAPL' in results and 'error' not in results['AAPL']:
        verify_calculated_ratios('AAPL', 'Your_Ratio_Name')  # Replace with your ratio name
    
    # Test agent interface
    test_agent_interface_integration()
    
    print("\n✅ Ratio calculation and verification complete!")
    
    # Summary
    print(f"\n📋 Summary:")
    for ticker, result in results.items():
        if 'error' in result:
            print(f"  ❌ {ticker}: {result['error']}")
        else:
            periods = result.get('periods', 0)
            categories = len(result.get('categories', []))
            print(f"  ✅ {ticker}: {periods} periods, {categories} categories")
```

---

## Complete Code Examples

### Example 1: Adding Price-to-Earnings (P/E) Ratio

```python
# 1. Add virtual field if needed (virtual_fields.py)
VIRTUAL_FIELDS = {
    # ... existing fields
    "earnings_per_share": [
        "net_income / shares_outstanding",
        "net_income / weighted_average_shares_outstanding"  # Fallback
    ],
    "market_cap": [
        "shares_outstanding * stock_price",
        "market_value_of_equity"
    ]
}

# 2. Add ratio definition (virtual_fields.py)
DEFAULT_RATIOS = {
    # ... existing ratios
    "Price_to_Earnings": {
        "formula": "stock_price / earnings_per_share",
        "description": "Price-to-Earnings Ratio - Market price per share divided by earnings per share",
        "category": "valuation"
    },
    "PE_Ratio_Alt": {
        "formula": "market_cap / net_income", 
        "description": "P/E Ratio (Alternative) - Market capitalization divided by net income",
        "category": "valuation"
    }
}

# 3. Initialize and calculate
from ratio_manager import initialize_default_ratios
from simple_ratio_calculator import SimpleRatioCalculator

# Initialize definitions
initialize_default_ratios()

# Calculate for companies
with SimpleRatioCalculator() as calc:
    result = calc.calculate_company_ratios('AAPL')
    print(f"Calculated ratios: {len(result.get('periods', []))} periods")

# 4. Test via agent interface
from agent_interface import get_ratios_for_agent
response = get_ratios_for_agent('AAPL', ['valuation'], 'latest')
if response.success:
    valuation_ratios = response.data.get('valuation', {}).get('ratios', {})
    for ratio_name, ratio_data in valuation_ratios.items():
        if 'Price_to_Earnings' in ratio_name:
            print(f"P/E Ratio: {ratio_data.get('value')}")
```

### Example 2: Adding Free Cash Flow Yield

```python
# 1. Add virtual field (virtual_fields.py)
VIRTUAL_FIELDS = {
    # ... existing fields
    "free_cash_flow_yield": [
        "free_cash_flow / market_cap",
        "free_cash_flow / enterprise_value"  # Alternative calculation
    ]
}

# 2. Add ratio definition (virtual_fields.py)
DEFAULT_RATIOS = {
    # ... existing ratios
    "Free_Cash_Flow_Yield": {
        "formula": "free_cash_flow / market_cap",
        "description": "Free Cash Flow Yield - Annual free cash flow as percentage of market capitalization",
        "category": "cash_flow"
    },
    "FCF_per_Share": {
        "formula": "free_cash_flow / shares_outstanding",
        "description": "Free Cash Flow per Share - Free cash flow divided by shares outstanding", 
        "category": "cash_flow"
    }
}
```

### Example 3: Adding Custom Growth Ratio

```python
# 1. Add to growth_ratio_calculator.py
growth_definitions = {
    # ... existing growth ratios
    "Free_Cash_Flow_Growth_YoY": {
        "formula": "YOY_GROWTH(free_cash_flow)",
        "description": "Year-over-year growth in free cash flow", 
        "category": "growth"
    },
    "Operating_Cash_Flow_Growth_YoY": {
        "formula": "YOY_GROWTH(net_cash_from_operating_activities)",
        "description": "Year-over-year growth in operating cash flow",
        "category": "growth"
    }
}

# 2. Update data collection method
def _get_all_financial_periods(self, ticker: str):
    financial_data_query = """
        SELECT 
            period_end_date, fiscal_year, fiscal_quarter,
            total_revenue, ebit, ebitda,
            free_cash_flow,  -- Add this
            net_cash_from_operating_activities,  -- Add this
            -- ... other fields
        FROM ltm_cash_flow_statements l
        WHERE l.ticker = %s
        ORDER BY period_end_date
    """
```

### Example 4: Adding Company-Specific Ratio

```python
#!/usr/bin/env python3
"""Add company-specific ratio for Apple"""

from ratio_manager import RatioManager
from models import RatioDefinition

def add_apple_specific_ratio():
    """Add custom ratio specific to Apple"""
    
    with RatioManager() as manager:
        # Get Apple's company ID
        company_info = manager.database.get_company_by_ticker('AAPL')
        if not company_info:
            print("❌ Apple not found in database")
            return False
        
        company_id = company_info['id']
        
        # Create Apple-specific ratio
        ratio_def = RatioDefinition(
            name="Services_Revenue_Ratio",
            company_id=company_id,
            formula="services_revenue / total_revenue",
            description="Apple Services revenue as percentage of total revenue",
            category="custom",
            created_by="apple_analysis"
        )
        
        success = manager.create_ratio_definition(ratio_def)
        if success:
            print("✅ Apple-specific ratio created successfully")
        else:
            print("❌ Failed to create Apple-specific ratio")
        
        return success

if __name__ == "__main__":
    add_apple_specific_ratio()
```

---

## Testing and Verification

### 1. Unit Tests for Virtual Fields

```python
#!/usr/bin/env python3
"""Unit tests for virtual field resolution"""

import unittest
from virtual_fields import VirtualFieldResolver

class TestVirtualFields(unittest.TestCase):
    
    def setUp(self):
        self.resolver = VirtualFieldResolver()
        self.test_data = {
            'total_revenue': 100000000.0,
            'net_income': 15000000.0,
            'total_stockholders_equity': 50000000.0,
            'cash_and_cash_equivalents': 10000000.0,
            'short_term_investments': 5000000.0,
            'total_debt': 25000000.0
        }
    
    def test_virtual_field_resolution(self):
        """Test that virtual fields resolve correctly"""
        resolved = self.resolver.resolve_virtual_fields(self.test_data.copy())
        
        # Check that cash_and_st_investments was calculated
        expected_cash = 10000000.0 + 5000000.0  # cash + short_term_investments
        self.assertEqual(resolved.get('cash_and_st_investments'), expected_cash)
    
    def test_ratio_calculation(self):
        """Test ratio calculation with virtual fields"""
        resolved = self.resolver.resolve_virtual_fields(self.test_data.copy())
        
        # Test ROE calculation
        roe_result = self.resolver._evaluate_expression(
            "net_income / total_stockholders_equity", 
            resolved
        )
        expected_roe = 15000000.0 / 50000000.0
        self.assertAlmostEqual(roe_result, expected_roe, places=6)
    
    def test_null_handling(self):
        """Test handling of null values"""
        test_data_with_nulls = self.test_data.copy()
        test_data_with_nulls['short_term_investments'] = None
        
        resolved = self.resolver.resolve_virtual_fields(test_data_with_nulls)
        
        # Should still calculate cash_and_st_investments using COALESCE
        self.assertIsNotNone(resolved.get('cash_and_st_investments'))
    
    def test_expression_validation(self):
        """Test expression component validation"""
        # Should pass - all fields available
        self.assertTrue(
            self.resolver._expression_has_valid_components(
                "net_income / total_stockholders_equity", 
                self.test_data
            )
        )
        
        # Should fail - missing field
        self.assertFalse(
            self.resolver._expression_has_valid_components(
                "missing_field / total_stockholders_equity",
                self.test_data
            )
        )

if __name__ == '__main__':
    unittest.main()
```

### 2. Integration Tests

```python
#!/usr/bin/env python3
"""Integration tests for ratio calculation system"""

import unittest
from simple_ratio_calculator import SimpleRatioCalculator
from agent_interface import get_ratios_for_agent
import json

class TestRatioIntegration(unittest.TestCase):
    
    def setUp(self):
        self.test_ticker = 'AAPL'
    
    def test_ratio_calculation_pipeline(self):
        """Test complete ratio calculation pipeline"""
        with SimpleRatioCalculator() as calc:
            result = calc.calculate_company_ratios(self.test_ticker)
            
            # Should not have errors
            self.assertNotIn('error', result)
            
            # Should have periods
            periods = result.get('periods', [])
            self.assertGreater(len(periods), 0)
            
            # Each period should have ratios
            for period in periods:
                ratios_by_category = period.get('ratios', {})
                self.assertGreater(len(ratios_by_category), 0)
                
                # Check ratio structure
                for category, data in ratios_by_category.items():
                    self.assertIn('ratios', data)
                    self.assertIn('category_name', data)
                    
                    for ratio in data['ratios']:
                        self.assertIn('name', ratio)
                        self.assertIn('value', ratio)
                        self.assertIn('calculation_inputs', ratio)
    
    def test_agent_interface_access(self):
        """Test agent interface returns calculated ratios"""
        response = get_ratios_for_agent(self.test_ticker, ['profitability'], 'latest')
        
        self.assertTrue(response.success)
        self.assertIn('profitability', response.data)
        
        profitability_data = response.data['profitability']
        self.assertIn('ratios', profitability_data)
        self.assertGreater(len(profitability_data['ratios']), 0)
    
    def test_new_ratio_presence(self):
        """Test that new ratios appear in results"""
        # Replace 'Your_Ratio_Name' with your actual ratio name
        target_ratio = 'ROE'  # Use existing ratio for test
        
        response = get_ratios_for_agent(self.test_ticker, ['profitability'], 'latest')
        
        if response.success:
            ratios = response.data.get('profitability', {}).get('ratios', {})
            
            # Check if target ratio exists
            ratio_found = any(target_ratio in ratio_key for ratio_key in ratios.keys())
            self.assertTrue(ratio_found, f"Ratio {target_ratio} not found in results")

if __name__ == '__main__':
    unittest.main()
```

### 3. Database Verification Queries

```sql
-- Check ratio definitions
SELECT 
    name, 
    category, 
    formula, 
    is_active,
    CASE WHEN company_id IS NULL THEN 'Global' ELSE 'Company-Specific' END as scope
FROM ratio_definitions 
WHERE is_active = true
ORDER BY category, name;

-- Check calculated ratios
SELECT 
    r.ticker,
    r.ratio_name,
    r.ratio_category,
    r.period_end_date,
    r.ratio_value,
    r.calculation_inputs
FROM calculated_ratios r
WHERE r.ticker = 'AAPL'
    AND r.ratio_name LIKE '%Your_Ratio%'  -- Replace with your ratio
ORDER BY r.period_end_date DESC;

-- Check ratio coverage by company
SELECT 
    ticker,
    COUNT(DISTINCT ratio_name) as unique_ratios,
    COUNT(*) as total_calculations,
    MAX(period_end_date) as latest_period
FROM calculated_ratios
GROUP BY ticker
ORDER BY unique_ratios DESC;

-- Check calculation inputs
SELECT 
    ratio_name,
    calculation_inputs,
    ratio_value
FROM calculated_ratios 
WHERE ticker = 'AAPL'
    AND calculation_inputs IS NOT NULL
    AND period_end_date = (
        SELECT MAX(period_end_date) 
        FROM calculated_ratios 
        WHERE ticker = 'AAPL'
    )
ORDER BY ratio_name;
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. "Field not found" Errors

**Problem**: Ratio calculation fails with field not found errors.

**Solutions**:
```python
# Check available fields in database
from database import FinancialDatabase

with FinancialDatabase() as db:
    with db.connection.cursor() as cursor:
        # Check income statement fields
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'income_statements'
        """)
        print("Income statement fields:", [row[0] for row in cursor.fetchall()])
        
        # Check balance sheet fields  
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'balance_sheets'
        """)
        print("Balance sheet fields:", [row[0] for row in cursor.fetchall()])
```

**Prevention**:
- Use exact field names from database schema
- Check `mapping.py` for available mapped fields
- Use virtual fields for derived calculations
- Verify field spelling and case sensitivity

#### 2. "Ratio not calculated" Issues

**Problem**: Ratio definition exists but no calculated values appear.

**Debug Steps**:
```python
# Check ratio definition exists
from database import FinancialDatabase

with FinancialDatabase() as db:
    with db.connection.cursor() as cursor:
        cursor.execute("""
            SELECT * FROM ratio_definitions 
            WHERE name = %s AND is_active = true
        """, ("Your_Ratio_Name",))
        
        result = cursor.fetchone()
        if result:
            print("✅ Ratio definition found")
        else:
            print("❌ Ratio definition missing or inactive")

# Check virtual field resolution
from virtual_fields import VirtualFieldResolver

resolver = VirtualFieldResolver()
test_data = {...}  # Your test financial data
resolved = resolver.resolve_virtual_fields(test_data)

# Test ratio formula
formula = "your_numerator / your_denominator"
has_components = resolver._expression_has_valid_components(formula, resolved)
print(f"Formula validation: {has_components}")

if has_components:
    result = resolver._evaluate_expression(formula, resolved)
    print(f"Calculation result: {result}")
```

**Common Causes**:
- Virtual field dependencies not resolved
- Null or zero denominators  
- Missing required financial data
- Formula syntax errors

#### 3. "Division by zero" Errors

**Problem**: Ratios fail when denominator is zero or null.

**Solutions**:
```python
# Add safeguards to formula
"safe_formula": """
    CASE 
        WHEN COALESCE(denominator, 0) = 0 THEN NULL 
        ELSE numerator / denominator 
    END
"""

# Or use COALESCE with small non-zero value
"safe_formula": "numerator / COALESCE(NULLIF(denominator, 0), 0.001)"
```

#### 4. "Agent interface missing ratio" Problems

**Problem**: Ratios calculated but don't appear in agent interface.

**Debug Steps**:
```python
# Check calculated_ratios table
from database import FinancialDatabase

with FinancialDatabase() as db:
    with db.connection.cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*) FROM calculated_ratios 
            WHERE ratio_name = %s AND ticker = %s
        """, ("Your_Ratio_Name", "AAPL"))
        
        count = cursor.fetchone()[0]
        print(f"Calculated ratios found: {count}")

# Check category filter
from agent_interface import get_ratios_for_agent

response = get_ratios_for_agent('AAPL', ['your_category'], 'latest')
if not response.success:
    print(f"Agent interface error: {response.error}")
```

**Common Causes**:
- Incorrect category name
- Category not included in filter
- Period filtering excluding ratios
- Database connection issues

#### 5. Virtual Field Resolution Failures

**Problem**: Virtual fields not resolving correctly.

**Debug Script**:
```python
#!/usr/bin/env python3
"""Debug virtual field resolution issues"""

from virtual_fields import VirtualFieldResolver
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

resolver = VirtualFieldResolver()

# Test specific virtual field
test_data = {
    # Add your test data here
}

print("Original data:")
for k, v in test_data.items():
    print(f"  {k}: {v}")

# Test step by step
print("\nVirtual field resolution:")
resolved = resolver.resolve_virtual_fields(test_data.copy())

print("\nResolved data:")
for k, v in resolved.items():
    if k not in test_data or test_data[k] != v:
        print(f"  {k}: {v} (NEW)")

# Test specific expression
expression = "your_virtual_field_expression"
print(f"\nTesting expression: {expression}")

has_components = resolver._expression_has_valid_components(expression, resolved)
print(f"Has valid components: {has_components}")

if has_components:
    try:
        result = resolver._evaluate_expression(expression, resolved)
        print(f"Expression result: {result}")
    except Exception as e:
        print(f"Expression error: {e}")
```

### Performance Issues

#### 1. Slow Ratio Calculations

**Optimization Strategies**:
- Batch process multiple companies
- Use database indexes on period_end_date, ticker
- Cache virtual field resolutions
- Limit calculation to recent periods only

```python
# Optimized batch calculation
def calculate_ratios_batch(tickers, max_periods=8):
    """Calculate ratios for multiple tickers with period limit"""
    
    for ticker in tickers:
        with SimpleRatioCalculator() as calc:
            # Modify calculator to limit periods
            result = calc.calculate_company_ratios(ticker)
            
            if 'periods' in result:
                # Keep only recent periods
                result['periods'] = result['periods'][:max_periods]
```

#### 2. Memory Usage with Large Datasets

**Solutions**:
- Process companies one at a time
- Use database cursors for large result sets
- Implement pagination for ratio queries
- Clear calculation caches regularly

### Data Quality Issues

#### 1. Inconsistent Financial Data

**Detection**:
```python
#!/usr/bin/env python3
"""Detect data quality issues"""

from database import FinancialDatabase

def check_data_quality(ticker):
    """Check for common data quality issues"""
    
    with FinancialDatabase() as db:
        with db.connection.cursor() as cursor:
            # Check for missing critical fields
            cursor.execute("""
                SELECT period_end_date,
                       total_revenue IS NULL as missing_revenue,
                       net_income IS NULL as missing_net_income,
                       total_stockholders_equity IS NULL as missing_equity
                FROM income_statements 
                WHERE ticker = %s
                ORDER BY period_end_date DESC
                LIMIT 10
            """, (ticker,))
            
            print(f"Data quality check for {ticker}:")
            for row in cursor.fetchall():
                period, missing_rev, missing_ni, missing_eq = row
                issues = []
                if missing_rev: issues.append("revenue")
                if missing_ni: issues.append("net_income") 
                if missing_eq: issues.append("equity")
                
                if issues:
                    print(f"  {period}: Missing {', '.join(issues)}")
                else:
                    print(f"  {period}: ✅ Complete")
```

#### 2. Ratio Value Validation

**Validation Rules**:
```python
def validate_ratio_values(ticker, ratio_name):
    """Validate calculated ratio values for reasonableness"""
    
    from database import FinancialDatabase
    
    with FinancialDatabase() as db:
        with db.connection.cursor() as cursor:
            cursor.execute("""
                SELECT ratio_value, period_end_date
                FROM calculated_ratios
                WHERE ticker = %s AND ratio_name = %s
                ORDER BY period_end_date DESC
                LIMIT 20
            """, (ticker, ratio_name))
            
            values = [row[0] for row in cursor.fetchall() if row[0] is not None]
            
            if values:
                avg_value = sum(values) / len(values)
                max_value = max(values)
                min_value = min(values)
                
                print(f"Ratio validation for {ticker} {ratio_name}:")
                print(f"  Average: {avg_value:.4f}")
                print(f"  Range: {min_value:.4f} to {max_value:.4f}")
                
                # Check for outliers (values > 3 standard deviations)
                import statistics
                if len(values) > 3:
                    stdev = statistics.stdev(values)
                    outliers = [v for v in values if abs(v - avg_value) > 3 * stdev]
                    if outliers:
                        print(f"  ⚠️ Potential outliers: {outliers}")
                    else:
                        print(f"  ✅ No outliers detected")
```

---

## Best Practices

### 1. Ratio Design Principles

#### **Use Meaningful Names**
```python
# ✅ Good - Clear and descriptive
"Return_on_Equity": {
    "formula": "net_income / total_stockholders_equity",
    "description": "Return on Equity - Net income as percentage of shareholders' equity"
}

# ❌ Bad - Unclear abbreviation
"ROE_v2": {
    "formula": "net_income / total_stockholders_equity", 
    "description": "ROE"
}
```

#### **Provide Comprehensive Descriptions**
```python
"EBITDA_Margin": {
    "formula": "ebitda / total_revenue",
    "description": "EBITDA Margin - Earnings before interest, taxes, depreciation and amortization as percentage of revenue. Measures operational profitability excluding non-cash and financing items.",
    "category": "profitability"
}
```

#### **Use Appropriate Categories**
- Group related ratios together
- Use standard category names
- Consider creating subcategories for complex analyses

### 2. Virtual Field Best Practices

#### **Order Expressions by Preference**
```python
"total_debt": [
    "long_term_debt + short_term_debt",  # Most accurate
    "total_long_term_debt",              # Good fallback
    "COALESCE(debt_total, 0)"           # Last resort
]
```

#### **Handle Missing Data Gracefully** 
```python
"working_capital": [
    "total_current_assets - total_current_liabilities",
    "COALESCE(current_assets, 0) - COALESCE(current_liabilities, 0)"
]
```

#### **Use COALESCE for Optional Fields**
```python
"total_operating_expenses": [
    "cost_of_revenue + COALESCE(sga_expense, 0) + COALESCE(research_and_development, 0)",
    "total_revenue - operating_income"
]
```

### 3. Formula Design Guidelines

#### **Avoid Division by Zero**
```python
# ✅ Safe formula with protection
"asset_turnover": "CASE WHEN total_assets > 0 THEN total_revenue / total_assets ELSE NULL END"

# ✅ Using COALESCE with non-zero default
"debt_to_equity": "total_debt / COALESCE(NULLIF(total_stockholders_equity, 0), 1)"
```

#### **Use Consistent Units**
```python
# ✅ Both in dollars
"revenue_per_employee": "total_revenue / employee_count"

# ✅ Both as percentages  
"margin_comparison": "net_margin - industry_avg_margin"
```

#### **Consider Ratio Directionality**
```python
# Document whether higher is better or worse
"Current_Ratio": {
    "formula": "total_current_assets / total_current_liabilities",
    "description": "Current Ratio - Higher values indicate better short-term liquidity (optimal range: 1.5-3.0)",
    "interpretation": "higher_is_better"
}
```

### 4. Testing Strategy

#### **Test with Real Data**
```python
def test_ratio_with_real_data():
    """Test ratios using actual company data"""
    test_companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    for ticker in test_companies:
        # Test calculation
        with SimpleRatioCalculator() as calc:
            result = calc.calculate_company_ratios(ticker)
            
            # Validate results
            assert 'error' not in result
            assert len(result.get('periods', [])) > 0
            
            # Check ratio reasonableness
            for period in result['periods']:
                for category, data in period['ratios'].items():
                    for ratio in data['ratios']:
                        value = ratio.get('value')
                        assert value is not None
                        assert not (value == float('inf') or value != value)  # Check for inf/NaN
```

#### **Test Edge Cases**
```python
def test_edge_cases():
    """Test ratios with edge case data"""
    
    edge_cases = [
        {'name': 'zero_revenue', 'data': {'total_revenue': 0, 'net_income': 1000}},
        {'name': 'negative_equity', 'data': {'total_stockholders_equity': -1000, 'net_income': 500}},
        {'name': 'null_values', 'data': {'total_revenue': None, 'net_income': None}}
    ]
    
    resolver = VirtualFieldResolver()
    
    for case in edge_cases:
        print(f"Testing {case['name']}...")
        try:
            resolved = resolver.resolve_virtual_fields(case['data'])
            # Test that resolution doesn't crash
            print(f"  ✅ {case['name']} handled gracefully")
        except Exception as e:
            print(f"  ❌ {case['name']} failed: {e}")
```

### 5. Documentation Standards

#### **Document Formula Logic**
```python
"Return_on_Invested_Capital": {
    "formula": "(ebit * (1 - 0.26)) / invested_capital",
    "description": "Return on Invested Capital - NOPAT divided by invested capital. Uses 26% corporate tax rate for NOPAT calculation. Measures efficiency of capital allocation.",
    "assumptions": ["26% corporate tax rate", "EBIT represents operating performance"],
    "category": "profitability"
}
```

#### **Include Interpretation Guidelines**
```python
"Quick_Ratio": {
    "formula": "(cash_and_st_investments + accounts_receivable) / total_current_liabilities",
    "description": "Quick Ratio - Most liquid assets divided by current liabilities",
    "interpretation": {
        "excellent": "> 1.5",
        "good": "1.0 - 1.5", 
        "concerning": "< 1.0",
        "notes": "Excludes inventory and prepaid expenses from current assets"
    },
    "category": "liquidity"
}
```

### 6. Maintenance and Updates

#### **Version Control for Ratios**
```python
# Track ratio definition changes
"Modified_Ratio_v2": {
    "formula": "updated_formula",
    "description": "Updated ratio description - v2.0",
    "version": "2.0",
    "changelog": "Updated formula to handle new accounting standards",
    "deprecated_versions": ["Modified_Ratio", "Modified_Ratio_v1"]
}
```

#### **Regular Validation**
```python
#!/usr/bin/env python3
"""Regular validation script for ratio quality"""

def monthly_ratio_validation():
    """Run monthly validation of ratio calculations"""
    
    validation_checks = [
        check_ratio_coverage,      # All companies have recent ratios
        check_ratio_reasonableness, # Values are within expected ranges
        check_calculation_inputs,   # Input values are properly stored
        check_category_distribution # Categories are balanced
    ]
    
    for check in validation_checks:
        try:
            check()
            print(f"✅ {check.__name__} passed")
        except Exception as e:
            print(f"❌ {check.__name__} failed: {e}")

if __name__ == "__main__":
    monthly_ratio_validation()
```

### 7. Performance Optimization

#### **Use Efficient Queries**
```sql
-- ✅ Efficient - uses indexes
SELECT ratio_name, ratio_value, period_end_date
FROM calculated_ratios 
WHERE ticker = 'AAPL' 
    AND period_end_date >= '2023-01-01'
ORDER BY period_end_date DESC;

-- ❌ Inefficient - no index usage
SELECT * FROM calculated_ratios 
WHERE UPPER(ticker) = 'AAPL'
    AND EXTRACT(YEAR FROM period_end_date) = 2023;
```

#### **Batch Operations**
```python
def calculate_ratios_efficient(tickers):
    """Efficient batch ratio calculation"""
    
    # Group tickers by data availability
    with FinancialDatabase() as db:
        available_tickers = []
        for ticker in tickers:
            if db.get_company_by_ticker(ticker):
                available_tickers.append(ticker)
    
    # Process in batches
    batch_size = 10
    for i in range(0, len(available_tickers), batch_size):
        batch = available_tickers[i:i + batch_size]
        
        for ticker in batch:
            try:
                with SimpleRatioCalculator() as calc:
                    calc.calculate_company_ratios(ticker)
            except Exception as e:
                print(f"Failed for {ticker}: {e}")
                continue
```

---

## Summary Checklist

When adding new ratios to the SECFinancialRAG system, follow this checklist to ensure success:

### ✅ Pre-Implementation
- [ ] Identify ratio type (standard, growth, or virtual field-dependent)
- [ ] Verify required fields exist in database schema
- [ ] Check if virtual fields need to be created
- [ ] Choose appropriate category and naming convention

### ✅ Implementation  
- [ ] Add virtual fields to `virtual_fields.py` (if needed)
- [ ] Add ratio definition to `DEFAULT_RATIOS` in `virtual_fields.py`
- [ ] Update `growth_ratio_calculator.py` (if growth ratio)
- [ ] Add computed fields to `mapping.py` (if needed)
- [ ] Update data models in `models.py` (if new fields)

### ✅ Database Setup
- [ ] Run initialization script to create ratio definitions
- [ ] Verify ratio definitions exist in `ratio_definitions` table
- [ ] Update database schema (if new fields required)

### ✅ Testing
- [ ] Test virtual field resolution with sample data
- [ ] Run ratio calculations for test companies
- [ ] Verify results in `calculated_ratios` table
- [ ] Test agent interface integration
- [ ] Validate ratio values for reasonableness

### ✅ Documentation
- [ ] Document ratio formulas and assumptions
- [ ] Add interpretation guidelines
- [ ] Update system documentation
- [ ] Create troubleshooting guides for new ratios

### ✅ Production Deployment
- [ ] Calculate ratios for all companies
- [ ] Monitor for errors and data quality issues
- [ ] Set up regular validation checks
- [ ] Plan for ongoing maintenance and updates

By following this comprehensive guide, you can successfully add new ratios to the SECFinancialRAG system without encountering errors and ensure they integrate seamlessly with the existing architecture.