# Ratio Definitions Integration

## Overview

This document describes the integration of ratio definitions into the final ResearchReportOutput JSON. Ratio definitions are automatically fetched from the SECFinancialRAG database and added to the final report after all domain agent iterations are complete.

## Changes Made

### 1. Fixed Ratio Display Names (data_fetcher.py)

**File**: `orchestration/data_fetcher.py` (Lines 420-425)

**Issue**: Ratio names displayed with " Q:" suffix (e.g., "Current Ratio Q:", "Quick Ratio Q:")

**Fix**: Added logic to remove " Q" suffix from display names:

```python
# Format ratio line
ratio_display_name = ratio_name.replace('_', ' ').title()
# Remove " Q" suffix if present (from quarterly ratio names)
if ratio_display_name.endswith(' Q'):
    ratio_display_name = ratio_display_name[:-2]
ratio_line = f"{ratio_display_name}:"
```

**Result**: Now displays as "Current Ratio:", "Quick Ratio:", etc.

---

### 2. Created Shared Ratio Definitions Utility

**Files Created**:
- `shared/__init__.py`
- `shared/ratio_definitions.py`

**Purpose**: Provides utility functions to fetch ratio definitions from SECFinancialRAG database and enrich ResearchReportOutput with this information.

#### Key Functions

##### `enrich_report_with_ratio_definitions(report_output, enable_debug=False)`

Main function that enriches a ResearchReportOutput with ratio definitions.

**Usage**:
```python
from shared.ratio_definitions import enrich_report_with_ratio_definitions

# After domain agent completes analysis
enriched_report = enrich_report_with_ratio_definitions(
    report_output=analysis_output,
    enable_debug=True
)
```

**What it does**:
1. Extracts unique ratio names from `chartable_ratios`
2. Extracts companies from `chartable_ratios` and `chartable_metrics`
3. Queries SECFinancialRAG database for ratio definitions
4. Adds `ratio_definitions` field to the report

**Ratio Definition Structure**:
```python
{
    'Current_Ratio': {
        'formula': 'current_assets / current_liabilities',
        'description': 'Measures ability to pay short-term debts',
        'category': 'liquidity',
        'interpretation': 'Values above 1.0 indicate sufficient liquidity',
        'company_specific': {
            'AAPL': 'Company-specific description if different from global',
            'MSFT': 'Company-specific description if different from global'
        }
    },
    'Quick_Ratio': {
        'formula': '(current_assets - inventory) / current_liabilities',
        'description': 'More conservative liquidity measure',
        'category': 'liquidity',
        'interpretation': 'Values above 1.0 indicate good short-term liquidity',
        'company_specific': {}
    }
}
```

##### `format_ratio_definitions_for_chart(ratio_definitions)`

Formats ratio definitions into a flat list suitable for chart display.

**Usage**:
```python
from shared.ratio_definitions import format_ratio_definitions_for_chart

formatted_defs = format_ratio_definitions_for_chart(
    report.ratio_definitions
)

# Output: List of dicts with 'ratio_name', 'formula', 'description', etc.
```

---

### 3. Updated ResearchReportOutput Schema

**File**: `config/schemas.py` (Line 263)

**Change**: Added optional `ratio_definitions` field:

```python
@dataclass
class ResearchReportOutput:
    # ... existing fields ...

    ratio_definitions: Optional[Dict[str, Dict[str, Any]]] = None  # Ratio definitions for charting (added post-analysis)
```

**Notes**:
- Field is optional (defaults to None)
- Populated AFTER all domain agent iterations complete
- Does not affect existing validation logic
- Automatically included in `to_dict()` and `to_json()` serialization

---

### 4. Integrated into Master Agent

**File**: `orchestration/master_agent.py` (Lines 20, 182-193)

**Change**: Added automatic enrichment after domain analysis completes:

```python
# Import at top
from shared.ratio_definitions import enrich_report_with_ratio_definitions

# In run_single_domain_analysis():
# Run analysis
analysis_output, review_history = await domain_agent.analyze(analysis_input)

# Enrich with ratio definitions (if ResearchReportOutput)
if isinstance(analysis_output, ResearchReportOutput):
    self.logger.info("Enriching report with ratio definitions")
    try:
        analysis_output = enrich_report_with_ratio_definitions(
            analysis_output,
            enable_debug=self.enable_debug
        )
        self.logger.info("Successfully enriched report with ratio definitions")
    except Exception as e:
        self.logger.warning(f"Failed to enrich with ratio definitions: {e}")
        # Continue with un-enriched report
```

**Behavior**:
- Automatically enriches ResearchReportOutput after all iterations complete
- Graceful fallback: continues with un-enriched report if enrichment fails
- Works transparently for all domain agents
- No changes required in domain agent or analyst agent code

---

## Usage Examples

### Example 1: Access Ratio Definitions in Final Report

```python
from orchestration.master_agent import MasterAgent

master = MasterAgent(enable_debug=True)

# Run analysis
result = await master.run_single_domain_analysis(
    company="AAPL",
    domain="liquidity",
    peers=["MSFT", "GOOGL"],
    user_focus="cash management and working capital efficiency",
    time_period="3 years"
)

# Access enriched report
analysis_output = result['analysis_output']

# Ratio definitions are now available
if analysis_output.ratio_definitions:
    for ratio_name, definition in analysis_output.ratio_definitions.items():
        print(f"Ratio: {ratio_name}")
        print(f"Formula: {definition['formula']}")
        print(f"Description: {definition['description']}")
        print(f"Category: {definition['category']}")
        print()
```

### Example 2: Use Definitions for Chart Display

```python
# Get formatted definitions for charts
from shared.ratio_definitions import format_ratio_definitions_for_chart

formatted_defs = format_ratio_definitions_for_chart(
    analysis_output.ratio_definitions
)

# Display in chart UI
for def_entry in formatted_defs:
    chart_caption = f"{def_entry['ratio_name']}: {def_entry['description']}"
    chart_formula = f"Formula: {def_entry['formula']}"
    chart_interpretation = def_entry['interpretation']

    # Add to chart metadata...
```

### Example 3: Export to JSON with Definitions

```python
# Convert to JSON (includes ratio_definitions automatically)
report_json = analysis_output.to_json()

# Save to file
import json
with open('analysis_report.json', 'w') as f:
    json.dump(json.loads(report_json), f, indent=2)
```

**Sample JSON Output**:
```json
{
  "run_id": "single_liquidity_AAPL_20251011_143022",
  "domain": "liquidity",
  "company": "AAPL",
  "ticker": "AAPL",
  "executive_summary": "...",
  "chartable_ratios": [
    {
      "ratio_name": "Current Ratio",
      "company_values": {...},
      "peer_values": {...},
      "interpretation": "...",
      "trend_direction": "improving"
    }
  ],
  "ratio_definitions": {
    "Current_Ratio": {
      "formula": "current_assets / current_liabilities",
      "description": "Measures a company's ability to pay short-term debts",
      "category": "liquidity",
      "interpretation": "Values above 1.0 indicate sufficient liquidity",
      "company_specific": {}
    }
  }
}
```

---

## Technical Details

### Database Integration

The utility integrates with SECFinancialRAG's database layer:

1. **Isolation**: Uses complete environment isolation when importing SECFinancialRAG
2. **Error Handling**: Graceful degradation if database is unavailable
3. **Caching**: SECFinancialRAG's internal caching applies
4. **Company-Specific Definitions**: Automatically detects and includes company-specific ratio definitions where they differ from global definitions

### Performance Considerations

- **When**: Runs AFTER all domain agent iterations complete (no impact on iteration loop)
- **Cost**: Minimal - only database queries, no LLM calls
- **Time**: Typically < 1 second for 4-5 ratios
- **Failure Mode**: Non-blocking - analysis continues if enrichment fails

### Extensibility

The utility can be extended to support:
- Custom ratio definition sources
- Additional metadata (calculation examples, industry benchmarks)
- Definition versioning/timestamps
- Multi-language descriptions

---

## Benefits

1. **Chart Clarity**: Charts can now display formula and description alongside values
2. **User Education**: Users understand what each ratio means
3. **Transparency**: Clear documentation of how ratios are calculated
4. **Company-Specific Context**: Highlights when companies use different definitions
5. **Audit Trail**: Definitions included in exported JSON for reproducibility

---

## Testing

To test the integration:

```python
# Run a simple analysis
from orchestration.master_agent import MasterAgent
import asyncio

async def test_ratio_definitions():
    master = MasterAgent(enable_debug=True)

    result = await master.run_single_domain_analysis(
        company="AAPL",
        domain="liquidity",
        peers=["MSFT"],
        user_focus="liquidity analysis",
        time_period="2 years"
    )

    analysis = result['analysis_output']

    # Check ratio definitions present
    assert analysis.ratio_definitions is not None
    print(f"Found {len(analysis.ratio_definitions)} ratio definitions")

    # Check structure
    for ratio_name, definition in analysis.ratio_definitions.items():
        assert 'formula' in definition
        assert 'description' in definition
        assert 'category' in definition
        print(f"✓ {ratio_name}: {definition['formula']}")

# Run test
asyncio.run(test_ratio_definitions())
```

---

## Migration Notes

**No Breaking Changes**:
- Existing code continues to work without modification
- `ratio_definitions` field is optional
- Backward compatible with AnalysisOutput

**To Adopt**:
- No code changes needed - enrichment happens automatically
- Access via `report.ratio_definitions` after analysis completes
- Use `format_ratio_definitions_for_chart()` for display formatting

---

## Troubleshooting

### Issue: `ratio_definitions` is None or empty

**Possible Causes**:
1. SECFinancialRAG database unavailable
2. No ratios in `chartable_ratios`
3. Ratio names don't match database entries

**Debug**:
```python
master = MasterAgent(enable_debug=True)  # Enable debug logging
# Check logs for "Enriching report with ratio definitions"
```

### Issue: Missing company-specific definitions

**Expected Behavior**: Company-specific definitions only included if they differ from global definition

**To Verify**:
```python
for ratio_name, definition in report.ratio_definitions.items():
    if definition.get('company_specific'):
        print(f"{ratio_name} has company-specific definitions")
```

---

## Future Enhancements

Potential future additions:
1. Definition caching across multiple analyses
2. Historical definition tracking (formulas can change over time)
3. Industry-specific ratio definitions
4. Benchmark value ranges for each ratio
5. Interactive definition tooltips in UI
6. Multi-language support for descriptions

---

## Summary

This integration provides automatic enrichment of financial analysis reports with comprehensive ratio definitions, making the output more transparent, educational, and useful for charting and visualization purposes. The implementation is non-invasive, performant, and backward compatible with existing code.
