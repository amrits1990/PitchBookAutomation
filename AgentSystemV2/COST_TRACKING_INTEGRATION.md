# LLM Cost Tracking Integration Guide

## Overview

This guide explains how to integrate the `llm_cost_tracker.py` module into the agent system to track costs across all LLM calls (analyst, reviewer, master agents).

## Changes Made

### 1. Created `config/llm_cost_tracker.py` ✅

Complete cost tracking system with:
- LLM pricing for 30+ models
- `TokenUsage` dataclass for individual calls
- `AgentCost` dataclass for per-agent tracking
- `RunCost` dataclass for complete run tracking
- `CostTracker` class for global cost management
- Helper functions to extract token usage from Agno responses

### 2. Updated `agents/base_agent.py` ✅

Changed `_run_agent()` method to:
```python
# OLD return type:
async def _run_agent(...) -> tuple[str, List[Dict[str, Any]]]:

# NEW return type:
async def _run_agent(...) -> tuple[str, List[Dict[str, Any]], TokenUsage]:
    # Extract token usage from Agno response
    token_usage = extract_token_usage_from_agno(response, self.model_id)
    return response_text, tool_calls_made, token_usage
```

## Remaining Integration Steps

### Step 1: Update analyst_agent.py

**File**: `agents/analyst_agent.py`
**Line**: 377

**Change**:
```python
# OLD (line 377):
response, tool_calls_captured = await self._run_agent(analysis_prompt, context)

# NEW:
response, tool_calls_captured, token_usage = await self._run_agent(analysis_prompt, context)
```

**Also add** to imports at top:
```python
from config.llm_cost_tracker import TokenUsage, AgentCost
```

**Modify** `analyze()` method to return TokenUsage:
```python
async def analyze(...) -> tuple[AnalysisOutput, TokenUsage]:
    ...
    response, tool_calls_captured, token_usage = await self._run_agent(analysis_prompt, context)
    ...
    return analysis_output, token_usage  # Return both
```

### Step 2: Update reviewer_agent.py

**File**: `agents/reviewer_agent.py`
**Line**: Search for `_run_agent` call (around line 174)

**Change**:
```python
# OLD:
response, _ = await self._run_agent(evaluation_prompt, context)

# NEW:
response, _, token_usage = await self._run_agent(evaluation_prompt, context)
```

**Modify** `evaluate()` method to return TokenUsage:
```python
async def evaluate(...) -> tuple[ReviewResult, TokenUsage]:
    ...
    response, _, token_usage = await self._run_agent(evaluation_prompt, context)
    ...
    return review_result, token_usage  # Return both
```

### Step 3: Update domain_agent.py

**File**: `agents/domain_agent.py`

**Add** to imports:
```python
from config.llm_cost_tracker import get_global_tracker, AgentCost, TokenUsage
```

**Modify** `analyze()` method:
```python
async def analyze(self, analysis_input: AnalysisInput, max_loops: int = None) -> tuple[AnalysisOutput, List[ReviewResult], RunCost]:
    # Initialize cost tracking
    tracker = get_global_tracker()
    run_cost = tracker.start_run(analysis_input.run_id, analysis_input.company, analysis_input.domain)

    # Inside the loop:
    while self.current_loops < self.max_loops:
        self.current_loops += 1

        # Step 1: Analyst Analysis
        current_analysis, analyst_token_usage = await self.analyst.analyze(...)

        # Track analyst cost
        analyst_cost = AgentCost(
            agent_name=f"{self.domain}_analyst",
            agent_type="analyst",
            loop_number=self.current_loops
        )
        analyst_cost.add_usage(analyst_token_usage)
        run_cost.add_analyst_cost(analyst_cost)

        # Step 2: Reviewer Evaluation
        review_result, reviewer_token_usage = await self.reviewer.evaluate(...)

        # Track reviewer cost
        reviewer_cost = AgentCost(
            agent_name=f"{self.domain}_reviewer",
            agent_type="reviewer",
            loop_number=self.current_loops
        )
        reviewer_cost.add_usage(reviewer_token_usage)
        run_cost.add_reviewer_cost(reviewer_cost)

        # Continue loop logic...

    # Finalize run
    run_cost.finalize()

    return current_analysis, review_history, run_cost
```

### Step 4: Update master_agent.py

**File**: `orchestration/master_agent.py`

**Add** to imports:
```python
from config.llm_cost_tracker import get_global_tracker, extract_token_usage_from_agno, AgentCost, format_cost_summary
```

**Modify** `run_single_domain_analysis()`:
```python
async def run_single_domain_analysis(self, ...):
    # Around line 179:
    analysis_output, review_history, run_cost = await domain_agent.analyze(analysis_input)

    # Track master agent costs (from coordination calls)
    tracker = get_global_tracker()
    master_run_cost = tracker.get_run(run_id)

    # (Master agent costs from line 131 and 205 coordination calls)
    # Extract token usage from master agent's coordination responses
    # Add to run_cost.add_master_cost(...)

    # Add cost info to execution result
    execution_result = {
        ...
        "run_cost": run_cost.get_breakdown(),
        "cost_breakdown": format_cost_summary(run_cost)
    }

    # Display cost summary
    self.logger.info("\n" + format_cost_summary(run_cost))

    return execution_result
```

### Step 5: Update comprehensive_demo.py

**File**: `examples/comprehensive_demo.py`

**Add** cost display after analysis:
```python
# After line 72 in demo_research_grade_liquidity_analysis():
if result["success"]:
    analysis = result["analysis_output"]

    # Display cost breakdown
    if "cost_breakdown" in result:
        print("\n" + result["cost_breakdown"])
```

## Token Extraction from Agno

The `extract_token_usage_from_agno()` function tries to extract tokens from:

1. `response.metrics` dict
2. `response.usage` dict
3. Parsing from `response.content` string

**If token extraction isn't working**, check Agno's response structure:
```python
# Add debug logging in base_agent.py _run_agent():
self.logger.debug(f"Agno response type: {type(response)}")
self.logger.debug(f"Agno response attributes: {dir(response)}")
if hasattr(response, 'metrics'):
    self.logger.debug(f"Metrics: {response.metrics}")
if hasattr(response, 'usage'):
    self.logger.debug(f"Usage: {response.usage}")
```

Then update `extract_token_usage_from_agno()` in `llm_cost_tracker.py` to match Agno's actual structure.

## Testing Cost Tracking

```python
from config.llm_cost_tracker import get_global_tracker

# Run analysis
result = await system.analyze_domain(...)

# Get cost summary
tracker = get_global_tracker()
summary = tracker.get_summary()
print(f"Total cost: {summary['total_cost']}")
print(f"Total runs: {summary['total_runs']}")
```

## Expected Output

After integration, you should see logs like:
```
INFO - LLM Usage: 2500 input + 1200 output = 3700 tokens ($0.0042)
INFO - Loop 1: Analysis completed
INFO - LLM Usage: 1800 input + 800 output = 2600 tokens ($0.0027)
INFO - Loop 1: Review completed

════════════════════════════════════════════════════════════════════
💰 LLM COST BREAKDOWN
════════════════════════════════════════════════════════════════════
Run ID: single_liquidity_AAPL_20251012_001234
Company: AAPL | Domain: liquidity
Duration: 2.3m

Total Cost: $0.0248
Total Tokens: 26,400 (Input: 14,200, Output: 12,200)

By Agent:
  📊 Analyst: $0.0156 (3 loops, 18,600 tokens)
  ✅ Reviewer: $0.0082 (3 loops, 7,200 tokens)
  🎯 Master: $0.0010 (2 calls, 600 tokens)
════════════════════════════════════════════════════════════════════
```

## Benefits

1. **Transparency**: See exactly what each agent costs
2. **Optimization**: Identify which loops/agents are expensive
3. **Budgeting**: Track costs across runs
4. **Debugging**: Understand token usage patterns
5. **Reporting**: Export cost data for analysis

## Model Pricing Updates

Update prices in `llm_cost_tracker.py` as providers change pricing:
```python
LLM_PRICING = {
    "openai/gpt-4o-mini": {
        "input": 0.150,   # Per 1M tokens
        "output": 0.600
    },
    # Add new models here
}
```

## Notes

- Costs are tracked in USD
- Prices are per 1 million tokens
- System automatically selects pricing based on model_id
- Falls back to default pricing if model not found
- All tracking is in-memory (doesn't persist across restarts)
- For production, consider adding database persistence
