# Critical Fixes for 3 Major Issues

## Issue 1: Reviewer Schema Bug (BLOCKING)
**File**: `config/schemas.py` line 604-672
**Problem**: Function `create_quality_evaluation_prompt()` tries to access `analysis.key_findings` which doesn't exist in `ResearchReportOutput`
**Fix**: Make function handle both schemas

## Issue 2: Data Contamination
**Problem**: Agent uses transcript data ($133B) instead of supplied_metrics
**Root Cause**: System prompt not explicit enough about data discipline
**Fix**: Add explicit data discipline rules at TOP of system prompt

## Issue 3: Shotgun Tool Calls
**Problem**: Agent makes 4-5 generic tool calls at once ("cash", "strategy", etc.)
**Root Cause**: Tool calling instructions are too verbose and confusing
**Fix**: Simplify tool calling to 3 simple rules

## Model Choice
User asked about GPT-4.5-mini → Already using gpt-4o-mini (best balance of cost/quality)

---

## Fix 1: Reviewer Schema Bug

```python
# Replace lines 596-672 in config/schemas.py

def create_quality_evaluation_prompt(analysis: AnalysisOutput, input_data: AnalysisInput) -> str:
    """Create evaluation prompt for reviewer - supports both schemas."""

    # Detect schema type
    is_research = isinstance(analysis, ResearchReportOutput)

    # Get findings from correct field
    if is_research:
        findings = [b.bullet_text for b in analysis.analysis_bullets]
        findings_count = len(findings)
    else:
        findings = analysis.key_findings
        findings_count = len(findings)

    # Format findings
    findings_text = "\n".join([f"{i+1}. {f[:250]}..." if len(f) > 250 else f"{i+1}. {f}"
                               for i, f in enumerate(findings)]) if findings else "(No findings)"

    # Tool calls
    tool_calls = analysis.tool_calls_made if analysis.tool_calls_made else []
    tool_text = "\n".join([f"  - {c.get('function', c if isinstance(c, str) else 'unknown')}"
                           for c in tool_calls[:10]]) if tool_calls else "(None)"

    # Qualitative evidence count
    qual_count = (sum(len(b.qualitative_sources) for b in analysis.analysis_bullets)
                  if is_research else len(analysis.qualitative_evidence))

    prompt = f"""
Evaluate analysis quality:

COMPANY: {analysis.company} | DOMAIN: {analysis.domain} | CONFIDENCE: {analysis.confidence_level}

SUMMARY: {analysis.executive_summary[:300]}...

FINDINGS ({findings_count}):
{findings_text}

TOOLS USED ({len(tool_calls)}):
{tool_text}

EVIDENCE: {qual_count} qualitative sources | {len(analysis.risk_factors)} risks | {len(analysis.recommendations)} recommendations

EVALUATE:
1. Evidence quality - Claims backed by data?
2. Logic - Conclusions follow from evidence?
3. Data usage - Supplied metrics used correctly?
4. Specificity - Findings actionable?
5. Risks - Comprehensive assessment?

Return: verdict ("approved"/"needs_revision"/"rejected"), quality_score (0.0-1.0), strengths (2-3), weaknesses (2-3).
"""
    return prompt
```

## Fix 2: Data Discipline - Add to VERY TOP of System Prompt

**File**: `agents/analyst_agent.py` - Add IMMEDIATELY after `<instructions>` tag

```
🚨 DATA DISCIPLINE RULES (CRITICAL):
1. QUANTITATIVE DATA ONLY from supplied_metrics and supplied_ratios JSON
2. NEVER extract financial numbers from tool outputs (transcripts, reports, news)
3. Tools provide QUALITATIVE context only (management commentary, strategy)
4. If you see "$133B" in transcript → IGNORE IT → Use supplied_metrics["cash_and_short_term_investments"]

VIOLATION EXAMPLE (WRONG): "Cash reached $133B per Q3 transcript"
CORRECT EXAMPLE: "Cash of $59.4B (supplied_metrics Q3-2025) represents..."
```

## Fix 3: Tool Calling Strategy - Replace Entire Section

**Current**: ~100 lines of verbose tool calling instructions
**Replace with**:

```
═══════════════════════════════════════════════════════════════
                    TOOL CALLING STRATEGY
═══════════════════════════════════════════════════════════════

THREE SIMPLE RULES:
1. START: Analyze supplied data FIRST (5-10 minutes of thinking)
2. IDENTIFY: Note 1-2 SPECIFIC questions that need qualitative context
3. CALL: Make ONE targeted tool call with a specific, detailed query

EXAMPLES:
❌ BAD: search_transcripts(query="cash")  ← Too vague
❌ BAD: search_transcripts(query="management commentary")  ← Too generic
✅ GOOD: search_transcripts(query="Did management discuss reasons for Q2-Q3 2025 working capital decline of 15% and any mitigation plans?")

❌ BAD: Making 5 tool calls in parallel at start
✅ GOOD: Analyze data → Identify gap → One targeted call → Integrate → Repeat if needed (max 6 total)

REMEMBER: Tools = CONTEXT, not DATA. Numbers come from supplied JSON.
```

---

## Why This Works

1. **Reviewer Fix**: Handles both schemas properly
2. **Data Discipline**: Rules at TOP + visual markers (🚨) + examples
3. **Tool Strategy**: Simple 3-rule system + clear examples

## Implementation Priority

1. CRITICAL: Fix reviewer schema bug (blocking analysis)
2. HIGH: Add data discipline rules to top of prompt
3. MEDIUM: Simplify tool calling section

## Alternative: If prompts still confuse LLM

Consider **few-shot examples** in system prompt instead of rules:

```
EXAMPLE ANALYSIS PROCESS:

Step 1: "I see cash_and_short_term_investments grew from $55.4B (Q1-2025) to $59.4B (Q3-2025) in supplied_metrics."

Step 2: "I need qualitative context for WHY cash increased. Let me make ONE targeted call..."

Tool call: search_transcripts(query="What did CFO say about Q2-Q3 2025 cash position increase and capital allocation strategy?")

Step 3: "CFO stated [quote]. Now I'll integrate: Cash of $59.4B (Q3-2025, supplied_metrics) increased 7% due to [CFO commentary]..."
```

This shows the LLM the EXACT pattern to follow.
