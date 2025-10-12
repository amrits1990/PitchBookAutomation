"""
Analyst Agent - Performs domain-specific financial analysis
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseFinancialAgent
from config.schemas import AnalysisInput, ResearchReportOutput, ReviewResult, get_schema_reference, get_persona_guidance
from config.domain_configs import get_domain_config
from config.settings import ANALYST_MODEL_ID, ANALYST_MAX_TOKENS
from config.llm_cost_tracker import TokenUsage


class AnalystAgent(BaseFinancialAgent):
    """
    Analyst agent that performs domain-specific financial analysis.
    Receives pre-calculated data and makes targeted tool calls for qualitative evidence.
    """
    
    def __init__(self, domain: str, enable_debug: bool = False):
        """
        Initialize analyst for specific domain.
        
        Args:
            domain: Domain type (liquidity, leverage, etc.)
            enable_debug: Enable debug logging
        """
        self.domain = domain
        self.domain_config = get_domain_config(domain)
        
        super().__init__(
            agent_name=f"{domain}_analyst",
            model_id=ANALYST_MODEL_ID,
            enable_debug=enable_debug,
            use_memory=False,  # Stateless for cost optimization
            max_tokens=ANALYST_MAX_TOKENS
        )
    
    def _get_agent_instructions(self) -> str:
        """Get analyst agent instructions for research-grade reports."""
        return f"""
You are an expert {self.domain} analyst producing research-grade financial analysis reports.
Your reports should read like those from top-tier investment research firms after days of comprehensive analysis.

═══════════════════════════════════════════════════════════════
                        🎯 MISSION
═══════════════════════════════════════════════════════════════

Produce a comprehensive, well-sourced financial analysis report that:
1. Selects 2-4 KEY RATIOS and 2-4 KEY METRICS worth charting
2. Builds 3-10 ANALYSIS BULLETS integrating quantitative + qualitative evidence
3. Provides DETAILED SOURCE ATTRIBUTION for every qualitative claim
4. Delivers actionable RECOMMENDATIONS and identifies KEY RISKS
5. Reads like an expert analyst's deep-dive research report

═══════════════════════════════════════════════════════════════
                🚨 DATA DISCIPLINE RULES (CRITICAL!) 🚨
═══════════════════════════════════════════════════════════════

⚠️ VIOLATION OF THESE RULES = ANALYSIS REJECTED

RULE 1: NUMBERS ONLY from supplied_metrics and supplied_ratios JSON
   ✅ CORRECT: "Cash of $59.4B (Q3-2025 supplied_metrics) increased..."
   ❌ WRONG: "Cash reached $133B per Q3 transcript..." ← NEVER do this!

RULE 2: Tools provide QUALITATIVE context ONLY (no numbers!)
   ✅ CORRECT: [search_transcripts] → "CFO discussed working capital efficiency"
   ❌ WRONG: [search_transcripts] → Extract "$133B cash" ← IGNORE THIS!

RULE 3: If transcript says "$133B" but supplied_metrics says "$59.4B" → USE $59.4B
   Transcripts are commentary. Supplied data is authoritative source of truth.

═══════════════════════════════════════════════════════════════
            STEP 1: SELECT CHART-WORTHY DATA (2-4 each)
═══════════════════════════════════════════════════════════════

From supplied data, select the MOST IMPORTANT ratios and metrics to highlight.

⚠️ CRITICAL RULES FOR CHART DATA:
1. TIME PERIOD: Include data for the FULL time period requested (e.g., "3 years" = 12 quarters)
2. RECENCY: ALWAYS use the MOST RECENT quarters available (sort by date, take latest)
3. PEER INCLUSION: Include ALL peers provided in peer_comparison (never exclude randomly)

📈 CHARTABLE RATIOS (Select 2-4):
   Selection criteria:
   - Shows clear trend (improving/declining/stable)
   - Relevant to {self.domain} domain analysis
   - Has peer comparison data for context
   - Tells an important story about company performance

   For each selected ratio:
   - Extract company values: Use MOST RECENT quarters from supplied_ratios (sorted by date)
   - Extract peer values: Include ALL peers from peer_comparison (don't skip any)
   - ⚠️ PEER COMPARISON USES TEMPORAL PERIODS:
     * Data organized as PERIOD 1, PERIOD 2, etc. (clustered by date proximity, not quarter labels)
     * ONLY compare companies within SAME period number (Period 1 AAPL vs Period 1 MSFT)
     * Each company keeps own fiscal quarter labels (AAPL Q3-2025 vs MSFT Q4-2025 in same period = OK)
     * Read the period_text in PEER COMPARISON section - it shows which data points are grouped
   - Determine trend_direction: improving/declining/stable/volatile
   - Write interpretation: What does this ratio tell us?

   ❌ WRONG: {{"Q1-2024": 1.5, "Q3-2024": 1.7, "Q1-2025": 1.8}} ← Missing quarters, not consecutive
   ✅ CORRECT: {{"Q3-2025": 1.8, "Q2-2025": 1.75, "Q1-2025": 1.7, "Q4-2024": 1.65}} ← Most recent, consecutive

📊 CHARTABLE METRICS (Select 2-4):
   Selection criteria:
   - Key performance indicator for {self.domain} domain
   - Shows meaningful trend over quarters
   - Supports your analysis narrative
   - Important for investment decision-making

   For each selected metric:
   - Extract values: Use MOST RECENT quarters from supplied_metrics (sorted by date)
   - Include enough quarters to cover the requested time period
   - Determine trend_direction: increasing/decreasing/stable/volatile
   - Write interpretation: What does this trend mean?

   ❌ WRONG: {{"Q3-2025": 55.4B, "Q2-2025": 48.5B, "Q1-2024": 67.8B}} ← Skips quarters, not consecutive
   ✅ CORRECT: {{"Q3-2025": 55.4B, "Q2-2025": 48.5B, "Q1-2025": 50.2B, "Q4-2024": 53.8B}} ← Most recent, consecutive

═══════════════════════════════════════════════════════════════
              STEP 2: TOOL STRATEGY (3 Simple Rules)
═══════════════════════════════════════════════════════════════

RULE 1: ANALYZE FIRST (5 mins of thinking about supplied data)
RULE 2: IDENTIFY 1-2 SPECIFIC questions needing qualitative context
RULE 3: MAKE ONE targeted tool call with detailed query

💡 IMPORTANT: You can query about ANY company - the subject company OR peer companies!
   This helps enrich peer comparisons with qualitative context.

EXAMPLES:
❌ BAD: search_transcripts(query="cash") ← Too vague!
❌ BAD: search_transcripts(query="management") ← Generic!
✅ GOOD: search_transcripts(query="Why did working capital decline 15% from Q2 to Q3 2025? Any mitigation plans discussed by CFO?") ← Specific!
✅ GOOD: search_transcripts(query="What did MSFT CFO say about their liquidity management strategy in latest earnings call?", company="MSFT") ← Peer query!
✅ GOOD: search_annual_reports(query="GOOGL's working capital management approach and cash allocation strategy", company="GOOGL") ← Peer query!

TOOLS AVAILABLE:
- search_transcripts → Management commentary, strategy (works for subject company AND peers)
- search_annual_reports → Risk factors, MD&A, business segments (works for subject company AND peers)
- search_news → Recent events, competitive dynamics (works for subject company AND peers)
- get_share_price_data → Market valuation context (works for subject company AND peers)

💡 TIP: Enrich your peer comparison analysis by querying peer companies directly!
   Example: If comparing AAPL to MSFT, you can call search_transcripts for both companies.

APPROACH: Analyze → Question → ONE targeted call → Integrate → Repeat if needed (max 6 total)

═══════════════════════════════════════════════════════════════
         STEP 3: BUILD ANALYSIS BULLETS (3-10 bullets)
═══════════════════════════════════════════════════════════════

Each analysis bullet must INTEGRATE quantitative + qualitative evidence.

📝 BULLET STRUCTURE:
   {{
     "bullet_text": "2-4 sentences making a specific analytical point",
     "quantitative_evidence": [
       "Specific numbers/ratios from supplied data (NOT from tools!)",
       "Example: Current Ratio: Q1-2024=1.5, Q2-2024=1.6, Q3-2024=1.7"
     ],
     "qualitative_sources": [
       {{
         "tool_name": "search_transcripts",
         "period": "Q3-2024",
         "speaker_name": "Luca Maestri",
         "chunk_summary": "CFO discussed cash management strategy and efficiency gains"
       }}
     ],
     "importance": "critical" | "high" | "moderate"
   }}

✅ GOOD EXAMPLE:
   "Apple's liquidity position has strengthened significantly over the past three
   quarters, with the current ratio improving from 1.5 to 1.7, driven by strategic
   cash management initiatives. The CFO emphasized disciplined working capital
   management and optimized cash conversion cycles as key priorities."

   Quantitative: ["Current Ratio: Q1-2024=1.5, Q2-2024=1.6, Q3-2024=1.7"]
   Sources: [search_transcripts, Q3-2024, Speaker: Luca Maestri, "working capital management"]
   Importance: critical

❌ BAD EXAMPLE:
   "Liquidity is good."  ← Too vague, no evidence, no sources

BULLET SELECTION:
- Start with 3 bullets minimum, up to 10 maximum
- Mark importance: critical (must-know), high (important), moderate (context)
- Cover different aspects of {self.domain} domain
- Each bullet should advance the investment thesis

═══════════════════════════════════════════════════════════════
          STEP 4: WRITE EXECUTIVE SUMMARY (3-5 sentences)
═══════════════════════════════════════════════════════════════

The executive summary should:
✅ Capture the 2-3 most important takeaways
✅ Reference key quantitative findings (but cite details in bullets)
✅ Provide investment perspective
✅ Be readable standalone (assume reader skips rest of report)
✅ Be at least 100 characters

Example:
"Apple demonstrates robust liquidity with current ratio improving from 1.5 to 1.7
over three quarters, outpacing key operating cycle metrics. Strategic cash management
initiatives have strengthened the balance sheet while maintaining operational flexibility.
Peer comparison shows Apple is converging toward industry leader levels, though still
below top-tier peers like Microsoft."

═══════════════════════════════════════════════════════════════
                    ⭐ QUALITY STANDARDS
═══════════════════════════════════════════════════════════════

Before submitting your analysis, verify:

✓ Chart data: 2-4 ratios + 2-4 metrics selected (not more, not less)
✓ Analysis bullets: 3-10 bullets, each with quant evidence + qual sources
✓ Source attribution: Every qualitative claim has DetailedSource with tool + metadata
✓ Data discipline: All financial numbers from supplied data (never from tools)
✓ Executive summary: 100+ characters, captures key takeaways
✓ Recommendations: 2-5 specific, actionable recommendations
✓ Risk factors: 2-5 key risks identified
✓ Confidence level: "high" | "medium" | "low" based on evidence quality

OUTPUT FORMAT: Use the ResearchReportOutput schema structure.
TOOL USAGE: Strategic and targeted (respect max_tool_calls limit).
CITATIONS: Every tool-derived insight must have full source attribution.

═══════════════════════════════════════════════════════════════
           📋 EXACT JSON SCHEMA (USE THESE FIELD NAMES!)
═══════════════════════════════════════════════════════════════

{{
  "run_id": "auto_generated",
  "domain": "{self.domain}",
  "company": "Company Name",
  "ticker": "TICK",
  "analysis_timestamp": "2025-01-15T10:00:00",
  "executive_summary": "3-5 sentence summary covering key takeaways and investment perspective (minimum 100 chars)",

  "chartable_ratios": [
    {{
      "ratio_name": "Current Ratio",  // ← USE "ratio_name" NOT "name"!
      "company_values": {{"Q3-2025": 1.8, "Q2-2025": 1.75, "Q1-2025": 1.7, "Q4-2024": 1.65}},  // ← MOST RECENT quarters first!
      "peer_values": {{  // ← REQUIRED: Include ALL peers (don't skip any)!
        "MSFT": {{"Q4-2025": 2.2, "Q3-2025": 2.15, "Q2-2025": 2.1, "Q1-2025": 2.0}},  // MSFT's fiscal quarters
        "GOOGL": {{"Q3-2025": 2.0, "Q2-2025": 1.98, "Q1-2025": 1.95, "Q4-2024": 1.9}},  // GOOGL's fiscal quarters
        "META": {{"Q3-2025": 1.85, "Q2-2025": 1.8, "Q1-2025": 1.78, "Q4-2024": 1.75}}  // Include ALL peers from list!
      }},
      "interpretation": "Brief interpretation of what this ratio trend means",
      "trend_direction": "improving"  // Options: "improving", "declining", "stable", "volatile"
    }}
    // ⚠️ Use fiscal_quarter + date from each PERIOD in peer comparison data (see period_text). Each company keeps own labels.
    // Include 2-4 ratios total
  ],

  "chartable_metrics": [
    {{
      "metric_name": "Cash and Equivalents",  // ← USE "metric_name" NOT "name"!
      "values": {{"Q3-2025": 55400000000, "Q2-2025": 48500000000, "Q1-2025": 50200000000, "Q4-2024": 53800000000}},  // ← MOST RECENT quarters first, consecutive!
      "unit": "USD",
      "trend_direction": "increasing",
      "interpretation": "Brief interpretation of this metric trend"
    }}
    // Include 2-4 metrics total
  ],

  "analysis_bullets": [
    {{
      "bullet_text": "2-4 sentences of in-depth analytical insight integrating quantitative data with qualitative context. Must be at least 200 characters to ensure sufficient depth and avoid superficial analysis. Explain root causes, connect dots between metrics, and provide forward-looking perspective.",
      "quantitative_evidence": [
        "Current Ratio: Q1-2024=1.5, Q2-2024=1.6, Q3-2024=1.7",
        "Cash increased 20% QoQ from $50B to $60B"
      ],
      "qualitative_sources": [
        {{
          "tool_name": "search_transcripts",
          "period": "Q3-2024",
          "speaker_name": "Luca Maestri",  // ← Include speaker for transcripts
          "chunk_summary": "CFO discussed working capital efficiency initiatives"
        }}
      ],
      "importance": "critical"  // Options: "critical", "high", "moderate"
    }}
    // Include 3-10 bullets, each 200+ chars
  ],

  "recommendations": ["Rec 1", "Rec 2"],  // 2-5 recommendations
  "risk_factors": ["Risk 1", "Risk 2"],  // 2-5 risks
  "confidence_level": "high",  // "high", "medium", or "low"
  "tool_calls_made": [],  // Auto-filled by system
  "data_sources_summary": "Brief summary of data sources used"
}}

CRITICAL FIELD NAMES:
- "ratio_name" (NOT "name")
- "metric_name" (NOT "name")
- "company_values" (NOT "values" for ratios)
- "peer_values" (REQUIRED - include ALL peers, no random exclusions)
- "bullet_text" (minimum 200 characters!)

CRITICAL DATA RULES:
- Use MOST RECENT quarters available (sort by date, take latest)
- Include enough consecutive quarters for requested time period
- Don't skip quarters (Q1→Q2→Q3→Q4, not Q1→Q3→Q1)
- Include ALL peers in peer_values (if 3 peers supplied, show all 3)
- Peer comparison: Extract fiscal_quarter from each PERIOD in period_text, ONLY compare within same period

Remember: You're producing a research report that an investment professional
would rely on for decision-making. Quality over quantity. Precision over vagueness.
"""
    
    async def analyze(self,
                     analysis_input: AnalysisInput,
                     previous_analysis: Optional[ResearchReportOutput] = None,
                     previous_feedback: Optional[ReviewResult] = None,
                     loop_iteration: int = 1) -> tuple[ResearchReportOutput, TokenUsage]:
        """
        Perform domain analysis based on input data.
        
        Args:
            analysis_input: Input data and parameters
            previous_analysis: Previous analysis if this is a revision
            previous_feedback: Feedback from reviewer if this is a revision
            loop_iteration: Current loop iteration number
            
        Returns:
            Complete analysis output
        """
        self.logger.info(f"Starting {self.domain} analysis for {analysis_input.company} (loop {loop_iteration})")

        # Log revision mode status
        if loop_iteration > 1 and previous_feedback:
            self.logger.info(f"🔄 REVISION MODE ACTIVE - Loop {loop_iteration}")
            self.logger.info(f"Previous verdict: {previous_feedback.verdict}, score: {previous_feedback.quality_score:.2f}")
            self.logger.info(f"Weaknesses to address: {len(previous_feedback.weaknesses)}")
            self.logger.info(f"Specific improvements requested: {len(previous_feedback.specific_improvements)}")
            for i, weakness in enumerate(previous_feedback.weaknesses[:3], 1):
                self.logger.info(f"  Weakness {i}: {weakness[:100]}")
        else:
            self.logger.info("📝 INITIAL ANALYSIS MODE - Loop 1")

        # Debug analysis input
        self.logger.debug(f"AnalysisInput run_id: {analysis_input.run_id}")
        self.logger.debug(f"AnalysisInput company: {analysis_input.company}")
        self.logger.debug(f"AnalysisInput supplied_metrics type: {type(analysis_input.supplied_metrics)}")
        self.logger.debug(f"AnalysisInput supplied_ratios type: {type(analysis_input.supplied_ratios)}")
        self.logger.debug(f"AnalysisInput peer_comparison type: {type(analysis_input.peer_comparison)}")
        
        try:
            # Build analysis prompt
            self.logger.debug("About to build analysis prompt...")
            analysis_prompt = self._build_analysis_prompt(
                analysis_input=analysis_input,
                previous_analysis=previous_analysis,
                previous_feedback=previous_feedback,
                loop_iteration=loop_iteration
            )
            self.logger.debug("Analysis prompt built successfully")
        except Exception as e:
            self.logger.error(f"Failed to build analysis prompt: {e}")
            raise
        
        # Add context for agent
        context = {
            "domain": self.domain,
            "company": analysis_input.company,
            "loop_iteration": loop_iteration,
            "schema_reference": get_schema_reference("ResearchReportOutput")  # Updated for research-grade reports
        }
        
        try:
            # Run analysis - now returns response, tool calls, and token usage
            response, tool_calls_captured, token_usage = await self._run_agent(analysis_prompt, context)

            self.logger.info(f"Agent returned {len(tool_calls_captured)} tool calls")

            # Parse response into ResearchReportOutput
            analysis_output = self._parse_analysis_response(
                response=response,
                analysis_input=analysis_input,
                captured_tool_calls=tool_calls_captured
            )

            self.logger.info(f"Analysis completed - confidence: {analysis_output.confidence_level}, tool_calls: {len(analysis_output.tool_calls_made)}")
            return analysis_output, token_usage

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            # Re-raise the exception - we must have valid ResearchReportOutput
            raise Exception(f"Analysis failed and could not produce ResearchReportOutput: {e}")
    
    def _build_analysis_prompt(self,
                              analysis_input: AnalysisInput,
                              previous_analysis: Optional[ResearchReportOutput],
                              previous_feedback: Optional[ReviewResult],
                              loop_iteration: int) -> str:
        """Build comprehensive analysis prompt (routes to revision prompt if loop > 1)."""

        # Route to streamlined revision prompt for subsequent loops
        if loop_iteration > 1 and previous_feedback and previous_analysis:
            self.logger.info(f"Using streamlined revision prompt for loop {loop_iteration}")
            return self._build_revision_prompt(analysis_input, previous_analysis, previous_feedback, loop_iteration)

        # Otherwise, use full initial analysis prompt
        self.logger.debug(f"Building initial analysis prompt for {analysis_input.company}")
        
        # Debug the input data structures
        self.logger.debug(f"supplied_metrics type: {type(analysis_input.supplied_metrics)}")
        self.logger.debug(f"supplied_ratios type: {type(analysis_input.supplied_ratios)}")
        self.logger.debug(f"peer_comparison type: {type(analysis_input.peer_comparison)}")
        
        if analysis_input.supplied_metrics:
            self.logger.debug(f"supplied_metrics keys: {list(analysis_input.supplied_metrics.keys())[:3]}")
        if analysis_input.supplied_ratios:
            self.logger.debug(f"supplied_ratios keys: {list(analysis_input.supplied_ratios.keys())[:3]}")
        if analysis_input.peer_comparison:
            self.logger.debug(f"peer_comparison content: {analysis_input.peer_comparison}")
        
        # Base prompt from domain configuration
        # Safe extraction of peer count
        peer_count = 0
        if analysis_input.peer_comparison and isinstance(analysis_input.peer_comparison, dict):
            peers = analysis_input.peer_comparison.get('peers', [])
            if isinstance(peers, list):
                peer_count = len(peers)
        
        self.logger.debug(f"Extracted peer_count: {peer_count}")
        
        # Prepare format parameters
        format_params = {
            'company': analysis_input.company,
            'user_focus': analysis_input.user_focus,
            'time_period': analysis_input.time_period,
            'persona': analysis_input.persona,
            'persona_guidance': get_persona_guidance(analysis_input.persona),
            'metrics_count': len(analysis_input.supplied_metrics) if analysis_input.supplied_metrics else 0,
            'ratios_count': len(analysis_input.supplied_ratios) if analysis_input.supplied_ratios else 0,
            'peer_count': peer_count,
            'max_tool_calls': analysis_input.max_tool_calls,
            'schema_ref': get_schema_reference("ResearchReportOutput")  # Updated for research-grade reports
        }
        
        self.logger.debug(f"Format parameters: {format_params}")
        
        try:
            self.logger.debug("About to format analyst_prompt_template...")
            base_prompt = self.domain_config.analyst_prompt_template.format(**format_params)
            self.logger.debug("Analyst prompt template formatted successfully")
        except Exception as e:
            self.logger.error(f"Error formatting analyst_prompt_template: {e}")
            self.logger.error(f"Template: {self.domain_config.analyst_prompt_template}")
            raise
        
        # Add supplied data details
        data_section = self._format_supplied_data(analysis_input)
        
        # Add revision context if this is not the first loop
        revision_section = ""
        revision_mode_instructions = ""

        if loop_iteration > 1 and previous_feedback:
            self.logger.info(f"Adding revision context to prompt (loop {loop_iteration})")
            revision_section = self._format_revision_context(previous_feedback, previous_analysis)
            self.logger.debug(f"Revision section length: {len(revision_section)} characters")

            # Add explicit REVISION MODE instructions
            revision_mode_instructions = f"""
═══════════════════════════════════════════════════════════════
                    🔄 REVISION MODE ACTIVE 🔄
═══════════════════════════════════════════════════════════════

⚠️ CRITICAL: This is Loop {loop_iteration}/{analysis_input.max_loops} - REVISION required, NOT fresh analysis!

REVISION APPROACH:
1. START with your previous analysis (provided below)
2. REVIEW your previous tool calls - see what you ALREADY asked
3. IDENTIFY which specific weaknesses the reviewer cited
4. MAKE NEW TARGETED TOOL CALLS with DIFFERENT queries to fill gaps
5. UPDATE the relevant sections of your analysis
6. MAINTAIN the strengths that were working

DO NOT:
❌ Generate a completely fresh analysis
❌ Ignore the reviewer's specific feedback
❌ Repeat the same weaknesses from the previous iteration
❌ Make random tool calls unrelated to reviewer feedback
❌ REPEAT tool calls with same/similar queries you already made

DO:
✅ Review your previous tool call history below
✅ Ask DIFFERENT questions using DIFFERENT search terms
✅ Focus new tool calls on addressing specific weaknesses
✅ Add missing data/evidence the reviewer requested
✅ Enhance peer comparisons if that was a weakness
✅ Improve recommendation specificity if needed
✅ Keep the strong elements from your previous work
"""

        # Combine sections
        full_prompt = f"""
{base_prompt}

SUPPLIED FINANCIAL DATA:
{data_section}

{revision_mode_instructions}

{revision_section}

{"🔄 REVISION TASK: Address the specific weaknesses above while maintaining strengths." if loop_iteration > 1 else ""}

Return {"REVISED" if loop_iteration > 1 else ""} analysis as valid JSON using ResearchReportOutput schema.
"""
        
        return full_prompt
    
    def _format_supplied_data(self, analysis_input: AnalysisInput) -> str:
        """Format supplied data for prompt."""
        self.logger.debug("Starting _format_supplied_data")
        sections = []
        
        # Metrics section
        try:
            self.logger.debug("Starting metrics section processing")
            if analysis_input.supplied_metrics and isinstance(analysis_input.supplied_metrics, dict):
                self.logger.debug(f"Processing {len(analysis_input.supplied_metrics)} metrics")
                metrics_preview = []
                for i, (metric, quarters) in enumerate(list(analysis_input.supplied_metrics.items())):
                    self.logger.debug(f"Processing metric {i}: {metric}, quarters type: {type(quarters)}")
                    if isinstance(quarters, dict):
                        try:
                            self.logger.debug(f"Quarters data keys: {list(quarters.keys())}")
                            historical_data = []
                            
                            # Handle real SECFinancialRAG format with 'values' and 'periods'
                            if 'values' in quarters and 'periods' in quarters:
                                values = quarters['values']
                                periods = quarters['periods']
                                self.logger.debug(f"Real data format - values: {len(values) if isinstance(values, list) else type(values)}, periods: {len(periods) if isinstance(periods, list) else type(periods)}")
                                
                                if isinstance(values, list) and isinstance(periods, list) and len(values) == len(periods):
                                    # Combine periods and values, sort by period (most recent first)
                                    combined = list(zip(periods, values))
                                    sorted_data = sorted(combined, key=lambda x: x[0], reverse=True)
                                    
                                    # Show up to 12 quarters for 3-year analysis (LTM data)
                                    for period, value in sorted_data[:12]:
                                        if value is not None:
                                            formatted_value = self._format_currency(float(value))
                                            historical_data.append(f"{period}: {formatted_value}")
                                else:
                                    self.logger.warning(f"Values/periods lists mismatch - values: {len(values) if isinstance(values, list) else 'not list'}, periods: {len(periods) if isinstance(periods, list) else 'not list'}")
                            
                            # Handle simulated format with quarter keys
                            elif any(key.startswith('Q') for key in quarters.keys()):
                                sorted_quarters = sorted(quarters.keys(), reverse=True)  # Most recent first
                                for quarter in sorted_quarters:
                                    value = quarters[quarter]
                                    formatted_value = self._format_currency(value)
                                    historical_data.append(f"{quarter}: {formatted_value}")
                            
                            if historical_data:
                                # Show all available quarters (up to 12 for 3 years)
                                trend_info = " | ".join(historical_data[:12])
                                metrics_preview.append(f"  {metric}: {trend_info}")
                                self.logger.debug(f"Successfully appended metric {i} with historical data")
                            else:
                                # Fallback for unexpected format
                                metrics_preview.append(f"  {metric}: {str(quarters)[:100]}...")
                                self.logger.debug(f"Used fallback format for metric {i}")
                        except Exception as e:
                            self.logger.error(f"Error processing metric {i} ({metric}): {e}")
                            metrics_preview.append(f"  {metric}: Data processing error")
                    else:
                        # Handle non-dict quarters data
                        self.logger.debug(f"Non-dict quarters data for {metric}: {type(quarters)}")
                        metrics_preview.append(f"  {metric}: {str(quarters)[:50]}...")
                
                self.logger.debug("Finished processing individual metrics, about to append to sections")
                if metrics_preview:
                    self.logger.debug(f"Appending metrics section with {len(metrics_preview)} metrics")
                    sections.append(f"METRICS ({len(analysis_input.supplied_metrics)} total):\n" + "\n".join(metrics_preview))
                self.logger.debug("Metrics section processing completed")
        except Exception as e:
            self.logger.error(f"Error processing metrics section: {e}")
            self.logger.exception("Full metrics section exception")
            sections.append("METRICS: Error processing metrics data")
        
        # Ratios section  
        try:
            self.logger.debug("Starting ratios section processing")
            if analysis_input.supplied_ratios and isinstance(analysis_input.supplied_ratios, dict):
                self.logger.debug(f"Processing {len(analysis_input.supplied_ratios)} ratios")
                ratios_preview = []
                for i, (ratio, quarters) in enumerate(list(analysis_input.supplied_ratios.items())):
                    self.logger.debug(f"Processing ratio {i}: {ratio}, quarters type: {type(quarters)}")
                    if isinstance(quarters, dict):
                        try:
                            historical_data = []
                            
                            # Handle SECFinancialRAG category format (e.g., "efficiency": {"Asset_Turnover_Q3-2025": {...}})
                            if isinstance(quarters, dict) and any('_Q' in key for key in quarters.keys()):
                                self.logger.debug(f"SECFinancialRAG category format detected for {ratio}")
                                # Extract individual ratios from the category
                                individual_ratios = []
                                for ratio_name, ratio_data in quarters.items():
                                    if isinstance(ratio_data, dict) and 'value' in ratio_data:
                                        value = float(ratio_data['value']) if hasattr(ratio_data['value'], '__float__') else ratio_data['value']
                                        period = ratio_data.get('period_end_date', ratio_name.split('_Q')[-1] if '_Q' in ratio_name else 'Unknown')
                                        individual_ratios.append((period, value, ratio_name.split('_')[0]))
                                
                                # Group by ratio type and show trends
                                from collections import defaultdict
                                ratio_groups = defaultdict(list)
                                for period, value, ratio_type in individual_ratios:
                                    ratio_groups[ratio_type].append((period, value))
                                
                                for ratio_type, values in ratio_groups.items():
                                    # Show up to 12 quarters for 3-year analysis
                                    sorted_values = sorted(values, key=lambda x: x[0], reverse=True)[:12]
                                    trend_line = " | ".join([f"{period}: {value:.2f}" for period, value in sorted_values])
                                    historical_data.append(f"    {ratio_type}: {trend_line}")
                                
                                if historical_data:
                                    ratios_preview.append(f"  {ratio} Category:")
                                    ratios_preview.extend(historical_data)
                            
                            # Handle regular format with 'values' and 'periods'
                            elif 'values' in quarters and 'periods' in quarters:
                                values = quarters['values']
                                periods = quarters['periods']
                                
                                if isinstance(values, list) and isinstance(periods, list) and len(values) == len(periods):
                                    combined = list(zip(periods, values))
                                    sorted_data = sorted(combined, key=lambda x: x[0], reverse=True)
                                    
                                    # Show up to 12 quarters for 3-year analysis
                                    for period, value in sorted_data[:12]:
                                        if value is not None:
                                            if isinstance(value, (int, float)):
                                                historical_data.append(f"{period}: {value:.2f}")
                                            else:
                                                historical_data.append(f"{period}: {str(value)}")
                                    
                                    if historical_data:
                                        trend_info = " | ".join(historical_data)
                                        ratios_preview.append(f"  {ratio}: {trend_info}")
                            
                            # Handle simulated format with quarter keys
                            elif any(key.startswith('Q') for key in quarters.keys()):
                                sorted_quarters = sorted(quarters.keys(), reverse=True)
                                for quarter in sorted_quarters:
                                    value = quarters[quarter]
                                    if isinstance(value, (int, float)):
                                        historical_data.append(f"{quarter}: {value:.2f}")
                                    else:
                                        historical_data.append(f"{quarter}: {str(value)}")
                                
                                if historical_data:
                                    # Show all available quarters (up to 12 for 3 years)
                                    trend_info = " | ".join(historical_data[:12])
                                    ratios_preview.append(f"  {ratio}: {trend_info}")
                            
                            # Fallback for unexpected formats
                            else:
                                ratios_preview.append(f"  {ratio}: {str(quarters)[:200]}...")
                            
                            self.logger.debug(f"Successfully processed ratio {i}: {ratio}")
                        except Exception as e:
                            self.logger.error(f"Error processing ratio {i} ({ratio}): {e}")
                            ratios_preview.append(f"  {ratio}: Data processing error")
                    else:
                        # Handle non-dict quarters data
                        self.logger.debug(f"Non-dict quarters data for ratio {ratio}: {type(quarters)}")
                        ratios_preview.append(f"  {ratio}: {str(quarters)[:50]}...")
                
                self.logger.debug("Finished processing individual ratios, about to append to sections")
                if ratios_preview:
                    sections.append(f"RATIOS ({len(analysis_input.supplied_ratios)} total):\n" + "\n".join(ratios_preview))
                self.logger.debug("Ratios section processing completed")
        except Exception as e:
            self.logger.error(f"Error processing ratios section: {e}")
            self.logger.exception("Full ratios section exception")
            sections.append("RATIOS: Error processing ratios data")
        
        # Peer comparison section
        try:
            self.logger.debug("Starting peer comparison section processing")
            if analysis_input.peer_comparison and isinstance(analysis_input.peer_comparison, dict):
                peers = analysis_input.peer_comparison.get('peers', [])
                self.logger.debug(f"Found peers: {peers}, type: {type(peers)}")

                # Check if we have the new period summaries format
                period_summaries = analysis_input.peer_comparison.get('period_summaries', [])

                if period_summaries and isinstance(period_summaries, list):
                    # NEW FORMAT: Period summaries with temporal clustering
                    self.logger.debug(f"Using new period summaries format with {len(period_summaries)} periods")

                    peer_lines = []
                    peer_lines.append(f"PEER COMPARISON ({len(peers)} companies): {', '.join(peers[:4])}")
                    peer_lines.append("")

                    # Add each period's formatted text
                    for period_summary in period_summaries:
                        period_text = period_summary.get('period_text', '')
                        if period_text:
                            # Add separator for visual clarity
                            peer_lines.append("─" * 60)
                            peer_lines.append(period_text)
                            peer_lines.append("─" * 60)
                            peer_lines.append("")

                    # Add summary information
                    summary = analysis_input.peer_comparison.get('summary', {})
                    if summary:
                        total_periods = summary.get('total_periods', 0)
                        ratios_compared = summary.get('ratios_compared', [])
                        peer_lines.append(f"Summary: {total_periods} temporal periods, {len(ratios_compared)} ratios compared")

                    sections.append("\n".join(peer_lines))
                    self.logger.debug(f"Generated peer comparison with {len(period_summaries)} periods")

                elif isinstance(peers, list) and len(peers) > 0:
                    # LEGACY FORMAT: Fallback for old format
                    self.logger.debug("Using legacy peer comparison format")
                    peer_lines = []
                    peer_lines.append(f"PEER COMPARISON ({len(peers)} companies): {', '.join(peers[:4])}")

                    # Extract and display comparative ratio values
                    # Look for keys that contain ratio comparison data
                    for key, value in analysis_input.peer_comparison.items():
                        if key in ['peers', 'focus_company', 'summary', 'raw_data', 'categories_compared']:
                            continue  # Skip metadata fields

                        # Handle various peer comparison data formats
                        if isinstance(value, dict):
                            # Check if it's a company-to-value mapping (e.g., {'AAPL': 0.136, 'MSFT': 0.185})
                            if all(isinstance(k, str) and isinstance(v, (int, float, type(None))) for k, v in value.items()):
                                # This is a direct ratio comparison
                                ratio_name = key.replace('_', ' ').title()
                                comparison_text = f"  {ratio_name}:"

                                # Show target company first if present
                                target_ticker = analysis_input.ticker
                                if target_ticker in value:
                                    target_val = value[target_ticker]
                                    if target_val is not None:
                                        comparison_text += f" {target_ticker}={target_val:.3f}"

                                # Show peer values
                                for peer in peers:
                                    if peer in value and peer != target_ticker:
                                        peer_val = value[peer]
                                        if peer_val is not None:
                                            comparison_text += f" | {peer}={peer_val:.3f}"

                                peer_lines.append(comparison_text)
                            else:
                                # More complex structure - show simplified version
                                peer_lines.append(f"  {key}: {str(value)[:100]}...")
                        elif isinstance(value, (list, str, int, float)):
                            # Simple value - display it
                            peer_lines.append(f"  {key}: {value}")

                    sections.append("\n".join(peer_lines))
                    self.logger.debug(f"Generated peer comparison with {len(peer_lines)} lines")
                else:
                    sections.append("PEER COMPARISON: Available (no peers listed)")

                self.logger.debug("Peer comparison section processing completed")
        except Exception as e:
            self.logger.error(f"Error processing peer comparison section: {e}")
            self.logger.exception("Full peer comparison section exception")
            sections.append("PEER COMPARISON: Error processing peer data")
        
        try:
            self.logger.debug("About to join sections and return")
            result = "\n\n".join(sections)
            self.logger.debug(f"Successfully joined {len(sections)} sections")
            return result
        except Exception as e:
            self.logger.error(f"Error joining sections: {e}")
            self.logger.exception("Section joining exception")
            return "Error formatting supplied data"
    
    def _format_revision_context(self,
                                previous_feedback: ReviewResult,
                                previous_analysis: Optional[ResearchReportOutput]) -> str:
        """Format revision context for improvement."""

        revision_text = f"""
═══════════════════════════════════════════════════════════════
              📋 REVIEWER FEEDBACK FROM PREVIOUS LOOP 📋
═══════════════════════════════════════════════════════════════

📊 PREVIOUS QUALITY SCORE: {previous_feedback.quality_score:.2f}/1.0
🔖 VERDICT: {previous_feedback.verdict.upper()}

───────────────────────────────────────────────────────────────
                    ✅ STRENGTHS TO MAINTAIN
───────────────────────────────────────────────────────────────
{chr(10).join([f"{i+1}. ✓ {strength}" for i, strength in enumerate(previous_feedback.strengths)])}

───────────────────────────────────────────────────────────────
                    ⚠️ CRITICAL WEAKNESSES TO FIX
───────────────────────────────────────────────────────────────
{chr(10).join([f"{i+1}. ❌ {weakness}" for i, weakness in enumerate(previous_feedback.weaknesses)])}

───────────────────────────────────────────────────────────────
                    🎯 SPECIFIC ACTION ITEMS
───────────────────────────────────────────────────────────────
{chr(10).join([f"{i+1}. 🔧 {improvement}" for i, improvement in enumerate(previous_feedback.specific_improvements)])}

───────────────────────────────────────────────────────────────
                    📝 YOUR PREVIOUS ANALYSIS SUMMARY
───────────────────────────────────────────────────────────────
"""

        # Add previous analysis summary if available
        if previous_analysis:
            # Extract findings from analysis_bullets
            previous_findings = [b.bullet_text for b in previous_analysis.analysis_bullets] if previous_analysis.analysis_bullets else []
            revision_text += f"""
Previous Analysis Bullets Count: {len(previous_findings)}
Previous Tool Calls Made: {len(previous_analysis.tool_calls_made)}
Previous Confidence: {previous_analysis.confidence_level}

First 3 Previous Findings:
{chr(10).join([f"  • {finding[:150]}..." if len(finding) > 150 else f"  • {finding}"
               for finding in previous_findings[:3]])}

"""

        # Add detailed tool call history to prevent redundant calls
        if previous_analysis and previous_analysis.tool_calls_made:
            revision_text += """
───────────────────────────────────────────────────────────────
        🔧 PREVIOUS TOOL CALLS (DO NOT REPEAT THESE!)
───────────────────────────────────────────────────────────────

⚠️ CRITICAL: You already made these tool calls. DO NOT repeat them!
   If you need more information, ask DIFFERENT questions or use DIFFERENT search terms.

"""
            # Format tool call history with details
            for i, tool_call in enumerate(previous_analysis.tool_calls_made, 1):
                if isinstance(tool_call, dict):
                    function_name = tool_call.get('function', 'unknown')
                    parameters = tool_call.get('parameters', {})

                    # Extract the query/search term if available
                    query = parameters.get('query', parameters.get('ticker', parameters.get('company', 'N/A')))

                    revision_text += f"{i}. {function_name}"
                    if query and query != 'N/A':
                        # Show first 100 chars of query
                        query_preview = query[:100] + "..." if len(str(query)) > 100 else query
                        revision_text += f"(query=\"{query_preview}\")"
                    revision_text += "\n"
                elif isinstance(tool_call, str):
                    revision_text += f"{i}. {tool_call}\n"

            revision_text += """
✅ WHAT TO DO INSTEAD:
   - Identify NEW gaps not addressed by previous calls
   - Ask DIFFERENT questions using DIFFERENT keywords
   - Target SPECIFIC information you're missing
   - Example: If you already asked about "cash strategy",
     now ask about "working capital cycle" or "liquidity risk mitigation"

"""

        revision_text += """
═══════════════════════════════════════════════════════════════
                      🎯 REVISION STRATEGY
═══════════════════════════════════════════════════════════════

STEP 1: Review weaknesses above - identify SPECIFIC gaps
STEP 2: Plan targeted tool calls to fill those SPECIFIC gaps
STEP 3: Execute tool calls and gather NEW evidence
STEP 4: UPDATE your analysis with new findings
STEP 5: MAINTAIN the strengths that reviewer liked

REMEMBER: This is a REVISION, not a fresh start!
"""

        return revision_text

    def _format_tool_call_history(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Format tool call history to show what has already been queried."""
        if not tool_calls:
            return "No previous tool calls"

        history_lines = []
        for i, tool_call in enumerate(tool_calls, 1):
            if isinstance(tool_call, dict):
                function_name = tool_call.get('function', 'unknown')
                parameters = tool_call.get('parameters', {})

                # Extract the query/search term if available
                query = parameters.get('query', parameters.get('ticker', parameters.get('company', 'N/A')))

                line = f"{i}. {function_name}"
                if query and query != 'N/A':
                    # Show first 100 chars of query
                    query_preview = query[:100] + "..." if len(str(query)) > 100 else query
                    line += f"(query=\"{query_preview}\")"
                history_lines.append(line)
            elif isinstance(tool_call, str):
                history_lines.append(f"{i}. {tool_call}")

        return "\n".join(history_lines)

    def _build_revision_prompt(self,
                               analysis_input: AnalysisInput,
                               previous_analysis: ResearchReportOutput,
                               previous_feedback: ReviewResult,
                               loop_iteration: int) -> str:
        """
        Build streamlined revision prompt for subsequent loops.
        More token-efficient than initial prompt - includes previous JSON instead of repeating data.
        """

        # Convert previous analysis to JSON for the LLM to refine
        previous_json = previous_analysis.to_json()

        revision_prompt = f"""
═══════════════════════════════════════════════════════════════
              🔄 REVISION MODE - Loop {loop_iteration}/{analysis_input.max_loops}
═══════════════════════════════════════════════════════════════

TASK: Refine your previous analysis based on reviewer feedback.

COMPANY: {analysis_input.company} ({analysis_input.ticker})
DOMAIN: {self.domain}
SCHEMA: {get_schema_reference("ResearchReportOutput")}

───────────────────────────────────────────────────────────────
                      📋 REVIEWER FEEDBACK
───────────────────────────────────────────────────────────────

VERDICT: {previous_feedback.verdict.upper()}
QUALITY SCORE: {previous_feedback.quality_score:.2f}/1.0

STRENGTHS (Keep These):
{chr(10).join([f"✓ {s}" for s in previous_feedback.strengths])}

WEAKNESSES (Fix These):
{chr(10).join([f"❌ {w}" for w in previous_feedback.weaknesses])}

SPECIFIC IMPROVEMENTS NEEDED:
{chr(10).join([f"{i+1}. {imp}" for i, imp in enumerate(previous_feedback.specific_improvements)])}

───────────────────────────────────────────────────────────────
            🔧 PREVIOUS TOOL CALLS (DO NOT REPEAT!)
───────────────────────────────────────────────────────────────

⚠️ You already made these tool calls. Make NEW calls with DIFFERENT queries:

{self._format_tool_call_history(previous_analysis.tool_calls_made) if previous_analysis.tool_calls_made else "No previous tool calls"}

✅ For revision: Ask DIFFERENT questions, use DIFFERENT search terms, target NEW gaps.

───────────────────────────────────────────────────────────────
                   📄 YOUR PREVIOUS ANALYSIS (JSON)
───────────────────────────────────────────────────────────────

{previous_json}

───────────────────────────────────────────────────────────────
                        🎯 REVISION STRATEGY
───────────────────────────────────────────────────────────────

STEP 1: Identify Gaps
   - Review weaknesses and improvement requests
   - Identify what specific evidence is missing
   - Determine which tools can fill those gaps

STEP 2: Targeted Tool Calls (NEW queries only!)
   - REVIEW previous tool calls listed above - DO NOT REPEAT them
   - Make ONLY NEW tool calls with DIFFERENT queries that address specific weaknesses
   - Do NOT call tools for information you already have
   - Use DIFFERENT search terms and questions than before
   - Focus on {analysis_input.max_tool_calls} most impactful NEW calls
   - Extract NEW qualitative evidence

STEP 3: Refine Your JSON
   - START with your previous JSON above
   - UPDATE sections that had weaknesses
   - ADD new evidence from tool calls
   - ENHANCE source attribution if that was weak
   - MAINTAIN the strengths the reviewer liked
   - Ensure chartable_ratios (2-4), chartable_metrics (2-4), analysis_bullets (3-10)

STEP 4: Quality Check
   - All numbers from supplied data (not tools)
   - Every qualitative claim has DetailedSource
   - Executive summary ≥100 chars
   - Recommendations (2-5) and risk_factors (2-5)

───────────────────────────────────────────────────────────────
                          ⚠️ CRITICAL RULES
───────────────────────────────────────────────────────────────

❌ DO NOT generate a completely fresh analysis
❌ DO NOT repeat the same weaknesses
❌ DO NOT extract financial numbers from tools
❌ DO NOT make unnecessary tool calls

✅ DO refine and improve your previous JSON
✅ DO address every weakness mentioned
✅ DO add missing source attributions
✅ DO keep what was working well

OUTPUT: Return the REFINED analysis as valid JSON using ResearchReportOutput schema.

NOTE: The supplied financial data (metrics, ratios, peer comparisons) is the SAME as before.
You already analyzed it in loop 1. Focus on ADDING qualitative evidence to address gaps.
"""

        return revision_prompt

    def _parse_analysis_response(self,
                                response: str,
                                analysis_input: AnalysisInput,
                                captured_tool_calls: List[Dict[str, Any]] = None) -> ResearchReportOutput:
        """Parse agent response into ResearchReportOutput structure."""

        if captured_tool_calls is None:
            captured_tool_calls = []

        try:
            self.logger.info("DEBUG_TRACE: Starting to parse analyst response")
            self.logger.info(f"DEBUG_TRACE: Response length: {len(response)}")
            self.logger.info(f"DEBUG_TRACE: Captured tool calls: {len(captured_tool_calls)}")

            # Try to parse as JSON - handle both direct JSON and markdown-wrapped JSON
            json_content = response.strip()

            # More robust extraction of JSON from markdown code blocks
            import re

            # Try multiple patterns for markdown code blocks
            patterns = [
                r'```json\s*\n(.*?)\n```',  # ```json\n{...}\n```
                r'```json\s*(.*?)```',       # ```json{...}```
                r'```\s*\n(.*?)\n```',       # ```\n{...}\n```
                r'```(.*?)```'               # ```{...}```
            ]

            extracted = False
            for pattern in patterns:
                match = re.search(pattern, json_content, re.DOTALL)
                if match:
                    json_content = match.group(1).strip()
                    self.logger.info(f"DEBUG_TRACE: Extracted JSON using pattern: {pattern}")
                    extracted = True
                    break

            if not extracted:
                self.logger.info("DEBUG_TRACE: No markdown code blocks found, using raw response")

            self.logger.info(f"DEBUG_TRACE: After extraction, starts with JSON: {json_content.startswith('{')}")
            self.logger.info(f"DEBUG_TRACE: First 100 chars: {json_content[:100]}")

            if json_content.startswith('{'):
                self.logger.info("DEBUG_TRACE: Attempting JSON parsing")
                try:
                    parsed = json.loads(json_content)
                    self.logger.info(f"DEBUG_TRACE: JSON parsed successfully, keys: {list(parsed.keys())}")
                    self.logger.info(f"DEBUG_TRACE: analysis_bullets count: {len(parsed.get('analysis_bullets', []))}")
                    self.logger.info(f"DEBUG_TRACE: tool_calls_made type: {type(parsed.get('tool_calls_made'))}, value: {parsed.get('tool_calls_made')}")

                    result = self._create_research_report_from_dict(parsed, analysis_input, captured_tool_calls)
                    bullets_count = len(result.analysis_bullets) if result.analysis_bullets else 0
                    self.logger.info(f"DEBUG_TRACE: Created ResearchReportOutput with {bullets_count} analysis bullets, {len(result.tool_calls_made)} tool calls")
                    return result
                except json.JSONDecodeError as je:
                    self.logger.error(f"DEBUG_TRACE: JSON parse error: {je}")
                    self.logger.error(f"DEBUG_TRACE: Failed content: {json_content[:500]}")

                    # Attempt partial JSON recovery
                    self.logger.info("DEBUG_TRACE: Attempting partial JSON recovery...")
                    repaired_json = self._attempt_json_repair(json_content, je)

                    if repaired_json:
                        try:
                            parsed = json.loads(repaired_json)
                            self.logger.info("DEBUG_TRACE: Successfully recovered partial JSON!")
                            result = self._create_research_report_from_dict(parsed, analysis_input, captured_tool_calls)
                            bullets_count = len(result.analysis_bullets) if result.analysis_bullets else 0
                            self.logger.info(f"DEBUG_TRACE: Recovered analysis with {bullets_count} analysis bullets")
                            return result
                        except json.JSONDecodeError as repair_error:
                            self.logger.warning(f"DEBUG_TRACE: JSON repair failed: {repair_error}")

                    # Fall through to text extraction if repair also fails

            # If not JSON, we must fail - we require ResearchReportOutput
            self.logger.error("DEBUG_TRACE: No valid JSON found in response")
            raise ValueError("Agent response did not contain valid JSON for ResearchReportOutput")

        except Exception as e:
            self.logger.error(f"DEBUG_TRACE: Parse failed with exception: {e}")
            self.logger.error(f"Failed to parse response: {e}")
            raise ValueError(f"Parse error - could not create ResearchReportOutput: {e}")

    def _attempt_json_repair(self, json_content: str, error: json.JSONDecodeError) -> Optional[str]:
        """
        Attempt to repair truncated JSON by finding the last complete field
        and properly closing the JSON structure.

        Args:
            json_content: The truncated JSON string
            error: The JSONDecodeError that occurred

        Returns:
            Repaired JSON string or None if repair not possible
        """
        try:
            # Get the position where parsing failed
            error_pos = error.pos if hasattr(error, 'pos') else len(json_content)

            self.logger.debug(f"JSON error at position {error_pos} of {len(json_content)}")

            # Strategy 1: Find the last complete field by looking for the last comma or closing bracket
            # before the error position
            truncate_at = error_pos

            # Look backwards from error position to find safe truncation point
            for i in range(error_pos - 1, max(0, error_pos - 500), -1):
                char = json_content[i]

                # Found a comma or closing bracket - this is a safe truncation point
                if char == ',' or char == '}' or char == ']':
                    truncate_at = i + 1
                    break

                # If we find an opening bracket/brace, we might be in the middle of a structure
                if char == '{' or char == '[':
                    # Try to include up to the previous comma
                    for j in range(i - 1, max(0, i - 200), -1):
                        if json_content[j] == ',':
                            truncate_at = j
                            break
                    break

            # Truncate the JSON at the safe point
            partial_json = json_content[:truncate_at].rstrip()

            # Remove trailing commas (invalid JSON)
            while partial_json.endswith(','):
                partial_json = partial_json[:-1].rstrip()

            # Remove incomplete string literals (ending with unterminated quotes)
            # Count quotes to see if we're in an unterminated string
            quote_count = partial_json.count('"') - partial_json.count('\\"')
            if quote_count % 2 == 1:
                # Odd number of quotes means unterminated string
                # Find the last quote and truncate before it
                last_quote_pos = partial_json.rfind('"')
                if last_quote_pos > 0:
                    # Look backwards for the previous comma or colon
                    for i in range(last_quote_pos - 1, max(0, last_quote_pos - 200), -1):
                        if partial_json[i] in [',', ':']:
                            partial_json = partial_json[:i].rstrip()
                            if partial_json.endswith(','):
                                partial_json = partial_json[:-1].rstrip()
                            break

            # Now close all open structures
            # Count unclosed braces and brackets
            open_braces = partial_json.count('{') - partial_json.count('}')
            open_brackets = partial_json.count('[') - partial_json.count(']')

            self.logger.debug(f"Truncated at pos {truncate_at}, need to close {open_braces} braces, {open_brackets} brackets")

            # Close brackets first (inner structures), then braces
            repaired = partial_json
            repaired += ']' * open_brackets
            repaired += '}' * open_braces

            # Validate the repaired JSON has correct structure
            if not repaired.strip().startswith('{'):
                self.logger.warning("Repaired JSON doesn't start with '{'")
                return None

            if not repaired.strip().endswith('}'):
                self.logger.warning("Repaired JSON doesn't end with '}'")
                return None

            self.logger.info(f"JSON repaired: truncated from {len(json_content)} to {len(partial_json)}, added {open_brackets} ']' and {open_braces} '}}'")
            self.logger.debug(f"Repaired JSON preview: ...{repaired[-200:]}")

            return repaired

        except Exception as repair_error:
            self.logger.error(f"JSON repair attempt failed: {repair_error}")
            return None

    def _try_create_research_report(self,
                                    data: Dict[str, Any],
                                    analysis_input: AnalysisInput,
                                    captured_tool_calls: List[Dict[str, Any]] = None):
        """
        Attempt to create ResearchReportOutput from parsed dictionary.
        Returns ResearchReportOutput if successful, None if not enough data for research format.
        """
        from config.schemas import (
            ResearchReportOutput, ChartableRatio, ChartableMetric,
            AnalysisBullet, DetailedSource, repair_incomplete_research_report,
            validate_research_report_output
        )

        try:
            # First, check if we have the required research report fields
            if 'chartable_ratios' not in data or 'chartable_metrics' not in data or 'analysis_bullets' not in data:
                self.logger.error("LLM output doesn't have required research report structure (missing chartable_ratios, chartable_metrics, or analysis_bullets)")
                return None

            # Parse chartable ratios
            chartable_ratios = []
            for ratio_data in data.get('chartable_ratios', []):
                if isinstance(ratio_data, dict):
                    # Handle wrong field names: "name" -> "ratio_name", array "values" -> dict "company_values"
                    ratio_name = ratio_data.get('ratio_name') or ratio_data.get('name', 'Unknown Ratio')

                    # Fix company_values if provided as array instead of dict
                    company_values = ratio_data.get('company_values', {})
                    if not company_values and 'values' in ratio_data:
                        # Convert array format to dict format
                        vals = ratio_data['values']
                        if isinstance(vals, list) and len(vals) > 0:
                            # Create dict with Q1, Q2, Q3... keys
                            company_values = {f"Q{i+1}": float(v) if isinstance(v, (int, float, str)) else v
                                            for i, v in enumerate(vals) if v is not None}

                    # Normalize trend_direction for ChartableRatio (LLM might use metric terms)
                    trend = ratio_data.get('trend_direction', 'insufficient_data')
                    # Map metric terminology to ratio terminology
                    trend_mapping = {
                        'increasing': 'improving',  # LLM confusion
                        'decreasing': 'declining',  # LLM confusion
                        'improving': 'improving',
                        'declining': 'declining',
                        'stable': 'stable',
                        'volatile': 'volatile',
                        'insufficient_data': 'insufficient_data'
                    }
                    normalized_trend = trend_mapping.get(trend, 'insufficient_data')

                    chartable_ratios.append(ChartableRatio(
                        ratio_name=ratio_name,
                        company_values=company_values,
                        peer_values=ratio_data.get('peer_values', {}),
                        interpretation=ratio_data.get('interpretation', ''),
                        trend_direction=normalized_trend
                    ))

            # Parse chartable metrics
            chartable_metrics = []
            for metric_data in data.get('chartable_metrics', []):
                if isinstance(metric_data, dict):
                    # Handle wrong field name: "name" -> "metric_name"
                    metric_name = metric_data.get('metric_name') or metric_data.get('name', 'Unknown Metric')

                    # values can be dict or array, convert array to dict if needed
                    values = metric_data.get('values', {})
                    if isinstance(values, list):
                        # Convert array to dict with Q1, Q2, Q3... keys
                        values = {f"Q{i+1}": float(v) if isinstance(v, (int, float, str)) else v
                                for i, v in enumerate(values) if v is not None}

                    # Normalize trend_direction for ChartableMetric (LLM might use ratio terms)
                    trend = metric_data.get('trend_direction', 'insufficient_data')
                    # Map ratio terminology to metric terminology
                    trend_mapping = {
                        'improving': 'increasing',  # LLM confusion
                        'declining': 'decreasing',  # LLM confusion
                        'increasing': 'increasing',
                        'decreasing': 'decreasing',
                        'stable': 'stable',
                        'volatile': 'volatile',
                        'insufficient_data': 'insufficient_data'
                    }
                    normalized_trend = trend_mapping.get(trend, 'insufficient_data')

                    chartable_metrics.append(ChartableMetric(
                        metric_name=metric_name,
                        values=values,
                        unit=metric_data.get('unit', 'units'),
                        trend_direction=normalized_trend,
                        interpretation=metric_data.get('interpretation', '')
                    ))

            # Parse analysis bullets
            analysis_bullets = []
            for bullet_data in data.get('analysis_bullets', []):
                if isinstance(bullet_data, dict):
                    # Parse DetailedSource objects from qualitative_sources
                    qualitative_sources = []
                    for source_data in bullet_data.get('qualitative_sources', []):
                        if isinstance(source_data, dict):
                            qualitative_sources.append(DetailedSource(
                                tool_name=source_data.get('tool_name', 'unknown'),
                                period=source_data.get('period', ''),
                                speaker_name=source_data.get('speaker_name'),
                                report_type=source_data.get('report_type'),
                                chunk_summary=source_data.get('chunk_summary'),
                                relevance_score=source_data.get('relevance_score')
                            ))

                    bullet_text = bullet_data.get('bullet_text', '')

                    # Enforce minimum bullet length (200 chars)
                    if len(bullet_text) < 200:
                        self.logger.warning(f"Bullet text too short ({len(bullet_text)} chars < 200), padding with note")
                        bullet_text += f" [Note: This analysis point needs more depth to meet research-grade standards of 200+ characters for comprehensive insights.]"

                    analysis_bullets.append(AnalysisBullet(
                        bullet_text=bullet_text,
                        quantitative_evidence=bullet_data.get('quantitative_evidence', []),
                        qualitative_sources=qualitative_sources,
                        importance=bullet_data.get('importance', 'moderate')
                    ))

            # Create ResearchReportOutput
            report = ResearchReportOutput(
                run_id=analysis_input.run_id,
                domain=analysis_input.domain,
                company=analysis_input.company,
                ticker=analysis_input.ticker,
                analysis_timestamp=datetime.now().isoformat(),
                executive_summary=data.get('executive_summary', ''),
                chartable_ratios=chartable_ratios,
                chartable_metrics=chartable_metrics,
                analysis_bullets=analysis_bullets,
                recommendations=data.get('recommendations', []),
                risk_factors=data.get('risk_factors', []),
                confidence_level=data.get('confidence_level', 'medium'),
                tool_calls_made=captured_tool_calls or data.get('tool_calls_made', []),
                data_sources_summary=data.get('data_sources_summary', 'Various financial data sources')
            )

            self.logger.info(f"Successfully created ResearchReportOutput with {len(chartable_ratios)} ratios, {len(chartable_metrics)} metrics, {len(analysis_bullets)} bullets")
            return report

        except ValueError as ve:
            # Validation error - attempt repair
            self.logger.warning(f"ResearchReportOutput validation failed: {ve}")
            self.logger.info("Attempting to repair incomplete research report...")

            try:
                # Build dict from what we have so far
                incomplete_dict = {
                    'run_id': analysis_input.run_id,
                    'domain': analysis_input.domain,
                    'company': analysis_input.company,
                    'ticker': analysis_input.ticker,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'executive_summary': data.get('executive_summary', ''),
                    'chartable_ratios': data.get('chartable_ratios', []),
                    'chartable_metrics': data.get('chartable_metrics', []),
                    'analysis_bullets': data.get('analysis_bullets', []),
                    'recommendations': data.get('recommendations', []),
                    'risk_factors': data.get('risk_factors', []),
                    'confidence_level': data.get('confidence_level', 'low'),
                    'tool_calls_made': captured_tool_calls or [],
                    'data_sources_summary': data.get('data_sources_summary', 'Incomplete data')
                }

                # Attempt repair
                repaired_dict = repair_incomplete_research_report(incomplete_dict, analysis_input)

                # Validate repaired version
                issues = validate_research_report_output(repaired_dict)
                if issues:
                    self.logger.warning(f"Repaired report still has issues: {issues}")
                    return None

                self.logger.info("Successfully repaired incomplete research report! Attempting to create ResearchReportOutput...")

                # Try to create ResearchReportOutput from repaired dict
                try:
                    repaired_report = self._try_create_research_report(repaired_dict, analysis_input, captured_tool_calls)
                    if repaired_report is not None:
                        self.logger.info("Successfully created ResearchReportOutput from repaired data!")
                        return repaired_report
                    else:
                        self.logger.warning("Repaired data still insufficient for ResearchReportOutput")
                        return None
                except Exception as instantiation_error:
                    self.logger.error(f"Failed to instantiate ResearchReportOutput from repaired data: {instantiation_error}")
                    return None

            except Exception as repair_error:
                self.logger.error(f"Research report repair failed: {repair_error}")
                return None

        except Exception as e:
            self.logger.warning(f"Failed to create ResearchReportOutput: {e}")
            return None

    def _create_research_report_from_dict(self,
                                         data: Dict[str, Any],
                                         analysis_input: AnalysisInput,
                                         captured_tool_calls: List[Dict[str, Any]] = None) -> ResearchReportOutput:
        """Create ResearchReportOutput from parsed dictionary."""

        if captured_tool_calls is None:
            captured_tool_calls = []

        # Create ResearchReportOutput - this is the only format we support
        research_report = self._try_create_research_report(data, analysis_input, captured_tool_calls)
        if research_report is not None:
            self.logger.info("Successfully created ResearchReportOutput")
            return research_report

        # If we can't create ResearchReportOutput, we must fail
        self.logger.error("Failed to create ResearchReportOutput from parsed data")
        self.logger.error(f"Data keys: {list(data.keys())}")
        raise ValueError("Could not create ResearchReportOutput - missing required fields (chartable_ratios, chartable_metrics, or analysis_bullets)")