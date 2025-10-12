"""
Compact Schemas for AgentSystemV2 - Token Optimized
"""

from __future__ import annotations  # Enable forward references for type hints

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class AnalysisInput:
    """Input data for domain analysis - pre-calculated metrics supplied upfront."""

    run_id: str
    company: str
    ticker: str
    domain: str
    time_period: str

    # Pre-calculated financial data
    supplied_metrics: Dict[str, Dict[str, float]]  # metric_name -> {quarter: value}
    supplied_ratios: Dict[str, Dict[str, float]]   # ratio_name -> {quarter: value}
    peer_comparison: Dict[str, Any]                # peer analysis data

    # Analysis parameters
    user_focus: str
    persona: str = "institutional_investor"  # Target audience: banker, cfo, trader, institutional_investor, retail_investor
    max_tool_calls: int = 6
    max_loops: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class ReviewResult:
    """Result from reviewer agent evaluation."""
    
    verdict: str  # "approved", "needs_revision", "rejected"
    quality_score: float  # 0.0 to 1.0
    
    # Feedback for improvement
    strengths: List[str]
    weaknesses: List[str]
    specific_improvements: List[str]
    
    # Loop control
    should_continue: bool
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatRefinementRequest:
    """User request for analysis refinement via chat."""

    original_analysis: ResearchReportOutput
    user_question: str
    refinement_type: str  # "clarification", "deeper_analysis", "alternative_view"
    max_loops: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MasterAnalysisRequest:
    """Request for master agent to coordinate multiple domains."""
    
    company: str
    ticker: str
    domains: List[str]
    user_focus: str = "comprehensive analysis"
    time_period: str = "3 years"
    
    # Coordination settings
    run_parallel: bool = True
    max_cost_per_domain: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DomainAgentConfig:
    """Configuration for a domain-specific agent."""

    domain: str
    required_metrics: List[str]
    required_ratios: List[str]
    peer_comparison_required: bool
    default_max_loops: int
    analyst_prompt_template: str
    reviewer_criteria: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
#                 RESEARCH-GRADE REPORT SCHEMAS
# ═══════════════════════════════════════════════════════════════

@dataclass
class DetailedSource:
    """Source attribution for qualitative insights - tracks where information came from."""

    tool_name: str  # e.g., "search_transcripts", "search_annual_reports"
    period: str  # e.g., "Q3-2024", "FY-2023"
    speaker_name: Optional[str] = None  # For transcripts - CEO name, CFO name, etc.
    report_type: Optional[str] = None  # For annual reports - "10-K", "10-Q"
    chunk_summary: Optional[str] = None  # Brief summary of the relevant chunk (~100 chars)
    relevance_score: Optional[float] = None  # How relevant this source is (0.0-1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_citation(self) -> str:
        """Format as inline citation."""
        parts = [self.tool_name, self.period]
        if self.speaker_name:
            parts.append(f"Speaker: {self.speaker_name}")
        if self.report_type:
            parts.append(self.report_type)
        return f"[{', '.join(parts)}]"


@dataclass
class ChartableRatio:
    """Financial ratio data formatted for charting - shows company vs peers over time."""

    ratio_name: str  # e.g., "Current Ratio", "Quick Ratio"
    company_values: Dict[str, float]  # {quarter: value} e.g., {"Q1-2024": 1.5, "Q2-2024": 1.6}
    peer_values: Dict[str, Dict[str, float]]  # {peer_ticker: {quarter: value}}
    interpretation: str  # What this ratio tells us about the company
    trend_direction: str  # "improving", "declining", "stable", "volatile"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self):
        """Validate ratio data."""
        if not self.company_values:
            raise ValueError(f"ChartableRatio '{self.ratio_name}' must have company values")
        if self.trend_direction not in ["improving", "declining", "stable", "volatile", "insufficient_data"]:
            raise ValueError(f"Invalid trend_direction: {self.trend_direction}")


@dataclass
class ChartableMetric:
    """Financial metric data formatted for charting - shows company performance over time."""

    metric_name: str  # e.g., "Revenue", "Net Income", "Operating Cash Flow"
    values: Dict[str, float]  # {quarter: value} e.g., {"Q1-2024": 95000000000, "Q2-2024": 98000000000}
    unit: str  # e.g., "USD", "USD millions", "shares"
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    interpretation: str  # What this metric trend tells us

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self):
        """Validate metric data."""
        if not self.values:
            raise ValueError(f"ChartableMetric '{self.metric_name}' must have values")
        if self.trend_direction not in ["increasing", "decreasing", "stable", "volatile", "insufficient_data"]:
            raise ValueError(f"Invalid trend_direction: {self.trend_direction}")


@dataclass
class AnalysisBullet:
    """Single analysis point integrating quantitative and qualitative evidence."""

    bullet_text: str  # The main analysis statement (2-4 sentences)
    quantitative_evidence: List[str]  # Numbers/ratios from supplied data (NOT from tools)
    qualitative_sources: List[DetailedSource]  # Sources supporting the narrative
    importance: str  # "critical", "high", "moderate"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert DetailedSource objects to dicts
        result['qualitative_sources'] = [s.to_dict() if hasattr(s, 'to_dict') else s
                                         for s in self.qualitative_sources]
        return result

    def __post_init__(self):
        """Validate bullet structure."""
        if len(self.bullet_text) < 200:
            raise ValueError("Analysis bullet must be at least 200 characters for research-grade depth")
        if self.importance not in ["critical", "high", "moderate", "low"]:
            raise ValueError(f"Invalid importance: {self.importance}")
        if not self.qualitative_sources:
            raise ValueError("Analysis bullet must have at least one qualitative source")


@dataclass
class ResearchReportOutput:
    """Research-grade financial analysis report - comprehensive structured output."""

    # Metadata
    run_id: str
    domain: str
    company: str
    ticker: str
    analysis_timestamp: str

    # Executive summary
    executive_summary: str  # 3-5 sentences capturing key takeaways

    # Chart-ready data (2-4 of each)
    chartable_ratios: List[ChartableRatio]  # 2-4 key ratios over time vs peers
    chartable_metrics: List[ChartableMetric]  # 2-4 key financial metrics over time

    # Structured analysis (3-10 bullets)
    analysis_bullets: List[AnalysisBullet]  # Each integrates quant + qual evidence

    # Recommendations and risks
    recommendations: List[str]  # 2-5 specific, actionable recommendations
    risk_factors: List[str]  # 2-5 key risks identified

    # Metadata
    confidence_level: str  # "high", "medium", "low"
    tool_calls_made: List[Dict[str, Any]]  # Track tool usage
    data_sources_summary: str  # Brief description of data sources used
    ratio_definitions: Optional[Dict[str, Dict[str, Any]]] = None  # Ratio definitions for charting (added post-analysis)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert nested dataclass objects to dicts
        result['chartable_ratios'] = [r.to_dict() if hasattr(r, 'to_dict') else r
                                      for r in self.chartable_ratios]
        result['chartable_metrics'] = [m.to_dict() if hasattr(m, 'to_dict') else m
                                       for m in self.chartable_metrics]
        result['analysis_bullets'] = [b.to_dict() if hasattr(b, 'to_dict') else b
                                      for b in self.analysis_bullets]
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def __post_init__(self):
        """Validate research report structure."""
        errors = []

        # Validate chart data counts (2-4 each)
        if not (2 <= len(self.chartable_ratios) <= 4):
            errors.append(f"Must have 2-4 chartable_ratios, got {len(self.chartable_ratios)}")
        if not (2 <= len(self.chartable_metrics) <= 4):
            errors.append(f"Must have 2-4 chartable_metrics, got {len(self.chartable_metrics)}")

        # Validate analysis bullets count (3-10)
        if not (3 <= len(self.analysis_bullets) <= 10):
            errors.append(f"Must have 3-10 analysis_bullets, got {len(self.analysis_bullets)}")

        # Validate recommendations and risks
        if not (2 <= len(self.recommendations) <= 5):
            errors.append(f"Should have 2-5 recommendations, got {len(self.recommendations)}")
        if not (2 <= len(self.risk_factors) <= 5):
            errors.append(f"Should have 2-5 risk_factors, got {len(self.risk_factors)}")

        # Validate confidence level
        if self.confidence_level not in ["high", "medium", "low"]:
            errors.append(f"Invalid confidence_level: {self.confidence_level}")

        # Validate executive summary length
        if len(self.executive_summary) < 100:
            errors.append("Executive summary must be at least 100 characters")

        if errors:
            raise ValueError(f"ResearchReportOutput validation failed: {'; '.join(errors)}")


# Schema reference strings for token optimization
ANALYSIS_INPUT_REF = "AnalysisInput{run_id,company,ticker,domain,time_period,supplied_metrics,supplied_ratios,peer_comparison,user_focus,max_tool_calls,max_loops}"

REVIEW_RESULT_REF = "ReviewResult{verdict,quality_score,strengths,weaknesses,specific_improvements,should_continue,reason}"

# Research-grade report schema references
DETAILED_SOURCE_REF = "DetailedSource{tool_name,period,speaker_name,report_type,chunk_summary,relevance_score}"

CHARTABLE_RATIO_REF = "ChartableRatio{ratio_name,company_values,peer_values,interpretation,trend_direction}"

CHARTABLE_METRIC_REF = "ChartableMetric{metric_name,values,unit,trend_direction,interpretation}"

ANALYSIS_BULLET_REF = "AnalysisBullet{bullet_text,quantitative_evidence,qualitative_sources,importance}"

RESEARCH_REPORT_OUTPUT_REF = "ResearchReportOutput{run_id,domain,company,ticker,analysis_timestamp,executive_summary,chartable_ratios[2-4],chartable_metrics[2-4],analysis_bullets[3-10],recommendations,risk_factors,confidence_level,tool_calls_made,data_sources_summary}"


def get_schema_reference(schema_name: str) -> str:
    """Get compact schema reference for prompt optimization."""
    schema_refs = {
        "AnalysisInput": ANALYSIS_INPUT_REF,
        "ReviewResult": REVIEW_RESULT_REF,
        "DetailedSource": DETAILED_SOURCE_REF,
        "ChartableRatio": CHARTABLE_RATIO_REF,
        "ChartableMetric": CHARTABLE_METRIC_REF,
        "AnalysisBullet": ANALYSIS_BULLET_REF,
        "ResearchReportOutput": RESEARCH_REPORT_OUTPUT_REF
    }
    return schema_refs.get(schema_name, schema_name)


def validate_analysis_output(output: Dict[str, Any]) -> List[str]:
    """Validate analysis output structure and return any issues."""
    issues = []

    required_fields = ["run_id", "domain", "company", "executive_summary", "key_findings", "confidence_level"]
    for field in required_fields:
        if field not in output:
            issues.append(f"Missing required field: {field}")

    # Validate confidence level
    if "confidence_level" in output:
        valid_confidence = ["high", "medium", "low"]
        if output["confidence_level"] not in valid_confidence:
            issues.append(f"Invalid confidence_level: {output['confidence_level']}. Must be one of {valid_confidence}")

    # Validate key findings
    if "key_findings" in output:
        if not isinstance(output["key_findings"], list) or len(output["key_findings"]) == 0:
            issues.append("key_findings must be a non-empty list")

    # Validate executive summary
    if "executive_summary" in output:
        if not isinstance(output["executive_summary"], str) or len(output["executive_summary"]) < 50:
            issues.append("executive_summary must be a string with at least 50 characters")

    return issues


def validate_research_report_output(output: Dict[str, Any]) -> List[str]:
    """Validate research report output structure and return any issues."""
    issues = []

    # Required fields
    required_fields = [
        "run_id", "domain", "company", "ticker", "analysis_timestamp",
        "executive_summary", "chartable_ratios", "chartable_metrics",
        "analysis_bullets", "recommendations", "risk_factors",
        "confidence_level", "data_sources_summary"
    ]

    for field in required_fields:
        if field not in output:
            issues.append(f"Missing required field: {field}")

    # Validate chartable_ratios (2-4)
    if "chartable_ratios" in output:
        if not isinstance(output["chartable_ratios"], list):
            issues.append("chartable_ratios must be a list")
        elif not (2 <= len(output["chartable_ratios"]) <= 4):
            issues.append(f"Must have 2-4 chartable_ratios, got {len(output['chartable_ratios'])}")

    # Validate chartable_metrics (2-4)
    if "chartable_metrics" in output:
        if not isinstance(output["chartable_metrics"], list):
            issues.append("chartable_metrics must be a list")
        elif not (2 <= len(output["chartable_metrics"]) <= 4):
            issues.append(f"Must have 2-4 chartable_metrics, got {len(output['chartable_metrics'])}")

    # Validate analysis_bullets (3-10)
    if "analysis_bullets" in output:
        if not isinstance(output["analysis_bullets"], list):
            issues.append("analysis_bullets must be a list")
        elif not (3 <= len(output["analysis_bullets"]) <= 10):
            issues.append(f"Must have 3-10 analysis_bullets, got {len(output['analysis_bullets'])}")

    # Validate recommendations (2-5)
    if "recommendations" in output:
        if not isinstance(output["recommendations"], list):
            issues.append("recommendations must be a list")
        elif not (2 <= len(output["recommendations"]) <= 5):
            issues.append(f"Should have 2-5 recommendations, got {len(output['recommendations'])}")

    # Validate risk_factors (2-5)
    if "risk_factors" in output:
        if not isinstance(output["risk_factors"], list):
            issues.append("risk_factors must be a list")
        elif not (2 <= len(output["risk_factors"]) <= 5):
            issues.append(f"Should have 2-5 risk_factors, got {len(output['risk_factors'])}")

    # Validate confidence level
    if "confidence_level" in output:
        valid_confidence = ["high", "medium", "low"]
        if output["confidence_level"] not in valid_confidence:
            issues.append(f"Invalid confidence_level: {output['confidence_level']}. Must be one of {valid_confidence}")

    # Validate executive summary
    if "executive_summary" in output:
        if not isinstance(output["executive_summary"], str) or len(output["executive_summary"]) < 100:
            issues.append("executive_summary must be a string with at least 100 characters")

    return issues


def repair_incomplete_research_report(output: Dict[str, Any], analysis_input: AnalysisInput) -> Dict[str, Any]:
    """
    Repair incomplete research report by filling in missing fields with reasonable defaults.
    Used for graceful degradation when LLM output is incomplete.
    """
    repaired = output.copy()

    # Fill in required metadata
    if "run_id" not in repaired:
        repaired["run_id"] = analysis_input.run_id
    if "domain" not in repaired:
        repaired["domain"] = analysis_input.domain
    if "company" not in repaired:
        repaired["company"] = analysis_input.company
    if "ticker" not in repaired:
        repaired["ticker"] = analysis_input.ticker
    if "analysis_timestamp" not in repaired:
        repaired["analysis_timestamp"] = datetime.now().isoformat()

    # Fill in missing analysis fields with minimal defaults
    if "executive_summary" not in repaired or len(repaired.get("executive_summary", "")) < 100:
        repaired["executive_summary"] = (
            f"Analysis of {analysis_input.company} ({analysis_input.ticker}) in the {analysis_input.domain} domain. "
            f"This analysis is incomplete and requires revision. Please review and refine the findings."
        )

    if "chartable_ratios" not in repaired or not isinstance(repaired["chartable_ratios"], list):
        repaired["chartable_ratios"] = []

    if "chartable_metrics" not in repaired or not isinstance(repaired["chartable_metrics"], list):
        repaired["chartable_metrics"] = []

    if "analysis_bullets" not in repaired or not isinstance(repaired["analysis_bullets"], list):
        repaired["analysis_bullets"] = []

    if "recommendations" not in repaired or not isinstance(repaired["recommendations"], list):
        repaired["recommendations"] = ["Requires further analysis"]

    if "risk_factors" not in repaired or not isinstance(repaired["risk_factors"], list):
        repaired["risk_factors"] = ["Incomplete analysis - data gaps present"]

    if "confidence_level" not in repaired:
        repaired["confidence_level"] = "low"

    if "data_sources_summary" not in repaired:
        repaired["data_sources_summary"] = "Analysis incomplete"

    if "tool_calls_made" not in repaired:
        repaired["tool_calls_made"] = []

    return repaired


def get_persona_guidance(persona: str) -> str:
    """
    Get audience-specific guidance for tailoring analysis language and depth.

    Args:
        persona: Target audience type (banker, cfo, trader, institutional_investor, retail_investor)

    Returns:
        String with persona-specific instructions
    """
    persona_map = {
        "banker": """
AUDIENCE: Commercial/Investment Banker
FOCUS AREAS:
- Credit risk assessment and debt capacity
- Covenant compliance and financial covenants
- Cash flow stability and debt service capability
- Collateral value and asset quality
- Refinancing risk and maturity profile

LANGUAGE STYLE:
- Technical and formal
- Emphasize risk factors and downside scenarios
- Quantify credit metrics (Interest Coverage, Debt/EBITDA, DSCR)
- Reference lending standards and industry benchmarks
- Highlight red flags and warning signs

DEPTH: Deep analysis of financial stability and credit worthiness""",

        "cfo": """
AUDIENCE: Chief Financial Officer / Finance Executive
FOCUS AREAS:
- Operational implications and actionable insights
- Comparison to industry benchmarks
- Strategic decision-making implications
- Board presentation readiness
- Forward-looking guidance and scenarios

LANGUAGE STYLE:
- Executive-level, strategic tone
- Balance quantitative rigor with strategic narrative
- Emphasize competitive positioning
- Frame recommendations in terms of strategic options
- Highlight both opportunities and risks

DEPTH: Strategic overview with supporting details for operational decisions""",

        "trader": """
AUDIENCE: Equity/Credit Trader
FOCUS AREAS:
- Near-term catalysts and trading signals
- Volatility drivers and market sentiment
- Relative value vs peers
- Technical levels and price action
- Earnings surprise potential

LANGUAGE STYLE:
- Concise and actionable
- Focus on what's changed recently
- Identify trading opportunities (long/short thesis)
- Highlight catalyst timeline (days/weeks ahead)
- Reference market consensus and surprises

DEPTH: High-level key takeaways with immediate market implications""",

        "institutional_investor": """
AUDIENCE: Institutional Investor (Asset Manager, Pension Fund, Endowment)
FOCUS AREAS:
- Long-term fundamental strength
- Competitive moat and sustainability
- Capital allocation strategy
- ESG considerations and governance
- Risk-adjusted return potential

LANGUAGE STYLE:
- Professional and comprehensive
- Balance quantitative analysis with qualitative factors
- Multi-year perspective (not trading-oriented)
- Benchmark to sector and market
- Emphasize durability of competitive advantages

DEPTH: Comprehensive research-grade analysis supporting investment thesis""",

        "retail_investor": """
AUDIENCE: Individual/Retail Investor
FOCUS AREAS:
- Investment thesis in plain language
- Key strengths and weaknesses clearly explained
- Realistic risk assessment
- Simple valuation metrics (P/E, dividend yield)
- Clear buy/hold/sell recommendation

LANGUAGE STYLE:
- Clear and accessible (avoid excessive jargon)
- Explain financial concepts when used
- Use analogies and real-world examples
- Focus on big picture, not minutiae
- Honest about uncertainties

DEPTH: Accessible analysis focusing on key investment drivers"""
    }

    # Default to institutional investor if persona not recognized
    return persona_map.get(persona, persona_map["institutional_investor"])


def create_quality_evaluation_prompt(analysis: ResearchReportOutput, input_data: AnalysisInput) -> str:
    """Create evaluation prompt for reviewer - expects ResearchReportOutput schema."""

    # Extract findings from analysis_bullets
    findings_list = [bullet.bullet_text for bullet in analysis.analysis_bullets] if analysis.analysis_bullets else []
    findings_count = len(findings_list)

    # Format findings (show more detail for reviewer)
    findings_text = "\n".join([f"{i+1}. {f}" for i, f in enumerate(findings_list)]) if findings_list else "(No findings)"

    # Format tool calls
    tool_calls = analysis.tool_calls_made if analysis.tool_calls_made else []
    tool_text = "\n".join([f"  - {c.get('function', c if isinstance(c, str) else 'unknown')}"
                           for c in tool_calls[:10]]) if tool_calls else "(None)"

    # Get qualitative evidence count from analysis_bullets
    qual_count = sum(len(b.qualitative_sources) for b in analysis.analysis_bullets) if analysis.analysis_bullets else 0

    # Format recommendations
    recs_text = "\n".join([f"  {i+1}. {r}" for i, r in enumerate(analysis.recommendations[:5])]) if analysis.recommendations else "(None)"

    prompt = f"""
═══════════════════════════════════════════════════════════════
                    🎯 USER'S ORIGINAL QUERY
═══════════════════════════════════════════════════════════════

COMPANY: {input_data.company} ({input_data.ticker})
DOMAIN: {input_data.domain}
USER FOCUS: {input_data.user_focus}
TIME PERIOD: {input_data.time_period}
TARGET AUDIENCE: {input_data.persona}

⚠️ CRITICAL: Does the analysis actually ANSWER this user query?
   Is it relevant, insightful, and actionable for this audience?

═══════════════════════════════════════════════════════════════
                    📊 ANALYSIS TO EVALUATE
═══════════════════════════════════════════════════════════════

CONFIDENCE LEVEL: {analysis.confidence_level}

EXECUTIVE SUMMARY:
{analysis.executive_summary}

ANALYSIS BULLETS ({findings_count}):
{findings_text}

RECOMMENDATIONS ({len(analysis.recommendations)}):
{recs_text}

RISK FACTORS: {len(analysis.risk_factors)} identified
TOOLS USED: {len(tool_calls)} tool calls | {qual_count} qualitative sources

═══════════════════════════════════════════════════════════════
              🔍 EVALUATION CRITERIA (Equal Weight)
═══════════════════════════════════════════════════════════════

A. CONTENT QUALITY (50% weight):
   1. Relevance: Does it directly address the user's focus query?
   2. Depth: Is the analysis deep and insightful, or superficial?
   3. Actionability: Can the {input_data.persona} make decisions from this?
   4. Completeness: Are all aspects of "{input_data.user_focus}" covered?
   5. Insight Quality: Does it provide non-obvious insights or just describe data?

B. TECHNICAL QUALITY (50% weight):
   6. Evidence: Claims backed by specific data from supplied metrics?
   7. Logic: Do conclusions logically follow from evidence?
   8. Data Discipline: Numbers from supplied_metrics/ratios (NOT tool outputs)?
   9. Source Attribution: Qualitative claims have proper sources?
   10. Risk Assessment: Comprehensive and relevant to the query?

═══════════════════════════════════════════════════════════════
                    ❓ KEY QUESTIONS TO ASK
═══════════════════════════════════════════════════════════════

RELEVANCE:
- If user asked about "cash strategy and working capital", does analysis cover this?
- If user wanted "comparison to Microsoft", is this comparison present and meaningful?
- Does it address the TIME PERIOD requested ({input_data.time_period})?

DEPTH:
- Are the insights surface-level ("cash increased") or deep ("cash increased due to working capital efficiency, evidenced by days payable outstanding improving from...")?
- Does it explain WHY trends occurred, not just WHAT happened?
- Are forward-looking implications discussed?

ACTIONABILITY:
- Can a {input_data.persona} use this analysis to make investment decisions?
- Are recommendations specific enough to act on?
- Does it provide clear investment implications?

═══════════════════════════════════════════════════════════════
                        📝 YOUR EVALUATION
═══════════════════════════════════════════════════════════════

Return JSON with:
- verdict: "approved" / "needs_revision" / "rejected"
- quality_score: 0.0-1.0 (consider BOTH content and technical quality)
- strengths: 2-3 specific strengths
- weaknesses: 2-3 specific weaknesses
- specific_improvements: 2-3 actionable improvement suggestions

SCORING GUIDANCE:
- 0.9-1.0: Answers query perfectly, deep insights, excellent evidence
- 0.8-0.9: Answers query well, good depth, solid evidence
- 0.7-0.8: Answers query adequately, moderate depth, acceptable evidence
- 0.6-0.7: Partially answers query, shallow analysis, weak evidence
- <0.6: Misses query focus, superficial, poor evidence
"""

    return prompt