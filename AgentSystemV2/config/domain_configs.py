"""
Domain Agent Configurations for AgentSystemV2
"""

from typing import Dict, List
from .schemas import DomainAgentConfig

# Liquidity Analysis Domain Configuration
LIQUIDITY_CONFIG = DomainAgentConfig(
    domain="liquidity",
    required_metrics=[
        "cash_and_short_term_investments",
        "current_assets", 
        "current_liabilities",
        "accounts_receivable",
        "inventory",
        "accounts_payable",
        "short_term_debt",
        "operating_cash_flow",
        "capital_expenditures"
    ],
    required_ratios=[
        "current_ratio",
        "quick_ratio", 
        "cash_ratio",
        "working_capital_turnover",
        "days_sales_outstanding",
        "days_inventory_outstanding",
        "days_payable_outstanding"
    ],
    peer_comparison_required=True,
    default_max_loops=3,
    analyst_prompt_template="""
DOMAIN: liquidity
COMPANY: {company}
ANALYSIS FOCUS: {user_focus}
TIME PERIOD: {time_period}
TARGET AUDIENCE: {persona}

SUPPLIED DATA SUMMARY:
- Financial metrics: {metrics_count} items
- Financial ratios: {ratios_count} items
- Peer comparison: {peer_count} companies

AUDIENCE-SPECIFIC GUIDANCE:
{persona_guidance}

Use schema: {schema_ref}
""",
    reviewer_criteria=[
        # Research report structure
        "Chart data selection: 2-4 chartable_ratios and 2-4 chartable_metrics selected",
        "Analysis bullets: 3-10 bullets, each integrating quantitative + qualitative evidence",
        "Executive summary: ≥100 chars, captures key takeaways",
        # Source attribution
        "Source attribution: Every qualitative claim has DetailedSource (tool, period, speaker/report)",
        "Tool-derived insights properly cited with metadata (quarter, speaker name for transcripts)",
        # Data discipline
        "Data discipline: All financial numbers from supplied data (NOT extracted from tools)",
        "Quantitative evidence references supplied_metrics and supplied_ratios only",
        # Domain-specific quality
        "Working capital trend analysis quality with root cause investigation",
        "Cash flow sustainability assessment with forward-looking perspective",
        "Peer comparison integration: selected ratios show company vs peers",
        # Tool usage strategy
        "Tool usage: Targeted and strategic (addresses specific questions, not blind calling)",
        "Tool calls aligned with identified gaps or investigative questions",
        # Recommendations and risks
        "Risk factor identification: 2-5 specific risks with evidence",
        "Recommendations: 2-5 actionable, specific recommendations tied to analysis",
        "Confidence level justified by evidence quality and completeness"
    ]
)

# Leverage Analysis Domain Configuration
LEVERAGE_CONFIG = DomainAgentConfig(
    domain="leverage",
    required_metrics=[
        "total_debt",
        "short_term_debt",
        "total_long_term_debt", 
        "total_equity",
        "total_assets",
        "interest_expense",
        "ebit",
        "ebitda",
        "operating_cash_flow"
    ],
    required_ratios=[
        "debt_to_equity",
        "debt_to_assets",
        "interest_coverage_ratio",
        "debt_service_coverage",
        "equity_ratio",
        "financial_leverage"
    ],
    peer_comparison_required=True,
    default_max_loops=3,
    analyst_prompt_template="""
DOMAIN: leverage
COMPANY: {company}
ANALYSIS FOCUS: {user_focus}
TIME PERIOD: {time_period}
TARGET AUDIENCE: {persona}

SUPPLIED DATA SUMMARY:
- Financial metrics: {metrics_count} items
- Financial ratios: {ratios_count} items
- Peer comparison: {peer_count} companies

AUDIENCE-SPECIFIC GUIDANCE:
{persona_guidance}

Use schema: {schema_ref}
""",
    reviewer_criteria=[
        # Research report structure
        "Chart data selection: 2-4 chartable_ratios and 2-4 chartable_metrics selected",
        "Analysis bullets: 3-10 bullets, each integrating quantitative + qualitative evidence",
        "Executive summary: ≥100 chars, captures key takeaways",
        # Source attribution
        "Source attribution: Every qualitative claim has DetailedSource (tool, period, speaker/report)",
        "Tool-derived insights properly cited with metadata",
        # Data discipline
        "Data discipline: All financial numbers from supplied data (NOT extracted from tools)",
        "Quantitative evidence references supplied_metrics and supplied_ratios only",
        # Domain-specific quality
        "Debt structure analysis completeness with maturity profile assessment",
        "Interest coverage assessment accuracy with trend analysis",
        "Peer leverage comparison: selected ratios show company vs industry peers",
        "Credit risk evaluation quality with downside scenarios",
        # Tool usage strategy
        "Tool usage: Targeted and strategic (not calling all tools blindly)",
        "Tool calls address specific debt strategy or refinancing questions",
        # Recommendations and risks
        "Risk factors: 2-5 specific leverage-related risks with evidence",
        "Recommendations: 2-5 actionable debt optimization strategies",
        "Debt capacity analysis depth and refinancing risk assessment"
    ]
)

# Working Capital Analysis Domain Configuration  
WORKING_CAPITAL_CONFIG = DomainAgentConfig(
    domain="working_capital",
    required_metrics=[
        "current_assets",
        "current_liabilities",
        "accounts_receivable", 
        "inventory",
        "accounts_payable",
        "total_revenue",
        "cost_of_goods_sold"
    ],
    required_ratios=[
        "working_capital_ratio",
        "working_capital_turnover",
        "cash_conversion_cycle",
        "days_sales_outstanding",
        "days_inventory_outstanding", 
        "days_payable_outstanding",
        "receivables_turnover",
        "inventory_turnover"
    ],
    peer_comparison_required=True,
    default_max_loops=3,
    analyst_prompt_template="""
DOMAIN: working_capital
COMPANY: {company}
ANALYSIS FOCUS: {user_focus}
TIME PERIOD: {time_period}
TARGET AUDIENCE: {persona}

SUPPLIED DATA SUMMARY:
- Financial metrics: {metrics_count} items
- Financial ratios: {ratios_count} items
- Peer comparison: {peer_count} companies

AUDIENCE-SPECIFIC GUIDANCE:
{persona_guidance}

Use schema: {schema_ref}
""",
    reviewer_criteria=[
        # Research report structure
        "Chart data selection: 2-4 chartable_ratios and 2-4 chartable_metrics selected",
        "Analysis bullets: 3-10 bullets, each integrating quantitative + qualitative evidence",
        "Executive summary: ≥100 chars, captures key takeaways",
        # Source attribution
        "Source attribution: Every qualitative claim has DetailedSource (tool, period, speaker/report)",
        "Tool-derived insights properly cited with metadata",
        # Data discipline
        "Data discipline: All financial numbers from supplied data (NOT extracted from tools)",
        "Quantitative evidence references supplied_metrics and supplied_ratios only",
        # Domain-specific quality
        "Cash conversion cycle analysis accuracy with component breakdown",
        "Component efficiency evaluation (DSO, DIO, DPO) with trend analysis",
        "Seasonal pattern recognition and impact assessment",
        "Peer comparison: selected efficiency ratios show company vs industry",
        # Tool usage strategy
        "Tool usage: Targeted to understand working capital management initiatives",
        "Tool calls address specific efficiency questions or operational changes",
        # Recommendations and risks
        "Risk factors: 2-5 specific working capital risks identified",
        "Optimization opportunity identification with quantified potential impact",
        "Recommendations: 2-5 actionable with implementation feasibility assessment"
    ]
)

# Operating Efficiency Domain Configuration
OPERATING_EFFICIENCY_CONFIG = DomainAgentConfig(
    domain="operating_efficiency", 
    required_metrics=[
        "revenue",
        "total_revenue",
        "gross_profit",
        "cost_of_goods_sold",
        "operating_expenses",
        "operating_income",
        "ebitda",
        "total_assets"
    ],
    required_ratios=[
        "gross_margin",
        "operating_margin",
        "ebitda_margin",
        "asset_turnover",
        "roa",
        "operating_efficiency_ratio"
    ],
    peer_comparison_required=True,
    default_max_loops=3,
    analyst_prompt_template="""
DOMAIN: operating_efficiency
COMPANY: {company}
ANALYSIS FOCUS: {user_focus}
TIME PERIOD: {time_period}
TARGET AUDIENCE: {persona}

SUPPLIED DATA SUMMARY:
- Financial metrics: {metrics_count} items
- Financial ratios: {ratios_count} items
- Peer comparison: {peer_count} companies

AUDIENCE-SPECIFIC GUIDANCE:
{persona_guidance}

Use schema: {schema_ref}
""",
    reviewer_criteria=[
        # Research report structure
        "Chart data selection: 2-4 chartable_ratios and 2-4 chartable_metrics selected",
        "Analysis bullets: 3-10 bullets, each integrating quantitative + qualitative evidence",
        "Executive summary: ≥100 chars, captures key takeaways",
        # Source attribution
        "Source attribution: Every qualitative claim has DetailedSource (tool, period, speaker/report)",
        "Tool-derived insights properly cited with metadata",
        # Data discipline
        "Data discipline: All financial numbers from supplied data (NOT extracted from tools)",
        "Quantitative evidence references supplied_metrics and supplied_ratios only",
        # Domain-specific quality
        "Margin trend analysis quality and depth with driver identification",
        "Asset utilization assessment completeness (turnover, ROA trends)",
        "Cost structure evaluation accuracy with fixed/variable breakdown context",
        "Competitive benchmarking: selected metrics show company vs industry leaders",
        # Tool usage strategy
        "Tool usage: Targeted to understand operational initiatives and efficiency drivers",
        "Tool calls address specific margin or productivity questions",
        # Recommendations and risks
        "Risk factors: 2-5 specific operational risks with impact assessment",
        "Operational improvement feasibility with expected benefits",
        "Recommendations: 2-5 strategic efficiency initiatives with clear rationale"
    ]
)

# Valuation Domain Configuration
VALUATION_CONFIG = DomainAgentConfig(
    domain="valuation",
    required_metrics=[
        "market_capitalization",
        "enterprise_value",
        "net_income",
        "free_cash_flow",
        "revenue",
        "ebitda",
        "book_value",
        "shares_outstanding"
    ],
    required_ratios=[
        "price_to_earnings",
        "price_to_book",
        "ev_to_ebitda",
        "price_to_sales",
        "peg_ratio",
        "ev_to_sales",
        "price_to_free_cash_flow"
    ],
    peer_comparison_required=True,
    default_max_loops=3,
    analyst_prompt_template="""
DOMAIN: valuation
COMPANY: {company}
ANALYSIS FOCUS: {user_focus}
TIME PERIOD: {time_period}
TARGET AUDIENCE: {persona}

SUPPLIED DATA SUMMARY:
- Financial metrics: {metrics_count} items
- Financial ratios: {ratios_count} items
- Peer comparison: {peer_count} companies

AUDIENCE-SPECIFIC GUIDANCE:
{persona_guidance}

Use schema: {schema_ref}
""",
    reviewer_criteria=[
        # Research report structure
        "Chart data selection: 2-4 chartable_ratios and 2-4 chartable_metrics selected",
        "Analysis bullets: 3-10 bullets, each integrating quantitative + qualitative evidence",
        "Executive summary: ≥100 chars, captures key investment thesis",
        # Source attribution
        "Source attribution: Every qualitative claim has DetailedSource (tool, period, speaker/report)",
        "Tool-derived insights properly cited with metadata",
        # Data discipline
        "Data discipline: All financial numbers from supplied data (NOT extracted from tools)",
        "Quantitative evidence references supplied_metrics and supplied_ratios only",
        # Domain-specific quality
        "Multi-metric valuation approach completeness (P/E, EV/EBITDA, P/B, etc.)",
        "Peer valuation comparison: selected multiples show relative positioning",
        "Growth prospect assessment with forward-looking catalysts",
        "Risk-adjusted valuation consideration with sensitivity analysis",
        # Tool usage strategy
        "Tool usage: Targeted to understand growth drivers, market sentiment, catalysts",
        "Tool calls address specific valuation questions or investor concerns",
        # Recommendations and risks
        "Risk factors: 2-5 valuation-specific risks (multiples compression, growth slowdown)",
        "Investment thesis clarity: clear buy/hold/sell recommendation with price targets",
        "Market context integration quality with macro and sector considerations"
    ]
)

# Domain configurations registry
DOMAIN_CONFIGS: Dict[str, DomainAgentConfig] = {
    "liquidity": LIQUIDITY_CONFIG,
    "leverage": LEVERAGE_CONFIG,
    "working_capital": WORKING_CAPITAL_CONFIG,
    "operating_efficiency": OPERATING_EFFICIENCY_CONFIG,
    "valuation": VALUATION_CONFIG
}

def get_domain_config(domain: str) -> DomainAgentConfig:
    """Get configuration for a specific domain."""
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CONFIGS.keys())}")
    return DOMAIN_CONFIGS[domain]

def get_available_domains() -> List[str]:
    """Get list of all available domain agents.""" 
    return list(DOMAIN_CONFIGS.keys())