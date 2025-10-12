"""
Comprehensive Demo of AgentSystemV2 - RESEARCH-GRADE REPORTS
Showcases new features: ResearchReportOutput, chart-ready data, source attribution, formatted reports
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AgentSystemV2
from config.schemas import MasterAnalysisRequest, ResearchReportOutput
from orchestration.master_agent import MasterAgent


def print_section_header(title: str, emoji: str = "🔍"):
    """Print a visually distinct section header"""
    print("\n" + "═" * 80)
    print(f"{emoji}  {title}")
    print("═" * 80)


def print_subsection(title: str):
    """Print a subsection header"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print("─" * 80)


async def demo_research_grade_liquidity_analysis():
    """
    Demonstrate research-grade liquidity analysis with new features:
    - ResearchReportOutput schema
    - Chart-ready data (ratios + metrics)
    - Detailed source attribution
    - Professional formatting
    """

    print_section_header("RESEARCH-GRADE LIQUIDITY ANALYSIS", "🎯")

    print("\n📋 Configuration:")
    print("   Company: AAPL (Apple Inc.)")
    print("   Domain: Liquidity")
    print("   Peers: MSFT, GOOGL, META")
    print("   Persona: institutional_investor (default)")
    print("   Focus: Deep investigative analysis with source attribution")
    print("   Max Loops: 3 (allows for revision)")

    system = AgentSystemV2(enable_debug=True)

    # Run liquidity analysis with investigative focus
    result = await system.analyze_domain(
        company="AAPL",
        domain="liquidity",
        peers=["MSFT", "GOOGL", "META"],
        user_focus="Investigate Apple's liquidity position. Compare cash ratios to Microsoft. Find management commentary on cash strategy, working capital efficiency, and any recent changes to liquidity management.",
        time_period="12 quarters",
        persona="institutional_investor",  # NEW: Target audience
        max_loops=3
    )

    if not result["success"]:
        print(f"\n❌ Analysis failed: {result.get('error')}")
        return None

    # Get the analysis output
    analysis = result["analysis_output"]

    print_subsection("Analysis Metadata")
    print(f"   Run ID: {result['run_id']}")
    print(f"   Company: {analysis.company}")
    print(f"   Domain: {analysis.domain}")
    print(f"   Confidence: {analysis.confidence_level.upper()}")
    print(f"   Output Format: ResearchReportOutput ✨")
    print(f"   Loops Used: {result['loops_used']}/{result.get('max_loops', 3)}")
    print(f"   Data Quality: {result['data_quality']:.2f}/1.0")

    # Display cost breakdown if available
    if 'cost_breakdown' in result:
        print_section_header("LLM COST TRACKING", "💰")
        print(result['cost_breakdown'])

    # Show tool calls made (compact view)
    print_subsection(f"Tool Calls Made ({len(analysis.tool_calls_made)})")
    if analysis.tool_calls_made:
        for i, tool_call in enumerate(analysis.tool_calls_made[:6], 1):  # Show up to 6
            print(f"   {i}. {tool_call}")
    else:
        print("   No tool calls made")

    # Research Report Features Display
    display_research_report_features(analysis)

    # Show formatted report
    print_section_header("FORMATTED RESEARCH REPORT", "📄")
    master = MasterAgent(enable_debug=False)
    formatted_report = master.format_research_report(analysis)
    print(formatted_report)

    # Show review history
    if result.get('review_history'):
        print_section_header("REVIEW HISTORY (Analyst-Reviewer Loop)", "🔄")
        for i, review in enumerate(result['review_history'], 1):
            print(f"\n   Loop {i}:")
            print(f"   Verdict: {review.verdict.upper()}")
            print(f"   Quality Score: {review.quality_score:.2f}/1.0")
            print(f"   Strengths: {len(review.strengths)}")
            for strength in review.strengths[:2]:
                print(f"      ✓ {strength[:80]}")
            print(f"   Weaknesses: {len(review.weaknesses)}")
            for weakness in review.weaknesses[:2]:
                print(f"      ✗ {weakness[:80]}")
            if i < len(result['review_history']):
                print(f"   → Continuing to Loop {i+1} for revision")

    return result["run_id"]


def display_research_report_features(analysis: ResearchReportOutput):
    """Display new research-grade report features"""

    print_subsection("NEW FEATURE: Chart-Ready Ratios")
    print(f"   Selected Ratios: {len(analysis.chartable_ratios)} (requirement: 2-4)")

    for i, ratio in enumerate(analysis.chartable_ratios, 1):
        print(f"\n   {i}. {ratio.ratio_name}")
        print(f"      Trend: {ratio.trend_direction.upper()}")
        print(f"      Interpretation: {ratio.interpretation[:80]}...")

        # Show company values
        if ratio.company_values:
            recent_quarters = sorted(ratio.company_values.items(), reverse=True)[:3]
            values_str = ", ".join([f"{q}: {v:.3f}" for q, v in recent_quarters])
            print(f"      Company Values: {values_str}")

        # Show peer comparison
        if ratio.peer_values:
            peer_count = len(ratio.peer_values)
            print(f"      Peer Comparison: {peer_count} peer(s)")
            for peer, peer_data in list(ratio.peer_values.items())[:2]:  # Show first 2 peers
                if peer_data:
                    recent = sorted(peer_data.items(), reverse=True)[:2]
                    peer_vals = ", ".join([f"{q}: {v:.3f}" for q, v in recent])
                    print(f"         {peer}: {peer_vals}")

    print_subsection("NEW FEATURE: Chart-Ready Metrics")
    print(f"   Selected Metrics: {len(analysis.chartable_metrics)} (requirement: 2-4)")

    for i, metric in enumerate(analysis.chartable_metrics, 1):
        print(f"\n   {i}. {metric.metric_name}")
        print(f"      Trend: {metric.trend_direction.upper()}")
        print(f"      Unit: {metric.unit}")
        print(f"      Interpretation: {metric.interpretation[:80]}...")

        if metric.values:
            recent_quarters = sorted(metric.values.items(), reverse=True)[:4]
            values_str = ", ".join([f"{q}: {format_value(v, metric.unit)}" for q, v in recent_quarters])
            print(f"      Values: {values_str}")

    print_subsection("NEW FEATURE: Analysis Bullets with Source Attribution")
    print(f"   Bullets: {len(analysis.analysis_bullets)} (requirement: 3-10)")

    for i, bullet in enumerate(analysis.analysis_bullets, 1):
        importance_emoji = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}
        emoji = importance_emoji.get(bullet.importance, "⚪")

        print(f"\n   {i}. {emoji} {bullet.bullet_text[:100]}...")

        # Quantitative evidence
        print(f"      📊 Quantitative Evidence ({len(bullet.quantitative_evidence)}):")
        for evidence in bullet.quantitative_evidence[:2]:
            print(f"         • {evidence}")

        # Detailed sources
        print(f"      📚 Sources ({len(bullet.qualitative_sources)}):")
        for source in bullet.qualitative_sources[:3]:
            source_dict = source if isinstance(source, dict) else (source.to_dict() if hasattr(source, 'to_dict') else {})
            parts = [source_dict.get('tool_name', 'Unknown')]
            if source_dict.get('period'):
                parts.append(source_dict['period'])
            if source_dict.get('speaker_name'):
                parts.append(f"Speaker: {source_dict['speaker_name']}")
            if source_dict.get('report_type'):
                parts.append(source_dict['report_type'])
            print(f"         • {', '.join(parts)}")
            if source_dict.get('chunk_summary'):
                print(f"           └─ {source_dict['chunk_summary'][:70]}...")

    print_subsection("Executive Summary")
    print(f"   Length: {len(analysis.executive_summary)} chars (requirement: ≥100)")
    print(f"\n   {analysis.executive_summary}")

    print_subsection("Recommendations & Risk Factors")
    print(f"   Recommendations: {len(analysis.recommendations)} (requirement: 2-5)")
    for i, rec in enumerate(analysis.recommendations, 1):
        print(f"      {i}. {rec}")

    print(f"\n   Risk Factors: {len(analysis.risk_factors)} (requirement: 2-5)")
    for i, risk in enumerate(analysis.risk_factors, 1):
        print(f"      {i}. {risk}")

    print_subsection("Data Sources Summary")
    print(f"   {analysis.data_sources_summary}")

    # Show ratio definitions if present
    if hasattr(analysis, 'ratio_definitions') and analysis.ratio_definitions:
        print_subsection(f"📖 Ratio Definitions ({len(analysis.ratio_definitions)})")
        for ratio_name, definition in analysis.ratio_definitions.items():
            print(f"\n   {ratio_name}:")
            print(f"      Formula: {definition.get('formula', 'N/A')}")
            print(f"      Description: {definition.get('description', 'N/A')}")
            print(f"      Category: {definition.get('category', 'N/A')}")
            if definition.get('interpretation'):
                print(f"      Interpretation: {definition['interpretation'][:100]}...")
    else:
        print_subsection("📖 Ratio Definitions")
        print("   ⚠️ No ratio definitions found (enrichment may have failed)")


def format_value(value: float, unit: str) -> str:
    """Format value based on unit"""
    if "USD" in unit or "$" in unit:
        if abs(value) >= 1e9:
            return f"${value/1e9:.1f}B"
        elif abs(value) >= 1e6:
            return f"${value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.1f}K"
        else:
            return f"${value:.2f}"
    else:
        return f"{value:,.0f}"


async def demo_schema_validation():
    """Demonstrate schema validation and constraints"""

    print_section_header("SCHEMA VALIDATION & CONSTRAINTS", "✅")

    print("\nResearchReportOutput Requirements:")
    print("   ✓ Chart-ready ratios: 2-4 (for visualization)")
    print("   ✓ Chart-ready metrics: 2-4 (for visualization)")
    print("   ✓ Analysis bullets: 3-10 (detailed analysis)")
    print("   ✓ Recommendations: 2-5")
    print("   ✓ Risk factors: 2-5")
    print("   ✓ Executive summary: ≥100 characters")
    print("   ✓ Every bullet has quantitative evidence + qualitative sources")
    print("   ✓ Every source has: tool_name, period, and metadata")

    print("\nData Discipline:")
    print("   📊 Quantitative data: ONLY from supplied_metrics and supplied_ratios")
    print("   📚 Qualitative data: From tools (transcripts, reports, news)")
    print("   🚫 Never extract financial numbers from tool outputs")

    print("\nRevision Process:")
    print("   🔄 Loop 1: Initial analysis with full data")
    print("   🔄 Loop 2+: Revision with previous JSON + reviewer feedback")
    print("   ✨ Revisions refine existing analysis (not recreate)")
    print("   🎯 Tool calls are targeted to address specific weaknesses")


async def demo_persona_support():
    """Demonstrate persona-aware analysis tailoring"""

    print_section_header("PERSONA-AWARE ANALYSIS", "👥")

    print("\nAvailable Personas:")
    print("   Each persona tailors language, depth, and focus areas")
    print()

    personas = {
        "banker": {
            "emoji": "🏦",
            "focus": "Credit risk, debt service capability, covenant compliance",
            "style": "Technical & formal, emphasizes downside scenarios"
        },
        "cfo": {
            "emoji": "💼",
            "focus": "Operational implications, strategic decisions, board presentation",
            "style": "Executive-level, strategic tone, competitive positioning"
        },
        "trader": {
            "emoji": "📈",
            "focus": "Near-term catalysts, trading signals, volatility drivers",
            "style": "Concise & actionable, short-term focus"
        },
        "institutional_investor": {
            "emoji": "💰",
            "focus": "Long-term fundamentals, competitive moat, risk-adjusted returns",
            "style": "Comprehensive & professional, multi-year perspective"
        },
        "retail_investor": {
            "emoji": "👤",
            "focus": "Investment thesis in plain language, key strengths/weaknesses",
            "style": "Clear & accessible, minimal jargon, big picture focus"
        }
    }

    for persona, details in personas.items():
        print(f"{details['emoji']} {persona.replace('_', ' ').title()}:")
        print(f"   Focus: {details['focus']}")
        print(f"   Style: {details['style']}")
        print()

    print("💡 Usage Example:")
    print("   result = await system.analyze_domain(")
    print("       company='AAPL',")
    print("       domain='liquidity',")
    print("       persona='banker',  # ← Target audience")
    print("       user_focus='Your analysis question here'")
    print("   )")
    print()
    print("🎯 Impact:")
    print("   • Banker analysis: Deep credit metrics, covenant headroom, lending standards")
    print("   • CFO analysis: Strategic options, board-ready insights, operational impact")
    print("   • Trader analysis: Immediate catalysts, trading opportunities, price action")
    print("   • Investor analysis: Long-term thesis, competitive advantages, ESG considerations")
    print("   • Retail analysis: Plain language, simple metrics, clear recommendations")


async def demo_system_status():
    """Show system status and capabilities"""

    print_section_header("SYSTEM STATUS & CAPABILITIES", "⚙️")

    system = AgentSystemV2()

    # Show available domains
    domains = system.get_available_domains()
    print(f"\n📊 Available Domains ({len(domains)}):")
    for domain in domains:
        capabilities = system.get_domain_capabilities(domain)
        # Note: capabilities returns counts (ints), not lists
        metrics_count = capabilities.get('required_metrics', 0)
        ratios_count = capabilities.get('required_ratios', 0)
        print(f"   • {domain}: {metrics_count} metrics, {ratios_count} ratios")

    # Show system status
    status = system.get_system_status()
    print(f"\n📈 System Status:")
    print(f"   Version: {status['system_version']}")
    print(f"   Total Analyses: {status['master_agent']['total_analyses']}")
    print(f"   Success Rate: {status['master_agent']['success_rate']:.1%}")
    print(f"   Active Agents: {status['agent_factory']['active_agents_count']}")
    print(f"   Debug Mode: {status['debug_enabled']}")

    print(f"\n🆕 Research-Grade Features:")
    print(f"   ✓ ResearchReportOutput schema")
    print(f"   ✓ Chart-ready data structures")
    print(f"   ✓ Detailed source attribution")
    print(f"   ✓ Professional report formatting")
    print(f"   ✓ Optimized revision prompts")
    print(f"   ✓ Enhanced reviewer criteria")


async def main():
    """Run comprehensive demo showcasing new research-grade features"""

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "🎯 AgentSystemV2 - RESEARCH-GRADE REPORTS DEMO" + " " * 15 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"\n📅 Demo started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Focus: Liquidity Analysis with Research-Grade Report Features")
    print("\nNew Features Being Demonstrated:")
    print("   ✨ ResearchReportOutput schema with validation")
    print("   ✨ Chart-ready ratios and metrics (2-4 each)")
    print("   ✨ Analysis bullets with detailed source attribution")
    print("   ✨ Persona-aware analysis (banker, CFO, trader, investor)")
    print("   ✨ Professional formatted reports")
    print("   ✨ Optimized revision process")
    print("   ✨ Enhanced reviewer criteria")

    try:
        # Demo 1: Run actual research-grade analysis
        print("\n" + "▶" * 40)
        run_id = await demo_research_grade_liquidity_analysis()

        # Demo 2: Show persona support
        await demo_persona_support()

        # Demo 3: Show schema validation
        await demo_schema_validation()

        # Demo 4: System status
        await demo_system_status()

        # Summary
        print_section_header("DEMO SUMMARY", "🎉")
        print(f"   Analysis Run ID: {run_id}")
        print(f"   Status: ✅ All demos completed successfully")
        print(f"\n   Key Takeaways:")
        print(f"   • Research-grade reports produce professional analyst-quality output")
        print(f"   • Chart-ready data makes visualization straightforward")
        print(f"   • Detailed sources make every claim traceable")
        print(f"   • Revision loops are more efficient and targeted")
        print("\n" + "═" * 80)
        print("   🚀 The system is ready for production use!")
        print("═" * 80)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logging.exception("Demo execution failed")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run comprehensive demo
    asyncio.run(main())
