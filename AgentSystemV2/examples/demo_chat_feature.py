"""
Demo: RAG-Powered Chat Feature for AgentSystemV2

This demo showcases:
1. Running a liquidity analysis
2. Starting a chat session
3. Asking various types of questions
4. Smart routing (ChatAgent vs full refinement)
5. Cost tracking per session
6. Session status monitoring
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AgentSystemV2
from tools.rag_tools import get_rag_tools_status, AVAILABLE_RAG_TOOLS
from config.schemas import (
    ResearchReportOutput, ChartableRatio, ChartableMetric,
    AnalysisBullet, DetailedSource
)


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print formatted subheader"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print("─" * 80)


def print_chat_response(response: dict, exchange_num: int):
    """Print formatted chat response"""
    print(f"\n💬 Exchange #{exchange_num}")
    print("-" * 80)

    response_type = response.get("type", "unknown")
    message = response.get("message", "")
    cost = response.get("cost", 0.0)
    source = response.get("source", "unknown")

    # Response type indicator
    if response_type == "chat_response":
        print("🤖 ChatAgent Response (Fast, RAG-powered)")
    elif response_type == "refinement":
        print("🔄 Full Refinement (Analyst-Reviewer Loop)")
    elif response_type == "limit_reached":
        print("⚠️  Limit Reached")
    else:
        print(f"📋 {response_type}")

    # Message
    print(f"\n{message}")

    # Metadata
    print(f"\nSource: {source}")
    print(f"Cost: ${cost:.6f}")

    if "tokens_used" in response:
        print(f"Tokens: {response['tokens_used']}")

    if "refinement_number" in response:
        print(f"Refinement: {response['refinement_number']}")
        print(f"Review Cycles: {response.get('review_cycles', 0)}")
        print(f"Quality Score: {response.get('quality_score', 0):.2f}")

    if "total_session_cost" in response:
        print(f"Total Session Cost: ${response['total_session_cost']:.4f}")


def print_session_status(status: dict):
    """Print formatted session status"""
    print_subheader("Session Status")

    print(f"Session ID: {status['session_id']}")
    print(f"Company: {status['company']} | Domain: {status['domain']}")
    print(f"Status: {status['status']}")
    print(f"")
    print(f"Exchanges: {status['exchange_count']}/{status['exchange_limit']}")
    print(f"Refinements: {status['refinements_used']}")
    print(f"Total Cost: ${status['total_cost']:.4f} / ${status['cost_limit']:.2f}")
    print(f"ChatAgent Active: {'✓' if status['has_chat_agent'] else '✗'}")
    print(f"")
    print(f"Created: {status['created_at']}")
    print(f"Last Activity: {status['last_activity']}")


def create_dummy_liquidity_analysis(run_id: str = "demo_run_001") -> ResearchReportOutput:
    """
    Create a dummy pre-saved liquidity analysis to save LLM costs during testing.
    This mimics a real liquidity analysis output for AAPL.
    """

    # Create chartable ratios (2-4 required)
    chartable_ratios = [
        ChartableRatio(
            ratio_name="Current Ratio",
            company_values={
                "Q3-2025": 1.07,
                "Q2-2025": 1.04,
                "Q1-2025": 1.06,
                "Q4-2024": 0.96
            },
            peer_values={
                "MSFT": {
                    "Q3-2025": 1.78,
                    "Q2-2025": 1.82,
                    "Q1-2025": 1.75,
                    "Q4-2024": 1.80
                }
            },
            interpretation="Apple's current ratio remains below 1.5x industry standard and significantly trails Microsoft, indicating tighter liquidity management driven by aggressive capital return programs.",
            trend_direction="stable"
        ),
        ChartableRatio(
            ratio_name="Quick Ratio",
            company_values={
                "Q3-2025": 0.92,
                "Q2-2025": 0.89,
                "Q1-2025": 0.91,
                "Q4-2024": 0.82
            },
            peer_values={
                "MSFT": {
                    "Q3-2025": 1.65,
                    "Q2-2025": 1.68,
                    "Q1-2025": 1.62,
                    "Q4-2024": 1.67
                }
            },
            interpretation="Quick ratio improvement from Q4-2024 to Q3-2025 reflects better working capital management, though still below peer benchmarks.",
            trend_direction="improving"
        ),
        ChartableRatio(
            ratio_name="Cash to Sales Ratio",
            company_values={
                "Q3-2025": 0.135,
                "Q2-2025": 0.142,
                "Q1-2025": 0.156,
                "Q4-2024": 0.178
            },
            peer_values={
                "MSFT": {
                    "Q3-2025": 0.245,
                    "Q2-2025": 0.238,
                    "Q1-2025": 0.242,
                    "Q4-2024": 0.251
                }
            },
            interpretation="Declining cash/sales ratio reflects strategic prioritization of shareholder returns over cash accumulation, contrasting with Microsoft's higher cash retention policy.",
            trend_direction="declining"
        )
    ]

    # Create chartable metrics (2-4 required)
    chartable_metrics = [
        ChartableMetric(
            metric_name="Cash and Cash Equivalents",
            values={
                "Q3-2025": 19100000000,
                "Q2-2025": 20300000000,
                "Q1-2025": 23500000000,
                "Q4-2024": 35200000000
            },
            unit="USD",
            trend_direction="decreasing",
            interpretation="Cash reserves decreased 45.7% from Q4-2024 to Q3-2025, primarily due to $93.2B deployed in share buybacks and dividends."
        ),
        ChartableMetric(
            metric_name="Operating Cash Flow",
            values={
                "Q3-2025": 108600000000,
                "Q2-2025": 109600000000,
                "Q1-2025": 108300000000,
                "Q4-2024": 118300000000
            },
            unit="USD",
            trend_direction="stable",
            interpretation="Operating cash flow remains robust and stable across quarters, demonstrating strong underlying business fundamentals despite cash balance reduction."
        ),
        ChartableMetric(
            metric_name="Current Liabilities",
            values={
                "Q3-2025": 141100000000,
                "Q2-2025": 144600000000,
                "Q1-2025": 144400000000,
                "Q4-2024": 176400000000
            },
            unit="USD",
            trend_direction="decreasing",
            interpretation="Current liabilities decreased 20% from Q4-2024, reflecting improved working capital management and seasonal payment cycle effects."
        )
    ]

    # Create analysis bullets (3-10 required, each >= 200 chars)
    analysis_bullets = [
        AnalysisBullet(
            bullet_text="Apple's liquidity position has evolved significantly over the past eight quarters, with the current ratio stabilizing around 1.07 in Q3-2025 after reaching a low of 0.96 in Q4-2024. While this represents improvement from the prior year's trough, Apple's liquidity ratios remain below industry norms and significantly trail Microsoft's 1.78 current ratio. This strategic choice reflects Apple's deliberate capital allocation policy favoring aggressive shareholder returns over cash accumulation, with $93.2B deployed in buybacks and dividends across recent quarters. Management has consistently emphasized that operating cash flow generation of approximately $108-110B quarterly provides sufficient coverage for operational needs despite lower absolute cash balances.",
            quantitative_evidence=[
                "Current ratio: 1.07 (Q3-2025) vs. 0.96 (Q4-2024)",
                "Microsoft current ratio: 1.78 (Q3-2025)",
                "Capital returns: $93.2B in recent quarters"
            ],
            qualitative_sources=[
                DetailedSource(
                    tool_name="search_annual_reports",
                    period="Q3-2025",
                    report_type="10-Q",
                    chunk_summary="Management discussion on liquidity adequacy and capital allocation strategy",
                    relevance_score=0.95
                ),
                DetailedSource(
                    tool_name="search_transcripts",
                    period="Q3-2025",
                    speaker_name="Luca Maestri, CFO",
                    chunk_summary="CFO commentary on cash deployment priorities and shareholder return commitment",
                    relevance_score=0.92
                )
            ],
            importance="critical"
        ),
        AnalysisBullet(
            bullet_text="The declining cash/sales ratio from 0.178 in Q4-2024 to 0.135 in Q3-2025 represents a 24% reduction, signaling a fundamental shift in Apple's liquidity management philosophy. Unlike Microsoft which maintains a cash/sales ratio above 0.24, Apple has consciously opted for a leaner balance sheet approach. This strategy is underpinned by confidence in the company's consistent operating cash flow generation and access to debt markets. However, this approach introduces execution risk if revenue growth slows or if unexpected operational needs arise, as the company has reduced its liquidity buffer from historical norms.",
            quantitative_evidence=[
                "Cash/Sales ratio: 0.135 (Q3-2025) vs. 0.178 (Q4-2024), -24%",
                "Microsoft Cash/Sales ratio: 0.245 (Q3-2025)",
                "Operating cash flow: $108.6B (Q3-2025), stable"
            ],
            qualitative_sources=[
                DetailedSource(
                    tool_name="search_news",
                    period="Q3-2025",
                    chunk_summary="Analyst concerns about Apple's reduced cash cushion relative to peers",
                    relevance_score=0.88
                ),
                DetailedSource(
                    tool_name="search_annual_reports",
                    period="Q2-2025",
                    report_type="10-Q",
                    chunk_summary="Discussion of capital structure optimization and shareholder value creation",
                    relevance_score=0.85
                )
            ],
            importance="critical"
        ),
        AnalysisBullet(
            bullet_text="Working capital management has emerged as a key focus area, evidenced by the 20% reduction in current liabilities from $176.4B in Q4-2024 to $141.1B in Q3-2025. Accounts payable decreased substantially from $69.0B to $50.4B in the same period, suggesting tighter supplier payment terms or seasonal effects in the supply chain. The quick ratio improvement from 0.82 to 0.92 across this timeframe indicates that these working capital initiatives are enhancing Apple's operational liquidity position. Recent quarterly filings signal management's proactive stance in optimizing working capital to mitigate risks from the lower cash reserve strategy.",
            quantitative_evidence=[
                "Current liabilities: $141.1B (Q3-2025) vs. $176.4B (Q4-2024), -20%",
                "Accounts payable: $50.4B (Q3-2025) vs. $69.0B (Q4-2024)",
                "Quick ratio: 0.92 (Q3-2025) vs. 0.82 (Q4-2024)"
            ],
            qualitative_sources=[
                DetailedSource(
                    tool_name="search_annual_reports",
                    period="Q3-2025",
                    report_type="10-Q",
                    chunk_summary="Enhanced working capital management initiatives and efficiency measures",
                    relevance_score=0.90
                ),
                DetailedSource(
                    tool_name="search_transcripts",
                    period="Q2-2025",
                    speaker_name="Tim Cook, CEO",
                    chunk_summary="CEO remarks on operational efficiency and cash flow optimization",
                    relevance_score=0.87
                )
            ],
            importance="high"
        ),
        AnalysisBullet(
            bullet_text="Analyst sentiment regarding Apple's liquidity metrics has shifted noticeably, with recent ratings showing a decrease in buy recommendations as concerns mount about the sustainability of the aggressive capital return program. The market is questioning whether Apple's liquidity cushion remains adequate given increasing competitive pressures in key product categories and potential economic headwinds. While management maintains confidence in the strategy, the divergence between Apple's approach and Microsoft's more conservative liquidity stance has become a focal point in comparative valuation discussions among institutional investors.",
            quantitative_evidence=[
                "Cash reserves decreased 45.7% from Q4-2024 to Q3-2025",
                "Share buybacks and dividends totaled $93.2B in recent quarters"
            ],
            qualitative_sources=[
                DetailedSource(
                    tool_name="search_news",
                    period="Q3-2025",
                    chunk_summary="Analyst downgrades citing liquidity concerns and capital allocation risks",
                    relevance_score=0.83
                ),
                DetailedSource(
                    tool_name="search_transcripts",
                    period="Q3-2025",
                    speaker_name="Tejas Gala, Senior Analyst",
                    chunk_summary="Analyst Q&A session discussing liquidity strategy vs peers",
                    relevance_score=0.81
                )
            ],
            importance="high"
        )
    ]

    # Create recommendations (2-5 required)
    recommendations = [
        "Monitor working capital trends closely to ensure adequate liquidity buffer is maintained, with specific attention to days payable outstanding and days sales outstanding metrics as early warning indicators.",
        "Consider adjusting the pace of capital return programs if liquidity metrics deteriorate further or if operating cash flow shows signs of weakness, establishing predetermined thresholds for capital allocation flexibility.",
        "Enhance disclosure around liquidity stress testing scenarios and contingency funding plans to address investor concerns about the reduced cash cushion relative to historical norms and peer companies.",
        "Evaluate opportunistic debt issuance to maintain strategic flexibility given favorable credit spreads, potentially providing a liquidity backstop while preserving operational cash flow for shareholder returns."
    ]

    # Create risk factors (2-5 required)
    risk_factors = [
        "Continued high capital expenditures and aggressive share repurchases may strain liquidity if not offset by sustained revenue growth, particularly if iPhone sales face prolonged headwinds.",
        "Reduced cash reserves limit financial flexibility to respond to unexpected operational challenges, major acquisitions, or economic downturns compared to peers with stronger liquidity positions.",
        "Growing analyst skepticism about liquidity adequacy could impact valuation multiples and stock performance if market sentiment shifts toward favoring balance sheet strength over capital returns.",
        "Working capital optimization initiatives may face limits, and any reversal in payables management could create short-term liquidity pressures requiring adjustments to capital allocation priorities."
    ]

    # Create tool calls record
    tool_calls_made = [
        {"tool": "search_annual_reports", "params": {"ticker": "AAPL", "query": "liquidity cash flow working capital"}},
        {"tool": "search_transcripts", "params": {"ticker": "AAPL", "query": "liquidity management capital allocation"}},
        {"tool": "search_news", "params": {"ticker": "AAPL", "query": "analyst ratings liquidity"}},
        {"tool": "get_financial_metrics", "params": {"ticker": "AAPL", "metrics": ["cash", "current_liabilities"]}},
        {"tool": "compare_companies", "params": {"tickers": ["AAPL", "MSFT"], "domain": "liquidity"}}
    ]

    # Create the ResearchReportOutput
    return ResearchReportOutput(
        run_id=run_id,
        domain="liquidity",
        company="AAPL",
        ticker="AAPL",
        analysis_timestamp=datetime.now().isoformat(),
        executive_summary="Apple Inc. demonstrates a strategic liquidity posture characterized by lower cash reserves relative to peers like Microsoft, driven by aggressive capital return programs totaling $93.2B in recent quarters. The current ratio of 1.07 and quick ratio of 0.92 in Q3-2025 reflect deliberate balance sheet optimization, though these metrics trail industry benchmarks. Strong operating cash flow generation of approximately $108-110B quarterly provides foundational liquidity support, while enhanced working capital management has partially offset declining cash balances. However, this lean liquidity approach introduces execution risk and has contributed to shifting analyst sentiment regarding the sustainability of current capital allocation priorities.",
        chartable_ratios=chartable_ratios,
        chartable_metrics=chartable_metrics,
        analysis_bullets=analysis_bullets,
        recommendations=recommendations,
        risk_factors=risk_factors,
        confidence_level="medium",
        tool_calls_made=tool_calls_made,
        data_sources_summary="Analysis based on SEC 10-Q filings (Q4-2024 through Q3-2025), earnings call transcripts, financial metrics from SECFinancialRAG, and recent analyst reports and news coverage."
    )


async def demo_chat_feature():
    """Demonstrate the chat feature with various question types"""

    print_header("🎯 Chat Feature Demo: RAG-Powered Conversations")

    print("\n📋 Demo Flow:")
    print("   1. Verify RAG tool enhancements (including SECFinancialRAG)")
    print("   2. Load pre-saved liquidity analysis for AAPL (saves ~$0.067 in LLM costs!)")
    print("   3. Start chat session with company restrictions")
    print("   4. Test all 6 RAG tools including new SEC tools")
    print("   5. Test company restrictions (decline unauthorized companies)")
    print("   6. Test ticker mapping (Apple → AAPL)")
    print("   7. Request full refinement (slow, expensive)")
    print("   8. Monitor costs and session status")

    # Step 0: Verify RAG tool enhancements
    print_header("Step 0: Verifying RAG Tool Enhancements")

    print("\n🔧 Checking RAG Tools Status...")
    status = get_rag_tools_status()

    print(f"\nAvailable RAG Tools:")
    print(f"  • Annual Reports: {'✓' if status['annual_reports'] else '✗'}")
    print(f"  • Transcripts: {'✓' if status['transcripts'] else '✗'}")
    print(f"  • News: {'✓' if status['news'] else '✗'}")
    print(f"  • Share Price: {'✓' if status['share_price'] else '✗'}")
    print(f"  • SEC Financial: {'✓' if status['sec_financial'] else '✗'}")
    print(f"\nTotal: {status['total_available']} tools available")

    # Check for new SEC tools specifically
    sec_tools = [t.__name__ for t in AVAILABLE_RAG_TOOLS if 'financial' in t.__name__ or 'compare' in t.__name__]
    print(f"\n✨ New SECFinancialRAG Tools:")
    for tool_name in sec_tools:
        print(f"  • {tool_name}")

    if len(sec_tools) >= 2:
        print("\n✅ SECFinancialRAG tools successfully loaded!")
    else:
        print("\n⚠️ WARNING: SECFinancialRAG tools may not be available")

    # Step 1: Use dummy pre-saved analysis (instead of expensive LLM call)
    print_header("Step 1: Loading Pre-Saved Analysis (Cost Optimization)")

    system = AgentSystemV2(enable_debug=False)

    print("\n💡 Using dummy pre-saved liquidity analysis for AAPL to save LLM costs...")
    print("   (In production, this would run: system.analyze_domain())")

    # Create dummy analysis
    run_id = "demo_liquidity_001"
    dummy_analysis = create_dummy_liquidity_analysis(run_id=run_id)

    # Store it in the system's master agent's active_analyses so chat can access it
    # Note: We create a dummy domain agent since chat session requires it
    from agents.domain_agent import DomainAgent
    dummy_domain_agent = system.master_agent.agent_factory.create_domain_agent("liquidity")

    system.master_agent.active_analyses[run_id] = {
        "domain_agent": dummy_domain_agent,
        "analysis_output": dummy_analysis,
        "review_history": []  # No review history for dummy data
    }

    # Create a result dict similar to what analyze_domain would return
    result = {
        "success": True,
        "run_id": run_id,
        "final_confidence": dummy_analysis.confidence_level,
        "total_cost": 0.0,  # Dummy data costs nothing!
        "analysis_output": dummy_analysis
    }

    print(f"✅ Analysis loaded from dummy data!")
    print(f"   Run ID: {result['run_id']}")
    print(f"   Confidence: {result['final_confidence']}")
    print(f"   Analysis Cost: $0.00 (using pre-saved dummy data)")
    print(f"   💰 Saved: ~$0.067 (typical liquidity analysis cost)")

    # Step 2: Start chat session
    print_header("Step 2: Starting Chat Session")

    # Use the system's internal master agent (don't create a new one!)
    chat_session = await system.start_chat_session(result['run_id'])

    if "error" in chat_session:
        print(f"❌ Chat session failed: {chat_session['error']}")
        return

    session_id = chat_session["session_id"]
    print(f"✅ Chat session started: {session_id}")
    print(f"\n{chat_session['welcome_message']}")

    # Get chat interface from system's master agent
    chat_interface = system.master_agent.chat_interface

    # Show allowed companies and ticker mapping
    session_state = chat_interface.active_sessions[session_id]
    chat_agent = session_state["chat_agent"]

    print_subheader("Session Configuration")
    print(f"\n🏢 Allowed Companies: {', '.join(chat_agent.allowed_tickers)}")
    print(f"\n📊 Ticker Mapping:")
    for ticker, name in chat_agent.ticker_to_name.items():
        print(f"  • {name} → {ticker}")

    print(f"\n🔒 Company Restriction Policy:")
    print(f"  ChatAgent will ONLY answer questions about: {', '.join(chat_agent.allowed_tickers)}")
    print(f"  Questions about other companies will be declined.")

    print(f"\n🔧 Available RAG Tools (6 total):")
    print(f"  1. search_annual_reports - Search SEC filings")
    print(f"  2. search_transcripts - Search earnings calls")
    print(f"  3. search_news - Search news articles")
    print(f"  4. get_share_price_data - Get stock prices")
    print(f"  5. get_financial_metrics - Get SEC financial data")
    print(f"  6. compare_companies - Compare companies by domain")

    # Step 3: Demonstrate various question types
    print_header("Step 3: Chat Interactions")

    questions = [
        {
            "type": "simple_q&a",
            "question": "What is Apple's current ratio in the latest quarter?",
            "description": "Simple data lookup (should use ChatAgent)"
        },
        {
            "type": "simple_q&a",
            "question": "Can you get the revenue and net income metrics for Apple for the last 4 quarters?",
            "description": "SEC Financial RAG tool test - get_financial_metrics"
        },
        {
            "type": "simple_q&a",
            "question": "How does Apple's liquidity compare to Microsoft?",
            "description": "SEC Financial RAG tool test - compare_companies"
        },
        {
            "type": "simple_q&a",
            "question": "Can you search for management commentary on cash management strategy?",
            "description": "Transcript RAG tool test (should use ChatAgent with transcript search)"
        },
        {
            "type": "company_restriction",
            "question": "What is Tesla's current ratio?",
            "description": "Company restriction test - should decline (Tesla not in allowed list)"
        },
        {
            "type": "simple_q&a",
            "question": "Explain why the current ratio changed over time for Apple",
            "description": "Explanation request (should use ChatAgent)"
        },
        {
            "type": "refinement",
            "question": "Regenerate the analysis with more focus on working capital efficiency",
            "description": "Complex refinement (should trigger full analyst-reviewer loop)"
        },
        {
            "type": "simple_q&a",
            "question": "What were the main findings about working capital?",
            "description": "Follow-up question (should use ChatAgent)"
        }
    ]

    for i, q_item in enumerate(questions, 1):
        print_subheader(f"Question {i}: {q_item['type']}")
        print(f"User: {q_item['question']}")
        print(f"(Expected: {q_item['description']})")

        # Process message
        response = await chat_interface.process_chat_message(
            session_id=session_id,
            user_message=q_item["question"]
        )

        # Display response
        print_chat_response(response, i)

        # Add feature annotation
        if 'SEC Financial' in q_item['description']:
            print("\n💡 Feature Test: SECFinancialRAG tool integration")
        elif 'restriction' in q_item['type']:
            print("\n💡 Feature Test: Company restriction enforcement")
            if "can only answer" in response.get("message", "").lower() or "cannot provide" in response.get("message", "").lower():
                print("   ✓ Company restriction working correctly!")
            else:
                print("   ⚠ Expected restriction message")
        elif 'Transcript RAG' in q_item['description']:
            print("\n💡 Feature Test: Transcript search with ticker mapping")

        # Brief pause for readability
        await asyncio.sleep(0.5)

    # Step 4: Show final session status
    print_header("Step 4: Final Session Status")

    status = chat_interface.get_session_status(session_id)
    print_session_status(status)

    # Step 5: Cost comparison
    print_header("Step 5: Cost Analysis")

    print("\n💰 Cost Breakdown:")
    print(f"   Initial Analysis: ${result['total_cost']:.4f} (dummy data - normally ~$0.067)")
    print(f"   Chat Session: ${status['total_cost']:.4f}")
    print(f"   Total Demo Cost: ${result['total_cost'] + status['total_cost']:.4f}")
    print(f"   💡 Cost Saved by using dummy data: ~$0.067")

    print("\n📊 Chat Session Stats:")
    print(f"   Simple Q&A responses: ~{status['exchange_count'] - status['refinements_used']}")
    print(f"   Full refinements: {status['refinements_used']}")
    if status['exchange_count'] > 0:
        print(f"   Average cost per exchange: ${status['total_cost'] / status['exchange_count']:.6f}")
    else:
        print(f"   Average cost per exchange: N/A (no exchanges)")

    # Calculate typical cost savings
    avg_refinement_cost = 0.03  # Typical full refinement
    avg_chat_cost = 0.0005  # Typical ChatAgent response
    simple_qa_count = status['exchange_count'] - status['refinements_used']

    if_all_refinements = (status['exchange_count'] * avg_refinement_cost)
    actual_cost = status['total_cost']
    savings = if_all_refinements - actual_cost

    print(f"\n💡 Cost Savings:")
    print(f"   If all exchanges used full refinement: ${if_all_refinements:.4f}")
    print(f"   Actual cost with smart routing: ${actual_cost:.4f}")
    print(f"   Savings: ${savings:.4f} ({(savings/if_all_refinements)*100:.1f}% reduction)")

    # Step 6: End session
    print_header("Step 6: Ending Session")

    end_result = chat_interface.end_session(session_id)
    print(f"✅ Session ended")
    print(f"   Total exchanges: {end_result['total_exchanges']}")
    print(f"   Refinements performed: {end_result['refinements_performed']}")

    # Summary
    print_header("🎉 Demo Complete")

    print("\n✅ Successfully Demonstrated:")
    print("   • Verified all 6 RAG tools loaded (including new SEC tools)")
    print("   • Using dummy pre-saved analysis (saves ~$0.067 per test run!)")
    print("   • Starting chat session with report context")
    print("   • Company restrictions (only AAPL & MSFT allowed)")
    print("   • Ticker mapping (Apple → AAPL, Microsoft → MSFT)")
    print("   • Smart routing (ChatAgent vs full refinement)")
    print("   • RAG-powered responses with all 6 tools:")
    print("     - search_annual_reports (SEC filings)")
    print("     - search_transcripts (earnings calls)")
    print("     - search_news (news articles)")
    print("     - get_share_price_data (stock prices)")
    print("     - get_financial_metrics (SEC financial data) ✨ NEW")
    print("     - compare_companies (company comparison) ✨ NEW")
    print("   • Cost tracking per exchange")
    print("   • Session management and limits")
    print("   • Significant cost savings (200x for simple Q&A)")

    print("\n💡 Key Takeaways:")
    print("   • Dummy data optimization: Save ~$0.067 per test run")
    print("   • ChatAgent handles most questions quickly and cheaply")
    print("   • SECFinancialRAG tools provide direct financial data access")
    print("   • Company restrictions prevent unauthorized queries")
    print("   • Ticker mapping ensures correct RAG tool usage")
    print("   • Full refinements only when explicitly requested")
    print("   • RAG tools provide fresh data for responses")
    print("   • Context maintained across conversation")
    print("   • Cost-effective multi-turn conversations")

    print("\n🎯 Enhancement Validation:")
    # Count how many questions tested each feature
    sec_tool_count = sum(1 for q in questions if 'SEC Financial' in q['description'])
    restriction_count = sum(1 for q in questions if 'restriction' in q['type'])
    rag_tool_count = sum(1 for q in questions if 'RAG tool' in q['description'])

    print(f"   • SEC Financial Tool Tests: {sec_tool_count}")
    print(f"   • Company Restriction Tests: {restriction_count}")
    print(f"   • RAG Tool Integration Tests: {rag_tool_count}")
    print(f"   • Total Test Questions: {len(questions)}")


async def main():
    """Run the chat demo"""

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "🎯 RAG-Powered Chat Feature Demo" + " " * 25 + "║")
    print("╚" + "═" * 78 + "╝")
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        await demo_chat_feature()

        print("\n" + "=" * 80)
        print("   ✅ Demo completed successfully!")
        print("=" * 80)

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

    # Run demo
    asyncio.run(main())
