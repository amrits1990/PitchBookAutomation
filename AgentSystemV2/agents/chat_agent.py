"""
Chat Agent - Lightweight conversational agent for post-report Q&A
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base_agent import BaseFinancialAgent
from config.schemas import ResearchReportOutput
from config.settings import DEFAULT_MODEL_ID
from config.llm_cost_tracker import TokenUsage


class ChatAgent(BaseFinancialAgent):
    """
    Lightweight chat agent for answering questions about completed analysis reports.

    Features:
    - RAG tool access for live data retrieval
    - Maintains report context and chat history
    - Cost-effective responses using cheaper model
    - Fast response times (2-5 seconds typical)

    Use Cases:
    - Answer clarifying questions about the report
    - Provide additional details on specific metrics
    - Search for relevant information using RAG tools
    - Explain findings in different ways
    """

    def __init__(self,
                 domain: str,
                 company: str,
                 report_context: str,
                 allowed_tickers: list = None,
                 ticker_to_name: dict = None,
                 enable_debug: bool = False):
        """
        Initialize chat agent with report context.

        Args:
            domain: Domain of the analysis (liquidity, leverage, etc.)
            company: Company ticker/name (target company)
            report_context: Compressed report summary for context
            allowed_tickers: List of allowed tickers (target + peers) for company restrictions
            ticker_to_name: Mapping of tickers to company names (e.g., {"AAPL": "Apple", "MSFT": "Microsoft"})
            enable_debug: Enable debug logging
        """
        self.domain = domain
        self.company = company
        self.report_context = report_context
        self.allowed_tickers = allowed_tickers or [company]  # Default: only target company
        self.ticker_to_name = ticker_to_name or {}

        # Use cheaper model for chat (cost-effective)
        chat_model = "openai/gpt-4o-mini"

        super().__init__(
            agent_name=f"{domain}_chat_agent",
            model_id=chat_model,
            enable_debug=enable_debug,
            use_memory=False,  # We'll manage context explicitly
            max_tokens=2000  # Shorter responses for chat
        )

        self.logger.info(f"ChatAgent initialized for {company} - {domain}")

    def _get_agent_instructions(self) -> str:
        """Get chat-optimized system instructions."""

        # Format allowed companies
        allowed_companies_str = ", ".join(self.allowed_tickers)
        ticker_mapping_str = "\n".join([f"  • {name} → {ticker}" for ticker, name in self.ticker_to_name.items()])
        if not ticker_mapping_str:
            ticker_mapping_str = f"  • {self.company} (target company)"

        return f"""
You are a financial analysis assistant helping users understand a {self.domain} analysis report for {self.company}.

═══════════════════════════════════════════════════════════════
                    🎯 YOUR ROLE
═══════════════════════════════════════════════════════════════

You're having a conversation about an ALREADY COMPLETED analysis report.
Your job is to:
1. Answer questions about the report clearly and concisely
2. Provide additional context or clarification
3. Use RAG tools to fetch supporting information when needed
4. Help users understand specific metrics, ratios, or findings

═══════════════════════════════════════════════════════════════
                    🚫 COMPANY RESTRICTIONS
═══════════════════════════════════════════════════════════════

IMPORTANT: You can ONLY answer questions about these companies:
{allowed_companies_str}

If a user asks about ANY other company, politely decline and explain that you can only discuss the companies in this report.

Example response for unauthorized company:
"I can only answer questions about {allowed_companies_str}, which are the companies covered in this report. I cannot provide information about other companies. Is there something specific you'd like to know about {self.company}?"

═══════════════════════════════════════════════════════════════
                    📊 REPORT CONTEXT
═══════════════════════════════════════════════════════════════

{self.report_context}

═══════════════════════════════════════════════════════════════
                    🏢 TICKER MAPPING
═══════════════════════════════════════════════════════════════

When calling RAG tools, ALWAYS use ticker symbols (not company names).
Here's the mapping:

{ticker_mapping_str}

If a user asks about "Apple", use "AAPL" in tool calls.
If a user asks about "Microsoft", use "MSFT" in tool calls.

═══════════════════════════════════════════════════════════════
                    🔧 AVAILABLE RAG TOOLS - DETAILED USAGE
═══════════════════════════════════════════════════════════════

You have 6 RAG tools available. Use the EXACT arguments shown below:

1️⃣  search_annual_reports(ticker, query, k=5, time_period="latest")
   Purpose: Search SEC 10-K/10-Q filings for specific information
   Arguments:
     • ticker (str): Company ticker (e.g., "AAPL")
     • query (str): What to search for (e.g., "debt covenants")
     • k (int, optional): Number of results (default: 5)
     • time_period (str, optional): "latest", "latest_10k", "latest_10q", "last_3_reports"
   Example: search_annual_reports("AAPL", "cash management strategy", k=3, time_period="latest")

2️⃣  search_transcripts(ticker, query, quarters_back=4, k=5)
   Purpose: Search earnings call transcripts for management commentary
   Arguments:
     • ticker (str): Company ticker (e.g., "MSFT")
     • query (str): What to search for (e.g., "liquidity strategy")
     • quarters_back (int, optional): How many quarters to search (default: 4)
     • k (int, optional): Number of results (default: 5)
   Example: search_transcripts("MSFT", "working capital management", quarters_back=4, k=5)

3️⃣  search_news(ticker, query, days_back=30)
   Purpose: Search recent news articles
   Arguments:
     • ticker (str): Company ticker (e.g., "AAPL")
     • query (str): What to search for (e.g., "acquisitions")
     • days_back (int, optional): Days of history (default: 30)
   Example: search_news("AAPL", "debt refinancing", days_back=60)

4️⃣  get_share_price_data(ticker, days_back=30)
   Purpose: Get stock price data, trends, and volatility
   Arguments:
     • ticker (str): Company ticker (e.g., "AAPL")
     • days_back (int, optional): Days of history (default: 30)
   Example: get_share_price_data("AAPL", days_back=90)

5️⃣  get_financial_metrics(ticker, metrics, period='latest')
   Purpose: Fetch specific financial metrics from SEC filings
   Arguments:
     • ticker (str): Company ticker (e.g., "AAPL")
     • metrics (list): Metric names (e.g., ["revenue", "net_income", "total_assets"])
     • period (str, optional): Period specification (default: 'latest')
         Valid formats:
         - 'latest': Most recent period only
         - 'FY2024': Specific fiscal year
         - 'Q2-2025': Specific quarter
         - 'last 8 quarters': Last n quarters (n=1-40)
         - 'last 3 financial years': Last n years (n=1-10)
   Available metrics: revenue, net_income, total_assets, total_liabilities,
                     stockholders_equity, operating_cash_flow, free_cash_flow, etc.
   Example: get_financial_metrics("AAPL", ["revenue", "net_income"], period='last 8 quarters')

6️⃣  compare_companies(ticker_list, categories, period='latest')
   Purpose: Compare financial ratios across multiple companies for specific categories
   Arguments:
     • ticker_list (list): List of tickers (e.g., ["AAPL", "MSFT"])
     • categories (list): List of ratio categories, e.g.:
         - "liquidity", "profitability", "leverage"
         - "efficiency", "valuation", "growth", "coverage"
     • period (str, optional): Period specification (default: 'latest')
         Valid formats:
         - 'latest': Most recent period only
         - 'FY2024': Specific fiscal year
         - 'Q2-2025': Specific quarter
         - 'last 4 quarters': Last n quarters (n=1-40)
         - 'last 3 financial years': Last n years (n=1-10)
   Example: compare_companies(["AAPL", "MSFT"], ["liquidity", "profitability"], period='latest')

═══════════════════════════════════════════════════════════════
                    🎯 WHEN TO USE EACH TOOL
═══════════════════════════════════════════════════════════════

Use search_annual_reports when:
  • User asks about financial statement details
  • User wants specific filing information
  • User asks about accounting policies, debt covenants, or legal matters

Use search_transcripts when:
  • User asks for management commentary
  • User wants CEO/CFO quotes or explanations
  • User asks about forward guidance or strategic initiatives

Use search_news when:
  • User asks about recent events
  • User wants market reaction or analyst opinions
  • User asks about M&A, partnerships, or announcements

Use get_share_price_data when:
  • User asks about stock performance
  • User wants price trends or volatility
  • User asks about market valuation changes

Use get_financial_metrics when:
  • User asks for specific raw metrics (revenue, income, assets)
  • User wants historical trends for specific metrics
  • User needs metric values not in the report

Use compare_companies when:
  • User asks for peer comparison
  • User wants to see how companies stack up by domain
  • User asks "How does X compare to Y in liquidity/leverage?"

═══════════════════════════════════════════════════════════════
                    💬 CONVERSATION GUIDELINES
═══════════════════════════════════════════════════════════════

DO:
✓ Be conversational and helpful
✓ Reference specific findings from the report
✓ Use RAG tools to support your answers
✓ ALWAYS use ticker symbols when calling tools (not company names!)
✓ Check if company is in allowed list before answering
✓ Admit when you need to make a tool call for current data
✓ Keep responses concise (2-4 paragraphs typical)
✓ Cite sources when using tool results

DON'T:
✗ Answer questions about companies not in the allowed list
✗ Use company names in tool calls (use tickers!)
✗ Regenerate the entire analysis (that's expensive!)
✗ Make up data - use supplied report context or tools
✗ Provide investment advice or recommendations beyond the report
✗ Ignore the report context - you're discussing THIS specific report

═══════════════════════════════════════════════════════════════
                    📝 RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

Structure your responses as:
1. Direct answer to the question
2. Supporting details from report or tools
3. Additional context if relevant
4. Suggest related questions if appropriate

Example:
"The report shows Apple's current ratio is 1.07 (Q2-2024), which indicates
adequate liquidity. This is based on current assets of $143B and current
liabilities of $133B. Compared to Microsoft's 1.35, Apple's position is slightly
weaker but still healthy. Would you like me to search for management commentary
on their liquidity strategy?"

Remember: You're a helpful assistant discussing an existing report, not creating new analysis.
"""

    async def respond_to_question(self,
                                  user_question: str,
                                  chat_history: list = None) -> tuple[str, List[Dict[str, Any]], TokenUsage]:
        """
        Respond to user question about the report.

        Args:
            user_question: User's question about the report
            chat_history: Previous chat exchanges for context

        Returns:
            Tuple of (response text, tool calls made, token usage)
        """
        self.logger.info(f"Processing chat question: {user_question[:50]}...")

        # Build context from chat history
        context_items = []
        if chat_history:
            # Include last 5 exchanges for context (sliding window)
            recent_history = chat_history[-5:]
            for exchange in recent_history:
                if exchange.get("type") == "user":
                    context_items.append(f"User: {exchange['message']}")
                elif exchange.get("type") == "assistant":
                    # Extract just the message content
                    msg = exchange.get("response", {}).get("message", "")
                    context_items.append(f"Assistant: {msg}")

        # Create prompt
        if context_items:
            conversation_context = "\n".join(context_items[-10:])  # Last 10 lines
            prompt = f"""
Previous conversation:
{conversation_context}

Current question: {user_question}

Please respond to the current question, considering the conversation context and the report information provided in your instructions.
"""
        else:
            prompt = f"""
User question: {user_question}

Please respond to this question about the {self.domain} analysis report for {self.company}.
"""

        try:
            # Run agent
            response_text, tool_calls, token_usage = await self._run_agent(prompt)

            # Log tool usage
            if tool_calls:
                self.logger.info(f"Chat agent made {len(tool_calls)} tool calls")
                for i, tool_call in enumerate(tool_calls, 1):
                    self.logger.debug(f"  Tool {i}: {tool_call}")

            # Add source citations if tools were used
            if tool_calls and response_text:
                citations = self._format_source_citations(tool_calls)
                if citations:
                    response_text = f"{response_text}\n\n{citations}"

            return response_text, tool_calls, token_usage

        except Exception as e:
            self.logger.error(f"Chat response failed: {e}")
            # Return error message with empty token usage
            error_response = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
            return error_response, [], TokenUsage(model_id=self.model_id)

    def _format_source_citations(self, tool_calls: List[Dict[str, Any]]) -> str:
        """
        Format tool calls as source citations to append to responses.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            Formatted citation string
        """
        if not tool_calls:
            return ""

        # Group tool calls by function name
        tool_summary = {}
        for tool_call in tool_calls:
            function_name = tool_call.get('function', 'unknown_tool')
            # Map internal function names to user-friendly names
            display_name = self._get_tool_display_name(function_name)

            if display_name not in tool_summary:
                tool_summary[display_name] = 0
            tool_summary[display_name] += 1

        # Format citations
        citation_parts = []
        for tool_name, count in tool_summary.items():
            if count == 1:
                citation_parts.append(tool_name)
            else:
                citation_parts.append(f"{tool_name} ({count}x)")

        if citation_parts:
            return f"**Sources:** {', '.join(citation_parts)}"
        return ""

    def _get_tool_display_name(self, function_name: str) -> str:
        """Map internal function names to user-friendly display names."""
        name_mapping = {
            'search_annual_reports': 'SEC Filings',
            'search_transcripts': 'Earnings Transcripts',
            'search_news': 'News Articles',
            'get_share_price_data': 'Stock Price Data',
            'get_financial_metrics': 'Financial Metrics',
            'compare_companies': 'Company Comparison'
        }
        return name_mapping.get(function_name, function_name.replace('_', ' ').title())

    def update_report_context(self, new_context: str):
        """
        Update the report context (useful if report is refined).

        Args:
            new_context: Updated report context
        """
        self.report_context = new_context
        self.logger.info("Report context updated")

        # Note: Agent instructions are set at initialization
        # For dynamic context updates, we inject it into the prompt instead


def create_report_chat_context(report: ResearchReportOutput, max_tokens: int = 2000) -> tuple[str, list, dict]:
    """
    Create compressed report context for chat agent (target: <2000 tokens).

    Args:
        report: ResearchReportOutput object
        max_tokens: Target token limit for context

    Returns:
        Tuple of (context_string, allowed_tickers, ticker_to_name_mapping)
    """
    context_lines = []

    # Basic info
    context_lines.append(f"Company: {report.company}")
    context_lines.append(f"Domain: {report.domain}")
    context_lines.append(f"Confidence: {report.confidence_level}")
    context_lines.append(f"Analysis Date: {report.analysis_timestamp[:10]}")
    context_lines.append("")

    # Executive summary
    context_lines.append("EXECUTIVE SUMMARY:")
    context_lines.append(report.executive_summary)
    context_lines.append("")

    # Key ratios
    if report.chartable_ratios:
        context_lines.append("KEY RATIOS:")
        for ratio in report.chartable_ratios[:3]:  # Top 3
            latest_quarters = sorted(ratio.company_values.items(), reverse=True)[:2]
            values_str = ", ".join([f"{q}: {v:.3f}" for q, v in latest_quarters])
            context_lines.append(f"  • {ratio.ratio_name}: {values_str}")
            context_lines.append(f"    Trend: {ratio.trend_direction}")
        context_lines.append("")

    # Key metrics
    if report.chartable_metrics:
        context_lines.append("KEY METRICS:")
        for metric in report.chartable_metrics[:3]:  # Top 3
            latest_quarters = sorted(metric.values.items(), reverse=True)[:2]
            values_str = ", ".join([f"{q}: {v:,.0f}" for q, v in latest_quarters])
            context_lines.append(f"  • {metric.metric_name}: {values_str}")
            context_lines.append(f"    Trend: {metric.trend_direction}")
        context_lines.append("")

    # Analysis bullets (condensed)
    if report.analysis_bullets:
        context_lines.append("KEY FINDINGS:")
        for i, bullet in enumerate(report.analysis_bullets[:5], 1):  # Top 5
            context_lines.append(f"  {i}. {bullet.bullet_text}")
        context_lines.append("")

    # Recommendations
    if report.recommendations:
        context_lines.append("RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations[:3], 1):  # Top 3
            context_lines.append(f"  {i}. {rec}")
        context_lines.append("")

    # Risk factors
    if report.risk_factors:
        context_lines.append("RISK FACTORS:")
        for i, risk in enumerate(report.risk_factors[:3], 1):  # Top 3
            context_lines.append(f"  {i}. {risk}")
        context_lines.append("")

    # Tool calls summary
    if hasattr(report, 'tool_calls_made') and report.tool_calls_made:
        tool_summary = {}
        for tool_call in report.tool_calls_made:
            # Handle both string and dict formats
            if isinstance(tool_call, dict):
                # Extract tool name from dict (check common keys)
                tool_name = tool_call.get('tool_name') or tool_call.get('name') or tool_call.get('function', 'unknown')
            elif isinstance(tool_call, str):
                # Extract tool name from string format "tool_name(args)"
                tool_name = tool_call.split('(')[0] if '(' in tool_call else tool_call
            else:
                # Unknown format, convert to string
                tool_name = str(tool_call)

            tool_summary[tool_name] = tool_summary.get(tool_name, 0) + 1

        context_lines.append("DATA SOURCES:")
        for tool_name, count in tool_summary.items():
            context_lines.append(f"  • {tool_name}: {count} calls")

    # Extract ticker information for chat restrictions
    allowed_tickers = [report.company]  # Start with target company
    ticker_to_name = {}

    # Add target company to mapping
    # Try to extract company name from various sources
    target_ticker = report.company
    ticker_to_name[target_ticker] = target_ticker  # Default: use ticker as name

    # Extract peer companies from chartable_ratios if available
    if hasattr(report, 'chartable_ratios') and report.chartable_ratios:
        # Look for peer data in ratio objects
        for ratio in report.chartable_ratios:
            if hasattr(ratio, 'peer_values') and ratio.peer_values:
                for peer_ticker, peer_data in ratio.peer_values.items():
                    if peer_ticker not in allowed_tickers:
                        allowed_tickers.append(peer_ticker)
                        # Try to extract peer name
                        if isinstance(peer_data, dict) and 'company_name' in peer_data:
                            ticker_to_name[peer_ticker] = peer_data['company_name']
                        else:
                            ticker_to_name[peer_ticker] = peer_ticker

    # Also check for peers_analyzed field (common pattern)
    if hasattr(report, 'peers_analyzed') and report.peers_analyzed:
        for peer in report.peers_analyzed:
            if isinstance(peer, dict):
                peer_ticker = peer.get('ticker') or peer.get('company')
                peer_name = peer.get('name') or peer_ticker
            else:
                peer_ticker = str(peer)
                peer_name = peer_ticker

            if peer_ticker and peer_ticker not in allowed_tickers:
                allowed_tickers.append(peer_ticker)
                ticker_to_name[peer_ticker] = peer_name

    # Try common company name mappings for well-known tickers
    common_mappings = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "AMZN": "Amazon",
        "META": "Meta",
        "TSLA": "Tesla",
        "NVDA": "NVIDIA",
        "JPM": "JPMorgan",
        "V": "Visa",
        "WMT": "Walmart"
    }

    # Update ticker_to_name with common mappings if ticker exists
    for ticker in allowed_tickers:
        if ticker in common_mappings and (ticker not in ticker_to_name or ticker_to_name[ticker] == ticker):
            ticker_to_name[ticker] = common_mappings[ticker]

    context_string = "\n".join(context_lines)

    return context_string, allowed_tickers, ticker_to_name
