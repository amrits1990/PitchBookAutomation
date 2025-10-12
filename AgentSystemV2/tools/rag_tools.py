"""
Clean RAG Tools Integration for AgentSystemV2
Provides simple, clean access to all RAG tools with proper error handling.
"""

import sys
import os
import logging
from typing import Dict, Any, List, Optional

# Add parent directories to path to access RAG systems (once at module level)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)

# Import RAG systems with clean error handling (once at module level)
try:
    from AnnualReportRAG.agent_interface import search_report_for_agent
    ANNUAL_RAG_AVAILABLE = True
    logger.debug("AnnualReportRAG imported successfully")
except ImportError as e:
    logger.warning(f"AnnualReportRAG not available: {e}")
    ANNUAL_RAG_AVAILABLE = False

try:
    from TranscriptRAG.agent_interface import search_transcripts_for_agent
    TRANSCRIPT_RAG_AVAILABLE = True
    logger.debug("TranscriptRAG imported successfully")
except ImportError as e:
    logger.warning(f"TranscriptRAG not available: {e}")
    TRANSCRIPT_RAG_AVAILABLE = False

try:
    from NewsRAG.agent_interface import search_news_for_agent
    NEWS_RAG_AVAILABLE = True
    logger.debug("NewsRAG imported successfully")
except ImportError as e:
    logger.warning(f"NewsRAG not available: {e}")
    NEWS_RAG_AVAILABLE = False

try:
    from SharePriceRAG.agent_interface import get_price_analysis_for_agent
    SHARE_PRICE_RAG_AVAILABLE = True
    logger.debug("SharePriceRAG imported successfully")
except ImportError as e:
    logger.warning(f"SharePriceRAG not available: {e}")
    SHARE_PRICE_RAG_AVAILABLE = False

try:
    from SECFinancialRAG.agent_interface import get_financial_metrics_for_agent, compare_companies_for_agent
    SEC_FINANCIAL_RAG_AVAILABLE = True
    logger.debug("SECFinancialRAG imported successfully")
except ImportError as e:
    logger.warning(f"SECFinancialRAG not available: {e}")
    SEC_FINANCIAL_RAG_AVAILABLE = False


# Wrapper functions that agents can use directly
def search_annual_reports(ticker: str, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None, time_period: str = "latest") -> Dict[str, Any]:
    """Search annual reports for a specific company and query.
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        query: Search query text
        k: Number of results to return (default: 5)
        filters: Optional search filters
        time_period: Time period to search. Accepts:
            - "latest" (default): Most recent filings
            - "latest_10k": Latest 10-K only
            - "latest_10q": Latest 10-Q only  
            - "latest_10k_and_10q": Latest 10-K and 10-Q
            - "last_3_reports": Last 3 reports (or any number)
            - Also accepts: "last 3 years", "last 5", "3 years", etc. (auto-converted)
    
    Returns:
        Dict with search results and metadata
    """
    if not ANNUAL_RAG_AVAILABLE:
        return {"error": "AnnualReportRAG not available", "success": False}
    
    try:
        # Validate and convert time_period parameter before calling the underlying function
        import re
        valid_time_periods = ["latest", "latest_10k_and_10q", "latest_10k", "latest_10q", "last_n_reports"]
        last_n_pattern = re.compile(r'^last_\d+_reports$')
        
        # Convert common agent variations to valid format
        original_time_period = time_period
        
        # Handle "last X years" -> "last_X_reports"
        if re.match(r'^last\s+\d+\s+years?$', time_period.lower()):
            number = re.search(r'\d+', time_period).group()
            time_period = f"last_{number}_reports"
            logger.info(f"Converted time_period from '{original_time_period}' to '{time_period}'")
        
        # Handle "last X" -> "last_X_reports"  
        elif re.match(r'^last\s+\d+$', time_period.lower()):
            number = re.search(r'\d+', time_period).group()
            time_period = f"last_{number}_reports"
            logger.info(f"Converted time_period from '{original_time_period}' to '{time_period}'")
        
        # Handle variations like "3 years", "5 reports", etc.
        elif re.match(r'^\d+\s+(years?|reports?)$', time_period.lower()):
            number = re.search(r'^\d+', time_period).group()
            time_period = f"last_{number}_reports"
            logger.info(f"Converted time_period from '{original_time_period}' to '{time_period}'")
        
        # Final validation
        if time_period not in valid_time_periods and not last_n_pattern.match(time_period):
            logger.warning(f"Invalid time_period '{time_period}', using 'latest' instead")
            time_period = "latest"
        
        result = search_report_for_agent(ticker, query, k, filters, time_period)
        return {"result": result, "success": True, "tool": "search_annual_reports"}
    except Exception as e:
        logger.error(f"Error in search_annual_reports: {e}")
        return {"error": str(e), "success": False, "tool": "search_annual_reports"}


def search_transcripts(ticker: str, query: str, quarters_back: int = 4, k: int = 5) -> Dict[str, Any]:
    """Search earnings call transcripts for a specific company and query."""
    if not TRANSCRIPT_RAG_AVAILABLE:
        return {"error": "TranscriptRAG not available", "success": False}

    try:
        # Fix: Use keyword arguments to avoid passing k to the wrong parameter
        result = search_transcripts_for_agent(
            ticker=ticker,
            query=query,
            quarters_back=quarters_back,
            k=k
        )
        return {"result": result, "success": True, "tool": "search_transcripts"}
    except Exception as e:
        logger.error(f"Error in search_transcripts: {e}")
        return {"error": str(e), "success": False, "tool": "search_transcripts"}


def search_news(ticker: str, query: str, days_back: int = 30) -> Dict[str, Any]:
    """Search news articles for a specific company and query."""
    if not NEWS_RAG_AVAILABLE:
        return {"error": "NewsRAG not available", "success": False}
    
    try:
        result = search_news_for_agent(ticker, query, days_back)
        return {"result": result, "success": True, "tool": "search_news"}
    except Exception as e:
        logger.error(f"Error in search_news: {e}")
        return {"error": str(e), "success": False, "tool": "search_news"}


def get_share_price_data(ticker: str, days_back: int = 30) -> Dict[str, Any]:
    """Get share price data and analysis for a specific company."""
    if not SHARE_PRICE_RAG_AVAILABLE:
        return {"error": "SharePriceRAG not available", "success": False}

    try:
        result = get_price_analysis_for_agent(ticker, days_back)
        return {"result": result, "success": True, "tool": "get_share_price_data"}
    except Exception as e:
        logger.error(f"Error in get_share_price_data: {e}")
        return {"error": str(e), "success": False, "tool": "get_share_price_data"}


def get_financial_metrics(ticker: str, metrics: List[str], period: str = 'latest') -> Dict[str, Any]:
    """
    Get financial metrics for a specific company from SEC filings.

    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        metrics: List of metric names (e.g., ['revenue', 'net_income', 'total_assets'])
        period: Period specification (default: 'latest'). Valid formats:
            - 'latest': Most recent period
            - 'FY2024': Specific fiscal year
            - 'Q2-2025': Specific quarter
            - 'last 8 quarters': Last n quarters (n=1-40)
            - 'last 3 financial years': Last n years (n=1-10)

    Returns:
        Dict with metrics data organized by period

    Example:
        get_financial_metrics('AAPL', ['revenue', 'net_income'], period='last 8 quarters')
    """
    if not SEC_FINANCIAL_RAG_AVAILABLE:
        return {"error": "SECFinancialRAG not available", "success": False}

    try:
        # Ensure period is a string
        period_str = str(period) if period else 'latest'
        response = get_financial_metrics_for_agent(ticker, metrics, period_str)

        # Handle FinancialAgentResponse dataclass
        if hasattr(response, 'success'):
            if response.success:
                return {"result": response.data, "success": True, "tool": "get_financial_metrics"}
            else:
                return {"error": response.error, "success": False, "tool": "get_financial_metrics"}
        else:
            # Fallback for dict responses
            return {"result": response, "success": True, "tool": "get_financial_metrics"}
    except Exception as e:
        logger.error(f"Error in get_financial_metrics: {e}")
        return {"error": str(e), "success": False, "tool": "get_financial_metrics"}


def compare_companies(ticker_list: List[str], categories: List[str], period: str = 'latest') -> Dict[str, Any]:
    """
    Compare financial ratios across multiple companies for specific categories.

    Args:
        ticker_list: List of company tickers (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        categories: List of ratio categories - e.g., ['liquidity', 'profitability', 'leverage',
                   'efficiency', 'valuation', 'growth', 'coverage']
        period: Period specification (default: 'latest'). Valid formats:
            - 'latest': Most recent period
            - 'FY2024': Specific fiscal year
            - 'Q2-2025': Specific quarter
            - 'last 4 quarters': Last n quarters (n=1-40)
            - 'last 3 financial years': Last n years (n=1-10)

    Returns:
        Dict with comparative ratio data across companies

    Example:
        compare_companies(['AAPL', 'MSFT'], ['liquidity', 'profitability'], period='latest')
    """
    if not SEC_FINANCIAL_RAG_AVAILABLE:
        return {"error": "SECFinancialRAG not available", "success": False}

    try:
        # Ensure period is a string
        period_str = str(period) if period else 'latest'
        response = compare_companies_for_agent(ticker_list, categories, period_str)

        # Handle FinancialAgentResponse dataclass
        if hasattr(response, 'success'):
            if response.success:
                return {"result": response.data, "success": True, "tool": "compare_companies"}
            else:
                return {"error": response.error, "success": False, "tool": "compare_companies"}
        else:
            # Fallback for dict responses
            return {"result": response, "success": True, "tool": "compare_companies"}
    except Exception as e:
        logger.error(f"Error in compare_companies: {e}")
        return {"error": str(e), "success": False, "tool": "compare_companies"}


# List of available tools for agents
AVAILABLE_RAG_TOOLS = []

if ANNUAL_RAG_AVAILABLE:
    AVAILABLE_RAG_TOOLS.append(search_annual_reports)

if TRANSCRIPT_RAG_AVAILABLE:
    AVAILABLE_RAG_TOOLS.append(search_transcripts)

if NEWS_RAG_AVAILABLE:
    AVAILABLE_RAG_TOOLS.append(search_news)

if SHARE_PRICE_RAG_AVAILABLE:
    AVAILABLE_RAG_TOOLS.append(get_share_price_data)

if SEC_FINANCIAL_RAG_AVAILABLE:
    AVAILABLE_RAG_TOOLS.append(get_financial_metrics)
    AVAILABLE_RAG_TOOLS.append(compare_companies)


def get_rag_tools_status() -> Dict[str, bool]:
    """Get the availability status of all RAG tools."""
    return {
        "annual_reports": ANNUAL_RAG_AVAILABLE,
        "transcripts": TRANSCRIPT_RAG_AVAILABLE,
        "news": NEWS_RAG_AVAILABLE,
        "share_price": SHARE_PRICE_RAG_AVAILABLE,
        "sec_financial": SEC_FINANCIAL_RAG_AVAILABLE,
        "total_available": len(AVAILABLE_RAG_TOOLS)
    }


# Export all tools
__all__ = [
    'search_annual_reports',
    'search_transcripts',
    'search_news',
    'get_share_price_data',
    'get_financial_metrics',
    'compare_companies',
    'AVAILABLE_RAG_TOOLS',
    'get_rag_tools_status'
]