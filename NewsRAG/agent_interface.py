"""
Agent-friendly interface for NewsRAG package
Provides simplified, structured outputs optimized for LangChain agents
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from .main import get_company_news_chunks
from .models import NewsCategory

logger = logging.getLogger(__name__)

# ---------- Thin Orchestration Functions (Moved to vector_enhanced_interface.py) ----------
# These functions are now handled by the vector-enhanced interface for better performance
# and persistent storage. The original implementations remain here as fallbacks.

def index_news_for_agent(ticker: str, days_back: int = 30, categories: Optional[List[str]] = None, max_articles: int = 50) -> Dict[str, Any]:
    """Fallback index function - used when vector enhancement unavailable"""
    categories = [c.lower() for c in (categories or ["general"])]
    result = get_company_news_chunks(companies=[ticker], categories=categories, days_back=days_back, max_articles_per_query=max_articles)
    if not result.get("success"):
        return {"success": False, "ticker": ticker, "chunks": [], "errors": result.get("errors", ["Failed to fetch news"])}
    return {"success": True, "ticker": ticker, "chunks": result.get("chunks", []), 
            "metadata": {"processing_time": result.get("processing_time_seconds", 0), "categories": categories, 
                        "days_back": days_back, "fetched_at": datetime.now().isoformat()}}


def search_news_for_agent(ticker: str, query: str, days_back: int = 30, k: int = 20, categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fallback search function - used when vector enhancement unavailable"""
    idx = index_news_for_agent(ticker, days_back=days_back, categories=categories)
    if not idx.get("success"):
        return idx
    chunks = idx.get("chunks", [])
    
    if not query:
        return {"success": True, "ticker": ticker, "query": query, "results": chunks[:k], "returned": min(k, len(chunks))}
    
    q, terms = query.lower(), [t for t in query.lower().split() if t]
    scored = [(sum((c.get("content", "") or "").lower().count(t) for t in terms) + (3 if q in (c.get("content", "") or "").lower() else 0), c) for c in chunks]
    scored = [sc for sc in scored if sc[0] > 0]
    top = [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)[:k]]
    
    return {"success": True, "ticker": ticker, "query": query, "results": top, "returned": len(top), 
            "total_candidates": len(chunks), "metadata": {"k": k, "days_back": days_back, "categories": categories or ["general"], 
                                                        "generated_at": datetime.now().isoformat()}}

# ---------- High-Level Wrapper Functions ----------

def get_news_for_agent(
    ticker: str, 
    days_back: int = 30,
    categories: List[str] = None,
    max_articles: int = 10
) -> Dict[str, Any]:
    """
    Agent-friendly news retrieval interface
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        days_back: Number of days to look back (default: 30)
        categories: List of news categories (default: ['general'])
        max_articles: Maximum number of articles to retrieve
        
    Returns:
        Structured dictionary optimized for agent consumption:
        {
            "success": bool,
            "ticker": str,
            "summary": str,              # Agent-consumable summary
            "key_points": List[str],     # Bullet points for analysis
            "sentiment": str,            # Overall sentiment
            "recent_developments": List[str],  # Recent key events
            "total_articles": int,
            "date_range": dict,
            "chunks": List[dict],        # Raw chunks for vector storage
            "metadata": dict             # Additional context
        }
    """
    
    try:
        # Set default categories if none provided
        if categories is None:
            categories = ["general"]
        
        # Validate and normalize categories
        valid_categories = []
        for cat in categories:
            try:
                valid_categories.append(cat.lower())
            except:
                continue
        
        if not valid_categories:
            valid_categories = ["general"]
        
        # Get raw news data
        result = get_company_news_chunks(
            companies=[ticker],
            categories=valid_categories,
            days_back=days_back,
            max_articles_per_query=max_articles
        )
        
        if not result["success"] or not result["chunks"]:
            return {
                "success": False,
                "ticker": ticker,
                "summary": f"No recent news found for {ticker} in the last {days_back} days",
                "key_points": [],
                "sentiment": "neutral",
                "recent_developments": [],
                "total_articles": 0,
                "date_range": None,
                "chunks": [],
                "metadata": {"error": "No news data available"},
                "errors": result.get("errors", ["No news data found"])
            }
        
        # Process chunks for agent consumption
        chunks = result["chunks"]
        processed_result = _process_chunks_for_agent(chunks, ticker, days_back)
        
        return {
            "success": True,
            "ticker": ticker,
            "summary": processed_result["summary"],
            "key_points": processed_result["key_points"],
            "sentiment": processed_result["sentiment"],
            "recent_developments": processed_result["recent_developments"],
            "total_articles": len(chunks),
            "date_range": result["summary"].get("date_range"),
            "chunks": chunks,  # For vector storage
            "metadata": {
                "processing_time": result.get("processing_time_seconds", 0),
                "companies_covered": result["summary"].get("companies_covered", []),
                "categories_searched": valid_categories
            },
            "errors": []
        }
        
    except Exception as e:
        error_msg = f"Error retrieving news for {ticker}: {str(e)}"
        logger.error(error_msg)
        
        return {
            "success": False,
            "ticker": ticker,
            "summary": f"Unable to retrieve news for {ticker} due to an error",
            "key_points": [],
            "sentiment": "neutral",
            "recent_developments": [],
            "total_articles": 0,
            "date_range": None,
            "chunks": [],
            "metadata": {"error": str(e)},
            "errors": [str(e)]
        }


def _process_chunks_for_agent(chunks: List[Dict], ticker: str, days_back: int) -> Dict[str, Any]:
    """
    Process raw news chunks into agent-friendly format
    """
    
    # Extract key information from chunks
    key_points = []
    recent_developments = []
    sentiment_scores = []
    
    # Sort chunks by date (most recent first)
    sorted_chunks = sorted(
        chunks, 
        key=lambda x: x.get("metadata", {}).get("published_date", ""), 
        reverse=True
    )
    
    # Process each chunk
    for i, chunk in enumerate(sorted_chunks[:10]):  # Limit to top 10 for summary
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        
        # Extract key points (first sentence of each article)
        if content:
            first_sentence = content.split('.')[0].strip()
            if len(first_sentence) > 20 and first_sentence not in key_points:
                key_points.append(f"• {first_sentence}")
        
        # Extract recent developments (last 7 days get special treatment)
        pub_date_str = metadata.get("published_date", "")
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                days_ago = (datetime.now().replace(tzinfo=pub_date.tzinfo) - pub_date).days
                
                if days_ago <= 7 and content:  # Recent developments
                    title = metadata.get("title", "")
                    if title and title not in recent_developments:
                        recent_developments.append(title)
            except:
                pass
        
        # Collect sentiment if available
        sentiment = metadata.get("sentiment", "neutral")
        if sentiment in ["positive", "negative", "neutral"]:
            sentiment_scores.append(sentiment)
    
    # Analyze overall sentiment
    overall_sentiment = _analyze_overall_sentiment(sentiment_scores)
    
    # Create summary
    summary = _create_news_summary(ticker, len(chunks), days_back, overall_sentiment, recent_developments)
    
    return {
        "summary": summary,
        "key_points": key_points[:8],  # Limit to 8 key points
        "sentiment": overall_sentiment,
        "recent_developments": recent_developments[:5]  # Limit to 5 recent developments
    }


def _analyze_overall_sentiment(sentiment_scores: List[str]) -> str:
    """
    Analyze overall sentiment from individual sentiment scores
    """
    if not sentiment_scores:
        return "neutral"
    
    positive_count = sentiment_scores.count("positive")
    negative_count = sentiment_scores.count("negative")
    neutral_count = sentiment_scores.count("neutral")
    
    total = len(sentiment_scores)
    
    # Determine overall sentiment
    if positive_count > negative_count and positive_count / total > 0.4:
        return "positive"
    elif negative_count > positive_count and negative_count / total > 0.4:
        return "negative"
    else:
        return "neutral"


def _create_news_summary(ticker: str, article_count: int, days_back: int, sentiment: str, recent_developments: List[str]) -> str:
    """
    Create a concise news summary for agent consumption
    """
    
    # Base summary
    summary_parts = [
        f"Found {article_count} news articles for {ticker} over the last {days_back} days."
    ]
    
    # Add sentiment
    if sentiment != "neutral":
        summary_parts.append(f"Overall news sentiment is {sentiment}.")
    
    # Add recent developments
    if recent_developments:
        summary_parts.append(f"Recent developments include: {', '.join(recent_developments[:3])}.")
    
    # Add guidance for further analysis
    if article_count > 0:
        summary_parts.append("Key themes cover business performance, market conditions, and strategic initiatives.")
    
    return " ".join(summary_parts)


def get_news_sentiment_analysis(ticker: str, days_back: int = 7) -> Dict[str, Any]:
    """
    Get focused sentiment analysis for a company
    
    Args:
        ticker: Company ticker symbol
        days_back: Days to analyze (default: 7 for recent sentiment)
        
    Returns:
        Sentiment-focused analysis for agents
    """
    
    news_data = get_news_for_agent(ticker, days_back=days_back, max_articles=20)
    
    if not news_data["success"]:
        return {
            "ticker": ticker,
            "sentiment": "neutral",
            "confidence": "low",
            "analysis": f"Unable to analyze sentiment for {ticker} due to lack of recent news",
            "supporting_evidence": []
        }
    
    # Enhanced sentiment analysis
    chunks = news_data["chunks"]
    positive_news = []
    negative_news = []
    
    for chunk in chunks:
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        sentiment = metadata.get("sentiment", "neutral")
        title = metadata.get("title", "")
        
        if sentiment == "positive" and title:
            positive_news.append(title)
        elif sentiment == "negative" and title:
            negative_news.append(title)
    
    # Confidence based on data availability
    confidence = "high" if len(chunks) >= 5 else "medium" if len(chunks) >= 2 else "low"
    
    return {
        "ticker": ticker,
        "sentiment": news_data["sentiment"],
        "confidence": confidence,
        "analysis": f"{ticker} sentiment over {days_back} days: {news_data['sentiment']} ({len(chunks)} articles analyzed)",
        "supporting_evidence": {
            "positive_headlines": positive_news[:3],
            "negative_headlines": negative_news[:3],
            "recent_developments": news_data["recent_developments"]
        },
        "recommendation": _get_sentiment_recommendation(news_data["sentiment"], confidence)
    }


def _get_sentiment_recommendation(sentiment: str, confidence: str) -> str:
    """Generate recommendation based on sentiment and confidence"""
    
    if confidence == "low":
        return "Insufficient recent news for reliable sentiment assessment. Consider expanding date range."
    
    recommendations = {
        "positive": "Recent news sentiment is favorable. Monitor for sustainability of positive trends.",
        "negative": "Recent news sentiment is concerning. Investigate specific issues and potential impact.",
        "neutral": "News sentiment is balanced. Look for underlying trends in business fundamentals."
    }
    
    return recommendations.get(sentiment, "Monitor news sentiment for changes in market perception.")