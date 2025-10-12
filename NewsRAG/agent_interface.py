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


def get_news_by_date_range_and_topic(
    company: str,
    start_date: str,
    end_date: str,
    topic: str,
    max_articles: int = 20
) -> Dict[str, Any]:
    """
    Enhanced agent function to retrieve news for any company between specific dates on a particular topic
    
    Args:
        company: Company name or ticker symbol (e.g., 'Apple', 'AAPL', 'Tesla', 'TSLA')
        start_date: Start date in 'YYYY-MM-DD' format (e.g., '2024-01-01')
        end_date: End date in 'YYYY-MM-DD' format (e.g., '2024-01-31')
        topic: News topic from available options: 'earnings', 'acquisitions', 'partnerships', 
               'products', 'leadership', 'regulatory', 'market', 'financial', 'business', 'general'
        max_articles: Maximum number of articles to retrieve (default: 20)
        
    Returns:
        Comprehensive dictionary optimized for agent consumption:
        {
            "success": bool,
            "company": str,
            "date_range": {"start": str, "end": str},
            "topic": str,
            "summary": str,                    # Agent-consumable overview
            "articles_found": int,
            "key_headlines": List[str],        # Important headlines
            "key_insights": List[str],         # Extracted insights
            "sentiment": str,                  # Overall sentiment
            "date_distribution": Dict,         # News distribution over time
            "source_breakdown": Dict,          # News sources analysis
            "chunks": List[dict],              # RAG-ready chunks
            "metadata": dict,                  # Processing info
            "available_topics": List[str],     # Valid topic options
            "errors": List[str]
        }
    """
    
    try:
        # Validate inputs
        validation_result = _validate_date_range_inputs(company, start_date, end_date, topic)
        if not validation_result["valid"]:
            return _create_error_response(company, start_date, end_date, topic, validation_result["errors"])
        
        # Calculate days back from date range
        from datetime import datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        current_dt = datetime.now()
        
        # Calculate days back from current date to start date
        days_back = (current_dt - start_dt).days
        days_back = max(1, days_back)  # Ensure at least 1 day
        
        logger.info(f"Getting news for {company} from {start_date} to {end_date} on topic '{topic}' (days_back: {days_back})")
        
        # Get raw news data using existing function
        result = get_company_news_chunks(
            companies=[company],
            categories=[topic],
            days_back=days_back,
            max_articles_per_query=max_articles
        )
        
        if not result["success"] or not result["chunks"]:
            return {
                "success": False,
                "company": company,
                "date_range": {"start": start_date, "end": end_date},
                "topic": topic,
                "summary": f"No news found for {company} between {start_date} and {end_date} on topic '{topic}'",
                "articles_found": 0,
                "key_headlines": [],
                "key_insights": [],
                "sentiment": "neutral",
                "date_distribution": {},
                "source_breakdown": {},
                "chunks": [],
                "metadata": {"error": "No news data available"},
                "available_topics": _get_available_topics(),
                "errors": result.get("errors", ["No news data found"])
            }
        
        # Filter chunks to exact date range
        filtered_chunks = _filter_chunks_by_date_range(result["chunks"], start_dt, end_dt)
        
        if not filtered_chunks:
            return {
                "success": False,
                "company": company,
                "date_range": {"start": start_date, "end": end_date},
                "topic": topic,
                "summary": f"No news found for {company} in the exact date range {start_date} to {end_date} on topic '{topic}'",
                "articles_found": 0,
                "key_headlines": [],
                "key_insights": [],
                "sentiment": "neutral",
                "date_distribution": {},
                "source_breakdown": {},
                "chunks": [],
                "metadata": {"note": f"Found {len(result['chunks'])} articles in broader range, but none in exact date range"},
                "available_topics": _get_available_topics(),
                "errors": ["No articles found in specified date range"]
            }
        
        # Process filtered chunks for enhanced agent consumption
        processed_result = _process_chunks_for_date_range_agent(
            filtered_chunks, company, start_date, end_date, topic
        )
        
        return {
            "success": True,
            "company": company,
            "date_range": {"start": start_date, "end": end_date},
            "topic": topic,
            "summary": processed_result["summary"],
            "articles_found": len(filtered_chunks),
            "key_headlines": processed_result["key_headlines"],
            "key_insights": processed_result["key_insights"],
            "sentiment": processed_result["sentiment"],
            "date_distribution": processed_result["date_distribution"],
            "source_breakdown": processed_result["source_breakdown"],
            "chunks": filtered_chunks,  # For vector storage
            "metadata": {
                "processing_time": result.get("processing_time_seconds", 0),
                "total_articles_retrieved": len(result["chunks"]),
                "articles_after_date_filtering": len(filtered_chunks),
                "companies_covered": result["summary"].get("companies_covered", []),
                "topic_searched": topic,
                "search_method": "date_range_filtered"
            },
            "available_topics": _get_available_topics(),
            "errors": []
        }
        
    except Exception as e:
        error_msg = f"Error retrieving news for {company}: {str(e)}"
        logger.error(error_msg)
        
        return _create_error_response(company, start_date, end_date, topic, [str(e)])


def _validate_date_range_inputs(company: str, start_date: str, end_date: str, topic: str) -> Dict[str, Any]:
    """Validate inputs for date range function"""
    errors = []
    
    # Validate company
    if not company or not isinstance(company, str) or len(company.strip()) == 0:
        errors.append("Company name/ticker is required and must be a non-empty string")
    
    # Validate dates
    try:
        from datetime import datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_dt > end_dt:
            errors.append("Start date must be before or equal to end date")
        
        if end_dt > datetime.now():
            errors.append("End date cannot be in the future")
        
        # Check if date range is reasonable (not more than 2 years)
        if (end_dt - start_dt).days > 730:
            errors.append("Date range too large. Maximum 2 years (730 days) allowed")
            
    except ValueError as e:
        errors.append(f"Invalid date format. Use 'YYYY-MM-DD' format. Error: {str(e)}")
    
    # Validate topic
    available_topics = _get_available_topics()
    if topic not in available_topics:
        errors.append(f"Invalid topic '{topic}'. Available topics: {', '.join(available_topics)}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def _get_available_topics() -> List[str]:
    """Get list of available news topics"""
    return [
        "earnings",      # Quarterly results, financial reports
        "acquisitions",  # Mergers, buyouts, deals
        "partnerships",  # Collaborations, alliances
        "products",      # Product launches, innovations
        "leadership",    # Executive changes, appointments
        "regulatory",    # Compliance, legal issues
        "market",        # Stock performance, analyst reports
        "financial",     # Funding, investments, IPOs
        "business",      # General business operations
        "general"        # All other news
    ]


def _filter_chunks_by_date_range(chunks: List[Dict], start_dt, end_dt) -> List[Dict]:
    """Filter chunks to only include those within the exact date range"""
    filtered_chunks = []
    
    for chunk in chunks:
        pub_date_str = chunk.get("metadata", {}).get("published_date", "")
        if pub_date_str:
            try:
                # Handle different date formats
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                
                # Remove timezone info for comparison if start_dt and end_dt are naive
                if start_dt.tzinfo is None and pub_date.tzinfo is not None:
                    pub_date = pub_date.replace(tzinfo=None)
                
                if start_dt <= pub_date <= end_dt:
                    filtered_chunks.append(chunk)
                    
            except Exception as e:
                logger.warning(f"Error parsing date '{pub_date_str}': {e}")
                continue
    
    return filtered_chunks


def _process_chunks_for_date_range_agent(
    chunks: List[Dict], 
    company: str, 
    start_date: str, 
    end_date: str, 
    topic: str
) -> Dict[str, Any]:
    """Process chunks for enhanced agent consumption with date range analysis"""
    
    # Extract key information
    key_headlines = []
    key_insights = []
    sentiment_scores = []
    date_distribution = {}
    source_breakdown = {}
    
    # Sort chunks by date (most recent first)
    sorted_chunks = sorted(
        chunks,
        key=lambda x: x.get("metadata", {}).get("published_date", ""),
        reverse=True
    )
    
    # Process each chunk
    for chunk in sorted_chunks:
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        
        # Extract headlines
        title = metadata.get("title", "")
        if title and title not in key_headlines:
            key_headlines.append(title)
        
        # Extract key insights (first 2 sentences of each article)
        if content:
            sentences = content.split('.')[:2]  # First 2 sentences
            insight = '. '.join(sentences).strip()
            if len(insight) > 30 and insight not in key_insights:
                key_insights.append(f"• {insight}")
        
        # Track sentiment
        sentiment = metadata.get("sentiment", "neutral")
        if sentiment in ["positive", "negative", "neutral"]:
            sentiment_scores.append(sentiment)
        
        # Track date distribution
        pub_date_str = metadata.get("published_date", "")
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                date_key = pub_date.strftime("%Y-%m-%d")
                date_distribution[date_key] = date_distribution.get(date_key, 0) + 1
            except:
                pass
        
        # Track source breakdown
        source = metadata.get("source", "unknown")
        source_breakdown[source] = source_breakdown.get(source, 0) + 1
    
    # Analyze overall sentiment
    overall_sentiment = _analyze_overall_sentiment(sentiment_scores)
    
    # Create summary
    summary = _create_date_range_summary(
        company, len(chunks), start_date, end_date, topic, overall_sentiment
    )
    
    return {
        "summary": summary,
        "key_headlines": key_headlines[:10],  # Top 10 headlines
        "key_insights": key_insights[:8],     # Top 8 insights
        "sentiment": overall_sentiment,
        "date_distribution": dict(sorted(date_distribution.items())),
        "source_breakdown": dict(sorted(source_breakdown.items(), key=lambda x: x[1], reverse=True))
    }


def _create_date_range_summary(
    company: str, 
    article_count: int, 
    start_date: str, 
    end_date: str, 
    topic: str, 
    sentiment: str
) -> str:
    """Create a concise summary for date range analysis"""
    
    summary_parts = [
        f"Found {article_count} {topic} news articles for {company} between {start_date} and {end_date}."
    ]
    
    if sentiment != "neutral":
        summary_parts.append(f"Overall news sentiment is {sentiment}.")
    
    if article_count > 0:
        summary_parts.append(f"Coverage focuses on {topic}-related developments and their impact on {company}.")
    
    return " ".join(summary_parts)


def _create_error_response(
    company: str, 
    start_date: str, 
    end_date: str, 
    topic: str, 
    errors: List[str]
) -> Dict[str, Any]:
    """Create standardized error response"""
    
    return {
        "success": False,
        "company": company,
        "date_range": {"start": start_date, "end": end_date},
        "topic": topic,
        "summary": f"Unable to retrieve news for {company} due to validation errors",
        "articles_found": 0,
        "key_headlines": [],
        "key_insights": [],
        "sentiment": "neutral",
        "date_distribution": {},
        "source_breakdown": {},
        "chunks": [],
        "metadata": {"validation_errors": errors},
        "available_topics": _get_available_topics(),
        "errors": errors
    }