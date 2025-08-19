"""
Agent-friendly interface for TranscriptRAG package
Provides earnings call insights extraction optimized for LangChain agents
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import re

# Try to import the main transcript function
try:
    from .get_transcript_chunks import get_transcript_chunks
except ImportError:
    try:
        from get_transcript_chunks import get_transcript_chunks
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("Could not import get_transcript_chunks - using fallback")
        get_transcript_chunks = None

logger = logging.getLogger(__name__)

# ---------- Thin Orchestration Functions (Moved to vector_enhanced_interface.py) ----------
# These functions are now handled by the vector-enhanced interface for better performance
# and persistent storage. The original implementations remain here as fallbacks.

def index_transcripts_for_agent(ticker: str, quarters_back: int = 4) -> Dict[str, Any]:
    """Fallback index function - used when vector enhancement unavailable"""
    if get_transcript_chunks is None:
        return {"success": False, "ticker": ticker, "chunks": [], "errors": ["Transcript processing unavailable"]}
    
    # Calculate start_date based on quarters_back (roughly 3 months per quarter)
    months_back = quarters_back * 3
    start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
    
    result = get_transcript_chunks(
        tickers=[ticker], 
        start_date=start_date,
        years_back=max(1, quarters_back // 4), 
        chunk_size=1000, 
        overlap=200,
        return_full_data=True  # Required to get chunks data
    )
    if result.get("status") != "success":
        return {"success": False, "ticker": ticker, "chunks": [], "errors": [result.get("message", "Failed to fetch transcripts")]}
    
    # Extract chunks from successful_transcripts
    chunks = []
    for transcript in result.get("successful_transcripts", []):
        if "transcript_dataset" in transcript:
            chunks.extend(transcript["transcript_dataset"].get("all_chunks", []))
    
    return {"success": True, "ticker": ticker, "chunks": chunks, 
            "metadata": {"quarters_back": quarters_back, "transcripts": len(result.get("successful_transcripts", [])), "fetched_at": datetime.now().isoformat()}}


def search_transcripts_for_agent(ticker: str, query: str, quarters_back: int = 4, k: int = 20, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fallback search function - used when vector enhancement unavailable"""
    idx = index_transcripts_for_agent(ticker, quarters_back=quarters_back)
    if not idx.get("success"):
        return idx
    chunks = idx.get("chunks", [])
    
    # Simple filtering
    filters = filters or {}
    filtered = [c for c in chunks if _transcript_matches_filters(c, filters)]
    
    if not query:
        top = filtered[:k]
    else:
        q, terms = query.lower(), [t for t in query.lower().split() if t]
        scored = [(sum((c.get("content", "") or c.get("text", "") or "").lower().count(t) for t in terms) + (3 if q in (c.get("content", "") or c.get("text", "") or "").lower() else 0), c) for c in filtered]
        scored = [sc for sc in scored if sc[0] > 0]
        top = [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)[:k]]
    
    return {"success": True, "ticker": ticker, "query": query, "results": top, "returned": len(top), 
            "total_candidates": len(filtered), "metadata": {"k": k, "filters": filters, "generated_at": datetime.now().isoformat()}}


def _transcript_matches_filters(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Helper function for basic transcript filtering"""
    if not filters:
        return True
    meta = chunk.get("metadata", {})
    if filters.get("section") and str(meta.get("section", "")).lower() != str(filters["section"]).lower():
        return False
    if filters.get("speaker") and str(meta.get("speaker", "")).lower() != str(filters["speaker"]).lower():
        return False
    if filters.get("quarter") and str(meta.get("quarter", "")).lower() != str(filters["quarter"]).lower():
        return False
    return True

# ---------- High-Level Wrapper Functions ----------

def get_transcript_insights_for_agent(
    ticker: str,
    quarters_back: int = 4,
    focus_areas: List[str] = None
) -> Dict[str, Any]:
    """
    Agent-friendly transcript insights extraction
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        quarters_back: Number of quarters to analyze (default: 4)
        focus_areas: Specific areas to focus on ['guidance', 'performance', 'outlook', 'risks']
        
    Returns:
        Structured dictionary optimized for agent consumption:
        {
            "success": bool,
            "ticker": str,
            "management_tone": str,           # "confident", "cautious", "mixed"
            "key_topics": List[str],          # Main discussion topics
            "guidance_changes": List[dict],   # Changes in guidance
            "analyst_concerns": List[str],    # Key analyst questions/concerns
            "strategic_initiatives": List[str], # New strategic announcements
            "financial_highlights": List[str],  # Key financial metrics mentioned
            "risk_factors": List[str],        # Risks discussed
            "outlook_summary": str,           # Forward-looking summary
            "sentiment_analysis": dict,       # Sentiment breakdown
            "quarters_analyzed": int,
            "raw_chunks": List[dict]          # Raw chunks for vector storage
        }
    """
    
    try:
        if get_transcript_chunks is None:
            return _create_transcript_error_response(ticker, "Transcript processing not available")
        
        # Set default focus areas
        if focus_areas is None:
            focus_areas = ["guidance", "performance", "outlook", "risks"]
        
        # Get transcript data
        transcript_result = get_transcript_chunks(
            tickers=[ticker],
            years_back=max(1, quarters_back // 4),  # Convert quarters to years
            chunk_size=1000,
            chunk_overlap=200
        )
        
        if not transcript_result.get("success", False) or not transcript_result.get("chunks"):
            return _create_transcript_error_response(
                ticker, 
                "No transcript data available or processing failed"
            )
        
        # Process chunks for insights
        chunks = transcript_result["chunks"]
        insights = _extract_insights_from_chunks(chunks, ticker, focus_areas)
        
        return {
            "success": True,
            "ticker": ticker,
            "management_tone": insights["management_tone"],
            "key_topics": insights["key_topics"],
            "guidance_changes": insights["guidance_changes"],
            "analyst_concerns": insights["analyst_concerns"],
            "strategic_initiatives": insights["strategic_initiatives"],
            "financial_highlights": insights["financial_highlights"],
            "risk_factors": insights["risk_factors"],
            "outlook_summary": insights["outlook_summary"],
            "sentiment_analysis": insights["sentiment_analysis"],
            "quarters_analyzed": quarters_back,
            "raw_chunks": chunks,  # For vector storage
            "metadata": {
                "total_chunks": len(chunks),
                "processing_timestamp": datetime.now().isoformat(),
                "focus_areas": focus_areas
            }
        }
        
    except Exception as e:
        error_msg = f"Error processing transcripts for {ticker}: {str(e)}"
        logger.error(error_msg)
        return _create_transcript_error_response(ticker, error_msg)


def _extract_insights_from_chunks(chunks: List[Dict], ticker: str, focus_areas: List[str]) -> Dict[str, Any]:
    """Extract structured insights from transcript chunks"""
    
    # Initialize insight containers
    insights = {
        "management_tone": "neutral",
        "key_topics": [],
        "guidance_changes": [],
        "analyst_concerns": [],
        "strategic_initiatives": [],
        "financial_highlights": [],
        "risk_factors": [],
        "outlook_summary": "",
        "sentiment_analysis": {}
    }
    
    # Separate management and Q&A content
    management_chunks = []
    qa_chunks = []
    
    for chunk in chunks:
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        section = metadata.get("section", "").lower()
        
        if "management" in section or "prepared" in section:
            management_chunks.append(chunk)
        elif "q&a" in section or "question" in section:
            qa_chunks.append(chunk)
        else:
            # Default to management if unclear
            management_chunks.append(chunk)
    
    # Extract insights based on focus areas
    if "guidance" in focus_areas:
        insights["guidance_changes"] = _extract_guidance_changes(management_chunks)
    
    if "performance" in focus_areas:
        insights["financial_highlights"] = _extract_financial_highlights(management_chunks)
    
    if "outlook" in focus_areas:
        insights["outlook_summary"] = _extract_outlook_summary(management_chunks)
    
    if "risks" in focus_areas:
        insights["risk_factors"] = _extract_risk_factors(management_chunks + qa_chunks)
    
    # Always extract these core insights
    insights["key_topics"] = _extract_key_topics(management_chunks + qa_chunks)
    insights["analyst_concerns"] = _extract_analyst_concerns(qa_chunks)
    insights["strategic_initiatives"] = _extract_strategic_initiatives(management_chunks)
    insights["management_tone"] = _analyze_management_tone(management_chunks)
    insights["sentiment_analysis"] = _analyze_sentiment(management_chunks, qa_chunks)
    
    return insights


def _extract_guidance_changes(chunks: List[Dict]) -> List[Dict]:
    """Extract guidance and forecast changes"""
    guidance_changes = []
    
    guidance_keywords = [
        "guidance", "outlook", "forecast", "expect", "anticipate",
        "target", "estimate", "project", "full year", "fy", "q1", "q2", "q3", "q4"
    ]
    
    for chunk in chunks:
        content = chunk.get("content", "").lower()
        
        # Look for guidance-related content
        for keyword in guidance_keywords:
            if keyword in content:
                # Extract surrounding context
                sentences = content.split('.')
                for sentence in sentences:
                    if keyword in sentence and len(sentence) > 30:
                        guidance_changes.append({
                            "type": "guidance_update",
                            "content": sentence.strip().capitalize(),
                            "quarter": chunk.get("metadata", {}).get("quarter", "unknown")
                        })
                        break
    
    # Remove duplicates and limit
    seen = set()
    unique_guidance = []
    for item in guidance_changes:
        content = item["content"]
        if content not in seen:
            seen.add(content)
            unique_guidance.append(item)
    
    return unique_guidance[:5]  # Limit to 5 most relevant


def _extract_financial_highlights(chunks: List[Dict]) -> List[str]:
    """Extract key financial metrics and highlights"""
    highlights = []
    
    financial_patterns = [
        r"revenue.{0,50}(\$[\d.,]+\s*(million|billion|m|b))",
        r"earnings.{0,50}(\$[\d.,]+)",
        r"margin.{0,30}(\d+\.?\d*%)",
        r"growth.{0,30}(\d+\.?\d*%)",
        r"increased?.{0,20}(\d+\.?\d*%)",
        r"decreased?.{0,20}(\d+\.?\d*%)"
    ]
    
    for chunk in chunks:
        content = chunk.get("content", "")
        
        # Look for financial metrics
        for pattern in financial_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                context_start = max(0, match.start() - 100)
                context_end = min(len(content), match.end() + 100)
                context = content[context_start:context_end].strip()
                
                if len(context) > 50:  # Ensure meaningful context
                    highlights.append(context)
    
    # Remove duplicates and clean up
    unique_highlights = list(set(highlights))
    return unique_highlights[:8]  # Limit to top 8


def _extract_outlook_summary(chunks: List[Dict]) -> str:
    """Extract forward-looking outlook summary"""
    
    outlook_keywords = [
        "outlook", "forward", "next quarter", "going forward", "expect",
        "anticipate", "future", "upcoming", "plan", "strategy"
    ]
    
    outlook_sentences = []
    
    for chunk in chunks:
        content = chunk.get("content", "")
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in outlook_keywords:
                if keyword in sentence_lower and len(sentence) > 40:
                    outlook_sentences.append(sentence.strip())
                    break
    
    # Combine and summarize
    if not outlook_sentences:
        return "No specific forward-looking guidance provided in recent transcripts."
    
    # Take the most relevant sentences
    relevant_outlook = outlook_sentences[:3]
    summary = " ".join(relevant_outlook)
    
    # Truncate if too long
    if len(summary) > 500:
        summary = summary[:500] + "..."
    
    return summary


def _extract_risk_factors(chunks: List[Dict]) -> List[str]:
    """Extract risk factors and concerns mentioned"""
    
    risk_keywords = [
        "risk", "concern", "challenge", "headwind", "uncertainty",
        "volatile", "pressure", "difficult", "impact", "threat"
    ]
    
    risk_factors = []
    
    for chunk in chunks:
        content = chunk.get("content", "")
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in risk_keywords:
                if keyword in sentence_lower and len(sentence) > 30:
                    risk_factors.append(sentence.strip())
                    break
    
    # Remove duplicates and return top risks
    unique_risks = list(set(risk_factors))
    return unique_risks[:6]


def _extract_key_topics(chunks: List[Dict]) -> List[str]:
    """Extract main discussion topics"""
    
    # Common business topics to look for
    topic_keywords = {
        "digital transformation": ["digital", "technology", "platform", "cloud"],
        "market expansion": ["market", "expansion", "growth", "international"],
        "product development": ["product", "innovation", "development", "launch"],
        "operational efficiency": ["efficiency", "cost", "optimization", "productivity"],
        "customer growth": ["customer", "user", "subscriber", "retention"],
        "competitive positioning": ["competition", "competitive", "market share"],
        "financial performance": ["revenue", "profit", "margin", "earnings"],
        "supply chain": ["supply", "manufacturing", "logistics", "inventory"]
    }
    
    topic_scores = {topic: 0 for topic in topic_keywords.keys()}
    
    # Score topics based on keyword frequency
    all_content = " ".join([chunk.get("content", "").lower() for chunk in chunks])
    
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            topic_scores[topic] += all_content.count(keyword)
    
    # Return top topics
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, score in sorted_topics if score > 0][:6]


def _extract_analyst_concerns(qa_chunks: List[Dict]) -> List[str]:
    """Extract key analyst questions and concerns"""
    
    concerns = []
    question_starters = ["question", "ask", "wondering", "concern", "clarify"]
    
    for chunk in qa_chunks:
        content = chunk.get("content", "")
        speaker = chunk.get("metadata", {}).get("speaker", "").lower()
        
        # Focus on analyst questions (not management responses)
        if "management" not in speaker and "ceo" not in speaker and "cfo" not in speaker:
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(starter in sentence_lower for starter in question_starters) and len(sentence) > 40:
                    concerns.append(sentence.strip())
    
    # Remove duplicates and limit
    unique_concerns = list(set(concerns))
    return unique_concerns[:5]


def _extract_strategic_initiatives(chunks: List[Dict]) -> List[str]:
    """Extract new strategic initiatives and announcements"""
    
    strategy_keywords = [
        "initiative", "strategy", "plan", "investment", "partnership",
        "acquisition", "expansion", "launch", "program", "focus"
    ]
    
    initiatives = []
    
    for chunk in chunks:
        content = chunk.get("content", "")
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in strategy_keywords) and len(sentence) > 40:
                initiatives.append(sentence.strip())
    
    # Remove duplicates and return top initiatives
    unique_initiatives = list(set(initiatives))
    return unique_initiatives[:5]


def _analyze_management_tone(chunks: List[Dict]) -> str:
    """Analyze overall management tone"""
    
    positive_words = [
        "strong", "growth", "positive", "optimistic", "confident", "pleased",
        "excellent", "successful", "improved", "increased", "momentum"
    ]
    
    negative_words = [
        "challenging", "difficult", "concern", "pressure", "decline", "decreased",
        "weak", "disappointing", "uncertainty", "headwind", "volatile"
    ]
    
    cautious_words = [
        "cautious", "careful", "monitoring", "watching", "uncertain", "variable",
        "depends", "conditions", "environment", "visibility"
    ]
    
    all_content = " ".join([chunk.get("content", "").lower() for chunk in chunks])
    
    positive_score = sum(all_content.count(word) for word in positive_words)
    negative_score = sum(all_content.count(word) for word in negative_words)
    cautious_score = sum(all_content.count(word) for word in cautious_words)
    
    # Determine dominant tone
    if positive_score > negative_score + cautious_score:
        return "confident"
    elif negative_score > positive_score + cautious_score:
        return "concerned" 
    elif cautious_score > positive_score:
        return "cautious"
    else:
        return "mixed"


def _analyze_sentiment(management_chunks: List[Dict], qa_chunks: List[Dict]) -> Dict[str, Any]:
    """Analyze sentiment across different sections"""
    
    management_tone = _analyze_management_tone(management_chunks)
    
    # Analyze Q&A tone separately
    qa_tone = "neutral"
    if qa_chunks:
        qa_tone = _analyze_management_tone(qa_chunks)  # Reuse same logic
    
    return {
        "management_sentiment": management_tone,
        "qa_sentiment": qa_tone,
        "overall_sentiment": management_tone,  # Weight management more heavily
        "confidence_level": "high" if len(management_chunks + qa_chunks) > 10 else "medium"
    }


def _create_transcript_error_response(ticker: str, error_msg: str) -> Dict[str, Any]:
    """Create standardized error response for transcript analysis"""
    return {
        "success": False,
        "ticker": ticker,
        "error": error_msg,
        "management_tone": "unknown",
        "key_topics": [],
        "guidance_changes": [],
        "analyst_concerns": [],
        "strategic_initiatives": [],
        "financial_highlights": [],
        "risk_factors": [],
        "outlook_summary": f"Unable to analyze transcript insights for {ticker}: {error_msg}",
        "sentiment_analysis": {},
        "quarters_analyzed": 0,
        "raw_chunks": [],
        "metadata": {"error": error_msg}
    }


def get_earnings_call_summary(ticker: str, latest_quarter_only: bool = True) -> Dict[str, Any]:
    """
    Get focused summary of latest earnings call
    
    Args:
        ticker: Company ticker symbol
        latest_quarter_only: Focus only on most recent quarter (default: True)
        
    Returns:
        Concise earnings call summary optimized for agents
    """
    
    quarters = 1 if latest_quarter_only else 2
    insights = get_transcript_insights_for_agent(ticker, quarters_back=quarters)
    
    if not insights["success"]:
        return {
            "ticker": ticker,
            "summary": f"Unable to retrieve earnings call summary for {ticker}",
            "key_points": [],
            "management_outlook": "unknown",
            "analyst_focus": [],
            "next_steps": []
        }
    
    # Create focused summary
    summary_parts = []
    
    # Management tone
    tone = insights["management_tone"]
    summary_parts.append(f"Management tone was {tone}")
    
    # Key highlights
    if insights["financial_highlights"]:
        summary_parts.append(f"Financial highlights: {insights['financial_highlights'][0]}")
    
    # Outlook
    if insights["outlook_summary"]:
        summary_parts.append(f"Outlook: {insights['outlook_summary'][:200]}...")
    
    return {
        "ticker": ticker,
        "summary": ". ".join(summary_parts),
        "key_points": insights["key_topics"][:5],
        "management_outlook": insights["outlook_summary"],
        "analyst_focus": insights["analyst_concerns"][:3],
        "next_steps": insights["strategic_initiatives"][:3],
        "sentiment": insights["sentiment_analysis"].get("overall_sentiment", "neutral")
    }