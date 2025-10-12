# RAG Agent Functions Implementation Guide

A comprehensive guide to all available agent functions across the PitchBookGenerator RAG ecosystem for building intelligent financial analysis systems.

## 📋 Table of Contents

- [Overview](#overview)
- [AnnualReportRAG](#annualreportrag)
- [TranscriptRAG](#transcriptrag)
- [SharePriceRAG](#sharepricerag)
- [SECFinancialRAG](#secfinancialrag)
- [NewsRAG](#newsrag)
- [Integration Patterns](#integration-patterns)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

---

## Overview

This document provides detailed specifications for all agent-ready functions across five specialized RAG systems. Each function returns structured dictionaries optimized for AI agent consumption with consistent error handling and metadata.

### Common Response Structure
All functions follow this pattern:
```python
{
    "success": bool,           # Operation success status
    "data": Any,              # Main response data
    "metadata": dict,         # Processing information
    "errors": List[str]       # Error messages (if any)
}
```

---

## AnnualReportRAG

### `get_annual_reports_for_agent(company, years_back=3, max_chunks=50)`

**Purpose**: Retrieves and processes annual reports (10-K filings) for company analysis.

**Parameters**:
- `company` (str): Company name or ticker symbol (e.g., "Apple", "AAPL")
- `years_back` (int): Number of recent years to retrieve (default: 3)
- `max_chunks` (int): Maximum chunks to return (default: 50)

**Returns**:
```python
{
    "success": bool,
    "company": str,
    "reports_found": int,
    "chunks_retrieved": int,
    "summary": str,                    # Agent-consumable overview
    "chunks": List[dict],             # RAG-ready content chunks
    "reports_metadata": List[dict],   # Filing metadata
    "content_types": dict,            # Section type distribution
    "date_range": dict,               # Coverage period
    "metadata": {
        "processing_time": float,
        "cik": str,                   # SEC company identifier
        "chunks_per_report": dict
    }
}
```

**Use Cases**:
- Long-term business strategy analysis
- Risk factor assessment
- Historical performance context
- Regulatory compliance review

---

## TranscriptRAG

### `get_transcripts_for_agent(company, quarters_back=4, max_chunks=30)`

**Purpose**: Retrieves earnings call transcripts for quarterly business insights.

**Parameters**:
- `company` (str): Company name or ticker symbol
- `quarters_back` (int): Number of recent quarters (default: 4)
- `max_chunks` (int): Maximum chunks to return (default: 30)

**Returns**:
```python
{
    "success": bool,
    "company": str,
    "transcripts_found": int,
    "chunks_retrieved": int,
    "summary": str,
    "chunks": List[dict],             # RAG-ready transcript chunks
    "quarters_covered": List[str],    # Quarter identifiers
    "transcript_metadata": List[dict],
    "content_distribution": dict,     # Q&A vs presentation split
    "date_range": dict,
    "metadata": {
        "processing_time": float,
        "fiscal_periods": List[dict],
        "data_sources": dict
    }
}
```

**Use Cases**:
- Recent business developments
- Management guidance analysis
- Quarterly performance insights
- Forward-looking statements

---

## SharePriceRAG

### `get_price_analysis_for_agent(ticker, days_back=90, include_peers=False, peer_tickers=None)`

**Purpose**: Comprehensive price trend analysis with trading signals and risk assessment.

**Parameters**:
- `ticker` (str): Stock ticker symbol (e.g., "AAPL")
- `days_back` (int): Analysis period in days (default: 90)
- `include_peers` (bool): Include peer comparison (default: False)
- `peer_tickers` (List[str]): List of peer tickers for comparison

**Returns**:
```python
{
    "success": bool,
    "ticker": str,
    "current_price": float,
    "price_trend": str,               # "upward", "downward", "sideways", "volatile"
    "volatility": str,                # "low", "medium", "high", "very_high"
    "performance_summary": str,       # Natural language summary
    "key_metrics": {
        "total_return_pct": float,
        "annualized_return_pct": float,
        "volatility_pct": float,
        "period_high": float,
        "period_low": float,
        "avg_volume": int
    },
    "trading_signals": List[str],     # Key observations
    "risk_assessment": str,           # "low", "medium", "high"
    "peer_comparison": dict,          # If include_peers=True
    "metadata": dict
}
```

### `compare_with_peers(ticker, peer_tickers, days_back=90)`

**Purpose**: Dedicated peer performance comparison and ranking.

**Parameters**:
- `ticker` (str): Target ticker to compare
- `peer_tickers` (List[str]): List of peer ticker symbols
- `days_back` (int): Analysis period (default: 90)

**Returns**:
```python
{
    "target_ticker": str,
    "target_return_pct": float,
    "peer_avg_return_pct": float,
    "relative_performance_pct": float,
    "rank": int,                      # Position in peer group
    "total_compared": int,
    "outperforming": bool,
    "peer_analysis": dict,            # Individual peer performance
    "summary": str                    # Natural language summary
}
```

### `get_raw_price_data_for_agent(ticker, days_back=90)`

**Purpose**: Raw price DataFrame for custom agent analysis.

**Parameters**:
- `ticker` (str): Stock ticker symbol
- `days_back` (int): Number of days to retrieve (default: 90)

**Returns**:
```python
{
    "success": bool,
    "ticker": str,
    "data": pd.DataFrame,             # OHLCV data indexed by date
    "records_count": int,
    "date_range": dict,
    "data_sources": dict,             # Cache vs API breakdown
    "cache_info": {
        "cache_hit_rate": float,
        "processing_time_seconds": float
    },
    "metadata": {
        "columns": List[str],         # ["open", "high", "low", "close", "volume"]
        "data_shape": tuple,
        "has_nulls": dict
    }
}
```

**Use Cases**:
- Custom technical analysis
- Statistical modeling
- Risk calculations
- Performance attribution

---

## SECFinancialRAG

### `get_financial_analysis_for_agent(ticker, years_back=3, analysis_type="comprehensive")`

**Purpose**: Financial metrics analysis with ratios and growth trends.

**Parameters**:
- `ticker` (str): Company ticker symbol
- `years_back` (int): Number of years to analyze (default: 3)
- `analysis_type` (str): "comprehensive", "ratios", "growth" (default: "comprehensive")

**Returns**:
```python
{
    "success": bool,
    "ticker": str,
    "analysis_type": str,
    "periods_analyzed": int,
    "financial_summary": str,
    "key_metrics": {
        "revenue_growth": float,
        "profit_margins": dict,
        "return_ratios": dict,
        "liquidity_ratios": dict,
        "leverage_ratios": dict
    },
    "growth_analysis": {
        "revenue_cagr": float,
        "earnings_cagr": float,
        "trend_direction": str
    },
    "financial_health": str,          # "strong", "stable", "concerning"
    "peer_benchmarks": dict,          # Industry comparisons
    "raw_financials": dict,           # Underlying data
    "metadata": dict
}
```

**Use Cases**:
- Financial health assessment
- Valuation analysis
- Credit risk evaluation
- Investment screening

---

## NewsRAG

### `get_news_for_agent(ticker, days_back=30, categories=None, max_articles=10)`

**Purpose**: Structured news analysis with sentiment and key insights.

**Parameters**:
- `ticker` (str): Company ticker symbol
- `days_back` (int): Lookback period (default: 30)
- `categories` (List[str]): News categories (default: ["general"])
- `max_articles` (int): Maximum articles to analyze (default: 10)

**Returns**:
```python
{
    "success": bool,
    "ticker": str,
    "summary": str,
    "key_points": List[str],          # Bullet points for analysis
    "sentiment": str,                 # "positive", "negative", "neutral"
    "recent_developments": List[str], # Recent key events
    "total_articles": int,
    "chunks": List[dict],            # RAG-ready content
    "metadata": dict
}
```

### `get_news_sentiment_analysis(ticker, days_back=7)`

**Purpose**: Focused sentiment analysis with confidence levels.

**Parameters**:
- `ticker` (str): Company ticker symbol
- `days_back` (int): Analysis period (default: 7)

**Returns**:
```python
{
    "ticker": str,
    "sentiment": str,
    "confidence": str,                # "low", "medium", "high"
    "analysis": str,
    "supporting_evidence": {
        "positive_headlines": List[str],
        "negative_headlines": List[str],
        "recent_developments": List[str]
    },
    "recommendation": str
}
```

### `get_news_by_date_range_and_topic(company, start_date, end_date, topic, max_articles=20)`

**Purpose**: Precise date range and topic-specific news retrieval.

**Parameters**:
- `company` (str): Company name or ticker
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str): End date in 'YYYY-MM-DD' format  
- `topic` (str): News topic from available options
- `max_articles` (int): Maximum articles (default: 20)

**Available Topics**: earnings, acquisitions, partnerships, products, leadership, regulatory, market, financial, business, general

**Returns**:
```python
{
    "success": bool,
    "company": str,
    "date_range": {"start": str, "end": str},
    "topic": str,
    "summary": str,
    "articles_found": int,
    "key_headlines": List[str],
    "key_insights": List[str],
    "sentiment": str,
    "date_distribution": dict,        # News distribution over time
    "source_breakdown": dict,         # News sources analysis
    "chunks": List[dict],
    "available_topics": List[str],
    "metadata": dict
}
```

**Use Cases**:
- Market sentiment analysis
- Event-driven research
- Competitive intelligence
- Risk monitoring

---

## Integration Patterns

### Multi-RAG Analysis Workflow

```python
def comprehensive_company_analysis(ticker: str):
    """Example multi-RAG agent workflow"""
    
    # 1. Get recent news sentiment
    news = get_news_sentiment_analysis(ticker, days_back=30)
    
    # 2. Analyze price trends
    price_analysis = get_price_analysis_for_agent(ticker, days_back=90)
    
    # 3. Get financial health
    financials = get_financial_analysis_for_agent(ticker, years_back=3)
    
    # 4. Recent business updates
    transcripts = get_transcripts_for_agent(ticker, quarters_back=2)
    
    # 5. Long-term strategy
    reports = get_annual_reports_for_agent(ticker, years_back=2)
    
    return {
        "ticker": ticker,
        "sentiment": news["sentiment"],
        "price_trend": price_analysis["price_trend"],
        "financial_health": financials["financial_health"],
        "recent_insights": transcripts["summary"],
        "strategy_context": reports["summary"]
    }
```

### Error Handling Pattern

```python
def safe_rag_call(rag_function, **kwargs):
    """Standard error handling for RAG functions"""
    try:
        result = rag_function(**kwargs)
        if result.get("success", False):
            return result
        else:
            errors = result.get("errors", ["Unknown error"])
            print(f"RAG function failed: {errors}")
            return None
    except Exception as e:
        print(f"Exception in RAG call: {e}")
        return None
```

---

## Error Handling

All functions implement consistent error handling:

### Common Error Types
- **Invalid Ticker/Company**: Non-existent or invalid symbols
- **Date Range Issues**: Invalid or future dates
- **Data Availability**: No data for requested period
- **API Limits**: Rate limiting or quota exceeded
- **Network Issues**: Connection or timeout errors

### Error Response Format
```python
{
    "success": false,
    "error": "Descriptive error message",
    "errors": ["Detailed error 1", "Detailed error 2"],
    "metadata": {"error_context": "additional_info"}
}
```

---

## Best Practices

### 1. Agent Function Selection
- **Real-time Analysis**: Use SharePriceRAG and NewsRAG
- **Fundamental Analysis**: Combine SECFinancialRAG and AnnualReportRAG
- **Recent Business Updates**: Use TranscriptRAG and recent NewsRAG
- **Comprehensive Research**: Use all RAGs with appropriate time horizons

### 2. Performance Optimization
- **Caching**: All RAGs implement intelligent caching
- **Batch Requests**: Group related queries when possible
- **Time Horizons**: Match analysis period to use case
- **Error Recovery**: Implement fallback strategies

### 3. Data Quality
- **Validation**: Check `success` field before processing
- **Completeness**: Verify data availability in `metadata`
- **Freshness**: Consider data timestamps for time-sensitive analysis
- **Source Tracking**: Use source information for data lineage

### 4. Agent Integration
- **Structured Outputs**: All functions return consistent dictionary structures
- **Metadata Utilization**: Use processing information for debugging
- **Error Propagation**: Handle errors gracefully in agent workflows
- **Response Caching**: Consider caching agent-level responses

---

## Example Agent Implementation

```python
class FinancialAnalysisAgent:
    """Example agent using multiple RAG systems"""
    
    def __init__(self):
        self.rags = {
            'news': NewsRAG,
            'price': SharePriceRAG, 
            'financial': SECFinancialRAG,
            'transcript': TranscriptRAG,
            'annual': AnnualReportRAG
        }
    
    def analyze_investment_opportunity(self, ticker: str):
        """Comprehensive investment analysis"""
        
        analysis = {}
        
        # Market sentiment
        sentiment = get_news_sentiment_analysis(ticker)
        analysis['sentiment'] = sentiment['sentiment']
        
        # Price momentum  
        price = get_price_analysis_for_agent(ticker)
        analysis['momentum'] = price['price_trend']
        
        # Financial health
        financials = get_financial_analysis_for_agent(ticker)
        analysis['health'] = financials['financial_health']
        
        # Generate recommendation
        return self._generate_recommendation(analysis)
    
    def _generate_recommendation(self, analysis):
        """Synthesize multi-RAG analysis into recommendation"""
        # Implementation specific to use case
        pass
```

---

This guide provides the complete specification for building sophisticated financial analysis agents using the PitchBookGenerator RAG ecosystem. Each function is designed for seamless integration with modern AI agent frameworks like LangChain, AutoGen, and custom implementations.