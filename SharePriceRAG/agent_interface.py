"""
Agent-friendly interface for SharePriceRAG package
Provides trend analysis and peer comparison optimized for LangChain agents
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta, date
from decimal import Decimal
import statistics

from .main import get_share_prices
from .config import SharePriceConfig

logger = logging.getLogger(__name__)


def get_price_analysis_for_agent(
    ticker: str,
    days_back: int = 90,
    include_peers: bool = False,
    peer_tickers: List[str] = None
) -> Dict[str, Any]:
    """
    Agent-friendly price analysis interface
    
    Args:
        ticker: Company ticker symbol (e.g., 'AAPL')
        days_back: Number of days to analyze (default: 90)
        include_peers: Whether to include peer comparison
        peer_tickers: List of peer tickers for comparison
        
    Returns:
        Structured dictionary optimized for agent consumption:
        {
            "success": bool,
            "ticker": str,
            "current_price": float,
            "price_trend": str,           # "upward", "downward", "sideways"
            "volatility": str,            # "high", "medium", "low"
            "performance_summary": str,   # Agent-readable summary
            "key_metrics": dict,          # Numerical metrics
            "peer_comparison": dict,      # Peer analysis if requested
            "trading_signals": List[str], # Key observations
            "risk_assessment": str,       # Risk level assessment
            "metadata": dict              # Additional context
        }
    """
    
    try:
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        # Get price data
        result = get_share_prices(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date
        )
        
        if not result.success or not result.price_data:
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Unable to retrieve price data for {ticker}",
                "current_price": None,
                "price_trend": "unknown",
                "volatility": "unknown",
                "performance_summary": f"No price data available for {ticker}",
                "key_metrics": {},
                "peer_comparison": {},
                "trading_signals": [],
                "risk_assessment": "unknown",
                "metadata": {"error": "No price data available"}
            }
        
        # Process price data for analysis
        ticker_data = result.get_ticker_data(ticker)
        if not ticker_data:
            return _create_error_response(ticker, "No price data found for ticker")
        
        # Sort by date
        ticker_data.sort(key=lambda x: x.date)
        
        # Perform analysis
        analysis = _analyze_price_data(ticker_data, ticker, days_back)
        
        # Add peer comparison if requested
        if include_peers and peer_tickers:
            peer_analysis = _perform_peer_comparison(ticker, peer_tickers, days_back)
            analysis["peer_comparison"] = peer_analysis
        else:
            analysis["peer_comparison"] = {}
        
        return analysis
        
    except Exception as e:
        error_msg = f"Error analyzing prices for {ticker}: {str(e)}"
        logger.error(error_msg)
        return _create_error_response(ticker, error_msg)


def _analyze_price_data(price_data: List, ticker: str, days_back: int) -> Dict[str, Any]:
    """Analyze price data and generate insights"""
    
    if not price_data:
        return _create_error_response(ticker, "No price data to analyze")
    
    # Extract prices and dates
    prices = [float(p.close_price) for p in price_data]
    volumes = [p.volume for p in price_data if p.volume]
    dates = [p.date for p in price_data]
    
    # Current and historical prices
    current_price = prices[-1]
    start_price = prices[0]
    
    # Calculate key metrics
    total_return = ((current_price - start_price) / start_price) * 100
    
    # Calculate volatility (standard deviation of daily returns)
    daily_returns = []
    for i in range(1, len(prices)):
        daily_return = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
        daily_returns.append(daily_return)
    
    volatility_pct = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
    
    # Trend analysis
    price_trend = _determine_price_trend(prices)
    volatility_level = _classify_volatility(volatility_pct)
    
    # Recent performance (last 30 days vs previous period)
    recent_performance = _analyze_recent_performance(prices, dates)
    
    # Trading signals
    trading_signals = _generate_trading_signals(prices, volumes, daily_returns)
    
    # Risk assessment
    risk_assessment = _assess_risk(volatility_pct, daily_returns)
    
    # Performance summary
    performance_summary = _create_performance_summary(
        ticker, total_return, price_trend, volatility_level, days_back
    )
    
    # Key metrics
    key_metrics = {
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(total_return * (365 / days_back), 2),
        "volatility_pct": round(volatility_pct, 2),
        "current_price": round(current_price, 2),
        "period_high": round(max(prices), 2),
        "period_low": round(min(prices), 2),
        "avg_volume": int(statistics.mean(volumes)) if volumes else 0,
        "data_points": len(prices),
        "date_range": {
            "start": dates[0].isoformat(),
            "end": dates[-1].isoformat()
        }
    }
    
    return {
        "success": True,
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "price_trend": price_trend,
        "volatility": volatility_level,
        "performance_summary": performance_summary,
        "key_metrics": key_metrics,
        "trading_signals": trading_signals,
        "risk_assessment": risk_assessment,
        "metadata": {
            "analysis_period_days": days_back,
            "data_source": "market_data",
            "analysis_timestamp": datetime.now().isoformat()
        }
    }


def _determine_price_trend(prices: List[float]) -> str:
    """Determine overall price trend"""
    if len(prices) < 5:
        return "insufficient_data"
    
    # Use linear regression approximation
    start_price = prices[0]
    end_price = prices[-1]
    mid_price = prices[len(prices)//2]
    
    # Calculate trend strength
    total_change = (end_price - start_price) / start_price * 100
    
    if total_change > 5:
        return "upward"
    elif total_change < -5:
        return "downward"
    else:
        # Check for sideways with volatility
        price_range = (max(prices) - min(prices)) / statistics.mean(prices) * 100
        if price_range < 10:
            return "sideways"
        else:
            return "volatile"


def _classify_volatility(volatility_pct: float) -> str:
    """Classify volatility level"""
    if volatility_pct < 1.5:
        return "low"
    elif volatility_pct < 3.0:
        return "medium"
    elif volatility_pct < 5.0:
        return "high"
    else:
        return "very_high"


def _analyze_recent_performance(prices: List[float], dates: List) -> Dict[str, Any]:
    """Analyze recent performance vs historical"""
    if len(prices) < 30:
        return {"insufficient_data": True}
    
    # Split into recent (last 30 days) and historical
    recent_prices = prices[-30:]
    historical_prices = prices[:-30]
    
    recent_return = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0]) * 100
    
    if historical_prices:
        historical_return = ((historical_prices[-1] - historical_prices[0]) / historical_prices[0]) * 100
        relative_performance = recent_return - historical_return
    else:
        relative_performance = recent_return
    
    return {
        "recent_30d_return": round(recent_return, 2),
        "relative_performance": round(relative_performance, 2),
        "accelerating": relative_performance > 0
    }


def _generate_trading_signals(prices: List[float], volumes: List, returns: List[float]) -> List[str]:
    """Generate key trading observations"""
    signals = []
    
    if len(prices) < 10:
        return ["Insufficient data for signal generation"]
    
    # Price momentum
    recent_trend = _determine_price_trend(prices[-20:])  # Last 20 days
    if recent_trend == "upward":
        signals.append("Recent upward momentum detected")
    elif recent_trend == "downward":
        signals.append("Recent downward pressure observed")
    
    # Volatility signals
    recent_volatility = statistics.stdev(returns[-10:]) if len(returns) >= 10 else 0
    overall_volatility = statistics.stdev(returns)
    
    if recent_volatility > overall_volatility * 1.5:
        signals.append("Increased volatility in recent trading")
    elif recent_volatility < overall_volatility * 0.5:
        signals.append("Volatility compression noted")
    
    # Volume analysis if available
    if volumes and len(volumes) > 20:
        recent_volume = statistics.mean(volumes[-10:])
        avg_volume = statistics.mean(volumes)
        
        if recent_volume > avg_volume * 1.3:
            signals.append("Above-average trading volume")
        elif recent_volume < avg_volume * 0.7:
            signals.append("Below-average trading volume")
    
    # Price extremes
    current_price = prices[-1]
    period_high = max(prices)
    period_low = min(prices)
    
    if current_price >= period_high * 0.95:
        signals.append("Trading near period high")
    elif current_price <= period_low * 1.05:
        signals.append("Trading near period low")
    
    return signals[:5]  # Limit to top 5 signals


def _assess_risk(volatility_pct: float, returns: List[float]) -> str:
    """Assess overall risk level"""
    
    # Base risk on volatility
    if volatility_pct < 2.0:
        base_risk = "low"
    elif volatility_pct < 4.0:
        base_risk = "medium"
    else:
        base_risk = "high"
    
    # Adjust for extreme moves
    if returns:
        max_daily_loss = min(returns)
        if max_daily_loss < -10:
            base_risk = "high"
        elif max_daily_loss < -5 and base_risk == "low":
            base_risk = "medium"
    
    return base_risk


def _create_performance_summary(ticker: str, total_return: float, trend: str, volatility: str, days: int) -> str:
    """Create agent-readable performance summary"""
    
    return_desc = "gained" if total_return > 0 else "declined"
    trend_desc = {
        "upward": "showing upward momentum",
        "downward": "under pressure",
        "sideways": "trading sideways",
        "volatile": "experiencing high volatility"
    }.get(trend, "showing mixed signals")
    
    summary = f"{ticker} has {return_desc} {abs(total_return):.1f}% over {days} days, {trend_desc}. "
    summary += f"Volatility is {volatility}. "
    
    # Add context
    if abs(total_return) > 20:
        summary += "This represents significant price movement."
    elif abs(total_return) < 5:
        summary += "Price movement has been relatively modest."
    
    return summary


def _perform_peer_comparison(ticker: str, peer_tickers: List[str], days_back: int) -> Dict[str, Any]:
    """Perform peer comparison analysis"""
    
    try:
        # Get peer data
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        all_tickers = [ticker] + peer_tickers
        peer_result = get_share_prices(
            tickers=all_tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        if not peer_result.success:
            return {"error": "Unable to retrieve peer data"}
        
        # Analyze each ticker
        comparisons = {}
        for t in all_tickers:
            ticker_data = peer_result.get_ticker_data(t)
            if ticker_data:
                prices = [float(p.close_price) for p in sorted(ticker_data, key=lambda x: x.date)]
                if len(prices) >= 2:
                    total_return = ((prices[-1] - prices[0]) / prices[0]) * 100
                    comparisons[t] = {
                        "return_pct": round(total_return, 2),
                        "current_price": round(prices[-1], 2)
                    }
        
        if ticker not in comparisons:
            return {"error": "Unable to analyze target ticker"}
        
        # Rank performance
        ticker_return = comparisons[ticker]["return_pct"]
        peer_returns = [comparisons[t]["return_pct"] for t in peer_tickers if t in comparisons]
        
        if not peer_returns:
            return {"error": "No valid peer data"}
        
        # Calculate relative performance
        avg_peer_return = statistics.mean(peer_returns)
        relative_performance = ticker_return - avg_peer_return
        
        # Determine ranking
        all_returns = [(t, comparisons[t]["return_pct"]) for t in comparisons.keys()]
        all_returns.sort(key=lambda x: x[1], reverse=True)
        
        ticker_rank = next(i for i, (t, _) in enumerate(all_returns, 1) if t == ticker)
        
        return {
            "target_ticker": ticker,
            "target_return_pct": ticker_return,
            "peer_avg_return_pct": round(avg_peer_return, 2),
            "relative_performance_pct": round(relative_performance, 2),
            "rank": ticker_rank,
            "total_compared": len(all_returns),
            "outperforming": relative_performance > 0,
            "peer_analysis": {
                t: data for t, data in comparisons.items() if t != ticker
            },
            "summary": _create_peer_summary(ticker, ticker_rank, len(all_returns), relative_performance)
        }
        
    except Exception as e:
        logger.error(f"Error in peer comparison: {e}")
        return {"error": str(e)}


def _create_peer_summary(ticker: str, rank: int, total: int, relative_perf: float) -> str:
    """Create peer comparison summary"""
    
    rank_desc = f"{rank} out of {total}"
    perf_desc = "outperforming" if relative_perf > 0 else "underperforming"
    
    if rank <= total * 0.25:
        performance_tier = "top quartile"
    elif rank <= total * 0.5:
        performance_tier = "second quartile" 
    elif rank <= total * 0.75:
        performance_tier = "third quartile"
    else:
        performance_tier = "bottom quartile"
    
    return f"{ticker} ranks {rank_desc} in peer group ({performance_tier}), {perf_desc} peer average by {abs(relative_perf):.1f}%"


def _create_error_response(ticker: str, error_msg: str) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "success": False,
        "ticker": ticker,
        "error": error_msg,
        "current_price": None,
        "price_trend": "unknown",
        "volatility": "unknown", 
        "performance_summary": f"Unable to analyze {ticker}: {error_msg}",
        "key_metrics": {},
        "peer_comparison": {},
        "trading_signals": [],
        "risk_assessment": "unknown",
        "metadata": {"error": error_msg}
    }


def compare_with_peers(ticker: str, peer_tickers: List[str], days_back: int = 90) -> Dict[str, Any]:
    """
    Dedicated peer comparison function
    
    Args:
        ticker: Target ticker to compare
        peer_tickers: List of peer ticker symbols
        days_back: Analysis period in days
        
    Returns:
        Peer comparison analysis optimized for agents
    """
    
    return _perform_peer_comparison(ticker, peer_tickers, days_back)