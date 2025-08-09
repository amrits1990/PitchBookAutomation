"""
Main interface for SharePriceRAG package
"""

import logging
import time
from typing import List, Union, Optional
from datetime import date, datetime

from .models import PriceQuery, PriceResult, PriceData
from .config import SharePriceConfig
from .database import SharePriceDatabase
from .price_client import SharePriceClient


def get_share_prices(
    tickers: Union[str, List[str]],
    start_date: Union[str, date],
    end_date: Union[str, date],
    include_weekends: bool = False,
    config: Optional[SharePriceConfig] = None,
    force_refresh: bool = False
) -> PriceResult:
    """
    Main interface function for retrieving share prices with intelligent caching.
    
    Takes ticker(s) and date range, checks PostgreSQL database for existing data,
    identifies missing gaps, fetches from free APIs, stores in DB, and returns
    comprehensive price data.
    
    Args:
        tickers: Single ticker string or list of ticker symbols
        start_date: Start date (YYYY-MM-DD string or date object)
        end_date: End date (YYYY-MM-DD string or date object)  
        include_weekends: Whether to include weekend dates (default: False)
        config: Optional SharePriceConfig (uses default if None)
        force_refresh: Force re-fetch even if data exists (default: False)
        
    Returns:
        PriceResult object with comprehensive data and statistics
        
    Example:
        >>> result = get_share_prices(
        ...     tickers=["AAPL", "MSFT", "GOOGL"],
        ...     start_date="2024-01-01", 
        ...     end_date="2024-01-31"
        ... )
        >>> print(f"Retrieved {result.total_records} records")
        >>> print(f"Cache hit rate: {result.cache_hit_rate:.2%}")
    """
    
    start_time = time.time()
    
    # Setup configuration
    if config is None:
        config = SharePriceConfig.default()
    
    # Setup logging
    from .logging_setup import setup_logging
    logger = setup_logging(config)
    
    # Normalize inputs
    if isinstance(tickers, str):
        tickers = [tickers]
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Create query object
    try:
        query = PriceQuery(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            include_weekends=include_weekends
        )
    except Exception as e:
        logger.error(f"Invalid query parameters: {e}")
        result = PriceResult(success=False, query=PriceQuery(tickers=[], start_date=start_date, end_date=end_date))
        result.add_error(f"Invalid query parameters: {e}")
        return result
    
    # Initialize result object
    result = PriceResult(success=True, query=query)
    
    logger.info(f"Starting price retrieval for {len(tickers)} tickers: {start_date} to {end_date}")
    
    try:
        # Initialize database
        database = SharePriceDatabase(config)
        
        if not database.initialize_database():
            result.add_error("Failed to initialize database")
            return result
        
        # Get existing data from database (unless force refresh)
        existing_data = []
        if not force_refresh:
            existing_data = database.get_existing_data(query)
            logger.info(f"Found {len(existing_data)} existing records in database")
            
            # Add existing data to result
            for data in existing_data:
                result.add_price_data(data, from_cache=True)
        
        # Find missing data ranges
        missing_ranges = database.find_missing_data_ranges(query)
        
        if force_refresh:
            # If force refresh, treat all requested data as missing
            from .models import MissingDataRange
            missing_ranges = [
                MissingDataRange(ticker=ticker, start_date=start_date, end_date=end_date)
                for ticker in tickers
            ]
            
        logger.info(f"Found {len(missing_ranges)} missing data ranges")
        result.missing_ranges = missing_ranges
        
        # Fetch missing data from APIs if needed
        if missing_ranges:
            if not config.has_api_access():
                result.add_error("No API access available - missing API keys")
                return result
            
            # Initialize API client
            client = SharePriceClient(config)
            
            # Fetch missing data
            logger.info(f"Fetching missing data from APIs...")
            new_data = client.fetch_missing_data(missing_ranges)
            
            if new_data:
                # Validate fetched data if configured
                if config.validate_prices:
                    new_data = _validate_price_data(new_data, config, logger)
                
                # Store new data in database
                stored_count = database.store_price_data(new_data)
                logger.info(f"Stored {stored_count} new records in database")
                
                # Add new data to result
                for data in new_data:
                    result.add_price_data(data, from_cache=False)
            else:
                result.add_warning("No new data was fetched from APIs")
        
        # Calculate final statistics
        result.calculate_cache_hit_rate()
        result.processing_time_seconds = time.time() - start_time
        
        logger.info(
            f"Completed: {result.total_records} records, "
            f"{result.cache_hit_rate:.1%} cache hit rate, "
            f"{result.processing_time_seconds:.2f}s"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_share_prices: {e}")
        result.add_error(f"Processing error: {e}")
        result.processing_time_seconds = time.time() - start_time
        return result


def _validate_price_data(price_data: List[PriceData], config: SharePriceConfig, logger) -> List[PriceData]:
    """Validate price data for suspicious values"""
    validated_data = []
    
    for data in price_data:
        try:
            # Basic validation
            if data.close_price is None or data.close_price <= 0:
                logger.warning(f"Invalid close price for {data.ticker} on {data.date}: {data.close_price}")
                continue
            
            # Check for extreme price changes (if we have previous data)
            # This is a simplified validation - could be enhanced with historical comparisons
            if data.open_price and data.close_price:
                daily_change = abs((float(data.close_price) - float(data.open_price)) / float(data.open_price)) * 100
                
                if daily_change > config.max_price_change_percent:
                    logger.warning(
                        f"Suspicious price change for {data.ticker} on {data.date}: "
                        f"{daily_change:.1f}% (max: {config.max_price_change_percent}%)"
                    )
                    # Still include the data but log the warning
            
            validated_data.append(data)
            
        except Exception as e:
            logger.warning(f"Validation error for {data.ticker} on {data.date}: {e}")
            continue
    
    logger.info(f"Validated {len(validated_data)}/{len(price_data)} price records")
    return validated_data


def get_ticker_summary(ticker: str, config: Optional[SharePriceConfig] = None) -> dict:
    """
    Get a summary of available data for a specific ticker
    
    Args:
        ticker: Stock ticker symbol
        config: Optional configuration
        
    Returns:
        Dictionary with ticker statistics and date ranges
    """
    if config is None:
        config = SharePriceConfig.default()
    
    database = SharePriceDatabase(config)
    
    if not database.initialize_database():
        return {"error": "Failed to initialize database"}
    
    # Get basic statistics
    stats = database.get_data_statistics(ticker.upper())
    
    # Get date range
    date_range = database.get_ticker_date_range(ticker.upper())
    
    return {
        "ticker": ticker.upper(),
        "statistics": stats,
        "date_range": {
            "start": date_range[0].isoformat() if date_range else None,
            "end": date_range[1].isoformat() if date_range else None
        },
        "timestamp": datetime.utcnow().isoformat()
    }


def cleanup_database(days_to_keep: int = 365, config: Optional[SharePriceConfig] = None) -> dict:
    """
    Clean up old price data from database
    
    Args:
        days_to_keep: Number of days of data to retain
        config: Optional configuration
        
    Returns:
        Dictionary with cleanup results
    """
    if config is None:
        config = SharePriceConfig.default()
    
    database = SharePriceDatabase(config)
    
    if not database.initialize_database():
        return {"error": "Failed to initialize database"}
    
    rows_deleted = database.cleanup_old_data(days_to_keep)
    
    return {
        "success": True,
        "rows_deleted": rows_deleted,
        "days_kept": days_to_keep,
        "timestamp": datetime.utcnow().isoformat()
    }


def health_check(config: Optional[SharePriceConfig] = None) -> dict:
    """
    Perform a health check of the system
    
    Args:
        config: Optional configuration
        
    Returns:
        Dictionary with system health status
    """
    if config is None:
        config = SharePriceConfig.default()
    
    health = {
        "database": False,
        "api_sources": {},
        "configuration": {},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Check database
    try:
        database = SharePriceDatabase(config)
        health["database"] = database.initialize_database()
    except Exception as e:
        health["database_error"] = str(e)
    
    # Check API sources
    try:
        client = SharePriceClient(config)
        health["api_sources"] = client.health_check()
    except Exception as e:
        health["api_error"] = str(e)
    
    # Configuration info
    health["configuration"] = {
        "has_alpha_vantage_key": bool(config.alpha_vantage_api_key),
        "yahoo_finance_enabled": config.use_yahoo_finance,
        "preferred_source": config.preferred_data_source,
        "rate_limits": {
            "per_minute": config.requests_per_minute,
            "per_day": config.requests_per_day
        }
    }
    
    return health

