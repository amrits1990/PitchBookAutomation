# SharePriceRAG - Intelligent Stock Price Retrieval with PostgreSQL Caching

A production-ready Python package for intelligent stock price data management with PostgreSQL caching and automatic fetching from reliable free APIs (Alpha Vantage, Yahoo Finance).

## Features

- **Intelligent Gap Detection**: Automatically identifies missing price data in your database
- **Multi-Source API Integration**: Uses Alpha Vantage (primary) and Yahoo Finance (fallback) 
- **PostgreSQL Caching**: Efficient local storage with automatic upserts
- **Bulk Ticker Support**: Handle multiple tickers simultaneously
- **Business Day Filtering**: Automatically excludes weekends unless specified
- **Rate Limiting**: Built-in rate limiting for API compliance
- **Comprehensive Statistics**: Cache hit rates, processing times, and data quality metrics
- **Production Ready**: Full error handling, logging, and validation

## Quick Start

```python
from SharePriceRAG import get_share_prices

# Basic usage - get prices for multiple tickers
result = get_share_prices(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(f"Retrieved {result.total_records} records")
print(f"Cache hit rate: {result.cache_hit_rate:.2%}")
print(f"Processing time: {result.processing_time_seconds:.2f}s")

# Access the price data
for price in result.price_data:
    print(f"{price.ticker} {price.date}: ${price.close_price}")
```

## Installation

1. Install package dependencies:
```bash
pip install -r requirements.txt
```

2. Setup PostgreSQL database:
```bash
# Create database
createdb shareprices

# Set environment variables
export POSTGRES_HOST="localhost"
export POSTGRES_DB="shareprices" 
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="your_password"
```

3. Get API keys (optional but recommended):
```bash
# Alpha Vantage (primary source)
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"

# Yahoo Finance works without API key (fallback)
export USE_YAHOO_FINANCE="true"
```

## Configuration

The package uses environment variables for configuration:

```bash
# Database (Required)
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"
export POSTGRES_DB="shareprices"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="your_password"

# APIs (Optional - Yahoo Finance works without key)
export ALPHA_VANTAGE_API_KEY="your_key_here"
export USE_YAHOO_FINANCE="true"

# Rate limiting
export API_REQUESTS_PER_MINUTE="5"
export API_REQUESTS_PER_DAY="500"

# Logging
export SHARE_PRICE_LOG_LEVEL="INFO"
```

## Advanced Usage

### Custom Configuration

```python
from SharePriceRAG import get_share_prices, SharePriceConfig

# Custom configuration
config = SharePriceConfig(
    db_host="your_db_host",
    alpha_vantage_api_key="your_key",
    requests_per_minute=10,
    validate_prices=True,
    max_price_change_percent=25.0
)

result = get_share_prices(
    tickers=["TSLA", "NVDA"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    config=config
)
```

### Force Refresh

```python
# Force re-fetch even if data exists in database
result = get_share_prices(
    tickers=["AAPL"],
    start_date="2024-01-01", 
    end_date="2024-01-31",
    force_refresh=True
)
```

### Include Weekends

```python
# Include weekend dates (useful for crypto or international markets)
result = get_share_prices(
    tickers=["BTC-USD"],
    start_date="2024-01-01",
    end_date="2024-01-07", 
    include_weekends=True
)
```

### Working with Individual Components

```python
from SharePriceRAG import SharePriceDatabase, SharePriceClient, SharePriceConfig

config = SharePriceConfig.default()

# Database operations
db = SharePriceDatabase(config)
db.initialize_database()

stats = db.get_data_statistics("AAPL")
print(f"AAPL records: {stats['total_records']}")

# API client operations  
client = SharePriceClient(config)
health = client.health_check()
print(f"APIs available: {health}")
```

## Output Format

The `get_share_prices` function returns a comprehensive `PriceResult` object:

```python
result = get_share_prices(["AAPL"], "2024-01-01", "2024-01-31")

# Main results
print(result.success)              # True/False
print(result.total_records)        # Total price records returned
print(result.records_from_cache)   # Records retrieved from database
print(result.records_fetched)      # Records fetched from APIs
print(result.cache_hit_rate)       # Percentage from cache (0.0 to 1.0)

# Price data (List of PriceData objects)
for price in result.price_data:
    print(f"{price.ticker} {price.date}:")
    print(f"  Open: ${price.open_price}")
    print(f"  High: ${price.high_price}")  
    print(f"  Low: ${price.low_price}")
    print(f"  Close: ${price.close_price}")
    print(f"  Volume: {price.volume}")
    print(f"  Source: {price.source}")

# Missing data tracking
for missing in result.missing_ranges:
    print(f"Missing: {missing.ticker} from {missing.start_date} to {missing.end_date}")

# Error handling
if result.errors:
    print(f"Errors: {result.errors}")
    
if result.warnings:
    print(f"Warnings: {result.warnings}")

# Convert to JSON for storage/API responses
json_data = result.to_dict()
```

## Utility Functions

### Ticker Summary

```python
from SharePriceRAG import get_ticker_summary

summary = get_ticker_summary("AAPL")
print(f"Date range: {summary['date_range']}")
print(f"Records: {summary['statistics']['total_records']}")
```

### Database Cleanup

```python
from SharePriceRAG import cleanup_database

# Keep only last 365 days of data
result = cleanup_database(days_to_keep=365)
print(f"Deleted {result['rows_deleted']} old records")
```

### Health Check

```python
from SharePriceRAG import health_check

health = health_check()
print(f"Database: {health['database']}")
print(f"APIs: {health['api_sources']}")
```

## Data Sources

### Alpha Vantage (Primary)
- **Pros**: High reliability, comprehensive data, good for production
- **Cons**: Requires API key, 500 requests/day limit on free tier
- **Rate Limit**: 5 requests/minute, 500/day (free tier)

### Yahoo Finance (Fallback)  
- **Pros**: No API key required, high rate limits, good coverage
- **Cons**: Less reliable, unofficial API, can be rate limited
- **Rate Limit**: ~200 requests/minute (unofficial)

## Database Schema

The package automatically creates the following PostgreSQL table:

```sql
CREATE TABLE share_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(12,4),
    high_price DECIMAL(12,4), 
    low_price DECIMAL(12,4),
    close_price DECIMAL(12,4) NOT NULL,
    adjusted_close DECIMAL(12,4),
    volume BIGINT,
    source VARCHAR(50) NOT NULL DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);
```

## Error Handling

The package includes comprehensive error handling:

- **API Errors**: Rate limiting, invalid responses, network issues
- **Database Errors**: Connection issues, schema problems, constraint violations  
- **Data Validation**: Suspicious price changes, missing required fields
- **Configuration Errors**: Missing credentials, invalid parameters

All errors are logged and included in the response for debugging.

## Logging

Configure logging level via environment variable:

```bash
export SHARE_PRICE_LOG_LEVEL="DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

## Production Considerations

- **Database**: Use connection pooling for high-concurrency applications
- **API Keys**: Store securely in environment variables, not in code
- **Rate Limits**: Monitor API usage to avoid hitting limits
- **Data Validation**: Enable price validation to catch suspicious data
- **Monitoring**: Use the health_check() function for monitoring
- **Backup**: Regular database backups recommended

## Contributing

1. Follow existing code structure and patterns
2. Add comprehensive error handling and logging  
3. Include type hints and docstrings
4. Test with multiple tickers and date ranges
5. Maintain backward compatibility

## Dependencies

- `psycopg2-binary`: PostgreSQL database adapter
- `requests`: HTTP library for API calls
- `yfinance`: Yahoo Finance API client
- `pandas`: Data manipulation (used by yfinance)
- `python-dateutil`: Date parsing utilities

## License

This package is part of the PitchBook Generator project.