# NewsRAG - News Retrieval and Processing for RAG Systems

A production-ready Python package for retrieving and processing news articles using the Tavily API, specifically designed for feeding into Retrieval-Augmented Generation (RAG) systems.

## Features

- **Company-Specific News Retrieval**: Search for news about specific companies or tickers  
- **Multiple Categories**: Support for various news categories (earnings, acquisitions, partnerships, etc.)
- **Tavily-Optimized Chunks**: Uses Tavily's built-in chunking (no redundant processing needed)
- **Rich Metadata**: Comprehensive metadata for each chunk including source, companies, dates, and more
- **Production Ready**: Built-in rate limiting, error handling, and logging
- **Simple Interface**: Easy-to-use main function with sensible defaults
- **Streamlined Architecture**: Direct chunk retrieval from Tavily (v2.0 simplified design)

## Quick Start

```python
from NewsRAG import get_company_news_chunks

# Basic usage
result = get_company_news_chunks(
    companies=["Apple", "AAPL"],
    categories=["earnings", "products"],
    days_back=7,
    output_file="apple_news.json"
)

print(f"Retrieved {result['summary']['chunks_generated']} chunks")
```

## Installation

1. Install the package requirements:
```bash
pip install -r requirements.txt
```

2. Set your Tavily API key:
```bash
export TAVILY_API_KEY="your_tavily_api_key_here"
```

## Configuration

The package uses environment variables for configuration:

```bash
# Required
export TAVILY_API_KEY="your_api_key"

# Optional configuration
export TAVILY_REQUESTS_PER_MINUTE="60"
export NEWS_CHUNK_SIZE="1000"
export NEWS_CHUNK_OVERLAP="200"
export NEWS_LOG_LEVEL="INFO"
```

## Supported News Categories

- `earnings`: Quarterly results, financial reports
- `acquisitions`: Mergers, buyouts, deals  
- `partnerships`: Collaborations, alliances
- `products`: Product launches, innovations
- `leadership`: Executive changes, appointments
- `regulatory`: Compliance, legal issues
- `market`: Stock performance, analyst reports
- `financial`: Funding, investments, IPOs
- `general`: All other news

## Output Format

The function returns a comprehensive JSON structure:

```json
{
  "success": true,
  "processing_time_seconds": 15.23,
  "summary": {
    "articles_retrieved": 25,
    "chunks_generated": 87,
    "total_word_count": 12450,
    "companies_covered": ["APPLE", "AAPL", "MICROSOFT"],
    "date_range": {
      "earliest": "2024-01-01T00:00:00",
      "latest": "2024-01-07T23:59:59"
    }
  },
  "chunks": [
    {
      "chunk_id": "article_123_chunk_0",
      "content": "Apple reported strong quarterly earnings...",
      "metadata": {
        "title": "Apple Beats Earnings Expectations",
        "source": "reuters.com",
        "published_date": "2024-01-05T10:30:00",
        "companies_mentioned": ["APPLE", "AAPL"],
        "category": "earnings",
        "url": "https://reuters.com/...",
        "word_count": 156,
        "chunk_position": "1/3"
      },
      "source_url": "https://reuters.com/...",
      "word_count": 156,
      "char_count": 892
    }
  ],
  "errors": [],
  "timestamp": "2024-01-07T12:00:00"
}
```

## Advanced Usage

### Custom Configuration

```python
from NewsRAG import get_company_news_chunks, NewsConfig

# Custom configuration
config = NewsConfig(
    chunk_size=800,
    chunk_overlap=150,
    min_word_count=100,
    max_articles_per_query=15
)

result = get_company_news_chunks(
    companies=["Tesla", "TSLA"],
    categories=["earnings", "products", "regulatory"],
    days_back=30,
    config=config
)
```

### Using Individual Components

```python
from NewsRAG import TavilyDirectClient, NewsConfig
from NewsRAG.models import NewsQuery, NewsCategory

# Setup
config = NewsConfig.default()
client = TavilyDirectClient(config)

# Create query
query = NewsQuery(
    companies=["Apple"],
    categories=[NewsCategory.EARNINGS],
    days_back=7
)

# Get chunks directly (no separate processing steps needed)
chunks = client.search_news_chunks(query)
```

## Rate Limiting

The package includes built-in rate limiting to respect Tavily API limits:
- Default: 60 requests per minute
- Automatic retry with exponential backoff
- Configurable via environment variables

## Error Handling

The package handles various error scenarios:
- API rate limiting
- Network timeouts
- Invalid API responses
- Content processing errors

All errors are logged and included in the response for debugging.

## Contributing

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include logging for debugging
4. Update tests for new features
5. Maintain backward compatibility

## Dependencies

- `tavily-python`: Tavily API client
- `python-dateutil`: Date parsing and handling

Optional dependencies for enhanced features:
- `nltk` or `spacy`: Advanced text processing
- `transformers`: Better company name extraction
- `redis`: Result caching

## License

This package is part of the PitchBook Generator project.

## Support

For issues and feature requests, please check the existing documentation or create an issue in the project repository.