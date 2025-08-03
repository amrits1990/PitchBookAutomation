# TranscriptRAG - Earnings Call Transcript Processing System

An independent system for processing earnings call transcripts and creating RAG-ready chunks, designed to work alongside the AnnualReportRAG system.

## Features

- 🔌 **Pluggable Data Sources**: Abstract interface supporting multiple transcript providers
- 📊 **Alpha Vantage Integration**: Built-in support for Alpha Vantage earnings data API
- 📝 **Section-Aware Processing**: Intelligent parsing of transcript sections (Opening Remarks, Q&A, etc.)
- 🗣️ **Speaker-Aware Chunking**: Optional chunking that respects speaker boundaries
- 📈 **Rich Metadata Extraction**: Comprehensive metadata including financial metrics and sentiment
- ⚙️ **Configurable Processing**: Environment-based configuration with validation
- 🔍 **RAG-Optimized Output**: Chunks optimized for retrieval and question-answering

## Quick Start

### 1. Environment Setup

```bash
# Required: Alpha Vantage API Key
export ALPHA_VANTAGE_API_KEY="your_api_key_here"

# Optional: Configuration
export DEFAULT_CHUNK_SIZE="800"
export DEFAULT_OVERLAP="150"
export ENABLE_SPEAKER_CHUNKING="true"
```

### 2. Basic Usage

```python
from TranscriptRAG import get_transcript_chunks

# Process transcripts for multiple companies
results = get_transcript_chunks(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2024-01-01',
    years_back=2,
    chunk_size=800,
    overlap=150,
    return_full_data=True
)

if results['status'] == 'success':
    for transcript in results['successful_transcripts']:
        print(f"{transcript['ticker']} {transcript['quarter']} - {transcript['chunk_count']} chunks")
```

## Architecture

### Core Components

1. **Data Source Interface** (`data_source_interface.py`)
   - Abstract interface for transcript providers
   - Standardized data structures
   - Registry for multiple sources

2. **Alpha Vantage Source** (`alpha_vantage_source.py`)
   - Alpha Vantage API integration
   - Rate limiting and error handling
   - Earnings data transformation

3. **Content Processor** (`transcript_content_processor.py`)
   - Transcript cleaning and parsing
   - Speaker identification
   - Section detection (Opening Remarks, Q&A, etc.)

4. **Metadata Extractor** (`transcript_metadata_extractor.py`)
   - Financial metrics extraction
   - Sentiment analysis
   - Topic identification

5. **Chunk Generator** (`transcript_chunk_generator.py`)
   - Standard text chunking
   - Speaker-aware chunking for Q&A sections
   - Overlap handling

6. **Configuration** (`transcript_config.py`)
   - Environment-based configuration
   - Parameter validation
   - Rate limiting settings

## API Reference

### `get_transcript_chunks()`

Main API function that mirrors the interface of `get_filing_chunks()` from AnnualReportRAG.

```python
def get_transcript_chunks(
    tickers: List[str],                    # Company ticker symbols
    start_date: str,                       # Start date (YYYY-MM-DD)
    years_back: int = 3,                   # Years to look back
    chunk_size: int = 800,                 # Chunk size (100-2000)
    overlap: int = 150,                    # Overlap between chunks
    limit_per_ticker: Optional[int] = None, # Max transcripts per ticker
    use_speaker_chunking: bool = None,     # Enable speaker-aware chunking
    return_full_data: bool = False,        # Include full datasets
    output_dir: str = None,                # Save results directory
    correlation_id: str = None             # Request tracking ID
) -> Dict
```

### Response Structure

```json
{
    "status": "success",
    "correlation_id": "uuid-string",
    "summary": {
        "total_tickers": 3,
        "total_transcripts_processed": 12,
        "total_transcripts_failed": 0,
        "processing_start_time": "2024-07-26T10:00:00",
        "processing_end_time": "2024-07-26T10:05:00"
    },
    "successful_transcripts": [
        {
            "ticker": "AAPL",
            "transcript_date": "2024-01-25",
            "quarter": "Q1",
            "fiscal_year": "2024",
            "transcript_type": "earnings_call",
            "chunk_count": 42,
            "processed_at": "2024-07-26T10:01:00"
        }
    ],
    "failed_transcripts": []
}
```

### Chunk Structure

Each chunk contains:

```json
{
    "chunk_id": 0,
    "global_chunk_id": 0,
    "text": "Management Remarks: We had a strong quarter...",
    "length": 756,
    "metadata": {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "quarter": "Q1",
        "fiscal_year": "2024",
        "section_name": "Management Remarks",
        "content_type": "transcript",
        "transcript_date": "2024-01-25T00:00:00",
        "chunk_length": 756,
        "speakers": ["CEO", "CFO"],
        "financial_metrics": {
            "eps_reported": 2.18,
            "eps_estimated": 2.10,
            "eps_surprise": 0.08
        }
    }
}
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPHA_VANTAGE_API_KEY` | **Required** | Alpha Vantage API key |
| `DEFAULT_CHUNK_SIZE` | `800` | Default chunk size |
| `DEFAULT_OVERLAP` | `150` | Default overlap between chunks |
| `MAX_TICKERS_PER_REQUEST` | `10` | Maximum tickers per API call |
| `MAX_TRANSCRIPTS_PER_TICKER` | `50` | Maximum transcripts per ticker |
| `ENABLE_SPEAKER_CHUNKING` | `true` | Enable speaker-aware chunking |
| `ALPHA_VANTAGE_RATE_LIMIT` | `5` | API calls per minute |
| `ALPHA_VANTAGE_DELAY` | `12` | Seconds between API calls |

## Data Sources

### Alpha Vantage

- **Data Type**: Earnings summaries with financial metrics
- **Coverage**: ~10 years of historical data
- **Rate Limits**: 5 calls/minute (free tier), 500 calls/day
- **Note**: Provides earnings data, not actual transcript text

### Adding New Data Sources

Implement the `TranscriptDataSource` interface:

```python
from data_source_interface import TranscriptDataSource, TranscriptData, TranscriptQuery

class CustomTranscriptSource(TranscriptDataSource):
    def get_transcripts(self, query: TranscriptQuery) -> List[TranscriptData]:
        # Implementation here
        pass
    
    def get_latest_transcript(self, ticker: str) -> Optional[TranscriptData]:
        # Implementation here
        pass
    
    # ... other required methods

# Register with the system
from data_source_interface import transcript_registry
transcript_registry.register_source('custom', CustomTranscriptSource())
```

## Error Handling

The system provides comprehensive error handling:

- **Validation Errors**: Invalid parameters, missing API keys
- **Network Errors**: API timeouts, connection issues
- **Rate Limiting**: Automatic delays, retry logic
- **Data Errors**: Malformed responses, missing data

All errors include correlation IDs for tracking and debugging.

## Comparison with AnnualReportRAG

| Feature | AnnualReportRAG | TranscriptRAG |
|---------|-----------------|---------------|
| **Data Source** | SEC EDGAR filings | Alpha Vantage earnings data |
| **Content Type** | Form 10-K/10-Q sections | Earnings call summaries |
| **Sectioning** | SEC filing sections | Transcript sections (Remarks, Q&A) |
| **Special Features** | Financial table extraction | Speaker-aware chunking |
| **Use Case** | Regulatory document analysis | Earnings call analysis |

## License

Part of the Capstone project for Gen AI Engineering Fellowship.