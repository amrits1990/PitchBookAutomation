# TranscriptRAG Agent Interface Examples

This directory contains interactive examples demonstrating the **agent-ready** TranscriptRAG interface with standardized responses, comprehensive error handling, and request tracking.

## 🚀 What's New in the Agent Interface

✅ **Standardized Response Format** - All functions return consistent JSON structure  
✅ **Error Codes** - Specific error codes (INVALID_INPUT, API_ERROR, etc.)  
✅ **Request Tracking** - Unique request IDs and processing time metrics  
✅ **Comprehensive Validation** - Input validation with detailed error messages  
✅ **Agent-Ready** - Designed for seamless integration with agentic frameworks

## 📋 Examples

### `ingest_transcripts.py` - Agent-Ready Ingestion
Interactive script demonstrating the `index_transcripts_for_agent()` function:
- Fetches and processes earnings call transcripts
- Shows standardized response format with success/error handling
- Displays request tracking (IDs, processing time)
- Comprehensive error reporting with error codes
- Saves detailed summaries with metadata

**Usage:**
```bash
cd TranscriptRAG/examples/
python ingest_transcripts.py
```

### `search_transcripts.py` - Agent-Ready Search
Interactive script demonstrating the `search_transcripts_for_agent()` function:
- Searches indexed transcript chunks
- Shows various search methods (vector_hybrid, vector_semantic, keyword)
- Demonstrates result evaluation and filtering
- Error handling with specific error codes
- Request tracking and performance metrics

**Usage:**
```bash
cd TranscriptRAG/examples/
python search_transcripts.py
```

## 🔧 Requirements

Before running the examples, ensure you have:

1. **API Key**: Set `ALPHA_VANTAGE_API_KEY` environment variable
2. **TranscriptRAG**: Properly installed with agent interface
3. **Python Packages**: All required dependencies installed

## 🎯 Testing the Agent Interface

These examples are perfect for testing the agent interface manually:

1. **Start with Ingestion**: Run `ingest_transcripts.py` to index some transcripts
2. **Test Search**: Run `search_transcripts.py` to search the indexed data
3. **Observe Response Format**: Notice the consistent JSON structure
4. **Test Error Cases**: Try invalid inputs to see error handling
5. **Check Metadata**: Look for request IDs and processing times

## 📊 Standardized Response Format

Both examples demonstrate the new standardized response format:

```json
{
  "success": true/false,
  "data": {
    // Function-specific successful response data
  },
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {} // Additional error context
  },
  "metadata": {
    "function": "function_name",
    "timestamp": "2024-01-01T12:00:00",
    "request_id": "uuid-string",
    "processing_time_ms": 1234.5
  }
}
```

## 🆔 Error Codes

The agent interface uses specific error codes for reliable error handling:

- `INVALID_INPUT` - Input validation failed
- `CONFIG_ERROR` - Configuration issues (missing API key, etc.)
- `API_ERROR` - External API failures
- `VECTOR_DB_ERROR` - Vector database issues
- `NOT_FOUND` - No data found for request
- `PROCESSING_ERROR` - Unexpected processing errors

## 💾 Output Files

Examples save results to organized directories:
- Ingestion summaries: `TranscriptRAG/data/summaries/`
- Search results: `TranscriptRAG/data/search_results/`

## 🧪 Manual Testing Scenarios

Try these scenarios to test the agent interface:

### ✅ Success Cases
- Ingest transcripts for a valid ticker (e.g., AAPL, MSFT)
- Search with simple queries like "revenue growth" or "guidance"
- Try different search methods and parameters

### ❌ Error Cases
- Empty ticker or invalid ticker format
- Empty search queries
- Invalid parameter ranges (k > 100, quarters_back > 20)
- Missing API key (unset ALPHA_VANTAGE_API_KEY)

### 📈 Performance Testing  
- Large number of results (k=100)
- Multiple quarters (quarters_back=8)
- Complex queries with multiple terms

Each test will show the standardized response format with appropriate success/error handling.