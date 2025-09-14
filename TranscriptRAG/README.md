# TranscriptRAG - Production-Ready Transcript Processing for AI Agents

A clean, efficient system for processing earnings call transcripts optimized for AI agents and LLM frameworks.

## 🤖 **Agent-Ready Features**

- **Standardized Responses**: Consistent error codes, metadata, and request tracking
- **Intelligent Quarter Selection**: See "Q1 2024", "Q3 2025" format with years
- **Accurate Scoring**: Fixed hybrid search showing proper relevance scores (no more 0.000)
- **Input Validation**: Comprehensive validation with helpful error messages
- **Vector Database**: LanceDB integration with transcript-specific metadata
- **Flexible Search**: Vector similarity + keyword search with quarter filtering

## 🚀 **Quick Start for Agents**

### Required Setup
```bash
# Set environment variable
export ALPHA_VANTAGE_API_KEY="your_api_key_here"

# Install dependencies  
pip install lancedb>=0.13.0 rank-bm25>=0.2.2 sentence-transformers
```

### Agent Interface Usage

```python
from TranscriptRAG import (
    index_transcripts_for_agent,
    search_transcripts_for_agent, 
    get_available_quarters_for_agent
)

# 1. Index transcripts for a company
result = index_transcripts_for_agent(
    ticker="AAPL",
    quarters_back=4  # Last 4 quarters
)

if result["success"]:
    print(f"Indexed {result['data']['chunk_count']} chunks")

# 2. Get available quarters (intelligent selection)
quarters_info = get_available_quarters_for_agent("AAPL")
if quarters_info["success"]:
    available = quarters_info["data"]["available_quarters"]
    print(f"Available: {available}")  # ['Q1 2024', 'Q4 2023', 'Q3 2023']

# 3. Search with specific quarters (RECOMMENDED)
search_result = search_transcripts_for_agent(
    ticker="AAPL",
    query="latest guidance and outlook",
    quarters=["Q1 2024", "Q4 2023"],  # Specific quarters with years
    k=10,
    search_method="vector_hybrid"
)

if search_result["success"]:
    for chunk in search_result["data"]["results"]:
        score = chunk["relevance_score"]  # Proper scoring (not 0.000)
        quarter = chunk["metadata"]["quarter"] 
        content = chunk["content"][:100]
        print(f"Score: {score:.3f} | {quarter} | {content}...")
```

## 📚 **Agent Interface Reference**

### `get_available_quarters_for_agent(ticker)`
Returns available quarters for intelligent selection.

**Response:**
```json
{
  "success": true,
  "data": {
    "ticker": "AAPL",
    "total_documents": 244,
    "available_quarters": ["Q1 2024", "Q4 2023", "Q3 2023", "Q2 2023"],
    "agent_recommendations": {
      "suggested_quarters_recent": ["Q1 2024", "Q4 2023", "Q3 2023"],
      "search_tips": ["Use specific quarters for focused results"]
    }
  }
}
```

### `search_transcripts_for_agent(ticker, query, quarters=None, k=20)`
Search with intelligent quarter selection and proper scoring.

**Parameters:**
- `ticker` (str): Company ticker (e.g., "AAPL")
- `query` (str): Search query
- `quarters` (list): Specific quarters like ["Q1 2024", "Q2 2024"] (RECOMMENDED)
- `quarters_back` (int): Legacy parameter (use quarters instead)
- `k` (int): Number of results (1-100)
- `search_method` (str): "vector_hybrid", "vector_semantic", "keyword"

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "content": "Management discussed strong revenue growth...",
        "metadata": {"quarter": "Q1 2024", "speaker": "CEO"},
        "relevance_score": 0.847,  // Proper scoring
        "chunk_id": "aapl_123"
      }
    ],
    "returned": 10,
    "search_method": "vector_hybrid",
    "quarters_searched": ["Q1 2024", "Q4 2023"],
    "available_quarters_info": {
      "available_quarters": ["Q1 2024", "Q4 2023", "Q3 2023"]
    }
  }
}
```

### `index_transcripts_for_agent(ticker, quarters_back=4)`
Index transcripts into vector database.

**Response:**
```json
{
  "success": true,
  "data": {
    "ticker": "AAPL",
    "chunk_count": 244,
    "transcripts_used": [
      {
        "quarter": "Q1 2024",
        "fiscal_year": "2024", 
        "chunk_count": 58
      }
    ],
    "vector_storage": {
      "success": true,
      "table_name": "transcripts_aapl"
    }
  }
}
```

## 🎯 **Agent Best Practices**

### 1. Always Check Available Quarters First
```python
# ✅ GOOD: Check what's available
quarters_info = get_available_quarters_for_agent("AAPL")
available = quarters_info["data"]["available_quarters"]

# Use specific quarters
search_result = search_transcripts_for_agent(
    ticker="AAPL", 
    query="guidance",
    quarters=available[:2]  # Most recent 2 quarters
)
```

### 2. Handle Errors Gracefully
```python
result = search_transcripts_for_agent("INVALID", "test")
if not result["success"]:
    error = result["error"]
    print(f"Error [{error['code']}]: {error['message']}")
    # Handle specific error codes: INVALID_INPUT, NOT_FOUND, etc.
```

### 3. Use Proper Scoring
```python
results = search_result["data"]["results"]
for chunk in results:
    score = chunk["relevance_score"]  # Always > 0 now
    if score > 0.7:
        print("High relevance chunk")
    elif score > 0.4:
        print("Medium relevance chunk")
```

## 🧪 **Interactive Examples**

Test the interface with examples:

```bash
cd TranscriptRAG/examples/

# Interactive transcript ingestion
python ingest_transcripts.py

# Interactive search with quarter selection
python search_transcripts.py
```

## 📁 **Project Structure**

```
TranscriptRAG/
├── __init__.py                    # Agent-ready API exports
├── agent_interface.py             # Primary agent interface
├── transcript_vector_store.py     # LanceDB vector database
├── get_transcript_chunks.py       # Core processing
├── examples/
│   ├── ingest_transcripts.py      # Example: indexing
│   └── search_transcripts.py      # Example: searching
└── README.md                      # This file
```

## ⚙️ **Configuration**

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ALPHA_VANTAGE_API_KEY` | **Required** | Alpha Vantage API key |
| `TRANSCRIPT_RAG_CHUNK_SIZE` | `800` | Chunk size for processing |
| `TRANSCRIPT_RAG_CHUNK_OVERLAP` | `150` | Overlap between chunks |

## 🔧 **Advanced Features**

### Search Filtering
```python
search_result = search_transcripts_for_agent(
    ticker="AAPL",
    query="iPhone sales",
    quarters=["Q1 2024"],
    k=10,
    filters={
        "section_name": "Q&A Session",  # Specific sections
        "speaker": "CEO"                # Specific speakers
    }
)
```

### Multiple Search Methods
```python
# Hybrid search (recommended) - combines semantic + keyword
search_transcripts_for_agent(..., search_method="vector_hybrid")

# Pure semantic search - meaning-based
search_transcripts_for_agent(..., search_method="vector_semantic") 

# Keyword search - exact term matching
search_transcripts_for_agent(..., search_method="keyword")
```

## 📊 **Error Codes**

| Code | Description | Solution |
|------|-------------|----------|
| `INVALID_INPUT` | Invalid parameters | Check ticker format, query length |
| `NOT_FOUND` | No data for ticker | Run `index_transcripts_for_agent()` first |
| `API_ERROR` | External API failure | Check API key, network connection |
| `VECTOR_DB_ERROR` | Database error | Check dependencies, disk space |

## 🚀 **Production Readiness**

✅ **Agent-Ready Features:**
- Standardized error codes and responses
- Comprehensive input validation  
- Request tracking with correlation IDs
- Proper relevance scoring (no more 0.000)
- Intelligent quarter selection with years
- Backwards compatibility maintained

✅ **Performance Optimized:**
- Vector database caching
- Efficient hybrid search
- Minimal dependencies
- Clean codebase without test artifacts

---

**Version 2.0.0** - Production-ready for agentic frameworks 🤖