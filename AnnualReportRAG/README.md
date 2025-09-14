# AnnualReportRAG

A production-ready Python package for downloading, processing, and indexing SEC annual reports and filings for Retrieval-Augmented Generation (RAG) applications. Features intelligent caching, dual granularity chunking, and agent-ready APIs.

## 🚀 Key Features

- **Agent-Ready APIs**: Simple functions designed for AI agent integration
- **Smart Caching**: Filing-level deduplication saves 70%+ on embedding costs
- **Dual Granularity Chunking**: Base chunks (1800 chars) + micro chunks (350 chars) for better retrieval
- **Intelligent Processing**: Automatic section detection, metadata extraction, and content cleaning
- **Vector Database Integration**: Built-in LanceDB support with hybrid search
- **Auto-Cleanup**: Removes raw files after successful indexing
- **Secure by Design**: Proper input validation and no credential exposure

## 📦 Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Set up your environment variables (copy from .env template)
cp .env.template .env
# Edit .env with your SEC_USER_AGENT email
```

## ⚙️ Configuration

All settings are configured via environment variables in your `.env` file:

```bash
# Required: SEC API requires a valid email for user identification
SEC_USER_AGENT=your.email@company.com

# AnnualReportRAG Chunking Configuration
ANNUAL_RAG_BASE_CHUNK_SIZE=1800          # Base chunk size for semantic context
ANNUAL_RAG_BASE_CHUNK_OVERLAP=250        # Base chunk overlap
ANNUAL_RAG_MICRO_CHUNK_SIZE=350          # Micro chunk size for fine-grained retrieval
ANNUAL_RAG_MICRO_CHUNK_OVERLAP=60        # Micro chunk overlap

# Feature Toggles
ANNUAL_RAG_ENABLE_DUAL_GRANULARITY=true  # Enable dual granularity expansion
ANNUAL_RAG_ENABLE_AUTO_CLEANUP=true      # Remove raw files after vector DB indexing
ANNUAL_RAG_ENABLE_VECTOR_CACHE=true      # Check for existing reports before processing

# Optional: LLM refinement for search queries
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_REFINEMENT_MODEL=openai/gpt-4.1-nano
```

## 🎯 Agent-Ready API

### Indexing Reports

```python
from AnnualReportRAG import index_reports_for_agent

# Index reports for a company
result = index_reports_for_agent(
    ticker="AAPL",
    years_back=3,
    filing_types=["10-K", "10-Q"],  # Optional: defaults to ["10-K", "10-Q"]
    force_refresh=False  # Optional: bypass cache if True
)

# Response structure
{
    "success": True,
    "ticker": "AAPL",
    "chunk_count": 1247,
    "filings_used": [...],
    "vector_storage": {
        "success": True,
        "source": "new_indexing",  # or "cache_hit"
        "chunks_indexed": 1247
    },
    "enhancements": {
        "dual_granularity": True,
        "auto_cleanup": True,
        "vector_cache": True
    }
}
```

### Searching Reports

```python
from AnnualReportRAG import search_report_for_agent

# Search with natural language queries
results = search_report_for_agent(
    ticker="AAPL",
    query="What was the revenue growth in Q3 2024?",
    k=10,  # Number of results
    filters={  # Optional metadata filtering
        "form_type": ["10-Q"],
        "fiscal_year": [2024],
        "section_name": ["Revenue", "Management Discussion"]
    },
    enable_llm_refinement=True  # Optional: refine query with LLM
)

# Response structure
{
    "success": True,
    "ticker": "AAPL",
    "query": "What was the revenue growth in Q3 2024?",
    "results": [
        {
            "content": "Revenue increased 5% year-over-year to $94.9 billion...",
            "form_type": "10-Q",
            "fiscal_year": 2024,
            "fiscal_quarter": "Q3",
            "section_name": "Revenue",
            "filing_date": "2024-08-01",
            "granularity": "base",  # "base" or "micro"
            "score": 0.94,
            "metadata": {...}
        }
    ],
    "total_found": 10,
    "search_metadata": {...}
}
```

## 📁 Directory Structure

```
AnnualReportRAG/
├── README.md                   # This file
├── __init__.py                 # Package exports
├── requirements.txt            # Dependencies
├── examples/                   # Usage examples
│   ├── ingest_reports.py      # Interactive ingestion example
│   └── search_reports.py      # Interactive search example
├── agent_interface.py          # Main agent-ready API functions
├── annual_vector_store.py      # Enhanced vector database operations
├── get_filing_chunks.py        # SEC filing download and processing
├── metadata_extractor.py       # Filing metadata extraction
├── content_processor.py        # Content cleaning and section detection
├── chunk_generator.py          # RAG-optimized chunking
├── config_manager.py           # Configuration management
├── filing_manager.py           # Filing organization utilities
├── filings/                    # Downloaded raw SEC filings (git-ignored)
├── rag_ready_data/            # Processed datasets (git-ignored)
└── data/                      # Vector database storage (git-ignored)
```

## 🔄 Caching & Efficiency

The system includes intelligent caching to avoid reprocessing:

- **Filing-Level Deduplication**: Checks vector DB for existing reports by metadata (form_type + fiscal_year + quarter)
- **Cost Savings**: Typically saves 70%+ on embedding costs for subsequent runs
- **Smart Cache Detection**: Automatically identifies partial cache hits and only processes missing reports
- **Auto-Cleanup**: Removes raw downloaded files after successful vector indexing

Example cache output:
```
📥 Downloaded 1132 chunks from 16 filings, checking for duplicates...
🔍 Found 12 existing filings in vector DB for AAPL
🔄 Smart deduplication: Skipped 12/16 already-cached filings (836 chunks)
💰 Saved embedding costs for 836 chunks (73.9% savings)
```

## 🛡️ Security Features

- **Input Validation**: Ticker symbols validated (1-5 alphabetic characters only)
- **Query Length Limits**: Prevents DoS attacks via massive queries
- **Secure Credential Handling**: API keys loaded from environment, never logged
- **No Code Injection**: No eval(), exec(), or unsafe subprocess calls
- **Parameterized Queries**: Safe database operations via LanceDB

## 🎮 Usage Examples

See the `examples/` directory for interactive examples:

```bash
cd AnnualReportRAG/examples/
python ingest_reports.py    # Interactive report ingestion
python search_reports.py    # Interactive report searching
```

## 🔧 Advanced Configuration

### Dual Granularity Chunking

When enabled, the system creates two types of chunks for optimal retrieval:

- **Base Chunks** (1800 chars): Provide semantic context for complex queries
- **Micro Chunks** (350 chars): Enable fine-grained retrieval for specific facts

### LLM Query Refinement

Enable intelligent query refinement for better search results:

```python
results = search_report_for_agent(
    ticker="AAPL",
    query="revenue trends",
    enable_llm_refinement=True,
    refinement_model="openai/gpt-4.1-nano"
)
```

The LLM will refine vague queries and suggest appropriate metadata filters.

## 📊 Supported Filing Types

- **10-K**: Annual reports with comprehensive company information
- **10-Q**: Quarterly reports with financial updates
- **8-K**: Current reports for material events (optional)

## 🔗 Integration with Agent Frameworks

This package is designed for easy integration with AI agent frameworks:

```python
# Example agent tool definition
def get_company_financial_data(ticker: str, query: str) -> dict:
    """Tool for agents to query SEC filings"""
    return search_report_for_agent(ticker=ticker, query=query, k=5)

def index_company_reports(ticker: str, years: int = 2) -> dict:
    """Tool for agents to index company reports"""
    return index_reports_for_agent(ticker=ticker, years_back=years)
```

## 🐛 Troubleshooting

### Common Issues

1. **SEC_USER_AGENT not set**: Ensure your email is configured in `.env`
2. **Rate limiting**: The system includes built-in rate limiting for SEC API compliance
3. **Large downloads**: Enable auto-cleanup to manage storage space
4. **Vector DB errors**: Check that the `data/` directory is writable

### Debug Logging

Set log level for detailed output:
```bash
LOG_LEVEL=DEBUG
```

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

This package is designed for production use in agentic frameworks. For issues or enhancements, please follow secure development practices and maintain the existing security standards.