# AnnualReportRAG

This module handles the processing of SEC annual reports and filings for RAG (Retrieval-Augmented Generation) applications.

## Directory Structure

```
AnnualReportRAG/
├── get_filing_chunks.py       # Main API for downloading and processing SEC filings
├── metadata_extractor.py      # Extracts metadata from SEC filings
├── content_processor.py       # Processes and cleans filing content
├── chunk_generator.py         # Generates RAG-ready text chunks
├── config_manager.py          # Configuration management
├── filings/                   # Downloaded raw SEC filings (excluded from git)
│   └── [ticker]/              # Company-specific filing directories
│       ├── 10-K/              # 10-K annual reports
│       ├── 10-Q/              # 10-Q quarterly reports
│       └── 8-K/               # 8-K current reports
├── rag_ready_data/            # Processed RAG datasets (excluded from git)
└── batch_processing_results.json  # Processing results log (excluded from git)
```

## Data Storage

- **Raw Filings**: Downloaded SEC filings are stored in the `filings/` directory, organized by ticker symbol and filing type
- **Processed Data**: RAG-ready chunks and metadata are stored in the `rag_ready_data/` directory
- **Results**: Processing results and logs are stored in `batch_processing_results.json`

All data directories are excluded from version control via `.gitignore` to keep the repository clean and avoid committing large data files.

## Usage

```python
from get_filing_chunks import get_filing_chunks_api

# Process SEC filings for companies
results = get_filing_chunks_api(
    tickers=['AAPL', 'MSFT'],
    start_date='2024-01-01',
    years_back=2,
    filing_types=['10-K', '10-Q']
)
```

The processed data will be automatically stored in the local `filings/` and `rag_ready_data/` directories within this module.
