# AnnualReportRAG Examples

Interactive examples for using the AnnualReportRAG package.

## Files

- **`ingest_reports.py`**: Interactive script to ingest SEC filings for a company
- **`search_reports.py`**: Interactive script to search through indexed reports

## Usage

Make sure you're in the examples directory and have set your environment variables:

```bash
cd AnnualReportRAG/examples/
python ingest_reports.py    # Index reports for a company
python search_reports.py    # Search indexed reports
```

## Environment Setup

Ensure your `.env` file is configured with at least:
```bash
SEC_USER_AGENT=your.email@company.com
```

## Example Workflow

1. First, ingest some reports:
   ```bash
   cd AnnualReportRAG/examples/
   python ingest_reports.py
   # Enter: AAPL
   # Years back: 2
   # Filing types: 10-K,10-Q
   ```

2. Then search them:
   ```bash
   python search_reports.py
   # Enter: AAPL
   # Query: What was the revenue growth in Q3?
   # Top-K: 10
   ```

## Output Files

Both scripts save their results as JSON files in organized data folders:
- `../data/summaries/ingestion_summary_TICKER_TIMESTAMP.json` (ingestion summaries)
- `../data/search_results/search_result_TICKER_TIMESTAMP.json` (search results)

These files can be analyzed or processed by other tools.