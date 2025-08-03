# Project Structure

This document outlines the complete structure of the PitchBookGenerator project.

```
PitchBookGenerator/
├── README.md                              # Main project documentation
├── requirements.txt                       # Main project dependencies  
├── .gitignore                            # Git ignore rules
├── .env.example                          # Example environment configuration
├── LICENSE                               # MIT License
├── setup.py                              # Automated setup script
├── run_sec_rag.py                        # Main CLI entry point (wrapper)
│
├── SECFinancialRAG/                      # Core financial data processing
│   ├── __init__.py                       # Package initialization
│   ├── requirements.txt                  # Package-specific dependencies
│   ├── .env.example                      # Environment template
│   ├── .gitignore                        # Package-specific ignores
│   ├── README.md                         # Package documentation
│   ├── sample_output.json                # Example API response
│   ├── sample_ltm_output.csv             # Example LTM export
│   ├── run_sec_rag.py                    # CLI interface
│   ├── main.py                           # Main processing logic
│   ├── database.py                       # PostgreSQL database operations
│   ├── models.py                         # Pydantic data models
│   ├── mapping.py                        # GAAP to database field mapping
│   ├── simplified_processor.py           # Financial statement processor
│   ├── sec_client.py                     # SEC API client
│   ├── ltm_calculator.py                 # LTM calculation engine
│   └── ltm_exports/                      # Generated LTM CSV files (ignored)
│
├── AnnualReportRAG/                      # Annual report text processing
│   ├── __init__.py                       # Package initialization
│   ├── requirements.txt                  # Package-specific dependencies
│   ├── .env.example                      # Environment template
│   ├── .gitignore                        # Package-specific ignores
│   ├── README.md                         # Package documentation
│   ├── sample_output.json                # Example API response
│   ├── get_filing_chunks.py              # Main API and processing
│   ├── metadata_extractor.py             # SEC filing metadata extraction
│   ├── content_processor.py              # Document content processing
│   ├── chunk_generator.py                # RAG chunk generation
│   ├── config_manager.py                 # Configuration management
│   ├── filings/                          # Downloaded SEC filings (ignored)
│   │   └── [ticker]/                     # Company-specific directories
│   │       ├── 10-K/                     # Annual reports
│   │       ├── 10-Q/                     # Quarterly reports
│   │       └── 8-K/                      # Current reports
│   ├── rag_ready_data/                   # Processed chunks (ignored)
│   └── batch_processing_results.json     # Processing logs (ignored)
│
├── TranscriptRAG/                        # Earnings transcript processing
│   ├── __init__.py                       # Package initialization
│   ├── requirements.txt                  # Package-specific dependencies
│   ├── .env.example                      # Environment template
│   ├── .gitignore                        # Package-specific ignores
│   ├── README.md                         # Package documentation
│   ├── sample_output.json                # Example API response
│   ├── sample_transcript_chunks.csv      # Example transcript chunks
│   ├── get_transcript_chunks.py          # Main API function
│   ├── data_source_interface.py          # Abstract data source interface
│   ├── alpha_vantage_source.py           # Alpha Vantage integration
│   ├── transcript_config.py              # Configuration management
│   ├── transcript_metadata_extractor.py  # Transcript metadata extraction
│   ├── transcript_content_processor.py   # Transcript content processing
│   ├── transcript_chunk_generator.py     # Transcript-specific chunking
│   ├── create_consolidated_chunks.py     # Chunk consolidation
│   └── transcript_results/               # Processed transcripts (ignored)
│
├── ltm_exports/                          # LTM CSV exports (ignored)
├── logs/                                 # Application logs (ignored)
└── __pycache__/                          # Python cache files (ignored)
```

## Package Responsibilities

### SECFinancialRAG
- **Primary Purpose**: Extract structured financial data from SEC filings
- **Data Types**: Income statements, balance sheets, cash flow statements  
- **Storage**: PostgreSQL database with normalized schemas
- **Output**: LTM CSV files, database records, financial metrics
- **Key Features**: GAAP mapping, data validation, smart deduplication

### AnnualReportRAG
- **Primary Purpose**: Process SEC documents for text-based RAG applications
- **Data Types**: Full document text, sections, paragraphs
- **Storage**: JSON files with text chunks and metadata
- **Output**: RAG-ready chunks with rich metadata
- **Key Features**: Section extraction, content cleaning, chunk optimization

### TranscriptRAG
- **Primary Purpose**: Process earnings call transcripts for conversational analysis
- **Data Types**: Transcript text, speaker segments, Q&A sections
- **Storage**: JSON files with speaker-aware chunks
- **Output**: Conversational chunks with financial context
- **Key Features**: Speaker identification, earnings metrics, Alpha Vantage integration

## Data Flow

1. **SECFinancialRAG**: SEC API → Financial Data → PostgreSQL → LTM CSV
2. **AnnualReportRAG**: SEC EDGAR → Document Text → Sections → RAG Chunks
3. **TranscriptRAG**: Alpha Vantage → Transcript Data → Speaker Segments → Conversational Chunks

## Integration Points

- All packages share similar API interfaces for consistency
- Common environment variable patterns across packages
- Standardized output formats (JSON responses)
- Compatible with downstream RAG and analysis systems
- Coordinated gitignore patterns to exclude large data files

## Development Workflow

1. **Individual Package Development**: Each package can be developed and tested independently
2. **Shared Standards**: Common patterns for configuration, logging, and error handling
3. **Git Workflow**: Package-specific branches with coordinated merges to main
4. **Documentation**: Each package maintains its own README with examples
5. **Testing**: Package-specific test suites with integration tests
