SEC Financial RAG Package

To run the code from the project root:
python run_sec_rag.py MSFT --ltm
python SECFinancialRAG/run_sec_rag.py MSFT --ltm

To run directly from this folder:
python run_sec_rag.py MSFT --ltm

Examples:
- Process single company: python run_sec_rag.py AAPL
- Process multiple companies: python run_sec_rag.py AAPL MSFT GOOGL
- With validation and LTM: python run_sec_rag.py AAPL --validate --ltm
- Help: python run_sec_rag.py --help
