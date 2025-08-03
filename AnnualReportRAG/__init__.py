"""
AnnualReportRAG Package

This package provides tools for downloading, processing, and chunking SEC annual reports
and filings for Retrieval-Augmented Generation (RAG) applications.
"""

from .get_filing_chunks import get_filing_chunks_api, process_multiple_companies_filings
from .metadata_extractor import MetadataExtractor
from .content_processor import ContentProcessor
from .chunk_generator import ChunkGenerator
from .config_manager import get_config

__version__ = "1.0.0"

__all__ = [
    "get_filing_chunks_api",
    "process_multiple_companies_filings",
    "MetadataExtractor",
    "ContentProcessor",
    "ChunkGenerator",
    "get_config"
]
