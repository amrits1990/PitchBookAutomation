"""
TranscriptRAG - Production-Ready Transcript Processing for Agentic Frameworks

A clean, efficient system for processing earnings call transcripts optimized for AI agents.
Provides standardized interfaces, proper error handling, and intelligent quarter selection.

Key Features:
- Agent-ready interfaces with standardized responses
- Quarter selection with year context (Q1 2024, Q3 2025)
- Hybrid vector + keyword search with accurate scoring
- Comprehensive input validation and error handling
- LanceDB vector database integration
"""

# Core processing function
from .get_transcript_chunks import get_transcript_chunks

# Data source interfaces  
from .data_source_interface import TranscriptData, TranscriptQuery, transcript_registry
from .alpha_vantage_source import register_alpha_vantage_source

# Configuration and validation
from .transcript_config import get_transcript_config, validate_transcript_environment

# Primary agent interfaces (RECOMMENDED FOR AGENTS)
from .agent_interface import (
    index_transcripts_for_agent,
    search_transcripts_for_agent, 
    get_available_quarters_for_agent,
    evaluate_transcript_search_results_with_llm,
)

__version__ = "2.0.0"  # Major version bump for agent-ready improvements
__author__ = "AI Assistant"

# Agent-Ready API (Primary Interface)
__all__ = [
    # === AGENT INTERFACES (RECOMMENDED) ===
    'index_transcripts_for_agent',          # Index transcripts with vector DB
    'search_transcripts_for_agent',         # Search with quarter selection & proper scoring
    'get_available_quarters_for_agent',     # Get available quarters for intelligent selection
    'evaluate_transcript_search_results_with_llm',  # LLM-powered result evaluation
    
    # === CORE PROCESSING ===
    'get_transcript_chunks',                # Direct chunk processing
    
    # === DATA SOURCES ===
    'TranscriptData',
    'TranscriptQuery', 
    'transcript_registry',
    'register_alpha_vantage_source',
    
    # === CONFIGURATION ===
    'get_transcript_config',
    'validate_transcript_environment',
]