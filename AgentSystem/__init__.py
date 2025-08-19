"""
AI Financial Analysis Agent System

A multi-agent system for comprehensive financial analysis using LangChain + LangGraph.
Provides specialist domain agents with vector database-backed RAG capabilities.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .vector_store import VectorStore
from .config import AgentConfig

__all__ = [
    "VectorStore",
    "AgentConfig",
]