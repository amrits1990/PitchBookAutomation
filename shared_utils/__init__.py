"""
Shared utilities for PitchBookGenerator RAG systems.

This package provides common utilities used across multiple RAG modules:
- Robust embedding model loading with error recovery
- Cache management utilities
- Diagnostic tools
"""

from .robust_embedding_loader import (
    RobustEmbeddingLoader,
    create_embedding_loader,
    diagnose_model_cache
)

__all__ = [
    'RobustEmbeddingLoader',
    'create_embedding_loader',
    'diagnose_model_cache'
]
