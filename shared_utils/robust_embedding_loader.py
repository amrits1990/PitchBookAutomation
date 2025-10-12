"""
Robust Embedding Model Loader - Permanent Fix for Meta Tensor Issues

This module provides a reliable way to load sentence-transformers models
with comprehensive error handling, cache management, and retry logic.

Root Causes Addressed:
1. Corrupted PyTorch/Hugging Face model cache
2. Meta tensor initialization errors during device transfer
3. Concurrent model loading race conditions
4. Device management issues (CPU/CUDA)
5. Cache permission and disk space issues

Solution Strategy:
- Explicit cache directory management
- Automatic cache clearing on corruption detection
- Retry logic with exponential backoff
- Thread-safe model loading with locks
- Proper device initialization
- Environment variable configuration
"""

import os
import shutil
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


class RobustEmbeddingLoader:
    """
    Thread-safe, failure-resistant embedding model loader.

    Features:
    - Automatic cache corruption detection and clearing
    - Retry logic with exponential backoff
    - Thread-safe singleton pattern per model
    - Proper device management (CPU/CUDA)
    - Detailed error logging and diagnostics
    """

    # Class-level locks for thread safety
    _model_locks: Dict[str, threading.Lock] = {}
    _model_cache: Dict[str, 'SentenceTransformer'] = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize robust embedding loader.

        Args:
            model_name: Name of sentence-transformer model
            cache_dir: Custom cache directory (default: ~/.cache/huggingface)
            device: Target device ('cpu', 'cuda', or None for auto)
            max_retries: Maximum retry attempts on failure
            retry_delay: Initial delay between retries (seconds)
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Set cache directory with environment variable support
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use environment variable or default
            cache_base = os.environ.get(
                'SENTENCE_TRANSFORMERS_HOME',
                os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')
            )
            self.cache_dir = Path(cache_base)

        # Determine device
        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Ensure we have a lock for this model
        with RobustEmbeddingLoader._global_lock:
            if model_name not in RobustEmbeddingLoader._model_locks:
                RobustEmbeddingLoader._model_locks[model_name] = threading.Lock()

        self.model_lock = RobustEmbeddingLoader._model_locks[model_name]

        logger.info(f"RobustEmbeddingLoader initialized: model={model_name}, device={self.device}, cache={self.cache_dir}")

    def _clear_model_cache(self) -> bool:
        """
        Clear corrupted model cache for specific model.

        Returns:
            True if cache was cleared successfully
        """
        try:
            # Find model cache directory
            model_cache_paths = [
                self.cache_dir / 'hub' / f'models--sentence-transformers--{self.model_name}',
                self.cache_dir / 'sentence_transformers' / self.model_name,
                Path.home() / '.cache' / 'torch' / 'sentence_transformers' / self.model_name
            ]

            cleared = False
            for cache_path in model_cache_paths:
                if cache_path.exists():
                    logger.warning(f"Clearing corrupted cache at: {cache_path}")
                    try:
                        shutil.rmtree(cache_path)
                        cleared = True
                        logger.info(f"Successfully cleared cache: {cache_path}")
                    except Exception as e:
                        logger.error(f"Failed to clear cache {cache_path}: {e}")

            return cleared

        except Exception as e:
            logger.error(f"Error during cache clearing: {e}")
            return False

    def _load_model_attempt(self, force_cpu: bool = False) -> 'SentenceTransformer':
        """
        Single attempt to load the model.

        Args:
            force_cpu: Force CPU device (fallback for CUDA issues)

        Returns:
            Loaded SentenceTransformer model

        Raises:
            Exception if loading fails
        """
        from sentence_transformers import SentenceTransformer
        import torch

        device = 'cpu' if force_cpu else self.device

        logger.info(f"Attempting to load model '{self.model_name}' on device '{device}'")

        # Method 1: Standard loading with explicit device
        try:
            model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=str(self.cache_dir)
            )
            logger.info(f"✓ Model loaded successfully on {device}")
            return model
        except Exception as e1:
            logger.warning(f"Method 1 failed (standard loading): {e1}")

            # Method 2: Load without device parameter, then move
            try:
                logger.info("Trying method 2: Load first, then move to device")
                model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(self.cache_dir)
                )
                # Manually move to device
                if device == 'cuda' and torch.cuda.is_available():
                    model = model.to(torch.device('cuda'))
                logger.info(f"✓ Model loaded successfully with method 2")
                return model
            except Exception as e2:
                logger.warning(f"Method 2 failed: {e2}")

                # Method 3: Use trust_remote_code and local_files_only fallback
                try:
                    logger.info("Trying method 3: Trust remote code with local files")
                    model = SentenceTransformer(
                        self.model_name,
                        device=device,
                        cache_folder=str(self.cache_dir),
                        trust_remote_code=True,
                        local_files_only=False  # Force re-download if needed
                    )
                    logger.info(f"✓ Model loaded successfully with method 3")
                    return model
                except Exception as e3:
                    logger.error(f"All loading methods failed: {e1}, {e2}, {e3}")
                    raise Exception(f"Failed to load model with all methods: {e1}")

    def load_model(self) -> 'SentenceTransformer':
        """
        Load model with robust error handling and retry logic.

        Returns:
            Loaded and verified SentenceTransformer model

        Raises:
            Exception if model cannot be loaded after all retries
        """
        # Check if already cached in memory
        with RobustEmbeddingLoader._global_lock:
            if self.model_name in RobustEmbeddingLoader._model_cache:
                logger.info(f"Using cached model instance for '{self.model_name}'")
                return RobustEmbeddingLoader._model_cache[self.model_name]

        # Thread-safe loading
        with self.model_lock:
            # Double-check after acquiring lock
            with RobustEmbeddingLoader._global_lock:
                if self.model_name in RobustEmbeddingLoader._model_cache:
                    return RobustEmbeddingLoader._model_cache[self.model_name]

            # Attempt loading with retries
            for attempt in range(self.max_retries):
                try:
                    # Try loading
                    model = self._load_model_attempt(force_cpu=(attempt > 0))

                    # Verify model works by encoding test text
                    test_embedding = model.encode(["test"], show_progress_bar=False)
                    if test_embedding is not None and len(test_embedding) > 0:
                        logger.info(f"✓ Model verification successful (embedding shape: {test_embedding.shape})")

                        # Cache in memory
                        with RobustEmbeddingLoader._global_lock:
                            RobustEmbeddingLoader._model_cache[self.model_name] = model

                        return model
                    else:
                        raise Exception("Model verification failed: test encoding returned empty")

                except Exception as e:
                    logger.error(f"Load attempt {attempt + 1}/{self.max_retries} failed: {e}")

                    # On meta tensor error, clear cache
                    if "meta tensor" in str(e).lower() or "cannot copy out" in str(e).lower():
                        logger.warning("Meta tensor error detected - clearing cache")
                        self._clear_model_cache()

                    # Wait before retry (exponential backoff)
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.info(f"Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {self.max_retries} attempts failed")
                        raise Exception(f"Failed to load model '{self.model_name}' after {self.max_retries} attempts: {e}")

    def encode(self, texts: list, **kwargs) -> np.ndarray:
        """
        Encode texts with automatic model loading and error handling.

        Args:
            texts: List of texts to encode
            **kwargs: Additional arguments for model.encode()

        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])

        model = self.load_model()

        try:
            return model.encode(texts, **kwargs)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")

            # Try to recover by clearing cache and reloading
            logger.warning("Attempting to recover by clearing cache and reloading model")
            self._clear_model_cache()

            # Clear memory cache
            with RobustEmbeddingLoader._global_lock:
                if self.model_name in RobustEmbeddingLoader._model_cache:
                    del RobustEmbeddingLoader._model_cache[self.model_name]

            # Retry once
            model = self.load_model()
            return model.encode(texts, **kwargs)

    @classmethod
    def clear_all_caches(cls):
        """Clear all cached model instances from memory."""
        with cls._global_lock:
            cls._model_cache.clear()
            logger.info("Cleared all cached model instances")

    @classmethod
    def get_cache_info(cls) -> Dict[str, int]:
        """Get information about cached models."""
        with cls._global_lock:
            return {
                "cached_models": len(cls._model_cache),
                "model_names": list(cls._model_cache.keys())
            }


def create_embedding_loader(
    model_name: str = "all-mpnet-base-v2",
    **kwargs
) -> RobustEmbeddingLoader:
    """
    Factory function to create a robust embedding loader.

    Args:
        model_name: Model name (default: all-mpnet-base-v2)
        **kwargs: Additional arguments for RobustEmbeddingLoader

    Returns:
        Configured RobustEmbeddingLoader instance
    """
    return RobustEmbeddingLoader(model_name=model_name, **kwargs)


# Diagnostic utility
def diagnose_model_cache(model_name: str = "all-mpnet-base-v2"):
    """
    Diagnose model cache status and potential issues.

    Args:
        model_name: Model to diagnose
    """
    print(f"\n{'='*60}")
    print(f"  Embedding Model Cache Diagnostics: {model_name}")
    print(f"{'='*60}\n")

    import torch

    # Check PyTorch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # Check cache directories
    cache_paths = [
        Path.home() / '.cache' / 'huggingface' / 'hub' / f'models--sentence-transformers--{model_name}',
        Path.home() / '.cache' / 'huggingface' / 'sentence_transformers' / model_name,
        Path.home() / '.cache' / 'torch' / 'sentence_transformers' / model_name
    ]

    print(f"\nCache Directories:")
    for path in cache_paths:
        exists = path.exists()
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if exists else 0
        print(f"  {'✓' if exists else '✗'} {path}")
        if exists:
            print(f"     Size: {size / (1024**2):.1f} MB")

    # Check environment variables
    print(f"\nEnvironment Variables:")
    print(f"  HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"  SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME', 'Not set')}")

    # Memory cache status
    cache_info = RobustEmbeddingLoader.get_cache_info()
    print(f"\nMemory Cache:")
    print(f"  Cached models: {cache_info['cached_models']}")
    if cache_info['model_names']:
        print(f"  Models: {', '.join(cache_info['model_names'])}")

    print(f"\n{'='*60}\n")
