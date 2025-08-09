"""
Centralized logging setup for SharePriceRAG package
"""

import os
import logging
import logging.handlers
from typing import Optional


def setup_logging(config) -> logging.Logger:
    """
    Setup logging with both console and file output
    
    Args:
        config: SharePriceConfig object with logging settings
        
    Returns:
        Configured logger instance
    """
    
    # Get log level
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger('sharepricerag')
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if config.enable_file_logging:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(config.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                config.log_file,
                maxBytes=config.log_max_bytes,
                backupCount=config.log_backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"File logging enabled: {config.log_file}")
            
        except Exception as e:
            # If file logging fails, just log to console
            logger.warning(f"Failed to setup file logging: {e}")
    
    return logger


def get_logger(name: str = 'sharepricerag') -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (defaults to 'sharepricerag')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)