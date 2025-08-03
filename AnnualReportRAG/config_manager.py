"""
Configuration management module for SEC filing processing
Centralized, secure, and environment-aware configuration
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


@dataclass
class ProcessingConfig:
    """Centralized configuration for SEC filing processing"""
    
    # API Configuration
    sec_user_agent: str = field(default_factory=lambda: os.getenv('SEC_USER_AGENT', ''))
    request_timeout: int = field(default_factory=lambda: int(os.getenv('REQUEST_TIMEOUT', '300')))
    
    # Processing Limits
    max_workers: int = field(default_factory=lambda: int(os.getenv('MAX_WORKERS', '4')))
    max_years_back: int = field(default_factory=lambda: int(os.getenv('MAX_YEARS_BACK', '5')))
    max_tickers_per_request: int = field(default_factory=lambda: int(os.getenv('MAX_TICKERS_PER_REQUEST', '10')))
    max_files_per_ticker: int = field(default_factory=lambda: int(os.getenv('MAX_FILES_PER_TICKER', '150')))
    max_file_size_mb: int = field(default_factory=lambda: int(os.getenv('MAX_FILE_SIZE_MB', '100')))
    
    # Chunking Configuration
    default_chunk_size: int = field(default_factory=lambda: int(os.getenv('DEFAULT_CHUNK_SIZE', '800')))
    default_overlap: int = field(default_factory=lambda: int(os.getenv('DEFAULT_OVERLAP', '150')))
    min_chunk_size: int = field(default_factory=lambda: int(os.getenv('MIN_CHUNK_SIZE', '100')))
    max_chunk_size: int = field(default_factory=lambda: int(os.getenv('MAX_CHUNK_SIZE', '5000')))
    min_overlap: int = field(default_factory=lambda: int(os.getenv('MIN_OVERLAP', '0')))
    max_overlap: int = field(default_factory=lambda: int(os.getenv('MAX_OVERLAP', '1000')))
    
    # Rate Limiting
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv('RATE_LIMIT_REQUESTS', '10')))
    rate_limit_window: int = field(default_factory=lambda: int(os.getenv('RATE_LIMIT_WINDOW', '60')))
    
    # Logging Configuration
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv('LOG_FILE'))
    environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'production'))
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_required_fields()
        self._validate_limits()
    
    def _validate_required_fields(self):
        """Validate required configuration fields"""
        if not self.sec_user_agent:
            raise ValueError("SEC_USER_AGENT environment variable must be set to a valid email address")
        
        # Basic email validation
        if '@' not in self.sec_user_agent or '.' not in self.sec_user_agent:
            raise ValueError("SEC_USER_AGENT must be a valid email address")
    
    def _validate_limits(self):
        """Validate configuration limits"""
        if not (1 <= self.max_years_back <= 10):
            raise ValueError("MAX_YEARS_BACK must be between 1 and 10")
        
        if not (1 <= self.max_workers <= 20):
            raise ValueError("MAX_WORKERS must be between 1 and 20")
        
        if not (100 <= self.min_chunk_size <= self.max_chunk_size <= 10000):
            raise ValueError("Invalid chunk size limits")
        
        if not (0 <= self.min_overlap <= self.max_overlap <= 2000):
            raise ValueError("Invalid overlap limits")
        
        if self.default_overlap >= self.default_chunk_size:
            raise ValueError("DEFAULT_OVERLAP must be less than DEFAULT_CHUNK_SIZE")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'sec_user_agent': self.sec_user_agent,
            'request_timeout': self.request_timeout,
            'max_workers': self.max_workers,
            'max_years_back': self.max_years_back,
            'max_tickers_per_request': self.max_tickers_per_request,
            'max_files_per_ticker': self.max_files_per_ticker,
            'max_file_size_mb': self.max_file_size_mb,
            'default_chunk_size': self.default_chunk_size,
            'default_overlap': self.default_overlap,
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size,
            'min_overlap': self.min_overlap,
            'max_overlap': self.max_overlap,
            'rate_limit_requests': self.rate_limit_requests,
            'rate_limit_window': self.rate_limit_window,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'environment': self.environment
        }


class ConfigurationProvider(ABC):
    """Abstract interface for configuration management"""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate configuration is complete and valid"""
        pass


class ConfigManager(ConfigurationProvider):
    """Configuration manager implementing the ConfigurationProvider interface"""
    
    def __init__(self):
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment"""
        try:
            self._config = ProcessingConfig()
        except Exception as e:
            raise ValueError(f"Configuration error: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary"""
        if self._config is None:
            self._load_config()
        return self._config.to_dict()
    
    def get_typed_config(self) -> ProcessingConfig:
        """Get typed configuration object"""
        if self._config is None:
            self._load_config()
        return self._config
    
    def validate_config(self) -> bool:
        """Validate configuration is complete and valid"""
        try:
            if self._config is None:
                self._load_config()
            return True
        except Exception:
            return False
    
    def reload_config(self):
        """Reload configuration from environment"""
        self._load_config()
    
    def validate_processing_params(self, chunk_size: int, overlap: int, years_back: int) -> bool:
        """Validate processing parameters against configuration limits"""
        config = self.get_typed_config()
        
        if not (config.min_chunk_size <= chunk_size <= config.max_chunk_size):
            return False
        
        if not (config.min_overlap <= overlap <= config.max_overlap):
            return False
        
        if overlap >= chunk_size:
            return False
        
        if not (1 <= years_back <= config.max_years_back):
            return False
        
        return True


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> ProcessingConfig:
    """Get configuration object (convenience function)"""
    return get_config_manager().get_typed_config()

def validate_environment() -> Dict[str, Any]:
    """Validate environment configuration and return status"""
    try:
        config_manager = get_config_manager()
        config = config_manager.get_typed_config()
        
        return {
            'valid': True,
            'config': config.to_dict(),
            'environment_variables_found': [
                key for key in [
                    'SEC_USER_AGENT', 'LOG_LEVEL', 'ENVIRONMENT',
                    'MAX_WORKERS', 'DEFAULT_CHUNK_SIZE', 'RATE_LIMIT_REQUESTS'
                ] if os.getenv(key)
            ]
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'required_variables': ['SEC_USER_AGENT'],
            'optional_variables': [
                'LOG_LEVEL', 'ENVIRONMENT', 'MAX_WORKERS', 
                'DEFAULT_CHUNK_SIZE', 'DEFAULT_OVERLAP',
                'RATE_LIMIT_REQUESTS', 'MAX_YEARS_BACK'
            ]
        }