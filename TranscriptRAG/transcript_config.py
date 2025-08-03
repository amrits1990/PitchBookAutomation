"""
Configuration management for transcript processing system
Handles settings, validation, and environment configuration
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


@dataclass
class TranscriptProcessingConfig:
    """Centralized configuration for transcript processing"""
    
    # API Configuration
    alpha_vantage_api_key: str = field(default_factory=lambda: os.getenv('ALPHA_VANTAGE_API_KEY', ''))
    request_timeout: int = field(default_factory=lambda: int(os.getenv('REQUEST_TIMEOUT', '30')))
    
    # Processing Limits
    max_workers: int = field(default_factory=lambda: int(os.getenv('MAX_WORKERS', '4')))
    max_years_back: int = field(default_factory=lambda: int(os.getenv('MAX_YEARS_BACK', '5')))
    max_tickers_per_request: int = field(default_factory=lambda: int(os.getenv('MAX_TICKERS_PER_REQUEST', '10')))
    max_transcripts_per_ticker: int = field(default_factory=lambda: int(os.getenv('MAX_TRANSCRIPTS_PER_TICKER', '50')))
    
    # Chunking Configuration
    default_chunk_size: int = field(default_factory=lambda: int(os.getenv('DEFAULT_CHUNK_SIZE', '800')))
    default_overlap: int = field(default_factory=lambda: int(os.getenv('DEFAULT_OVERLAP', '150')))
    min_chunk_size: int = field(default_factory=lambda: int(os.getenv('MIN_CHUNK_SIZE', '100')))
    max_chunk_size: int = field(default_factory=lambda: int(os.getenv('MAX_CHUNK_SIZE', '2000')))
    min_overlap: int = field(default_factory=lambda: int(os.getenv('MIN_OVERLAP', '0')))
    max_overlap: int = field(default_factory=lambda: int(os.getenv('MAX_OVERLAP', '500')))
    
    # Alpha Vantage Specific
    alpha_vantage_rate_limit: int = field(default_factory=lambda: int(os.getenv('ALPHA_VANTAGE_RATE_LIMIT', '5')))
    alpha_vantage_delay: int = field(default_factory=lambda: int(os.getenv('ALPHA_VANTAGE_DELAY', '12')))
    
    # Content Processing
    enable_speaker_chunking: bool = field(default_factory=lambda: os.getenv('ENABLE_SPEAKER_CHUNKING', 'true').lower() == 'true')
    min_content_length: int = field(default_factory=lambda: int(os.getenv('MIN_CONTENT_LENGTH', '50')))
    max_content_length: int = field(default_factory=lambda: int(os.getenv('MAX_CONTENT_LENGTH', '1000000')))
    
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
        if not self.alpha_vantage_api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable must be set")
        
        if len(self.alpha_vantage_api_key) < 10:
            raise ValueError("ALPHA_VANTAGE_API_KEY appears to be invalid (too short)")
    
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
        
        if not (1 <= self.alpha_vantage_rate_limit <= 100):
            raise ValueError("ALPHA_VANTAGE_RATE_LIMIT must be between 1 and 100")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'alpha_vantage_api_key': '***REDACTED***',  # Don't expose API key
            'request_timeout': self.request_timeout,
            'max_workers': self.max_workers,
            'max_years_back': self.max_years_back,
            'max_tickers_per_request': self.max_tickers_per_request,
            'max_transcripts_per_ticker': self.max_transcripts_per_ticker,
            'default_chunk_size': self.default_chunk_size,
            'default_overlap': self.default_overlap,
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size,
            'min_overlap': self.min_overlap,
            'max_overlap': self.max_overlap,
            'alpha_vantage_rate_limit': self.alpha_vantage_rate_limit,
            'alpha_vantage_delay': self.alpha_vantage_delay,
            'enable_speaker_chunking': self.enable_speaker_chunking,
            'min_content_length': self.min_content_length,
            'max_content_length': self.max_content_length,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'environment': self.environment
        }


class TranscriptConfigurationProvider(ABC):
    """Abstract interface for transcript configuration management"""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate configuration is complete and valid"""
        pass


class TranscriptConfigManager(TranscriptConfigurationProvider):
    """Configuration manager for transcript processing system"""
    
    def __init__(self):
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment"""
        try:
            self._config = TranscriptProcessingConfig()
        except Exception as e:
            raise ValueError(f"Transcript configuration error: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary"""
        if self._config is None:
            self._load_config()
        return self._config.to_dict()
    
    def get_typed_config(self) -> TranscriptProcessingConfig:
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
    
    def get_data_source_config(self, source_name: str) -> Dict[str, Any]:
        """Get configuration for specific data source"""
        config = self.get_typed_config()
        
        if source_name.lower() == 'alpha_vantage':
            return {
                'api_key': config.alpha_vantage_api_key,
                'rate_limit': config.alpha_vantage_rate_limit,
                'delay': config.alpha_vantage_delay,
                'timeout': config.request_timeout
            }
        
        return {}
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get chunking-specific configuration"""
        config = self.get_typed_config()
        
        return {
            'default_chunk_size': config.default_chunk_size,
            'default_overlap': config.default_overlap,
            'min_chunk_size': config.min_chunk_size,
            'max_chunk_size': config.max_chunk_size,
            'min_overlap': config.min_overlap,
            'max_overlap': config.max_overlap,
            'enable_speaker_chunking': config.enable_speaker_chunking
        }


# Global configuration manager instance
_transcript_config_manager = None

def get_transcript_config_manager() -> TranscriptConfigManager:
    """Get global transcript configuration manager instance"""
    global _transcript_config_manager
    if _transcript_config_manager is None:
        _transcript_config_manager = TranscriptConfigManager()
    return _transcript_config_manager

def get_transcript_config() -> TranscriptProcessingConfig:
    """Get configuration object (convenience function)"""
    return get_transcript_config_manager().get_typed_config()

def validate_transcript_environment() -> Dict[str, Any]:
    """Validate transcript environment configuration and return status"""
    try:
        config_manager = get_transcript_config_manager()
        config = config_manager.get_typed_config()
        
        return {
            'valid': True,
            'config': config.to_dict(),
            'environment_variables_found': [
                key for key in [
                    'ALPHA_VANTAGE_API_KEY', 'LOG_LEVEL', 'ENVIRONMENT',
                    'MAX_WORKERS', 'DEFAULT_CHUNK_SIZE', 'ALPHA_VANTAGE_RATE_LIMIT'
                ] if os.getenv(key)
            ]
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'required_variables': ['ALPHA_VANTAGE_API_KEY'],
            'optional_variables': [
                'LOG_LEVEL', 'ENVIRONMENT', 'MAX_WORKERS', 
                'DEFAULT_CHUNK_SIZE', 'DEFAULT_OVERLAP',
                'ALPHA_VANTAGE_RATE_LIMIT', 'MAX_YEARS_BACK'
            ]
        }