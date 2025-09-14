"""
Data source abstraction interface for transcript sourcing
Provides pluggable architecture for different transcript data providers
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TranscriptData:
    """Standardized transcript data structure"""
    ticker: str
    company_name: str
    transcript_date: datetime
    quarter: str
    fiscal_year: str
    transcript_type: str  # "earnings_call", "conference_call", etc.
    content: str
    participants: List[str]
    raw_data: Dict[str, Any]
    source: str
    metadata: Dict[str, Any]


@dataclass
class TranscriptQuery:
    """Query parameters for transcript retrieval"""
    ticker: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    quarters_back: Optional[int] = None
    limit: Optional[int] = None
    transcript_type: Optional[str] = None


class TranscriptDataSource(ABC):
    """Abstract interface for transcript data sources"""
    
    @abstractmethod
    def get_transcripts(self, query: TranscriptQuery) -> List[TranscriptData]:
        """
        Retrieve transcripts based on query parameters
        
        Args:
            query: TranscriptQuery with search parameters
            
        Returns:
            List of TranscriptData objects
            
        Raises:
            DataSourceError: If retrieval fails
        """
        pass
    
    @abstractmethod
    def get_latest_transcript(self, ticker: str) -> Optional[TranscriptData]:
        """
        Get the most recent transcript for a ticker
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            TranscriptData object or None if not found
        """
        pass
    
    @abstractmethod
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if ticker is supported by this data source
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            True if ticker is valid and supported
        """
        pass
    
    @abstractmethod
    def get_supported_date_range(self) -> tuple[datetime, datetime]:
        """
        Get the date range supported by this data source
        
        Returns:
            Tuple of (earliest_date, latest_date)
        """
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get rate limiting information for this data source
        
        Returns:
            Dictionary with rate limit details
        """
        pass


class DataSourceError(Exception):
    """Custom exception for data source errors"""
    
    def __init__(self, message: str, error_code: str = None, source: str = None):
        super().__init__(message)
        self.error_code = error_code
        self.source = source
        self.timestamp = datetime.now()


class DataSourceRegistry:
    """Registry for managing multiple data sources"""
    
    def __init__(self):
        self._sources: Dict[str, TranscriptDataSource] = {}
        self._default_source: Optional[str] = None
    
    def register_source(self, name: str, source: TranscriptDataSource, 
                       is_default: bool = False):
        """Register a new data source"""
        self._sources[name] = source
        if is_default or not self._default_source:
            self._default_source = name
    
    def get_source(self, name: str = None) -> TranscriptDataSource:
        """Get a data source by name, or default if no name provided"""
        if name is None:
            name = self._default_source
        
        if name not in self._sources:
            raise DataSourceError(f"Unknown data source: {name}")
        
        return self._sources[name]
    
    def list_sources(self) -> List[str]:
        """List all registered data sources"""
        return list(self._sources.keys())
    
    def get_default_source(self) -> Optional[str]:
        """Get the default data source name"""
        return self._default_source


# Global registry instance
transcript_registry = DataSourceRegistry()