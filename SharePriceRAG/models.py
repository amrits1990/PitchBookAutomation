"""
Data models for SharePriceRAG package
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from decimal import Decimal


@dataclass
class PriceData:
    """Represents daily stock price data"""
    ticker: str
    date: date
    open_price: Optional[Decimal] = None
    high_price: Optional[Decimal] = None
    low_price: Optional[Decimal] = None
    close_price: Decimal = None
    adjusted_close: Optional[Decimal] = None
    volume: Optional[int] = None
    
    # Data source tracking
    source: str = "unknown"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        # Ensure ticker is uppercase
        self.ticker = self.ticker.upper()
        
        # If adjusted_close not provided, use close_price
        if self.adjusted_close is None:
            self.adjusted_close = self.close_price


@dataclass  
class DateRange:
    """Represents a date range"""
    start_date: date
    end_date: date
    
    def __post_init__(self):
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")
    
    def contains(self, check_date: date) -> bool:
        """Check if date falls within range"""
        return self.start_date <= check_date <= self.end_date
    
    def days_count(self) -> int:
        """Number of days in range"""
        return (self.end_date - self.start_date).days + 1


@dataclass
class PriceQuery:
    """Query parameters for price retrieval"""
    tickers: List[str]
    start_date: date
    end_date: date
    include_weekends: bool = False
    
    def __post_init__(self):
        # Normalize tickers to uppercase
        self.tickers = [ticker.strip().upper() for ticker in self.tickers]
        
        # Validate date range
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")
    
    @property
    def date_range(self) -> DateRange:
        """Get date range object"""
        return DateRange(self.start_date, self.end_date)


@dataclass
class MissingDataRange:
    """Represents a range of missing data for a ticker"""
    ticker: str
    start_date: date
    end_date: date
    days_missing: int = 0
    
    def __post_init__(self):
        self.days_missing = (self.end_date - self.start_date).days + 1


@dataclass
class PriceResult:
    """Result of price data retrieval operation"""
    success: bool
    query: PriceQuery
    
    # Results data
    price_data: List[PriceData] = field(default_factory=list)
    
    # Statistics
    total_records: int = 0
    records_from_cache: int = 0
    records_fetched: int = 0
    
    # Missing data tracking
    missing_ranges: List[MissingDataRange] = field(default_factory=list)
    
    # Performance tracking  
    processing_time_seconds: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add an error to the result"""
        self.errors.append(error)
        if self.success and self.errors:
            self.success = False
    
    def add_warning(self, warning: str):
        """Add a warning to the result"""
        self.warnings.append(warning)
    
    def add_price_data(self, data: PriceData, from_cache: bool = False):
        """Add price data to result"""
        self.price_data.append(data)
        self.total_records += 1
        
        if from_cache:
            self.records_from_cache += 1
        else:
            self.records_fetched += 1
    
    def calculate_cache_hit_rate(self):
        """Calculate cache hit rate"""
        if self.total_records > 0:
            self.cache_hit_rate = self.records_from_cache / self.total_records
        else:
            self.cache_hit_rate = 0.0
    
    def get_ticker_data(self, ticker: str) -> List[PriceData]:
        """Get price data for specific ticker"""
        return [data for data in self.price_data if data.ticker.upper() == ticker.upper()]
    
    def get_date_range_data(self, start_date: date, end_date: date) -> List[PriceData]:
        """Get price data within date range"""
        return [
            data for data in self.price_data 
            if start_date <= data.date <= end_date
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'success': self.success,
            'query': {
                'tickers': self.query.tickers,
                'start_date': self.query.start_date.isoformat(),
                'end_date': self.query.end_date.isoformat(),
                'include_weekends': self.query.include_weekends
            },
            'statistics': {
                'total_records': self.total_records,
                'records_from_cache': self.records_from_cache,
                'records_fetched': self.records_fetched,
                'cache_hit_rate': round(self.cache_hit_rate, 4),
                'processing_time_seconds': round(self.processing_time_seconds, 4)
            },
            'price_data': [
                {
                    'ticker': data.ticker,
                    'date': data.date.isoformat(),
                    'open': float(data.open_price) if data.open_price else None,
                    'high': float(data.high_price) if data.high_price else None,
                    'low': float(data.low_price) if data.low_price else None,
                    'close': float(data.close_price) if data.close_price else None,
                    'adjusted_close': float(data.adjusted_close) if data.adjusted_close else None,
                    'volume': data.volume,
                    'source': data.source
                }
                for data in self.price_data
            ],
            'missing_ranges': [
                {
                    'ticker': range_data.ticker,
                    'start_date': range_data.start_date.isoformat(),
                    'end_date': range_data.end_date.isoformat(),
                    'days_missing': range_data.days_missing
                }
                for range_data in self.missing_ranges
            ],
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': datetime.utcnow().isoformat()
        }