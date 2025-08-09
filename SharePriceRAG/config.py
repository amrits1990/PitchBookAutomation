"""
Configuration settings for SharePriceRAG package
"""

import os
from dataclasses import dataclass
from typing import List

# Load .env file if available
try:
    from dotenv import load_dotenv
    # Try to load .env from package directory
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # Fallback to default .env loading
        load_dotenv()
except ImportError:
    # python-dotenv not installed, skip
    pass


@dataclass
class SharePriceConfig:
    """Configuration class for SharePriceRAG package"""
    
    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "shareprices"
    db_user: str = "postgres"
    db_password: str = ""
    
    # API Configuration (Alpha Vantage as primary)
    alpha_vantage_api_key: str = ""
    
    # Yahoo Finance (backup - no API key needed)
    use_yahoo_finance: bool = True
    
    # Rate limiting
    requests_per_minute: int = 5  # Alpha Vantage free tier limit
    requests_per_day: int = 500   # Alpha Vantage free tier limit
    
    # Data preferences
    preferred_data_source: str = "alpha_vantage"  # alpha_vantage, yahoo_finance
    
    # Caching settings
    cache_expiry_days: int = 1    # How long to cache data before refresh
    max_batch_size: int = 100     # Maximum tickers per batch
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/sharepricerag.log"
    enable_file_logging: bool = True
    log_max_bytes: int = 10485760  # 10MB
    log_backup_count: int = 5
    
    # Data validation
    validate_prices: bool = True
    max_price_change_percent: float = 50.0  # Flag suspicious price changes
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        
        # Load database config from environment if not provided
        if not self.db_password:
            self.db_password = os.getenv("POSTGRES_PASSWORD", "")
            
        if not self.db_user:
            self.db_user = os.getenv("POSTGRES_USER", "postgres")
            
        if not self.db_host:
            self.db_host = os.getenv("POSTGRES_HOST", "localhost")
            
        if not self.db_name:
            self.db_name = os.getenv("POSTGRES_DB", "shareprices")
        
        # Load API key from environment if not provided
        if not self.alpha_vantage_api_key:
            self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        
        # Validate rate limiting
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
            
        if self.requests_per_day <= 0:
            raise ValueError("requests_per_day must be positive")
    
    @classmethod
    def from_env(cls) -> 'SharePriceConfig':
        """Create configuration from environment variables"""
        return cls(
            db_host=os.getenv("POSTGRES_HOST", "localhost"),
            db_port=int(os.getenv("POSTGRES_PORT", "5432")),
            db_name=os.getenv("POSTGRES_DB", "shareprices"),
            db_user=os.getenv("POSTGRES_USER", "postgres"),
            db_password=os.getenv("POSTGRES_PASSWORD", ""),
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
            use_yahoo_finance=os.getenv("USE_YAHOO_FINANCE", "true").lower() == "true",
            requests_per_minute=int(os.getenv("API_REQUESTS_PER_MINUTE", "5")),
            requests_per_day=int(os.getenv("API_REQUESTS_PER_DAY", "500")),
            preferred_data_source=os.getenv("PREFERRED_DATA_SOURCE", "alpha_vantage"),
            cache_expiry_days=int(os.getenv("CACHE_EXPIRY_DAYS", "1")),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "100")),
            log_level=os.getenv("SHARE_PRICE_LOG_LEVEL", "INFO"),
            log_file=os.getenv("SHARE_PRICE_LOG_FILE", "logs/sharepricerag.log"),
            enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true",
            log_max_bytes=int(os.getenv("LOG_MAX_BYTES", "10485760")),
            log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            validate_prices=os.getenv("VALIDATE_PRICES", "true").lower() == "true",
            max_price_change_percent=float(os.getenv("MAX_PRICE_CHANGE_PERCENT", "50.0"))
        )
    
    @classmethod
    def default(cls) -> 'SharePriceConfig':
        """Create default configuration"""
        return cls.from_env()
    
    @property
    def database_url(self) -> str:
        """Get database connection URL"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    def has_api_access(self) -> bool:
        """Check if we have access to at least one data source"""
        return bool(self.alpha_vantage_api_key) or self.use_yahoo_finance


# Supported stock exchanges and their trading days
SUPPORTED_EXCHANGES = {
    "NYSE": {
        "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "holidays": [
            # Major US holidays (can be extended)
            "2024-01-01",  # New Year's Day
            "2024-01-15",  # Martin Luther King Jr. Day
            "2024-02-19",  # Presidents' Day
            "2024-03-29",  # Good Friday
            "2024-05-27",  # Memorial Day
            "2024-06-19",  # Juneteenth
            "2024-07-04",  # Independence Day
            "2024-09-02",  # Labor Day
            "2024-11-28",  # Thanksgiving
            "2024-12-25",  # Christmas Day
        ]
    },
    "NASDAQ": {
        "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "holidays": [
            # Same as NYSE
            "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29",
            "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
            "2024-11-28", "2024-12-25"
        ]
    }
}

# Free API endpoints and their characteristics
API_SOURCES = {
    "alpha_vantage": {
        "name": "Alpha Vantage",
        "requires_key": True,
        "free_tier_limit": 500,  # requests per day
        "rate_limit": 5,  # requests per minute
        "reliability": "high",
        "data_quality": "high"
    },
    "yahoo_finance": {
        "name": "Yahoo Finance",
        "requires_key": False,
        "free_tier_limit": 2000,  # requests per hour
        "rate_limit": 200,  # requests per minute
        "reliability": "medium",
        "data_quality": "medium"
    }
}