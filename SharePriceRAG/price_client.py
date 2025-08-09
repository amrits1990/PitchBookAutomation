"""
API clients for fetching stock price data from free sources
"""

import logging
import time
import requests
from typing import List, Optional, Dict, Any
from datetime import date, datetime, timedelta
from decimal import Decimal
import yfinance as yf

from .models import PriceData, MissingDataRange
from .config import SharePriceConfig


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded"""
    pass


class SharePriceClient:
    """Client for fetching share price data from multiple free APIs"""
    
    def __init__(self, config: SharePriceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Rate limiting tracking
        self._last_request_time = 0
        self._request_count = 0
        self._daily_request_count = 0
        self._daily_reset_time = time.time()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        from .logging_setup import setup_logging
        setup_logging(self.config)
        self.logger = logging.getLogger('sharepricerag.price_client')
    
    def fetch_missing_data(self, missing_ranges: List[MissingDataRange]) -> List[PriceData]:
        """Fetch price data for missing ranges"""
        all_price_data = []
        
        for missing_range in missing_ranges:
            try:
                price_data = self._fetch_ticker_data(
                    missing_range.ticker,
                    missing_range.start_date,
                    missing_range.end_date
                )
                all_price_data.extend(price_data)
                
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {missing_range.ticker}: {e}")
                continue
        
        return all_price_data
    
    def _fetch_ticker_data(self, ticker: str, start_date: date, end_date: date) -> List[PriceData]:
        """Fetch price data for a single ticker"""
        
        # Try Alpha Vantage first if API key is available
        if self.config.alpha_vantage_api_key and self.config.preferred_data_source == "alpha_vantage":
            try:
                return self._fetch_from_alpha_vantage(ticker, start_date, end_date)
            except Exception as e:
                self.logger.warning(f"Alpha Vantage failed for {ticker}: {e}")
                
        # Fallback to Yahoo Finance
        if self.config.use_yahoo_finance:
            try:
                return self._fetch_from_yahoo_finance(ticker, start_date, end_date)
            except Exception as e:
                self.logger.warning(f"Yahoo Finance failed for {ticker}: {e}")
        
        self.logger.error(f"All data sources failed for {ticker}")
        return []
    
    def _fetch_from_alpha_vantage(self, ticker: str, start_date: date, end_date: date) -> List[PriceData]:
        """Fetch data from Alpha Vantage API"""
        self._check_rate_limits()
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': ticker,
            'outputsize': 'full',
            'apikey': self.config.alpha_vantage_api_key
        }
        
        self.logger.info(f"Fetching {ticker} from Alpha Vantage: {start_date} to {end_date}")
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        self._update_rate_limits()
        
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            raise Exception(f"Alpha Vantage API error: {data['Error Message']}")
        
        if "Note" in data:
            raise RateLimitError(f"Alpha Vantage rate limit: {data['Note']}")
            
        if "Time Series (Daily)" not in data:
            raise Exception(f"Invalid response format from Alpha Vantage")
        
        # Parse the response
        time_series = data["Time Series (Daily)"]
        price_data = []
        
        for date_str, daily_data in time_series.items():
            price_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Filter by date range
            if start_date <= price_date <= end_date:
                price_data.append(
                    PriceData(
                        ticker=ticker,
                        date=price_date,
                        open_price=Decimal(daily_data["1. open"]),
                        high_price=Decimal(daily_data["2. high"]),
                        low_price=Decimal(daily_data["3. low"]),
                        close_price=Decimal(daily_data["4. close"]),
                        adjusted_close=Decimal(daily_data["5. adjusted close"]),
                        volume=int(daily_data["6. volume"]),
                        source="alpha_vantage"
                    )
                )
        
        self.logger.info(f"Retrieved {len(price_data)} records for {ticker} from Alpha Vantage")
        return price_data
    
    def _fetch_from_yahoo_finance(self, ticker: str, start_date: date, end_date: date) -> List[PriceData]:
        """Fetch data from Yahoo Finance using yfinance"""
        self.logger.info(f"Fetching {ticker} from Yahoo Finance: {start_date} to {end_date}")
        
        try:
            # Create ticker object
            ticker_obj = yf.Ticker(ticker)
            
            # Fetch historical data
            hist = ticker_obj.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),  # yfinance end is exclusive
                auto_adjust=False,
                prepost=False,
                actions=False
            )
            
            if hist.empty:
                raise Exception(f"No data returned for {ticker}")
            
            price_data = []
            
            for date_index, row in hist.iterrows():
                price_date = date_index.date()
                
                # Skip if not in our date range (defensive)
                if not (start_date <= price_date <= end_date):
                    continue
                
                price_data.append(
                    PriceData(
                        ticker=ticker,
                        date=price_date,
                        open_price=Decimal(str(row['Open'])) if not pd.isna(row['Open']) else None,
                        high_price=Decimal(str(row['High'])) if not pd.isna(row['High']) else None,
                        low_price=Decimal(str(row['Low'])) if not pd.isna(row['Low']) else None,
                        close_price=Decimal(str(row['Close'])),
                        adjusted_close=Decimal(str(row['Adj Close'])) if not pd.isna(row['Adj Close']) else None,
                        volume=int(row['Volume']) if not pd.isna(row['Volume']) else None,
                        source="yahoo_finance"
                    )
                )
            
            self.logger.info(f"Retrieved {len(price_data)} records for {ticker} from Yahoo Finance")
            return price_data
            
        except Exception as e:
            raise Exception(f"Yahoo Finance error for {ticker}: {e}")
    
    def _check_rate_limits(self):
        """Check if we can make a request without exceeding rate limits"""
        current_time = time.time()
        
        # Reset daily counter if needed
        if current_time - self._daily_reset_time >= 86400:  # 24 hours
            self._daily_request_count = 0
            self._daily_reset_time = current_time
        
        # Check daily limit
        if self._daily_request_count >= self.config.requests_per_day:
            raise RateLimitError("Daily API request limit exceeded")
        
        # Check per-minute limit
        if self._last_request_time > 0:
            time_since_last = current_time - self._last_request_time
            if time_since_last < 60 / self.config.requests_per_minute:
                sleep_time = (60 / self.config.requests_per_minute) - time_since_last
                self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
    
    def _update_rate_limits(self):
        """Update rate limiting counters after making a request"""
        self._last_request_time = time.time()
        self._request_count += 1
        self._daily_request_count += 1
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if a ticker exists and is tradeable"""
        try:
            # Use Yahoo Finance for quick validation
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # Check if we got valid info
            if not info or 'symbol' not in info:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Ticker validation failed for {ticker}: {e}")
            return False
    
    def get_latest_price(self, ticker: str) -> Optional[PriceData]:
        """Get the most recent price for a ticker"""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=7)  # Last week
            
            price_data = self._fetch_ticker_data(ticker, start_date, end_date)
            
            if price_data:
                # Return the most recent data
                return max(price_data, key=lambda x: x.date)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get latest price for {ticker}: {e}")
            return None
    
    def get_supported_exchanges(self) -> List[str]:
        """Get list of supported stock exchanges"""
        return list(self.config.supported_exchanges.keys()) if hasattr(self.config, 'supported_exchanges') else ["NYSE", "NASDAQ"]
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of all data sources"""
        health = {
            'alpha_vantage': False,
            'yahoo_finance': False,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check Alpha Vantage
        if self.config.alpha_vantage_api_key:
            try:
                # Simple API call to check connectivity
                url = "https://www.alphavantage.co/query"
                params = {
                    'function': 'TIME_SERIES_INTRADAY',
                    'symbol': 'MSFT',
                    'interval': '1min',
                    'outputsize': 'compact',
                    'apikey': self.config.alpha_vantage_api_key
                }
                response = requests.get(url, params=params, timeout=10)
                health['alpha_vantage'] = response.status_code == 200
            except Exception:
                pass
        
        # Check Yahoo Finance
        try:
            ticker = yf.Ticker("MSFT")
            info = ticker.info
            health['yahoo_finance'] = bool(info and 'symbol' in info)
        except Exception:
            pass
        
        return health


# Import pandas for Yahoo Finance data handling
try:
    import pandas as pd
except ImportError:
    # Fallback for pandas operations
    class pd:
        @staticmethod
        def isna(value):
            return value is None or (isinstance(value, float) and value != value)  # NaN check