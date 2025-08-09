"""
SharePriceRAG Package - Share Price Retrieval and Caching

A production-ready package for intelligent share price data management with PostgreSQL caching
and automatic fetching from reliable free APIs.

Key Features:
- Smart gap detection in price data
- Automatic fetching from free APIs (Alpha Vantage, Yahoo Finance)
- PostgreSQL database caching
- Bulk ticker support
- Date range filtering
- Production-ready error handling
"""

from .price_client import SharePriceClient
from .database import SharePriceDatabase
from .models import PriceData, PriceQuery, PriceResult
from .config import SharePriceConfig
from .main import get_share_prices, get_ticker_summary, health_check, cleanup_database

__version__ = "1.0.0"
__author__ = "PitchBook Generator"

# Main interface function
__all__ = [
    'get_share_prices',
    'get_ticker_summary',
    'health_check',
    'cleanup_database',
    'SharePriceClient',
    'SharePriceDatabase',
    'SharePriceConfig',
    'PriceData',
    'PriceQuery',
    'PriceResult'
]