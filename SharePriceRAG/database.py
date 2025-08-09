"""
PostgreSQL database operations for SharePriceRAG package
"""

import logging
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from typing import List, Optional, Dict, Any, Tuple
from datetime import date, datetime, timedelta
from contextlib import contextmanager
from decimal import Decimal

from .models import PriceData, PriceQuery, DateRange, MissingDataRange
from .config import SharePriceConfig


class SharePriceDatabase:
    """Manages PostgreSQL database operations for share price data"""
    
    def __init__(self, config: SharePriceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        from .logging_setup import setup_logging
        setup_logging(self.config)
        self.logger = logging.getLogger('sharepricerag.database')
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password,
                cursor_factory=RealDictCursor
            )
            yield conn
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
                
    def initialize_database(self) -> bool:
        """Create tables if they don't exist"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create main price data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS share_prices (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(20) NOT NULL,
                        date DATE NOT NULL,
                        open_price DECIMAL(12,4),
                        high_price DECIMAL(12,4),
                        low_price DECIMAL(12,4),
                        close_price DECIMAL(12,4) NOT NULL,
                        adjusted_close DECIMAL(12,4),
                        volume BIGINT,
                        source VARCHAR(50) NOT NULL DEFAULT 'unknown',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(ticker, date)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_share_prices_ticker 
                    ON share_prices(ticker)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_share_prices_date 
                    ON share_prices(date)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_share_prices_ticker_date 
                    ON share_prices(ticker, date)
                """)
                
                # Create trigger for updated_at
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql'
                """)
                
                cursor.execute("""
                    DROP TRIGGER IF EXISTS update_share_prices_updated_at ON share_prices
                """)
                
                cursor.execute("""
                    CREATE TRIGGER update_share_prices_updated_at 
                    BEFORE UPDATE ON share_prices 
                    FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column()
                """)
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            return False
    
    def get_existing_data(self, query: PriceQuery) -> List[PriceData]:
        """Get existing price data for the query"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT ticker, date, open_price, high_price, low_price, 
                           close_price, adjusted_close, volume, source, 
                           created_at, updated_at
                    FROM share_prices 
                    WHERE ticker = ANY(%s) 
                    AND date BETWEEN %s AND %s
                    ORDER BY ticker, date
                """, (query.tickers, query.start_date, query.end_date))
                
                results = cursor.fetchall()
                
                return [
                    PriceData(
                        ticker=row['ticker'],
                        date=row['date'],
                        open_price=row['open_price'],
                        high_price=row['high_price'],
                        low_price=row['low_price'],
                        close_price=row['close_price'],
                        adjusted_close=row['adjusted_close'],
                        volume=row['volume'],
                        source=row['source'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    for row in results
                ]
                
        except Exception as e:
            self.logger.error(f"Error getting existing data: {e}")
            return []
    
    def find_missing_data_ranges(self, query: PriceQuery) -> List[MissingDataRange]:
        """Find missing data ranges for each ticker"""
        missing_ranges = []
        
        try:
            existing_data = self.get_existing_data(query)
            
            # Group by ticker
            ticker_data = {}
            for data in existing_data:
                if data.ticker not in ticker_data:
                    ticker_data[data.ticker] = []
                ticker_data[data.ticker].append(data.date)
            
            # Find missing ranges for each ticker
            for ticker in query.tickers:
                existing_dates = set(ticker_data.get(ticker, []))
                
                # Generate all expected dates (excluding weekends if needed)
                expected_dates = self._generate_business_days(
                    query.start_date, 
                    query.end_date,
                    include_weekends=query.include_weekends
                )
                
                missing_dates = sorted(set(expected_dates) - existing_dates)
                
                if missing_dates:
                    # Group consecutive missing dates into ranges
                    ranges = self._group_consecutive_dates(missing_dates)
                    
                    for start_date, end_date in ranges:
                        missing_ranges.append(
                            MissingDataRange(
                                ticker=ticker,
                                start_date=start_date,
                                end_date=end_date
                            )
                        )
                        
        except Exception as e:
            self.logger.error(f"Error finding missing data ranges: {e}")
            
        return missing_ranges
    
    def _generate_business_days(self, start_date: date, end_date: date, 
                               include_weekends: bool = False) -> List[date]:
        """Generate list of business days between dates"""
        dates = []
        current = start_date
        
        while current <= end_date:
            if include_weekends or current.weekday() < 5:  # Monday = 0, Sunday = 6
                dates.append(current)
            current += timedelta(days=1)
            
        return dates
    
    def _group_consecutive_dates(self, dates: List[date]) -> List[Tuple[date, date]]:
        """Group consecutive dates into ranges"""
        if not dates:
            return []
        
        ranges = []
        start_date = dates[0]
        end_date = dates[0]
        
        for i in range(1, len(dates)):
            if dates[i] == end_date + timedelta(days=1):
                end_date = dates[i]
            else:
                ranges.append((start_date, end_date))
                start_date = dates[i]
                end_date = dates[i]
        
        ranges.append((start_date, end_date))
        return ranges
    
    def store_price_data(self, price_data: List[PriceData]) -> int:
        """Store price data in database with upsert logic"""
        if not price_data:
            return 0
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare data for batch insert/update
                values = [
                    (
                        data.ticker,
                        data.date,
                        data.open_price,
                        data.high_price,
                        data.low_price,
                        data.close_price,
                        data.adjusted_close,
                        data.volume,
                        data.source
                    )
                    for data in price_data
                ]
                
                # Use ON CONFLICT for upsert
                execute_values(
                    cursor,
                    """
                    INSERT INTO share_prices 
                    (ticker, date, open_price, high_price, low_price, 
                     close_price, adjusted_close, volume, source)
                    VALUES %s
                    ON CONFLICT (ticker, date)
                    DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        adjusted_close = EXCLUDED.adjusted_close,
                        volume = EXCLUDED.volume,
                        source = EXCLUDED.source,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    values
                )
                
                rows_affected = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Stored {rows_affected} price records")
                return rows_affected
                
        except Exception as e:
            self.logger.error(f"Error storing price data: {e}")
            return 0
    
    def get_ticker_date_range(self, ticker: str) -> Optional[Tuple[date, date]]:
        """Get the date range of available data for a ticker"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT MIN(date) as min_date, MAX(date) as max_date
                    FROM share_prices 
                    WHERE ticker = %s
                """, (ticker,))
                
                result = cursor.fetchone()
                
                if result and result['min_date'] and result['max_date']:
                    return (result['min_date'], result['max_date'])
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting ticker date range: {e}")
            return None
    
    def get_data_statistics(self, ticker: str = None) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Base query
                where_clause = "WHERE ticker = %s" if ticker else ""
                params = [ticker] if ticker else []
                
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT ticker) as unique_tickers,
                        MIN(date) as earliest_date,
                        MAX(date) as latest_date,
                        COUNT(DISTINCT source) as sources_used
                    FROM share_prices 
                    {where_clause}
                """, params)
                
                stats = dict(cursor.fetchone())
                
                # Get source breakdown
                cursor.execute(f"""
                    SELECT source, COUNT(*) as count
                    FROM share_prices 
                    {where_clause}
                    GROUP BY source
                    ORDER BY count DESC
                """, params)
                
                stats['source_breakdown'] = dict(cursor.fetchall())
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> int:
        """Clean up old price data"""
        try:
            cutoff_date = date.today() - timedelta(days=days_to_keep)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM share_prices 
                    WHERE date < %s
                """, (cutoff_date,))
                
                rows_deleted = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {rows_deleted} old records")
                return rows_deleted
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return 0