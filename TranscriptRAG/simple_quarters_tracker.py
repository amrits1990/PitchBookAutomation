"""
Simple Quarters Tracker for TranscriptRAG

Tracks which quarters have been ingested for each ticker to avoid redundant API calls.
Provides intelligent caching and quarter management functionality.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from pathlib import Path


class SimpleQuartersTracker:
    """
    Simple file-based tracker for managing quarter ingestion status.
    
    Tracks which quarters have been successfully ingested for each ticker
    to enable intelligent caching and avoid redundant API calls.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize the quarters tracker.
        
        Args:
            cache_file: Path to cache file. If None, uses default location.
        """
        if cache_file is None:
            # Create cache file in TranscriptRAG data directory
            base_dir = Path(__file__).parent
            data_dir = base_dir / "data"
            data_dir.mkdir(exist_ok=True)
            cache_file = str(data_dir / "quarters_cache.json")
        
        self.cache_file = cache_file
        self.cache_data = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache data from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Ensure proper structure
                    if not isinstance(data, dict):
                        return {"tickers": {}, "last_updated": datetime.now().isoformat()}
                    if "tickers" not in data:
                        data["tickers"] = {}
                    return data
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            print(f"Warning: Could not load quarters cache: {e}")
        
        # Return empty cache structure
        return {"tickers": {}, "last_updated": datetime.now().isoformat()}
    
    def _save_cache(self) -> bool:
        """Save cache data to file."""
        try:
            self.cache_data["last_updated"] = datetime.now().isoformat()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, default=str)
            return True
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not save quarters cache: {e}")
            return False
    
    def get_ingested_quarters(self, ticker: str) -> List[str]:
        """
        Get list of quarters already ingested for a ticker.
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            List of quarter strings like ["Q1 2024", "Q4 2023"]
        """
        ticker = ticker.upper()
        ticker_data = self.cache_data.get("tickers", {}).get(ticker, {})
        return ticker_data.get("ingested_quarters", [])
    
    def add_quarter(self, ticker: str, quarter_str: str) -> bool:
        """
        Add a single quarter as ingested for a ticker.
        
        Args:
            ticker: Company ticker symbol  
            quarter_str: Quarter string like "Q1 2024"
            
        Returns:
            True if successfully added, False otherwise
        """
        ticker = ticker.upper()
        
        if "tickers" not in self.cache_data:
            self.cache_data["tickers"] = {}
        
        if ticker not in self.cache_data["tickers"]:
            self.cache_data["tickers"][ticker] = {
                "ingested_quarters": [],
                "last_ingestion": None
            }
        
        ticker_data = self.cache_data["tickers"][ticker]
        
        if quarter_str not in ticker_data["ingested_quarters"]:
            ticker_data["ingested_quarters"].append(quarter_str)
            ticker_data["last_ingestion"] = datetime.now().isoformat()
            
            # Sort quarters chronologically (newest first)
            ticker_data["ingested_quarters"] = self._sort_quarters(ticker_data["ingested_quarters"])
            
            return self._save_cache()
        
        return True  # Already exists
    
    def add_multiple_quarters(self, ticker: str, quarter_strings: List[str]) -> bool:
        """
        Add multiple quarters as ingested for a ticker.
        
        Args:
            ticker: Company ticker symbol
            quarter_strings: List of quarter strings like ["Q1 2024", "Q4 2023"]
            
        Returns:
            True if successfully added, False otherwise
        """
        ticker = ticker.upper()
        
        if "tickers" not in self.cache_data:
            self.cache_data["tickers"] = {}
        
        if ticker not in self.cache_data["tickers"]:
            self.cache_data["tickers"][ticker] = {
                "ingested_quarters": [],
                "last_ingestion": None
            }
        
        ticker_data = self.cache_data["tickers"][ticker]
        
        # Add new quarters that aren't already tracked
        added_any = False
        for quarter_str in quarter_strings:
            if quarter_str not in ticker_data["ingested_quarters"]:
                ticker_data["ingested_quarters"].append(quarter_str)
                added_any = True
        
        if added_any:
            ticker_data["last_ingestion"] = datetime.now().isoformat()
            
            # Sort quarters chronologically (newest first)
            ticker_data["ingested_quarters"] = self._sort_quarters(ticker_data["ingested_quarters"])
            
            return self._save_cache()
        
        return True  # Nothing new to add
    
    def remove_quarter(self, ticker: str, quarter_str: str) -> bool:
        """
        Remove a quarter from ingested list (for cache invalidation).
        
        Args:
            ticker: Company ticker symbol
            quarter_str: Quarter string like "Q1 2024"
            
        Returns:
            True if successfully removed, False otherwise
        """
        ticker = ticker.upper()
        ticker_data = self.cache_data.get("tickers", {}).get(ticker, {})
        
        if not ticker_data:
            return True  # Nothing to remove
        
        ingested = ticker_data.get("ingested_quarters", [])
        if quarter_str in ingested:
            ingested.remove(quarter_str)
            ticker_data["last_ingestion"] = datetime.now().isoformat()
            return self._save_cache()
        
        return True  # Wasn't there anyway
    
    def clear_ticker(self, ticker: str) -> bool:
        """
        Clear all ingested quarters for a ticker.
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            True if successfully cleared, False otherwise
        """
        ticker = ticker.upper()
        
        if ticker in self.cache_data.get("tickers", {}):
            del self.cache_data["tickers"][ticker]
            return self._save_cache()
        
        return True  # Nothing to clear
    
    def generate_quarters_to_fetch(self, ticker: str, quarters_back: int) -> List[str]:
        """
        Generate list of quarters that need to be fetched for a ticker.
        
        Uses Alpha Vantage Overview API to get fiscal year information and latest quarter,
        then generates proper fiscal quarters based on company-specific fiscal calendar.
        
        Args:
            ticker: Company ticker symbol
            quarters_back: Number of recent quarters requested
            
        Returns:
            List of quarter strings that need to be fetched
        """
        # Use fiscal-aware quarter generation from Alpha Vantage source
        expected_quarters = self._generate_recent_quarters_fiscal_aware(ticker, quarters_back)
        
        # Get already ingested quarters
        ingested_quarters = self.get_ingested_quarters(ticker)
        
        # Find quarters that need to be fetched
        quarters_to_fetch = []
        for quarter in expected_quarters:
            if quarter not in ingested_quarters:
                quarters_to_fetch.append(quarter)
        
        return quarters_to_fetch
    
    def _generate_recent_quarters_fiscal_aware(self, ticker: str, quarters_back: int) -> List[str]:
        """
        Generate recent quarters using Alpha Vantage Overview API for fiscal year awareness.
        
        Args:
            ticker: Company ticker symbol
            quarters_back: Number of recent quarters requested
            
        Returns:
            List of quarter strings in "Q1 2024" format
        """
        try:
            # Import Alpha Vantage source to get fiscal quarter logic
            try:
                from .alpha_vantage_source import AlphaVantageTranscriptSource
            except ImportError:
                from alpha_vantage_source import AlphaVantageTranscriptSource
            
            # Create Alpha Vantage source instance
            alpha_vantage = AlphaVantageTranscriptSource()
            
            # Get fiscal-aware quarters using Alpha Vantage's logic
            recent_quarters = alpha_vantage._get_recent_quarters_sequence(ticker, quarters_back)
            
            # Convert from "2025Q1" format to "Q1 2025" format
            converted_quarters = []
            for quarter_str in recent_quarters:
                if 'Q' in quarter_str:
                    # Split "2025Q1" -> ["2025", "Q1"] -> "Q1 2025"
                    parts = quarter_str.split('Q')
                    if len(parts) == 2:
                        year = parts[0]
                        quarter = f"Q{parts[1]}"
                        converted_quarters.append(f"{quarter} {year}")
                    else:
                        # Fallback to original format
                        converted_quarters.append(quarter_str)
                else:
                    converted_quarters.append(quarter_str)
            
            return converted_quarters
            
        except Exception as e:
            print(f"Warning: Could not get fiscal-aware quarters for {ticker}: {e}")
            # Fallback to calendar-based quarters if fiscal logic fails
            return self._generate_recent_quarters(quarters_back)
    
    def _generate_recent_quarters(self, quarters_back: int) -> List[str]:
        """
        Generate recent quarters based on current date.
        
        Args:
            quarters_back: Number of quarters to go back
            
        Returns:
            List of quarter strings like ["Q3 2024", "Q2 2024", "Q1 2024"]
        """
        current_date = datetime.now()
        quarters = []
        
        for i in range(quarters_back):
            # Go back i quarters (3 months each)
            quarter_date = current_date - timedelta(days=i * 90)
            quarter_num = ((quarter_date.month - 1) // 3) + 1
            quarter_str = f"Q{quarter_num} {quarter_date.year}"
            quarters.append(quarter_str)
        
        return quarters
    
    def _sort_quarters(self, quarters: List[str]) -> List[str]:
        """
        Sort quarters chronologically (newest first).
        
        Args:
            quarters: List of quarter strings
            
        Returns:
            Sorted list of quarter strings
        """
        def quarter_sort_key(quarter_str):
            try:
                # Parse "Q1 2024" -> (2024, 1)
                parts = quarter_str.split()
                if len(parts) != 2:
                    return (0, 0)  # Put invalid entries at the end
                
                quarter_part = parts[0]  # "Q1"
                year_part = parts[1]     # "2024"
                
                if quarter_part.startswith('Q') and quarter_part[1:].isdigit():
                    quarter_num = int(quarter_part[1:])
                    year = int(year_part)
                    return (year, quarter_num)
                else:
                    return (0, 0)
            except (ValueError, IndexError):
                return (0, 0)
        
        # Sort by year and quarter (newest first)
        return sorted(quarters, key=quarter_sort_key, reverse=True)
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        tickers_data = self.cache_data.get("tickers", {})
        
        total_tickers = len(tickers_data)
        total_quarters = sum(
            len(data.get("ingested_quarters", [])) 
            for data in tickers_data.values()
        )
        
        last_updated = self.cache_data.get("last_updated", "unknown")
        
        return {
            "total_tickers": total_tickers,
            "total_quarters_cached": total_quarters,
            "last_updated": last_updated,
            "cache_file": self.cache_file,
            "file_exists": os.path.exists(self.cache_file)
        }
    
    def list_all_tickers(self) -> List[str]:
        """
        Get list of all tickers that have cached quarters.
        
        Returns:
            List of ticker symbols
        """
        return list(self.cache_data.get("tickers", {}).keys())
    
    def track_quarters(self, ticker: str, quarters: List[str]) -> Dict:
        """
        Legacy method for backwards compatibility.
        
        Args:
            ticker: Company ticker symbol
            quarters: List of quarter strings
            
        Returns:
            Status dictionary
        """
        success = self.add_multiple_quarters(ticker, quarters)
        return {
            "success": success,
            "tracked": len(quarters) if quarters else 0
        }
    
    def get_tracked_quarters(self, ticker: str) -> List[str]:
        """
        Legacy method for backwards compatibility.
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            List of tracked quarter strings
        """
        return self.get_ingested_quarters(ticker)


# Convenience functions for backwards compatibility
def create_simple_quarters_tracker(cache_file: Optional[str] = None) -> SimpleQuartersTracker:
    """Create a SimpleQuartersTracker instance."""
    return SimpleQuartersTracker(cache_file)


def get_default_tracker() -> SimpleQuartersTracker:
    """Get a default tracker instance."""
    return SimpleQuartersTracker()


if __name__ == "__main__":
    # Example usage
    tracker = SimpleQuartersTracker()
    
    print("=== Simple Quarters Tracker Test ===")
    print(f"Cache stats: {tracker.get_cache_stats()}")
    
    # Test adding quarters
    test_ticker = "AAPL"
    test_quarters = ["Q3 2024", "Q2 2024", "Q1 2024"]
    
    print(f"\nAdding quarters for {test_ticker}: {test_quarters}")
    success = tracker.add_multiple_quarters(test_ticker, test_quarters)
    print(f"Success: {success}")
    
    # Test getting ingested quarters
    ingested = tracker.get_ingested_quarters(test_ticker)
    print(f"Ingested quarters: {ingested}")
    
    # Test generating quarters to fetch
    quarters_to_fetch = tracker.generate_quarters_to_fetch(test_ticker, 6)
    print(f"Quarters to fetch (6 back): {quarters_to_fetch}")
    
    print(f"\nFinal cache stats: {tracker.get_cache_stats()}")