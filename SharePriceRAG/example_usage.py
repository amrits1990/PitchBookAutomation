"""
Example usage script for SharePriceRAG package
"""

import os
import sys
import json
from datetime import date, datetime, timedelta

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from SharePriceRAG import get_share_prices, get_ticker_summary, health_check, cleanup_database
from SharePriceRAG import SharePriceConfig


def example_basic_usage():
    """Basic usage example - single ticker"""
    print("\n=== Example 1: Basic Usage - Single Ticker ===")
    
    result = get_share_prices(
        tickers="AAPL",
        start_date="2024-01-01",
        end_date="2025-08-08"
    )
    
    if result.success:
        print(f"✅ Success! Retrieved {result.total_records} records")
        print(f"   Cache hit rate: {result.cache_hit_rate:.1%}")
        print(f"   Processing time: {result.processing_time_seconds:.2f}s")
        print(f"   Records from cache: {result.records_from_cache}")
        print(f"   Records fetched: {result.records_fetched}")
        
        # Show sample data
        if result.price_data:
            sample = result.price_data[0]
            print(f"\n   Sample data:")
            print(f"   {sample.ticker} {sample.date}: ${sample.close_price} (Source: {sample.source})")
    else:
        print(f"❌ Failed: {result.errors}")


# def example_multiple_tickers():
#     """Multiple tickers example"""
#     print("\n=== Example 2: Multiple Tickers ===")
    
#     tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
#     result = get_share_prices(
#         tickers=tickers,
#         start_date="2024-01-15",
#         end_date="2024-01-19"  # One week
#     )
    
#     if result.success:
#         print(f"✅ Retrieved data for {len(tickers)} tickers")
#         print(f"   Total records: {result.total_records}")
#         print(f"   Cache hit rate: {result.cache_hit_rate:.1%}")
        
#         # Show data by ticker
#         for ticker in tickers:
#             ticker_data = result.get_ticker_data(ticker)
#             if ticker_data:
#                 latest = max(ticker_data, key=lambda x: x.date)
#                 print(f"   {ticker}: ${latest.close_price} on {latest.date} (Source: {latest.source})")
#             else:
#                 print(f"   {ticker}: No data available")
                
#         # Show any missing ranges
#         if result.missing_ranges:
#             print(f"\n   Missing data ranges:")
#             for missing in result.missing_ranges:
#                 print(f"   - {missing.ticker}: {missing.start_date} to {missing.end_date} ({missing.days_missing} days)")
#     else:
#         print(f"❌ Failed: {result.errors}")


# def example_custom_config():
#     """Example with custom configuration"""
#     print("\n=== Example 3: Custom Configuration ===")
    
#     # Custom config
#     config = SharePriceConfig(
#         preferred_data_source="yahoo_finance",  # Use Yahoo Finance first
#         validate_prices=True,
#         max_price_change_percent=20.0,  # Flag price changes over 20%
#         requests_per_minute=10  # Higher rate limit if you have paid API
#     )
    
#     result = get_share_prices(
#         tickers=["NVDA", "AMD"],
#         start_date="2024-01-01",
#         end_date="2024-01-07",
#         config=config
#     )
    
#     if result.success:
#         print(f"✅ Custom config example completed")
#         print(f"   Records: {result.total_records}")
#         print(f"   Warnings: {len(result.warnings)}")
        
#         if result.warnings:
#             for warning in result.warnings[:3]:  # Show first 3 warnings
#                 print(f"   ⚠️  {warning}")
#     else:
#         print(f"❌ Failed: {result.errors}")


# def example_force_refresh():
#     """Example with force refresh"""
#     print("\n=== Example 4: Force Refresh (Re-fetch from APIs) ===")
    
#     result = get_share_prices(
#         tickers="AAPL",
#         start_date="2024-01-01",
#         end_date="2024-01-05",
#         force_refresh=True  # Force re-fetch even if data exists
#     )
    
#     if result.success:
#         print(f"✅ Force refresh completed")
#         print(f"   All {result.total_records} records fetched from APIs")
#         print(f"   Cache hit rate: {result.cache_hit_rate:.1%} (should be 0%)")
#     else:
#         print(f"❌ Failed: {result.errors}")


# def example_include_weekends():
#     """Example including weekend dates"""
#     print("\n=== Example 5: Include Weekends (for Crypto/24-7 Markets) ===")
    
#     result = get_share_prices(
#         tickers="BTC-USD",  # Bitcoin (if supported)
#         start_date="2024-01-06",  # Saturday
#         end_date="2024-01-07",    # Sunday
#         include_weekends=True
#     )
    
#     if result.success:
#         print(f"✅ Weekend data example")
#         print(f"   Records: {result.total_records}")
        
#         for price in result.price_data:
#             day_name = price.date.strftime("%A")
#             print(f"   {price.ticker} {price.date} ({day_name}): ${price.close_price}")
#     else:
#         print(f"❌ Failed (expected for traditional stocks): {result.errors}")


# def example_ticker_summary():
#     """Example of ticker summary"""
#     print("\n=== Example 6: Ticker Summary ===")
    
#     summary = get_ticker_summary("AAPL")
    
#     if "error" not in summary:
#         print(f"✅ AAPL Summary:")
#         print(f"   Total records: {summary['statistics'].get('total_records', 0)}")
#         print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
#         print(f"   Sources: {summary['statistics'].get('source_breakdown', {})}")
#     else:
#         print(f"❌ Summary failed: {summary['error']}")


# def example_health_check():
#     """Example of system health check"""
#     print("\n=== Example 7: Health Check ===")
    
#     health = health_check()
    
#     print(f"🏥 System Health:")
#     print(f"   Database: {'✅' if health['database'] else '❌'}")
#     print(f"   Alpha Vantage: {'✅' if health['api_sources'].get('alpha_vantage') else '❌'}")
#     print(f"   Yahoo Finance: {'✅' if health['api_sources'].get('yahoo_finance') else '❌'}")
#     print(f"   Has Alpha Vantage key: {'✅' if health['configuration']['has_alpha_vantage_key'] else '❌'}")
#     print(f"   Preferred source: {health['configuration']['preferred_source']}")


# def example_json_export():
#     """Example of exporting results to JSON"""
#     print("\n=== Example 8: JSON Export ===")
    
#     result = get_share_prices(
#         tickers=["MSFT", "GOOGL"],
#         start_date="2024-01-02",
#         end_date="2024-01-03"
#     )
    
#     if result.success:
#         # Convert to JSON
#         json_data = result.to_dict()
        
#         # Save to file
#         filename = f"share_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#         with open(filename, 'w') as f:
#             json.dump(json_data, f, indent=2, default=str)
        
#         print(f"✅ Results exported to {filename}")
#         print(f"   File size: {os.path.getsize(filename)} bytes")
#         print(f"   Contains {len(json_data['price_data'])} price records")
#     else:
#         print(f"❌ Export failed: {result.errors}")


# def example_date_range_analysis():
#     """Example analyzing date ranges and gaps"""
#     print("\n=== Example 9: Date Range Analysis ===")
    
#     # Request a longer date range to see gap detection in action
#     result = get_share_prices(
#         tickers="AAPL",
#         start_date="2023-12-01",  # Longer range
#         end_date="2024-01-31"
#     )
    
#     if result.success:
#         print(f"✅ Date range analysis:")
#         print(f"   Requested: 2023-12-01 to 2024-01-31")
#         print(f"   Total records: {result.total_records}")
        
#         # Analyze by month
#         if result.price_data:
#             dates = [p.date for p in result.price_data]
#             earliest = min(dates)
#             latest = max(dates)
#             print(f"   Actual range: {earliest} to {latest}")
            
#             # Count by source
#             sources = {}
#             for price in result.price_data:
#                 sources[price.source] = sources.get(price.source, 0) + 1
            
#             print(f"   Data sources:")
#             for source, count in sources.items():
#                 print(f"   - {source}: {count} records")
#     else:
#         print(f"❌ Analysis failed: {result.errors}")


# def main():
#     """Run all examples"""
#     print("🚀 SharePriceRAG - Example Usage Script")
#     print("=" * 50)
    
#     # Check if .env file exists
#     env_file = os.path.join(os.path.dirname(__file__), '.env')
#     if not os.path.exists(env_file):
#         print("⚠️  .env file not found. Some examples may fail without API keys.")
#         print("   Create a .env file with your configuration (see .env.example)")
    
#     try:
#         # Run examples
#         example_basic_usage()
#         example_multiple_tickers()
#         example_custom_config()
#         example_force_refresh()
#         example_include_weekends()
#         example_ticker_summary()
#         example_health_check()
#         example_json_export()
#         example_date_range_analysis()
        
#         print("\n" + "=" * 50)
#         print("✅ All examples completed!")
#         print("\n💡 Tips:")
#         print("   - Set ALPHA_VANTAGE_API_KEY in .env for better rate limits")
#         print("   - Yahoo Finance works without API key but may be rate limited")
#         print("   - Check the generated JSON file for complete data structure")
#         print("   - Use health_check() to verify your setup")
        
#     except ImportError as e:
#         print(f"❌ Import error: {e}")
#         print("   Make sure SharePriceRAG is in your Python path")
#         print("   Run: pip install -r requirements.txt")
#     except Exception as e:
#         print(f"❌ Unexpected error: {e}")
#         import traceback
#         traceback.print_exc()


def main():
    """Run basic example"""
    print("🚀 SharePriceRAG - Example Usage Script")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_file):
        print("⚠️  .env file not found. Some examples may fail without API keys.")
        print("   Create a .env file with your configuration (see .env.example)")
    
    try:
        # Run basic example only for now
        example_basic_usage()
        
        print("\n" + "=" * 50)
        print("✅ Basic example completed!")
        print("\n💡 Tips:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Set up PostgreSQL database")
        print("   - Copy .env.example to .env and configure")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure dependencies are installed: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()