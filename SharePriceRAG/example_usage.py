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
from SharePriceRAG.agent_interface import get_price_analysis_for_agent, compare_with_peers, get_raw_price_data_for_agent


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


def demo_price_analysis_agent():
    """Interactive demo of price analysis agent function"""
    print("\n=== Agent Function: Price Analysis ===")
    print("This function provides comprehensive price trend analysis for agents")
    
    # Get user input
    ticker = input("\nEnter ticker symbol (e.g., AAPL): ").strip().upper()
    if not ticker:
        print("❌ Ticker symbol is required")
        return
    
    try:
        days_back = input("Enter analysis period in days (default 90): ").strip()
        days_back = int(days_back) if days_back else 90
    except ValueError:
        days_back = 90
    
    include_peers = input("Include peer comparison? (y/N): ").strip().lower() == 'y'
    peer_tickers = []
    
    if include_peers:
        peers_input = input("Enter peer tickers separated by commas (e.g., MSFT,GOOGL): ").strip()
        if peers_input:
            peer_tickers = [p.strip().upper() for p in peers_input.split(',')]
    
    print(f"\n🔍 Analyzing {ticker} for {days_back} days...")
    if peer_tickers:
        print(f"   Including peers: {', '.join(peer_tickers)}")
    
    try:
        result = get_price_analysis_for_agent(
            ticker=ticker,
            days_back=days_back,
            include_peers=include_peers,
            peer_tickers=peer_tickers
        )
        
        if result["success"]:
            print(f"\n✅ Analysis Complete for {result['ticker']}!")
            print("=" * 50)
            
            # Core metrics
            print(f"📊 PRICE ANALYSIS:")
            print(f"   Current Price: ${result['current_price']}")
            print(f"   Price Trend: {result['price_trend'].title()}")
            print(f"   Volatility: {result['volatility'].title()}")
            print(f"   Risk Assessment: {result['risk_assessment'].title()}")
            
            # Performance summary
            print(f"\n📈 PERFORMANCE SUMMARY:")
            print(f"   {result['performance_summary']}")
            
            # Key metrics
            metrics = result['key_metrics']
            print(f"\n🔢 KEY METRICS:")
            print(f"   Total Return: {metrics['total_return_pct']}%")
            print(f"   Annualized Return: {metrics['annualized_return_pct']}%")
            print(f"   Volatility: {metrics['volatility_pct']}%")
            print(f"   Period High: ${metrics['period_high']}")
            print(f"   Period Low: ${metrics['period_low']}")
            print(f"   Data Points: {metrics['data_points']}")
            
            # Trading signals
            if result['trading_signals']:
                print(f"\n🎯 TRADING SIGNALS:")
                for signal in result['trading_signals']:
                    print(f"   • {signal}")
            
            # Peer comparison
            if result.get('peer_comparison') and 'error' not in result['peer_comparison']:
                peer = result['peer_comparison']
                print(f"\n👥 PEER COMPARISON:")
                print(f"   Target Return: {peer['target_return_pct']}%")
                print(f"   Peer Average: {peer['peer_avg_return_pct']}%")
                print(f"   Relative Performance: {peer['relative_performance_pct']}%")
                print(f"   Ranking: {peer['rank']} out of {peer['total_compared']}")
                print(f"   Status: {'✅ Outperforming' if peer['outperforming'] else '📉 Underperforming'}")
                print(f"   Summary: {peer['summary']}")
            
            # Metadata
            meta = result['metadata']
            print(f"\n📋 ANALYSIS INFO:")
            print(f"   Analysis Period: {meta['analysis_period_days']} days")
            print(f"   Data Source: {meta['data_source']}")
            print(f"   Analysis Time: {meta['analysis_timestamp'][:19]}")
            
        else:
            print(f"\n❌ Analysis Failed:")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")


def demo_peer_comparison_agent():
    """Interactive demo of peer comparison agent function"""
    print("\n=== Agent Function: Peer Comparison ===")
    print("This function provides dedicated peer performance comparison")
    
    # Get user input
    ticker = input("\nEnter target ticker symbol (e.g., AAPL): ").strip().upper()
    if not ticker:
        print("❌ Ticker symbol is required")
        return
    
    peers_input = input("Enter peer tickers separated by commas (e.g., MSFT,GOOGL,AMZN): ").strip()
    if not peers_input:
        print("❌ Peer tickers are required")
        return
    
    peer_tickers = [p.strip().upper() for p in peers_input.split(',')]
    
    try:
        days_back = input("Enter analysis period in days (default 90): ").strip()
        days_back = int(days_back) if days_back else 90
    except ValueError:
        days_back = 90
    
    print(f"\n🔍 Comparing {ticker} with peers: {', '.join(peer_tickers)}")
    print(f"   Analysis period: {days_back} days")
    
    try:
        result = compare_with_peers(
            ticker=ticker,
            peer_tickers=peer_tickers,
            days_back=days_back
        )
        
        if "error" not in result:
            print(f"\n✅ Peer Comparison Complete!")
            print("=" * 50)
            
            # Target performance
            print(f"🎯 TARGET PERFORMANCE:")
            print(f"   {result['target_ticker']}: {result['target_return_pct']}% return")
            
            # Peer average
            print(f"\n👥 PEER AVERAGE:")
            print(f"   Average Return: {result['peer_avg_return_pct']}%")
            print(f"   Relative Performance: {result['relative_performance_pct']}%")
            
            # Ranking
            print(f"\n🏆 RANKING:")
            print(f"   Position: {result['rank']} out of {result['total_compared']}")
            print(f"   Status: {'✅ Outperforming' if result['outperforming'] else '📉 Underperforming'}")
            
            # Individual peer analysis
            if result.get('peer_analysis'):
                print(f"\n📊 INDIVIDUAL PEER PERFORMANCE:")
                for peer_ticker, peer_data in result['peer_analysis'].items():
                    print(f"   {peer_ticker}: {peer_data['return_pct']}% (${peer_data['current_price']})")
            
            # Summary
            print(f"\n📝 SUMMARY:")
            print(f"   {result['summary']}")
            
        else:
            print(f"\n❌ Peer Comparison Failed:")
            print(f"   Error: {result['error']}")
            
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")


def demo_raw_price_data_agent():
    """Interactive demo of raw price data agent function"""
    print("\n=== Agent Function: Raw Price Data ===")
    print("This function provides raw price DataFrame for custom agent analysis")
    
    # Get user input
    ticker = input("\nEnter ticker symbol (e.g., AAPL): ").strip().upper()
    if not ticker:
        print("❌ Ticker symbol is required")
        return
    
    try:
        days_back = input("Enter number of days to retrieve (default 90): ").strip()
        days_back = int(days_back) if days_back else 90
    except ValueError:
        days_back = 90
    
    print(f"\n🔍 Retrieving {days_back} days of raw price data for {ticker}...")
    
    try:
        result = get_raw_price_data_for_agent(
            ticker=ticker,
            days_back=days_back
        )
        
        if result["success"]:
            print(f"\n✅ Raw Data Retrieved for {result['ticker']}!")
            print("=" * 50)
            
            # Data overview
            df = result['data']
            print(f"📊 DATAFRAME OVERVIEW:")
            print(f"   Shape: {result['metadata']['data_shape']} (rows, columns)")
            print(f"   Records Count: {result['records_count']}")
            print(f"   Columns: {', '.join(result['metadata']['columns'])}")
            print(f"   Date Range: {result['date_range']['start']} to {result['date_range']['end']}")
            
            # Data sources
            if result['data_sources']:
                print(f"\n📡 DATA SOURCES:")
                for source, count in result['data_sources'].items():
                    print(f"   {source}: {count} records")
            
            # Cache information
            cache = result['cache_info']
            print(f"\n💾 CACHE PERFORMANCE:")
            print(f"   Cache Hit Rate: {cache['cache_hit_rate']:.1%}")
            print(f"   Records from Cache: {cache['records_from_cache']}")
            print(f"   Records Fetched: {cache['records_fetched']}")
            print(f"   Processing Time: {cache['processing_time_seconds']:.2f}s")
            
            # Sample data preview
            if not df.empty:
                print(f"\n📈 SAMPLE DATA (First 5 rows):")
                print(df.head().to_string())
                
                # Basic statistics
                print(f"\n📊 BASIC STATISTICS:")
                print(f"   Latest Close: ${df['close'].iloc[-1]:.2f}")
                print(f"   Period High: ${df['close'].max():.2f}")
                print(f"   Period Low: ${df['close'].min():.2f}")
                print(f"   Average Volume: {df['volume'].mean():,.0f}" if df['volume'].notna().any() else "   Average Volume: N/A")
                
                # Data quality
                null_counts = result['metadata']['has_nulls']
                if any(count > 0 for count in null_counts.values()):
                    print(f"\n⚠️  DATA QUALITY (Missing Values):")
                    for col, count in null_counts.items():
                        if count > 0:
                            print(f"   {col}: {count} missing values")
                else:
                    print(f"\n✅ DATA QUALITY: No missing values")
            
            # Agent usage suggestions
            print(f"\n🤖 AGENT USAGE SUGGESTIONS:")
            print(f"   • Use df['close'] for price analysis")
            print(f"   • Calculate returns with df['close'].pct_change()")
            print(f"   • Analyze volume patterns with df['volume']")
            print(f"   • Compute technical indicators (RSI, MA, etc.)")
            print(f"   • DataFrame is indexed by date for easy time-series operations")
            
        else:
            print(f"\n❌ Data Retrieval Failed:")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")


def interactive_agent_demo():
    """Interactive demo for SharePriceRAG agent functions"""
    print("🤖 SharePriceRAG - Agent Functions Interactive Demo")
    print("=" * 55)
    
    # Check environment
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_file):
        print("⚠️  .env file not found. Some functions may fail without proper configuration.")
        print("   Create a .env file with your database and API settings")
    
    while True:
        print("\n🎯 Available Agent Functions:")
        print("1. Price Analysis for Agent - Comprehensive trend analysis with trading signals")
        print("2. Compare with Peers - Performance ranking against peer companies")
        print("3. Raw Price Data for Agent - Get DataFrame for custom analysis")
        print("4. Exit")
        
        choice = input("\nSelect an agent function (1-4): ").strip()
        
        if choice == '1':
            demo_price_analysis_agent()
        elif choice == '2':
            demo_peer_comparison_agent()
        elif choice == '3':
            demo_raw_price_data_agent()
        elif choice == '4':
            print("\n👋 Goodbye! Thanks for trying SharePriceRAG agent functions.")
            break
        else:
            print("❌ Invalid choice. Please select 1, 2, 3, or 4.")
        
        # Ask if user wants to continue
        if choice in ['1', '2', '3']:
            continue_demo = input("\nTry another agent function? (Y/n): ").strip().lower()
            if continue_demo == 'n':
                print("\n👋 Demo completed! Thanks for trying SharePriceRAG.")
                break


def main():
    """Run interactive agent demo"""
    try:
        interactive_agent_demo()
        
        print("\n" + "=" * 55)
        print("✅ SharePriceRAG Agent Demo Completed!")
        print("\n💡 Agent Integration Tips:")
        print("   • All functions return structured dictionaries")
        print("   • Error handling is built-in with standardized responses")
        print("   • Use get_price_analysis_for_agent() for trend analysis")
        print("   • Use compare_with_peers() for performance benchmarking")
        print("   • Use get_raw_price_data_for_agent() for custom DataFrame analysis")
        print("   • All outputs are optimized for LangChain/agent consumption")
        print("\n🔧 Setup Requirements:")
        print("   • PostgreSQL database configured")
        print("   • Alpha Vantage API key (optional but recommended)")
        print("   • Internet connection for data fetching")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure dependencies are installed: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()