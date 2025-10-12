"""
Example usage of the NewsRAG package
"""

import os
import sys
import json

# Add the parent directory to Python path so we can import NewsRAG
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Load environment variables from .env file if it exists
def load_env_file():
    # Look for .env file in the same directory as this script (NewsRAG folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(current_dir, '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
        print(f"✅ Loaded environment variables from: {env_file}")
    else:
        print(f"ℹ️  No .env file found at: {env_file}")

load_env_file()

from NewsRAG import get_company_news_chunks, NewsConfig
from NewsRAG.models import NewsCategory
from NewsRAG.agent_interface import (
    get_news_for_agent,
    get_news_sentiment_analysis,
    get_news_by_date_range_and_topic
)


def main():
    """Demonstrate NewsRAG functionality"""
    
    print("=== NewsRAG Example Usage ===\n")
    
    # Check if API key is set
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ TAVILY_API_KEY environment variable not set!")
        print("Please set your Tavily API key using one of these methods:")
        print("1. PowerShell: $env:TAVILY_API_KEY = 'your_key_here'")
        print("2. Create a .env file in the NewsRAG folder with: TAVILY_API_KEY=your_key_here")
        print("3. Set permanently: [Environment]::SetEnvironmentVariable('TAVILY_API_KEY', 'your_key_here', 'User')")
        print("\nGet your Tavily API key from: https://tavily.com/")
        return
    
    # Example 1: Basic usage
    print("1. Basic news retrieval for Apple")
    print("-" * 50)
    
    try:
        result = get_company_news_chunks(
            companies=["Apple", "AAPL"],
            categories=["earnings", "products", "general"],
            days_back=7,
            max_articles_per_query=5,
            output_file="apple_news_example.json"
        )
        
        if result["success"]:
            summary = result["summary"]
            print(f"✅ Success!")
            print(f"   Articles retrieved: {summary['articles_retrieved']}")
            print(f"   Chunks generated: {summary['chunks_generated']}")
            print(f"   Total words: {summary['total_word_count']}")
            print(f"   Companies covered: {summary['companies_covered']}")
            print(f"   Processing time: {result['processing_time_seconds']:.2f}s")
            
            # Show sample chunk
            if result["chunks"]:
                chunk = result["chunks"][0]
                print(f"\n   Sample chunk:")
                print(f"   Title: {chunk['metadata']['title'][:60]}...")
                print(f"   Content preview: {chunk['content'][:100]}...")
                print(f"   Source: {chunk['metadata']['source']}")
        else:
            print(f"❌ Failed: {result['errors']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Agent Interface - Basic News Retrieval
    print("2. Agent Interface - Basic News Retrieval")
    print("-" * 50)
    
    try:
        agent_result = get_news_for_agent(
            ticker="NVDA",
            days_back=7,
            categories=["general", "earnings"],
            max_articles=5
        )
        
        if agent_result["success"]:
            print(f"✅ Agent Interface Success!")
            print(f"   Ticker: {agent_result['ticker']}")
            print(f"   Total articles: {agent_result['total_articles']}")
            print(f"   Sentiment: {agent_result['sentiment']}")
            print(f"   Summary: {agent_result['summary'][:100]}...")
            
            # Show key points
            if agent_result["key_points"]:
                print(f"   Key Points:")
                for point in agent_result["key_points"][:3]:
                    print(f"     {point[:80]}...")
            
            # Show recent developments
            if agent_result["recent_developments"]:
                print(f"   Recent Developments:")
                for dev in agent_result["recent_developments"][:2]:
                    print(f"     - {dev[:60]}...")
        else:
            print(f"❌ Agent Interface Failed: {agent_result['errors']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Agent Interface - Sentiment Analysis
    print("3. Agent Interface - Sentiment Analysis")
    print("-" * 50)
    
    try:
        sentiment_result = get_news_sentiment_analysis(
            ticker="TSLA",
            days_back=14
        )
        
        print(f"✅ Sentiment Analysis Complete!")
        print(f"   Ticker: {sentiment_result['ticker']}")
        print(f"   Sentiment: {sentiment_result['sentiment']}")
        print(f"   Confidence: {sentiment_result['confidence']}")
        print(f"   Analysis: {sentiment_result['analysis']}")
        print(f"   Recommendation: {sentiment_result['recommendation'][:100]}...")
        
        # Show supporting evidence
        evidence = sentiment_result["supporting_evidence"]
        if evidence["positive_headlines"]:
            print(f"   Positive Headlines:")
            for headline in evidence["positive_headlines"]:
                print(f"     + {headline[:70]}...")
        
        if evidence["negative_headlines"]:
            print(f"   Negative Headlines:")
            for headline in evidence["negative_headlines"]:
                print(f"     - {headline[:70]}...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Agent Interface - Date Range and Topic Search
    print("4. Agent Interface - Date Range and Topic Search")
    print("-" * 50)
    
    try:
        date_range_result = get_news_by_date_range_and_topic(
            company="Apple",
            start_date="2024-01-01",
            end_date="2024-01-31", 
            topic="products",
            max_articles=10
        )
        
        if date_range_result["success"]:
            print(f"✅ Date Range Search Success!")
            print(f"   Company: {date_range_result['company']}")
            print(f"   Date Range: {date_range_result['date_range']['start']} to {date_range_result['date_range']['end']}")
            print(f"   Topic: {date_range_result['topic']}")
            print(f"   Articles Found: {date_range_result['articles_found']}")
            print(f"   Sentiment: {date_range_result['sentiment']}")
            print(f"   Summary: {date_range_result['summary'][:120]}...")
            
            # Show key headlines
            if date_range_result["key_headlines"]:
                print(f"   Key Headlines:")
                for headline in date_range_result["key_headlines"][:3]:
                    print(f"     📰 {headline[:80]}...")
            
            # Show insights
            if date_range_result["key_insights"]:
                print(f"   Key Insights:")
                for insight in date_range_result["key_insights"][:2]:
                    print(f"     💡 {insight[:80]}...")
            
            # Show date distribution
            if date_range_result["date_distribution"]:
                print(f"   Date Distribution:")
                for date, count in list(date_range_result["date_distribution"].items())[:3]:
                    print(f"     {date}: {count} articles")
            
            # Show source breakdown
            if date_range_result["source_breakdown"]:
                print(f"   Top Sources:")
                for source, count in list(date_range_result["source_breakdown"].items())[:3]:
                    print(f"     {source}: {count} articles")
            
            print(f"   Available Topics: {', '.join(date_range_result['available_topics'])}")
        else:
            print(f"❌ Date Range Search Failed: {date_range_result['errors']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 5: Agent Workflow Simulation
    print("5. Simulated Agent Workflow - Company Analysis")
    print("-" * 50)
    
    try:
        print("🤖 Agent Task: Analyze Tesla's recent news sentiment and product developments")
        
        # Step 1: Get overall news sentiment
        print("\n   Step 1: Checking overall sentiment...")
        sentiment = get_news_sentiment_analysis("TSLA", days_back=30)
        print(f"   Agent Decision: Sentiment is {sentiment['sentiment']} with {sentiment['confidence']} confidence")
        
        # Step 2: Get specific product news in date range
        print("\n   Step 2: Searching for product developments...")
        products = get_news_by_date_range_and_topic(
            company="Tesla",
            start_date="2024-01-01",
            end_date="2024-02-29",
            topic="products",
            max_articles=15
        )
        
        if products["success"]:
            print(f"   Agent Finding: Found {products['articles_found']} product-related articles")
            if products["key_headlines"]:
                print(f"   Agent Analysis: Latest product headline: {products['key_headlines'][0][:100]}...")
        
        # Step 3: Get comprehensive overview
        print("\n   Step 3: Getting comprehensive overview...")
        overview = get_news_for_agent("TSLA", days_back=21, max_articles=20)
        
        if overview["success"]:
            print(f"   Agent Summary: {overview['summary'][:150]}...")
            print(f"   Agent Conclusion: {len(overview['recent_developments'])} recent developments identified")
        
        print("\n   🎯 Agent Workflow Complete!")
        print(f"   📊 Total Data Points: Sentiment analysis + {products.get('articles_found', 0)} product articles + {overview.get('total_articles', 0)} general articles")
        
    except Exception as e:
        print(f"❌ Agent Workflow Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # # Example 6: Multiple companies and categories
    # print("6. Multiple companies - Tech giants")
    # print("-" * 50)
    
    # try:
    #     result = get_company_news_chunks(
    #         companies=["Apple", "Microsoft", "Google", "Amazon"],
    #         categories=["earnings", "acquisitions", "partnerships"],
    #         days_back=14,
    #         max_articles_per_query=3,
    #         output_file="tech_giants_news.json"
    #     )
        
    #     if result["success"]:
    #         summary = result["summary"]
    #         print(f"✅ Success!")
    #         print(f"   Articles: {summary['articles_retrieved']}")
    #         print(f"   Chunks: {summary['chunks_generated']}")
    #         print(f"   Companies found: {', '.join(summary['companies_covered'][:5])}")
            
    #         # Show category breakdown
    #         categories_found = {}
    #         for chunk in result["chunks"]:
    #             cat = chunk["metadata"].get("category", "unknown")
    #             categories_found[cat] = categories_found.get(cat, 0) + 1
            
    #         print(f"   Categories found: {categories_found}")
    #     else:
    #         print(f"❌ Failed: {result['errors']}")
            
    # except Exception as e:
    #     print(f"❌ Error: {e}")
    
    # print("\n" + "="*60 + "\n")
    
    # # Example 3: Custom configuration
    # print("3. Custom configuration example")
    # print("-" * 50)
    
    # try:
    #     # Create custom config
    #     config = NewsConfig(
    #         chunk_size=800,          # Smaller chunks
    #         chunk_overlap=150,       # Less overlap
    #         min_word_count=75,       # Lower minimum
    #         max_articles_per_query=8 # More articles
    #     )
        
    #     result = get_company_news_chunks(
    #         companies=["Tesla"],
    #         categories=["financial", "regulatory", "products"],
    #         days_back=30,
    #         config=config,
    #         output_file="tesla_custom_config.json"
    #     )
        
    #     if result["success"]:
    #         summary = result["summary"]
    #         print(f"✅ Success with custom config!")
    #         print(f"   Articles: {summary['articles_retrieved']}")
    #         print(f"   Chunks: {summary['chunks_generated']}")
    #         print(f"   Avg words per chunk: {summary['total_word_count'] // summary['chunks_generated'] if summary['chunks_generated'] > 0 else 0}")
            
    #         # Show chunk size distribution
    #         chunk_sizes = [chunk["word_count"] for chunk in result["chunks"]]
    #         if chunk_sizes:
    #             print(f"   Chunk size range: {min(chunk_sizes)}-{max(chunk_sizes)} words")
    #     else:
    #         print(f"❌ Failed: {result['errors']}")
            
    # except Exception as e:
    #     print(f"❌ Error: {e}")
    
    # print("\n" + "="*60 + "\n")
    
    # # Example 4: Financial sector news
    # print("4. Financial sector news")
    # print("-" * 50)
    
    # try:
    #     banks = ["JPMorgan", "Bank of America", "Wells Fargo", "Goldman Sachs"]
        
    #     result = get_company_news_chunks(
    #         companies=banks,
    #         categories=["earnings", "regulatory", "financial"],
    #         days_back=21,
    #         max_articles_per_query=4,
    #         output_file="bank_news.json"
    #     )
        
    #     if result["success"]:
    #         summary = result["summary"]
    #         print(f"✅ Financial sector analysis complete!")
    #         print(f"   Banks covered: {len(summary['companies_covered'])} companies")
    #         print(f"   News chunks: {summary['chunks_generated']}")
            
    #         # Show date range
    #         if summary["date_range"]["earliest"] and summary["date_range"]["latest"]:
    #             print(f"   Date range: {summary['date_range']['earliest'][:10]} to {summary['date_range']['latest'][:10]}")
                
    #         # Recent articles
    #         recent_articles = []
    #         for chunk in result["chunks"][:3]:
    #             title = chunk["metadata"]["title"]
    #             date = chunk["metadata"]["published_date"][:10] if chunk["metadata"]["published_date"] else "Unknown"
    #             recent_articles.append(f"{date}: {title[:50]}...")
            
    #         if recent_articles:
    #             print(f"   Recent headlines:")
    #             for article in recent_articles:
    #                 print(f"     - {article}")
    #     else:
    #         print(f"❌ Failed: {result['errors']}")
            
    # except Exception as e:
    #     print(f"❌ Error: {e}")
    
    print("=== Agent Interface Examples Completed ===")
    print("\nAgent Functions Demonstrated:")
    print("✅ get_news_for_agent() - Structured news retrieval with agent-friendly output")
    print("✅ get_news_sentiment_analysis() - Focused sentiment analysis with confidence levels")
    print("✅ get_news_by_date_range_and_topic() - Precise date range and topic filtering")
    print("✅ Agent Workflow Simulation - Multi-step analysis combining all functions")
    print("\nGenerated files:")
    print("- apple_news_example.json")
    print("\nAll agent functions return structured dictionaries optimized for:")
    print("- LangChain agent consumption")
    print("- Error handling and validation")
    print("- Sentiment analysis and insights")
    print("- RAG-ready chunks for vector storage")


if __name__ == "__main__":
    main()