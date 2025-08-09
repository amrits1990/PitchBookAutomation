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
    
    # # Example 2: Multiple companies and categories
    # print("2. Multiple companies - Tech giants")
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
    
    # print("\n=== Examples completed ===")
    # print("\nGenerated files:")
    # print("- apple_news_example.json")
    # print("- tech_giants_news.json") 
    # print("- tesla_custom_config.json")
    # print("- bank_news.json")
    # print("\nThese files contain the full RAG-ready news data with chunks and metadata.")


if __name__ == "__main__":
    main()