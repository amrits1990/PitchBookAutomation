"""
Agent-Style Financial RAG Interface Demo for SECFinancialRAG

This demo simulates how AI agents will interact with the SECFinancialRAG system:
1. Uses standardized FinancialAgentResponse format
2. Demonstrates natural language query processing with OpenRouter LLM
3. Shows exact response formats that will be provided to agents as context
4. Tests all agent interface functions with realistic scenarios
5. Provides both structured data and natural language interpretations

This represents the actual interface that AI agents will use to query financial data.
"""

import logging
import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
from dataclasses import asdict
from dotenv import load_dotenv

# Add the parent directory to Python path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agent_interface import (
        get_financial_metrics_for_agent,
        get_ratios_for_agent,
        compare_companies_for_agent,
        get_ratio_definition_for_agent,
        get_available_ratio_categories_for_agent,
        FinancialAgentResponse
    )
    from standalone_interface import (
        get_financial_data,
        get_multiple_companies_data,
        get_company_ratios_only
    )
    from database import FinancialDatabase
    from models import Company
    print("✓ Successfully imported agent interface and package modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please run from the SECFinancialRAG directory or install the package")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs for demo clarity
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# Load environment variables from parent directory  
_current_dir = Path(__file__).parent
_parent_env_file = _current_dir.parent.parent / '.env'
if _parent_env_file.exists():
    load_dotenv(_parent_env_file)
    logger.debug(f"Loaded environment from {_parent_env_file}")
else:
    logger.warning(f"Parent .env file not found at {_parent_env_file}")

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1/chat/completions'
DEMO_LLM_MODEL = os.getenv('SEC_FINANCIAL_DEMO_LLM_MODEL', 'anthropic/claude-3.5-sonnet')

class FinancialAgentSimulator:
    """Simulates how an AI agent would interact with the financial RAG system"""
    
    def __init__(self):
        self.openrouter_available = bool(OPENROUTER_API_KEY)
        if not self.openrouter_available:
            print("⚠️  OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable for LLM responses.")
        else:
            print(f"✓ Demo LLM configured: {DEMO_LLM_MODEL}")
    
    def get_llm_response(self, system_prompt: str, user_query: str, context_data: Dict) -> str:
        """Get natural language response from LLM using financial data as context"""
        if not self.openrouter_available:
            return "[LLM Response unavailable - OpenRouter API key not configured]"
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"User Query: {user_query}\n\nFinancial Data Context: {json.dumps(context_data, indent=2, default=str)}"
                }
            ]
            
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": DEMO_LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 800,
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"[LLM Error: {response.status_code}]"
                
        except Exception as e:
            return f"[LLM Error: {str(e)}]"
    
    def simulate_agent_response(self, raw_response: FinancialAgentResponse, 
                               user_query: str, response_type: str) -> Dict[str, Any]:
        """Simulate complete agent response with LLM interpretation"""
        
        # Convert FinancialAgentResponse to dict for JSON serialization
        if hasattr(raw_response, '__dict__'):
            response_dict = asdict(raw_response)
        else:
            response_dict = raw_response
        
        system_prompts = {
            "metrics": "You are a financial analyst AI. Analyze the provided financial metrics data and give a concise, insightful response to the user's query. Focus on key insights, trends, and what the numbers mean for the business.",
            "ratios": "You are a financial analyst AI. Analyze the provided financial ratios data and give a comprehensive assessment. Explain what the ratios indicate about the company's financial health, performance, and compare to typical industry standards where relevant.",
            "comparison": "You are a financial analyst AI. Analyze the company comparison data and provide insights about relative performance. Highlight key differences, competitive positions, and what the metrics reveal about each company's strengths and weaknesses.",
            "definition": "You are a financial education AI. Explain the ratio definition and provide practical guidance on interpretation. Make it educational and actionable.",
            "categories": "You are a financial data AI. Explain the available ratio categories and help the user understand what types of analysis they can perform with each category."
        }
        
        # Get LLM interpretation
        llm_response = self.get_llm_response(
            system_prompts.get(response_type, system_prompts["metrics"]),
            user_query,
            response_dict
        )
        
        # Create agent-style response
        agent_response = {
            "query": user_query,
            "response_type": response_type,
            "structured_data": response_dict,
            "natural_language_response": llm_response,
            "agent_metadata": {
                "processing_time": datetime.utcnow().isoformat(),
                "data_freshness": "real_time",
                "confidence_score": 0.95 if raw_response.success else 0.1,
                "sources_used": ["SEC_EDGAR", "Company_Filings"]
            }
        }
        
        return agent_response


def demo_financial_metrics_query(ticker: str, metrics: List[str], period: str = 'LTM') -> Dict[str, Any]:
    """
    Demo: Agent-style financial metrics query with LLM interpretation
    
    This demonstrates exactly how agents will interact with financial data:
    1. Call agent interface function
    2. Get standardized FinancialAgentResponse
    3. Process with LLM for natural language response
    4. Return complete agent-style response
    """
    print(f"\n=== AGENT DEMO: Financial Metrics Query ===")
    print(f"Query: Get {', '.join(metrics)} for {ticker.upper()} ({period})")
    print(f"Agent Function: get_financial_metrics_for_agent()")
    
    # Step 1: Call agent interface (this is what agents will do)
    raw_response = get_financial_metrics_for_agent(ticker, metrics, period)
    
    # Step 2: Simulate agent processing with LLM
    simulator = FinancialAgentSimulator()
    user_query = f"What are the {', '.join(metrics)} for {ticker} in the {period} period?"
    agent_response = simulator.simulate_agent_response(raw_response, user_query, "metrics")
    
    # Step 3: Display results (this is what would be provided to user)
    print(f"\n--- RAW API RESPONSE ---")
    print(f"Success: {raw_response.success}")
    if raw_response.success:
        print(f"Data Keys: {list(raw_response.data.keys())}")
        print(f"Metadata: {raw_response.metadata.get('ticker', 'N/A')} - {raw_response.metadata.get('total_periods', 0)} periods")
        print(f"LLM Mapping Used: {raw_response.metadata.get('field_mapping_used', False)}")
        
        # Show field mapping details
        mapping_details = raw_response.metadata.get('mapping_details', {})
        print(f"\n--- FIELD MAPPING DETAILS ---")
        for metric, details in mapping_details.items():
            print(f"{metric}:")
            print(f"  Confidence: {details.get('confidence', 0):.2f}")
            print(f"  Method: {details.get('mapping_method', 'Unknown')}")
            if details.get('direct_field'):
                print(f"  Direct Field: {details['direct_field']}")
            if details.get('calculated_from'):
                print(f"  Calculated From: {details['calculated_from']}")
                print(f"  Calculation: {details.get('calculation_method', 'N/A')}")
        
        # Show LLM suggestions if any
        suggestions = raw_response.metadata.get('llm_suggestions', [])
        if suggestions:
            print(f"\n--- LLM SUGGESTIONS ---")
            for suggestion in suggestions:
                print(f"  - {suggestion}")
    else:
        print(f"Error: {raw_response.error}")
    
    print(f"\n--- AGENT RESPONSE (What user gets) ---")
    print(f"Query: {agent_response['query']}")
    print(f"Response Type: {agent_response['response_type']}")
    print(f"\nNatural Language Response:")
    print(agent_response['natural_language_response'])
    
    print(f"\n--- STRUCTURED DATA (JSON Context) ---")
    print(json.dumps(agent_response['structured_data'], indent=2, default=str))
    
    return agent_response


def demo_financial_ratios_query(ticker: str, categories: List[str] = None, period: str = 'LTM') -> Dict[str, Any]:
    """
    Demo: Agent-style financial ratios query with LLM interpretation
    
    This shows exactly how agents will query and interpret financial ratios
    """
    print(f"\n=== AGENT DEMO: Financial Ratios Query ===")
    categories_str = ', '.join(categories) if categories else 'all categories'
    print(f"Query: Get {categories_str} ratios for {ticker.upper()} ({period})")
    print(f"Agent Function: get_ratios_for_agent()")
    
    # Step 1: Call agent interface
    raw_response = get_ratios_for_agent(ticker, categories, period)
    
    # Step 2: Simulate agent processing
    simulator = FinancialAgentSimulator()
    user_query = f"Show me the {categories_str} for {ticker}. How is their financial performance?"
    agent_response = simulator.simulate_agent_response(raw_response, user_query, "ratios")
    
    # Step 3: Display results
    print(f"\n--- RAW API RESPONSE ---")
    print(f"Success: {raw_response.success}")
    if raw_response.success:
        print(f"Categories Found: {raw_response.metadata.get('categories_found', [])}")
        print(f"Total Ratios: {raw_response.metadata.get('total_ratios', 0)}")
    else:
        print(f"Error: {raw_response.error}")
    
    print(f"\n--- AGENT RESPONSE (What user gets) ---")
    print(f"Query: {agent_response['query']}")
    print(f"\nNatural Language Response:")
    print(agent_response['natural_language_response'])
    
    print(f"\n--- STRUCTURED DATA (JSON Context) ---")
    print(json.dumps(agent_response['structured_data'], indent=2, default=str))
    
    return agent_response


def demo_ratio_definition_query(ratio_name: str, ticker: str = None) -> Dict[str, Any]:
    """
    Demo: Agent-style ratio definition query with educational response
    
    This shows how agents will help users understand financial ratios
    """
    print(f"\n=== AGENT DEMO: Ratio Definition Query ===")
    print(f"Query: Explain '{ratio_name}' ratio{' for ' + ticker if ticker else ''}")
    print(f"Agent Function: get_ratio_definition_for_agent()")
    
    # Step 1: Call agent interface
    raw_response = get_ratio_definition_for_agent(ratio_name, ticker)
    
    # Step 2: Simulate agent processing
    simulator = FinancialAgentSimulator()
    user_query = f"What does the {ratio_name} ratio mean and how should I interpret it?"
    agent_response = simulator.simulate_agent_response(raw_response, user_query, "definition")
    
    # Step 3: Display results
    print(f"\n--- RAW API RESPONSE ---")
    print(f"Success: {raw_response.success}")
    if raw_response.success:
        print(f"Ratio Found: {raw_response.data.get('name', 'N/A')}")
        print(f"Category: {raw_response.data.get('category', 'N/A')}")
        print(f"Formula: {raw_response.data.get('formula', 'N/A')}")
    else:
        print(f"Error: {raw_response.error}")
    
    print(f"\n--- AGENT RESPONSE (What user gets) ---")
    print(f"Query: {agent_response['query']}")
    print(f"\nNatural Language Response:")
    print(agent_response['natural_language_response'])
    
    print(f"\n--- STRUCTURED DATA (JSON Context) ---")
    print(json.dumps(agent_response['structured_data'], indent=2, default=str))
    
    return agent_response


def demo_available_ratio_categories_query() -> Dict[str, Any]:
    """
    Demo: Agent-style query for available ratio categories
    
    This shows how agents can discover what ratio categories are available
    """
    print(f"\\n=== AGENT DEMO: Available Ratio Categories Query ===")
    print(f"Query: What ratio categories are available?")
    print(f"Agent Function: get_available_ratio_categories_for_agent()")
    
    # Step 1: Call agent interface
    raw_response = get_available_ratio_categories_for_agent()
    
    # Step 2: Simulate agent processing
    simulator = FinancialAgentSimulator()
    user_query = f"What types of financial ratio categories can I query?"
    agent_response = simulator.simulate_agent_response(raw_response, user_query, "categories")
    
    # Step 3: Display results
    print(f"\\n--- RAW API RESPONSE ---")
    print(f"Success: {raw_response.success}")
    if raw_response.success:
        print(f"Total Categories: {raw_response.metadata.get('total_categories', 0)}")
        print(f"Usage Note: {raw_response.metadata.get('usage_note', 'N/A')}")
        print(f"Period Formats: {raw_response.metadata.get('period_formats', 'N/A')}")
        
        print(f"\\n--- AVAILABLE CATEGORIES ---")
        if raw_response.data:
            for category, info in raw_response.data.items():
                print(f"{category.upper()}:")
                print(f"  Description: {info.get('description', 'N/A')}")
                examples = info.get('examples', [])
                print(f"  Example Ratios: {', '.join(examples[:3])}...")
    else:
        print(f"Error: {raw_response.error}")
    
    print(f"\\n--- AGENT RESPONSE (What user gets) ---")
    print(f"Query: {agent_response['query']}")
    print(f"\\nNatural Language Response:")
    print(agent_response['natural_language_response'])
    
    print(f"\\n--- STRUCTURED DATA (JSON Context) ---")
    print(json.dumps(agent_response['structured_data'], indent=2, default=str))
    
    return agent_response


def demo_companies_comparison_query(tickers: List[str], categories: List[str], period: str = 'latest') -> Dict[str, Any]:
    """
    Demo: Agent-style company comparison query with competitive analysis
    
    This shows how agents will perform comparative ratio analysis across companies
    """
    print(f"\n=== AGENT DEMO: Company Comparison Query ===")
    tickers_str = ', '.join([t.upper() for t in tickers])
    categories_str = ', '.join(categories)
    print(f"Query: Compare {tickers_str} on {categories_str} ratios ({period})")
    print(f"Agent Function: compare_companies_for_agent()")
    
    # Step 1: Call agent interface
    raw_response = compare_companies_for_agent(tickers, categories, period)
    
    # Step 2: Simulate agent processing
    simulator = FinancialAgentSimulator()
    user_query = f"Compare {tickers_str} companies on {categories_str} ratios. Which company is performing better and why?"
    agent_response = simulator.simulate_agent_response(raw_response, user_query, "comparison")
    
    # Step 3: Display results
    print(f"\n--- RAW API RESPONSE ---")
    print(f"Success: {raw_response.success}")
    if raw_response.success:
        print(f"Companies Analyzed: {raw_response.metadata.get('successful_tickers', [])}")
        print(f"Categories Compared: {raw_response.metadata.get('categories_requested', [])}")
        if 'failed_tickers' in raw_response.metadata:
            print(f"Failed Companies: {raw_response.metadata.get('failed_tickers', [])}")
    else:
        print(f"Error: {raw_response.error}")
    
    print(f"\n--- AGENT RESPONSE (What user gets) ---")
    print(f"Query: {agent_response['query']}")
    print(f"\nNatural Language Response:")
    print(agent_response['natural_language_response'])
    
    print(f"\n--- STRUCTURED DATA (JSON Context) ---")
    print(json.dumps(agent_response['structured_data'], indent=2, default=str))
    
    return agent_response


def demo_comprehensive_agent_scenario():
    """
    Demo: Complete agent interaction scenario covering all major use cases
    
    This simulates a realistic conversation flow with an AI agent
    """
    print(f"\n" + "="*80)
    print(f"COMPREHENSIVE AGENT INTERACTION DEMO")
    print(f"Simulating realistic user conversation with financial AI agent")
    print(f"="*80)
    
    scenarios = [
        {
            'description': "User asks about Apple's recent performance",
            'function': lambda: demo_financial_metrics_query(
                'AAPL', 
                ['total_revenue', 'net_income', 'operating_income'], 
                'LTM'
            )
        },
        {
            'description': "User wants to understand Apple's financial health",
            'function': lambda: demo_financial_ratios_query(
                'AAPL', 
                ['profitability', 'liquidity'], 
                'LTM'
            )
        },
        {
            'description': "User asks what ROE means",
            'function': lambda: demo_ratio_definition_query('ROE', 'AAPL')
        },
        {
            'description': "User wants to compare tech giants",
            'function': lambda: demo_companies_comparison_query(
                ['AAPL', 'MSFT', 'GOOGL'], 
                ['profitability', 'liquidity'], 
                'latest'
            )
        }
    ]
    
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[SCENARIO {i}] {scenario['description']}")
        print("-" * 60)
        try:
            result = scenario['function']()
            results.append(result)
        except Exception as e:
            print(f"ERROR in scenario {i}: {e}")
            results.append(None)
        
        if i < len(scenarios):
            input("\nPress Enter to continue to next scenario...")
    
    print(f"\n" + "="*80)
    print(f"DEMO COMPLETE - {len([r for r in results if r])} successful scenarios")
    print(f"="*80)
    
    return results


def interactive_agent_demo():
    """Interactive mode for testing agent-style financial queries"""
    print("SECFinancialRAG - Agent-Style Interface Demo")
    print("=" * 55)
    print("This demo shows exactly how AI agents will interact with financial data")
    print("All responses use standardized FinancialAgentResponse format with LLM interpretation\n")
    
    while True:
        print("\nAvailable Agent Demo Options:")
        print("   1. Financial Metrics Query (get_financial_metrics_for_agent)")
        print("   2. Financial Ratios Query (get_ratios_for_agent)") 
        print("   3. Ratio Definition Query (get_ratio_definition_for_agent)")
        print("   4. Company Comparison (compare_companies_for_agent)")
        print("   5. Available Ratio Categories (get_available_ratio_categories_for_agent)")
        print("   6. Comprehensive Demo Scenario")
        print("   7. Exit")
        
        choice = input("\n   Select option (1-7): ").strip()
        
        if choice == "1":
            ticker = input("   Enter ticker: ").strip().upper()
            if ticker:
                metrics_input = input("   Enter metrics (comma-separated): ").strip()
                metrics = [m.strip() for m in metrics_input.split(',')] if metrics_input else ['total_revenue', 'net_income']
                period = input("   Period (latest/FY2023/Q1-2024/last 3 quarters/last 2 financial years): ").strip() or "latest"
                demo_financial_metrics_query(ticker, metrics, period)
        
        elif choice == "2":
            ticker = input("   Enter ticker: ").strip().upper()
            if ticker:
                categories_input = input("   Categories (comma-separated, or Enter for all): ").strip()
                categories = [c.strip() for c in categories_input.split(',')] if categories_input else None
                period = input("   Period (latest/FY2023/Q1-2024/last 3 quarters/last 2 financial years): ").strip() or "latest"
                demo_financial_ratios_query(ticker, categories, period)
        
        elif choice == "3":
            ratio_name = input("   Enter ratio name (e.g., ROE, Current_Ratio): ").strip()
            if ratio_name:
                ticker = input("   Enter ticker (optional): ").strip().upper() or None
                demo_ratio_definition_query(ratio_name, ticker)
        
        elif choice == "4":
            tickers_input = input("   Enter tickers (comma-separated): ").strip()
            if tickers_input:
                tickers = [t.strip().upper() for t in tickers_input.split(',')]
                categories_input = input("   Enter ratio categories (comma-separated, e.g., profitability,liquidity): ").strip()
                categories = [c.strip() for c in categories_input.split(',')] if categories_input else ['profitability']
                period = input("   Period (latest/FY2023/Q1-2024/last 3 quarters/last 2 financial years): ").strip() or "latest"
                demo_companies_comparison_query(tickers, categories, period)
        
        elif choice == "5":
            demo_available_ratio_categories_query()
        
        elif choice == "6":
            demo_comprehensive_agent_scenario()
        
        elif choice == "7":
            print("\nAgent demo session ended")
            break
        
        else:
            print("   ERROR - Invalid choice. Please select 1-7.")


if __name__ == "__main__":
    print(f"\n🤖 SECFinancialRAG Agent Interface Demo")
    print(f"This demonstrates exactly how AI agents will interact with financial data\n")
    
    # Show configuration
    if OPENROUTER_API_KEY:
        print(f"✅ OpenRouter API configured - LLM responses enabled")
        print(f"   Agent LLM (field mapping): {os.getenv('SEC_FINANCIAL_AGENT_LLM_MODEL', 'anthropic/claude-3.5-sonnet')}")
        print(f"   Demo LLM (explanations): {DEMO_LLM_MODEL}")
    else:
        print(f"⚠️  OpenRouter API not configured - Set OPENROUTER_API_KEY for LLM responses")
    
    print(f"   Environment loaded from: {_parent_env_file}")
    
    try:
        interactive_agent_demo()
    except KeyboardInterrupt:
        print("\n\nAgent demo cancelled by user")
    except Exception as e:
        print(f"\nERROR - Unexpected error: {e}")
        logger.exception("Unexpected error in agent demo")