"""
Vector-enhanced interface for NewsRAG
This provides the same API but with vector database backend
"""

import sys
import os

# Add AgentSystem to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AgentSystem'))

try:
    from AgentSystem.vector_store import enhance_rag_package
    from .agent_interface import index_news_for_agent, search_news_for_agent
    
    # Create vector-enhanced versions
    enhanced_index_news, enhanced_search_news = enhance_rag_package(
        index_func=index_news_for_agent,
        search_func=search_news_for_agent,
        table_name="news",
        content_field="content"  # NewsRAG uses 'content' field
    )
    
    # Export enhanced functions with same names for drop-in replacement
    index_news_for_agent_vector = enhanced_index_news
    search_news_for_agent_vector = enhanced_search_news
    
except ImportError as e:
    print(f"Vector enhancement not available: {e}")
    # Fall back to original functions
    from .agent_interface import index_news_for_agent, search_news_for_agent
    
    index_news_for_agent_vector = index_news_for_agent
    search_news_for_agent_vector = search_news_for_agent