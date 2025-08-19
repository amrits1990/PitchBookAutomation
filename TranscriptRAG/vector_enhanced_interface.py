"""
Vector-enhanced interface for TranscriptRAG
This provides the same API but with vector database backend
"""

import sys
import os

# Add AgentSystem to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AgentSystem'))

try:
    from AgentSystem.vector_store import enhance_rag_package
    from .agent_interface import index_transcripts_for_agent, search_transcripts_for_agent
    
    # Create vector-enhanced versions
    enhanced_index_transcripts, enhanced_search_transcripts = enhance_rag_package(
        index_func=index_transcripts_for_agent,
        search_func=search_transcripts_for_agent,
        table_name="transcripts",
        content_field="content"  # TranscriptRAG uses 'content' field
    )
    
    # Export enhanced functions with same names for drop-in replacement
    index_transcripts_for_agent_vector = enhanced_index_transcripts
    search_transcripts_for_agent_vector = enhanced_search_transcripts
    
except ImportError as e:
    print(f"Vector enhancement not available: {e}")
    # Fall back to original functions
    from .agent_interface import index_transcripts_for_agent, search_transcripts_for_agent
    
    index_transcripts_for_agent_vector = index_transcripts_for_agent
    search_transcripts_for_agent_vector = search_transcripts_for_agent