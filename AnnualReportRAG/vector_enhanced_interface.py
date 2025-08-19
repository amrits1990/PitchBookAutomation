"""
Vector-enhanced interface for AnnualReportRAG
This provides the same API but with vector database backend
"""

import sys
import os

# Add AgentSystem to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AgentSystem'))

try:
    from AgentSystem.vector_store import enhance_rag_package
    from .agent_interface import index_reports_for_agent, search_report_for_agent
    
    # Create vector-enhanced versions
    enhanced_index_reports, enhanced_search_reports = enhance_rag_package(
        index_func=index_reports_for_agent,
        search_func=search_report_for_agent,
        table_name="annual_reports",
        content_field="text"  # AnnualReportRAG uses 'text' field
    )
    
    # Export enhanced functions with same names for drop-in replacement
    index_reports_for_agent_vector = enhanced_index_reports
    search_report_for_agent_vector = enhanced_search_reports
    
except ImportError as e:
    print(f"Vector enhancement not available: {e}")
    # Fall back to original functions
    from .agent_interface import index_reports_for_agent, search_report_for_agent
    
    index_reports_for_agent_vector = index_reports_for_agent
    search_report_for_agent_vector = search_report_for_agent