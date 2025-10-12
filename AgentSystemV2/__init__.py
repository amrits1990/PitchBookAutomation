"""
AgentSystemV2 - Simplified Domain Agent System

A lightweight, cost-optimized financial analysis system with:
- Configurable analyst-reviewer loops
- Multi-domain parallel analysis
- Interactive chat refinements
- Schema-based token optimization
- Agno framework integration
"""

from .main import AgentSystemV2, analyze_liquidity, analyze_comprehensive_financial
from .orchestration.master_agent import MasterAgent
from .orchestration.agent_factory import AgentFactory, create_domain_agent
from .agents.domain_agent import DomainAgent
from .tools.chat_interface import ChatInterface
from .config.schemas import AnalysisInput, ReviewResult, MasterAnalysisRequest, ResearchReportOutput
from .config.domain_configs import get_available_domains, get_domain_config

__version__ = "2.0.0"
__author__ = "AgentSystemV2 Team"

# Main exports
__all__ = [
    "AgentSystemV2",
    "MasterAgent",
    "DomainAgent",
    "ChatInterface",
    "AgentFactory",
    "create_domain_agent",
    "analyze_liquidity",
    "analyze_comprehensive_financial",
    "get_available_domains",
    "get_domain_config",
    "AnalysisInput",
    "ResearchReportOutput",
    "ReviewResult",
    "MasterAnalysisRequest"
]