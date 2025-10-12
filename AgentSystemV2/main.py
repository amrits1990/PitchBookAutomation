"""
AgentSystemV2 - Main Entry Point
Simplified Domain Agent System with Analyst-Reviewer Loops
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from orchestration.master_agent import MasterAgent
from orchestration.agent_factory import get_agent_factory
from config.schemas import MasterAnalysisRequest
from config.domain_configs import get_available_domains
from config.settings import DEBUG_MODE, LOG_LEVEL


# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AgentSystemV2:
    """
    Main AgentSystemV2 interface providing simplified access to domain agents.
    
    Features:
    - Individual domain analysis with analyst-reviewer loops
    - Multi-domain parallel analysis with master coordination  
    - Interactive chat for post-analysis refinements
    - Cost optimization and quality controls
    """
    
    def __init__(self, enable_debug: bool = DEBUG_MODE):
        """
        Initialize AgentSystemV2.
        
        Args:
            enable_debug: Enable debug logging
        """
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize master agent
        self.master_agent = MasterAgent(enable_debug=enable_debug)
        self.agent_factory = get_agent_factory(enable_debug=enable_debug)
        
        self.logger.info("AgentSystemV2 initialized")
    
    async def analyze_domain(self,
                           company: str,
                           domain: str,
                           peers: List[str] = None,
                           user_focus: str = "comprehensive analysis",
                           time_period: str = "3 years",
                           persona: str = "institutional_investor",
                           max_loops: int = 3) -> Dict[str, Any]:
        """
        Analyze single domain with configurable analyst-reviewer loops.

        Args:
            company: Company ticker (e.g., "AAPL")
            domain: Domain type (liquidity, leverage, working_capital, operating_efficiency, valuation)
            peers: List of peer company tickers
            user_focus: Specific focus for analysis
            time_period: Analysis time period
            persona: Target audience (banker, cfo, trader, institutional_investor, retail_investor)
            max_loops: Maximum analyst-reviewer loops

        Returns:
            Complete domain analysis results
        """
        return await self.master_agent.run_single_domain_analysis(
            company=company,
            domain=domain,
            peers=peers or [],
            user_focus=user_focus,
            time_period=time_period,
            persona=persona,
            max_loops=max_loops
        )
    
    async def analyze_comprehensive(self,
                                  company: str,
                                  domains: List[str] = None,
                                  user_focus: str = "comprehensive financial analysis",
                                  time_period: str = "3 years",
                                  run_parallel: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive multi-domain analysis.
        
        Args:
            company: Company ticker
            domains: List of domains to analyze (default: all available)
            user_focus: Overall analysis focus
            time_period: Analysis time period
            run_parallel: Whether to run domains in parallel
            
        Returns:
            Comprehensive multi-domain analysis with master synthesis
        """
        if domains is None:
            domains = get_available_domains()
        
        request = MasterAnalysisRequest(
            company=company,
            ticker=company,
            domains=domains,
            user_focus=user_focus,
            time_period=time_period,
            run_parallel=run_parallel,
            max_cost_per_domain=1.0
        )
        
        return await self.master_agent.run_multi_domain_analysis(request)
    
    async def start_chat_session(self, run_id: str) -> Dict[str, Any]:
        """
        Start interactive chat session for completed analysis.
        
        Args:
            run_id: Analysis run ID from previous analysis
            
        Returns:
            Chat session initialization result
        """
        return await self.master_agent.start_chat_session(run_id)
    
    async def chat_with_analysis(self, 
                               session_id: str,
                               user_message: str,
                               refinement_type: str = "clarification") -> Dict[str, Any]:
        """
        Send message to chat session.
        
        Args:
            session_id: Chat session ID
            user_message: User's question or request
            refinement_type: Type of refinement (clarification, deeper_analysis, alternative_view)
            
        Returns:
            Chat response with analysis or refinement
        """
        return await self.master_agent.chat_interface.process_chat_message(
            session_id=session_id,
            user_message=user_message,
            refinement_type=refinement_type
        )
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domain types."""
        return get_available_domains()
    
    def get_domain_capabilities(self, domain: str) -> Dict[str, Any]:
        """Get capabilities and requirements for specific domain."""
        return self.agent_factory.get_domain_capabilities(domain)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        master_summary = self.master_agent.get_execution_summary()
        factory_status = self.agent_factory.get_factory_status()
        
        return {
            "system_version": "AgentSystemV2",
            "available_domains": self.get_available_domains(),
            "master_agent": master_summary,
            "agent_factory": factory_status,
            "debug_enabled": self.enable_debug
        }


# Convenience functions for direct usage
async def analyze_liquidity(company: str, 
                          peers: List[str] = None,
                          user_focus: str = "cash runway and working capital trends") -> Dict[str, Any]:
    """Quick liquidity analysis."""
    system = AgentSystemV2()
    return await system.analyze_domain(
        company=company,
        domain="liquidity",
        peers=peers,
        user_focus=user_focus
    )

async def analyze_comprehensive_financial(company: str, 
                                        user_focus: str = "investment analysis") -> Dict[str, Any]:
    """Quick comprehensive financial analysis."""
    system = AgentSystemV2()
    return await system.analyze_comprehensive(
        company=company,
        user_focus=user_focus,
        run_parallel=True
    )


# CLI Interface
async def main():
    """Main CLI interface for AgentSystemV2."""
    
    print("🤖 AgentSystemV2 - Simplified Domain Agent System")
    print("=" * 60)
    
    system = AgentSystemV2(enable_debug=True)
    
    # Display available domains
    domains = system.get_available_domains()
    print(f"📊 Available Domains: {', '.join(domains)}")
    
    # Example usage
    print("\n🔍 Running example analysis...")
    
    try:
        # Single domain analysis
        result = await system.analyze_domain(
            company="AAPL",
            domain="liquidity",
            peers=["MSFT", "GOOGL"],
            user_focus="cash management and liquidity trends",
            max_loops=2
        )
        
        if result["success"]:
            analysis = result["analysis_output"]
            print(f"\n✅ Analysis completed:")
            print(f"   Company: {analysis.company}")
            print(f"   Domain: {analysis.domain}")
            print(f"   Confidence: {analysis.confidence_level}")
            print(f"   Loops used: {result['loops_used']}")
            print(f"   Key findings: {len(analysis.key_findings)}")
            
            # Start chat session
            print(f"\n💬 Starting chat session...")
            chat_result = await system.start_chat_session(result["run_id"])
            
            if "session_id" in chat_result:
                print(f"   Chat session: {chat_result['session_id']}")
                
                # Example chat interaction
                chat_response = await system.chat_with_analysis(
                    session_id=chat_result["session_id"],
                    user_message="What are the main liquidity risks for Apple?",
                    refinement_type="clarification"
                )
                
                print(f"   Chat response type: {chat_response.get('type')}")
        else:
            print(f"❌ Analysis failed: {result.get('error')}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # System status
    status = system.get_system_status()
    print(f"\n📈 System Status:")
    print(f"   Total analyses: {status['master_agent']['total_analyses']}")
    print(f"   Success rate: {status['master_agent']['success_rate']:.1%}")
    print(f"   Active agents: {status['agent_factory']['active_agents_count']}")


if __name__ == "__main__":
    asyncio.run(main())