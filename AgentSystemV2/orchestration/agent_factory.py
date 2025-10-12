"""
Agent Factory - Creates and manages domain agents
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from agents.domain_agent import DomainAgent
from config.domain_configs import get_available_domains, get_domain_config
from config.settings import AVAILABLE_DOMAINS, DEFAULT_DOMAIN


class AgentFactory:
    """
    Factory for creating and managing domain agents.
    Provides unified interface for domain agent creation and lifecycle management.
    """
    
    def __init__(self, enable_debug: bool = False):
        """
        Initialize agent factory.
        
        Args:
            enable_debug: Enable debug logging for all created agents
        """
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(__name__)
        
        if enable_debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Active agents registry
        self.active_agents: Dict[str, DomainAgent] = {}
        self.agent_creation_history: List[Dict[str, str]] = []
        
        self.logger.info(f"AgentFactory initialized with {len(AVAILABLE_DOMAINS)} available domains")
    
    def create_domain_agent(self, domain: str) -> DomainAgent:
        """
        Create a domain agent for the specified domain.
        
        Args:
            domain: Domain type (liquidity, leverage, etc.)
            
        Returns:
            Configured domain agent
            
        Raises:
            ValueError: If domain is not supported
        """
        if domain not in get_available_domains():
            raise ValueError(f"Unsupported domain: {domain}. Available: {get_available_domains()}")
        
        self.logger.info(f"Creating domain agent for: {domain}")
        
        # Create domain agent
        domain_agent = DomainAgent(
            domain=domain,
            enable_debug=self.enable_debug
        )
        
        # Register agent
        agent_key = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_agents[agent_key] = domain_agent
        
        # Record creation
        self.agent_creation_history.append({
            "agent_key": agent_key,
            "domain": domain,
            "created_at": datetime.now().isoformat(),
            "status": "created"
        })
        
        self.logger.info(f"Domain agent created: {agent_key}")
        return domain_agent
    
    def get_domain_agent(self, domain: str, create_if_missing: bool = True) -> Optional[DomainAgent]:
        """
        Get existing domain agent or optionally create new one.
        
        Args:
            domain: Domain type
            create_if_missing: Create new agent if none exists
            
        Returns:
            Domain agent instance or None
        """
        # Look for existing agent for this domain
        for agent_key, agent in self.active_agents.items():
            if agent.domain == domain:
                self.logger.debug(f"Found existing agent for {domain}: {agent_key}")
                return agent
        
        # Create new agent if requested
        if create_if_missing:
            self.logger.info(f"No existing agent for {domain}, creating new one")
            return self.create_domain_agent(domain)
        
        return None
    
    def create_multiple_agents(self, domains: List[str]) -> Dict[str, DomainAgent]:
        """
        Create multiple domain agents for parallel execution.
        
        Args:
            domains: List of domain types to create agents for
            
        Returns:
            Dictionary mapping domain -> agent
        """
        agents = {}
        
        for domain in domains:
            try:
                agent = self.create_domain_agent(domain)
                agents[domain] = agent
                self.logger.info(f"Created agent for {domain}")
            except Exception as e:
                self.logger.error(f"Failed to create agent for {domain}: {e}")
                # Continue creating other agents
        
        self.logger.info(f"Created {len(agents)}/{len(domains)} requested agents")
        return agents
    
    def validate_domain_support(self, domains: List[str]) -> Dict[str, bool]:
        """
        Validate which domains are supported.
        
        Args:
            domains: List of domains to validate
            
        Returns:
            Dictionary mapping domain -> is_supported
        """
        available = get_available_domains()
        validation_results = {}
        
        for domain in domains:
            validation_results[domain] = domain in available
        
        return validation_results
    
    def get_domain_capabilities(self, domain: str) -> Dict[str, any]:
        """
        Get capabilities and configuration for a domain.
        
        Args:
            domain: Domain to get capabilities for
            
        Returns:
            Domain capabilities and configuration info
        """
        if domain not in get_available_domains():
            return {"supported": False, "error": f"Domain {domain} not supported"}
        
        config = get_domain_config(domain)
        
        return {
            "supported": True,
            "domain": domain,
            "description": config.description if hasattr(config, 'description') else 'No description',
            "required_metrics": len(config.required_metrics),
            "required_ratios": len(config.required_ratios),
            "peer_comparison_required": config.peer_comparison_required,
            "default_max_loops": config.default_max_loops,
            "reviewer_criteria_count": len(config.reviewer_criteria)
        }
    
    def get_factory_status(self) -> Dict[str, any]:
        """Get current factory status and statistics."""
        
        active_domains = list(set([agent.domain for agent in self.active_agents.values()]))
        
        return {
            "active_agents_count": len(self.active_agents),
            "active_domains": active_domains,
            "total_agents_created": len(self.agent_creation_history),
            "available_domains": get_available_domains(),
            "debug_enabled": self.enable_debug,
            "last_agent_created": self.agent_creation_history[-1]["created_at"] if self.agent_creation_history else None
        }
    
    def cleanup_agents(self, max_age_hours: int = 24) -> int:
        """
        Clean up old agents to free memory.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of agents cleaned up
        """
        current_time = datetime.now()
        agents_to_remove = []
        
        for agent_key in self.active_agents:
            # Extract timestamp from agent key
            try:
                timestamp_str = agent_key.split('_')[-2] + '_' + agent_key.split('_')[-1]
                agent_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                age_hours = (current_time - agent_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    agents_to_remove.append(agent_key)
            except Exception as e:
                self.logger.warning(f"Could not parse timestamp for agent {agent_key}: {e}")
        
        # Remove old agents
        for agent_key in agents_to_remove:
            del self.active_agents[agent_key]
            self.logger.info(f"Cleaned up old agent: {agent_key}")
        
        return len(agents_to_remove)
    
    def reset_factory(self):
        """Reset factory to clean state."""
        self.active_agents.clear()
        self.agent_creation_history.clear()
        self.logger.info("Agent factory reset to clean state")


# Global factory instance
_global_factory: Optional[AgentFactory] = None

def get_agent_factory(enable_debug: bool = False) -> AgentFactory:
    """Get global agent factory instance (singleton pattern)."""
    global _global_factory
    
    if _global_factory is None:
        _global_factory = AgentFactory(enable_debug=enable_debug)
    
    return _global_factory

def create_domain_agent(domain: str, enable_debug: bool = False) -> DomainAgent:
    """Convenience function to create domain agent."""
    factory = get_agent_factory(enable_debug=enable_debug)
    return factory.create_domain_agent(domain)