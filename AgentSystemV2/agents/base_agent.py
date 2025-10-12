"""
Base Agent Class for AgentSystemV2
"""

import logging
import json
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.memory import MemoryManager

from config.settings import OPENROUTER_API_KEY, DEFAULT_MODEL_ID, DEFAULT_MAX_TOKENS
from config.llm_cost_tracker import TokenUsage, extract_token_usage_from_agno


class BaseFinancialAgent(ABC):
    """Base class for all financial analysis agents."""

    def __init__(self,
                 agent_name: str,
                 model_id: str = DEFAULT_MODEL_ID,
                 enable_debug: bool = False,
                 use_memory: bool = False,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        """
        Initialize base agent.

        Args:
            agent_name: Name identifier for the agent
            model_id: OpenRouter model identifier
            enable_debug: Enable debug logging
            use_memory: Whether to use agno memory system
            max_tokens: Maximum tokens for model completion
        """
        self.agent_name = agent_name
        self.model_id = model_id
        self.enable_debug = enable_debug
        self.use_memory = use_memory
        self.max_tokens = max_tokens
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        if enable_debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Initialize agno agent
        self.agent = self._create_agent()
        
        self.logger.info(f"{agent_name} initialized with model {model_id}")
    
    def _create_agent(self) -> Agent:
        """Create and configure the agno agent."""

        # Get RAG tools
        tools = self._get_rag_tools()

        agent_config = {
            "name": self.agent_name,
            "model": OpenRouter(
                id=self.model_id,
                api_key=OPENROUTER_API_KEY,
                max_completion_tokens=self.max_tokens
            ),
            "instructions": self._get_agent_instructions(),
            "tools": tools,
            "debug_mode": self.enable_debug,
            "markdown": False
        }
        
        # Add memory if requested
        if self.use_memory:
            agent_config["memory_manager"] = MemoryManager()
        
        return Agent(**agent_config)
    
    def _get_rag_tools(self):
        """Get RAG tools for investigative analysis using clean import approach."""
        try:
            from tools.rag_tools import AVAILABLE_RAG_TOOLS, get_rag_tools_status
            
            # Get status for logging
            status = get_rag_tools_status()
            self.logger.info(f"RAG Tools Status: {status}")
            
            # Return all available tools
            self.logger.info(f"Loaded {len(AVAILABLE_RAG_TOOLS)} RAG tools for agent")
            return AVAILABLE_RAG_TOOLS
            
        except Exception as e:
            self.logger.error(f"Failed to load RAG tools: {e}")
            return []
    
    @abstractmethod
    def _get_agent_instructions(self) -> str:
        """Get the system instructions for this agent."""
        pass
    
    async def _run_agent(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> tuple[str, List[Dict[str, Any]], TokenUsage]:
        """
        Run the agent with the given prompt.

        Args:
            prompt: The prompt to send to the agent
            context: Optional context dictionary

        Returns:
            Tuple of (agent response as string, tool calls made, token usage)
        """
        try:
            # Add context to prompt if provided
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                full_prompt = f"{context_str}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Run agent
            response = await self.agent.arun(full_prompt)

            # Extract token usage from Agno response
            token_usage = extract_token_usage_from_agno(response, self.model_id)

            # Extract tool calls from the agent's run response
            tool_calls_made = []
            try:
                # The response from agno might have tool_calls information
                if hasattr(response, 'messages'):
                    # Extract tool calls from messages
                    for msg in response.messages:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_calls_made.append({
                                    "function": tool_call.function.name if hasattr(tool_call, 'function') else str(tool_call),
                                    "parameters": json.loads(tool_call.function.arguments) if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments') else {},
                                    "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else ""
                                })

                # Alternative: Check if the agent itself has a session or run history
                if hasattr(self.agent, 'run_response') and self.agent.run_response:
                    run_resp = self.agent.run_response
                    if hasattr(run_resp, 'tools') and run_resp.tools:
                        for tool_result in run_resp.tools:
                            tool_calls_made.append({
                                "function": tool_result.get('name', 'unknown'),
                                "parameters": tool_result.get('arguments', {}),
                                "result": str(tool_result.get('result', ''))[:200]  # Truncate long results
                            })

                self.logger.debug(f"Captured {len(tool_calls_made)} tool calls from agent execution")

            except Exception as tool_extract_error:
                self.logger.warning(f"Could not extract tool calls: {tool_extract_error}")

            # Extract text response
            response_text = ""
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            # Log cost information if tokens were tracked
            if token_usage.total_tokens > 0:
                cost = token_usage.calculate_cost()
                self.logger.info(f"LLM Usage: {token_usage.input_tokens} input + {token_usage.output_tokens} output = {token_usage.total_tokens} tokens (${cost:.4f})")

            return response_text, tool_calls_made, token_usage

        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            raise
    
    def _validate_required_fields(self, data: Dict[str, Any], required_fields: list) -> list:
        """Validate that required fields are present in data."""
        missing = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing.append(field)
        return missing
    
    def _format_currency(self, amount: float) -> str:
        """Format currency amount for display."""
        if abs(amount) >= 1e9:
            return f"${amount/1e9:.1f}B"
        elif abs(amount) >= 1e6:
            return f"${amount/1e6:.1f}M"
        elif abs(amount) >= 1e3:
            return f"${amount/1e3:.1f}K"
        else:
            return f"${amount:.2f}"
    
    def _format_percentage(self, ratio: float) -> str:
        """Format ratio as percentage."""
        return f"{ratio*100:.1f}%"
    
    def _extract_latest_quarter_data(self, quarterly_data: Dict[str, Any]) -> tuple:
        """Extract latest quarter data and value."""
        if not quarterly_data:
            return None, None
        
        # Handle different data structures
        # Case 1: Simple quarter -> value mapping
        if all(isinstance(v, (int, float)) for v in quarterly_data.values()):
            sorted_quarters = sorted(quarterly_data.keys(), reverse=True)
            latest_quarter = sorted_quarters[0]
            latest_value = quarterly_data[latest_quarter]
            return latest_quarter, latest_value
        
        # Case 2: Complex ratio data structure from SECFinancialRAG
        # Keys like "Cash_Ratio_Q4-2022", values are dicts with {'value': Decimal, 'period_end_date': ...}
        latest_quarter = None
        latest_value = None
        latest_date = None
        
        for key, data in quarterly_data.items():
            if isinstance(data, dict) and 'value' in data:
                # Extract quarter from key (e.g., "Cash_Ratio_Q4-2022" -> "Q4-2022")
                quarter = None
                if '_Q' in key:
                    quarter = key.split('_Q')[1]  # Get "4-2022" part
                    quarter = f"Q{quarter}"  # Make it "Q4-2022"
                elif 'period_end_date' in data:
                    # Use period_end_date as fallback
                    quarter = data['period_end_date']
                
                if quarter:
                    # Extract numeric value
                    value = data['value']
                    if hasattr(value, '__float__'):  # Handle Decimal objects
                        value = float(value)
                    elif isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            continue
                    
                    # Keep the latest (most recent) entry
                    if latest_quarter is None or quarter > latest_quarter:
                        latest_quarter = quarter
                        latest_value = value
                        latest_date = data.get('period_end_date')
        
        # If we found complex data, return it
        if latest_quarter and latest_value is not None:
            return latest_quarter, latest_value
        
        # Fallback: if no valid data found, return None
        return None, None
    
    def _calculate_trend(self, quarterly_data: Dict[str, float], periods: int = 4) -> str:
        """Calculate trend direction over specified periods."""
        if not quarterly_data or len(quarterly_data) < 2:
            return "insufficient_data"
        
        # Sort quarters chronologically
        sorted_items = sorted(quarterly_data.items())
        
        if len(sorted_items) < periods:
            periods = len(sorted_items)
        
        # Compare recent periods
        recent_values = [item[1] for item in sorted_items[-periods:]]
        
        # Simple trend calculation
        if len(recent_values) >= 2:
            start_value = recent_values[0]
            end_value = recent_values[-1]
            
            if end_value > start_value * 1.05:  # 5% threshold
                return "improving"
            elif end_value < start_value * 0.95:  # 5% threshold
                return "declining" 
            else:
                return "stable"
        
        return "unknown"