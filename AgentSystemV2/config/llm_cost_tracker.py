"""
LLM Cost Tracker - Track token usage and costs across all agents
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# LLM Pricing (USD per 1M tokens)
# Updated as of 2025 - verify with OpenRouter/provider pricing pages
LLM_PRICING = {
    # OpenAI Models
    "openai/gpt-4o": {
        "input": 2.50,   # $2.50 per 1M input tokens
        "output": 10.00  # $10.00 per 1M output tokens
    },
    "openai/gpt-4o-mini": {
        "input": 0.150,   # $0.15 per 1M input tokens
        "output": 0.600   # $0.60 per 1M output tokens
    },
    "openai/gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00
    },
    "openai/gpt-3.5-turbo": {
        "input": 0.50,
        "output": 1.50
    },

    # Anthropic Models
    "anthropic/claude-3.5-sonnet": {
        "input": 3.00,
        "output": 15.00
    },
    "anthropic/claude-3-opus": {
        "input": 15.00,
        "output": 75.00
    },
    "anthropic/claude-3-sonnet": {
        "input": 3.00,
        "output": 15.00
    },
    "anthropic/claude-3-haiku": {
        "input": 0.25,
        "output": 1.25
    },

    # Google Models
    "google/gemini-pro-1.5": {
        "input": 1.25,
        "output": 5.00
    },
    "google/gemini-flash-1.5": {
        "input": 0.075,
        "output": 0.30
    },

    # X.AI Models
    "x-ai/grok-2": {
        "input": 2.00,
        "output": 10.00
    },
    "x-ai/grok-beta": {
        "input": 5.00,
        "output": 15.00
    },
    "x-ai/grok-4-fast": {
        "input": 0.50,
        "output": 1.50
    },

    # Meta Models
    "meta-llama/llama-3.1-405b": {
        "input": 2.70,
        "output": 2.70
    },
    "meta-llama/llama-3.1-70b": {
        "input": 0.52,
        "output": 0.75
    },
    "meta-llama/llama-3.1-8b": {
        "input": 0.18,
        "output": 0.18
    },

    # Mistral Models
    "mistralai/mistral-large": {
        "input": 3.00,
        "output": 9.00
    },
    "mistralai/mistral-medium": {
        "input": 2.70,
        "output": 8.10
    },

    # DeepSeek Models
    "deepseek/deepseek-chat": {
        "input": 0.14,
        "output": 0.28
    },

    # Default fallback pricing (conservative estimate)
    "default": {
        "input": 1.00,
        "output": 3.00
    }
}


@dataclass
class TokenUsage:
    """Token usage for a single LLM call"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def calculate_cost(self) -> float:
        """Calculate cost for this token usage"""
        return calculate_cost(self.input_tokens, self.output_tokens, self.model_id)


@dataclass
class AgentCost:
    """Cost tracking for a specific agent"""
    agent_name: str
    agent_type: str  # "analyst", "reviewer", "master"
    loop_number: Optional[int] = None

    # Token counts
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # Costs
    total_cost: float = 0.0

    # Individual calls
    calls: List[TokenUsage] = field(default_factory=list)

    def add_usage(self, usage: TokenUsage):
        """Add token usage from a single call"""
        self.calls.append(usage)
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost += usage.calculate_cost()


@dataclass
class RunCost:
    """Complete cost tracking for an entire analysis run"""
    run_id: str
    company: str
    domain: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    # Agent costs
    analyst_costs: List[AgentCost] = field(default_factory=list)
    reviewer_costs: List[AgentCost] = field(default_factory=list)
    master_costs: List[AgentCost] = field(default_factory=list)

    # Totals
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    def add_analyst_cost(self, cost: AgentCost):
        """Add analyst agent cost"""
        self.analyst_costs.append(cost)
        self._update_totals()

    def add_reviewer_cost(self, cost: AgentCost):
        """Add reviewer agent cost"""
        self.reviewer_costs.append(cost)
        self._update_totals()

    def add_master_cost(self, cost: AgentCost):
        """Add master agent cost"""
        self.master_costs.append(cost)
        self._update_totals()

    def _update_totals(self):
        """Recalculate total costs"""
        all_costs = self.analyst_costs + self.reviewer_costs + self.master_costs

        self.total_input_tokens = sum(c.total_input_tokens for c in all_costs)
        self.total_output_tokens = sum(c.total_output_tokens for c in all_costs)
        self.total_tokens = sum(c.total_tokens for c in all_costs)
        self.total_cost = sum(c.total_cost for c in all_costs)

    def finalize(self):
        """Mark run as complete"""
        self.end_time = datetime.now().isoformat()
        self._update_totals()

    def get_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown"""
        return {
            "run_id": self.run_id,
            "company": self.company,
            "domain": self.domain,
            "duration": self._calculate_duration(),
            "total_cost": f"${self.total_cost:.4f}",
            "total_tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_tokens
            },
            "analyst": {
                "loops": len(self.analyst_costs),
                "cost": f"${sum(c.total_cost for c in self.analyst_costs):.4f}",
                "tokens": sum(c.total_tokens for c in self.analyst_costs)
            },
            "reviewer": {
                "loops": len(self.reviewer_costs),
                "cost": f"${sum(c.total_cost for c in self.reviewer_costs):.4f}",
                "tokens": sum(c.total_tokens for c in self.reviewer_costs)
            },
            "master": {
                "calls": len(self.master_costs),
                "cost": f"${sum(c.total_cost for c in self.master_costs):.4f}",
                "tokens": sum(c.total_tokens for c in self.master_costs)
            }
        }

    def _calculate_duration(self) -> str:
        """Calculate run duration"""
        if not self.end_time:
            return "In progress"

        try:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            duration = (end - start).total_seconds()

            if duration < 60:
                return f"{duration:.1f}s"
            elif duration < 3600:
                return f"{duration/60:.1f}m"
            else:
                return f"{duration/3600:.1f}h"
        except:
            return "Unknown"


class CostTracker:
    """
    Central cost tracker for all LLM usage in the system.

    Usage:
        tracker = CostTracker()
        run_cost = tracker.start_run("run_123", "AAPL", "liquidity")

        # Track analyst loop 1
        analyst_cost = AgentCost("liquidity_analyst", "analyst", loop_number=1)
        usage = extract_token_usage_from_agno(response, "openai/gpt-4o-mini")
        analyst_cost.add_usage(usage)
        run_cost.add_analyst_cost(analyst_cost)

        # Get breakdown
        breakdown = run_cost.get_breakdown()
    """

    def __init__(self):
        self.runs: Dict[str, RunCost] = {}
        self.logger = logging.getLogger(__name__)

    def start_run(self, run_id: str, company: str, domain: str) -> RunCost:
        """Start tracking a new analysis run"""
        run_cost = RunCost(run_id=run_id, company=company, domain=domain)
        self.runs[run_id] = run_cost
        self.logger.info(f"Started cost tracking for {run_id}")
        return run_cost

    def get_run(self, run_id: str) -> Optional[RunCost]:
        """Get run cost tracker"""
        return self.runs.get(run_id)

    def finalize_run(self, run_id: str) -> Optional[RunCost]:
        """Finalize a run and return cost breakdown"""
        run_cost = self.runs.get(run_id)
        if run_cost:
            run_cost.finalize()
            self.logger.info(f"Finalized cost tracking for {run_id}: ${run_cost.total_cost:.4f}")
        return run_cost

    def get_total_cost(self) -> float:
        """Get total cost across all runs"""
        return sum(run.total_cost for run in self.runs.values())

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked runs"""
        return {
            "total_runs": len(self.runs),
            "total_cost": f"${self.get_total_cost():.4f}",
            "total_tokens": sum(run.total_tokens for run in self.runs.values()),
            "runs": [run.get_breakdown() for run in self.runs.values()]
        }


# Utility Functions

def get_model_pricing(model_id: str) -> Dict[str, float]:
    """
    Get pricing for a model.

    Args:
        model_id: Model identifier (e.g., "openai/gpt-4o-mini")

    Returns:
        Dictionary with 'input' and 'output' prices per 1M tokens
    """
    pricing = LLM_PRICING.get(model_id)
    if pricing:
        return pricing

    # Try without provider prefix
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
    for key in LLM_PRICING:
        if model_name in key:
            return LLM_PRICING[key]

    logger.warning(f"No pricing found for model '{model_id}', using default")
    return LLM_PRICING["default"]


def calculate_cost(input_tokens: int, output_tokens: int, model_id: str) -> float:
    """
    Calculate cost for a given token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_id: Model identifier

    Returns:
        Cost in USD
    """
    pricing = get_model_pricing(model_id)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def extract_token_usage_from_agno(response: Any, model_id: str) -> TokenUsage:
    """
    Extract token usage from Agno agent response.

    Agno stores metrics in response.metrics as a Metrics object.

    Args:
        response: Agno agent response object (RunOutput)
        model_id: Model identifier used

    Returns:
        TokenUsage object
    """
    try:
        # Try to extract from various possible locations in Agno response
        input_tokens = 0
        output_tokens = 0

        # Method 1: Check response.metrics (Agno RunOutput has a Metrics object)
        if hasattr(response, 'metrics') and response.metrics is not None:
            metrics = response.metrics

            # Agno's Metrics object has attributes (not a dict)
            if hasattr(metrics, 'input_tokens'):
                input_tokens = metrics.input_tokens or 0
            if hasattr(metrics, 'output_tokens'):
                output_tokens = metrics.output_tokens or 0

            # Fallback to dict-style access if attributes don't exist
            if input_tokens == 0 and output_tokens == 0 and isinstance(metrics, dict):
                input_tokens = metrics.get('input_tokens', 0) or metrics.get('prompt_tokens', 0)
                output_tokens = metrics.get('output_tokens', 0) or metrics.get('completion_tokens', 0)

        # Method 2: Check response.usage (for compatibility)
        if hasattr(response, 'usage') and response.usage is not None and (input_tokens == 0 and output_tokens == 0):
            usage = response.usage
            if isinstance(usage, dict):
                input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
                output_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)

        total_tokens = input_tokens + output_tokens

        if total_tokens == 0:
            logger.debug(f"No token usage found in response for model {model_id}")
        else:
            logger.debug(f"Extracted token usage: {input_tokens} input + {output_tokens} output = {total_tokens} total")

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model_id=model_id
        )

    except Exception as e:
        logger.warning(f"Failed to extract token usage from Agno response: {e}")
        return TokenUsage(model_id=model_id)


def format_cost_summary(run_cost: RunCost) -> str:
    """
    Format cost summary as a readable string.

    Args:
        run_cost: RunCost object

    Returns:
        Formatted string
    """
    breakdown = run_cost.get_breakdown()

    lines = []
    lines.append("=" * 80)
    lines.append("💰 LLM COST BREAKDOWN")
    lines.append("=" * 80)
    lines.append(f"Run ID: {breakdown['run_id']}")
    lines.append(f"Company: {breakdown['company']} | Domain: {breakdown['domain']}")
    lines.append(f"Duration: {breakdown['duration']}")
    lines.append("")
    lines.append(f"Total Cost: {breakdown['total_cost']}")
    lines.append(f"Total Tokens: {breakdown['total_tokens']['total']:,} " +
                f"(Input: {breakdown['total_tokens']['input']:,}, Output: {breakdown['total_tokens']['output']:,})")
    lines.append("")
    lines.append("By Agent:")
    lines.append(f"  📊 Analyst: {breakdown['analyst']['cost']} " +
                f"({breakdown['analyst']['loops']} loops, {breakdown['analyst']['tokens']:,} tokens)")
    lines.append(f"  ✅ Reviewer: {breakdown['reviewer']['cost']} " +
                f"({breakdown['reviewer']['loops']} loops, {breakdown['reviewer']['tokens']:,} tokens)")
    lines.append(f"  🎯 Master: {breakdown['master']['cost']} " +
                f"({breakdown['master']['calls']} calls, {breakdown['master']['tokens']:,} tokens)")
    lines.append("=" * 80)

    return "\n".join(lines)


# Global cost tracker instance
_global_tracker = None

def get_global_tracker() -> CostTracker:
    """Get or create global cost tracker instance"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker
