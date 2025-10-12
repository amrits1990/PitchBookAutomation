"""
AgentSystemV2 Configuration - Simplified Domain Agent System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Model configuration for different agent types
ANALYST_MODEL_ID = os.getenv("ANALYST_MODEL", "x-ai/grok-4-fast")  # Cost-optimized
REVIEWER_MODEL_ID = os.getenv("REVIEWER_MODEL", "openai/gpt-4o-mini")  # Cost-optimized  
MASTER_MODEL_ID = os.getenv("MASTER_MODEL", "openai/gpt-4o-mini")  # Strong reasoning
WRITER_MODEL_ID = os.getenv("WRITER_MODEL", "openai/gpt-4o-mini")  # Quality output

# Default fallback model
DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL", "openai/gpt-4o-mini")

# Token limits for model responses (max_tokens/max_completion_tokens)
ANALYST_MAX_TOKENS = int(os.getenv("ANALYST_MAX_TOKENS", "8000"))
REVIEWER_MAX_TOKENS = int(os.getenv("REVIEWER_MAX_TOKENS", "8000"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "8000"))

# Cost limits (USD)
MAX_COST_PER_ANALYSIS = float(os.getenv("MAX_COST_PER_ANALYSIS", "3.0"))
MAX_ANALYST_COST = float(os.getenv("MAX_ANALYST_COST", "1.0"))
MAX_REVIEWER_COST = float(os.getenv("MAX_REVIEWER_COST", "0.5"))
MAX_MASTER_COST = float(os.getenv("MAX_MASTER_COST", "1.0"))
MAX_WRITER_COST = float(os.getenv("MAX_WRITER_COST", "0.5"))

# Loop configuration
DEFAULT_MAX_LOOPS = int(os.getenv("DEFAULT_MAX_LOOPS", "3"))
CHAT_MAX_LOOPS = int(os.getenv("CHAT_MAX_LOOPS", "2"))  # Reduced for chat refinements
MIN_LOOPS = 1
MAX_LOOPS_HARD_LIMIT = 5

# Chat configuration
CHAT_AGENT_MODEL = os.getenv("CHAT_AGENT_MODEL", "openai/gpt-4o-mini")  # Cost-effective chat model
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "2000"))  # Shorter responses for chat
MAX_CHAT_EXCHANGES = int(os.getenv("MAX_CHAT_EXCHANGES", "20"))  # Max exchanges per session
MAX_CHAT_REFINEMENTS = int(os.getenv("MAX_CHAT_REFINEMENTS", "3"))  # Max expensive refinements
CHAT_HISTORY_WINDOW = int(os.getenv("CHAT_HISTORY_WINDOW", "10"))  # Exchanges to keep in context
CHAT_CONTEXT_MAX_TOKENS = int(os.getenv("CHAT_CONTEXT_MAX_TOKENS", "2000"))  # Max tokens for report context
MAX_CHAT_COST_PER_SESSION = float(os.getenv("MAX_CHAT_COST_PER_SESSION", "0.50"))  # USD limit per session

# Budget and quality controls
ENABLE_COST_TRACKING = os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true"
ENABLE_BUDGET_GUARDS = os.getenv("ENABLE_BUDGET_GUARDS", "true").lower() == "true"
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.8"))

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG_MODE = os.getenv("DEBUG", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# External RAG system paths (relative to PitchBookGenerator)
RAG_BASE_PATH = "../"
SEC_FINANCIAL_RAG_PATH = RAG_BASE_PATH + "SECFinancialRAG"
ANNUAL_REPORT_RAG_PATH = RAG_BASE_PATH + "AnnualReportRAG"
TRANSCRIPT_RAG_PATH = RAG_BASE_PATH + "TranscriptRAG"
NEWS_RAG_PATH = RAG_BASE_PATH + "NewsRAG" 
SHARE_PRICE_RAG_PATH = RAG_BASE_PATH + "SharePriceRAG"

# Domain configuration
AVAILABLE_DOMAINS = ["liquidity", "leverage", "working_capital", "operating_efficiency", "valuation"]
DEFAULT_DOMAIN = "liquidity"

# Schema optimization settings
USE_SCHEMA_REFERENCES = True
TOKEN_OPTIMIZATION_ENABLED = True
COMPRESS_PROMPTS = True