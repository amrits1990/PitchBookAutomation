# AgentSystemV2 - Simplified Domain Agent System

A lightweight, cost-optimized financial analysis system with configurable analyst-reviewer loops, multi-domain coordination, and interactive chat refinements.

## 🎯 Key Features

### Core Capabilities
- **Domain-Specific Analysis**: Liquidity, leverage, working capital, operating efficiency, and valuation analysis
- **Analyst-Reviewer Loops**: Configurable quality control with automated feedback cycles
- **Multi-Domain Coordination**: Parallel or sequential analysis across multiple domains
- **Interactive Chat**: Post-analysis refinements and follow-up questions
- **Cost Optimization**: Schema references, agno memory, and strategic model routing

### Technical Architecture
- **Agno Framework**: Built on agno agents with memory management
- **Modular Design**: Cleanly separated concerns with factory pattern
- **Schema-Based**: Token-optimized prompts with structured I/O
- **Async/Parallel**: High-performance concurrent execution
- **Error Resilient**: Graceful degradation and fallback mechanisms

## 🚀 Quick Start

### Installation

```bash
# Clone or copy AgentSystemV2 directory
cd AgentSystemV2

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional - Model Configuration
ANALYST_MODEL=openai/gpt-4o-mini
REVIEWER_MODEL=openai/gpt-4o-mini
MASTER_MODEL=openai/gpt-4o

# Optional - Cost Controls
MAX_COST_PER_ANALYSIS=3.0
DEFAULT_MAX_LOOPS=3
CHAT_MAX_LOOPS=2
```

### Basic Usage

```python
import asyncio
from main import AgentSystemV2

async def main():
    # Initialize system
    system = AgentSystemV2()
    
    # Single domain analysis
    result = await system.analyze_domain(
        company="AAPL",
        domain="liquidity",
        peers=["MSFT", "GOOGL"],
        user_focus="cash management trends",
        max_loops=3
    )
    
    # Multi-domain analysis
    comprehensive = await system.analyze_comprehensive(
        company="AAPL",
        domains=["liquidity", "leverage", "operating_efficiency"],
        run_parallel=True
    )
    
    # Interactive chat
    chat_session = await system.start_chat_session(result["run_id"])
    response = await system.chat_with_analysis(
        session_id=chat_session["session_id"],
        user_message="What are the main liquidity risks?"
    )

asyncio.run(main())
```

## 📊 Available Domains

### 1. Liquidity Analysis
- **Focus**: Short-term obligations and cash flow management
- **Metrics**: Cash, current assets/liabilities, working capital components
- **Ratios**: Current ratio, quick ratio, cash conversion cycle
- **Use Case**: Cash runway analysis, working capital efficiency

### 2. Leverage Analysis
- **Focus**: Capital structure and financial risk assessment
- **Metrics**: Debt structure, interest expense, equity, assets
- **Ratios**: Debt-to-equity, interest coverage, financial leverage
- **Use Case**: Credit analysis, debt capacity evaluation

### 3. Working Capital Analysis
- **Focus**: Working capital efficiency and cash conversion
- **Metrics**: Receivables, inventory, payables, revenue, COGS
- **Ratios**: Working capital turnover, DSO, DIO, DPO
- **Use Case**: Operational efficiency, seasonal pattern analysis

### 4. Operating Efficiency Analysis
- **Focus**: Operational performance and margin analysis
- **Metrics**: Revenue, costs, operating income, assets
- **Ratios**: Margin analysis, asset turnover, productivity metrics
- **Use Case**: Profitability analysis, operational benchmarking

### 5. Valuation Analysis
- **Focus**: Investment attractiveness and valuation metrics
- **Metrics**: Market cap, enterprise value, earnings, cash flow
- **Ratios**: P/E, P/B, EV/EBITDA, PEG ratio
- **Use Case**: Investment decisions, relative valuation

## 🔄 Analysis Flow

### Single Domain Flow
```
User Request → Data Fetching → Domain Agent Creation → Analyst-Reviewer Loop → Final Output → Chat Available
                    ↓
              Supply upfront data → Analyst analyzes → Reviewer evaluates → Loop until approved
```

### Multi-Domain Flow
```
User Request → Master Agent Planning → Parallel/Sequential Domain Execution → Synthesis → Unified Report
                    ↓
              Domain 1, Domain 2, Domain 3... → Cross-domain insights → Master recommendations
```

### Chat Refinement Flow
```
Analysis Complete → Chat Session Start → User Questions → Refinement Analysis → Updated Response
                         ↓
              Simple Q&A OR Deep Refinement (analyst-reviewer loop with reduced cycles)
```

## ⚙️ Configuration

### Loop Configuration
- **Default Max Loops**: 3 (configurable per analysis)
- **Chat Max Loops**: 2 (reduced for efficiency)
- **Hard Limit**: 5 loops maximum
- **Quality Threshold**: 0.8 for approval

### Cost Optimization
- **Schema References**: ~40% token reduction
- **Model Routing**: Cheap models for bulk work, strong models for quality
- **Memory Management**: Agno memory for context retention
- **Targeted Tool Calls**: Strategic RAG system usage

### Quality Controls
- **Local Evaluator**: Rules-based validation (no LLM cost)
- **Reviewer Agent**: Quality assessment and feedback
- **Confidence Scoring**: Evidence-based confidence levels
- **Cross-Domain Validation**: Multi-domain consistency checks

## 📁 Directory Structure

```
AgentSystemV2/
├── config/                 # Configuration and schemas
│   ├── settings.py         # Environment and model configuration
│   ├── schemas.py          # Data schemas and validation
│   └── domain_configs.py   # Domain-specific configurations
├── agents/                 # Agent implementations
│   ├── base_agent.py       # Base agent class
│   ├── analyst_agent.py    # Analysis execution agent
│   ├── reviewer_agent.py   # Quality review agent
│   └── domain_agent.py     # Domain coordinator (analyst + reviewer)
├── orchestration/          # System coordination
│   ├── master_agent.py     # Master coordinator for multi-domain
│   ├── agent_factory.py    # Agent creation and management
│   └── data_fetcher.py     # Upfront data fetching
├── tools/                  # Supporting tools
│   └── chat_interface.py   # Interactive chat system
├── examples/               # Usage examples
│   └── comprehensive_demo.py
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🎯 Usage Patterns

### Investment Analysis
```python
# Comprehensive investment evaluation
result = await system.analyze_comprehensive(
    company="AAPL",
    domains=["liquidity", "leverage", "valuation"],
    user_focus="institutional investment decision",
    run_parallel=True
)
```

### Credit Analysis
```python
# Focus on creditworthiness
result = await system.analyze_domain(
    company="AAPL",
    domain="leverage",
    peers=["MSFT", "GOOGL"],
    user_focus="credit risk assessment and debt capacity",
    max_loops=3
)
```

### Operational Review
```python
# Operational efficiency deep dive
result = await system.analyze_domain(
    company="AAPL",
    domain="operating_efficiency",
    user_focus="margin improvement opportunities",
    max_loops=2
)
```

### Interactive Analysis
```python
# Analysis with follow-up refinements
result = await system.analyze_domain(company="AAPL", domain="liquidity")
chat = await system.start_chat_session(result["run_id"])

# Ask follow-up questions
response1 = await system.chat_with_analysis(
    session_id=chat["session_id"],
    user_message="What specific liquidity risks should we monitor?"
)

response2 = await system.chat_with_analysis(
    session_id=chat["session_id"], 
    user_message="Provide deeper analysis on working capital trends",
    refinement_type="deeper_analysis"
)
```

## 🛠️ Development

### Running Examples
```bash
# Run comprehensive demo
python examples/comprehensive_demo.py

# Run main CLI interface
python main.py
```

### Testing
```bash
# Run tests (when available)
pytest tests/

# Run specific domain test
python -m pytest tests/test_domain_agent.py -v
```

### Customization

#### Adding New Domains
1. Define domain configuration in `config/domain_configs.py`
2. Add required metrics and ratios
3. Define analyst prompt template
4. Set reviewer criteria
5. Update available domains list

#### Custom Analysis Focus
```python
# Custom focus for specific use case
result = await system.analyze_domain(
    company="AAPL",
    domain="liquidity",
    user_focus="acquisition financing - cash available for M&A activity",
    max_loops=3
)
```

## 📊 Performance & Cost

### Expected Performance
- **Single Domain**: 30-60 seconds
- **Multi-Domain (3 domains)**: 45-90 seconds (parallel)
- **Chat Response**: 5-15 seconds
- **Cost per Analysis**: $0.50-$2.00 (depending on complexity)

### Cost Optimization Features
- Schema references reduce tokens by ~40%
- Model routing optimizes cost/quality tradeoff
- Local evaluator provides free quality checks
- Agno memory reduces context repetition
- Configurable loop limits control maximum cost

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies installed
2. **API Key Issues**: Check OPENROUTER_API_KEY in environment
3. **Memory Warnings**: Normal with agno - system works correctly
4. **Analysis Failures**: Check logs for specific error details

### Debug Mode
```python
# Enable detailed logging
system = AgentSystemV2(enable_debug=True)
```

### System Status
```python
# Check system health
status = system.get_system_status()
print(status)
```

## 🚧 Limitations

- External RAG systems integration requires manual setup
- Simulated financial data used when external systems unavailable
- Chat refinements limited to 3 per session for cost control
- Multi-domain synthesis quality depends on individual domain success

## 🔮 Future Enhancements

- Real-time financial data integration
- Advanced cross-domain dependency modeling
- Custom report generation and formatting
- Enhanced cost tracking and budgeting
- Performance analytics and optimization insights

## 📞 Support

For questions, issues, or contributions:
1. Check the examples directory for usage patterns
2. Review configuration options in settings.py
3. Enable debug mode for detailed logging
4. Refer to the simplified agent system plan for architecture details

---

**AgentSystemV2** - Built for efficiency, designed for scalability, optimized for cost-effectiveness.