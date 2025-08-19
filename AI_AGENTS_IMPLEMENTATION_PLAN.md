# AI Financial Analysis Agent System - Implementation Plan

## 🎯 Project Overview

Build a multi-agent financial analysis system using LangChain + LangGraph where specialist agents analyze companies across different domains (Liquidity, Leverage, Working Capital, Valuation, Operations, Strategy). Each agent provides standalone and peer-comparative analysis with chat capabilities. The system includes a master agent for comprehensive reporting.

### Key Features
- **6 Specialist Domain Agents** + 1 Master Agent
- **Interactive Chat** with each agent (with memory)
- **Vector Database Storage** for efficient data reuse (News, Transcripts, Annual Reports)
- **FastAPI Backend** with RESTful services
- **Modular Architecture** for easy maintenance and testing
- **Comprehensive Analysis** with charts and structured outputs

### Key Change Summary (Thin Interface + Search-First)
- Agent interfaces are thin orchestration layers that index raw chunks with rich metadata and retrieve via search. No lossy pre-summarization during ingestion.
- Retrieval is hybrid (semantic + BM25) against LanceDB with metadata filters (ticker, section, form, date).
- Existing get_*_insights_for_agent remain as wrappers over index_* and search_* for convenience; core retrieval always uses search_*.

---

## 🏗️ System Architecture

```
PitchBookGenerator/
├── AgentSystem/                 # Main agent system package
│   ├── __init__.py
│   ├── config/                  # Configuration management
│   ├── vector_store/           # Vector database operations
│   ├── tools/                  # LangChain tool wrappers for RAG packages
│   ├── agents/                 # Domain specialist agents
│   ├── chat/                   # Chat management system
│   ├── workflows/              # LangGraph workflows
│   ├── api/                    # FastAPI application
│   └── utils/                  # Utility functions
```

### Thin Agent Interface Pattern
- Index: Pull cached or live data and store all chunks in vector DB with rich metadata.
- Search: Hybrid search with filters; return top-k chunks verbatim for LLM reasoning.
- Health/Status: Coverage checks, last indexed time, counts.

Standard interface per RAG package
- AnnualReportRAG
  - index_reports_for_agent(ticker: str, prefer_cached: bool = True, years_back: int = 2, filing_types: List[str] = ["10-K","10-Q"]) -> dict
  - search_report_for_agent(ticker: str, query: str, k: int = 20, filters: dict = None) -> dict
  - get_report_insights_for_agent(...) -> dict  [wrapper built on search_*]
- NewsRAG
  - index_news_for_agent(ticker: str, days_back: int = 30) -> dict
  - search_news_for_agent(ticker: str, query: str, k: int = 20, filters: dict = None) -> dict
  - get_news_for_agent(...) -> dict  [wrapper built on search_*]
- TranscriptRAG
  - index_transcripts_for_agent(ticker: str, quarters_back: int = 4) -> dict
  - search_transcripts_for_agent(ticker: str, query: str, k: int = 20, filters: dict = None) -> dict
  - get_transcript_insights_for_agent(...) -> dict  [wrapper built on search_*]
- SharePriceRAG
  - Analytical only (no vector DB). Provides get_price_analysis_for_agent and compare_with_peers.

Vector store standard schema
- id, content, embedding, ticker, company, source (e.g., 10-K/10-Q/news/transcript), section_name, filing_date/published_date/quarter, global_chunk_id, chunk_id, metadata (JSON). 

---

# 📅 Sprint-Based Implementation Plan

## Sprint 1: Foundation & RAG Package Interfaces (Week 1-2)
**Duration:** 10 days  
**Goal:** Make all RAG interfaces thin orchestration layers and finalize contracts

### 🔧 RAG Package Interface Changes (Thin Layer)

#### 1.1 SECFinancialRAG Package ✅ (No change)
- Analytical API remains the same (already tool-ready)

```python
# Already available:
from SECFinancialRAG import get_financial_data
result = get_financial_data(ticker="AAPL", include_ratios=True)
```

#### 1.2 NewsRAG Interface Adjustments
- Add standardized thin interface:
  - index_news_for_agent(ticker, days_back)
  - search_news_for_agent(ticker, query, k=20, filters)
- Keep get_news_for_agent as a wrapper that issues a set of searches to produce a high-level summary; do not truncate or alter chunk text before indexing.

#### 1.3 SharePriceRAG Package Improvements

**Issues to Fix:**
- Add trend analysis capabilities
- Create agent-friendly summary methods
- Add peer comparison utilities

**Changes Needed:**
```python
# Add to SharePriceRAG/main.py
def get_price_analysis_for_agent(ticker: str, days_back: int = 90) -> dict:
    """Agent-friendly price analysis"""
    return {
        "current_price": float,
        "price_trend": str,      # "upward", "downward", "sideways"
        "volatility": str,       # "high", "medium", "low"
        "performance_summary": str,
        "key_metrics": dict      # For agent consumption
    }

def compare_with_peers(ticker: str, peer_tickers: List[str]) -> dict:
    """Peer price comparison"""
    pass
```

#### 1.4 TranscriptRAG Interface Adjustments
- Add standardized thin interface:
  - index_transcripts_for_agent(ticker, quarters_back)
  - search_transcripts_for_agent(ticker, query, k=20, filters)
- Keep get_transcript_insights_for_agent as a wrapper built on search_*.

#### 1.5 AnnualReportRAG Interface Adjustments
- New thin interface:
  - index_reports_for_agent(ticker, prefer_cached=True, years_back=2, filing_types=["10-K","10-Q"])
  - search_report_for_agent(ticker, query, k=20, filters={form_type, section_name, date_range})
- Keep get_report_insights_for_agent and get_risk_factor_analysis as optional wrappers that run search_* with predefined queries and assemble outputs. Do not pre-trim chunks.

### 1.6 Foundation Setup Tasks

#### Directory Structure Creation
```bash
mkdir -p AgentSystem/{config,vector_store,tools,agents,chat,workflows,api,utils}
touch AgentSystem/__init__.py
# Create all subdirectory __init__.py files
```

#### Core Configuration Files
- `config/agent_config.py` - Agent definitions and personalities
- `config/model_config.py` - LLM configurations  
- `config/vector_config.py` - Vector database and embedding settings
- `config/api_config.py` - FastAPI settings

```python
# config/vector_config.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass 
class VectorConfig:
    """Configuration for vector database and embeddings"""
    
    # Embedding Model Configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_alternative: str = "sentence-transformers/all-mpnet-base-v2"  # Higher quality option
    embedding_dimension: int = 384  # For all-MiniLM-L6-v2
    
    # Database Configuration
    db_path: str = "./vector_db"
    
    # Search Configuration
    hybrid_search_enabled: bool = True
    semantic_weight: float = 0.7  # Weight for semantic similarity
    keyword_weight: float = 0.3   # Weight for keyword matching (BM25)
    
    # Coverage thresholds for cached data usage
    news_coverage_threshold: float = 80.0      # Use cached if >80% coverage
    transcript_coverage_threshold: float = 90.0 # Higher threshold for transcripts
    report_coverage_threshold: float = 95.0     # Highest for annual reports
    
    # Search result limits
    default_search_limit: int = 20
    max_search_limit: int = 100
    
    # BM25 Parameters
    bm25_k1: float = 1.5  # Term frequency saturation parameter
    bm25_b: float = 0.75  # Field length normalization parameter
    
    @classmethod
    def for_production(cls) -> 'VectorConfig':
        """Production-optimized configuration"""
        return cls(
            embedding_model_name="sentence-transformers/all-mpnet-base-v2",  # Higher quality
            embedding_dimension=768,
            semantic_weight=0.8,  # Emphasize semantic search in production
            keyword_weight=0.2
        )
    
    @classmethod  
    def for_development(cls) -> 'VectorConfig':
        """Development-optimized configuration (faster)"""
        return cls(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster
            embedding_dimension=384,
            semantic_weight=0.6,  # More balanced for testing
            keyword_weight=0.4
        )

# Embedding model comparison
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "dimensions": 384,
        "size_mb": 90,
        "speed": "fast",
        "quality": "good",
        "use_case": "development, real-time search"
    },
    "all-mpnet-base-v2": {
        "dimensions": 768, 
        "size_mb": 420,
        "speed": "medium",
        "quality": "high",
        "use_case": "production, better accuracy"
    },
    "all-distilroberta-v1": {
        "dimensions": 768,
        "size_mb": 290,
        "speed": "medium",
        "quality": "high", 
        "use_case": "balanced speed/quality"
    }
}
```

#### Dependencies Installation
```python
# requirements.txt additions
fastapi>=0.104.1
uvicorn>=0.24.0
lancedb>=0.3.4
sentence-transformers>=2.2.2
langchain>=0.1.0
langgraph>=0.0.25
pydantic>=2.0.0
python-multipart>=0.0.5
rank-bm25>=0.2.2  # For keyword search
Pillow>=10.0.0    # For future image processing in investor presentations
```

### Sprint 1 Testing Criteria
- [ ] All RAG packages have standardized tool interfaces
- [ ] Foundation directory structure is complete
- [ ] Basic configuration files are implemented
- [ ] All dependencies are installed and compatible
- [ ] Each RAG package can be imported and called successfully

---

## Sprint 2: Vector Database & Hybrid Search (Week 2-3)
**Duration:** 7 days  
**Goal:** Implement vector storage and hybrid search; wire AnnualReportRAG end-to-end

### 🎯 Vector Database Strategy: LanceDB + Hybrid Search

#### Why LanceDB over ChromaDB?

**LanceDB Advantages:**
- **Native Multimodal Support**: Perfect for future investor presentation RAG with images
- **Superior Performance**: Columnar storage format optimized for large-scale data
- **Hybrid Search Support**: Easy to implement semantic + keyword hybrid via reranking and metadata filtering
- **Better Scalability**: Handles production workloads more efficiently
- **Advanced Filtering**: Better support for complex metadata queries

**Embedding Model Choice: HuggingFace sentence-transformers**
- **all-MiniLM-L6-v2** (Primary): Fast, 384-dim, good for development
- **all-mpnet-base-v2** (Production): Higher quality, 768-dim, better accuracy
- **Free and Open Source**: No API costs, runs locally
- **Financial Domain Ready**: Pre-trained on diverse text, works well for financial documents

**Hybrid Search Benefits:**
- **Semantic Search**: Captures contextual meaning (e.g., "liquidity crisis" matches "cash flow problems")  
- **Keyword Search**: Exact matches for specific terms (e.g., "current ratio", "EBITDA")
- **BM25 Re-ranking**: Proven algorithm for keyword relevance
- **Configurable Weights**: Adjust semantic vs keyword importance per use case

**Future Multimodal Capabilities:**
```python
# Future investor presentation support
class InvestorPresentationStore(BaseVectorStore):
    def __init__(self):
        super().__init__("investor_presentation_data")
        
        # Text embeddings
        self.text_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Image embeddings (future)
        # self.image_model = SentenceTransformer('clip-ViT-B-32')
        
    def store_presentation_slides(self, slides: List[Dict]):
        """Store both slide text and images with cross-modal search"""
        # LanceDB natively supports multimodal embeddings
        # Can search "revenue growth chart" and find relevant slides
        pass
```

### 2.1 Vector Database Implementation

#### LanceDB Setup with Hybrid Search
```python
# vector_store/base_store.py
from abc import ABC, abstractmethod
import lancedb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import numpy as np

class BaseVectorStore(ABC):
    """Abstract base class for vector storage with hybrid search capabilities"""
    
    def __init__(self, collection_name: str):
        # Initialize LanceDB
        self.db = lancedb.connect("./vector_db")
        self.collection_name = collection_name
        
        # Initialize embedding model (HuggingFace)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        # Initialize BM25 for keyword search
        self.bm25 = None
        self.documents = []
        
        self._create_table_if_not_exists()
    
    def _create_table_if_not_exists(self):
        """Create LanceDB table with schema"""
        try:
            self.table = self.db.open_table(self.collection_name)
        except FileNotFoundError:
            # Create new table with schema
            schema = [
                {"name": "id", "type": "string"},
                {"name": "content", "type": "string"},
                {"name": "embedding", "type": f"array({self.embedding_dim})"},
                {"name": "company", "type": "string"},
                {"name": "date", "type": "string"},
                {"name": "source", "type": "string"},
                {"name": "category", "type": "string"},
                {"name": "metadata", "type": "string"}  # JSON string
            ]
            
            # Create with empty data
            import pandas as pd
            empty_df = pd.DataFrame({col["name"]: [] for col in schema})
            empty_df["embedding"] = empty_df["embedding"].astype(object)
            
            self.table = self.db.create_table(self.collection_name, empty_df)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using HuggingFace model"""
        return self.embedding_model.encode(text).tolist()
    
    def _update_bm25_index(self):
        """Update BM25 index with current documents"""
        if self.documents:
            tokenized_docs = [doc.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
    
    @abstractmethod
    def store_data(self, data: List[Dict], metadata: Dict) -> bool:
        """Store data with metadata"""
        pass
    
    def hybrid_search(self, query: str, company: str = None, limit: int = 10, 
                     semantic_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict]:
        """
        Hybrid search combining semantic similarity and keyword matching
        
        Args:
            query: Search query
            company: Optional company filter
            limit: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        """
        
        # Semantic search using LanceDB
        query_embedding = self._generate_embedding(query)
        
        # Build filter conditions
        filter_conditions = []
        if company:
            filter_conditions.append(f"company = '{company}'")
        
        where_clause = " AND ".join(filter_conditions) if filter_conditions else None
        
        semantic_results = self.table.search(query_embedding) \
            .where(where_clause) \
            .limit(limit * 2) \
            .to_pandas()
        
        # Keyword search using BM25
        if self.bm25:
            query_tokens = query.split()
            keyword_scores = self.bm25.get_scores(query_tokens)
            
            # Combine scores and re-rank
            combined_results = self._combine_and_rerank(
                semantic_results, keyword_scores, 
                semantic_weight, keyword_weight
            )
        else:
            combined_results = semantic_results
        
        return combined_results.head(limit).to_dict('records')
    
    def _combine_and_rerank(self, semantic_results, keyword_scores, 
                          semantic_weight, keyword_weight):
        """Combine semantic and keyword scores for re-ranking"""
        
        # Normalize semantic scores (distance to similarity)
        semantic_results['semantic_score'] = 1 / (1 + semantic_results['_distance'])
        
        # Normalize keyword scores
        max_keyword = max(keyword_scores) if keyword_scores else 1
        normalized_keyword = [score / max_keyword for score in keyword_scores]
        
        # Add keyword scores to results (matching by index)
        if len(normalized_keyword) >= len(semantic_results):
            semantic_results['keyword_score'] = normalized_keyword[:len(semantic_results)]
        else:
            semantic_results['keyword_score'] = 0
        
        # Calculate combined score
        semantic_results['combined_score'] = (
            semantic_weight * semantic_results['semantic_score'] + 
            keyword_weight * semantic_results['keyword_score']
        )
        
        # Sort by combined score
        return semantic_results.sort_values('combined_score', ascending=False)
    
    @abstractmethod  
    def query_with_overlap(self, company: str, start_date: date, end_date: date) -> Dict:
        """Query with date overlap detection"""
        pass
    
    def _analyze_coverage(self, existing_data: Dict, start_date: date, end_date: date) -> Dict:
        """Analyze data coverage percentage"""
        # Implementation for coverage analysis
        pass
```

#### Specific Vector Stores
```python
# vector_store/news_store.py
import pandas as pd
import json
from datetime import datetime, timedelta

class NewsVectorStore(BaseVectorStore):
    def __init__(self):
        super().__init__("news_data")
    
    def store_news_data(self, news_chunks: List[Dict], company: str, date_range: Dict):
        """Store news chunks with company/date metadata and embeddings"""
        
        data_to_store = []
        documents_for_bm25 = []
        
        for chunk in news_chunks:
            content = chunk.get("content", "")
            embedding = self._generate_embedding(content)
            
            data_to_store.append({
                "id": chunk.get("chunk_id", f"{company}_{datetime.now().timestamp()}"),
                "content": content,
                "embedding": embedding,
                "company": company.upper(),
                "date": chunk.get("date", datetime.now().isoformat()),
                "source": chunk.get("source", "unknown"),
                "category": chunk.get("category", "general"),
                "metadata": json.dumps({
                    "title": chunk.get("title", ""),
                    "url": chunk.get("url", ""),
                    "sentiment": chunk.get("sentiment", "neutral")
                })
            })
            
            documents_for_bm25.append(content)
        
        # Store in LanceDB
        if data_to_store:
            df = pd.DataFrame(data_to_store)
            self.table.add(df)
            
            # Update BM25 index
            self.documents.extend(documents_for_bm25)
            self._update_bm25_index()
        
        return True
    
    def get_cached_news(self, company: str, days_back: int) -> Optional[Dict]:
        """Retrieve cached news with coverage analysis"""
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Query recent news for company
        results = self.table.search() \
            .where(f"company = '{company.upper()}' AND date >= '{cutoff_date.isoformat()}'") \
            .limit(1000) \
            .to_pandas()
        
        if len(results) == 0:
            return None
        
        # Analyze coverage
        total_days = days_back
        unique_dates = set(pd.to_datetime(results['date']).dt.date)
        coverage_percentage = (len(unique_dates) / total_days) * 100
        
        return {
            "has_data": True,
            "coverage_percentage": coverage_percentage,
            "total_chunks": len(results),
            "date_range": {
                "start": results['date'].min(),
                "end": results['date'].max()
            },
            "chunks": results.to_dict('records')
        }
    
    def search_news(self, query: str, company: str = None, days_back: int = 30) -> List[Dict]:
        """Hybrid search for news with company and date filters"""
        
        # Add date filter to hybrid search
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            # Combine with existing company filter logic in hybrid_search
        
        return self.hybrid_search(query, company=company, limit=20)

# vector_store/transcript_store.py  
class TranscriptVectorStore(BaseVectorStore):
    def __init__(self):
        super().__init__("transcript_data")
    
    def store_transcript_data(self, transcript_chunks: List[Dict], company: str, quarter: str):
        """Store transcript chunks with speaker and quarter metadata"""
        
        data_to_store = []
        documents_for_bm25 = []
        
        for chunk in transcript_chunks:
            content = chunk.get("content", "")
            embedding = self._generate_embedding(content)
            
            data_to_store.append({
                "id": chunk.get("chunk_id", f"{company}_{quarter}_{len(data_to_store)}"),
                "content": content,
                "embedding": embedding,
                "company": company.upper(),
                "date": chunk.get("date", ""),
                "source": "earnings_call",
                "category": chunk.get("section", "unknown"),  # "management", "qa", etc.
                "metadata": json.dumps({
                    "quarter": quarter,
                    "speaker": chunk.get("speaker", ""),
                    "speaker_role": chunk.get("speaker_role", ""),
                    "section": chunk.get("section", "")
                })
            })
            
            documents_for_bm25.append(content)
        
        if data_to_store:
            df = pd.DataFrame(data_to_store)
            self.table.add(df)
            
            self.documents.extend(documents_for_bm25)
            self._update_bm25_index()
        
        return True

# vector_store/report_store.py
class ReportVectorStore(BaseVectorStore):
    def __init__(self):
        super().__init__("report_data")
    
    def store_report_data(self, report_chunks: List[Dict], company: str, report_type: str, filing_date: str):
        """Store annual report chunks with section metadata"""
        
        data_to_store = []
        documents_for_bm25 = []
        
        for chunk in report_chunks:
            content = chunk.get("content", "")
            embedding = self._generate_embedding(content)
            
            data_to_store.append({
                "id": chunk.get("chunk_id", f"{company}_{report_type}_{len(data_to_store)}"),
                "content": content,
                "embedding": embedding,
                "company": company.upper(),
                "date": filing_date,
                "source": report_type,  # "10-K", "10-Q", etc.
                "category": chunk.get("section", "unknown"),
                "metadata": json.dumps({
                    "section_title": chunk.get("section_title", ""),
                    "page_number": chunk.get("page_number", ""),
                    "report_type": report_type,
                    "filing_date": filing_date
                })
            })
            
            documents_for_bm25.append(content)
        
        if data_to_store:
            df = pd.DataFrame(data_to_store)
            self.table.add(df)
            
            self.documents.extend(documents_for_bm25)
            self._update_bm25_index()
        
        return True

# vector_store/investor_presentation_store.py (Future enhancement)
class InvestorPresentationStore(BaseVectorStore):
    """Future: Multimodal store for investor presentations with images"""
    
    def __init__(self):
        super().__init__("investor_presentation_data")
        # Will include image embedding models for slides
        # LanceDB handles multimodal data natively
        
    def store_presentation_data(self, presentation_data: List[Dict], company: str):
        """Store both text and image data from investor presentations"""
        # Future implementation for multimodal embeddings
        pass
```

### 2.2 LangChain Tool Wrappers

#### Base Tool Class
```python
# tools/base_tool.py
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

class BaseFinancialTool(BaseTool, ABC):
    """Abstract base for financial analysis tools"""
    
    def __init__(self):
        super().__init__()
        self._setup_tool()
    
    @abstractmethod
    def _setup_tool(self):
        """Setup tool-specific configurations"""
        pass
    
    def _handle_errors(self, func, *args, **kwargs):
        """Standardized error handling for all tools"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    # Added: common formatting helpers used by tool snippets
    def _format_for_agent(self, result: dict) -> str:
        if not isinstance(result, dict):
            return str(result)
        if not result.get("success", True):
            return f"Error: {result.get('error') or result.get('message', 'Unknown error')}"

        # For successful results, return a structured summary
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def _format_cached_results(self, cached: dict) -> str:
        payload = {
            "success": True,
            "source": "cache",
            "coverage": cached.get("coverage_percentage"),
            "total_chunks": cached.get("total_chunks"),
            "date_range": cached.get("date_range"),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
```

#### Specific Tool Implementations
```python
# tools/financial_tool.py
class FinancialDataTool(BaseFinancialTool):
    name = "financial_data"
    description = """Get comprehensive financial data including income statements, 
    balance sheets, cash flow, and calculated ratios for liquidity, leverage, 
    and profitability analysis."""
    
    def _run(self, ticker: str, include_ratios: bool = True) -> str:
        from SECFinancialRAG import get_financial_data
        result = self._handle_errors(get_financial_data, ticker=ticker, include_ratios=include_ratios)
        return self._format_for_agent(result)

# tools/news_tool.py
class NewsAnalysisTool(BaseFinancialTool):
    name = "news_analysis"
    description = """Get recent news and sentiment analysis for a company. 
    Checks local vector database first to avoid API rate limits."""
    
    def __init__(self):
        super().__init__()
        self.vector_store = NewsVectorStore()
    
    def _run(self, ticker: str, days_back: int = 30) -> str:
        # Check vector store first
        cached_data = self.vector_store.get_cached_news(ticker, days_back)
        
        if cached_data and cached_data["coverage_percentage"] > 80:
            return self._format_cached_results(cached_data)
        
        # Fetch new data if needed
        from NewsRAG import get_company_news_chunks
        result = self._handle_errors(
            get_company_news_chunks, 
            companies=[ticker], 
            days_back=days_back
        )
        
        if result["success"]:
            # Store in vector database
            self.vector_store.store_news_data(
                result["chunks"], ticker, 
                {"days_back": days_back}
            )
        
        return self._format_for_agent(result)
```

### 2.3 Query Manager for Smart Data Retrieval
```python
# vector_store/query_manager.py
class QueryManager:
    """Manages intelligent querying across vector stores"""
    
    def __init__(self):
        self.news_store = NewsVectorStore()
        self.transcript_store = TranscriptVectorStore() 
        self.report_store = ReportVectorStore()
    
    def get_comprehensive_data(self, ticker: str, analysis_type: str) -> Dict:
        """Get data from multiple sources based on analysis type"""
        pass
    
    def check_data_freshness(self, ticker: str) -> Dict:
        """Check freshness of data across all stores"""
        pass
```

### Sprint 2 Testing Criteria
- [ ] Vector database stores and retrieves data correctly
- [ ] Date overlap detection works accurately  
- [ ] All 5 tool wrappers are implemented and functional
- [ ] Tools can handle errors gracefully
- [ ] Vector storage prevents redundant API calls
- [ ] Query manager coordinates data retrieval effectively

---

## Sprint 3: Domain Specialist Agents (Week 3-4)  
**Duration:** 7 days  
**Goal:** Implement 6 specialist agents with domain expertise

### 3.1 Base Agent with Chat Capability

```python
# agents/base_agent.py
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from abc import ABC, abstractmethod

class BaseAgentWithChat(ABC):
    """Base class for all domain specialist agents"""
    
    def __init__(self, domain: str, tools: List[BaseTool]):
        self.domain = domain
        self.tools = tools
        self.agent_executor = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Initialize LangChain agent with tools"""
        system_prompt = self._get_system_prompt()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create agent with tools
        agent = create_tool_calling_agent(
            llm=self._get_llm(),
            tools=self.tools, 
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5
        )
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Domain-specific system prompt"""
        pass
    
    @abstractmethod  
    def analyze_company(self, ticker: str, peers: List[str] = None) -> Dict:
        """Perform structured domain analysis"""
        pass
    
    def chat(self, message: str, context: Dict = None) -> str:
        """Interactive chat with domain expertise"""
        enhanced_input = self._enhance_with_context(message, context)
        
        response = self.agent_executor.invoke({
            "input": enhanced_input
        })
        
        return response["output"]
    
    def _enhance_with_context(self, message: str, context: Dict) -> str:
        """Add analysis context to chat message"""
        if not context:
            return message
        
        context_str = f"\nContext from previous analysis:\n{context}\n\nUser question: {message}"
        return context_str
```

### 3.2 Specialist Agent Implementations

#### Liquidity Agent
```python
# agents/liquidity_agent.py
class LiquidityAgent(BaseAgentWithChat):
    def __init__(self):
        tools = [
            FinancialDataTool(),
            PriceDataTool(), 
            NewsAnalysisTool()
        ]
        super().__init__("liquidity", tools)
    
    def _get_system_prompt(self) -> str:
        return """You are a Senior Credit Analyst specializing in liquidity analysis.

Your expertise includes:
- Cash position and cash flow analysis  
- Working capital management assessment
- Short-term debt obligations evaluation
- Current ratio, quick ratio, cash ratio analysis
- Operating cash flow trends and patterns
- Seasonal liquidity variations

Always provide:
1. Specific numerical metrics with context
2. Trends over time (improving/deteriorating)  
3. Industry benchmark comparisons when available
4. Risk assessment and implications
5. Actionable insights for management

Use tools to get current financial data, recent news affecting liquidity, 
and stock price movements that might indicate market concerns."""

    def analyze_company(self, ticker: str, peers: List[str] = None) -> Dict:
        """Comprehensive liquidity analysis"""
        
        # Get financial data
        financial_data = self.agent_executor.invoke({
            "input": f"Get comprehensive financial data for {ticker} including liquidity ratios"
        })
        
        # Analyze trends and peer comparison
        analysis_prompt = f"""
        Analyze liquidity position for {ticker}:
        1. Calculate and interpret key liquidity ratios
        2. Assess working capital trends
        3. Evaluate cash flow adequacy
        4. Compare with peers: {peers if peers else 'industry average'}
        5. Identify liquidity risks and strengths
        
        Provide structured analysis with specific metrics and insights.
        """
        
        analysis = self.agent_executor.invoke({"input": analysis_prompt})
        
        return {
            "domain": "liquidity",
            "ticker": ticker, 
            "analysis": analysis["output"],
            "key_metrics": self._extract_key_metrics(analysis["output"]),
            "risk_level": self._assess_risk_level(analysis["output"]),
            "recommendations": self._extract_recommendations(analysis["output"])
        }
```

#### Leverage Agent  
```python
# agents/leverage_agent.py
class LeverageAgent(BaseAgentWithChat):
    def __init__(self):
        tools = [
            FinancialDataTool(),
            TranscriptAnalysisTool(),
            NewsAnalysisTool()
        ]
        super().__init__("leverage", tools)
    
    def _get_system_prompt(self) -> str:
        return """You are a Fixed Income Analyst specializing in leverage and debt analysis.

Your expertise includes:
- Debt structure and composition analysis
- Interest coverage and debt service capability  
- Leverage ratios (debt-to-equity, debt-to-assets, debt-to-EBITDA)
- Credit risk assessment
- Covenant compliance monitoring
- Refinancing risk evaluation

Always provide:
1. Detailed debt metrics with trends
2. Interest rate and maturity analysis
3. Covenant headroom assessment  
4. Industry leverage comparisons
5. Credit risk rating implications
6. Future debt capacity analysis

Focus on both quantitative metrics and qualitative factors affecting creditworthiness."""
```

#### Valuation Agent
```python  
# agents/valuation_agent.py
class ValuationAgent(BaseAgentWithChat):
    def __init__(self):
        tools = [
            FinancialDataTool(),
            PriceDataTool(),
            NewsAnalysisTool(),
            TranscriptAnalysisTool()
        ]
        super().__init__("valuation", tools)
    
    def _get_system_prompt(self) -> str:
        return """You are a Senior Equity Analyst specializing in company valuation.

Your expertise includes:
- Multiple valuation methodologies (DCF, multiples, asset-based)
- P/E, P/B, EV/EBITDA, PEG ratio analysis
- Growth rate assessment and sustainability
- Comparable company analysis
- Market sentiment and pricing efficiency
- Intrinsic value vs market value assessment

Always provide:
1. Multiple valuation approaches with ranges
2. Key assumptions and sensitivity analysis
3. Peer group comparisons with justification
4. Market timing and sentiment considerations  
5. Upside/downside scenarios
6. Investment recommendation with rationale

Consider both fundamental analysis and market dynamics in your assessments."""
```

#### Working Capital Agent
```python
# agents/working_capital_agent.py  
class WorkingCapitalAgent(BaseAgentWithChat):
    def _get_system_prompt(self) -> str:
        return """You are an Operations Finance Analyst specializing in working capital management.

Your expertise includes:
- Days Sales Outstanding (DSO) analysis
- Days Inventory Outstanding (DIO) trends  
- Days Payable Outstanding (DPO) optimization
- Cash conversion cycle evaluation
- Seasonal working capital patterns
- Supply chain finance implications

Focus on operational efficiency and cash flow optimization."""
```

#### Operational Performance Agent
```python
# agents/operational_agent.py
class OperationalAgent(BaseAgentWithChat):  
    def _get_system_prompt(self) -> str:
        return """You are a Management Consultant specializing in operational performance analysis.

Your expertise includes:
- Revenue growth and composition analysis
- Margin trends and cost structure
- Operational efficiency metrics
- Market share and competitive positioning  
- Productivity and scale economics
- Business model sustainability

Provide insights on operational strengths, improvement opportunities, and competitive advantages."""
```

#### Strategy Agent
```python
# agents/strategy_agent.py
class StrategyAgent(BaseAgentWithChat):
    def _get_system_prompt(self) -> str:
        return """You are a Strategy Consultant specializing in corporate strategy analysis.

Your expertise includes:
- Strategic direction and vision assessment
- Market positioning and competitive moats
- Growth strategy evaluation (organic vs inorganic)
- Capital allocation effectiveness
- ESG and sustainability initiatives
- Innovation and digital transformation

Focus on long-term value creation and strategic competitive advantages."""
```

### Sprint 3 Testing Criteria
- [ ] All 6 specialist agents are implemented with proper system prompts
- [ ] Each agent can access appropriate tools for their domain
- [ ] Agents can perform structured company analysis
- [ ] Chat functionality works with domain expertise  
- [ ] Agents provide domain-specific insights and metrics
- [ ] Error handling works across all agents

---

## Sprint 4: Chat System & Memory Management (Week 4-5)
**Duration:** 7 days  
**Goal:** Implement comprehensive chat system with persistent memory

### 4.1 Chat Management System

```python
# chat/chat_manager.py
from typing import Dict, List, Optional
import uuid
from datetime import datetime
import json

class ChatManager:
    """Manages chat sessions across all agents"""
    
    def __init__(self):
        self.active_sessions: Dict[str, ChatSession] = {}
        self.conversation_store = ConversationStore()
    
    def create_session(self, agent_type: str, user_id: str = None) -> str:
        """Create new chat session"""
        session_id = str(uuid.uuid4())
        
        session = ChatSession(
            session_id=session_id,
            agent_type=agent_type,
            user_id=user_id,
            created_at=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        return session_id
    
    def send_message(self, session_id: str, message: str, context: Dict = None) -> str:
        """Send message to agent and get response"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Add context if provided (from analysis results)
        if context:
            session.add_context(context)
        
        # Get agent response
        agent = self._get_agent(session.agent_type)
        response = agent.chat(message, session.get_context())
        
        # Store conversation
        session.add_exchange(message, response)
        self.conversation_store.save_exchange(session_id, message, response)
        
        return response
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get chat history for session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].get_history()
        
        return self.conversation_store.load_history(session_id)
    
    def list_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user"""
        return self.conversation_store.get_user_sessions(user_id)
```

### 4.2 Chat Session Model
```python
# chat/models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

@dataclass
class ChatExchange:
    timestamp: datetime
    user_message: str
    agent_response: str
    context_used: Optional[Dict] = None

@dataclass
class ChatSession:
    session_id: str
    agent_type: str
    user_id: Optional[str]
    created_at: datetime
    exchanges: List[ChatExchange] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_exchange(self, user_message: str, agent_response: str):
        """Add new message exchange"""
        exchange = ChatExchange(
            timestamp=datetime.now(),
            user_message=user_message,
            agent_response=agent_response,
            context_used=self.context.copy() if self.context else None
        )
        self.exchanges.append(exchange)
    
    def add_context(self, context: Dict):
        """Add analysis context to session"""
        self.context.update(context)
    
    def get_context(self) -> Dict:
        """Get current session context"""
        return self.context
    
    def get_history(self) -> List[Dict]:
        """Get formatted chat history"""
        return [
            {
                "timestamp": exchange.timestamp.isoformat(),
                "user": exchange.user_message,
                "agent": exchange.agent_response
            }
            for exchange in self.exchanges
        ]
```

### 4.3 Persistent Storage
```python
# chat/conversation_store.py
import sqlite3
import json
from datetime import datetime
from typing import List, Dict

class ConversationStore:
    """SQLite-based conversation storage"""
    
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    user_id TEXT,
                    created_at TIMESTAMP,
                    last_activity TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    user_message TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    context_data TEXT,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
                )
            """)
            
            conn.commit()
    
    def save_exchange(self, session_id: str, user_message: str, agent_response: str, context: Dict = None):
        """Save message exchange"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO chat_exchanges 
                (session_id, timestamp, user_message, agent_response, context_data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                datetime.now(),
                user_message,
                agent_response,
                json.dumps(context) if context else None
            ))
            
            # Update session last activity
            cursor.execute("""
                UPDATE chat_sessions 
                SET last_activity = ?
                WHERE session_id = ?
            """, (datetime.now(), session_id))
            
            conn.commit()
```

### 4.4 Memory Integration with Agents
```python
# agents/base_agent.py (enhancement)
class BaseAgentWithChat(ABC):
    def __init__(self, domain: str, tools: List[BaseTool]):
        self.domain = domain
        self.tools = tools
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            return_messages=True
        )
        self._setup_agent()
    
    def chat(self, message: str, context: Dict = None) -> str:
        """Enhanced chat with memory and context"""
        
        # Build enhanced input with context and memory
        enhanced_input = self._build_enhanced_input(message, context)
        
        # Get response from agent
        response = self.agent_executor.invoke({
            "input": enhanced_input,
            "chat_history": self.memory.chat_memory.messages
        })
        
        # Update memory
        self.memory.chat_memory.add_user_message(message)
        self.memory.chat_memory.add_ai_message(response["output"])
        
        return response["output"]
    
    def _build_enhanced_input(self, message: str, context: Dict) -> str:
        """Build enhanced input with context and conversation history"""
        parts = []
        
        if context:
            parts.append(f"Analysis Context:\n{self._format_context(context)}")
        
        parts.append(f"User Question: {message}")
        
        return "\n\n".join(parts)
```

### Sprint 4 Testing Criteria
- [ ] Chat sessions can be created and managed
- [ ] Message exchanges are stored persistently  
- [ ] Agents maintain conversation context
- [ ] Analysis context enhances chat responses
- [ ] Chat history can be retrieved and displayed
- [ ] Memory limits prevent context overflow

---

## Sprint 5: Master Agent & LangGraph Workflow (Week 5-6)
**Duration:** 7 days
**Goal:** Implement master orchestration agent and workflow management

### 5.1 Master Agent Implementation

```python
# agents/master_agent.py
from typing import Dict, List, Any
from langchain.schema import SystemMessage

class MasterAgent(BaseAgentWithChat):
    """Master orchestration agent for comprehensive analysis"""
    
    def __init__(self):
        tools = [
            FinancialDataTool(),
            NewsAnalysisTool(), 
            PriceDataTool(),
            TranscriptAnalysisTool(),
            ReportAnalysisTool()
        ]
        super().__init__("master", tools)
        
        # Initialize specialist agents for delegation
        self.specialist_agents = {
            "liquidity": LiquidityAgent(),
            "leverage": LeverageAgent(), 
            "valuation": ValuationAgent(),
            "working_capital": WorkingCapitalAgent(),
            "operational": OperationalAgent(),
            "strategy": StrategyAgent()
        }
    
    def _get_system_prompt(self) -> str:
        return """You are a Senior Financial Analyst and Portfolio Manager with expertise across all domains of financial analysis.

Your role is to:
1. Coordinate comprehensive company analysis across multiple domains
2. Synthesize insights from specialist analyses into cohesive narratives  
3. Provide strategic recommendations tailored to user personas (investor, banker, CFO)
4. Identify cross-domain patterns, risks, and opportunities
5. Generate executive summaries and detailed reports

You have access to all financial data tools and can delegate specific domain analysis to specialist agents when needed.

Always consider:
- User's perspective and information needs
- Interconnections between different analysis domains
- Risk-return trade-offs and investment implications
- Industry context and competitive positioning
- Timing and market environment factors

Provide clear, actionable insights with supporting evidence."""

    def conduct_comprehensive_analysis(self, ticker: str, peers: List[str], user_persona: str) -> Dict:
        """Orchestrate comprehensive analysis across all domains"""
        
        analysis_results = {}
        
        # Delegate to specialist agents
        for domain, agent in self.specialist_agents.items():
            try:
                domain_analysis = agent.analyze_company(ticker, peers)
                analysis_results[domain] = domain_analysis
            except Exception as e:
                analysis_results[domain] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        # Synthesize results
        synthesis = self._synthesize_analysis(analysis_results, ticker, peers, user_persona)
        
        return {
            "ticker": ticker,
            "peers": peers,
            "user_persona": user_persona,
            "domain_analyses": analysis_results,
            "synthesis": synthesis,
            "recommendations": self._generate_recommendations(synthesis, user_persona),
            "executive_summary": self._create_executive_summary(synthesis),
            "risk_assessment": self._assess_overall_risk(analysis_results)
        }
    
    def _synthesize_analysis(self, domain_results: Dict, ticker: str, peers: List[str], persona: str) -> str:
        """Synthesize specialist analyses into cohesive narrative"""
        
        synthesis_prompt = f"""
        Synthesize the following domain analyses for {ticker} into a comprehensive assessment:

        Domain Analyses:
        {self._format_domain_results(domain_results)}

        Peer Companies: {peers}
        Target Audience: {persona}

        Please provide:
        1. Overall financial health assessment
        2. Key strengths and competitive advantages  
        3. Primary risks and concerns
        4. Cross-domain insights and patterns
        5. Peer comparison summary
        6. Outlook and key factors to monitor

        Tailor the analysis for a {persona} perspective.
        """
        
        response = self.agent_executor.invoke({"input": synthesis_prompt})
        return response["output"]
```

### 5.2 LangGraph Workflow Implementation

```python
# workflows/analysis_workflow.py
from langgraph import StateGraph, END
from typing import Dict, Any, List
from pydantic import BaseModel

class AnalysisState(BaseModel):
    # Input parameters
    ticker: str
    peers: List[str] = []
    user_persona: str = "investor"
    
    # Analysis results  
    domain_analyses: Dict[str, Any] = {}
    synthesis: str = ""
    final_report: Dict[str, Any] = {}
    
    # Chat integration
    chat_sessions: Dict[str, str] = {}  # domain -> session_id
    
    # Status tracking
    current_step: str = "initialized"
    errors: List[str] = []
    warnings: List[str] = []

def create_analysis_workflow(chat_manager: ChatManager):
    """Create LangGraph workflow for comprehensive analysis"""
    
    workflow = StateGraph(AnalysisState)
    
    # Analysis nodes
    workflow.add_node("data_validation", validate_inputs)
    workflow.add_node("liquidity_analysis", create_liquidity_analysis_node())
    workflow.add_node("leverage_analysis", create_leverage_analysis_node()) 
    workflow.add_node("valuation_analysis", create_valuation_analysis_node())
    workflow.add_node("working_capital_analysis", create_wc_analysis_node())
    workflow.add_node("operational_analysis", create_ops_analysis_node())
    workflow.add_node("strategy_analysis", create_strategy_analysis_node())
    workflow.add_node("master_synthesis", create_master_synthesis_node())
    workflow.add_node("report_generation", generate_final_report)
    workflow.add_node("chat_setup", setup_chat_sessions)
    
    # Define workflow edges
    workflow.set_entry_point("data_validation")
    workflow.add_edge("data_validation", "liquidity_analysis")
    workflow.add_edge("data_validation", "leverage_analysis")
    workflow.add_edge("data_validation", "valuation_analysis") 
    workflow.add_edge("data_validation", "working_capital_analysis")
    workflow.add_edge("data_validation", "operational_analysis")
    workflow.add_edge("data_validation", "strategy_analysis")
    
    # All domain analyses feed to master synthesis
    workflow.add_edge("liquidity_analysis", "master_synthesis")
    workflow.add_edge("leverage_analysis", "master_synthesis") 
    workflow.add_edge("valuation_analysis", "master_synthesis")
    workflow.add_edge("working_capital_analysis", "master_synthesis")
    workflow.add_edge("operational_analysis", "master_synthesis")
    workflow.add_edge("strategy_analysis", "master_synthesis")
    
    workflow.add_edge("master_synthesis", "report_generation")
    workflow.add_edge("report_generation", "chat_setup")
    workflow.add_edge("chat_setup", END)
    
    return workflow

def validate_inputs(state: AnalysisState) -> AnalysisState:
    """Validate input parameters"""
    if not state.ticker:
        state.errors.append("Ticker symbol is required")
        return state
    
    # Add ticker validation logic
    state.current_step = "validation_complete"
    return state

def create_liquidity_analysis_node():
    """Factory function for liquidity analysis node"""
    liquidity_agent = LiquidityAgent()
    
    def liquidity_analysis(state: AnalysisState) -> AnalysisState:
        try:
            analysis = liquidity_agent.analyze_company(state.ticker, state.peers)
            state.domain_analyses["liquidity"] = analysis
        except Exception as e:
            state.errors.append(f"Liquidity analysis failed: {str(e)}")
        
        return state
    
    return liquidity_analysis

# Similar factory functions for other domains...

def setup_chat_sessions(state: AnalysisState) -> AnalysisState:
    """Setup chat sessions for each domain agent"""
    chat_manager = ChatManager()
    
    for domain in ["liquidity", "leverage", "valuation", "working_capital", "operational", "strategy", "master"]:
        session_id = chat_manager.create_session(domain)
        
        # Add analysis context to each session
        if domain in state.domain_analyses:
            context = {
                "analysis_results": state.domain_analyses[domain],
                "company": state.ticker,
                "peers": state.peers
            }
            chat_manager.send_message(session_id, "Analysis context loaded", context)
        
        state.chat_sessions[domain] = session_id
    
    return state
```

### 5.3 Workflow Execution Engine
```python
# workflows/workflow_engine.py
class WorkflowEngine:
    """Manages workflow execution and state"""
    
    def __init__(self):
        self.chat_manager = ChatManager()
        self.workflow = create_analysis_workflow(self.chat_manager)
    
    async def run_analysis(self, ticker: str, peers: List[str] = None, user_persona: str = "investor") -> Dict:
        """Execute complete analysis workflow"""
        
        initial_state = AnalysisState(
            ticker=ticker,
            peers=peers or [],
            user_persona=user_persona
        )
        
        # Run workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        return {
            "success": len(final_state.errors) == 0,
            "ticker": final_state.ticker,
            "analysis_results": final_state.domain_analyses,
            "synthesis": final_state.synthesis,
            "final_report": final_state.final_report,
            "chat_sessions": final_state.chat_sessions,
            "errors": final_state.errors,
            "warnings": final_state.warnings
        }
    
    def get_chat_response(self, domain: str, session_id: str, message: str) -> str:
        """Get chat response from specific domain agent"""
        return self.chat_manager.send_message(session_id, message)
```

### Sprint 5 Testing Criteria
- [ ] Master agent can orchestrate comprehensive analysis
- [ ] LangGraph workflow executes all analysis steps
- [ ] Specialist agent results are properly synthesized  
- [ ] Chat sessions are initialized with analysis context
- [ ] Workflow handles errors gracefully
- [ ] Analysis results are structured and complete

---

## Sprint 6: FastAPI Backend Services (Week 6-7)
**Duration:** 7 days
**Goal:** Create comprehensive REST API for frontend integration

### 6.1 FastAPI Application Structure

```python
# api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from .routers import analysis, chat, health
from ..workflows.workflow_engine import WorkflowEngine
from ..chat.chat_manager import ChatManager

# Global instances
workflow_engine = None
chat_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global workflow_engine, chat_manager
    
    # Startup
    workflow_engine = WorkflowEngine()
    chat_manager = ChatManager()
    
    yield
    
    # Cleanup
    # Close database connections, cleanup resources

app = FastAPI(
    title="Financial Analysis Agent System API",
    description="AI-powered financial analysis with domain specialist agents",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])

@app.get("/")
async def root():
    return {"message": "Financial Analysis Agent System API", "version": "1.0.0"}
```

### 6.2 Analysis API Endpoints

```python
# api/routers/analysis.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import time  # Added

router = APIRouter()

class AnalysisRequest(BaseModel):
    ticker: str
    peers: Optional[List[str]] = []
    user_persona: str = "investor"
    domains: Optional[List[str]] = None  # Specific domains to analyze

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    ticker: str
    peers: List[str]
    user_persona: str
    domain_analyses: Dict[str, Any]
    synthesis: Optional[str] = None
    chat_sessions: Dict[str, str]
    errors: List[str] = []
    warnings: List[str] = []

@router.post("/start", response_model=Dict[str, str])
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start comprehensive company analysis"""
    
    try:
        # Validate ticker
        if not request.ticker or len(request.ticker) > 10:
            raise HTTPException(status_code=400, detail="Invalid ticker symbol")
        
        # Start analysis in background
        analysis_id = f"{request.ticker}_{int(time.time())}"
        
        background_tasks.add_task(
            run_analysis_task,
            analysis_id,
            request.ticker,
            request.peers,
            request.user_persona
        )
        
        return {
            "analysis_id": analysis_id,
            "status": "started",
            "message": "Analysis started. Use /status endpoint to check progress."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get analysis progress and results"""
    
    # Check analysis status in storage/cache
    status = get_analysis_from_storage(analysis_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return status

@router.get("/results/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_results(analysis_id: str):
    """Get complete analysis results"""
    
    results = get_analysis_from_storage(analysis_id)
    
    if not results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if results["status"] != "completed":
        raise HTTPException(status_code=202, detail="Analysis still in progress")
    
    return results

@router.post("/quick-analysis")
async def quick_analysis(request: AnalysisRequest):
    """Synchronous analysis for simple use cases"""
    
    try:
        # Run workflow synchronously (with timeout)
        results = await asyncio.wait_for(
            workflow_engine.run_analysis(
                request.ticker, 
                request.peers, 
                request.user_persona
            ),
            timeout=300  # 5 minute timeout
        )
        
        return results
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Analysis timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task function
async def run_analysis_task(analysis_id: str, ticker: str, peers: List[str], persona: str):
    """Background analysis execution"""
    
    try:
        # Update status
        update_analysis_status(analysis_id, "running", {"step": "initializing"})
        
        # Run analysis
        results = await workflow_engine.run_analysis(ticker, peers, persona)
        
        # Store results
        results["analysis_id"] = analysis_id
        results["status"] = "completed"
        store_analysis_results(analysis_id, results)
        
    except Exception as e:
        # Store error
        error_result = {
            "analysis_id": analysis_id,
            "status": "failed", 
            "error": str(e)
        }
        store_analysis_results(analysis_id, error_result)
```

### 6.3 Chat API Endpoints

```python
# api/routers/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime  # Added

router = APIRouter()

class ChatSessionRequest(BaseModel):
    agent_type: str  # liquidity, leverage, valuation, etc.
    user_id: Optional[str] = None

class ChatMessageRequest(BaseModel):
    session_id: str
    message: str
    
class ChatMessageResponse(BaseModel):
    session_id: str
    response: str
    timestamp: str

class ChatHistoryResponse(BaseModel):
    session_id: str
    agent_type: str
    history: List[Dict[str, str]]

@router.post("/sessions", response_model=Dict[str, str])
async def create_chat_session(request: ChatSessionRequest):
    """Create new chat session with specific agent"""
    
    valid_agents = ["master", "liquidity", "leverage", "valuation", "working_capital", "operational", "strategy"]
    
    if request.agent_type not in valid_agents:
        raise HTTPException(status_code=400, detail=f"Invalid agent type. Choose from: {valid_agents}")
    
    try:
        session_id = chat_manager.create_session(request.agent_type, request.user_id)
        
        return {
            "session_id": session_id,
            "agent_type": request.agent_type,
            "status": "created"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/message", response_model=ChatMessageResponse)
async def send_chat_message(request: ChatMessageRequest):
    """Send message to agent and get response"""
    
    try:
        response = chat_manager.send_message(request.session_id, request.message)
        
        return ChatMessageResponse(
            session_id=request.session_id,
            response=response,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Get chat history for session"""
    
    try:
        history = chat_manager.get_session_history(session_id)
        
        # Get session info
        session = chat_manager.active_sessions.get(session_id)
        agent_type = session.agent_type if session else "unknown"
        
        return ChatHistoryResponse(
            session_id=session_id,
            agent_type=agent_type,
            history=history
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/user/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all chat sessions for a user"""
    
    try:
        sessions = chat_manager.list_user_sessions(user_id)
        return {"user_id": user_id, "sessions": sessions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete chat session"""
    
    try:
        if session_id in chat_manager.active_sessions:
            del chat_manager.active_sessions[session_id]
        
        # Could also delete from persistent storage
        
        return {"session_id": session_id, "status": "deleted"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 6.4 Health Check & System Status

```python
# api/routers/health.py
from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime

router = APIRouter()

@router.get("/")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed system health check"""
    
    health_status = {
        "api": "healthy",
        "timestamp": datetime.now().isoformat()
    }
    
    # Check RAG packages
    try:
        from SECFinancialRAG import get_financial_data
        test_result = get_financial_data("AAPL", test_mode=True)
        health_status["sec_financial_rag"] = "healthy" if test_result else "unhealthy"
    except Exception as e:
        health_status["sec_financial_rag"] = f"error: {str(e)}"
    
    # Check vector databases
    try:
        from ..vector_store.news_store import NewsVectorStore
        news_store = NewsVectorStore()
        health_status["vector_db"] = "healthy"
    except Exception as e:
        health_status["vector_db"] = f"error: {str(e)}"
    
    # Check chat system
    try:
        chat_sessions_count = len(chat_manager.active_sessions) if chat_manager else 0
        health_status["chat_system"] = {
            "status": "healthy",
            "active_sessions": chat_sessions_count
        }
    except Exception as e:
        health_status["chat_system"] = f"error: {str(e)}"
    
    return health_status

@router.get("/agents")
async def check_agents():
    """Check status of all agents"""
    
    agent_status = {}
    agent_types = ["master", "liquidity", "leverage", "valuation", "working_capital", "operational", "strategy"]
    
    for agent_type in agent_types:
        try:
            # Simple agent health check
            test_session = chat_manager.create_session(agent_type)
            response = chat_manager.send_message(test_session, "Health check")
            agent_status[agent_type] = "healthy" if response else "unhealthy"
            
            # Clean up test session
            if test_session in chat_manager.active_sessions:
                del chat_manager.active_sessions[test_session]
                
        except Exception as e:
            agent_status[agent_type] = f"error: {str(e)}"
    
    return {"agents": agent_status}
```

### 6.5 API Configuration & Deployment

```bash
# .env.production
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database Configuration  
POSTGRES_HOST=your_postgres_host
POSTGRES_DB=financial_analysis
VECTOR_DB_PATH=/app/data/vector_db

# RAG Package API Keys
ALPHA_VANTAGE_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### 6.6 Index/Search Endpoints for Vector Stores
Add explicit endpoints to align with thin interface pattern. These can be simple wrappers that call index_* and search_* in each RAG package or the corresponding VectorStore services.

Examples (contracts):
- Reports
  - POST /api/v1/reports/index { ticker, years_back, filing_types, prefer_cached }
  - POST /api/v1/reports/search { ticker, query, k, filters: { form_type, section_name, date_range } }
- News
  - POST /api/v1/news/index { ticker, days_back }
  - POST /api/v1/news/search { ticker, query, k, days_back }
- Transcripts
  - POST /api/v1/transcripts/index { ticker, quarters_back }
  - POST /api/v1/transcripts/search { ticker, query, k, filters: { quarter, section } }

These endpoints should return top-k verbatim chunks with metadata and citation fields.

---

## Sprint 7: Integration Testing & Documentation (Week 7-8)
**Duration:** 7 days
**Goal:** End-to-end testing, performance optimization, and comprehensive documentation

### 7.1 Testing Framework Setup

```python
# tests/conftest.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from AgentSystem.api.main import app
from AgentSystem.chat.conversation_store import ConversationStore

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def test_db():
    """Test database setup"""
    test_store = ConversationStore(":memory:")  # In-memory SQLite
    return test_store

@pytest.fixture
def sample_analysis_data():
    """Sample data for testing"""
    return {
        "ticker": "AAPL",
        "peers": ["MSFT", "GOOGL"],
        "user_persona": "investor"
    }
```

### 7.2 Component Tests

```python
# tests/test_agents.py
import pytest
from AgentSystem.agents.liquidity_agent import LiquidityAgent

class TestLiquidityAgent:
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        agent = LiquidityAgent()
        assert agent.domain == "liquidity"
        assert len(agent.tools) > 0
        assert agent.agent_executor is not None
    
    @pytest.mark.asyncio
    async def test_company_analysis(self):
        """Test company analysis functionality"""
        agent = LiquidityAgent()
        result = agent.analyze_company("AAPL")
        
        assert "domain" in result
        assert result["domain"] == "liquidity"
        assert "ticker" in result
        assert "analysis" in result

# tests/test_vector_store.py
class TestVectorStore:
    def test_news_storage_and_retrieval(self):
        """Test news vector storage"""
        from AgentSystem.vector_store.news_store import NewsVectorStore
        
        store = NewsVectorStore()
        
        # Test data storage
        test_chunks = [
            {
                "chunk_id": "test_1",
                "content": "Apple reports strong quarterly earnings",
                "source": "test_source"
            }
        ]
        
        result = store.store_news_data(test_chunks, "AAPL", {"days_back": 30})
        assert result is True
        
        # Test data retrieval
        cached_data = store.get_cached_news("AAPL", 30)
        assert cached_data is not None

# tests/test_chat_system.py
class TestChatSystem:
    def test_session_creation(self, test_db):
        """Test chat session creation"""
        from AgentSystem.chat.chat_manager import ChatManager
        
        chat_manager = ChatManager()
        session_id = chat_manager.create_session("liquidity", "test_user")
        
        assert session_id is not None
        assert session_id in chat_manager.active_sessions
    
    def test_message_exchange(self, test_db):
        """Test message sending and receiving"""
        from AgentSystem.chat.chat_manager import ChatManager
        
        chat_manager = ChatManager()
        session_id = chat_manager.create_session("liquidity")
        
        response = chat_manager.send_message(session_id, "What is current ratio?")
        
        assert response is not None
        assert len(response) > 0
```

### 7.3 API Integration Tests

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

class TestAnalysisAPI:
    def test_start_analysis(self, client):
        """Test analysis initiation"""
        response = client.post("/api/v1/analysis/start", json={
            "ticker": "AAPL",
            "peers": ["MSFT"],
            "user_persona": "investor"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] == "started"
    
    def test_analysis_status(self, client):
        """Test analysis status checking"""
        # First start an analysis
        start_response = client.post("/api/v1/analysis/start", json={
            "ticker": "AAPL",
            "user_persona": "investor"
        })
        
        analysis_id = start_response.json()["analysis_id"]
        
        # Check status
        status_response = client.get(f"/api/v1/analysis/status/{analysis_id}")
        
        assert status_response.status_code in [200, 202]

class TestChatAPI:
    def test_create_chat_session(self, client):
        """Test chat session creation via API"""
        response = client.post("/api/v1/chat/sessions", json={
            "agent_type": "liquidity",
            "user_id": "test_user"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
    
    def test_send_chat_message(self, client):
        """Test sending chat message via API"""
        # Create session first
        session_response = client.post("/api/v1/chat/sessions", json={
            "agent_type": "master"
        })
        
        session_id = session_response.json()["session_id"]
        
        # Send message
        message_response = client.post("/api/v1/chat/message", json={
            "session_id": session_id,
            "message": "Analyze Apple's financial health"
        })
        
        assert message_response.status_code == 200
        assert "response" in message_response.json()
```

### 7.4 Performance Tests

```python
# tests/test_performance.py
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    @pytest.mark.asyncio
    async def test_concurrent_analyses(self, client):
        """Test multiple simultaneous analyses"""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        start_time = time.time()
        
        # Start multiple analyses concurrently
        tasks = []
        for ticker in tickers:
            response = client.post("/api/v1/analysis/start", json={
                "ticker": ticker,
                "user_persona": "investor"
            })
            tasks.append(response.json()["analysis_id"])
        
        # Wait for completion (simplified)
        await asyncio.sleep(10)  # In real test, poll for completion
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete reasonably quickly
        assert duration < 300  # 5 minutes max
        assert len(tasks) == len(tickers)
    
    def test_chat_response_time(self, client):
        """Test chat response performance"""
        # Create session
        session_response = client.post("/api/v1/chat/sessions", json={
            "agent_type": "master"
        })
        session_id = session_response.json()["session_id"]
        
        # Test response times
        messages = [
            "What is Apple's current ratio?",
            "How is their debt situation?", 
            "Compare with Microsoft"
        ]
        
        response_times = []
        for message in messages:
            start_time = time.time()
            
            response = client.post("/api/v1/chat/message", json={
                "session_id": session_id,
                "message": message
            })
            
            end_time = time.time()
            response_times.append(end_time - start_time)
            
            assert response.status_code == 200
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 30  # 30 seconds max average
```

### 7.5 Documentation Generation

```python
# docs/generate_docs.py
import json
from fastapi.openapi.utils import get_openapi
from AgentSystem.api.main import app

def generate_openapi_spec():
    """Generate OpenAPI specification"""
    openapi_schema = get_openapi(
        title="Financial Analysis Agent System API",
        version="1.0.0",
        description="AI-powered financial analysis with domain specialist agents",
        routes=app.routes,
    )
    
    with open("docs/api_spec.json", "w") as f:
        json.dump(openapi_schema, f, ensure_ascii=False, indent=2)
    
    print("OpenAPI specification generated: docs/api_spec.json")

if __name__ == "__main__":
    generate_openapi_spec()
```

### 7.6 API Documentation Template

```markdown
# API Documentation

## Overview
The Financial Analysis Agent System provides AI-powered financial analysis through domain specialist agents.

## Authentication
Currently no authentication required (development mode).

## Base URL
```
http://localhost:8000/api/v1
```

## Endpoints

### Analysis Endpoints

#### Start Analysis
```http
POST /analysis/start
```

Start comprehensive financial analysis for a company.

**Request Body:**
```json
{
  "ticker": "AAPL",
  "peers": ["MSFT", "GOOGL"],
  "user_persona": "investor"
}
```

**Response:**
```json
{
  "analysis_id": "AAPL_1703123456",
  "status": "started",
  "message": "Analysis started. Use /status endpoint to check progress."
}
```

#### Check Analysis Status
```http
GET /analysis/status/{analysis_id}
```

**Response:**
```json
{
  "analysis_id": "AAPL_1703123456",
  "status": "running",
  "progress": {
    "completed_domains": ["liquidity", "leverage"],
    "remaining_domains": ["valuation", "operational"]
  }
}
```

### Chat Endpoints

#### Create Chat Session
```http
POST /chat/sessions
```

**Request Body:**
```json
{
  "agent_type": "liquidity",
  "user_id": "optional_user_id"
}
```

#### Send Message
```http
POST /chat/message
```

**Request Body:**
```json
{
  "session_id": "uuid-session-id",
  "message": "What is Apple's current ratio?"
}
```

## Error Handling

All endpoints return errors in this format:
```json
{
  "detail": "Error message description"
}
```

Common HTTP status codes:
- `200`: Success
- `202`: Accepted (for async operations)
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error
```

### Sprint 7 Testing Criteria
- [ ] All unit tests pass with >90% code coverage
- [ ] Integration tests verify end-to-end functionality
- [ ] Performance tests validate system scalability  
- [ ] API documentation is complete and accurate
- [ ] Load testing shows acceptable response times
- [ ] Error handling works consistently across all components

---

## 🚀 Deployment & Production Setup

### Production Checklist
- [ ] Environment variables configured
- [ ] Database connections secured
- [ ] API rate limiting implemented
- [ ] Logging and monitoring setup
- [ ] Error tracking configured
- [ ] Docker containers built
- [ ] Health checks operational

### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY AgentSystem ./AgentSystem
EXPOSE 8000

CMD ["uvicorn", "AgentSystem.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration
```bash
# .env.production
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database Configuration  
POSTGRES_HOST=your_postgres_host
POSTGRES_DB=financial_analysis
VECTOR_DB_PATH=/app/data/vector_db

# RAG Package API Keys
ALPHA_VANTAGE_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### 6.6 Index/Search Endpoints for Vector Stores
Add explicit endpoints to align with thin interface pattern. These can be simple wrappers that call index_* and search_* in each RAG package or the corresponding VectorStore services.

Examples (contracts):
- Reports
  - POST /api/v1/reports/index { ticker, years_back, filing_types, prefer_cached }
  - POST /api/v1/reports/search { ticker, query, k, filters: { form_type, section_name, date_range } }
- News
  - POST /api/v1/news/index { ticker, days_back }
  - POST /api/v1/news/search { ticker, query, k, days_back }
- Transcripts
  - POST /api/v1/transcripts/index { ticker, quarters_back }
  - POST /api/v1/transcripts/search { ticker, query, k, filters: { quarter, section } }

These endpoints should return top-k verbatim chunks with metadata and citation fields.

---

## Sprint 7: Integration Testing & Documentation (Week 7-8)
**Duration:** 7 days
**Goal:** End-to-end testing, performance optimization, and comprehensive documentation

### 7.1 Testing Framework Setup

```python
# tests/conftest.py
import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from AgentSystem.api.main import app
from AgentSystem.chat.conversation_store import ConversationStore

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def test_db():
    """Test database setup"""
    test_store = ConversationStore(":memory:")  # In-memory SQLite
    return test_store

@pytest.fixture
def sample_analysis_data():
    """Sample data for testing"""
    return {
        "ticker": "AAPL",
        "peers": ["MSFT", "GOOGL"],
        "user_persona": "investor"
    }
```

### 7.2 Component Tests

```python
# tests/test_agents.py
import pytest
from AgentSystem.agents.liquidity_agent import LiquidityAgent

class TestLiquidityAgent:
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        agent = LiquidityAgent()
        assert agent.domain == "liquidity"
        assert len(agent.tools) > 0
        assert agent.agent_executor is not None
    
    @pytest.mark.asyncio
    async def test_company_analysis(self):
        """Test company analysis functionality"""
        agent = LiquidityAgent()
        result = agent.analyze_company("AAPL")
        
        assert "domain" in result
        assert result["domain"] == "liquidity"
        assert "ticker" in result
        assert "analysis" in result

# tests/test_vector_store.py
class TestVectorStore:
    def test_news_storage_and_retrieval(self):
        """Test news vector storage"""
        from AgentSystem.vector_store.news_store import NewsVectorStore
        
        store = NewsVectorStore()
        
        # Test data storage
        test_chunks = [
            {
                "chunk_id": "test_1",
                "content": "Apple reports strong quarterly earnings",
                "source": "test_source"
            }
        ]
        
        result = store.store_news_data(test_chunks, "AAPL", {"days_back": 30})
        assert result is True
        
        # Test data retrieval
        cached_data = store.get_cached_news("AAPL", 30)
        assert cached_data is not None

# tests/test_chat_system.py
class TestChatSystem:
    def test_session_creation(self, test_db):
        """Test chat session creation"""
        from AgentSystem.chat.chat_manager import ChatManager
        
        chat_manager = ChatManager()
        session_id = chat_manager.create_session("liquidity", "test_user")
        
        assert session_id is not None
        assert session_id in chat_manager.active_sessions
    
    def test_message_exchange(self, test_db):
        """Test message sending and receiving"""
        from AgentSystem.chat.chat_manager import ChatManager
        
        chat_manager = ChatManager()
        session_id = chat_manager.create_session("liquidity")
        
        response = chat_manager.send_message(session_id, "What is current ratio?")
        
        assert response is not None
        assert len(response) > 0
```

### 7.3 API Integration Tests

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

class TestAnalysisAPI:
    def test_start_analysis(self, client):
        """Test analysis initiation"""
        response = client.post("/api/v1/analysis/start", json={
            "ticker": "AAPL",
            "peers": ["MSFT"],
            "user_persona": "investor"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] == "started"
    
    def test_analysis_status(self, client):
        """Test analysis status checking"""
        # First start an analysis
        start_response = client.post("/api/v1/analysis/start", json={
            "ticker": "AAPL",
            "user_persona": "investor"
        })
        
        analysis_id = start_response.json()["analysis_id"]
        
        # Check status
        status_response = client.get(f"/api/v1/analysis/status/{analysis_id}")
        
        assert status_response.status_code in [200, 202]

class TestChatAPI:
    def test_create_chat_session(self, client):
        """Test chat session creation via API"""
        response = client.post("/api/v1/chat/sessions", json={
            "agent_type": "liquidity",
            "user_id": "test_user"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
    
    def test_send_chat_message(self, client):
        """Test sending chat message via API"""
        # Create session first
        session_response = client.post("/api/v1/chat/sessions", json={
            "agent_type": "master"
        })
        
        session_id = session_response.json()["session_id"]
        
        # Send message
        message_response = client.post("/api/v1/chat/message", json={
            "session_id": session_id,
            "message": "Analyze Apple's financial health"
        })
        
        assert message_response.status_code == 200
        assert "response" in message_response.json()
```

### 7.4 Performance Tests

```python
# tests/test_performance.py
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    @pytest.mark.asyncio
    async def test_concurrent_analyses(self, client):
        """Test multiple simultaneous analyses"""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        start_time = time.time()
        
        # Start multiple analyses concurrently
        tasks = []
        for ticker in tickers:
            response = client.post("/api/v1/analysis/start", json={
                "ticker": ticker,
                "user_persona": "investor"
            })
            tasks.append(response.json()["analysis_id"])
        
        # Wait for completion (simplified)
        await asyncio.sleep(10)  # In real test, poll for completion
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete reasonably quickly
        assert duration < 300  # 5 minutes max
        assert len(tasks) == len(tickers)
    
    def test_chat_response_time(self, client):
        """Test chat response performance"""
        # Create session
        session_response = client.post("/api/v1/chat/sessions", json={
            "agent_type": "master"
        })
        session_id = session_response.json()["session_id"]
        
        # Test response times
        messages = [
            "What is Apple's current ratio?",
            "How is their debt situation?", 
            "Compare with Microsoft"
        ]
        
        response_times = []
        for message in messages:
            start_time = time.time()
            
            response = client.post("/api/v1/chat/message", json={
                "session_id": session_id,
                "message": message
            })
            
            end_time = time.time()
            response_times.append(end_time - start_time)
            
            assert response.status_code == 200
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 30  # 30 seconds max average
```

### 7.5 Documentation Generation

```python
# docs/generate_docs.py
import json
from fastapi.openapi.utils import get_openapi
from AgentSystem.api.main import app

def generate_openapi_spec():
    """Generate OpenAPI specification"""
    openapi_schema = get_openapi(
        title="Financial Analysis Agent System API",
        version="1.0.0",
        description="AI-powered financial analysis with domain specialist agents",
        routes=app.routes,
    )
    
    with open("docs/api_spec.json", "w") as f:
        json.dump(openapi_schema, f, ensure_ascii=False, indent=2)
    
    print("OpenAPI specification generated: docs/api_spec.json")

if __name__ == "__main__":
    generate_openapi_spec()
```

### 7.6 API Documentation Template

```markdown
# API Documentation

## Overview
The Financial Analysis Agent System provides AI-powered financial analysis through domain specialist agents.

## Authentication
Currently no authentication required (development mode).

## Base URL
```
http://localhost:8000/api/v1
```

## Endpoints

### Analysis Endpoints

#### Start Analysis
```http
POST /analysis/start
```

Start comprehensive financial analysis for a company.

**Request Body:**
```json
{
  "ticker": "AAPL",
  "peers": ["MSFT", "GOOGL"],
  "user_persona": "investor"
}
```

**Response:**
```json
{
  "analysis_id": "AAPL_1703123456",
  "status": "started",
  "message": "Analysis started. Use /status endpoint to check progress."
}
```

#### Check Analysis Status
```http
GET /analysis/status/{analysis_id}
```

**Response:**
```json
{
  "analysis_id": "AAPL_1703123456",
  "status": "running",
  "progress": {
    "completed_domains": ["liquidity", "leverage"],
    "remaining_domains": ["valuation", "operational"]
  }
}
```

### Chat Endpoints

#### Create Chat Session
```http
POST /chat/sessions
```

**Request Body:**
```json
{
  "agent_type": "liquidity",
  "user_id": "optional_user_id"
}
```

#### Send Message
```http
POST /chat/message
```

**Request Body:**
```json
{
  "session_id": "uuid-session-id",
  "message": "What is Apple's current ratio?"
}
```

## Error Handling

All endpoints return errors in this format:
```json
{
  "detail": "Error message description"
}
```

Common HTTP status codes:
- `200`: Success
- `202`: Accepted (for async operations)
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error
```

### Sprint 7 Testing Criteria
- [ ] All unit tests pass with >90% code coverage
- [ ] Integration tests verify end-to-end functionality
- [ ] Performance tests validate system scalability  
- [ ] API documentation is complete and accurate
- [ ] Load testing shows acceptable response times
- [ ] Error handling works consistently across all components

---

## 🚀 Deployment & Production Setup

### Production Checklist
- [ ] Environment variables configured
- [ ] Database connections secured
- [ ] API rate limiting implemented
- [ ] Logging and monitoring setup
- [ ] Error tracking configured
- [ ] Docker containers built
- [ ] Health checks operational

### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY AgentSystem ./AgentSystem
EXPOSE 8000

CMD ["uvicorn", "AgentSystem.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration
```bash
# .env.production
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database Configuration  
POSTGRES_HOST=your_postgres_host
POSTGRES_DB=financial_analysis
VECTOR_DB_PATH=/app/data/vector_db

# RAG Package API Keys
ALPHA_VANTAGE_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### 6.6 Index/Search Endpoints for Vector Stores
Add explicit endpoints to align with thin interface pattern. These can be simple wrappers that call index_* and search_* in each RAG package or the corresponding VectorStore services.

Examples (contracts):
- Reports
  - POST /api/v1/reports/index { ticker, years_back, filing_types, prefer_cached }
  - POST /api/v1/reports/search { ticker, query, k, filters: { form_type, section_name, date_range } }
- News
  - POST /api/v1/news/index { ticker, days_back }
  - POST /api/v1/news/search { ticker, query, k, days_back }
- Transcripts
  - POST /api/v1/transcripts/index { ticker, quarters_back }
  - POST /api/v1/transcripts/search { ticker, query, k, filters: { quarter, section } }

These endpoints should return top-k verbatim chunks with metadata and citation fields.

---