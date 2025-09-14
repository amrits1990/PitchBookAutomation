"""
Transcript Query Enhancement Module
Expands user queries with financial synonyms and domain-specific terminology for better BM25 search results
"""

import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class TranscriptQueryEnhancer:
    """Enhances search queries with financial synonyms and domain-specific terms"""
    
    def __init__(self):
        # Financial terminology mapping for earnings call transcripts
        self.keyword_mapping = {
            # Forward-looking terms
            "forward-looking": ["outlook", "forecast", "guidance", "projection", "estimate", "expected", "anticipate", "predict", "future", "roadmap", "vision", "planning"],
            "outlook": ["guidance", "forecast", "projection", "forward-looking", "estimate", "expected", "anticipate", "future outlook", "going forward"],
            "guidance": ["outlook", "forecast", "projection", "forward-looking", "estimate", "expected", "target", "goal", "roadmap"],
            "forecast": ["outlook", "guidance", "projection", "estimate", "expected", "anticipated", "predicted", "forecasted"],
            "estimates": ["guidance", "outlook", "forecast", "projection", "expected", "anticipated", "target", "consensus"],
            
            # Financial performance
            "revenue": ["sales", "income", "top line", "net sales", "total revenue", "gross sales", "receipts"],
            "earnings": ["profit", "net income", "bottom line", "earnings per share", "EPS", "profitability", "net profit"],
            "margin": ["gross margin", "operating margin", "profit margin", "EBITDA margin", "net margin", "margins", "profitability"],
            "growth": ["expansion", "increase", "improvement", "uptick", "acceleration", "momentum", "gains", "rising"],
            "decline": ["decrease", "drop", "fall", "contraction", "deceleration", "slowdown", "weakness", "deterioration"],
            
            # Capital allocation
            "repurchase": ["buyback", "buy back", "share repurchase", "stock buyback", "share purchase", "repo", "capital return"],
            "buyback": ["repurchase", "buy back", "share repurchase", "stock buyback", "share purchase", "repo", "capital return"],
            "dividend": ["payout", "distribution", "dividend payment", "cash dividend", "shareholder return", "yield"],
            "capex": ["capital expenditure", "capital spending", "investments", "infrastructure spending", "facility investments", "equipment purchases"],
            "cash flow": ["operating cash flow", "free cash flow", "FCF", "cash generation", "cash from operations", "liquidity"],
            
            # Operations
            "supply chain": ["supply", "logistics", "manufacturing", "production", "sourcing", "procurement", "vendor", "supplier"],
            "inventory": ["stock", "stockpile", "goods", "merchandise", "finished goods", "raw materials", "components"],
            "demand": ["customer demand", "market demand", "consumption", "appetite", "interest", "purchasing", "adoption"],
            "pricing": ["price", "pricing strategy", "cost", "pricing power", "price realization", "pricing environment"],
            
            # Market dynamics
            "competition": ["competitive", "competitor", "rival", "market share", "competitive landscape", "competitive pressure"],
            "market share": ["share", "market position", "competitive position", "penetration", "footprint"],
            "expansion": ["growth", "scaling", "rollout", "launch", "enter", "penetrate", "extend"],
            "acquisition": ["M&A", "merger", "purchase", "deal", "transaction", "strategic investment", "takeover"],
            
            # Technology & Innovation
            "innovation": ["R&D", "research", "development", "technology", "breakthrough", "advancement", "cutting-edge"],
            "AI": ["artificial intelligence", "machine learning", "ML", "intelligent", "smart", "automation", "cognitive"],
            "digital": ["digitization", "digital transformation", "online", "electronic", "virtual", "cloud"],
            "platform": ["ecosystem", "infrastructure", "framework", "system", "architecture", "solution"],
            
            # Cost management
            "cost": ["expense", "spending", "expenditure", "investment", "outlay", "cost structure", "operating expense"],
            "efficiency": ["optimization", "productivity", "streamlining", "rationalization", "cost savings", "operational efficiency"],
            "restructuring": ["reorganization", "cost reduction", "optimization", "rightsizing", "transformation"],
            
            # Geographic & Segments
            "international": ["global", "overseas", "foreign", "worldwide", "abroad", "cross-border", "multinational"],
            "domestic": ["US", "United States", "home market", "local", "national", "stateside"],
            "emerging markets": ["developing markets", "growth markets", "international expansion", "new markets"],
            "segment": ["division", "business unit", "category", "vertical", "line of business", "product line"],
            
            # Financial metrics
            "EBITDA": ["earnings before interest tax depreciation amortization", "operating earnings", "core earnings"],
            "ROI": ["return on investment", "returns", "profitability", "investment returns"],
            "leverage": ["debt", "borrowing", "financial leverage", "debt ratio", "gearing"],
            "working capital": ["current assets", "liquidity", "cash conversion", "operating capital"],
            
            # Sentiment & Performance
            "strong": ["robust", "solid", "healthy", "impressive", "excellent", "outstanding", "exceptional"],
            "weak": ["soft", "challenging", "difficult", "poor", "disappointing", "concerning", "pressured"],
            "momentum": ["trend", "trajectory", "direction", "progress", "acceleration", "pace", "speed"],
            "headwinds": ["challenges", "obstacles", "pressures", "difficulties", "constraints", "barriers"],
            "tailwinds": ["opportunities", "benefits", "support", "favorable conditions", "positive drivers"],
            
            # Timeframes
            "quarterly": ["Q1", "Q2", "Q3", "Q4", "quarter", "three months", "90 days"],
            "annual": ["yearly", "year-over-year", "YoY", "full year", "FY", "12 months"],
            "near-term": ["short-term", "immediate", "next quarter", "coming months", "upcoming"],
            "long-term": ["strategic", "multi-year", "future", "long-range", "sustained", "ongoing"],
        }
        
        # Build reverse mapping for faster lookup
        self.reverse_mapping = defaultdict(set)
        for primary_term, synonyms in self.keyword_mapping.items():
            self.reverse_mapping[primary_term.lower()].add(primary_term)
            for synonym in synonyms:
                self.reverse_mapping[synonym.lower()].add(primary_term)
                # Also add the synonym itself
                self.reverse_mapping[synonym.lower()].add(synonym)
    
    def expand_query(self, query: str, max_expansions: int = 3) -> str:
        """
        Expand query with relevant financial synonyms
        
        Args:
            query: Original search query
            max_expansions: Maximum number of synonyms to add per detected term
            
        Returns:
            Expanded query string with synonyms
        """
        query_lower = query.lower()
        expanded_terms = set()
        original_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Find matching terms and their expansions
        for word in original_words:
            if word in self.reverse_mapping:
                # Get primary terms this word maps to
                primary_terms = self.reverse_mapping[word]
                
                for primary_term in primary_terms:
                    if primary_term in self.keyword_mapping:
                        # Add synonyms for this primary term
                        synonyms = self.keyword_mapping[primary_term][:max_expansions]
                        expanded_terms.update(synonyms)
        
        # Also check for multi-word phrases
        expanded_terms.update(self._find_phrase_expansions(query_lower, max_expansions))
        
        # Remove words that are already in the original query and remove duplicates
        expanded_terms = expanded_terms - original_words
        
        # Create expanded query with limited terms to avoid noise
        if expanded_terms:
            # Limit to most relevant terms to prevent query bloat
            limited_terms = sorted(list(expanded_terms))[:6]  # Max 6 additional terms
            expansion_str = " ".join(limited_terms)
            return f"{query} {expansion_str}"
        
        return query
    
    def _find_phrase_expansions(self, query_lower: str, max_expansions: int) -> Set[str]:
        """Find expansions for multi-word phrases in the query"""
        expansions = set()
        
        # Check for key phrases
        phrase_patterns = {
            r'\bforward[- ]looking\b': 'forward-looking',
            r'\bshare repurchase\b': 'repurchase',
            r'\bbuy[- ]?back\b': 'buyback',
            r'\bcash flow\b': 'cash flow',
            r'\bmarket share\b': 'market share',
            r'\bsupply chain\b': 'supply chain',
            r'\bcapital expenditure\b': 'capex',
            r'\boperating margin\b': 'margin',
            r'\bnet income\b': 'earnings',
        }
        
        for pattern, key in phrase_patterns.items():
            if re.search(pattern, query_lower):
                if key in self.keyword_mapping:
                    synonyms = self.keyword_mapping[key][:max_expansions]
                    expansions.update(synonyms)
        
        return expansions
    
    def get_query_suggestions(self, query: str) -> List[Tuple[str, List[str]]]:
        """
        Get suggestions for query improvement based on detected financial terms
        
        Args:
            query: Original search query
            
        Returns:
            List of tuples: (detected_term, suggested_synonyms)
        """
        query_lower = query.lower()
        suggestions = []
        
        for word in re.findall(r'\b\w+\b', query_lower):
            if word in self.reverse_mapping:
                primary_terms = self.reverse_mapping[word]
                for primary_term in primary_terms:
                    if primary_term in self.keyword_mapping:
                        synonyms = self.keyword_mapping[primary_term][:5]
                        suggestions.append((word, synonyms))
        
        return suggestions
    
    def enhance_bm25_query(self, query: str, max_synonyms: int = 2, boost_important: bool = False) -> str:
        """
        Enhance query specifically for BM25 search with controlled synonym expansion
        
        Args:
            query: Original query
            max_synonyms: Maximum synonyms to add per detected term
            boost_important: Whether to add weight to important financial terms (disabled by default)
            
        Returns:
            Enhanced query optimized for BM25
        """
        # Use more conservative expansion to reduce noise
        expanded_query = self.expand_query(query, max_expansions=max_synonyms)
        
        # Only boost if specifically requested, and do it more conservatively
        if boost_important:
            important_terms = ['guidance', 'outlook', 'forecast', 'earnings', 'revenue']
            query_words = set(expanded_query.lower().split())
            
            boosted_terms = []
            for term in important_terms:
                if term in query_words:
                    boosted_terms.append(term)  # Add only once, don't repeat
            
            if boosted_terms:
                # Remove duplicates and add unique boosted terms
                unique_boosted = list(set(boosted_terms) - query_words)
                if unique_boosted:
                    expanded_query += " " + " ".join(unique_boosted)
        
        return expanded_query
    
    def get_financial_context_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        Extract financial context and return relevant keyword categories
        
        Args:
            query: Search query
            
        Returns:
            Dictionary mapping context categories to relevant keywords
        """
        query_lower = query.lower()
        context_keywords = defaultdict(list)
        
        # Define context categories
        categories = {
            'performance': ['revenue', 'earnings', 'margin', 'growth', 'decline'],
            'forward_looking': ['outlook', 'guidance', 'forecast', 'estimates'],
            'capital_allocation': ['repurchase', 'buyback', 'dividend', 'capex'],
            'operations': ['supply chain', 'inventory', 'demand', 'pricing'],
            'market': ['competition', 'market share', 'expansion'],
            'financial_health': ['cash flow', 'EBITDA', 'leverage', 'working capital']
        }
        
        for category, terms in categories.items():
            for term in terms:
                if term in self.keyword_mapping:
                    # Check if any synonym of this term appears in query
                    all_terms = [term] + self.keyword_mapping[term]
                    for check_term in all_terms:
                        if check_term.lower() in query_lower:
                            context_keywords[category].extend(self.keyword_mapping[term][:3])
                            break
        
        return dict(context_keywords)


# Example usage and testing
if __name__ == "__main__":
    enhancer = TranscriptQueryEnhancer()
    
    # Test queries
    test_queries = [
        "What is Apple's outlook for next quarter?",
        "Tell me about revenue growth and margins",
        "Any updates on share buyback program?",
        "How is supply chain performing?",
        "What are the forward-looking estimates?"
    ]
    
    print("🔍 Query Enhancement Examples:")
    print("=" * 60)
    
    for query in test_queries:
        expanded = enhancer.expand_query(query)
        bm25_enhanced = enhancer.enhance_bm25_query(query)
        suggestions = enhancer.get_query_suggestions(query)
        
        print(f"\nOriginal: {query}")
        print(f"Expanded: {expanded}")
        print(f"BM25 Enhanced: {bm25_enhanced}")
        if suggestions:
            print(f"Suggestions: {suggestions}")
        print("-" * 40)