"""
Transcript metadata extraction module
Extracts and enriches metadata from transcript data for enhanced searchability
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from data_source_interface import TranscriptData


@dataclass
class EnrichedTranscriptMetadata:
    """Enriched metadata extracted from transcript content"""
    
    # Basic metadata
    ticker: str
    company_name: str
    transcript_date: datetime
    quarter: str
    fiscal_year: str
    transcript_type: str
    source: str
    
    # Content metadata
    content_length: int
    word_count: int
    participant_count: int
    participants: List[str]
    
    # Financial metrics (if available)
    eps_reported: Optional[float] = None
    eps_estimated: Optional[float] = None
    eps_surprise: Optional[float] = None
    revenue_mentioned: bool = False
    guidance_mentioned: bool = False
    
    # Extracted entities
    key_topics: List[str] = None
    sentiment_indicators: List[str] = None
    financial_terms: List[str] = None
    
    # Processing metadata
    extraction_timestamp: str = None
    processing_version: str = "1.0"
    
    def __post_init__(self):
        if self.key_topics is None:
            self.key_topics = []
        if self.sentiment_indicators is None:
            self.sentiment_indicators = []
        if self.financial_terms is None:
            self.financial_terms = []
        if self.extraction_timestamp is None:
            self.extraction_timestamp = datetime.now().isoformat()


class TranscriptMetadataExtractor:
    """Extracts and enriches metadata from transcript data"""
    
    def __init__(self):
        # Financial terms patterns
        self.financial_terms_patterns = [
            r'\b(?:revenue|sales|income|profit|loss|margin|ebitda|cash flow)\b',
            r'\b(?:earnings|eps|dividend|guidance|outlook|forecast)\b',
            r'\b(?:growth|expansion|acquisition|merger|investment)\b',
            r'\b(?:market share|competition|competitive|strategy)\b',
            r'\b(?:cost|expense|spending|capex|opex)\b'
        ]
        
        # Sentiment indicators
        self.positive_sentiment_patterns = [
            r'\b(?:strong|positive|growth|increase|improve|better|excellent|outstanding)\b',
            r'\b(?:exceed|beat|above|higher|record|milestone)\b',
            r'\b(?:confident|optimistic|pleased|successful|progress)\b'
        ]
        
        self.negative_sentiment_patterns = [
            r'\b(?:weak|decline|decrease|lower|below|miss|challenge|difficult)\b',
            r'\b(?:concern|risk|issue|problem|headwind|pressure)\b',
            r'\b(?:uncertain|cautious|disappointed|struggle)\b'
        ]
        
        # Key business topics
        self.topic_patterns = {
            'technology': r'\b(?:digital|technology|innovation|AI|automation|software|platform)\b',
            'market_expansion': r'\b(?:expansion|market|international|global|new market)\b',
            'operations': r'\b(?:operations|efficiency|productivity|supply chain|manufacturing)\b',
            'customer': r'\b(?:customer|client|user|consumer|satisfaction|retention)\b',
            'regulatory': r'\b(?:regulation|compliance|regulatory|policy|government)\b',
            'sustainability': r'\b(?:sustainability|environmental|ESG|climate|green)\b'
        }
    
    def extract_metadata(self, transcript_data: TranscriptData) -> EnrichedTranscriptMetadata:
        """
        Extract enriched metadata from transcript data
        
        Args:
            transcript_data: TranscriptData object
            
        Returns:
            EnrichedTranscriptMetadata with extracted information
        """
        content = transcript_data.content.lower()
        
        # Basic content metrics
        content_length = len(transcript_data.content)
        word_count = len(transcript_data.content.split())
        participant_count = len(transcript_data.participants)
        
        # Extract financial metrics from existing metadata
        eps_data = transcript_data.metadata.get('eps_data', {})
        eps_reported = self._safe_float_convert(eps_data.get('reported'))
        eps_estimated = self._safe_float_convert(eps_data.get('estimated'))
        eps_surprise = self._safe_float_convert(eps_data.get('surprise'))
        
        # Check for revenue and guidance mentions
        revenue_mentioned = bool(re.search(r'\b(?:revenue|sales|income)\b', content))
        guidance_mentioned = bool(re.search(r'\b(?:guidance|outlook|forecast|expect)\b', content))
        
        # Extract financial terms
        financial_terms = self._extract_patterns(content, self.financial_terms_patterns)
        
        # Extract sentiment indicators
        positive_sentiment = self._extract_patterns(content, self.positive_sentiment_patterns)
        negative_sentiment = self._extract_patterns(content, self.negative_sentiment_patterns)
        sentiment_indicators = positive_sentiment + negative_sentiment
        
        # Extract key topics
        key_topics = []
        for topic, pattern in self.topic_patterns.items():
            if re.search(pattern, content):
                key_topics.append(topic)
        
        return EnrichedTranscriptMetadata(
            ticker=transcript_data.ticker,
            company_name=transcript_data.company_name,
            transcript_date=transcript_data.transcript_date,
            quarter=transcript_data.quarter,
            fiscal_year=transcript_data.fiscal_year,
            transcript_type=transcript_data.transcript_type,
            source=transcript_data.source,
            content_length=content_length,
            word_count=word_count,
            participant_count=participant_count,
            participants=transcript_data.participants,
            eps_reported=eps_reported,
            eps_estimated=eps_estimated,
            eps_surprise=eps_surprise,
            revenue_mentioned=revenue_mentioned,
            guidance_mentioned=guidance_mentioned,
            key_topics=key_topics,
            sentiment_indicators=sentiment_indicators,
            financial_terms=financial_terms
        )
    
    def _safe_float_convert(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == '':
            return None
        try:
            if isinstance(value, str):
                # Remove common non-numeric characters
                cleaned = re.sub(r'[^\d.-]', '', value)
                return float(cleaned) if cleaned else None
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _extract_patterns(self, content: str, patterns: List[str]) -> List[str]:
        """Extract unique matches from content using regex patterns"""
        matches = set()
        for pattern in patterns:
            found = re.findall(pattern, content, re.IGNORECASE)
            matches.update(found)
        return list(matches)
    
    def extract_earnings_insights(self, transcript_data: TranscriptData) -> Dict[str, Any]:
        """
        Extract specific earnings call insights
        
        Args:
            transcript_data: TranscriptData object
            
        Returns:
            Dictionary with earnings-specific insights
        """
        content = transcript_data.content.lower()
        insights = {}
        
        # Performance indicators
        beat_consensus = any(word in content for word in ['beat', 'exceed', 'above consensus'])
        missed_estimates = any(word in content for word in ['miss', 'below', 'disappointed'])
        
        insights['performance'] = {
            'beat_consensus': beat_consensus,
            'missed_estimates': missed_estimates,
            'eps_surprise_positive': transcript_data.metadata.get('eps_data', {}).get('surprise', 0) > 0
        }
        
        # Forward-looking statements
        guidance_raised = bool(re.search(r'\b(?:raise|raising|increased|improving)\b.*\b(?:guidance|outlook)\b', content))
        guidance_lowered = bool(re.search(r'\b(?:lower|lowering|reduced|cut)\b.*\b(?:guidance|outlook)\b', content))
        
        insights['guidance'] = {
            'guidance_provided': transcript_data.metadata.get('guidance_mentioned', False),
            'guidance_raised': guidance_raised,
            'guidance_lowered': guidance_lowered
        }
        
        # Business highlights
        insights['business_highlights'] = {
            'new_products_mentioned': bool(re.search(r'\b(?:new product|launch|introduce)\b', content)),
            'partnerships_mentioned': bool(re.search(r'\b(?:partnership|alliance|collaboration)\b', content)),
            'acquisitions_mentioned': bool(re.search(r'\b(?:acquisition|acquire|merger|bought)\b', content)),
            'expansion_mentioned': bool(re.search(r'\b(?:expansion|expand|growing|growth)\b', content))
        }
        
        return insights
    
    def create_search_metadata(self, enriched_metadata: EnrichedTranscriptMetadata) -> Dict[str, Any]:
        """
        Create optimized metadata for search and retrieval
        
        Args:
            enriched_metadata: EnrichedTranscriptMetadata object
            
        Returns:
            Dictionary optimized for search indexing
        """
        return {
            'ticker': enriched_metadata.ticker,
            'company_name': enriched_metadata.company_name,
            'quarter': enriched_metadata.quarter,
            'fiscal_year': enriched_metadata.fiscal_year,
            'transcript_type': enriched_metadata.transcript_type,
            'source': enriched_metadata.source,
            'date': enriched_metadata.transcript_date.isoformat(),
            'content_stats': {
                'length': enriched_metadata.content_length,
                'word_count': enriched_metadata.word_count,
                'participant_count': enriched_metadata.participant_count
            },
            'financial_metrics': {
                'eps_reported': enriched_metadata.eps_reported,
                'eps_estimated': enriched_metadata.eps_estimated,
                'eps_surprise': enriched_metadata.eps_surprise,
                'revenue_mentioned': enriched_metadata.revenue_mentioned,
                'guidance_mentioned': enriched_metadata.guidance_mentioned
            },
            'topics': enriched_metadata.key_topics,
            'sentiment': enriched_metadata.sentiment_indicators,
            'financial_terms': enriched_metadata.financial_terms,
            'searchable_text': f"{enriched_metadata.company_name} {enriched_metadata.ticker} {enriched_metadata.quarter} {enriched_metadata.fiscal_year} {' '.join(enriched_metadata.key_topics)}".lower()
        }