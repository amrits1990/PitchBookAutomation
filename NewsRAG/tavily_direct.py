"""
Simplified Tavily client that returns chunks directly (no separate chunking needed)
"""

import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from .models import NewsQuery, NewsCategory, NewsChunk
from .config import NewsConfig, NEWS_CATEGORY_TERMS, TRUSTED_NEWS_DOMAINS

try:
    from tavily import TavilyClient
except ImportError:
    raise ImportError("tavily package is required. Install with: pip install tavily-python")


logger = logging.getLogger(__name__)


class TavilyDirectClient:
    """Simplified client that converts Tavily results directly to NewsChunk objects"""
    
    def __init__(self, config: NewsConfig):
        self.config = config
        self.client = TavilyClient(api_key=config.tavily_api_key)
        
        # Rate limiting
        self._last_request_time = 0
        self._requests_this_minute = 0
        self._minute_start = time.time()
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        
        # Reset minute counter if needed
        if current_time - self._minute_start >= 60:
            self._requests_this_minute = 0
            self._minute_start = current_time
        
        # Check if we need to wait
        if self._requests_this_minute >= self.config.requests_per_minute:
            sleep_time = 60 - (current_time - self._minute_start) + 1
            logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
            self._requests_this_minute = 0
            self._minute_start = time.time()
        
        # Minimum time between requests
        time_since_last = current_time - self._last_request_time
        min_interval = 60 / self.config.requests_per_minute
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
    
    def search_news_chunks(self, query: NewsQuery) -> List[NewsChunk]:
        """Search for news and return ready-to-use chunks directly"""
        all_chunks = []
        
        try:
            for company in query.companies:
                for category in query.categories:
                    logger.info(f"Searching news for {company} in category {category.value}")
                    
                    # Build search query
                    search_query = self._build_search_query(company, category)
                    
                    # Get chunks for this company/category combination
                    company_chunks = self._search_single_query(search_query, query, company, category)
                    all_chunks.extend(company_chunks)
                    
                    logger.info(f"Retrieved {len(company_chunks)} chunks for {company} - {category.value}")
            
            # Remove duplicates based on URL
            unique_chunks = self._deduplicate_chunks(all_chunks)
            logger.info(f"Total unique chunks retrieved: {len(unique_chunks)}")
            
            return unique_chunks
            
        except Exception as e:
            logger.error(f"Error searching news: {e}")
            raise
    
    def _build_search_query(self, company: str, category: NewsCategory) -> str:
        """Build search query string for Tavily API"""
        
        # Base query with company name and business focus
        query_parts = [f"business updates for {company}"]
        
        # Add category-specific terms
        category_terms = NEWS_CATEGORY_TERMS.get(category.value, [])
        if category_terms and category.value != 'general':
            # Add specific category focus
            category_query = " OR ".join(category_terms[:2])  # Limit to top 2 terms
            query_parts.append(f"({category_query})")
        
        return " AND ".join(query_parts)
    
    def _search_single_query(self, search_query: str, query: NewsQuery, company: str, category: NewsCategory) -> List[NewsChunk]:
        """Perform a single search query and return chunks directly"""
        
        self._wait_for_rate_limit()
        
        # Map days_back to Tavily time_range
        time_range = self._get_time_range(query.days_back)
        
        # Prepare domain list for Tavily
        include_domains = TRUSTED_NEWS_DOMAINS
        
        try:
            logger.debug(f"Sending request to Tavily API: {search_query}")
            
            response = self.client.search(
                query=search_query,
                topic="news",
                search_depth="advanced",
                max_results=query.max_articles_per_query,
                time_range=time_range,
                include_answer=False,  # We don't need the AI-generated answer
                chunks_per_source=4,   # Get multiple chunks per source
                include_domains=include_domains
            )
            
            self._requests_this_minute += 1
            self._last_request_time = time.time()
            
            return self._parse_to_chunks(response, company, category)
            
        except Exception as e:
            logger.error(f"Tavily API error: {e}")
            return []
    
    def _get_time_range(self, days_back: int) -> str:
        """Map days_back to Tavily time_range parameter"""
        if days_back <= 1:
            return "day"
        elif days_back <= 7:
            return "week"  
        elif days_back <= 30:
            return "month"
        elif days_back <= 365:
            return "year"
        else:
            return "year"  # Default to year for longer periods
    
    def _parse_to_chunks(self, response: Dict[str, Any], company: str, category: NewsCategory) -> List[NewsChunk]:
        """Parse Tavily response directly to NewsChunk objects"""
        chunks = []
        
        results = response.get("results", [])
        
        for i, result in enumerate(results):
            try:
                # Extract basic fields
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")
                
                # Skip if no meaningful content
                word_count = len(content.split())
                if word_count < self.config.min_word_count:
                    continue
                
                # Parse published date
                published_date = self._parse_published_date(result.get("published_date"))
                
                # Extract metadata
                source = self._extract_source_from_url(url)
                score = result.get("score", 0.0)
                tavily_id = result.get("id", f"tavily_chunk_{i}")
                
                # Create comprehensive metadata
                metadata = {
                    # Article metadata
                    "title": title,
                    "source": source,
                    "author": None,
                    "published_date": published_date.isoformat() if published_date else None,
                    "url": url,
                    
                    # Company and category info
                    "companies_mentioned": [company],
                    "category": category.value,
                    
                    # Tavily-specific metadata
                    "tavily_id": tavily_id,
                    "tavily_score": score,
                    
                    # Processing metadata
                    "retrieval_timestamp": datetime.utcnow().isoformat(),
                    "processing_timestamp": datetime.utcnow().isoformat(),
                    
                    # RAG-specific fields
                    "content_type": "news_article",
                    "language": "en",
                    "domain": source,
                    "chunk_source": "tavily_direct"  # Indicate this came directly from Tavily
                }
                
                # Create chunk ID
                chunk_id = f"tavily_{hashlib.md5(f'{url}_{i}'.encode()).hexdigest()[:12]}_chunk_{i}"
                
                # Create NewsChunk object
                chunk = NewsChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata,
                    source_article_id=tavily_id,
                    source_url=url,
                    chunk_index=i,
                    total_chunks=1,  # Each Tavily result is treated as a standalone chunk
                    word_count=word_count,
                    char_count=len(content)
                )
                
                chunks.append(chunk)
                logger.debug(f"Created chunk: {chunk_id} ({word_count} words)")
                    
            except Exception as e:
                logger.warning(f"Error parsing chunk result: {e}")
                continue
        
        return chunks
    
    def _parse_published_date(self, date_str) -> Optional[datetime]:
        """Parse published date from various formats"""
        if not date_str:
            return datetime.utcnow()
            
        try:
            if isinstance(date_str, str):
                # Handle various date formats
                for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"]:
                    try:
                        return datetime.strptime(date_str.replace("Z", "+00:00"), fmt)
                    except:
                        continue
        except:
            pass
            
        return datetime.utcnow()
    
    def _extract_source_from_url(self, url: str) -> str:
        """Extract source domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except:
            return "unknown"
    
    def _deduplicate_chunks(self, chunks: List[NewsChunk]) -> List[NewsChunk]:
        """Remove duplicate chunks based on URL and content similarity"""
        seen_urls = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Use URL as primary deduplication key
            if chunk.source_url not in seen_urls:
                seen_urls.add(chunk.source_url)
                unique_chunks.append(chunk)
        
        return unique_chunks