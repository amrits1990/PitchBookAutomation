"""
Main interface for the NewsRAG package
"""

import logging
import json
import time
import os
from typing import List, Dict, Any, Union, Optional
from datetime import datetime

from .models import NewsQuery, NewsCategory, NewsProcessingResult
from .config import NewsConfig
from .tavily_direct import TavilyDirectClient


logger = logging.getLogger(__name__)


def get_company_news_chunks(
    companies: Union[str, List[str]],
    categories: Union[str, List[str], List[NewsCategory]] = None,
    days_back: int = 30,
    max_articles_per_query: int = 10,
    config: Optional[NewsConfig] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to retrieve and process news for companies into RAG-ready chunks
    
    Args:
        companies: Company name(s) or ticker symbol(s)
        categories: News categories to search for
        days_back: Number of days to look back for news
        max_articles_per_query: Maximum articles per company/category combination
        config: Configuration object (uses default if None)
        output_file: Optional file path to save results as JSON
    
    Returns:
        Dictionary containing processing results and chunks
    """
    start_time = time.time()
    
    # Setup logging
    if config is None:
        config = NewsConfig.default()
    
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "newsrag_logs.txt")
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger.info("Starting NewsRAG processing")
    
    try:
        # Normalize inputs
        if isinstance(companies, str):
            companies = [companies]
        
        if categories is None:
            categories = [NewsCategory.GENERAL]
        elif isinstance(categories, str):
            categories = [NewsCategory(categories.lower())]
        elif isinstance(categories, list) and categories and isinstance(categories[0], str):
            categories = [NewsCategory(cat.lower()) for cat in categories]
        
        # Create query
        query = NewsQuery(
            companies=companies,
            categories=categories,
            days_back=days_back,
            max_articles_per_query=max_articles_per_query
        )
        
        logger.info(f"Processing news for companies: {query.companies}")
        logger.info(f"Categories: {[cat.value for cat in query.categories]}")
        logger.info(f"Date range: {days_back} days back")
        # Log the query text to the log file
        try:
            with open(log_file_path, "a", encoding="utf-8") as logf:
                logf.write(f"\n[QUERY] {str(query)}\n")
        except Exception as log_exc:
            logger.warning(f"Failed to write query to log file: {log_exc}")
        
        # Initialize simplified client
        client = TavilyDirectClient(config)
        
        # Create result object
        result = NewsProcessingResult(
            success=True,
            query=query,
            articles_retrieved=0,
            chunks_generated=0
        )
        
        # Single step: Retrieve news chunks directly from Tavily
        logger.info("Retrieving news chunks from Tavily...")
        chunks = client.search_news_chunks(query)
        
        if not chunks:
            logger.warning("No news chunks retrieved")
            result.success = False
            result.add_error("No news chunks found for the given query")
            return _format_result(result, start_time)
        
        logger.info(f"Retrieved {len(chunks)} ready-to-use chunks")
        
        # Add chunks to result
        for chunk in chunks:
            result.add_chunk(chunk)
        
        # Since Tavily returns chunks directly, we count chunks as "articles" for compatibility
        result.articles_retrieved = len(chunks)
        
        # Calculate summary statistics
        result.companies_covered = list(set([
            company for chunk in chunks 
            for company in chunk.metadata.get("companies_mentioned", [])
        ]))
        
        # Extract date range from chunks
        dates = []
        for chunk in chunks:
            pub_date_str = chunk.metadata.get("published_date")
            if pub_date_str:
                try:
                    dates.append(datetime.fromisoformat(pub_date_str.replace("Z", "+00:00")))
                except:
                    pass
        
        if dates:
            result.date_range = {
                "earliest": min(dates),
                "latest": max(dates)
            }
        
        processing_time = time.time() - start_time
        result.processing_time_seconds = processing_time
        
        logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
        logger.info(f"Generated {result.chunks_generated} chunks from {result.articles_retrieved} articles")
        
        # Format result
        formatted_result = _format_result(result, start_time)
        
        # Save to file if requested
        if output_file:
            _save_result_to_file(formatted_result, output_file)
            logger.info(f"Results saved to {output_file}")
        
        return formatted_result
        
    except Exception as e:
        error_msg = f"Error in news processing: {str(e)}"
        logger.error(error_msg)
        
        result = NewsProcessingResult(
            success=False,
            query=query if 'query' in locals() else None,
            articles_retrieved=0,
            chunks_generated=0
        )
        result.add_error(error_msg)
        
        return _format_result(result, start_time)


def _format_result(result: NewsProcessingResult, start_time: float) -> Dict[str, Any]:
    """Format result for return"""
    
    # Convert chunks to dictionary format (simplified since chunks are already ready)
    chunks_dict = [
        {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "source_article_id": chunk.source_article_id,
            "source_url": chunk.source_url,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "word_count": chunk.word_count,
            "char_count": chunk.char_count,
            "processed_timestamp": chunk.processed_timestamp.isoformat()
        }
        for chunk in result.chunks
    ]
    
    formatted_result = {
        "success": result.success,
        "processing_time_seconds": time.time() - start_time,
        "summary": {
            "articles_retrieved": result.articles_retrieved,
            "chunks_generated": result.chunks_generated,
            "total_word_count": result.total_word_count,
            "companies_covered": result.companies_covered,
            "date_range": {
                "earliest": result.date_range["earliest"].isoformat() if result.date_range and result.date_range["earliest"] else None,
                "latest": result.date_range["latest"].isoformat() if result.date_range and result.date_range["latest"] else None
            } if result.date_range else None
        },
        "query": {
            "companies": result.query.companies if result.query else [],
            "categories": [cat.value for cat in result.query.categories] if result.query else [],
            "days_back": result.query.days_back if result.query else 0,
            "max_articles_per_query": result.query.max_articles_per_query if result.query else 0
        } if result.query else {},
        "chunks": chunks_dict,
        "errors": result.errors,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return formatted_result


def _save_result_to_file(result: Dict[str, Any], file_path: str):
    """Save result to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving result to file {file_path}: {e}")
        raise