"""
Main API for transcript processing and chunking
Provides get_transcript_chunks function similar to get_filing_chunks
"""

import os
import json
import time
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from functools import wraps

# Import all the modules we created
try:
    from .data_source_interface import (
        TranscriptQuery, DataSourceError, transcript_registry
    )
    from .alpha_vantage_source import register_alpha_vantage_source
    from .transcript_metadata_extractor import TranscriptMetadataExtractor
    from .transcript_content_processor import TranscriptContentProcessor
    from .transcript_chunk_generator import TranscriptChunkGenerator
    from .transcript_config import (
        get_transcript_config, get_transcript_config_manager,
        validate_transcript_environment
    )
    from .simple_quarters_tracker import SimpleQuartersTracker
except ImportError:
    # Fallback for direct script execution
    from data_source_interface import (
        TranscriptQuery, DataSourceError, transcript_registry
    )
    from alpha_vantage_source import register_alpha_vantage_source
    from transcript_metadata_extractor import TranscriptMetadataExtractor
    from transcript_content_processor import TranscriptContentProcessor
    from transcript_chunk_generator import TranscriptChunkGenerator
    from transcript_config import (
        get_transcript_config, get_transcript_config_manager,
        validate_transcript_environment
    )
    from simple_quarters_tracker import SimpleQuartersTracker

# Load environment variables from TranscriptRAG .env file
try:
    from dotenv import load_dotenv
    from pathlib import Path
    # Load .env file from TranscriptRAG directory
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)
except ImportError:
    # dotenv not available, will use system environment variables
    pass


class TranscriptProcessor:
    """Main transcript processor orchestrating all components"""
    
    def __init__(self):
        self.config = get_transcript_config()
        self.metadata_extractor = TranscriptMetadataExtractor()
        self.content_processor = TranscriptContentProcessor()
        self.chunk_generator = TranscriptChunkGenerator()
        
        # Initialize data sources
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """Initialize and register available data sources"""
        try:
            # Register Alpha Vantage if API key is available
            if self.config.alpha_vantage_api_key:
                register_alpha_vantage_source(
                    api_key=self.config.alpha_vantage_api_key,
                    is_default=True
                )
        except Exception as e:
            print(f"Warning: Failed to initialize Alpha Vantage source: {e}")
    
    def process_transcript_data(self, transcript_data, chunk_size: int = 800, 
                               overlap: int = 150, use_speaker_chunking: bool = None) -> Dict:
        """
        Process a single transcript into chunks with metadata
        
        Args:
            transcript_data: TranscriptData object
            chunk_size: Size of chunks
            overlap: Overlap between chunks
            use_speaker_chunking: Whether to use speaker-aware chunking
            
        Returns:
            Dictionary with processed chunks and metadata
        """
        if use_speaker_chunking is None:
            use_speaker_chunking = self.config.enable_speaker_chunking
        
        # Extract enriched metadata
        enriched_metadata = self.metadata_extractor.extract_metadata(transcript_data)
        
        # Process content
        processed_content = self.content_processor.process_transcript_content(transcript_data)
        
        # Create section-based chunks
        section_chunks_data = self.content_processor.create_section_chunks_metadata(
            processed_content, transcript_data
        )
        
        all_chunks = []
        chunk_id_counter = 0
        
        # Process each section using specialized chunking strategies
        for section_data in section_chunks_data:
            section_text = section_data['text']
            section_metadata = section_data['metadata']
            section_name = section_metadata.get('section_name', '')
            
            if 'Q&A' in section_name:
                # Use Q&A-specific chunking that groups questions with answers
                section_chunks = self.chunk_generator.create_qa_grouped_chunks(
                    section_text, section_metadata, chunk_size=1200, overlap=0
                )
            elif 'Opening Remarks' in section_name:
                # Use no-overlap chunking for opening remarks (preserves JSON entry boundaries)
                section_chunks = self.chunk_generator.create_opening_remarks_chunks(
                    section_text, section_metadata, chunk_size=chunk_size
                )
            else:
                # Fallback to regular chunking for other sections
                section_chunks = self.chunk_generator.create_transcript_chunks(
                    section_text, section_metadata, chunk_size, overlap
                )
            
            # Update global chunk IDs
            for chunk in section_chunks:
                chunk['global_chunk_id'] = chunk_id_counter
                chunk['metadata']['global_chunk_id'] = chunk_id_counter
                chunk_id_counter += 1
            
            all_chunks.extend(section_chunks)
        
        # Create final dataset
        return {
            'transcript_metadata': enriched_metadata,
            'chunk_count': len(all_chunks),
            'chunks': all_chunks,
            'created_at': datetime.now().isoformat(),
            'processing_metadata': processed_content['processing_metadata'],
            'chunk_settings': {
                'chunk_size': chunk_size,
                'overlap': overlap,
                'use_speaker_chunking': use_speaker_chunking
            }
        }


# Rate limiting decorator
def rate_limit(calls_per_minute: int = 5):
    """Rate limiting decorator for API calls"""
    last_called = [0.0]
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < 60.0 / calls_per_minute:
                time.sleep((60.0 / calls_per_minute) - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Simple audit logger
class TranscriptAuditLogger:
    @staticmethod
    def log_request(correlation_id: str, action: str, details: Dict):
        """Log audit events with correlation ID"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'correlation_id': correlation_id,
            'action': action,
            'details': details,
            'service': 'transcript_processor'
        }
        print(f"TRANSCRIPT_AUDIT: {json.dumps(log_entry)}")
    
    @staticmethod
    def log_error(correlation_id: str, error_type: str, message: str):
        """Log errors with correlation ID"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'correlation_id': correlation_id,
            'level': 'ERROR',
            'error_type': error_type,
            'message': message,
            'service': 'transcript_processor'
        }
        print(f"TRANSCRIPT_ERROR: {json.dumps(log_entry)}")


@rate_limit(calls_per_minute=5)  # Match Alpha Vantage free tier limits
def process_single_ticker_transcripts(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    processor: TranscriptProcessor,
    chunk_size: int,
    overlap: int,
    limit: Optional[int],
    correlation_id: str,
    expected_quarters: Optional[List[str]] = None
) -> Dict:
    """Process transcripts for a single ticker with intelligent caching"""
    
    ticker_results = {'success': [], 'failures': []}
    
    # Get configuration to check if intelligent caching is enabled
    config = get_transcript_config()
    
    try:
        # Initialize simple quarters tracker
        quarters_tracker = SimpleQuartersTracker()
        
        # Use expected quarters if provided (fiscal-aware), otherwise fall back to old logic
        if expected_quarters:
            print(f"  📋 Using fiscal-aware quarters: {expected_quarters}")
            
            # Convert fiscal quarters format (2025Q4) to simple format (Q4 2025) for quarters tracker
            simple_format_quarters = []
            for fq in expected_quarters:
                if 'Q' in fq:
                    year_str, quarter_str = fq.split('Q')
                    simple_format_quarters.append(f"Q{quarter_str} {year_str}")
            
            # Check which quarters are already ingested
            ingested_quarters = quarters_tracker.get_ingested_quarters(ticker)
            quarters_to_fetch = [q for q in simple_format_quarters if q not in ingested_quarters]
            
            print(f"  🔍 Expected quarters: {simple_format_quarters}")
            print(f"  ✅ Already ingested: {ingested_quarters}")
            print(f"  📥 Missing quarters: {quarters_to_fetch}")
            
        else:
            # Calculate quarters_back from date range 
            days_diff = (end_date - start_date).days
            quarters_back = max(1, days_diff // 90)  # ~90 days per quarter
            
            # Check which quarters need to be fetched (old logic)
            quarters_to_fetch = quarters_tracker.generate_quarters_to_fetch(ticker, quarters_back)
        
        if not quarters_to_fetch:
            total_quarters = len(expected_quarters) if expected_quarters else quarters_back
            print(f"  ✅ All {total_quarters} quarters already ingested for {ticker}")
            print(f"  🚀 Skipping API calls - no new quarters to fetch")
            
            # Instead of returning empty results, create cache hit response
            # Check what quarters are available from cache
            try:
                # Get the quarters that were expected to be cached
                cached_quarters = expected_quarters if expected_quarters else []
                if not cached_quarters:
                    # Generate quarters based on quarters_back if expected_quarters not available
                    current_date = datetime.now()
                    cached_quarters = []
                    for i in range(quarters_back):
                        # Go back i quarters (3 months each)
                        quarter_date = current_date - timedelta(days=i * 90)
                        quarter_num = ((quarter_date.month - 1) // 3) + 1
                        cached_quarters.append(f"Q{quarter_num} {quarter_date.year}")
                
                # Create cache hit responses for each cached quarter
                cache_hit_results = []
                for quarter_str in cached_quarters:
                    # Parse quarter string for metadata
                    parts = quarter_str.split()
                    if len(parts) == 2:
                        quarter_part = parts[0]  # "Q1" 
                        year_part = parts[1]     # "2025"
                        quarter_num = quarter_part[1:] if quarter_part.startswith('Q') else "1"
                        
                        cache_hit_result = {
                            'ticker': ticker.upper(),
                            'quarter': quarter_num,
                            'fiscal_year': year_part,
                            'transcript_date': f"{year_part}-{(int(quarter_num)-1)*3+3:02d}-15T00:00:00",  # Approximate date
                            'transcript_type': 'earnings_call',
                            'processed_at': datetime.now().isoformat(),
                            'source': 'cache_hit',
                            'transcript_dataset': {
                                'chunks': [],  # We don't need to load actual chunks for cache hit
                                'chunk_count': 0,  # Will be updated by vector store if needed
                                'metadata': {
                                    'source': 'vector_database_cache',
                                    'ticker': ticker.upper(),
                                    'quarter': quarter_num,
                                    'fiscal_year': year_part
                                }
                            }
                        }
                        cache_hit_results.append(cache_hit_result)
                
                # Add cache hit results to ticker_results
                ticker_results['success'].extend(cache_hit_results)
                print(f"  📊 Created cache hit response for {len(cache_hit_results)} quarters")
                
            except Exception as cache_error:
                print(f"  ⚠️ Error creating cache hit response: {cache_error}")
            
            return ticker_results
        
        print(f"  🎯 Need to fetch {len(quarters_to_fetch)} quarters: {quarters_to_fetch}")
        
        # Get data source
        data_source = transcript_registry.get_source()
        
        # Process each quarter individually
        successful_quarters = []
        
        for quarter_str in quarters_to_fetch:
            try:
                print(f"    📥 Fetching transcript for {quarter_str}...")
                
                # Parse quarter string (e.g., "Q1 2025" -> quarter="1", year="2025")
                parts = quarter_str.split()
                if len(parts) != 2:
                    print(f"      ❌ Invalid quarter format: {quarter_str}")
                    continue
                
                quarter_part = parts[0]  # "Q1"
                year_part = parts[1]     # "2025"
                
                if not quarter_part.startswith('Q') or not quarter_part[1:].isdigit():
                    print(f"      ❌ Invalid quarter format: {quarter_part}")
                    continue
                
                quarter_num = quarter_part[1:]  # "1"
                
                # Create Alpha Vantage query for specific quarter
                params = {
                    'function': 'EARNINGS_CALL_TRANSCRIPT',
                    'symbol': ticker.upper(),
                    'quarter': f"{year_part}Q{quarter_num}"  # "2025Q1"
                }
                
                # Make direct API call
                api_data = data_source._make_request(params)
                transcript_data = data_source._parse_transcript_data(api_data, ticker, f"{year_part}Q{quarter_num}")
                
                if transcript_data:
                    print(f"      ✓ Processing transcript data...")
                    
                    # Process transcript into chunks
                    processed_result = processor.process_transcript_data(
                        transcript_data=transcript_data,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                    
                    # Add to success results
                    result_data = {
                        'ticker': ticker,
                        'transcript_date': transcript_data.transcript_date.isoformat(),
                        'quarter': transcript_data.quarter,
                        'fiscal_year': transcript_data.fiscal_year,
                        'transcript_type': transcript_data.transcript_type,
                        'transcript_dataset': processed_result,
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    ticker_results['success'].append(result_data)
                    successful_quarters.append(quarter_str)
                    
                    print(f"      ✓ Successfully processed {quarter_str}")
                else:
                    print(f"      ❌ No transcript data found for {quarter_str}")
                    
            except Exception as e:
                print(f"      ❌ Failed to process {quarter_str}: {str(e)}")
                error_detail = {
                    'ticker': ticker,
                    'quarter_string': quarter_str,
                    'error': f'Processing failed: {str(e)}',
                    'error_time': datetime.now().isoformat(),
                    'correlation_id': correlation_id
                }
                ticker_results['failures'].append(error_detail)
        
        # Mark successful quarters as ingested
        if successful_quarters:
            success = quarters_tracker.add_multiple_quarters(ticker, successful_quarters)
            if success:
                print(f"    💾 Updated ingestion tracking: {len(successful_quarters)} quarters")
            else:
                print(f"    ⚠️ Failed to update ingestion tracking")
        
        print(f"  🎉 Ingestion complete: {len(successful_quarters)} quarters processed")
        
    except DataSourceError as e:
        TranscriptAuditLogger.log_error(
            correlation_id, 'DATA_SOURCE_ERROR', 
            f"Failed to fetch transcripts for {ticker}"
        )
        error_detail = {
            'ticker': ticker,
            'error': f"Failed to fetch transcripts: {e.error_code}",
            'error_time': datetime.now().isoformat(),
            'correlation_id': correlation_id
        }
        ticker_results['failures'].append(error_detail)
        print(f"  ✗ Failed to fetch transcripts for {ticker}: {e.error_code}")
    
    except Exception as e:
        TranscriptAuditLogger.log_error(
            correlation_id, 'NETWORK_ERROR', 
            f"Unexpected error for {ticker}"
        )
        error_detail = {
            'ticker': ticker,
            'error': f"Unexpected error occurred",
            'error_time': datetime.now().isoformat(),
            'correlation_id': correlation_id
        }
        ticker_results['failures'].append(error_detail)
        print(f"  ✗ Unexpected error for {ticker}")
    
    return ticker_results


def get_transcript_chunks(
    tickers: List[str],
    start_date: str,
    quarters_back: int = 4,
    chunk_size: int = 800,
    overlap: int = 150,
    limit_per_ticker: Optional[int] = None,
    use_speaker_chunking: bool = None,
    return_full_data: bool = False,
    output_dir: str = None,
    correlation_id: str = None,
    expected_quarters: Optional[List[str]] = None
) -> Dict:
    """
    Main API function to get transcript chunks for multiple tickers
    
    Environment Variables Required:
        ALPHA_VANTAGE_API_KEY: Valid Alpha Vantage API key
    
    Args:
        tickers: List of company ticker symbols (1-5 letters each)
        start_date: Start date in 'YYYY-MM-DD' format
        quarters_back: Number of quarters to look back (1-20)
        chunk_size: Size of text chunks for RAG (100-2000)
        overlap: Overlap between chunks (0 to chunk_size-1)
        limit_per_ticker: Maximum transcripts per ticker (optional)
        use_speaker_chunking: Enable speaker-aware chunking for Q&A sections
        return_full_data: If True, returns full datasets in response
        output_dir: Directory to save files (optional)
        correlation_id: Request correlation ID for tracking
    
    Returns:
        Dictionary with processing results and optionally full datasets
        On error: Dictionary with 'status': 'error' and error details
    """
    
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    
    try:
        # Validate configuration
        config = get_transcript_config()
        config_manager = get_transcript_config_manager()
        
        if not config_manager.validate_config():
            raise ValueError("Transcript configuration validation failed")
        
        # Validate input parameters
        if not tickers or not isinstance(tickers, list):
            raise ValueError("tickers must be a non-empty list")
        
        if len(tickers) > config.max_tickers_per_request:
            raise ValueError(f"Too many tickers. Maximum allowed: {config.max_tickers_per_request}")
        
        # Validate tickers
        for ticker in tickers:
            if not isinstance(ticker, str) or not ticker.isalpha() or not (1 <= len(ticker) <= 5):
                raise ValueError(f"Invalid ticker '{ticker}'. Must be 1-5 letters.")
        
        # Validate processing parameters
        if not config_manager.validate_processing_params(chunk_size, overlap, quarters_back):
            raise ValueError("Invalid processing parameters")
        
        # Parse dates properly - start_date is the actual start date, not end date
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            # End date is current date (we're looking for transcripts from start_date to now)
            end_dt = datetime.now()
        except ValueError:
            raise ValueError("start_date must be in format 'YYYY-MM-DD'")
        
        # Initialize processor
        processor = TranscriptProcessor()
        
        # Initialize results
        results = {
            'success': [],
            'failures': [],
            'correlation_id': correlation_id,
            'summary': {
                'total_tickers': len(tickers),
                'total_transcripts_processed': 0,
                'total_transcripts_failed': 0,
                'processing_start_time': datetime.now().isoformat(),
                'correlation_id': correlation_id,
                'parameters': {
                    'tickers': tickers,
                    'start_date': start_date,
                    'quarters_back': quarters_back,
                    'chunk_size': chunk_size,
                    'overlap': overlap,
                    'use_speaker_chunking': use_speaker_chunking
                }
            }
        }
        
        print(f"Starting transcript processing for {len(tickers)} companies...")
        print(f"Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        
        # Process each ticker
        for ticker in tickers:
            print(f"\n{'='*50}")
            print(f"Processing {ticker}")
            print(f"{'='*50}")
            
            ticker_results = process_single_ticker_transcripts(
                ticker=ticker.upper(),
                start_date=start_dt,
                end_date=end_dt,
                processor=processor,
                chunk_size=chunk_size,
                overlap=overlap,
                limit=limit_per_ticker,
                correlation_id=correlation_id,
                expected_quarters=expected_quarters
            )
            
            # Add ticker results to overall results
            results['success'].extend(ticker_results['success'])
            results['failures'].extend(ticker_results['failures'])
        
        # Update summary
        results['summary']['total_transcripts_processed'] = len(results['success'])
        results['summary']['total_transcripts_failed'] = len(results['failures'])
        results['summary']['processing_end_time'] = datetime.now().isoformat()
        
        # Save results if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            results_file = os.path.join(output_dir, 'transcript_processing_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            print(f"Results saved to: {results_file}")
        
        print(f"\n{'='*60}")
        print(f"TRANSCRIPT PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total transcripts processed successfully: {results['summary']['total_transcripts_processed']}")
        print(f"Total transcripts failed: {results['summary']['total_transcripts_failed']}")
        
        # Prepare API response
        api_response = {
            'status': 'success',
            'correlation_id': correlation_id,
            'summary': results['summary'],
            'successful_transcripts': [],
            'failed_transcripts': results['failures']
        }
        
        # Process successful results
        for success in results['success']:
            transcript_info = {
                'ticker': success['ticker'],
                'transcript_date': success['transcript_date'],
                'quarter': success['quarter'],
                'fiscal_year': success['fiscal_year'],
                'transcript_type': success['transcript_type'],
                'chunk_count': success['transcript_dataset']['chunk_count'],
                'processed_at': success['processed_at']
            }
            
            # Include source if it exists (for cache hit tracking)
            if 'source' in success:
                transcript_info['source'] = success['source']
            
            # Include full dataset if requested
            if return_full_data:
                transcript_info['transcript_dataset'] = success['transcript_dataset']
            
            api_response['successful_transcripts'].append(transcript_info)
        
        return api_response
        
    except ValueError as e:
        # Input validation errors are safe to show
        TranscriptAuditLogger.log_error(correlation_id, 'VALIDATION_ERROR', str(e))
        return {
            'status': 'error',
            'error_code': 'VALIDATION_ERROR',
            'message': str(e),
            'correlation_id': correlation_id,
            'error_time': datetime.now().isoformat()
        }
    except Exception as e:
        # Internal errors should not expose details
        TranscriptAuditLogger.log_error(correlation_id, 'PROCESSING_ERROR', 'Internal processing error')
        return {
            'status': 'error',
            'error_code': 'PROCESSING_ERROR',
            'message': 'An error occurred during transcript processing',
            'correlation_id': correlation_id,
            'error_time': datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    # Example call
    tickers = ['AAPL']  # Add more tickers as needed
    start_date = '2025-03-01'
    
    print("Example Transcript API call:")
    print(f"Processing tickers: {tickers}")
    print(f"Start date: {start_date}")
    print(f"Quarters back: 4")
    
    # Call the API
    results = get_transcript_chunks(
        tickers=tickers,
        start_date=start_date,
        quarters_back=4,
        return_full_data=False,  # Set to True to get full datasets
        output_dir= './transcript_results',  # Specify output directory
    )
    
    print("\nAPI Response Summary:")
    print(f"Status: {results['status']}")
    if results['status'] == 'success':
        print(f"Successful transcripts: {len(results['successful_transcripts'])}")
        print(f"Failed transcripts: {len(results['failed_transcripts'])}")
        
        for transcript in results['successful_transcripts']:
            print(f"  - {transcript['ticker']} {transcript['quarter']} {transcript['fiscal_year']} - {transcript['chunk_count']} chunks")
    else:
        print(f"Error: {results.get('message', 'Unknown error')}")
        if 'error_code' in results:
            print(f"Error Code: {results['error_code']}")