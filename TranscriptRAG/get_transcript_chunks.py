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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try loading from current directory first, then parent directory
    load_dotenv()
    load_dotenv('../.env')  # Load from parent directory where .env exists
except ImportError:
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
        
        # Process each section
        for section_data in section_chunks_data:
            section_text = section_data['text']
            section_metadata = section_data['metadata']
            
            if use_speaker_chunking and 'Q&A' in section_metadata.get('section_name', ''):
                # Use speaker-aware chunking for Q&A sections
                section_chunks = self.chunk_generator.create_speaker_aware_chunks(
                    section_text, section_metadata, chunk_size, overlap
                )
            else:
                # Use regular chunking
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
    correlation_id: str
) -> Dict:
    """Process transcripts for a single ticker"""
    
    ticker_results = {'success': [], 'failures': []}
    
    try:
        # Get data source
        data_source = transcript_registry.get_source()
        
        # Create query
        query = TranscriptQuery(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        # Fetch transcripts
        print(f"  Fetching transcripts for {ticker}...")
        transcripts = data_source.get_transcripts(query)
        
        print(f"    Found {len(transcripts)} transcripts")
        
        # Process each transcript
        for transcript_data in transcripts:
            try:
                print(f"    Processing: {transcript_data.quarter} {transcript_data.fiscal_year}")
                
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
                print(f"    ✓ Successfully processed {transcript_data.quarter} {transcript_data.fiscal_year}")
                
            except Exception as e:
                TranscriptAuditLogger.log_error(
                    correlation_id, 'PROCESSING_ERROR', 
                    f"Failed to process transcript for {ticker}"
                )
                error_detail = {
                    'ticker': ticker,
                    'transcript_date': transcript_data.transcript_date.isoformat(),
                    'quarter': transcript_data.quarter,
                    'error': 'Processing failed - see logs for details',
                    'error_time': datetime.now().isoformat(),
                    'correlation_id': correlation_id
                }
                ticker_results['failures'].append(error_detail)
                print(f"    ✗ Failed to process {transcript_data.quarter} {transcript_data.fiscal_year}")
        
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
    years_back: int = 3,
    chunk_size: int = 800,
    overlap: int = 150,
    limit_per_ticker: Optional[int] = None,
    use_speaker_chunking: bool = None,
    return_full_data: bool = False,
    output_dir: str = None,
    correlation_id: str = None
) -> Dict:
    """
    Main API function to get transcript chunks for multiple tickers
    
    Environment Variables Required:
        ALPHA_VANTAGE_API_KEY: Valid Alpha Vantage API key
    
    Args:
        tickers: List of company ticker symbols (1-5 letters each)
        start_date: Start date in 'YYYY-MM-DD' format
        years_back: Number of years to look back (1-10)
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
    
    # Audit log the request
    TranscriptAuditLogger.log_request(correlation_id, 'get_transcript_chunks_start', {
        'tickers': tickers,
        'start_date': start_date,
        'years_back': years_back,
        'chunk_size': chunk_size,
        'overlap': overlap
    })
    
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
        if not config_manager.validate_processing_params(chunk_size, overlap, years_back):
            raise ValueError("Invalid processing parameters")
        
        # Parse dates
        try:
            end_dt = datetime.strptime(start_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=365 * years_back)
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
                    'years_back': years_back,
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
                correlation_id=correlation_id
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
    print(f"Years back: 2")
    
    # Call the API
    results = get_transcript_chunks(
        tickers=tickers,
        start_date=start_date,
        years_back=1,
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