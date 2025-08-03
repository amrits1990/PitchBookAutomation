import os
import json
import time
import uuid
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from secedgar import filings, FilingType
from metadata_extractor import MetadataExtractor
from content_processor import ContentProcessor
from chunk_generator import ChunkGenerator

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
content_processor = ContentProcessor()
metadata_extractor = MetadataExtractor()
chunk_generator = ChunkGenerator()
# Import with fallback for missing modules
try:
    from config_manager import get_config, validate_environment
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    print("Warning: Config manager not available, using basic configuration")
    CONFIG_MANAGER_AVAILABLE = False
    
    def get_config():
        class BasicConfig:
            def __init__(self):
                self.sec_user_agent = os.getenv('SEC_USER_AGENT', 'user@example.com')
                self.min_chunk_size = 100
                self.max_chunk_size = 5000
                self.min_overlap = 0
                self.max_overlap = 1000
                self.max_files_per_ticker = 150
        return BasicConfig()

# Proper SECFilingCleanerChunker with metadata extraction and content cleaning
class SECFilingCleanerChunker:
    """Full-featured SEC filing processor with metadata extraction and content cleaning"""
    
    def __init__(self):
        self.config = get_config()
    
    def extract_metadata_from_header(self, file_path: str, content: str) -> Dict:
        """Extract metadata from SEC filing header"""
        metadata = metadata_extractor.extract_filing_metadata(file_path, content)
        return metadata
    
    
    def create_rag_dataset_for_filing(self, file_path: str, output_dir: str = 'rag_ready_data', 
                                    chunk_size: int = 800, overlap: int = 150) -> Dict:
        """Create a complete RAG dataset from SEC filing using section-based processing"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()
            
            # Extract metadata from header
            filing_metadata = self.extract_metadata_from_header(file_path, raw_content)
            section_patterns = content_processor.get_section_patterns(filing_metadata.get('form_type'))
            
            all_chunks = []
            chunk_id_counter = 0
            
            print(f"Creating RAG dataset for: {filing_metadata.get('company_name', 'Unknown')} ({filing_metadata.get('ticker', 'N/A')})")
            
            # Process each section individually
            section_names = [item['name'] for item in section_patterns]
            
            for section_name in section_names:
                try:
                    section_data = content_processor.extract_section_content(file_path, section_name, filing_metadata)
                    
                    if section_data['metadata']['section_found'] and section_data['text']:
                        section_chunks = chunk_generator.create_rag_chunks(
                            section_data['text'],
                            section_data['metadata'],
                            chunk_size=chunk_size,
                            overlap=overlap
                        )
                        
                        # Update chunk IDs
                        for chunk in section_chunks:
                            chunk['global_chunk_id'] = chunk_id_counter
                            chunk['metadata']['global_chunk_id'] = chunk_id_counter
                            chunk_id_counter += 1
                        
                        all_chunks.extend(section_chunks)
                        print(f"  ✓ {section_name}: {len(section_chunks)} chunks")
                    
                except Exception as e:
                    print(f"  ✗ Failed processing {section_name}: {e}")
            
            # Process unclassified content
            try:
                print("Processing unclassified content...")
                unclassified_data = content_processor.extract_unclassified_content(file_path, filing_metadata)
                
                if unclassified_data['metadata']['section_found'] and unclassified_data['text']:
                    unclassified_chunks = chunk_generator.create_rag_chunks(
                        unclassified_data['text'],
                        unclassified_data['metadata'],
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                    
                    # Update chunk IDs
                    for chunk in unclassified_chunks:
                        chunk['global_chunk_id'] = chunk_id_counter
                        chunk['metadata']['global_chunk_id'] = chunk_id_counter
                        chunk_id_counter += 1
                    
                    all_chunks.extend(unclassified_chunks)
                    print(f"  ✓ Unclassified Content: {len(unclassified_chunks)} chunks")
                else:
                    print("  ℹ No unclassified content found")
                    
            except Exception as e:
                print(f"  ✗ Failed processing unclassified content: {e}")
            
            # Create final dataset
            result = {
                'filing_metadata': filing_metadata,
                'chunk_count': len(all_chunks),
                'chunks': all_chunks,
                'created_at': datetime.now().isoformat(),
                'chunk_settings': {
                    'chunk_size': chunk_size,
                    'overlap': overlap
                }
            }
            
            print(f"    Total chunks created: {len(all_chunks)}")
            
            return result
            
        except Exception as e:
            print(f"Error processing filing: {e}")
            return {
                'filing_metadata': {'error': str(e)},
                'chunk_count': 0,
                'chunks': [],
                'created_at': datetime.now().isoformat()
            }

# Rate limiting decorator
def rate_limit(calls_per_minute=10):
    """Rate limiting decorator to prevent API abuse"""
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
class AuditLogger:
    @staticmethod
    def log_request(correlation_id: str, action: str, details: Dict):
        """Log audit events with correlation ID"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'correlation_id': correlation_id,
            'action': action,
            'details': details
        }
        print(f"AUDIT: {json.dumps(log_entry)}")
    
    @staticmethod
    def log_error(correlation_id: str, error_type: str, message: str):
        """Log errors with correlation ID"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'correlation_id': correlation_id,
            'level': 'ERROR',
            'error_type': error_type,
            'message': message
        }
        print(f"ERROR: {json.dumps(log_entry)}")

def process_multiple_companies_filings(
    tickers: List[str],
    start_date: str,
    years_back: int = 3,
    filing_types: List[str] = None,
    chunk_size: int = 1000,
    overlap: int = 150,
    output_base_dir: str = None,
    cleanup_after_processing: bool = False,
    correlation_id: str = None  # Add correlation ID parameter
) -> Dict:
    """
    Process SEC filings for multiple companies and return all cleaned JSONs.
    
    Environment Variables Required:
        SEC_USER_AGENT: Valid email address for SEC API access
    
    Args:
        tickers: List of company ticker symbols (e.g., ['AAPL', 'MSFT']) - 1-5 letters each
        start_date: Start date in format 'YYYY-MM-DD'
        years_back: Number of years to go back (1-10, default: 3)
        filing_types: List of filing types (['10-K', '10-Q', '8-K'], default: ['10-K', '10-Q'])
        chunk_size: Size of text chunks for RAG (100-5000, default: 1000)
        overlap: Overlap between chunks (0 to chunk_size-1, default: 150)
        output_base_dir: Base directory for outputs (default: current directory)
    
    Returns:
        Dictionary containing:
        - success: List of successfully processed filings with their data
        - failures: List of failed processing attempts with error details
        - summary: Processing summary statistics
    
    Raises:
        ValueError: For invalid input parameters or missing environment variables
    """
    
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    # Audit log the request
    AuditLogger.log_request(correlation_id, 'process_filings_start', {
        'tickers': tickers,
        'start_date': start_date,
        'years_back': years_back,
        'filing_types': filing_types
    })
    
    # Validate configuration using config manager
    try:
        config = get_config()
    except Exception as e:
        AuditLogger.log_error(correlation_id, 'CONFIGURATION_ERROR', str(e))
        raise ValueError(f"Configuration error: {e}")
    
    # Check required environment variable
    if not config.sec_user_agent or config.sec_user_agent == 'user@example.com':
        AuditLogger.log_error(correlation_id, 'CONFIGURATION_ERROR', 'SEC_USER_AGENT not set')
        raise ValueError("SEC_USER_AGENT environment variable must be set to a valid email address")
    
    # Simple input validation
    if not tickers or not isinstance(tickers, list):
        raise ValueError("tickers must be a non-empty list")
    
    # Validate tickers (1-5 uppercase letters)
    for ticker in tickers:
        if not isinstance(ticker, str) or not ticker.isalpha() or not (1 <= len(ticker) <= 5):
            raise ValueError(f"Invalid ticker '{ticker}'. Must be 1-5 letters.")
    
    # Validate years_back
    if not isinstance(years_back, int) or not (1 <= years_back <= 10):
        raise ValueError("years_back must be between 1 and 10")
    
    # Validate chunk parameters using config manager
    if not isinstance(chunk_size, int) or not (config.min_chunk_size <= chunk_size <= config.max_chunk_size):
        raise ValueError(f"chunk_size must be between {config.min_chunk_size} and {config.max_chunk_size}")
    
    if not isinstance(overlap, int) or not (config.min_overlap <= overlap < chunk_size):
        raise ValueError(f"overlap must be between {config.min_overlap} and {chunk_size-1}")
    
    if filing_types is None:
        filing_types = ['10-K', '10-Q']
    
    # Validate filing types
    valid_filing_types = {'10-K', '10-Q', '8-K'}
    for filing_type in filing_types:
        if filing_type not in valid_filing_types:
            raise ValueError(f"Invalid filing type '{filing_type}'. Must be one of: {valid_filing_types}")
    
    if output_base_dir is None:
        output_base_dir = os.getcwd()
    
    # Parse dates
    try:
        end_dt = datetime.strptime(start_date, '%Y-%m-%d')  # This is the end of our date range
        start_dt = end_dt - timedelta(days=365 * years_back)  # This is the start of our date range
    except ValueError:
        raise ValueError("start_date must be in format 'YYYY-MM-DD'")
    
    # Initialize results with correlation ID
    results = {
        'success': [],
        'failures': [],
        'correlation_id': correlation_id,
        'summary': {
            'total_tickers': len(tickers),
            'total_filings_processed': 0,
            'total_filings_failed': 0,
            'processing_start_time': datetime.now().isoformat(),
            'correlation_id': correlation_id,
            'parameters': {
                'tickers': tickers,
                'start_date': start_date,
                'years_back': years_back,
                'filing_types': filing_types,
                'chunk_size': chunk_size,
                'overlap': overlap
            }
        }
    }
    
    # Initialize cleaner
    cleaner = SECFilingCleanerChunker()
    
    print(f"Starting batch processing for {len(tickers)} companies...")
    print(f"Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    print(f"Filing types: {filing_types}")
    
    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Processing {ticker}")
        print(f"{'='*50}")
        
        ticker_results = process_single_company(
            ticker=ticker,
            start_dt=start_dt,
            end_dt=end_dt,
            filing_types=filing_types,
            cleaner=cleaner,
            chunk_size=chunk_size,
            overlap=overlap,
            output_base_dir=output_base_dir,
            correlation_id=correlation_id
        )
        
        # Add ticker results to overall results
        results['success'].extend(ticker_results['success'])
        results['failures'].extend(ticker_results['failures'])
    
    # Update summary
    results['summary']['total_filings_processed'] = len(results['success'])
    results['summary']['total_filings_failed'] = len(results['failures'])
    results['summary']['processing_end_time'] = datetime.now().isoformat()
    
    # Save overall results in AnnualReportRAG folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    batch_results_file = os.path.join(script_dir, 'batch_processing_results.json')
    with open(batch_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total filings processed successfully: {results['summary']['total_filings_processed']}")
    print(f"Total filings failed: {results['summary']['total_filings_failed']}")
    print(f"Results saved to: {batch_results_file}")
    
    if cleanup_after_processing:
        print(f"\n{'='*60}")
        print("STARTING CLEANUP")
        print(f"{'='*60}")
        
        
        # Perform cleanup
        cleanup_results = SECFilingCleanerChunker.cleanup_after_batch_processing(
            batch_results_file=batch_results_file,
            base_dir=script_dir,  # Use AnnualReportRAG directory as base
            force_cleanup=False  # Only cleanup if processing was successful
        )
        
        # Add cleanup results to batch results
        results['cleanup_results'] = cleanup_results
        
        # Update the results file with cleanup info
        with open(batch_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        if cleanup_results.get('cleanup_performed'):
            print("✅ Cleanup completed successfully!")
        else:
            print(f"⚠️  Cleanup skipped: {cleanup_results.get('reason', 'Unknown reason')}")

    return results

@rate_limit(calls_per_minute=int(os.getenv('RATE_LIMIT_REQUESTS', '10')))  # Apply rate limiting to SEC API calls
def process_single_company(
    ticker: str,
    start_dt: datetime,
    end_dt: datetime,
    filing_types: List[str],
    cleaner: SECFilingCleanerChunker,
    chunk_size: int,
    overlap: int,
    output_base_dir: str,
    correlation_id: str
) -> Dict:
    """Process all filings for a single company"""
    
    # Get config for this function
    config = get_config()
    
    company_results = {'success': [], 'failures': []}
    
    # Create company-specific directories within AnnualReportRAG folder
    # Get the directory where this script is located (AnnualReportRAG folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filings_dir = os.path.join(script_dir, 'filings', ticker.lower())
    rag_dir = os.path.join(script_dir, 'rag_ready_data')
    os.makedirs(filings_dir, exist_ok=True)
    os.makedirs(rag_dir, exist_ok=True)
    
    for filing_type in filing_types:
        try:
            print(f"  Downloading {filing_type} filings for {ticker}...")
            
            # Download filings
            filing_type_mapping = {
                '10-K': FilingType.FILING_10K,
                '10-Q': FilingType.FILING_10Q,
                '8-K': FilingType.FILING_8K,
                # Add other types as needed
            }
            
            filing_type_enum = filing_type_mapping.get(filing_type)
            
            filing_downloader = filings(
                cik_lookup=[ticker],
                filing_type=filing_type_enum,
                start_date=start_dt,
                end_date=end_dt,
                user_agent=config.sec_user_agent,
                count=min(config.max_files_per_ticker, 150)  # Use config limits
            )
            
            # Create filing type directory
            type_dir = os.path.join(filings_dir, filing_type)
            os.makedirs(type_dir, exist_ok=True)
            
            # Download to directory
            filing_downloader.save(type_dir)
            
            # Process downloaded files
            downloaded_files = []
            for root, dirs, files in os.walk(type_dir):
                for file in files:
                    if file.endswith('.txt'):
                        downloaded_files.append(os.path.join(root, file))
            
            print(f"    Found {len(downloaded_files)} {filing_type} files")
            
            # Process each downloaded file
            for file_path in downloaded_files:
                try:
                    print(f"    Processing: {os.path.basename(file_path)}")
                    
                    # Create RAG dataset
                    rag_dataset = cleaner.create_rag_dataset_for_filing(
                        file_path=file_path,
                        output_dir=rag_dir,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                    
                    # Add to success results
                    filing_result = {
                        'ticker': ticker,
                        'filing_type': filing_type,
                        'file_path': file_path,
                        'rag_dataset': rag_dataset,
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    company_results['success'].append(filing_result)
                    print(f"    ✓ Successfully processed {os.path.basename(file_path)}")
                    
                except Exception as e:
                    AuditLogger.log_error(correlation_id, 'PROCESSING_ERROR', f"Failed to process {os.path.basename(file_path)}")
                    error_detail = {
                        'ticker': ticker,
                        'filing_type': filing_type,
                        'file_path': os.path.basename(file_path),  # Only basename for security
                        'error': 'Processing failed - see logs for details',  # Don't expose internal errors
                        'error_time': datetime.now().isoformat(),
                        'correlation_id': correlation_id
                    }
                    company_results['failures'].append(error_detail)
                    print(f"    ✗ Failed to process {os.path.basename(file_path)}: Processing error")
            
        except Exception as e:
            AuditLogger.log_error(correlation_id, 'NETWORK_ERROR', f"Failed to download {filing_type} for {ticker}")
            error_detail = {
                'ticker': ticker,
                'filing_type': filing_type,
                'error': f"Failed to download {filing_type} filings - network or server error",
                'error_time': datetime.now().isoformat(),
                'correlation_id': correlation_id
            }
            company_results['failures'].append(error_detail)
            print(f"  ✗ Failed to download {filing_type} filings for {ticker}: Network error")
    
    return company_results

def get_filing_chunks_api(
    tickers: List[str],
    start_date: str,
    years_back: int = 3,
    filing_types: List[str] = None,
    chunk_size: int = 800,
    overlap: int = 150,
    return_full_data: bool = False,
    output_dir: str = None,
    correlation_id: str = None
) -> Dict:
    """
    Main API function that can be called by third-party applications.
    
    Environment Variables Required:
        SEC_USER_AGENT: Valid email address for SEC API access
    
    Args:
        tickers: List of company ticker symbols (1-5 letters each)
        start_date: Start date in 'YYYY-MM-DD' format
        years_back: Number of years to look back (1-10)
        filing_types: Types of filings to process (['10-K', '10-Q', '8-K'])
        chunk_size: Size of text chunks for RAG (100-5000)
        overlap: Overlap between chunks (0 to chunk_size-1)
        return_full_data: If True, returns full RAG datasets in response
        output_dir: Directory to save files (optional)
    
    Returns:
        Dictionary with processing results and optionally full datasets
        On error: Dictionary with 'status': 'error' and error details
    """
    
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    try:
        # Process all filings
        results = process_multiple_companies_filings(
            tickers=tickers,
            start_date=start_date,
            years_back=years_back,
            filing_types=filing_types,
            chunk_size=chunk_size,
            overlap=overlap,
            output_base_dir=output_dir,
            correlation_id=correlation_id
        )
        
        # Prepare API response
        api_response = {
            'status': 'success',
            'correlation_id': correlation_id,
            'summary': results['summary'],
            'successful_filings': [],
            'failed_filings': results['failures']
        }
        
        # Process successful results
        for success in results['success']:
            filing_info = {
                'ticker': success['ticker'],
                'filing_type': success['filing_type'],
                'company_name': success['rag_dataset']['filing_metadata'].get('company_name'),
                'fiscal_year': success['rag_dataset']['filing_metadata'].get('fiscal_year'),
                'filing_date': success['rag_dataset']['filing_metadata'].get('filing_date'),
                'chunk_count': success['rag_dataset']['chunk_count'],
                'processed_at': success['processed_at']
            }
            
            # Include full dataset if requested
            if return_full_data:
                filing_info['rag_dataset'] = success['rag_dataset']
            
            api_response['successful_filings'].append(filing_info)
        
        return api_response
        
    except ValueError as e:
        # Input validation errors are safe to show
        AuditLogger.log_error(correlation_id, 'VALIDATION_ERROR', str(e))
        return {
            'status': 'error',
            'error_code': 'VALIDATION_ERROR',
            'message': str(e),
            'correlation_id': correlation_id,
            'error_time': datetime.now().isoformat()
        }
    except Exception as e:
        # Internal errors should not expose details
        AuditLogger.log_error(correlation_id, 'PROCESSING_ERROR', 'Internal processing error')
        return {
            'status': 'error',
            'error_code': 'PROCESSING_ERROR',
            'message': 'An error occurred during processing',
            'correlation_id': correlation_id,
            'error_time': datetime.now().isoformat()
        }

# Example usage
if __name__ == "__main__":
    # Example call
    tickers = ['WERN']  # Add more tickers as needed
    start_date = '2025-07-01'
    
    print("Example API call:")
    print(f"Processing tickers: {tickers}")
    print(f"Start date: {start_date}")
    print(f"Years back: 2")
    
    # Call the API
    results = get_filing_chunks_api(
        tickers=tickers,
        start_date=start_date,
        years_back=1,
        filing_types=['10-Q'],  # Start with just 10-K for testing
        return_full_data=False  # Set to True to get full datasets
    )
    
    print("\nAPI Response Summary:")
    print(f"Status: {results['status']}")
    if results['status'] == 'success':
        print(f"Successful filings: {len(results['successful_filings'])}")
        print(f"Failed filings: {len(results['failed_filings'])}")
        
        for filing in results['successful_filings']:
            print(f"  - {filing['ticker']} {filing['filing_type']} ({filing['fiscal_year']}) - {filing['chunk_count']} chunks")
    else:
        print(f"Error: {results.get('message', 'Unknown error')}")
        if 'error_code' in results:
            print(f"Error Code: {results['error_code']}")
