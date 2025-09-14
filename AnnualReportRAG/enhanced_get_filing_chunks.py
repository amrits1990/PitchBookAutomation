"""
Enhanced get_filing_chunks with intelligent filing management
Integrates with filing_manager to handle date-aware cleanup and processing
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

try:
    from .get_filing_chunks import get_filing_chunks_api as original_get_filing_chunks
    from .filing_manager import auto_cleanup_before_processing, FilingManager
except ImportError:
    from get_filing_chunks import get_filing_chunks_api as original_get_filing_chunks
    from filing_manager import auto_cleanup_before_processing, FilingManager

logger = logging.getLogger(__name__)

def get_filing_chunks_with_cleanup(
    tickers: List[str],
    start_date: str,
    years_back: int = 2,
    filing_types: Optional[List[str]] = None,
    chunk_size: int = 800,
    overlap: int = 150,
    return_full_data: bool = False,
    output_dir: Optional[str] = None,
    correlation_id: Optional[str] = None,
    auto_cleanup: bool = True,
    cleanup_older_than_years: int = 3
) -> Dict[str, Any]:
    """
    Enhanced filing chunks retrieval with intelligent cleanup
    
    Args:
        tickers: List of company ticker symbols
        start_date: Start date in YYYY-MM-DD format
        years_back: Years to look back for analysis
        filing_types: Types of filings to retrieve (defaults to ['10-K', '10-Q'])
        chunk_size: Size of text chunks (100-2000)
        overlap: Overlap between chunks (50-500)
        return_full_data: Whether to include full datasets
        output_dir: Output directory for processed files
        correlation_id: Request tracking ID
        auto_cleanup: Whether to clean old filings automatically
        cleanup_older_than_years: Clean filings older than this many years
        
    Returns:
        Enhanced results with cleanup statistics
    """
    
    filing_types = filing_types or ['10-K', '10-Q']
    
    logger.info(f"Enhanced filing retrieval: tickers={tickers}, years_back={years_back}, "
               f"auto_cleanup={auto_cleanup}, cleanup_older_than_years={cleanup_older_than_years}")
    
    # Step 1: Perform auto-cleanup for each ticker if enabled
    cleanup_results = {}
    total_files_cleaned = 0
    total_size_cleaned_mb = 0
    
    if auto_cleanup:
        logger.info("Performing auto-cleanup of old filings...")
        
        for ticker in tickers:
            try:
                valid_files, cleanup_stats = auto_cleanup_before_processing(
                    ticker=ticker,
                    years_back=cleanup_older_than_years,  # Clean older than specified years
                    filing_types=filing_types,
                    cleanup_enabled=True
                )
                
                cleanup_results[ticker] = cleanup_stats
                
                # Aggregate cleanup statistics
                if cleanup_stats.get('cleanup_performed'):
                    total_files_cleaned += cleanup_stats.get('files_deleted', 0)
                    total_size_cleaned_mb += cleanup_stats.get('total_size_to_delete_mb', 0)
                
                logger.info(f"Cleanup for {ticker}: "
                           f"{cleanup_stats.get('files_deleted', 0)} files deleted, "
                           f"{cleanup_stats.get('valid_files_for_processing', 0)} files valid for processing")
                
            except Exception as e:
                logger.warning(f"Cleanup failed for {ticker}: {e}")
                cleanup_results[ticker] = {'error': str(e), 'cleanup_performed': False}
    
    # Step 2: Call original filing retrieval function
    logger.info("Calling original filing retrieval API...")
    
    try:
        # Call the original function
        original_result = original_get_filing_chunks(
            tickers=tickers,
            start_date=start_date,
            years_back=years_back,
            filing_types=filing_types,
            chunk_size=chunk_size,
            overlap=overlap,
            return_full_data=return_full_data,
            output_dir=output_dir,
            correlation_id=correlation_id
        )
        
        # Enhance the result with cleanup information
        if isinstance(original_result, dict):
            enhanced_result = original_result.copy()
            
            # Add cleanup metadata
            enhanced_result['cleanup_metadata'] = {
                'auto_cleanup_enabled': auto_cleanup,
                'cleanup_older_than_years': cleanup_older_than_years,
                'total_files_cleaned': total_files_cleaned,
                'total_size_cleaned_mb': round(total_size_cleaned_mb, 2),
                'per_ticker_cleanup': cleanup_results,
                'cleanup_performed_at': datetime.now().isoformat()
            }
            
            # Add enhancement flag
            enhanced_result['enhanced_processing'] = True
            enhanced_result['filing_management_version'] = '1.0'
            
            logger.info(f"Enhanced processing complete: "
                       f"{total_files_cleaned} old files cleaned ({total_size_cleaned_mb:.2f} MB)")
            
            return enhanced_result
        else:
            # If original result is not a dict, wrap it
            return {
                'status': 'success',
                'original_result': original_result,
                'cleanup_metadata': {
                    'auto_cleanup_enabled': auto_cleanup,
                    'total_files_cleaned': total_files_cleaned,
                    'per_ticker_cleanup': cleanup_results
                },
                'enhanced_processing': True
            }
    
    except Exception as e:
        logger.error(f"Enhanced filing retrieval failed: {e}")
        return {
            'status': 'error',
            'message': f"Enhanced filing retrieval failed: {str(e)}",
            'cleanup_metadata': {
                'auto_cleanup_enabled': auto_cleanup,
                'cleanup_results': cleanup_results
            },
            'enhanced_processing': True
        }

def analyze_filing_storage(ticker: str = None) -> Dict[str, Any]:
    """
    Analyze current filing storage and provide recommendations
    
    Args:
        ticker: Specific ticker to analyze (if None, analyzes all tickers)
        
    Returns:
        Analysis results and cleanup recommendations
    """
    
    logger.info(f"Analyzing filing storage for ticker: {ticker or 'ALL'}")
    
    manager = FilingManager()
    base_dir = manager.base_dir
    
    if not base_dir.exists():
        return {
            'status': 'no_filings',
            'message': 'No filings directory found',
            'recommendations': ['No action needed - no filings stored locally']
        }
    
    # Analyze specific ticker or all tickers
    analysis_results = {}
    total_size_mb = 0
    total_filings = 0
    recommendations = []
    
    if ticker:
        # Analyze specific ticker
        tickers_to_analyze = [ticker.lower()]
    else:
        # Find all ticker directories
        tickers_to_analyze = []
        for item in base_dir.iterdir():
            if item.is_dir():
                tickers_to_analyze.append(item.name)
    
    for ticker_dir_name in tickers_to_analyze:
        ticker_upper = ticker_dir_name.upper()
        stats = manager.get_ticker_filing_stats(ticker_upper)
        
        if stats['total_filings'] > 0:
            analysis_results[ticker_upper] = stats
            total_size_mb += stats['total_size_mb']
            total_filings += stats['total_filings']
            
            # Generate recommendations for this ticker
            if stats['total_filings'] > 20:
                recommendations.append(f"{ticker_upper}: Consider cleaning old filings "
                                     f"({stats['total_filings']} files, {stats['total_size_mb']:.2f} MB)")
            
            # Check date range
            if stats['oldest_filing'] and stats['newest_filing']:
                oldest = datetime.fromisoformat(stats['oldest_filing'])
                if oldest < datetime.now() - timedelta(days=3*365):  # Older than 3 years
                    recommendations.append(f"{ticker_upper}: Has filings older than 3 years "
                                         f"(oldest: {oldest.strftime('%Y-%m-%d')})")
    
    # Overall recommendations
    if total_size_mb > 100:  # Over 100 MB
        recommendations.append(f"Overall: Consider cleanup - {total_size_mb:.2f} MB total storage used")
    
    if total_filings > 100:
        recommendations.append(f"Overall: Large number of filings ({total_filings}) - consider automated cleanup")
    
    if not recommendations:
        recommendations.append("No immediate cleanup needed - filing storage looks healthy")
    
    return {
        'status': 'success',
        'analysis_date': datetime.now().isoformat(),
        'total_tickers': len(analysis_results),
        'total_filings': total_filings,
        'total_size_mb': round(total_size_mb, 2),
        'per_ticker_stats': analysis_results,
        'recommendations': recommendations,
        'cleanup_commands': [
            f"# Clean old filings for specific ticker:",
            f"from AnnualReportRAG.filing_manager import clean_old_filings_for_ticker",
            f"clean_old_filings_for_ticker('TICKER', years_back=2, dry_run=True)",
            f"",
            f"# Use enhanced filing retrieval with auto-cleanup:",
            f"from AnnualReportRAG.enhanced_get_filing_chunks import get_filing_chunks_with_cleanup",
            f"result = get_filing_chunks_with_cleanup(['TICKER'], '2023-01-01', auto_cleanup=True)"
        ]
    }

# Backward compatibility wrapper
def get_filing_chunks_api(*args, **kwargs):
    """Backward compatible API that uses enhanced version by default"""
    
    # Check if enhanced processing is explicitly disabled
    disable_enhanced = kwargs.pop('disable_enhanced', False)
    auto_cleanup = kwargs.pop('auto_cleanup', True)
    
    if disable_enhanced:
        # Use original function
        return original_get_filing_chunks(*args, **kwargs)
    else:
        # Use enhanced version with cleanup
        return get_filing_chunks_with_cleanup(*args, auto_cleanup=auto_cleanup, **kwargs)

# Main entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "analyze":
            ticker = sys.argv[2] if len(sys.argv) > 2 else None
            analysis = analyze_filing_storage(ticker)
            
            print("\n=== FILING STORAGE ANALYSIS ===")
            print(f"Total tickers: {analysis['total_tickers']}")
            print(f"Total filings: {analysis['total_filings']}")
            print(f"Total size: {analysis['total_size_mb']} MB")
            print("\n--- Recommendations ---")
            for rec in analysis['recommendations']:
                print(f"• {rec}")
            
            if analysis['per_ticker_stats']:
                print("\n--- Per-Ticker Details ---")
                for ticker, stats in analysis['per_ticker_stats'].items():
                    print(f"{ticker}: {stats['total_filings']} filings, "
                          f"{stats['total_size_mb']} MB, "
                          f"range: {stats['date_range'] or 'unknown'}")
        
        elif command == "cleanup":
            ticker = sys.argv[2] if len(sys.argv) > 2 else None
            years_back = int(sys.argv[3]) if len(sys.argv) > 3 else 2
            dry_run = "--dry-run" in sys.argv
            
            if not ticker:
                print("Usage: python enhanced_get_filing_chunks.py cleanup TICKER [YEARS_BACK] [--dry-run]")
                sys.exit(1)
            
            from filing_manager import clean_old_filings_for_ticker
            result = clean_old_filings_for_ticker(ticker, years_back, dry_run=dry_run)
            
            print(f"\n=== CLEANUP RESULTS FOR {ticker.upper()} ===")
            print(f"Files to delete: {result['files_to_delete']}")
            print(f"Files to keep: {result['files_to_keep']}")
            print(f"Size to delete: {result['total_size_to_delete_mb']} MB")
            print(f"Dry run: {result['dry_run']}")
            
            if not dry_run:
                print(f"Files deleted: {result['files_deleted']}")
                print(f"Deletion errors: {result['deletion_errors']}")
        
        else:
            print("Available commands:")
            print("  analyze [TICKER]    - Analyze filing storage")
            print("  cleanup TICKER [YEARS_BACK] [--dry-run]  - Clean old filings")
    else:
        print("Enhanced filing chunks retrieval with auto-cleanup")
        print("Use: python enhanced_get_filing_chunks.py analyze")