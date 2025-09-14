"""
Filing Manager for AnnualReportRAG
Manages filing downloads and cleanup based on date ranges and user requirements
"""

import os
import re
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FilingManager:
    """Manages SEC filing downloads and cleanup with date-aware filtering"""
    
    def __init__(self, base_filings_dir: str = None):
        """
        Initialize filing manager
        
        Args:
            base_filings_dir: Base directory for filings (defaults to AnnualReportRAG/filings)
        """
        if base_filings_dir:
            self.base_dir = Path(base_filings_dir)
        else:
            # Default to the filings directory relative to this script
            script_dir = Path(__file__).parent
            self.base_dir = script_dir / "filings"
        
        self.base_dir.mkdir(exist_ok=True)
    
    def get_ticker_filing_dir(self, ticker: str) -> Path:
        """Get filing directory for a specific ticker"""
        return self.base_dir / ticker.lower()
    
    def extract_filing_date_from_filename(self, filename: str) -> Optional[datetime]:
        """
        Extract filing date from SEC filename
        
        SEC filenames follow pattern: CIK-YY-NNNNNN.txt where YY is year
        Example: 0000320193-24-000123.txt = filed in 2024
        
        Args:
            filename: SEC filing filename
            
        Returns:
            Estimated filing date or None if cannot parse
        """
        try:
            # Pattern: CIK-YY-NNNNNN.txt
            match = re.match(r'\d+-(\d{2})-\d+\.txt$', filename)
            if not match:
                return None
            
            year_suffix = match.group(1)
            
            # Convert 2-digit year to 4-digit year
            # 22-24 = 2022-2024, 00-21 = 2000-2021 (though unlikely for recent filings)
            year_int = int(year_suffix)
            if year_int >= 22:  # Assume 2022+ for 22-99
                full_year = 2000 + year_int
            else:  # Assume 2000s for 00-21  
                full_year = 2000 + year_int
            
            # Return January 1st of that year as approximation
            # (SEC filings don't encode exact date in filename)
            return datetime(full_year, 1, 1)
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not parse filing date from {filename}: {e}")
            return None
    
    def get_filing_file_info(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get information about all downloaded filing files for a ticker
        
        Args:
            ticker: Company ticker symbol
            
        Returns:
            List of filing file information dictionaries
        """
        ticker_dir = self.get_ticker_filing_dir(ticker)
        if not ticker_dir.exists():
            return []
        
        filing_files = []
        
        # Walk through ticker directory structure
        for root, dirs, files in os.walk(ticker_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(ticker_dir)
                    
                    # Extract filing type from path structure
                    path_parts = relative_path.parts
                    filing_type = "unknown"
                    if len(path_parts) >= 3:  # e.g., 10-K/AAPL/10-K/filename.txt
                        filing_type = path_parts[0]
                    
                    # Extract filing date from filename
                    filing_date = self.extract_filing_date_from_filename(file)
                    
                    # Get file stats
                    stat = file_path.stat()
                    
                    filing_info = {
                        'filename': file,
                        'filepath': str(file_path),
                        'relative_path': str(relative_path),
                        'filing_type': filing_type,
                        'estimated_filing_date': filing_date,
                        'file_size_bytes': stat.st_size,
                        'downloaded_at': datetime.fromtimestamp(stat.st_mtime)
                    }
                    
                    filing_files.append(filing_info)
        
        # Sort by estimated filing date (newest first)
        filing_files.sort(
            key=lambda x: x['estimated_filing_date'] or datetime.min, 
            reverse=True
        )
        
        return filing_files
    
    def filter_filings_by_date_range(
        self, 
        ticker: str, 
        years_back: int = 2,
        filing_types: Optional[List[str]] = None,
        reference_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter downloaded filings by date range and types
        
        Args:
            ticker: Company ticker symbol
            years_back: Years to look back from reference date
            filing_types: Types of filings to include (e.g., ['10-K', '10-Q'])
            reference_date: Reference date (defaults to today)
            
        Returns:
            List of filings within the specified date range and types
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        cutoff_date = reference_date - timedelta(days=years_back * 365)
        
        logger.info(f"Filtering {ticker} filings: cutoff_date={cutoff_date.strftime('%Y-%m-%d')}, "
                   f"filing_types={filing_types}")
        
        all_filings = self.get_filing_file_info(ticker)
        filtered_filings = []
        
        for filing in all_filings:
            # Check filing type filter
            if filing_types and filing['filing_type'] not in filing_types:
                continue
            
            # Check date filter
            filing_date = filing['estimated_filing_date']
            if filing_date and filing_date >= cutoff_date:
                filtered_filings.append(filing)
            elif not filing_date:
                # If we can't determine date, include it (better safe than sorry)
                logger.warning(f"Including filing with unknown date: {filing['filename']}")
                filtered_filings.append(filing)
        
        logger.info(f"Filtered to {len(filtered_filings)}/{len(all_filings)} filings for {ticker}")
        
        return filtered_filings
    
    def clean_old_filings(
        self, 
        ticker: str, 
        years_back: int = 2,
        filing_types: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Clean old filing files that are outside the specified date range
        
        Args:
            ticker: Company ticker symbol
            years_back: Years to keep from current date
            filing_types: Types of filings to clean (None = all types)
            dry_run: If True, only report what would be deleted
            
        Returns:
            Cleanup statistics and details
        """
        cutoff_date = datetime.now() - timedelta(days=years_back * 365)
        
        logger.info(f"Cleaning {ticker} filings older than {cutoff_date.strftime('%Y-%m-%d')} "
                   f"(dry_run={dry_run})")
        
        all_filings = self.get_filing_file_info(ticker)
        files_to_delete = []
        files_to_keep = []
        total_size_to_delete = 0
        
        for filing in all_filings:
            # Check if this filing type should be cleaned
            if filing_types and filing['filing_type'] not in filing_types:
                files_to_keep.append(filing)
                continue
            
            # Check if filing is old
            filing_date = filing['estimated_filing_date']
            should_delete = False
            
            if filing_date and filing_date < cutoff_date:
                should_delete = True
            elif not filing_date:
                # For files without clear dates, check download date
                download_date = filing['downloaded_at']
                if download_date < cutoff_date:
                    should_delete = True
            
            if should_delete:
                files_to_delete.append(filing)
                total_size_to_delete += filing['file_size_bytes']
            else:
                files_to_keep.append(filing)
        
        # Perform deletion if not dry run
        deleted_files = []
        deletion_errors = []
        
        if not dry_run and files_to_delete:
            for filing in files_to_delete:
                try:
                    file_path = Path(filing['filepath'])
                    if file_path.exists():
                        file_path.unlink()
                        deleted_files.append(filing)
                        logger.info(f"Deleted old filing: {filing['filename']}")
                    else:
                        logger.warning(f"File not found for deletion: {filing['filepath']}")
                except Exception as e:
                    error_info = {'filing': filing, 'error': str(e)}
                    deletion_errors.append(error_info)
                    logger.error(f"Error deleting {filing['filename']}: {e}")
        
        # Clean empty directories
        if not dry_run:
            self._clean_empty_directories(self.get_ticker_filing_dir(ticker))
        
        cleanup_stats = {
            'ticker': ticker.upper(),
            'cutoff_date': cutoff_date.isoformat(),
            'years_back': years_back,
            'filing_types_cleaned': filing_types,
            'dry_run': dry_run,
            'total_files_found': len(all_filings),
            'files_to_delete': len(files_to_delete),
            'files_to_keep': len(files_to_keep),
            'total_size_to_delete_mb': round(total_size_to_delete / 1024 / 1024, 2),
            'files_deleted': len(deleted_files) if not dry_run else 0,
            'deletion_errors': len(deletion_errors),
            'cleanup_performed_at': datetime.now().isoformat()
        }
        
        if dry_run:
            logger.info(f"DRY RUN: Would delete {len(files_to_delete)} files "
                       f"({cleanup_stats['total_size_to_delete_mb']} MB)")
        else:
            logger.info(f"Cleanup complete: deleted {len(deleted_files)} files, "
                       f"{len(deletion_errors)} errors")
        
        return cleanup_stats
    
    def _clean_empty_directories(self, base_path: Path):
        """Recursively remove empty directories"""
        try:
            for root, dirs, files in os.walk(base_path, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        if not any(dir_path.iterdir()):  # Directory is empty
                            dir_path.rmdir()
                            logger.debug(f"Removed empty directory: {dir_path}")
                    except OSError:
                        pass  # Directory not empty or other error
        except Exception as e:
            logger.warning(f"Error cleaning empty directories: {e}")
    
    def get_ticker_filing_stats(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive filing statistics for a ticker"""
        filings = self.get_filing_file_info(ticker)
        
        if not filings:
            return {
                'ticker': ticker.upper(),
                'total_filings': 0,
                'filing_types': {},
                'date_range': None,
                'total_size_mb': 0,
                'oldest_filing': None,
                'newest_filing': None
            }
        
        # Calculate statistics
        filing_types = {}
        total_size = 0
        dates_with_data = []
        
        for filing in filings:
            # Count by filing type
            filing_type = filing['filing_type']
            filing_types[filing_type] = filing_types.get(filing_type, 0) + 1
            
            # Sum file sizes
            total_size += filing['file_size_bytes']
            
            # Collect dates
            if filing['estimated_filing_date']:
                dates_with_data.append(filing['estimated_filing_date'])
        
        # Date range analysis
        date_range = None
        oldest_filing = None
        newest_filing = None
        
        if dates_with_data:
            dates_with_data.sort()
            oldest_filing = dates_with_data[0].isoformat()
            newest_filing = dates_with_data[-1].isoformat()
            date_range = f"{oldest_filing} to {newest_filing}"
        
        return {
            'ticker': ticker.upper(),
            'total_filings': len(filings),
            'filing_types': filing_types,
            'date_range': date_range,
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'oldest_filing': oldest_filing,
            'newest_filing': newest_filing,
            'filings_with_dates': len(dates_with_data),
            'filings_without_dates': len(filings) - len(dates_with_data)
        }

# Convenience functions for external use
def clean_old_filings_for_ticker(
    ticker: str, 
    years_back: int = 2, 
    filing_types: Optional[List[str]] = None,
    dry_run: bool = False,
    filings_dir: str = None
) -> Dict[str, Any]:
    """
    Clean old filings for a specific ticker
    
    Args:
        ticker: Company ticker symbol
        years_back: Years to keep from current date
        filing_types: Types of filings to clean (None = all types)
        dry_run: If True, only report what would be deleted
        filings_dir: Base filings directory (defaults to ./filings)
        
    Returns:
        Cleanup statistics
    """
    manager = FilingManager(filings_dir)
    return manager.clean_old_filings(ticker, years_back, filing_types, dry_run)

def get_valid_filings_for_processing(
    ticker: str,
    years_back: int = 2,
    filing_types: Optional[List[str]] = None,
    filings_dir: str = None
) -> List[str]:
    """
    Get list of valid filing file paths for processing based on date range
    
    Args:
        ticker: Company ticker symbol
        years_back: Years to look back
        filing_types: Types of filings to include
        filings_dir: Base filings directory
        
    Returns:
        List of valid filing file paths
    """
    manager = FilingManager(filings_dir)
    valid_filings = manager.filter_filings_by_date_range(ticker, years_back, filing_types)
    
    return [filing['filepath'] for filing in valid_filings]

def auto_cleanup_before_processing(
    ticker: str,
    years_back: int = 2,
    filing_types: Optional[List[str]] = None,
    filings_dir: str = None,
    cleanup_enabled: bool = True
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Auto-cleanup old filings and return valid files for processing
    
    Args:
        ticker: Company ticker symbol
        years_back: Years to keep/process
        filing_types: Types of filings to process
        filings_dir: Base filings directory
        cleanup_enabled: Whether to actually perform cleanup
        
    Returns:
        Tuple of (valid_file_paths, cleanup_stats)
    """
    manager = FilingManager(filings_dir)
    
    # Get stats before cleanup
    pre_stats = manager.get_ticker_filing_stats(ticker)
    
    # Perform cleanup if enabled
    cleanup_stats = {'cleanup_performed': False}
    if cleanup_enabled:
        cleanup_stats = manager.clean_old_filings(
            ticker=ticker,
            years_back=years_back, 
            filing_types=filing_types,
            dry_run=False
        )
        cleanup_stats['cleanup_performed'] = True
    
    # Get valid files for processing
    valid_files = get_valid_filings_for_processing(
        ticker=ticker,
        years_back=years_back,
        filing_types=filing_types,
        filings_dir=filings_dir
    )
    
    # Add pre/post stats
    cleanup_stats.update({
        'pre_cleanup_stats': pre_stats,
        'valid_files_for_processing': len(valid_files),
        'valid_file_paths': valid_files
    })
    
    logger.info(f"Auto-cleanup for {ticker}: {len(valid_files)} valid files, "
               f"cleanup_enabled={cleanup_enabled}")
    
    return valid_files, cleanup_stats