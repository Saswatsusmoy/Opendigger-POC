"""
Utility functions for the OpenDigger Repository Labeling POC.
Contains common helper functions used across multiple modules.
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
from functools import wraps
import hashlib
import unicodedata

from .config import get_config

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logging(
    name: str = "opendigger_poc",
    level: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """Set up logging configuration for the application.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    config = get_config()
    
    # Use provided level or default from config
    log_level = level or config.logging.level
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        config.logging.format,
        datefmt=config.logging.date_format
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    _logger = logger
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Optional logger name. If None, returns the global logger.
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(name)
    
    if _logger is None:
        return setup_logging()
    
    return _logger


def safe_json_load(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Safely load JSON data from a file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data or None if loading fails
    """
    logger = get_logger()
    
    try:
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"JSON file does not exist: {path}")
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None


def safe_json_save(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> bool:
    """Safely save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
        indent: JSON indentation
        ensure_ascii: Whether to ensure ASCII encoding
        
    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()
    
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        
        logger.debug(f"Successfully saved JSON to {path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def clean_text(text: str) -> str:
    """Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_keywords(
    text: str,
    min_length: int = 3,
    max_keywords: int = 20,
    exclude_common: bool = True
) -> List[str]:
    """Extract keywords from text using simple pattern matching.
    
    Args:
        text: Text to extract keywords from
        min_length: Minimum keyword length
        max_keywords: Maximum number of keywords to return
        exclude_common: Whether to exclude common words
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Common words to exclude
    common_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
        'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use', 'way', 'will',
        'with', 'this', 'that', 'have', 'from', 'they', 'know', 'want', 'been',
        'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just',
        'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them',
        'well', 'were'
    } if exclude_common else set()
    
    # Extract words using regex
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', text.lower())
    
    # Filter and count words
    word_counts = {}
    for word in words:
        if (len(word) >= min_length and 
            word not in common_words and
            not word.isdigit()):
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]


def validate_repository_data(repo_data: Dict[str, Any]) -> bool:
    """Validate repository data structure.
    
    Args:
        repo_data: Repository data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['name', 'platform']
    
    if not isinstance(repo_data, dict):
        return False
    
    # Check required fields
    for field in required_fields:
        if field not in repo_data:
            return False
    
    # Validate field types
    if not isinstance(repo_data.get('name'), str):
        return False
    
    if not isinstance(repo_data.get('platform'), str):
        return False
    
    return True


def calculate_confidence_score(
    base_score: float,
    factors: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Calculate a confidence score based on multiple factors.
    
    Args:
        base_score: Base confidence score (0.0 to 1.0)
        factors: Dictionary of factor names to values
        weights: Optional weights for each factor
        
    Returns:
        Calculated confidence score (0.0 to 1.0)
    """
    if not factors:
        return max(0.0, min(1.0, base_score))
    
    # Default weights
    default_weights = {factor: 1.0 for factor in factors.keys()}
    weights = weights or default_weights
    
    # Calculate weighted score
    weighted_sum = 0.0
    total_weight = 0.0
    
    for factor, value in factors.items():
        weight = weights.get(factor, 1.0)
        weighted_sum += value * weight
        total_weight += weight
    
    if total_weight == 0:
        return max(0.0, min(1.0, base_score))
    
    # Combine base score with weighted factors
    factor_score = weighted_sum / total_weight
    final_score = (base_score + factor_score) / 2.0
    
    return max(0.0, min(1.0, final_score))


def generate_hash(data: Union[str, Dict[str, Any]]) -> str:
    """Generate a hash for data deduplication.
    
    Args:
        data: Data to hash (string or dictionary)
        
    Returns:
        SHA-256 hash string
    """
    if isinstance(data, dict):
        # Sort dictionary for consistent hashing
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode('utf-8')).hexdigest()


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable[[F], F]:
    """Decorator to retry function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            return None  # Should never reach here
        
        return wrapper
    return decorator


def timing_decorator(func: F) -> F:
    """Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper


def batch_process(
    items: List[T],
    batch_size: int,
    process_func: Callable[[List[T]], Any],
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Any]:
    """Process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        process_func: Function to process each batch
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of results from processing each batch
    """
    logger = get_logger()
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
        
        try:
            result = process_func(batch)
            results.append(result)
            
            if progress_callback:
                progress_callback(batch_num, total_batches)
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            results.append(None)
    
    return results


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """Format timestamp as ISO string.
    
    Args:
        timestamp: Unix timestamp. If None, uses current time.
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = time.time()
    
    return datetime.fromtimestamp(timestamp).isoformat()


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except (OSError, FileNotFoundError):
        return 0


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, with later ones taking precedence.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def filter_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out None values from dictionary.
    
    Args:
        data: Dictionary to filter
        
    Returns:
        Dictionary with None values removed
    """
    return {k: v for k, v in data.items() if v is not None}


def normalize_label(label: str) -> str:
    """Normalize a label string for consistency.
    
    Args:
        label: Raw label string
        
    Returns:
        Normalized label
    """
    if not label:
        return ""
    
    # Convert to lowercase
    label = label.lower().strip()
    
    # Replace spaces and special characters with hyphens
    label = re.sub(r'[^\w\-]', '-', label)
    
    # Remove multiple consecutive hyphens
    label = re.sub(r'-+', '-', label)
    
    # Remove leading/trailing hyphens
    label = label.strip('-')
    
    return label


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid URL, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the process
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.logger = get_logger()
    
    def update(self, increment: int = 1) -> None:
        """Update progress.
        
        Args:
            increment: Number of items processed
        """
        self.current += increment
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        
        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed_time / self.current) * (self.total - self.current)
        else:
            eta = 0
        
        self.logger.info(
            f"{self.description}: {self.current}/{self.total} "
            f"({percentage:.1f}%) - ETA: {eta:.1f}s"
        )
    
    def finish(self) -> None:
        """Mark progress as finished."""
        total_time = time.time() - self.start_time
        self.logger.info(
            f"{self.description} completed: {self.total} items in {total_time:.1f}s"
        ) 