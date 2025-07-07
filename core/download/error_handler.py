"""
Download error handling utilities
Single responsibility: Categorize and handle download errors with retry logic
"""

import time
from typing import Tuple, Callable, Any
from enum import Enum


class ErrorCategory(Enum):
    """Categories of download errors"""
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    NOT_FOUND = "not_found"
    ACCESS_DENIED = "access_denied"
    PROXY_SSL = "proxy_ssl"
    UNKNOWN = "unknown"


class DownloadErrorHandler:
    """Handle download errors with intelligent categorization and retry logic"""
    
    def __init__(self):
        self.error_patterns = {
            ErrorCategory.NETWORK: ['network', 'timeout', 'connection', 'temporary'],
            ErrorCategory.RATE_LIMIT: ['403', '429', 'rate limit', 'too many requests'],
            ErrorCategory.NOT_FOUND: ['404', 'not found', 'unavailable'],
            ErrorCategory.ACCESS_DENIED: ['401', 'unauthorized', 'private', 'restricted'],
            ErrorCategory.PROXY_SSL: ['proxy', 'ssl', 'certificate']
        }
        
        self.retry_settings = {
            ErrorCategory.NETWORK: (True, 30),
            ErrorCategory.RATE_LIMIT: (True, 60),
            ErrorCategory.NOT_FOUND: (False, 0),
            ErrorCategory.ACCESS_DENIED: (False, 0),
            ErrorCategory.PROXY_SSL: (True, 10),
            ErrorCategory.UNKNOWN: (True, 20)
        }
    
    def categorize_download_error(self, error_msg: str) -> Tuple[ErrorCategory, bool, int]:
        """
        Categorize download errors to determine retry strategy
        
        Args:
            error_msg (str): Error message to categorize
            
        Returns:
            tuple: (category, is_retryable, suggested_wait_time)
        """
        error_msg_lower = error_msg.lower()
        
        for category, patterns in self.error_patterns.items():
            if any(keyword in error_msg_lower for keyword in patterns):
                is_retryable, wait_time = self.retry_settings[category]
                return category, is_retryable, wait_time
        
        # Default to unknown category
        is_retryable, wait_time = self.retry_settings[ErrorCategory.UNKNOWN]
        return ErrorCategory.UNKNOWN, is_retryable, wait_time
    
    def intelligent_retry_download(
        self, 
        download_func: Callable, 
        max_retries: int = 3, 
        initial_wait: int = 5
    ) -> Any:
        """
        Intelligent retry mechanism with exponential backoff
        
        Args:
            download_func (callable): Function to retry
            max_retries (int): Maximum number of retries
            initial_wait (int): Initial wait time in seconds
            
        Returns:
            Any: Result of successful download function call
            
        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return download_func()
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                category, is_retryable, suggested_wait = self.categorize_download_error(error_msg)
                
                if attempt == max_retries:
                    print(f"Download failed after {max_retries} retries: {error_msg}")
                    break
                
                if not is_retryable:
                    print(f"Non-retryable error ({category.value}): {error_msg}")
                    break
                
                wait_time = min(initial_wait * (2 ** attempt), suggested_wait)
                print(f"Download attempt {attempt + 1} failed ({category.value}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        raise last_exception if last_exception else Exception("Max retries exceeded")
    
    def get_error_suggestion(self, error_msg: str) -> str:
        """
        Get user-friendly suggestion based on error category
        
        Args:
            error_msg (str): Error message
            
        Returns:
            str: User-friendly suggestion
        """
        category, is_retryable, _ = self.categorize_download_error(error_msg)
        
        suggestions = {
            ErrorCategory.NETWORK: "Check your internet connection and try again.",
            ErrorCategory.RATE_LIMIT: "You've hit a rate limit. Wait a few minutes before trying again.",
            ErrorCategory.NOT_FOUND: "The video was not found or is no longer available.",
            ErrorCategory.ACCESS_DENIED: "The video is private or requires authentication.",
            ErrorCategory.PROXY_SSL: "Check your proxy settings or SSL certificate configuration.",
            ErrorCategory.UNKNOWN: "An unexpected error occurred. Check the video URL and try again."
        }
        
        base_suggestion = suggestions.get(category, suggestions[ErrorCategory.UNKNOWN])
        
        if is_retryable:
            return f"{base_suggestion} This error is automatically retryable."
        else:
            return f"{base_suggestion} Manual intervention may be required."
    
    def create_retry_decorator(self, max_retries: int = 3, initial_wait: int = 5):
        """
        Create a decorator for automatic retry functionality
        
        Args:
            max_retries (int): Maximum number of retries
            initial_wait (int): Initial wait time
            
        Returns:
            callable: Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return self.intelligent_retry_download(
                    lambda: func(*args, **kwargs),
                    max_retries=max_retries,
                    initial_wait=initial_wait
                )
            return wrapper
        return decorator