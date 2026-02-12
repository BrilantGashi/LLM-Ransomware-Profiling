"""
Error Handler Module for LLM API Calls
Implements robust error handling with exponential backoff retry logic.

Features:
- Automatic retry on transient errors (timeout, rate limit)
- Exponential backoff algorithm
- Comprehensive error logging for debugging
- Error report generation for issue tracking
"""


import logging
import time
from typing import Callable, Any, Optional, Dict, List
from datetime import datetime
from functools import wraps


try:
    from openai import APIError, APITimeoutError, RateLimitError, APIConnectionError
except ImportError:
    class APIError(Exception):
        pass
    class APITimeoutError(Exception):
        pass
    class RateLimitError(Exception):
        pass
    class APIConnectionError(Exception):
        pass


logger = logging.getLogger("ErrorHandler")


class ErrorHandler:
    """
    Centralized error handling for LLM API calls.
    
    Implements retry logic with exponential backoff for transient errors.
    
    Attributes:
        max_retries (int): Maximum number of retry attempts
        backoff_factor (int): Exponential backoff multiplier (default: 2)
        error_log (List[Dict]): Chronological log of all errors encountered
    
    Example:
        >>> handler = ErrorHandler(max_retries=3, backoff_factor=2)
        >>> result = handler.with_retry(client.generate_response, messages)
    """
    
    def __init__(self, max_retries: int = 3, backoff_factor: int = 2):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum retry attempts (default: 3)
            backoff_factor: Exponential backoff multiplier (default: 2)
                           Wait time = backoff_factor ^ attempt_number
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.error_log: List[Dict[str, Any]] = []
        
        logger.info(
            f"ErrorHandler initialized: max_retries={max_retries}, "
            f"backoff_factor={backoff_factor}"
        )
    
    def with_retry(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Execute a function with automatic retry on transient errors.
        
        Implements exponential backoff strategy:
        - Attempt 1: immediate
        - Attempt 2: wait 2^0 = 1 second
        - Attempt 3: wait 2^1 = 2 seconds
        - Attempt 4: wait 2^2 = 4 seconds
        
        Transient errors (retried):
        - APITimeoutError: Request timeout
        - RateLimitError: Rate limit exceeded
        - APIConnectionError: Network connectivity issues
        
        Permanent errors (not retried):
        - APIError: Authentication, invalid model, etc.
        - Other exceptions: Unexpected errors
        
        Args:
            func: Function to execute (typically LLM API call)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        
        Returns:
            The return value of func if successful
        
        Raises:
            Exception: If all retry attempts fail or permanent error occurs
            
        Example:
            >>> handler = ErrorHandler()
            >>> response = handler.with_retry(
            ...     client.chat.completions.create,
            ...     model="qwen3",
            ...     messages=[{"role": "user", "content": "Hello"}]
            ... )
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(
                        f"Success on retry attempt {attempt + 1}/{self.max_retries}"
                    )
                
                return result
            
            except (APITimeoutError, RateLimitError, APIConnectionError) as e:
                last_exception = e
                error_type = type(e).__name__
                
                if attempt == self.max_retries - 1:
                    self._log_error(
                        error_type=error_type,
                        message=str(e),
                        attempt=attempt + 1,
                        function_name=func.__name__,
                        is_final=True
                    )
                    logger.error(
                        f"All {self.max_retries} retry attempts failed for "
                        f"{func.__name__}: {error_type}"
                    )
                    raise
                
                wait_time = self.backoff_factor ** attempt
                
                self._log_error(
                    error_type=error_type,
                    message=str(e),
                    attempt=attempt + 1,
                    function_name=func.__name__,
                    is_final=False
                )
                
                logger.warning(
                    f"Retry {attempt + 1}/{self.max_retries} after {wait_time}s "
                    f"- {error_type}: {str(e)[:100]}"
                )
                
                time.sleep(wait_time)
            
            except APIError as e:
                self._log_error(
                    error_type="APIError",
                    message=str(e),
                    attempt=attempt + 1,
                    function_name=func.__name__,
                    is_final=True,
                    is_permanent=True
                )
                
                logger.error(
                    f"Non-recoverable API error in {func.__name__}: {e}"
                )
                raise
            
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                
                self._log_error(
                    error_type=error_type,
                    message=str(e),
                    attempt=attempt + 1,
                    function_name=func.__name__,
                    is_final=True,
                    is_permanent=True
                )
                
                logger.error(
                    f"Unexpected error in {func.__name__}: {error_type} - {e}",
                    exc_info=True
                )
                
                if attempt == self.max_retries - 1:
                    raise
        
        raise last_exception
    
    def _log_error(
        self,
        error_type: str,
        message: str,
        attempt: int,
        function_name: str = "unknown",
        is_final: bool = False,
        is_permanent: bool = False
    ) -> None:
        """
        Log error details for debugging and reporting.
        
        Args:
            error_type: Type of exception (e.g., "APITimeoutError")
            message: Error message text
            attempt: Which retry attempt number (1-indexed)
            function_name: Name of function that failed
            is_final: Whether this was the final retry attempt
            is_permanent: Whether error is non-recoverable
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'message': message[:500],
            'attempt': attempt,
            'max_attempts': self.max_retries,
            'function': function_name,
            'is_final': is_final,
            'is_permanent': is_permanent
        }
        
        self.error_log.append(error_entry)
        logger.debug(f"Error logged: {error_entry}")
    
    def get_error_report(self, max_entries: int = 20) -> str:
        """
        Generate formatted error report for issue tracking.
        
        Args:
            max_entries: Maximum number of recent errors to include (default: 20)
        
        Returns:
            str: Formatted error report with timestamps and details
        """
        if not self.error_log:
            return "No errors recorded."
        
        recent_errors = self.error_log[-max_entries:]
        
        lines = [
            "",
            "=" * 70,
            "ERROR REPORT",
            "=" * 70,
            f"Total errors logged: {len(self.error_log)}",
            f"Showing last {len(recent_errors)} entries",
            "=" * 70,
            ""
        ]
        
        for i, err in enumerate(recent_errors, 1):
            lines.append(f"Entry {i}:")
            lines.append(f"Timestamp: {err['timestamp']}")
            lines.append(
                f"Error Type: {err['error_type']} "
                f"(Attempt {err['attempt']}/{err['max_attempts']})"
            )
            
            if err.get('is_permanent'):
                lines.append("Status: NON-RECOVERABLE")
            elif err.get('is_final'):
                lines.append("Status: FINAL ATTEMPT FAILED")
            else:
                lines.append("Status: Retrying...")
            
            lines.append(f"Function: {err['function']}")
            lines.append(f"Message: {err['message']}")
            lines.append("-" * 70)
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("SUMMARY STATISTICS")
        lines.append("=" * 70)
        
        error_types = {}
        for err in recent_errors:
            err_type = err['error_type']
            error_types[err_type] = error_types.get(err_type, 0) + 1
        
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            lines.append(f"  {err_type}: {count}")
        
        lines.append("=" * 70)
        lines.append("")
        
        return "\n".join(lines)
    
    def clear_log(self) -> None:
        """
        Clear the error log.
        Useful for starting fresh between different pipeline runs.
        """
        num_cleared = len(self.error_log)
        self.error_log.clear()
        logger.info(f"Error log cleared ({num_cleared} entries removed)")
    
    def get_error_count(self) -> int:
        """
        Get total number of errors logged.
        
        Returns:
            int: Total error count
        """
        return len(self.error_log)
    
    def get_error_stats(self) -> Dict[str, int]:
        """
        Get statistics about error types encountered.
        
        Returns:
            Dict[str, int]: Mapping of error types to occurrence counts
        
        Example:
            >>> handler.get_error_stats()
            {'APITimeoutError': 5, 'RateLimitError': 2, 'APIError': 1}
        """
        stats = {}
        for err in self.error_log:
            err_type = err['error_type']
            stats[err_type] = stats.get(err_type, 0) + 1
        return stats


def retry_on_error(max_retries: int = 3, backoff_factor: int = 2):
    """
    Decorator for automatic retry on function errors.
    
    Convenience decorator that wraps a function with retry logic.
    
    Args:
        max_retries: Maximum retry attempts
        backoff_factor: Exponential backoff multiplier
    
    Returns:
        Decorated function with retry capability
    
    Example:
        >>> @retry_on_error(max_retries=5)
        >>> def call_api(model, messages):
        ...     return client.chat.completions.create(model=model, messages=messages)
        
        >>> result = call_api("qwen3", messages)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler(
                max_retries=max_retries,
                backoff_factor=backoff_factor
            )
            return handler.with_retry(func, *args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    """
    Example demonstrating error handler usage.
    """
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("ERROR HANDLER DEMO")
    print("=" * 70)
    
    handler = ErrorHandler(max_retries=3, backoff_factor=2)
    
    def flaky_api_call(fail_count: int = 2):
        """Simulates API that fails first N times, then succeeds."""
        if not hasattr(flaky_api_call, 'call_count'):
            flaky_api_call.call_count = 0
        
        flaky_api_call.call_count += 1
        
        if flaky_api_call.call_count <= fail_count:
            print(f"  [Simulated failure {flaky_api_call.call_count}/{fail_count}]")
            raise APITimeoutError("Simulated timeout error")
        
        print(f"  [Success on attempt {flaky_api_call.call_count}]")
        return {"status": "success", "data": "API response"}
    
    print("\nTesting successful retry...")
    try:
        result = handler.with_retry(flaky_api_call, fail_count=2)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    print("\nError Report:")
    print(handler.get_error_report())
    
    print("Error Statistics:")
    stats = handler.get_error_stats()
    for err_type, count in stats.items():
        print(f"  {err_type}: {count}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
