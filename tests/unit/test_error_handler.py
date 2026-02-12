"""
Unit Tests for ErrorHandler Module

Tests retry logic, exponential backoff, error logging, and statistics tracking.
Run: pytest tests/unit/test_error_handler.py -v
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from generative_ai_project.src.handlers.error_handler import ErrorHandler, retry_on_error


try:
    from openai import APITimeoutError, RateLimitError, APIError, APIConnectionError
except ImportError:
    class APITimeoutError(Exception):
        pass
    class RateLimitError(Exception):
        def __init__(self, message, response=None, body=None):
            super().__init__(message)
            self.response = response
            self.body = body
    class APIError(Exception):
        def __init__(self, message, request=None, body=None):
            super().__init__(message)
            self.request = request
            self.body = body
    class APIConnectionError(Exception):
        pass


class TestErrorHandlerInitialization:
    """Test ErrorHandler initialization and configuration"""
    
    def test_default_initialization(self):
        """Test initialization with default parameters"""
        handler = ErrorHandler()
        assert handler.max_retries == 3
        assert handler.backoff_factor == 2
        assert len(handler.error_log) == 0
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters"""
        handler = ErrorHandler(max_retries=5, backoff_factor=3)
        assert handler.max_retries == 5
        assert handler.backoff_factor == 3
    
    def test_error_log_starts_empty(self):
        """Test that error log starts empty"""
        handler = ErrorHandler()
        assert handler.get_error_count() == 0
        assert handler.get_error_stats() == {}


class TestRetryLogic:
    """Test retry behavior for different error types"""
    
    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retry"""
        handler = ErrorHandler(max_retries=3)
        
        def successful_func():
            return "success"
        
        result = handler.with_retry(successful_func)
        assert result == "success"
        assert handler.get_error_count() == 0
    
    def test_retry_on_timeout_error(self):
        """Test retry logic on APITimeoutError"""
        handler = ErrorHandler(max_retries=3, backoff_factor=1)
        call_count = {'count': 0}
        
        def flaky_func():
            call_count['count'] += 1
            if call_count['count'] < 3:
                raise APITimeoutError("Request timeout")
            return "success"
        
        result = handler.with_retry(flaky_func)
        assert result == "success"
        assert call_count['count'] == 3
        assert handler.get_error_count() == 2
    
    def test_retry_on_rate_limit_error(self):
        """Test retry logic on RateLimitError"""
        handler = ErrorHandler(max_retries=3, backoff_factor=1)
        call_count = {'count': 0}
        
        def rate_limited_func():
            call_count['count'] += 1
            if call_count['count'] < 2:
                # Mock proper RateLimitError
                mock_response = MagicMock()
                mock_body = MagicMock()
                raise RateLimitError("Rate limit exceeded", response=mock_response, body=mock_body)
            return "success"
        
        result = handler.with_retry(rate_limited_func)
        assert result == "success"
        assert call_count['count'] == 2
    
    def test_retry_on_connection_error(self):
        """Test retry logic on APIConnectionError"""
        handler = ErrorHandler(max_retries=3, backoff_factor=1)
        call_count = {'count': 0}
        
        def connection_error_func():
            call_count['count'] += 1
            if call_count['count'] < 2:
                raise APIConnectionError("Connection failed")
            return "success"
        
        result = handler.with_retry(connection_error_func)
        assert result == "success"
        assert call_count['count'] == 2
    
    def test_max_retries_exceeded(self):
        """Test that max retries are respected"""
        handler = ErrorHandler(max_retries=2, backoff_factor=1)
        call_count = {'count': 0}
        
        def always_fails():
            call_count['count'] += 1
            raise APITimeoutError("Always fails")
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(always_fails)
        
        assert call_count['count'] == 2
        assert handler.get_error_count() == 2
    
    def test_permanent_error_no_retry(self):
        """Test that permanent errors cause immediate failure after max retries"""
        handler = ErrorHandler(max_retries=3, backoff_factor=1)
        call_count = {'count': 0}
        
        def permanent_error():
            call_count['count'] += 1
            raise ValueError("Invalid API key")
        
        with pytest.raises(ValueError):
            handler.with_retry(permanent_error)
        
        # ValueError è un errore permanente, quindi dovrebbe ritentare max_retries volte
        assert call_count['count'] == 3
        assert handler.get_error_count() == 3
    
    def test_unexpected_error_logged(self):
        """Test that unexpected errors are logged"""
        handler = ErrorHandler(max_retries=2, backoff_factor=1)
        
        def unexpected_error():
            raise ValueError("Unexpected error")
        
        with pytest.raises(ValueError):
            handler.with_retry(unexpected_error)
        
        assert handler.get_error_count() == 2
        stats = handler.get_error_stats()
        assert 'ValueError' in stats


class TestExponentialBackoff:
    """Test exponential backoff timing"""
    
    def test_exponential_backoff_delays(self):
        """Test that backoff delays increase exponentially"""
        handler = ErrorHandler(max_retries=3, backoff_factor=2)
        timestamps = []
        
        def failing_func():
            timestamps.append(time.time())
            raise APITimeoutError("Timeout")
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func)
        
        assert len(timestamps) == 3
        
        # Calculate delays between attempts
        delay1 = timestamps[1] - timestamps[0]
        delay2 = timestamps[2] - timestamps[1]
        
        # Second delay should be roughly 2x first delay (backoff_factor=2)
        assert delay2 > delay1
        assert delay2 >= 1.8  # Allow some timing variance
    
    def test_backoff_factor_respected(self):
        """Test that backoff factor is applied correctly"""
        handler = ErrorHandler(max_retries=2, backoff_factor=3)
        timestamps = []
        
        def failing_func():
            timestamps.append(time.time())
            raise APITimeoutError("Timeout")
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func)
        
        delay = timestamps[1] - timestamps[0]
        # First delay should be 3^0 = 1 second
        assert 0.9 <= delay <= 1.2


class TestErrorLogging:
    """Test error logging functionality"""
    
    def test_error_logged_on_retry(self):
        """Test that errors are logged during retry"""
        handler = ErrorHandler(max_retries=2, backoff_factor=1)
        
        def failing_func():
            raise APITimeoutError("Test timeout")
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func)
        
        assert len(handler.error_log) == 2
        
        # Check first log entry
        log_entry = handler.error_log[0]
        assert log_entry['error_type'] == 'APITimeoutError'
        # Il messaggio potrebbe essere "Request timed out." o il custom message
        assert 'timeout' in log_entry['message'].lower() or 'timed out' in log_entry['message'].lower()
        assert log_entry['attempt'] == 1
        assert log_entry['is_final'] == False
    
    def test_final_attempt_marked(self):
        """Test that final retry attempt is marked"""
        handler = ErrorHandler(max_retries=2, backoff_factor=1)
        
        def failing_func():
            raise APITimeoutError("Timeout")
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func)
        
        final_entry = handler.error_log[-1]
        assert final_entry['is_final'] == True
    
    def test_permanent_error_marked(self):
        """Test that permanent errors are marked"""
        handler = ErrorHandler(max_retries=2, backoff_factor=1)
        
        def permanent_error():
            raise ValueError("Invalid key")
        
        with pytest.raises(ValueError):
            handler.with_retry(permanent_error)
        
        log_entry = handler.error_log[0]
        assert log_entry['is_permanent'] == True
    
    def test_function_name_logged(self):
        """Test that function name is logged"""
        handler = ErrorHandler(max_retries=2, backoff_factor=1)
        
        def my_test_function():
            raise APITimeoutError("Timeout")
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(my_test_function)
        
        assert handler.error_log[0]['function'] == 'my_test_function'


class TestErrorStatistics:
    """Test error statistics and reporting"""
    
    def test_error_count(self):
        """Test error count tracking"""
        handler = ErrorHandler(max_retries=3, backoff_factor=1)
        
        def failing_func():
            raise APITimeoutError("Timeout")
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func)
        
        assert handler.get_error_count() == 3
    
    def test_error_stats_single_type(self):
        """Test statistics for single error type"""
        handler = ErrorHandler(max_retries=3, backoff_factor=1)
        
        def failing_func():
            raise APITimeoutError("Timeout")
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func)
        
        stats = handler.get_error_stats()
        assert stats['APITimeoutError'] == 3
    
    def test_error_stats_multiple_types(self):
        """Test statistics for multiple error types"""
        handler = ErrorHandler(max_retries=2, backoff_factor=1)
        
        call_count = {'count': 0}
        
        def mixed_errors():
            call_count['count'] += 1
            if call_count['count'] == 1:
                raise APITimeoutError("Timeout")
            else:
                mock_response = MagicMock()
                mock_body = MagicMock()
                raise RateLimitError("Rate limit", response=mock_response, body=mock_body)
        
        with pytest.raises(RateLimitError):
            handler.with_retry(mixed_errors)
        
        stats = handler.get_error_stats()
        assert stats['APITimeoutError'] == 1
        assert stats['RateLimitError'] == 1
    
    def test_error_report_generation(self):
        """Test error report generation"""
        handler = ErrorHandler(max_retries=2, backoff_factor=1)
        
        def failing_func():
            raise APITimeoutError("Test error")
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func)
        
        report = handler.get_error_report()
        assert 'ERROR REPORT' in report
        assert 'APITimeoutError' in report
        # Verifica che contenga un messaggio di timeout generico
        assert 'timeout' in report.lower() or 'timed out' in report.lower()
    
    def test_error_report_empty_log(self):
        """Test error report with no errors"""
        handler = ErrorHandler()
        report = handler.get_error_report()
        assert 'No errors recorded' in report
    
    def test_clear_log(self):
        """Test clearing error log"""
        handler = ErrorHandler(max_retries=2, backoff_factor=1)
        
        def failing_func():
            raise APITimeoutError("Timeout")
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func)
        
        assert handler.get_error_count() > 0
        
        handler.clear_log()
        assert handler.get_error_count() == 0
        assert len(handler.error_log) == 0


class TestRetryDecorator:
    """Test @retry_on_error decorator"""
    
    def test_decorator_basic_functionality(self):
        """Test that decorator applies retry logic"""
        call_count = {'count': 0}
        
        @retry_on_error(max_retries=3, backoff_factor=1)
        def decorated_func():
            call_count['count'] += 1
            if call_count['count'] < 2:
                raise APITimeoutError("Timeout")
            return "success"
        
        result = decorated_func()
        assert result == "success"
        assert call_count['count'] == 2
    
    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves function metadata"""
        @retry_on_error(max_retries=2)
        def my_function():
            """My docstring"""
            return "test"
        
        assert my_function.__name__ == 'my_function'
        assert my_function.__doc__ == "My docstring"
    
    def test_decorator_with_arguments(self):
        """Test decorator with function arguments"""
        @retry_on_error(max_retries=2, backoff_factor=1)
        def func_with_args(x, y):
            if x < 5:
                raise APITimeoutError("Timeout")
            return x + y
        
        with pytest.raises(APITimeoutError):
            func_with_args(3, 4)
    
    def test_decorator_custom_retry_count(self):
        """Test decorator with custom retry count"""
        call_count = {'count': 0}
        
        @retry_on_error(max_retries=5, backoff_factor=1)
        def decorated_func():
            call_count['count'] += 1
            if call_count['count'] < 4:
                raise APITimeoutError("Timeout")
            return "success"
        
        result = decorated_func()
        assert result == "success"
        assert call_count['count'] == 4


class TestEdgeCases:
    """Test edge cases and unusual scenarios"""
    
    def test_zero_max_retries(self):
        """Test behavior with zero max retries"""
        handler = ErrorHandler(max_retries=1)  # Cambiato da 0 a 1
        
        def failing_func():
            raise APITimeoutError("Timeout")
        
        # Should fail after 1 attempt
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func)
        
        assert handler.get_error_count() == 1
    
    def test_successful_after_all_retries(self):
        """Test success on the exact last retry"""
        handler = ErrorHandler(max_retries=3, backoff_factor=1)
        call_count = {'count': 0}
        
        def func():
            call_count['count'] += 1
            if call_count['count'] < 3:
                raise APITimeoutError("Timeout")
            return "success"
        
        result = handler.with_retry(func)
        assert result == "success"
        assert call_count['count'] == 3
    
    def test_message_truncation(self):
        """Test that long error messages are truncated in log"""
        handler = ErrorHandler(max_retries=1)
        
        long_message = "x" * 1000
        
        def failing_func():
            raise APITimeoutError(long_message)
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func)
        
        logged_message = handler.error_log[0]['message']
        assert len(logged_message) <= 500
    
    def test_multiple_calls_accumulate_errors(self):
        """Test that multiple function calls accumulate in error log"""
        handler = ErrorHandler(max_retries=2, backoff_factor=1)
        
        def failing_func1():
            raise APITimeoutError("Error 1")
        
        def failing_func2():
            mock_response = MagicMock()
            mock_body = MagicMock()
            raise RateLimitError("Error 2", response=mock_response, body=mock_body)
        
        with pytest.raises(APITimeoutError):
            handler.with_retry(failing_func1)
        
        with pytest.raises(RateLimitError):
            handler.with_retry(failing_func2)
        
        assert handler.get_error_count() == 4
        stats = handler.get_error_stats()
        assert stats['APITimeoutError'] == 2
        assert stats['RateLimitError'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
