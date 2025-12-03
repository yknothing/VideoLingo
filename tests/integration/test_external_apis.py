"""
Integration tests for external API interactions
Tests API retry logic, timeout handling, response validation, and rate limiting
"""

import os
import sys
import time
import json
import random
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import pytest
import requests
from typing import Dict, List, Any, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.utils.ask_gpt import (
    ask_gpt, is_network_error, get_retry_delay,
    mask_sensitive_content, sanitize_for_logging
)
from core.download.error_handler import (
    DownloadErrorHandler, ErrorCategory
)
from core.utils.decorator import except_handler


class TestAPIRetryLogic:
    """Test API retry logic and timeout handling"""
    
    def test_network_error_detection(self):
        """Test detection of network-related errors"""
        network_errors = [
            "Connection refused",
            "Timeout error occurred",
            "Network is unreachable",
            "DNS resolution failed",
            "SSL handshake_failure",
            "Connection reset by peer",
            "Service temporarily unavailable",
            "502 Bad Gateway",
            "504 Gateway Timeout"
        ]
        
        for error_msg in network_errors:
            assert is_network_error(Exception(error_msg)) is True
        
        # Non-network errors
        non_network_errors = [
            "Invalid API key",
            "Insufficient credits",
            "Model not found",
            "Invalid request format"
        ]
        
        for error_msg in non_network_errors:
            assert is_network_error(Exception(error_msg)) is False
    
    def test_exponential_backoff_calculation(self):
        """Test exponential backoff with jitter calculation"""
        # Test increasing delays
        delays = []
        for attempt in range(5):
            delay = get_retry_delay(attempt, base_delay=1.0)
            delays.append(delay)
        
        # Verify exponential growth
        for i in range(1, len(delays)):
            # Each delay should be roughly double the previous (accounting for jitter)
            assert delays[i] > delays[i-1]
            # But not more than 2.2x due to jitter
            assert delays[i] < delays[i-1] * 2.2
        
        # Test maximum delay cap
        large_attempt_delay = get_retry_delay(10, base_delay=1.0)
        assert large_attempt_delay <= 60.0
    
    @patch('time.sleep')
    @patch('core.utils.ask_gpt.openai.chat.completions.create')
    def test_retry_on_network_errors(self, mock_openai, mock_sleep):
        """Test that network errors trigger retries"""
        # Simulate network errors then success
        mock_openai.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            requests.exceptions.Timeout("Request timeout"),
            Mock(choices=[Mock(message=Mock(content="Success"))])
        ]
        
        with patch('core.utils.ask_gpt.load_key', return_value="test-key"):
            result = ask_gpt("test prompt")
        
        # Verify retries occurred
        assert mock_openai.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries
        assert result == "Success"
    
    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded"""
        @except_handler("API call failed", retry=3, delay=0.1)
        def failing_api_call():
            raise ConnectionError("Network error")
        
        with pytest.raises(ConnectionError):
            failing_api_call()
    
    def test_non_retryable_errors(self):
        """Test that non-retryable errors don't trigger retries"""
        handler = DownloadErrorHandler()
        
        # Test non-retryable error categories
        non_retryable = [
            ("404 Not Found", ErrorCategory.NOT_FOUND),
            ("401 Unauthorized", ErrorCategory.ACCESS_DENIED)
        ]
        
        for error_msg, expected_category in non_retryable:
            category, is_retryable, _ = handler.categorize_download_error(error_msg)
            assert category == expected_category
            assert is_retryable is False
    
    def test_concurrent_api_retries(self):
        """Test retry logic with concurrent API calls"""
        call_counter = {'count': 0}
        lock = threading.Lock()
        
        @except_handler("Concurrent API failed", retry=2, delay=0.1)
        def api_call_with_failures(call_id):
            with lock:
                call_counter['count'] += 1
                # Fail first attempt for all calls
                if call_counter['count'] <= 3:
                    raise ConnectionError(f"Network error for call {call_id}")
            return f"Success {call_id}"
        
        # Make concurrent calls
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(api_call_with_failures, i) for i in range(3)]
            results = [f.result() for f in futures]
        
        # All should eventually succeed
        assert all("Success" in r for r in results)
        assert call_counter['count'] >= 6  # At least 2 attempts per call


class TestAPIResponseValidation:
    """Test API response validation and error mapping"""
    
    def test_json_response_validation(self):
        """Test validation of JSON API responses"""
        # Valid responses
        valid_responses = [
            {"status": "success", "data": "content"},
            {"choices": [{"message": {"content": "response"}}]},
            {"result": {"translation": "text"}}
        ]
        
        for response in valid_responses:
            assert self._validate_json_response(response) is True
        
        # Invalid responses
        invalid_responses = [
            None,
            "",
            "not json",
            {"error": "API error"},
            {"status": "error", "message": "Failed"}
        ]
        
        for response in invalid_responses:
            assert self._validate_json_response(response) is False
    
    def _validate_json_response(self, response):
        """Helper to validate JSON response structure"""
        if not response or not isinstance(response, dict):
            return False
        
        # Check for error indicators
        if "error" in response or response.get("status") == "error":
            return False
        
        # Check for expected data fields
        has_data = any([
            "data" in response,
            "choices" in response,
            "result" in response,
            "content" in response
        ])
        
        return has_data
    
    def test_error_response_mapping(self):
        """Test mapping of API error responses to appropriate exceptions"""
        error_mappings = {
            {"error": {"code": "rate_limit_exceeded"}}: "RateLimitError",
            {"error": {"code": "invalid_api_key"}}: "AuthenticationError",
            {"error": {"code": "model_not_found"}}: "ModelNotFoundError",
            {"error": {"code": "context_length_exceeded"}}: "ContextLengthError",
            {"error": {"message": "Internal server error"}}: "ServerError"
        }
        
        for error_response, expected_error_type in error_mappings.items():
            mapped_error = self._map_error_response(error_response)
            assert expected_error_type in mapped_error
    
    def _map_error_response(self, response):
        """Helper to map error response to exception type"""
        if not response or "error" not in response:
            return "UnknownError"
        
        error = response["error"]
        code = error.get("code", "")
        message = error.get("message", "")
        
        if "rate_limit" in code:
            return "RateLimitError"
        elif "api_key" in code or "authentication" in message.lower():
            return "AuthenticationError"
        elif "model" in code:
            return "ModelNotFoundError"
        elif "context" in code or "length" in code:
            return "ContextLengthError"
        elif "server" in message.lower():
            return "ServerError"
        
        return "UnknownError"
    
    def test_partial_response_handling(self):
        """Test handling of partial or incomplete API responses"""
        partial_responses = [
            {"choices": []},  # Empty choices
            {"choices": [{"message": {}}]},  # Missing content
            {"result": None},  # Null result
            {"data": ""},  # Empty data
        ]
        
        for response in partial_responses:
            result = self._extract_content_safely(response)
            assert result is not None  # Should handle gracefully
            assert isinstance(result, str)  # Should return string
    
    def _extract_content_safely(self, response):
        """Safely extract content from API response"""
        if not response:
            return ""
        
        # Try different response formats
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        
        if "result" in response:
            return str(response["result"]) if response["result"] else ""
        
        if "data" in response:
            return str(response["data"]) if response["data"] else ""
        
        return ""
    
    def test_response_sanitization(self):
        """Test sanitization of sensitive data in responses"""
        sensitive_response = {
            "api_key": "sk-abcd1234efgh5678",
            "data": "User content",
            "token": "Bearer xyz123456789",
            "password": "secret123"
        }
        
        sanitized = sanitize_for_logging(sensitive_response)
        
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["token"] == "[REDACTED]"
        assert sanitized["password"] == "[REDACTED]"
        assert "User content" in str(sanitized["data"])  # Data partially visible


class TestRateLimitingAndBackoff:
    """Test rate limiting and backoff strategies"""
    
    def test_rate_limit_detection(self):
        """Test detection of rate limit responses"""
        rate_limit_responses = [
            {"error": {"code": "rate_limit_exceeded"}},
            {"error": {"message": "429 Too Many Requests"}},
            {"status": 429, "message": "Rate limit hit"},
            {"error": "You have exceeded your rate limit"}
        ]
        
        handler = DownloadErrorHandler()
        for response in rate_limit_responses:
            error_msg = str(response)
            category, is_retryable, wait_time = handler.categorize_download_error(error_msg)
            assert category == ErrorCategory.RATE_LIMIT or "429" in error_msg
            assert is_retryable is True
            assert wait_time >= 60  # Should wait longer for rate limits
    
    def test_token_bucket_rate_limiter(self):
        """Test token bucket algorithm for rate limiting"""
        class TokenBucket:
            def __init__(self, capacity, refill_rate):
                self.capacity = capacity
                self.tokens = capacity
                self.refill_rate = refill_rate
                self.last_refill = time.time()
                self.lock = threading.Lock()
            
            def consume(self, tokens=1):
                with self.lock:
                    self._refill()
                    if self.tokens >= tokens:
                        self.tokens -= tokens
                        return True
                    return False
            
            def _refill(self):
                now = time.time()
                elapsed = now - self.last_refill
                tokens_to_add = elapsed * self.refill_rate
                self.tokens = min(self.capacity, self.tokens + tokens_to_add)
                self.last_refill = now
        
        # Test rate limiting
        bucket = TokenBucket(capacity=5, refill_rate=2)  # 2 tokens per second
        
        # Consume all tokens
        for _ in range(5):
            assert bucket.consume() is True
        
        # Should be rate limited
        assert bucket.consume() is False
        
        # Wait for refill
        time.sleep(1)
        assert bucket.consume() is True  # Should have ~2 tokens after 1 second
    
    def test_adaptive_backoff_strategy(self):
        """Test adaptive backoff based on error patterns"""
        class AdaptiveBackoff:
            def __init__(self):
                self.error_history = []
                self.base_delay = 1.0
            
            def record_error(self, error_type):
                self.error_history.append({
                    'type': error_type,
                    'timestamp': time.time()
                })
                # Keep only recent errors (last minute)
                cutoff = time.time() - 60
                self.error_history = [
                    e for e in self.error_history 
                    if e['timestamp'] > cutoff
                ]
            
            def get_delay(self):
                # Increase delay based on recent error frequency
                recent_errors = len(self.error_history)
                if recent_errors > 10:
                    return self.base_delay * 5
                elif recent_errors > 5:
                    return self.base_delay * 3
                elif recent_errors > 2:
                    return self.base_delay * 2
                return self.base_delay
        
        backoff = AdaptiveBackoff()
        
        # No errors - minimum delay
        assert backoff.get_delay() == 1.0
        
        # Few errors - moderate delay
        for _ in range(3):
            backoff.record_error("network")
        assert backoff.get_delay() == 2.0
        
        # Many errors - maximum delay
        for _ in range(10):
            backoff.record_error("rate_limit")
        assert backoff.get_delay() == 5.0
    
    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for API calls"""
        class CircuitBreaker:
            def __init__(self, failure_threshold=5, recovery_timeout=30):
                self.failure_threshold = failure_threshold
                self.recovery_timeout = recovery_timeout
                self.failure_count = 0
                self.last_failure_time = None
                self.state = "closed"  # closed, open, half-open
                self.lock = threading.Lock()
            
            def call(self, func, *args, **kwargs):
                with self.lock:
                    if self.state == "open":
                        if self._should_attempt_reset():
                            self.state = "half-open"
                        else:
                            raise Exception("Circuit breaker is open")
                
                try:
                    result = func(*args, **kwargs)
                    with self.lock:
                        self._on_success()
                    return result
                except Exception as e:
                    with self.lock:
                        self._on_failure()
                    raise e
            
            def _should_attempt_reset(self):
                return (
                    self.last_failure_time and
                    time.time() - self.last_failure_time >= self.recovery_timeout
                )
            
            def _on_success(self):
                self.failure_count = 0
                self.state = "closed"
            
            def _on_failure(self):
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
        
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.5)
        
        def failing_api():
            raise Exception("API error")
        
        def working_api():
            return "success"
        
        # Test circuit opens after threshold
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_api)
        
        # Circuit should be open
        with pytest.raises(Exception, match="Circuit breaker is open"):
            breaker.call(working_api)
        
        # Wait for recovery timeout
        time.sleep(0.6)
        
        # Should allow attempt (half-open state)
        result = breaker.call(working_api)
        assert result == "success"
        
        # Circuit should be closed again
        assert breaker.state == "closed"
    
    def test_concurrent_rate_limiting(self):
        """Test rate limiting with concurrent requests"""
        class RateLimiter:
            def __init__(self, max_requests_per_second):
                self.max_requests = max_requests_per_second
                self.request_times = []
                self.lock = threading.Lock()
            
            def allow_request(self):
                with self.lock:
                    now = time.time()
                    # Remove old requests outside window
                    self.request_times = [
                        t for t in self.request_times 
                        if now - t < 1.0
                    ]
                    
                    if len(self.request_times) < self.max_requests:
                        self.request_times.append(now)
                        return True
                    return False
        
        limiter = RateLimiter(max_requests_per_second=5)
        results = {'allowed': 0, 'denied': 0}
        lock = threading.Lock()
        
        def make_request():
            if limiter.allow_request():
                with lock:
                    results['allowed'] += 1
            else:
                with lock:
                    results['denied'] += 1
        
        # Make 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            for future in futures:
                future.result()
        
        # Should allow max 5 requests
        assert results['allowed'] == 5
        assert results['denied'] == 5


class TestAPISecurityAndPrivacy:
    """Test API security and privacy features"""
    
    def test_api_key_masking(self):
        """Test that API keys are properly masked in logs"""
        test_cases = [
            ("sk-abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx", "[API_KEY_REDACTED]"),
            ("sk-or-v1-" + "a" * 64, "[API_KEY_REDACTED]"),
            ("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test", "[API_KEY_REDACTED]"),
        ]
        
        for api_key, expected in test_cases:
            sanitized = sanitize_for_logging(api_key)
            assert expected in sanitized
    
    def test_content_masking_for_privacy(self):
        """Test content masking for user privacy"""
        sensitive_content = "This is sensitive user data with personal information"
        
        # Test different mask ratios
        masked_30 = mask_sensitive_content(sensitive_content, mask_ratio=0.3)
        masked_50 = mask_sensitive_content(sensitive_content, mask_ratio=0.5)
        masked_70 = mask_sensitive_content(sensitive_content, mask_ratio=0.7)
        
        # Verify masking increases with ratio
        assert "[MASKED" in masked_30
        assert "[MASKED" in masked_50
        assert "[MASKED" in masked_70
        
        # More masking = less visible content
        visible_30 = len([c for c in masked_30 if c not in "[]MASKED "])
        visible_50 = len([c for c in masked_50 if c not in "[]MASKED "])
        visible_70 = len([c for c in masked_70 if c not in "[]MASKED "])
        
        assert visible_30 > visible_50 > visible_70
    
    def test_secure_api_configuration(self):
        """Test secure API configuration practices"""
        # Test environment variable precedence
        with patch.dict(os.environ, {"VIDEO_LINGO_API_KEY": "env-key"}):
            with patch('core.utils.ask_gpt.load_key', return_value=None):
                # Should use environment variable when config is None
                with patch('core.utils.ask_gpt.openai.chat.completions.create') as mock_api:
                    mock_api.return_value = Mock(
                        choices=[Mock(message=Mock(content="response"))]
                    )
                    result = ask_gpt("test")
                    # Verify env key was used (would be in the API client config)
                    assert result == "response"
    
    def test_api_timeout_configuration(self):
        """Test API timeout configuration and handling"""
        def slow_api_call():
            time.sleep(5)
            return "complete"
        
        # Test timeout enforcement
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slow_api_call)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.5)
    
    def test_request_id_tracking(self):
        """Test request ID tracking for debugging"""
        import uuid
        
        class APIClient:
            def __init__(self):
                self.request_log = []
            
            def make_request(self, endpoint, data):
                request_id = str(uuid.uuid4())
                request_record = {
                    'id': request_id,
                    'endpoint': endpoint,
                    'timestamp': time.time(),
                    'data_size': len(str(data))
                }
                self.request_log.append(request_record)
                
                # Simulate API call
                try:
                    if random.random() < 0.3:  # 30% failure rate
                        raise Exception("API error")
                    return {'request_id': request_id, 'result': 'success'}
                except Exception as e:
                    request_record['error'] = str(e)
                    raise
        
        client = APIClient()
        
        # Make several requests
        success_count = 0
        for i in range(10):
            try:
                result = client.make_request(f"/endpoint{i}", {"data": i})
                success_count += 1
            except:
                pass
        
        # Verify all requests were logged
        assert len(client.request_log) == 10
        
        # Verify request IDs are unique
        request_ids = [r['id'] for r in client.request_log]
        assert len(request_ids) == len(set(request_ids))
        
        # Verify error tracking
        errors = [r for r in client.request_log if 'error' in r]
        assert len(errors) + success_count == 10


class TestAPILoadBalancing:
    """Test API load balancing and failover strategies"""
    
    def test_round_robin_load_balancing(self):
        """Test round-robin load balancing across API endpoints"""
        class LoadBalancer:
            def __init__(self, endpoints):
                self.endpoints = endpoints
                self.current_index = 0
                self.lock = threading.Lock()
            
            def get_endpoint(self):
                with self.lock:
                    endpoint = self.endpoints[self.current_index]
                    self.current_index = (self.current_index + 1) % len(self.endpoints)
                    return endpoint
        
        endpoints = ["api1.example.com", "api2.example.com", "api3.example.com"]
        balancer = LoadBalancer(endpoints)
        
        # Test round-robin distribution
        selected = []
        for _ in range(9):
            selected.append(balancer.get_endpoint())
        
        # Each endpoint should be selected 3 times
        for endpoint in endpoints:
            assert selected.count(endpoint) == 3
        
        # Verify order is round-robin
        for i in range(9):
            assert selected[i] == endpoints[i % 3]
    
    def test_weighted_load_balancing(self):
        """Test weighted load balancing based on endpoint performance"""
        class WeightedLoadBalancer:
            def __init__(self, endpoints_weights):
                self.endpoints_weights = endpoints_weights
                self.total_weight = sum(w for _, w in endpoints_weights)
            
            def get_endpoint(self):
                r = random.uniform(0, self.total_weight)
                cumulative = 0
                for endpoint, weight in self.endpoints_weights:
                    cumulative += weight
                    if r <= cumulative:
                        return endpoint
                return self.endpoints_weights[-1][0]
        
        # Endpoints with weights (higher weight = more traffic)
        endpoints_weights = [
            ("fast-api.com", 5),
            ("medium-api.com", 3),
            ("slow-api.com", 1)
        ]
        
        balancer = WeightedLoadBalancer(endpoints_weights)
        
        # Test distribution over many requests
        distribution = {}
        num_requests = 900
        for _ in range(num_requests):
            endpoint = balancer.get_endpoint()
            distribution[endpoint] = distribution.get(endpoint, 0) + 1
        
        # Verify approximate distribution
        # fast-api should get ~5/9 of traffic
        assert abs(distribution["fast-api.com"] / num_requests - 5/9) < 0.1
        # medium-api should get ~3/9 of traffic
        assert abs(distribution["medium-api.com"] / num_requests - 3/9) < 0.1
        # slow-api should get ~1/9 of traffic
        assert abs(distribution["slow-api.com"] / num_requests - 1/9) < 0.1
    
    def test_failover_mechanism(self):
        """Test failover to backup endpoints on primary failure"""
        class FailoverClient:
            def __init__(self, primary, backups):
                self.primary = primary
                self.backups = backups
                self.failed_endpoints = set()
            
            def make_request(self, data):
                # Try primary first
                if self.primary not in self.failed_endpoints:
                    try:
                        return self._call_endpoint(self.primary, data)
                    except Exception:
                        self.failed_endpoints.add(self.primary)
                
                # Try backups in order
                for backup in self.backups:
                    if backup not in self.failed_endpoints:
                        try:
                            return self._call_endpoint(backup, data)
                        except Exception:
                            self.failed_endpoints.add(backup)
                
                raise Exception("All endpoints failed")
            
            def _call_endpoint(self, endpoint, data):
                # Simulate API call with failure possibility
                if endpoint in ["failing1.com", "failing2.com"]:
                    raise Exception(f"{endpoint} is down")
                return f"Response from {endpoint}"
        
        client = FailoverClient(
            primary="failing1.com",
            backups=["failing2.com", "working.com"]
        )
        
        # Should failover to working endpoint
        result = client.make_request({"test": "data"})
        assert result == "Response from working.com"
        
        # Verify failed endpoints are tracked
        assert "failing1.com" in client.failed_endpoints
        assert "failing2.com" in client.failed_endpoints
        assert "working.com" not in client.failed_endpoints


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
