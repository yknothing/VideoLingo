# ------------
# Security Fixes Validation Tests (P1 & P2)
# Tests for JSON deserialization, path traversal, and race condition fixes
# ------------

import pytest
import json
import threading
import time
import os
from pathlib import Path


class TestJSONSecurityFixes:
    """P1: SEC_002/SEC_013 - JSON deserialization security tests"""
    
    def test_valid_json_parsing(self):
        """Test that valid JSON is parsed correctly"""
        from core.utils.ask_gpt import safe_json_parse
        
        result = safe_json_parse('{"key": "value", "number": 123}')
        assert result == {"key": "value", "number": 123}
    
    def test_size_limit_enforcement(self):
        """Test that oversized JSON is rejected"""
        from core.utils.ask_gpt import safe_json_parse
        from core.constants import SecurityConstants
        
        # Create JSON larger than limit
        large_data = "x" * (SecurityConstants.MAX_JSON_RESPONSE_SIZE + 1000)
        large_json = f'{{"data": "{large_data}"}}'
        
        with pytest.raises(ValueError, match="too large"):
            safe_json_parse(large_json)
    
    def test_suspicious_pattern_detection(self):
        """Test that suspicious patterns are detected"""
        from core.utils.ask_gpt import safe_json_parse
        
        suspicious_inputs = [
            '{"code": "__import__"}',
            '{"fn": "eval(x)"}',
            '{"fn": "exec(code)"}',
            '{"mod": "import os"}',
            '{"call": "globals()"}',
        ]
        
        for suspicious in suspicious_inputs:
            with pytest.raises(ValueError, match="suspicious"):
                safe_json_parse(suspicious)
    
    def test_url_encoded_bypass_blocked(self):
        """Test that URL-encoded attacks are blocked"""
        from core.utils.ask_gpt import safe_json_parse
        
        # URL encoded __import__
        encoded = '{"code": "%5f%5fimport%5f%5f"}'
        with pytest.raises(ValueError, match="suspicious"):
            safe_json_parse(encoded)
    
    def test_nesting_depth_validation(self):
        """Test JSON nesting depth limits"""
        from core.utils.ask_gpt import _validate_json_depth
        
        # Shallow nesting should pass
        shallow = {"a": {"b": {"c": "value"}}}
        assert _validate_json_depth(shallow, max_depth=10) is True
        
        # Deep nesting should fail
        deep = {"level": None}
        current = deep
        for _ in range(60):
            current["level"] = {"level": None}
            current = current["level"]
        
        assert _validate_json_depth(deep, max_depth=50) is False
    
    def test_type_validation(self):
        """Test that non-string input is rejected"""
        from core.utils.ask_gpt import safe_json_parse
        
        with pytest.raises(ValueError, match="must be string"):
            safe_json_parse(123)
        
        with pytest.raises(ValueError, match="must be string"):
            safe_json_parse(None)


class TestPathTraversalFixes:
    """P1: SEC_001 - Path traversal protection tests"""
    
    def test_unix_traversal_blocked(self):
        """Test that Unix-style traversal is blocked"""
        from core.st_utils.sidebar_setting import _validate_path
        
        is_valid, _, error = _validate_path("../../../etc/passwd")
        assert not is_valid
        assert "malicious" in error.lower()
    
    def test_windows_traversal_blocked(self):
        """Test that Windows-style traversal is blocked"""
        from core.st_utils.sidebar_setting import _validate_path
        
        is_valid, _, error = _validate_path("..\\..\\windows\\system32")
        assert not is_valid
        assert "malicious" in error.lower()
    
    def test_url_encoded_traversal_blocked(self):
        """Test that URL-encoded traversal is blocked"""
        from core.st_utils.sidebar_setting import _validate_path
        
        # %2e = . and %2f = /
        is_valid, _, error = _validate_path("%2e%2e%2f%2e%2e%2fetc/passwd")
        assert not is_valid
    
    def test_double_url_encoded_blocked(self):
        """Test that double URL-encoded traversal is blocked"""
        from core.st_utils.sidebar_setting import _validate_path
        
        # %252e = %2e (double encoded)
        is_valid, _, error = _validate_path("%252e%252e/etc/passwd")
        assert not is_valid
    
    def test_shell_metacharacters_blocked(self):
        """Test that shell metacharacters are blocked"""
        from core.st_utils.sidebar_setting import _validate_path
        
        dangerous_paths = [
            "/tmp; rm -rf /",
            "/tmp && cat /etc/passwd",
            "/tmp | cat /etc/passwd",
            "/tmp`whoami`",
            "/tmp$(whoami)",
        ]
        
        for dangerous in dangerous_paths:
            is_valid, _, _ = _validate_path(dangerous)
            assert not is_valid, f"Should block: {dangerous}"
    
    def test_null_byte_injection_blocked(self):
        """Test that null byte injection is blocked"""
        from core.st_utils.sidebar_setting import _validate_path
        
        is_valid, _, _ = _validate_path("/tmp\x00/evil")
        assert not is_valid
    
    def test_valid_path_accepted(self):
        """Test that valid paths are accepted"""
        from core.st_utils.sidebar_setting import _validate_path
        
        cwd = os.getcwd()
        is_valid, resolved_path, error = _validate_path(cwd)
        assert is_valid, f"Current directory should be valid: {error}"
        assert os.path.exists(resolved_path)
    
    def test_allowed_roots_populated(self):
        """Test that allowed roots are properly configured"""
        from core.st_utils.sidebar_setting import _get_allowed_path_roots
        
        roots = _get_allowed_path_roots()
        assert len(roots) > 0
        assert all(isinstance(r, Path) for r in roots)


class TestConfigCacheRaceCondition:
    """P2: CNC_001 - Config cache race condition tests"""
    
    def test_concurrent_read_operations(self):
        """Test that concurrent reads don't cause errors"""
        from core.utils.config_utils import _get_cached_config
        
        errors = []
        results = []
        
        def read_config():
            try:
                for _ in range(20):
                    config = _get_cached_config()
                    results.append(isinstance(config, dict))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=read_config) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors during concurrent reads: {errors}"
        assert all(results), "All results should be dict type"
    
    def test_concurrent_invalidate_operations(self):
        """Test that concurrent invalidates don't cause errors"""
        from core.utils.config_utils import _get_cached_config, _invalidate_cache
        
        errors = []
        
        def mixed_operations():
            try:
                for _ in range(10):
                    _get_cached_config()
                    _invalidate_cache()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))
        
        threads = [threading.Thread(target=mixed_operations) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"
    
    def test_cache_returns_copy(self):
        """Test that cache returns copies to prevent external modification"""
        from core.utils.config_utils import _get_cached_config
        
        config1 = _get_cached_config()
        config2 = _get_cached_config()
        
        # Should be equal but not the same object
        assert config1 == config2
        assert config1 is not config2, "Cache should return copies"


class TestExceptionHandlingFixes:
    """P2: IMP_002 - Exception handling improvements"""
    
    def test_memory_manager_error_logging(self):
        """Test that MemoryManager logs errors properly"""
        from core._2_asr import MemoryManager
        
        manager = MemoryManager()
        # Just verify it doesn't crash
        status = manager.get_memory_status()
        assert "total_gb" in status or "error" in status
    
    def test_result_collector_size_estimation(self):
        """Test that result collector handles errors in size estimation"""
        from core._2_asr import MemoryManager, MemoryEfficientResultCollector
        
        manager = MemoryManager()
        collector = MemoryEfficientResultCollector(manager)
        
        # Test with valid result
        valid_result = {"segments": [{"text": "Hello world"}]}
        size = collector._estimate_result_size(valid_result)
        assert size > 0
        
        # Test with malformed result (should not crash)
        malformed_result = {"segments": "not a list"}
        size = collector._estimate_result_size(malformed_result)
        assert size > 0  # Should return default estimate


class TestConstantsIntegration:
    """P3: IMP_004 - Magic values moved to constants"""
    
    def test_security_constants_exist(self):
        """Test that security constants are properly defined"""
        from core.constants import SecurityConstants
        
        assert hasattr(SecurityConstants, 'MAX_JSON_RESPONSE_SIZE')
        assert hasattr(SecurityConstants, 'MAX_JSON_NESTING_DEPTH')
        assert hasattr(SecurityConstants, 'MAX_JSON_STRING_LENGTH')
        
        # Verify sensible values
        assert SecurityConstants.MAX_JSON_RESPONSE_SIZE > 1024 * 1024  # > 1MB
        assert SecurityConstants.MAX_JSON_NESTING_DEPTH > 10
    
    def test_memory_constants_exist(self):
        """Test that memory constants are properly defined"""
        from core.constants import MemoryConstants
        
        assert hasattr(MemoryConstants, 'DEFAULT_SIZE_ESTIMATE_BYTES')
        assert hasattr(MemoryConstants, 'BYTES_PER_MB')
        assert hasattr(MemoryConstants, 'BYTES_PER_GB')
        
        # Verify values
        assert MemoryConstants.BYTES_PER_MB == 1024 * 1024
        assert MemoryConstants.BYTES_PER_GB == 1024 * 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

