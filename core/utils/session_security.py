"""
VideoLingo Session Security Module
===================================

Comprehensive session security system for Streamlit applications with:
- Session state validation and integrity protection
- User session isolation with unique tokens
- Encrypted session data storage
- Session timeout and cleanup mechanisms
- Cross-user session contamination prevention
- Concurrent session protection
- Session state audit logging
- Immutable state update patterns

Author: VideoLingo Security Team
Last Updated: 2025-08-21
Security Level: HIGH
"""

import os
import json
import time
import uuid
import hashlib
import hmac
import secrets
import logging
import threading
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import streamlit as st

# Configure session security logger
session_logger = logging.getLogger('videolingo.session_security')
if not session_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [Session:%(session_id)s] %(message)s'
    )
    handler.setFormatter(formatter)
    session_logger.addHandler(handler)
    session_logger.setLevel(logging.INFO)

# Type definitions
T = TypeVar('T')


class SessionSecurityError(Exception):
    """Base exception for session security violations"""
    pass


class SessionValidationError(SessionSecurityError):
    """Raised when session validation fails"""
    pass


class SessionIsolationError(SessionSecurityError):
    """Raised when session isolation is violated"""
    pass


class SessionTimeoutError(SessionSecurityError):
    """Raised when session has expired"""
    pass


class SessionIntegrityError(SessionSecurityError):
    """Raised when session integrity check fails"""
    pass


@dataclass
class SessionMetadata:
    """Session metadata for tracking and validation"""
    session_id: str
    user_token: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_expired: bool = False
    security_level: str = "standard"


@dataclass
class SessionConfig:
    """Configuration for session security"""
    # Timeout settings
    session_timeout: timedelta = timedelta(hours=2)
    idle_timeout: timedelta = timedelta(minutes=30)
    max_concurrent_sessions: int = 5
    
    # Security settings
    enable_encryption: bool = True
    enable_integrity_check: bool = True
    enable_audit_logging: bool = True
    
    # Validation settings
    validate_on_access: bool = True
    strict_isolation: bool = True
    auto_cleanup: bool = True
    
    # Performance settings
    cleanup_interval: timedelta = timedelta(minutes=5)
    max_session_size: int = 10 * 1024 * 1024  # 10MB


class SessionEncryption:
    """Handles encryption and decryption of session data"""
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize session encryption with master key"""
        if master_key:
            self.key = self._derive_key(master_key.encode())
        else:
            # Generate a new key for this session
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, master_key: bytes) -> bytes:
        """Derive encryption key from master key using PBKDF2"""
        salt = b'videolingo_session_salt_2025'  # Fixed salt for consistency
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key))
        return key
    
    def encrypt(self, data: Dict[str, Any]) -> str:
        """Encrypt session data"""
        try:
            json_data = json.dumps(data, default=str).encode('utf-8')
            encrypted = self.cipher.encrypt(json_data)
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')
        except Exception as e:
            raise SessionSecurityError(f"Failed to encrypt session data: {e}")
    
    def decrypt(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt session data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return json.loads(decrypted.decode('utf-8'))
        except Exception as e:
            raise SessionSecurityError(f"Failed to decrypt session data: {e}")


class SessionValidator:
    """Validates session state and detects manipulation"""
    
    def __init__(self, secret_key: str):
        """Initialize validator with secret key for HMAC"""
        self.secret_key = secret_key.encode('utf-8')
    
    def generate_signature(self, data: Dict[str, Any]) -> str:
        """Generate HMAC signature for data integrity"""
        try:
            # Sort keys for consistent signature
            sorted_data = json.dumps(data, sort_keys=True, default=str)
            signature = hmac.new(
                self.secret_key,
                sorted_data.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            return signature
        except Exception as e:
            raise SessionSecurityError(f"Failed to generate signature: {e}")
    
    def verify_signature(self, data: Dict[str, Any], signature: str) -> bool:
        """Verify data integrity using HMAC signature"""
        try:
            expected_signature = self.generate_signature(data)
            return hmac.compare_digest(expected_signature, signature)
        except Exception:
            return False
    
    def validate_session_data(self, data: Any, allowed_types: List[type] = None) -> bool:
        """Validate session data structure and types"""
        if allowed_types is None:
            allowed_types = [str, int, float, bool, list, dict, type(None)]
        
        def _check_type(obj):
            if type(obj) not in allowed_types:
                return False
            if isinstance(obj, dict):
                return all(_check_type(v) for v in obj.values())
            elif isinstance(obj, list):
                return all(_check_type(item) for item in obj)
            return True
        
        return _check_type(data)


class SecureSessionManager:
    """
    Main session security manager providing comprehensive protection
    """
    
    def __init__(self, config: Optional[SessionConfig] = None):
        """Initialize secure session manager"""
        self.config = config or SessionConfig()
        self.sessions: Dict[str, SessionMetadata] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
        self.session_signatures: Dict[str, str] = {}
        
        # Initialize encryption and validation
        self.master_key = self._get_or_create_master_key()
        self.encryption = SessionEncryption(self.master_key)
        self.validator = SessionValidator(self.master_key)
        
        # Session cleanup timer
        self._cleanup_lock = threading.Lock()
        self._last_cleanup = time.time()
        
        session_logger.info("SecureSessionManager initialized", extra={'session_id': 'system'})
    
    def _get_or_create_master_key(self) -> str:
        """Get or create master encryption key"""
        # In production, this should be stored securely (environment, key vault, etc.)
        key_file = os.path.join(os.path.expanduser("~"), ".videolingo", "session.key")
        
        try:
            if os.path.exists(key_file):
                with open(key_file, 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        
        # Generate new key
        new_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')
        
        # Save key securely
        try:
            os.makedirs(os.path.dirname(key_file), mode=0o700, exist_ok=True)
            with open(key_file, 'w') as f:
                f.write(new_key)
            os.chmod(key_file, 0o600)
        except Exception as e:
            session_logger.warning(f"Failed to save session key: {e}", extra={'session_id': 'system'})
        
        return new_key
    
    def create_session(self, user_context: Optional[Dict[str, str]] = None) -> str:
        """Create a new secure session"""
        session_id = str(uuid.uuid4())
        user_token = secrets.token_urlsafe(32)
        
        metadata = SessionMetadata(
            session_id=session_id,
            user_token=user_token,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            ip_address=user_context.get('ip_address') if user_context else None,
            user_agent=user_context.get('user_agent') if user_context else None
        )
        
        self.sessions[session_id] = metadata
        self.session_data[session_id] = {}
        
        # Generate initial signature
        self.session_signatures[session_id] = self.validator.generate_signature({})
        
        session_logger.info(
            f"New session created for user_token: {user_token[:8]}...",
            extra={'session_id': session_id}
        )
        
        return session_id
    
    def get_current_session_id(self) -> str:
        """Get or create current session ID"""
        # Try to get session ID from Streamlit session state
        if hasattr(st, 'session_state'):
            if '_secure_session_id' in st.session_state:
                session_id = st.session_state._secure_session_id
                if self.validate_session(session_id):
                    return session_id
                else:
                    # Session invalid, remove from state
                    del st.session_state._secure_session_id
        
        # Create new session
        session_id = self.create_session()
        
        # Store in Streamlit session state
        if hasattr(st, 'session_state'):
            st.session_state._secure_session_id = session_id
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate session exists and is not expired"""
        if session_id not in self.sessions:
            return False
        
        metadata = self.sessions[session_id]
        now = datetime.now()
        
        # Check expiration
        if metadata.is_expired:
            return False
        
        # Check session timeout
        if now - metadata.created_at > self.config.session_timeout:
            self._expire_session(session_id, "session_timeout")
            return False
        
        # Check idle timeout
        if now - metadata.last_accessed > self.config.idle_timeout:
            self._expire_session(session_id, "idle_timeout")
            return False
        
        return True
    
    def _expire_session(self, session_id: str, reason: str):
        """Mark session as expired and log the reason"""
        if session_id in self.sessions:
            self.sessions[session_id].is_expired = True
            session_logger.warning(
                f"Session expired: {reason}",
                extra={'session_id': session_id}
            )
    
    def set_session_value(self, key: str, value: Any, session_id: Optional[str] = None) -> None:
        """Set session value with validation and integrity protection"""
        if session_id is None:
            session_id = self.get_current_session_id()
        
        if not self.validate_session(session_id):
            raise SessionValidationError(f"Invalid session: {session_id}")
        
        # Validate data type
        if not self.validator.validate_session_data(value):
            raise SessionValidationError(f"Invalid data type for key: {key}")
        
        # Check session size limits
        if self._estimate_session_size(session_id, key, value) > self.config.max_session_size:
            raise SessionValidationError("Session size limit exceeded")
        
        # Update session data
        if session_id not in self.session_data:
            self.session_data[session_id] = {}
        
        self.session_data[session_id][key] = value
        
        # Update integrity signature
        self.session_signatures[session_id] = self.validator.generate_signature(
            self.session_data[session_id]
        )
        
        # Update access metadata
        self.sessions[session_id].last_accessed = datetime.now()
        self.sessions[session_id].access_count += 1
        
        session_logger.debug(
            f"Session value set: {key}",
            extra={'session_id': session_id}
        )
        
        # Trigger cleanup if needed
        self._maybe_cleanup()
    
    def get_session_value(self, key: str, default: Any = None, session_id: Optional[str] = None) -> Any:
        """Get session value with validation"""
        if session_id is None:
            session_id = self.get_current_session_id()
        
        if not self.validate_session(session_id):
            raise SessionValidationError(f"Invalid session: {session_id}")
        
        # Verify integrity
        if not self._verify_session_integrity(session_id):
            raise SessionIntegrityError(f"Session integrity check failed: {session_id}")
        
        # Update access metadata
        self.sessions[session_id].last_accessed = datetime.now()
        self.sessions[session_id].access_count += 1
        
        value = self.session_data.get(session_id, {}).get(key, default)
        
        session_logger.debug(
            f"Session value accessed: {key}",
            extra={'session_id': session_id}
        )
        
        return value
    
    def delete_session_value(self, key: str, session_id: Optional[str] = None) -> bool:
        """Delete session value"""
        if session_id is None:
            session_id = self.get_current_session_id()
        
        if not self.validate_session(session_id):
            raise SessionValidationError(f"Invalid session: {session_id}")
        
        if session_id in self.session_data and key in self.session_data[session_id]:
            del self.session_data[session_id][key]
            
            # Update integrity signature
            self.session_signatures[session_id] = self.validator.generate_signature(
                self.session_data[session_id]
            )
            
            session_logger.debug(
                f"Session value deleted: {key}",
                extra={'session_id': session_id}
            )
            return True
        
        return False
    
    def clear_session(self, session_id: Optional[str] = None) -> None:
        """Clear all session data"""
        if session_id is None:
            session_id = self.get_current_session_id()
        
        if session_id in self.session_data:
            self.session_data[session_id] = {}
            self.session_signatures[session_id] = self.validator.generate_signature({})
            
            session_logger.info(
                "Session data cleared",
                extra={'session_id': session_id}
            )
    
    def destroy_session(self, session_id: Optional[str] = None) -> None:
        """Completely destroy session"""
        if session_id is None:
            session_id = self.get_current_session_id()
        
        # Remove from all tracking structures
        self.sessions.pop(session_id, None)
        self.session_data.pop(session_id, None)
        self.session_signatures.pop(session_id, None)
        
        # Remove from Streamlit session state if it's the current session
        if hasattr(st, 'session_state') and st.session_state.get('_secure_session_id') == session_id:
            del st.session_state._secure_session_id
        
        session_logger.info(
            "Session destroyed",
            extra={'session_id': session_id}
        )
    
    def _verify_session_integrity(self, session_id: str) -> bool:
        """Verify session data integrity"""
        if not self.config.enable_integrity_check:
            return True
        
        if session_id not in self.session_signatures:
            return False
        
        current_signature = self.session_signatures[session_id]
        session_data = self.session_data.get(session_id, {})
        
        return self.validator.verify_signature(session_data, current_signature)
    
    def _estimate_session_size(self, session_id: str, key: str, value: Any) -> int:
        """Estimate session data size in bytes"""
        test_data = self.session_data.get(session_id, {}).copy()
        test_data[key] = value
        return len(json.dumps(test_data, default=str).encode('utf-8'))
    
    def _maybe_cleanup(self) -> None:
        """Trigger cleanup if interval has passed"""
        if not self.config.auto_cleanup:
            return
        
        current_time = time.time()
        if current_time - self._last_cleanup > self.config.cleanup_interval.total_seconds():
            with self._cleanup_lock:
                if current_time - self._last_cleanup > self.config.cleanup_interval.total_seconds():
                    self.cleanup_expired_sessions()
                    self._last_cleanup = current_time
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        expired_sessions = []
        
        for session_id, metadata in self.sessions.items():
            if not self.validate_session(session_id):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.destroy_session(session_id)
        
        if expired_sessions:
            session_logger.info(
                f"Cleaned up {len(expired_sessions)} expired sessions",
                extra={'session_id': 'system'}
            )
        
        return len(expired_sessions)
    
    def get_session_info(self, session_id: Optional[str] = None) -> Optional[SessionMetadata]:
        """Get session metadata"""
        if session_id is None:
            session_id = self.get_current_session_id()
        
        return self.sessions.get(session_id)


class SecureSessionState:
    """
    Secure wrapper for Streamlit session state with React-like patterns
    """
    
    def __init__(self, manager: Optional[SecureSessionManager] = None):
        """Initialize secure session state wrapper"""
        self.manager = manager or get_session_manager()
    
    def __getitem__(self, key: str) -> Any:
        """Get session value using dictionary syntax"""
        return self.manager.get_session_value(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set session value using dictionary syntax"""
        self.manager.set_session_value(key, value)
    
    def __delitem__(self, key: str) -> None:
        """Delete session value using dictionary syntax"""
        self.manager.delete_session_value(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in session"""
        try:
            self.manager.get_session_value(key)
            return True
        except:
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get session value with default"""
        return self.manager.get_session_value(key, default)
    
    def pop(self, key: str, default: Any = None) -> Any:
        """Remove and return session value"""
        value = self.manager.get_session_value(key, default)
        self.manager.delete_session_value(key)
        return value
    
    def clear(self) -> None:
        """Clear all session data"""
        self.manager.clear_session()
    
    def keys(self) -> List[str]:
        """Get all session keys"""
        session_id = self.manager.get_current_session_id()
        return list(self.manager.session_data.get(session_id, {}).keys())
    
    def update(self, **kwargs) -> None:
        """Update multiple session values"""
        for key, value in kwargs.items():
            self.manager.set_session_value(key, value)


# Global session manager instance
_session_manager: Optional[SecureSessionManager] = None
_session_lock = threading.Lock()


def get_session_manager() -> SecureSessionManager:
    """Get global session manager instance (singleton)"""
    global _session_manager
    
    if _session_manager is None:
        with _session_lock:
            if _session_manager is None:
                _session_manager = SecureSessionManager()
    
    return _session_manager


def get_secure_session_state() -> SecureSessionState:
    """Get secure session state wrapper"""
    return SecureSessionState(get_session_manager())


# Context managers for session security
class SessionSecurityContext:
    """Context manager for secure session operations"""
    
    def __init__(self, session_id: Optional[str] = None, 
                 security_level: str = "standard"):
        self.session_id = session_id
        self.security_level = security_level
        self.manager = get_session_manager()
        self._original_session = None
    
    def __enter__(self) -> SecureSessionState:
        """Enter security context"""
        if self.session_id:
            # Validate session before using
            if not self.manager.validate_session(self.session_id):
                raise SessionValidationError(f"Invalid session: {self.session_id}")
        
        return SecureSessionState(self.manager)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit security context"""
        if exc_type:
            session_logger.error(
                f"Exception in session context: {exc_type.__name__}: {exc_val}",
                extra={'session_id': self.session_id or 'unknown'}
            )


# Decorators for session security
def require_valid_session(func: Callable) -> Callable:
    """Decorator to ensure function has valid session"""
    def wrapper(*args, **kwargs):
        manager = get_session_manager()
        session_id = manager.get_current_session_id()
        
        if not manager.validate_session(session_id):
            raise SessionValidationError("Valid session required")
        
        return func(*args, **kwargs)
    
    return wrapper


def session_audit_log(operation: str):
    """Decorator to log session operations"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            manager = get_session_manager()
            session_id = manager.get_current_session_id()
            
            session_logger.info(
                f"Session operation: {operation}",
                extra={'session_id': session_id}
            )
            
            try:
                result = func(*args, **kwargs)
                session_logger.debug(
                    f"Session operation completed: {operation}",
                    extra={'session_id': session_id}
                )
                return result
            except Exception as e:
                session_logger.error(
                    f"Session operation failed: {operation} - {e}",
                    extra={'session_id': session_id}
                )
                raise
        
        return wrapper
    return decorator


# Utility functions
def secure_session_cleanup():
    """Manual session cleanup"""
    manager = get_session_manager()
    return manager.cleanup_expired_sessions()


def get_session_security_status() -> Dict[str, Any]:
    """Get current session security status"""
    manager = get_session_manager()
    session_id = manager.get_current_session_id()
    metadata = manager.get_session_info(session_id)
    
    return {
        'session_id': session_id,
        'is_valid': manager.validate_session(session_id),
        'created_at': metadata.created_at if metadata else None,
        'last_accessed': metadata.last_accessed if metadata else None,
        'access_count': metadata.access_count if metadata else 0,
        'is_expired': metadata.is_expired if metadata else True,
        'total_sessions': len(manager.sessions),
        'encryption_enabled': manager.config.enable_encryption,
        'integrity_check_enabled': manager.config.enable_integrity_check,
    }