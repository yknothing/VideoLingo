"""
Test Data Cleanup and Lifecycle Management
Automatic cleanup mechanisms for test data with safety checks
Prevents accumulation of test data and maintains clean test environment
"""

import os
import shutil
import time
import threading
from pathlib import Path
from typing import List, Dict, Set, Optional, Callable, Pattern
from datetime import datetime, timedelta
from fnmatch import fnmatch
import re
import weakref

from .base import AbstractDataCleaner, DataScope, DataCategory

class SafeDataCleaner(AbstractDataCleaner):
    """Safe data cleaner with multiple protection mechanisms"""
    
    def __init__(self, base_path: Path, protected_paths: Optional[List[str]] = None):
        self.base_path = Path(base_path)
        self.protected_paths = set(protected_paths or [])
        self._lock = threading.RLock()
        
        # Add default protected paths
        self.protected_paths.update([
            str(Path.home()),
            "/",
            "/usr",
            "/etc",
            "/var",
            "/System",
            "/Applications"
        ])
        
        # Tracking
        self._tracked_files: Set[Path] = set()
        self._tracked_dirs: Set[Path] = set()
        self._scope_tracking: Dict[DataScope, Set[Path]] = {
            scope: set() for scope in DataScope
        }
    
    def register_file(self, file_path: Path, scope: DataScope = DataScope.TEMPORARY) -> None:
        """Register a file for cleanup tracking"""
        with self._lock:
            file_path = Path(file_path).resolve()
            self._tracked_files.add(file_path)
            self._scope_tracking[scope].add(file_path)
    
    def register_directory(self, dir_path: Path, scope: DataScope = DataScope.TEMPORARY) -> None:
        """Register a directory for cleanup tracking"""
        with self._lock:
            dir_path = Path(dir_path).resolve()
            self._tracked_dirs.add(dir_path)
            self._scope_tracking[scope].add(dir_path)
    
    def cleanup_by_scope(self, scope: DataScope) -> int:
        """Cleanup data by scope"""
        count = 0
        with self._lock:
            paths_to_clean = list(self._scope_tracking[scope])
            
            for path in paths_to_clean:
                if self._safe_remove(path):
                    count += 1
                    # Remove from all tracking
                    self._tracked_files.discard(path)
                    self._tracked_dirs.discard(path)
                    for scope_set in self._scope_tracking.values():
                        scope_set.discard(path)
            
            self._scope_tracking[scope].clear()
        
        return count
    
    def cleanup_by_age(self, max_age_hours: int) -> int:
        """Cleanup data older than specified hours"""
        count = 0
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self._lock:
            paths_to_check = list(self._tracked_files | self._tracked_dirs)
            
            for path in paths_to_check:
                try:
                    if path.exists() and path.stat().st_mtime < cutoff_time:
                        if self._safe_remove(path):
                            count += 1
                            self._remove_from_tracking(path)
                except Exception:
                    continue
        
        return count
    
    def cleanup_by_pattern(self, pattern: str) -> int:
        """Cleanup data matching pattern"""
        count = 0
        
        with self._lock:
            paths_to_check = list(self._tracked_files | self._tracked_dirs)
            
            for path in paths_to_check:
                if fnmatch(path.name, pattern) or fnmatch(str(path), pattern):
                    if self._safe_remove(path):
                        count += 1
                        self._remove_from_tracking(path)
        
        return count
    
    def cleanup_all(self) -> int:
        """Cleanup all tracked test data"""
        count = 0
        
        with self._lock:
            # Clean files first
            files_to_clean = list(self._tracked_files)
            for file_path in files_to_clean:
                if self._safe_remove(file_path):
                    count += 1
            
            # Clean directories
            dirs_to_clean = list(self._tracked_dirs)
            # Sort by depth (deepest first)
            dirs_to_clean.sort(key=lambda p: len(p.parts), reverse=True)
            
            for dir_path in dirs_to_clean:
                if self._safe_remove(dir_path):
                    count += 1
            
            # Clear all tracking
            self._tracked_files.clear()
            self._tracked_dirs.clear()
            for scope_set in self._scope_tracking.values():
                scope_set.clear()
        
        return count
    
    def cleanup_empty_directories(self) -> int:
        """Remove empty directories in tracked paths"""
        count = 0
        
        # Find all directories under base_path
        if not self.base_path.exists():
            return 0
        
        # Get all directories, sorted by depth (deepest first)
        dirs = []
        for root, dirnames, filenames in os.walk(self.base_path, topdown=False):
            dirs.append(Path(root))
        
        for dir_path in dirs:
            if dir_path == self.base_path:
                continue  # Never remove base path
            
            try:
                if dir_path.exists() and self._is_empty_directory(dir_path):
                    if self._safe_remove(dir_path):
                        count += 1
            except Exception:
                continue
        
        return count
    
    def _safe_remove(self, path: Path) -> bool:
        """Safely remove file or directory with protection checks"""
        try:
            path = path.resolve()
            
            # Protection checks
            if not self._is_safe_to_remove(path):
                return False
            
            if path.is_file():
                path.unlink()
                return True
            elif path.is_dir():
                shutil.rmtree(path)
                return True
            
        except Exception as e:
            print(f"Warning: Could not remove {path}: {e}")
        
        return False
    
    def _is_safe_to_remove(self, path: Path) -> bool:
        """Check if path is safe to remove"""
        path = path.resolve()
        
        # Check protected paths
        for protected in self.protected_paths:
            protected_path = Path(protected).resolve()
            try:
                # Check if path is or is under protected path
                if path == protected_path or protected_path in path.parents:
                    return False
            except Exception:
                continue
        
        # Must be under base_path
        try:
            if self.base_path not in path.parents and path \!= self.base_path:
                return False
        except Exception:
            return False
        
        # Additional safety checks
        path_str = str(path)
        unsafe_patterns = [
            r'^/[^/]*$',  # Root level directories
            r'.*/(bin|sbin|usr|etc|var|opt)(/.*)?$',  # System directories
            r'.*\.py$',  # Python source files (unless explicitly tracked)
        ]
        
        for pattern in unsafe_patterns:
            if re.match(pattern, path_str):
                # Allow if explicitly tracked
                if path not in self._tracked_files and path not in self._tracked_dirs:
                    return False
        
        return True
    
    def _is_empty_directory(self, path: Path) -> bool:
        """Check if directory is empty"""
        try:
            return path.is_dir() and not any(path.iterdir())
        except Exception:
            return False
    
    def _remove_from_tracking(self, path: Path) -> None:
        """Remove path from all tracking sets"""
        self._tracked_files.discard(path)
        self._tracked_dirs.discard(path)
        for scope_set in self._scope_tracking.values():
            scope_set.discard(path)
    
    def get_stats(self) -> Dict[str, int]:
        """Get cleanup statistics"""
        with self._lock:
            return {
                'tracked_files': len(self._tracked_files),
                'tracked_dirs': len(self._tracked_dirs),
                'session_scope': len(self._scope_tracking[DataScope.SESSION]),
                'module_scope': len(self._scope_tracking[DataScope.TEST_MODULE]),
                'function_scope': len(self._scope_tracking[DataScope.TEST_FUNCTION]),
                'temporary_scope': len(self._scope_tracking[DataScope.TEMPORARY])
            }

class ScheduledCleaner:
    """Scheduled cleanup with automatic execution"""
    
    def __init__(self, cleaner: SafeDataCleaner, interval_minutes: int = 30):
        self.cleaner = cleaner
        self.interval = interval_minutes * 60
        self._timer: Optional[threading.Timer] = None
        self._running = False
    
    def start(self) -> None:
        """Start scheduled cleanup"""
        if not self._running:
            self._running = True
            self._schedule_next()
    
    def stop(self) -> None:
        """Stop scheduled cleanup"""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
    
    def _schedule_next(self) -> None:
        """Schedule next cleanup"""
        if self._running:
            self._timer = threading.Timer(self.interval, self._execute_cleanup)
            self._timer.daemon = True
            self._timer.start()
    
    def _execute_cleanup(self) -> None:
        """Execute cleanup tasks"""
        try:
            # Cleanup temporary and expired data
            temp_count = self.cleaner.cleanup_by_scope(DataScope.TEMPORARY)
            old_count = self.cleaner.cleanup_by_age(24)  # 24 hours
            empty_dirs = self.cleaner.cleanup_empty_directories()
            
            if temp_count > 0 or old_count > 0 or empty_dirs > 0:
                print(f"Cleaned up: {temp_count} temp, {old_count} old, {empty_dirs} empty dirs")
                
        except Exception as e:
            print(f"Scheduled cleanup error: {e}")
        
        finally:
            self._schedule_next()

class TestSessionCleaner:
    """Cleanup manager for test sessions"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.cleaner = SafeDataCleaner(base_path)
        self.scheduled_cleaner = ScheduledCleaner(self.cleaner)
        
        # Track session state
        self._session_active = False
        self._module_stack: List[str] = []
        self._function_stack: List[str] = []
        
        # Auto-cleanup on exit
        weakref.finalize(self, self._final_cleanup)
    
    def start_session(self) -> None:
        """Start test session"""
        self._session_active = True
        self.scheduled_cleaner.start()
    
    def end_session(self) -> None:
        """End test session and cleanup"""
        self._session_active = False
        self.scheduled_cleaner.stop()
        
        # Cleanup all scopes except SESSION
        self.cleaner.cleanup_by_scope(DataScope.TEMPORARY)
        self.cleaner.cleanup_by_scope(DataScope.TEST_FUNCTION)
        self.cleaner.cleanup_by_scope(DataScope.TEST_MODULE)
        
        # Clean empty directories
        self.cleaner.cleanup_empty_directories()
    
    def start_module(self, module_name: str) -> None:
        """Start test module"""
        self._module_stack.append(module_name)
    
    def end_module(self, module_name: str) -> None:
        """End test module and cleanup"""
        if module_name in self._module_stack:
            self._module_stack.remove(module_name)
        
        # Cleanup module scope
        self.cleaner.cleanup_by_scope(DataScope.TEST_MODULE)
    
    def start_function(self, function_name: str) -> None:
        """Start test function"""
        self._function_stack.append(function_name)
    
    def end_function(self, function_name: str) -> None:
        """End test function and cleanup"""
        if function_name in self._function_stack:
            self._function_stack.remove(function_name)
        
        # Cleanup function scope
        self.cleaner.cleanup_by_scope(DataScope.TEST_FUNCTION)
    
    def register_temp_file(self, file_path: Path) -> None:
        """Register temporary file for cleanup"""
        self.cleaner.register_file(file_path, DataScope.TEMPORARY)
    
    def register_temp_dir(self, dir_path: Path) -> None:
        """Register temporary directory for cleanup"""
        self.cleaner.register_directory(dir_path, DataScope.TEMPORARY)
    
    def emergency_cleanup(self) -> int:
        """Emergency cleanup of all test data"""
        print("Performing emergency cleanup of all test data...")
        return self.cleaner.cleanup_all()
    
    def _final_cleanup(self) -> None:
        """Final cleanup on exit"""
        try:
            self.scheduled_cleaner.stop()
            self.cleaner.cleanup_by_scope(DataScope.TEMPORARY)
            self.cleaner.cleanup_empty_directories()
        except Exception:
            pass

class CleanupHooks:
    """Hooks for integrating cleanup with test frameworks"""
    
    def __init__(self, session_cleaner: TestSessionCleaner):
        self.session_cleaner = session_cleaner
        self._registered_hooks: List[Callable] = []
    
    def pytest_configure(self, config) -> None:
        """pytest configuration hook"""
        self.session_cleaner.start_session()
    
    def pytest_unconfigure(self, config) -> None:
        """pytest unconfiguration hook"""
        self.session_cleaner.end_session()
    
    def pytest_runtest_setup(self, item) -> None:
        """pytest test setup hook"""
        module_name = item.module.__name__ if item.module else "unknown"
        function_name = item.name
        
        self.session_cleaner.start_module(module_name)
        self.session_cleaner.start_function(function_name)
    
    def pytest_runtest_teardown(self, item) -> None:
        """pytest test teardown hook"""
        module_name = item.module.__name__ if item.module else "unknown"
        function_name = item.name
        
        self.session_cleaner.end_function(function_name)
        self.session_cleaner.end_module(module_name)
    
    def register_custom_hook(self, hook: Callable) -> None:
        """Register custom cleanup hook"""
        self._registered_hooks.append(hook)
    
    def execute_custom_hooks(self) -> None:
        """Execute all custom hooks"""
        for hook in self._registered_hooks:
            try:
                hook()
            except Exception as e:
                print(f"Custom hook error: {e}")

# Global session cleaner
_session_cleaner: Optional[TestSessionCleaner] = None

def get_session_cleaner(base_path: Optional[Path] = None) -> TestSessionCleaner:
    """Get global session cleaner"""
    global _session_cleaner
    if _session_cleaner is None:
        if base_path is None:
            base_path = Path(__file__).parent / "test_data"
        _session_cleaner = TestSessionCleaner(base_path)
    return _session_cleaner

def initialize_session_cleaner(base_path: Path) -> TestSessionCleaner:
    """Initialize global session cleaner"""
    global _session_cleaner
    _session_cleaner = TestSessionCleaner(base_path)
    return _session_cleaner

def cleanup_all_test_data(base_path: Optional[Path] = None) -> int:
    """Emergency cleanup of all test data"""
    cleaner = get_session_cleaner(base_path)
    return cleaner.emergency_cleanup()
EOF < /dev/null