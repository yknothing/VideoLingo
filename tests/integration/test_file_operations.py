"""
Integration tests for file operations
Tests concurrent file access, cleanup, resource management, and disk space handling
"""

import os
import sys
import time
import shutil
import tempfile
import threading
import multiprocessing
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, patch, MagicMock
import pytest
import psutil
from typing import List, Dict, Optional, Tuple
import fcntl  # For file locking on Unix
import random
import gc
import weakref

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.utils import get_temp_path, get_output_path
from core.utils.onekeycleanup import cleanup_temp_files
from core.utils.path_adapter import sanitize_path, validate_path_security


class TestConcurrentFileAccess:
    """Test concurrent file access patterns and race conditions"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp(prefix="file_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_concurrent_read_operations(self, temp_dir):
        """Test multiple threads reading the same file concurrently"""
        test_file = os.path.join(temp_dir, "shared_read.txt")
        test_content = "Test content for concurrent reading\n" * 1000
        
        # Create test file
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        results = []
        errors = []
        
        def read_file(thread_id):
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                    results.append((thread_id, len(content)))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Launch concurrent readers
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_file, i) for i in range(10)]
            for future in futures:
                future.result()
        
        # Verify all reads succeeded
        assert len(errors) == 0
        assert len(results) == 10
        # All should read the same content length
        content_lengths = [length for _, length in results]
        assert all(length == content_lengths[0] for length in content_lengths)
    
    def test_concurrent_write_with_locking(self, temp_dir):
        """Test concurrent writes with proper file locking"""
        test_file = os.path.join(temp_dir, "locked_write.txt")
        lock_file = test_file + ".lock"
        
        class FileLock:
            def __init__(self, lock_path):
                self.lock_path = lock_path
                self.lock_fd = None
            
            def __enter__(self):
                self.lock_fd = open(self.lock_path, 'w')
                # Try to acquire exclusive lock
                while True:
                    try:
                        fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except IOError:
                        time.sleep(0.01)
                return self
            
            def __exit__(self, *args):
                if self.lock_fd:
                    fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                    self.lock_fd.close()
                    try:
                        os.remove(self.lock_path)
                    except:
                        pass
        
        write_order = []
        
        def write_with_lock(thread_id):
            with FileLock(lock_file):
                with open(test_file, 'a') as f:
                    f.write(f"Thread {thread_id} writing\n")
                    write_order.append(thread_id)
                    time.sleep(0.01)  # Simulate work
        
        # Launch concurrent writers
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_with_lock, i) for i in range(5)]
            for future in futures:
                future.result()
        
        # Verify all writes completed
        assert len(write_order) == 5
        
        # Verify file content
        with open(test_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 5
        
        # Each thread should have written exactly once
        for i in range(5):
            assert f"Thread {i} writing\n" in lines
    
    def test_atomic_file_operations(self, temp_dir):
        """Test atomic file write operations to prevent partial writes"""
        target_file = os.path.join(temp_dir, "atomic_target.json")
        
        def atomic_write(filepath, data):
            """Write data atomically using temp file and rename"""
            temp_file = filepath + '.tmp.' + str(os.getpid())
            try:
                # Write to temp file
                with open(temp_file, 'w') as f:
                    json.dump(data, f)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                
                # Atomic rename
                os.rename(temp_file, filepath)
            except Exception:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise
        
        def concurrent_atomic_write(thread_id):
            data = {"thread": thread_id, "timestamp": time.time()}
            atomic_write(target_file, data)
        
        # Launch concurrent atomic writes
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_atomic_write, i) for i in range(10)]
            for future in futures:
                future.result()
        
        # Verify final file is valid JSON (not corrupted)
        with open(target_file, 'r') as f:
            data = json.load(f)
        
        assert "thread" in data
        assert "timestamp" in data
    
    def test_file_handle_exhaustion_prevention(self, temp_dir):
        """Test prevention of file handle exhaustion"""
        max_handles = 100
        opened_files = []
        
        try:
            # Try to open many files
            for i in range(max_handles):
                f = open(os.path.join(temp_dir, f"file_{i}.txt"), 'w')
                opened_files.append(f)
        except OSError as e:
            # Should handle gracefully
            assert "Too many open files" in str(e) or "No file descriptors" in str(e)
        finally:
            # Clean up
            for f in opened_files:
                f.close()
        
        # Test with context manager (automatic cleanup)
        file_count = 0
        for i in range(max_handles):
            try:
                with open(os.path.join(temp_dir, f"ctx_file_{i}.txt"), 'w') as f:
                    f.write("test")
                    file_count += 1
            except OSError:
                break
        
        # Should have successfully created some files
        assert file_count > 0
    
    def test_concurrent_directory_operations(self, temp_dir):
        """Test concurrent directory creation and deletion"""
        base_dir = os.path.join(temp_dir, "concurrent_dirs")
        os.makedirs(base_dir, exist_ok=True)
        
        results = {'created': [], 'errors': []}
        lock = threading.Lock()
        
        def create_and_remove_dir(thread_id):
            dir_path = os.path.join(base_dir, f"dir_{thread_id}")
            try:
                # Create directory
                os.makedirs(dir_path, exist_ok=True)
                
                # Write a file in it
                file_path = os.path.join(dir_path, "test.txt")
                with open(file_path, 'w') as f:
                    f.write(f"Thread {thread_id}")
                
                # Verify and clean up
                assert os.path.exists(file_path)
                shutil.rmtree(dir_path)
                
                with lock:
                    results['created'].append(thread_id)
            except Exception as e:
                with lock:
                    results['errors'].append((thread_id, str(e)))
        
        # Launch concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_and_remove_dir, i) for i in range(10)]
            for future in futures:
                future.result()
        
        # All operations should succeed
        assert len(results['created']) == 10
        assert len(results['errors']) == 0


class TestFileCleanupAndResourceManagement:
    """Test file cleanup and resource management"""
    
    @pytest.fixture
    def cleanup_dir(self):
        """Create a directory for cleanup testing"""
        cleanup_dir = tempfile.mkdtemp(prefix="cleanup_test_")
        yield cleanup_dir
        shutil.rmtree(cleanup_dir, ignore_errors=True)
    
    def test_automatic_cleanup_on_exception(self, cleanup_dir):
        """Test that files are cleaned up even when exceptions occur"""
        temp_files = []
        
        class ManagedFile:
            def __init__(self, filepath):
                self.filepath = filepath
                self.file = None
            
            def __enter__(self):
                self.file = open(self.filepath, 'w')
                return self.file
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.file:
                    self.file.close()
                # Clean up file on exception
                if exc_type is not None:
                    try:
                        os.remove(self.filepath)
                    except:
                        pass
        
        # Test normal operation
        normal_file = os.path.join(cleanup_dir, "normal.txt")
        with ManagedFile(normal_file) as f:
            f.write("normal operation")
        assert os.path.exists(normal_file)
        
        # Test exception handling
        error_file = os.path.join(cleanup_dir, "error.txt")
        try:
            with ManagedFile(error_file) as f:
                f.write("will be deleted")
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # File should be cleaned up
        assert not os.path.exists(error_file)
    
    def test_temp_file_lifecycle_management(self, cleanup_dir):
        """Test temporary file lifecycle and cleanup"""
        import atexit
        
        class TempFileManager:
            def __init__(self, base_dir):
                self.base_dir = base_dir
                self.temp_files = set()
                # Register cleanup on exit
                atexit.register(self.cleanup_all)
            
            def create_temp_file(self, prefix="temp_", suffix=".tmp"):
                """Create a managed temporary file"""
                fd, filepath = tempfile.mkstemp(
                    prefix=prefix, suffix=suffix, dir=self.base_dir
                )
                os.close(fd)  # Close the file descriptor
                self.temp_files.add(filepath)
                return filepath
            
            def cleanup_file(self, filepath):
                """Clean up a specific file"""
                if filepath in self.temp_files:
                    try:
                        os.remove(filepath)
                        self.temp_files.remove(filepath)
                    except OSError:
                        pass
            
            def cleanup_all(self):
                """Clean up all managed files"""
                for filepath in list(self.temp_files):
                    self.cleanup_file(filepath)
        
        manager = TempFileManager(cleanup_dir)
        
        # Create some temp files
        temp_files = [manager.create_temp_file() for _ in range(5)]
        
        # Verify files exist
        for filepath in temp_files:
            assert os.path.exists(filepath)
        
        # Clean up specific file
        manager.cleanup_file(temp_files[0])
        assert not os.path.exists(temp_files[0])
        
        # Clean up all remaining
        manager.cleanup_all()
        for filepath in temp_files[1:]:
            assert not os.path.exists(filepath)
    
    def test_resource_leak_detection(self):
        """Test detection and prevention of resource leaks"""
        leaked_resources = []
        
        class FileResource:
            def __init__(self, filepath):
                self.filepath = filepath
                self.file = open(filepath, 'w')
                self.closed = False
            
            def write(self, data):
                if not self.closed:
                    self.file.write(data)
            
            def close(self):
                if not self.closed:
                    self.file.close()
                    self.closed = True
            
            def __del__(self):
                if not self.closed:
                    # Resource leak detected
                    leaked_resources.append(self.filepath)
                    self.close()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create resource without proper cleanup
            resource1 = FileResource(os.path.join(temp_dir, "leak1.txt"))
            resource1.write("data")
            # No close() called - leak!
            
            # Create resource with proper cleanup
            resource2 = FileResource(os.path.join(temp_dir, "proper.txt"))
            resource2.write("data")
            resource2.close()
            
            # Force garbage collection
            del resource1
            del resource2
            gc.collect()
        
        # Should detect the leak
        assert len(leaked_resources) == 1
        assert "leak1.txt" in leaked_resources[0]
    
    def test_weak_reference_cleanup(self, cleanup_dir):
        """Test cleanup using weak references"""
        cleaned_up = []
        
        class TempFileWithWeakRef:
            def __init__(self, filepath):
                self.filepath = filepath
                with open(filepath, 'w') as f:
                    f.write("temp data")
            
            def __del__(self):
                try:
                    os.remove(self.filepath)
                    cleaned_up.append(self.filepath)
                except:
                    pass
        
        # Create temp file
        temp_path = os.path.join(cleanup_dir, "weakref_temp.txt")
        temp_file = TempFileWithWeakRef(temp_path)
        
        # Create weak reference
        weak_ref = weakref.ref(temp_file)
        
        # File should exist
        assert os.path.exists(temp_path)
        assert weak_ref() is not None
        
        # Delete strong reference
        del temp_file
        gc.collect()
        
        # Weak reference should be dead and file cleaned up
        assert weak_ref() is None
        assert not os.path.exists(temp_path)
        assert temp_path in cleaned_up
    
    def test_cleanup_order_dependencies(self, cleanup_dir):
        """Test cleanup with order dependencies"""
        class OrderedCleanup:
            def __init__(self):
                self.cleanup_stack = []
            
            def register(self, cleanup_func, priority=0):
                """Register cleanup function with priority (higher = cleanup first)"""
                self.cleanup_stack.append((priority, cleanup_func))
                self.cleanup_stack.sort(key=lambda x: x[0], reverse=True)
            
            def cleanup(self):
                """Execute cleanup in priority order"""
                for _, func in self.cleanup_stack:
                    try:
                        func()
                    except Exception:
                        pass
        
        cleanup_order = []
        cleaner = OrderedCleanup()
        
        # Create files with dependencies
        db_file = os.path.join(cleanup_dir, "database.db")
        log_file = os.path.join(cleanup_dir, "app.log")
        cache_file = os.path.join(cleanup_dir, "cache.tmp")
        
        for filepath in [db_file, log_file, cache_file]:
            with open(filepath, 'w') as f:
                f.write("data")
        
        # Register cleanup in specific order
        # Cache should be cleaned first, then log, then database
        cleaner.register(lambda: (os.remove(cache_file), cleanup_order.append("cache")), priority=3)
        cleaner.register(lambda: (os.remove(log_file), cleanup_order.append("log")), priority=2)
        cleaner.register(lambda: (os.remove(db_file), cleanup_order.append("db")), priority=1)
        
        # Execute cleanup
        cleaner.cleanup()
        
        # Verify cleanup order
        assert cleanup_order == ["cache", "log", "db"]
        
        # Verify files are cleaned
        assert not os.path.exists(cache_file)
        assert not os.path.exists(log_file)
        assert not os.path.exists(db_file)


class TestDiskSpaceManagement:
    """Test disk space handling and quota management"""
    
    def test_disk_space_monitoring(self):
        """Test monitoring of available disk space"""
        # Get disk usage for current directory
        usage = psutil.disk_usage('.')
        
        # Verify we can get disk statistics
        assert usage.total > 0
        assert usage.used >= 0
        assert usage.free >= 0
        assert 0 <= usage.percent <= 100
        
        # Calculate required space for operation
        required_space_mb = 100
        required_bytes = required_space_mb * 1024 * 1024
        
        # Check if enough space available
        has_space = usage.free > required_bytes
        assert isinstance(has_space, bool)
    
    def test_quota_enforcement(self, tmp_path):
        """Test enforcement of disk space quotas"""
        class DiskQuotaManager:
            def __init__(self, base_dir, quota_bytes):
                self.base_dir = base_dir
                self.quota_bytes = quota_bytes
            
            def get_usage(self):
                """Calculate total size of files in base_dir"""
                total = 0
                for dirpath, dirnames, filenames in os.walk(self.base_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total += os.path.getsize(filepath)
                        except:
                            pass
                return total
            
            def check_quota(self, size_to_add):
                """Check if adding size would exceed quota"""
                current_usage = self.get_usage()
                return (current_usage + size_to_add) <= self.quota_bytes
            
            def write_with_quota(self, filename, data):
                """Write file only if quota allows"""
                data_size = len(data.encode('utf-8')) if isinstance(data, str) else len(data)
                
                if not self.check_quota(data_size):
                    raise IOError(f"Quota exceeded: {data_size} bytes would exceed {self.quota_bytes} quota")
                
                filepath = os.path.join(self.base_dir, filename)
                mode = 'w' if isinstance(data, str) else 'wb'
                with open(filepath, mode) as f:
                    f.write(data)
                
                return filepath
        
        # Set 1MB quota
        quota_manager = DiskQuotaManager(str(tmp_path), 1024 * 1024)
        
        # Write small file - should succeed
        small_data = "x" * 1000  # 1KB
        filepath1 = quota_manager.write_with_quota("small.txt", small_data)
        assert os.path.exists(filepath1)
        
        # Try to write large file exceeding quota - should fail
        large_data = "x" * (2 * 1024 * 1024)  # 2MB
        with pytest.raises(IOError, match="Quota exceeded"):
            quota_manager.write_with_quota("large.txt", large_data)
    
    def test_automatic_cleanup_on_low_space(self, tmp_path):
        """Test automatic cleanup when disk space is low"""
        class AutoCleanupManager:
            def __init__(self, base_dir, min_free_bytes):
                self.base_dir = base_dir
                self.min_free_bytes = min_free_bytes
                self.file_registry = {}  # filepath -> (size, priority, age)
            
            def register_file(self, filepath, priority=0):
                """Register a file with cleanup priority (lower = delete first)"""
                try:
                    stat = os.stat(filepath)
                    self.file_registry[filepath] = (
                        stat.st_size,
                        priority,
                        stat.st_mtime
                    )
                except:
                    pass
            
            def cleanup_if_needed(self):
                """Clean up files if free space is below minimum"""
                usage = psutil.disk_usage(self.base_dir)
                
                if usage.free >= self.min_free_bytes:
                    return []
                
                # Sort files by priority (ascending) and age (oldest first)
                files_to_delete = sorted(
                    self.file_registry.items(),
                    key=lambda x: (x[1][1], x[1][2])
                )
                
                deleted = []
                freed_space = 0
                needed_space = self.min_free_bytes - usage.free
                
                for filepath, (size, priority, age) in files_to_delete:
                    if freed_space >= needed_space:
                        break
                    
                    try:
                        os.remove(filepath)
                        deleted.append(filepath)
                        freed_space += size
                        del self.file_registry[filepath]
                    except:
                        pass
                
                return deleted
        
        # Create manager with high minimum (to trigger cleanup in test)
        manager = AutoCleanupManager(str(tmp_path), 10 * 1024 * 1024 * 1024)  # 10GB
        
        # Create test files with different priorities
        test_files = []
        for i in range(5):
            filepath = tmp_path / f"file_{i}.txt"
            filepath.write_text("x" * 1000)
            test_files.append(str(filepath))
            # Lower priority files should be deleted first
            manager.register_file(str(filepath), priority=i)
        
        # Trigger cleanup (will delete due to high minimum)
        deleted = manager.cleanup_if_needed()
        
        # Should delete lowest priority files first
        assert len(deleted) > 0
        # Verify lower priority files were deleted first
        if len(deleted) < len(test_files):
            remaining_files = [f for f in test_files if f not in deleted]
            # Remaining files should have higher indices (higher priority)
            remaining_indices = [int(Path(f).stem.split('_')[1]) for f in remaining_files]
            deleted_indices = [int(Path(f).stem.split('_')[1]) for f in deleted]
            assert min(remaining_indices) > max(deleted_indices) if deleted_indices else True
    
    def test_large_file_handling(self, tmp_path):
        """Test handling of large files efficiently"""
        class LargeFileHandler:
            def __init__(self, chunk_size=1024*1024):  # 1MB chunks
                self.chunk_size = chunk_size
            
            def write_large_file(self, filepath, size_bytes):
                """Write large file in chunks to avoid memory issues"""
                chunks_written = 0
                bytes_written = 0
                
                with open(filepath, 'wb') as f:
                    while bytes_written < size_bytes:
                        chunk_size = min(self.chunk_size, size_bytes - bytes_written)
                        # Write zeros (efficient for testing)
                        f.write(b'\0' * chunk_size)
                        bytes_written += chunk_size
                        chunks_written += 1
                
                return chunks_written
            
            def read_large_file_chunked(self, filepath):
                """Read large file in chunks"""
                chunks_read = 0
                total_bytes = 0
                
                with open(filepath, 'rb') as f:
                    while True:
                        chunk = f.read(self.chunk_size)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        chunks_read += 1
                
                return chunks_read, total_bytes
            
            def copy_large_file_efficiently(self, src, dst):
                """Copy large file efficiently using chunks"""
                chunks_copied = 0
                
                with open(src, 'rb') as fsrc:
                    with open(dst, 'wb') as fdst:
                        while True:
                            chunk = fsrc.read(self.chunk_size)
                            if not chunk:
                                break
                            fdst.write(chunk)
                            chunks_copied += 1
                
                return chunks_copied
        
        handler = LargeFileHandler(chunk_size=1024)  # 1KB chunks for testing
        
        # Create a "large" file (100KB for testing)
        large_file = tmp_path / "large.dat"
        size = 100 * 1024
        chunks_written = handler.write_large_file(str(large_file), size)
        
        assert large_file.exists()
        assert large_file.stat().st_size == size
        assert chunks_written == 100  # 100 chunks of 1KB
        
        # Read the file
        chunks_read, bytes_read = handler.read_large_file_chunked(str(large_file))
        assert chunks_read == chunks_written
        assert bytes_read == size
        
        # Copy the file
        copy_file = tmp_path / "large_copy.dat"
        chunks_copied = handler.copy_large_file_efficiently(
            str(large_file), str(copy_file)
        )
        assert chunks_copied == chunks_written
        assert copy_file.stat().st_size == size
    
    def test_temp_space_reservation(self, tmp_path):
        """Test reservation of temporary space for operations"""
        class TempSpaceReserver:
            def __init__(self, base_dir):
                self.base_dir = base_dir
                self.reservations = {}
            
            def reserve_space(self, operation_id, size_bytes):
                """Reserve space for an operation"""
                # Create a placeholder file to reserve space
                placeholder = os.path.join(self.base_dir, f".reserve_{operation_id}")
                
                try:
                    # Check available space
                    usage = psutil.disk_usage(self.base_dir)
                    if usage.free < size_bytes:
                        raise IOError(f"Insufficient space: need {size_bytes}, have {usage.free}")
                    
                    # Create placeholder file
                    with open(placeholder, 'wb') as f:
                        f.seek(size_bytes - 1)
                        f.write(b'\0')
                    
                    self.reservations[operation_id] = placeholder
                    return True
                except Exception as e:
                    # Cleanup on failure
                    if os.path.exists(placeholder):
                        os.remove(placeholder)
                    raise e
            
            def use_reservation(self, operation_id, actual_file):
                """Replace reservation with actual file"""
                if operation_id not in self.reservations:
                    raise ValueError(f"No reservation for {operation_id}")
                
                placeholder = self.reservations[operation_id]
                
                # Remove placeholder
                os.remove(placeholder)
                
                # Now the space is available for actual file
                del self.reservations[operation_id]
                return True
            
            def release_reservation(self, operation_id):
                """Release unused reservation"""
                if operation_id in self.reservations:
                    placeholder = self.reservations[operation_id]
                    if os.path.exists(placeholder):
                        os.remove(placeholder)
                    del self.reservations[operation_id]
        
        reserver = TempSpaceReserver(str(tmp_path))
        
        # Reserve space
        operation_id = "test_op_1"
        reserved_size = 10 * 1024  # 10KB
        assert reserver.reserve_space(operation_id, reserved_size)
        
        # Verify placeholder exists
        placeholder_path = tmp_path / f".reserve_{operation_id}"
        assert placeholder_path.exists()
        assert placeholder_path.stat().st_size == reserved_size
        
        # Use the reservation
        actual_file = tmp_path / "actual.dat"
        actual_file.write_bytes(b'x' * 5000)  # Write less than reserved
        assert reserver.use_reservation(operation_id, str(actual_file))
        
        # Placeholder should be gone
        assert not placeholder_path.exists()
        
        # Test releasing unused reservation
        operation_id2 = "test_op_2"
        reserver.reserve_space(operation_id2, 5000)
        reserver.release_reservation(operation_id2)
        assert not (tmp_path / f".reserve_{operation_id2}").exists()


class TestFileIntegrityAndSafety:
    """Test file integrity and safety mechanisms"""
    
    def test_checksum_verification(self, tmp_path):
        """Test file integrity using checksums"""
        class FileIntegrityChecker:
            def __init__(self):
                self.checksums = {}
            
            def calculate_checksum(self, filepath, algorithm='sha256'):
                """Calculate checksum of a file"""
                hash_func = hashlib.new(algorithm)
                with open(filepath, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hash_func.update(chunk)
                return hash_func.hexdigest()
            
            def save_checksum(self, filepath):
                """Calculate and save checksum"""
                checksum = self.calculate_checksum(filepath)
                self.checksums[filepath] = checksum
                return checksum
            
            def verify_integrity(self, filepath):
                """Verify file integrity against saved checksum"""
                if filepath not in self.checksums:
                    return False, "No checksum found"
                
                current_checksum = self.calculate_checksum(filepath)
                expected_checksum = self.checksums[filepath]
                
                if current_checksum == expected_checksum:
                    return True, "File intact"
                else:
                    return False, f"Checksum mismatch: {current_checksum} != {expected_checksum}"
        
        checker = FileIntegrityChecker()
        
        # Create test file
        test_file = tmp_path / "data.txt"
        original_content = "Original content for integrity check"
        test_file.write_text(original_content)
        
        # Save checksum
        original_checksum = checker.save_checksum(str(test_file))
        assert len(original_checksum) == 64  # SHA256 produces 64 hex chars
        
        # Verify integrity - should pass
        is_valid, message = checker.verify_integrity(str(test_file))
        assert is_valid
        assert message == "File intact"
        
        # Corrupt the file
        test_file.write_text("Corrupted content")
        
        # Verify integrity - should fail
        is_valid, message = checker.verify_integrity(str(test_file))
        assert not is_valid
        assert "Checksum mismatch" in message
    
    def test_safe_file_replacement(self, tmp_path):
        """Test safe file replacement with backup"""
        class SafeFileReplacer:
            def __init__(self):
                self.backup_suffix = '.backup'
            
            def replace_file(self, filepath, new_content):
                """Safely replace file content with backup"""
                backup_path = filepath + self.backup_suffix
                
                try:
                    # Create backup if file exists
                    if os.path.exists(filepath):
                        shutil.copy2(filepath, backup_path)
                    
                    # Write new content to temp file
                    temp_path = filepath + '.tmp'
                    with open(temp_path, 'w') as f:
                        f.write(new_content)
                        f.flush()
                        os.fsync(f.fileno())
                    
                    # Atomic rename
                    os.replace(temp_path, filepath)
                    
                    # Remove backup on success
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    
                    return True
                
                except Exception as e:
                    # Restore from backup on failure
                    if os.path.exists(backup_path):
                        shutil.copy2(backup_path, filepath)
                        os.remove(backup_path)
                    raise e
        
        replacer = SafeFileReplacer()
        
        # Create original file
        test_file = tmp_path / "important.conf"
        original = "original configuration"
        test_file.write_text(original)
        
        # Replace successfully
        new_content = "new configuration"
        assert replacer.replace_file(str(test_file), new_content)
        assert test_file.read_text() == new_content
        
        # Verify backup was cleaned up
        assert not (tmp_path / "important.conf.backup").exists()
    
    def test_symlink_and_hardlink_safety(self, tmp_path):
        """Test safe handling of symbolic and hard links"""
        # Create a regular file
        target_file = tmp_path / "target.txt"
        target_file.write_text("target content")
        
        # Create symbolic link
        symlink = tmp_path / "symlink.txt"
        try:
            os.symlink(str(target_file), str(symlink))
            
            # Test symlink detection
            assert os.path.islink(str(symlink))
            assert not os.path.islink(str(target_file))
            
            # Read through symlink
            assert symlink.read_text() == "target content"
            
            # Get real path
            real_path = os.path.realpath(str(symlink))
            assert real_path == str(target_file.resolve())
        except OSError:
            # Skip on systems that don't support symlinks
            pytest.skip("Symlinks not supported")
        
        # Create hard link
        hardlink = tmp_path / "hardlink.txt"
        try:
            os.link(str(target_file), str(hardlink))
            
            # Both should have same inode
            assert os.stat(str(target_file)).st_ino == os.stat(str(hardlink)).st_ino
            
            # Modification through hard link affects original
            hardlink.write_text("modified content")
            assert target_file.read_text() == "modified content"
        except OSError:
            # Skip on systems that don't support hard links
            pytest.skip("Hard links not supported")
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks"""
        base_dir = "/safe/base/directory"
        
        unsafe_paths = [
            "../../../etc/passwd",
            "subdir/../../sensitive.txt",
            "/etc/passwd",  # Absolute path
            "~/../../root/secret",  # Home directory traversal
            "subdir/../../../etc/shadow"
        ]
        
        safe_paths = [
            "normal_file.txt",
            "subdir/file.txt",
            "deep/nested/path/file.txt"
        ]
        
        def is_safe_path(base_dir, user_path):
            """Check if user path is safe (within base_dir)"""
            # Resolve to absolute path
            full_path = os.path.abspath(os.path.join(base_dir, user_path))
            # Check if it's within base_dir
            return full_path.startswith(os.path.abspath(base_dir))
        
        # Test unsafe paths
        for unsafe in unsafe_paths:
            assert not is_safe_path(base_dir, unsafe), f"Path {unsafe} should be unsafe"
        
        # Test safe paths
        for safe in safe_paths:
            assert is_safe_path(base_dir, safe), f"Path {safe} should be safe"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
