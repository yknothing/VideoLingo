# Comprehensive Test Suite for Exception Handling Decorators
# Tests core/utils/decorator.py - Targeting 85% branch coverage

import pytest
import time
import os
from unittest.mock import patch, Mock, call
from functools import wraps

try:
    from core.utils.decorator import (
        except_handler,
        check_file_exists,
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestExceptHandlerDecorator:
    """Comprehensive tests for except_handler decorator"""

    def test_except_handler_no_exception(self):
        """Test decorator when function executes successfully"""

        @except_handler("Test error", retry=0)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_except_handler_single_retry_success(self):
        """Test decorator with single retry that succeeds"""
        call_count = 0

        @except_handler("Test error", retry=1, delay=0.01)
        def function_with_retry():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return "success on retry"

        with patch("time.sleep") as mock_sleep:
            result = function_with_retry()
            assert result == "success on retry"
            assert call_count == 2
            mock_sleep.assert_called_once_with(0.01)

    def test_except_handler_multiple_retries_success(self):
        """Test decorator with multiple retries that eventually succeeds"""
        call_count = 0

        @except_handler("Test error", retry=3, delay=0.01)
        def function_with_multiple_retries():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError(f"Call {call_count} fails")
            return "success after multiple retries"

        with patch("time.sleep") as mock_sleep, patch("rich.print") as mock_print:
            result = function_with_multiple_retries()
            assert result == "success after multiple retries"
            assert call_count == 3

            # Verify sleep was called for each retry
            expected_sleep_calls = [call(0.01), call(0.02)]  # Exponential backoff
            mock_sleep.assert_has_calls(expected_sleep_calls)

            # Verify error messages were printed
            assert mock_print.call_count == 2

    def test_except_handler_all_retries_exhausted(self):
        """Test decorator when all retries are exhausted"""

        @except_handler("Test error", retry=2, delay=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with patch("time.sleep") as mock_sleep, patch(
            "rich.print"
        ) as mock_print, pytest.raises(ValueError, match="Always fails"):
            always_fails()

            # Verify sleep was called for each retry attempt
            expected_sleep_calls = [call(0.01), call(0.02)]
            mock_sleep.assert_has_calls(expected_sleep_calls)

            # Verify error messages were printed for each retry
            assert mock_print.call_count == 3  # 2 retries + 1 final failure

    def test_except_handler_exponential_backoff(self):
        """Test exponential backoff delay calculation"""

        @except_handler("Test error", retry=3, delay=1.0)
        def always_fails():
            raise ValueError("Always fails")

        with patch("time.sleep") as mock_sleep, patch("rich.print"), pytest.raises(
            ValueError
        ):
            always_fails()

            # Verify exponential backoff: delay * (2^i)
            expected_delays = [1.0, 2.0, 4.0]
            actual_delays = [call_args[0][0] for call_args in mock_sleep.call_args_list]
            assert actual_delays == expected_delays

    def test_except_handler_default_return_value(self):
        """Test decorator with default return value on failure"""

        @except_handler("Test error", retry=1, delay=0.01, default_return="default")
        def always_fails():
            raise ValueError("Always fails")

        with patch("time.sleep"), patch("rich.print"):
            result = always_fails()
            assert result == "default"

    def test_except_handler_default_return_none(self):
        """Test decorator with None as default return value"""

        @except_handler("Test error", retry=1, delay=0.01, default_return=None)
        def always_fails():
            raise ValueError("Always fails")

        with patch("time.sleep"), patch("rich.print"):
            result = always_fails()
            assert result is None

    def test_except_handler_zero_retries(self):
        """Test decorator with zero retries"""

        @except_handler("Test error", retry=0)
        def fails_immediately():
            raise ValueError("Immediate failure")

        with patch("rich.print") as mock_print, pytest.raises(
            ValueError, match="Immediate failure"
        ):
            fails_immediately()

            # Should print error once and raise immediately
            mock_print.assert_called_once()

    def test_except_handler_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata"""

        @except_handler("Test error", retry=0)
        def original_function():
            """Original docstring"""
            return "original"

        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original docstring"

    def test_except_handler_with_function_arguments(self):
        """Test decorator with function that takes arguments"""

        @except_handler("Test error", retry=1, delay=0.01)
        def function_with_args(a, b, keyword=None):
            if a == "fail":
                raise ValueError("Argument-based failure")
            return f"args: {a}, {b}, keyword: {keyword}"

        # Test successful call
        result = function_with_args("success", "test", keyword="value")
        assert result == "args: success, test, keyword: value"

        # Test failure and retry
        call_count = 0

        @except_handler("Test error", retry=1, delay=0.01)
        def function_with_retry_args(x):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return f"retry success: {x}"

        with patch("time.sleep"), patch("rich.print"):
            result = function_with_retry_args("test_arg")
            assert result == "retry success: test_arg"
            assert call_count == 2

    def test_except_handler_different_exception_types(self):
        """Test decorator handling different types of exceptions"""
        exception_types = [
            ValueError("Value error"),
            TypeError("Type error"),
            RuntimeError("Runtime error"),
            KeyError("Key error"),
            Exception("Generic exception"),
        ]

        for exc in exception_types:

            @except_handler("Test error", retry=1, delay=0.01)
            def function_raising_exception():
                raise exc

            with patch("time.sleep"), patch("rich.print") as mock_print, pytest.raises(
                type(exc), match=str(exc)
            ):
                function_raising_exception()

                # Verify error message contains exception details
                error_calls = [
                    str(call_args) for call_args in mock_print.call_args_list
                ]
                assert any(str(exc) in call_str for call_str in error_calls)

    def test_except_handler_custom_error_message(self):
        """Test decorator with custom error message"""
        custom_message = "Custom error occurred during processing"

        @except_handler(custom_message, retry=1, delay=0.01)
        def failing_function():
            raise ValueError("Original error")

        with patch("time.sleep"), patch("rich.print") as mock_print, pytest.raises(
            ValueError
        ):
            failing_function()

            # Verify custom error message is used
            printed_messages = [
                str(call_args[0][0]) for call_args in mock_print.call_args_list
            ]
            assert any(custom_message in msg for msg in printed_messages)

    def test_except_handler_retry_counter_display(self):
        """Test that retry counter is correctly displayed in error messages"""

        @except_handler("Test error", retry=3, delay=0.01)
        def always_fails():
            raise ValueError("Test failure")

        with patch("time.sleep"), patch("rich.print") as mock_print, pytest.raises(
            ValueError
        ):
            always_fails()

            # Extract retry counters from printed messages
            printed_args = [call_args[0][0] for call_args in mock_print.call_args_list]

            # Should show retry 1/3, 2/3, 3/3, then final failure
            retry_patterns = ["retry: 1/3", "retry: 2/3", "retry: 3/3", "retry: 4/3"]
            for i, pattern in enumerate(retry_patterns):
                if i < len(printed_args):
                    assert pattern in str(printed_args[i])

    def test_except_handler_nested_decorators(self):
        """Test decorator works correctly when nested with other decorators"""

        def another_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return f"decorated: {result}"

            return wrapper

        @another_decorator
        @except_handler("Test error", retry=1, delay=0.01)
        def nested_function():
            return "success"

        result = nested_function()
        assert result == "decorated: success"

        # Test with exception
        call_count = 0

        @another_decorator
        @except_handler("Test error", retry=1, delay=0.01)
        def nested_function_with_exception():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return "retry success"

        with patch("time.sleep"), patch("rich.print"):
            result = nested_function_with_exception()
            assert result == "decorated: retry success"

    def test_except_handler_thread_safety(self):
        """Test decorator thread safety with concurrent executions"""
        import threading

        results = []
        errors = []

        @except_handler("Test error", retry=2, delay=0.001)
        def concurrent_function(thread_id):
            if thread_id % 2 == 0:  # Even threads fail once
                if not hasattr(concurrent_function, f"called_{thread_id}"):
                    setattr(concurrent_function, f"called_{thread_id}", True)
                    raise ValueError(f"Thread {thread_id} fails first time")
            return f"success_{thread_id}"

        def worker(thread_id):
            try:
                with patch("time.sleep"), patch("rich.print"):
                    result = concurrent_function(thread_id)
                    results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)

        # Verify results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5

        for thread_id, result in results:
            assert result == f"success_{thread_id}"


class TestCheckFileExistsDecorator:
    """Comprehensive tests for check_file_exists decorator"""

    def test_check_file_exists_file_not_exists(self):
        """Test decorator when file doesn't exist - function should execute"""
        call_count = 0

        @check_file_exists("/nonexistent/file.txt")
        def function_to_execute():
            nonlocal call_count
            call_count += 1
            return "executed"

        with patch("os.path.exists", return_value=False):
            result = function_to_execute()
            assert result == "executed"
            assert call_count == 1

    def test_check_file_exists_file_exists(self):
        """Test decorator when file exists - function should be skipped"""
        call_count = 0

        @check_file_exists("/existing/file.txt")
        def function_to_skip():
            nonlocal call_count
            call_count += 1
            return "should not execute"

        with patch("os.path.exists", return_value=True), patch(
            "rich.print"
        ) as mock_print:
            result = function_to_skip()
            assert result is None  # Function returns None when skipped
            assert call_count == 0  # Function was not called

            # Verify warning message was printed
            mock_print.assert_called_once()
            warning_message = str(mock_print.call_args[0][0])
            assert "already exists" in warning_message
            assert "/existing/file.txt" in warning_message
            assert "function_to_skip" in warning_message

    def test_check_file_exists_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata"""

        @check_file_exists("/test/file.txt")
        def original_function():
            """Original docstring"""
            return "original"

        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original docstring"

    def test_check_file_exists_with_function_arguments(self):
        """Test decorator with function that takes arguments"""

        @check_file_exists("/test/file.txt")
        def function_with_args(a, b, keyword=None):
            return f"args: {a}, {b}, keyword: {keyword}"

        # Test when file doesn't exist - function executes normally
        with patch("os.path.exists", return_value=False):
            result = function_with_args("test1", "test2", keyword="value")
            assert result == "args: test1, test2, keyword: value"

        # Test when file exists - function is skipped regardless of arguments
        with patch("os.path.exists", return_value=True), patch("rich.print"):
            result = function_with_args("test1", "test2", keyword="value")
            assert result is None

    def test_check_file_exists_different_file_paths(self):
        """Test decorator with different file path formats"""
        test_paths = [
            "/absolute/path/file.txt",
            "relative/path/file.txt",
            "./current/dir/file.txt",
            "../parent/dir/file.txt",
            "~/home/file.txt",
            "C:\\Windows\\file.txt",  # Windows path
            "",  # Empty path
        ]

        for file_path in test_paths:

            @check_file_exists(file_path)
            def test_function():
                return f"executed for {file_path}"

            # Test file exists case
            with patch("os.path.exists", return_value=True), patch(
                "rich.print"
            ) as mock_print:
                result = test_function()
                assert result is None

                # Verify correct file path is used in warning
                warning_message = str(mock_print.call_args[0][0])
                assert file_path in warning_message or (
                    not file_path and "File <>" in warning_message
                )

    def test_check_file_exists_os_path_exists_exception(self):
        """Test decorator behavior when os.path.exists raises exception"""

        @check_file_exists("/test/file.txt")
        def test_function():
            return "executed despite exception"

        # If os.path.exists raises exception, function should still execute
        with patch("os.path.exists", side_effect=OSError("Permission denied")):
            result = test_function()
            assert result == "executed despite exception"

    def test_check_file_exists_nested_decorators(self):
        """Test decorator works correctly when nested with other decorators"""

        def another_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if result is not None:
                    return f"decorated: {result}"
                return result

            return wrapper

        @another_decorator
        @check_file_exists("/test/file.txt")
        def nested_function():
            return "success"

        # Test when file doesn't exist
        with patch("os.path.exists", return_value=False):
            result = nested_function()
            assert result == "decorated: success"

        # Test when file exists
        with patch("os.path.exists", return_value=True), patch("rich.print"):
            result = nested_function()
            assert result is None  # Neither decorator nor function executed

    def test_check_file_exists_multiple_files_same_function(self):
        """Test using decorator multiple times with different files"""

        # This tests the decorator factory behavior
        def create_test_function(file_path):
            @check_file_exists(file_path)
            def test_function():
                return f"executed for {file_path}"

            return test_function

        func1 = create_test_function("/path1/file.txt")
        func2 = create_test_function("/path2/file.txt")

        with patch("os.path.exists") as mock_exists, patch("rich.print") as mock_print:
            # First file exists, second doesn't
            mock_exists.side_effect = lambda path: path == "/path1/file.txt"

            result1 = func1()
            result2 = func2()

            assert result1 is None  # File exists, function skipped
            assert (
                result2 == "executed for /path2/file.txt"
            )  # File doesn't exist, function executed

            # Verify warning only for first function
            mock_print.assert_called_once()
            warning_message = str(mock_print.call_args[0][0])
            assert "/path1/file.txt" in warning_message

    def test_check_file_exists_thread_safety(self):
        """Test decorator thread safety with concurrent file checks"""
        import threading

        results = []

        @check_file_exists("/shared/file.txt")
        def concurrent_function(thread_id):
            return f"executed_by_thread_{thread_id}"

        def worker(thread_id):
            # Mock different file existence for different threads
            with patch("os.path.exists") as mock_exists, patch("rich.print"):
                # Even thread IDs have existing files, odd don't
                mock_exists.return_value = thread_id % 2 == 0

                result = concurrent_function(thread_id)
                results.append((thread_id, result))

        # Start multiple threads
        threads = []
        for i in range(6):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)

        # Verify results
        assert len(results) == 6

        for thread_id, result in results:
            if thread_id % 2 == 0:  # Even threads (file exists)
                assert result is None
            else:  # Odd threads (file doesn't exist)
                assert result == f"executed_by_thread_{thread_id}"


class TestDecoratorInteraction:
    """Test interaction between both decorators"""

    def test_both_decorators_combined(self):
        """Test combining except_handler and check_file_exists decorators"""
        call_count = 0

        @except_handler("Combined decorator error", retry=1, delay=0.01)
        @check_file_exists("/test/file.txt")
        def combined_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return "success after retry"

        # Test when file doesn't exist and function fails then succeeds
        with patch("os.path.exists", return_value=False), patch("time.sleep"), patch(
            "rich.print"
        ):
            result = combined_function()
            assert result == "success after retry"
            assert call_count == 2

        # Reset for next test
        call_count = 0

        # Test when file exists - function should be skipped entirely
        with patch("os.path.exists", return_value=True), patch("rich.print"):
            result = combined_function()
            assert result is None
            assert call_count == 0  # Function never called due to file existence

    def test_decorator_order_matters(self):
        """Test that decorator order affects behavior"""
        call_count = 0

        # Order 1: check_file_exists first, then except_handler
        @check_file_exists("/test/file.txt")
        @except_handler("Error in order 1", retry=1, delay=0.01)
        def function_order1():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        # Order 2: except_handler first, then check_file_exists
        @except_handler("Error in order 2", retry=1, delay=0.01)
        @check_file_exists("/test/file.txt")
        def function_order2():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        call_count = 0

        # Test order 1 - file exists, should skip entirely (no exception handling needed)
        with patch("os.path.exists", return_value=True), patch("rich.print"):
            result = function_order1()
            assert result is None
            assert call_count == 0

        call_count = 0

        # Test order 2 - file exists, should still skip (check happens inside except_handler)
        with patch("os.path.exists", return_value=True), patch("rich.print"):
            result = function_order2()
            assert result is None
            assert call_count == 0


class TestRealWorldScenarios:
    """Test decorators in realistic usage scenarios"""

    def test_file_processing_scenario(self):
        """Test decorators in a file processing scenario"""

        @except_handler("File processing failed", retry=2, delay=0.01)
        @check_file_exists("/output/processed_file.txt")
        def process_file(input_path, output_path):
            # Simulate file processing
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # Simulate processing that might fail
            import random

            if random.random() < 0.3:  # 30% chance of failure
                raise RuntimeError("Processing error occurred")

            return f"Processed {input_path} -> {output_path}"

        # Test successful processing (output doesn't exist yet)
        with patch("os.path.exists") as mock_exists, patch("time.sleep"), patch(
            "rich.print"
        ), patch(
            "random.random", return_value=0.5
        ):  # No random failure
            mock_exists.side_effect = (
                lambda path: path == "/input/file.txt"
            )  # Input exists, output doesn't

            result = process_file("/input/file.txt", "/output/processed_file.txt")
            assert "Processed" in result

        # Test skipped processing (output already exists)
        with patch("os.path.exists") as mock_exists, patch("rich.print"):
            mock_exists.return_value = True  # Both input and output exist

            result = process_file("/input/file.txt", "/output/processed_file.txt")
            assert result is None  # Skipped due to existing output

    def test_api_request_scenario(self):
        """Test decorators in an API request scenario"""

        @except_handler("API request failed", retry=3, delay=0.1)
        def make_api_request(endpoint, retries_made=None):
            if retries_made is None:
                retries_made = []

            # Simulate network issues
            if len(retries_made) < 2:
                retries_made.append(len(retries_made))
                raise ConnectionError(f"Network error {len(retries_made)}")

            return {"status": "success", "endpoint": endpoint}

        with patch("time.sleep"), patch("rich.print"):
            result = make_api_request("https://api.example.com/data")
            assert result["status"] == "success"
            assert result["endpoint"] == "https://api.example.com/data"

    def test_data_backup_scenario(self):
        """Test decorators in a data backup scenario"""

        @except_handler("Backup failed", retry=1, delay=0.01, default_return=False)
        @check_file_exists("/backup/data_backup.zip")
        def create_backup(source_dir):
            # Simulate backup creation
            if not os.path.exists(source_dir):
                raise FileNotFoundError(f"Source directory not found: {source_dir}")

            # Simulate successful backup
            return True

        # Test backup creation when backup doesn't exist
        with patch("os.path.exists") as mock_exists, patch("time.sleep"), patch(
            "rich.print"
        ):
            mock_exists.side_effect = (
                lambda path: path == "/data/source"
            )  # Source exists, backup doesn't

            result = create_backup("/data/source")
            assert result is True

        # Test skipped backup when backup already exists
        with patch("os.path.exists", return_value=True), patch("rich.print"):
            result = create_backup("/data/source")
            assert result is None  # Skipped due to existing backup
