"""
Comprehensive test suite for main download functions in core/_1_ytdlp.py
Tests download_video_ytdlp, download_via_python_api, download_via_command, find_video_files
"""

import pytest
import os
import tempfile
import shutil
import subprocess
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from pathlib import Path
import json
import time
import sys
import re
from contextlib import contextmanager

# Import the module under test
try:
    from core._1_ytdlp import (
        download_video_ytdlp,
        download_via_python_api,
        download_via_command,
        find_video_files,
        find_best_video_file,
    )
except ImportError as e:
    pytest.skip(
        f"Could not import ytdlp download functions: {e}", allow_module_level=True
    )


class TestDownloadVideoYtdlp:
    """Test main download_video_ytdlp function"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_url = "https://www.youtube.com/watch?v=test123"

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.ensure_storage_dirs")
    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.intelligent_retry_download")
    @patch("core._1_ytdlp.validate_video_file")
    @patch("shutil.disk_usage")
    def test_download_video_ytdlp_success(
        self,
        mock_disk_usage,
        mock_validate,
        mock_retry,
        mock_load_key,
        mock_ensure_dirs,
        mock_get_paths,
    ):
        """Test successful video download"""
        # Setup mocks
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = "1080"
        mock_disk_usage.return_value = Mock(free=5 * 1024**3)  # 5GB free
        mock_retry.return_value = os.path.join(self.temp_dir, "downloaded.mp4")
        mock_validate.return_value = (True, "File is valid")

        # Create test downloaded file
        test_file = os.path.join(self.temp_dir, "downloaded.mp4")
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))  # 10MB

        # Execute
        result = download_video_ytdlp(
            self.test_url, save_path=self.temp_dir, resolution="1080"
        )

        # Verify
        assert result == test_file
        mock_ensure_dirs.assert_called_once()
        mock_retry.assert_called_once()
        mock_validate.assert_called_once_with(test_file)

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    @patch("shutil.disk_usage")
    def test_download_video_ytdlp_low_disk_space_warning(
        self, mock_disk_usage, mock_load_key, mock_get_paths
    ):
        """Test warning when disk space is low"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = "1080"
        mock_disk_usage.return_value = Mock(free=500 * 1024**2)  # 500MB free (< 1GB)

        with patch("core._1_ytdlp.intelligent_retry_download") as mock_retry:
            test_file = os.path.join(self.temp_dir, "downloaded.mp4")
            with open(test_file, "wb") as f:
                f.write(b"x" * (10 * 1024 * 1024))

            mock_retry.return_value = test_file

            with patch("core._1_ytdlp.validate_video_file") as mock_validate:
                mock_validate.return_value = (True, "Valid")

                with patch("core._1_ytdlp.rprint") as mock_print:
                    result = download_video_ytdlp(self.test_url)

                    # Should warn about low disk space
                    warning_calls = [
                        call
                        for call in mock_print.call_args_list
                        if "Warning: Low disk space" in str(call)
                    ]
                    assert len(warning_calls) > 0

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    @patch("shutil.disk_usage")
    def test_download_video_ytdlp_disk_usage_error(
        self, mock_disk_usage, mock_load_key, mock_get_paths
    ):
        """Test handling disk usage check error"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = "1080"
        mock_disk_usage.side_effect = OSError("Permission denied")

        with patch("core._1_ytdlp.intelligent_retry_download") as mock_retry:
            test_file = os.path.join(self.temp_dir, "downloaded.mp4")
            with open(test_file, "wb") as f:
                f.write(b"x" * (10 * 1024 * 1024))

            mock_retry.return_value = test_file

            with patch("core._1_ytdlp.validate_video_file") as mock_validate:
                mock_validate.return_value = (True, "Valid")

                # Should not raise exception despite disk usage error
                result = download_video_ytdlp(self.test_url)
                assert result == test_file

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.intelligent_retry_download")
    @patch("core._1_ytdlp.find_most_recent_video_file")
    def test_download_video_ytdlp_fallback_file_search(
        self, mock_find_recent, mock_retry, mock_load_key, mock_get_paths
    ):
        """Test fallback to file search when download returns invalid path"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = "1080"
        mock_retry.return_value = "nonexistent_file.mp4"  # Invalid path

        test_file = os.path.join(self.temp_dir, "found_file.mp4")
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        mock_find_recent.return_value = test_file

        with patch("core._1_ytdlp.validate_video_file") as mock_validate:
            mock_validate.return_value = (True, "Valid")

            result = download_video_ytdlp(self.test_url)

            assert result == test_file
            mock_find_recent.assert_called_once_with(self.temp_dir)

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.intelligent_retry_download")
    @patch("core._1_ytdlp.find_most_recent_video_file")
    def test_download_video_ytdlp_file_not_found_error(
        self, mock_find_recent, mock_retry, mock_load_key, mock_get_paths
    ):
        """Test error when no downloaded file can be found"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = "1080"
        mock_retry.return_value = None
        mock_find_recent.return_value = None

        with pytest.raises(Exception, match="Unable to locate downloaded file"):
            download_video_ytdlp(self.test_url)

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.intelligent_retry_download")
    def test_download_video_ytdlp_validation_warning(
        self, mock_retry, mock_load_key, mock_get_paths
    ):
        """Test handling of file validation warnings"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = "1080"

        test_file = os.path.join(self.temp_dir, "downloaded.mp4")
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        mock_retry.return_value = test_file

        with patch("core._1_ytdlp.validate_video_file") as mock_validate:
            mock_validate.return_value = (False, "File corrupted")

            with patch("core._1_ytdlp.rprint") as mock_print:
                result = download_video_ytdlp(self.test_url)

                # Should still return the file but print warning
                assert result == test_file
                warning_calls = [
                    call
                    for call in mock_print.call_args_list
                    if "validation failed" in str(call)
                ]
                assert len(warning_calls) > 0

    def test_download_video_ytdlp_uses_defaults(self):
        """Test that function uses default paths and resolution when not specified"""
        with patch(
            "core.utils.config_utils.get_storage_paths"
        ) as mock_get_paths, patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "core._1_ytdlp.intelligent_retry_download"
        ) as mock_retry:
            mock_get_paths.return_value = {"input": "/default/input"}
            mock_load_key.return_value = "720"  # Default resolution

            test_file = os.path.join(self.temp_dir, "downloaded.mp4")
            with open(test_file, "wb") as f:
                f.write(b"x" * (10 * 1024 * 1024))

            mock_retry.return_value = test_file

            with patch("core._1_ytdlp.validate_video_file") as mock_validate:
                mock_validate.return_value = (True, "Valid")

                # Call without save_path or resolution
                download_video_ytdlp(self.test_url)

                # Should use loaded defaults
                mock_get_paths.assert_called_once()
                mock_load_key.assert_called_with("youtube_resolution")


class TestDownloadViaPythonApi:
    """Test download_via_python_api function"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_url = "https://www.youtube.com/watch?v=test123"

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core._1_ytdlp.update_ytdlp")
    @patch("core.utils.config_utils.load_key")
    def test_download_via_python_api_success(self, mock_load_key, mock_update_ytdlp):
        """Test successful download via Python API"""
        # Setup mocks
        mock_ydl_class = MagicMock()
        mock_ydl_instance = MagicMock()
        mock_ydl_class.return_value.__enter__ = Mock(return_value=mock_ydl_instance)
        mock_ydl_class.return_value.__exit__ = Mock(return_value=None)
        mock_update_ytdlp.return_value = mock_ydl_class

        mock_load_key.side_effect = lambda key: {
            "youtube.cookies_path": None,
            "youtube.proxy": None,
        }.get(key, None)

        # Create test file that will be "downloaded"
        test_file = os.path.join(self.temp_dir, "test_video.mp4")
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        with patch("core._1_ytdlp.find_most_recent_video_file") as mock_find_recent:
            mock_find_recent.return_value = test_file

            # Execute
            result = download_via_python_api(self.test_url, self.temp_dir, "1080")

            # Verify
            assert result == test_file
            mock_ydl_instance.download.assert_called_once_with([self.test_url])

    @patch("core._1_ytdlp.update_ytdlp")
    @patch("core.utils.config_utils.load_key")
    def test_download_via_python_api_with_progress_callback(
        self, mock_load_key, mock_update_ytdlp
    ):
        """Test download with progress callback"""
        mock_ydl_class = MagicMock()
        mock_ydl_instance = MagicMock()
        mock_ydl_class.return_value.__enter__ = Mock(return_value=mock_ydl_instance)
        mock_ydl_class.return_value.__exit__ = Mock(return_value=None)
        mock_update_ytdlp.return_value = mock_ydl_class

        mock_load_key.side_effect = lambda key: {
            "youtube.cookies_path": None,
            "youtube.proxy": None,
        }.get(key, None)

        progress_callback = Mock()
        test_file = os.path.join(self.temp_dir, "test_video.mp4")
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        with patch("core._1_ytdlp.find_most_recent_video_file") as mock_find_recent:
            mock_find_recent.return_value = test_file

            # Simulate download progress
            def mock_download(urls):
                # Simulate progress hook call
                call_args = mock_ydl_class.call_args[0][0]  # Get ydl_opts
                progress_hooks = call_args.get("progress_hooks", [])
                if progress_hooks:
                    progress_hook = progress_hooks[0]
                    # Simulate downloading progress
                    progress_hook(
                        {
                            "status": "downloading",
                            "downloaded_bytes": 5000000,
                            "total_bytes": 10000000,
                            "speed": 1000000,
                            "eta": 5,
                        }
                    )
                    # Simulate finished
                    progress_hook({"status": "finished", "filename": test_file})

            mock_ydl_instance.download.side_effect = mock_download

            result = download_via_python_api(
                self.test_url, self.temp_dir, "1080", progress_callback
            )

            assert result == test_file
            # Progress callback should have been called
            assert progress_callback.call_count >= 1

    @patch("core._1_ytdlp.update_ytdlp")
    @patch("core.utils.config_utils.load_key")
    def test_download_via_python_api_with_cookies(
        self, mock_load_key, mock_update_ytdlp
    ):
        """Test download with cookies configuration"""
        mock_ydl_class = MagicMock()
        mock_ydl_instance = MagicMock()
        mock_ydl_class.return_value.__enter__ = Mock(return_value=mock_ydl_instance)
        mock_ydl_class.return_value.__exit__ = Mock(return_value=None)
        mock_update_ytdlp.return_value = mock_ydl_class

        cookies_file = os.path.join(self.temp_dir, "cookies.txt")
        with open(cookies_file, "w") as f:
            f.write("# Netscape HTTP Cookie File")

        mock_load_key.side_effect = lambda key: {
            "youtube.cookies_path": cookies_file,
            "youtube.proxy": None,
        }.get(key, None)

        test_file = os.path.join(self.temp_dir, "test_video.mp4")
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        with patch("core._1_ytdlp.find_most_recent_video_file") as mock_find_recent:
            mock_find_recent.return_value = test_file

            download_via_python_api(self.test_url, self.temp_dir, "1080")

            # Check that cookies were configured
            call_args = mock_ydl_class.call_args[0][0]  # Get ydl_opts
            # Should try browser cookies first, then fallback to file
            assert "cookiesfrombrowser" in call_args or "cookiefile" in call_args

    @patch("core._1_ytdlp.update_ytdlp")
    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.validate_proxy_url")
    def test_download_via_python_api_with_proxy(
        self, mock_validate_proxy, mock_load_key, mock_update_ytdlp
    ):
        """Test download with proxy configuration"""
        mock_ydl_class = MagicMock()
        mock_ydl_instance = MagicMock()
        mock_ydl_class.return_value.__enter__ = Mock(return_value=mock_ydl_instance)
        mock_ydl_class.return_value.__exit__ = Mock(return_value=None)
        mock_update_ytdlp.return_value = mock_ydl_class

        proxy_url = "http://proxy.example.com:8080"
        mock_load_key.side_effect = lambda key: {
            "youtube.cookies_path": None,
            "youtube.proxy": proxy_url,
        }.get(key, None)
        mock_validate_proxy.return_value = True

        test_file = os.path.join(self.temp_dir, "test_video.mp4")
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        with patch("core._1_ytdlp.find_most_recent_video_file") as mock_find_recent:
            mock_find_recent.return_value = test_file

            download_via_python_api(self.test_url, self.temp_dir, "1080")

            # Check that proxy was configured
            call_args = mock_ydl_class.call_args[0][0]  # Get ydl_opts
            assert call_args.get("proxy") == proxy_url

    @patch("core._1_ytdlp.update_ytdlp")
    @patch("core.utils.config_utils.load_key")
    def test_download_via_python_api_download_failure(
        self, mock_load_key, mock_update_ytdlp
    ):
        """Test handling of download failure"""
        mock_ydl_class = MagicMock()
        mock_ydl_instance = MagicMock()
        mock_ydl_class.return_value.__enter__ = Mock(return_value=mock_ydl_instance)
        mock_ydl_class.return_value.__exit__ = Mock(return_value=None)
        mock_update_ytdlp.return_value = mock_ydl_class

        mock_load_key.side_effect = lambda key: {
            "youtube.cookies_path": None,
            "youtube.proxy": None,
        }.get(key, None)

        # Mock download to raise exception
        mock_ydl_instance.download.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            download_via_python_api(self.test_url, self.temp_dir, "1080")

    @patch("core._1_ytdlp.update_ytdlp")
    @patch("core.utils.config_utils.load_key")
    def test_download_via_python_api_no_file_found(
        self, mock_load_key, mock_update_ytdlp
    ):
        """Test error when no downloaded file can be found"""
        mock_ydl_class = MagicMock()
        mock_ydl_instance = MagicMock()
        mock_ydl_class.return_value.__enter__ = Mock(return_value=mock_ydl_instance)
        mock_ydl_class.return_value.__exit__ = Mock(return_value=None)
        mock_update_ytdlp.return_value = mock_ydl_class

        mock_load_key.side_effect = lambda key: {
            "youtube.cookies_path": None,
            "youtube.proxy": None,
        }.get(key, None)

        with patch("core._1_ytdlp.find_most_recent_video_file") as mock_find_recent:
            mock_find_recent.return_value = None

            with pytest.raises(Exception, match="could not locate the downloaded file"):
                download_via_python_api(self.test_url, self.temp_dir, "1080")


class TestDownloadViaCommand:
    """Test download_via_command function"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_url = "https://www.youtube.com/watch?v=test123"

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("subprocess.Popen")
    @patch("core.utils.config_utils.get_storage_paths")
    def test_download_via_command_success(self, mock_get_paths, mock_popen):
        """Test successful download via command"""
        mock_get_paths.return_value = {"input": self.temp_dir}

        # Create test file
        test_file = os.path.join(self.temp_dir, "test_video.mp4")
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        # Mock subprocess
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout.readline.side_effect = [
            "[download] Destination: test_video.mp4\n",
            "[download] 100% of 10.00MiB\n",
            "",  # End of output
        ]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        with patch("core._1_ytdlp.find_most_recent_video_file") as mock_find_recent:
            mock_find_recent.return_value = test_file

            result = download_via_command(self.test_url, self.temp_dir, "1080")

            assert result == test_file

            # Verify command was called with correct arguments
            call_args = mock_popen.call_args[0][0]  # Get cmd argument
            assert "yt-dlp" in call_args
            assert "--format" in call_args
            assert self.test_url in call_args

    @patch("subprocess.Popen")
    @patch("core.utils.config_utils.get_storage_paths")
    def test_download_via_command_with_progress_callback(
        self, mock_get_paths, mock_popen
    ):
        """Test download with progress callback"""
        mock_get_paths.return_value = {"input": self.temp_dir}

        test_file = os.path.join(self.temp_dir, "test_video.mp4")
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        # Mock subprocess with progress output
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout.readline.side_effect = [
            "[download] Destination: test_video.mp4\n",
            "[download]  45.2% of  117.60MiB at    2.34MiB/s ETA 00:32\n",
            "[download] 100% of 117.60MiB\n",
            "",
        ]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        progress_callback = Mock()

        with patch("core._1_ytdlp.find_most_recent_video_file") as mock_find_recent:
            mock_find_recent.return_value = test_file

            result = download_via_command(
                self.test_url, self.temp_dir, "1080", progress_callback
            )

            assert result == test_file
            # Progress callback should have been called with progress data
            progress_calls = [
                call
                for call in progress_callback.call_args_list
                if call[0][0].get("status") == "downloading"
            ]
            assert len(progress_calls) > 0

    @patch("subprocess.Popen")
    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.validate_proxy_url")
    def test_download_via_command_with_proxy(
        self, mock_validate_proxy, mock_load_key, mock_get_paths, mock_popen
    ):
        """Test download with proxy configuration"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        proxy_url = "http://proxy.example.com:8080"
        mock_load_key.return_value = proxy_url
        mock_validate_proxy.return_value = True

        test_file = os.path.join(self.temp_dir, "test_video.mp4")
        with open(test_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout.readline.side_effect = [""]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        with patch("core._1_ytdlp.find_most_recent_video_file") as mock_find_recent:
            mock_find_recent.return_value = test_file

            download_via_command(self.test_url, self.temp_dir, "1080")

            # Check that proxy was added to command
            call_args = mock_popen.call_args[0][0]
            assert "--proxy" in call_args
            proxy_index = call_args.index("--proxy")
            assert call_args[proxy_index + 1] == proxy_url

    @patch("subprocess.Popen")
    @patch("core.utils.config_utils.get_storage_paths")
    def test_download_via_command_failure(self, mock_get_paths, mock_popen):
        """Test handling of command failure"""
        mock_get_paths.return_value = {"input": self.temp_dir}

        # Mock subprocess to return error
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.stdout.readline.side_effect = ["ERROR: Video not found\n", ""]
        mock_process.wait.return_value = 1
        mock_popen.return_value = mock_process

        with pytest.raises(Exception, match="yt-dlp command failed"):
            download_via_command(self.test_url, self.temp_dir, "1080")

    @patch("subprocess.Popen")
    @patch("core.utils.config_utils.get_storage_paths")
    def test_download_via_command_no_file_found(self, mock_get_paths, mock_popen):
        """Test error when no downloaded file can be found"""
        mock_get_paths.return_value = {"input": self.temp_dir}

        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.stdout.readline.side_effect = [""]
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        with patch("core._1_ytdlp.find_most_recent_video_file") as mock_find_recent:
            mock_find_recent.return_value = None

            with pytest.raises(Exception, match="could not locate the downloaded file"):
                download_via_command(self.test_url, self.temp_dir, "1080")

    @patch("subprocess.Popen")
    def test_download_via_command_path_sanitization(self, mock_popen):
        """Test path sanitization in output parsing"""
        mock_process = Mock()
        mock_process.returncode = 0

        # Test various output patterns with potentially unsafe paths
        test_outputs = [
            "[download] Destination: ../../../etc/passwd\n",
            '[Merger] Merging formats into "../../malicious.mp4"\n',
            "[download] /absolute/path/video.mp4 has already been downloaded\n",
        ]

        for output_line in test_outputs:
            mock_process.stdout.readline.side_effect = [output_line, ""]
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            with patch("core._1_ytdlp.find_most_recent_video_file") as mock_find_recent:
                test_file = os.path.join(self.temp_dir, "safe_video.mp4")
                with open(test_file, "wb") as f:
                    f.write(b"x" * 1024)
                mock_find_recent.return_value = test_file

                # Should handle path sanitization and not fail
                result = download_via_command(self.test_url, self.temp_dir, "1080")
                assert result == test_file


class TestFindVideoFiles:
    """Test find_video_files function"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    def test_find_video_files_single_file(self, mock_load_key, mock_get_paths):
        """Test finding single video file"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = ["mp4", "avi", "mkv"]

        # Create single video file
        video_file = os.path.join(self.temp_dir, "test_video.mp4")
        with open(video_file, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024))

        result = find_video_files(self.temp_dir)
        assert result == video_file

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    def test_find_video_files_no_files(self, mock_load_key, mock_get_paths):
        """Test error when no video files found"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = ["mp4", "avi", "mkv"]

        with pytest.raises(ValueError, match="No video files found"):
            find_video_files(self.temp_dir)

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.find_best_video_file")
    def test_find_video_files_multiple_files(
        self, mock_find_best, mock_load_key, mock_get_paths
    ):
        """Test finding best file when multiple exist"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = ["mp4", "avi", "mkv"]

        # Create multiple video files
        video_files = [
            os.path.join(self.temp_dir, "video1.mp4"),
            os.path.join(self.temp_dir, "video2.mp4"),
            os.path.join(self.temp_dir, "video3.mp4"),
        ]

        for video_file in video_files:
            with open(video_file, "wb") as f:
                f.write(b"x" * (10 * 1024 * 1024))

        best_file = video_files[1]  # Middle file is "best"
        mock_find_best.return_value = best_file

        result = find_video_files(self.temp_dir)
        assert result == best_file
        mock_find_best.assert_called_once()

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    def test_find_video_files_skip_non_video_extensions(
        self, mock_load_key, mock_get_paths
    ):
        """Test skipping non-video file extensions"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = ["mp4", "avi"]

        # Create mixed files
        video_file = os.path.join(self.temp_dir, "video.mp4")
        text_file = os.path.join(self.temp_dir, "readme.txt")
        image_file = os.path.join(self.temp_dir, "thumb.jpg")

        for file_path in [video_file, text_file, image_file]:
            with open(file_path, "wb") as f:
                f.write(b"x" * 1024)

        result = find_video_files(self.temp_dir)
        assert result == video_file

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    def test_find_video_files_skip_excluded_patterns(
        self, mock_load_key, mock_get_paths
    ):
        """Test skipping explicitly excluded file patterns"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = [
            "mp4",
            "jpg",
        ]  # jpg allowed but should be excluded

        # Create files including excluded patterns
        video_file = os.path.join(self.temp_dir, "video.mp4")
        jpg_file = os.path.join(self.temp_dir, "image.jpg")
        json_file = os.path.join(self.temp_dir, "data.json")

        for file_path in [video_file, jpg_file, json_file]:
            with open(file_path, "wb") as f:
                f.write(b"x" * 1024)

        result = find_video_files(self.temp_dir)
        assert result == video_file

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    def test_find_video_files_filter_output_files(self, mock_load_key, mock_get_paths):
        """Test filtering out output files from old structure"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = ["mp4"]

        # Create files including ones that should be filtered out
        input_file = os.path.join(self.temp_dir, "input_video.mp4")
        output_file1 = os.path.join(self.temp_dir, "output", "output_video.mp4")
        output_file2 = os.path.join(
            self.temp_dir, "output_video.mp4"
        )  # Starts with output

        # Create directories
        os.makedirs(os.path.dirname(output_file1), exist_ok=True)

        for file_path in [input_file, output_file1, output_file2]:
            with open(file_path, "wb") as f:
                f.write(b"x" * 1024)

        result = find_video_files(self.temp_dir)
        assert result == input_file

    @patch("core.utils.config_utils.get_storage_paths")
    @patch("core.utils.config_utils.load_key")
    @patch("core._1_ytdlp.find_best_video_file")
    @patch("core._1_ytdlp.validate_video_file")
    def test_find_video_files_no_valid_files(
        self, mock_validate, mock_find_best, mock_load_key, mock_get_paths
    ):
        """Test error when no valid files found after validation"""
        mock_get_paths.return_value = {"input": self.temp_dir}
        mock_load_key.return_value = ["mp4"]
        mock_find_best.return_value = None
        mock_validate.return_value = (False, "Invalid file")

        # Create invalid video files
        for i in range(3):
            video_file = os.path.join(self.temp_dir, f"video{i}.mp4")
            with open(video_file, "wb") as f:
                f.write(b"x" * 1024)  # Small invalid files

        with pytest.raises(
            ValueError, match="No valid video files found after validation"
        ):
            find_video_files(self.temp_dir)

    def test_find_video_files_windows_path_normalization(self):
        """Test Windows path normalization"""
        if not sys.platform.startswith("win"):
            pytest.skip("Windows path test only relevant on Windows")

        with patch(
            "core.utils.config_utils.get_storage_paths"
        ) as mock_get_paths, patch(
            "core.utils.config_utils.load_key"
        ) as mock_load_key, patch(
            "glob.glob"
        ) as mock_glob:
            mock_get_paths.return_value = {"input": self.temp_dir}
            mock_load_key.return_value = ["mp4"]

            # Mock glob to return Windows-style paths
            windows_path = "C:\\temp\\video.mp4"
            mock_glob.return_value = [windows_path]

            with patch("os.path.isfile") as mock_isfile:
                mock_isfile.return_value = True

                result = find_video_files(self.temp_dir)
                # Should convert backslashes to forward slashes
                assert "/" in result or result == windows_path


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
