import glob
import os
import re
import subprocess
import sys

from core.utils import *
from core.utils.security_utils import (
    sanitize_filename,
    sanitize_path,
    validate_proxy_url,
)

# Import modular components for enhanced functionality
try:
    from core.download import (
        VideoFileValidator,
        VideoFormatSelector,
        DownloadErrorHandler,
        PartialDownloadCleaner,
    )

    MODULAR_COMPONENTS_AVAILABLE = True
except ImportError:
    # Fallback to legacy implementation if modular components not available
    MODULAR_COMPONENTS_AVAILABLE = False


def update_ytdlp():
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"]
        )
        if "yt_dlp" in sys.modules:
            del sys.modules["yt_dlp"]
        rprint("[green]yt-dlp updated[/green]")
    except subprocess.CalledProcessError as e:
        rprint("[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]")
    from yt_dlp import YoutubeDL

    return YoutubeDL


# ------------
# Enhanced file validation and cleanup functions
# ------------


def validate_video_file(file_path, expected_min_size_mb=1):
    """
    Validate video file integrity and size
    Returns: (is_valid, error_message)
    """
    # Use modular validator if available for enhanced validation
    if MODULAR_COMPONENTS_AVAILABLE:
        try:
            validator = VideoFileValidator(min_size_mb=expected_min_size_mb)
            return validator.validate_video_file(file_path)
        except Exception:
            # Fallback to legacy implementation
            pass

    # Legacy validation implementation
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb < expected_min_size_mb:
        return False, f"File too small ({file_size_mb:.1f}MB), likely incomplete"

    # Check if file is a partial download (common extensions)
    if file_path.endswith((".part", ".tmp", ".download")):
        return False, "File appears to be a partial download"

    # Basic file format validation
    try:
        import subprocess

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, "File format validation failed"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # If ffprobe not available, skip detailed validation
        pass

    return True, "File is valid"


def cleanup_partial_downloads(save_path):
    """
    Clean up partial downloads and temporary files
    """
    # Use modular cleaner if available for enhanced cleanup
    if MODULAR_COMPONENTS_AVAILABLE:
        try:
            cleaner = PartialDownloadCleaner()
            return cleaner.cleanup_partial_downloads(save_path)
        except Exception:
            # Fallback to legacy implementation
            pass

    # Legacy cleanup implementation
    partial_patterns = ["*.part", "*.tmp", "*.download", "*.f*", "*.ytdl"]
    cleaned_files = []

    for pattern in partial_patterns:
        for file_path in glob.glob(os.path.join(save_path, pattern)):
            try:
                os.remove(file_path)
                cleaned_files.append(os.path.basename(file_path))
            except OSError as e:
                rprint(f"[yellow]Warning: Could not remove {file_path}: {e}[/yellow]")

    if cleaned_files:
        rprint(f"[blue]Cleaned up partial files: {', '.join(cleaned_files)}[/blue]")

    return cleaned_files


def find_most_recent_video_file(save_path):
    """
    Find the most recently created video file in the directory
    This is used as a fallback when download path detection fails
    """
    from core.utils.config_utils import load_key

    # Use modular validator if available for enhanced file finding
    if MODULAR_COMPONENTS_AVAILABLE:
        try:
            allowed_formats = load_key("allowed_video_formats")
            validator = VideoFileValidator()
            return validator.find_most_recent_video_file(save_path, allowed_formats)
        except Exception:
            # Fallback to legacy implementation
            pass

    # Legacy implementation
    allowed_formats = load_key("allowed_video_formats")
    video_files = []

    for file_path in glob.glob(os.path.join(save_path, "*")):
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file_path)[1][1:].lower()
        if ext not in allowed_formats:
            continue

        # Skip obvious non-video files
        if file_path.endswith(
            (".jpg", ".jpeg", ".png", ".txt", ".json", ".part", ".tmp")
        ):
            continue

        # Validate file
        is_valid, error_msg = validate_video_file(file_path)
        if not is_valid:
            continue

        # Get file modification time
        mtime = os.path.getmtime(file_path)
        video_files.append((file_path, mtime))

    if not video_files:
        return None

    # Return the most recently modified file
    most_recent = max(video_files, key=lambda x: x[1])[0]
    return most_recent


def find_best_video_file(save_path, allowed_formats):
    """
    Find the best video file when multiple files exist
    Priority: largest valid file with proper format
    """
    # Use modular validator if available for enhanced file finding
    if MODULAR_COMPONENTS_AVAILABLE:
        try:
            validator = VideoFileValidator()
            return validator.find_best_video_file(save_path, allowed_formats)
        except Exception:
            # Fallback to legacy implementation
            pass

    # Legacy implementation
    candidates = []

    for file_path in glob.glob(os.path.join(save_path, "*")):
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file_path)[1][1:].lower()
        if ext not in allowed_formats:
            continue

        # Validate file
        is_valid, error_msg = validate_video_file(file_path)
        if not is_valid:
            rprint(
                f"[yellow]Skipping invalid file {os.path.basename(file_path)}: {error_msg}[/yellow]"
            )
            continue

        file_size = os.path.getsize(file_path)
        candidates.append((file_path, file_size))

    if not candidates:
        return None

    # Return largest valid file
    best_file = max(candidates, key=lambda x: x[1])[0]
    return best_file


def get_optimal_format(resolution):
    """
    获取针对VideoLingo场景优化的视频格式
    优先选择H.264+AAC组合，确保最佳处理性能
    """
    # Use modular format selector if available for enhanced format selection
    if MODULAR_COMPONENTS_AVAILABLE:
        try:
            selector = VideoFormatSelector()
            return selector.get_optimal_format(str(resolution))
        except Exception:
            # Fallback to legacy implementation
            pass

    # Legacy format selection implementation
    if resolution == "best":
        # 最佳质量：按分辨率优先级选择最高可用分辨率
        return "bestvideo[height<=2160][vcodec^=avc1]/bestvideo[height<=2160][vcodec^=h264]/bestvideo[height<=1440][vcodec^=avc1]/bestvideo[height<=1440][vcodec^=h264]/bestvideo[height<=1080][vcodec^=avc1]/bestvideo[height<=1080][vcodec^=h264]/bestvideo[height<=720][vcodec^=avc1]/bestvideo[height<=720][vcodec^=h264]/bestvideo[ext=mp4]/bestvideo+bestaudio[acodec^=mp4a]/bestaudio[ext=m4a]/bestaudio/best[vcodec^=avc1]/best[vcodec^=h264]/best[ext=mp4]/best"
    else:
        # 指定分辨率：优先H.264编码
        height_filter = f"[height<={resolution}]"
        video_part = f"bestvideo{height_filter}[vcodec^=avc1]/bestvideo{height_filter}[vcodec^=h264]/bestvideo{height_filter}[ext=mp4]/bestvideo{height_filter}"
        audio_part = "bestaudio[acodec^=mp4a]/bestaudio[ext=m4a]/bestaudio"
        fallback = f"best{height_filter}[vcodec^=avc1]/best{height_filter}[vcodec^=h264]/best{height_filter}[ext=mp4]/best{height_filter}"
        return f"{video_part}+{audio_part}/{fallback}"


def categorize_download_error(error_msg):
    """
    Categorize download errors to determine retry strategy
    Returns: (category, is_retryable, suggested_wait_time)
    """
    # Use modular error handler if available for enhanced error categorization
    if MODULAR_COMPONENTS_AVAILABLE:
        try:
            handler = DownloadErrorHandler()
            category, is_retryable, wait_time = handler.categorize_download_error(
                error_msg
            )
            return category.value, is_retryable, wait_time
        except Exception:
            # Fallback to legacy implementation
            pass

    # Legacy error categorization implementation
    error_msg_lower = error_msg.lower()

    if any(
        keyword in error_msg_lower
        for keyword in ["network", "timeout", "connection", "temporary"]
    ):
        return "network", True, 30
    elif any(
        keyword in error_msg_lower
        for keyword in ["403", "429", "rate limit", "too many requests"]
    ):
        return "rate_limit", True, 60
    elif any(
        keyword in error_msg_lower for keyword in ["404", "not found", "unavailable"]
    ):
        return "not_found", False, 0
    elif any(
        keyword in error_msg_lower
        for keyword in ["401", "unauthorized", "private", "restricted"]
    ):
        return "access_denied", False, 0
    elif any(keyword in error_msg_lower for keyword in ["proxy", "ssl", "certificate"]):
        return "proxy_ssl", True, 10
    else:
        return "unknown", True, 20


def intelligent_retry_download(download_func, max_retries=3, initial_wait=5):
    """
    Intelligent retry mechanism with exponential backoff
    """
    # Use modular error handler if available for enhanced retry logic
    if MODULAR_COMPONENTS_AVAILABLE:
        try:
            handler = DownloadErrorHandler()
            return handler.intelligent_retry_download(
                download_func, max_retries, initial_wait
            )
        except Exception:
            # Fallback to legacy implementation
            pass

    # Legacy retry implementation
    import time

    for attempt in range(max_retries + 1):
        try:
            return download_func()
        except Exception as e:
            error_msg = str(e)
            category, is_retryable, suggested_wait = categorize_download_error(
                error_msg
            )

            if attempt == max_retries:
                rprint(
                    f"[red]Download failed after {max_retries} retries: {error_msg}[/red]"
                )
                raise e

            if not is_retryable:
                rprint(f"[red]Non-retryable error ({category}): {error_msg}[/red]")
                raise e

            wait_time = min(initial_wait * (2**attempt), suggested_wait)
            rprint(
                f"[yellow]Download attempt {attempt + 1} failed ({category}). Retrying in {wait_time}s...[/yellow]"
            )
            time.sleep(wait_time)

    raise Exception("Max retries exceeded")


def download_video_ytdlp(url, save_path=None, resolution=None, progress_callback=None):
    from core.utils.config_utils import get_storage_paths, ensure_storage_dirs, load_key

    # Use configured paths and resolution if not specified
    if save_path is None:
        paths = get_storage_paths()
        save_path = paths["input"]
    if resolution is None:
        resolution = load_key("youtube_resolution")

    # Debug resolution parameter (can be removed in production)
    print(
        f"[DEBUG] download_video_ytdlp called with resolution: '{resolution}' (type: {type(resolution)})"
    )

    # 资源监控：检查磁盘空间
    try:
        import shutil
        free_bytes = shutil.disk_usage(save_path).free
        free_gb = free_bytes / (1024 * 1024 * 1024)
        if free_gb < 1.0:  # 少于1GB可用空间时警告
            rprint(f"[yellow]Warning: Low disk space ({free_gb:.1f}GB available). Download may fail.[/yellow]")
    except Exception as e:
        rprint(f"[yellow]Could not check disk space: {e}[/yellow]")

    # Ensure all storage directories exist
    ensure_storage_dirs()
    os.makedirs(save_path, exist_ok=True)

    # Define the download function for retry mechanism
    def attempt_download():
        try:
            return download_via_command(
                url, save_path, resolution, progress_callback, outtmpl=None
            )
        except Exception as e:
            rprint(
                f"[yellow]External yt-dlp failed: {e}. Trying Python API...[/yellow]"
            )
            # Fallback to Python API
            return download_via_python_api(
                url, save_path, resolution, progress_callback, outtmpl=None
            )

    # Execute download with intelligent retry & get downloaded path
    downloaded_file = intelligent_retry_download(attempt_download, max_retries=3)

    # Enhanced validation and fallback logic
    if not downloaded_file or not os.path.exists(downloaded_file):
        rprint(
            f"[yellow]Download function returned invalid path, searching for downloaded file...[/yellow]"
        )
        # Fallback: search for the most recent video file in the directory
        try:
            downloaded_file = find_most_recent_video_file(save_path)
            if downloaded_file:
                rprint(
                    f"[green]Found downloaded file: {os.path.basename(downloaded_file)}[/green]"
                )
            else:
                raise Exception("No video file found after download completion")
        except Exception as fallback_error:
            rprint(f"[red]Fallback file search failed: {fallback_error}[/red]")
            raise Exception(
                f"Download failed: Unable to locate downloaded file. Original error: {fallback_error}"
            )

    # Final validation
    if not os.path.exists(downloaded_file):
        raise Exception(f"Download failed: File not found at {downloaded_file}")

    # Validate file integrity
    is_valid, validation_msg = validate_video_file(downloaded_file)
    if not is_valid:
        rprint(
            f"[yellow]Warning: Downloaded file validation failed: {validation_msg}[/yellow]"
        )
        # Don't raise error for validation issues, let user decide

    file_size = os.path.getsize(downloaded_file) / (1024 * 1024)
    rprint(
        f"[green]Download completed successfully: {os.path.basename(downloaded_file)} ({file_size:.1f}MB)[/green]"
    )

    return downloaded_file


def download_via_python_api(
    url, save_path, resolution, progress_callback=None, outtmpl=None
):
    """
    Download using Python yt-dlp API with progress callback
    """
    from core.utils.config_utils import load_key

    downloaded_holder = {"path": None}

    def progress_hook(d):
        if progress_callback and d["status"] == "downloading":
            try:
                downloaded = d.get("downloaded_bytes", 0)
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                if total and total > 0:
                    progress = downloaded / total
                    speed = d.get("speed", 0) or 0
                    eta = d.get("eta", 0) or 0
                    
                    # 资源监控：检查下载速度和剩余时间
                    speed_mbps = (speed / (1024 * 1024)) if speed else 0
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    
                    progress_data = {
                        "progress": progress,
                        "downloaded": downloaded,
                        "total": total,
                        "speed": speed,
                        "eta": eta,
                        "status": "downloading",
                        "downloaded_mb": downloaded_mb,
                        "total_mb": total_mb,
                        "speed_mbps": speed_mbps
                    }
                    
                    # 低速警告
                    if speed_mbps > 0 and speed_mbps < 0.1:  # 小于0.1MB/s
                        progress_data["warning"] = f"Slow download speed: {speed_mbps:.2f}MB/s"
                    
                    progress_callback(progress_data)
            except Exception as e:
                # 记录但不阻止下载
                pass
        elif progress_callback and d["status"] == "finished":
            progress_callback({"progress": 1.0, "status": "finished"})
            downloaded_holder["path"] = d.get("filename")

    ydl_opts = {
        "format": get_optimal_format(resolution),
        "outtmpl": outtmpl or f"{save_path}/%(title).200s.%(ext)s",
        "restrictfilenames": True,  # Enable filename restriction for compatibility
        "windowsfilenames": True,  # Use Windows-safe filenames for cross-platform compatibility
        # Force proper file timestamps
        "postprocessor_args": {"default": ["-atime", "-mtime"]},
        "noplaylist": True,
        "writethumbnail": True,
        "postprocessors": [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}],
        # Add headers to avoid bot detection
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        # Add extra options to bypass bot detection
        "extractor_args": {
            "youtube": {
                "player_client": ["ios", "android", "web"],
                "player_skip": ["webpage"],
                "skip": ["hls", "dash"],
            }
        },
        # Enhanced retry settings for network stability
        "retries": 15,
        "fragment_retries": 15,
        "file_access_retries": 15,
        "continuedl": True,  # Enable resume capability
        "retry_sleep_functions": {
            "http": lambda n: min(4 * n, 30),  # Exponential backoff for HTTP errors
            "fragment": lambda n: min(2 * n, 10),  # Backoff for fragment errors
        },
        "socket_timeout": 30,  # Socket timeout
        "sleep_interval": 1,  # Sleep between requests
        "max_sleep_interval": 5,  # Max sleep between requests
        "sleep_interval_subtitles": 1,  # Sleep interval for subtitles
        "progress_hooks": [progress_hook] if progress_callback else [],
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}],
        "writesubtitles": False,  # Don't download subtitles by default
        "writeautomaticsub": False,  # Don't download auto-generated subtitles
    }

    # Read Youtube Cookie File
    cookies_path = load_key("youtube.cookies_path")
    try:
        # Always try browser cookies first as they're more current
        ydl_opts["cookiesfrombrowser"] = ("chrome",)
    except:
        # If browser cookies fail, try explicit cookie file
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts["cookiefile"] = str(cookies_path)
        else:
            # Continue without cookies as last resort
            pass

    # NEW: Read proxy setting from config if provided
    try:
        proxy_url = load_key("youtube.proxy")
        if proxy_url and validate_proxy_url(proxy_url):
            ydl_opts["proxy"] = proxy_url
        elif proxy_url:
            rprint(f"[yellow]Warning: Invalid proxy URL format: {proxy_url}[/yellow]")
    except KeyError:
        # No proxy configured, silently ignore
        pass

    # Fallback to environment variables if no proxy set yet
    if "proxy" not in ydl_opts:
        proxy_env = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
        )
        if proxy_env and validate_proxy_url(proxy_env):
            ydl_opts["proxy"] = proxy_env
        elif proxy_env:
            rprint(
                f"[yellow]Warning: Invalid proxy URL format in environment: {proxy_env}[/yellow]"
            )

    # Get YoutubeDL class after updating
    YoutubeDL = update_ytdlp()

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        # No cleanup - preserve files per user requirements
        raise e

    # Enhanced file path resolution with fallback
    final_path = None
    if downloaded_holder["path"] and os.path.exists(downloaded_holder["path"]):
        final_path = downloaded_holder["path"]
    else:
        # Fallback: search for the most recent video file
        rprint(
            "[yellow]Could not determine file path from Python API, searching directory...[/yellow]"
        )
        final_path = find_most_recent_video_file(save_path)

    if not final_path:
        raise Exception("Download completed but could not locate the downloaded file")

    # Update timestamp only for the downloaded file
    if final_path and os.path.exists(final_path):
        import time

        current_time = time.time()
        os.utime(final_path, (current_time, current_time))

    rprint(
        f"[green]Download completed successfully: {os.path.basename(final_path)}[/green]"
    )
    return final_path


def download_via_command(
    url, save_path=None, resolution="1080", progress_callback=None, outtmpl=None
):
    """Try using external yt-dlp command for better stability"""
    import subprocess
    import re
    from core.utils.config_utils import get_storage_paths

    # Use configured paths if not specified
    if save_path is None:
        paths = get_storage_paths()
        save_path = paths["input"]

    # Debug external command (can be removed in production)
    print(f"[DEBUG] download_via_command using resolution: '{resolution}'")

    cmd = [
        "yt-dlp",
        "--cookies-from-browser",
        "chrome",
        "--format",
        get_optimal_format(resolution),
        "--output",
        outtmpl or f"{save_path}/%(title).200s.%(ext)s",
        "--restrict-filenames",  # Enable filename restriction for compatibility
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "--no-part",  # Prevent .part files
        "--no-mtime",  # Don't preserve modification time
        url,
    ]

    # Add proxy if configured
    try:
        proxy_url = load_key("youtube.proxy")
        if proxy_url and validate_proxy_url(proxy_url):
            cmd.extend(["--proxy", proxy_url])
        elif proxy_url:
            rprint(f"[yellow]Warning: Invalid proxy URL format: {proxy_url}[/yellow]")
    except:
        # Try environment proxy
        proxy_env = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy_env and validate_proxy_url(proxy_env):
            cmd.extend(["--proxy", proxy_env])
        elif proxy_env:
            rprint(
                f"[yellow]Warning: Invalid proxy URL format in environment: {proxy_env}[/yellow]"
            )

    dest_file_holder = {"path": None}

    def parse_progress_line(line, progress_callback):
        """Parse yt-dlp progress line and call callback"""
        if progress_callback and "[download]" in line:
            # Parse progress like: [download]  45.2% of  117.60MiB at    2.34MiB/s ETA 00:32
            progress_match = re.search(
                r"\[download\]\s+(\d+\.?\d*)%\s+of\s+([\d\.]+\w+)", line
            )
            if progress_match:
                percent = float(progress_match.group(1)) / 100.0
                total_size = progress_match.group(2)

                speed_match = re.search(r"at\s+([\d\.]+\w+/s)", line)
                eta_match = re.search(r"ETA\s+(\d+:\d+)", line)

                progress_callback(
                    {
                        "progress": percent,
                        "total_size_str": total_size,
                        "speed_str": speed_match.group(1) if speed_match else "",
                        "eta_str": eta_match.group(1) if eta_match else "",
                        "status": "downloading",
                    }
                )

        # Enhanced destination file detection with more robust patterns
        if dest_file_holder["path"] is None:
            # Pattern 1: [download] Destination: filename
            m = re.search(r"\[download\]\s+Destination:\s+(.+)", line)
            if not m:
                # Pattern 2: [Merger] Merging formats into "filename"
                m = re.search(r'\[Merger\]\s+Merging formats into\s+"([^"]+)"', line)
            if not m:
                # Pattern 3: [download] filename has already been downloaded
                m = re.search(
                    r"\[download\]\s+(.+)\s+has already been downloaded", line
                )
            if not m:
                # Pattern 4: [ffmpeg] Merging formats into "filename"
                m = re.search(r'\[ffmpeg\]\s+Merging formats into\s+"([^"]+)"', line)
            if not m:
                # Pattern 5: [download] 100% of filename at speed
                m = re.search(r"\[download\]\s+100%\s+of\s+([^\s]+)", line)
            if m:
                detected_path = m.group(1).strip()
                # Security: Use safe path handling to prevent directory traversal
                try:
                    if os.path.isabs(detected_path):
                        # For absolute paths, ensure they're within expected directories
                        safe_path = sanitize_path(
                            save_path, os.path.basename(detected_path)
                        )
                        dest_file_holder["path"] = str(safe_path)
                    else:
                        # For relative paths, sanitize and join safely
                        safe_path = sanitize_path(save_path, detected_path)
                        dest_file_holder["path"] = str(safe_path)
                except Exception as e:
                    # Fallback to safe filename generation
                    rprint(
                        f"[yellow]Path sanitization failed: {e}. Using fallback.[/yellow]"
                    )
                    safe_filename = sanitize_filename(os.path.basename(detected_path))
                    dest_file_holder["path"] = os.path.join(save_path, safe_filename)

    # Run process with real-time output parsing
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
    )

    output_lines = []
    for line in iter(process.stdout.readline, ""):
        output_lines.append(line)
        if progress_callback:
            parse_progress_line(line.strip(), progress_callback)

    process.wait()

    if process.returncode != 0:
        error_output = "".join(output_lines)
        # No cleanup - preserve files per user requirements
        raise Exception(f"yt-dlp command failed: {error_output}")

    if progress_callback:
        progress_callback({"progress": 1.0, "status": "finished"})

    # Enhanced file path resolution with fallback
    final_path = None
    if dest_file_holder["path"] and os.path.exists(dest_file_holder["path"]):
        final_path = dest_file_holder["path"]
    else:
        # Fallback: search for the most recent video file
        rprint(
            "[yellow]Could not determine file path from output, searching directory...[/yellow]"
        )
        final_path = find_most_recent_video_file(save_path)

    if not final_path:
        raise Exception("Download completed but could not locate the downloaded file")

    # Update timestamp only for the downloaded file
    if final_path and os.path.exists(final_path):
        import time

        current_time = time.time()
        os.utime(final_path, (current_time, current_time))

    rprint(
        f"[green]Download completed successfully: {os.path.basename(final_path)}[/green]"
    )
    return final_path


def find_video_files(save_path=None):
    """
    Enhanced video file finder with better multi-file handling
    """
    from core.utils.config_utils import get_storage_paths, load_key

    if save_path is None:
        paths = get_storage_paths()
        save_path = paths["input"]

    allowed_formats = load_key("allowed_video_formats")

    # No cleanup - preserve files per user requirements

    # Find all video files
    video_files = []
    for file_path in glob.glob(os.path.join(save_path, "*")):
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file_path)[1][1:].lower()
        if ext not in allowed_formats:
            continue

        # Skip obvious non-video files
        if file_path.endswith((".jpg", ".jpeg", ".png", ".txt", ".json")):
            continue

        video_files.append(file_path)

    # change \\ to /, this happen on windows
    if sys.platform.startswith("win"):
        video_files = [file.replace("\\", "/") for file in video_files]

    # Filter out output files from old structure
    video_files = [
        file
        for file in video_files
        if not any(file.startswith(prefix) for prefix in ["output/output", "/output"])
    ]

    if len(video_files) == 0:
        raise ValueError(
            "No video files found. Please check if the download was successful."
        )

    if len(video_files) == 1:
        return video_files[0]

    # Multiple files found - find the best one
    rprint(
        f"[yellow]Multiple video files found ({len(video_files)}). Selecting the best one...[/yellow]"
    )

    best_file = find_best_video_file(save_path, allowed_formats)

    if best_file is None:
        # If no valid files found, show detailed info
        rprint("[red]No valid video files found. File details:[/red]")
        for file_path in video_files:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            is_valid, error_msg = validate_video_file(file_path)
            rprint(
                f"  - {os.path.basename(file_path)}: {size_mb:.1f}MB, Valid: {is_valid}, Error: {error_msg}"
            )
        raise ValueError(
            "No valid video files found after validation. Please check the download and try again."
        )

    # Preserve all files - no cleanup per user requirements

    rprint(
        f"[green]Selected video file: {os.path.basename(best_file)} ({os.path.getsize(best_file)/(1024*1024):.1f}MB)[/green]"
    )
    return best_file


if __name__ == "__main__":
    # Example usage
    url = input("Please enter the URL of the video you want to download: ")
    resolution = input(
        "Please enter the desired resolution (360/480/720/1080, default 1080): "
    )
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
