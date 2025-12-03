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
from core.utils.observability import log_event, time_block, inc_counter, observe_histogram


def validate_download_url(url):
    """严格的URL验证，防止命令注入攻击"""
    if not url or not isinstance(url, str):
        raise ValueError("Invalid URL: must be non-empty string")
    
    # 长度限制
    if len(url) > 2048:
        raise ValueError("URL too long")
    
    # 白名单验证 - 只允许支持的视频平台
    allowed_patterns = [
        r'^https?://(?:www\.)?youtube\.com/watch\?v=[a-zA-Z0-9_-]+',
        r'^https?://youtu\.be/[a-zA-Z0-9_-]+',
        r'^https?://(?:www\.)?bilibili\.com/video/[a-zA-Z0-9]+',
        r'^https?://(?:www\.)?twitter\.com/\w+/status/\d+',
        r'^https?://(?:www\.)?tiktok\.com/@[\w\.]+/video/\d+',
    ]
    
    url_stripped = url.strip()
    if not any(re.match(pattern, url_stripped) for pattern in allowed_patterns):
        raise ValueError(f"URL not from allowed domain or invalid format: {url_stripped}")
    
    # 检查危险字符 - 防止命令注入
    dangerous_chars = [';', '&', '|', '`', '$', '(', ')', '<', '>', '"', "'", '\\']
    if any(char in url_stripped for char in dangerous_chars):
        raise ValueError(f"URL contains dangerous characters that could be used for injection")
    
    # 检查控制字符
    if any(ord(char) < 32 for char in url_stripped):
        raise ValueError("URL contains control characters")
    
    return url_stripped


def safe_resolve_download_path(detected_path, base_dir):
    """安全的路径解析，防止目录遍历攻击"""
    import time
    
    if not detected_path or not base_dir:
        raise ValueError("Invalid path parameters")
    
    # 移除引号
    clean_path = detected_path.strip('"\'')
    
    # 只使用文件名，丢弃所有路径信息
    safe_filename = os.path.basename(clean_path)
    
    # 清理文件名，移除危险字符
    safe_filename = re.sub(r'[^\w\-_\.]', '_', safe_filename)
    
    # 检查文件名长度
    if len(safe_filename) > 255:
        safe_filename = safe_filename[:255]
    
    # 如果文件名为空或只有点，生成安全的默认名
    if not safe_filename or safe_filename in ['.', '..'] or safe_filename.startswith('.'):
        safe_filename = f"download_{int(time.time())}.mp4"
    
    # 构建安全路径
    safe_path = os.path.join(base_dir, safe_filename)
    
    # 规范化路径
    safe_path = os.path.normpath(safe_path)
    
    # 确保路径在基目录内（防止目录遍历）
    base_dir_norm = os.path.normpath(base_dir)
    if not safe_path.startswith(base_dir_norm + os.sep) and safe_path != base_dir_norm:
        raise ValueError(f"Path outside allowed directory: {safe_path}")
    
    return safe_path


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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
        if "yt_dlp" in sys.modules:
            del sys.modules["yt_dlp"]
        rprint("[green]yt-dlp updated[/green]")
        log_event("info", "yt-dlp updated", stage="download", op="update_ytdlp")
    except subprocess.CalledProcessError as e:
        rprint("[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]")
        log_event("warning", f"update yt-dlp failed: {e}", stage="download", op="update_ytdlp")
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
        if file_path.endswith((".jpg", ".jpeg", ".png", ".txt", ".json", ".part", ".tmp")):
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

    # Simplified format selection to reduce processing time and hanging
    if resolution == "best":
        return "bestvideo[height<=2160]+bestaudio/best"
    else:
        # Simple format selection with height limit
        height_limit = f"[height<={resolution}]"
        return f"bestvideo{height_limit}+bestaudio/best{height_limit}"


def categorize_download_error(error_msg):
    """
    Categorize download errors to determine retry strategy
    Returns: (category, is_retryable, suggested_wait_time)
    """
    # Use modular error handler if available for enhanced error categorization
    if MODULAR_COMPONENTS_AVAILABLE:
        try:
            handler = DownloadErrorHandler()
            category, is_retryable, wait_time = handler.categorize_download_error(error_msg)
            return category.value, is_retryable, wait_time
        except Exception:
            # Fallback to legacy implementation
            pass

    # Legacy error categorization implementation
    error_msg_lower = error_msg.lower()

    # Enhanced: Check for unsupported URL/platform errors first
    if any(
        keyword in error_msg_lower
        for keyword in [
            "unsupported url",
            "unsupported site",
            "no suitable formats found",
            "no video formats found",
        ]
    ):
        return "unsupported_platform", False, 0
    elif any(
        keyword in error_msg_lower for keyword in ["network", "timeout", "connection", "temporary"]
    ):
        return "network", True, 30
    elif any(
        keyword in error_msg_lower for keyword in ["403", "429", "rate limit", "too many requests"]
    ):
        return "rate_limit", True, 60
    elif any(keyword in error_msg_lower for keyword in ["404", "not found", "unavailable"]):
        return "not_found", False, 0
    elif any(
        keyword in error_msg_lower for keyword in ["401", "unauthorized", "private", "restricted"]
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
            return handler.intelligent_retry_download(download_func, max_retries, initial_wait)
        except Exception:
            # Fallback to legacy implementation
            pass

    # Legacy retry implementation
    import time

    log_event(
        "info",
        "download retry start",
        stage="download",
        op="retry",
        max_retries=max_retries,
        initial_wait=initial_wait,
    )
    for attempt in range(max_retries + 1):
        try:
            with time_block("download attempt", stage="download", op="attempt", attempt=attempt):
                result = download_func()
            if attempt > 0:
                inc_counter(
                    "download.retry_success", 1, stage="download", op="retry", attempt=attempt
                )
            return result
        except Exception as e:
            error_msg = str(e)
            category, is_retryable, suggested_wait = categorize_download_error(error_msg)

            if attempt == max_retries:
                rprint(f"[red]Download failed after {max_retries} retries: {error_msg}[/red]")
                log_event(
                    "error",
                    "download failed after max retries",
                    stage="download",
                    op="retry",
                    attempt=attempt,
                    category=category,
                    error=error_msg,
                )
                inc_counter("download.error", 1, stage="download", category=category)
                raise e

            if not is_retryable:
                rprint(f"[red]Non-retryable error ({category}): {error_msg}[/red]")
                log_event(
                    "error",
                    "non-retryable error",
                    stage="download",
                    op="retry",
                    attempt=attempt,
                    category=category,
                    error=error_msg,
                )
                inc_counter("download.error", 1, stage="download", category=category)
                raise e

            wait_time = min(initial_wait * (2**attempt), suggested_wait)
            rprint(
                f"[yellow]Download attempt {attempt + 1} failed ({category}). Retrying in {wait_time}s...[/yellow]"
            )
            log_event(
                "warning",
                "download attempt failed",
                stage="download",
                op="retry",
                attempt=attempt,
                category=category,
                wait_time=wait_time,
            )
            observe_histogram(
                "download.retry_wait_seconds",
                wait_time,
                stage="download",
                op="retry",
                attempt=attempt,
            )
            time.sleep(wait_time)

    raise Exception("Max retries exceeded")


def get_playlist_info(url):
    """
    Get playlist information for RSS/podcast links without downloading
    Returns: (title, episodes_list) or None if failed
    """
    from core.utils.config_utils import load_key

    try:
        YoutubeDL = update_ytdlp()

        ydl_opts = {
            "extract_flat": True,  # Don't download, just get info
            "quiet": True,
            "no_warnings": True,
        }

        # Add cookies/proxy as in download function
        try:
            ydl_opts["cookiesfrombrowser"] = ("chrome",)
        except:
            cookies_path = load_key("youtube.cookies_path")
            if cookies_path and os.path.exists(cookies_path):
                ydl_opts["cookiefile"] = str(cookies_path)

        try:
            proxy_url = load_key("youtube.proxy")
            if proxy_url and validate_proxy_url(proxy_url):
                ydl_opts["proxy"] = proxy_url
        except:
            pass

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if info.get("_type") == "playlist":
                episodes = []
                for entry in info.get("entries", []):
                    if entry:
                        episodes.append(
                            {
                                "title": entry.get("title", "Unknown Title"),
                                "url": entry.get("url") or entry.get("webpage_url"),
                                "duration": entry.get("duration"),
                                "upload_date": entry.get("upload_date"),
                            }
                        )

                return {
                    "title": info.get("title", "Unknown Podcast"),
                    "episodes": episodes,
                    "total_count": len(episodes),
                }
    except Exception as e:
        rprint(f"[yellow]Failed to get playlist info: {e}[/yellow]")
        return None

    return None


def download_with_timeout(download_func, timeout_seconds=600, progress_callback=None):
    """
    Execute download function with timeout and stuck detection
    """
    import threading
    import time
    
    result = {"success": False, "data": None, "error": None}
    last_progress_time = {"time": time.time(), "progress": 0}
    
    def monitor_progress():
        """Monitor for stuck downloads"""
        while not result["success"] and result["error"] is None:
            time.sleep(30)  # Check every 30 seconds
            current_time = time.time()
            time_since_progress = current_time - last_progress_time["time"]
            
            # If no progress for 300 seconds (5 minutes), consider stuck
            if time_since_progress > 300:
                result["error"] = Exception(f"Download appears stuck - no progress for {time_since_progress:.0f} seconds")
                break
    
    def wrapped_progress_callback(progress_data):
        """Wrapper to track progress timing"""
        if progress_callback:
            progress_callback(progress_data)
        
        # Update progress tracking
        current_progress = progress_data.get("progress", 0)
        if current_progress > last_progress_time["progress"]:
            last_progress_time["time"] = time.time()
            last_progress_time["progress"] = current_progress
    
    def download_thread():
        """Run download in separate thread"""
        try:
            data = download_func(wrapped_progress_callback)
            result["data"] = data
            result["success"] = True
        except Exception as e:
            result["error"] = e
    
    # Start download and monitoring threads
    dl_thread = threading.Thread(target=download_thread)
    monitor_thread = threading.Thread(target=monitor_progress)
    
    dl_thread.start()
    monitor_thread.start()
    
    # Wait for download with timeout
    dl_thread.join(timeout=timeout_seconds)
    
    # Check results
    if dl_thread.is_alive():
        result["error"] = Exception(f"Download timed out after {timeout_seconds} seconds")
    
    # Cleanup monitoring thread
    if monitor_thread.is_alive():
        # Let monitor thread know to exit
        result["success"] = True
        monitor_thread.join(timeout=5)
    
    if result["error"]:
        raise result["error"]
    
    return result["data"]



def download_video_ytdlp(
    url, save_path=None, resolution=None, progress_callback=None, playlist_items=None
):
    from core.utils.config_utils import get_storage_paths, ensure_storage_dirs, load_key

    # SECURITY: 验证URL安全性，防止命令注入
    try:
        safe_url = validate_download_url(url)
    except ValueError as e:
        raise ValueError(f"Security validation failed: {e}")
    
    # Use configured paths and resolution if not specified
    if save_path is None:
        paths = get_storage_paths()
        save_path = paths["input"]
    if resolution is None:
        resolution = load_key("youtube_resolution")

    # Debug resolution parameter (can be removed in production)
    log_event(
        "debug",
        f"download_video_ytdlp params: url={url[:50]}... resolution={resolution} type={type(resolution)}",
        stage="download",
        op="params",
    )
    
    # Validate progress callback if provided
    if progress_callback:
        try:
            progress_callback({"progress": 0.01, "status": "initializing"})
        except Exception as callback_error:
            rprint(f"[yellow]Warning: Progress callback test failed: {callback_error}[/yellow]")
            # Don't fail the download, just proceed without callbacks
            progress_callback = None

    # 资源监控：检查磁盘空间
    try:
        import shutil

        free_bytes = shutil.disk_usage(save_path).free
        free_gb = free_bytes / (1024 * 1024 * 1024)
        if free_gb < 1.0:  # 少于1GB可用空间时警告
            rprint(
                f"[yellow]Warning: Low disk space ({free_gb:.1f}GB available). Download may fail.[/yellow]"
            )
            log_event(
                "warning", "low disk space", stage="download", op="precheck", free_gb=float(free_gb)
            )
    except Exception as e:
        rprint(f"[yellow]Could not check disk space: {e}[/yellow]")
        log_event("warning", f"disk space check failed: {e}", stage="download", op="precheck")

    # Ensure all storage directories exist
    ensure_storage_dirs()
    os.makedirs(save_path, exist_ok=True)

    # Define the download function for retry mechanism
    def attempt_download(progress_cb=None):
        try:
            with time_block("download via command", stage="download", op="command"):
                return download_via_command(
                    url,
                    save_path,
                    resolution,
                    progress_cb or progress_callback,
                    outtmpl=None,
                    playlist_items=playlist_items,
                )
        except Exception as e:
            rprint(f"[yellow]External yt-dlp failed: {e}. Trying Python API...[/yellow]")
            log_event("warning", f"external yt-dlp failed: {e}", stage="download", op="command")
            # Fallback to Python API
            with time_block("download via python api", stage="download", op="python_api"):
                return download_via_python_api(
                    url,
                    save_path,
                    resolution,
                    progress_cb or progress_callback,
                    outtmpl=None,
                    playlist_items=playlist_items,
                )

    # Execute download with timeout and intelligent retry & get downloaded path
    log_event("info", "download start", stage="download", op="download", url=str(url)[:120])
    rprint(f"[blue]Starting download for URL: {url[:50]}...[/blue]")
    
    with time_block("download total", stage="download", op="total"):
        try:
            # Use timeout wrapper for download with stuck detection
            def download_func(cb):
                return attempt_download(progress_cb=cb)
                
            downloaded_file = download_with_timeout(
                download_func, 
                timeout_seconds=1200,  # 20 minute timeout
                progress_callback=progress_callback
            )
        except Exception as timeout_error:
            rprint(f"[red]Download with timeout failed: {timeout_error}[/red]")
            # Fallback to basic retry without timeout (but with shorter retry limit)
            try:
                rprint("[yellow]Attempting basic retry as last resort...[/yellow]")
                downloaded_file = intelligent_retry_download(lambda: attempt_download(), max_retries=2)
            except Exception as basic_error:
                rprint(f"[red]Basic retry also failed: {basic_error}[/red]")
                raise timeout_error

    # Enhanced validation and fallback logic
    if not downloaded_file or not os.path.exists(downloaded_file):
        rprint(
            f"[yellow]Download function returned invalid path, searching for downloaded file...[/yellow]"
        )
        log_event(
            "warning", "downloaded path invalid, try fallback search", stage="download", op="post"
        )
        # Fallback: search for the most recent video file in the directory
        try:
            downloaded_file = find_most_recent_video_file(save_path)
            if downloaded_file:
                rprint(f"[green]Found downloaded file: {os.path.basename(downloaded_file)}[/green]")
                log_event(
                    "info",
                    "fallback found downloaded file",
                    stage="download",
                    op="post",
                    file=os.path.basename(downloaded_file),
                )
            else:
                raise Exception("No video file found after download completion")
        except Exception as fallback_error:
            rprint(f"[red]Fallback file search failed: {fallback_error}[/red]")
            log_event(
                "error", f"fallback search failed: {fallback_error}", stage="download", op="post"
            )
            raise Exception(
                f"Download failed: Unable to locate downloaded file. Original error: {fallback_error}"
            )

    # Final validation
    if not os.path.exists(downloaded_file):
        raise Exception(f"Download failed: File not found at {downloaded_file}")

    # Validate file integrity
    is_valid, validation_msg = validate_video_file(downloaded_file)
    if not is_valid:
        rprint(f"[yellow]Warning: Downloaded file validation failed: {validation_msg}[/yellow]")
        log_event(
            "warning",
            f"downloaded file validation failed: {validation_msg}",
            stage="download",
            op="validate",
        )
        # Don't raise error for validation issues, let user decide

    file_size = os.path.getsize(downloaded_file) / (1024 * 1024)
    rprint(
        f"[green]Download completed successfully: {os.path.basename(downloaded_file)} ({file_size:.1f}MB)[/green]"
    )
    log_event(
        "info",
        "download completed",
        stage="download",
        op="complete",
        file=os.path.basename(downloaded_file),
        size_mb=float(file_size),
    )
    inc_counter("download.success", 1, stage="download")
    observe_histogram("download.file_size_mb", float(file_size), stage="download")

    return downloaded_file


def download_via_python_api(
    url, save_path, resolution, progress_callback=None, outtmpl=None, playlist_items=None
):
    """
    Download using Python yt-dlp API with progress callback
    """
    from core.utils.config_utils import load_key

    downloaded_holder = {"path": None}

    def progress_hook(d):
        if not progress_callback:
            return
        
        # Progress tracking for monitoring
            
        status = d.get("status", "unknown")
        
        try:
            if status == "downloading":
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
                        "speed_mbps": speed_mbps,
                    }

                    # 低速警告
                    if speed_mbps > 0 and speed_mbps < 0.1:  # 小于0.1MB/s
                        progress_data["warning"] = f"Slow download speed: {speed_mbps:.2f}MB/s"

                    progress_callback(progress_data)
                else:
                    # Handle downloading without size info
                    progress_callback({
                        "progress": 0.1, 
                        "status": "downloading", 
                        "message": "Downloading (size unknown)..."
                    })
                    
            elif status == "finished":
                progress_callback({"progress": 1.0, "status": "finished"})
                downloaded_holder["path"] = d.get("filename")
                
            elif status in ["error", "failed"]:
                error_msg = d.get("error", d.get("msg", "Download failed"))
                progress_callback({
                    "progress": 0, 
                    "status": "error", 
                    "message": f"Error: {error_msg}"
                })
                
            elif status in ["preparing", "extracting"]:
                progress_callback({
                    "progress": 0.05, 
                    "status": "preparing", 
                    "message": f"{status.title()}..."
                })
                
            elif status in ["processing", "postprocessing"]:
                progress_callback({
                    "progress": 0.95, 
                    "status": "processing", 
                    "message": "Processing video..."
                })
                
            else:
                # Handle any other status
                progress_callback({
                    "progress": 0.1, 
                    "status": "working", 
                    "message": f"{status.title()}..."
                })
                
        except Exception as e:
            # Log progress callback errors but don't stop download
            print(f"Progress callback error: {e}")
            import traceback
            traceback.print_exc()

    ydl_opts = {
        "format": get_optimal_format(resolution),
        "outtmpl": outtmpl or f"{save_path}/%(title).200s.%(ext)s",
        "restrictfilenames": True,  # Enable filename restriction for compatibility
        "windowsfilenames": True,  # Use Windows-safe filenames for cross-platform compatibility
        # Force proper file timestamps
        "postprocessor_args": {"default": ["-atime", "-mtime"]},
        "noplaylist": False if playlist_items else True,
        "writethumbnail": True,
        "postprocessors": [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}],
        # Add headers to avoid bot detection
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        # Improved extractor args to handle YouTube API issues
        "extractor_args": {
            "youtube": {
                "player_client": ["web", "ios"],  # Prioritize web client, fallback to ios
                "player_skip": ["dash"],  # Skip DASH when problematic
                "skip": ["hls"] if resolution != "best" else [],  # Allow HLS for best quality
            }
        },
        # Enhanced retry settings for network stability and slow connections
        "retries": 20,  # Increased retries for better reliability
        "fragment_retries": 20,  # More retries for fragment downloads
        "file_access_retries": 20,  # More file access retries
        "continuedl": True,  # Enable resume capability
        "retry_sleep_functions": {
            "http": lambda n: min(5 * n, 60),  # More aggressive backoff for HTTP errors
            "fragment": lambda n: min(3 * n, 20),  # Longer backoff for fragment errors
            "file_access": lambda n: min(2 * n, 10),  # File access backoff
        },
        "socket_timeout": 60,  # Longer socket timeout for slow connections
        "read_timeout": 120,  # Read timeout for slow responses
        "sleep_interval": 2,  # Longer sleep between requests to avoid rate limiting
        "max_sleep_interval": 10,  # Max sleep between requests
        "sleep_interval_subtitles": 2,  # Sleep interval for subtitles
        # Add buffer size optimization for slow connections
        "http_chunk_size": 1024 * 1024,  # 1MB chunks for better performance
        "progress_hooks": [progress_hook] if progress_callback else [],
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}],
        "writesubtitles": False,  # Don't download subtitles by default
        "writeautomaticsub": False,  # Don't download auto-generated subtitles
    }

    # Add playlist items selection if specified
    if playlist_items:
        ydl_opts["playlist_items"] = playlist_items

    # Read Youtube Cookie File with improved handling
    cookies_path = load_key("youtube.cookies_path")
    cookies_configured = False
    
    # Try different browser cookie sources
    browsers_to_try = ["chrome", "firefox", "safari", "edge"]
    
    for browser in browsers_to_try:
        try:
            ydl_opts["cookiesfrombrowser"] = (browser,)
            cookies_configured = True
            rprint(f"[blue]Using cookies from {browser}[/blue]")
            break
        except:
            continue
    
    if not cookies_configured:
        # If browser cookies fail, try explicit cookie file
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts["cookiefile"] = str(cookies_path)
            cookies_configured = True
            rprint("[blue]Using cookies from file[/blue]")
        else:
            # Continue without cookies - may limit access to some content
            rprint("[yellow]No cookies available - some content may be inaccessible[/yellow]")

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
            ydl.download([safe_url])
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

    rprint(f"[green]Download completed successfully: {os.path.basename(final_path)}[/green]")
    return final_path


def download_via_command(
    url,
    save_path=None,
    resolution="1080",
    progress_callback=None,
    outtmpl=None,
    playlist_items=None,
):
    """Try using external yt-dlp command for better stability"""
    import subprocess
    import re
    from core.utils.config_utils import get_storage_paths

    # SECURITY: 验证URL安全性，防止命令注入
    try:
        safe_url = validate_download_url(url)
    except ValueError as e:
        raise ValueError(f"Security validation failed: {e}")

    # Use configured paths if not specified
    if save_path is None:
        paths = get_storage_paths()
        save_path = paths["input"]

    # Debug external command (structured)
    log_event(
        "debug",
        f"download_via_command resolution={resolution}",
        stage="download",
        op="command_params",
    )

    cmd = [
        "yt-dlp",
        "--cookies-from-browser",
        "chrome",
        "--format",
        get_optimal_format(resolution),
        "--output",
        outtmpl or f"{save_path}/%(title).200s.%(ext)s",
        "--restrict-filenames",  # Enable filename restriction for compatibility
        "--no-playlist" if not playlist_items else "--yes-playlist",
        "--merge-output-format",
        "mp4",
        "--no-part",  # Prevent .part files
        "--no-mtime",  # Don't preserve modification time
        safe_url,  # 使用验证后的安全URL
    ]

    # Add playlist items selection if specified
    if playlist_items:
        cmd.extend(["--playlist-items", playlist_items])

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
        """Parse yt-dlp progress line and call callback with enhanced status handling"""
        if not progress_callback:
            return
            
        line_lower = line.lower()
        
        # Handle different types of status messages
        if "[download]" in line_lower:
            # Parse progress like: [download]  45.2% of  117.60MiB at    2.34MiB/s ETA 00:32
            progress_match = re.search(r"\[download\]\s+(\d+\.?\d*)%\s+of\s+([\d\.]+\w+)", line)
            if progress_match:
                percent = float(progress_match.group(1)) / 100.0
                total_size = progress_match.group(2)

                speed_match = re.search(r"at\s+([\d\.]+\w+/s)", line)
                eta_match = re.search(r"ETA\s+(\d+:\d+)", line)
                
                speed_str = speed_match.group(1) if speed_match else ""
                
                # Parse speed for warnings
                warning = None
                if speed_str:
                    try:
                        # Extract numeric value and unit
                        speed_match_numeric = re.search(r"([\d\.]+)(\w+)/s", speed_str)
                        if speed_match_numeric:
                            speed_val = float(speed_match_numeric.group(1))
                            speed_unit = speed_match_numeric.group(2).lower()
                            
                            # Convert to MB/s for comparison
                            speed_mbps = speed_val
                            if speed_unit == "kib" or speed_unit == "kb":
                                speed_mbps = speed_val / 1024
                            elif speed_unit == "gib" or speed_unit == "gb":
                                speed_mbps = speed_val * 1024
                                
                            if speed_mbps < 0.1:  # Less than 0.1 MB/s
                                warning = f"Slow download speed: {speed_str}"
                    except:
                        pass

                progress_data = {
                    "progress": percent,
                    "total_size_str": total_size,
                    "speed_str": speed_str,
                    "eta_str": eta_match.group(1) if eta_match else "",
                    "status": "downloading",
                }
                
                if warning:
                    progress_data["warning"] = warning

                progress_callback(progress_data)
            elif "destination:" in line_lower:
                progress_callback({
                    "progress": 0.1, 
                    "status": "preparing", 
                    "message": "Preparing download..."
                })
            elif "sleeping" in line_lower:
                progress_callback({
                    "progress": 0.05, 
                    "status": "preparing", 
                    "message": "Waiting as required by site..."
                })
        elif any(keyword in line_lower for keyword in ["extracting", "downloading webpage", "downloading player"]):
            progress_callback({
                "progress": 0.05, 
                "status": "extracting", 
                "message": "Extracting video information..."
            })
        elif any(keyword in line_lower for keyword in ["merging", "merger", "ffmpeg"]):
            progress_callback({
                "progress": 0.95, 
                "status": "processing", 
                "message": "Processing video files..."
            })
        elif "error" in line_lower and progress_callback:
            progress_callback({
                "progress": 0, 
                "status": "error", 
                "message": "Download error occurred"
            })

        # Enhanced destination file detection with more robust patterns
        if dest_file_holder["path"] is None:
            # Pattern 1: [download] Destination: filename
            m = re.search(r"\[download\]\s+Destination:\s+(.+)", line)
            if not m:
                # Pattern 2: [Merger] Merging formats into "filename"
                m = re.search(r'\[Merger\]\s+Merging formats into\s+"([^"]+)"', line)
            if not m:
                # Pattern 3: [download] filename has already been downloaded
                m = re.search(r"\[download\]\s+(.+)\s+has already been downloaded", line)
            if not m:
                # Pattern 4: [ffmpeg] Merging formats into "filename"
                m = re.search(r'\[ffmpeg\]\s+Merging formats into\s+"([^"]+)"', line)
            if not m:
                # Pattern 5: [download] 100% of filename at speed
                m = re.search(r"\[download\]\s+100%\s+of\s+([^\s]+)", line)
            if m:
                detected_path = m.group(1).strip()
                # SECURITY: 使用增强的安全路径解析，防止目录遍历攻击
                try:
                    safe_path = safe_resolve_download_path(detected_path, save_path)
                    dest_file_holder["path"] = str(safe_path)
                    rprint(f"[green]安全路径解析成功: {os.path.basename(safe_path)}[/green]")
                except Exception as e:
                    # 如果路径解析失败，生成安全的后备文件名
                    rprint(f"[yellow]Path sanitization failed: {e}. Using fallback.[/yellow]")
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
        rprint("[yellow]Could not determine file path from output, searching directory...[/yellow]")
        final_path = find_most_recent_video_file(save_path)

    if not final_path:
        raise Exception("Download completed but could not locate the downloaded file")

    # Update timestamp only for the downloaded file
    if final_path and os.path.exists(final_path):
        import time

        current_time = time.time()
        os.utime(final_path, (current_time, current_time))

    rprint(f"[green]Download completed successfully: {os.path.basename(final_path)}[/green]")
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
        raise ValueError("No video files found. Please check if the download was successful.")

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
    resolution = input("Please enter the desired resolution (360/480/720/1080, default 1080): ")
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
