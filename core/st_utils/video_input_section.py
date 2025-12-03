"""
VideoLingo 视频输入部分 - 重构版本
支持下载和上传两种方式的标签页切换
解耦合视频源与处理功能
"""

import os
import streamlit as st
import logging
import time
try:
    from core._1_ytdlp import download_video_ytdlp, get_playlist_info
except ImportError as e:
    st.error(f"❌ Failed to import download functions: {e}")
    download_video_ytdlp = None
from core.utils.config_utils import get_storage_paths, load_key
from core.utils.video_manager import get_video_manager
from core.utils.file_security import (
    FileSecurityValidator, 
    FileSecurityError, 
    get_user_friendly_error_message
)
from core.utils.session_security import (
    get_secure_session_state,
    SessionSecurityContext,
    require_valid_session,
    session_audit_log,
    SessionValidationError,
    SessionIntegrityError,
    session_logger
)
from core.constants import ConfigContract
from translations.translations import translate as t

# Configure upload security logger
upload_logger = logging.getLogger('videolingo.upload')
if not upload_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    upload_logger.addHandler(handler)
    upload_logger.setLevel(logging.INFO)


def render_video_input_section():
    """渲染视频输入部分，支持下载和上传两种方式"""
    st.header(t("a. Video Input"))

    # 创建标签页
    tab1, tab2 = st.tabs([t("📥 Download Video"), t("📁 Upload Video")])

    current_video_id = None

    with tab1:
        # 下载视频页面
        current_video_id = handle_download_tab()

    with tab2:
        # 上传视频页面
        current_video_id = handle_upload_tab()

    return current_video_id


def handle_download_tab():
    """处理下载视频标签页"""
    video_mgr = get_video_manager()

    # 检查是否有通过下载获得的视频
    try:
        if "download_video_id" in st.session_state:
            video_id = st.session_state.download_video_id
            video_paths = video_mgr.get_video_paths(video_id)
            if os.path.exists(video_paths["input"]):
                st.video(video_paths["input"])
                st.success(f"Video ID: {video_id}")

                # 当前视频指示器
                st.info(f"🎬 **当前活跃视频**: {os.path.basename(video_paths['input'])}")

                # 切换视频按钮组
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 切换到新视频", key="switch_download_video", use_container_width=True):
                        # 不删除文件，只清除session状态以允许新的下载
                        del st.session_state.download_video_id
                        st.success("已切换到新视频模式，可以下载新视频")
                        st.rerun()
                with col2:
                    if st.button(
                        "🗑️ 删除当前视频", key="delete_download_video", use_container_width=True
                    ):
                        try:
                            # 完全删除当前视频和相关处理文件
                            video_mgr.safe_overwrite_temp_files(video_id)
                            video_mgr.safe_overwrite_output_files(video_id)
                            if os.path.exists(video_paths["input"]):
                                os.remove(video_paths["input"])
                            del st.session_state.download_video_id
                            st.success("视频及相关文件已删除")
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {str(e)}")

                return video_id
    except Exception as e:
        st.error(f"Error loading download video: {str(e)}")
        return None

    # 下载界面
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input(t("Enter YouTube link:"), key="download_url_input")

        # Platform support hint
        st.info(t("Platform Support Hint"))

        # URL type detection with enhanced warnings
        if url:
            detected_platform = detect_url_platform(url)
            if detected_platform:
                if is_unsupported_platform(detected_platform):
                    st.warning(f"⚠️ {t('Unsupported Platform Detected')}: {detected_platform}")
                    st.info(t("Podcast Platform Alternatives Hint"))
                    alternatives = get_supported_podcast_alternatives()
                    with st.expander(t("View Supported Alternatives")):
                        for alt in alternatives:
                            st.write(alt)
                else:
                    st.success(f"🎯 {t('URL Type Detected')}: {detected_platform}")
    with col2:
        res_dict = {
            "360p": "360",
            "720p": "720",
            "1080p": "1080",
            "1440p": "1440",
            "2160p (4K)": "2160",
            "Best Quality": "best",
        }
        target_res = load_key("youtube_resolution")
        res_options = list(res_dict.keys())

        default_idx = 0
        for i, (display_name, res_value) in enumerate(res_dict.items()):
            if res_value == str(target_res):
                default_idx = i
                break

        res_display = st.selectbox(
            t("Resolution"), options=res_options, index=default_idx, key="download_resolution"
        )
        res = res_dict[res_display]

    # 清理旧的播客信息（当URL变化时）
    if url:
        # 清理其他URL的播客信息，只保留当前URL的
        keys_to_remove = [
            key
            for key in st.session_state.keys()
            if key.startswith("podcast_info_") and key != f"podcast_info_{url}"
        ]
        for key in keys_to_remove:
            del st.session_state[key]

    # RSS播客集数选择功能
    if url and is_rss_url(url):
        # 检查是否已经获取了播客信息
        if f"podcast_info_{url}" not in st.session_state:
            if st.button(t("Get Podcast Info"), key="get_podcast_info", use_container_width=True):
                with st.spinner(t("Getting podcast info...")):
                    podcast_info = get_playlist_info(url)
                    if podcast_info:
                        st.session_state[f"podcast_info_{url}"] = podcast_info
                        st.rerun()
                    else:
                        st.error(t("Failed to get podcast info, please check RSS URL"))
                return None
        else:
            # 显示播客信息和集数选择界面
            podcast_info = st.session_state[f"podcast_info_{url}"]
            st.success(
                f"{t('Podcast')}: {podcast_info['title']} ({podcast_info['total_count']}{t('episodes')})"
            )

            # 集数选择选项
            selection_type = st.radio(
                t("Select download range:"),
                [
                    t("Latest 1 episode"),
                    t("Latest 5 episodes"),
                    t("Custom range"),
                    t("Download all"),
                ],
                key="podcast_selection_type",
            )

            playlist_items = None
            if selection_type == t("Latest 1 episode"):
                playlist_items = "1"
            elif selection_type == t("Latest 5 episodes"):
                playlist_items = "1-5"
            elif selection_type == t("Custom range"):
                col1, col2 = st.columns(2)
                with col1:
                    start_ep = st.number_input(
                        t("Start episode"),
                        min_value=1,
                        max_value=podcast_info["total_count"],
                        value=1,
                        key="start_episode",
                    )
                with col2:
                    end_ep = st.number_input(
                        t("End episode"),
                        min_value=start_ep,
                        max_value=podcast_info["total_count"],
                        value=min(start_ep + 4, podcast_info["total_count"]),
                        key="end_episode",
                    )
                playlist_items = f"{start_ep}-{end_ep}" if start_ep != end_ep else str(start_ep)
            # 全部下载时 playlist_items 保持 None

            # 显示即将下载的集数信息
            if playlist_items:
                st.info(f"{t('Will download episodes:')} {playlist_items}")
            else:
                st.info(f"{t('Will download all episodes:')} {podcast_info['total_count']}")

            # 将playlist_items存储到session_state用于下载
            st.session_state["selected_playlist_items"] = playlist_items

    # 并发保护：检查是否正在下载
    download_in_progress = st.session_state.get("download_in_progress", False)

    if download_in_progress:
        st.warning("⚠️ 下载正在进行中，请等待完成...")
        if st.button("🛑 取消下载", key="cancel_download"):
            st.session_state.download_in_progress = False
            st.rerun()
    elif st.button(t("Download Video"), key="download_video_button", use_container_width=True):
        if url:
            # 清空之前的错误状态和重置session状态
            if "download_error" in st.session_state:
                del st.session_state["download_error"]
            
            # 设置下载状态保护
            st.session_state.download_in_progress = True
            # 创建进度跟踪
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                info_text = st.empty()

            def progress_callback(progress_data):
                try:
                    progress = progress_data.get("progress", 0)
                    status = progress_data.get("status", "downloading")
                    message = progress_data.get("message", "")

                    if status == "initializing":
                        progress_bar.progress(0.01)
                        status_text.text("🚀 Initializing download...")
                        info_text.text("Setting up download parameters...")
                        
                    elif status in ["preparing", "extracting"]:
                        progress_bar.progress(min(progress, 0.1))
                        status_text.text(f"🔍 {message or status.title()}...")
                        info_text.text("Extracting video information...")
                        
                    elif status == "working":
                        progress_bar.progress(min(progress, 0.2))
                        status_text.text(f"⚙️ {message or 'Processing'}...")
                        info_text.text("Working on your request...")
                        
                    elif status == "downloading":
                        progress_bar.progress(min(progress, 0.99))
                        if progress > 0:
                            status_text.text(f"⬬ Downloading... {progress:.1%}")
                        else:
                            status_text.text("⬬ Downloading... 0.0%")

                        # 更详细的进度信息
                        if "speed_str" in progress_data and "eta_str" in progress_data:
                            speed = progress_data.get("speed_str", "")
                            eta = progress_data.get("eta_str", "")
                            total_size = progress_data.get("total_size_str", "")
                            info_text.text(f"Size: {total_size} | Speed: {speed} | ETA: {eta}")
                        elif "speed_mbps" in progress_data:
                            speed_mbps = progress_data.get("speed_mbps", 0)
                            downloaded_mb = progress_data.get("downloaded_mb", 0)
                            total_mb = progress_data.get("total_mb", 0)
                            if total_mb > 0:
                                info_text.text(f"Downloaded: {downloaded_mb:.1f}MB / {total_mb:.1f}MB | Speed: {speed_mbps:.2f}MB/s")
                            else:
                                info_text.text(f"Downloaded: {downloaded_mb:.1f}MB | Speed: {speed_mbps:.2f}MB/s")
                        elif message:
                            info_text.text(message)
                        else:
                            info_text.text("Downloading video data...")
                            
                        # Show warnings for slow speeds
                        warning = progress_data.get("warning", "")
                        if warning:
                            info_text.text(f"⚠️ {warning}")
                            
                    elif status == "processing":
                        progress_bar.progress(min(progress, 0.99))
                        status_text.text("🔧 Processing video...")
                        info_text.text("Converting and optimizing...")
                        
                    elif status == "finished":
                        progress_bar.progress(1.0)
                        status_text.text("✅ Download completed successfully!")
                        info_text.text("Video ready for processing")
                        
                    elif status == "error":
                        progress_bar.progress(0)
                        status_text.text("❌ Download error occurred")
                        info_text.text(message or "An error occurred during download")
                        
                    else:
                        # Handle any unknown status
                        progress_bar.progress(min(progress, 0.99))
                        status_text.text(f"📋 {message or status.title()}...")
                        info_text.text("Processing your request...")
                        
                except Exception as callback_error:
                    # 记录回调错误但不中断下载过程
                    print(f"Progress callback error: {callback_error}")
                    # 存储错误信息以便调试
                    st.session_state["progress_callback_error"] = str(callback_error)
                    # Fallback display
                    try:
                        progress_bar.progress(0.1)
                        status_text.text("⚠️ Progress display error")
                        info_text.text(f"Callback error: {callback_error}")
                    except:
                        pass

            status_text.text("🚀 Starting download...")
            info_text.text("Initializing...")

            try:
                # 添加调试信息确认参数正确传递
                st.info(f"🚀 Starting download with resolution: '{res}'")
                st.info(f"📱 URL: {url[:50]}...")

                # Enhanced download with title preservation and concurrent protection
                try:
                    # 检查下载函数是否可用
                    if download_video_ytdlp is None:
                        raise ImportError("Download function not available due to import error")
                    
                    # 获取播客集数选择参数
                    playlist_items = st.session_state.get("selected_playlist_items", None)
                    
                    # 确保URL和res参数不为空
                    if not url or not url.strip():
                        raise ValueError("URL cannot be empty")
                    if not res:
                        res = "1080"  # 默认分辨率
                        
                    # 添加更详细的初始状态
                    status_text.text("🔄 Initializing download process...")
                    info_text.text(f"Target resolution: {res}")
                    
                    # 确保进度回调函数正常工作
                    try:
                        progress_callback({"progress": 0.0, "status": "downloading"})
                    except Exception as callback_test_error:
                        st.warning(f"Progress callback test failed: {callback_test_error}")
                    
                    downloaded_video = download_video_ytdlp(
                        url.strip(),  # 确保URL没有前后空格
                        resolution=res,
                        progress_callback=progress_callback,
                        playlist_items=playlist_items,
                    )
                    video_id = video_mgr.register_video(downloaded_video)
                    st.session_state.download_video_id = video_id

                    # Show success with preserved title
                    file_size = os.path.getsize(downloaded_video) / (1024 * 1024)
                    original_title = os.path.splitext(os.path.basename(downloaded_video))[0]
                    st.success(f"✅ Video downloaded and registered! ID: {video_id}")
                    st.info(f"🎥 标题: {original_title}")
                    st.info(f"📁 文件: {os.path.basename(downloaded_video)} ({file_size:.1f}MB)")

                    # 清除下载状态
                    st.session_state.download_in_progress = False
                    st.rerun()
                except Exception as download_error:
                    # 安全清除下载状态和存储错误信息
                    with SessionSecurityContext() as secure_session:
                        secure_session["download_in_progress"] = False
                        secure_session["download_error"] = str(download_error)
                    raise download_error

            except Exception as e:
                # 确保安全清除下载状态
                with SessionSecurityContext() as secure_session:
                    secure_session["download_in_progress"] = False

                progress_bar.progress(0)
                status_text.text(f"❌ Download failed")
                error_details = str(e)
                info_text.text(f"Error: {error_details}")

                # Enhanced error handling with context-specific messages
                error_msg = str(e)
                st.error(f"❌ Download failed: {error_msg}")
                
                # 添加调试信息显示
                if "progress_callback_error" in st.session_state:
                    st.warning(f"⚠️ Progress callback error: {st.session_state['progress_callback_error']}")
                
                # 显示详细错误信息用于调试
                with st.expander("🔍 Debug Information"):
                    st.code(f"URL: {url}")
                    st.code(f"Resolution: {res}")
                    st.code(f"Error Details: {error_details}")
                    if "download_error" in st.session_state:
                        st.code(f"Previous Download Error: {st.session_state['download_error']}")

                # Check if this is an unsupported platform error
                if any(
                    keyword in error_msg.lower()
                    for keyword in [
                        "unsupported url",
                        "unsupported site",
                        "no suitable formats found",
                    ]
                ):
                    detected_platform = detect_url_platform(url) if url else None
                    st.warning(t("Unsupported Platform Error Message"))

                    if detected_platform and is_unsupported_platform(detected_platform):
                        st.info(f"📱 {t('Detected Platform')}: {detected_platform}")

                    st.info(t("Podcast Platform Alternatives Hint"))
                    alternatives = get_supported_podcast_alternatives()
                    with st.expander(t("View Supported Alternatives")):
                        for alt in alternatives:
                            st.write(alt)

                    st.info(t("Alternative Solution Suggestion"))

                # Provide other context-specific troubleshooting
                elif "could not locate" in error_msg.lower():
                    st.info(
                        "💡 The download may have completed but the file couldn't be found. Check your input directory."
                    )
                elif "validation failed" in error_msg.lower():
                    st.info(
                        "💡 The downloaded file may be corrupted. Try downloading again or check your internet connection."
                    )
                elif "network" in error_msg.lower() or "timeout" in error_msg.lower():
                    st.info(
                        "💡 Network issue detected. Check your internet connection and try again."
                    )
                else:
                    st.info("💡 Try checking the video URL and your internet connection.")
        else:
            st.warning("Please enter a YouTube URL")
            # 确保没有意外的下载状态残留
            if "download_in_progress" in st.session_state:
                st.session_state.download_in_progress = False

    return None


@require_valid_session
@session_audit_log("handle_upload_tab")
def handle_upload_tab():
    """处理上传视频标签页 - 使用安全会话管理"""
    video_mgr = get_video_manager()
    paths = get_storage_paths()

    # 使用安全会话状态
    with SessionSecurityContext() as secure_session:
        # 检查是否有通过上传获得的视频
        try:
            video_id = secure_session.get("upload_video_id")
            if video_id:
                # 验证视频ID格式和权限
                if not _validate_video_id_access(video_id, "upload"):
                    secure_session.pop("upload_video_id", None)
                    st.error("🔒 Session security violation detected. Video access denied.")
                    st.rerun()
                    return None
                    
                video_paths = video_mgr.get_video_paths(video_id)
                if os.path.exists(video_paths["input"]):
                    st.video(video_paths["input"])
                    st.success(f"Video ID: {video_id}")

                    # 当前视频指示器
                    st.info(f"🎬 **当前活跃视频**: {os.path.basename(video_paths['input'])}")

                    # 切换视频按钮组
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 切换到新视频", key="switch_upload_video", use_container_width=True):
                            # 安全清除session状态
                            _secure_video_cleanup(secure_session, "upload_video_id")
                            st.success("已切换到新视频模式，可以上传新视频")
                            st.rerun()
                    with col2:
                        if st.button("🗑️ 删除当前视频", key="delete_upload_video", use_container_width=True):
                            try:
                                # 完全删除当前视频和相关处理文件
                                video_mgr.safe_overwrite_temp_files(video_id)
                                video_mgr.safe_overwrite_output_files(video_id)
                                if os.path.exists(video_paths["input"]):
                                    os.remove(video_paths["input"])
                                _secure_video_cleanup(secure_session, "upload_video_id")
                                st.success("视频及相关文件已删除")
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {str(e)}")

                    return video_id
        except (SessionValidationError, SessionIntegrityError) as e:
            st.error(f"🔒 Session security error: {str(e)}")
            secure_session.clear()
            st.rerun()
            return None
        except Exception as e:
            st.error(f"Error loading upload video: {str(e)}")
            return None

    # Security-enhanced upload interface with comprehensive validation
    with SessionSecurityContext() as secure_session:
        upload_in_progress = secure_session.get("upload_in_progress", False)
    
    # Display security information to users
    with st.expander("📋 Upload Requirements & Security Info", expanded=False):
        st.markdown("""
        **Supported File Types:**
        - 🎥 Video: MP4, AVI, MOV, MKV, WebM, FLV, WMV
        - 🎵 Audio: MP3, WAV, FLAC, AAC, M4A, OGG, WMA
        
        **Security Features:**
        - ✅ File content validation (magic number verification)
        - ✅ Size limits enforced (max 2GB)
        - ✅ Filename sanitization for security
        - ✅ MIME type verification
        - ✅ Secure file storage with restricted permissions
        
        **File Size Limits:**
        - Maximum: 2GB per file
        - Minimum: 1KB
        
        ⚡ **Tip:** For best performance, use MP4 or MP3 files when possible.
        """)

        if upload_in_progress:
            st.warning("⚠️ 上传正在进行中，请等待完成...")
            if st.button("🛑 取消上传", key="cancel_upload"):
                secure_session["upload_in_progress"] = False
                temp_file = secure_session.get("upload_temp_file")
                if temp_file:
                    # Clean up temp file
                    try:
                        os.remove(temp_file)
                        secure_session.pop("upload_temp_file", None)
                    except:
                        pass
                st.rerun()
            uploaded_file = None
        else:
            # Enhanced file uploader with security hints
            uploaded_file = st.file_uploader(
            t("Upload video file"),
            type=load_key("allowed_video_formats") + load_key("allowed_audio_formats"),
            key="upload_file_uploader",
            help="Upload a video or audio file. All files are automatically scanned for security."
        )
        
        # Display upload progress and security status
        if uploaded_file is not None:
            # Show file info before processing
            file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
            st.info(f"📁 File: {uploaded_file.name} ({file_size_mb:.1f} MB)")
            
            # Security status indicator
            with st.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success("🔒 Security Scanning")
                with col2:
                    st.info("📝 Validating Format")
                with col3:
                    st.info("💾 Preparing Storage")

    if uploaded_file:
        # Set upload protection state using secure session
        with SessionSecurityContext() as secure_session:
            secure_session["upload_in_progress"] = True
        upload_start_time = time.time()
        
        # Create progress indicators
        progress_container = st.container()
        with progress_container:
            security_progress = st.progress(0)
            security_status = st.empty()
            security_details = st.empty()
        
        try:
            # Log upload attempt
            upload_logger.info(f"Upload attempt: {uploaded_file.name} ({len(uploaded_file.getbuffer())} bytes)")
            
            # Phase 1: Security Validation (0-30%)
            security_status.text("🔒 Performing security validation...")
            security_progress.progress(0.1)
            
            # Get file data for validation
            file_data = uploaded_file.getbuffer()
            file_size = len(file_data)
            original_filename = uploaded_file.name
            
            # Initialize security validator with configuration settings
            max_size = load_key(ConfigContract.K_SECURITY_MAX_UPLOAD_SIZE)
            enable_virus_scan = load_key(ConfigContract.K_SECURITY_ENABLE_VIRUS_SCAN)
            validator = FileSecurityValidator(
                max_file_size=max_size,
                enable_virus_scan=enable_virus_scan
            )
            
            security_progress.progress(0.2)
            security_details.text("Validating filename and content...")
            
            # Comprehensive security validation
            try:
                sanitized_filename, detected_mime_type = validator.validate_upload(
                    file_data[:1024],  # First 1KB for magic number check
                    original_filename,
                    file_size
                )
                security_progress.progress(0.3)
                security_details.text(f"✅ Security validation passed - {detected_mime_type}")
                
            except FileSecurityError as security_error:
                # Handle security errors with user-friendly messages
                security_progress.progress(0)
                with SessionSecurityContext() as temp_session:
                    temp_session["upload_in_progress"] = False
                
                error_info = get_user_friendly_error_message(security_error)
                st.error(f"{error_info['title']}")
                st.warning(error_info['message'])
                st.info(f"💡 {error_info['suggestion']}")
                
                # Log security violation
                upload_logger.warning(f"Security validation failed for {original_filename}: {security_error}")
                
                # Show technical details in expander for debugging
                with st.expander("🔍 Technical Details"):
                    st.code(f"Original filename: {original_filename}")
                    st.code(f"File size: {file_size} bytes")
                    st.code(f"Error type: {type(security_error).__name__}")
                    st.code(f"Error details: {str(security_error)}")
                
                return None
            
            # Phase 2: Secure File Storage (30-70%)
            security_status.text("💾 Creating secure file storage...")
            security_progress.progress(0.4)
            
            # Create secure temporary file and store path securely
            _, ext = os.path.splitext(sanitized_filename)
            secure_temp_path = validator.create_secure_temp_file(suffix=ext)
            with SessionSecurityContext() as temp_session:
                temp_session["upload_temp_file"] = secure_temp_path
            
            security_progress.progress(0.5)
            security_details.text("Writing file with secure permissions...")
            
            # Write file securely
            try:
                with open(secure_temp_path, "wb") as f:
                    f.write(file_data)
                
                # Verify file was written correctly
                if not os.path.exists(secure_temp_path):
                    raise FileSecurityError("Failed to write file securely")
                    
                written_size = os.path.getsize(secure_temp_path)
                if written_size != file_size:
                    raise FileSecurityError(f"File size mismatch: expected {file_size}, got {written_size}")
                    
                security_progress.progress(0.6)
                security_details.text("✅ Secure storage completed")
                
            except Exception as storage_error:
                # Clean up on storage failure
                if os.path.exists(secure_temp_path):
                    os.remove(secure_temp_path)
                raise FileSecurityError(f"Secure storage failed: {storage_error}")
            
            # Phase 3: Virus Scanning (70-80%)
            if validator.enable_virus_scan:
                security_status.text("🛡️ Scanning for threats...")
                security_progress.progress(0.7)
                
                try:
                    is_clean = validator.scan_for_virus(secure_temp_path)
                    if not is_clean:
                        # Clean up infected file
                        os.remove(secure_temp_path)
                        raise FileSecurityError("File failed virus scan")
                    
                    security_progress.progress(0.8)
                    security_details.text("✅ Virus scan completed - file is clean")
                    
                except Exception as virus_error:
                    # Clean up on virus scan failure
                    if os.path.exists(secure_temp_path):
                        os.remove(secure_temp_path)
                    raise FileSecurityError(f"Virus scan failed: {virus_error}")
            else:
                security_progress.progress(0.8)
                security_details.text("ℹ️ Virus scanning disabled")
            
            # Phase 4: Final Processing (80-100%)
            security_status.text("⚙️ Finalizing upload...")
            security_progress.progress(0.85)
            
            # Move to final location
            final_filename = f"upload_{int(time.time())}_{sanitized_filename}"
            final_path = os.path.join(paths["input"], final_filename)
            
            # Ensure target directory exists
            os.makedirs(paths["input"], exist_ok=True)
            
            # Move file to final location
            import shutil
            shutil.move(secure_temp_path, final_path)
            
            # Set appropriate permissions
            os.chmod(final_path, 0o644)
            
            security_progress.progress(0.9)
            security_details.text("Registering with video management system...")
            
            # Register with video management system and store securely
            video_id = video_mgr.register_video(final_path)
            with SessionSecurityContext() as temp_session:
                temp_session["upload_video_id"] = video_id
            
            security_progress.progress(1.0)
            security_status.text("✅ Upload completed successfully!")
            
            # Calculate upload time
            upload_time = time.time() - upload_start_time
            
            # Display success information
            original_title = os.path.splitext(original_filename)[0]
            file_size_mb = file_size / (1024 * 1024)
            
            st.success(f"✅ Video uploaded and registered! ID: {video_id}")
            
            # Detailed upload summary
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🎥 Original: {original_title}")
                st.info(f"📁 Stored as: {final_filename}")
                st.info(f"📊 Size: {file_size_mb:.1f} MB")
            with col2:
                st.info(f"🔒 MIME Type: {detected_mime_type}")
                st.info(f"⏱️ Upload Time: {upload_time:.1f}s")
                st.info(f"🆔 Video ID: {video_id}")
            
            # Handle audio to video conversion if needed
            if ext.lower() in load_key("allowed_audio_formats"):
                st.info("🔄 Converting audio file to video format...")
                try:
                    from core.st_utils.download_video_section import convert_audio_to_video
                    video_paths = video_mgr.get_video_paths(video_id)
                    convert_audio_to_video(video_paths["input"])
                    st.success("✅ Audio to video conversion completed")
                except Exception as convert_error:
                    st.warning(f"⚠️ Audio conversion failed: {convert_error}")
                    upload_logger.warning(f"Audio conversion failed for {video_id}: {convert_error}")
            
            # Log successful upload
            upload_logger.info(f"Upload successful: {original_filename} -> {video_id} ({file_size_mb:.1f}MB, {upload_time:.1f}s)")
            
            # Clean up session state securely
            with SessionSecurityContext() as temp_session:
                temp_session.pop("upload_temp_file", None)
                temp_session["upload_in_progress"] = False
            
            st.rerun()
            
        except FileSecurityError:
            # Security errors already handled above
            pass
            
        except Exception as e:
            # Handle unexpected errors with secure session cleanup
            with SessionSecurityContext() as error_session:
                error_session["upload_in_progress"] = False
                
                # Clean up any temporary files securely
                temp_file = error_session.get("upload_temp_file")
                if temp_file:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        error_session.pop("upload_temp_file", None)
                    except:
                        pass
            
            # Clean up final file if it exists
            if "final_path" in locals() and os.path.exists(final_path):
                try:
                    os.remove(final_path)
                except:
                    pass
            
            # Display user-friendly error
            st.error("❌ Upload failed due to an unexpected error")
            st.warning(f"Error details: {str(e)}")
            
            # Technical details for debugging
            with st.expander("🔍 Technical Error Details"):
                st.code(f"Filename: {uploaded_file.name if uploaded_file else 'Unknown'}")
                st.code(f"Error type: {type(e).__name__}")
                st.code(f"Error message: {str(e)}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    st.code(f"Traceback: {traceback.format_exc()}")
            
            # Log error
            upload_logger.error(f"Upload failed for {uploaded_file.name if uploaded_file else 'unknown'}: {e}")
            
            st.info("💡 Please try again. If the problem persists, check that your file is a valid video/audio file and not corrupted.")

    return None


def get_current_video_id():
    """获取当前活跃的视频ID"""
    # 优先返回当前标签页的视频
    if "download_video_id" in st.session_state:
        return st.session_state.download_video_id
    elif "upload_video_id" in st.session_state:
        return st.session_state.upload_video_id
    else:
        return None


def clear_video_session():
    """清空视频会话状态"""
    if "download_video_id" in st.session_state:
        del st.session_state.download_video_id
    if "upload_video_id" in st.session_state:
        del st.session_state.upload_video_id


def is_rss_url(url):
    """检测是否为RSS播客链接"""
    if not url:
        return False
    url_lower = url.lower()
    rss_patterns = [".rss", "/rss", "feed://", "feeds.", "/feed/", ".xml", "podcast"]
    return any(pattern in url_lower for pattern in rss_patterns)


def detect_url_platform(url):
    """检测URL对应的平台类型，包含对不支持平台的识别"""
    if not url:
        return None

    url_lower = url.lower()

    # 定义常见支持平台的URL模式
    supported_platform_patterns = {
        "YouTube": ["youtube.com", "youtu.be"],
        "Bilibili": ["bilibili.com", "b23.tv"],
        "TikTok": ["tiktok.com"],
        "Twitter/X": ["twitter.com", "x.com"],
        "Facebook": ["facebook.com", "fb.com"],
        "Instagram": ["instagram.com"],
        "Twitch": ["twitch.tv"],
        "SoundCloud": ["soundcloud.com"],
        "Spotify": ["spotify.com"],
        "Apple Podcasts": ["podcasts.apple.com"],
        "Podcast RSS": [".rss", "/rss", "feed://", "podcast"],
    }

    # 定义已知不支持的播客平台
    unsupported_podcast_platforms = {
        "Everand (Unsupported)": ["everand.com"],
        "Audible (Unsupported)": ["audible.com", "audible.co.uk", "audible.de"],
        "Podcast Addict (Unsupported)": ["podcastaddict.com"],
        "Stitcher (Unsupported)": ["stitcher.com"],
        "iHeartRadio (Unsupported)": ["iheart.com", "iheartradio.com"],
    }

    # 首先检查不支持的平台
    for platform, patterns in unsupported_podcast_platforms.items():
        for pattern in patterns:
            if pattern in url_lower:
                return platform

    # 然后检查支持的平台
    for platform, patterns in supported_platform_patterns.items():
        for pattern in patterns:
            if pattern in url_lower:
                return platform

    # 通用媒体检测
    if any(ext in url_lower for ext in [".mp4", ".mp3", ".wav", ".m4a", ".webm"]):
        return "Direct Media File"

    return "Other Platform (1000+ supported)"


def is_unsupported_platform(platform_name):
    """检查平台是否不支持"""
    if not platform_name:
        return False
    return "(Unsupported)" in platform_name


def get_supported_podcast_alternatives():
    """获取支持的播客平台替代方案"""
    return [
        "• Apple Podcasts (podcasts.apple.com)",
        "• Spotify Podcasts (open.spotify.com/show/...)",
        "• SoundCloud (soundcloud.com)",
        "• YouTube Podcasts (youtube.com)",
        "• RSS Feed URLs (*.rss, feed://...)",
        "• Direct MP3/M4A URLs",
    ]


# 安全会话工具函数
def _validate_video_id_access(video_id: str, source_type: str) -> bool:
    """验证视频ID的访问权限和格式"""
    if not video_id or not isinstance(video_id, str):
        return False
    
    # 验证视频ID格式（UUID或安全标识符）
    import re
    uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    safe_id_pattern = r'^[a-zA-Z0-9_-]{8,64}$'
    
    if not (re.match(uuid_pattern, video_id) or re.match(safe_id_pattern, video_id)):
        session_logger.warning(f"Invalid video ID format: {video_id}")
        return False
    
    # 验证来源类型
    if source_type not in ["download", "upload"]:
        session_logger.warning(f"Invalid source type: {source_type}")
        return False
    
    try:
        # 验证视频管理器中是否存在
        video_mgr = get_video_manager()
        video_paths = video_mgr.get_video_paths(video_id)
        return os.path.exists(video_paths["input"])
    except Exception as e:
        session_logger.warning(f"Video validation error: {e}")
        return False


def _validate_podcast_info(podcast_info) -> bool:
    """验证播客信息的结构和内容"""
    if not isinstance(podcast_info, dict):
        return False
    
    required_fields = ['title', 'total_count']
    for field in required_fields:
        if field not in podcast_info:
            return False
    
    # 验证数据类型
    if not isinstance(podcast_info['title'], str) or not isinstance(podcast_info['total_count'], int):
        return False
    
    # 验证数据范围
    if len(podcast_info['title']) > 500 or podcast_info['total_count'] < 0 or podcast_info['total_count'] > 10000:
        return False
    
    return True


def _secure_video_cleanup(secure_session, video_key: str):
    """安全清理视频相关的session数据"""
    try:
        secure_session.pop(video_key, None)
        # 清理相关的状态
        if video_key == "download_video_id":
            secure_session.pop("download_in_progress", None)
            secure_session.pop("download_error", None)
            secure_session.pop("progress_callback_error", None)
            secure_session.pop("selected_playlist_items", None)
        elif video_key == "upload_video_id":
            secure_session.pop("upload_in_progress", None)
            secure_session.pop("upload_temp_file", None)
        
        session_logger.info(f"Secure cleanup completed for {video_key}")
    except Exception as e:
        session_logger.error(f"Error during secure cleanup: {e}")
