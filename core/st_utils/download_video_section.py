import os
import re
import shutil
import subprocess
from time import sleep
import threading
import queue
import glob

import streamlit as st
from core._1_ytdlp import download_video_ytdlp, find_video_files, cleanup_partial_downloads, validate_video_file
from core.utils import *
from core.utils.config_utils import get_storage_paths, ensure_storage_dirs
from core.utils.video_manager import get_video_manager
from core.utils.security_utils import generate_safe_filename, sanitize_filename, validate_url
from translations.translations import translate as t

def download_video_section():
    st.header(t("a. Download or Upload Video"))
    with st.container(border=True):
        # Initialize storage directories and video manager
        ensure_storage_dirs()
        paths = get_storage_paths()
        video_mgr = get_video_manager()
        
        try:
            video_file = find_video_files()
            
            # Display existing video preview and info
            file_size_mb = os.path.getsize(video_file) / (1024 * 1024)
            is_valid, validation_msg = validate_video_file(video_file)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.video(video_file)
            with col2:
                st.info(f"📁 **File:** {os.path.basename(video_file)}")
                st.info(f"📊 **Size:** {file_size_mb:.1f} MB")
                if is_valid:
                    st.success(f"✅ **Status:** Valid")
                else:
                    st.warning(f"⚠️ **Status:** {validation_msg}")
                    
                # Add warning if this might not be the intended video
                st.warning("⚠️ **Note:** This video may not correspond to your current URL. Use 'Delete and Reselect' to download the correct video.")
            
            if st.button(t("Delete and Reselect"), key="delete_video_button"):
                # 新架构: 基于视频ID安全地重置处理状态
                try:
                    current_video_id = video_mgr.get_current_video_id()
                    if current_video_id:
                        # 只覆盖temp和output，保留input
                        video_mgr.safe_overwrite_temp_files(current_video_id)
                        video_mgr.safe_overwrite_output_files(current_video_id)
                        st.success(f"已重置视频 {current_video_id} 的处理状态")
                    else:
                        # 如果没有ID，删除当前视频文件
                        os.remove(video_file)
                        st.success("视频文件已删除")
                except Exception as e:
                    st.error(f"重置失败: {str(e)}")
                    # 降级到删除文件
                    try:
                        os.remove(video_file)
                        st.success("视频文件已删除")
                    except:
                        st.error("无法删除视频文件，请手动删除")
                sleep(1)
                st.rerun()
            
            st.markdown("---")  # Add divider between existing video and download form
            st.markdown("### Download New Video")

            # Download section
            col1, col2 = st.columns([3, 1])
            with col1:
                url = st.text_input(t("Enter YouTube link:"))
            with col2:
                res_dict = {
                    "360p": "360",
                    "1080p": "1080",
                    "Best": "best"
                }
                target_res = load_key("youtube_resolution")
                res_options = list(res_dict.keys())
                
                # Find the correct default index based on config value
                default_idx = 0  # default to 360p
                for i, (display_name, res_value) in enumerate(res_dict.items()):
                    if res_value == str(target_res):
                        default_idx = i
                        break
                
                res_display = st.selectbox(t("Resolution"), options=res_options, index=default_idx)
                res = res_dict[res_display]
                
            if st.button(t("Download Video"), key="download_video_button"):
                if url:
                    # Enhanced download with better progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    info_text = st.empty()
                    
                    # Create download state tracking
                    download_state = {
                        'phase': 'initializing',
                        'error': None,
                        'completed': False
                    }
                    
                    def progress_callback(progress_data):
                        try:
                            progress = progress_data.get('progress', 0)
                            status = progress_data.get('status', 'downloading')
                            
                            if status == 'downloading':
                                download_state['phase'] = 'downloading'
                                progress_bar.progress(min(progress, 0.99))  # Keep some margin for completion
                                
                                # Format status message
                                if 'speed_str' in progress_data and 'eta_str' in progress_data:
                                    speed = progress_data.get('speed_str', '')
                                    eta = progress_data.get('eta_str', '')
                                    total_size = progress_data.get('total_size_str', '')
                                    status_text.text(f"⬇️ Downloading... {progress:.1%}")
                                    info_text.text(f"Size: {total_size} | Speed: {speed} | ETA: {eta}")
                                else:
                                    downloaded = progress_data.get('downloaded', 0)
                                    total = progress_data.get('total', 0)
                                    speed = progress_data.get('speed', 0)
                                    
                                    if total > 0:
                                        status_text.text(f"⬇️ Downloading... {progress:.1%}")
                                        if speed > 0:
                                            speed_mb = speed / (1024 * 1024)
                                            info_text.text(f"Downloaded: {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB | Speed: {speed_mb:.1f}MB/s")
                                        else:
                                            info_text.text(f"Downloaded: {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB")
                                    else:
                                        status_text.text("⬇️ Downloading...")
                                        
                            elif status == 'finished':
                                download_state['phase'] = 'validating'
                                progress_bar.progress(1.0)
                                status_text.text("✅ Download completed successfully!")
                                info_text.text("Validating file...")
                                
                        except Exception as e:
                            pass  # Ignore callback errors
                    
                    status_text.text("🚀 Starting download...")
                    info_text.text("Initializing...")
                    
                    try:
                        st.info(f"🚀 Starting download with resolution: '{res}'")
                        
                        # No cleanup needed - preserve existing files per user requirements
                        st.info("📥 Starting download (existing files will be preserved)...")
                        
                        downloaded_video = download_video_ytdlp(url, resolution=res, progress_callback=progress_callback)
                        
                        # Enhanced post-download validation
                        status_text.text("🔍 Validating downloaded video...")
                        info_text.text("Checking file integrity...")
                        
                        if not downloaded_video:
                            st.error("❌ Download failed: No file path returned from downloader")
                            st.info("This might indicate a network issue or the video is not accessible")
                            return
                        
                        if not os.path.exists(downloaded_video):
                            st.error(f"❌ Download failed: File not found at {downloaded_video}")
                            st.info("The download may have been interrupted or failed silently")
                            return
                        
                        # Validate the downloaded video
                        is_valid, validation_msg = validate_video_file(downloaded_video)
                        file_size_mb = os.path.getsize(downloaded_video) / (1024 * 1024)
                        
                        if is_valid:
                            # 新架构: 下载完成后注册到视频管理系统
                            video_id = video_mgr.register_video(downloaded_video)
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Download and validation completed!")
                            info_text.text(f"File: {os.path.basename(downloaded_video)} ({file_size_mb:.1f}MB)")
                            
                            st.success(f"🎉 Video downloaded successfully!")
                            st.info(f"📁 File: {os.path.basename(downloaded_video)} ({file_size_mb:.1f}MB)")
                            st.info(f"📝 Video registered with ID: {video_id}")
                            st.info(f"🔗 Source URL: {url}")
                        else:
                            st.warning(f"⚠️ Download completed but validation failed: {validation_msg}")
                            st.info(f"📊 File size: {file_size_mb:.1f}MB")
                            st.info("The video may still be usable, but please check the quality.")
                        
                        sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = str(e)
                        progress_bar.progress(0)
                        status_text.text(f"❌ Download failed")
                        info_text.text(f"Error: {error_msg}")
                        
                        # Enhanced error categorization
                        if "yt-dlp command failed" in error_msg:
                            st.error("❌ **Download Tool Error**: The download tool encountered an error.")
                            st.info("💡 **Possible solutions:**")
                            st.write("- Check your internet connection")
                            st.write("- Verify the YouTube URL is correct and accessible")
                            st.write("- Try a different resolution")
                            st.write("- Check if the video is region-restricted")
                        elif "proxy" in error_msg.lower() or "network" in error_msg.lower():
                            st.error("❌ **Network Error**: Connection problem detected.")
                            st.info("💡 **Possible solutions:**")
                            st.write("- Check your internet connection")
                            st.write("- Verify proxy settings if you're using one")
                            st.write("- Try again in a few minutes")
                        elif "403" in error_msg or "401" in error_msg:
                            st.error("❌ **Access Denied**: The video may be private or restricted.")
                            st.info("💡 **Possible solutions:**")
                            st.write("- Check if the video is public")
                            st.write("- Try logging into YouTube in your browser")
                            st.write("- Check if cookies are properly configured")
                        else:
                            st.error(f"❌ **Unknown Error**: {error_msg}")
                            st.info("💡 **Try these steps:**")
                            st.write("- Clean up partial downloads using the button above")
                            st.write("- Try a different video URL")
                            st.write("- Check the system logs for more details")
                            
                        # Option to clean up partial downloads
                        if st.button("🧹 Clean up and retry", key="cleanup_retry"):
                            cleanup_partial_downloads(paths['input'])
                            st.success("Partial downloads cleaned up. Please try again.")
                            st.rerun()
                            
                else:
                    st.warning("Please enter a YouTube URL")

            uploaded_file = st.file_uploader(t("Or upload video"), type=load_key("allowed_video_formats") + load_key("allowed_audio_formats"))
            if uploaded_file:
                # 新架构: 基于ID的文件管理 + 安全文件名处理
                original_name = uploaded_file.name
                _, ext = os.path.splitext(original_name)
                
                # Generate safe filename using UUID to prevent path traversal and conflicts
                safe_filename = generate_safe_filename(original_name, ext.lower())
                
                # Show upload progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📤 Uploading file...")
                progress_bar.progress(0.3)
                
                # 创建临时文件 - 使用安全的UUID文件名
                temp_path = os.path.join(paths['input'], f"upload_{safe_filename}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar.progress(0.6)
                status_text.text("🔍 Validating uploaded file...")
                
                try:
                    # Validate uploaded file
                    is_valid, validation_msg = validate_video_file(temp_path)
                    file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                    
                    if is_valid:
                        progress_bar.progress(0.8)
                        status_text.text("📝 Registering video...")
                        
                        # 注册到视频管理系统
                        video_id = video_mgr.register_video(temp_path)
                        
                        progress_bar.progress(0.9)
                        
                        # 如果是音频文件，转换为视频
                        if ext.lower() in load_key("allowed_audio_formats"):
                            status_text.text("🎵➡️🎬 Converting audio to video...")
                            video_paths = video_mgr.get_video_paths(video_id)
                            convert_audio_to_video(video_paths['input'])
                            
                        progress_bar.progress(1.0)
                        status_text.text("✅ Upload completed successfully!")
                        
                        st.success(f"🎉 Video uploaded successfully! ID: {video_id}")
                        st.info(f"📊 Original: {original_name} ({file_size_mb:.1f}MB)")
                        st.info(f"🔒 Stored as: {safe_filename}")
                        
                    else:
                        st.error(f"❌ **File Validation Failed**: {validation_msg}")
                        st.info(f"📊 File size: {file_size_mb:.1f}MB")
                        st.info("💡 Please check if the file is a valid video format and not corrupted.")
                        
                    # 删除临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.error(f"❌ Video upload failed: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                st.rerun()
            else:
                return False

        except Exception as e:
            # Enhanced error handling for video file detection
            error_msg = str(e)
            if "No video files found" in error_msg:
                st.info("📥 No video found. Please download or upload a video.")
            elif "Multiple video files found" in error_msg:
                st.warning("⚠️ Multiple video files detected. The system will automatically select the best one.")
            elif "No valid video files found" in error_msg:
                st.error("❌ Downloaded files appear to be corrupted or incomplete. Please try downloading again.")
                # Show cleanup option
                if st.button("🧹 Clean up partial downloads", key="cleanup_partial"):
                    cleanup_partial_downloads(paths['input'])
                    st.success("Partial downloads cleaned up. Please try downloading again.")
                    st.rerun()
            else:
                st.error(f"❌ Error: {error_msg}")
            
            # Download section
            col1, col2 = st.columns([3, 1])
            with col1:
                url = st.text_input(t("Enter YouTube link:"))
            with col2:
                res_dict = {
                    "360p": "360",
                    "1080p": "1080",
                    "Best": "best"
                }
                target_res = load_key("youtube_resolution")
                res_options = list(res_dict.keys())
                
                # Find the correct default index based on config value
                default_idx = 0  # default to 360p
                for i, (display_name, res_value) in enumerate(res_dict.items()):
                    if res_value == str(target_res):
                        default_idx = i
                        break
                
                res_display = st.selectbox(t("Resolution"), options=res_options, index=default_idx)
                res = res_dict[res_display]
                
            if st.button(t("Download Video"), key="download_video_button"):
                if url and validate_url(url):
                    # Enhanced download with better progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    info_text = st.empty()
                    
                    # Create download state tracking
                    download_state = {
                        'phase': 'initializing',
                        'error': None,
                        'completed': False
                    }
                    
                    def progress_callback(progress_data):
                        try:
                            progress = progress_data.get('progress', 0)
                            status = progress_data.get('status', 'downloading')
                            
                            if status == 'downloading':
                                download_state['phase'] = 'downloading'
                                progress_bar.progress(min(progress, 0.99))  # Keep some margin for completion
                                
                                # Format status message
                                if 'speed_str' in progress_data and 'eta_str' in progress_data:
                                    speed = progress_data.get('speed_str', '')
                                    eta = progress_data.get('eta_str', '')
                                    total_size = progress_data.get('total_size_str', '')
                                    status_text.text(f"⬇️ Downloading... {progress:.1%}")
                                    info_text.text(f"Size: {total_size} | Speed: {speed} | ETA: {eta}")
                                else:
                                    downloaded = progress_data.get('downloaded', 0)
                                    total = progress_data.get('total', 0)
                                    speed = progress_data.get('speed', 0)
                                    
                                    if total > 0:
                                        status_text.text(f"⬇️ Downloading... {progress:.1%}")
                                        if speed > 0:
                                            speed_mb = speed / (1024 * 1024)
                                            info_text.text(f"Downloaded: {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB | Speed: {speed_mb:.1f}MB/s")
                                        else:
                                            info_text.text(f"Downloaded: {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB")
                                    else:
                                        status_text.text("⬇️ Downloading...")
                                        
                            elif status == 'finished':
                                download_state['phase'] = 'validating'
                                progress_bar.progress(1.0)
                                status_text.text("✅ Download completed successfully!")
                                info_text.text("Validating file...")
                                
                        except Exception as e:
                            pass  # Ignore callback errors
                    
                    status_text.text("🚀 Starting download...")
                    info_text.text("Initializing...")
                    
                    try:
                        st.info(f"🚀 Starting download with resolution: '{res}'")
                        
                        # No cleanup needed - preserve existing files per user requirements
                        st.info("📥 Starting download (existing files will be preserved)...")
                        
                        downloaded_video = download_video_ytdlp(url, resolution=res, progress_callback=progress_callback)
                        
                        # Enhanced post-download validation
                        status_text.text("🔍 Validating downloaded video...")
                        info_text.text("Checking file integrity...")
                        
                        if not downloaded_video:
                            st.error("❌ Download failed: No file path returned from downloader")
                            st.info("This might indicate a network issue or the video is not accessible")
                            return
                        
                        if not os.path.exists(downloaded_video):
                            st.error(f"❌ Download failed: File not found at {downloaded_video}")
                            st.info("The download may have been interrupted or failed silently")
                            return
                        
                        # Validate the downloaded video
                        is_valid, validation_msg = validate_video_file(downloaded_video)
                        file_size_mb = os.path.getsize(downloaded_video) / (1024 * 1024)
                        
                        if is_valid:
                            # 新架构: 下载完成后注册到视频管理系统
                            video_id = video_mgr.register_video(downloaded_video)
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Download and validation completed!")
                            info_text.text(f"File: {os.path.basename(downloaded_video)} ({file_size_mb:.1f}MB)")
                            
                            st.success(f"🎉 Video downloaded successfully!")
                            st.info(f"📁 File: {os.path.basename(downloaded_video)} ({file_size_mb:.1f}MB)")
                            st.info(f"📝 Video registered with ID: {video_id}")
                            st.info(f"🔗 Source URL: {url}")
                        else:
                            st.warning(f"⚠️ Download completed but validation failed: {validation_msg}")
                            st.info(f"📊 File size: {file_size_mb:.1f}MB")
                            st.info("The video may still be usable, but please check the quality.")
                        
                        sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = str(e)
                        progress_bar.progress(0)
                        status_text.text(f"❌ Download failed")
                        info_text.text(f"Error: {error_msg}")
                        
                        # Enhanced error categorization
                        if "yt-dlp command failed" in error_msg:
                            st.error("❌ **Download Tool Error**: The download tool encountered an error.")
                            st.info("💡 **Possible solutions:**")
                            st.write("- Check your internet connection")
                            st.write("- Verify the YouTube URL is correct and accessible")
                            st.write("- Try a different resolution")
                            st.write("- Check if the video is region-restricted")
                        elif "proxy" in error_msg.lower() or "network" in error_msg.lower():
                            st.error("❌ **Network Error**: Connection problem detected.")
                            st.info("💡 **Possible solutions:**")
                            st.write("- Check your internet connection")
                            st.write("- Verify proxy settings if you're using one")
                            st.write("- Try again in a few minutes")
                        elif "403" in error_msg or "401" in error_msg:
                            st.error("❌ **Access Denied**: The video may be private or restricted.")
                            st.info("💡 **Possible solutions:**")
                            st.write("- Check if the video is public")
                            st.write("- Try logging into YouTube in your browser")
                            st.write("- Check if cookies are properly configured")
                        else:
                            st.error(f"❌ **Unknown Error**: {error_msg}")
                            st.info("💡 **Try these steps:**")
                            st.write("- Clean up partial downloads using the button above")
                            st.write("- Try a different video URL")
                            st.write("- Check the system logs for more details")
                            
                        # Option to clean up partial downloads
                        if st.button("🧹 Clean up and retry", key="cleanup_retry"):
                            cleanup_partial_downloads(paths['input'])
                            st.success("Partial downloads cleaned up. Please try again.")
                            st.rerun()
                            
                elif url and not validate_url(url):
                    st.error("❌ Invalid URL format. Please enter a valid YouTube URL.")
                else:
                    st.warning("Please enter a YouTube URL")

            uploaded_file = st.file_uploader(t("Or upload video"), type=load_key("allowed_video_formats") + load_key("allowed_audio_formats"))
            if uploaded_file:
                # 新架构: 基于ID的文件管理 + 安全文件名处理
                original_name = uploaded_file.name
                _, ext = os.path.splitext(original_name)
                
                # Generate safe filename using UUID to prevent path traversal and conflicts
                safe_filename = generate_safe_filename(original_name, ext.lower())
                
                # Show upload progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📤 Uploading file...")
                progress_bar.progress(0.3)
                
                # 创建临时文件 - 使用安全的UUID文件名
                temp_path = os.path.join(paths['input'], f"upload_{safe_filename}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar.progress(0.6)
                status_text.text("🔍 Validating uploaded file...")
                
                try:
                    # Validate uploaded file
                    is_valid, validation_msg = validate_video_file(temp_path)
                    file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                    
                    if is_valid:
                        progress_bar.progress(0.8)
                        status_text.text("📝 Registering video...")
                        
                        # 注册到视频管理系统
                        video_id = video_mgr.register_video(temp_path)
                        
                        progress_bar.progress(0.9)
                        
                        # 如果是音频文件，转换为视频
                        if ext.lower() in load_key("allowed_audio_formats"):
                            status_text.text("🎵➡️🎬 Converting audio to video...")
                            video_paths = video_mgr.get_video_paths(video_id)
                            convert_audio_to_video(video_paths['input'])
                            
                        progress_bar.progress(1.0)
                        status_text.text("✅ Upload completed successfully!")
                        
                        st.success(f"🎉 Video uploaded successfully! ID: {video_id}")
                        st.info(f"📊 Original: {original_name} ({file_size_mb:.1f}MB)")
                        st.info(f"🔒 Stored as: {safe_filename}")
                        
                    else:
                        st.error(f"❌ **File Validation Failed**: {validation_msg}")
                        st.info(f"📊 File size: {file_size_mb:.1f}MB")
                        st.info("💡 Please check if the file is a valid video format and not corrupted.")
                        
                    # 删除临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.error(f"❌ Video upload failed: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                st.rerun()
            else:
                return False

def convert_audio_to_video(audio_file: str) -> str:
    paths = get_storage_paths()
    output_video = os.path.join(paths['input'], 'black_screen.mp4')
    if not os.path.exists(output_video):
        print(f"🎵➡️🎬 Converting audio to video with FFmpeg ......")
        ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=640x360', '-i', audio_file, '-shortest', '-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p', output_video]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"🎵➡️🎬 Converted <{audio_file}> to <{output_video}> with FFmpeg\n")
        # delete audio file
        os.remove(audio_file)
    return output_video
