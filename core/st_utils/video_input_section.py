"""
VideoLingo 视频输入部分 - 重构版本
支持下载和上传两种方式的标签页切换
解耦合视频源与处理功能
"""

import os
import re
import streamlit as st
from core._1_ytdlp import download_video_ytdlp, find_video_files
from core.utils.config_utils import get_storage_paths, ensure_storage_dirs, load_key
from core.utils.video_manager import get_video_manager
from translations.translations import translate as t

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
        if 'download_video_id' in st.session_state:
            video_id = st.session_state.download_video_id
            video_paths = video_mgr.get_video_paths(video_id)
            if os.path.exists(video_paths['input']):
                st.video(video_paths['input'])
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
                    if st.button("🗑️ 删除当前视频", key="delete_download_video", use_container_width=True):
                        try:
                            # 完全删除当前视频和相关处理文件
                            video_mgr.safe_overwrite_temp_files(video_id)
                            video_mgr.safe_overwrite_output_files(video_id)
                            if os.path.exists(video_paths['input']):
                                os.remove(video_paths['input'])
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
    with col2:
        res_dict = {
            "360p": "360",
            "720p": "720", 
            "1080p": "1080",
            "1440p": "1440",
            "2160p (4K)": "2160",
            "Best Quality": "best"
        }
        target_res = load_key("youtube_resolution")
        res_options = list(res_dict.keys())
        
        default_idx = 0
        for i, (display_name, res_value) in enumerate(res_dict.items()):
            if res_value == str(target_res):
                default_idx = i
                break
        
        res_display = st.selectbox(t("Resolution"), options=res_options, index=default_idx, key="download_resolution")
        res = res_dict[res_display]
    
    # 并发保护：检查是否正在下载
    download_in_progress = st.session_state.get('download_in_progress', False)
    
    if download_in_progress:
        st.warning("⚠️ 下载正在进行中，请等待完成...")
        if st.button("🛑 取消下载", key="cancel_download"):
            st.session_state.download_in_progress = False
            st.rerun()
    elif st.button(t("Download Video"), key="download_video_button", use_container_width=True):
        if url:
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
                    progress = progress_data.get('progress', 0)
                    status = progress_data.get('status', 'downloading')
                    
                    if status == 'downloading':
                        progress_bar.progress(min(progress, 0.99))
                        status_text.text(f"⬬ Downloading... {progress:.1%}")
                        
                        # 更详细的进度信息
                        if 'speed_str' in progress_data and 'eta_str' in progress_data:
                            speed = progress_data.get('speed_str', '')
                            eta = progress_data.get('eta_str', '')
                            total_size = progress_data.get('total_size_str', '')
                            info_text.text(f"Size: {total_size} | Speed: {speed} | ETA: {eta}")
                    elif status == 'finished':
                        progress_bar.progress(1.0)
                        status_text.text("✅ Download completed successfully!")
                        info_text.text("")
                except Exception as callback_error:
                    # 记录回调错误但不中断下载过程
                    print(f"Progress callback error: {callback_error}")
            
            status_text.text("🚀 Starting download...")
            info_text.text("Initializing...")
            
            try:
                st.info(f"🚀 Starting download with resolution: '{res}'")
                
                # Enhanced download with title preservation and concurrent protection
                try:
                    downloaded_video = download_video_ytdlp(url, resolution=res, progress_callback=progress_callback)
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
                    # 清除下载状态
                    st.session_state.download_in_progress = False
                    raise download_error
                
            except Exception as e:
                # 确保清除下载状态
                st.session_state.download_in_progress = False
                
                progress_bar.progress(0)
                status_text.text(f"❌ Download failed")
                info_text.text(f"Error: {str(e)}")
                
                # Enhanced error handling with context-specific messages
                error_msg = str(e)
                st.error(f"❌ Download failed: {error_msg}")
                
                # Provide context-specific troubleshooting
                if "could not locate" in error_msg.lower():
                    st.info("💡 The download may have completed but the file couldn't be found. Check your input directory.")
                elif "validation failed" in error_msg.lower():
                    st.info("💡 The downloaded file may be corrupted. Try downloading again or check your internet connection.")
                elif "network" in error_msg.lower() or "timeout" in error_msg.lower():
                    st.info("💡 Network issue detected. Check your internet connection and try again.")
                else:
                    st.info("💡 Try checking the video URL and your internet connection.")
        else:
            st.warning("Please enter a YouTube URL")
    
    return None

def handle_upload_tab():
    """处理上传视频标签页"""
    video_mgr = get_video_manager()
    paths = get_storage_paths()
    
    # 检查是否有通过上传获得的视频
    try:
        if 'upload_video_id' in st.session_state:
            video_id = st.session_state.upload_video_id
            video_paths = video_mgr.get_video_paths(video_id)
            if os.path.exists(video_paths['input']):
                st.video(video_paths['input'])
                st.success(f"Video ID: {video_id}")
                
                # 当前视频指示器
                st.info(f"🎬 **当前活跃视频**: {os.path.basename(video_paths['input'])}")
                
                # 切换视频按钮组
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 切换到新视频", key="switch_upload_video", use_container_width=True):
                        # 不删除文件，只清除session状态以允许新的上传
                        del st.session_state.upload_video_id
                        st.success("已切换到新视频模式，可以上传新视频")
                        st.rerun()
                with col2:
                    if st.button("🗑️ 删除当前视频", key="delete_upload_video", use_container_width=True):
                        try:
                            # 完全删除当前视频和相关处理文件
                            video_mgr.safe_overwrite_temp_files(video_id)
                            video_mgr.safe_overwrite_output_files(video_id)
                            if os.path.exists(video_paths['input']):
                                os.remove(video_paths['input'])
                            del st.session_state.upload_video_id
                            st.success("视频及相关文件已删除")
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {str(e)}")
                
                return video_id
    except Exception as e:
        st.error(f"Error loading upload video: {str(e)}")
        return None
    
    # 上传界面 - 添加并发保护
    upload_in_progress = st.session_state.get('upload_in_progress', False)
    
    if upload_in_progress:
        st.warning("⚠️ 上传正在进行中，请等待完成...")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader(
            t("Upload video file"), 
            type=load_key("allowed_video_formats") + load_key("allowed_audio_formats"),
            key="upload_file_uploader"
        )
    
    if uploaded_file:
        # 设置上传状态保护
        st.session_state.upload_in_progress = True
        
        try:
            # 处理上传的文件
            raw_name = uploaded_file.name.replace(' ', '_')
            name, ext = os.path.splitext(raw_name)
            clean_name = re.sub(r'[^\w\-_\.]', '', name) + ext.lower()
            
            # 创建临时文件
            temp_path = os.path.join(paths['input'], f"temp_{clean_name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 注册到视频管理系统
            video_id = video_mgr.register_video(temp_path)
            st.session_state.upload_video_id = video_id
            
            # 保持原始文件名
            original_name = os.path.splitext(uploaded_file.name)[0]
            st.success(f"✅ Video uploaded and registered! ID: {video_id}")
            st.info(f"🎥 标题: {original_name}")
            st.info(f"📁 文件: {clean_name}")
            
            # 如果是音频文件，转换为视频
            if ext.lower() in load_key("allowed_audio_formats"):
                from core.st_utils.download_video_section import convert_audio_to_video
                video_paths = video_mgr.get_video_paths(video_id)
                convert_audio_to_video(video_paths['input'])
            
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # 清除上传状态
            st.session_state.upload_in_progress = False
            st.rerun()
            
        except Exception as e:
            # 确保清除上传状态
            st.session_state.upload_in_progress = False
            st.error(f"Video registration failed: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
    
    return None

def get_current_video_id():
    """获取当前活跃的视频ID"""
    # 优先返回当前标签页的视频
    if 'download_video_id' in st.session_state:
        return st.session_state.download_video_id
    elif 'upload_video_id' in st.session_state:
        return st.session_state.upload_video_id
    else:
        return None

def clear_video_session():
    """清空视频会话状态"""
    if 'download_video_id' in st.session_state:
        del st.session_state.download_video_id
    if 'upload_video_id' in st.session_state:
        del st.session_state.upload_video_id