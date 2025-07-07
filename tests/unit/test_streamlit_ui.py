# Unit Tests for Streamlit UI Components
# Tests core/st_utils/ modules

import pytest
from unittest.mock import patch, Mock, MagicMock
import streamlit as st

@pytest.mark.ui
class TestVideoInputSection:
    """Test suite for video input UI components"""
    
    def test_render_video_input_section_tabs(self, mock_streamlit):
        """Test video input section tab rendering"""
        from core.st_utils.video_input_section import render_video_input_section
        
        with patch('streamlit.tabs') as mock_tabs, \
             patch('core.st_utils.video_input_section.handle_download_tab', return_value=None), \
             patch('core.st_utils.video_input_section.handle_upload_tab', return_value=None):
            
            mock_tabs.return_value = [Mock(), Mock()]  # Two tab contexts
            
            result = render_video_input_section()
            
            assert mock_tabs.called
            # Should create tabs for download and upload
            call_args = mock_tabs.call_args[0]
            assert len(call_args[0]) == 2  # Two tabs
    
    def test_handle_download_tab_with_existing_video(self, mock_streamlit, temp_config_dir):
        """Test download tab with existing video"""
        from core.st_utils.video_input_section import handle_download_tab
        
        mock_video_id = "test_video_123"
        
        with patch('streamlit.session_state', {'download_video_id': mock_video_id}), \
             patch('core.utils.video_manager.get_video_manager') as mock_video_mgr, \
             patch('os.path.exists', return_value=True), \
             patch('streamlit.video'), \
             patch('streamlit.success'), \
             patch('streamlit.button', return_value=False):
            
            # Mock video manager
            mock_mgr = Mock()
            mock_mgr.get_video_paths.return_value = {
                'input': str(temp_config_dir / 'test_video.mp4')
            }
            mock_video_mgr.return_value = mock_mgr
            
            result = handle_download_tab()
            
            assert result == mock_video_id
            assert mock_mgr.get_video_paths.called
    
    def test_handle_download_tab_new_download(self, mock_streamlit, temp_config_dir):
        """Test download tab with new video download"""
        from core.st_utils.video_input_section import handle_download_tab
        
        with patch('streamlit.session_state', {}), \
             patch('streamlit.text_input', return_value="https://youtube.com/watch?v=test"), \
             patch('streamlit.selectbox', return_value="1080p"), \
             patch('streamlit.button', return_value=True), \
             patch('streamlit.container'), \
             patch('streamlit.progress'), \
             patch('streamlit.empty'), \
             patch('core._1_ytdlp.download_video_ytdlp', return_value="/path/to/video.mp4"), \
             patch('core.utils.video_manager.get_video_manager') as mock_video_mgr, \
             patch('os.path.exists', return_value=True):
            
            mock_mgr = Mock()
            mock_mgr.register_video.return_value = "new_video_123"
            mock_video_mgr.return_value = mock_mgr
            
            result = handle_download_tab()
            
            # Should trigger download process
            assert mock_mgr.register_video.called
    
    def test_handle_upload_tab_with_file(self, mock_streamlit, temp_config_dir):
        """Test upload tab with file upload"""
        from core.st_utils.video_input_section import handle_upload_tab
        
        # Mock uploaded file
        mock_file = Mock()
        mock_file.name = "test_video.mp4"
        mock_file.getbuffer.return_value = b"fake_video_content" * 1000
        
        with patch('streamlit.session_state', {}), \
             patch('streamlit.file_uploader', return_value=mock_file), \
             patch('core.utils.video_manager.get_video_manager') as mock_video_mgr, \
             patch('core.utils.config_utils.get_storage_paths', return_value={'input': str(temp_config_dir)}), \
             patch('core.utils.config_utils.load_key', return_value=['mp4', 'avi']), \
             patch('streamlit.success'), \
             patch('streamlit.rerun'):
            
            mock_mgr = Mock()
            mock_mgr.register_video.return_value = "uploaded_video_123"
            mock_video_mgr.return_value = mock_mgr
            
            result = handle_upload_tab()
            
            assert mock_mgr.register_video.called
    
    def test_handle_upload_tab_no_file(self, mock_streamlit):
        """Test upload tab without file"""
        from core.st_utils.video_input_section import handle_upload_tab
        
        with patch('streamlit.session_state', {}), \
             patch('streamlit.file_uploader', return_value=None):
            
            result = handle_upload_tab()
            
            assert result is None


@pytest.mark.ui
class TestSidebarSettings:
    """Test suite for sidebar settings UI"""
    
    def test_page_setting_basic_rendering(self, mock_streamlit):
        """Test basic page setting rendering"""
        from core.st_utils.sidebar_setting import page_setting
        
        with patch('streamlit.sidebar.header'), \
             patch('streamlit.sidebar.selectbox', return_value="English"), \
             patch('streamlit.sidebar.checkbox', return_value=False), \
             patch('streamlit.sidebar.slider', return_value=1.0), \
             patch('core.utils.config_utils.load_key', return_value="default_value"), \
             patch('core.utils.config_utils.save_key'):
            
            # Should not raise any exceptions
            page_setting()
    
    def test_page_setting_language_selection(self, mock_streamlit):
        """Test language selection in settings"""
        from core.st_utils.sidebar_setting import page_setting
        
        with patch('streamlit.sidebar.selectbox') as mock_selectbox, \
             patch('core.utils.config_utils.load_key', return_value="English"), \
             patch('core.utils.config_utils.save_key') as mock_save_key, \
             patch('streamlit.sidebar.header'), \
             patch('streamlit.sidebar.checkbox', return_value=False), \
             patch('streamlit.sidebar.slider', return_value=1.0):
            
            mock_selectbox.return_value = "Chinese (Simplified)"
            
            page_setting()
            
            # Should save language selection
            assert mock_selectbox.called
    
    def test_page_setting_api_configuration(self, mock_streamlit):
        """Test API configuration in settings"""
        from core.st_utils.sidebar_setting import page_setting
        
        with patch('streamlit.sidebar.text_input') as mock_text_input, \
             patch('streamlit.sidebar.selectbox', return_value="gpt-4"), \
             patch('core.utils.config_utils.load_key', return_value=""), \
             patch('core.utils.config_utils.save_key') as mock_save_key, \
             patch('streamlit.sidebar.header'), \
             patch('streamlit.sidebar.checkbox', return_value=False), \
             patch('streamlit.sidebar.slider', return_value=1.0):
            
            mock_text_input.return_value = "new-api-key"
            
            page_setting()
            
            # Should handle API key input
            assert mock_text_input.called
    
    def test_page_setting_advanced_options(self, mock_streamlit):
        """Test advanced options in settings"""
        from core.st_utils.sidebar_setting import page_setting
        
        with patch('streamlit.sidebar.expander') as mock_expander, \
             patch('streamlit.sidebar.checkbox', return_value=True), \
             patch('streamlit.sidebar.slider', return_value=0.7), \
             patch('core.utils.config_utils.load_key', return_value=1.0), \
             patch('core.utils.config_utils.save_key'), \
             patch('streamlit.sidebar.header'), \
             patch('streamlit.sidebar.selectbox', return_value="English"):
            
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock()
            
            page_setting()
            
            # Should create advanced options expander
            assert mock_expander.called


@pytest.mark.ui
class TestMainStreamlitApp:
    """Test suite for main Streamlit application"""
    
    def test_main_app_initialization(self, mock_streamlit):
        """Test main app initialization"""
        from st import main
        
        with patch('streamlit.columns', return_value=[Mock(), Mock()]), \
             patch('streamlit.image'), \
             patch('streamlit.markdown'), \
             patch('core.st_utils.sidebar_setting.page_setting'), \
             patch('core.st_utils.video_input_section.render_video_input_section', return_value=None), \
             patch('st.text_processing_section'), \
             patch('st.audio_processing_section'):
            
            # Should not raise any exceptions
            main()
    
    def test_text_processing_section_no_video(self, mock_streamlit):
        """Test text processing section without video"""
        from st import text_processing_section
        
        with patch('streamlit.header'), \
             patch('streamlit.container'), \
             patch('streamlit.warning'):
            
            result = text_processing_section(current_video_id=None)
            
            assert result is False
    
    def test_text_processing_section_with_video(self, mock_streamlit, temp_config_dir):
        """Test text processing section with video"""
        from st import text_processing_section
        
        mock_video_id = "test_video_123"
        
        with patch('streamlit.header'), \
             patch('streamlit.container'), \
             patch('streamlit.markdown'), \
             patch('streamlit.button', return_value=False), \
             patch('core.utils.video_manager.get_video_manager') as mock_video_mgr, \
             patch('os.path.exists', return_value=False):
            
            mock_mgr = Mock()
            mock_mgr.get_output_file.return_value = str(temp_config_dir / 'output_sub.mp4')
            mock_video_mgr.return_value = mock_mgr
            
            result = text_processing_section(current_video_id=mock_video_id)
            
            # Should show processing button when no output exists
            assert mock_mgr.get_output_file.called
    
    def test_audio_processing_section_prerequisites(self, mock_streamlit, temp_config_dir):
        """Test audio processing section prerequisites"""
        from st import audio_processing_section
        
        mock_video_id = "test_video_123"
        
        with patch('streamlit.header'), \
             patch('streamlit.container'), \
             patch('streamlit.warning'), \
             patch('core.utils.video_manager.get_video_manager') as mock_video_mgr, \
             patch('os.path.exists', return_value=False):
            
            mock_mgr = Mock()
            mock_mgr.get_output_file.return_value = str(temp_config_dir / 'output_sub.mp4')
            mock_video_mgr.return_value = mock_mgr
            
            result = audio_processing_section(current_video_id=mock_video_id)
            
            # Should require subtitles first
            assert result is False
    
    def test_get_current_video_id_from_session(self, mock_streamlit):
        """Test getting current video ID from session state"""
        from st import get_current_video_id
        
        test_cases = [
            ({'download_video_id': 'download_123'}, 'download_123'),
            ({'upload_video_id': 'upload_456'}, 'upload_456'),
            ({'download_video_id': 'download_123', 'upload_video_id': 'upload_456'}, 'download_123'),  # Download has priority
            ({}, None)
        ]
        
        for session_state, expected in test_cases:
            with patch('streamlit.session_state', session_state):
                result = get_current_video_id()
                assert result == expected


@pytest.mark.ui
class TestUIStateManagement:
    """Test suite for UI state management"""
    
    def test_session_state_persistence(self, mock_streamlit):
        """Test session state persistence across UI interactions"""
        
        # Mock session state
        session_state = {}
        
        with patch('streamlit.session_state', session_state):
            from core.st_utils.video_input_section import handle_download_tab
            
            # Simulate setting a download video ID
            session_state['download_video_id'] = 'test_video_123'
            
            with patch('core.utils.video_manager.get_video_manager') as mock_video_mgr, \
                 patch('os.path.exists', return_value=True), \
                 patch('streamlit.video'), \
                 patch('streamlit.success'), \
                 patch('streamlit.button', return_value=False):
                
                mock_mgr = Mock()
                mock_mgr.get_video_paths.return_value = {'input': '/path/to/video.mp4'}
                mock_video_mgr.return_value = mock_mgr
                
                result = handle_download_tab()
                
                # Should maintain video ID in session
                assert result == 'test_video_123'
                assert session_state['download_video_id'] == 'test_video_123'
    
    def test_ui_error_handling(self, mock_streamlit):
        """Test UI error handling"""
        from core.st_utils.video_input_section import handle_download_tab
        
        with patch('streamlit.session_state', {}), \
             patch('streamlit.text_input', return_value="https://youtube.com/watch?v=test"), \
             patch('streamlit.selectbox', return_value="1080p"), \
             patch('streamlit.button', return_value=True), \
             patch('streamlit.container'), \
             patch('streamlit.progress'), \
             patch('streamlit.empty'), \
             patch('streamlit.error') as mock_error, \
             patch('core._1_ytdlp.download_video_ytdlp', side_effect=Exception("Download failed")):
            
            result = handle_download_tab()
            
            # Should handle errors gracefully
            assert mock_error.called
            assert result is None
    
    def test_ui_progress_tracking(self, mock_streamlit):
        """Test UI progress tracking during operations"""
        from core.st_utils.video_input_section import handle_download_tab
        
        progress_values = []
        
        def mock_progress_update(value):
            progress_values.append(value)
        
        with patch('streamlit.session_state', {}), \
             patch('streamlit.text_input', return_value="https://youtube.com/watch?v=test"), \
             patch('streamlit.selectbox', return_value="1080p"), \
             patch('streamlit.button', return_value=True), \
             patch('streamlit.container'), \
             patch('streamlit.progress') as mock_progress, \
             patch('streamlit.empty'), \
             patch('core._1_ytdlp.download_video_ytdlp', return_value="/path/to/video.mp4"), \
             patch('core.utils.video_manager.get_video_manager') as mock_video_mgr:
            
            mock_progress_bar = Mock()
            mock_progress_bar.progress = mock_progress_update
            mock_progress.return_value = mock_progress_bar
            
            mock_mgr = Mock()
            mock_mgr.register_video.return_value = "new_video_123"
            mock_video_mgr.return_value = mock_mgr
            
            handle_download_tab()
            
            # Should update progress during download
            assert mock_progress.called
    
    def test_ui_responsive_design(self, mock_streamlit):
        """Test UI responsive design elements"""
        from core.st_utils.video_input_section import render_video_input_section
        
        with patch('streamlit.tabs') as mock_tabs, \
             patch('streamlit.columns') as mock_columns, \
             patch('core.st_utils.video_input_section.handle_download_tab', return_value=None), \
             patch('core.st_utils.video_input_section.handle_upload_tab', return_value=None):
            
            mock_tabs.return_value = [Mock(), Mock()]
            mock_columns.return_value = [Mock(), Mock()]
            
            render_video_input_section()
            
            # Should create responsive layout elements
            assert mock_tabs.called
    
    @pytest.mark.parametrize("video_id,expected_sections", [
        (None, 0),  # No sections available without video
        ("test_video_123", 2),  # Both text and audio sections with video
    ])
    def test_conditional_section_display(self, mock_streamlit, video_id, expected_sections):
        """Test conditional display of processing sections"""
        from st import text_processing_section, audio_processing_section
        
        sections_displayed = 0
        
        with patch('streamlit.header'), \
             patch('streamlit.container'), \
             patch('streamlit.markdown'), \
             patch('streamlit.warning'), \
             patch('core.utils.video_manager.get_video_manager') as mock_video_mgr:
            
            if video_id:
                mock_mgr = Mock()
                mock_mgr.get_output_file.return_value = "/path/to/output.mp4"
                mock_video_mgr.return_value = mock_mgr
                
                with patch('os.path.exists', return_value=False), \
                     patch('streamlit.button', return_value=False):
                    
                    # Text processing section
                    result1 = text_processing_section(current_video_id=video_id)
                    if not result1 and video_id:  # Shows processing button
                        sections_displayed += 1
                    
                    # Audio processing section
                    result2 = audio_processing_section(current_video_id=video_id)
                    if not result2 and video_id:  # Shows warning about prerequisites
                        sections_displayed += 1
            else:
                text_processing_section(current_video_id=video_id)
                audio_processing_section(current_video_id=video_id)
        
        # Should display appropriate number of sections
        assert sections_displayed <= expected_sections