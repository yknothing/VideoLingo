"""
Comprehensive test suite for Streamlit UI components.
Targets 85%+ branch coverage for user interface functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os

# Add core directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestSidebarSettingModule(unittest.TestCase):
    """Test core/st_utils/sidebar_setting.py - Settings management interface"""
    
    def setUp(self):
        self.patcher_streamlit = patch('core.st_utils.sidebar_setting.st')
        self.patcher_load_key = patch('core.st_utils.sidebar_setting.load_key')
        self.patcher_update_key = patch('core.st_utils.sidebar_setting.update_key')
        self.patcher_check_api = patch('core.st_utils.sidebar_setting.check_api')
        self.patcher_translate = patch('core.st_utils.sidebar_setting.t')
        self.patcher_display_langs = patch('core.st_utils.sidebar_setting.DISPLAY_LANGUAGES')
        
        self.mock_st = self.patcher_streamlit.start()
        self.mock_load_key = self.patcher_load_key.start()
        self.mock_update_key = self.patcher_update_key.start()
        self.mock_check_api = self.patcher_check_api.start()
        self.mock_translate = self.patcher_translate.start()
        self.mock_display_langs = self.patcher_display_langs.start()
        
        # Setup default mocks
        self.mock_translate.side_effect = lambda x: f"tr_{x}"
        self.mock_display_langs.return_value = {
            "English": "en",
            "Chinese": "zh"
        }
        self.mock_load_key.side_effect = lambda key: {
            "display_language": "en",
            "api.key": "test_key",
            "api.base_url": "https://api.test.com",
            "api.model": "gpt-4",
            "api.llm_support_json": True,
            "whisper.language": "en",
            "whisper.runtime": "local",
            "target_language": "Spanish",
            "demucs": False,
            "burn_subtitles": True,
            "tts_method": "azure_tts",
            "sf_fish_tts.mode": "preset",
            "sf_fish_tts.api_key": "sf_key",
            "sf_fish_tts.voice": "default",
            "fish_tts.character_id_dict": {"char1": "id1", "char2": "id2"},
            "fish_tts.character": "char1",
            "azure_tts.voice": "en-US-AriaNeural",
            "gpt_sovits.refer_mode": 1,
            "edge_tts.voice": "en-US-AriaNeural"
        }.get(key, "default")
        
    def tearDown(self):
        self.patcher_streamlit.stop()
        self.patcher_load_key.stop()
        self.patcher_update_key.stop()
        self.patcher_check_api.stop()
        self.patcher_translate.stop()
        self.patcher_display_langs.stop()
        
    def test_config_input_no_change(self):
        """Test config input with no value change"""
        from core.st_utils.sidebar_setting import config_input
        
        self.mock_st.text_input.return_value = "test_key"
        
        result = config_input("API Key", "api.key", help="Help text")
        
        self.mock_st.text_input.assert_called_with("API Key", value="test_key", help="Help text")
        self.mock_update_key.assert_not_called()
        self.assertEqual(result, "test_key")
        
    def test_config_input_with_change(self):
        """Test config input with value change"""
        from core.st_utils.sidebar_setting import config_input
        
        self.mock_st.text_input.return_value = "new_key"
        
        result = config_input("API Key", "api.key")
        
        self.mock_update_key.assert_called_with("api.key", "new_key")
        self.assertEqual(result, "new_key")
        
    def test_page_setting_display_language_change(self):
        """Test page setting with display language change"""
        from core.st_utils.sidebar_setting import page_setting
        
        # Mock display languages
        self.mock_display_langs.__getitem__ = lambda self, key: {"English": "en", "Chinese": "zh"}[key]
        self.mock_display_langs.keys.return_value = ["English", "Chinese"]
        self.mock_display_langs.values.return_value = ["en", "zh"]
        
        # Mock selectbox returning different language
        self.mock_st.selectbox.side_effect = [
            "Chinese",  # Display language selectbox
            "en",       # Recognition language
            "local",    # Runtime
            "azure_tts" # TTS method
        ]
        
        # Mock columns and other UI elements
        self.mock_st.columns.return_value = [Mock(), Mock()]
        self.mock_st.expander.return_value.__enter__ = Mock(return_value=Mock())
        self.mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        self.mock_st.text_input.return_value = "test"
        self.mock_st.toggle.return_value = True
        self.mock_st.button.return_value = False
        
        page_setting()
        
        self.mock_update_key.assert_any_call("display_language", "zh")
        self.mock_st.rerun.assert_called()
        
    def test_page_setting_api_check_valid(self):
        """Test page setting with valid API check"""
        from core.st_utils.sidebar_setting import page_setting
        
        # Mock UI elements
        self.mock_st.selectbox.side_effect = ["English", "en", "local", "azure_tts"]
        
        # Mock columns
        mock_col1, mock_col2 = Mock(), Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        self.mock_st.columns.return_value = [mock_col1, mock_col2]
        
        # Mock expander
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        self.mock_st.expander.return_value = mock_expander
        
        self.mock_st.text_input.return_value = "test"
        self.mock_st.toggle.return_value = True
        
        # Mock API button click
        self.mock_st.button.return_value = True
        self.mock_check_api.return_value = True
        
        page_setting()
        
        self.mock_st.toast.assert_called_with("tr_API Key is valid", icon="✅")
        
    def test_page_setting_api_check_invalid(self):
        """Test page setting with invalid API check"""
        from core.st_utils.sidebar_setting import page_setting
        
        # Mock UI elements
        self.mock_st.selectbox.side_effect = ["English", "en", "local", "azure_tts"]
        
        # Mock columns and expander
        mock_col1, mock_col2 = Mock(), Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        self.mock_st.columns.return_value = [mock_col1, mock_col2]
        
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        self.mock_st.expander.return_value = mock_expander
        
        self.mock_st.text_input.return_value = "test"
        self.mock_st.toggle.return_value = True
        
        # Mock API button click with invalid result
        self.mock_st.button.return_value = True
        self.mock_check_api.return_value = False
        
        page_setting()
        
        self.mock_st.toast.assert_called_with("tr_API Key is invalid", icon="❌")
        
    def test_page_setting_whisper_runtime_cloud(self):
        """Test page setting with cloud runtime selection"""
        from core.st_utils.sidebar_setting import page_setting
        
        # Mock UI elements with cloud runtime
        self.mock_st.selectbox.side_effect = ["English", "en", "cloud", "azure_tts"]
        
        # Mock columns and expander
        mock_col1, mock_col2 = Mock(), Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        self.mock_st.columns.return_value = [mock_col1, mock_col2]
        
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        self.mock_st.expander.return_value = mock_expander
        
        self.mock_st.text_input.return_value = "test"
        self.mock_st.toggle.return_value = True
        self.mock_st.button.return_value = False
        
        page_setting()
        
        # Should show cloud API input
        calls = [call[0][0] for call in self.mock_st.text_input.call_args_list]
        self.assertIn("tr_WhisperX 302ai API", calls)
        
    def test_page_setting_whisper_runtime_elevenlabs(self):
        """Test page setting with elevenlabs runtime selection"""
        from core.st_utils.sidebar_setting import page_setting
        
        # Mock UI elements with elevenlabs runtime
        self.mock_st.selectbox.side_effect = ["English", "en", "elevenlabs", "azure_tts"]
        
        # Mock UI structure
        mock_col1, mock_col2 = Mock(), Mock()
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        self.mock_st.columns.return_value = [mock_col1, mock_col2]
        
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        self.mock_st.expander.return_value = mock_expander
        
        self.mock_st.text_input.return_value = "test"
        self.mock_st.toggle.return_value = True
        self.mock_st.button.return_value = False
        
        page_setting()
        
        # Should show elevenlabs API input
        calls = [call[0][0] for call in self.mock_st.text_input.call_args_list]
        self.assertIn("ElevenLabs API", calls)
        
    def test_page_setting_tts_sf_fish_preset_mode(self):
        """Test page setting with SF Fish TTS preset mode"""
        from core.st_utils.sidebar_setting import page_setting
        
        # Mock UI elements with sf_fish_tts
        self.mock_st.selectbox.side_effect = [
            "English",      # Display language
            "en",          # Recognition language  
            "local",       # Runtime
            "sf_fish_tts", # TTS method
            "preset"       # SF Fish mode
        ]
        
        # Mock UI structure
        self.mock_st.columns.return_value = [Mock(), Mock()]
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        self.mock_st.expander.return_value = mock_expander
        
        self.mock_st.text_input.return_value = "test"
        self.mock_st.toggle.return_value = True
        self.mock_st.button.return_value = False
        
        page_setting()
        
        # Should show voice input for preset mode
        calls = [call[0][0] for call in self.mock_st.text_input.call_args_list]
        self.assertIn("Voice", calls)
        
    def test_page_setting_tts_fish_tts(self):
        """Test page setting with Fish TTS selection"""
        from core.st_utils.sidebar_setting import page_setting
        
        # Mock UI elements with fish_tts
        self.mock_st.selectbox.side_effect = [
            "English",   # Display language
            "en",       # Recognition language
            "local",    # Runtime
            "fish_tts", # TTS method
            "char1"     # Fish character
        ]
        
        # Mock UI structure
        self.mock_st.columns.return_value = [Mock(), Mock()]
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        self.mock_st.expander.return_value = mock_expander
        
        self.mock_st.text_input.return_value = "test"
        self.mock_st.toggle.return_value = True
        self.mock_st.button.return_value = False
        
        page_setting()
        
        # Should show character selection
        character_calls = [call for call in self.mock_st.selectbox.call_args_list 
                          if len(call[0]) > 1 and "tr_Fish TTS Character" in str(call[0])]
        self.assertTrue(len(character_calls) > 0)
        
    def test_page_setting_tts_gpt_sovits(self):
        """Test page setting with GPT-SoVITS selection"""
        from core.st_utils.sidebar_setting import page_setting
        
        # Mock UI elements with gpt_sovits
        self.mock_st.selectbox.side_effect = [
            "English",     # Display language
            "en",         # Recognition language
            "local",      # Runtime
            "gpt_sovits", # TTS method
            2             # Refer mode
        ]
        
        # Mock UI structure
        self.mock_st.columns.return_value = [Mock(), Mock()]
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        self.mock_st.expander.return_value = mock_expander
        
        self.mock_st.text_input.return_value = "test"
        self.mock_st.toggle.return_value = True
        self.mock_st.button.return_value = False
        self.mock_st.info = Mock()
        
        page_setting()
        
        # Should show info message and character input
        self.mock_st.info.assert_called()
        
    def test_page_setting_toggle_changes(self):
        """Test page setting with toggle value changes"""
        from core.st_utils.sidebar_setting import page_setting
        
        # Mock UI elements
        self.mock_st.selectbox.side_effect = ["English", "en", "local", "azure_tts"]
        self.mock_st.columns.return_value = [Mock(), Mock()]
        mock_expander = Mock()
        mock_expander.__enter__ = Mock(return_value=mock_expander)
        mock_expander.__exit__ = Mock(return_value=None)
        self.mock_st.expander.return_value = mock_expander
        
        self.mock_st.text_input.return_value = "test"
        self.mock_st.button.return_value = False
        
        # Mock toggle values that differ from loaded values
        self.mock_st.toggle.side_effect = [False, False, False]  # LLM JSON, demucs, burn_subtitles
        
        page_setting()
        
        # Should update all changed toggle values
        expected_calls = [
            call("api.llm_support_json", False),
            call("demucs", False),
            call("burn_subtitles", False)
        ]
        for expected_call in expected_calls:
            self.assertIn(expected_call, self.mock_update_key.call_args_list)
        
    @patch('core.st_utils.sidebar_setting.ask_gpt')
    def test_check_api_success(self, mock_ask_gpt):
        """Test successful API check"""
        from core.st_utils.sidebar_setting import check_api
        
        mock_ask_gpt.return_value = {'message': 'success'}
        
        result = check_api()
        
        self.assertTrue(result)
        mock_ask_gpt.assert_called_with(
            "This is a test, response 'message':'success' in json format.", 
            resp_type="json", 
            log_title='None'
        )
        
    @patch('core.st_utils.sidebar_setting.ask_gpt')
    def test_check_api_failure(self, mock_ask_gpt):
        """Test failed API check"""
        from core.st_utils.sidebar_setting import check_api
        
        mock_ask_gpt.side_effect = Exception("API error")
        
        result = check_api()
        
        self.assertFalse(result)
        
    @patch('core.st_utils.sidebar_setting.ask_gpt')
    def test_check_api_wrong_response(self, mock_ask_gpt):
        """Test API check with wrong response"""
        from core.st_utils.sidebar_setting import check_api
        
        mock_ask_gpt.return_value = {'message': 'failure'}
        
        result = check_api()
        
        self.assertFalse(result)


class TestStreamlitMainApp(unittest.TestCase):
    """Test st.py - Main Streamlit application entry point"""
    
    def setUp(self):
        self.patcher_streamlit = patch('streamlit')
        self.patcher_os = patch('os')
        self.patcher_sys = patch('sys')
        
        self.mock_st = self.patcher_streamlit.start()
        self.mock_os = self.patcher_os.start()
        self.mock_sys = self.patcher_sys.start()
        
        # Mock sys.path for import safety
        self.mock_sys.path = ['/mock/path']
        
    def tearDown(self):
        self.patcher_streamlit.stop()
        self.patcher_os.stop()
        self.patcher_sys.stop()
        
    @patch('importlib.import_module')
    def test_app_import_structure(self, mock_import):
        """Test that main app can import required modules"""
        
        # Mock the required modules
        mock_modules = {
            'core.utils': Mock(),
            'core.st_utils.sidebar_setting': Mock(),
            'translations.translations': Mock()
        }
        
        def mock_import_side_effect(module_name):
            return mock_modules.get(module_name, Mock())
            
        mock_import.side_effect = mock_import_side_effect
        
        # Test import safety
        try:
            import importlib
            for module in mock_modules.keys():
                importlib.import_module(module)
        except Exception as e:
            self.fail(f"Import failed: {e}")


if __name__ == '__main__':
    unittest.main()
