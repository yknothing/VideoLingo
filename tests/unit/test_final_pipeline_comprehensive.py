"""
Comprehensive test suite for final core processing pipeline modules (_5 through _12).
Targets 85%+ branch coverage for subtitle processing, audio merging, and video output.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import pandas as pd
import pytest
import os
import sys
import tempfile
import subprocess
from io import StringIO

# Add core directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class TestSplitSubModule(unittest.TestCase):
    """Test core/_5_split_sub.py - Subtitle splitting and alignment"""
    
    def setUp(self):
        self.patcher_load_key = patch('core._5_split_sub.load_key')
        self.patcher_ask_gpt = patch('core._5_split_sub.ask_gpt')
        self.patcher_split_sentence = patch('core._5_split_sub.split_sentence')
        self.patcher_get_align_prompt = patch('core._5_split_sub.get_align_prompt')
        self.patcher_get_joiner = patch('core._5_split_sub.get_joiner')
        self.patcher_console = patch('core._5_split_sub.console')
        
        self.mock_load_key = self.patcher_load_key.start()
        self.mock_ask_gpt = self.patcher_ask_gpt.start()
        self.mock_split_sentence = self.patcher_split_sentence.start()
        self.mock_get_align_prompt = self.patcher_get_align_prompt.start()
        self.mock_get_joiner = self.patcher_get_joiner.start()
        self.mock_console = self.patcher_console.start()
        
        # Setup default mocks
        self.mock_load_key.side_effect = lambda key: {
            "whisper.language": "en",
            "whisper.detected_language": "en",
            "subtitle": {"max_length": 50, "target_multiplier": 1.0},
            "max_workers": 4
        }.get(key, "default")
        
        self.mock_get_joiner.return_value = Mock()
        self.mock_get_joiner.return_value.join = lambda x: " ".join(x)
        
    def tearDown(self):
        self.patcher_load_key.stop()
        self.patcher_ask_gpt.stop()
        self.patcher_split_sentence.stop()
        self.patcher_get_align_prompt.stop()
        self.patcher_get_joiner.stop()
        self.patcher_console.stop()
        
    def test_calc_len_chinese_chars(self):
        """Test character length calculation for Chinese"""
        from core._5_split_sub import calc_len
        
        # Chinese characters should have weight 1.75
        result = calc_len("你好")
        self.assertEqual(result, 3.5)  # 2 chars * 1.75
        
    def test_calc_len_english_chars(self):
        """Test character length calculation for English"""
        from core._5_split_sub import calc_len
        
        # English characters should have weight 1
        result = calc_len("hello")
        self.assertEqual(result, 5.0)  # 5 chars * 1
        
    def test_calc_len_empty_string(self):
        """Test character length calculation for empty string"""
        from core._5_split_sub import calc_len
        
        result = calc_len("")
        self.assertEqual(result, 0.0)
        
    def test_align_subs_success(self):
        """Test successful subtitle alignment"""
        from core._5_split_sub import align_subs
        
        # Mock GPT response
        align_response = {
            'align': [
                {'target_part_1': 'First part'},
                {'target_part_2': 'Second part'}
            ]
        }
        self.mock_ask_gpt.return_value = align_response
        self.mock_get_align_prompt.return_value = "mock prompt"
        
        src_sub = "Original subtitle"
        tr_sub = "Translated subtitle"
        src_part = "First part\nSecond part"
        
        src_parts, tr_parts, tr_remerged = align_subs(src_sub, tr_sub, src_part)
        
        self.assertEqual(src_parts, ['First part', 'Second part'])
        self.assertEqual(tr_parts, ['First part', 'Second part'])
        self.assertEqual(tr_remerged, 'First part Second part')


class TestGenSubModule(unittest.TestCase):
    """Test core/_6_gen_sub.py - Subtitle generation with timestamps"""
    
    def setUp(self):
        self.patcher_load_key = patch('core._6_gen_sub.load_key')
        self.patcher_console = patch('core._6_gen_sub.console')
        self.patcher_os_makedirs = patch('core._6_gen_sub.os.makedirs')
        
        self.mock_load_key = self.patcher_load_key.start()
        self.mock_console = self.patcher_console.start()
        self.mock_makedirs = self.patcher_os_makedirs.start()
        
    def tearDown(self):
        self.patcher_load_key.stop()
        self.patcher_console.stop()
        self.patcher_os_makedirs.stop()
        
    def test_convert_to_srt_format(self):
        """Test SRT time format conversion"""
        from core._6_gen_sub import convert_to_srt_format
        
        result = convert_to_srt_format(1.5, 3.75)
        expected = "00:00:01,500 --> 00:00:03,750"
        self.assertEqual(result, expected)
        
    def test_remove_punctuation(self):
        """Test punctuation removal"""
        from core._6_gen_sub import remove_punctuation
        
        result = remove_punctuation("Hello, world! How are you?")
        expected = "Hello world How are you"
        self.assertEqual(result, expected)
        
    def test_clean_translation(self):
        """Test translation cleaning function"""
        from core._6_gen_sub import clean_translation
        
        result = clean_translation("Hello。World，")
        expected = "Hello World"  # Should remove Chinese punctuation
        self.assertEqual(result.replace('.', '').strip(), expected)


class TestMergeAudioModule(unittest.TestCase):
    """Test core/_11_merge_audio.py - Audio merging functionality"""
    
    def setUp(self):
        self.patcher_console = patch('core._11_merge_audio.console')
        self.patcher_subprocess = patch('core._11_merge_audio.subprocess')
        self.patcher_audio_segment = patch('core._11_merge_audio.AudioSegment')
        self.patcher_os = patch('core._11_merge_audio.os')
        
        self.mock_console = self.patcher_console.start()
        self.mock_subprocess = self.patcher_subprocess.start()
        self.mock_audio_segment = self.patcher_audio_segment.start()
        self.mock_os = self.patcher_os.start()
        
        # Setup AudioSegment mocks
        self.mock_audio_segment.silent.return_value = Mock()
        self.mock_audio_segment.from_mp3.return_value = Mock()
        
    def tearDown(self):
        self.patcher_console.stop()
        self.patcher_subprocess.stop()
        self.patcher_audio_segment.stop()
        self.patcher_os.stop()
        
    def test_load_and_flatten_data(self):
        """Test Excel data loading and flattening"""
        from core._11_merge_audio import load_and_flatten_data
        
        # Mock DataFrame with eval-able strings
        mock_df = pd.DataFrame({
            'lines': ['["line1", "line2"]', '["line3"]'],
            'new_sub_times': ['[(0.0, 1.0), (1.0, 2.0)]', '[(2.0, 3.0)]']
        })
        
        with patch('pandas.read_excel', return_value=mock_df):
            df, lines, times = load_and_flatten_data('test.xlsx')
            
            expected_lines = ["line1", "line2", "line3"]
            expected_times = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
            
            self.assertEqual(lines, expected_lines)
            self.assertEqual(times, expected_times)
            
    def test_get_audio_files(self):
        """Test audio file path generation"""
        from core._11_merge_audio import get_audio_files
        
        # Mock DataFrame
        mock_df = pd.DataFrame({
            'number': [1, 2],
            'lines': ['["line1", "line2"]', '["line3"]']
        })
        
        result = get_audio_files(mock_df)
        
        expected = [
            'output/audio_segs/1_0.wav',
            'output/audio_segs/1_1.wav',
            'output/audio_segs/2_0.wav'
        ]
        self.assertEqual(result, expected)
        
    def test_process_audio_segment(self):
        """Test single audio segment processing"""
        from core._11_merge_audio import process_audio_segment
        
        # Mock file operations
        self.mock_os.remove = Mock()
        mock_audio = Mock()
        self.mock_audio_segment.from_mp3.return_value = mock_audio
        
        result = process_audio_segment('test.wav')
        
        # Should run ffmpeg and return audio segment
        self.mock_subprocess.run.assert_called_once()
        self.mock_audio_segment.from_mp3.assert_called_once()
        self.assertEqual(result, mock_audio)


if __name__ == '__main__':
    unittest.main()
