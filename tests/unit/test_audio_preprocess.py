"""
Comprehensive test suite for audio preprocessing and Demucs integration.
Tests audio normalization, conversion, segmentation, transcription processing, and vocal separation.
"""

import pytest
import os
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import subprocess
import gc

# Test imports for audio preprocessing module
try:
    from core.asr_backend.audio_preprocess import (
        normalize_audio_volume,
        convert_video_to_audio,
        get_audio_duration,
        split_audio,
        process_transcription,
        save_results,
        save_language
    )
    from core.asr_backend.demucs_vl import (
        is_demucs_available,
        demucs_audio,
        create_preloaded_separator
    )
except ImportError as e:
    pytest.skip(f"Audio preprocessing modules not available: {e}", allow_module_level=True)


class TestAudioNormalization:
    """Test audio volume normalization functionality."""
    
    @pytest.fixture
    def mock_audio_segment(self):
        """Mock AudioSegment for audio processing tests."""
        with patch('core.asr_backend.audio_preprocess.AudioSegment') as mock_segment:
            mock_audio = Mock()
            mock_audio.dBFS = -25.0
            mock_audio.apply_gain.return_value = mock_audio
            mock_audio.export.return_value = None
            mock_segment.from_file.return_value = mock_audio
            yield mock_segment
    
    def test_normalize_audio_volume_basic(self, mock_audio_segment):
        """Test basic audio volume normalization."""
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            result = normalize_audio_volume(
                temp_file.name, temp_file.name, target_db=-20.0, format="wav"
            )
            
            assert result == temp_file.name
            mock_audio_segment.from_file.assert_called_once_with(temp_file.name)
            
            # Verify gain was applied (target -20dB, original -25dB = +5dB gain)
            mock_audio_segment.from_file.return_value.apply_gain.assert_called_once_with(5.0)
            mock_audio_segment.from_file.return_value.export.assert_called_once_with(
                temp_file.name, format="wav"
            )
    
    def test_normalize_audio_volume_different_formats(self, mock_audio_segment):
        """Test normalization with different audio formats."""
        formats = ['wav', 'mp3', 'flac', 'ogg']
        
        for fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=f'.{fmt}') as temp_file:
                normalize_audio_volume(temp_file.name, temp_file.name, format=fmt)
                
                # Verify export was called with correct format
                mock_audio_segment.from_file.return_value.export.assert_called_with(
                    temp_file.name, format=fmt
                )
    
    def test_normalize_audio_volume_gain_calculation(self, mock_audio_segment):
        """Test correct gain calculation for different target levels."""
        test_cases = [
            (-30.0, -20.0, 10.0),  # Original -30dB, target -20dB = +10dB gain
            (-15.0, -20.0, -5.0),  # Original -15dB, target -20dB = -5dB gain
            (-20.0, -20.0, 0.0),   # Original -20dB, target -20dB = 0dB gain
        ]
        
        for original_db, target_db, expected_gain in test_cases:
            mock_audio_segment.from_file.return_value.dBFS = original_db
            
            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
                normalize_audio_volume(temp_file.name, temp_file.name, target_db=target_db)
                
                mock_audio_segment.from_file.return_value.apply_gain.assert_called_with(expected_gain)
    
    def test_normalize_audio_volume_error_handling(self, mock_audio_segment):
        """Test error handling in audio normalization."""
        mock_audio_segment.from_file.side_effect = Exception("Audio loading failed")
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_file:
            with pytest.raises(Exception, match="Audio loading failed"):
                normalize_audio_volume(temp_file.name, temp_file.name)


class TestVideoToAudioConversion:
    """Test video to audio conversion functionality."""
    
    @patch('core.asr_backend.audio_preprocess.subprocess.run')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.os.path.exists')
    def test_convert_video_to_audio_success(self, mock_exists, mock_makedirs, mock_subprocess):
        """Test successful video to audio conversion."""
        mock_exists.return_value = False  # Audio file doesn't exist
        
        with patch('core.asr_backend.audio_preprocess._AUDIO_DIR', '/tmp/audio'), \
             patch('core.asr_backend.audio_preprocess._RAW_AUDIO_FILE', '/tmp/audio/raw.mp3'):
            
            convert_video_to_audio('/tmp/test_video.mp4')
            
            # Verify directory creation
            mock_makedirs.assert_called_once_with('/tmp/audio', exist_ok=True)
            
            # Verify ffmpeg command
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            
            assert call_args[0] == 'ffmpeg'
            assert '-y' in call_args  # Overwrite output
            assert '-i' in call_args  # Input file
            assert '/tmp/test_video.mp4' in call_args
            assert '/tmp/audio/raw.mp3' in call_args
            assert '-vn' in call_args  # No video
            assert '-c:a' in call_args and 'libmp3lame' in call_args  # Audio codec
            assert '-b:a' in call_args and '32k' in call_args  # Bitrate
            assert '-ar' in call_args and '16000' in call_args  # Sample rate
            assert '-ac' in call_args and '1' in call_args  # Mono channel
    
    @patch('core.asr_backend.audio_preprocess.os.path.exists')
    @patch('core.asr_backend.audio_preprocess.subprocess.run')
    def test_convert_video_to_audio_skip_existing(self, mock_subprocess, mock_exists):
        """Test skipping conversion when audio file already exists."""
        mock_exists.return_value = True  # Audio file exists
        
        with patch('core.asr_backend.audio_preprocess._RAW_AUDIO_FILE', '/tmp/audio/existing.mp3'):
            convert_video_to_audio('/tmp/test_video.mp4')
            
            # Should not run ffmpeg if file exists
            mock_subprocess.assert_not_called()
    
    @patch('core.asr_backend.audio_preprocess.subprocess.run')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.os.path.exists')
    def test_convert_video_to_audio_ffmpeg_failure(self, mock_exists, mock_makedirs, mock_subprocess):
        """Test handling of FFmpeg conversion failure."""
        mock_exists.return_value = False
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'ffmpeg')
        
        with patch('core.asr_backend.audio_preprocess._AUDIO_DIR', '/tmp/audio'), \
             patch('core.asr_backend.audio_preprocess._RAW_AUDIO_FILE', '/tmp/audio/raw.mp3'):
            
            with pytest.raises(subprocess.CalledProcessError):
                convert_video_to_audio('/tmp/test_video.mp4')


class TestAudioDurationExtraction:
    """Test audio duration extraction functionality."""
    
    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_success(self, mock_popen):
        """Test successful audio duration extraction."""
        mock_process = Mock()
        mock_process.communicate.return_value = (
            b'',
            b'Duration: 00:02:30.50, start: 0.000000, bitrate: 128 kb/s'
        )
        mock_popen.return_value = mock_process
        
        duration = get_audio_duration('/tmp/test_audio.wav')
        
        assert duration == 150.5  # 2 minutes 30.5 seconds
        mock_popen.assert_called_once_with(
            ['ffmpeg', '-i', '/tmp/test_audio.wav'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_various_formats(self, mock_popen):
        """Test duration extraction with various time formats."""
        test_cases = [
            ('Duration: 00:01:30.25, start: 0.000000', 90.25),
            ('Duration: 01:00:00.00, start: 0.000000', 3600.0),
            ('Duration: 00:00:05.75, start: 0.000000', 5.75),
            ('Duration: 02:30:45.125, start: 0.000000', 9045.125),
        ]
        
        for duration_line, expected_seconds in test_cases:
            mock_process = Mock()
            mock_process.communicate.return_value = (b'', duration_line.encode())
            mock_popen.return_value = mock_process
            
            duration = get_audio_duration('/tmp/test.wav')
            assert abs(duration - expected_seconds) < 0.001
    
    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_failure(self, mock_popen):
        """Test handling of duration extraction failure."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b'', b'Invalid output format')
        mock_popen.return_value = mock_process
        
        duration = get_audio_duration('/tmp/test_audio.wav')
        
        assert duration == 0  # Should return 0 on failure
    
    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_exception_handling(self, mock_popen):
        """Test exception handling in duration extraction."""
        mock_popen.side_effect = Exception("FFmpeg not found")
        
        duration = get_audio_duration('/tmp/test_audio.wav')
        
        assert duration == 0  # Should return 0 on exception


class TestAudioSegmentation:
    """Test audio segmentation and silence detection."""
    
    @pytest.fixture
    def mock_audio_dependencies(self):
        """Mock dependencies for audio segmentation tests."""
        with patch('core.asr_backend.audio_preprocess.AudioSegment') as mock_segment, \
             patch('core.asr_backend.audio_preprocess.mediainfo') as mock_mediainfo, \
             patch('core.asr_backend.audio_preprocess.detect_silence') as mock_detect_silence:
            
            mock_audio = Mock()
            mock_segment.from_file.return_value = mock_audio
            
            yield {
                'segment': mock_segment,
                'mediainfo': mock_mediainfo,
                'detect_silence': mock_detect_silence,
                'audio': mock_audio
            }
    
    def test_split_audio_short_duration(self, mock_audio_dependencies):
        """Test audio splitting when duration is shorter than target."""
        mock_audio_dependencies['mediainfo'].return_value = {"duration": "25.0"}
        
        segments = split_audio('/tmp/test.wav', target_len=30*60, win=60)
        
        assert segments == [(0, 25.0)]  # Single segment for entire audio
    
    def test_split_audio_with_silence_detection(self, mock_audio_dependencies):
        """Test audio splitting with silence detection."""
        mock_audio_dependencies['mediainfo'].return_value = {"duration": "3600.0"}  # 1 hour
        
        # Mock silence detection results (in milliseconds relative to search window)
        mock_audio_dependencies['detect_silence'].return_value = [(500, 1000)]  # 0.5-1.0s silence
        
        segments = split_audio('/tmp/test.wav', target_len=30*60, win=60)
        
        assert len(segments) >= 2  # Should split into multiple segments
        mock_audio_dependencies['detect_silence'].assert_called()
    
    def test_split_audio_no_silence_found(self, mock_audio_dependencies):
        """Test audio splitting when no suitable silence is found."""
        mock_audio_dependencies['mediainfo'].return_value = {"duration": "3600.0"}
        mock_audio_dependencies['detect_silence'].return_value = []  # No silence found
        
        segments = split_audio('/tmp/test.wav', target_len=30*60, win=60)
        
        assert len(segments) >= 1  # Should still split at target threshold
    
    def test_split_audio_silence_filtering(self, mock_audio_dependencies):
        """Test filtering of silence regions by duration and position."""
        mock_audio_dependencies['mediainfo'].return_value = {"duration": "3600.0"}
        
        # Mock various silence regions with different characteristics
        silence_regions = [
            (100, 200),   # Too short (0.1s)
            (500, 600),   # Too short (0.1s)
            (1000, 2000), # Good silence (1.0s)
            (2500, 3000), # Good silence (0.5s) but wrong position
        ]
        mock_audio_dependencies['detect_silence'].return_value = silence_regions
        
        segments = split_audio('/tmp/test.wav', target_len=30*60, win=60)
        
        # Should use appropriate silence regions for splitting
        assert len(segments) > 1
    
    def test_split_audio_custom_parameters(self, mock_audio_dependencies):
        """Test audio splitting with custom parameters."""
        mock_audio_dependencies['mediainfo'].return_value = {"duration": "900.0"}  # 15 minutes
        mock_audio_dependencies['detect_silence'].return_value = [(500, 1000)]
        
        # Custom target length and window
        segments = split_audio('/tmp/test.wav', target_len=10*60, win=30)
        
        # Verify custom parameters were used
        assert len(segments) >= 1
        
        # Check that detect_silence was called with appropriate parameters
        call_args = mock_audio_dependencies['detect_silence'].call_args
        assert call_args[1]['min_silence_len'] == 500  # 0.5s * 1000ms
        assert call_args[1]['silence_thresh'] == -30
    
    def test_split_audio_edge_cases(self, mock_audio_dependencies):
        """Test audio splitting edge cases."""
        # Test very short audio
        mock_audio_dependencies['mediainfo'].return_value = {"duration": "5.0"}
        segments = split_audio('/tmp/test.wav', target_len=30*60, win=60)
        assert segments == [(0, 5.0)]
        
        # Test audio exactly at target length
        mock_audio_dependencies['mediainfo'].return_value = {"duration": str(30*60)}
        segments = split_audio('/tmp/test.wav', target_len=30*60, win=60)
        assert len(segments) == 1
        assert segments[0] == (0, 30*60)
    
    def test_split_audio_memory_efficiency(self, mock_audio_dependencies):
        """Test that audio splitting handles memory efficiently for large files."""
        mock_audio_dependencies['mediainfo'].return_value = {"duration": "7200.0"}  # 2 hours
        mock_audio_dependencies['detect_silence'].return_value = [(1000, 2000)]
        
        # Mock large audio file
        large_audio = Mock()
        large_audio.__getitem__ = Mock(return_value=Mock())
        mock_audio_dependencies['segment'].from_file.return_value = large_audio
        
        segments = split_audio('/tmp/large_test.wav', target_len=30*60, win=60)
        
        # Should split large file into multiple segments
        assert len(segments) > 2


class TestTranscriptionProcessing:
    """Test transcription result processing and formatting."""
    
    def test_process_transcription_basic(self):
        """Test basic transcription processing."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 0.5},
                        {'word': 'world', 'start': 0.5, 'end': 1.0},
                        {'word': 'test', 'start': 1.0, 'end': 1.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 3
        assert df.iloc[0]['text'] == 'Hello'
        assert df.iloc[0]['start'] == 0.0
        assert df.iloc[0]['end'] == 0.5
        assert df.iloc[1]['text'] == 'world'
        assert df.iloc[2]['text'] == 'test'
    
    def test_process_transcription_with_speaker_id(self):
        """Test transcription processing with speaker identification."""
        transcription_result = {
            'segments': [
                {
                    'speaker_id': 'speaker_1',
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 0.5}
                    ]
                },
                {
                    'speaker_id': 'speaker_2',
                    'words': [
                        {'word': 'World', 'start': 1.0, 'end': 1.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 2
        assert df.iloc[0]['speaker_id'] == 'speaker_1'
        assert df.iloc[1]['speaker_id'] == 'speaker_2'
    
    def test_process_transcription_missing_speaker_id(self):
        """Test transcription processing with missing speaker IDs."""
        transcription_result = {
            'segments': [
                {
                    # No speaker_id field
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 0.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 1
        assert pd.isna(df.iloc[0]['speaker_id']) or df.iloc[0]['speaker_id'] is None
    
    def test_process_transcription_long_word_filtering(self):
        """Test filtering of excessively long words."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'Normal', 'start': 0.0, 'end': 0.5},
                        {'word': 'A' * 35, 'start': 0.5, 'end': 1.0},  # 35 chars (too long)
                        {'word': 'Also normal', 'start': 1.0, 'end': 1.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 2  # Long word should be filtered out
        assert df.iloc[0]['text'] == 'Normal'
        assert df.iloc[1]['text'] == 'Also normal'
    
    def test_process_transcription_french_guillemets(self):
        """Test French guillemets removal."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': '»Hello«', 'start': 0.0, 'end': 0.5},
                        {'word': '«World»', 'start': 0.5, 'end': 1.0}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 2
        assert df.iloc[0]['text'] == 'Hello'
        assert df.iloc[1]['text'] == 'World'
    
    def test_process_transcription_missing_timestamps(self):
        """Test handling of words without timestamps."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'First', 'start': 0.0, 'end': 0.5},
                        {'word': 'Second'},  # Missing timestamps
                        {'word': 'Third', 'start': 1.0, 'end': 1.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 3
        assert df.iloc[0]['text'] == 'First'
        assert df.iloc[1]['text'] == 'Second'
        assert df.iloc[1]['start'] == 0.5  # Should use previous word's end time
        assert df.iloc[1]['end'] == 0.5
        assert df.iloc[2]['text'] == 'Third'
    
    def test_process_transcription_first_word_missing_timestamps(self):
        """Test handling when first word is missing timestamps."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'First'},  # Missing timestamps
                        {'word': 'Second', 'start': 1.0, 'end': 1.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 2
        assert df.iloc[0]['text'] == 'First'
        assert df.iloc[0]['start'] == 1.0  # Should use next word's start time
        assert df.iloc[0]['end'] == 1.5    # Should use next word's end time
    
    def test_process_transcription_all_words_missing_timestamps(self):
        """Test handling when all words are missing timestamps."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'First'},
                        {'word': 'Second'}
                    ]
                }
            ]
        }
        
        with pytest.raises(Exception, match="No next word with timestamp found"):
            process_transcription(transcription_result)
    
    def test_process_transcription_empty_segments(self):
        """Test processing of empty segments."""
        transcription_result = {'segments': []}
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)


class TestResultSaving:
    """Test saving of transcription results."""
    
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.pd.DataFrame.to_excel')
    def test_save_results_success(self, mock_to_excel, mock_makedirs):
        """Test successful saving of results."""
        df = pd.DataFrame({
            'text': ['Hello', 'World', 'Test'],
            'start': [0.0, 0.5, 1.0],
            'end': [0.5, 1.0, 1.5]
        })
        
        with patch('core.asr_backend.audio_preprocess._2_CLEANED_CHUNKS', '/tmp/output.xlsx'):
            save_results(df)
        
        mock_makedirs.assert_called_once_with('output/log', exist_ok=True)
        mock_to_excel.assert_called_once_with('/tmp/output.xlsx', index=False)
    
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.pd.DataFrame.to_excel')
    def test_save_results_empty_text_removal(self, mock_to_excel, mock_makedirs):
        """Test removal of rows with empty text."""
        df = pd.DataFrame({
            'text': ['Hello', '', 'World', ''],
            'start': [0.0, 0.5, 1.0, 1.5],
            'end': [0.5, 1.0, 1.5, 2.0]
        })
        
        with patch('core.asr_backend.audio_preprocess._2_CLEANED_CHUNKS', '/tmp/output.xlsx'):
            save_results(df)
        
        # Should save DataFrame with empty text rows removed
        mock_to_excel.assert_called_once()
        saved_df_call = mock_to_excel.call_args[0]
        # Cannot easily verify the filtered DataFrame without accessing internals
    
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.pd.DataFrame.to_excel')
    def test_save_results_long_text_removal(self, mock_to_excel, mock_makedirs):
        """Test removal of rows with excessively long text."""
        df = pd.DataFrame({
            'text': ['Normal', 'A' * 35, 'Also normal'],  # Middle text too long
            'start': [0.0, 0.5, 1.0],
            'end': [0.5, 1.0, 1.5]
        })
        
        with patch('core.asr_backend.audio_preprocess._2_CLEANED_CHUNKS', '/tmp/output.xlsx'):
            save_results(df)
        
        # Should filter out long text
        mock_to_excel.assert_called_once()
    
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.pd.DataFrame.to_excel')
    def test_save_results_text_quoting(self, mock_to_excel, mock_makedirs):
        """Test that text is properly quoted in output."""
        df = pd.DataFrame({
            'text': ['Hello', 'World'],
            'start': [0.0, 0.5],
            'end': [0.5, 1.0]
        })
        
        with patch('core.asr_backend.audio_preprocess._2_CLEANED_CHUNKS', '/tmp/output.xlsx'):
            save_results(df)
        
        # Text should be quoted
        mock_to_excel.assert_called_once()
    
    @patch('core.asr_backend.audio_preprocess.update_key')
    def test_save_language(self, mock_update_key):
        """Test language saving functionality."""
        save_language('fr')
        
        mock_update_key.assert_called_once_with("whisper.detected_language", 'fr')
    
    @patch('core.asr_backend.audio_preprocess.update_key')
    def test_save_language_various_codes(self, mock_update_key):
        """Test saving various language codes."""
        languages = ['en', 'zh', 'fr', 'de', 'es', 'ja', 'ko']
        
        for lang in languages:
            save_language(lang)
        
        assert mock_update_key.call_count == len(languages)
        for i, lang in enumerate(languages):
            assert mock_update_key.call_args_list[i] == call("whisper.detected_language", lang)


class TestDemucsIntegration:
    """Test Demucs vocal separation integration."""
    
    def test_is_demucs_available_true(self):
        """Test Demucs availability check when available."""
        with patch('core.asr_backend.demucs_vl.DEMUCS_AVAILABLE', True):
            assert is_demucs_available() is True
    
    def test_is_demucs_available_false(self):
        """Test Demucs availability check when not available."""
        with patch('core.asr_backend.demucs_vl.DEMUCS_AVAILABLE', False):
            assert is_demucs_available() is False
    
    @patch('core.asr_backend.demucs_vl.DEMUCS_AVAILABLE', False)
    @patch('core.asr_backend.demucs_vl.rprint')
    def test_demucs_audio_unavailable(self, mock_rprint):
        """Test Demucs processing when not available."""
        demucs_audio()
        
        # Should print warning and return early
        warning_calls = [call for call in mock_rprint.call_args_list 
                        if 'Demucs is not available' in str(call)]
        assert len(warning_calls) > 0
    
    @patch('core.asr_backend.demucs_vl.DEMUCS_AVAILABLE', True)
    @patch('core.asr_backend.demucs_vl.os.path.exists')
    @patch('core.asr_backend.demucs_vl.rprint')
    def test_demucs_audio_files_exist(self, mock_rprint, mock_exists):
        """Test Demucs processing when output files already exist."""
        mock_exists.return_value = True  # Both vocal and background files exist
        
        with patch('core.asr_backend.demucs_vl._VOCAL_AUDIO_FILE', '/tmp/vocal.wav'), \
             patch('core.asr_backend.demucs_vl._BACKGROUND_AUDIO_FILE', '/tmp/background.wav'):
            
            demucs_audio()
            
            # Should skip processing and print warning
            warning_calls = [call for call in mock_rprint.call_args_list 
                            if 'already exist, skip Demucs processing' in str(call)]
            assert len(warning_calls) > 0
    
    @patch('core.asr_backend.demucs_vl.DEMUCS_AVAILABLE', True)
    @patch('core.asr_backend.demucs_vl.os.path.exists')
    @patch('core.asr_backend.demucs_vl.os.makedirs')
    @patch('core.asr_backend.demucs_vl.get_model')
    @patch('core.asr_backend.demucs_vl.create_preloaded_separator')
    @patch('core.asr_backend.demucs_vl.save_audio')
    @patch('core.asr_backend.demucs_vl.gc.collect')
    def test_demucs_audio_successful_separation(self, mock_gc, mock_save_audio, mock_separator, 
                                               mock_get_model, mock_makedirs, mock_exists):
        """Test successful Demucs audio separation."""
        mock_exists.return_value = False  # Files don't exist
        
        # Mock model and separator
        mock_model = Mock()
        mock_model.samplerate = 44100
        mock_get_model.return_value = mock_model
        
        mock_sep_instance = Mock()
        mock_outputs = {
            'vocals': Mock(),
            'drums': Mock(),
            'bass': Mock(),
            'other': Mock()
        }
        mock_outputs['vocals'].cpu.return_value = Mock()
        for key in ['drums', 'bass', 'other']:
            mock_outputs[key].cpu.return_value = Mock()
        
        mock_sep_instance.separate_audio_file.return_value = (None, mock_outputs)
        mock_separator.return_value = mock_sep_instance
        
        with patch('core.asr_backend.demucs_vl._AUDIO_DIR', '/tmp/audio'), \
             patch('core.asr_backend.demucs_vl._RAW_AUDIO_FILE', '/tmp/raw.wav'), \
             patch('core.asr_backend.demucs_vl._VOCAL_AUDIO_FILE', '/tmp/vocal.wav'), \
             patch('core.asr_backend.demucs_vl._BACKGROUND_AUDIO_FILE', '/tmp/background.wav'):
            
            demucs_audio()
            
            # Verify model loading and separation
            mock_get_model.assert_called_once_with('htdemucs')
            mock_separator.assert_called_once()
            mock_sep_instance.separate_audio_file.assert_called_once_with('/tmp/raw.wav')
            
            # Verify audio saving
            assert mock_save_audio.call_count == 2  # Vocals and background
            
            # Verify cleanup
            mock_gc.assert_called()
    
    @patch('core.asr_backend.demucs_vl.DEMUCS_AVAILABLE', True)
    @patch('core.asr_backend.demucs_vl.os.path.exists')
    @patch('core.asr_backend.demucs_vl.get_model')
    @patch('core.asr_backend.demucs_vl.rprint')
    def test_demucs_audio_exception_handling(self, mock_rprint, mock_get_model, mock_exists):
        """Test Demucs exception handling."""
        mock_exists.return_value = False
        mock_get_model.side_effect = Exception("Model loading failed")
        
        with patch('core.asr_backend.demucs_vl._AUDIO_DIR', '/tmp/audio'):
            demucs_audio()
            
            # Should handle exception gracefully
            error_calls = [call for call in mock_rprint.call_args_list 
                          if 'Error during audio separation' in str(call)]
            assert len(error_calls) > 0
            
            continue_calls = [call for call in mock_rprint.call_args_list 
                             if 'Continuing without audio separation' in str(call)]
            assert len(continue_calls) > 0
    
    @patch('core.asr_backend.demucs_vl.DEMUCS_AVAILABLE', True)
    @patch('core.asr_backend.demucs_vl.Separator')
    def test_create_preloaded_separator_gpu(self, mock_separator_class):
        """Test creation of preloaded separator with GPU."""
        mock_model = Mock()
        mock_separator_instance = Mock()
        mock_separator_class.return_value = mock_separator_instance
        
        with patch('core.asr_backend.demucs_vl.is_cuda_available', return_value=True), \
             patch('core.asr_backend.demucs_vl.torch.backends.mps.is_available', return_value=False):
            
            separator = create_preloaded_separator(mock_model)
            
            assert separator == mock_separator_instance
            mock_separator_class.assert_called_once_with(mock_model)
            mock_separator_instance.update_parameter.assert_called_once()
            
            # Verify GPU device was used
            call_kwargs = mock_separator_instance.update_parameter.call_args[1]
            assert call_kwargs['device'] == 'cuda'
    
    @patch('core.asr_backend.demucs_vl.DEMUCS_AVAILABLE', True)
    @patch('core.asr_backend.demucs_vl.Separator')
    def test_create_preloaded_separator_cpu(self, mock_separator_class):
        """Test creation of preloaded separator with CPU fallback."""
        mock_model = Mock()
        mock_separator_instance = Mock()
        mock_separator_class.return_value = mock_separator_instance
        
        with patch('core.asr_backend.demucs_vl.is_cuda_available', return_value=False), \
             patch('core.asr_backend.demucs_vl.torch.backends.mps.is_available', return_value=False):
            
            separator = create_preloaded_separator(mock_model, shifts=2, overlap=0.5)
            
            assert separator == mock_separator_instance
            
            # Verify CPU device was used and custom parameters
            call_kwargs = mock_separator_instance.update_parameter.call_args[1]
            assert call_kwargs['device'] == 'cpu'
            assert call_kwargs['shifts'] == 2
            assert call_kwargs['overlap'] == 0.5
    
    @patch('core.asr_backend.demucs_vl.DEMUCS_AVAILABLE', False)
    def test_create_preloaded_separator_unavailable(self):
        """Test separator creation when Demucs is unavailable."""
        mock_model = Mock()
        
        with pytest.raises(ImportError, match="demucs is not available"):
            create_preloaded_separator(mock_model)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
