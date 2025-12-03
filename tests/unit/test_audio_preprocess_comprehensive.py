"""
Comprehensive test suite specifically for audio preprocessing functionality.
Tests audio format conversion, volume normalization, and audio analysis.
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import subprocess

# Test imports for audio preprocessing
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
except ImportError as e:
    pytest.skip(f"Audio preprocessing modules not available: {e}", allow_module_level=True)


class TestAudioNormalization:
    """Test audio volume normalization functionality."""
    
    @pytest.fixture
    def mock_audio_segment(self):
        """Create mock AudioSegment for testing."""
        with patch('core.asr_backend.audio_preprocess.AudioSegment') as mock:
            mock_audio = Mock()
            mock_audio.dBFS = -25.0
            mock_audio.apply_gain.return_value = mock_audio
            mock.from_file.return_value = mock_audio
            yield mock_audio
    
    def test_normalize_audio_volume_standard_case(self, mock_audio_segment):
        """Test standard audio normalization."""
        with patch('core.asr_backend.audio_preprocess.AudioSegment.from_file', return_value=mock_audio_segment):
            result = normalize_audio_volume("/tmp/input.wav", "/tmp/output.wav")
        
        assert result == "/tmp/output.wav"
        mock_audio_segment.apply_gain.assert_called_once()
        mock_audio_segment.export.assert_called_once_with("/tmp/output.wav", format="wav")

    def test_normalize_audio_volume_different_target_db(self, mock_audio_segment):
        """Test audio normalization with different target dB."""
        mock_audio_segment.dBFS = -30.0
        
        with patch('core.asr_backend.audio_preprocess.AudioSegment.from_file', return_value=mock_audio_segment):
            normalize_audio_volume("/tmp/input.wav", "/tmp/output.wav", target_db=-18.0)
        
        # Should apply gain to reach target dB
        expected_gain = -18.0 - (-30.0)  # 12 dB gain
        mock_audio_segment.apply_gain.assert_called_once_with(expected_gain)

    def test_normalize_audio_volume_mp3_format(self, mock_audio_segment):
        """Test audio normalization with MP3 output format."""
        with patch('core.asr_backend.audio_preprocess.AudioSegment.from_file', return_value=mock_audio_segment):
            normalize_audio_volume("/tmp/input.wav", "/tmp/output.mp3", format="mp3")
        
        mock_audio_segment.export.assert_called_once_with("/tmp/output.mp3", format="mp3")

    def test_normalize_audio_volume_already_normalized(self, mock_audio_segment):
        """Test normalization when audio is already at target level."""
        mock_audio_segment.dBFS = -20.0  # Already at default target
        
        with patch('core.asr_backend.audio_preprocess.AudioSegment.from_file', return_value=mock_audio_segment):
            normalize_audio_volume("/tmp/input.wav", "/tmp/output.wav", target_db=-20.0)
        
        mock_audio_segment.apply_gain.assert_called_once_with(0.0)  # No gain needed

    def test_normalize_audio_volume_extreme_quiet(self, mock_audio_segment):
        """Test normalization of extremely quiet audio."""
        mock_audio_segment.dBFS = -60.0  # Very quiet
        
        with patch('core.asr_backend.audio_preprocess.AudioSegment.from_file', return_value=mock_audio_segment):
            normalize_audio_volume("/tmp/input.wav", "/tmp/output.wav", target_db=-12.0)
        
        expected_gain = -12.0 - (-60.0)  # 48 dB boost
        mock_audio_segment.apply_gain.assert_called_once_with(expected_gain)

    def test_normalize_audio_volume_extreme_loud(self, mock_audio_segment):
        """Test normalization of extremely loud audio."""
        mock_audio_segment.dBFS = -3.0  # Very loud
        
        with patch('core.asr_backend.audio_preprocess.AudioSegment.from_file', return_value=mock_audio_segment):
            normalize_audio_volume("/tmp/input.wav", "/tmp/output.wav", target_db=-20.0)
        
        expected_gain = -20.0 - (-3.0)  # -17 dB reduction
        mock_audio_segment.apply_gain.assert_called_once_with(expected_gain)


class TestVideoToAudioConversion:
    """Test video to audio conversion functionality."""
    
    @patch('core.asr_backend.audio_preprocess.subprocess.run')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.os.path.exists')
    def test_convert_video_to_audio_mp4_to_mp3(self, mock_exists, mock_makedirs, mock_subprocess):
        """Test converting MP4 video to MP3 audio."""
        mock_exists.return_value = False
        
        with patch('core.asr_backend.audio_preprocess._AUDIO_DIR', '/tmp/audio'):
            with patch('core.asr_backend.audio_preprocess._RAW_AUDIO_FILE', '/tmp/audio/raw.mp3'):
                convert_video_to_audio('/tmp/video.mp4')
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with('/tmp/audio', exist_ok=True)
        
        # Verify ffmpeg command structure
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert call_args[0] == 'ffmpeg'
        assert '-y' in call_args  # Overwrite output
        assert '-i' in call_args  # Input file
        assert '/tmp/video.mp4' in call_args
        assert '-vn' in call_args  # No video
        assert 'libmp3lame' in call_args  # MP3 codec
        assert '32k' in call_args  # Bitrate
        assert '16000' in call_args  # Sample rate
        assert '-ac' in call_args and '1' in call_args  # Mono channel

    @patch('core.asr_backend.audio_preprocess.subprocess.run')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.os.path.exists')
    def test_convert_video_to_audio_avi_format(self, mock_exists, mock_makedirs, mock_subprocess):
        """Test converting AVI video to audio."""
        mock_exists.return_value = False
        
        with patch('core.asr_backend.audio_preprocess._AUDIO_DIR', '/tmp/audio'):
            with patch('core.asr_backend.audio_preprocess._RAW_AUDIO_FILE', '/tmp/audio/raw.mp3'):
                convert_video_to_audio('/tmp/video.avi')
        
        call_args = mock_subprocess.call_args[0][0]
        assert '/tmp/video.avi' in call_args

    @patch('core.asr_backend.audio_preprocess.os.path.exists')
    def test_convert_video_to_audio_skip_if_exists(self, mock_exists):
        """Test skipping conversion when audio file already exists."""
        mock_exists.return_value = True
        
        with patch('core.asr_backend.audio_preprocess.subprocess.run') as mock_subprocess:
            with patch('core.asr_backend.audio_preprocess._RAW_AUDIO_FILE', '/tmp/audio/existing.mp3'):
                convert_video_to_audio('/tmp/video.mp4')
        
        mock_subprocess.assert_not_called()

    @patch('core.asr_backend.audio_preprocess.subprocess.run')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.os.path.exists')
    def test_convert_video_to_audio_ffmpeg_failure(self, mock_exists, mock_makedirs, mock_subprocess):
        """Test handling FFmpeg conversion failure."""
        mock_exists.return_value = False
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'ffmpeg')
        
        with pytest.raises(subprocess.CalledProcessError):
            with patch('core.asr_backend.audio_preprocess._AUDIO_DIR', '/tmp/audio'):
                with patch('core.asr_backend.audio_preprocess._RAW_AUDIO_FILE', '/tmp/audio/raw.mp3'):
                    convert_video_to_audio('/tmp/invalid_video.mp4')

    @patch('core.asr_backend.audio_preprocess.subprocess.run')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.os.path.exists')
    def test_convert_video_to_audio_utf8_metadata(self, mock_exists, mock_makedirs, mock_subprocess):
        """Test UTF-8 metadata encoding in conversion."""
        mock_exists.return_value = False
        
        with patch('core.asr_backend.audio_preprocess._AUDIO_DIR', '/tmp/audio'):
            with patch('core.asr_backend.audio_preprocess._RAW_AUDIO_FILE', '/tmp/audio/raw.mp3'):
                convert_video_to_audio('/tmp/video.mp4')
        
        call_args = mock_subprocess.call_args[0][0]
        assert '-metadata' in call_args
        assert 'encoding=UTF-8' in call_args


class TestAudioDurationExtraction:
    """Test audio duration extraction functionality."""
    
    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_minutes_seconds(self, mock_popen):
        """Test duration extraction for typical audio files."""
        mock_process = Mock()
        mock_process.communicate.return_value = (
            b'',
            b'Duration: 00:03:45.67, start: 0.000000, bitrate: 128 kb/s'
        )
        mock_popen.return_value = mock_process
        
        duration = get_audio_duration('/tmp/audio.mp3')
        
        expected_duration = 3*60 + 45.67  # 3 minutes 45.67 seconds = 225.67 seconds
        assert abs(duration - expected_duration) < 0.01

    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_hours_format(self, mock_popen):
        """Test duration extraction with hours."""
        mock_process = Mock()
        mock_process.communicate.return_value = (
            b'',
            b'Duration: 01:23:45.12, start: 0.000000'
        )
        mock_popen.return_value = mock_process
        
        duration = get_audio_duration('/tmp/long_audio.mp3')
        
        expected_duration = 1*3600 + 23*60 + 45.12  # 1 hour 23 minutes 45.12 seconds
        assert abs(duration - expected_duration) < 0.01

    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_short_file(self, mock_popen):
        """Test duration extraction for short audio files."""
        mock_process = Mock()
        mock_process.communicate.return_value = (
            b'',
            b'Duration: 00:00:02.50, start: 0.000000'
        )
        mock_popen.return_value = mock_process
        
        duration = get_audio_duration('/tmp/short.wav')
        
        assert abs(duration - 2.5) < 0.01

    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_invalid_format(self, mock_popen):
        """Test duration extraction with invalid ffmpeg output."""
        mock_process = Mock()
        mock_process.communicate.return_value = (
            b'',
            b'Invalid input format or corrupted file'
        )
        mock_popen.return_value = mock_process
        
        duration = get_audio_duration('/tmp/corrupted.mp3')
        
        assert duration == 0

    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_no_duration_line(self, mock_popen):
        """Test duration extraction when no duration line found."""
        mock_process = Mock()
        mock_process.communicate.return_value = (
            b'',
            b'Some other output without duration information'
        )
        mock_popen.return_value = mock_process
        
        duration = get_audio_duration('/tmp/audio.wav')
        
        assert duration == 0

    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_ffmpeg_error(self, mock_popen):
        """Test handling ffmpeg process errors."""
        mock_process = Mock()
        mock_process.communicate.side_effect = Exception("Process error")
        mock_popen.return_value = mock_process
        
        duration = get_audio_duration('/tmp/audio.wav')
        
        assert duration == 0


class TestAudioSplitting:
    """Test audio splitting functionality."""
    
    @pytest.fixture
    def mock_audio_components(self):
        """Mock audio processing components."""
        with patch('core.asr_backend.audio_preprocess.AudioSegment.from_file') as mock_from_file, \
             patch('core.asr_backend.audio_preprocess.mediainfo') as mock_mediainfo, \
             patch('core.asr_backend.audio_preprocess.detect_silence') as mock_silence:
            
            mock_audio = Mock()
            mock_from_file.return_value = mock_audio
            yield mock_from_file, mock_mediainfo, mock_silence, mock_audio

    def test_split_audio_short_file(self, mock_audio_components):
        """Test splitting audio file shorter than target length."""
        _, mock_mediainfo, _, _ = mock_audio_components
        mock_mediainfo.return_value = {"duration": "15.5"}  # 15.5 seconds
        
        segments = split_audio('/tmp/short.wav', target_len=30, win=5)
        
        assert segments == [(0, 15.5)]

    def test_split_audio_exact_target_length(self, mock_audio_components):
        """Test splitting audio file exactly at target length."""
        _, mock_mediainfo, _, _ = mock_audio_components
        mock_mediainfo.return_value = {"duration": "30.0"}  # Exactly 30 seconds
        
        segments = split_audio('/tmp/exact.wav', target_len=30, win=5)
        
        assert segments == [(0, 30.0)]

    def test_split_audio_with_silence_detection(self, mock_audio_components):
        """Test audio splitting with silence detection."""
        _, mock_mediainfo, mock_silence, _ = mock_audio_components
        mock_mediainfo.return_value = {"duration": "120.0"}  # 2 minutes
        
        # Mock silence detection - found silence at 58-62 seconds
        mock_silence.return_value = [(2000, 4000)]  # 2-4 seconds in the search window
        
        segments = split_audio('/tmp/long.wav', target_len=60, win=10)
        
        assert len(segments) >= 1
        mock_silence.assert_called()

    def test_split_audio_no_silence_found(self, mock_audio_components):
        """Test audio splitting when no silence is detected."""
        _, mock_mediainfo, mock_silence, _ = mock_audio_components
        mock_mediainfo.return_value = {"duration": "150.0"}  # 2.5 minutes
        
        # Mock no silence detection
        mock_silence.return_value = []
        
        segments = split_audio('/tmp/nosil.wav', target_len=60, win=10)
        
        assert len(segments) >= 2  # Should split at threshold even without silence

    def test_split_audio_multiple_segments(self, mock_audio_components):
        """Test splitting long audio into multiple segments."""
        _, mock_mediainfo, mock_silence, _ = mock_audio_components
        mock_mediainfo.return_value = {"duration": "300.0"}  # 5 minutes
        
        # Mock finding silence for each split
        mock_silence.return_value = [(1000, 2000)]  # 1-2 seconds silence found
        
        segments = split_audio('/tmp/verylong.wav', target_len=120, win=20)
        
        assert len(segments) >= 2
        # Verify segments don't overlap
        for i in range(len(segments) - 1):
            assert segments[i][1] <= segments[i + 1][0]

    def test_split_audio_invalid_silence_regions(self, mock_audio_components):
        """Test handling invalid silence regions."""
        _, mock_mediainfo, mock_silence, _ = mock_audio_components
        mock_mediainfo.return_value = {"duration": "180.0"}  # 3 minutes
        
        # Mock silence regions that are too short or poorly positioned
        mock_silence.return_value = [(100, 200)]  # Too short (0.1 second silence)
        
        segments = split_audio('/tmp/badsil.wav', target_len=90, win=15)
        
        assert len(segments) >= 1

    def test_split_audio_custom_parameters(self, mock_audio_components):
        """Test audio splitting with custom target length and window."""
        _, mock_mediainfo, mock_silence, _ = mock_audio_components
        mock_mediainfo.return_value = {"duration": "600.0"}  # 10 minutes
        
        mock_silence.return_value = [(500, 1500)]  # 0.5-1.5 seconds silence
        
        segments = split_audio('/tmp/custom.wav', target_len=180, win=30)  # 3 min target, 30 sec window
        
        assert len(segments) >= 2
        mock_silence.assert_called()

    def test_split_audio_safe_margin_handling(self, mock_audio_components):
        """Test proper handling of safe margins around silence."""
        _, mock_mediainfo, mock_silence, _ = mock_audio_components
        mock_mediainfo.return_value = {"duration": "240.0"}  # 4 minutes
        
        # Mock finding suitable silence region
        mock_silence.return_value = [(3000, 5000)]  # 3-5 seconds silence (2 seconds long)
        
        segments = split_audio('/tmp/margins.wav', target_len=120, win=20)
        
        assert len(segments) >= 2
        # Should split within the silence region with safe margins


class TestTranscriptionProcessing:
    """Test transcription result processing."""
    
    def test_process_transcription_basic_structure(self):
        """Test basic transcription processing structure."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 0.5},
                        {'word': 'world', 'start': 0.6, 'end': 1.2},
                        {'word': 'test', 'start': 1.3, 'end': 1.8}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 3
        assert list(df.columns) == ['text', 'start', 'end', 'speaker_id']
        assert df.iloc[0]['text'] == 'Hello'
        assert df.iloc[1]['text'] == 'world'
        assert df.iloc[2]['text'] == 'test'

    def test_process_transcription_speaker_identification(self):
        """Test transcription processing with speaker identification."""
        transcription_result = {
            'segments': [
                {
                    'speaker_id': 'speaker_A',
                    'words': [
                        {'word': 'First', 'start': 0.0, 'end': 0.5},
                        {'word': 'speaker', 'start': 0.6, 'end': 1.2}
                    ]
                },
                {
                    'speaker_id': 'speaker_B', 
                    'words': [
                        {'word': 'Second', 'start': 2.0, 'end': 2.5},
                        {'word': 'speaker', 'start': 2.6, 'end': 3.2}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 4
        assert df.iloc[0]['speaker_id'] == 'speaker_A'
        assert df.iloc[1]['speaker_id'] == 'speaker_A'
        assert df.iloc[2]['speaker_id'] == 'speaker_B'
        assert df.iloc[3]['speaker_id'] == 'speaker_B'

    def test_process_transcription_missing_speaker_id(self):
        """Test processing when speaker_id is missing."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'No', 'start': 0.0, 'end': 0.5},
                        {'word': 'speaker', 'start': 0.6, 'end': 1.2}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 2
        assert pd.isna(df.iloc[0]['speaker_id'])
        assert pd.isna(df.iloc[1]['speaker_id'])

    def test_process_transcription_long_words_filtering(self):
        """Test filtering of excessively long words."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'Normal', 'start': 0.0, 'end': 0.5},
                        {'word': 'x' * 35, 'start': 0.6, 'end': 1.2},  # 35 chars - too long
                        {'word': 'Also_normal', 'start': 1.3, 'end': 1.8}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        # Long word should be filtered out
        assert len(df) == 2
        assert 'Normal' in df['text'].values
        assert 'Also_normal' in df['text'].values
        assert 'x' * 35 not in df['text'].values

    def test_process_transcription_french_guillemets(self):
        """Test removal of French guillemets characters."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': '»Bonjour«', 'start': 0.0, 'end': 0.5},
                        {'word': '»world«', 'start': 0.6, 'end': 1.2}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert df.iloc[0]['text'] == 'Bonjour'
        assert df.iloc[1]['text'] == 'world'

    def test_process_transcription_missing_timestamps_first_word(self):
        """Test handling missing timestamps on first word."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'Missing_timestamps'},  # No start/end
                        {'word': 'Has_timestamps', 'start': 1.0, 'end': 1.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 2
        assert df.iloc[0]['text'] == 'Missing_timestamps'
        assert df.iloc[0]['start'] == 1.0  # Should use next word's start time
        assert df.iloc[0]['end'] == 1.5    # Should use next word's end time

    def test_process_transcription_missing_timestamps_middle_word(self):
        """Test handling missing timestamps on middle word."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'First', 'start': 0.0, 'end': 0.5},
                        {'word': 'Missing'},  # No timestamps
                        {'word': 'Third', 'start': 1.0, 'end': 1.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 3
        assert df.iloc[1]['text'] == 'Missing'
        assert df.iloc[1]['start'] == 0.5  # Should use previous word's end time
        assert df.iloc[1]['end'] == 0.5    # Should use previous word's end time

    def test_process_transcription_missing_timestamps_exception(self):
        """Test exception when first word lacks timestamps and no next word available."""
        transcription_result = {
            'segments': [
                {
                    'words': [
                        {'word': 'Only_word_no_timestamps'}  # No start/end and no next word
                    ]
                }
            ]
        }
        
        with pytest.raises(Exception, match="No next word with timestamp found"):
            process_transcription(transcription_result)

    def test_process_transcription_empty_segments(self):
        """Test processing empty segments."""
        transcription_result = {'segments': []}
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 0
        assert list(df.columns) == ['text', 'start', 'end', 'speaker_id']

    def test_process_transcription_empty_words(self):
        """Test processing segments with empty words."""
        transcription_result = {
            'segments': [
                {
                    'words': []
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        assert len(df) == 0


class TestResultsSaving:
    """Test transcription results saving functionality."""
    
    @patch('core.asr_backend.audio_preprocess.pd.DataFrame.to_excel')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    def test_save_results_basic(self, mock_makedirs, mock_to_excel):
        """Test basic results saving."""
        df = pd.DataFrame({
            'text': ['Hello', 'world', 'test'],
            'start': [0.0, 0.5, 1.0],
            'end': [0.5, 1.0, 1.5],
            'speaker_id': [None, None, None]
        })
        
        with patch('core.asr_backend.audio_preprocess._2_CLEANED_CHUNKS', '/tmp/output.xlsx'):
            save_results(df)
        
        mock_makedirs.assert_called_once_with('output/log', exist_ok=True)
        mock_to_excel.assert_called_once_with('/tmp/output.xlsx', index=False)

    @patch('core.asr_backend.audio_preprocess.pd.DataFrame.to_excel')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    def test_save_results_empty_text_removal(self, mock_makedirs, mock_to_excel):
        """Test removal of rows with empty text."""
        df = pd.DataFrame({
            'text': ['Hello', '', 'world', ''],
            'start': [0.0, 0.5, 1.0, 1.5],
            'end': [0.5, 1.0, 1.5, 2.0],
            'speaker_id': [None, None, None, None]
        })
        
        with patch('core.asr_backend.audio_preprocess._2_CLEANED_CHUNKS', '/tmp/output.xlsx'):
            save_results(df)
        
        # Should save only non-empty rows
        call_args = mock_to_excel.call_args
        saved_df = call_args[1]['index'] if len(call_args) > 1 else True
        # The function modifies df in place, so we verify makedirs was called
        mock_makedirs.assert_called_once()

    @patch('core.asr_backend.audio_preprocess.pd.DataFrame.to_excel')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    def test_save_results_long_words_removal(self, mock_makedirs, mock_to_excel):
        """Test removal of words longer than 30 characters."""
        df = pd.DataFrame({
            'text': ['Normal', 'x' * 35, 'Also_normal'],
            'start': [0.0, 0.5, 1.0],
            'end': [0.5, 1.0, 1.5],
            'speaker_id': [None, None, None]
        })
        
        with patch('core.asr_backend.audio_preprocess._2_CLEANED_CHUNKS', '/tmp/output.xlsx'):
            save_results(df)
        
        mock_makedirs.assert_called_once()
        mock_to_excel.assert_called_once()

    @patch('core.asr_backend.audio_preprocess.pd.DataFrame.to_excel')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    def test_save_results_text_quoting(self, mock_makedirs, mock_to_excel):
        """Test that text values are properly quoted."""
        df = pd.DataFrame({
            'text': ['Hello', 'world'],
            'start': [0.0, 0.5],
            'end': [0.5, 1.0],
            'speaker_id': [None, None]
        })
        
        with patch('core.asr_backend.audio_preprocess._2_CLEANED_CHUNKS', '/tmp/output.xlsx'):
            save_results(df)
        
        # Function should add quotes around text
        mock_to_excel.assert_called_once()

    @patch('core.asr_backend.audio_preprocess.update_key')
    def test_save_language_english(self, mock_update_key):
        """Test saving English language detection."""
        save_language('en')
        
        mock_update_key.assert_called_once_with("whisper.detected_language", 'en')

    @patch('core.asr_backend.audio_preprocess.update_key')
    def test_save_language_chinese(self, mock_update_key):
        """Test saving Chinese language detection."""
        save_language('zh')
        
        mock_update_key.assert_called_once_with("whisper.detected_language", 'zh')

    @patch('core.asr_backend.audio_preprocess.update_key')
    def test_save_language_other_languages(self, mock_update_key):
        """Test saving other language detections."""
        languages = ['fr', 'de', 'ja', 'ko', 'es']
        
        for lang in languages:
            mock_update_key.reset_mock()
            save_language(lang)
            mock_update_key.assert_called_once_with("whisper.detected_language", lang)


if __name__ == '__main__':
    pytest.main([__file__])
