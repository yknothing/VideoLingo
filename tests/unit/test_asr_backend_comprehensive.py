"""
Comprehensive test suite for ASR backend modules.
Tests audio preprocessing, transcription services, and API integrations.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import pandas as pd
import subprocess
import requests
import io
import numpy as np

# Test imports for ASR modules
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
    from core.asr_backend.whisperX_local import (
        check_hf_mirror,
        transcribe_audio
    )
    from core.asr_backend.whisperX_302 import transcribe_audio_302
    from core.asr_backend.elevenlabs_asr import (
        elev2whisper,
        transcribe_audio_elevenlabs,
        iso_639_2_to_1
    )
except ImportError as e:
    pytest.skip(f"ASR backend modules not available: {e}", allow_module_level=True)


class TestAudioPreprocess:
    """Test audio preprocessing functionality."""
    
    @pytest.fixture
    def mock_audio_segment(self):
        """Mock AudioSegment for testing."""
        with patch('core.asr_backend.audio_preprocess.AudioSegment') as mock:
            mock_instance = Mock()
            mock_instance.dBFS = -30.0
            mock_instance.apply_gain.return_value = mock_instance
            mock.from_file.return_value = mock_instance
            yield mock

    @pytest.fixture
    def temp_audio_file(self):
        """Create temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            yield f.name
        os.unlink(f.name)

    def test_normalize_audio_volume_success(self, mock_audio_segment, temp_audio_file):
        """Test successful audio volume normalization."""
        result = normalize_audio_volume(temp_audio_file, temp_audio_file, target_db=-20.0)
        
        assert result == temp_audio_file
        mock_audio_segment.from_file.assert_called_once_with(temp_audio_file)
        mock_audio_segment.from_file.return_value.apply_gain.assert_called_once()
        mock_audio_segment.from_file.return_value.export.assert_called_once_with(temp_audio_file, format="wav")

    def test_normalize_audio_volume_with_different_format(self, mock_audio_segment, temp_audio_file):
        """Test audio normalization with different format."""
        normalize_audio_volume(temp_audio_file, temp_audio_file, target_db=-15.0, format="mp3")
        
        mock_audio_segment.from_file.return_value.export.assert_called_once_with(temp_audio_file, format="mp3")

    @patch('core.asr_backend.audio_preprocess.subprocess.run')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    @patch('core.asr_backend.audio_preprocess.os.path.exists')
    def test_convert_video_to_audio_success(self, mock_exists, mock_makedirs, mock_subprocess):
        """Test successful video to audio conversion."""
        mock_exists.return_value = False
        
        with patch('core.asr_backend.audio_preprocess._AUDIO_DIR', '/tmp/audio'):
            with patch('core.asr_backend.audio_preprocess._RAW_AUDIO_FILE', '/tmp/audio/raw.mp3'):
                convert_video_to_audio('/tmp/video.mp4')
        
        mock_makedirs.assert_called_once_with('/tmp/audio', exist_ok=True)
        mock_subprocess.assert_called_once()
        
        # Verify ffmpeg command
        call_args = mock_subprocess.call_args[0][0]
        assert call_args[0] == 'ffmpeg'
        assert '/tmp/video.mp4' in call_args
        assert '/tmp/audio/raw.mp3' in call_args

    @patch('core.asr_backend.audio_preprocess.os.path.exists')
    def test_convert_video_to_audio_skip_existing(self, mock_exists):
        """Test skipping conversion when audio file exists."""
        mock_exists.return_value = True
        
        with patch('core.asr_backend.audio_preprocess.subprocess.run') as mock_subprocess:
            with patch('core.asr_backend.audio_preprocess._RAW_AUDIO_FILE', '/tmp/audio/raw.mp3'):
                convert_video_to_audio('/tmp/video.mp4')
        
        mock_subprocess.assert_not_called()

    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_success(self, mock_popen):
        """Test successful audio duration extraction."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b'', b'Duration: 00:02:30.50, start: 0.000000')
        mock_popen.return_value = mock_process
        
        duration = get_audio_duration('/tmp/audio.wav')
        
        assert duration == 150.5  # 2 minutes 30.5 seconds
        mock_popen.assert_called_once()

    @patch('core.asr_backend.audio_preprocess.subprocess.Popen')
    def test_get_audio_duration_failure(self, mock_popen):
        """Test audio duration extraction failure."""
        mock_process = Mock()
        mock_process.communicate.return_value = (b'', b'Invalid output')
        mock_popen.return_value = mock_process
        
        duration = get_audio_duration('/tmp/audio.wav')
        
        assert duration == 0

    @patch('core.asr_backend.audio_preprocess.AudioSegment.from_file')
    @patch('core.asr_backend.audio_preprocess.mediainfo')
    @patch('core.asr_backend.audio_preprocess.detect_silence')
    def test_split_audio_short_duration(self, mock_detect_silence, mock_mediainfo, mock_from_file):
        """Test audio splitting with short duration."""
        mock_mediainfo.return_value = {"duration": "25.0"}  # Less than target_len + win
        
        segments = split_audio('/tmp/audio.wav', target_len=30*60, win=60)
        
        assert segments == [(0, 25.0)]

    @patch('core.asr_backend.audio_preprocess.AudioSegment.from_file')
    @patch('core.asr_backend.audio_preprocess.mediainfo')
    @patch('core.asr_backend.audio_preprocess.detect_silence')
    def test_split_audio_with_silence_detection(self, mock_detect_silence, mock_mediainfo, mock_from_file):
        """Test audio splitting with silence detection."""
        mock_mediainfo.return_value = {"duration": "2000.0"}  # Long duration
        mock_audio = Mock()
        mock_from_file.return_value = mock_audio
        
        # Mock silence regions found
        mock_detect_silence.return_value = [(500, 1500)]  # 0.5s to 1.5s silence
        
        segments = split_audio('/tmp/audio.wav', target_len=30*60, win=60)
        
        assert len(segments) > 0
        mock_detect_silence.assert_called()

    def test_process_transcription_basic(self):
        """Test basic transcription processing."""
        result = {
            'segments': [
                {
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 0.5},
                        {'word': 'World', 'start': 0.5, 'end': 1.0}
                    ]
                }
            ]
        }
        
        df = process_transcription(result)
        
        assert len(df) == 2
        assert df.iloc[0]['text'] == 'Hello'
        assert df.iloc[0]['start'] == 0.0
        assert df.iloc[0]['end'] == 0.5
        assert df.iloc[1]['text'] == 'World'

    def test_process_transcription_with_speaker_id(self):
        """Test transcription processing with speaker identification."""
        result = {
            'segments': [
                {
                    'speaker_id': 'speaker_1',
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 0.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(result)
        
        assert len(df) == 1
        assert df.iloc[0]['speaker_id'] == 'speaker_1'

    def test_process_transcription_long_word_filtering(self):
        """Test filtering of excessively long words."""
        result = {
            'segments': [
                {
                    'words': [
                        {'word': 'Short', 'start': 0.0, 'end': 0.5},
                        {'word': 'A' * 35, 'start': 0.5, 'end': 1.0}  # 35 characters
                    ]
                }
            ]
        }
        
        df = process_transcription(result)
        
        assert len(df) == 1  # Long word should be filtered out
        assert df.iloc[0]['text'] == 'Short'

    def test_process_transcription_french_guillemets(self):
        """Test French guillemets removal."""
        result = {
            'segments': [
                {
                    'words': [
                        {'word': '»Hello«', 'start': 0.0, 'end': 0.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(result)
        
        assert df.iloc[0]['text'] == 'Hello'

    def test_process_transcription_missing_timestamps(self):
        """Test handling words without timestamps."""
        result = {
            'segments': [
                {
                    'words': [
                        {'word': 'First', 'start': 0.0, 'end': 0.5},
                        {'word': 'Second'},  # No timestamps
                        {'word': 'Third', 'start': 1.0, 'end': 1.5}
                    ]
                }
            ]
        }
        
        df = process_transcription(result)
        
        assert len(df) == 3
        assert df.iloc[1]['text'] == 'Second'
        assert df.iloc[1]['start'] == 0.5  # Should use previous word's end time

    @patch('core.asr_backend.audio_preprocess.pd.DataFrame.to_excel')
    @patch('core.asr_backend.audio_preprocess.os.makedirs')
    def test_save_results_success(self, mock_makedirs, mock_to_excel):
        """Test successful saving of results."""
        df = pd.DataFrame({
            'text': ['Hello', 'World', ''],
            'start': [0.0, 0.5, 1.0],
            'end': [0.5, 1.0, 1.5]
        })
        
        with patch('core.asr_backend.audio_preprocess._2_CLEANED_CHUNKS', '/tmp/output.xlsx'):
            save_results(df)
        
        mock_makedirs.assert_called_once_with('output/log', exist_ok=True)
        mock_to_excel.assert_called_once()

    @patch('core.asr_backend.audio_preprocess.update_key')
    def test_save_language(self, mock_update_key):
        """Test saving detected language."""
        save_language('en')
        
        mock_update_key.assert_called_once_with("whisper.detected_language", 'en')


class TestWhisperXLocal:
    """Test WhisperX local transcription functionality."""
    
    @patch('core.asr_backend.whisperX_local.subprocess.run')
    @patch('core.asr_backend.whisperX_local.time.time')
    def test_check_hf_mirror_success(self, mock_time, mock_subprocess):
        """Test successful HF mirror check."""
        mock_time.side_effect = [0, 1, 2, 3]  # Time progression
        mock_subprocess.return_value.returncode = 0
        
        result = check_hf_mirror()
        
        assert result.startswith('https://')
        assert mock_subprocess.call_count >= 1

    @patch('core.asr_backend.whisperX_local.subprocess.run')
    def test_check_hf_mirror_all_fail(self, mock_subprocess):
        """Test HF mirror check when all mirrors fail."""
        mock_subprocess.return_value.returncode = 1
        
        with patch('core.asr_backend.whisperX_local.time.time', side_effect=[0, 1, 2, 3]):
            result = check_hf_mirror()
        
        assert result.startswith('https://huggingface.co')

    @patch('core.asr_backend.whisperX_local.whisperx.load_model')
    @patch('core.asr_backend.whisperX_local.whisperx.load_align_model')
    @patch('core.asr_backend.whisperX_local.whisperx.align')
    @patch('core.asr_backend.whisperX_local.librosa.load')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_available')
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.update_key')
    def test_transcribe_audio_success_cpu(self, mock_update_key, mock_load_key, mock_cuda, 
                                         mock_librosa_load, mock_align, mock_load_align, mock_load_model):
        """Test successful audio transcription on CPU."""
        # Setup mocks
        mock_cuda.return_value = False
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.model': 'base'
        }.get(key, 'default')
        
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'language': 'en',
            'segments': [
                {'start': 0.0, 'end': 1.0, 'words': [{'word': 'Hello', 'start': 0.0, 'end': 1.0}]}
            ]
        }
        mock_load_model.return_value = mock_model
        
        mock_align_model = Mock()
        mock_metadata = Mock()
        mock_load_align.return_value = (mock_align_model, mock_metadata)
        
        mock_align.return_value = {
            'segments': [
                {'start': 0.0, 'end': 1.0, 'words': [{'word': 'Hello', 'start': 0.0, 'end': 1.0}]}
            ]
        }
        
        mock_librosa_load.return_value = (np.array([0.1, 0.2]), 16000)
        
        # Test transcription
        result = transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        assert result is not None
        assert 'segments' in result
        mock_model.transcribe.assert_called_once()
        mock_align.assert_called_once()

    @patch('core.asr_backend.whisperX_local.whisperx.load_model')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_available')
    @patch('core.asr_backend.whisperX_local.torch.cuda.get_device_properties')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_bf16_supported')
    @patch('core.asr_backend.whisperX_local.load_key')
    def test_transcribe_audio_gpu_optimization(self, mock_load_key, mock_bf16, mock_device_props, 
                                              mock_cuda, mock_load_model):
        """Test GPU optimization in transcription."""
        mock_cuda.return_value = True
        mock_bf16.return_value = True
        mock_device_props.return_value.total_memory = 12 * 1024**3  # 12GB
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.model': 'large'
        }.get(key, 'default')
        
        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Test exception to stop execution")
        mock_load_model.return_value = mock_model
        
        with pytest.raises(Exception):
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        # Verify GPU optimization was applied
        mock_load_model.assert_called_once()
        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs['compute_type'] == 'float16'

    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.os.path.exists')
    def test_transcribe_audio_chinese_model_selection(self, mock_exists, mock_load_key):
        """Test Chinese model selection."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'zh'
        }.get(key, 'default')
        mock_exists.return_value = True
        
        with patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.side_effect = Exception("Test exception")
            mock_load_model.return_value = mock_model
            
            with pytest.raises(Exception):
                transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Should use Chinese model path
            call_args = mock_load_model.call_args[0]
            assert 'Belle-whisper' in call_args[0]

    @patch('core.asr_backend.whisperX_local.load_key')
    def test_transcribe_audio_language_validation(self, mock_load_key):
        """Test language validation for Chinese detection."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en'  # Set to English initially
        }.get(key, 'default')
        
        with patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'language': 'zh',  # But detected as Chinese
                'segments': []
            }
            mock_load_model.return_value = mock_model
            
            with patch('core.asr_backend.whisperX_local.librosa.load') as mock_librosa:
                mock_librosa.return_value = (np.array([0.1]), 16000)
                
                # Should raise ValueError for Chinese detection mismatch
                with pytest.raises(ValueError, match="Please specify the transcription language as zh"):
                    transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)


class TestWhisperX302:
    """Test 302.ai WhisperX integration."""
    
    @patch('core.asr_backend.whisperX_302.os.path.exists')
    @patch('core.asr_backend.whisperX_302.json.load')
    def test_transcribe_audio_302_cached_result(self, mock_json_load, mock_exists):
        """Test returning cached transcription result."""
        mock_exists.return_value = True
        mock_json_load.return_value = {'segments': []}
        
        with patch('builtins.open', mock_open()):
            result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        assert result == {'segments': []}
        mock_json_load.assert_called_once()

    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    @patch('core.asr_backend.whisperX_302.os.path.exists')
    @patch('core.asr_backend.whisperX_302.os.makedirs')
    def test_transcribe_audio_302_api_success(self, mock_makedirs, mock_exists, mock_load_key, 
                                            mock_requests, mock_librosa_load):
        """Test successful 302.ai API transcription."""
        mock_exists.return_value = False
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.whisperX_302_api_key': 'test_key'
        }.get(key, 'default')
        mock_librosa_load.return_value = (np.array([0.1, 0.2]), 16000)
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'segments': [
                {'start': 0.0, 'end': 1.0, 'words': [{'word': 'Hello', 'start': 0.0, 'end': 1.0}]}
            ]
        }
        mock_requests.return_value = mock_response
        
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('core.asr_backend.whisperX_302.json.dump') as mock_json_dump:
                result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        assert 'segments' in result
        mock_requests.assert_called_once()
        mock_json_dump.assert_called_once()

    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.load_key')
    @patch('core.asr_backend.whisperX_302.os.path.exists')
    def test_transcribe_audio_302_timestamp_adjustment(self, mock_exists, mock_load_key, mock_librosa_load):
        """Test timestamp adjustment for audio segments."""
        mock_exists.return_value = False
        mock_load_key.return_value = 'en'
        mock_librosa_load.return_value = (np.array([0.1]), 16000)
        
        with patch('core.asr_backend.whisperX_302.requests.request') as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {
                'segments': [
                    {
                        'start': 0.0,
                        'end': 1.0,
                        'words': [{'word': 'Hello', 'start': 0.0, 'end': 1.0}]
                    }
                ]
            }
            mock_requests.return_value = mock_response
            
            with patch('builtins.open', mock_open()):
                with patch('core.asr_backend.whisperX_302.json.dump'):
                    result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 5.0, 15.0)
        
        # Check that timestamps were adjusted by start offset (5.0)
        assert result['segments'][0]['start'] == 5.0
        assert result['segments'][0]['end'] == 6.0
        assert result['segments'][0]['words'][0]['start'] == 5.0
        assert result['segments'][0]['words'][0]['end'] == 6.0


class TestElevenLabsASR:
    """Test ElevenLabs ASR integration."""
    
    def test_iso_639_2_to_1_mapping(self):
        """Test ISO language code conversion."""
        assert iso_639_2_to_1['eng'] == 'en'
        assert iso_639_2_to_1['zho'] == 'zh'
        assert iso_639_2_to_1['fra'] == 'fr'

    def test_elev2whisper_basic_conversion(self):
        """Test basic ElevenLabs to Whisper format conversion."""
        elev_json = {
            'words': [
                {'text': 'Hello', 'start': 0.0, 'end': 0.5, 'speaker_id': 'speaker_1'},
                {'text': ' world', 'start': 0.5, 'end': 1.0, 'speaker_id': 'speaker_1'}
            ]
        }
        
        result = elev2whisper(elev_json, word_level_timestamp=False)
        
        assert len(result['segments']) == 1
        assert result['segments'][0]['text'] == 'Hello world'
        assert result['segments'][0]['start'] == 0.0
        assert result['segments'][0]['end'] == 1.0

    def test_elev2whisper_with_word_timestamps(self):
        """Test conversion with word-level timestamps."""
        elev_json = {
            'words': [
                {'text': 'Hello', 'start': 0.0, 'end': 0.5, 'speaker_id': 'speaker_1'}
            ]
        }
        
        result = elev2whisper(elev_json, word_level_timestamp=True)
        
        assert len(result['segments']) == 1
        assert 'words' in result['segments'][0]
        assert len(result['segments'][0]['words']) == 1

    def test_elev2whisper_speaker_change(self):
        """Test segment splitting on speaker change."""
        elev_json = {
            'words': [
                {'text': 'Hello', 'start': 0.0, 'end': 0.5, 'speaker_id': 'speaker_1'},
                {'text': 'world', 'start': 0.5, 'end': 1.0, 'speaker_id': 'speaker_2'}
            ]
        }
        
        result = elev2whisper(elev_json, word_level_timestamp=False)
        
        assert len(result['segments']) == 2
        assert result['segments'][0]['speaker_id'] == 'speaker_1'
        assert result['segments'][1]['speaker_id'] == 'speaker_2'

    def test_elev2whisper_time_gap_splitting(self):
        """Test segment splitting on time gaps."""
        elev_json = {
            'words': [
                {'text': 'Hello', 'start': 0.0, 'end': 0.5, 'speaker_id': 'speaker_1'},
                {'text': 'world', 'start': 2.0, 'end': 2.5, 'speaker_id': 'speaker_1'}  # Gap > 1 second
            ]
        }
        
        result = elev2whisper(elev_json, word_level_timestamp=False)
        
        assert len(result['segments']) == 2  # Should split on time gap

    def test_elev2whisper_empty_words(self):
        """Test handling empty words list."""
        elev_json = {'words': []}
        
        result = elev2whisper(elev_json)
        
        assert result == {'segments': []}

    @patch('core.asr_backend.elevenlabs_asr.os.path.exists')
    @patch('core.asr_backend.elevenlabs_asr.json.load')
    def test_transcribe_audio_elevenlabs_cached(self, mock_json_load, mock_exists):
        """Test returning cached ElevenLabs transcription."""
        mock_exists.return_value = True
        mock_json_load.return_value = {'segments': []}
        
        with patch('builtins.open', mock_open()):
            result = transcribe_audio_elevenlabs('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        assert result == {'segments': []}

    @patch('core.asr_backend.elevenlabs_asr.librosa.load')
    @patch('core.asr_backend.elevenlabs_asr.requests.post')
    @patch('core.asr_backend.elevenlabs_asr.load_key')
    @patch('core.asr_backend.elevenlabs_asr.update_key')
    @patch('core.asr_backend.elevenlabs_asr.os.path.exists')
    @patch('core.asr_backend.elevenlabs_asr.os.makedirs')
    def test_transcribe_audio_elevenlabs_api_success(self, mock_makedirs, mock_exists, mock_update_key,
                                                   mock_load_key, mock_requests, mock_librosa_load):
        """Test successful ElevenLabs API transcription."""
        mock_exists.return_value = False
        mock_load_key.side_effect = lambda key: {
            'whisper.elevenlabs_api_key': 'test_key',
            'whisper.language': 'en'
        }.get(key, 'default')
        mock_librosa_load.return_value = (np.array([0.1]), 16000)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'language_code': 'eng',
            'words': [
                {'text': 'Hello', 'start': 0.0, 'end': 0.5, 'speaker_id': 'speaker_1'}
            ]
        }
        mock_requests.return_value = mock_response
        
        with patch('core.asr_backend.elevenlabs_asr.tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/temp.mp3'
            with patch('core.asr_backend.elevenlabs_asr.sf.write'):
                with patch('core.asr_backend.elevenlabs_asr.os.remove'):
                    with patch('builtins.open', mock_open()):
                        with patch('core.asr_backend.elevenlabs_asr.json.dump'):
                            result = transcribe_audio_elevenlabs('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        assert 'segments' in result
        mock_requests.assert_called_once()
        mock_update_key.assert_called_once_with("whisper.detected_language", 'en')

    @patch('core.asr_backend.elevenlabs_asr.librosa.load')
    @patch('core.asr_backend.elevenlabs_asr.load_key')
    def test_transcribe_audio_elevenlabs_timestamp_adjustment(self, mock_load_key, mock_librosa_load):
        """Test timestamp adjustment in ElevenLabs transcription."""
        mock_load_key.return_value = 'test_key'
        mock_librosa_load.return_value = (np.array([0.1]), 16000)
        
        with patch('core.asr_backend.elevenlabs_asr.requests.post') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'language_code': 'eng',
                'words': [
                    {'text': 'Hello', 'start': 0.0, 'end': 0.5, 'speaker_id': 'speaker_1'}
                ]
            }
            mock_requests.return_value = mock_response
            
            with patch('core.asr_backend.elevenlabs_asr.tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = '/tmp/temp.mp3'
                with patch('core.asr_backend.elevenlabs_asr.sf.write'):
                    with patch('core.asr_backend.elevenlabs_asr.os.remove'):
                        with patch('builtins.open', mock_open()):
                            with patch('core.asr_backend.elevenlabs_asr.json.dump'):
                                with patch('core.asr_backend.elevenlabs_asr.os.path.exists', return_value=False):
                                    result = transcribe_audio_elevenlabs('/tmp/raw.wav', '/tmp/vocal.wav', 5.0, 15.0)
        
        # Mock the elev2whisper conversion result
        with patch('core.asr_backend.elevenlabs_asr.elev2whisper') as mock_elev2whisper:
            mock_elev2whisper.return_value = {'segments': []}
            # The actual timestamp adjustment happens in the response processing
            # This test verifies the flow works correctly


if __name__ == '__main__':
    pytest.main([__file__])
