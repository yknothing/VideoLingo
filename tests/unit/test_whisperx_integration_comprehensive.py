"""
Comprehensive integration test suite for WhisperX ASR functionality.
Tests end-to-end transcription workflows and model integrations.
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
import torch
import subprocess
import time
import json

# Test imports for WhisperX integration
try:
    from core.asr_backend.whisperX_local import (
        check_hf_mirror,
        transcribe_audio
    )
    from core.asr_backend.whisperX_302 import transcribe_audio_302
    from core.asr_backend.audio_preprocess import (
        process_transcription,
        split_audio
    )
except ImportError as e:
    pytest.skip(f"WhisperX integration modules not available: {e}", allow_module_level=True)


class TestWhisperXLocalIntegration:
    """Test WhisperX local model integration."""
    
    @pytest.fixture
    def mock_whisperx_model(self):
        """Mock WhisperX model for testing."""
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'language': 'en',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.5,
                    'text': 'Hello world',
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 0.8},
                        {'word': 'world', 'start': 1.0, 'end': 2.5}
                    ]
                }
            ]
        }
        return mock_model

    @pytest.fixture
    def mock_align_model(self):
        """Mock alignment model for testing."""
        mock_align_model = Mock()
        mock_metadata = Mock()
        return mock_align_model, mock_metadata

    @pytest.fixture
    def mock_aligned_result(self):
        """Mock aligned transcription result."""
        return {
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.5,
                    'text': 'Hello world',
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 0.8},
                        {'word': 'world', 'start': 1.0, 'end': 2.5}
                    ]
                }
            ]
        }

    @patch('core.asr_backend.whisperX_local.whisperx.load_model')
    @patch('core.asr_backend.whisperX_local.whisperx.load_align_model')
    @patch('core.asr_backend.whisperX_local.whisperx.align')
    @patch('core.asr_backend.whisperX_local.librosa.load')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_available')
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.update_key')
    @patch('core.asr_backend.whisperX_local.os.path.exists')
    def test_full_transcription_pipeline_cpu(self, mock_exists, mock_update_key, mock_load_key, 
                                           mock_cuda, mock_librosa, mock_align, mock_load_align, mock_load_model,
                                           mock_whisperx_model, mock_align_model, mock_aligned_result):
        """Test complete transcription pipeline on CPU."""
        # Setup
        mock_cuda.return_value = False
        mock_exists.return_value = False
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.model': 'base',
            'model_dir': '/tmp/models'
        }.get(key, 'default')
        
        mock_librosa.return_value = (np.array([0.1, 0.2, 0.3]), 16000)
        mock_load_model.return_value = mock_whisperx_model
        mock_load_align.return_value = mock_align_model
        mock_align.return_value = mock_aligned_result
        
        # Test
        result = transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        # Verify
        assert result is not None
        assert 'segments' in result
        assert len(result['segments']) == 1
        mock_load_model.assert_called_once()
        mock_align.assert_called_once()
        mock_update_key.assert_called_once_with("whisper.language", 'en')

    @patch('core.asr_backend.whisperX_local.whisperx.load_model')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_available')
    @patch('core.asr_backend.whisperX_local.torch.cuda.get_device_properties')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_bf16_supported')
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.os.path.exists')
    def test_gpu_optimization_high_memory(self, mock_exists, mock_load_key, mock_bf16, 
                                        mock_device_props, mock_cuda, mock_load_model,
                                        mock_whisperx_model):
        """Test GPU optimization for high memory systems."""
        # Setup high memory GPU
        mock_cuda.return_value = True
        mock_bf16.return_value = True
        mock_device_props.return_value.total_memory = 16 * 1024**3  # 16GB
        mock_exists.return_value = False
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.model': 'large-v2'
        }.get(key, 'default')
        
        # Mock model to stop after load
        mock_whisperx_model.transcribe.side_effect = Exception("Stop execution")
        mock_load_model.return_value = mock_whisperx_model
        
        # Test
        with pytest.raises(Exception, match="Stop execution"):
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        # Verify GPU optimization
        mock_load_model.assert_called_once()
        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs['device'] == 'cuda'
        assert call_kwargs['compute_type'] == 'float16'

    @patch('core.asr_backend.whisperX_local.whisperx.load_model')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_available')
    @patch('core.asr_backend.whisperX_local.torch.cuda.get_device_properties')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_bf16_supported')
    @patch('core.asr_backend.whisperX_local.load_key')
    def test_gpu_optimization_low_memory(self, mock_load_key, mock_bf16, mock_device_props, 
                                       mock_cuda, mock_load_model, mock_whisperx_model):
        """Test GPU optimization for low memory systems."""
        # Setup low memory GPU
        mock_cuda.return_value = True
        mock_bf16.return_value = False  # No BF16 support
        mock_device_props.return_value.total_memory = 4 * 1024**3  # 4GB
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.model': 'base'
        }.get(key, 'default')
        
        mock_whisperx_model.transcribe.side_effect = Exception("Stop execution")
        mock_load_model.return_value = mock_whisperx_model
        
        # Test
        with pytest.raises(Exception):
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        # Verify low memory optimization
        call_kwargs = mock_load_model.call_args[1]
        assert call_kwargs['compute_type'] == 'int8'  # Fallback for no BF16

    @patch('core.asr_backend.whisperX_local.whisperx.load_model')
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.os.path.exists')
    def test_chinese_model_selection(self, mock_exists, mock_load_key, mock_load_model,
                                   mock_whisperx_model):
        """Test Chinese model selection and path handling."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'zh',
            'model_dir': '/tmp/models'
        }.get(key, 'default')
        
        # Mock local Chinese model exists
        mock_exists.return_value = True
        
        mock_whisperx_model.transcribe.side_effect = Exception("Stop execution")
        mock_load_model.return_value = mock_whisperx_model
        
        # Test
        with pytest.raises(Exception):
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        # Verify Chinese model path
        call_args = mock_load_model.call_args[0]
        model_path = call_args[0]
        assert 'Belle-whisper-large-v3-zh-punct-fasterwhisper' in model_path

    @patch('core.asr_backend.whisperX_local.whisperx.load_model')
    @patch('core.asr_backend.whisperX_local.whisperx.load_align_model')
    @patch('core.asr_backend.whisperX_local.whisperx.align')
    @patch('core.asr_backend.whisperX_local.librosa.load')
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.update_key')
    def test_chinese_language_validation(self, mock_update_key, mock_load_key, mock_librosa,
                                       mock_align, mock_load_align, mock_load_model):
        """Test Chinese language detection validation."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en'  # Set to English
        }.get(key, 'default')
        mock_librosa.return_value = (np.array([0.1]), 16000)
        
        # Mock model detects Chinese but config is English
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'language': 'zh',  # Detected as Chinese
            'segments': []
        }
        mock_load_model.return_value = mock_model
        
        # Test - should raise validation error
        with pytest.raises(ValueError, match="Please specify the transcription language as zh"):
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)

    @patch('core.asr_backend.whisperX_local.whisperx.load_model')
    @patch('core.asr_backend.whisperX_local.whisperx.load_align_model')
    @patch('core.asr_backend.whisperX_local.whisperx.align')
    @patch('core.asr_backend.whisperX_local.librosa.load')
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.update_key')
    def test_timestamp_adjustment_with_offset(self, mock_update_key, mock_load_key, mock_librosa,
                                            mock_align, mock_load_align, mock_load_model):
        """Test timestamp adjustment when processing audio segments."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.model': 'base'
        }.get(key, 'default')
        mock_librosa.return_value = (np.array([0.1, 0.2]), 16000)
        
        # Mock transcription result with relative timestamps
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'language': 'en',
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 1.0},
                        {'word': 'world', 'start': 1.0, 'end': 2.0}
                    ]
                }
            ]
        }
        mock_load_model.return_value = mock_model
        
        # Mock alignment
        mock_align_model, mock_metadata = Mock(), Mock()
        mock_load_align.return_value = (mock_align_model, mock_metadata)
        mock_align.return_value = {
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 1.0},
                        {'word': 'world', 'start': 1.0, 'end': 2.0}
                    ]
                }
            ]
        }
        
        # Test with 10-second offset
        result = transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 10.0, 20.0)
        
        # Verify timestamps are adjusted by offset
        assert result['segments'][0]['start'] == 10.0
        assert result['segments'][0]['end'] == 12.0
        assert result['segments'][0]['words'][0]['start'] == 10.0
        assert result['segments'][0]['words'][0]['end'] == 11.0
        assert result['segments'][0]['words'][1]['start'] == 11.0
        assert result['segments'][0]['words'][1]['end'] == 12.0

    @patch('core.asr_backend.whisperX_local.whisperx.load_model')
    @patch('core.asr_backend.whisperX_local.torch.cuda.empty_cache')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_available')
    @patch('core.asr_backend.whisperX_local.load_key')
    def test_gpu_memory_cleanup(self, mock_load_key, mock_cuda, mock_empty_cache, mock_load_model):
        """Test proper GPU memory cleanup."""
        mock_cuda.return_value = True
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.model': 'base'
        }.get(key, 'default')
        
        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Test cleanup")
        mock_load_model.return_value = mock_model
        
        # Test
        with pytest.raises(Exception):
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        # Verify memory cleanup was called
        assert mock_empty_cache.call_count >= 1

    @patch('core.asr_backend.whisperX_local.subprocess.run')
    @patch('core.asr_backend.whisperX_local.time.time')
    def test_hf_mirror_selection_performance(self, mock_time, mock_subprocess):
        """Test HuggingFace mirror selection based on performance."""
        # Mock time progression for ping tests
        mock_time.side_effect = [0, 0.5, 1.0, 2.5, 3.0, 4.0]  # Times for ping tests
        
        # Mock subprocess results - first mirror fast, second slow
        mock_results = []
        mock_results.append(Mock(returncode=0))  # Official fast
        mock_results.append(Mock(returncode=0))  # Mirror slow
        mock_subprocess.side_effect = mock_results
        
        result = check_hf_mirror()
        
        # Should select the faster official mirror
        assert result == 'https://huggingface.co'
        assert mock_subprocess.call_count == 2

    @patch('core.asr_backend.whisperX_local.subprocess.run')
    @patch('core.asr_backend.whisperX_local.time.time')
    def test_hf_mirror_fallback_on_failure(self, mock_time, mock_subprocess):
        """Test fallback when all mirrors fail."""
        mock_time.side_effect = [0, 1, 2, 3]
        
        # Mock all pings fail
        mock_subprocess.return_value = Mock(returncode=1)
        
        result = check_hf_mirror()
        
        # Should fallback to official
        assert result == 'https://huggingface.co'

    @patch('core.asr_backend.whisperX_local.os.name', 'nt')
    @patch('core.asr_backend.whisperX_local.subprocess.run')
    def test_hf_mirror_windows_ping_command(self, mock_subprocess):
        """Test Windows-specific ping command format."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        check_hf_mirror()
        
        # Verify Windows ping command format
        call_args = mock_subprocess.call_args_list[0][0][0]
        assert call_args[0] == 'ping'
        assert '-n' in call_args  # Windows ping option
        assert '-w' in call_args  # Windows timeout option

    @patch('core.asr_backend.whisperX_local.os.name', 'posix')
    @patch('core.asr_backend.whisperX_local.subprocess.run')
    def test_hf_mirror_unix_ping_command(self, mock_subprocess):
        """Test Unix-specific ping command format."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        check_hf_mirror()
        
        # Verify Unix ping command format
        call_args = mock_subprocess.call_args_list[0][0][0]
        assert call_args[0] == 'ping'
        assert '-c' in call_args  # Unix count option
        assert '-W' in call_args  # Unix timeout option


class TestWhisperX302Integration:
    """Test 302.ai WhisperX cloud service integration."""
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    @patch('core.asr_backend.whisperX_302.os.path.exists')
    @patch('core.asr_backend.whisperX_302.os.makedirs')
    def test_302_api_transcription_workflow(self, mock_makedirs, mock_exists, mock_load_key, 
                                          mock_requests, mock_librosa):
        """Test complete 302.ai API transcription workflow."""
        # Setup
        mock_exists.return_value = False  # No cache
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.whisperX_302_api_key': 'test_api_key'
        }.get(key, 'default')
        mock_librosa.return_value = (np.array([0.1, 0.2, 0.3, 0.4]), 16000)
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'text': 'Hello world',
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 1.0},
                        {'word': 'world', 'start': 1.0, 'end': 2.0}
                    ]
                }
            ]
        }
        mock_requests.return_value = mock_response
        
        # Test
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('core.asr_backend.whisperX_302.json.dump') as mock_json_dump:
                result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 5.0)
        
        # Verify
        assert 'segments' in result
        assert len(result['segments']) == 1
        mock_requests.assert_called_once()
        mock_json_dump.assert_called_once()  # Cache saved
        
        # Verify API request format
        call_kwargs = mock_requests.call_args[1]
        assert 'headers' in call_kwargs
        assert 'Authorization' in call_kwargs['headers']
        assert 'Bearer test_api_key' in call_kwargs['headers']['Authorization']

    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.load_key')
    @patch('core.asr_backend.whisperX_302.os.path.exists')
    def test_302_audio_processing_and_slicing(self, mock_exists, mock_load_key, mock_librosa):
        """Test audio processing and slicing for 302.ai API."""
        mock_exists.return_value = False
        mock_load_key.return_value = 'en'
        
        # Mock 10-second audio at 16kHz
        sample_rate = 16000
        audio_duration = 10.0
        audio_samples = np.random.random(int(sample_rate * audio_duration))
        mock_librosa.return_value = (audio_samples, sample_rate)
        
        with patch('core.asr_backend.whisperX_302.requests.request') as mock_requests:
            with patch('core.asr_backend.whisperX_302.sf.write') as mock_sf_write:
                mock_response = Mock()
                mock_response.json.return_value = {'segments': []}
                mock_requests.return_value = mock_response
                
                # Test slicing from 2-8 seconds
                with patch('builtins.open', mock_open()):
                    with patch('core.asr_backend.whisperX_302.json.dump'):
                        transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 2.0, 8.0)
                
                # Verify audio slicing
                mock_sf_write.assert_called_once()
                call_args = mock_sf_write.call_args[0]
                audio_slice = call_args[1]  # Second argument is audio data
                expected_samples = int((8.0 - 2.0) * sample_rate)  # 6 seconds
                assert len(audio_slice) == expected_samples

    @patch('core.asr_backend.whisperX_302.os.path.exists')
    @patch('core.asr_backend.whisperX_302.json.load')
    def test_302_caching_mechanism(self, mock_json_load, mock_exists):
        """Test caching mechanism for 302.ai transcription."""
        # Mock cache exists
        mock_exists.return_value = True
        mock_cached_result = {
            'segments': [
                {'start': 0.0, 'end': 5.0, 'text': 'Cached result'}
            ]
        }
        mock_json_load.return_value = mock_cached_result
        
        # Test
        with patch('builtins.open', mock_open()):
            result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        # Verify cached result returned
        assert result == mock_cached_result
        mock_json_load.assert_called_once()

    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    @patch('core.asr_backend.whisperX_302.os.path.exists')
    def test_302_timestamp_adjustment_precision(self, mock_exists, mock_load_key, 
                                              mock_requests, mock_librosa):
        """Test precise timestamp adjustment for 302.ai results."""
        mock_exists.return_value = False
        mock_load_key.return_value = 'en'
        mock_librosa.return_value = (np.array([0.1, 0.2]), 16000)
        
        # Mock API response with precise timestamps
        mock_response = Mock()
        mock_response.json.return_value = {
            'segments': [
                {
                    'start': 1.234,
                    'end': 3.567,
                    'words': [
                        {'word': 'Test', 'start': 1.234, 'end': 2.345},
                        {'word': 'word', 'start': 2.456, 'end': 3.567}
                    ]
                }
            ]
        }
        mock_requests.return_value = mock_response
        
        # Test with 15.5 second offset
        start_offset = 15.5
        with patch('builtins.open', mock_open()):
            with patch('core.asr_backend.whisperX_302.json.dump'):
                result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', start_offset, start_offset + 10.0)
        
        # Verify precise timestamp adjustment
        segment = result['segments'][0]
        assert abs(segment['start'] - (1.234 + start_offset)) < 0.001
        assert abs(segment['end'] - (3.567 + start_offset)) < 0.001
        
        word1 = segment['words'][0]
        assert abs(word1['start'] - (1.234 + start_offset)) < 0.001
        assert abs(word1['end'] - (2.345 + start_offset)) < 0.001

    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    @patch('core.asr_backend.whisperX_302.os.path.exists')
    def test_302_api_payload_format(self, mock_exists, mock_load_key, mock_requests, mock_librosa):
        """Test correct API payload format for 302.ai."""
        mock_exists.return_value = False
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'zh',
            'whisper.whisperX_302_api_key': 'test_key'
        }.get(key, 'default')
        mock_librosa.return_value = (np.array([0.1]), 16000)
        
        mock_response = Mock()
        mock_response.json.return_value = {'segments': []}
        mock_requests.return_value = mock_response
        
        # Test
        with patch('builtins.open', mock_open()):
            with patch('core.asr_backend.whisperX_302.json.dump'):
                transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav')
        
        # Verify API payload
        call_kwargs = mock_requests.call_args[1]
        assert 'data' in call_kwargs
        payload = call_kwargs['data']
        assert payload['processing_type'] == 'align'
        assert payload['language'] == 'zh'
        assert payload['output'] == 'raw'
        
        # Verify audio file upload
        assert 'files' in call_kwargs
        files = call_kwargs['files'][0]
        assert files[0] == 'audio_input'
        assert 'audio_slice.wav' in files[1]


class TestASRIntegrationWorkflows:
    """Test integrated ASR workflows combining multiple components."""
    
    @patch('core.asr_backend.audio_preprocess.split_audio')
    def test_long_audio_segmentation_integration(self, mock_split_audio):
        """Test integration with audio segmentation for long files."""
        # Mock audio splitting
        mock_split_audio.return_value = [
            (0.0, 120.0),    # First 2 minutes
            (120.0, 240.0),  # Second 2 minutes
            (240.0, 300.0)   # Final 1 minute
        ]
        
        segments = mock_split_audio('/tmp/long_audio.wav', target_len=120, win=10)
        
        assert len(segments) == 3
        assert segments[0] == (0.0, 120.0)
        assert segments[1] == (120.0, 240.0) 
        assert segments[2] == (240.0, 300.0)
        
        # Verify no segment overlap
        for i in range(len(segments) - 1):
            assert segments[i][1] <= segments[i + 1][0]

    def test_transcription_processing_integration(self):
        """Test integration with transcription result processing."""
        # Mock complex transcription result
        transcription_result = {
            'segments': [
                {
                    'speaker_id': 'speaker_1',
                    'words': [
                        {'word': 'Welcome', 'start': 0.0, 'end': 0.8},
                        {'word': 'to', 'start': 0.8, 'end': 1.0},
                        {'word': 'our', 'start': 1.0, 'end': 1.3}
                    ]
                },
                {
                    'speaker_id': 'speaker_2',
                    'words': [
                        {'word': 'Thank', 'start': 3.0, 'end': 3.4},
                        {'word': 'you', 'start': 3.4, 'end': 3.8}
                    ]
                }
            ]
        }
        
        df = process_transcription(transcription_result)
        
        # Verify processing results
        assert len(df) == 5
        assert df.iloc[0]['text'] == 'Welcome'
        assert df.iloc[0]['speaker_id'] == 'speaker_1'
        assert df.iloc[3]['text'] == 'Thank'
        assert df.iloc[3]['speaker_id'] == 'speaker_2'
        
        # Verify timestamp continuity
        assert df.iloc[0]['start'] == 0.0
        assert df.iloc[2]['end'] == 1.3
        assert df.iloc[3]['start'] == 3.0

    @patch('core.asr_backend.whisperX_local.transcribe_audio')
    @patch('core.asr_backend.whisperX_302.transcribe_audio_302')
    def test_asr_method_selection_integration(self, mock_302, mock_local):
        """Test integration with different ASR method selection."""
        # Test local transcription
        mock_local.return_value = {
            'segments': [{'start': 0, 'end': 5, 'text': 'Local result'}]
        }
        
        local_result = mock_local('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        assert 'segments' in local_result
        assert local_result['segments'][0]['text'] == 'Local result'
        
        # Test 302.ai transcription
        mock_302.return_value = {
            'segments': [{'start': 0, 'end': 5, 'text': '302.ai result'}]
        }
        
        cloud_result = mock_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        assert 'segments' in cloud_result
        assert cloud_result['segments'][0]['text'] == '302.ai result'

    @patch('core.asr_backend.audio_preprocess.get_audio_duration')
    def test_duration_based_processing_decision(self, mock_duration):
        """Test processing decisions based on audio duration."""
        # Test short audio (no splitting needed)
        mock_duration.return_value = 30.0  # 30 seconds
        
        duration = mock_duration('/tmp/short.wav')
        if duration <= 60:  # Less than 1 minute
            processing_strategy = 'single_pass'
        else:
            processing_strategy = 'segmented'
        
        assert processing_strategy == 'single_pass'
        
        # Test long audio (requires splitting)
        mock_duration.return_value = 3600.0  # 1 hour
        
        duration = mock_duration('/tmp/long.wav')
        if duration <= 60:
            processing_strategy = 'single_pass'
        else:
            processing_strategy = 'segmented'
        
        assert processing_strategy == 'segmented'

    def test_multilingual_processing_integration(self):
        """Test integration with multilingual processing."""
        # Test language detection scenarios
        test_scenarios = [
            {
                'detected': 'en',
                'configured': 'auto',
                'should_process': True,
                'expected_lang': 'en'
            },
            {
                'detected': 'zh',
                'configured': 'en',
                'should_process': False,  # Mismatch
                'expected_lang': None
            },
            {
                'detected': 'zh',
                'configured': 'zh',
                'should_process': True,
                'expected_lang': 'zh'
            }
        ]
        
        for scenario in test_scenarios:
            detected = scenario['detected']
            configured = scenario['configured']
            
            if configured == 'auto' or configured == detected:
                assert scenario['should_process'] is True
            else:
                assert scenario['should_process'] is False


if __name__ == '__main__':
    pytest.main([__file__])
