"""
Comprehensive test suite for WhisperX local API integration.
Tests local model loading, GPU optimization, language handling, and alignment processing.
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

# Test imports for WhisperX local module
try:
    from core.asr_backend.whisperX_local import (
        check_hf_mirror,
        transcribe_audio
    )
except ImportError as e:
    pytest.skip(f"WhisperX local module not available: {e}", allow_module_level=True)


class TestHuggingFaceMirrorCheck:
    """Test HuggingFace mirror checking and optimization."""
    
    @patch('core.asr_backend.whisperX_local.subprocess.run')
    @patch('core.asr_backend.whisperX_local.time.time')
    def test_check_hf_mirror_official_fastest(self, mock_time, mock_subprocess):
        """Test mirror check when official HF is fastest."""
        # Mock time progression: start, official ping, mirror ping
        mock_time.side_effect = [0, 1, 2, 3, 4]
        
        # Mock successful pings with official being faster
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # Official HF ping (1 second)
            Mock(returncode=0)   # Mirror ping (2 seconds)
        ]
        
        result = check_hf_mirror()
        
        assert result == "https://huggingface.co"
        assert mock_subprocess.call_count == 2
    
    @patch('core.asr_backend.whisperX_local.subprocess.run')
    @patch('core.asr_backend.whisperX_local.time.time')
    def test_check_hf_mirror_mirror_fastest(self, mock_time, mock_subprocess):
        """Test mirror check when mirror is fastest."""
        # Mock time progression with mirror being faster
        mock_time.side_effect = [0, 2, 4, 5, 6]
        
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # Official HF ping (2 seconds)
            Mock(returncode=0)   # Mirror ping (1 second)
        ]
        
        result = check_hf_mirror()
        
        assert result == "https://hf-mirror.com"
        assert mock_subprocess.call_count == 2
    
    @patch('core.asr_backend.whisperX_local.subprocess.run')
    @patch('core.asr_backend.whisperX_local.time.time')
    def test_check_hf_mirror_all_fail(self, mock_time, mock_subprocess):
        """Test mirror check when all mirrors fail."""
        mock_time.side_effect = [0, 1, 2, 3, 4]
        
        # Mock all pings failing
        mock_subprocess.side_effect = [
            Mock(returncode=1),  # Official fails
            Mock(returncode=1)   # Mirror fails
        ]
        
        result = check_hf_mirror()
        
        assert result == "https://huggingface.co"  # Should default to official
        assert mock_subprocess.call_count == 2
    
    @patch('core.asr_backend.whisperX_local.subprocess.run')
    @patch('core.asr_backend.whisperX_local.os.name', 'nt')
    def test_check_hf_mirror_windows_ping_command(self, mock_subprocess):
        """Test Windows-specific ping command."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('core.asr_backend.whisperX_local.time.time', side_effect=[0, 1, 2, 3]):
            check_hf_mirror()
        
        # Verify Windows ping command was used
        ping_calls = mock_subprocess.call_args_list
        assert any('ping' in call[0][0][0] for call in ping_calls)
        assert any('-n' in call[0][0] for call in ping_calls)  # Windows flag
        assert any('-w' in call[0][0] for call in ping_calls)  # Windows timeout
    
    @patch('core.asr_backend.whisperX_local.subprocess.run')
    @patch('core.asr_backend.whisperX_local.os.name', 'posix')
    def test_check_hf_mirror_linux_ping_command(self, mock_subprocess):
        """Test Linux-specific ping command."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('core.asr_backend.whisperX_local.time.time', side_effect=[0, 1, 2, 3]):
            check_hf_mirror()
        
        # Verify Linux ping command was used
        ping_calls = mock_subprocess.call_args_list
        assert any('ping' in call[0][0][0] for call in ping_calls)
        assert any('-c' in call[0][0] for call in ping_calls)  # Linux flag
        assert any('-W' in call[0][0] for call in ping_calls)  # Linux timeout
    
    @patch('core.asr_backend.whisperX_local.rprint')
    def test_check_hf_mirror_exception_handling(self, mock_rprint):
        """Test exception handling in mirror check."""
        with patch('core.asr_backend.whisperX_local.subprocess.run', side_effect=Exception("Network error")):
            result = check_hf_mirror()
        
        # Should return None and handle exception gracefully
        assert result is None


class TestWhisperXDeviceOptimization:
    """Test WhisperX device detection and optimization."""
    
    @pytest.fixture
    def mock_base_dependencies(self):
        """Mock base dependencies for transcription tests."""
        with patch('core.asr_backend.whisperX_local.check_hf_mirror', return_value='https://huggingface.co'), \
             patch('core.asr_backend.whisperX_local.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_local.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_local.os.path.exists') as mock_exists:
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.model': 'base'
            }.get(key, 'default')
            mock_librosa.return_value = (np.array([0.1, 0.2, 0.3]), 16000)
            mock_exists.return_value = False  # Use remote model
            
            yield {
                'load_key': mock_load_key,
                'librosa': mock_librosa,
                'exists': mock_exists
            }
    
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_available')
    @patch('core.asr_backend.whisperX_local.torch.cuda.get_device_properties')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_bf16_supported')
    def test_gpu_optimization_high_memory(self, mock_bf16, mock_device_props, mock_cuda, mock_base_dependencies):
        """Test GPU optimization with high memory GPU."""
        mock_cuda.return_value = True
        mock_bf16.return_value = True
        
        mock_device = Mock()
        mock_device.total_memory = 12 * 1024**3  # 12GB
        mock_device_props.return_value = mock_device
        
        with patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model, \
             patch('core.asr_backend.whisperX_local.whisperx.load_align_model') as mock_load_align, \
             patch('core.asr_backend.whisperX_local.whisperx.align') as mock_align, \
             patch('core.asr_backend.whisperX_local.update_key'):
            
            # Setup model mocks
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'language': 'en',
                'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
            }
            mock_load_model.return_value = mock_model
            
            mock_align_model = Mock()
            mock_metadata = Mock()
            mock_load_align.return_value = (mock_align_model, mock_metadata)
            mock_align.return_value = {'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]}
            
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Verify GPU optimizations were applied
            mock_load_model.assert_called_once()
            call_kwargs = mock_load_model.call_args[1]
            assert call_kwargs['device'] == 'cuda'
            assert call_kwargs['compute_type'] == 'float16'  # bf16 supported
    
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_available')
    @patch('core.asr_backend.whisperX_local.torch.cuda.get_device_properties')
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_bf16_supported')
    def test_gpu_optimization_low_memory(self, mock_bf16, mock_device_props, mock_cuda, mock_base_dependencies):
        """Test GPU optimization with low memory GPU."""
        mock_cuda.return_value = True
        mock_bf16.return_value = False  # No bf16 support
        
        mock_device = Mock()
        mock_device.total_memory = 4 * 1024**3  # 4GB (low memory)
        mock_device_props.return_value = mock_device
        
        with patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model, \
             patch('core.asr_backend.whisperX_local.whisperx.load_align_model') as mock_load_align, \
             patch('core.asr_backend.whisperX_local.whisperx.align') as mock_align, \
             patch('core.asr_backend.whisperX_local.update_key'):
            
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'language': 'en',
                'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
            }
            mock_load_model.return_value = mock_model
            
            mock_align_model = Mock()
            mock_metadata = Mock()
            mock_load_align.return_value = (mock_align_model, mock_metadata)
            mock_align.return_value = {'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]}
            
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Verify low-memory optimizations
            call_kwargs = mock_load_model.call_args[1]
            assert call_kwargs['device'] == 'cuda'
            assert call_kwargs['compute_type'] == 'int8'  # Lower precision for low memory
    
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_available')
    def test_cpu_fallback(self, mock_cuda, mock_base_dependencies):
        """Test CPU fallback when CUDA is not available."""
        mock_cuda.return_value = False
        
        with patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model, \
             patch('core.asr_backend.whisperX_local.whisperx.load_align_model') as mock_load_align, \
             patch('core.asr_backend.whisperX_local.whisperx.align') as mock_align, \
             patch('core.asr_backend.whisperX_local.update_key'):
            
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'language': 'en',
                'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
            }
            mock_load_model.return_value = mock_model
            
            mock_align_model = Mock()
            mock_metadata = Mock()
            mock_load_align.return_value = (mock_align_model, mock_metadata)
            mock_align.return_value = {'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]}
            
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Verify CPU settings
            call_kwargs = mock_load_model.call_args[1]
            assert call_kwargs['device'] == 'cpu'
            assert call_kwargs['compute_type'] == 'int8'


class TestWhisperXLanguageHandling:
    """Test language detection and handling in WhisperX."""
    
    @pytest.fixture
    def mock_transcription_dependencies(self):
        """Mock dependencies for language testing."""
        with patch('core.asr_backend.whisperX_local.check_hf_mirror', return_value='https://huggingface.co'), \
             patch('core.asr_backend.whisperX_local.torch.cuda.is_available', return_value=False), \
             patch('core.asr_backend.whisperX_local.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model, \
             patch('core.asr_backend.whisperX_local.whisperx.load_align_model') as mock_load_align, \
             patch('core.asr_backend.whisperX_local.whisperx.align') as mock_align:
            
            mock_librosa.return_value = (np.array([0.1, 0.2]), 16000)
            
            # Setup align model mock
            mock_align_model = Mock()
            mock_metadata = Mock()
            mock_load_align.return_value = (mock_align_model, mock_metadata)
            mock_align.return_value = {'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]}
            
            yield {
                'librosa': mock_librosa,
                'load_model': mock_load_model,
                'load_align': mock_load_align,
                'align': mock_align
            }
    
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.update_key')
    def test_chinese_language_detection_success(self, mock_update_key, mock_load_key, mock_transcription_dependencies):
        """Test successful Chinese language detection with matching config."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'zh',  # Config set to Chinese
            'whisper.model': 'base'
        }.get(key, 'default')
        
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'language': 'zh',  # Detected as Chinese (matches config)
            'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
        }
        mock_transcription_dependencies['load_model'].return_value = mock_model
        
        result = transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        assert result is not None
        mock_update_key.assert_called_with('whisper.language', 'zh')
    
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.update_key')
    def test_chinese_language_detection_mismatch(self, mock_update_key, mock_load_key, mock_transcription_dependencies):
        """Test Chinese language detection mismatch error."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',  # Config set to English
            'whisper.model': 'base'
        }.get(key, 'default')
        
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'language': 'zh',  # But detected as Chinese
            'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
        }
        mock_transcription_dependencies['load_model'].return_value = mock_model
        
        with pytest.raises(ValueError, match="Please specify the transcription language as zh and try again!"):
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
    
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.update_key')
    def test_auto_language_detection(self, mock_update_key, mock_load_key, mock_transcription_dependencies):
        """Test automatic language detection."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'auto',  # Auto detection
            'whisper.model': 'base'
        }.get(key, 'default')
        
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'language': 'fr',  # Detected as French
            'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
        }
        mock_transcription_dependencies['load_model'].return_value = mock_model
        
        result = transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        assert result is not None
        mock_update_key.assert_called_with('whisper.language', 'fr')
    
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.os.path.exists')
    def test_chinese_model_selection(self, mock_exists, mock_load_key, mock_transcription_dependencies):
        """Test special Chinese model selection."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'zh',
            'model_dir': '/tmp/models'
        }.get(key, 'default')
        mock_exists.return_value = True  # Local Chinese model exists
        
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'language': 'zh',
            'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
        }
        mock_transcription_dependencies['load_model'].return_value = mock_model
        
        transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        # Verify Chinese-specific model was used
        call_args = mock_transcription_dependencies['load_model'].call_args[0]
        assert 'Belle-whisper' in call_args[0]


class TestWhisperXModelManagement:
    """Test model loading, caching, and management."""
    
    @pytest.fixture
    def mock_model_dependencies(self):
        """Mock dependencies for model testing."""
        with patch('core.asr_backend.whisperX_local.check_hf_mirror', return_value='https://huggingface.co'), \
             patch('core.asr_backend.whisperX_local.torch.cuda.is_available', return_value=False), \
             patch('core.asr_backend.whisperX_local.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_local.whisperx.load_align_model') as mock_load_align, \
             patch('core.asr_backend.whisperX_local.whisperx.align') as mock_align, \
             patch('core.asr_backend.whisperX_local.update_key'):
            
            mock_librosa.return_value = (np.array([0.1, 0.2]), 16000)
            
            mock_align_model = Mock()
            mock_metadata = Mock()
            mock_load_align.return_value = (mock_align_model, mock_metadata)
            mock_align.return_value = {'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]}
            
            yield
    
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.os.path.exists')
    def test_local_model_usage(self, mock_exists, mock_load_key, mock_model_dependencies):
        """Test usage of locally cached model."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.model': 'large-v2',
            'model_dir': '/tmp/models'
        }.get(key, 'default')
        mock_exists.return_value = True  # Local model exists
        
        with patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'language': 'en',
                'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
            }
            mock_load_model.return_value = mock_model
            
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Should use local model path
            call_args = mock_load_model.call_args[0]
            assert '/tmp/models/large-v2' in call_args[0]
    
    @patch('core.asr_backend.whisperX_local.load_key')
    @patch('core.asr_backend.whisperX_local.os.path.exists')
    def test_remote_model_download(self, mock_exists, mock_load_key, mock_model_dependencies):
        """Test downloading remote model when local doesn't exist."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.model': 'large-v2',
            'model_dir': '/tmp/models'
        }.get(key, 'default')
        mock_exists.return_value = False  # Local model doesn't exist
        
        with patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model:
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'language': 'en',
                'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
            }
            mock_load_model.return_value = mock_model
            
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Should use remote model name
            call_args = mock_load_model.call_args[0]
            assert call_args[0] == 'large-v2'
            
            # Should specify download root
            call_kwargs = mock_load_model.call_args[1]
            assert 'download_root' in call_kwargs
            assert call_kwargs['download_root'] == '/tmp/models'
    
    @patch('core.asr_backend.whisperX_local.torch.cuda.empty_cache')
    def test_gpu_memory_cleanup(self, mock_empty_cache, mock_model_dependencies):
        """Test GPU memory cleanup after processing."""
        with patch('core.asr_backend.whisperX_local.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model:
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.model': 'base'
            }.get(key, 'default')
            
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'language': 'en',
                'segments': [{'start': 0.0, 'end': 1.0, 'words': []}]
            }
            mock_load_model.return_value = mock_model
            
            transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Verify GPU memory was cleared
            assert mock_empty_cache.call_count >= 2  # After transcription and alignment


class TestWhisperXTimestampAlignment:
    """Test timestamp alignment and adjustment functionality."""
    
    @pytest.fixture
    def mock_alignment_dependencies(self):
        """Mock dependencies for alignment testing."""
        with patch('core.asr_backend.whisperX_local.check_hf_mirror', return_value='https://huggingface.co'), \
             patch('core.asr_backend.whisperX_local.torch.cuda.is_available', return_value=False), \
             patch('core.asr_backend.whisperX_local.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model, \
             patch('core.asr_backend.whisperX_local.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_local.update_key'):
            
            mock_librosa.return_value = (np.array([0.1, 0.2]), 16000)
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.model': 'base'
            }.get(key, 'default')
            
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'language': 'en',
                'segments': [
                    {
                        'start': 0.0,
                        'end': 2.0,
                        'words': [
                            {'word': 'Hello', 'start': 0.0, 'end': 1.0},
                            {'word': 'World', 'start': 1.0, 'end': 2.0}
                        ]
                    }
                ]
            }
            mock_load_model.return_value = mock_model
            
            yield
    
    @patch('core.asr_backend.whisperX_local.whisperx.load_align_model')
    @patch('core.asr_backend.whisperX_local.whisperx.align')
    def test_timestamp_alignment_process(self, mock_align, mock_load_align, mock_alignment_dependencies):
        """Test timestamp alignment with vocal audio."""
        mock_align_model = Mock()
        mock_metadata = Mock()
        mock_load_align.return_value = (mock_align_model, mock_metadata)
        
        # Mock aligned result
        mock_align.return_value = {
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'words': [
                        {'word': 'Hello', 'start': 0.5, 'end': 1.2},
                        {'word': 'World', 'start': 1.3, 'end': 1.8}
                    ]
                }
            ]
        }
        
        result = transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 5.0, 15.0)
        
        # Verify alignment was performed
        mock_load_align.assert_called_once()
        mock_align.assert_called_once()
        
        # Check alignment parameters
        align_call_args = mock_align.call_args[0]
        assert align_call_args[1] == mock_align_model
        assert align_call_args[2] == mock_metadata
    
    def test_timestamp_adjustment_for_segments(self, mock_alignment_dependencies):
        """Test timestamp adjustment for audio segments."""
        with patch('core.asr_backend.whisperX_local.whisperx.load_align_model') as mock_load_align, \
             patch('core.asr_backend.whisperX_local.whisperx.align') as mock_align:
            
            mock_align_model = Mock()
            mock_metadata = Mock()
            mock_load_align.return_value = (mock_align_model, mock_metadata)
            
            # Mock aligned result with relative timestamps
            mock_align.return_value = {
                'segments': [
                    {
                        'start': 0.0,
                        'end': 2.0,
                        'words': [
                            {'word': 'Hello', 'start': 0.5, 'end': 1.2},
                            {'word': 'World', 'start': 1.3, 'end': 1.8}
                        ]
                    }
                ]
            }
            
            # Process segment starting at 10.0 seconds
            result = transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 10.0, 20.0)
            
            # Verify timestamps were adjusted by start offset
            segment = result['segments'][0]
            assert segment['start'] == 10.0  # 0.0 + 10.0
            assert segment['end'] == 12.0    # 2.0 + 10.0
            
            word1 = segment['words'][0]
            assert word1['start'] == 10.5   # 0.5 + 10.0
            assert word1['end'] == 11.2     # 1.2 + 10.0


class TestWhisperXErrorHandling:
    """Test error handling in WhisperX operations."""
    
    @patch('core.asr_backend.whisperX_local.check_hf_mirror')
    def test_mirror_check_failure_handling(self, mock_check_mirror):
        """Test handling of mirror check failure."""
        mock_check_mirror.return_value = None  # Mirror check failed
        
        with patch('core.asr_backend.whisperX_local.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_local.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model:
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.model': 'base'
            }.get(key, 'default')
            mock_librosa.return_value = (np.array([0.1]), 16000)
            
            # Should still proceed with transcription despite mirror check failure
            mock_model = Mock()
            mock_model.transcribe.side_effect = Exception("Test exception")
            mock_load_model.return_value = mock_model
            
            with pytest.raises(Exception):
                transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
    
    @patch('core.asr_backend.whisperX_local.whisperx.load_model')
    def test_model_loading_failure(self, mock_load_model):
        """Test handling of model loading failure."""
        mock_load_model.side_effect = Exception("Failed to load model")
        
        with patch('core.asr_backend.whisperX_local.check_hf_mirror', return_value='https://huggingface.co'), \
             patch('core.asr_backend.whisperX_local.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_local.librosa.load') as mock_librosa:
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.model': 'base'
            }.get(key, 'default')
            mock_librosa.return_value = (np.array([0.1]), 16000)
            
            with pytest.raises(Exception, match="Failed to load model"):
                transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
    
    @patch('core.asr_backend.whisperX_local.librosa.load')
    def test_audio_loading_failure(self, mock_librosa):
        """Test handling of audio loading failure."""
        mock_librosa.side_effect = Exception("Failed to load audio")
        
        with patch('core.asr_backend.whisperX_local.check_hf_mirror', return_value='https://huggingface.co'), \
             patch('core.asr_backend.whisperX_local.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.model': 'base'
            }.get(key, 'default')
            
            with pytest.raises(Exception, match="Failed to load audio"):
                transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
    
    def test_transcription_failure_cleanup(self):
        """Test resource cleanup when transcription fails."""
        with patch('core.asr_backend.whisperX_local.check_hf_mirror', return_value='https://huggingface.co'), \
             patch('core.asr_backend.whisperX_local.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_local.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model, \
             patch('core.asr_backend.whisperX_local.torch.cuda.empty_cache') as mock_empty_cache:
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.model': 'base'
            }.get(key, 'default')
            mock_librosa.return_value = (np.array([0.1]), 16000)
            
            mock_model = Mock()
            mock_model.transcribe.side_effect = Exception("Transcription failed")
            mock_load_model.return_value = mock_model
            
            with pytest.raises(Exception, match="Transcription failed"):
                transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Verify cleanup was attempted
            mock_empty_cache.assert_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
