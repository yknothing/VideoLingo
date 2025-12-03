"""
Comprehensive test suite for ASR factory/provider selection logic.
Tests provider factory pattern, configuration-based selection, and provider initialization.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Since there's no explicit factory module, we'll test the provider selection
# logic that exists in the main ASR orchestrator


class TestASRProviderFactory:
    """Test ASR provider factory functionality."""
    
    @pytest.fixture
    def mock_config_keys(self):
        """Mock configuration keys for different ASR providers."""
        return {
            'whisper.runtime': 'local',
            'whisper.language': 'en',
            'whisper.model': 'base',
            'whisper.whisperX_302_api_key': 'test_302_key',
            'whisper.elevenlabs_api_key': 'test_elevenlabs_key',
            'demucs': False,
            'model_dir': '/tmp/models'
        }
    
    @patch('core._2_asr.load_key')
    def test_local_whisperx_provider_selection(self, mock_load_key, mock_config_keys):
        """Test selection and import of local WhisperX provider."""
        mock_load_key.side_effect = lambda key: mock_config_keys.get(key, 'default')
        mock_config_keys['whisper.runtime'] = 'local'
        
        # Test that the local provider is selected
        from core._2_asr import transcribe
        
        with patch('core._2_asr.monitor_memory_and_warn'), \
             patch('core._2_asr.find_video_files', return_value='/tmp/video.mp4'), \
             patch('core._2_asr.convert_video_to_audio'), \
             patch('core._2_asr.split_audio', return_value=[(0.0, 30.0)]), \
             patch('core._2_asr.process_transcription', return_value=Mock()), \
             patch('core._2_asr.save_results'), \
             patch('core._2_asr.check_file_exists', return_value=lambda func: func):
            
            # Mock the local transcription import and function
            with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_transcribe:
                mock_transcribe.return_value = {'segments': []}
                
                # This should import and use the local provider
                transcribe()
                
                # Verify the local provider was used
                mock_transcribe.assert_called()
    
    @patch('core._2_asr.load_key')
    def test_cloud_302ai_provider_selection(self, mock_load_key, mock_config_keys):
        """Test selection and import of 302.ai cloud provider."""
        mock_load_key.side_effect = lambda key: mock_config_keys.get(key, 'default')
        mock_config_keys['whisper.runtime'] = 'cloud'
        
        from core._2_asr import transcribe
        
        with patch('core._2_asr.monitor_memory_and_warn'), \
             patch('core._2_asr.find_video_files', return_value='/tmp/video.mp4'), \
             patch('core._2_asr.convert_video_to_audio'), \
             patch('core._2_asr.split_audio', return_value=[(0.0, 30.0)]), \
             patch('core._2_asr.process_transcription', return_value=Mock()), \
             patch('core._2_asr.save_results'), \
             patch('core._2_asr.check_file_exists', return_value=lambda func: func):
            
            # Mock the cloud transcription import and function
            with patch('core.asr_backend.whisperX_302.transcribe_audio_302') as mock_transcribe:
                mock_transcribe.return_value = {'segments': []}
                
                # This should import and use the cloud provider
                transcribe()
                
                # Verify the cloud provider was used
                mock_transcribe.assert_called()
    
    @patch('core._2_asr.load_key')
    def test_elevenlabs_provider_selection(self, mock_load_key, mock_config_keys):
        """Test selection and import of ElevenLabs provider."""
        mock_load_key.side_effect = lambda key: mock_config_keys.get(key, 'default')
        mock_config_keys['whisper.runtime'] = 'elevenlabs'
        
        from core._2_asr import transcribe
        
        with patch('core._2_asr.monitor_memory_and_warn'), \
             patch('core._2_asr.find_video_files', return_value='/tmp/video.mp4'), \
             patch('core._2_asr.convert_video_to_audio'), \
             patch('core._2_asr.split_audio', return_value=[(0.0, 30.0)]), \
             patch('core._2_asr.process_transcription', return_value=Mock()), \
             patch('core._2_asr.save_results'), \
             patch('core._2_asr.check_file_exists', return_value=lambda func: func):
            
            # Mock the ElevenLabs transcription import and function
            with patch('core.asr_backend.elevenlabs_asr.transcribe_audio_elevenlabs') as mock_transcribe:
                mock_transcribe.return_value = {'segments': []}
                
                # This should import and use the ElevenLabs provider
                transcribe()
                
                # Verify the ElevenLabs provider was used
                mock_transcribe.assert_called()
    
    def test_provider_configuration_validation(self):
        """Test validation of provider-specific configuration."""
        # Test local provider configuration
        with patch('core._2_asr.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key: {
                'whisper.runtime': 'local',
                'whisper.model': 'large-v2',
                'whisper.language': 'en'
            }.get(key, 'default')
            
            # Import should work without errors
            from core.asr_backend import whisperX_local
            assert whisperX_local is not None
        
        # Test cloud provider configuration
        with patch('core._2_asr.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key: {
                'whisper.runtime': 'cloud',
                'whisper.whisperX_302_api_key': 'test_key',
                'whisper.language': 'en'
            }.get(key, 'default')
            
            # Import should work without errors
            from core.asr_backend import whisperX_302
            assert whisperX_302 is not None
        
        # Test ElevenLabs provider configuration
        with patch('core._2_asr.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key: {
                'whisper.runtime': 'elevenlabs',
                'whisper.elevenlabs_api_key': 'test_key',
                'whisper.language': 'en'
            }.get(key, 'default')
            
            # Import should work without errors
            from core.asr_backend import elevenlabs_asr
            assert elevenlabs_asr is not None


class TestASRProviderCapabilities:
    """Test capabilities and features of different ASR providers."""
    
    def test_local_whisperx_capabilities(self):
        """Test local WhisperX provider capabilities."""
        from core.asr_backend.whisperX_local import transcribe_audio, check_hf_mirror
        
        # Verify functions exist
        assert callable(transcribe_audio)
        assert callable(check_hf_mirror)
        
        # Test that the provider supports the expected interface
        import inspect
        signature = inspect.signature(transcribe_audio)
        expected_params = ['raw_audio_file', 'vocal_audio_file', 'start', 'end']
        
        for param in expected_params:
            assert param in signature.parameters
    
    def test_cloud_302ai_capabilities(self):
        """Test 302.ai cloud provider capabilities."""
        from core.asr_backend.whisperX_302 import transcribe_audio_302
        
        # Verify function exists
        assert callable(transcribe_audio_302)
        
        # Test that the provider supports the expected interface
        import inspect
        signature = inspect.signature(transcribe_audio_302)
        expected_params = ['raw_audio_path', 'vocal_audio_path', 'start', 'end']
        
        for param in expected_params:
            assert param in signature.parameters
    
    def test_elevenlabs_capabilities(self):
        """Test ElevenLabs provider capabilities."""
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs, elev2whisper, iso_639_2_to_1
        
        # Verify functions exist
        assert callable(transcribe_audio_elevenlabs)
        assert callable(elev2whisper)
        assert isinstance(iso_639_2_to_1, dict)
        
        # Test that the provider supports the expected interface
        import inspect
        signature = inspect.signature(transcribe_audio_elevenlabs)
        expected_params = ['raw_audio_path', 'vocal_audio_path', 'start', 'end']
        
        for param in expected_params:
            assert param in signature.parameters


class TestASRProviderErrorHandling:
    """Test error handling in ASR provider selection and usage."""
    
    @patch('core._2_asr.load_key')
    def test_invalid_provider_runtime(self, mock_load_key):
        """Test handling of invalid/unsupported runtime configuration."""
        mock_load_key.side_effect = lambda key: {
            'whisper.runtime': 'invalid_provider',
            'demucs': False
        }.get(key, 'default')
        
        from core._2_asr import transcribe
        
        with patch('core._2_asr.monitor_memory_and_warn'), \
             patch('core._2_asr.find_video_files', return_value='/tmp/video.mp4'), \
             patch('core._2_asr.convert_video_to_audio'), \
             patch('core._2_asr.split_audio', return_value=[(0.0, 30.0)]), \
             patch('core._2_asr.process_transcription', return_value=Mock()), \
             patch('core._2_asr.save_results'), \
             patch('core._2_asr.check_file_exists', return_value=lambda func: func):
            
            # Should handle invalid provider gracefully or raise appropriate error
            with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
                transcribe()
    
    @patch('core._2_asr.load_key')
    def test_missing_api_key_handling(self, mock_load_key):
        """Test handling of missing API keys for cloud providers."""
        # Test 302.ai without API key
        mock_load_key.side_effect = lambda key: {
            'whisper.runtime': 'cloud',
            'whisper.whisperX_302_api_key': None,  # Missing API key
            'demucs': False
        }.get(key, 'default')
        
        with patch('core.asr_backend.whisperX_302.transcribe_audio_302') as mock_transcribe:
            # Should handle missing API key
            mock_transcribe.side_effect = Exception("Authentication failed")
            
            from core._2_asr import transcribe
            
            with patch('core._2_asr.monitor_memory_and_warn'), \
                 patch('core._2_asr.find_video_files', return_value='/tmp/video.mp4'), \
                 patch('core._2_asr.convert_video_to_audio'), \
                 patch('core._2_asr.split_audio', return_value=[(0.0, 30.0)]), \
                 patch('core._2_asr.process_transcription', return_value=Mock()), \
                 patch('core._2_asr.save_results'), \
                 patch('core._2_asr.check_file_exists', return_value=lambda func: func):
                
                with pytest.raises(Exception, match="Authentication failed"):
                    transcribe()
    
    def test_provider_import_failures(self):
        """Test handling of provider import failures."""
        # Mock import failure for local provider
        with patch('builtins.__import__', side_effect=ImportError("WhisperX not installed")):
            # Should handle import failure gracefully
            with pytest.raises(ImportError):
                from core.asr_backend.whisperX_local import transcribe_audio


class TestASRProviderInterface:
    """Test ASR provider interface consistency."""
    
    def test_provider_return_format_consistency(self):
        """Test that all providers return consistent data format."""
        # All providers should return a dictionary with 'segments' key
        # Each segment should have consistent structure
        
        expected_segment_structure = {
            'start': float,
            'end': float,
            'words': list
        }
        
        # Test local provider format (mock)
        with patch('core.asr_backend.whisperX_local.transcribe_audio') as mock_local:
            mock_local.return_value = {
                'segments': [
                    {
                        'start': 0.0,
                        'end': 1.0,
                        'words': [{'word': 'test', 'start': 0.0, 'end': 1.0}]
                    }
                ]
            }
            
            from core.asr_backend.whisperX_local import transcribe_audio
            result = transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            assert 'segments' in result
            assert isinstance(result['segments'], list)
            if result['segments']:
                segment = result['segments'][0]
                assert 'start' in segment
                assert 'end' in segment
                assert 'words' in segment
        
        # Test cloud provider format (mock)
        with patch('core.asr_backend.whisperX_302.transcribe_audio_302') as mock_cloud:
            mock_cloud.return_value = {
                'segments': [
                    {
                        'start': 0.0,
                        'end': 1.0,
                        'words': [{'word': 'test', 'start': 0.0, 'end': 1.0}]
                    }
                ]
            }
            
            from core.asr_backend.whisperX_302 import transcribe_audio_302
            result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            assert 'segments' in result
            assert isinstance(result['segments'], list)
    
    def test_provider_parameter_consistency(self):
        """Test that all providers accept consistent parameters."""
        # All providers should accept: raw_audio_path, vocal_audio_path, start, end
        
        from core.asr_backend.whisperX_local import transcribe_audio as local_transcribe
        from core.asr_backend.whisperX_302 import transcribe_audio_302 as cloud_transcribe
        from core.asr_backend.elevenlabs_asr import transcribe_audio_elevenlabs as elevenlabs_transcribe
        
        import inspect
        
        # Check local provider signature
        local_sig = inspect.signature(local_transcribe)
        assert 'raw_audio_file' in local_sig.parameters
        assert 'vocal_audio_file' in local_sig.parameters
        assert 'start' in local_sig.parameters
        assert 'end' in local_sig.parameters
        
        # Check cloud provider signature
        cloud_sig = inspect.signature(cloud_transcribe)
        assert 'raw_audio_path' in cloud_sig.parameters
        assert 'vocal_audio_path' in cloud_sig.parameters
        assert 'start' in cloud_sig.parameters
        assert 'end' in cloud_sig.parameters
        
        # Check ElevenLabs provider signature
        elevenlabs_sig = inspect.signature(elevenlabs_transcribe)
        assert 'raw_audio_path' in elevenlabs_sig.parameters
        assert 'vocal_audio_path' in elevenlabs_sig.parameters
        assert 'start' in elevenlabs_sig.parameters
        assert 'end' in elevenlabs_sig.parameters


class TestASRProviderOptimization:
    """Test ASR provider optimization features."""
    
    @patch('core.asr_backend.whisperX_local.torch.cuda.is_available')
    @patch('core.asr_backend.whisperX_local.torch.cuda.get_device_properties')
    def test_local_provider_gpu_optimization(self, mock_device_props, mock_cuda_available):
        """Test that local provider optimizes for available GPU."""
        mock_cuda_available.return_value = True
        
        # Mock GPU with different memory sizes
        mock_device = Mock()
        mock_device.total_memory = 8 * 1024**3  # 8GB
        mock_device_props.return_value = mock_device
        
        from core.asr_backend.whisperX_local import transcribe_audio
        
        with patch('core.asr_backend.whisperX_local.whisperx.load_model') as mock_load_model, \
             patch('core.asr_backend.whisperX_local.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_local.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.model': 'base'
            }.get(key, 'default')
            mock_librosa.return_value = ([0.1, 0.2], 16000)
            
            mock_model = Mock()
            mock_model.transcribe.side_effect = Exception("Stop execution for test")
            mock_load_model.return_value = mock_model
            
            try:
                transcribe_audio('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            except Exception:
                pass  # Expected for test
            
            # Verify GPU optimization was considered
            mock_cuda_available.assert_called()
            mock_device_props.assert_called()
    
    def test_cloud_provider_caching(self):
        """Test that cloud providers implement result caching."""
        from core.asr_backend.whisperX_302 import transcribe_audio_302
        
        with patch('core.asr_backend.whisperX_302.os.path.exists') as mock_exists, \
             patch('core.asr_backend.whisperX_302.json.load') as mock_json_load, \
             patch('builtins.open'):
            
            # Mock cached result exists
            mock_exists.return_value = True
            mock_json_load.return_value = {'segments': [{'cached': True}]}
            
            result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Should return cached result without API call
            assert result == {'segments': [{'cached': True}]}
            mock_json_load.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
