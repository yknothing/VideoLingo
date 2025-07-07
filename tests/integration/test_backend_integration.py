# Integration Tests for Backend Systems
# Tests ASR and TTS backend integrations

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

@pytest.mark.integration
class TestASRBackendIntegration:
    """Integration tests for ASR backend systems"""
    
    def test_whisperx_local_integration(self, mock_torch, mock_whisperx, mock_audio_file):
        """Test WhisperX local backend integration"""
        from core.asr_backend.whisperX_local import transcribe_audio
        
        with patch('core.asr_backend.whisperX_local.load_audio') as mock_load_audio, \
             patch('core.utils.config_utils.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'asr.whisper.model': 'base',
                'asr.whisper.language': 'auto',
                'asr.whisper.device': 'cpu'
            }.get(key, default)
            
            mock_load_audio.return_value = Mock()  # Mock audio array
            
            result = transcribe_audio(mock_audio_file)
            
            assert 'segments' in result
            assert len(result['segments']) > 0
            assert 'text' in result['segments'][0]
    
    def test_whisperx_cloud_integration(self, mock_requests, mock_audio_file):
        """Test WhisperX cloud backend integration"""
        from core.asr_backend.whisperX_302 import transcribe_audio_cloud
        
        # Mock successful API response
        mock_response = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Hello world",
                    "words": [
                        {"start": 0.0, "end": 0.5, "word": "Hello"},
                        {"start": 0.6, "end": 1.0, "word": "world"}
                    ]
                }
            ],
            "language": "en"
        }
        
        mock_requests['post'].return_value.json.return_value = mock_response
        mock_requests['post'].return_value.status_code = 200
        
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'asr.cloud.api_key': 'test-api-key',
                'asr.cloud.endpoint': 'https://api.test.com/transcribe'
            }.get(key, default)
            
            result = transcribe_audio_cloud(mock_audio_file)
            
            assert result == mock_response
            assert mock_requests['post'].called
    
    def test_elevenlabs_asr_integration(self, mock_requests, mock_audio_file):
        """Test ElevenLabs ASR integration"""
        from core.asr_backend.elevenlabs_asr import transcribe_with_elevenlabs
        
        mock_response = {
            "text": "Hello world, this is a test transcription.",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello world"},
                {"start": 2.5, "end": 5.0, "text": "this is a test transcription."}
            ]
        }
        
        mock_requests['post'].return_value.json.return_value = mock_response
        mock_requests['post'].return_value.status_code = 200
        
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'asr.elevenlabs.api_key': 'test-elevenlabs-key'
            }.get(key, default)
            
            result = transcribe_with_elevenlabs(mock_audio_file)
            
            assert 'text' in result
            assert 'segments' in result
            assert len(result['segments']) == 2
    
    def test_demucs_audio_separation(self, mock_subprocess, mock_audio_file, temp_config_dir):
        """Test Demucs audio separation integration"""
        from core.asr_backend.demucs_vl import separate_vocals
        
        # Mock successful demucs process
        mock_subprocess['run'].return_value.returncode = 0
        
        # Create mock separated audio file
        separated_audio = temp_config_dir / 'vocals.wav'
        separated_audio.write_text('mock separated audio')
        
        with patch('core.asr_backend.demucs_vl.find_separated_audio', return_value=str(separated_audio)):
            result = separate_vocals(mock_audio_file)
            
            assert result == str(separated_audio)
            assert mock_subprocess['run'].called
    
    def test_audio_preprocessing_pipeline(self, mock_audio_file, temp_config_dir):
        """Test complete audio preprocessing pipeline"""
        from core.asr_backend.audio_preprocess import preprocess_audio
        
        with patch('core.asr_backend.audio_preprocess.normalize_audio') as mock_normalize, \
             patch('core.asr_backend.audio_preprocess.reduce_noise') as mock_denoise, \
             patch('subprocess.run') as mock_subprocess:
            
            mock_normalize.return_value = mock_audio_file
            mock_denoise.return_value = mock_audio_file
            mock_subprocess.return_value = Mock(returncode=0)
            
            result = preprocess_audio(mock_audio_file)
            
            assert result is not None
            assert mock_normalize.called
            assert mock_denoise.called


@pytest.mark.integration
class TestTTSBackendIntegration:
    """Integration tests for TTS backend systems"""
    
    def test_openai_tts_integration(self, mock_requests, temp_config_dir):
        """Test OpenAI TTS backend integration"""
        from core.tts_backend.openai_tts import generate_speech
        
        # Mock successful TTS response
        mock_audio_content = b'fake_audio_content' * 1000
        mock_requests['post'].return_value.content = mock_audio_content
        mock_requests['post'].return_value.status_code = 200
        
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.openai.api_key': 'test-openai-key',
                'tts.openai.model': 'tts-1',
                'tts.openai.voice': 'alloy'
            }.get(key, default)
            
            output_file = temp_config_dir / 'test_output.mp3'
            
            result = generate_speech("Hello world", str(output_file))
            
            assert result is True
            assert mock_requests['post'].called
    
    def test_azure_tts_integration(self, mock_requests, temp_config_dir):
        """Test Azure TTS backend integration"""
        from core.tts_backend.azure_tts import synthesize_speech
        
        # Mock Azure token and synthesis responses
        mock_requests['post'].side_effect = [
            Mock(status_code=200, text='mock_access_token'),  # Token request
            Mock(status_code=200, content=b'fake_audio_content' * 1000)  # Synthesis request
        ]
        
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.azure.api_key': 'test-azure-key',
                'tts.azure.region': 'eastus',
                'tts.azure.voice': 'en-US-AriaNeural'
            }.get(key, default)
            
            output_file = temp_config_dir / 'test_azure_output.wav'
            
            result = synthesize_speech("Hello world", str(output_file))
            
            assert result is True
            assert mock_requests['post'].call_count == 2
    
    def test_edge_tts_integration(self, mock_subprocess, temp_config_dir):
        """Test Edge TTS backend integration"""
        from core.tts_backend.edge_tts import generate_edge_speech
        
        mock_subprocess['run'].return_value.returncode = 0
        
        output_file = temp_config_dir / 'test_edge_output.mp3'
        
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.edge.voice': 'en-US-AriaNeural',
                'tts.edge.rate': '+0%',
                'tts.edge.pitch': '+0Hz'
            }.get(key, default)
            
            result = generate_edge_speech("Hello world", str(output_file))
            
            assert result is True
            assert mock_subprocess['run'].called
    
    def test_gpt_sovits_integration(self, mock_requests, temp_config_dir):
        """Test GPT-SoVITS backend integration"""
        from core.tts_backend.gpt_sovits_tts import generate_sovits_speech
        
        mock_audio_content = b'fake_sovits_audio' * 1000
        mock_requests['post'].return_value.content = mock_audio_content
        mock_requests['post'].return_value.status_code = 200
        
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.gpt_sovits.api_url': 'http://localhost:9880',
                'tts.gpt_sovits.reference_audio': 'reference.wav',
                'tts.gpt_sovits.reference_text': 'Reference text'
            }.get(key, default)
            
            output_file = temp_config_dir / 'test_sovits_output.wav'
            
            result = generate_sovits_speech("Hello world", str(output_file))
            
            assert result is True
            assert mock_requests['post'].called
    
    def test_fish_tts_integration(self, mock_requests, temp_config_dir):
        """Test Fish TTS backend integration"""
        from core.tts_backend.fish_tts import generate_fish_speech
        
        mock_audio_content = b'fake_fish_audio' * 1000
        mock_requests['post'].return_value.content = mock_audio_content
        mock_requests['post'].return_value.status_code = 200
        
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.fish.api_key': 'test-fish-key',
                'tts.fish.model': 'fish-speech-1',
                'tts.fish.voice': 'default'
            }.get(key, default)
            
            output_file = temp_config_dir / 'test_fish_output.wav'
            
            result = generate_fish_speech("Hello world", str(output_file))
            
            assert result is True
            assert mock_requests['post'].called
    
    def test_tts_main_router_fallback(self, temp_config_dir):
        """Test TTS main router with fallback logic"""
        from core.tts_backend.tts_main import generate_audio_with_fallback
        
        with patch('core.tts_backend.openai_tts.generate_speech', side_effect=Exception("OpenAI failed")), \
             patch('core.tts_backend.azure_tts.synthesize_speech', return_value=True), \
             patch('core.utils.config_utils.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.primary_engine': 'openai',
                'tts.fallback_engines': ['azure', 'edge'],
                'tts.enable_fallback': True
            }.get(key, default)
            
            output_file = temp_config_dir / 'test_fallback_output.wav'
            
            result = generate_audio_with_fallback("Hello world", str(output_file))
            
            # Should succeed using fallback
            assert result is True
    
    def test_duration_estimation_integration(self):
        """Test TTS duration estimation"""
        from core.tts_backend.estimate_duration import estimate_speech_duration
        
        test_cases = [
            ("Hello world", 1.0, 2.0),  # (text, min_duration, max_duration)
            ("This is a longer sentence with more words.", 2.0, 4.0),
            ("Short", 0.5, 1.5)
        ]
        
        for text, min_dur, max_dur in test_cases:
            duration = estimate_speech_duration(text)
            assert min_dur <= duration <= max_dur
            assert isinstance(duration, float)
    
    def test_custom_tts_plugin_integration(self, temp_config_dir):
        """Test custom TTS plugin system"""
        from core.tts_backend.custom_tts import load_custom_tts_plugin, execute_custom_tts
        
        # Create mock custom TTS plugin
        plugin_file = temp_config_dir / 'custom_tts_plugin.py'
        plugin_content = '''
def generate_speech(text, output_file, **kwargs):
    """Custom TTS implementation"""
    with open(output_file, 'wb') as f:
        f.write(f"Custom TTS: {text}".encode())
    return True

def get_plugin_info():
    return {
        "name": "TestTTS",
        "version": "1.0.0",
        "supported_languages": ["en", "zh"]
    }
'''
        plugin_file.write_text(plugin_content)
        
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.custom.plugin_path': str(plugin_file),
                'tts.custom.enabled': True
            }.get(key, default)
            
            # Test plugin loading
            plugin = load_custom_tts_plugin()
            assert plugin is not None
            
            # Test plugin execution
            output_file = temp_config_dir / 'custom_output.wav'
            result = execute_custom_tts(plugin, "Hello custom TTS", str(output_file))
            
            assert result is True
            assert output_file.exists()
            assert "Custom TTS: Hello custom TTS" in output_file.read_text()


@pytest.mark.integration
class TestBackendErrorHandling:
    """Test error handling in backend systems"""
    
    def test_asr_backend_error_recovery(self, mock_audio_file):
        """Test ASR backend error recovery"""
        from core.asr_backend.whisperX_local import transcribe_audio
        
        with patch('whisperx.load_model', side_effect=Exception("Model loading failed")), \
             patch('core.asr_backend.whisperX_302.transcribe_audio_cloud') as mock_cloud_fallback:
            
            mock_cloud_fallback.return_value = {
                "segments": [{"start": 0, "end": 1, "text": "fallback result"}]
            }
            
            # Should fallback to cloud when local fails
            with patch('core.utils.config_utils.load_key') as mock_load_key:
                mock_load_key.side_effect = lambda key, default=None: {
                    'asr.enable_fallback': True,
                    'asr.cloud.api_key': 'test-key'
                }.get(key, default)
                
                result = transcribe_audio(mock_audio_file)
                
                assert 'segments' in result
                assert result['segments'][0]['text'] == "fallback result"
    
    def test_tts_backend_timeout_handling(self, temp_config_dir):
        """Test TTS backend timeout handling"""
        from core.tts_backend.openai_tts import generate_speech
        
        import requests
        
        with patch('requests.post', side_effect=requests.Timeout("Request timeout")), \
             patch('core.utils.config_utils.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.openai.api_key': 'test-key',
                'tts.timeout': 5,
                'tts.max_retries': 2
            }.get(key, default)
            
            output_file = temp_config_dir / 'timeout_test.mp3'
            
            # Should handle timeout gracefully
            result = generate_speech("Hello world", str(output_file))
            assert result is False  # Should fail gracefully
    
    def test_backend_rate_limit_handling(self, temp_config_dir):
        """Test backend rate limit handling"""
        from core.tts_backend.openai_tts import generate_speech
        
        import requests
        
        # Mock rate limit response
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {'Retry-After': '1'}
        
        # Success on retry
        success_response = Mock()
        success_response.status_code = 200
        success_response.content = b'audio_content'
        
        with patch('requests.post', side_effect=[rate_limit_response, success_response]), \
             patch('time.sleep') as mock_sleep, \
             patch('core.utils.config_utils.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.openai.api_key': 'test-key',
                'tts.handle_rate_limits': True
            }.get(key, default)
            
            output_file = temp_config_dir / 'rate_limit_test.mp3'
            
            result = generate_speech("Hello world", str(output_file))
            
            # Should succeed after retry
            assert result is True
            assert mock_sleep.called  # Should have waited
    
    def test_backend_configuration_validation(self):
        """Test backend configuration validation"""
        from core.tts_backend.tts_main import validate_tts_config
        from core.asr_backend.whisperX_local import validate_asr_config
        
        # Test valid TTS config
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.engine': 'openai',
                'tts.openai.api_key': 'test-key',
                'tts.openai.voice': 'alloy'
            }.get(key, default)
            
            is_valid, errors = validate_tts_config()
            assert is_valid is True
            assert len(errors) == 0
        
        # Test invalid TTS config
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'tts.engine': 'openai'
                # Missing API key
            }.get(key, default)
            
            is_valid, errors = validate_tts_config()
            assert is_valid is False
            assert len(errors) > 0
        
        # Test valid ASR config
        with patch('core.utils.config_utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key, default=None: {
                'asr.backend': 'whisperx_local',
                'asr.whisper.model': 'base',
                'asr.whisper.device': 'cpu'
            }.get(key, default)
            
            is_valid, errors = validate_asr_config()
            assert is_valid is True
            assert len(errors) == 0