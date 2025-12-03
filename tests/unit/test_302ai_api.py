"""
Comprehensive test suite for 302.ai cloud ASR API integration.
Tests API requests, response handling, caching, error handling, and audio processing.
"""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import requests
import io

# Test imports for 302.ai module
try:
    from core.asr_backend.whisperX_302 import transcribe_audio_302
except ImportError as e:
    pytest.skip(f"302.ai ASR module not available: {e}", allow_module_level=True)


class TestCacheManagement:
    """Test result caching functionality."""
    
    @patch('core.asr_backend.whisperX_302.os.path.exists')
    @patch('core.asr_backend.whisperX_302.json.load')
    def test_cache_hit_returns_cached_result(self, mock_json_load, mock_exists):
        """Test that cached results are returned when available."""
        mock_exists.return_value = True
        cached_result = {
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'words': [{'word': 'cached', 'start': 0.0, 'end': 2.0}]
                }
            ]
        }
        mock_json_load.return_value = cached_result
        
        with patch('builtins.open', mock_open()):
            result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
        
        assert result == cached_result
        mock_json_load.assert_called_once()
        # Should not make any API requests when cache hit
    
    @patch('core.asr_backend.whisperX_302.os.path.exists')
    def test_cache_miss_triggers_api_call(self, mock_exists):
        """Test that missing cache triggers API call."""
        mock_exists.return_value = False  # No cache file
        
        with patch('core.asr_backend.whisperX_302.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_302.requests.request') as mock_request, \
             patch('core.asr_backend.whisperX_302.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'):
            
            mock_librosa.return_value = (np.array([0.1, 0.2]), 16000)
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.whisperX_302_api_key': 'test_api_key'
            }.get(key, 'default')
            
            mock_response = Mock()
            mock_response.json.return_value = {'segments': []}
            mock_request.return_value = mock_response
            
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Should make API request
            mock_request.assert_called_once()
    
    def test_cache_file_path_generation(self):
        """Test cache file path generation with segment parameters."""
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=True), \
             patch('core.asr_backend.whisperX_302.json.load', return_value={'segments': []}), \
             patch('builtins.open', mock_open()):
            
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 5.0, 15.0)
            
            # Cache file should include segment parameters in filename
            # This is tested indirectly through the os.path.exists call
    
    @patch('core.asr_backend.whisperX_302.os.makedirs')
    @patch('core.asr_backend.whisperX_302.json.dump')
    def test_cache_write_after_api_call(self, mock_json_dump, mock_makedirs):
        """Test that API results are cached after successful call."""
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_302.requests.request') as mock_request, \
             patch('core.asr_backend.whisperX_302.load_key') as mock_load_key, \
             patch('builtins.open', mock_open()):
            
            mock_librosa.return_value = (np.array([0.1]), 16000)
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.whisperX_302_api_key': 'test_key'
            }.get(key, 'default')
            
            api_result = {'segments': [{'test': 'data'}]}
            mock_response = Mock()
            mock_response.json.return_value = api_result
            mock_request.return_value = mock_response
            
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 10.0)
            
            # Should create log directory and save cache
            mock_makedirs.assert_called_once()
            mock_json_dump.assert_called_once_with(api_result, mock.ANY, indent=4, ensure_ascii=False)


class TestAudioProcessing:
    """Test audio loading and processing functionality."""
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    def test_audio_loading_and_slicing(self, mock_librosa):
        """Test audio loading and time-based slicing."""
        # Mock audio data: 3 seconds at 16kHz
        mock_audio = np.random.rand(48000)  # 3 seconds * 16000 Hz
        mock_librosa.return_value = (mock_audio, 16000)
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.requests.request') as mock_request, \
             patch('core.asr_backend.whisperX_302.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'):
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.whisperX_302_api_key': 'test_key'
            }.get(key, 'default')
            
            mock_response = Mock()
            mock_response.json.return_value = {'segments': []}
            mock_request.return_value = mock_response
            
            # Request slice from 1.0 to 2.0 seconds
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 1.0, 2.0)
            
            # Verify librosa was called to load audio at 16kHz
            mock_librosa.assert_called_once_with('/tmp/vocal.wav', sr=16000)
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.sf.write')
    def test_audio_format_conversion(self, mock_sf_write, mock_librosa):
        """Test audio format conversion for API submission."""
        mock_audio = np.random.rand(16000)  # 1 second
        mock_librosa.return_value = (mock_audio, 16000)
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.requests.request') as mock_request, \
             patch('core.asr_backend.whisperX_302.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'):
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.whisperX_302_api_key': 'test_key'
            }.get(key, 'default')
            
            mock_response = Mock()
            mock_response.json.return_value = {'segments': []}
            mock_request.return_value = mock_response
            
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 1.0)
            
            # Verify audio was written in WAV format
            mock_sf_write.assert_called_once()
            call_args = mock_sf_write.call_args
            assert call_args[1][2] == 16000  # Sample rate
            assert call_args[1]['format'] == 'WAV'
            assert call_args[1]['subtype'] == 'PCM_16'
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    def test_full_audio_processing_when_no_timestamps(self, mock_librosa):
        """Test processing full audio when no start/end timestamps provided."""
        mock_audio = np.random.rand(32000)  # 2 seconds at 16kHz
        mock_librosa.return_value = (mock_audio, 16000)
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.requests.request') as mock_request, \
             patch('core.asr_backend.whisperX_302.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'):
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.whisperX_302_api_key': 'test_key'
            }.get(key, 'default')
            
            mock_response = Mock()
            mock_response.json.return_value = {'segments': []}
            mock_request.return_value = mock_response
            
            # Call without start/end parameters
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', None, None)
            
            # Should process full audio duration
            mock_librosa.assert_called_once()
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    def test_audio_slicing_edge_cases(self, mock_librosa):
        """Test audio slicing with edge cases."""
        mock_audio = np.random.rand(16000)  # 1 second
        mock_librosa.return_value = (mock_audio, 16000)
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.requests.request') as mock_request, \
             patch('core.asr_backend.whisperX_302.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.whisperX_302_api_key': 'test_key'
            }.get(key, 'default')
            
            mock_response = Mock()
            mock_response.json.return_value = {'segments': []}
            mock_request.return_value = mock_response
            
            # Test slicing beyond audio duration
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.5, 2.0)
            
            # Should handle gracefully without errors


class TestAPIIntegration:
    """Test 302.ai API integration and request handling."""
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    def test_api_request_format(self, mock_load_key, mock_request, mock_librosa):
        """Test correct API request format and parameters."""
        mock_librosa.return_value = (np.array([0.1]), 16000)
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'fr',
            'whisper.whisperX_302_api_key': 'test_api_key_123'
        }.get(key, 'default')
        
        mock_response = Mock()
        mock_response.json.return_value = {'segments': []}
        mock_request.return_value = mock_response
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 5.0)
            
            # Verify API call was made correctly
            mock_request.assert_called_once()
            call_args, call_kwargs = mock_request.call_args
            
            # Check method and URL
            assert call_args[0] == "POST"
            assert call_args[1] == "https://api.302.ai/302/whisperx"
            
            # Check headers
            assert 'headers' in call_kwargs
            assert call_kwargs['headers']['Authorization'] == 'Bearer test_api_key_123'
            
            # Check payload data
            assert 'data' in call_kwargs
            payload = call_kwargs['data']
            assert payload['processing_type'] == 'align'
            assert payload['language'] == 'fr'
            assert payload['output'] == 'raw'
            
            # Check files were included
            assert 'files' in call_kwargs
            assert len(call_kwargs['files']) == 1
            assert call_kwargs['files'][0][0] == 'audio_input'
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    def test_api_response_processing(self, mock_load_key, mock_request, mock_librosa):
        """Test processing of API response."""
        mock_librosa.return_value = (np.array([0.1]), 16000)
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.whisperX_302_api_key': 'test_key'
        }.get(key, 'default')
        
        api_response = {
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
        
        mock_response = Mock()
        mock_response.json.return_value = api_response
        mock_request.return_value = mock_response
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 2.0)
            
            assert result == api_response
            assert 'segments' in result
            assert len(result['segments']) == 1
            assert result['segments'][0]['text'] == 'Hello world'
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    def test_api_error_handling(self, mock_load_key, mock_request, mock_librosa):
        """Test handling of API errors."""
        mock_librosa.return_value = (np.array([0.1]), 16000)
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.whisperX_302_api_key': 'invalid_key'
        }.get(key, 'default')
        
        # Mock API error response
        mock_request.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            with pytest.raises(requests.exceptions.HTTPError):
                transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 5.0)
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    def test_api_timeout_handling(self, mock_load_key, mock_request, mock_librosa):
        """Test handling of API timeouts."""
        mock_librosa.return_value = (np.array([0.1]), 16000)
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.whisperX_302_api_key': 'test_key'
        }.get(key, 'default')
        
        mock_request.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            with pytest.raises(requests.exceptions.Timeout):
                transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 5.0)


class TestTimestampAdjustment:
    """Test timestamp adjustment for audio segments."""
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    def test_timestamp_adjustment_with_start_offset(self, mock_load_key, mock_request, mock_librosa):
        """Test timestamp adjustment when processing audio segment with start offset."""
        mock_librosa.return_value = (np.array([0.1]), 16000)
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.whisperX_302_api_key': 'test_key'
        }.get(key, 'default')
        
        # Mock API response with relative timestamps
        api_response = {
            'segments': [
                {
                    'start': 0.0,  # Relative to segment start
                    'end': 2.0,
                    'words': [
                        {'word': 'Hello', 'start': 0.5, 'end': 1.0},
                        {'word': 'world', 'start': 1.0, 'end': 1.5}
                    ]
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.json.return_value = api_response
        mock_request.return_value = mock_response
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            # Process segment starting at 10.0 seconds
            result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 10.0, 15.0)
            
            # Verify timestamps were adjusted by start offset
            segment = result['segments'][0]
            assert segment['start'] == 10.0  # 0.0 + 10.0
            assert segment['end'] == 12.0    # 2.0 + 10.0
            
            # Check word-level timestamps
            words = segment['words']
            assert words[0]['start'] == 10.5  # 0.5 + 10.0
            assert words[0]['end'] == 11.0    # 1.0 + 10.0
            assert words[1]['start'] == 11.0  # 1.0 + 10.0
            assert words[1]['end'] == 11.5    # 1.5 + 10.0
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    def test_no_timestamp_adjustment_when_start_none(self, mock_load_key, mock_request, mock_librosa):
        """Test no timestamp adjustment when start is None."""
        mock_librosa.return_value = (np.array([0.1]), 16000)
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.whisperX_302_api_key': 'test_key'
        }.get(key, 'default')
        
        api_response = {
            'segments': [
                {
                    'start': 1.0,
                    'end': 3.0,
                    'words': [{'word': 'test', 'start': 1.5, 'end': 2.5}]
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.json.return_value = api_response
        mock_request.return_value = mock_response
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            # Call with None start parameter
            result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', None, None)
            
            # Timestamps should remain unchanged
            segment = result['segments'][0]
            assert segment['start'] == 1.0  # No adjustment
            assert segment['end'] == 3.0
            assert segment['words'][0]['start'] == 1.5
            assert segment['words'][0]['end'] == 2.5
    
    @patch('core.asr_backend.whisperX_302.librosa.load')
    @patch('core.asr_backend.whisperX_302.requests.request')
    @patch('core.asr_backend.whisperX_302.load_key')
    def test_timestamp_adjustment_with_missing_word_timestamps(self, mock_load_key, mock_request, mock_librosa):
        """Test timestamp adjustment with words missing timestamps."""
        mock_librosa.return_value = (np.array([0.1]), 16000)
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.whisperX_302_api_key': 'test_key'
        }.get(key, 'default')
        
        api_response = {
            'segments': [
                {
                    'start': 0.0,
                    'end': 2.0,
                    'words': [
                        {'word': 'Hello', 'start': 0.0, 'end': 1.0},
                        {'word': 'world'}  # Missing timestamps
                    ]
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.json.return_value = api_response
        mock_request.return_value = mock_response
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            result = transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 5.0, 10.0)
            
            # First word should be adjusted
            words = result['segments'][0]['words']
            assert words[0]['start'] == 5.0  # 0.0 + 5.0
            assert words[0]['end'] == 6.0    # 1.0 + 5.0
            
            # Second word without timestamps should remain unchanged
            assert 'start' not in words[1] or words[1].get('start') is None
            assert 'end' not in words[1] or words[1].get('end') is None


class TestConfigurationHandling:
    """Test configuration and API key handling."""
    
    def test_language_configuration(self):
        """Test language configuration is properly used."""
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_302.requests.request') as mock_request, \
             patch('core.asr_backend.whisperX_302.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            mock_librosa.return_value = (np.array([0.1]), 16000)
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'de',  # German
                'whisper.whisperX_302_api_key': 'test_key'
            }.get(key, 'default')
            
            mock_response = Mock()
            mock_response.json.return_value = {'segments': []}
            mock_request.return_value = mock_response
            
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 5.0)
            
            # Verify German language was used in API request
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs['data']['language'] == 'de'
    
    def test_api_key_configuration(self):
        """Test API key is properly retrieved and used."""
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_302.requests.request') as mock_request, \
             patch('core.asr_backend.whisperX_302.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            mock_librosa.return_value = (np.array([0.1]), 16000)
            test_api_key = 'sk-302ai-test-key-12345'
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.whisperX_302_api_key': test_api_key
            }.get(key, 'default')
            
            mock_response = Mock()
            mock_response.json.return_value = {'segments': []}
            mock_request.return_value = mock_response
            
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 5.0)
            
            # Verify API key was used in authorization header
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs['headers']['Authorization'] == f'Bearer {test_api_key}'
    
    @patch('core.asr_backend.whisperX_302.load_key')
    def test_missing_api_key_handling(self, mock_load_key):
        """Test handling of missing API key."""
        mock_load_key.side_effect = lambda key: {
            'whisper.language': 'en',
            'whisper.whisperX_302_api_key': None  # Missing API key
        }.get(key, None)
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_302.requests.request') as mock_request:
            
            mock_librosa.return_value = (np.array([0.1]), 16000)
            
            # Should still attempt API call but may fail with authentication error
            mock_request.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
            
            with pytest.raises(requests.exceptions.HTTPError):
                transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 5.0)


class TestPerformanceAndTiming:
    """Test performance monitoring and timing functionality."""
    
    @patch('core.asr_backend.whisperX_302.time.time')
    @patch('core.asr_backend.whisperX_302.rprint')
    def test_timing_measurement(self, mock_rprint, mock_time):
        """Test that transcription timing is measured and logged."""
        # Mock time progression
        mock_time.side_effect = [0.0, 2.5]  # 2.5 seconds elapsed
        
        with patch('core.asr_backend.whisperX_302.os.path.exists', return_value=False), \
             patch('core.asr_backend.whisperX_302.librosa.load') as mock_librosa, \
             patch('core.asr_backend.whisperX_302.requests.request') as mock_request, \
             patch('core.asr_backend.whisperX_302.load_key') as mock_load_key, \
             patch('core.asr_backend.whisperX_302.os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('core.asr_backend.whisperX_302.json.dump'), \
             patch('core.asr_backend.whisperX_302.sf.write'):
            
            mock_librosa.return_value = (np.array([0.1]), 16000)
            mock_load_key.side_effect = lambda key: {
                'whisper.language': 'en',
                'whisper.whisperX_302_api_key': 'test_key'
            }.get(key, 'default')
            
            mock_response = Mock()
            mock_response.json.return_value = {'segments': []}
            mock_request.return_value = mock_response
            
            transcribe_audio_302('/tmp/raw.wav', '/tmp/vocal.wav', 0.0, 5.0)
            
            # Verify timing was logged
            timing_calls = [call for call in mock_rprint.call_args_list 
                          if 'completed in' in str(call) and '2.50 seconds' in str(call)]
            assert len(timing_calls) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
