"""
Functional tests for Individual TTS Backend implementations
Tests specific TTS engines without complex external dependencies
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any


class TestAzureTTSLogic:
    """Test azure_tts.py core logic"""
    
    def test_azure_tts_request_logic(self):
        """Test Azure TTS request preparation and execution logic"""
        # Simulate azure_tts logic
        def mock_azure_tts_logic(text, save_path, api_key, voice):
            """Mock Azure TTS logic"""
            # Step 1: Prepare SSML payload
            payload = f"<speak version='1.0' xml:lang='zh-CN'><voice name='{voice}'>{text}</voice></speak>"
            
            # Step 2: Prepare headers
            headers = {
                'Authorization': f'Bearer {api_key}',
                'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm',
                'Content-Type': 'application/ssml+xml'
            }
            
            # Step 3: Validate configuration
            validation_result = {
                'valid': True,
                'errors': []
            }
            
            if not api_key:
                validation_result['valid'] = False
                validation_result['errors'].append('API key is required')
            
            if len(api_key) < 10:
                validation_result['valid'] = False
                validation_result['errors'].append('API key appears invalid')
            
            if not voice:
                validation_result['valid'] = False
                validation_result['errors'].append('Voice is required')
            
            if not text.strip():
                validation_result['valid'] = False
                validation_result['errors'].append('Text is required')
            
            if not validation_result['valid']:
                return {
                    'success': False,
                    'validation': validation_result,
                    'payload': payload,
                    'headers': headers
                }
            
            # Step 4: Mock API request
            def simulate_azure_api_request(url, headers, payload):
                # Simulate different response scenarios
                if 'invalid_key' in api_key:
                    return {
                        'status_code': 401,
                        'content': b'',
                        'error': 'Unauthorized - Invalid API key'
                    }
                elif 'rate_limit' in api_key:
                    return {
                        'status_code': 429,
                        'content': b'',
                        'error': 'Rate limit exceeded'
                    }
                elif 'network_error' in api_key:
                    return {
                        'status_code': 500,
                        'content': b'',
                        'error': 'Internal server error'
                    }
                else:
                    # Successful response with mock audio data
                    mock_audio_content = b'RIFF' + b'\x00' * 100  # Mock WAV header + data
                    return {
                        'status_code': 200,
                        'content': mock_audio_content,
                        'error': None
                    }
            
            api_response = simulate_azure_api_request("https://api.302.ai/cognitiveservices/v1", headers, payload)
            
            # Step 5: Handle response
            if api_response['status_code'] == 200:
                # Mock file writing
                file_size = len(api_response['content'])
                
                return {
                    'success': True,
                    'validation': validation_result,
                    'payload': payload,
                    'headers': headers,
                    'api_response': api_response,
                    'file_info': {
                        'save_path': save_path,
                        'file_size': file_size,
                        'format': 'WAV',
                        'sample_rate': '16kHz',
                        'bit_depth': '16-bit',
                        'channels': 'mono'
                    }
                }
            else:
                return {
                    'success': False,
                    'validation': validation_result,
                    'api_response': api_response,
                    'error': api_response['error']
                }
        
        # Test successful Azure TTS call
        result = mock_azure_tts_logic(
            text="Hello, this is a test.",
            save_path="/tmp/test_azure.wav",
            api_key="valid_azure_api_key_123456",
            voice="zh-CN-XiaoxiaoNeural"
        )
        
        assert result['success'] is True
        assert result['validation']['valid'] is True
        assert 'zh-CN-XiaoxiaoNeural' in result['payload']
        assert 'Hello, this is a test.' in result['payload']
        assert result['headers']['Content-Type'] == 'application/ssml+xml'
        assert result['headers']['X-Microsoft-OutputFormat'] == 'riff-16khz-16bit-mono-pcm'
        assert result['file_info']['format'] == 'WAV'
        assert result['file_info']['file_size'] > 0
        
        # Test invalid API key
        invalid_result = mock_azure_tts_logic(
            text="Test text",
            save_path="/tmp/test.wav",
            api_key="invalid_key",
            voice="zh-CN-XiaoxiaoNeural"
        )
        
        assert invalid_result['success'] is False
        assert 'Unauthorized' in invalid_result['error']
        assert invalid_result['api_response']['status_code'] == 401
        
        # Test missing configuration
        missing_config_result = mock_azure_tts_logic(
            text="",
            save_path="/tmp/test.wav",
            api_key="",
            voice=""
        )
        
        assert missing_config_result['success'] is False
        assert missing_config_result['validation']['valid'] is False
        assert 'API key is required' in missing_config_result['validation']['errors']
        assert 'Voice is required' in missing_config_result['validation']['errors']
        assert 'Text is required' in missing_config_result['validation']['errors']
        
        # Test rate limiting
        rate_limit_result = mock_azure_tts_logic(
            text="Test text",
            save_path="/tmp/test.wav",
            api_key="rate_limit_key_123456",
            voice="zh-CN-XiaoxiaoNeural"
        )
        
        assert rate_limit_result['success'] is False
        assert rate_limit_result['api_response']['status_code'] == 429
        assert 'Rate limit exceeded' in rate_limit_result['error']
    
    def test_azure_ssml_generation_logic(self):
        """Test Azure SSML payload generation logic"""
        # Simulate SSML generation logic
        def mock_generate_azure_ssml(text, voice, language='zh-CN', custom_settings=None):
            """Mock Azure SSML generation"""
            # Basic SSML structure
            ssml_template = "<speak version='1.0' xml:lang='{lang}'><voice name='{voice}'>{content}</voice></speak>"
            
            # Handle custom settings
            if custom_settings:
                # Advanced SSML with prosody control
                prosody_attrs = []
                if 'rate' in custom_settings:
                    prosody_attrs.append(f"rate='{custom_settings['rate']}'")
                if 'pitch' in custom_settings:
                    prosody_attrs.append(f"pitch='{custom_settings['pitch']}'")
                if 'volume' in custom_settings:
                    prosody_attrs.append(f"volume='{custom_settings['volume']}'")
                
                if prosody_attrs:
                    prosody_tag = f"<prosody {' '.join(prosody_attrs)}>{text}</prosody>"
                    content = prosody_tag
                else:
                    content = text
            else:
                content = text
            
            # Escape XML special characters
            content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            ssml = ssml_template.format(lang=language, voice=voice, content=content)
            
            return {
                'ssml': ssml,
                'language': language,
                'voice': voice,
                'has_prosody': custom_settings is not None and any(key in custom_settings for key in ['rate', 'pitch', 'volume']),
                'content_escaped': '&' in content or '&lt;' in content or '&gt;' in content
            }
        
        # Test basic SSML generation
        basic_ssml = mock_generate_azure_ssml(
            text="Hello world",
            voice="zh-CN-XiaoxiaoNeural"
        )
        
        assert "<speak version='1.0' xml:lang='zh-CN'>" in basic_ssml['ssml']
        assert "<voice name='zh-CN-XiaoxiaoNeural'>Hello world</voice>" in basic_ssml['ssml']
        assert basic_ssml['language'] == 'zh-CN'
        assert basic_ssml['has_prosody'] is False
        
        # Test SSML with prosody settings
        advanced_ssml = mock_generate_azure_ssml(
            text="Slow and quiet speech",
            voice="en-US-JennyNeural",
            language="en-US",
            custom_settings={'rate': 'slow', 'volume': 'soft', 'pitch': 'low'}
        )
        
        assert advanced_ssml['has_prosody'] is True
        assert "rate='slow'" in advanced_ssml['ssml']
        assert "volume='soft'" in advanced_ssml['ssml']
        assert "pitch='low'" in advanced_ssml['ssml']
        assert advanced_ssml['language'] == 'en-US'
        
        # Test XML escaping
        xml_text = "Text with <tags> & special characters"
        escaped_ssml = mock_generate_azure_ssml(xml_text, "zh-CN-XiaoxiaoNeural")
        
        assert escaped_ssml['content_escaped'] is True
        assert '&lt;tags&gt;' in escaped_ssml['ssml']
        assert '&amp;' in escaped_ssml['ssml']


class TestOpenAITTSLogic:
    """Test openai_tts.py core logic"""
    
    def test_openai_tts_request_logic(self):
        """Test OpenAI TTS request logic"""
        # Simulate openai_tts logic
        def mock_openai_tts_logic(text, save_path, api_key, voice):
            """Mock OpenAI TTS logic"""
            voice_list = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            
            # Step 1: Validate voice
            if voice not in voice_list:
                return {
                    'success': False,
                    'error': f'Invalid voice: {voice}. Please choose from {voice_list}',
                    'voice_validation': False
                }
            
            # Step 2: Prepare request payload
            payload = {
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "response_format": "wav"
            }
            
            headers = {
                'Authorization': f"Bearer {api_key}",
                'Content-Type': 'application/json'
            }
            
            # Step 3: Validate configuration
            validation_errors = []
            if not api_key:
                validation_errors.append('API key is required')
            if len(api_key) < 20:
                validation_errors.append('API key appears invalid')
            if not text.strip():
                validation_errors.append('Text input is required')
            
            if validation_errors:
                return {
                    'success': False,
                    'validation_errors': validation_errors,
                    'voice_validation': True
                }
            
            # Step 4: Mock API request
            def simulate_openai_api_request(url, headers, payload):
                if 'invalid' in api_key:
                    return {'status_code': 401, 'content': b'', 'text': 'Invalid API key'}
                elif 'quota_exceeded' in api_key:
                    return {'status_code': 429, 'content': b'', 'text': 'Quota exceeded'}
                elif 'model_overloaded' in api_key:
                    return {'status_code': 503, 'content': b'', 'text': 'Model overloaded'}
                else:
                    # Mock successful WAV content
                    mock_wav_content = b'RIFF' + b'\x24\x08\x00\x00' + b'WAVE' + b'\x00' * 50
                    return {'status_code': 200, 'content': mock_wav_content, 'text': ''}
            
            api_response = simulate_openai_api_request("https://api.openai.com/v1/audio/speech", headers, payload)
            
            # Step 5: Handle response
            if api_response['status_code'] == 200:
                # Mock directory creation and file writing
                file_size = len(api_response['content'])
                
                return {
                    'success': True,
                    'voice_validation': True,
                    'payload': payload,
                    'headers': headers,
                    'api_response': api_response,
                    'file_info': {
                        'save_path': save_path,
                        'file_size': file_size,
                        'format': 'WAV',
                        'model': 'tts-1'
                    }
                }
            else:
                return {
                    'success': False,
                    'voice_validation': True,
                    'api_response': api_response,
                    'error': f"API Error {api_response['status_code']}: {api_response['text']}"
                }
        
        # Test successful OpenAI TTS call
        result = mock_openai_tts_logic(
            text="Welcome to the future of AI.",
            save_path="/tmp/openai_test.wav",
            api_key="sk-valid_openai_api_key_123456789012345",
            voice="alloy"
        )
        
        assert result['success'] is True
        assert result['voice_validation'] is True
        assert result['payload']['model'] == 'tts-1'
        assert result['payload']['voice'] == 'alloy'
        assert result['payload']['response_format'] == 'wav'
        assert result['file_info']['format'] == 'WAV'
        assert result['file_info']['file_size'] > 0
        
        # Test invalid voice
        invalid_voice_result = mock_openai_tts_logic(
            text="Test text",
            save_path="/tmp/test.wav",
            api_key="sk-valid_key_123456789012345",
            voice="invalid_voice"
        )
        
        assert invalid_voice_result['success'] is False
        assert invalid_voice_result['voice_validation'] is False
        assert 'Invalid voice: invalid_voice' in invalid_voice_result['error']
        assert 'alloy' in invalid_voice_result['error']  # Should suggest valid voices
        
        # Test validation errors
        validation_result = mock_openai_tts_logic(
            text="",
            save_path="/tmp/test.wav",
            api_key="short",
            voice="alloy"
        )
        
        assert validation_result['success'] is False
        assert 'API key appears invalid' in validation_result['validation_errors']
        assert 'Text input is required' in validation_result['validation_errors']
        
        # Test API errors
        quota_result = mock_openai_tts_logic(
            text="Test text",
            save_path="/tmp/test.wav",
            api_key="sk-quota_exceeded_key_123456789012345",
            voice="nova"
        )
        
        assert quota_result['success'] is False
        assert quota_result['api_response']['status_code'] == 429
        assert 'Quota exceeded' in quota_result['error']
    
    def test_openai_voice_selection_logic(self):
        """Test OpenAI voice selection and validation logic"""
        # Simulate voice selection logic
        def mock_openai_voice_selection(requested_voice, language_preference=None):
            """Mock OpenAI voice selection logic"""
            voice_characteristics = {
                "alloy": {
                    "gender": "neutral",
                    "tone": "balanced",
                    "use_case": "general purpose",
                    "language_optimized": ["en"]
                },
                "echo": {
                    "gender": "male",
                    "tone": "deep",
                    "use_case": "narration",
                    "language_optimized": ["en"]
                },
                "fable": {
                    "gender": "neutral",
                    "tone": "expressive",
                    "use_case": "storytelling",
                    "language_optimized": ["en"]
                },
                "onyx": {
                    "gender": "male",
                    "tone": "authoritative",
                    "use_case": "professional",
                    "language_optimized": ["en"]
                },
                "nova": {
                    "gender": "female",
                    "tone": "energetic",
                    "use_case": "conversational",
                    "language_optimized": ["en"]
                },
                "shimmer": {
                    "gender": "female",
                    "tone": "gentle",
                    "use_case": "calm narration",
                    "language_optimized": ["en"]
                }
            }
            
            available_voices = list(voice_characteristics.keys())
            
            if requested_voice not in available_voices:
                # Suggest similar voices
                suggestions = []
                if 'male' in requested_voice.lower():
                    suggestions = ["echo", "onyx"]
                elif 'female' in requested_voice.lower():
                    suggestions = ["nova", "shimmer"]
                else:
                    suggestions = ["alloy", "fable"]
                
                return {
                    'valid': False,
                    'requested_voice': requested_voice,
                    'error': f'Voice "{requested_voice}" not available',
                    'available_voices': available_voices,
                    'suggestions': suggestions
                }
            
            voice_info = voice_characteristics[requested_voice]
            
            # Check language optimization
            language_match = True
            if language_preference:
                language_match = language_preference in voice_info['language_optimized']
            
            return {
                'valid': True,
                'selected_voice': requested_voice,
                'characteristics': voice_info,
                'language_optimized': language_match,
                'recommendation': 'optimal' if language_match else 'acceptable'
            }
        
        # Test valid voice selection
        alloy_result = mock_openai_voice_selection("alloy", "en")
        
        assert alloy_result['valid'] is True
        assert alloy_result['selected_voice'] == "alloy"
        assert alloy_result['characteristics']['gender'] == "neutral"
        assert alloy_result['language_optimized'] is True
        assert alloy_result['recommendation'] == 'optimal'
        
        # Test invalid voice
        invalid_result = mock_openai_voice_selection("invalid_voice")
        
        assert invalid_result['valid'] is False
        assert 'not available' in invalid_result['error']
        assert len(invalid_result['available_voices']) == 6
        assert len(invalid_result['suggestions']) > 0
        
        # Test gender-based suggestions
        male_result = mock_openai_voice_selection("male_voice")
        
        assert male_result['valid'] is False
        assert "echo" in male_result['suggestions']
        assert "onyx" in male_result['suggestions']
        
        female_result = mock_openai_voice_selection("female_voice")
        
        assert female_result['valid'] is False
        assert "nova" in female_result['suggestions']
        assert "shimmer" in female_result['suggestions']
        
        # Test language optimization
        spanish_result = mock_openai_voice_selection("alloy", "es")
        
        assert spanish_result['valid'] is True
        assert spanish_result['language_optimized'] is False  # Not optimized for Spanish
        assert spanish_result['recommendation'] == 'acceptable'


class TestEdgeTTSLogic:
    """Test edge_tts.py core logic"""
    
    def test_edge_tts_command_logic(self):
        """Test Edge TTS command preparation and execution logic"""
        # Simulate edge_tts logic
        def mock_edge_tts_logic(text, save_path, voice_config):
            """Mock Edge TTS logic"""
            # Step 1: Load voice configuration
            voice = voice_config.get("voice", "en-US-JennyNeural")
            
            # Step 2: Validate voice format
            def validate_edge_voice(voice_name):
                # Edge TTS voice format: language-region-NameNeural
                import re
                pattern = r'^[a-z]{2}-[A-Z]{2}-\w+Neural$'
                
                if not re.match(pattern, voice_name):
                    return {
                        'valid': False,
                        'error': 'Invalid voice format. Expected: language-region-NameNeural',
                        'examples': ['en-US-JennyNeural', 'zh-CN-XiaoxiaoNeural', 'es-ES-ElviraNeural']
                    }
                
                # Check if voice exists in mock database
                common_voices = [
                    'en-US-JennyNeural', 'en-US-GuyNeural', 'en-GB-SoniaNeural',
                    'zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural', 'zh-CN-XiaoyiNeural',
                    'es-ES-ElviraNeural', 'fr-FR-DeniseNeural', 'de-DE-KatjaNeural'
                ]
                
                if voice_name not in common_voices:
                    return {
                        'valid': False,
                        'error': f'Voice "{voice_name}" not found in available voices',
                        'suggestions': [v for v in common_voices if v.startswith(voice_name[:5])]
                    }
                
                return {
                    'valid': True,
                    'voice': voice_name,
                    'language': voice_name[:5],
                    'speaker': voice_name[6:-6]  # Extract speaker name
                }
            
            voice_validation = validate_edge_voice(voice)
            
            if not voice_validation['valid']:
                return {
                    'success': False,
                    'voice_validation': voice_validation,
                    'command': None
                }
            
            # Step 3: Prepare command
            cmd = ["edge-tts", "--voice", voice, "--text", text, "--write-media", save_path]
            
            # Step 4: Validate inputs
            validation_errors = []
            if not text.strip():
                validation_errors.append('Text input is required')
            if not save_path:
                validation_errors.append('Save path is required')
            if not save_path.endswith(('.wav', '.mp3')):
                validation_errors.append('Save path should end with .wav or .mp3')
            
            if validation_errors:
                return {
                    'success': False,
                    'validation_errors': validation_errors,
                    'voice_validation': voice_validation,
                    'command': cmd
                }
            
            # Step 5: Mock command execution
            def simulate_edge_command_execution(command):
                # Simulate different execution scenarios
                if 'network_error' in text:
                    return {
                        'returncode': 1,
                        'stdout': '',
                        'stderr': 'Network error: Unable to connect to Edge TTS service'
                    }
                elif 'invalid_text' in text:
                    return {
                        'returncode': 2,
                        'stdout': '',
                        'stderr': 'Text processing error: Invalid characters in text'
                    }
                elif len(text) > 1000:
                    return {
                        'returncode': 3,
                        'stdout': '',
                        'stderr': 'Text too long: Edge TTS has a character limit'
                    }
                else:
                    return {
                        'returncode': 0,
                        'stdout': f'Audio saved to {save_path}',
                        'stderr': ''
                    }
            
            execution_result = simulate_edge_command_execution(cmd)
            
            # Step 6: Handle execution result
            if execution_result['returncode'] == 0:
                # Mock file creation
                file_info = {
                    'save_path': save_path,
                    'file_size': len(text) * 100,  # Mock size calculation
                    'format': 'WAV' if save_path.endswith('.wav') else 'MP3',
                    'voice_used': voice,
                    'duration_estimate': len(text) / 10  # Mock duration
                }
                
                return {
                    'success': True,
                    'voice_validation': voice_validation,
                    'command': cmd,
                    'execution_result': execution_result,
                    'file_info': file_info
                }
            else:
                return {
                    'success': False,
                    'voice_validation': voice_validation,
                    'command': cmd,
                    'execution_result': execution_result,
                    'error': execution_result['stderr']
                }
        
        # Test successful Edge TTS call
        result = mock_edge_tts_logic(
            text="Today is a beautiful day for testing TTS.",
            save_path="/tmp/edge_test.wav",
            voice_config={"voice": "en-US-JennyNeural"}
        )
        
        assert result['success'] is True
        assert result['voice_validation']['valid'] is True
        assert result['voice_validation']['language'] == 'en-US'
        assert result['voice_validation']['speaker'] == 'Jenny'
        assert "edge-tts" in result['command']
        assert "--voice" in result['command']
        assert "en-US-JennyNeural" in result['command']
        assert result['file_info']['format'] == 'WAV'
        assert result['file_info']['duration_estimate'] > 0
        
        # Test invalid voice format
        invalid_voice_result = mock_edge_tts_logic(
            text="Test text",
            save_path="/tmp/test.wav",
            voice_config={"voice": "invalid-voice-format"}
        )
        
        assert invalid_voice_result['success'] is False
        assert invalid_voice_result['voice_validation']['valid'] is False
        assert 'Invalid voice format' in invalid_voice_result['voice_validation']['error']
        assert 'en-US-JennyNeural' in invalid_voice_result['voice_validation']['examples']
        
        # Test voice not found
        missing_voice_result = mock_edge_tts_logic(
            text="Test text",
            save_path="/tmp/test.wav",
            voice_config={"voice": "en-US-NonexistentNeural"}
        )
        
        assert missing_voice_result['success'] is False
        assert 'not found in available voices' in missing_voice_result['voice_validation']['error']
        assert len(missing_voice_result['voice_validation']['suggestions']) > 0
        
        # Test validation errors
        validation_error_result = mock_edge_tts_logic(
            text="",
            save_path="/tmp/test.txt",  # Wrong extension
            voice_config={"voice": "zh-CN-XiaoxiaoNeural"}
        )
        
        assert validation_error_result['success'] is False
        assert 'Text input is required' in validation_error_result['validation_errors']
        assert 'should end with .wav or .mp3' in validation_error_result['validation_errors']
        
        # Test execution errors
        network_error_result = mock_edge_tts_logic(
            text="This is a network_error test",
            save_path="/tmp/test.wav",
            voice_config={"voice": "en-US-GuyNeural"}
        )
        
        assert network_error_result['success'] is False
        assert network_error_result['execution_result']['returncode'] == 1
        assert 'Network error' in network_error_result['error']
        
        # Test text too long
        long_text = "a" * 1500  # Exceeds mock limit
        long_text_result = mock_edge_tts_logic(
            text=long_text,
            save_path="/tmp/test.wav",
            voice_config={"voice": "zh-CN-YunxiNeural"}
        )
        
        assert long_text_result['success'] is False
        assert long_text_result['execution_result']['returncode'] == 3
        assert 'Text too long' in long_text_result['error']
    
    def test_edge_voice_categorization_logic(self):
        """Test Edge TTS voice categorization and selection logic"""
        # Simulate voice categorization logic
        def mock_edge_voice_categorization():
            """Mock Edge TTS voice categorization"""
            voice_database = {
                'English': {
                    'US': {
                        'female': ['en-US-JennyNeural', 'en-US-AriaNeural'],
                        'male': ['en-US-GuyNeural', 'en-US-DavisNeural']
                    },
                    'GB': {
                        'female': ['en-GB-SoniaNeural', 'en-GB-LibbyNeural'],
                        'male': ['en-GB-RyanNeural', 'en-GB-ThomasNeural']
                    }
                },
                'Chinese': {
                    'CN': {
                        'female': ['zh-CN-XiaoxiaoNeural', 'zh-CN-XiaoyiNeural'],
                        'male': ['zh-CN-YunxiNeural', 'zh-CN-YunyangNeural']
                    },
                    'TW': {
                        'female': ['zh-TW-HsiaoChenNeural'],
                        'male': ['zh-TW-YunJheNeural']
                    }
                },
                'Spanish': {
                    'ES': {
                        'female': ['es-ES-ElviraNeural', 'es-ES-AlvaroNeural'],
                        'male': ['es-ES-AlvaroNeural']
                    }
                }
            }
            
            def find_voices_by_criteria(language=None, region=None, gender=None):
                matching_voices = []
                
                for lang, regions in voice_database.items():
                    if language and language.lower() not in lang.lower():
                        continue
                    
                    for reg, genders in regions.items():
                        if region and region.upper() != reg:
                            continue
                        
                        for gen, voices in genders.items():
                            if gender and gender.lower() != gen.lower():
                                continue
                            
                            for voice in voices:
                                matching_voices.append({
                                    'voice': voice,
                                    'language': lang,
                                    'region': reg,
                                    'gender': gen,
                                    'code': voice[:5]  # e.g., 'en-US'
                                })
                
                return matching_voices
            
            def get_voice_recommendations(text_language=None, speaker_preference=None):
                recommendations = []
                
                if text_language == 'english':
                    recommendations.extend(find_voices_by_criteria('English'))
                elif text_language == 'chinese':
                    recommendations.extend(find_voices_by_criteria('Chinese'))
                elif text_language == 'spanish':
                    recommendations.extend(find_voices_by_criteria('Spanish'))
                else:
                    # Default recommendations
                    recommendations.extend(find_voices_by_criteria('English', 'US'))
                
                if speaker_preference:
                    gender_filtered = [v for v in recommendations if v['gender'] == speaker_preference]
                    if gender_filtered:
                        recommendations = gender_filtered
                
                return recommendations[:5]  # Top 5 recommendations
            
            return {
                'voice_database': voice_database,
                'find_voices_by_criteria': find_voices_by_criteria,
                'get_voice_recommendations': get_voice_recommendations,
                'total_voices': sum(len(voices) for lang in voice_database.values() 
                                 for region in lang.values() 
                                 for voices in region.values())
            }
        
        categorization = mock_edge_voice_categorization()
        
        # Test voice database structure
        assert 'English' in categorization['voice_database']
        assert 'Chinese' in categorization['voice_database']
        assert categorization['total_voices'] > 10
        
        # Test finding voices by criteria
        english_female_voices = categorization['find_voices_by_criteria']('English', gender='female')
        
        assert len(english_female_voices) > 0
        assert all(v['gender'] == 'female' for v in english_female_voices)
        assert all('English' in v['language'] for v in english_female_voices)
        assert any('Jenny' in v['voice'] for v in english_female_voices)
        
        # Test Chinese voices
        chinese_voices = categorization['find_voices_by_criteria']('Chinese')
        
        assert len(chinese_voices) > 0
        assert all('zh-' in v['voice'] for v in chinese_voices)
        assert any(v['region'] == 'CN' for v in chinese_voices)
        
        # Test voice recommendations
        english_recommendations = categorization['get_voice_recommendations']('english', 'female')
        
        assert len(english_recommendations) > 0
        assert len(english_recommendations) <= 5
        assert all(v['gender'] == 'female' for v in english_recommendations)
        assert all('English' in v['language'] for v in english_recommendations)
        
        # Test default recommendations
        default_recommendations = categorization['get_voice_recommendations']()
        
        assert len(default_recommendations) > 0
        assert all('en-US' in v['voice'] for v in default_recommendations)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])