"""
Dynamic Test Data Generators
Programmatic generation of test data to avoid large static files
Specialized generators for VideoLingo's various data types
"""

import random
import string
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import io
import wave
import struct

from .base import (
    AbstractDataGenerator, DataCategory, DataFormat, DataScope, 
    DataMetadata, DataGenerationError
)

class ConfigGenerator(AbstractDataGenerator[Dict[str, Any]]):
    """Generator for VideoLingo configuration variants"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        if seed:
            random.seed(seed)
    
    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate configuration data"""
        variant = kwargs.get('variant', 'default')
        
        base_config = self._get_base_config()
        
        if variant == 'minimal':
            return self._generate_minimal_config(base_config)
        elif variant == 'full':
            return self._generate_full_config(base_config)
        elif variant == 'openai':
            return self._generate_openai_config(base_config)
        elif variant == 'openrouter':
            return self._generate_openrouter_config(base_config)
        elif variant == 'azure':
            return self._generate_azure_config(base_config)
        elif variant == 'error_prone':
            return self._generate_error_prone_config(base_config)
        else:
            return base_config
    
    def get_metadata_template(self) -> DataMetadata:
        """Get metadata template for config data"""
        return DataMetadata(
            name="",
            category=DataCategory.CONFIG,
            format=DataFormat.YAML,
            scope=DataScope.TEST_MODULE,
            description="Generated VideoLingo configuration",
            tags=["config", "generated"]
        )
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Get base configuration structure"""
        return {
            'api': {
                'key': 'sk-test-key-placeholder',
                'base_url': 'https://api.openai.com/v1',
                'model': 'gpt-3.5-turbo',
                'llm_support_json': True
            },
            'max_workers': 2,
            'target_language': '简体中文',
            'display_language': 'zh-CN',
            'youtube_resolution': '1080',
            'demucs': True,
            'burn_subtitles': True,
            'ffmpeg_gpu': False,
            'whisper': {
                'model': 'large-v3',
                'language': 'en',
                'runtime': 'local'
            },
            'tts_method': 'openai_tts',
            'subtitle': {
                'max_length': 75,
                'target_multiplier': 1.2
            },
            'summary_length': 8000,
            'max_split_length': 20,
            'reflect_translate': True,
            'pause_before_translate': False,
            'batch_translate_size': 15,
            'speed_factor': {
                'min': 1.0,
                'accept': 1.2,
                'max': 1.4
            },
            'min_subtitle_duration': 2.5,
            'min_trim_duration': 3.5,
            'tolerance': 1.5,
            'video_storage': {
                'base_path': '',
                'input_dir': 'input',
                'temp_dir': 'temp',
                'output_dir': 'output'
            }
        }
    
    def _generate_minimal_config(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Generate minimal configuration for testing"""
        return {
            'api': {
                'key': 'test-minimal-key',
                'base_url': 'https://test.example.com/v1',
                'model': 'test-model'
            },
            'max_workers': 1,
            'target_language': 'English',
            'tts_method': 'edge_tts'
        }
    
    def _generate_full_config(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Generate full configuration with all options"""
        config = base.copy()
        
        # Add additional optional configurations
        config.update({
            'openai_tts': {
                'api_key': f'sk-openai-{self._random_string(32)}',
                'voice': random.choice(['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']),
                'model': 'tts-1-hd'
            },
            'azure_tts': {
                'api_key': f'azure-key-{self._random_string(32)}',
                'region': random.choice(['eastus', 'westus2', 'northeurope']),
                'voice': 'zh-CN-XiaoxiaoNeural'
            },
            'sf_fish_tts': {
                'api_key': f'fish-{self._random_string(24)}',
                'model': 'fish-speech-1'
            },
            'youtube': {
                'cookies_path': '/path/to/cookies.txt',
                'proxy': 'http://proxy.example.com:8080'
            },
            'model_dir': '/custom/model/cache'
        })
        
        return config
    
    def _generate_openrouter_config(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Generate OpenRouter-specific configuration"""
        config = base.copy()
        config['api'] = {
            'key': f'sk-or-v1-{self._random_string(32)}',
            'base_url': 'https://openrouter.ai/api/v1',
            'model': random.choice([
                'anthropic/claude-3.5-sonnet',
                'openai/gpt-4-turbo',
                'google/gemini-pro',
                'mistralai/mistral-large'
            ]),
            'llm_support_json': True
        }
        return config
    
    def _generate_azure_config(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Azure-specific configuration"""
        config = base.copy()
        config['api'] = {
            'key': f'azure-key-{self._random_string(32)}',
            'base_url': f'https://{self._random_string(8)}.openai.azure.com/',
            'model': 'gpt-35-turbo',
            'llm_support_json': True
        }
        config['azure_tts'] = {
            'api_key': f'azure-tts-{self._random_string(32)}',
            'region': random.choice(['eastus', 'westus2', 'northeurope']),
            'voice': 'zh-CN-XiaoxiaoNeural'
        }
        return config
    
    def _generate_error_prone_config(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration that might cause common errors"""
        config = base.copy()
        
        # Introduce potential issues for testing error handling
        error_scenarios = random.choice([
            'missing_api_key',
            'invalid_url',
            'unsupported_language',
            'invalid_tts_method'
        ])
        
        if error_scenarios == 'missing_api_key':
            config['api']['key'] = ''
        elif error_scenarios == 'invalid_url':
            config['api']['base_url'] = 'not-a-valid-url'
        elif error_scenarios == 'unsupported_language':
            config['target_language'] = 'UnsupportedLanguage'
        elif error_scenarios == 'invalid_tts_method':
            config['tts_method'] = 'non_existent_tts'
        
        return config
    
    def _random_string(self, length: int) -> str:
        """Generate random string for keys and IDs"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class APIResponseGenerator(AbstractDataGenerator[Dict[str, Any]]):
    """Generator for API response mock data"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        if seed:
            random.seed(seed)
    
    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate API response data"""
        api_type = kwargs.get('api_type', 'openai')
        scenario = kwargs.get('scenario', 'success')
        
        if api_type == 'openai':
            return self._generate_openai_response(scenario)
        elif api_type == 'azure':
            return self._generate_azure_response(scenario)
        elif api_type == 'whisper':
            return self._generate_whisper_response(scenario)
        elif api_type == 'tts':
            return self._generate_tts_response(scenario)
        else:
            raise DataGenerationError(f"Unsupported API type: {api_type}")
    
    def get_metadata_template(self) -> DataMetadata:
        """Get metadata template for API response data"""
        return DataMetadata(
            name="",
            category=DataCategory.API_RESPONSE,
            format=DataFormat.JSON,
            scope=DataScope.TEST_FUNCTION,
            description="Generated API response mock",
            tags=["api", "mock", "generated"]
        )
    
    def _generate_openai_response(self, scenario: str) -> Dict[str, Any]:
        """Generate OpenAI API response"""
        if scenario == 'success':
            return {
                'id': f'chatcmpl-{self._random_id()}',
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': 'gpt-3.5-turbo',
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': self._generate_translation_content()
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': random.randint(100, 500),
                    'completion_tokens': random.randint(50, 200),
                    'total_tokens': random.randint(150, 700)
                }
            }
        elif scenario == 'error':
            return {
                'error': {
                    'message': 'Rate limit exceeded',
                    'type': 'rate_limit_error',
                    'code': 'rate_limit_exceeded'
                }
            }
        elif scenario == 'timeout':
            return {
                'error': {
                    'message': 'Request timeout',
                    'type': 'timeout_error',
                    'code': 'timeout'
                }
            }
        else:
            raise DataGenerationError(f"Unknown scenario: {scenario}")
    
    def _generate_whisper_response(self, scenario: str) -> Dict[str, Any]:
        """Generate Whisper ASR response"""
        if scenario == 'success':
            return {
                'text': self._generate_transcript_text(),
                'segments': self._generate_segments(),
                'language': 'en'
            }
        elif scenario == 'empty':
            return {
                'text': '',
                'segments': [],
                'language': 'en'
            }
        else:
            return {'error': 'ASR processing failed'}
    
    def _generate_tts_response(self, scenario: str) -> Dict[str, Any]:
        """Generate TTS response"""
        if scenario == 'success':
            return {
                'audio_data': self._generate_mock_audio_data(),
                'format': 'mp3',
                'duration': random.uniform(1.0, 10.0),
                'sample_rate': 24000
            }
        else:
            return {'error': 'TTS generation failed'}
    
    def _generate_translation_content(self) -> str:
        """Generate sample translation content"""
        translations = [
            "这是一个测试翻译。",
            "机器学习正在改变世界。",
            "人工智能的未来充满可能性。",
            "技术进步让生活更美好。"
        ]
        return random.choice(translations)
    
    def _generate_transcript_text(self) -> str:
        """Generate sample transcript text"""
        transcripts = [
            "Hello, this is a test audio transcription.",
            "Welcome to the world of artificial intelligence.",
            "Today we will discuss machine learning algorithms.",
            "Thank you for watching this educational video."
        ]
        return random.choice(transcripts)
    
    def _generate_segments(self) -> List[Dict[str, Any]]:
        """Generate transcript segments"""
        segments = []
        start_time = 0.0
        
        for i in range(random.randint(3, 8)):
            duration = random.uniform(2.0, 5.0)
            end_time = start_time + duration
            
            segments.append({
                'id': i,
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'text': f'This is segment {i + 1} of the transcript.',
                'words': self._generate_word_timestamps(start_time, end_time)
            })
            
            start_time = end_time + random.uniform(0.1, 0.5)
        
        return segments
    
    def _generate_word_timestamps(self, start: float, end: float) -> List[Dict[str, Any]]:
        """Generate word-level timestamps"""
        words = ['This', 'is', 'segment', 'of', 'the', 'transcript']
        word_duration = (end - start) / len(words)
        
        word_timestamps = []
        for i, word in enumerate(words):
            word_start = start + i * word_duration
            word_end = word_start + word_duration * 0.8
            
            word_timestamps.append({
                'word': word,
                'start': round(word_start, 3),
                'end': round(word_end, 3),
                'confidence': random.uniform(0.8, 1.0)
            })
        
        return word_timestamps
    
    def _generate_mock_audio_data(self) -> str:
        """Generate mock audio data (base64 encoded)"""
        import base64
        mock_data = b'MOCK_AUDIO_DATA' + bytes(random.randint(0, 255) for _ in range(1024))
        return base64.b64encode(mock_data).decode('ascii')
    
    def _random_id(self) -> str:
        """Generate random ID"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

class MediaSampleGenerator(AbstractDataGenerator[bytes]):
    """Generator for media sample files"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        if seed:
            random.seed(seed)
    
    def generate(self, **kwargs) -> bytes:
        """Generate media sample data"""
        media_type = kwargs.get('media_type', 'audio')
        size = kwargs.get('size', 'small')
        
        if media_type == 'audio':
            return self._generate_audio_sample(size)
        elif media_type == 'video':
            return self._generate_video_sample(size)
        elif media_type == 'image':
            return self._generate_image_sample(size)
        else:
            raise DataGenerationError(f"Unsupported media type: {media_type}")
    
    def get_metadata_template(self) -> DataMetadata:
        """Get metadata template for media sample data"""
        return DataMetadata(
            name="",
            category=DataCategory.MEDIA_SAMPLE,
            format=DataFormat.BINARY,
            scope=DataScope.TEST_MODULE,
            description="Generated media sample",
            tags=["media", "sample", "generated"]
        )
    
    def _generate_audio_sample(self, size: str) -> bytes:
        """Generate WAV audio sample"""
        duration_map = {'small': 1, 'medium': 5, 'large': 30}
        duration = duration_map.get(size, 1)
        
        sample_rate = 16000
        frequency = 440  # A4 note
        
        # Generate sine wave
        samples = []
        for i in range(int(sample_rate * duration)):
            sample = int(32767 * 0.3 * random.uniform(-1, 1))  # Add some noise
            samples.append(sample)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
        
        return buffer.getvalue()
    
    def _generate_video_sample(self, size: str) -> bytes:
        """Generate mock video file"""
        size_map = {'small': 1024, 'medium': 10240, 'large': 102400}
        file_size = size_map.get(size, 1024)
        
        # Create a mock MP4-like structure
        header = b'ftypisom'  # Mock MP4 header
        padding = bytes(random.randint(0, 255) for _ in range(file_size - len(header)))
        
        return header + padding
    
    def _generate_image_sample(self, size: str) -> bytes:
        """Generate mock image file"""
        # Simple BMP structure for testing
        size_map = {'small': 64, 'medium': 256, 'large': 1024}
        dimension = size_map.get(size, 64)
        
        # BMP header (simplified)
        header = b'BM'  # BMP signature
        file_size = 54 + dimension * dimension * 3  # Header + RGB data
        
        bmp_data = header + struct.pack('<I', file_size)
        bmp_data += b'\x00\x00\x00\x00'  # Reserved
        bmp_data += struct.pack('<I', 54)  # Offset to image data
        bmp_data += struct.pack('<I', 40)  # Header size
        bmp_data += struct.pack('<II', dimension, dimension)  # Width, Height
        bmp_data += struct.pack('<HH', 1, 24)  # Planes, bits per pixel
        
        # Add padding to reach expected size
        padding_size = file_size - len(bmp_data)
        bmp_data += bytes(random.randint(0, 255) for _ in range(padding_size))
        
        return bmp_data

class LanguageDataGenerator(AbstractDataGenerator[Dict[str, Any]]):
    """Generator for language-specific test data"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        if seed:
            random.seed(seed)
    
    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate language test data"""
        language = kwargs.get('language', 'en')
        data_type = kwargs.get('data_type', 'sentences')
        
        if data_type == 'sentences':
            return self._generate_sentences(language)
        elif data_type == 'translations':
            return self._generate_translation_pairs(language)
        elif data_type == 'terminology':
            return self._generate_terminology(language)
        else:
            raise DataGenerationError(f"Unsupported language data type: {data_type}")
    
    def get_metadata_template(self) -> DataMetadata:
        """Get metadata template for language data"""
        return DataMetadata(
            name="",
            category=DataCategory.LANGUAGE_DATA,
            format=DataFormat.JSON,
            scope=DataScope.TEST_MODULE,
            description="Generated language test data",
            tags=["language", "text", "generated"]
        )
    
    def _generate_sentences(self, language: str) -> Dict[str, Any]:
        """Generate sample sentences for a language"""
        sentences_data = {
            'en': [
                "This is a sample English sentence.",
                "Machine learning is transforming the world.",
                "The future of artificial intelligence is bright.",
                "Technology makes our lives easier and better."
            ],
            'zh': [
                "这是一个中文示例句子。",
                "机器学习正在改变世界。",
                "人工智能的未来一片光明。",
                "技术让我们的生活更加便利美好。"
            ],
            'es': [
                "Esta es una oración de ejemplo en español.",
                "El aprendizaje automático está transformando el mundo.",
                "El futuro de la inteligencia artificial es brillante.",
                "La tecnología hace nuestras vidas más fáciles y mejores."
            ]
        }
        
        return {
            'language': language,
            'sentences': sentences_data.get(language, sentences_data['en']),
            'count': len(sentences_data.get(language, sentences_data['en']))
        }
    
    def _generate_translation_pairs(self, target_language: str) -> Dict[str, Any]:
        """Generate translation pairs"""
        pairs = {
            'zh': [
                {'source': 'Hello, world\!', 'target': '你好，世界！'},
                {'source': 'How are you?', 'target': '你好吗？'},
                {'source': 'Thank you very much.', 'target': '非常感谢。'},
                {'source': 'See you later.', 'target': '再见。'}
            ],
            'es': [
                {'source': 'Hello, world\!', 'target': '¡Hola, mundo\!'},
                {'source': 'How are you?', 'target': '¿Cómo estás?'},
                {'source': 'Thank you very much.', 'target': 'Muchas gracias.'},
                {'source': 'See you later.', 'target': 'Hasta luego.'}
            ]
        }
        
        return {
            'source_language': 'en',
            'target_language': target_language,
            'pairs': pairs.get(target_language, pairs['zh'])
        }
    
    def _generate_terminology(self, language: str) -> Dict[str, Any]:
        """Generate terminology glossary"""
        terms = {
            'zh': {
                'machine learning': '机器学习',
                'artificial intelligence': '人工智能',
                'neural network': '神经网络',
                'deep learning': '深度学习'
            },
            'es': {
                'machine learning': 'aprendizaje automático',
                'artificial intelligence': 'inteligencia artificial',
                'neural network': 'red neuronal',
                'deep learning': 'aprendizaje profundo'
            }
        }
        
        return {
            'language': language,
            'terminology': terms.get(language, {}),
            'count': len(terms.get(language, {}))
        }

class ErrorScenarioGenerator(AbstractDataGenerator[Dict[str, Any]]):
    """Generator for error scenarios and edge cases"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        if seed:
            random.seed(seed)
    
    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate error scenario data"""
        error_type = kwargs.get('error_type', 'api_error')
        
        if error_type == 'api_error':
            return self._generate_api_error()
        elif error_type == 'file_error':
            return self._generate_file_error()
        elif error_type == 'network_error':
            return self._generate_network_error()
        elif error_type == 'validation_error':
            return self._generate_validation_error()
        else:
            raise DataGenerationError(f"Unsupported error type: {error_type}")
    
    def get_metadata_template(self) -> DataMetadata:
        """Get metadata template for error scenario data"""
        return DataMetadata(
            name="",
            category=DataCategory.ERROR_SCENARIO,
            format=DataFormat.JSON,
            scope=DataScope.TEST_FUNCTION,
            description="Generated error scenario",
            tags=["error", "testing", "generated"]
        )
    
    def _generate_api_error(self) -> Dict[str, Any]:
        """Generate API error scenarios"""
        errors = [
            {
                'error_code': 429,
                'error_message': 'Rate limit exceeded',
                'retry_after': random.randint(60, 300),
                'type': 'rate_limit'
            },
            {
                'error_code': 401,
                'error_message': 'Invalid API key',
                'type': 'authentication'
            },
            {
                'error_code': 503,
                'error_message': 'Service unavailable',
                'type': 'service_error'
            },
            {
                'error_code': 400,
                'error_message': 'Invalid request format',
                'type': 'validation_error'
            }
        ]
        
        return random.choice(errors)
    
    def _generate_file_error(self) -> Dict[str, Any]:
        """Generate file operation errors"""
        errors = [
            {
                'error_type': 'file_not_found',
                'file_path': '/nonexistent/path/file.mp4',
                'error_message': 'The specified file does not exist'
            },
            {
                'error_type': 'permission_denied',
                'file_path': '/restricted/directory/file.mp4',
                'error_message': 'Permission denied accessing file'
            },
            {
                'error_type': 'disk_full',
                'error_message': 'No space left on device'
            },
            {
                'error_type': 'corrupted_file',
                'file_path': '/path/to/corrupted.mp4',
                'error_message': 'File appears to be corrupted or invalid format'
            }
        ]
        
        return random.choice(errors)
    
    def _generate_network_error(self) -> Dict[str, Any]:
        """Generate network-related errors"""
        errors = [
            {
                'error_type': 'connection_timeout',
                'timeout_seconds': random.randint(10, 60),
                'error_message': 'Connection timed out'
            },
            {
                'error_type': 'dns_resolution',
                'hostname': f'invalid-{self._random_string(8)}.com',
                'error_message': 'Name or service not known'
            },
            {
                'error_type': 'connection_refused',
                'port': random.randint(8000, 9000),
                'error_message': 'Connection refused'
            }
        ]
        
        return random.choice(errors)
    
    def _generate_validation_error(self) -> Dict[str, Any]:
        """Generate validation errors"""
        errors = [
            {
                'field': 'api_key',
                'value': '',
                'error_message': 'API key cannot be empty'
            },
            {
                'field': 'target_language',
                'value': 'InvalidLanguage',
                'error_message': 'Unsupported target language'
            },
            {
                'field': 'max_workers',
                'value': -1,
                'error_message': 'Max workers must be positive integer'
            }
        ]
        
        return random.choice(errors)
    
    def _random_string(self, length: int) -> str:
        """Generate random string"""
        return ''.join(random.choices(string.ascii_lowercase, k=length))
EOF < /dev/null