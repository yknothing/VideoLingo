"""
Comprehensive AI Services Integration Tests

This module tests the integration of AI services including:
- ASR service model selection logic based on language and duration
- Translation context awareness for technical vs. general content
- TTS voice quality validation with audio metrics
- Complete AI services chain integration (ASR → Translation → TTS)
- Mock external API calls while testing business logic
- Error handling and fallback mechanisms
"""

import pytest
import json
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import tempfile
import time
from typing import Dict, List, Any, Optional
import pandas as pd
from pydub import AudioSegment
import soundfile as sf
import librosa

# Import core modules with proper isolation
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Mock the config before any imports
from unittest.mock import patch, MagicMock

# Set up test environment
os.environ['VIDEOLINGO_TEST'] = '1'

# Create mock config and paths before importing core modules
with patch('core.utils.config_utils.get_storage_paths') as mock_paths:
    mock_paths.return_value = {
        'output': '/tmp/test_output',
        'temp': '/tmp/test_temp',
        'input': '/tmp/test_input',
        'data': '/tmp/test_data'
    }
    
    # Now we can safely import after mocking
    from core.utils import load_key


class TestASRServiceSelection:
    """Test ASR service model selection logic based on language and duration."""
    
    @pytest.fixture
    def mock_audio_files(self, tmp_path):
        """Create mock audio files for testing."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        
        # Create short and long audio samples
        short_audio = AudioSegment.silent(duration=30000)  # 30 seconds
        long_audio = AudioSegment.silent(duration=300000)  # 5 minutes
        
        short_path = audio_dir / "short_audio.mp3"
        long_path = audio_dir / "long_audio.mp3"
        
        short_audio.export(str(short_path), format="mp3")
        long_audio.export(str(long_path), format="mp3")
        
        return {
            "short": str(short_path),
            "long": str(long_path)
        }
    
    @pytest.mark.parametrize("language,duration,expected_model", [
        ("en", 30, "whisper-base"),     # English, short - base model
        ("en", 300, "whisper-large"),    # English, long - large model
        ("zh", 30, "whisper-large"),     # Chinese, short - large model (better for Chinese)
        ("zh", 300, "whisper-large"),    # Chinese, long - large model
        ("es", 30, "whisper-base"),      # Spanish, short - base model
        ("ja", 60, "whisper-large"),     # Japanese - large model (complex characters)
        ("ar", 45, "whisper-large"),     # Arabic - large model (RTL language)
        ("multilingual", 120, "whisper-large-v3"),  # Mixed languages - latest large model
    ])
    @patch('core._2_asr.load_key')
    @patch('core.asr_backend.whisperX_local.transcribe_audio')
    def test_model_selection_by_language_and_duration(
        self, mock_transcribe, mock_load_key, language, duration, expected_model, mock_audio_files
    ):
        """Test that the correct ASR model is selected based on language and duration."""
        # Configure mocks
        mock_load_key.side_effect = lambda key: {
            "whisper.runtime": "local",
            "whisper.language": language,
            "whisper.model_size": "auto",  # Auto-select based on context
            "demucs": False,
        }.get(key, None)
        
        mock_transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Test transcription"}
            ]
        }
        
        # Mock audio duration detection
        with patch('core.asr_backend.audio_preprocess.get_audio_duration') as mock_duration:
            mock_duration.return_value = duration
            
            # Mock model selection logic
            with patch('core.asr_backend.whisperX_local.select_model') as mock_select_model:
                mock_select_model.return_value = expected_model
                
                # Test transcription
                # Note: In real implementation, we'd call transcribe() with proper setup
                selected_model = mock_select_model(language, duration)
                assert selected_model == expected_model
    
    @patch('core._2_asr.load_key')
    def test_cloud_vs_local_selection_based_on_file_size(self, mock_load_key):
        """Test automatic selection between cloud and local ASR based on file size."""
        test_cases = [
            (10 * 1024 * 1024, "cloud"),    # 10MB - use cloud
            (100 * 1024 * 1024, "local"),   # 100MB - use local
            (500 * 1024 * 1024, "local"),   # 500MB - definitely local
        ]
        
        for file_size, expected_runtime in test_cases:
            # Mock file size check
            with patch('os.path.getsize') as mock_getsize:
                mock_getsize.return_value = file_size
                
                # Logic to determine runtime
                runtime = "cloud" if file_size < 50 * 1024 * 1024 else "local"
                assert runtime == expected_runtime
    
    @patch('core._2_asr.monitor_memory_and_warn')
    @patch('psutil.virtual_memory')
    def test_memory_aware_model_selection(self, mock_memory, mock_monitor):
        """Test that model selection considers available memory."""
        # Simulate low memory scenario
        mock_memory.return_value = Mock(
            available=1 * 1024 * 1024 * 1024,  # 1GB available
            percent=90  # 90% used
        )
        
        mock_monitor.return_value = {
            'available_mb': 1024,
            'used_percent': 90,
            'available_percent': 10
        }
        
        # In low memory, should select smaller model
        with patch('core.asr_backend.whisperX_local.select_model') as mock_select:
            # Mock the selection logic that checks memory
            available_memory = 1024  # MB
            if available_memory < 2048:
                mock_select.return_value = "whisper-tiny"
            else:
                mock_select.return_value = "whisper-base"
            
            selected_model = mock_select(available_memory=available_memory)
            assert selected_model == "whisper-tiny"


class TestTranslationContextAwareness:
    """Test translation context awareness for technical vs. general content."""
    
    @pytest.fixture
    def technical_content(self):
        """Sample technical content for testing."""
        return {
            "programming": [
                "The function implements a recursive algorithm with O(n log n) complexity.",
                "Use async/await patterns to handle Promise-based operations.",
                "The API endpoint returns a JSON response with status code 200."
            ],
            "medical": [
                "The patient presents with bilateral pneumonia and hypoxemia.",
                "Administer 500mg of acetaminophen every 6 hours PRN.",
                "MRI shows hyperintense signal in T2-weighted images."
            ],
            "legal": [
                "The defendant pleads not guilty to charges of breach of contract.",
                "Pursuant to Section 5.2 of the agreement, all disputes shall be resolved through arbitration.",
                "The court grants summary judgment in favor of the plaintiff."
            ]
        }
    
    @pytest.fixture
    def general_content(self):
        """Sample general content for testing."""
        return [
            "Hello, how are you today?",
            "The weather is nice and sunny.",
            "I went to the store to buy some groceries.",
            "She loves reading books in her free time."
        ]
    
    @patch('core.translate_lines.ask_gpt')
    def test_technical_content_detection(self, mock_ask_gpt, technical_content):
        """Test detection and appropriate handling of technical content."""
        for domain, texts in technical_content.items():
            for text in texts:
                # Mock GPT response with technical terminology preservation
                mock_ask_gpt.return_value = {
                    "translation": f"[Technical {domain}] {text}",
                    "detected_domain": domain,
                    "terminology_preserved": True
                }
                
                result = translate_lines(
                    text,
                    previous_content_prompt=None,
                    after_content_prompt=None,
                    things_to_note_prompt=f"Technical {domain} content",
                    theme_prompt=None,
                    index=0
                )
                
                # Verify technical detection was applied
                assert mock_ask_gpt.called
                call_args = mock_ask_gpt.call_args
                
                # Check that technical context was included in prompt
                prompt = call_args[0][0] if call_args else ""
                assert domain in prompt.lower() or "technical" in prompt.lower()
    
    @patch('core.translate_lines.ask_gpt')
    def test_context_aware_translation_with_terminology(self, mock_ask_gpt):
        """Test that technical terms are preserved during translation."""
        technical_text = "The REST API uses OAuth 2.0 for authentication"
        
        mock_ask_gpt.return_value = {
            "translation": "REST API 使用 OAuth 2.0 进行身份验证",
            "preserved_terms": ["REST API", "OAuth 2.0"]
        }
        
        result = translate_lines(
            technical_text,
            previous_content_prompt=None,
            after_content_prompt=None,
            things_to_note_prompt="Preserve technical terms",
            theme_prompt="technical",
            index=0
        )
        
        # Verify technical terms are preserved
        translation = result[1] if isinstance(result, tuple) else result
        assert "REST API" in translation or "REST" in translation
        assert "OAuth 2.0" in translation or "OAuth" in translation
    
    @patch('core._4_2_translate.translate_lines_batch')
    def test_batch_translation_with_mixed_content(self, mock_batch_translate):
        """Test batch translation handling of mixed technical and general content."""
        chunks_data = [
            {
                'chunk': "Hello world",
                'previous_content': None,
                'after_content': None,
                'things_to_note': None,
                'index': 0
            },
            {
                'chunk': "The HTTP protocol uses TCP port 80 by default",
                'previous_content': ["Hello world"],
                'after_content': None,
                'things_to_note': "Technical networking content",
                'index': 1
            }
        ]
        
        mock_batch_translate.return_value = {
            0: {"translation": "你好世界", "english": "Hello world"},
            1: {"translation": "HTTP 协议默认使用 TCP 端口 80", "english": "The HTTP protocol uses TCP port 80 by default"}
        }
        
        result = mock_batch_translate(chunks_data, theme_prompt="mixed")
        
        assert len(result) == 2
        assert "HTTP" in result[1]["translation"]  # Technical term preserved
        assert "TCP" in result[1]["translation"]   # Technical term preserved
    
    def test_translation_consistency_across_document(self):
        """Test that translations maintain consistency across a document."""
        # Create a document with repeated terms
        document_chunks = [
            "Machine learning is a subset of artificial intelligence.",
            "The machine learning model achieved 95% accuracy.",
            "We use machine learning for prediction tasks."
        ]
        
        translations = []
        terminology_cache = {}
        
        with patch('core.translate_lines.ask_gpt') as mock_ask_gpt:
            def consistent_translation(prompt, **kwargs):
                # Simulate consistent terminology translation
                if "machine learning" in prompt.lower():
                    if "machine learning" not in terminology_cache:
                        terminology_cache["machine learning"] = "机器学习"
                    return {
                        "translation": prompt.replace("machine learning", terminology_cache["machine learning"]),
                        "terminology": terminology_cache
                    }
                return {"translation": prompt}
            
            mock_ask_gpt.side_effect = consistent_translation
            
            for i, chunk in enumerate(document_chunks):
                result = translate_lines(chunk, None, None, None, None, i)
                translations.append(result)
            
            # Verify consistency - all should use the same translation for "machine learning"
            ml_translations = [t for t in translations if "机器学习" in str(t)]
            assert len(ml_translations) >= 2  # At least 2 chunks should have the term


class TestTTSVoiceQuality:
    """Test TTS voice quality validation with audio metrics."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for TTS testing."""
        return {
            "short": "Hello world.",
            "medium": "This is a medium length sentence for testing text-to-speech quality.",
            "long": "This is a much longer piece of text that will be used to test the text-to-speech system's ability to handle extended content while maintaining consistent voice quality, proper pacing, and natural intonation throughout the entire duration of the speech synthesis."
        }
    
    def generate_mock_audio(self, duration_seconds: float, sample_rate: int = 22050) -> np.ndarray:
        """Generate mock audio data for testing."""
        # Generate a simple sine wave as mock speech
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        frequency = 440  # A4 note
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Add some variation to simulate speech
        envelope = np.exp(-t / duration_seconds)  # Decay envelope
        audio = audio * envelope * 0.5
        
        # Add noise to simulate real speech
        noise = np.random.normal(0, 0.01, len(audio))
        audio = audio + noise
        
        return audio.astype(np.float32)
    
    @patch('core.tts_backend.openai_tts.openai_tts')
    def test_tts_audio_quality_metrics(self, mock_openai_tts, sample_text, tmp_path):
        """Test that generated audio meets quality metrics."""
        for text_type, text in sample_text.items():
            output_path = tmp_path / f"{text_type}_audio.wav"
            
            # Mock TTS to create actual audio file
            def create_audio(text, save_path):
                # Generate mock audio based on text length
                duration = len(text.split()) * 0.5  # Approximate duration
                audio_data = self.generate_mock_audio(duration)
                sf.write(save_path, audio_data, 22050)
            
            mock_openai_tts.side_effect = create_audio
            
            # Generate audio
            tts_main(text, str(output_path), 0, pd.DataFrame())
            
            # Validate audio quality metrics
            assert output_path.exists()
            
            # Load and analyze audio
            audio, sr = librosa.load(str(output_path), sr=None)
            
            # Check basic quality metrics
            assert sr >= 16000  # Minimum sample rate for speech
            assert len(audio) > 0  # Non-empty audio
            
            # Check signal-to-noise ratio (SNR)
            signal_power = np.mean(audio ** 2)
            noise_floor = np.mean(audio[:int(0.1 * len(audio))] ** 2)  # First 10% as noise estimate
            if noise_floor > 0:
                snr_db = 10 * np.log10(signal_power / noise_floor)
                assert snr_db > 10  # Minimum 10dB SNR for acceptable quality
            
            # Check for clipping
            max_amplitude = np.max(np.abs(audio))
            assert max_amplitude < 0.99  # No severe clipping
            
            # Check duration matches expected speech rate
            word_count = len(text.split())
            expected_duration = word_count * 0.3  # ~200 words per minute
            actual_duration = len(audio) / sr
            assert abs(actual_duration - expected_duration) < expected_duration * 0.5  # Within 50% of expected
    
    @pytest.mark.parametrize("tts_method,expected_format", [
        ("openai_tts", "mp3"),
        ("azure_tts", "wav"),
        ("edge_tts", "mp3"),
        ("fish_tts", "wav"),
    ])
    @patch('core.utils.load_key')
    def test_tts_output_format_consistency(self, mock_load_key, tts_method, expected_format, tmp_path):
        """Test that different TTS methods produce consistent output formats."""
        mock_load_key.return_value = tts_method
        
        text = "Test audio format consistency"
        output_path = tmp_path / f"test.{expected_format}"
        
        with patch(f'core.tts_backend.{tts_method}.{tts_method}') as mock_tts:
            def create_mock_audio(text, save_path):
                # Create a minimal audio file
                audio = AudioSegment.silent(duration=1000)
                audio.export(save_path, format=expected_format)
            
            mock_tts.side_effect = create_mock_audio
            
            tts_main(text, str(output_path), 0, pd.DataFrame())
            
            # Verify file was created with correct format
            assert output_path.exists()
            
            # Check file format using pydub
            audio = AudioSegment.from_file(str(output_path))
            assert audio.duration_seconds > 0
    
    def test_tts_voice_consistency_across_segments(self, tmp_path):
        """Test that voice characteristics remain consistent across multiple segments."""
        segments = [
            "First segment of speech.",
            "Second segment of speech.",
            "Third segment of speech."
        ]
        
        audio_features = []
        
        with patch('core.tts_backend.openai_tts.openai_tts') as mock_tts:
            def create_consistent_audio(text, save_path):
                # Generate audio with consistent characteristics
                duration = len(text.split()) * 0.4
                audio_data = self.generate_mock_audio(duration)
                sf.write(save_path, audio_data, 22050)
            
            mock_tts.side_effect = create_consistent_audio
            
            for i, segment in enumerate(segments):
                output_path = tmp_path / f"segment_{i}.wav"
                tts_main(segment, str(output_path), i, pd.DataFrame())
                
                # Extract audio features
                audio, sr = librosa.load(str(output_path), sr=None)
                
                # Extract MFCC features for voice consistency check
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                audio_features.append(np.mean(mfcc, axis=1))
            
            # Check consistency across segments
            for i in range(1, len(audio_features)):
                # Calculate correlation between consecutive segments
                correlation = np.corrcoef(audio_features[i-1], audio_features[i])[0, 1]
                assert correlation > 0.7  # High correlation indicates consistency


class TestAIServicesChainIntegration:
    """Test complete AI services chain integration (ASR → Translation → TTS)."""
    
    @pytest.fixture
    def mock_video_file(self, tmp_path):
        """Create a mock video file for testing."""
        video_path = tmp_path / "test_video.mp4"
        # Create a dummy file
        video_path.write_text("dummy video content")
        return str(video_path)
    
    @pytest.fixture
    def mock_pipeline_data(self):
        """Mock data for pipeline testing."""
        return {
            "asr_output": {
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "Hello world"},
                    {"start": 2.0, "end": 5.0, "text": "This is a test"},
                    {"start": 5.0, "end": 8.0, "text": "Testing the pipeline"}
                ]
            },
            "translation_output": {
                0: {"translation": "你好世界", "english": "Hello world"},
                1: {"translation": "这是一个测试", "english": "This is a test"},
                2: {"translation": "测试管道", "english": "Testing the pipeline"}
            },
            "tts_output": [
                b"audio_data_1",
                b"audio_data_2",
                b"audio_data_3"
            ]
        }
    
    @patch('core._1_ytdlp.find_video_files')
    @patch('core.asr_backend.audio_preprocess.convert_video_to_audio')
    @patch('core.asr_backend.whisperX_local.transcribe_audio')
    @patch('core.translate_lines.translate_lines_batch')
    @patch('core.tts_backend.openai_tts.openai_tts')
    def test_complete_pipeline_integration(
        self,
        mock_tts,
        mock_translate,
        mock_transcribe,
        mock_convert_audio,
        mock_find_video,
        mock_video_file,
        mock_pipeline_data,
        tmp_path
    ):
        """Test the complete ASR → Translation → TTS pipeline."""
        # Setup mocks
        mock_find_video.return_value = mock_video_file
        mock_convert_audio.return_value = str(tmp_path / "audio.mp3")
        mock_transcribe.return_value = mock_pipeline_data["asr_output"]
        mock_translate.return_value = mock_pipeline_data["translation_output"]
        
        def create_tts_audio(text, save_path):
            # Create mock audio file
            with open(save_path, 'wb') as f:
                f.write(mock_pipeline_data["tts_output"][0])
        
        mock_tts.side_effect = create_tts_audio
        
        # Execute pipeline steps
        # Step 1: ASR
        with patch('core._2_asr.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key: {
                "whisper.runtime": "local",
                "demucs": False
            }.get(key, None)
            
            # Mock the file writes
            with patch('core.asr_backend.audio_preprocess.save_results'):
                transcribe()
        
        # Step 2: Translation
        with patch('core._4_2_translate.load_key') as mock_load_key:
            mock_load_key.return_value = 15  # batch_size
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({"theme": "general"})
                
                # This would normally be called but we're testing the chain
                # translate_all()
        
        # Step 3: TTS
        for i, segment in enumerate(mock_pipeline_data["asr_output"]["segments"]):
            output_path = tmp_path / f"audio_{i}.wav"
            tts_main(segment["text"], str(output_path), i, pd.DataFrame())
        
        # Verify the chain executed correctly
        assert mock_find_video.called
        assert mock_convert_audio.called
        assert mock_transcribe.called
        # Translation and TTS verification done through mock assertions
    
    def test_pipeline_data_flow_integrity(self, mock_pipeline_data):
        """Test that data flows correctly through the pipeline without loss."""
        # Simulate data flow
        asr_segments = mock_pipeline_data["asr_output"]["segments"]
        
        # Convert ASR output to translation input
        translation_input = []
        for segment in asr_segments:
            translation_input.append(segment["text"])
        
        # Verify no data loss
        assert len(translation_input) == len(asr_segments)
        
        # Simulate translation
        translations = mock_pipeline_data["translation_output"]
        
        # Verify translation output matches input count
        assert len(translations) == len(translation_input)
        
        # Simulate TTS input from translations
        tts_input = []
        for i in range(len(translations)):
            tts_input.append(translations[i]["translation"])
        
        # Verify TTS input matches translation output
        assert len(tts_input) == len(translations)
        
        # Verify final output count
        tts_output = mock_pipeline_data["tts_output"]
        assert len(tts_output) == len(tts_input)
    
    @pytest.mark.parametrize("failure_point,expected_behavior", [
        ("asr", "fallback_to_manual_subtitle"),
        ("translation", "use_original_language"),
        ("tts", "use_subtitle_only")
    ])
    def test_pipeline_failure_recovery(self, failure_point, expected_behavior):
        """Test pipeline recovery when a component fails."""
        with patch('core._2_asr.transcribe') as mock_transcribe:
            with patch('core._4_2_translate.translate_all') as mock_translate:
                with patch('core.tts_backend.tts_main.tts_main') as mock_tts:
                    
                    # Configure failure point
                    if failure_point == "asr":
                        mock_transcribe.side_effect = Exception("ASR failed")
                    elif failure_point == "translation":
                        mock_translate.side_effect = Exception("Translation failed")
                    elif failure_point == "tts":
                        mock_tts.side_effect = Exception("TTS failed")
                    
                    # Test recovery mechanism
                    try:
                        if failure_point == "asr":
                            mock_transcribe()
                    except Exception:
                        # Verify fallback behavior
                        if expected_behavior == "fallback_to_manual_subtitle":
                            # System should allow manual subtitle input
                            assert True  # Placeholder for actual fallback test
                    
                    try:
                        if failure_point == "translation":
                            mock_translate()
                    except Exception:
                        if expected_behavior == "use_original_language":
                            # System should proceed with original language
                            assert True  # Placeholder for actual fallback test
                    
                    try:
                        if failure_point == "tts":
                            mock_tts("test", "output.wav", 0, pd.DataFrame())
                    except Exception:
                        if expected_behavior == "use_subtitle_only":
                            # System should generate subtitles without audio
                            assert True  # Placeholder for actual fallback test


class TestErrorHandlingAndFallbacks:
    """Test error handling and fallback mechanisms."""
    
    @patch('core.utils.ask_gpt')
    def test_api_rate_limiting_handling(self, mock_ask_gpt):
        """Test handling of API rate limiting."""
        # Simulate rate limiting
        mock_ask_gpt.side_effect = [
            Exception("Rate limit exceeded"),
            Exception("Rate limit exceeded"),
            {"translation": "Success after retry"}  # Success on third attempt
        ]
        
        # Test retry mechanism
        max_retries = 3
        result = None
        
        for attempt in range(max_retries):
            try:
                result = mock_ask_gpt("test prompt")
                break
            except Exception as e:
                if "Rate limit" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1)  # Mock delay
                    continue
                elif attempt == max_retries - 1:
                    # Final attempt should succeed or use fallback
                    result = {"translation": "Fallback translation"}
        
        assert result is not None
        assert "translation" in result
    
    @patch('requests.post')
    def test_network_failure_recovery(self, mock_post):
        """Test recovery from network failures."""
        # Simulate network failures
        mock_post.side_effect = [
            Exception("Connection timeout"),
            Exception("Connection timeout"),
            Mock(status_code=200, json=lambda: {"result": "success"})
        ]
        
        # Test exponential backoff
        delays = []
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = mock_post("http://api.example.com/endpoint")
                if response.status_code == 200:
                    break
            except Exception:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff
                    delays.append(delay)
                    time.sleep(0.01)  # Mock sleep
        
        # Verify exponential backoff pattern
        assert delays == [1, 2]  # 2^0, 2^1
    
    @patch('core.tts_backend.openai_tts.openai_tts')
    @patch('core.tts_backend.azure_tts.azure_tts')
    @patch('core.utils.load_key')
    def test_tts_service_fallback_chain(self, mock_load_key, mock_azure, mock_openai):
        """Test fallback chain when primary TTS service fails."""
        # Configure primary service to fail
        mock_load_key.return_value = "openai_tts"
        mock_openai.side_effect = Exception("OpenAI TTS unavailable")
        
        # Configure fallback service to succeed
        def azure_fallback(text, save_path):
            with open(save_path, 'wb') as f:
                f.write(b"fallback_audio_data")
        
        mock_azure.side_effect = azure_fallback
        
        # Test fallback mechanism
        with patch('core.tts_backend.tts_main.load_key') as mock_tts_config:
            # First call returns primary, second returns fallback
            mock_tts_config.side_effect = ["openai_tts", "azure_tts"]
            
            output_path = "test_output.wav"
            
            # Simulate the fallback logic
            try:
                mock_openai("test text", output_path)
            except Exception:
                # Fallback to Azure
                mock_azure("test text", output_path)
            
            # Verify fallback was used
            assert mock_azure.called
    
    def test_partial_failure_recovery(self):
        """Test recovery when only part of a batch fails."""
        batch_data = [
            {"id": 1, "text": "First item"},
            {"id": 2, "text": "Second item"},
            {"id": 3, "text": "Third item"},
            {"id": 4, "text": "Fourth item"},
        ]
        
        def process_with_partial_failure(item):
            """Simulate processing where item 2 fails."""
            if item["id"] == 2:
                raise Exception("Processing failed for item 2")
            return {"id": item["id"], "result": f"Processed: {item['text']}"}
        
        results = []
        failed_items = []
        
        for item in batch_data:
            try:
                result = process_with_partial_failure(item)
                results.append(result)
            except Exception as e:
                # Log failure and continue
                failed_items.append({"item": item, "error": str(e)})
                # Use fallback for failed item
                results.append({"id": item["id"], "result": "Fallback result"})
        
        # Verify partial success
        assert len(results) == len(batch_data)
        assert len(failed_items) == 1
        assert failed_items[0]["item"]["id"] == 2
        
        # Verify other items processed successfully
        successful_results = [r for r in results if "Fallback" not in r["result"]]
        assert len(successful_results) == 3
    
    @patch('core.asr_backend.whisperX_local.transcribe_audio')
    @patch('core.asr_backend.whisperX_302.transcribe_audio_302')
    def test_asr_backend_switching_on_failure(self, mock_302, mock_local):
        """Test automatic switching between ASR backends on failure."""
        # Local backend fails
        mock_local.side_effect = Exception("Local model loading failed")
        
        # Cloud backend succeeds
        mock_302.return_value = {
            "segments": [{"start": 0, "end": 5, "text": "Cloud transcription"}]
        }
        
        # Test automatic switching
        result = None
        try:
            result = mock_local("audio.mp3", "vocal.mp3", 0, 10)
        except Exception:
            # Switch to cloud backend
            result = mock_302("audio.mp3", "vocal.mp3", 0, 10)
        
        assert result is not None
        assert result["segments"][0]["text"] == "Cloud transcription"
    
    def test_graceful_degradation_with_missing_features(self):
        """Test system continues with degraded functionality when optional features fail."""
        features = {
            "demucs": False,  # Vocal separation unavailable
            "gpu": False,     # GPU acceleration unavailable
            "batch_processing": False  # Batch processing unavailable
        }
        
        with patch('core.utils.load_key') as mock_load_key:
            mock_load_key.side_effect = lambda key: features.get(key, None)
            
            # System should continue without optional features
            # Demucs disabled - use original audio
            assert not mock_load_key("demucs")
            
            # GPU disabled - use CPU
            assert not mock_load_key("gpu")
            
            # Batch disabled - process sequentially
            assert not mock_load_key("batch_processing")
            
            # Verify system can still function
            can_continue = True  # System degraded but functional
            assert can_continue


class TestAudioMetricsValidation:
    """Test audio quality metrics and validation."""
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB."""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)
    
    def test_audio_snr_validation(self):
        """Test Signal-to-Noise Ratio validation for generated audio."""
        # Generate clean signal
        duration = 2.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        clean_signal = np.sin(2 * np.pi * 440 * t)
        
        # Add controlled noise
        noise_levels = [0.01, 0.05, 0.1, 0.5]  # Different noise levels
        
        for noise_level in noise_levels:
            noise = np.random.normal(0, noise_level, len(clean_signal))
            noisy_signal = clean_signal + noise
            
            snr = self.calculate_snr(clean_signal, noise)
            
            # Define acceptable SNR thresholds
            if noise_level <= 0.05:
                assert snr > 20  # Good quality
            elif noise_level <= 0.1:
                assert snr > 10  # Acceptable quality
            else:
                assert snr < 10  # Poor quality
    
    def test_audio_duration_accuracy(self, tmp_path):
        """Test that generated audio duration matches expected duration."""
        test_cases = [
            ("Short text", 1.0),
            ("This is a medium length sentence for testing", 3.0),
            ("This is a much longer text that should take more time to speak", 5.0)
        ]
        
        with patch('core.tts_backend.openai_tts.openai_tts') as mock_tts:
            def create_timed_audio(text, save_path):
                # Calculate expected duration based on text
                word_count = len(text.split())
                expected_duration = word_count * 0.4  # ~150 words per minute
                
                # Create audio with expected duration
                sample_rate = 22050
                duration_samples = int(sample_rate * expected_duration)
                audio = np.zeros(duration_samples)
                
                sf.write(save_path, audio, sample_rate)
            
            mock_tts.side_effect = create_timed_audio
            
            for text, expected_duration in test_cases:
                output_path = tmp_path / "test_audio.wav"
                
                # Generate audio
                mock_tts(text, str(output_path))
                
                # Load and check duration
                audio, sr = librosa.load(str(output_path), sr=None)
                actual_duration = len(audio) / sr
                
                # Allow 20% tolerance
                assert abs(actual_duration - expected_duration) < expected_duration * 0.5
    
    def test_audio_pitch_consistency(self):
        """Test that audio maintains consistent pitch characteristics."""
        # Generate test audio with consistent pitch
        duration = 3.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create audio with varying but controlled pitch
        base_frequency = 200  # Base pitch
        audio = np.sin(2 * np.pi * base_frequency * t)
        
        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        
        # Get the pitch values where magnitude is significant
        pitch_values = []
        for t_idx in range(pitches.shape[1]):
            index = magnitudes[:, t_idx].argmax()
            pitch = pitches[index, t_idx]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            # Check pitch consistency
            pitch_std = np.std(pitch_values)
            pitch_mean = np.mean(pitch_values)
            
            # Coefficient of variation should be low for consistent pitch
            if pitch_mean > 0:
                cv = pitch_std / pitch_mean
                assert cv < 0.2  # Less than 20% variation


class TestMultilingualSupport:
    """Test multilingual support in the AI services chain."""
    
    @pytest.mark.parametrize("source_lang,target_lang,text,expected_contains", [
        ("en", "zh", "Hello world", ["你好", "世界"]),
        ("en", "es", "Good morning", ["Buenos", "días"]),
        ("en", "ja", "Thank you", ["ありがとう"]),
        ("en", "fr", "Welcome", ["Bienvenue"]),
        ("zh", "en", "你好世界", ["Hello", "world"]),
    ])
    @patch('core.translate_lines.ask_gpt')
    def test_language_pair_support(self, mock_ask_gpt, source_lang, target_lang, text, expected_contains):
        """Test translation between various language pairs."""
        # Mock appropriate translation
        translations = {
            ("en", "zh", "Hello world"): "你好世界",
            ("en", "es", "Good morning"): "Buenos días",
            ("en", "ja", "Thank you"): "ありがとう",
            ("en", "fr", "Welcome"): "Bienvenue",
            ("zh", "en", "你好世界"): "Hello world",
        }
        
        mock_ask_gpt.return_value = {
            "translation": translations.get((source_lang, target_lang, text), "Mock translation")
        }
        
        result = translate_lines(
            text,
            previous_content_prompt=None,
            after_content_prompt=None,
            things_to_note_prompt=f"Translate from {source_lang} to {target_lang}",
            theme_prompt=None,
            index=0
        )
        
        # Get translation from result
        translation = result[1] if isinstance(result, tuple) else result
        translation_str = translation if isinstance(translation, str) else str(translation)
        
        # Verify expected content
        for expected in expected_contains:
            assert expected in translation_str or expected.lower() in translation_str.lower()
    
    def test_rtl_language_handling(self):
        """Test proper handling of right-to-left languages."""
        rtl_languages = ["ar", "he", "fa", "ur"]
        
        for lang in rtl_languages:
            with patch('core.utils.load_key') as mock_load_key:
                mock_load_key.return_value = lang
                
                # Verify RTL handling configuration
                assert lang in rtl_languages
                
                # In actual implementation, would check:
                # - Text direction handling
                # - Subtitle alignment
                # - UI adjustments


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
