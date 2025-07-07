# Unit Tests for Translation System
# Tests core/translate_lines.py

import pytest
import json
from unittest.mock import patch, Mock, MagicMock

from core.translate_lines import (
    translate_lines, get_valid_line_jsons, 
    batch_translate_chunk, validate_translation_result,
    format_translation_prompt, extract_terminology
)

class TestTranslateLines:
    """Test suite for translation functionality"""
    
    @pytest.fixture
    def sample_lines(self):
        """Sample line data for testing"""
        return [
            {
                "line_id": 1,
                "start_time": 0.0,
                "end_time": 2.5,
                "text": "Hello world, this is a test.",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "Hello"},
                    {"start": 0.6, "end": 1.0, "word": "world"},
                    {"start": 1.2, "end": 1.4, "word": "this"},
                    {"start": 1.5, "end": 1.7, "word": "is"},
                    {"start": 1.8, "end": 1.9, "word": "a"},
                    {"start": 2.0, "end": 2.4, "word": "test"}
                ]
            },
            {
                "line_id": 2,
                "start_time": 2.5,
                "end_time": 5.0,
                "text": "Welcome to VideoLingo translation system.",
                "words": [
                    {"start": 2.5, "end": 2.8, "word": "Welcome"},
                    {"start": 2.9, "end": 3.1, "word": "to"},
                    {"start": 3.2, "end": 3.7, "word": "VideoLingo"},
                    {"start": 3.8, "end": 4.3, "word": "translation"},
                    {"start": 4.4, "end": 4.9, "word": "system"}
                ]
            }
        ]
    
    @pytest.fixture
    def mock_translation_response(self):
        """Mock translation response from LLM"""
        return {
            "translations": [
                {
                    "line_id": 1,
                    "faithful_translation": "你好世界，这是一个测试。",
                    "expressive_translation": "你好世界，这是个测试。",
                    "confidence": 0.95
                },
                {
                    "line_id": 2,
                    "faithful_translation": "欢迎来到VideoLingo翻译系统。",
                    "expressive_translation": "欢迎使用VideoLingo翻译系统。",
                    "confidence": 0.92
                }
            ]
        }
    
    def test_translate_lines_success(self, sample_lines, mock_translation_response, temp_config_dir):
        """Test successful translation of lines"""
        with patch('core.translate_lines.ask_gpt_with_result_cache', return_value=mock_translation_response), \
             patch('core.translate_lines.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'translation.target_language': 'Chinese (Simplified)',
                'translation.chunk_size': 1000,
                'translation.batch_size': 10
            }.get(key, default)
            
            result = translate_lines(sample_lines)
            
            assert len(result) == 2
            assert result[0]['line_id'] == 1
            assert result[0]['faithful_translation'] == "你好世界，这是一个测试。"
            assert result[0]['expressive_translation'] == "你好世界，这是个测试。"
            assert result[1]['line_id'] == 2
    
    def test_translate_lines_empty_input(self):
        """Test translation with empty input"""
        result = translate_lines([])
        assert result == []
    
    def test_translate_lines_batch_processing(self, temp_config_dir):
        """Test batch processing with large number of lines"""
        # Create many lines to test batching
        many_lines = []
        for i in range(25):  # More than typical batch size
            many_lines.append({
                "line_id": i,
                "start_time": i * 2.0,
                "end_time": (i + 1) * 2.0,
                "text": f"This is line number {i}.",
                "words": [{"start": i * 2.0, "end": (i + 1) * 2.0, "word": f"line{i}"}]
            })
        
        mock_response = {
            "translations": [
                {
                    "line_id": i,
                    "faithful_translation": f"这是第{i}行。",
                    "expressive_translation": f"这是第{i}行。",
                    "confidence": 0.9
                } for i in range(25)
            ]
        }
        
        with patch('core.translate_lines.ask_gpt_with_result_cache', return_value=mock_response), \
             patch('core.translate_lines.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'translation.target_language': 'Chinese (Simplified)',
                'translation.chunk_size': 1000,
                'translation.batch_size': 10
            }.get(key, default)
            
            result = translate_lines(many_lines)
            
            assert len(result) == 25
            # Verify all lines were translated
            for i, translation in enumerate(result):
                assert translation['line_id'] == i
                assert translation['faithful_translation'] == f"这是第{i}行。"
    
    def test_get_valid_line_jsons(self, sample_lines):
        """Test extraction of valid line JSONs"""
        # Add some invalid lines
        invalid_lines = sample_lines + [
            {"invalid": "line", "missing_required_fields": True},
            {"line_id": 3, "text": "Valid line", "start_time": 5.0, "end_time": 6.0, "words": []}
        ]
        
        valid_lines = get_valid_line_jsons(invalid_lines)
        
        # Should filter out invalid lines
        assert len(valid_lines) == 3  # 2 original + 1 valid added
        for line in valid_lines:
            assert 'line_id' in line
            assert 'text' in line
            assert 'start_time' in line
            assert 'end_time' in line
    
    def test_batch_translate_chunk_success(self, sample_lines, mock_translation_response):
        """Test successful batch translation of a chunk"""
        with patch('core.translate_lines.ask_gpt_with_result_cache', return_value=mock_translation_response):
            result = batch_translate_chunk(sample_lines, "Chinese (Simplified)")
            
            assert len(result) == 2
            assert all('faithful_translation' in item for item in result)
            assert all('expressive_translation' in item for item in result)
    
    def test_batch_translate_chunk_api_error(self, sample_lines):
        """Test batch translation with API error"""
        with patch('core.translate_lines.ask_gpt_with_result_cache', side_effect=Exception("API Error")):
            with pytest.raises(Exception):
                batch_translate_chunk(sample_lines, "Chinese (Simplified)")
    
    def test_validate_translation_result_valid(self, mock_translation_response):
        """Test validation of valid translation result"""
        is_valid, errors = validate_translation_result(mock_translation_response, [1, 2])
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_translation_result_missing_translations(self):
        """Test validation with missing translation key"""
        invalid_response = {"wrong_key": []}
        is_valid, errors = validate_translation_result(invalid_response, [1, 2])
        
        assert is_valid is False
        assert any("missing 'translations'" in error for error in errors)
    
    def test_validate_translation_result_missing_line_ids(self, mock_translation_response):
        """Test validation with missing line IDs"""
        # Remove one translation
        mock_translation_response['translations'] = mock_translation_response['translations'][:1]
        
        is_valid, errors = validate_translation_result(mock_translation_response, [1, 2])
        
        assert is_valid is False
        assert any("Missing translation for line_id: 2" in error for error in errors)
    
    def test_validate_translation_result_invalid_structure(self):
        """Test validation with invalid translation structure"""
        invalid_response = {
            "translations": [
                {"line_id": 1},  # Missing required fields
                {"faithful_translation": "test"}  # Missing line_id
            ]
        }
        
        is_valid, errors = validate_translation_result(invalid_response, [1, 2])
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_format_translation_prompt(self, sample_lines):
        """Test translation prompt formatting"""
        target_language = "Chinese (Simplified)"
        context = "This is a technical video about programming."
        
        prompt = format_translation_prompt(sample_lines, target_language, context)
        
        assert target_language in prompt
        assert "Hello world" in prompt
        assert "VideoLingo" in prompt
        if context:
            assert context in prompt
        
        # Should contain proper JSON structure
        assert '"line_id"' in prompt
        assert '"text"' in prompt
    
    def test_format_translation_prompt_no_context(self, sample_lines):
        """Test prompt formatting without context"""
        prompt = format_translation_prompt(sample_lines, "Spanish")
        
        assert "Spanish" in prompt
        assert prompt is not None
        assert len(prompt) > 0
    
    def test_extract_terminology_from_lines(self, sample_lines):
        """Test terminology extraction from lines"""
        # Add technical terms
        technical_lines = sample_lines + [
            {
                "line_id": 3,
                "start_time": 5.0,
                "end_time": 7.0,
                "text": "API authentication and OAuth implementation.",
                "words": []
            }
        ]
        
        terminology = extract_terminology(technical_lines)
        
        assert isinstance(terminology, list)
        # Should extract technical terms
        technical_terms = ['API', 'OAuth', 'VideoLingo']
        extracted_terms = [term['term'] for term in terminology]
        
        for term in technical_terms:
            if term in ' '.join([line['text'] for line in technical_lines]):
                assert any(term.lower() in extracted_term.lower() for extracted_term in extracted_terms)
    
    def test_translation_with_context_integration(self, sample_lines, temp_config_dir):
        """Test translation with context integration"""
        context = "This video is about software development and programming concepts."
        
        mock_response = {
            "translations": [
                {
                    "line_id": 1,
                    "faithful_translation": "你好世界，这是一个测试。",
                    "expressive_translation": "你好世界，这是个测试。",
                    "confidence": 0.95,
                    "context_used": True
                },
                {
                    "line_id": 2,
                    "faithful_translation": "欢迎来到VideoLingo翻译系统。",
                    "expressive_translation": "欢迎使用VideoLingo翻译系统。",
                    "confidence": 0.92,
                    "context_used": True
                }
            ]
        }
        
        with patch('core.translate_lines.ask_gpt_with_result_cache', return_value=mock_response) as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'translation.target_language': 'Chinese (Simplified)',
                'translation.chunk_size': 1000,
                'translation.use_context': True
            }.get(key, default)
            
            result = translate_lines(sample_lines, context=context)
            
            # Verify context was included in the prompt
            call_args = mock_ask_gpt.call_args[0]
            assert context in call_args[0]  # First argument should be the prompt
            
            assert len(result) == 2
            assert all('faithful_translation' in item for item in result)
    
    def test_translation_error_recovery(self, sample_lines, temp_config_dir):
        """Test translation error recovery mechanisms"""
        # Mock partial failure - some translations succeed, others fail
        partial_response = {
            "translations": [
                {
                    "line_id": 1,
                    "faithful_translation": "你好世界，这是一个测试。",
                    "expressive_translation": "你好世界，这是个测试。",
                    "confidence": 0.95
                }
                # Missing translation for line_id 2
            ]
        }
        
        with patch('core.translate_lines.ask_gpt_with_result_cache', return_value=partial_response), \
             patch('core.translate_lines.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'translation.target_language': 'Chinese (Simplified)',
                'translation.chunk_size': 1000,
                'translation.retry_failed': True
            }.get(key, default)
            
            # Should handle partial failures gracefully
            with pytest.raises(Exception):  # Or handle gracefully based on implementation
                translate_lines(sample_lines)
    
    def test_translation_performance_monitoring(self, sample_lines, temp_config_dir):
        """Test translation performance monitoring"""
        import time
        
        def slow_mock_response(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow response
            return {
                "translations": [
                    {
                        "line_id": line['line_id'],
                        "faithful_translation": f"Translation for {line['text']}",
                        "expressive_translation": f"Expressive translation for {line['text']}",
                        "confidence": 0.9
                    } for line in sample_lines
                ]
            }
        
        with patch('core.translate_lines.ask_gpt_with_result_cache', side_effect=slow_mock_response), \
             patch('core.translate_lines.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'translation.target_language': 'Chinese (Simplified)',
                'translation.chunk_size': 1000
            }.get(key, default)
            
            start_time = time.time()
            result = translate_lines(sample_lines)
            duration = time.time() - start_time
            
            # Should complete in reasonable time
            assert duration < 5.0  # Adjust based on expected performance
            assert len(result) == len(sample_lines)
    
    @pytest.mark.parametrize("target_language,expected_in_prompt", [
        ("Chinese (Simplified)", "Chinese (Simplified)"),
        ("Spanish", "Spanish"),
        ("French", "French"),
        ("Japanese", "Japanese"),
        ("German", "German")
    ])
    def test_translation_language_support(self, sample_lines, target_language, expected_in_prompt):
        """Test translation support for different languages"""
        mock_response = {
            "translations": [
                {
                    "line_id": line['line_id'],
                    "faithful_translation": f"Faithful in {target_language}",
                    "expressive_translation": f"Expressive in {target_language}",
                    "confidence": 0.9
                } for line in sample_lines
            ]
        }
        
        with patch('core.translate_lines.ask_gpt_with_result_cache', return_value=mock_response) as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'translation.target_language': target_language,
                'translation.chunk_size': 1000
            }.get(key, default)
            
            result = translate_lines(sample_lines)
            
            # Verify language was included in prompt
            call_args = mock_ask_gpt.call_args[0]
            assert expected_in_prompt in call_args[0]
            
            assert len(result) == len(sample_lines)
    
    def test_translation_confidence_handling(self, sample_lines, temp_config_dir):
        """Test handling of translation confidence scores"""
        low_confidence_response = {
            "translations": [
                {
                    "line_id": 1,
                    "faithful_translation": "低置信度翻译",
                    "expressive_translation": "低置信度翻译",
                    "confidence": 0.3  # Low confidence
                },
                {
                    "line_id": 2,
                    "faithful_translation": "高置信度翻译",
                    "expressive_translation": "高置信度翻译",
                    "confidence": 0.95  # High confidence
                }
            ]
        }
        
        with patch('core.translate_lines.ask_gpt_with_result_cache', return_value=low_confidence_response), \
             patch('core.translate_lines.load_key') as mock_load_key:
            
            mock_load_key.side_effect = lambda key, default=None: {
                'translation.target_language': 'Chinese (Simplified)',
                'translation.chunk_size': 1000,
                'translation.min_confidence': 0.5
            }.get(key, default)
            
            result = translate_lines(sample_lines)
            
            # Implementation should handle low confidence appropriately
            assert len(result) == 2
            
            # Check confidence scores are preserved
            for translation in result:
                assert 'confidence' in translation
                assert isinstance(translation['confidence'], (int, float))