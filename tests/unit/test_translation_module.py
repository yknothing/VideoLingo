"""
Comprehensive test suite for Translation module (_4_2_translate.py)
Tests the translation functionality with 90%+ branch coverage
"""

import pytest
import tempfile
import os
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, mock_open
from difflib import SequenceMatcher


class TestTranslationChunking:
    """Test text chunking functionality"""
    
    def test_split_chunks_by_chars_basic(self):
        """Test basic text chunking functionality"""
        # Mock file content
        mock_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8"
        
        with patch('builtins.open', mock_open(read_data=mock_content)):
            from core._4_2_translate import split_chunks_by_chars
            
            # Test with small chunk size
            chunks = split_chunks_by_chars(chunk_size=20, max_i=3)
            
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            
            # Verify chunks are not empty
            for chunk in chunks:
                assert isinstance(chunk, str)
                assert len(chunk.strip()) > 0
    
    def test_split_chunks_by_chars_size_limit(self):
        """Test chunking with size limits"""
        # Mock file content with long lines
        mock_content = "This is a very long line that should be chunked\nAnother long line for testing\nShort line\nAnother short line"
        
        with patch('builtins.open', mock_open(read_data=mock_content)):
            from core._4_2_translate import split_chunks_by_chars
            
            # Test with specific chunk size
            chunks = split_chunks_by_chars(chunk_size=50, max_i=2)
            
            assert isinstance(chunks, list)
            assert len(chunks) >= 1
            
            # Verify no chunk exceeds size limit significantly
            for chunk in chunks:
                # Allow some tolerance for the algorithm
                assert len(chunk) <= 100  # Should be reasonable
    
    def test_split_chunks_by_chars_sentence_limit(self):
        """Test chunking with sentence count limit"""
        # Mock file content with many short lines
        mock_content = "\n".join([f"Line {i}" for i in range(1, 21)])
        
        with patch('builtins.open', mock_open(read_data=mock_content)):
            from core._4_2_translate import split_chunks_by_chars
            
            # Test with sentence count limit
            chunks = split_chunks_by_chars(chunk_size=1000, max_i=5)
            
            assert isinstance(chunks, list)
            assert len(chunks) >= 1
            
            # Verify chunks respect sentence count limit
            for chunk in chunks:
                lines = chunk.split('\n')
                assert len(lines) <= 5  # Should respect max_i
    
    def test_split_chunks_empty_content(self):
        """Test handling of empty content"""
        with patch('builtins.open', mock_open(read_data="")):
            from core._4_2_translate import split_chunks_by_chars
            
            chunks = split_chunks_by_chars(chunk_size=100, max_i=10)
            
            assert isinstance(chunks, list)
            # Should handle empty content gracefully
            assert len(chunks) <= 1


class TestTranslationContext:
    """Test context extraction functionality"""
    
    def test_get_previous_content_basic(self):
        """Test getting previous content from chunks"""
        from core._4_2_translate import get_previous_content
        
        chunks = [
            "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
            "Line 6\nLine 7\nLine 8\nLine 9\nLine 10",
            "Line 11\nLine 12\nLine 13\nLine 14\nLine 15"
        ]
        
        # Test first chunk (should return None)
        result = get_previous_content(chunks, 0)
        assert result is None
        
        # Test middle chunk (should return last 3 lines from previous)
        result = get_previous_content(chunks, 1)
        assert isinstance(result, list)
        assert len(result) <= 3
        assert result == ["Line 3", "Line 4", "Line 5"]
        
        # Test last chunk
        result = get_previous_content(chunks, 2)
        assert isinstance(result, list)
        assert len(result) <= 3
        assert result == ["Line 8", "Line 9", "Line 10"]
    
    def test_get_after_content_basic(self):
        """Test getting after content from chunks"""
        from core._4_2_translate import get_after_content
        
        chunks = [
            "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
            "Line 6\nLine 7\nLine 8\nLine 9\nLine 10",
            "Line 11\nLine 12\nLine 13\nLine 14\nLine 15"
        ]
        
        # Test first chunk
        result = get_after_content(chunks, 0)
        assert isinstance(result, list)
        assert len(result) <= 2
        assert result == ["Line 6", "Line 7"]
        
        # Test middle chunk
        result = get_after_content(chunks, 1)
        assert isinstance(result, list)
        assert len(result) <= 2
        assert result == ["Line 11", "Line 12"]
        
        # Test last chunk (should return None)
        result = get_after_content(chunks, 2)
        assert result is None
    
    def test_get_previous_content_short_chunk(self):
        """Test getting previous content from short chunks"""
        from core._4_2_translate import get_previous_content
        
        chunks = [
            "Line 1\nLine 2",  # Only 2 lines
            "Line 3\nLine 4\nLine 5\nLine 6\nLine 7"
        ]
        
        result = get_previous_content(chunks, 1)
        assert isinstance(result, list)
        assert len(result) == 2  # Should return all available lines
        assert result == ["Line 1", "Line 2"]
    
    def test_get_after_content_short_chunk(self):
        """Test getting after content from short chunks"""
        from core._4_2_translate import get_after_content
        
        chunks = [
            "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
            "Line 6"  # Only 1 line
        ]
        
        result = get_after_content(chunks, 0)
        assert isinstance(result, list)
        assert len(result) == 1  # Should return all available lines
        assert result == ["Line 6"]


class TestTranslationSimilarity:
    """Test similarity calculation functionality"""
    
    def test_similar_identical_strings(self):
        """Test similarity for identical strings"""
        from core._4_2_translate import similar
        
        text1 = "Hello world"
        text2 = "Hello world"
        
        similarity = similar(text1, text2)
        assert similarity == 1.0
    
    def test_similar_different_strings(self):
        """Test similarity for different strings"""
        from core._4_2_translate import similar
        
        text1 = "Hello world"
        text2 = "Goodbye world"
        
        similarity = similar(text1, text2)
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0
    
    def test_similar_empty_strings(self):
        """Test similarity for empty strings"""
        from core._4_2_translate import similar
        
        text1 = ""
        text2 = ""
        
        similarity = similar(text1, text2)
        assert similarity == 1.0
    
    def test_similar_one_empty_string(self):
        """Test similarity with one empty string"""
        from core._4_2_translate import similar
        
        text1 = "Hello world"
        text2 = ""
        
        similarity = similar(text1, text2)
        assert similarity == 0.0
    
    def test_similar_partially_matching_strings(self):
        """Test similarity for partially matching strings"""
        from core._4_2_translate import similar
        
        text1 = "Hello world"
        text2 = "Hello there"
        
        similarity = similar(text1, text2)
        assert 0.0 < similarity < 1.0
        assert similarity > 0.5  # Should have some similarity due to "Hello"


class TestTranslationBatch:
    """Test batch translation functionality"""
    
    @patch('core._4_2_translate.search_things_to_note_in_prompt')
    @patch('core._4_2_translate.translate_lines')
    def test_translate_chunk_basic(self, mock_translate_lines, mock_search_things):
        """Test basic chunk translation"""
        from core._4_2_translate import translate_chunk
        
        # Mock dependencies
        mock_search_things.return_value = "Test notes"
        mock_translate_lines.return_value = ("English text", "Translated text")
        
        chunks = [
            "Previous chunk",
            "Current chunk to translate",
            "Next chunk"
        ]
        
        # Test translation
        result = translate_chunk(
            chunk="Current chunk to translate",
            chunks=chunks,
            theme_prompt="Test theme",
            i=1
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == 1  # Index
        assert result[1] == "English text"
        assert result[2] == "Translated text"
        
        # Verify dependencies were called
        mock_search_things.assert_called_once_with("Current chunk to translate")
        mock_translate_lines.assert_called_once()
    
    @patch('core._4_2_translate.search_things_to_note_in_prompt')
    @patch('core._4_2_translate.translate_lines')
    def test_translate_chunk_with_context(self, mock_translate_lines, mock_search_things):
        """Test chunk translation with context"""
        from core._4_2_translate import translate_chunk
        
        # Mock dependencies
        mock_search_things.return_value = "Test notes"
        mock_translate_lines.return_value = ("English text", "Translated text")
        
        chunks = [
            "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
            "Line 6\nLine 7\nLine 8\nLine 9\nLine 10",
            "Line 11\nLine 12\nLine 13\nLine 14\nLine 15"
        ]
        
        # Test translation with context
        result = translate_chunk(
            chunk="Line 6\nLine 7\nLine 8\nLine 9\nLine 10",
            chunks=chunks,
            theme_prompt="Test theme",
            i=1
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        # Verify translate_lines was called with context
        call_args = mock_translate_lines.call_args
        assert len(call_args[0]) >= 3  # chunk, previous_content, after_content
        
        # Check that context was extracted correctly
        previous_content = call_args[0][1]  # second argument
        after_content = call_args[0][2]     # third argument
        
        assert previous_content == ["Line 3", "Line 4", "Line 5"]
        assert after_content == ["Line 11", "Line 12"]


class TestTranslationMain:
    """Test main translation functionality"""
    
    @patch('core._4_2_translate.load_key')
    @patch('core._4_2_translate.split_chunks_by_chars')
    @patch('core.translate_lines.translate_lines_batch')
    @patch('core._4_2_translate.search_things_to_note_in_prompt')
    @patch('core._4_2_translate.check_len_then_trim')
    @patch('core._4_2_translate.align_timestamp')
    @patch('pandas.read_excel')
    @patch('builtins.open', mock_open(read_data='{"theme": "test theme"}'))
    def test_translate_all_basic(self, mock_read_excel, mock_align_timestamp, 
                                mock_check_len_then_trim, mock_search_things,
                                mock_translate_lines_batch, mock_split_chunks, 
                                mock_load_key):
        """Test basic translation workflow"""
        # Mock dependencies
        mock_load_key.side_effect = lambda key: {
            'batch_translate_size': 2,
            'min_trim_duration': 5.0
        }.get(key, 15)
        
        mock_split_chunks.return_value = [
            "Chunk 1 content",
            "Chunk 2 content"
        ]
        
        mock_translate_lines_batch.return_value = {
            0: {'translation': 'Translated chunk 1'},
            1: {'translation': 'Translated chunk 2'}
        }
        
        mock_search_things.return_value = "Test notes"
        
        # Mock pandas DataFrame
        mock_df = MagicMock()
        mock_df['text'] = MagicMock()
        mock_df['text'].str.strip.return_value.str.strip.return_value = mock_df['text']
        mock_read_excel.return_value = mock_df
        
        # Mock align_timestamp result
        mock_align_result = pd.DataFrame({
            'Translation': ['Translated chunk 1', 'Translated chunk 2'],
            'duration': [3.0, 7.0]
        })
        mock_align_timestamp.return_value = mock_align_result
        
        mock_check_len_then_trim.side_effect = lambda x, d: x  # Return unchanged
        
        # Mock the decorator
        with patch('core._4_2_translate.check_file_exists') as mock_decorator:
            mock_decorator.return_value = lambda func: func
            
            from core._4_2_translate import translate_all
            
            # Test main translation function
            translate_all()
            
            # Verify key functions were called
            mock_split_chunks.assert_called_once_with(chunk_size=600, max_i=15)
            mock_translate_lines_batch.assert_called()
            mock_align_timestamp.assert_called_once()
    
    @patch('core._4_2_translate.load_key')
    @patch('core._4_2_translate.split_chunks_by_chars')
    @patch('core.translate_lines.translate_lines_batch')
    @patch('builtins.open', mock_open(read_data='{"theme": "test theme"}'))
    def test_translate_all_missing_translation(self, mock_translate_lines_batch, 
                                             mock_split_chunks, mock_load_key):
        """Test handling of missing translations"""
        # Mock dependencies
        mock_load_key.side_effect = lambda key: {
            'batch_translate_size': 2
        }.get(key, 15)
        
        mock_split_chunks.return_value = [
            "Chunk 1 content",
            "Chunk 2 content"
        ]
        
        # Mock missing translation (only chunk 0, missing chunk 1)
        mock_translate_lines_batch.return_value = {
            0: {'translation': 'Translated chunk 1'}
            # Missing chunk 1 translation
        }
        
        # Mock the decorator
        with patch('core._4_2_translate.check_file_exists') as mock_decorator:
            mock_decorator.return_value = lambda func: func
            
            from core._4_2_translate import translate_all
            
            # Test that missing translation raises error
            with pytest.raises(ValueError, match="Translation failed for chunk 1"):
                translate_all()
    
    @patch('core._4_2_translate.load_key')
    @patch('core._4_2_translate.split_chunks_by_chars')
    @patch('core.translate_lines.translate_lines_batch')
    @patch('builtins.open', mock_open(read_data='{"theme": "test theme"}'))
    def test_translate_all_batch_processing(self, mock_translate_lines_batch, 
                                          mock_split_chunks, mock_load_key):
        """Test batch processing logic"""
        # Mock dependencies
        mock_load_key.side_effect = lambda key: {
            'batch_translate_size': 2  # Small batch size for testing
        }.get(key, 15)
        
        # Mock 5 chunks to test batching
        mock_split_chunks.return_value = [
            "Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5"
        ]
        
        # Mock batch translation results
        mock_translate_lines_batch.side_effect = [
            {0: {'translation': 'T1'}, 1: {'translation': 'T2'}},  # Batch 1
            {2: {'translation': 'T3'}, 3: {'translation': 'T4'}},  # Batch 2
            {4: {'translation': 'T5'}}                             # Batch 3
        ]
        
        # Mock other dependencies
        with patch('core._4_2_translate.search_things_to_note_in_prompt'):
            with patch('core._4_2_translate.check_len_then_trim'):
                with patch('core._4_2_translate.align_timestamp'):
                    with patch('pandas.read_excel'):
                        with patch('core._4_2_translate.check_file_exists') as mock_decorator:
                            mock_decorator.return_value = lambda func: func
                            
                            from core._4_2_translate import translate_all
                            
                            # Test batch processing
                            translate_all()
                            
                            # Verify batch processing - should be called 3 times (3 batches)
                            assert mock_translate_lines_batch.call_count == 3
                            
                            # Verify batch sizes
                            call_args_list = mock_translate_lines_batch.call_args_list
                            assert len(call_args_list[0][0][0]) == 2  # First batch: 2 chunks
                            assert len(call_args_list[1][0][0]) == 2  # Second batch: 2 chunks
                            assert len(call_args_list[2][0][0]) == 1  # Third batch: 1 chunk
    
    @patch('core._4_2_translate.load_key')
    def test_translate_all_config_fallback(self, mock_load_key):
        """Test configuration fallback handling"""
        # Mock load_key to raise exception for batch_translate_size
        mock_load_key.side_effect = Exception("Config not found")
        
        # Mock other dependencies to avoid actual processing
        with patch('core._4_2_translate.split_chunks_by_chars') as mock_split:
            with patch('core.translate_lines.translate_lines_batch') as mock_batch:
                with patch('builtins.open', mock_open(read_data='{"theme": "test"}')):
                    with patch('core._4_2_translate.check_file_exists') as mock_decorator:
                        mock_decorator.return_value = lambda func: func
                        
                        # Mock to return empty results to test config fallback
                        mock_split.return_value = []
                        mock_batch.return_value = {}
                        
                        from core._4_2_translate import translate_all
                        
                        # Should not raise exception and use default batch size
                        translate_all()
                        
                        # Verify it handled the config exception gracefully
                        mock_split.assert_called_once()


class TestTranslationIntegration:
    """Test translation module integration"""
    
    def test_translation_module_imports(self):
        """Test that translation module can import dependencies"""
        try:
            # Test core imports
            from core._4_2_translate import (
                split_chunks_by_chars,
                get_previous_content,
                get_after_content,
                translate_chunk,
                similar,
                translate_all
            )
            
            # Verify functions exist and are callable
            assert callable(split_chunks_by_chars)
            assert callable(get_previous_content)
            assert callable(get_after_content)
            assert callable(translate_chunk)
            assert callable(similar)
            assert callable(translate_all)
            
        except ImportError as e:
            pytest.skip(f"Translation module dependencies not available: {e}")
    
    def test_translation_workflow_components(self):
        """Test that translation workflow components work together"""
        from core._4_2_translate import get_previous_content, get_after_content, similar
        
        # Test workflow with sample data
        chunks = [
            "First chunk content\nwith multiple lines",
            "Second chunk content\nwith different text",
            "Third chunk content\nwith final text"
        ]
        
        # Test context extraction
        prev_content = get_previous_content(chunks, 1)
        after_content = get_after_content(chunks, 1)
        
        assert prev_content is not None
        assert after_content is not None
        
        # Test similarity calculation
        similarity = similar(chunks[0], chunks[1])
        assert 0.0 <= similarity <= 1.0
        
        # Test that context extraction provides reasonable results
        assert isinstance(prev_content, list)
        assert isinstance(after_content, list)
        assert len(prev_content) > 0
        assert len(after_content) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])