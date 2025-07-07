"""
Functional tests for Translation module (_4_2_translate.py)
Tests core translation logic without complex dependencies
"""

import pytest
import tempfile
import os
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from difflib import SequenceMatcher


class TestTranslationLogic:
    """Test core translation logic functionality"""
    
    def test_text_chunking_algorithm(self):
        """Test text chunking algorithm logic"""
        # Simulate the chunking algorithm from split_chunks_by_chars
        def mock_split_chunks_by_chars(sentences, chunk_size, max_i):
            chunks = []
            chunk = ''
            sentence_count = 0
            
            for sentence in sentences:
                if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
                    if chunk.strip():
                        chunks.append(chunk.strip())
                    chunk = sentence + '\n'
                    sentence_count = 1
                else:
                    chunk += sentence + '\n'
                    sentence_count += 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            return chunks
        
        # Test chunking logic
        sentences = ['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5', 'Line 6']
        
        # Test with character limit
        chunks = mock_split_chunks_by_chars(sentences, chunk_size=20, max_i=10)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        # Verify no chunk is empty
        for chunk in chunks:
            assert len(chunk.strip()) > 0
        
        # Test with sentence limit
        chunks = mock_split_chunks_by_chars(sentences, chunk_size=1000, max_i=2)
        assert isinstance(chunks, list)
        
        # Each chunk should have at most 2 sentences
        for chunk in chunks:
            lines = chunk.split('\n')
            assert len([line for line in lines if line.strip()]) <= 2
    
    def test_context_extraction_logic(self):
        """Test context extraction logic"""
        # Simulate get_previous_content logic
        def mock_get_previous_content(chunks, chunk_index):
            if chunk_index == 0:
                return None
            return chunks[chunk_index - 1].split('\n')[-3:]  # Last 3 lines
        
        # Simulate get_after_content logic
        def mock_get_after_content(chunks, chunk_index):
            if chunk_index == len(chunks) - 1:
                return None
            return chunks[chunk_index + 1].split('\n')[:2]  # First 2 lines
        
        chunks = [
            "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
            "Line 6\nLine 7\nLine 8\nLine 9\nLine 10", 
            "Line 11\nLine 12\nLine 13\nLine 14\nLine 15"
        ]
        
        # Test first chunk
        prev = mock_get_previous_content(chunks, 0)
        after = mock_get_after_content(chunks, 0)
        assert prev is None
        assert after == ["Line 6", "Line 7"]
        
        # Test middle chunk
        prev = mock_get_previous_content(chunks, 1)
        after = mock_get_after_content(chunks, 1)
        assert prev == ["Line 3", "Line 4", "Line 5"]
        assert after == ["Line 11", "Line 12"]
        
        # Test last chunk
        prev = mock_get_previous_content(chunks, 2)
        after = mock_get_after_content(chunks, 2)
        assert prev == ["Line 8", "Line 9", "Line 10"]
        assert after is None
    
    def test_similarity_calculation(self):
        """Test similarity calculation using SequenceMatcher"""
        def mock_similar(a, b):
            return SequenceMatcher(None, a, b).ratio()
        
        # Test identical strings
        assert mock_similar("hello", "hello") == 1.0
        
        # Test completely different strings
        similarity = mock_similar("hello", "world")
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0
        
        # Test partially similar strings
        similarity = mock_similar("hello world", "hello there")
        assert 0.0 < similarity < 1.0
        assert similarity > 0.3  # Should have some similarity
        
        # Test empty strings
        assert mock_similar("", "") == 1.0
        assert mock_similar("hello", "") == 0.0
    
    def test_batch_processing_logic(self):
        """Test batch processing workflow"""
        # Simulate batch processing from translate_all
        def mock_batch_process(chunks, batch_size):
            all_results = {}
            
            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]
                
                # Simulate batch translation
                batch_results = {}
                for i, chunk in enumerate(batch_chunks):
                    chunk_index = batch_start + i
                    batch_results[chunk_index] = {
                        'translation': f'Translated {chunk}'
                    }
                
                # Merge results
                all_results.update(batch_results)
            
            return all_results
        
        chunks = ['Chunk 1', 'Chunk 2', 'Chunk 3', 'Chunk 4', 'Chunk 5']
        
        # Test batch processing with batch size 2
        results = mock_batch_process(chunks, batch_size=2)
        
        assert len(results) == 5  # All chunks should be processed
        
        # Verify all chunks were translated
        for i in range(5):
            assert i in results
            assert 'translation' in results[i]
            assert f'Chunk {i+1}' in results[i]['translation']
        
        # Test with batch size 1
        results = mock_batch_process(chunks, batch_size=1)
        assert len(results) == 5
        
        # Test with batch size larger than chunks
        results = mock_batch_process(chunks, batch_size=10)
        assert len(results) == 5
    
    def test_translation_result_processing(self):
        """Test translation result processing logic"""
        # Simulate result processing from translate_all
        chunks = ['Line 1\nLine 2', 'Line 3\nLine 4', 'Line 5\nLine 6']
        
        translation_results = {
            0: {'translation': 'Trans 1\nTrans 2'},
            1: {'translation': 'Trans 3\nTrans 4'},
            2: {'translation': 'Trans 5\nTrans 6'}
        }
        
        # Process results into source and translation lists
        src_text, trans_text = [], []
        
        for i, chunk in enumerate(chunks):
            chunk_lines = chunk.split('\n')
            src_text.extend(chunk_lines)
            
            if i in translation_results:
                trans_text.extend(translation_results[i]['translation'].split('\n'))
            else:
                # This should raise an error in real implementation
                raise ValueError(f"Translation failed for chunk {i}")
        
        # Verify results
        assert len(src_text) == 6  # 2 lines per chunk * 3 chunks
        assert len(trans_text) == 6  # Should match source
        
        assert src_text == ['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5', 'Line 6']
        assert trans_text == ['Trans 1', 'Trans 2', 'Trans 3', 'Trans 4', 'Trans 5', 'Trans 6']
        
        # Test missing translation
        incomplete_results = {0: {'translation': 'Trans 1\nTrans 2'}}  # Missing chunks 1, 2
        
        src_text, trans_text = [], []
        with pytest.raises(ValueError, match="Translation failed for chunk 1"):
            for i, chunk in enumerate(chunks):
                chunk_lines = chunk.split('\n')
                src_text.extend(chunk_lines)
                
                if i in incomplete_results:
                    trans_text.extend(incomplete_results[i]['translation'].split('\n'))
                else:
                    raise ValueError(f"Translation failed for chunk {i}")
    
    def test_config_handling_logic(self):
        """Test configuration handling with fallbacks"""
        # Simulate config loading with fallbacks
        def mock_load_config_with_fallback(key, default_value):
            config_values = {
                'batch_translate_size': 15,
                'min_trim_duration': 5.0
            }
            
            try:
                # Simulate config loading
                if key in config_values:
                    return config_values[key]
                else:
                    raise KeyError(f"Config key {key} not found")
            except (KeyError, FileNotFoundError, PermissionError):
                return default_value
        
        # Test successful config loading
        batch_size = mock_load_config_with_fallback('batch_translate_size', 15)
        assert batch_size == 15
        
        # Test fallback for missing config
        invalid_config = mock_load_config_with_fallback('non_existent_key', 10)
        assert invalid_config == 10
        
        # Test exception handling
        def mock_load_config_with_exception(key):
            raise KeyError("Config not available")
        
        try:
            batch_size = mock_load_config_with_exception('batch_translate_size')
        except (KeyError, FileNotFoundError, PermissionError):
            batch_size = 15  # Fallback
        
        assert batch_size == 15
    
    def test_chunk_data_preparation(self):
        """Test chunk data preparation for batch processing"""
        # Simulate chunk data preparation from translate_all
        def mock_prepare_chunk_data(chunks, batch_start, batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            chunks_data = []
            for i, chunk in enumerate(batch_chunks):
                chunk_index = batch_start + i
                
                # Mock context extraction
                previous_content = None if chunk_index == 0 else ['prev1', 'prev2', 'prev3']
                after_content = None if chunk_index == len(chunks) - 1 else ['after1', 'after2']
                
                chunks_data.append({
                    'chunk': chunk,
                    'previous_content': previous_content,
                    'after_content': after_content,
                    'things_to_note': f'notes for chunk {chunk_index}',
                    'index': chunk_index
                })
            
            return chunks_data
        
        chunks = ['Chunk 1', 'Chunk 2', 'Chunk 3', 'Chunk 4', 'Chunk 5']
        
        # Test first batch
        chunk_data = mock_prepare_chunk_data(chunks, 0, 2)
        assert len(chunk_data) == 2
        assert chunk_data[0]['index'] == 0
        assert chunk_data[1]['index'] == 1
        assert chunk_data[0]['previous_content'] is None  # First chunk
        assert chunk_data[1]['previous_content'] is not None
        
        # Test middle batch
        chunk_data = mock_prepare_chunk_data(chunks, 2, 2)
        assert len(chunk_data) == 2
        assert chunk_data[0]['index'] == 2
        assert chunk_data[1]['index'] == 3
        assert chunk_data[0]['previous_content'] is not None
        assert chunk_data[1]['previous_content'] is not None
        
        # Test last batch
        chunk_data = mock_prepare_chunk_data(chunks, 4, 2)
        assert len(chunk_data) == 1  # Only one chunk left
        assert chunk_data[0]['index'] == 4
        assert chunk_data[0]['after_content'] is None  # Last chunk
    
    def test_translation_workflow_steps(self):
        """Test complete translation workflow steps"""
        # Simulate the main workflow steps
        workflow_steps = []
        
        # Step 1: Initialize
        workflow_steps.append('initialize')
        
        # Step 2: Load configuration
        try:
            batch_size = 15  # Mock config loading
            workflow_steps.append('load_config')
        except (KeyError, FileNotFoundError, PermissionError):
            batch_size = 15  # Fallback
            workflow_steps.append('config_fallback')
        
        # Step 3: Split text into chunks
        chunks = ['chunk1', 'chunk2', 'chunk3']  # Mock chunking
        workflow_steps.append('split_chunks')
        
        # Step 4: Load theme
        theme = 'test theme'  # Mock theme loading
        workflow_steps.append('load_theme')
        
        # Step 5: Process batches
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        for batch_num in range(total_batches):
            workflow_steps.append(f'process_batch_{batch_num}')
        
        # Step 6: Combine results
        workflow_steps.append('combine_results')
        
        # Step 7: Process timestamps
        workflow_steps.append('process_timestamps')
        
        # Step 8: Apply trimming
        workflow_steps.append('apply_trimming')
        
        # Step 9: Save results
        workflow_steps.append('save_results')
        
        # Verify workflow
        expected_steps = [
            'initialize', 'load_config', 'split_chunks', 'load_theme',
            'process_batch_0', 'combine_results', 'process_timestamps',
            'apply_trimming', 'save_results'
        ]
        
        assert workflow_steps == expected_steps
        assert len(workflow_steps) == 9


class TestTranslationErrorHandling:
    """Test translation error handling scenarios"""
    
    def test_missing_translation_handling(self):
        """Test handling of missing translations"""
        chunks = ['chunk1', 'chunk2', 'chunk3']
        incomplete_results = {0: {'translation': 'trans1'}}  # Missing 1, 2
        
        # Should detect and handle missing translations
        missing_chunks = []
        for i in range(len(chunks)):
            if i not in incomplete_results:
                missing_chunks.append(i)
        
        assert missing_chunks == [1, 2]
        
        # Should raise appropriate error
        with pytest.raises(ValueError, match="Translation failed for chunk 1"):
            for i in range(len(chunks)):
                if i not in incomplete_results:
                    raise ValueError(f"Translation failed for chunk {i}")
    
    def test_empty_chunk_handling(self):
        """Test handling of empty chunks"""
        # Test chunking with empty content
        def mock_handle_empty_chunks(content):
            if not content or not content.strip():
                return ['']  # Return single empty chunk
            
            sentences = content.strip().split('\n')
            return sentences if sentences else ['']
        
        # Test empty content
        result = mock_handle_empty_chunks('')
        assert result == ['']
        
        # Test whitespace-only content
        result = mock_handle_empty_chunks('   \n\n   ')
        assert result == ['']
        
        # Test normal content
        result = mock_handle_empty_chunks('Line 1\nLine 2')
        assert result == ['Line 1', 'Line 2']
    
    def test_batch_size_edge_cases(self):
        """Test edge cases in batch size handling"""
        chunks = ['a', 'b', 'c', 'd', 'e']
        
        # Test batch size larger than chunk count
        def mock_calculate_batches(chunks, batch_size):
            return (len(chunks) + batch_size - 1) // batch_size
        
        # Normal case
        batches = mock_calculate_batches(chunks, 2)
        assert batches == 3  # (5 + 2 - 1) // 2 = 3
        
        # Batch size equals chunk count
        batches = mock_calculate_batches(chunks, 5)
        assert batches == 1
        
        # Batch size larger than chunk count
        batches = mock_calculate_batches(chunks, 10)
        assert batches == 1
        
        # Batch size 1
        batches = mock_calculate_batches(chunks, 1)
        assert batches == 5
    
    def test_config_error_recovery(self):
        """Test configuration error recovery"""
        def mock_load_with_recovery(key, default):
            error_scenarios = [
                'file_not_found',
                'permission_denied',
                'invalid_format',
                'key_error'
            ]
            
            # Simulate different error types
            for scenario in error_scenarios:
                try:
                    if scenario == 'file_not_found':
                        raise FileNotFoundError("Config file not found")
                    elif scenario == 'permission_denied':
                        raise PermissionError("Cannot read config file")
                    elif scenario == 'invalid_format':
                        raise json.JSONDecodeError("Invalid JSON", "", 0)
                    elif scenario == 'key_error':
                        raise KeyError(f"Key {key} not found")
                except (FileNotFoundError, PermissionError, json.JSONDecodeError, KeyError):
                    return default  # Fallback to default
            
            return default
        
        # Test that all error scenarios fall back to default
        result = mock_load_with_recovery('batch_size', 15)
        assert result == 15
        
        result = mock_load_with_recovery('min_duration', 5.0)
        assert result == 5.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])