# Comprehensive Unit Tests for Translation Module
import pytest
import pandas as pd
import json
from unittest.mock import patch, mock_open, MagicMock
from core._4_2_translate import split_chunks_by_chars, similar, get_previous_content

class TestChunkProcessing:
    def test_split_chunks_by_chars_basic(self):
        sample_text = """First sentence.
Second sentence.
Third sentence."""
        with patch('builtins.open', mock_open(read_data=sample_text)):
            with patch('core._4_2_translate._3_2_SPLIT_BY_MEANING', '/fake/path'):
                chunks = split_chunks_by_chars(chunk_size=20, max_i=2)
                assert isinstance(chunks, list)
                assert len(chunks) > 0

    def test_similar_function(self):
        assert similar('hello', 'hello') == 1.0
        similarity = similar('hello', 'world')
        assert 0.0 <= similarity < 1.0

    def test_get_previous_content_first_chunk(self):
        chunks = ['chunk1', 'chunk2', 'chunk3']
        previous = get_previous_content(chunks, 0)
        assert previous is None

    def test_get_after_content_last_chunk(self):
        from core._4_2_translate import get_after_content
        chunks = ['chunk1', 'chunk2', 'chunk3']
        after = get_after_content(chunks, 2)  # Last chunk
        assert after is None

    def test_translate_chunk_basic(self):
        from core._4_2_translate import translate_chunk
        chunk = "Hello world."
        chunks = [chunk]
        theme_prompt = "Test theme"
        
        with patch('core._4_2_translate.search_things_to_note_in_prompt', return_value="Notes"):
            with patch('core._4_2_translate.translate_lines', return_value=("English text", "Translated text")):
                result = translate_chunk(chunk, chunks, theme_prompt, 0)
                assert len(result) == 3  # Returns (index, english, translation)
                assert result[0] == 0
