"""
Comprehensive test suite for translation and processing modules.
Targets 85%+ branch coverage for translation workflow and semantic processing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import sys
import os

# Add core directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestTranslateLinesModule(unittest.TestCase):
    """Test core/translate_lines.py - Line-by-line translation functionality"""
    
    def setUp(self):
        self.patcher_load_key = patch('core.translate_lines.load_key')
        self.patcher_ask_gpt = patch('core.translate_lines.ask_gpt')
        self.patcher_console = patch('core.translate_lines.console')
        self.patcher_progress = patch('core.translate_lines.Progress')
        
        self.mock_load_key = self.patcher_load_key.start()
        self.mock_ask_gpt = self.patcher_ask_gpt.start()
        self.mock_console = self.patcher_console.start()
        self.mock_progress = self.patcher_progress.start()
        
        # Setup default mock values
        self.mock_load_key.side_effect = lambda key: {
            "target_language": "Spanish",
            "max_workers": 4
        }.get(key, "default")
        
        # Mock progress bar
        self.mock_progress_instance = Mock()
        self.mock_progress.return_value.__enter__.return_value = self.mock_progress_instance
        
    def tearDown(self):
        self.patcher_load_key.stop()
        self.patcher_ask_gpt.stop()
        self.patcher_console.stop()
        self.patcher_progress.stop()
        
    def test_translate_line_success(self):
        """Test successful single line translation"""
        from core.translate_lines import translate_line
        
        self.mock_ask_gpt.return_value = {"translation": "Hola mundo"}
        
        result = translate_line("Hello world", "Spanish")
        
        self.assertEqual(result, "Hola mundo")
        self.mock_ask_gpt.assert_called_once()
        
    def test_translate_line_empty_input(self):
        """Test translation with empty input"""
        from core.translate_lines import translate_line
        
        result = translate_line("", "Spanish")
        
        self.assertEqual(result, "")
        self.mock_ask_gpt.assert_not_called()
        
    def test_translate_line_whitespace_input(self):
        """Test translation with whitespace-only input"""
        from core.translate_lines import translate_line
        
        result = translate_line("   \n\t  ", "Spanish")
        
        self.assertEqual(result, "")
        self.mock_ask_gpt.assert_not_called()
        
    def test_translate_line_api_error(self):
        """Test translation with API error"""
        from core.translate_lines import translate_line
        
        self.mock_ask_gpt.side_effect = Exception("API Error")
        
        result = translate_line("Hello world", "Spanish")
        
        self.assertEqual(result, "Hello world")  # Should return original on error
        
    def test_translate_line_invalid_response(self):
        """Test translation with invalid API response"""
        from core.translate_lines import translate_line
        
        self.mock_ask_gpt.return_value = {"error": "Invalid response"}
        
        result = translate_line("Hello world", "Spanish")
        
        self.assertEqual(result, "Hello world")  # Should return original on error
        
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_translate_lines_batch_success(self, mock_executor):
        """Test successful batch translation"""
        from core.translate_lines import translate_lines_batch
        
        # Mock executor
        mock_future1 = Mock()
        mock_future1.result.return_value = "Hola"
        mock_future2 = Mock()
        mock_future2.result.return_value = "Mundo"
        
        mock_executor.return_value.__enter__.return_value.submit.side_effect = [mock_future1, mock_future2]
        
        lines = ["Hello", "World"]
        target_lang = "Spanish"
        
        result = translate_lines_batch(lines, target_lang)
        
        expected = ["Hola", "Mundo"]
        self.assertEqual(result, expected)
        
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_translate_lines_batch_with_progress(self, mock_executor):
        """Test batch translation with progress tracking"""
        from core.translate_lines import translate_lines_batch
        
        # Mock executor and futures
        mock_futures = []
        for i in range(3):
            mock_future = Mock()
            mock_future.result.return_value = f"Translation {i}"
            mock_futures.append(mock_future)
        
        mock_executor.return_value.__enter__.return_value.submit.side_effect = mock_futures
        
        lines = ["Line 1", "Line 2", "Line 3"]
        
        result = translate_lines_batch(lines, "Spanish", show_progress=True)
        
        expected = ["Translation 0", "Translation 1", "Translation 2"]
        self.assertEqual(result, expected)
        
        # Should update progress bar
        self.assertEqual(self.mock_progress_instance.advance.call_count, 3)
        
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_translate_lines_batch_mixed_results(self, mock_executor):
        """Test batch translation with mixed success/failure"""
        from core.translate_lines import translate_lines_batch
        
        # Mock one successful, one failed future
        mock_future1 = Mock()
        mock_future1.result.return_value = "Success"
        mock_future2 = Mock()
        mock_future2.result.side_effect = Exception("Failed")
        
        mock_executor.return_value.__enter__.return_value.submit.side_effect = [mock_future1, mock_future2]
        
        lines = ["Hello", "World"]
        
        result = translate_lines_batch(lines, "Spanish")
        
        # Should handle mixed results gracefully
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Success")
        
    def test_get_translation_prompt(self):
        """Test translation prompt generation"""
        from core.translate_lines import get_translation_prompt
        
        prompt = get_translation_prompt("Hello world", "Spanish")
        
        self.assertIn("Hello world", prompt)
        self.assertIn("Spanish", prompt)
        self.assertIn("translation", prompt.lower())
        
    def test_validate_translation_response(self):
        """Test translation response validation"""
        from core.translate_lines import validate_translation_response
        
        # Valid response
        valid_response = {"translation": "Hola mundo"}
        result = validate_translation_response(valid_response)
        self.assertEqual(result["status"], "success")
        
        # Invalid response - missing translation key
        invalid_response = {"error": "No translation"}
        result = validate_translation_response(invalid_response)
        self.assertEqual(result["status"], "error")
        
        # Invalid response - empty translation
        empty_response = {"translation": ""}
        result = validate_translation_response(empty_response)
        self.assertEqual(result["status"], "error")


class TestSplitMeaningModule(unittest.TestCase):
    """Test core/_3_2_split_meaning.py - Semantic sentence splitting"""
    
    def setUp(self):
        self.patcher_ask_gpt = patch('core._3_2_split_meaning.ask_gpt')
        self.patcher_load_key = patch('core._3_2_split_meaning.load_key')
        self.patcher_get_split_prompt = patch('core._3_2_split_meaning.get_split_meaning_prompt')
        
        self.mock_ask_gpt = self.patcher_ask_gpt.start()
        self.mock_load_key = self.patcher_load_key.start()
        self.mock_get_split_prompt = self.patcher_get_split_prompt.start()
        
        # Setup defaults
        self.mock_load_key.return_value = "Spanish"
        self.mock_get_split_prompt.return_value = "Split this sentence"
        
    def tearDown(self):
        self.patcher_ask_gpt.stop()
        self.patcher_load_key.stop()
        self.patcher_get_split_prompt.stop()
        
    def test_split_sentence_success_two_parts(self):
        """Test successful sentence splitting into two parts"""
        from core._3_2_split_meaning import split_sentence
        
        self.mock_ask_gpt.return_value = {
            "part1": "First part",
            "part2": "Second part"
        }
        
        result = split_sentence("Long sentence that needs splitting", num_parts=2)
        
        expected = "First part\nSecond part"
        self.assertEqual(result, expected)
        
    def test_split_sentence_success_three_parts(self):
        """Test successful sentence splitting into three parts"""
        from core._3_2_split_meaning import split_sentence
        
        self.mock_ask_gpt.return_value = {
            "part1": "First part",
            "part2": "Second part", 
            "part3": "Third part"
        }
        
        result = split_sentence("Very long sentence that needs multiple splits", num_parts=3)
        
        expected = "First part\nSecond part\nThird part"
        self.assertEqual(result, expected)
        
    def test_split_sentence_invalid_response(self):
        """Test sentence splitting with invalid GPT response"""
        from core._3_2_split_meaning import split_sentence
        
        self.mock_ask_gpt.return_value = {"error": "Invalid response"}
        
        # Should handle error gracefully and return original
        result = split_sentence("Original sentence", num_parts=2)
        
        self.assertIn("Original sentence", result)
        
    def test_split_sentence_missing_parts(self):
        """Test sentence splitting with missing parts in response"""
        from core._3_2_split_meaning import split_sentence
        
        # Missing part2
        self.mock_ask_gpt.return_value = {
            "part1": "First part"
        }
        
        result = split_sentence("Original sentence", num_parts=2)
        
        # Should handle missing parts
        self.assertIsInstance(result, str)
        
    def test_split_sentence_empty_parts(self):
        """Test sentence splitting with empty parts"""
        from core._3_2_split_meaning import split_sentence
        
        self.mock_ask_gpt.return_value = {
            "part1": "",
            "part2": "Second part"
        }
        
        result = split_sentence("Original sentence", num_parts=2)
        
        expected = "\nSecond part"
        self.assertEqual(result, expected)
        
    def test_split_sentence_api_error(self):
        """Test sentence splitting with API error"""
        from core._3_2_split_meaning import split_sentence
        
        self.mock_ask_gpt.side_effect = Exception("API Error")
        
        result = split_sentence("Original sentence", num_parts=2)
        
        # Should return original sentence on error
        self.assertEqual(result, "Original sentence")
        
    def test_validate_split_response_success(self):
        """Test split response validation success"""
        from core._3_2_split_meaning import validate_split_response
        
        valid_response = {
            "part1": "First part",
            "part2": "Second part"
        }
        
        result = validate_split_response(valid_response, num_parts=2)
        
        self.assertEqual(result["status"], "success")
        
    def test_validate_split_response_missing_parts(self):
        """Test split response validation with missing parts"""
        from core._3_2_split_meaning import validate_split_response
        
        invalid_response = {
            "part1": "First part"
            # Missing part2
        }
        
        result = validate_split_response(invalid_response, num_parts=2)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Missing required parts", result["message"])
        
    def test_validate_split_response_empty_parts(self):
        """Test split response validation with empty parts"""
        from core._3_2_split_meaning import validate_split_response
        
        invalid_response = {
            "part1": "",
            "part2": "Second part"
        }
        
        result = validate_split_response(invalid_response, num_parts=2)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Empty parts found", result["message"])


class TestSummarizeModule(unittest.TestCase):
    """Test core/_4_1_summarize.py - Content summarization"""
    
    def setUp(self):
        self.patcher_ask_gpt = patch('core._4_1_summarize.ask_gpt')
        self.patcher_load_key = patch('core._4_1_summarize.load_key')
        self.patcher_read_excel = patch('pandas.read_excel')
        self.patcher_to_excel = patch('pandas.DataFrame.to_excel')
        self.patcher_console = patch('core._4_1_summarize.console')
        
        self.mock_ask_gpt = self.patcher_ask_gpt.start()
        self.mock_load_key = self.patcher_load_key.start()
        self.mock_read_excel = self.patcher_read_excel.start()
        self.mock_to_excel = self.patcher_to_excel.start()
        self.mock_console = self.patcher_console.start()
        
        # Setup defaults
        self.mock_load_key.return_value = "Spanish"
        
    def tearDown(self):
        self.patcher_ask_gpt.stop()
        self.patcher_load_key.stop()
        self.patcher_read_excel.stop()
        self.patcher_to_excel.stop()
        self.patcher_console.stop()
        
    def test_summarize_content_success(self):
        """Test successful content summarization"""
        from core._4_1_summarize import summarize_content
        
        self.mock_ask_gpt.return_value = {
            "summary": "This is a summary of the content",
            "key_points": ["Point 1", "Point 2", "Point 3"]
        }
        
        content = "Long content that needs to be summarized..."
        target_language = "Spanish"
        
        result = summarize_content(content, target_language)
        
        self.assertIn("summary", result)
        self.assertIn("key_points", result)
        self.assertEqual(result["summary"], "This is a summary of the content")
        
    def test_summarize_content_empty_input(self):
        """Test summarization with empty content"""
        from core._4_1_summarize import summarize_content
        
        result = summarize_content("", "Spanish")
        
        # Should return empty summary for empty input
        self.assertEqual(result["summary"], "")
        self.assertEqual(result["key_points"], [])
        
    def test_summarize_content_api_error(self):
        """Test summarization with API error"""
        from core._4_1_summarize import summarize_content
        
        self.mock_ask_gpt.side_effect = Exception("API Error")
        
        result = summarize_content("Content to summarize", "Spanish")
        
        # Should handle error gracefully
        self.assertIn("summary", result)
        self.assertIn("error", result["summary"].lower())
        
    def test_summarize_content_invalid_response(self):
        """Test summarization with invalid API response"""
        from core._4_1_summarize import summarize_content
        
        self.mock_ask_gpt.return_value = {"error": "Invalid response"}
        
        result = summarize_content("Content to summarize", "Spanish")
        
        # Should handle invalid response
        self.assertIn("summary", result)
        
    def test_chunk_content_by_sentences(self):
        """Test content chunking by sentences"""
        from core._4_1_summarize import chunk_content_by_sentences
        
        content = "First sentence. Second sentence. Third sentence. Fourth sentence."
        max_chunk_size = 30  # Small size to force chunking
        
        chunks = chunk_content_by_sentences(content, max_chunk_size)
        
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks
        for chunk in chunks:
            self.assertLessEqual(len(chunk), max_chunk_size * 2)  # Allow some overflow
            
    def test_chunk_content_single_long_sentence(self):
        """Test content chunking with single long sentence"""
        from core._4_1_summarize import chunk_content_by_sentences
        
        content = "This is a very long sentence without any periods that exceeds the maximum chunk size"
        max_chunk_size = 20
        
        chunks = chunk_content_by_sentences(content, max_chunk_size)
        
        # Should still return at least one chunk even if sentence is too long
        self.assertGreaterEqual(len(chunks), 1)
        
    @patch('core._4_1_summarize.summarize_content')
    def test_summarize_main_success(self, mock_summarize):
        """Test main summarization function"""
        from core._4_1_summarize import summarize_main
        
        # Mock Excel data
        mock_df = pd.DataFrame({
            'text': ['Content line 1', 'Content line 2', 'Content line 3']
        })
        self.mock_read_excel.return_value = mock_df
        
        # Mock summarization result
        mock_summarize.return_value = {
            "summary": "Combined summary",
            "key_points": ["Point 1", "Point 2"]
        }
        
        summarize_main()
        
        # Should read input file and save summary
        self.mock_read_excel.assert_called()
        mock_summarize.assert_called()
        self.mock_to_excel.assert_called()
        
    @patch('core._4_1_summarize.summarize_content')
    def test_summarize_main_large_content(self, mock_summarize):
        """Test main summarization with large content requiring chunking"""
        from core._4_1_summarize import summarize_main
        
        # Mock large Excel data
        large_content = ["Very long content " * 100 for _ in range(50)]
        mock_df = pd.DataFrame({'text': large_content})
        self.mock_read_excel.return_value = mock_df
        
        # Mock summarization results for chunks
        mock_summarize.side_effect = [
            {"summary": f"Summary {i}", "key_points": [f"Point {i}"]}
            for i in range(10)  # Simulate multiple chunks
        ]
        
        summarize_main()
        
        # Should call summarize multiple times for chunks
        self.assertGreater(mock_summarize.call_count, 1)
        
    def test_validate_summary_response_success(self):
        """Test summary response validation success"""
        from core._4_1_summarize import validate_summary_response
        
        valid_response = {
            "summary": "Valid summary",
            "key_points": ["Point 1", "Point 2"]
        }
        
        result = validate_summary_response(valid_response)
        
        self.assertEqual(result["status"], "success")
        
    def test_validate_summary_response_missing_summary(self):
        """Test summary response validation with missing summary"""
        from core._4_1_summarize import validate_summary_response
        
        invalid_response = {
            "key_points": ["Point 1", "Point 2"]
        }
        
        result = validate_summary_response(invalid_response)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Missing summary", result["message"])
        
    def test_validate_summary_response_empty_summary(self):
        """Test summary response validation with empty summary"""
        from core._4_1_summarize import validate_summary_response
        
        invalid_response = {
            "summary": "",
            "key_points": ["Point 1"]
        }
        
        result = validate_summary_response(invalid_response)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Empty summary", result["message"])


if __name__ == '__main__':
    unittest.main()
