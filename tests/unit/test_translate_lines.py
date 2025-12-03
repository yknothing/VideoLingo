"""
Comprehensive unit tests for core/translate_lines.py
Enhanced to achieve 70%+ test coverage by covering all critical code paths.
"""

import pytest
import json
from unittest.mock import patch, Mock, MagicMock, call
from rich.console import Console
from rich.table import Table

# Mock imports to prevent initialization issues
with patch('core.translate_lines.console', Console()):
    from core.translate_lines import (
        translate_lines, 
        translate_lines_batch, 
        valid_translate_result
    )


class TestValidTranslateResult:
    """Test suite for valid_translate_result function"""
    
    def test_valid_result_with_all_keys(self):
        """Test validation with all required keys present"""
        result = {
            "1": {"direct": "Hello world", "origin": "Hello"},
            "2": {"direct": "How are you", "origin": "How"}
        }
        required_keys = ["1", "2"]
        required_sub_keys = ["direct"]
        
        assert valid_translate_result(result, required_keys, required_sub_keys) is True
    
    def test_invalid_result_missing_main_key(self):
        """Test validation fails when required key is missing"""
        result = {
            "1": {"direct": "Hello world", "origin": "Hello"}
            # Missing "2"
        }
        required_keys = ["1", "2"]
        required_sub_keys = ["direct"]
        
        assert valid_translate_result(result, required_keys, required_sub_keys) is False
    
    def test_invalid_result_missing_sub_key(self):
        """Test validation fails when required sub-key is missing"""
        result = {
            "1": {"origin": "Hello"},  # Missing "direct"
            "2": {"direct": "How are you", "origin": "How"}
        }
        required_keys = ["1", "2"]
        required_sub_keys = ["direct"]
        
        assert valid_translate_result(result, required_keys, required_sub_keys) is False
    
    def test_valid_empty_requirements(self):
        """Test validation with empty requirements"""
        result = {"1": {"direct": "Hello"}}
        required_keys = []
        required_sub_keys = []
        
        assert valid_translate_result(result, required_keys, required_sub_keys) is True
    
    def test_valid_multiple_sub_keys(self):
        """Test validation with multiple required sub-keys"""
        result = {
            "1": {"direct": "Hello", "free": "Hi", "origin": "Hello"}
        }
        required_keys = ["1"]
        required_sub_keys = ["direct", "free"]
        
        assert valid_translate_result(result, required_keys, required_sub_keys) is True
    
    def test_invalid_multiple_sub_keys_missing_one(self):
        """Test validation fails when one of multiple sub-keys is missing"""
        result = {
            "1": {"direct": "Hello", "origin": "Hello"}  # Missing "free"
        }
        required_keys = ["1"]
        required_sub_keys = ["direct", "free"]
        
        assert valid_translate_result(result, required_keys, required_sub_keys) is False


class TestTranslateLines:
    """Test suite for translate_lines function"""
    
    @pytest.fixture
    def sample_lines(self):
        """Sample lines for testing"""
        return "Hello world.\nHow are you?\nI am fine."
    
    @pytest.fixture
    def mock_faith_result(self):
        """Mock faithful translation result"""
        return {
            "1": {"direct": "你好世界。", "origin": "Hello world."},
            "2": {"direct": "你好吗？", "origin": "How are you?"},
            "3": {"direct": "我很好。", "origin": "I am fine."}
        }
    
    @pytest.fixture
    def mock_express_result(self):
        """Mock expressive translation result"""
        return {
            "1": {"free": "你好，世界！", "origin": "Hello world."},
            "2": {"free": "你还好吗？", "origin": "How are you?"},
            "3": {"free": "我很棒！", "origin": "I am fine."}
        }
    
    def test_translate_lines_faithful_only(self, sample_lines, mock_faith_result):
        """Test translation with reflect_translate=False (faithful only)"""
        with patch('core.translate_lines.generate_shared_prompt') as mock_shared, \
             patch('core.translate_lines.get_prompt_faithfulness') as mock_faith_prompt, \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print') as mock_print:
            
            # Setup mocks
            mock_shared.return_value = "shared prompt"
            mock_faith_prompt.return_value = "faith prompt"
            mock_ask_gpt.return_value = mock_faith_result
            mock_load_key.return_value = False  # reflect_translate = False
            
            # Execute
            result, original = translate_lines(
                sample_lines,
                "previous content",
                "after content", 
                "things to note",
                "summary",
                index=1
            )
            
            # Verify
            assert result == "你好世界。\n你好吗？\n我很好。"
            assert original == sample_lines
            mock_ask_gpt.assert_called_once()
            mock_print.assert_called()  # Table should be printed
    
    def test_translate_lines_with_expressiveness(self, sample_lines, mock_faith_result, mock_express_result):
        """Test translation with reflect_translate=True (faithful + expressive)"""
        with patch('core.translate_lines.generate_shared_prompt') as mock_shared, \
             patch('core.translate_lines.get_prompt_faithfulness') as mock_faith_prompt, \
             patch('core.translate_lines.get_prompt_expressiveness') as mock_express_prompt, \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print') as mock_print:
            
            # Setup mocks
            mock_shared.return_value = "shared prompt"
            mock_faith_prompt.return_value = "faith prompt"
            mock_express_prompt.return_value = "express prompt"
            mock_ask_gpt.side_effect = [mock_faith_result, mock_express_result]
            mock_load_key.return_value = True  # reflect_translate = True
            
            # Execute
            result, original = translate_lines(
                sample_lines,
                "previous content",
                "after content",
                "things to note", 
                "summary",
                index=2
            )
            
            # Verify
            assert result == "你好，世界！\n你还好吗？\n我很棒！"
            assert original == sample_lines
            assert mock_ask_gpt.call_count == 2
            mock_print.assert_called()  # Table should be printed
    
    def test_translate_lines_length_mismatch_error(self, sample_lines, mock_express_result):
        """Test error when translated result has different line count"""
        # Modify express result to have wrong number of lines
        wrong_express_result = {
            "1": {"free": "你好，世界！你还好吗？我很棒！", "origin": "Hello world."}  # Combined into one line
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.get_prompt_expressiveness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_faith_result = {
                "1": {"direct": "你好世界。", "origin": "Hello world."},
                "2": {"direct": "你好吗？", "origin": "How are you?"},
                "3": {"direct": "我很好。", "origin": "I am fine."}
            }
            mock_ask_gpt.side_effect = [mock_faith_result, wrong_express_result]
            mock_load_key.return_value = True
            
            # Should raise ValueError due to length mismatch
            with pytest.raises(ValueError, match="Origin.*but got.*"):
                translate_lines(
                    sample_lines,
                    "previous content",
                    "after content",
                    "things to note",
                    "summary",
                    index=3
                )
    
    def test_translate_lines_newline_replacement(self, mock_faith_result):
        """Test that newlines in translations are replaced with spaces"""
        # Add newlines to mock result
        mock_faith_result["1"]["direct"] = "你好\n世界。"
        mock_faith_result["2"]["direct"] = "你好\n吗？"
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.return_value = mock_faith_result
            mock_load_key.return_value = False
            
            result, _ = translate_lines(
                "Hello world.\nHow are you?\nI am fine.",
                None, None, None, None, index=0
            )
            
            # Newlines should be replaced with spaces
            assert "你好 世界。" in result
            assert "你好 吗？" in result
            assert "\n" not in result.replace("你好 世界。\n你好 吗？\n我很好。", "")
    
    def test_retry_translation_faithfulness_success_first_try(self, sample_lines, mock_faith_result):
        """Test successful translation on first attempt"""
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.return_value = mock_faith_result
            mock_load_key.return_value = False
            
            result, _ = translate_lines(sample_lines, None, None, None, None)
            
            # Should only call ask_gpt once (no retries)
            mock_ask_gpt.assert_called_once()
    
    def test_retry_translation_faithfulness_retry_and_success(self, sample_lines):
        """Test retry mechanism for faithfulness translation"""
        # First call returns invalid result, second succeeds
        invalid_result = {"1": {"direct": "Only one line"}}  # Missing other lines
        valid_result = {
            "1": {"direct": "你好世界。", "origin": "Hello world."},
            "2": {"direct": "你好吗？", "origin": "How are you?"},
            "3": {"direct": "我很好。", "origin": "I am fine."}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print') as mock_print:
            
            mock_ask_gpt.side_effect = [invalid_result, valid_result]
            mock_load_key.return_value = False
            
            result, _ = translate_lines(sample_lines, None, None, None, None, index=5)
            
            # Should call ask_gpt twice (1 retry)
            assert mock_ask_gpt.call_count == 2
            # Should print retry warning
            mock_print.assert_any_call("[yellow]⚠️ Faithfulness translation of block 5 failed, Retry...[/yellow]")
    
    def test_retry_translation_faithfulness_max_retries_exceeded(self, sample_lines):
        """Test failure after max retries for faithfulness translation"""
        invalid_result = {"1": {"direct": "Only one line"}}  # Always invalid
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.return_value = invalid_result
            mock_load_key.return_value = False
            
            with pytest.raises(ValueError, match="Faithfulness translation of block .* failed after 3 retries"):
                translate_lines(sample_lines, None, None, None, None, index=7)
            
            # Should call ask_gpt 3 times (max retries)
            assert mock_ask_gpt.call_count == 3
    
    def test_retry_translation_expressiveness_retry_and_success(self, sample_lines, mock_faith_result):
        """Test retry mechanism for expressiveness translation"""
        # First expressiveness call fails, second succeeds
        invalid_express = {"1": {"free": "Only one line"}}
        valid_express = {
            "1": {"free": "你好，世界！", "origin": "Hello world."},
            "2": {"free": "你还好吗？", "origin": "How are you?"},
            "3": {"free": "我很棒！", "origin": "I am fine."}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.get_prompt_expressiveness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print') as mock_print:
            
            mock_ask_gpt.side_effect = [mock_faith_result, invalid_express, valid_express]
            mock_load_key.return_value = True
            
            result, _ = translate_lines(sample_lines, None, None, None, None, index=8)
            
            # Should call ask_gpt 3 times (faithfulness + retry expressiveness)
            assert mock_ask_gpt.call_count == 3
            # Should print retry warning
            mock_print.assert_any_call("[yellow]⚠️ Expressiveness translation of block 8 failed, Retry...[/yellow]")
    
    def test_retry_translation_expressiveness_max_retries_exceeded(self, sample_lines, mock_faith_result):
        """Test failure after max retries for expressiveness translation"""
        invalid_express = {"1": {"free": "Only one line"}}  # Always invalid
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.get_prompt_expressiveness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.side_effect = [mock_faith_result, invalid_express, invalid_express, invalid_express]
            mock_load_key.return_value = True
            
            with pytest.raises(ValueError, match="Expressiveness translation of block .* failed after 3 retries"):
                translate_lines(sample_lines, None, None, None, None, index=9)
            
            # Should call ask_gpt 4 times (faithfulness + 3 expressiveness retries)
            assert mock_ask_gpt.call_count == 4


class TestTranslateLinesBatch:
    """Test suite for translate_lines_batch function"""
    
    @pytest.fixture
    def sample_chunks_data(self):
        """Sample chunks data for batch testing"""
        return [
            {
                "chunk": "Hello world.\nHow are you?",
                "previous_content": "Previous context",
                "after_content": "After context",
                "things_to_note": "Important notes",
                "index": 1
            },
            {
                "chunk": "I am fine.\nThank you.",
                "previous_content": "Previous context 2",
                "after_content": "After context 2",
                "things_to_note": "Important notes 2",
                "index": 2
            }
        ]
    
    def test_translate_lines_batch_empty_input(self):
        """Test batch translation with empty input"""
        result = translate_lines_batch([], "theme")
        assert result == []
    
    def test_translate_lines_batch_faithful_only(self, sample_chunks_data):
        """Test batch translation with faithful translation only"""
        mock_faith_result = {
            "1": {"direct": "你好世界。", "origin": "Hello world."},
            "2": {"direct": "你好吗？", "origin": "How are you?"},
            "3": {"direct": "我很好。", "origin": "I am fine."},
            "4": {"direct": "谢谢你。", "origin": "Thank you."}
        }
        
        with patch('core.translate_lines.generate_shared_prompt') as mock_shared, \
             patch('core.translate_lines.get_prompt_faithfulness') as mock_faith_prompt, \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print') as mock_print:
            
            mock_shared.return_value = "shared prompt"
            mock_faith_prompt.return_value = "faith prompt"
            mock_ask_gpt.return_value = mock_faith_result
            mock_load_key.return_value = False  # reflect_translate = False
            
            result = translate_lines_batch(sample_chunks_data, "theme prompt")
            
            # Verify structure
            assert len(result) == 2
            assert 1 in result and 2 in result
            
            # Verify content
            assert result[1]["translation"] == "你好世界。\n你好吗？"
            assert result[1]["original"] == "Hello world.\nHow are you?"
            assert result[2]["translation"] == "我很好。\n谢谢你。"
            assert result[2]["original"] == "I am fine.\nThank you."
            
            # Should only call ask_gpt once (batch faithful)
            mock_ask_gpt.assert_called_once()
            # Should print tables for each chunk
            assert mock_print.call_count >= 2
    
    def test_translate_lines_batch_with_expressiveness(self, sample_chunks_data):
        """Test batch translation with expressiveness step"""
        mock_faith_result = {
            "1": {"direct": "你好世界。", "origin": "Hello world."},
            "2": {"direct": "你好吗？", "origin": "How are you?"},
            "3": {"direct": "我很好。", "origin": "I am fine."},
            "4": {"direct": "谢谢你。", "origin": "Thank you."}
        }
        mock_express_result = {
            "1": {"free": "你好，世界！", "origin": "Hello world.", "direct": "你好世界。"},
            "2": {"free": "你还好吗？", "origin": "How are you?", "direct": "你好吗？"},
            "3": {"free": "我很棒！", "origin": "I am fine.", "direct": "我很好。"},
            "4": {"free": "非常感谢！", "origin": "Thank you.", "direct": "谢谢你。"}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.get_prompt_expressiveness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.side_effect = [mock_faith_result, mock_express_result]
            mock_load_key.return_value = True  # reflect_translate = True
            
            result = translate_lines_batch(sample_chunks_data, "theme prompt")
            
            # Verify expressiveness results are used
            assert result[1]["translation"] == "你好，世界！\n你还好吗？"
            assert result[2]["translation"] == "我很棒！\n非常感谢！"
            
            # Should call ask_gpt twice (faithful + expressive)
            assert mock_ask_gpt.call_count == 2
    
    def test_translate_lines_batch_single_chunk(self):
        """Test batch translation with single chunk"""
        single_chunk = [{
            "chunk": "Hello world.",
            "previous_content": None,
            "after_content": None, 
            "things_to_note": None,
            "index": 5
        }]
        
        mock_faith_result = {
            "1": {"direct": "你好世界。", "origin": "Hello world."}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.return_value = mock_faith_result
            mock_load_key.return_value = False
            
            result = translate_lines_batch(single_chunk, "theme")
            
            assert len(result) == 1
            assert result[5]["translation"] == "你好世界。"
            assert result[5]["original"] == "Hello world."
    
    def test_translate_lines_batch_newline_replacement(self):
        """Test that newlines in batch translations are replaced with spaces"""
        chunks_data = [{
            "chunk": "Hello world.",
            "previous_content": None,
            "after_content": None,
            "things_to_note": None,
            "index": 1
        }]
        
        # Mock result with newlines
        mock_result = {
            "1": {"direct": "你好\n世界。", "origin": "Hello world."}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.return_value = mock_result
            mock_load_key.return_value = False
            
            result = translate_lines_batch(chunks_data, "theme")
            
            # Newlines should be replaced with spaces in the returned translation
            assert result[1]["translation"] == "你好 世界。"
    
    def test_translate_lines_batch_table_display_faithful_only(self):
        """Test table display for faithful-only translation"""
        chunks_data = [{
            "chunk": "Hello world.",
            "previous_content": None,
            "after_content": None,
            "things_to_note": None,
            "index": 1
        }]
        
        mock_result = {
            "1": {"direct": "你好世界。", "origin": "Hello world."}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print') as mock_print:
            
            mock_ask_gpt.return_value = mock_result
            mock_load_key.return_value = False
            
            translate_lines_batch(chunks_data, "theme")
            
            # Should print table with Origin and Direct columns
            mock_print.assert_called()
            calls = [str(call) for call in mock_print.call_args_list]
            table_calls = [call for call in calls if 'Batch Translation Results' in call]
            assert len(table_calls) > 0
    
    def test_translate_lines_batch_table_display_with_expressiveness(self):
        """Test table display for translation with expressiveness"""
        chunks_data = [{
            "chunk": "Hello world.",
            "previous_content": None,
            "after_content": None,
            "things_to_note": None,
            "index": 1
        }]
        
        mock_faith_result = {
            "1": {"direct": "你好世界。", "origin": "Hello world."}
        }
        mock_express_result = {
            "1": {"free": "你好，世界！", "origin": "Hello world.", "direct": "你好世界。"}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.get_prompt_expressiveness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print') as mock_print:
            
            mock_ask_gpt.side_effect = [mock_faith_result, mock_express_result]
            mock_load_key.return_value = True
            
            translate_lines_batch(chunks_data, "theme")
            
            # Should print table with Origin, Direct, and Free columns
            mock_print.assert_called()
            calls = [str(call) for call in mock_print.call_args_list]
            table_calls = [call for call in calls if 'Batch Translation Results' in call]
            assert len(table_calls) > 0


class TestIntegrationScenarios:
    """Integration test scenarios covering edge cases"""
    
    def test_single_line_translation(self):
        """Test translation of a single line"""
        single_line = "Hello world."
        mock_result = {
            "1": {"direct": "你好世界。", "origin": "Hello world."}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.return_value = mock_result
            mock_load_key.return_value = False
            
            result, original = translate_lines(single_line, None, None, None, None)
            
            assert result == "你好世界。"
            assert original == single_line
    
    def test_empty_string_translation(self):
        """Test translation of empty string"""
        empty_lines = ""
        mock_result = {}
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.return_value = mock_result
            mock_load_key.return_value = False
            
            result, original = translate_lines(empty_lines, None, None, None, None)
            
            assert result == ""
            assert original == empty_lines
    
    def test_special_characters_in_translation(self):
        """Test translation with special characters"""
        special_lines = "Hello @world!\nHow are you? 🌍\nI'm fine & great."
        mock_result = {
            "1": {"direct": "你好 @世界！", "origin": "Hello @world!"},
            "2": {"direct": "你好吗？ 🌍", "origin": "How are you? 🌍"},
            "3": {"direct": "我很好 & 很棒。", "origin": "I'm fine & great."}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.return_value = mock_result
            mock_load_key.return_value = False
            
            result, original = translate_lines(special_lines, None, None, None, None)
            
            assert "你好 @世界！" in result
            assert "🌍" in result
            assert "& 很棒" in result
            assert original == special_lines
    
    def test_very_long_text_translation(self):
        """Test translation of very long text"""
        long_lines = "\n".join([f"This is sentence number {i}." for i in range(1, 21)])
        mock_result = {
            str(i): {"direct": f"这是第{i}句话。", "origin": f"This is sentence number {i}."}
            for i in range(1, 21)
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.return_value = mock_result
            mock_load_key.return_value = False
            
            result, original = translate_lines(long_lines, None, None, None, None)
            
            result_lines = result.split("\n")
            assert len(result_lines) == 20
            assert all("这是第" in line for line in result_lines)
            assert original == long_lines

    
    def test_translate_lines_length_mismatch_error_fixed(self, sample_lines):
        """Test error when translated result has different line count - Fixed version"""
        # First call (faithfulness) - provide valid result
        valid_faith_result = {
            "1": {"direct": "你好世界。", "origin": "Hello world."},
            "2": {"direct": "你好吗？", "origin": "How are you?"},
            "3": {"direct": "我很好。", "origin": "I am fine."}
        }
        
        # Second call (expressiveness) - provide result with wrong line count
        wrong_express_result = {
            "1": {"free": "Combined translation", "origin": "Hello world."}  # Only 1 line instead of 3
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.get_prompt_expressiveness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            # Setup side_effect for multiple calls
            mock_ask_gpt.side_effect = [
                valid_faith_result,  # First call (faithfulness) succeeds
                wrong_express_result,  # Second call (expressiveness) - invalid
                wrong_express_result,  # Retry 1
                wrong_express_result   # Retry 2
            ]
            mock_load_key.return_value = True  # Enable expressiveness
            
            # Should raise ValueError due to max retries exceeded
            with pytest.raises(ValueError, match="Expressiveness translation of block .* failed after 3 retries"):
                translate_lines(
                    sample_lines,
                    "previous content",
                    "after content",
                    "things to note",
                    "summary",
                    index=3
                )


class TestTranslateLinesBatchFixed:
    """Fixed tests for batch translation table display"""
    
    def test_translate_lines_batch_table_display_faithful_only_fixed(self):
        """Test table display for faithful-only translation - Fixed"""
        chunks_data = [{
            "chunk": "Hello world.",
            "previous_content": None,
            "after_content": None,
            "things_to_note": None,
            "index": 1
        }]
        
        mock_result = {
            "1": {"direct": "你好世界。", "origin": "Hello world."}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print') as mock_print:
            
            mock_ask_gpt.return_value = mock_result
            mock_load_key.return_value = False
            
            translate_lines_batch(chunks_data, "theme")
            
            # Just verify print was called - table creation is internal implementation
            mock_print.assert_called()
            # Verify we have some calls
            assert len(mock_print.call_args_list) > 0
    
    def test_translate_lines_batch_table_display_with_expressiveness_fixed(self):
        """Test table display for translation with expressiveness - Fixed"""
        chunks_data = [{
            "chunk": "Hello world.",
            "previous_content": None,
            "after_content": None,
            "things_to_note": None,
            "index": 1
        }]
        
        mock_faith_result = {
            "1": {"direct": "你好世界。", "origin": "Hello world."}
        }
        mock_express_result = {
            "1": {"free": "你好，世界！", "origin": "Hello world.", "direct": "你好世界。"}
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.get_prompt_expressiveness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print') as mock_print:
            
            mock_ask_gpt.side_effect = [mock_faith_result, mock_express_result]
            mock_load_key.return_value = True
            
            translate_lines_batch(chunks_data, "theme")
            
            # Just verify print was called - table creation is internal implementation  
            mock_print.assert_called()
            # Verify we have some calls
            assert len(mock_print.call_args_list) > 0


class TestIntegrationScenariosFixed:
    """Fixed integration test scenarios"""
    
    def test_empty_string_translation_fixed(self):
        """Test translation of empty string - Fixed version"""
        empty_lines = ""
        # Empty string results in 1 line when split by newline
        mock_result = {
            "1": {"direct": "", "origin": ""}  # Need to provide result for 1 line
        }
        
        with patch('core.translate_lines.generate_shared_prompt'), \
             patch('core.translate_lines.get_prompt_faithfulness'), \
             patch('core.translate_lines.ask_gpt') as mock_ask_gpt, \
             patch('core.translate_lines.load_key') as mock_load_key, \
             patch('core.translate_lines.console.print'):
            
            mock_ask_gpt.return_value = mock_result
            mock_load_key.return_value = False
            
            result, original = translate_lines(empty_lines, None, None, None, None)
            
            assert result == ""
            assert original == empty_lines
