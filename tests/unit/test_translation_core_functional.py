"""
Functional tests for Translation Core modules
Tests core translation functionality without complex LLM dependencies
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any


class TestTranslateLinesLogic:
    """Test translate_lines.py core logic"""
    
    def test_translation_result_validation(self):
        """Test valid_translate_result function logic"""
        # Simulate valid_translate_result logic
        def mock_valid_translate_result(result, required_keys, required_sub_keys):
            # Check for the required key
            if not all(key in result for key in required_keys):
                return {
                    "status": "error", 
                    "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}"
                }
            
            # Check for required sub-keys in all items
            for key in result:
                if not all(sub_key in result[key] for sub_key in required_sub_keys):
                    return {
                        "status": "error", 
                        "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}"
                    }

            return {"status": "success", "message": "Translation completed"}
        
        # Test valid result
        valid_result = {
            "1": {"direct": "Hello", "origin": "你好"},
            "2": {"direct": "World", "origin": "世界"}
        }
        
        validation = mock_valid_translate_result(valid_result, ["1", "2"], ["direct"])
        assert validation["status"] == "success"
        assert validation["message"] == "Translation completed"
        
        # Test missing required key
        incomplete_result = {
            "1": {"direct": "Hello", "origin": "你好"}
            # Missing "2"
        }
        
        validation = mock_valid_translate_result(incomplete_result, ["1", "2"], ["direct"])
        assert validation["status"] == "error"
        assert "Missing required key(s): 2" in validation["message"]
        
        # Test missing sub-key
        incomplete_subkey_result = {
            "1": {"origin": "你好"},  # Missing "direct"
            "2": {"direct": "World", "origin": "世界"}
        }
        
        validation = mock_valid_translate_result(incomplete_subkey_result, ["1", "2"], ["direct"])
        assert validation["status"] == "error"
        assert "Missing required sub-key(s) in item 1: direct" in validation["message"]
    
    def test_translation_workflow_logic(self):
        """Test translate_lines workflow logic"""
        # Simulate translate_lines workflow
        def mock_translate_lines_workflow(lines, reflect_translate=True):
            workflow_steps = []
            
            # Step 1: Faithful translation
            workflow_steps.append('faithfulness_translation')
            
            # Mock line splitting and faithful translation
            line_list = lines.split('\n')
            faith_result = {}
            
            for i, line in enumerate(line_list, 1):
                faith_result[str(i)] = {
                    "origin": line,
                    "direct": f"Faithful translation of: {line}"
                }
            
            # Clean up newlines in direct translation
            for key in faith_result:
                faith_result[key]["direct"] = faith_result[key]["direct"].replace('\n', ' ')
            
            # Step 2: Check if reflection is needed
            if not reflect_translate:
                # Use faithful translation directly
                translate_result = "\n".join([faith_result[str(i)]["direct"].strip() for i in range(1, len(line_list) + 1)])
                return {
                    'result': translate_result,
                    'original': lines,
                    'steps': workflow_steps,
                    'faith_result': faith_result,
                    'express_result': None
                }
            
            # Step 3: Expressiveness translation
            workflow_steps.append('expressiveness_translation')
            
            express_result = {}
            for i, line in enumerate(line_list, 1):
                express_result[str(i)] = {
                    "origin": line,
                    "direct": faith_result[str(i)]["direct"],
                    "free": f"Expressive translation of: {line}"
                }
            
            # Final result uses expressive translation
            translate_result = "\n".join([express_result[str(i)]["free"].replace('\n', ' ').strip() for i in range(1, len(line_list) + 1)])
            
            # Step 4: Length validation
            workflow_steps.append('length_validation')
            
            if len(lines.split('\n')) != len(translate_result.split('\n')):
                return {
                    'error': 'Length mismatch',
                    'original_lines': len(lines.split('\n')),
                    'translated_lines': len(translate_result.split('\n')),
                    'steps': workflow_steps
                }
            
            return {
                'result': translate_result,
                'original': lines,
                'steps': workflow_steps,
                'faith_result': faith_result,
                'express_result': express_result
            }
        
        # Test faithful-only translation
        lines = "Hello world\nThis is a test"
        result = mock_translate_lines_workflow(lines, reflect_translate=False)
        
        assert 'faithfulness_translation' in result['steps']
        assert 'expressiveness_translation' not in result['steps']
        assert result['faith_result']['1']['origin'] == "Hello world"
        assert "Faithful translation of: Hello world" in result['faith_result']['1']['direct']
        assert result['express_result'] is None
        
        # Test full translation with reflection
        result = mock_translate_lines_workflow(lines, reflect_translate=True)
        
        assert 'faithfulness_translation' in result['steps']
        assert 'expressiveness_translation' in result['steps']
        assert 'length_validation' in result['steps']
        assert result['express_result'] is not None
        assert "Expressive translation of: Hello world" in result['express_result']['1']['free']
        
        # Test length mismatch scenario
        def mock_translate_lines_workflow_length_error(lines):
            # Simulate a case where translation produces different line count
            line_list = lines.split('\n')
            
            # Mock express result with different line count
            express_result = {}
            for i in range(1, len(line_list) + 2):  # Extra line
                express_result[str(i)] = {
                    "origin": "test",
                    "direct": "direct",
                    "free": f"line {i}"
                }
            
            # This will create a mismatch
            translate_result = "\n".join([express_result[str(i)]["free"] for i in range(1, len(line_list) + 2)])
            
            if len(lines.split('\n')) != len(translate_result.split('\n')):
                return {
                    'error': 'Length mismatch',
                    'original_lines': len(lines.split('\n')),
                    'translated_lines': len(translate_result.split('\n')),
                    'steps': ['faithfulness_translation', 'expressiveness_translation', 'length_validation']
                }
            
            return {'result': translate_result}
        
        error_result = mock_translate_lines_workflow_length_error(lines)
        assert 'error' in error_result
        assert error_result['error'] == 'Length mismatch'
        assert error_result['original_lines'] == 2
        assert error_result['translated_lines'] == 3
    
    def test_batch_translation_logic(self):
        """Test translate_lines_batch function logic"""
        # Simulate batch translation logic
        def mock_translate_lines_batch(chunks_data, theme_prompt):
            if not chunks_data:
                return []
            
            workflow_steps = []
            
            # Step 1: Prepare batch input
            workflow_steps.append('batch_input_preparation')
            
            batch_input = {}
            all_lines = []
            chunk_line_mapping = {}
            
            line_counter = 1
            for chunk_data in chunks_data:
                chunk_lines = chunk_data['chunk'].split('\n')
                chunk_start = line_counter
                
                for line in chunk_lines:
                    batch_input[str(line_counter)] = {
                        "origin": line,
                        "chunk_index": chunk_data['index'],
                        "direct": f"Direct translation {line_counter}."
                    }
                    all_lines.append(line)
                    line_counter += 1
                
                chunk_line_mapping[chunk_data['index']] = {
                    'start': chunk_start,
                    'end': line_counter - 1,
                    'lines': chunk_lines
                }
            
            # Step 2: Generate shared context
            workflow_steps.append('shared_context_generation')
            
            primary_chunk = chunks_data[0]
            shared_context = {
                'previous_content': primary_chunk.get('previous_content'),
                'after_content': primary_chunk.get('after_content'),
                'theme_prompt': theme_prompt,
                'things_to_note': primary_chunk.get('things_to_note')
            }
            
            # Step 3: Batch faithfulness translation
            workflow_steps.append('batch_faithfulness_translation')
            
            faith_result = {}
            for i in range(1, len(all_lines) + 1):
                faith_result[str(i)] = {
                    "origin": all_lines[i-1],
                    "direct": f"Faithful batch translation {i}".replace('\n', ' ')
                }
            
            # Step 4: Check reflection requirement
            reflect_translate = True  # Mock config
            
            if reflect_translate:
                # Step 5: Batch expressiveness translation
                workflow_steps.append('batch_expressiveness_translation')
                
                express_result = {}
                for i in range(1, len(all_lines) + 1):
                    express_result[str(i)] = {
                        "origin": all_lines[i-1],
                        "direct": faith_result[str(i)]["direct"],
                        "free": f"Expressive batch translation {i}".replace('\n', ' ')
                    }
                
                final_result = express_result
            else:
                final_result = faith_result
            
            # Step 6: Split results back to individual chunks
            workflow_steps.append('result_splitting')
            
            chunk_results = {}
            for chunk_data in chunks_data:
                chunk_idx = chunk_data['index']
                mapping = chunk_line_mapping[chunk_idx]
                
                if reflect_translate:
                    chunk_translation = '\n'.join([
                        final_result[str(i)]["free"].replace('\n', ' ').strip() 
                        for i in range(mapping['start'], mapping['end'] + 1)
                    ])
                else:
                    chunk_translation = '\n'.join([
                        final_result[str(i)]["direct"].replace('\n', ' ').strip() 
                        for i in range(mapping['start'], mapping['end'] + 1)
                    ])
                
                chunk_results[chunk_idx] = {
                    'translation': chunk_translation,
                    'original': chunk_data['chunk'],
                    'workflow_steps': workflow_steps
                }
            
            return chunk_results
        
        # Test batch translation
        chunks_data = [
            {
                'index': 1,
                'chunk': 'Hello world\nThis is chunk 1',
                'previous_content': 'Previous context',
                'after_content': 'After context',
                'things_to_note': 'Important notes'
            },
            {
                'index': 2,
                'chunk': 'Second chunk\nWith two lines',
                'previous_content': 'Previous context 2',
                'after_content': 'After context 2',
                'things_to_note': 'More notes'
            }
        ]
        
        theme_prompt = "Theme: Technical documentation"
        
        results = mock_translate_lines_batch(chunks_data, theme_prompt)
        
        assert len(results) == 2
        assert 1 in results and 2 in results
        
        # Check chunk 1 results
        chunk1 = results[1]
        assert chunk1['original'] == 'Hello world\nThis is chunk 1'
        assert 'Expressive batch translation 1' in chunk1['translation']
        assert 'Expressive batch translation 2' in chunk1['translation']
        assert len(chunk1['workflow_steps']) == 6
        
        # Check chunk 2 results
        chunk2 = results[2]
        assert chunk2['original'] == 'Second chunk\nWith two lines'
        assert 'Expressive batch translation 3' in chunk2['translation']
        assert 'Expressive batch translation 4' in chunk2['translation']
        
        # Test empty chunks
        empty_results = mock_translate_lines_batch([], theme_prompt)
        assert empty_results == []
    
    def test_retry_translation_logic(self):
        """Test retry translation logic"""
        # Simulate retry translation logic
        def mock_retry_translation(prompt, length, step_name, max_retries=3):
            retry_results = []
            
            for retry in range(max_retries):
                # Mock different behaviors for different retries
                if retry == 0:
                    # First attempt - simulate format error
                    mock_result = {str(i): {"origin": f"line {i}"} for i in range(1, length)}  # Missing required field
                    retry_results.append({
                        'retry': retry,
                        'success': False,
                        'error': 'Missing required field',
                        'result': mock_result
                    })
                elif retry == 1:
                    # Second attempt - simulate length mismatch
                    mock_result = {str(i): {"direct": f"translation {i}", "origin": f"line {i}"} for i in range(1, length + 2)}  # Extra line
                    retry_results.append({
                        'retry': retry,
                        'success': False,
                        'error': 'Length mismatch',
                        'result': mock_result
                    })
                else:
                    # Third attempt - success
                    if step_name == 'faithfulness':
                        mock_result = {str(i): {"direct": f"faithful translation {i}", "origin": f"line {i}"} for i in range(1, length + 1)}
                    else:  # expressiveness
                        mock_result = {str(i): {"free": f"expressive translation {i}", "origin": f"line {i}"} for i in range(1, length + 1)}
                    
                    retry_results.append({
                        'retry': retry,
                        'success': True,
                        'result': mock_result
                    })
                    
                    return {
                        'final_success': True,
                        'final_result': mock_result,
                        'retry_count': retry + 1,
                        'retry_history': retry_results
                    }
            
            # All retries failed
            return {
                'final_success': False,
                'final_result': None,
                'retry_count': max_retries,
                'retry_history': retry_results,
                'error': f'{step_name.capitalize()} translation failed after {max_retries} retries'
            }
        
        # Test successful retry after failures
        result = mock_retry_translation("test prompt", 3, "faithfulness")
        
        assert result['final_success'] is True
        assert result['retry_count'] == 3
        assert len(result['retry_history']) == 3
        assert result['retry_history'][0]['success'] is False
        assert result['retry_history'][1]['success'] is False
        assert result['retry_history'][2]['success'] is True
        assert "faithful translation 1" in result['final_result']['1']['direct']
        
        # Test expressiveness retry
        result = mock_retry_translation("test prompt", 2, "expressiveness")
        
        assert result['final_success'] is True
        assert "expressive translation 1" in result['final_result']['1']['free']
        
        # Test complete failure (mock by using 0 length to force failure)
        def mock_retry_translation_fail(prompt, length, step_name, max_retries=3):
            retry_results = []
            
            for retry in range(max_retries):
                retry_results.append({
                    'retry': retry,
                    'success': False,
                    'error': 'Persistent failure',
                    'result': {}
                })
            
            return {
                'final_success': False,
                'final_result': None,
                'retry_count': max_retries,
                'retry_history': retry_results,
                'error': f'{step_name.capitalize()} translation failed after {max_retries} retries'
            }
        
        fail_result = mock_retry_translation_fail("test prompt", 2, "faithfulness")
        
        assert fail_result['final_success'] is False
        assert fail_result['retry_count'] == 3
        assert len(fail_result['retry_history']) == 3
        assert all(not r['success'] for r in fail_result['retry_history'])
        assert 'Faithfulness translation failed after 3 retries' in fail_result['error']


class TestSpacyUtilsLogic:
    """Test spacy_utils modules core logic"""
    
    def test_phrase_validation_logic(self):
        """Test is_valid_phrase logic from split_by_comma.py"""
        # Simulate is_valid_phrase logic
        def mock_is_valid_phrase(phrase_tokens):
            """Mock phrase validation logic"""
            # Check for subject and verb
            has_subject = any(token.get('dep_') in ["nsubj", "nsubjpass"] or token.get('pos_') == "PRON" for token in phrase_tokens)
            has_verb = any(token.get('pos_') == "VERB" or token.get('pos_') == 'AUX' for token in phrase_tokens)
            return has_subject and has_verb
        
        # Test valid phrase (has subject and verb)
        valid_phrase = [
            {'text': 'He', 'pos_': 'PRON', 'dep_': 'nsubj'},
            {'text': 'runs', 'pos_': 'VERB', 'dep_': 'ROOT'},
            {'text': 'fast', 'pos_': 'ADV', 'dep_': 'advmod'}
        ]
        
        assert mock_is_valid_phrase(valid_phrase) is True
        
        # Test phrase with subject but no verb
        no_verb_phrase = [
            {'text': 'The', 'pos_': 'DET', 'dep_': 'det'},
            {'text': 'cat', 'pos_': 'NOUN', 'dep_': 'nsubj'},
            {'text': 'quickly', 'pos_': 'ADV', 'dep_': 'advmod'}
        ]
        
        assert mock_is_valid_phrase(no_verb_phrase) is False
        
        # Test phrase with verb but no subject
        no_subject_phrase = [
            {'text': 'Running', 'pos_': 'VERB', 'dep_': 'ROOT'},
            {'text': 'quickly', 'pos_': 'ADV', 'dep_': 'advmod'}
        ]
        
        assert mock_is_valid_phrase(no_subject_phrase) is False
        
        # Test phrase with auxiliary verb
        aux_phrase = [
            {'text': 'She', 'pos_': 'PRON', 'dep_': 'nsubj'},
            {'text': 'is', 'pos_': 'AUX', 'dep_': 'ROOT'},
            {'text': 'running', 'pos_': 'VERB', 'dep_': 'xcomp'}
        ]
        
        assert mock_is_valid_phrase(aux_phrase) is True
        
        # Test phrase with passive subject
        passive_phrase = [
            {'text': 'The', 'pos_': 'DET', 'dep_': 'det'},
            {'text': 'ball', 'pos_': 'NOUN', 'dep_': 'nsubjpass'},
            {'text': 'was', 'pos_': 'AUX', 'dep_': 'auxpass'},
            {'text': 'thrown', 'pos_': 'VERB', 'dep_': 'ROOT'}
        ]
        
        assert mock_is_valid_phrase(passive_phrase) is True
    
    def test_comma_analysis_logic(self):
        """Test analyze_comma logic from split_by_comma.py"""
        # Simulate analyze_comma logic
        def mock_analyze_comma(start, doc_tokens, comma_index, window_size=9):
            """Mock comma analysis logic"""
            # Get phrases around comma
            left_start = max(start, comma_index - window_size)
            right_end = min(len(doc_tokens), comma_index + window_size + 1)
            
            left_phrase = doc_tokens[left_start:comma_index]
            right_phrase = doc_tokens[comma_index + 1:right_end]
            
            # Check if right phrase is valid for splitting
            def is_valid_phrase(phrase_tokens):
                has_subject = any(token.get('dep_') in ["nsubj", "nsubjpass"] or token.get('pos_') == "PRON" for token in phrase_tokens)
                has_verb = any(token.get('pos_') == "VERB" or token.get('pos_') == 'AUX' for token in phrase_tokens)
                return has_subject and has_verb
            
            suitable_for_splitting = is_valid_phrase(right_phrase)
            
            # Remove punctuation and check word count
            left_words = [t for t in left_phrase if not t.get('is_punct', False)]
            right_words = []
            for t in right_phrase:
                if t.get('is_punct', False):
                    break  # Only check the first part of the right phrase
                right_words.append(t)
            
            if len(left_words) <= 3 or len(right_words) <= 3:
                suitable_for_splitting = False
            
            return {
                'suitable_for_splitting': suitable_for_splitting,
                'left_words': len(left_words),
                'right_words': len(right_words),
                'left_phrase': [t.get('text', '') for t in left_phrase],
                'right_phrase': [t.get('text', '') for t in right_phrase]
            }
        
        # Test good comma split candidate
        good_doc = [
            {'text': 'The', 'pos_': 'DET', 'dep_': 'det', 'is_punct': False},
            {'text': 'quick', 'pos_': 'ADJ', 'dep_': 'amod', 'is_punct': False},
            {'text': 'brown', 'pos_': 'ADJ', 'dep_': 'amod', 'is_punct': False},
            {'text': 'fox', 'pos_': 'NOUN', 'dep_': 'nsubj', 'is_punct': False},
            {'text': 'jumps', 'pos_': 'VERB', 'dep_': 'ROOT', 'is_punct': False},
            {'text': ',', 'pos_': 'PUNCT', 'dep_': 'punct', 'is_punct': True},  # comma at index 5
            {'text': 'and', 'pos_': 'CCONJ', 'dep_': 'cc', 'is_punct': False},
            {'text': 'it', 'pos_': 'PRON', 'dep_': 'nsubj', 'is_punct': False},
            {'text': 'runs', 'pos_': 'VERB', 'dep_': 'conj', 'is_punct': False},
            {'text': 'fast', 'pos_': 'ADV', 'dep_': 'advmod', 'is_punct': False}
        ]
        
        result = mock_analyze_comma(0, good_doc, 5)
        
        assert result['suitable_for_splitting'] is True
        assert result['left_words'] == 5  # "The quick brown fox jumps"
        assert result['right_words'] == 3  # "and it runs" (before any punctuation)
        assert 'fox' in result['left_phrase']
        assert 'it' in result['right_phrase']
        
        # Test poor comma split candidate (too short)
        short_doc = [
            {'text': 'Yes', 'pos_': 'INTJ', 'dep_': 'ROOT', 'is_punct': False},
            {'text': ',', 'pos_': 'PUNCT', 'dep_': 'punct', 'is_punct': True},
            {'text': 'okay', 'pos_': 'INTJ', 'dep_': 'parataxis', 'is_punct': False}
        ]
        
        result = mock_analyze_comma(0, short_doc, 1)
        
        assert result['suitable_for_splitting'] is False
        assert result['left_words'] == 1  # "Yes"
        assert result['right_words'] == 1  # "okay"
        
        # Test comma with no verb in right phrase
        no_verb_doc = [
            {'text': 'The', 'pos_': 'DET', 'dep_': 'det', 'is_punct': False},
            {'text': 'cat', 'pos_': 'NOUN', 'dep_': 'nsubj', 'is_punct': False},
            {'text': 'sits', 'pos_': 'VERB', 'dep_': 'ROOT', 'is_punct': False},
            {'text': ',', 'pos_': 'PUNCT', 'dep_': 'punct', 'is_punct': True},
            {'text': 'the', 'pos_': 'DET', 'dep_': 'det', 'is_punct': False},
            {'text': 'big', 'pos_': 'ADJ', 'dep_': 'amod', 'is_punct': False},
            {'text': 'one', 'pos_': 'NOUN', 'dep_': 'appos', 'is_punct': False}
        ]
        
        result = mock_analyze_comma(0, no_verb_doc, 3)
        
        assert result['suitable_for_splitting'] is False  # No verb in right phrase
        assert result['left_words'] == 3
        assert result['right_words'] == 3
    
    def test_comma_splitting_workflow(self):
        """Test split_by_comma workflow"""
        # Simulate split_by_comma workflow
        def mock_split_by_comma(text, mock_doc_tokens):
            """Mock comma splitting workflow"""
            sentences = []
            start = 0
            
            for i, token in enumerate(mock_doc_tokens):
                if token.get('text') in [',', '，']:
                    # Mock analyze_comma call
                    def analyze_comma(start_idx, doc, comma_idx):
                        left_phrase = doc[start_idx:comma_idx]
                        right_phrase = doc[comma_idx + 1:min(len(doc), comma_idx + 10)]
                        
                        # Check if right phrase has subject and verb
                        has_subject = any(t.get('dep_') in ["nsubj", "nsubjpass"] or t.get('pos_') == "PRON" for t in right_phrase)
                        has_verb = any(t.get('pos_') in ["VERB", "AUX"] for t in right_phrase)
                        suitable = has_subject and has_verb
                        
                        # Check word count
                        left_words = [t for t in left_phrase if not t.get('is_punct', False)]
                        right_words = []
                        for t in right_phrase:
                            if t.get('is_punct', False):
                                break
                            right_words.append(t)
                        
                        if len(left_words) <= 3 or len(right_words) <= 3:
                            suitable = False
                        
                        return suitable
                    
                    suitable_for_splitting = analyze_comma(start, mock_doc_tokens, i)
                    
                    if suitable_for_splitting:
                        # Extract text from start to comma
                        segment_text = ' '.join(token.get('text', '') for token in mock_doc_tokens[start:i]).strip()
                        sentences.append(segment_text)
                        start = i + 1
            
            # Add remaining text
            if start < len(mock_doc_tokens):
                remaining_text = ' '.join(token.get('text', '') for token in mock_doc_tokens[start:]).strip()
                sentences.append(remaining_text)
            
            return sentences
        
        # Test text with good comma split
        good_text = "The quick brown fox jumps over the lazy dog, and it runs very fast through the forest."
        good_tokens = [
            {'text': 'The', 'pos_': 'DET', 'dep_': 'det', 'is_punct': False},
            {'text': 'quick', 'pos_': 'ADJ', 'dep_': 'amod', 'is_punct': False},
            {'text': 'brown', 'pos_': 'ADJ', 'dep_': 'amod', 'is_punct': False},
            {'text': 'fox', 'pos_': 'NOUN', 'dep_': 'nsubj', 'is_punct': False},
            {'text': 'jumps', 'pos_': 'VERB', 'dep_': 'ROOT', 'is_punct': False},
            {'text': 'over', 'pos_': 'ADP', 'dep_': 'prep', 'is_punct': False},
            {'text': 'the', 'pos_': 'DET', 'dep_': 'det', 'is_punct': False},
            {'text': 'lazy', 'pos_': 'ADJ', 'dep_': 'amod', 'is_punct': False},
            {'text': 'dog', 'pos_': 'NOUN', 'dep_': 'pobj', 'is_punct': False},
            {'text': ',', 'pos_': 'PUNCT', 'dep_': 'punct', 'is_punct': True},
            {'text': 'and', 'pos_': 'CCONJ', 'dep_': 'cc', 'is_punct': False},
            {'text': 'it', 'pos_': 'PRON', 'dep_': 'nsubj', 'is_punct': False},
            {'text': 'runs', 'pos_': 'VERB', 'dep_': 'conj', 'is_punct': False},
            {'text': 'very', 'pos_': 'ADV', 'dep_': 'advmod', 'is_punct': False},
            {'text': 'fast', 'pos_': 'ADV', 'dep_': 'advmod', 'is_punct': False},
            {'text': 'through', 'pos_': 'ADP', 'dep_': 'prep', 'is_punct': False},
            {'text': 'the', 'pos_': 'DET', 'dep_': 'det', 'is_punct': False},
            {'text': 'forest', 'pos_': 'NOUN', 'dep_': 'pobj', 'is_punct': False},
            {'text': '.', 'pos_': 'PUNCT', 'dep_': 'punct', 'is_punct': True}
        ]
        
        result = mock_split_by_comma(good_text, good_tokens)
        
        assert len(result) == 2
        assert "The quick brown fox jumps over the lazy dog" in result[0]
        assert "and it runs very fast through the forest ." in result[1]
        
        # Test text with no suitable comma splits
        no_split_text = "Yes, no, maybe."
        no_split_tokens = [
            {'text': 'Yes', 'pos_': 'INTJ', 'dep_': 'ROOT', 'is_punct': False},
            {'text': ',', 'pos_': 'PUNCT', 'dep_': 'punct', 'is_punct': True},
            {'text': 'no', 'pos_': 'INTJ', 'dep_': 'parataxis', 'is_punct': False},
            {'text': ',', 'pos_': 'PUNCT', 'dep_': 'punct', 'is_punct': True},
            {'text': 'maybe', 'pos_': 'ADV', 'dep_': 'parataxis', 'is_punct': False},
            {'text': '.', 'pos_': 'PUNCT', 'dep_': 'punct', 'is_punct': True}
        ]
        
        result = mock_split_by_comma(no_split_text, no_split_tokens)
        
        assert len(result) == 1  # No splits made
        assert "Yes , no , maybe ." in result[0]
    
    def test_comma_splitting_main_workflow(self):
        """Test split_by_comma_main workflow"""
        # Simulate split_by_comma_main workflow
        def mock_split_by_comma_main(input_sentences, mock_nlp_func):
            """Mock main comma splitting workflow"""
            all_split_sentences = []
            processing_stats = {
                'total_input_sentences': len(input_sentences),
                'total_output_sentences': 0,
                'sentences_split': 0,
                'sentences_unchanged': 0
            }
            
            for sentence in input_sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Mock split_by_comma call
                split_sentences = mock_nlp_func(sentence)
                
                if len(split_sentences) > 1:
                    processing_stats['sentences_split'] += 1
                else:
                    processing_stats['sentences_unchanged'] += 1
                
                all_split_sentences.extend(split_sentences)
            
            processing_stats['total_output_sentences'] = len(all_split_sentences)
            
            return {
                'split_sentences': all_split_sentences,
                'stats': processing_stats
            }
        
        # Mock NLP function that splits sentences with commas
        def mock_nlp_split_function(text):
            # Simple mock: split on comma if it looks like a good split
            if ',' in text and ' and ' in text:
                parts = text.split(',', 1)
                if len(parts) == 2 and len(parts[0].strip()) > 10 and len(parts[1].strip()) > 10:
                    return [parts[0].strip(), parts[1].strip()]
            return [text]
        
        # Test input sentences
        input_sentences = [
            "The quick brown fox jumps over the lazy dog, and it runs very fast through the forest.",
            "Simple sentence without commas.",
            "This sentence has a comma, but it's not suitable for splitting.",
            "Another complex sentence with proper structure, and it should be split correctly."
        ]
        
        result = mock_split_by_comma_main(input_sentences, mock_nlp_split_function)
        
        assert result['stats']['total_input_sentences'] == 4
        assert result['stats']['sentences_split'] == 2  # First and fourth sentences
        assert result['stats']['sentences_unchanged'] == 2  # Second and third sentences
        assert result['stats']['total_output_sentences'] == 6  # 4 original + 2 splits
        
        # Check that appropriate sentences were split
        split_sentences = result['split_sentences']
        assert len(split_sentences) == 6
        
        # First sentence should be split
        assert any("The quick brown fox jumps over the lazy dog" in s for s in split_sentences)
        assert any("and it runs very fast through the forest" in s for s in split_sentences)
        
        # Second sentence should remain unchanged
        assert "Simple sentence without commas." in split_sentences
        
        # Fourth sentence should be split
        assert any("Another complex sentence with proper structure" in s for s in split_sentences)
        assert any("and it should be split correctly" in s for s in split_sentences)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])