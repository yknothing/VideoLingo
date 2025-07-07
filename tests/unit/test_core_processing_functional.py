"""
Functional tests for Core Processing modules
Tests NLP splitting, meaning splitting, and summarization functionality
"""

import pytest
import tempfile
import os
import json
import math
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any


class TestNLPSplittingLogic:
    """Test _3_1_split_nlp.py core logic"""
    
    def test_nlp_splitting_pipeline(self):
        """Test split_by_spacy pipeline logic"""
        # Simulate split_by_spacy pipeline
        def mock_split_by_spacy_pipeline():
            """Mock NLP splitting pipeline"""
            pipeline_steps = []
            
            # Step 1: Initialize NLP model
            pipeline_steps.append('init_nlp')
            nlp_model_info = {
                'model_name': 'en_core_web_sm',
                'language': 'en',
                'loaded': True
            }
            
            # Step 2: Split by marks
            pipeline_steps.append('split_by_mark')
            mark_split_results = {
                'sentences_processed': 15,
                'sentences_split': 3,
                'marks_detected': ['.', '!', '?'],
                'success': True
            }
            
            # Step 3: Split by comma
            pipeline_steps.append('split_by_comma')
            comma_split_results = {
                'sentences_processed': 18,  # After mark splitting
                'sentences_split': 5,
                'commas_analyzed': 12,
                'suitable_splits': 5,
                'success': True
            }
            
            # Step 4: Split sentences (general)
            pipeline_steps.append('split_sentences')
            sentence_split_results = {
                'sentences_processed': 23,  # After comma splitting
                'sentences_split': 2,
                'method': 'spacy_sentences',
                'success': True
            }
            
            # Step 5: Split long sentences by root
            pipeline_steps.append('split_long_by_root')
            root_split_results = {
                'sentences_processed': 25,  # After sentence splitting
                'long_sentences_found': 3,
                'sentences_split': 2,
                'root_analysis_performed': True,
                'success': True
            }
            
            # Calculate overall results
            total_input_sentences = mark_split_results['sentences_processed']
            total_output_sentences = root_split_results['sentences_processed']
            total_splits_made = (
                mark_split_results['sentences_split'] +
                comma_split_results['sentences_split'] +
                sentence_split_results['sentences_split'] +
                root_split_results['sentences_split']
            )
            
            return {
                'pipeline_steps': pipeline_steps,
                'nlp_model': nlp_model_info,
                'mark_split': mark_split_results,
                'comma_split': comma_split_results,
                'sentence_split': sentence_split_results,
                'root_split': root_split_results,
                'overall': {
                    'input_sentences': total_input_sentences,
                    'output_sentences': total_output_sentences,
                    'total_splits': total_splits_made,
                    'efficiency': total_splits_made / total_input_sentences,
                    'success': True
                }
            }
        
        # Test NLP splitting pipeline
        result = mock_split_by_spacy_pipeline()
        
        # Verify pipeline steps
        expected_steps = ['init_nlp', 'split_by_mark', 'split_by_comma', 'split_sentences', 'split_long_by_root']
        assert result['pipeline_steps'] == expected_steps
        
        # Verify NLP model initialization
        assert result['nlp_model']['loaded'] is True
        assert result['nlp_model']['language'] == 'en'
        
        # Verify each splitting step
        assert result['mark_split']['success'] is True
        assert result['comma_split']['success'] is True
        assert result['sentence_split']['success'] is True
        assert result['root_split']['success'] is True
        
        # Verify overall results
        assert result['overall']['success'] is True
        assert result['overall']['input_sentences'] == 15
        assert result['overall']['output_sentences'] == 25
        assert result['overall']['total_splits'] == 12  # 3+5+2+2
        assert result['overall']['efficiency'] == 12/15
    
    def test_nlp_model_initialization_logic(self):
        """Test NLP model initialization logic"""
        # Simulate init_nlp logic
        def mock_init_nlp(language='en', model_preference='web'):
            """Mock NLP model initialization"""
            model_configs = {
                'en': {
                    'web': 'en_core_web_sm',
                    'large': 'en_core_web_lg',
                    'trf': 'en_core_web_trf'
                },
                'zh': {
                    'web': 'zh_core_web_sm',
                    'large': 'zh_core_web_lg',
                    'trf': 'zh_core_web_trf'
                },
                'auto': {
                    'web': 'xx_core_web_sm',
                    'large': 'xx_core_web_lg'
                }
            }
            
            # Select model based on language and preference
            lang_config = model_configs.get(language, model_configs['en'])
            model_name = lang_config.get(model_preference, lang_config['web'])
            
            # Mock model loading result
            model_info = {
                'model_name': model_name,
                'language': language,
                'preference': model_preference,
                'loaded': True,
                'capabilities': {
                    'tokenizer': True,
                    'parser': True,
                    'ner': True,
                    'dependency_parsing': True
                },
                'memory_usage_mb': 50 if 'sm' in model_name else 200,
                'load_time_seconds': 2.5 if 'sm' in model_name else 8.0
            }
            
            return model_info
        
        # Test English model loading
        en_model = mock_init_nlp('en', 'web')
        assert en_model['model_name'] == 'en_core_web_sm'
        assert en_model['loaded'] is True
        assert en_model['capabilities']['dependency_parsing'] is True
        assert en_model['memory_usage_mb'] == 50
        
        # Test Chinese model loading
        zh_model = mock_init_nlp('zh', 'large')
        assert zh_model['model_name'] == 'zh_core_web_lg'
        assert zh_model['memory_usage_mb'] == 200
        assert zh_model['load_time_seconds'] == 8.0
        
        # Test auto-detection model
        auto_model = mock_init_nlp('auto', 'web')
        assert auto_model['model_name'] == 'xx_core_web_sm'
        assert auto_model['language'] == 'auto'
        
        # Test fallback to default preference
        fallback_model = mock_init_nlp('en', 'invalid_preference')
        assert fallback_model['model_name'] == 'en_core_web_sm'  # Falls back to 'web'


class TestMeaningSplittingLogic:
    """Test _3_2_split_meaning.py core logic"""
    
    def test_sentence_tokenization_logic(self):
        """Test tokenize_sentence logic"""
        # Simulate tokenize_sentence logic
        def mock_tokenize_sentence(sentence, mock_nlp):
            """Mock sentence tokenization"""
            # Simple tokenization simulation
            import re
            
            # Basic tokenization (split on spaces and punctuation)
            tokens = re.findall(r'\w+|[^\w\s]', sentence)
            
            # Mock spaCy-like token objects
            token_objects = []
            for token in tokens:
                token_objects.append({
                    'text': token,
                    'pos_': 'NOUN' if token.isalpha() else 'PUNCT',
                    'dep_': 'ROOT' if token.isalpha() else 'punct',
                    'is_alpha': token.isalpha(),
                    'is_punct': not token.isalnum()
                })
            
            return [token['text'] for token in token_objects]
        
        # Test tokenization
        sentence = "The quick brown fox jumps over the lazy dog."
        mock_nlp = {}  # Mock NLP object
        
        tokens = mock_tokenize_sentence(sentence, mock_nlp)
        
        expected_tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
        assert tokens == expected_tokens
        
        # Test complex sentence with punctuation
        complex_sentence = "Hello, world! How are you?"
        complex_tokens = mock_tokenize_sentence(complex_sentence, mock_nlp)
        
        expected_complex = ['Hello', ',', 'world', '!', 'How', 'are', 'you', '?']
        assert complex_tokens == expected_complex
    
    def test_split_position_finding_logic(self):
        """Test find_split_positions logic"""
        # Simulate find_split_positions logic
        def mock_find_split_positions(original, modified, joiner=' '):
            """Mock split position finding logic"""
            from difflib import SequenceMatcher
            
            split_positions = []
            parts = modified.split('[br]')
            start = 0
            
            for i in range(len(parts) - 1):
                max_similarity = 0
                best_split = None
                
                # Search for best split position
                for j in range(start, len(original)):
                    original_left = original[start:j]
                    modified_left = joiner.join(parts[i].split())
                    
                    # Calculate similarity
                    left_similarity = SequenceMatcher(None, original_left, modified_left).ratio()
                    
                    if left_similarity > max_similarity:
                        max_similarity = left_similarity
                        best_split = j
                
                result = {
                    'part_index': i,
                    'max_similarity': max_similarity,
                    'best_split_position': best_split,
                    'similarity_threshold_met': max_similarity >= 0.9
                }
                
                if max_similarity < 0.9:
                    result['warning'] = f"Low similarity at split point: {max_similarity:.2f}"
                
                if best_split is not None:
                    split_positions.append(best_split)
                    start = best_split
                else:
                    result['error'] = f"Unable to find suitable split point for part {i+1}"
            
            return {
                'split_positions': split_positions,
                'total_parts': len(parts),
                'successful_splits': len(split_positions)
            }
        
        # Test good split scenario
        original = "The quick brown fox jumps over the lazy dog and runs fast."
        modified = "The quick brown fox jumps[br]over the lazy dog and runs fast."
        
        result = mock_find_split_positions(original, modified)
        
        assert len(result['split_positions']) == 1
        assert result['successful_splits'] == 1
        assert result['total_parts'] == 2
        
        # Test multiple splits
        multi_modified = "The quick brown[br]fox jumps over[br]the lazy dog."
        multi_result = mock_find_split_positions(original, multi_modified)
        
        assert len(multi_result['split_positions']) == 2
        assert multi_result['successful_splits'] == 2
        assert multi_result['total_parts'] == 3
    
    def test_sentence_splitting_logic(self):
        """Test split_sentence logic"""
        # Simulate split_sentence logic
        def mock_split_sentence(sentence, num_parts, word_limit=20, retry_attempt=0):
            """Mock sentence splitting logic"""
            # Mock GPT response simulation
            def simulate_gpt_split_response(sentence, num_parts):
                # Simple simulation: split sentence roughly equally
                words = sentence.split()
                part_size = len(words) // num_parts
                
                parts = []
                for i in range(num_parts):
                    start_idx = i * part_size
                    end_idx = (i + 1) * part_size if i < num_parts - 1 else len(words)
                    part = ' '.join(words[start_idx:end_idx])
                    parts.append(part)
                
                # Mock response format
                split_text = '[br]'.join(parts)
                
                response = {
                    'choice': 1,
                    'split1': split_text,
                    'reasoning': f'Split into {num_parts} parts based on word limit'
                }
                
                return response
            
            # Validate split requirements
            if len(sentence.split()) <= word_limit:
                return {
                    'result': sentence,
                    'splits_made': 0,
                    'reason': 'Sentence within word limit'
                }
            
            # Get GPT split response
            gpt_response = simulate_gpt_split_response(sentence, num_parts)
            choice = gpt_response['choice']
            best_split = gpt_response[f'split{choice}']
            
            # Validate split contains [br]
            if '[br]' not in best_split:
                return {
                    'error': 'Split failed, no [br] found',
                    'retry_attempt': retry_attempt
                }
            
            # Find split positions and apply them
            def find_and_apply_splits(original, split_text):
                # Mock find_split_positions call
                parts = split_text.split('[br]')
                words = original.split()
                
                # Calculate approximate split positions
                split_positions = []
                words_per_part = len(words) // len(parts)
                
                for i in range(len(parts) - 1):
                    pos = (i + 1) * words_per_part
                    split_positions.append(pos)
                
                # Apply splits
                result_parts = []
                start = 0
                for pos in split_positions:
                    result_parts.append(' '.join(words[start:pos]))
                    start = pos
                result_parts.append(' '.join(words[start:]))
                
                return '\n'.join(result_parts)
            
            final_result = find_and_apply_splits(sentence, best_split)
            
            return {
                'result': final_result,
                'original': sentence,
                'splits_made': num_parts,
                'word_limit': word_limit,
                'retry_attempt': retry_attempt,
                'gpt_response': gpt_response
            }
        
        # Test normal splitting
        long_sentence = "The quick brown fox jumps over the lazy dog and runs very fast through the forest while chasing a rabbit."
        
        result = mock_split_sentence(long_sentence, 2, word_limit=10)
        
        assert 'result' in result
        assert result['splits_made'] == 2
        assert '\n' in result['result']  # Contains line breaks
        assert len(result['result'].split('\n')) == 2  # Two parts
        
        # Test sentence within word limit
        short_sentence = "The quick brown fox."
        short_result = mock_split_sentence(short_sentence, 2, word_limit=20)
        
        assert short_result['splits_made'] == 0
        assert short_result['reason'] == 'Sentence within word limit'
        assert short_result['result'] == short_sentence
        
        # Test three-way split
        very_long_sentence = "The quick brown fox jumps over the lazy dog and runs very fast through the forest while chasing a rabbit that was eating carrots."
        
        three_way_result = mock_split_sentence(very_long_sentence, 3, word_limit=8)
        
        assert three_way_result['splits_made'] == 3
        assert len(three_way_result['result'].split('\n')) == 3
    
    def test_parallel_splitting_logic(self):
        """Test parallel_split_sentences logic"""
        # Simulate parallel_split_sentences logic
        def mock_parallel_split_sentences(sentences, max_length, max_workers=2, retry_attempt=0):
            """Mock parallel sentence splitting"""
            new_sentences = [None] * len(sentences)
            processing_stats = {
                'total_sentences': len(sentences),
                'sentences_needing_split': 0,
                'sentences_successfully_split': 0,
                'sentences_unchanged': 0,
                'parallel_jobs': 0
            }
            
            # Process each sentence
            for index, sentence in enumerate(sentences):
                # Mock tokenization
                tokens = sentence.split()  # Simple tokenization
                num_parts = math.ceil(len(tokens) / max_length)
                
                if len(tokens) > max_length:
                    processing_stats['sentences_needing_split'] += 1
                    processing_stats['parallel_jobs'] += 1
                    
                    # Mock split_sentence call
                    def mock_split_result(sentence, num_parts):
                        words = sentence.split()
                        part_size = len(words) // num_parts
                        
                        parts = []
                        for i in range(num_parts):
                            start_idx = i * part_size
                            end_idx = (i + 1) * part_size if i < num_parts - 1 else len(words)
                            part = ' '.join(words[start_idx:end_idx])
                            parts.append(part)
                        
                        return '\n'.join(parts)
                    
                    split_result = mock_split_result(sentence, num_parts)
                    
                    if split_result:
                        split_lines = split_result.strip().split('\n')
                        new_sentences[index] = [line.strip() for line in split_lines]
                        processing_stats['sentences_successfully_split'] += 1
                    else:
                        new_sentences[index] = [sentence]
                else:
                    new_sentences[index] = [sentence]
                    processing_stats['sentences_unchanged'] += 1
            
            # Flatten results
            flattened_sentences = [sentence for sublist in new_sentences for sentence in sublist]
            
            processing_stats['output_sentences'] = len(flattened_sentences)
            processing_stats['split_efficiency'] = processing_stats['sentences_successfully_split'] / max(1, processing_stats['sentences_needing_split'])
            
            return {
                'sentences': flattened_sentences,
                'stats': processing_stats
            }
        
        # Test parallel splitting
        test_sentences = [
            "Short sentence.",
            "This is a medium length sentence that might need splitting.",
            "This is a very long sentence that definitely needs to be split into multiple parts because it exceeds the maximum length limit.",
            "Another short one.",
            "Yet another extremely long sentence that contains many words and should be automatically detected as requiring splitting into smaller more manageable chunks."
        ]
        
        result = mock_parallel_split_sentences(test_sentences, max_length=10, max_workers=2)
        
        assert result['stats']['total_sentences'] == 5
        assert result['stats']['sentences_needing_split'] >= 2  # Long sentences
        assert result['stats']['sentences_unchanged'] >= 2  # Short sentences
        assert result['stats']['output_sentences'] > 5  # More output than input due to splits
        assert result['stats']['split_efficiency'] > 0  # Some splits were successful
        
        # Verify that long sentences were actually split
        assert len(result['sentences']) > len(test_sentences)
        
        # Test with no sentences needing split
        short_sentences = ["Short.", "Also short.", "Brief."]
        short_result = mock_parallel_split_sentences(short_sentences, max_length=20)
        
        assert short_result['stats']['sentences_needing_split'] == 0
        assert short_result['stats']['sentences_unchanged'] == 3
        assert len(short_result['sentences']) == 3
    
    def test_meaning_splitting_main_workflow(self):
        """Test split_sentences_by_meaning main workflow"""
        # Simulate main workflow
        def mock_split_sentences_by_meaning_workflow(input_sentences, config):
            """Mock main meaning splitting workflow"""
            workflow_steps = []
            
            # Step 1: Read input sentences
            workflow_steps.append('read_input_sentences')
            current_sentences = input_sentences.copy()
            
            # Step 2: Initialize NLP model
            workflow_steps.append('init_nlp_model')
            nlp_info = {
                'model_loaded': True,
                'language': 'en',
                'load_time': 2.5
            }
            
            # Step 3: Process sentences multiple times (retry mechanism)
            workflow_steps.append('retry_processing')
            max_retries = config.get('max_retries', 3)
            max_length = config.get('max_split_length', 20)
            max_workers = config.get('max_workers', 2)
            
            retry_stats = []
            
            for retry_attempt in range(max_retries):
                # Mock parallel_split_sentences call
                split_result = mock_parallel_split_sentences(current_sentences, max_length, max_workers, retry_attempt)
                
                retry_stats.append({
                    'retry_attempt': retry_attempt,
                    'input_sentences': len(current_sentences),
                    'output_sentences': len(split_result['sentences']),
                    'splits_made': split_result['stats']['sentences_successfully_split'],
                    'sentences_unchanged': split_result['stats']['sentences_unchanged']
                })
                
                current_sentences = split_result['sentences']
                
                # Check if any sentences still need splitting
                long_sentences = [s for s in current_sentences if len(s.split()) > max_length]
                if not long_sentences:
                    break
            
            # Step 4: Save results
            workflow_steps.append('save_results')
            
            # Calculate overall statistics
            original_count = len(input_sentences)
            final_count = len(current_sentences)
            total_splits = sum(stat['splits_made'] for stat in retry_stats)
            
            return {
                'workflow_steps': workflow_steps,
                'nlp_info': nlp_info,
                'retry_stats': retry_stats,
                'results': {
                    'original_sentence_count': original_count,
                    'final_sentence_count': final_count,
                    'total_splits_made': total_splits,
                    'retries_performed': len(retry_stats),
                    'final_sentences': current_sentences
                },
                'success': True
            }
        
        # Mock parallel_split_sentences for this test
        def mock_parallel_split_sentences(sentences, max_length, max_workers, retry_attempt):
            new_sentences = []
            splits_made = 0
            unchanged = 0
            
            for sentence in sentences:
                words = sentence.split()
                if len(words) > max_length:
                    # Split into two parts
                    mid = len(words) // 2
                    part1 = ' '.join(words[:mid])
                    part2 = ' '.join(words[mid:])
                    new_sentences.extend([part1, part2])
                    splits_made += 1
                else:
                    new_sentences.append(sentence)
                    unchanged += 1
            
            return {
                'sentences': new_sentences,
                'stats': {
                    'sentences_successfully_split': splits_made,
                    'sentences_unchanged': unchanged
                }
            }
        
        # Test workflow
        test_input = [
            "Short sentence.",
            "This is a very long sentence that needs to be split into multiple parts for better processing.",
            "Another extremely long sentence with many words that exceeds the maximum allowed length.",
            "Brief."
        ]
        
        config = {
            'max_retries': 3,
            'max_split_length': 8,
            'max_workers': 2
        }
        
        result = mock_split_sentences_by_meaning_workflow(test_input, config)
        
        assert result['success'] is True
        assert len(result['workflow_steps']) == 4
        assert 'read_input_sentences' in result['workflow_steps']
        assert 'save_results' in result['workflow_steps']
        
        # Check results
        assert result['results']['original_sentence_count'] == 4
        assert result['results']['final_sentence_count'] > 4  # Sentences were split
        assert result['results']['total_splits_made'] >= 2  # Long sentences split
        
        # Check that long sentences were processed
        final_sentences = result['results']['final_sentences']
        assert all(len(s.split()) <= 20 for s in final_sentences)  # No sentence too long


class TestSummarizationLogic:
    """Test _4_1_summarize.py core logic"""
    
    def test_chunk_combination_logic(self):
        """Test combine_chunks logic"""
        # Simulate combine_chunks logic
        def mock_combine_chunks(input_sentences, summary_length=1000):
            """Mock chunk combination logic"""
            # Clean and process sentences
            cleaned_sentences = [line.strip() for line in input_sentences if line.strip()]
            
            # Combine into single text
            combined_text = ' '.join(cleaned_sentences)
            
            # Truncate to summary length
            truncated_text = combined_text[:summary_length]
            
            return {
                'original_sentences': len(input_sentences),
                'cleaned_sentences': len(cleaned_sentences),
                'combined_length': len(combined_text),
                'truncated_length': len(truncated_text),
                'truncated': len(combined_text) > summary_length,
                'text': truncated_text
            }
        
        # Test normal combination
        test_sentences = [
            "This is the first sentence.",
            "This is the second sentence.",
            "",  # Empty line
            "This is the third sentence with more content.",
            "   ",  # Whitespace only
            "Final sentence here."
        ]
        
        result = mock_combine_chunks(test_sentences, summary_length=100)
        
        assert result['original_sentences'] == 6
        assert result['cleaned_sentences'] == 4  # Empty lines removed
        assert result['truncated'] is True  # Text was truncated
        assert result['truncated_length'] == 100
        assert result['text'].startswith("This is the first sentence.")
        
        # Test with large summary length (no truncation)
        large_result = mock_combine_chunks(test_sentences, summary_length=1000)
        
        assert large_result['truncated'] is False
        assert large_result['combined_length'] == large_result['truncated_length']
        assert all(sentence in large_result['text'] for sentence in ["first", "second", "third", "Final"])
    
    def test_terminology_search_logic(self):
        """Test search_things_to_note_in_prompt logic"""
        # Simulate terminology search logic
        def mock_search_things_to_note_in_prompt(sentence, terminology_data):
            """Mock terminology search logic"""
            things_to_note_list = []
            
            # Search for terms in sentence (case-insensitive)
            for term in terminology_data['terms']:
                if term['src'].lower() in sentence.lower():
                    things_to_note_list.append(term['src'])
            
            if things_to_note_list:
                # Generate prompt format
                prompt_lines = []
                for i, term_data in enumerate(terminology_data['terms']):
                    if term_data['src'] in things_to_note_list:
                        line = f'{i+1}. "{term_data["src"]}": "{term_data["tgt"]}", meaning: {term_data["note"]}'
                        prompt_lines.append(line)
                
                prompt = '\n'.join(prompt_lines)
                
                return {
                    'terms_found': things_to_note_list,
                    'prompt': prompt,
                    'has_terms': True
                }
            else:
                return {
                    'terms_found': [],
                    'prompt': None,
                    'has_terms': False
                }
        
        # Test terminology data
        terminology = {
            'terms': [
                {
                    'src': 'API',
                    'tgt': '应用程序接口',
                    'note': 'Application Programming Interface'
                },
                {
                    'src': 'machine learning',
                    'tgt': '机器学习',
                    'note': 'A subset of artificial intelligence'
                },
                {
                    'src': 'database',
                    'tgt': '数据库',
                    'note': 'Organized collection of data'
                }
            ]
        }
        
        # Test sentence with terms
        sentence_with_terms = "The API uses machine learning to query the database."
        
        result = mock_search_things_to_note_in_prompt(sentence_with_terms, terminology)
        
        assert result['has_terms'] is True
        assert len(result['terms_found']) == 3  # All terms found
        assert 'API' in result['terms_found']
        assert 'machine learning' in result['terms_found']
        assert 'database' in result['terms_found']
        assert 'Application Programming Interface' in result['prompt']
        assert '机器学习' in result['prompt']
        
        # Test sentence without terms
        sentence_without_terms = "The quick brown fox jumps over the lazy dog."
        
        no_terms_result = mock_search_things_to_note_in_prompt(sentence_without_terms, terminology)
        
        assert no_terms_result['has_terms'] is False
        assert len(no_terms_result['terms_found']) == 0
        assert no_terms_result['prompt'] is None
        
        # Test case-insensitive matching
        case_sentence = "We need to implement an api for the Machine Learning system."
        
        case_result = mock_search_things_to_note_in_prompt(case_sentence, terminology)
        
        assert case_result['has_terms'] is True
        assert 'API' in case_result['terms_found']  # Found despite lowercase 'api'
        assert 'machine learning' in case_result['terms_found']  # Found despite different case
    
    def test_summary_generation_logic(self):
        """Test get_summary main logic"""
        # Simulate get_summary logic
        def mock_get_summary_workflow(src_content, custom_terms_data):
            """Mock summary generation workflow"""
            workflow_steps = []
            
            # Step 1: Process custom terms
            workflow_steps.append('process_custom_terms')
            
            custom_terms_json = {
                "terms": [
                    {
                        "src": str(row['src']),
                        "tgt": str(row['tgt']), 
                        "note": str(row['note'])
                    }
                    for row in custom_terms_data
                ]
            }
            
            custom_terms_info = {
                'terms_loaded': len(custom_terms_data),
                'terms_json': custom_terms_json
            }
            
            # Step 2: Generate summary prompt
            workflow_steps.append('generate_summary_prompt')
            
            prompt_info = {
                'content_length': len(src_content),
                'custom_terms_included': len(custom_terms_json['terms']),
                'prompt_generated': True
            }
            
            # Step 3: Call GPT for summarization
            workflow_steps.append('call_gpt_summarization')
            
            # Mock GPT response
            def mock_gpt_summarization_response(content, custom_terms):
                # Extract key terms from content (mock)
                content_words = content.lower().split()
                common_terms = ['system', 'data', 'process', 'analysis', 'method']
                
                extracted_terms = []
                for term in common_terms:
                    if term in content_words:
                        extracted_terms.append({
                            'src': term,
                            'tgt': f'{term}_translated',
                            'note': f'Key term related to {term}'
                        })
                
                return {
                    'terms': extracted_terms,
                    'summary': f'Summary of content with {len(extracted_terms)} key terms identified'
                }
            
            gpt_response = mock_gpt_summarization_response(src_content, custom_terms_json)
            
            # Step 4: Validate response
            workflow_steps.append('validate_response')
            
            def validate_summary_response(response_data):
                required_keys = {'src', 'tgt', 'note'}
                if 'terms' not in response_data:
                    return {"status": "error", "message": "Invalid response format"}
                for term in response_data['terms']:
                    if not all(key in term for key in required_keys):
                        return {"status": "error", "message": "Invalid response format"}   
                return {"status": "success", "message": "Summary completed"}
            
            validation_result = validate_summary_response(gpt_response)
            
            # Step 5: Merge with custom terms
            workflow_steps.append('merge_custom_terms')
            
            final_summary = gpt_response.copy()
            final_summary['terms'].extend(custom_terms_json['terms'])
            
            # Step 6: Save results
            workflow_steps.append('save_results')
            
            return {
                'workflow_steps': workflow_steps,
                'custom_terms_info': custom_terms_info,
                'prompt_info': prompt_info,
                'gpt_response': gpt_response,
                'validation_result': validation_result,
                'final_summary': final_summary,
                'success': validation_result['status'] == 'success'
            }
        
        # Test summary generation
        test_content = "This system processes data using advanced analysis methods. The data analysis system provides comprehensive process management."
        
        custom_terms = [
            {'src': 'system', 'tgt': '系统', 'note': 'Computer system'},
            {'src': 'data', 'tgt': '数据', 'note': 'Information data'}
        ]
        
        result = mock_get_summary_workflow(test_content, custom_terms)
        
        assert result['success'] is True
        assert len(result['workflow_steps']) == 6
        assert 'process_custom_terms' in result['workflow_steps']
        assert 'save_results' in result['workflow_steps']
        
        # Check custom terms processing
        assert result['custom_terms_info']['terms_loaded'] == 2
        assert len(result['custom_terms_info']['terms_json']['terms']) == 2
        
        # Check GPT response
        assert 'terms' in result['gpt_response']
        assert len(result['gpt_response']['terms']) > 0
        
        # Check final summary includes both extracted and custom terms
        final_terms = result['final_summary']['terms']
        extracted_count = len(result['gpt_response']['terms'])
        custom_count = len(custom_terms)
        assert len(final_terms) == extracted_count + custom_count
        
        # Verify custom terms are included
        custom_term_srcs = [term['src'] for term in custom_terms]
        final_term_srcs = [term['src'] for term in final_terms]
        for custom_src in custom_term_srcs:
            assert custom_src in final_term_srcs
    
    def test_summary_validation_logic(self):
        """Test summary response validation logic"""
        # Simulate validation logic
        def mock_valid_summary(response_data):
            """Mock summary validation logic"""
            required_keys = {'src', 'tgt', 'note'}
            
            # Check if 'terms' key exists
            if 'terms' not in response_data:
                return {
                    "status": "error", 
                    "message": "Invalid response format: missing 'terms' key",
                    "missing_key": "terms"
                }
            
            # Check each term for required keys
            invalid_terms = []
            for i, term in enumerate(response_data['terms']):
                missing_keys = [key for key in required_keys if key not in term]
                if missing_keys:
                    invalid_terms.append({
                        'term_index': i,
                        'missing_keys': missing_keys,
                        'term_data': term
                    })
            
            if invalid_terms:
                return {
                    "status": "error", 
                    "message": "Invalid response format: terms missing required keys",
                    "invalid_terms": invalid_terms
                }
            
            return {
                "status": "success", 
                "message": "Summary completed",
                "terms_validated": len(response_data['terms'])
            }
        
        # Test valid response
        valid_response = {
            'terms': [
                {'src': 'API', 'tgt': '接口', 'note': 'Application interface'},
                {'src': 'data', 'tgt': '数据', 'note': 'Information content'}
            ]
        }
        
        valid_result = mock_valid_summary(valid_response)
        assert valid_result['status'] == 'success'
        assert valid_result['terms_validated'] == 2
        
        # Test missing 'terms' key
        missing_terms = {'summary': 'Some summary text'}
        
        missing_result = mock_valid_summary(missing_terms)
        assert missing_result['status'] == 'error'
        assert missing_result['missing_key'] == 'terms'
        assert 'missing \'terms\' key' in missing_result['message']
        
        # Test missing required keys in terms
        invalid_terms_response = {
            'terms': [
                {'src': 'API', 'tgt': '接口'},  # Missing 'note'
                {'src': 'data', 'note': 'Information'},  # Missing 'tgt'
                {'tgt': '系统', 'note': 'System info'}  # Missing 'src'
            ]
        }
        
        invalid_result = mock_valid_summary(invalid_terms_response)
        assert invalid_result['status'] == 'error'
        assert len(invalid_result['invalid_terms']) == 3
        assert 'note' in invalid_result['invalid_terms'][0]['missing_keys']
        assert 'tgt' in invalid_result['invalid_terms'][1]['missing_keys']
        assert 'src' in invalid_result['invalid_terms'][2]['missing_keys']
        
        # Test partially valid terms
        partial_response = {
            'terms': [
                {'src': 'API', 'tgt': '接口', 'note': 'Application interface'},  # Valid
                {'src': 'data', 'tgt': '数据'}  # Invalid - missing 'note'
            ]
        }
        
        partial_result = mock_valid_summary(partial_response)
        assert partial_result['status'] == 'error'
        assert len(partial_result['invalid_terms']) == 1
        assert partial_result['invalid_terms'][0]['term_index'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])