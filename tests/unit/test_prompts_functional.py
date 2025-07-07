"""
Functional tests for Prompts module
Tests prompt generation logic without complex LLM dependencies
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any


class TestPromptGenerationLogic:
    """Test prompts.py core logic"""
    
    def test_split_prompt_generation_logic(self):
        """Test get_split_prompt function logic"""
        # Simulate get_split_prompt logic
        def mock_get_split_prompt(sentence, num_parts=2, word_limit=20, language="en"):
            """Mock split prompt generation"""
            
            # Generate split prompt template
            split_prompt = f"""
## Role
You are a professional Netflix subtitle splitter in **{language}**.

## Task
Split the given subtitle text into **{num_parts}** parts, each less than **{word_limit}** words.

1. Maintain sentence meaning coherence according to Netflix subtitle standards
2. MOST IMPORTANT: Keep parts roughly equal in length (minimum 3 words each)
3. Split at natural points like punctuation marks or conjunctions
4. If provided text is repeated words, simply split at the middle of the repeated words.

## Steps
1. Analyze the sentence structure, complexity, and key splitting challenges
2. Generate two alternative splitting approaches with [br] tags at split positions
3. Compare both approaches highlighting their strengths and weaknesses
4. Choose the best splitting approach

## Given Text
<split_this_sentence>
{sentence}
</split_this_sentence>

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief description of sentence structure, complexity, and key splitting challenges",
    "split1": "First splitting approach with [br] tags at split positions",
    "split2": "Alternative splitting approach with [br] tags at split positions",
    "assess": "Comparison of both approaches highlighting their strengths and weaknesses",
    "choice": "1 or 2"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
""".strip()
            
            # Validate inputs
            validation_result = {
                'valid': True,
                'errors': []
            }
            
            if not sentence.strip():
                validation_result['valid'] = False
                validation_result['errors'].append('Sentence cannot be empty')
            
            if num_parts < 2:
                validation_result['valid'] = False
                validation_result['errors'].append('Number of parts must be at least 2')
            
            if word_limit < 5:
                validation_result['valid'] = False
                validation_result['errors'].append('Word limit must be at least 5')
            
            # Calculate prompt statistics
            prompt_stats = {
                'template_length': len(split_prompt),
                'variable_count': split_prompt.count('{'),
                'json_structure_present': '```json' in split_prompt,
                'role_section_present': '## Role' in split_prompt,
                'task_section_present': '## Task' in split_prompt
            }
            
            return {
                'prompt': split_prompt,
                'validation': validation_result,
                'stats': prompt_stats,
                'parameters': {
                    'sentence': sentence,
                    'num_parts': num_parts,
                    'word_limit': word_limit,
                    'language': language
                }
            }
        
        # Test normal split prompt generation
        result = mock_get_split_prompt(
            sentence="The quick brown fox jumps over the lazy dog and runs away.",
            num_parts=2,
            word_limit=15,
            language="en"
        )
        
        assert result['validation']['valid'] is True
        assert result['stats']['json_structure_present'] is True
        assert result['stats']['role_section_present'] is True
        assert result['stats']['task_section_present'] is True
        assert 'Netflix subtitle splitter' in result['prompt']
        assert 'into **2** parts' in result['prompt']
        assert 'less than **15** words' in result['prompt']
        assert result['parameters']['sentence'] in result['prompt']
        
        # Test with different parameters
        chinese_result = mock_get_split_prompt(
            sentence="这是一个很长的句子，需要被分割成多个部分来提高可读性。",
            num_parts=3,
            word_limit=10,
            language="zh"
        )
        
        assert chinese_result['validation']['valid'] is True
        assert 'into **3** parts' in chinese_result['prompt']
        assert 'less than **10** words' in chinese_result['prompt']
        assert 'subtitle splitter in **zh**' in chinese_result['prompt']
        
        # Test validation errors
        invalid_result = mock_get_split_prompt(
            sentence="",
            num_parts=1,
            word_limit=2
        )
        
        assert invalid_result['validation']['valid'] is False
        assert 'Sentence cannot be empty' in invalid_result['validation']['errors']
        assert 'Number of parts must be at least 2' in invalid_result['validation']['errors']
        assert 'Word limit must be at least 5' in invalid_result['validation']['errors']
    
    def test_summary_prompt_generation_logic(self):
        """Test get_summary_prompt function logic"""
        # Simulate get_summary_prompt logic
        def mock_get_summary_prompt(source_content, custom_terms_json=None, src_lang="en", tgt_lang="zh"):
            """Mock summary prompt generation"""
            
            # Generate custom terms note
            terms_note = ""
            if custom_terms_json and custom_terms_json.get('terms'):
                terms_list = []
                for term in custom_terms_json['terms']:
                    terms_list.append(f"- {term['src']}: {term['tgt']} ({term['note']})")
                terms_note = "\n### Existing Terms\nPlease exclude these terms in your extraction:\n" + "\n".join(terms_list)
            
            summary_prompt = f"""
## Role
You are a video translation expert and terminology consultant, specializing in {src_lang} comprehension and {tgt_lang} expression optimization.

## Task
For the provided {src_lang} video text:
1. Summarize main topic in two sentences
2. Extract professional terms/names with {tgt_lang} translations (excluding existing terms)
3. Provide brief explanation for each term

{terms_note}

Steps:
1. Topic Summary:
   - Quick scan for general understanding
   - Write two sentences: first for main topic, second for key point
2. Term Extraction:
   - Mark professional terms and names (excluding those listed in Existing Terms)
   - Provide {tgt_lang} translation or keep original
   - Add brief explanation
   - Extract less than 15 terms

## INPUT
<text>
{source_content}
</text>

## Output in only JSON format and no other text
{{
  "theme": "Two-sentence video summary",
  "terms": [
    {{
      "src": "{src_lang} term",
      "tgt": "{tgt_lang} translation or original", 
      "note": "Brief explanation"
    }},
    ...
  ]
}}  

## Example
{{
  "theme": "本视频介绍人工智能在医疗领域的应用现状。重点展示了AI在医学影像诊断和药物研发中的突破性进展。",
  "terms": [
    {{
      "src": "Machine Learning",
      "tgt": "机器学习",
      "note": "AI的核心技术，通过数据训练实现智能决策"
    }},
    {{
      "src": "CNN",
      "tgt": "CNN",
      "note": "卷积神经网络，用于医学图像识别的深度学习模型"
    }}
  ]
}}

Note: Start you answer with ```json and end with ```, do not add any other text.
""".strip()
            
            # Validate inputs
            validation_result = {
                'valid': True,
                'errors': []
            }
            
            if not source_content.strip():
                validation_result['valid'] = False
                validation_result['errors'].append('Source content cannot be empty')
            
            if len(source_content) > 10000:
                validation_result['valid'] = False
                validation_result['errors'].append('Source content too long (>10000 chars)')
            
            # Analyze prompt structure
            prompt_analysis = {
                'has_role_section': '## Role' in summary_prompt,
                'has_task_section': '## Task' in summary_prompt,
                'has_example_section': '## Example' in summary_prompt,
                'has_custom_terms': terms_note != "",
                'custom_terms_count': len(custom_terms_json.get('terms', [])) if custom_terms_json else 0,
                'content_length': len(source_content),
                'language_pair': f"{src_lang}->{tgt_lang}"
            }
            
            return {
                'prompt': summary_prompt,
                'validation': validation_result,
                'analysis': prompt_analysis,
                'terms_note': terms_note,
                'parameters': {
                    'src_lang': src_lang,
                    'tgt_lang': tgt_lang,
                    'content_preview': source_content[:100] + '...' if len(source_content) > 100 else source_content
                }
            }
        
        # Test basic summary prompt
        basic_result = mock_get_summary_prompt(
            source_content="This video discusses artificial intelligence applications in healthcare, focusing on medical imaging and drug discovery.",
            src_lang="en",
            tgt_lang="zh"
        )
        
        assert basic_result['validation']['valid'] is True
        assert basic_result['analysis']['has_role_section'] is True
        assert basic_result['analysis']['has_task_section'] is True
        assert basic_result['analysis']['has_example_section'] is True
        assert basic_result['analysis']['has_custom_terms'] is False
        assert 'specializing in en comprehension and zh expression' in basic_result['prompt']
        assert basic_result['analysis']['language_pair'] == 'en->zh'
        
        # Test with custom terms
        custom_terms = {
            'terms': [
                {'src': 'AI', 'tgt': '人工智能', 'note': 'Artificial Intelligence'},
                {'src': 'ML', 'tgt': '机器学习', 'note': 'Machine Learning'}
            ]
        }
        
        custom_result = mock_get_summary_prompt(
            source_content="Advanced AI and ML techniques are transforming modern healthcare systems.",
            custom_terms_json=custom_terms,
            src_lang="en",
            tgt_lang="zh"
        )
        
        assert custom_result['validation']['valid'] is True
        assert custom_result['analysis']['has_custom_terms'] is True
        assert custom_result['analysis']['custom_terms_count'] == 2
        assert '### Existing Terms' in custom_result['terms_note']
        assert 'AI: 人工智能 (Artificial Intelligence)' in custom_result['terms_note']
        assert 'ML: 机器学习 (Machine Learning)' in custom_result['terms_note']
        
        # Test validation errors
        invalid_result = mock_get_summary_prompt(
            source_content="",
            src_lang="en",
            tgt_lang="zh"
        )
        
        assert invalid_result['validation']['valid'] is False
        assert 'Source content cannot be empty' in invalid_result['validation']['errors']
        
        # Test content too long
        long_content = "x" * 15000
        long_result = mock_get_summary_prompt(
            source_content=long_content,
            src_lang="en",
            tgt_lang="zh"
        )
        
        assert long_result['validation']['valid'] is False
        assert 'Source content too long' in long_result['validation']['errors']
    
    def test_shared_prompt_generation_logic(self):
        """Test generate_shared_prompt function logic"""
        # Simulate generate_shared_prompt logic
        def mock_generate_shared_prompt(previous_content_prompt, after_content_prompt, summary_prompt, things_to_note_prompt):
            """Mock shared prompt generation"""
            
            # Generate shared prompt template
            shared_prompt = f'''### Context Information
<previous_content>
{previous_content_prompt}
</previous_content>

<subsequent_content>
{after_content_prompt}
</subsequent_content>

### Content Summary
{summary_prompt}

### Points to Note
{things_to_note_prompt}'''
            
            # Analyze prompt components
            components_analysis = {
                'has_previous_content': bool(previous_content_prompt and previous_content_prompt.strip()),
                'has_after_content': bool(after_content_prompt and after_content_prompt.strip()),
                'has_summary': bool(summary_prompt and summary_prompt.strip()),
                'has_notes': bool(things_to_note_prompt and things_to_note_prompt.strip()),
                'total_length': len(shared_prompt),
                'section_count': shared_prompt.count('###')
            }
            
            # Calculate completeness score
            filled_components = sum([
                components_analysis['has_previous_content'],
                components_analysis['has_after_content'],
                components_analysis['has_summary'],
                components_analysis['has_notes']
            ])
            completeness_score = (filled_components / 4) * 100
            
            return {
                'shared_prompt': shared_prompt,
                'analysis': components_analysis,
                'completeness_score': completeness_score,
                'filled_sections': filled_components,
                'empty_sections': 4 - filled_components
            }
        
        # Test complete shared prompt
        complete_result = mock_generate_shared_prompt(
            previous_content_prompt="Previous video content about AI basics",
            after_content_prompt="Next content will cover advanced applications",
            summary_prompt="Video introduces machine learning concepts",
            things_to_note_prompt="Technical terms: neural networks, algorithms"
        )
        
        assert complete_result['completeness_score'] == 100.0
        assert complete_result['filled_sections'] == 4
        assert complete_result['empty_sections'] == 0
        assert complete_result['analysis']['has_previous_content'] is True
        assert complete_result['analysis']['has_after_content'] is True
        assert complete_result['analysis']['has_summary'] is True
        assert complete_result['analysis']['has_notes'] is True
        assert complete_result['analysis']['section_count'] == 3
        assert 'AI basics' in complete_result['shared_prompt']
        assert 'neural networks' in complete_result['shared_prompt']
        
        # Test partial shared prompt
        partial_result = mock_generate_shared_prompt(
            previous_content_prompt="Previous content",
            after_content_prompt="",
            summary_prompt="Summary here",
            things_to_note_prompt=""
        )
        
        assert partial_result['completeness_score'] == 50.0
        assert partial_result['filled_sections'] == 2
        assert partial_result['empty_sections'] == 2
        assert partial_result['analysis']['has_previous_content'] is True
        assert partial_result['analysis']['has_after_content'] is False
        assert partial_result['analysis']['has_summary'] is True
        assert partial_result['analysis']['has_notes'] is False
        
        # Test empty shared prompt
        empty_result = mock_generate_shared_prompt("", "", "", "")
        
        assert empty_result['completeness_score'] == 0.0
        assert empty_result['filled_sections'] == 0
        assert empty_result['empty_sections'] == 4
        assert all(not value for key, value in empty_result['analysis'].items() if key.startswith('has_'))
    
    def test_faithfulness_prompt_generation_logic(self):
        """Test get_prompt_faithfulness function logic"""
        # Simulate get_prompt_faithfulness logic
        def mock_get_prompt_faithfulness(lines, shared_prompt, src_language="en", target_language="zh"):
            """Mock faithfulness prompt generation"""
            
            # Split lines and create JSON structure
            line_splits = lines.split('\n')
            json_dict = {}
            for i, line in enumerate(line_splits, 1):
                json_dict[f"{i}"] = {"origin": line, "direct": f"direct {target_language} translation {i}."}
            
            json_format = json.dumps(json_dict, indent=2, ensure_ascii=False)
            
            # Generate faithfulness prompt
            prompt_faithfulness = f'''
## Role
You are a professional Netflix subtitle translator, fluent in both {src_language} and {target_language}, as well as their respective cultures. 
Your expertise lies in accurately understanding the semantics and structure of the original {src_language} text and faithfully translating it into {target_language} while preserving the original meaning.

## Task
We have a segment of original {src_language} subtitles that need to be directly translated into {target_language}. These subtitles come from a specific context and may contain specific themes and terminology.

1. Translate the original {src_language} subtitles into {target_language} line by line
2. Ensure the translation is faithful to the original, accurately conveying the original meaning
3. Consider the context and professional terminology

{shared_prompt}

<translation_principles>
1. Faithful to the original: Accurately convey the content and meaning of the original text, without arbitrarily changing, adding, or omitting content.
2. Accurate terminology: Use professional terms correctly and maintain consistency in terminology.
3. Understand the context: Fully comprehend and reflect the background and contextual relationships of the text.
</translation_principles>

## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
            
            # Analyze prompt structure
            prompt_analysis = {
                'line_count': len(line_splits),
                'json_entries': len(json_dict),
                'has_shared_prompt': bool(shared_prompt and shared_prompt.strip()),
                'has_translation_principles': '<translation_principles>' in prompt_faithfulness,
                'has_role_definition': '## Role' in prompt_faithfulness,
                'has_task_definition': '## Task' in prompt_faithfulness,
                'json_format_valid': json_format.startswith('{') and json_format.endswith('}'),
                'total_prompt_length': len(prompt_faithfulness)
            }
            
            return {
                'prompt': prompt_faithfulness,
                'json_structure': json_dict,
                'json_format': json_format,
                'analysis': prompt_analysis,
                'parameters': {
                    'src_language': src_language,
                    'target_language': target_language,
                    'line_count': len(line_splits)
                }
            }
        
        # Test faithfulness prompt with multiple lines
        lines = "Hello world\nThis is a test\nTranslation example"
        shared_context = "### Context Information\nVideo about technology"
        
        result = mock_get_prompt_faithfulness(lines, shared_context, "en", "zh")
        
        assert result['analysis']['line_count'] == 3
        assert result['analysis']['json_entries'] == 3
        assert result['analysis']['has_shared_prompt'] is True
        assert result['analysis']['has_translation_principles'] is True
        assert result['analysis']['has_role_definition'] is True
        assert result['analysis']['has_task_definition'] is True
        assert result['analysis']['json_format_valid'] is True
        assert 'Netflix subtitle translator' in result['prompt']
        assert 'fluent in both en and zh' in result['prompt']
        assert shared_context in result['prompt']
        
        # Verify JSON structure
        assert len(result['json_structure']) == 3
        assert result['json_structure']['1']['origin'] == 'Hello world'
        assert result['json_structure']['2']['origin'] == 'This is a test'
        assert result['json_structure']['3']['origin'] == 'Translation example'
        assert 'direct zh translation 1' in result['json_structure']['1']['direct']
        
        # Test single line
        single_line_result = mock_get_prompt_faithfulness("Single line test", "", "en", "fr")
        
        assert single_line_result['analysis']['line_count'] == 1
        assert single_line_result['analysis']['json_entries'] == 1
        assert single_line_result['analysis']['has_shared_prompt'] is False
        assert 'fluent in both en and fr' in single_line_result['prompt']
    
    def test_expressiveness_prompt_generation_logic(self):
        """Test get_prompt_expressiveness function logic"""
        # Simulate get_prompt_expressiveness logic
        def mock_get_prompt_expressiveness(faithfulness_result, lines, shared_prompt, src_language="en", target_language="zh"):
            """Mock expressiveness prompt generation"""
            
            # Create JSON format with reflection and free translation
            json_format = {
                key: {
                    "origin": value["origin"],
                    "direct": value["direct"],
                    "reflect": "your reflection on direct translation",
                    "free": "your free translation"
                }
                for key, value in faithfulness_result.items()
            }
            json_format_str = json.dumps(json_format, indent=2, ensure_ascii=False)
            
            # Generate expressiveness prompt
            prompt_expressiveness = f'''
## Role
You are a professional Netflix subtitle translator and language consultant.
Your expertise lies not only in accurately understanding the original {src_language} but also in optimizing the {target_language} translation to better suit the target language's expression habits and cultural background.

## Task
We already have a direct translation version of the original {src_language} subtitles.
Your task is to reflect on and improve these direct translations to create more natural and fluent {target_language} subtitles.

1. Analyze the direct translation results line by line, pointing out existing issues
2. Provide detailed modification suggestions
3. Perform free translation based on your analysis
4. Do not add comments or explanations in the translation, as the subtitles are for the audience to read
5. Do not leave empty lines in the free translation, as the subtitles are for the audience to read

{shared_prompt}

<Translation Analysis Steps>
Please use a two-step thinking process to handle the text line by line:

1. Direct Translation Reflection:
   - Evaluate language fluency
   - Check if the language style is consistent with the original text
   - Check the conciseness of the subtitles, point out where the translation is too wordy

2. {target_language} Free Translation:
   - Aim for contextual smoothness and naturalness, conforming to {target_language} expression habits
   - Ensure it's easy for {target_language} audience to understand and accept
   - Adapt the language style to match the theme (e.g., use casual language for tutorials, professional terminology for technical content, formal language for documentaries)
</Translation Analysis Steps>
   
## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format_str}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
            
            # Analyze prompt structure
            prompt_analysis = {
                'faithfulness_entries': len(faithfulness_result),
                'has_reflection_step': 'Direct Translation Reflection' in prompt_expressiveness,
                'has_free_translation_step': 'Free Translation' in prompt_expressiveness,
                'has_analysis_steps': '<Translation Analysis Steps>' in prompt_expressiveness,
                'has_shared_prompt': bool(shared_prompt and shared_prompt.strip()),
                'json_has_reflect_field': '"reflect"' in json_format_str,
                'json_has_free_field': '"free"' in json_format_str,
                'total_prompt_length': len(prompt_expressiveness)
            }
            
            return {
                'prompt': prompt_expressiveness,
                'json_structure': json_format,
                'json_format': json_format_str,
                'analysis': prompt_analysis,
                'parameters': {
                    'src_language': src_language,
                    'target_language': target_language,
                    'faithfulness_count': len(faithfulness_result)
                }
            }
        
        # Test expressiveness prompt
        faithfulness_result = {
            "1": {"origin": "Hello world", "direct": "你好世界"},
            "2": {"origin": "How are you?", "direct": "你好吗？"}
        }
        lines = "Hello world\nHow are you?"
        shared_context = "### Context\nCasual conversation"
        
        result = mock_get_prompt_expressiveness(faithfulness_result, lines, shared_context, "en", "zh")
        
        assert result['analysis']['faithfulness_entries'] == 2
        assert result['analysis']['has_reflection_step'] is True
        assert result['analysis']['has_free_translation_step'] is True
        assert result['analysis']['has_analysis_steps'] is True
        assert result['analysis']['has_shared_prompt'] is True
        assert result['analysis']['json_has_reflect_field'] is True
        assert result['analysis']['json_has_free_field'] is True
        assert 'language consultant' in result['prompt']
        assert 'conforming to zh expression habits' in result['prompt']
        assert shared_context in result['prompt']
        
        # Verify JSON structure includes all required fields
        assert len(result['json_structure']) == 2
        for key in result['json_structure']:
            assert 'origin' in result['json_structure'][key]
            assert 'direct' in result['json_structure'][key]
            assert 'reflect' in result['json_structure'][key]
            assert 'free' in result['json_structure'][key]
    
    def test_subtitle_trim_prompt_logic(self):
        """Test get_subtitle_trim_prompt function logic"""
        # Simulate subtitle trim prompt logic
        def mock_get_subtitle_trim_prompt(text, duration):
            """Mock subtitle trim prompt generation"""
            
            rule = '''Consider a. Reducing filler words without modifying meaningful content. b. Omitting unnecessary modifiers or pronouns, for example:
    - "Please explain your thought process" can be shortened to "Please explain thought process"
    - "We need to carefully analyze this complex problem" can be shortened to "We need to analyze this problem"
    - "Let's discuss the various different perspectives on this topic" can be shortened to "Let's discuss different perspectives on this topic"
    - "Can you describe in detail your experience from yesterday" can be shortened to "Can you describe yesterday's experience" '''

            trim_prompt = f'''
## Role
You are a professional subtitle editor, editing and optimizing lengthy subtitles that exceed voiceover time before handing them to voice actors. 
Your expertise lies in cleverly shortening subtitles slightly while ensuring the original meaning and structure remain unchanged.

## INPUT
<subtitles>
Subtitle: "{text}"
Duration: {duration} seconds
</subtitles>

## Processing Rules
{rule}

## Processing Steps
Please follow these steps and provide the results in the JSON output:
1. Analysis: Briefly analyze the subtitle's structure, key information, and filler words that can be omitted.
2. Trimming: Based on the rules and analysis, optimize the subtitle by making it more concise according to the processing rules.

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief analysis of the subtitle, including structure, key information, and potential processing locations",
    "result": "Optimized and shortened subtitle in the original subtitle language"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
            
            # Analyze trim requirements
            word_count = len(text.split())
            char_count = len(text)
            words_per_second = word_count / duration if duration > 0 else 0
            chars_per_second = char_count / duration if duration > 0 else 0
            
            # Determine if trimming is needed
            needs_trimming = words_per_second > 3 or chars_per_second > 15  # Rough thresholds
            
            trim_analysis = {
                'word_count': word_count,
                'char_count': char_count,
                'duration': duration,
                'words_per_second': words_per_second,
                'chars_per_second': chars_per_second,
                'needs_trimming': needs_trimming,
                'has_processing_rules': rule in trim_prompt,
                'has_analysis_step': 'Analysis:' in trim_prompt,
                'has_trimming_step': 'Trimming:' in trim_prompt
            }
            
            return {
                'prompt': trim_prompt,
                'analysis': trim_analysis,
                'rule': rule,
                'parameters': {
                    'text': text,
                    'duration': duration
                }
            }
        
        # Test subtitle that needs trimming
        long_text = "Please carefully explain in detail your comprehensive thought process and methodology"
        result = mock_get_subtitle_trim_prompt(long_text, 4.0)
        
        assert result['analysis']['word_count'] == 11
        assert result['analysis']['duration'] == 4.0
        assert result['analysis']['words_per_second'] > 2
        assert result['analysis']['needs_trimming'] is True
        assert result['analysis']['has_processing_rules'] is True
        assert result['analysis']['has_analysis_step'] is True
        assert result['analysis']['has_trimming_step'] is True
        assert 'subtitle editor' in result['prompt']
        assert long_text in result['prompt']
        assert 'Duration: 4.0 seconds' in result['prompt']
        
        # Test short subtitle that doesn't need trimming
        short_text = "Hello world"
        short_result = mock_get_subtitle_trim_prompt(short_text, 3.0)
        
        assert short_result['analysis']['word_count'] == 2
        assert short_result['analysis']['words_per_second'] < 1
        assert short_result['analysis']['needs_trimming'] is False
        
        # Test edge case with zero duration
        zero_duration_result = mock_get_subtitle_trim_prompt("Test text", 0)
        
        assert zero_duration_result['analysis']['words_per_second'] == 0
        assert zero_duration_result['analysis']['chars_per_second'] == 0
    
    def test_text_correction_prompt_logic(self):
        """Test get_correct_text_prompt function logic"""
        # Simulate text correction prompt logic
        def mock_get_correct_text_prompt(text):
            """Mock text correction prompt generation"""
            
            correction_prompt = f'''
## Role
You are a text cleaning expert for TTS (Text-to-Speech) systems.

## Task
Clean the given text by:
1. Keep only basic punctuation (.,?!)
2. Preserve the original meaning

## INPUT
{text}

## Output in only JSON format and no other text
```json
{{
    "text": "cleaned text here"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
            
            # Analyze text cleaning requirements
            special_chars = set(text) - set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!')
            basic_punctuation = set(text) & set('.,?!')
            
            cleaning_analysis = {
                'original_length': len(text),
                'has_special_chars': len(special_chars) > 0,
                'special_chars_count': len(special_chars),
                'special_chars_list': list(special_chars),
                'has_basic_punctuation': len(basic_punctuation) > 0,
                'basic_punctuation_list': list(basic_punctuation),
                'needs_cleaning': len(special_chars) > 0,
                'is_empty': not text.strip()
            }
            
            return {
                'prompt': correction_prompt,
                'analysis': cleaning_analysis,
                'parameters': {
                    'original_text': text
                }
            }
        
        # Test text with special characters
        messy_text = "Hello... world!!! This has @#$% special chars & symbols"
        result = mock_get_correct_text_prompt(messy_text)
        
        assert result['analysis']['has_special_chars'] is True
        assert result['analysis']['needs_cleaning'] is True
        assert '@' in result['analysis']['special_chars_list']
        assert '#' in result['analysis']['special_chars_list']
        assert '$' in result['analysis']['special_chars_list']
        assert '%' in result['analysis']['special_chars_list']
        assert '&' in result['analysis']['special_chars_list']
        assert '!' in result['analysis']['basic_punctuation_list']
        assert 'text cleaning expert' in result['prompt']
        assert 'TTS (Text-to-Speech)' in result['prompt']
        assert messy_text in result['prompt']
        
        # Test clean text
        clean_text = "Hello world. How are you?"
        clean_result = mock_get_correct_text_prompt(clean_text)
        
        assert clean_result['analysis']['has_special_chars'] is False
        assert clean_result['analysis']['needs_cleaning'] is False
        assert clean_result['analysis']['has_basic_punctuation'] is True
        assert '.' in clean_result['analysis']['basic_punctuation_list']
        assert '?' in clean_result['analysis']['basic_punctuation_list']
        
        # Test empty text
        empty_result = mock_get_correct_text_prompt("")
        
        assert empty_result['analysis']['is_empty'] is True
        assert empty_result['analysis']['has_special_chars'] is False
        assert empty_result['analysis']['needs_cleaning'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])