# Comprehensive Unit Tests for NLP Split Module
import pytest
from unittest.mock import patch, MagicMock
from core._3_1_split_nlp import split_by_spacy
from core.spacy_utils.load_nlp_model import get_spacy_model, init_nlp

class TestSpacyModelLoading:
    def test_get_spacy_model_supported_language(self):
        spacy_model_map = {'en': 'en_core_web_md', 'zh': 'zh_core_web_md'}
        
        with patch('core.spacy_utils.load_nlp_model.SPACY_MODEL_MAP', spacy_model_map):
            model = get_spacy_model('en')
            assert model == 'en_core_web_md'

    def test_get_spacy_model_unsupported_language(self):
        spacy_model_map = {'en': 'en_core_web_md'}
        
        with patch('core.spacy_utils.load_nlp_model.SPACY_MODEL_MAP', spacy_model_map):
            with patch('core.spacy_utils.load_nlp_model.rprint') as mock_print:
                model = get_spacy_model('unsupported')
                assert model == 'en_core_web_md'
                mock_print.assert_called_once()

    def test_init_nlp_english_language(self):
        mock_nlp = MagicMock()
        
        with patch('core.spacy_utils.load_nlp_model.load_key', return_value='en'):
            with patch('core.spacy_utils.load_nlp_model.spacy.load', return_value=mock_nlp):
                with patch('core.spacy_utils.load_nlp_model.rprint'):
                    result = init_nlp()
                    assert result == mock_nlp

class TestMainSplitFunction:
    def test_split_by_spacy_complete_pipeline(self):
        mock_nlp = MagicMock()
        
        with patch('core._3_1_split_nlp.init_nlp', return_value=mock_nlp):
            with patch('core._3_1_split_nlp.split_by_mark'):
                with patch('core._3_1_split_nlp.split_by_comma_main'):
                    with patch('core._3_1_split_nlp.split_sentences_main'):
                        with patch('core._3_1_split_nlp.split_long_by_root_main'):
                            with patch('core._3_1_split_nlp.check_file_exists', lambda x: lambda f: f):
                                split_by_spacy()
