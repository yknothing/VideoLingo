"""
Comprehensive test suite for remaining utility modules.
Targets 85%+ branch coverage for path handling, security, and NLP utilities.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call, mock_open
import sys
import os
import tempfile
import shutil

# Add core directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestPathAdapterModule(unittest.TestCase):
    """Test core/utils/path_adapter.py - Cross-platform path handling"""
    
    def setUp(self):
        self.patcher_os = patch('core.utils.path_adapter.os')
        self.patcher_platform = patch('core.utils.path_adapter.platform')
        self.patcher_pathlib = patch('core.utils.path_adapter.Path')
        
        self.mock_os = self.patcher_os.start()
        self.mock_platform = self.patcher_platform.start()
        self.mock_pathlib = self.patcher_pathlib.start()
        
        # Setup platform defaults
        self.mock_platform.system.return_value = "Linux"
        self.mock_os.path.sep = "/"
        self.mock_os.path.exists.return_value = True
        
    def tearDown(self):
        self.patcher_os.stop()
        self.patcher_platform.stop()
        self.patcher_pathlib.stop()
        
    def test_normalize_path_linux(self):
        """Test path normalization on Linux"""
        from core.utils.path_adapter import normalize_path
        
        self.mock_platform.system.return_value = "Linux"
        self.mock_os.path.normpath.return_value = "/home/user/file.txt"
        
        result = normalize_path("/home/user//file.txt")
        
        self.mock_os.path.normpath.assert_called_with("/home/user//file.txt")
        self.assertEqual(result, "/home/user/file.txt")
        
    def test_normalize_path_windows(self):
        """Test path normalization on Windows"""
        from core.utils.path_adapter import normalize_path
        
        self.mock_platform.system.return_value = "Windows"
        self.mock_os.path.normpath.return_value = "C:\\Users\\user\\file.txt"
        
        result = normalize_path("C:\\Users\\user\\\\file.txt")
        
        self.mock_os.path.normpath.assert_called_with("C:\\Users\\user\\\\file.txt")
        self.assertEqual(result, "C:\\Users\\user\\file.txt")
        
    def test_normalize_path_macos(self):
        """Test path normalization on macOS"""
        from core.utils.path_adapter import normalize_path
        
        self.mock_platform.system.return_value = "Darwin"
        self.mock_os.path.normpath.return_value = "/Users/user/file.txt"
        
        result = normalize_path("/Users/user//file.txt")
        
        self.mock_os.path.normpath.assert_called_with("/Users/user//file.txt")
        self.assertEqual(result, "/Users/user/file.txt")
        
    def test_safe_join_paths_success(self):
        """Test safe path joining"""
        from core.utils.path_adapter import safe_join_paths
        
        self.mock_os.path.join.return_value = "/base/sub/file.txt"
        self.mock_os.path.abspath.return_value = "/base/sub/file.txt"
        self.mock_os.path.commonpath.return_value = "/base"
        
        result = safe_join_paths("/base", "sub", "file.txt")
        
        self.assertEqual(result, "/base/sub/file.txt")
        
    def test_safe_join_paths_directory_traversal(self):
        """Test safe path joining with directory traversal attempt"""
        from core.utils.path_adapter import safe_join_paths
        
        # Simulate directory traversal attempt
        self.mock_os.path.join.return_value = "/base/../etc/passwd"
        self.mock_os.path.abspath.return_value = "/etc/passwd"
        self.mock_os.path.commonpath.side_effect = ValueError("Different paths")
        
        with self.assertRaises(ValueError):
            safe_join_paths("/base", "..", "..", "etc", "passwd")
            
    def test_create_directory_structure_success(self):
        """Test successful directory structure creation"""
        from core.utils.path_adapter import create_directory_structure
        
        self.mock_os.path.exists.return_value = False
        self.mock_os.makedirs = Mock()
        
        create_directory_structure("/new/directory/structure")
        
        self.mock_os.makedirs.assert_called_with("/new/directory/structure", exist_ok=True)
        
    def test_create_directory_structure_already_exists(self):
        """Test directory structure creation when already exists"""
        from core.utils.path_adapter import create_directory_structure
        
        self.mock_os.path.exists.return_value = True
        self.mock_os.makedirs = Mock()
        
        create_directory_structure("/existing/directory")
        
        # Should still call makedirs with exist_ok=True
        self.mock_os.makedirs.assert_called_with("/existing/directory", exist_ok=True)
        
    def test_create_directory_structure_permission_error(self):
        """Test directory structure creation with permission error"""
        from core.utils.path_adapter import create_directory_structure
        
        self.mock_os.makedirs.side_effect = PermissionError("Permission denied")
        
        with self.assertRaises(PermissionError):
            create_directory_structure("/protected/directory")
            
    def test_get_relative_path_success(self):
        """Test successful relative path calculation"""
        from core.utils.path_adapter import get_relative_path
        
        self.mock_os.path.relpath.return_value = "sub/file.txt"
        
        result = get_relative_path("/base/sub/file.txt", "/base")
        
        self.mock_os.path.relpath.assert_called_with("/base/sub/file.txt", "/base")
        self.assertEqual(result, "sub/file.txt")
        
    def test_get_relative_path_same_path(self):
        """Test relative path calculation for same path"""
        from core.utils.path_adapter import get_relative_path
        
        self.mock_os.path.relpath.return_value = "."
        
        result = get_relative_path("/base", "/base")
        
        self.assertEqual(result, ".")
        
    def test_ensure_path_encoding_utf8(self):
        """Test path encoding ensuring UTF-8"""
        from core.utils.path_adapter import ensure_path_encoding
        
        # Test with string path
        result = ensure_path_encoding("/path/with/unicode/文件.txt")
        
        self.assertIsInstance(result, str)
        self.assertIn("文件.txt", result)
        
    def test_ensure_path_encoding_bytes(self):
        """Test path encoding with bytes input"""
        from core.utils.path_adapter import ensure_path_encoding
        
        # Test with bytes path
        result = ensure_path_encoding(b"/path/file.txt")
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, "/path/file.txt")


class TestSecurityUtilsModule(unittest.TestCase):
    """Test core/utils/security_utils.py - Security validation functions"""
    
    def setUp(self):
        self.patcher_os = patch('core.utils.security_utils.os')
        self.patcher_re = patch('core.utils.security_utils.re')
        
        self.mock_os = self.patcher_os.start()
        self.mock_re = self.patcher_re.start()
        
    def tearDown(self):
        self.patcher_os.stop()
        self.patcher_re.stop()
        
    def test_validate_file_path_safe(self):
        """Test file path validation with safe path"""
        from core.utils.security_utils import validate_file_path
        
        self.mock_os.path.abspath.return_value = "/safe/path/file.txt"
        self.mock_os.path.exists.return_value = True
        
        result = validate_file_path("/safe/path/file.txt", allowed_dirs=["/safe"])
        
        self.assertTrue(result)
        
    def test_validate_file_path_directory_traversal(self):
        """Test file path validation with directory traversal"""
        from core.utils.security_utils import validate_file_path
        
        self.mock_os.path.abspath.return_value = "/etc/passwd"
        
        result = validate_file_path("../../../etc/passwd", allowed_dirs=["/safe"])
        
        self.assertFalse(result)
        
    def test_validate_file_path_nonexistent(self):
        """Test file path validation with non-existent file"""
        from core.utils.security_utils import validate_file_path
        
        self.mock_os.path.abspath.return_value = "/safe/nonexistent.txt"
        self.mock_os.path.exists.return_value = False
        
        result = validate_file_path("/safe/nonexistent.txt", allowed_dirs=["/safe"], must_exist=True)
        
        self.assertFalse(result)
        
    def test_validate_file_path_optional_existence(self):
        """Test file path validation with optional existence"""
        from core.utils.security_utils import validate_file_path
        
        self.mock_os.path.abspath.return_value = "/safe/newfile.txt"
        self.mock_os.path.exists.return_value = False
        
        result = validate_file_path("/safe/newfile.txt", allowed_dirs=["/safe"], must_exist=False)
        
        self.assertTrue(result)
        
    def test_sanitize_filename_safe(self):
        """Test filename sanitization with safe filename"""
        from core.utils.security_utils import sanitize_filename
        
        result = sanitize_filename("safe_filename.txt")
        
        self.assertEqual(result, "safe_filename.txt")
        
    def test_sanitize_filename_with_unsafe_chars(self):
        """Test filename sanitization with unsafe characters"""
        from core.utils.security_utils import sanitize_filename
        
        # Mock regex substitution
        self.mock_re.sub.return_value = "unsafe_filename.txt"
        
        result = sanitize_filename("unsafe/file<name>.txt")
        
        self.mock_re.sub.assert_called()
        self.assertEqual(result, "unsafe_filename.txt")
        
    def test_sanitize_filename_empty_result(self):
        """Test filename sanitization resulting in empty string"""
        from core.utils.security_utils import sanitize_filename
        
        self.mock_re.sub.return_value = ""
        
        result = sanitize_filename("///<<<>>>")
        
        self.assertEqual(result, "unnamed_file")
        
    def test_validate_url_safe(self):
        """Test URL validation with safe URL"""
        from core.utils.security_utils import validate_url
        
        result = validate_url("https://api.openai.com/v1/chat/completions")
        
        self.assertTrue(result)
        
    def test_validate_url_unsafe_scheme(self):
        """Test URL validation with unsafe scheme"""
        from core.utils.security_utils import validate_url
        
        result = validate_url("file:///etc/passwd")
        
        self.assertFalse(result)
        
    def test_validate_url_malformed(self):
        """Test URL validation with malformed URL"""
        from core.utils.security_utils import validate_url
        
        result = validate_url("not-a-url")
        
        self.assertFalse(result)
        
    def test_validate_api_key_valid(self):
        """Test API key validation with valid key"""
        from core.utils.security_utils import validate_api_key
        
        result = validate_api_key("sk-1234567890abcdef")
        
        self.assertTrue(result)
        
    def test_validate_api_key_too_short(self):
        """Test API key validation with too short key"""
        from core.utils.security_utils import validate_api_key
        
        result = validate_api_key("short")
        
        self.assertFalse(result)
        
    def test_validate_api_key_empty(self):
        """Test API key validation with empty key"""
        from core.utils.security_utils import validate_api_key
        
        result = validate_api_key("")
        
        self.assertFalse(result)
        
    def test_validate_api_key_none(self):
        """Test API key validation with None key"""
        from core.utils.security_utils import validate_api_key
        
        result = validate_api_key(None)
        
        self.assertFalse(result)
        
    def test_check_file_permissions_readable(self):
        """Test file permission check for readable file"""
        from core.utils.security_utils import check_file_permissions
        
        self.mock_os.access.return_value = True
        
        result = check_file_permissions("/path/file.txt", "read")
        
        self.mock_os.access.assert_called_with("/path/file.txt", self.mock_os.R_OK)
        self.assertTrue(result)
        
    def test_check_file_permissions_writable(self):
        """Test file permission check for writable file"""
        from core.utils.security_utils import check_file_permissions
        
        self.mock_os.access.return_value = True
        
        result = check_file_permissions("/path/file.txt", "write")
        
        self.mock_os.access.assert_called_with("/path/file.txt", self.mock_os.W_OK)
        self.assertTrue(result)
        
    def test_check_file_permissions_executable(self):
        """Test file permission check for executable file"""
        from core.utils.security_utils import check_file_permissions
        
        self.mock_os.access.return_value = True
        
        result = check_file_permissions("/path/script.sh", "execute")
        
        self.mock_os.access.assert_called_with("/path/script.sh", self.mock_os.X_OK)
        self.assertTrue(result)
        
    def test_check_file_permissions_invalid_mode(self):
        """Test file permission check with invalid mode"""
        from core.utils.security_utils import check_file_permissions
        
        result = check_file_permissions("/path/file.txt", "invalid")
        
        self.assertFalse(result)
        
    def test_escape_shell_command_safe(self):
        """Test shell command escaping with safe command"""
        from core.utils.security_utils import escape_shell_command
        
        result = escape_shell_command("ls -la")
        
        self.assertIsInstance(result, str)
        self.assertIn("ls", result)
        
    def test_escape_shell_command_with_injection(self):
        """Test shell command escaping with injection attempt"""
        from core.utils.security_utils import escape_shell_command
        
        dangerous_cmd = "ls; rm -rf /"
        result = escape_shell_command(dangerous_cmd)
        
        # Should escape or remove dangerous parts
        self.assertNotIn("rm -rf", result)


class TestSpacyUtilsModule(unittest.TestCase):
    """Test core/spacy_utils/* - NLP utility functions"""
    
    def setUp(self):
        self.patcher_spacy = patch('spacy')
        self.patcher_load_key = patch('core.spacy_utils.language_detector.load_key')
        
        self.mock_spacy = self.patcher_spacy.start()
        self.mock_load_key = self.patcher_load_key.start()
        
        # Mock spaCy model
        self.mock_nlp = Mock()
        self.mock_spacy.load.return_value = self.mock_nlp
        
        # Mock document and tokens
        self.mock_doc = Mock()
        self.mock_token1 = Mock()
        self.mock_token1.text = "Hello"
        self.mock_token1.pos_ = "NOUN"
        self.mock_token1.is_alpha = True
        
        self.mock_token2 = Mock()
        self.mock_token2.text = "world"
        self.mock_token2.pos_ = "NOUN"
        self.mock_token2.is_alpha = True
        
        self.mock_doc.__iter__ = Mock(return_value=iter([self.mock_token1, self.mock_token2]))
        self.mock_nlp.return_value = self.mock_doc
        
    def tearDown(self):
        self.patcher_spacy.stop()
        self.patcher_load_key.stop()
        
    def test_detect_language_english(self):
        """Test language detection for English text"""
        from core.spacy_utils.language_detector import detect_language
        
        # Mock language detection result
        self.mock_load_key.return_value = "en"
        
        result = detect_language("Hello world, this is a test.")
        
        self.assertEqual(result, "en")
        
    def test_detect_language_auto_detection(self):
        """Test automatic language detection"""
        from core.spacy_utils.language_detector import detect_language
        
        # Mock auto detection
        self.mock_load_key.return_value = "auto"
        
        with patch('core.spacy_utils.language_detector.langdetect') as mock_langdetect:
            mock_langdetect.detect.return_value = "es"
            
            result = detect_language("Hola mundo, esto es una prueba.")
            
            self.assertEqual(result, "es")
            
    def test_detect_language_fallback(self):
        """Test language detection with fallback"""
        from core.spacy_utils.language_detector import detect_language
        
        self.mock_load_key.return_value = "auto"
        
        with patch('core.spacy_utils.language_detector.langdetect') as mock_langdetect:
            mock_langdetect.detect.side_effect = Exception("Detection failed")
            
            result = detect_language("Some text")
            
            self.assertEqual(result, "en")  # Should fallback to English
            
    def test_tokenize_text_success(self):
        """Test successful text tokenization"""
        from core.spacy_utils.tokenizer import tokenize_text
        
        with patch('core.spacy_utils.tokenizer.load_spacy_model') as mock_load:
            mock_load.return_value = self.mock_nlp
            
            tokens = tokenize_text("Hello world", "en")
            
            expected = ["Hello", "world"]
            self.assertEqual(tokens, expected)
            
    def test_tokenize_text_empty_input(self):
        """Test text tokenization with empty input"""
        from core.spacy_utils.tokenizer import tokenize_text
        
        tokens = tokenize_text("", "en")
        
        self.assertEqual(tokens, [])
        
    def test_tokenize_text_model_loading_error(self):
        """Test text tokenization with model loading error"""
        from core.spacy_utils.tokenizer import tokenize_text
        
        with patch('core.spacy_utils.tokenizer.load_spacy_model') as mock_load:
            mock_load.side_effect = Exception("Model not found")
            
            tokens = tokenize_text("Hello world", "en")
            
            # Should handle error gracefully
            self.assertEqual(tokens, [])
            
    def test_extract_entities_success(self):
        """Test successful named entity extraction"""
        from core.spacy_utils.entity_extractor import extract_entities
        
        # Mock entities
        mock_ent1 = Mock()
        mock_ent1.text = "OpenAI"
        mock_ent1.label_ = "ORG"
        
        mock_ent2 = Mock()
        mock_ent2.text = "New York"
        mock_ent2.label_ = "GPE"
        
        self.mock_doc.ents = [mock_ent1, mock_ent2]
        
        with patch('core.spacy_utils.entity_extractor.load_spacy_model') as mock_load:
            mock_load.return_value = self.mock_nlp
            
            entities = extract_entities("OpenAI is located in New York", "en")
            
            expected = [
                {"text": "OpenAI", "label": "ORG"},
                {"text": "New York", "label": "GPE"}
            ]
            self.assertEqual(entities, expected)
            
    def test_extract_entities_no_entities(self):
        """Test entity extraction with no entities found"""
        from core.spacy_utils.entity_extractor import extract_entities
        
        self.mock_doc.ents = []
        
        with patch('core.spacy_utils.entity_extractor.load_spacy_model') as mock_load:
            mock_load.return_value = self.mock_nlp
            
            entities = extract_entities("This text has no entities", "en")
            
            self.assertEqual(entities, [])
            
    def test_extract_entities_model_error(self):
        """Test entity extraction with model error"""
        from core.spacy_utils.entity_extractor import extract_entities
        
        with patch('core.spacy_utils.entity_extractor.load_spacy_model') as mock_load:
            mock_load.side_effect = Exception("Model error")
            
            entities = extract_entities("Some text", "en")
            
            self.assertEqual(entities, [])
            
    def test_load_spacy_model_success(self):
        """Test successful spaCy model loading"""
        from core.spacy_utils.model_loader import load_spacy_model
        
        model = load_spacy_model("en")
        
        self.mock_spacy.load.assert_called_with("en_core_web_sm")
        self.assertEqual(model, self.mock_nlp)
        
    def test_load_spacy_model_fallback(self):
        """Test spaCy model loading with fallback"""
        from core.spacy_utils.model_loader import load_spacy_model
        
        # First call fails, second succeeds
        self.mock_spacy.load.side_effect = [OSError("Model not found"), self.mock_nlp]
        
        model = load_spacy_model("zh")
        
        # Should try specific model first, then fallback
        calls = self.mock_spacy.load.call_args_list
        self.assertEqual(len(calls), 2)
        
    def test_load_spacy_model_all_fail(self):
        """Test spaCy model loading when all attempts fail"""
        from core.spacy_utils.model_loader import load_spacy_model
        
        self.mock_spacy.load.side_effect = OSError("No models available")
        
        with self.assertRaises(OSError):
            load_spacy_model("invalid")


if __name__ == '__main__':
    unittest.main()
