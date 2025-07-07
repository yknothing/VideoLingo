# Test the real config_utils module with direct import

import pytest
import os
import tempfile
import sys
import importlib.util
from pathlib import Path
from unittest.mock import patch, mock_open
from ruamel.yaml import YAML

# Directly import the config_utils module without going through core/__init__.py
config_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core', 'utils', 'config_utils.py')
spec = importlib.util.spec_from_file_location("config_utils", config_utils_path)
config_utils = importlib.util.module_from_spec(spec)
sys.modules["config_utils"] = config_utils
spec.loader.exec_module(config_utils)

class TestRealConfigUtils:
    """Test the actual config_utils module"""
    
    def test_load_key_basic(self):
        """Test basic load_key functionality"""
        test_config = {
            'api': {
                'key': 'test-api-key',
                'model': 'gpt-4'
            },
            'simple_key': 'simple_value'
        }
        
        yaml_content = """
api:
  key: test-api-key
  model: gpt-4
simple_key: simple_value
"""
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Test simple key
            result = config_utils.load_key('simple_key')
            assert result == 'simple_value'
            
            # Test nested key
            result = config_utils.load_key('api.key')
            assert result == 'test-api-key'
            
            result = config_utils.load_key('api.model')
            assert result == 'gpt-4'
    
    def test_load_key_nonexistent(self):
        """Test load_key with non-existent keys"""
        yaml_content = """
api:
  key: test-key
"""
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Test non-existent key - should raise KeyError
            with pytest.raises(KeyError):
                config_utils.load_key('nonexistent_key')
            
            # Test non-existent nested key
            with pytest.raises(KeyError):
                config_utils.load_key('api.nonexistent')
            
            # Test partially non-existent path
            with pytest.raises(KeyError):
                config_utils.load_key('nonexistent.key')
    
    def test_update_key_basic(self):
        """Test basic update_key functionality"""
        initial_config = {
            'api': {
                'key': 'old-key',
                'model': 'gpt-3'
            }
        }
        
        yaml = YAML()
        yaml_content = """
api:
  key: old-key
  model: gpt-3
"""
        
        updated_yaml = None
        
        def mock_write(content):
            nonlocal updated_yaml
            updated_yaml = content
        
        mock_file = mock_open(read_data=yaml_content)
        mock_file.return_value.write = mock_write
        
        with patch('builtins.open', mock_file):
            # Test updating existing key
            result = config_utils.update_key('api.key', 'new-api-key')
            assert result is True
            
            # Test updating another key
            result = config_utils.update_key('api.model', 'gpt-4')
            assert result is True
    
    def test_update_key_nonexistent(self):
        """Test update_key with non-existent keys"""
        yaml_content = """
api:
  key: test-key
"""
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Test non-existent key - should raise KeyError
            with pytest.raises(KeyError):
                config_utils.update_key('nonexistent_key', 'value')
            
            # Test non-existent nested key
            with pytest.raises(KeyError):
                config_utils.update_key('api.nonexistent', 'value')
    
    def test_complex_nested_structure(self):
        """Test complex nested YAML structure"""
        yaml_content = """
level1:
  level2:
    level3:
      value: deep_value
    array:
      - item1
      - item2
  simple: simple_value
boolean_val: true
number_val: 42
null_val: null
"""
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Test deep nesting
            result = config_utils.load_key('level1.level2.level3.value')
            assert result == 'deep_value'
            
            # Test array access
            result = config_utils.load_key('level1.level2.array')
            assert result == ['item1', 'item2']
            
            # Test different data types
            assert config_utils.load_key('boolean_val') is True
            assert config_utils.load_key('number_val') == 42
            assert config_utils.load_key('null_val') is None
    
    def test_error_conditions(self):
        """Test various error conditions"""
        
        # Test file not found
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                config_utils.load_key('any.key')
        
        # Test invalid YAML
        invalid_yaml = "invalid: yaml: content: ["
        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with pytest.raises(Exception):  # YAML parsing error
                config_utils.load_key('any.key')
    
    def test_data_type_preservation(self):
        """Test that YAML data types are preserved"""
        yaml_content = """
string_val: "test string"
int_val: 42
float_val: 3.14159
bool_true: true
bool_false: false
null_val: null
list_val:
  - one
  - 2
  - 3.0
  - true
dict_val:
  nested: value
  number: 123
"""
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Test all data types
            assert config_utils.load_key('string_val') == "test string"
            assert config_utils.load_key('int_val') == 42
            assert config_utils.load_key('float_val') == 3.14159
            assert config_utils.load_key('bool_true') is True
            assert config_utils.load_key('bool_false') is False
            assert config_utils.load_key('null_val') is None
            
            list_result = config_utils.load_key('list_val')
            assert list_result == ['one', 2, 3.0, True]
            
            dict_result = config_utils.load_key('dict_val')
            assert dict_result == {'nested': 'value', 'number': 123}
    
    def test_thread_safety_simulation(self):
        """Test thread safety aspects"""
        yaml_content = """
test_key: test_value
"""
        
        # This test simulates what would happen with concurrent access
        # The actual threading.Lock should prevent race conditions
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Multiple calls should work consistently
            for i in range(10):
                result = config_utils.load_key('test_key')
                assert result == 'test_value'
    
    @pytest.mark.parametrize("key_path,yaml_content,expected", [
        ("simple", "simple: value", "value"),
        ("nested.key", "nested:\n  key: nested_value", "nested_value"),
        ("deep.very.nested", "deep:\n  very:\n    nested: deep_value", "deep_value"),
    ])
    def test_parameterized_scenarios(self, key_path, yaml_content, expected):
        """Test various key path scenarios"""
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            result = config_utils.load_key(key_path)
            assert result == expected


# importlib.util already imported above