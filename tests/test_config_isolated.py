# Isolated test for config_utils to check actual coverage

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, Mock
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the actual module we want to test
from core.utils.config_utils import load_key, save_key

class TestConfigUtilsReal:
    """Test actual config_utils module"""
    
    def test_load_key_with_mock_file(self):
        """Test load_key with mocked file"""
        mock_config = {
            'api': {
                'key': 'test-api-key',
                'model': 'gpt-4'
            },
            'paths': {
                'input': '/test/input'
            }
        }
        
        with patch('core.utils.config_utils.load_all_config', return_value=mock_config):
            # Test nested key access
            result = load_key('api.key')
            assert result == 'test-api-key'
            
            # Test another nested key
            result = load_key('api.model')
            assert result == 'gpt-4'
            
            # Test non-existent key
            result = load_key('nonexistent.key')
            assert result is None
            
            # Test with default value
            result = load_key('nonexistent.key', default='default_value')
            assert result == 'default_value'
    
    def test_save_key_functionality(self):
        """Test save_key functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, 'test_config.yaml')
            
            # Create initial config
            initial_config = {'existing': {'key': 'value'}}
            with open(config_file, 'w') as f:
                yaml.dump(initial_config, f)
            
            with patch('core.utils.config_utils.CONFIG_PATH', config_file):
                # Test saving new key
                result = save_key('new.nested.key', 'new_value')
                assert result is True
                
                # Verify it was saved
                with open(config_file, 'r') as f:
                    saved_config = yaml.safe_load(f)
                
                assert saved_config['new']['nested']['key'] == 'new_value'
                assert saved_config['existing']['key'] == 'value'  # Existing preserved
    
    def test_complex_nested_operations(self):
        """Test complex nested key operations"""
        mock_config = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 'deep_value'
                    }
                }
            },
            'array_value': [1, 2, 3],
            'boolean_value': True,
            'null_value': None
        }
        
        with patch('core.utils.config_utils.load_all_config', return_value=mock_config):
            # Test deep nesting
            assert load_key('level1.level2.level3.value') == 'deep_value'
            
            # Test partial path
            level2_result = load_key('level1.level2')
            assert level2_result == {'level3': {'value': 'deep_value'}}
            
            # Test array value
            assert load_key('array_value') == [1, 2, 3]
            
            # Test boolean value
            assert load_key('boolean_value') is True
            
            # Test null value
            assert load_key('null_value') is None
    
    def test_error_conditions(self):
        """Test error handling conditions"""
        
        # Test with invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            f.flush()
            invalid_file = f.name
        
        try:
            with patch('core.utils.config_utils.CONFIG_PATH', invalid_file):
                # Should handle invalid YAML gracefully
                result = load_key('any.key')
                # Implementation might return None or empty dict
                assert result is None or result == {}
        finally:
            os.unlink(invalid_file)
        
        # Test with non-existent file
        with patch('core.utils.config_utils.CONFIG_PATH', '/nonexistent/file.yaml'):
            result = load_key('any.key')
            assert result is None
    
    def test_data_type_preservation(self):
        """Test that data types are preserved correctly"""
        test_config = {
            'string_val': 'test_string',
            'int_val': 42,
            'float_val': 3.14,
            'bool_val': True,
            'null_val': None,
            'list_val': [1, 'two', 3.0],
            'dict_val': {'nested': 'value'}
        }
        
        with patch('core.utils.config_utils.load_all_config', return_value=test_config):
            # Test type preservation
            assert isinstance(load_key('string_val'), str)
            assert isinstance(load_key('int_val'), int)
            assert isinstance(load_key('float_val'), float)
            assert isinstance(load_key('bool_val'), bool)
            assert load_key('null_val') is None
            assert isinstance(load_key('list_val'), list)
            assert isinstance(load_key('dict_val'), dict)
            
            # Test actual values
            assert load_key('string_val') == 'test_string'
            assert load_key('int_val') == 42
            assert load_key('float_val') == 3.14
            assert load_key('bool_val') is True
            assert load_key('list_val') == [1, 'two', 3.0]
            assert load_key('dict_val') == {'nested': 'value'}
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        edge_config = {
            'empty_string': '',
            'empty_dict': {},
            'empty_list': [],
            'zero': 0,
            'false': False,
            'special_chars': 'áéíóú!@#$%^&*()',
            'unicode': '你好世界 🌍',
            'very': {
                'deeply': {
                    'nested': {
                        'key': {
                            'with': {
                                'many': {
                                    'levels': 'deep_value'
                                }
                            }
                        }
                    }
                }
            }
        }
        
        with patch('core.utils.config_utils.load_all_config', return_value=edge_config):
            # Test empty values
            assert load_key('empty_string') == ''
            assert load_key('empty_dict') == {}
            assert load_key('empty_list') == []
            assert load_key('zero') == 0
            assert load_key('false') is False
            
            # Test special characters
            assert load_key('special_chars') == 'áéíóú!@#$%^&*()'
            assert load_key('unicode') == '你好世界 🌍'
            
            # Test very deep nesting
            deep_result = load_key('very.deeply.nested.key.with.many.levels')
            assert deep_result == 'deep_value'
    
    @pytest.mark.parametrize("key_path,expected", [
        ("simple", "value"),
        ("nested.key", "nested_value"),
        ("nonexistent", None),
        ("partial.nonexistent", None),
    ])
    def test_parameterized_key_access(self, key_path, expected):
        """Test parameterized key access patterns"""
        test_config = {
            'simple': 'value',
            'nested': {
                'key': 'nested_value'
            }
        }
        
        with patch('core.utils.config_utils.load_all_config', return_value=test_config):
            result = load_key(key_path)
            assert result == expected