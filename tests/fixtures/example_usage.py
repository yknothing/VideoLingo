"""
Example Usage of VideoLingo Test Data Management System

This file demonstrates how to use the test data management system
in various testing scenarios.
"""

import pytest
from pathlib import Path
from typing import Dict, Any

# Import the test data management system
from tests.fixtures import (
    initialize_test_data_system,
    get_videolingo_fixtures,
    load_sample_config,
    load_sample_api_response,
    DataCategory,
    DataScope,
    get_cache_manager,
    get_session_cleaner
)
from tests.fixtures.generators import ConfigGenerator, APIResponseGenerator

class TestDataManagementExamples:
    """Example usage patterns for the test data management system"""
    
    @classmethod
    def setup_class(cls):
        """Set up test data system for the class"""
        cls.data_manager = initialize_test_data_system(Path("/tmp/videolingo_test_data"))
        cls.fixtures = get_videolingo_fixtures(cls.data_manager)
        cls.cache = get_cache_manager()
        cls.cleaner = get_session_cleaner()
    
    def test_basic_config_loading(self):
        """Example: Loading predefined configuration samples"""
        
        # Load minimal configuration
        minimal_config = load_sample_config("minimal")
        assert minimal_config['api']['key'] == "test-api-key"
        assert minimal_config['max_workers'] == 1
        
        # Load OpenRouter configuration
        openrouter_config = load_sample_config("openrouter")
        assert "openrouter.ai" in openrouter_config['api']['base_url']
        assert openrouter_config['api']['model'] == "anthropic/claude-3.5-sonnet"
        
        # Load full-featured configuration
        full_config = load_sample_config("full_featured")
        assert 'openai_tts' in full_config
        assert 'azure_tts' in full_config
        assert full_config['max_workers'] == 4
    
    def test_dynamic_config_generation(self):
        """Example: Generating configuration data dynamically"""
        
        generator = ConfigGenerator(seed=42)  # Reproducible results
        
        # Generate different configuration variants
        test_configs = [
            generator.generate(variant="minimal"),
            generator.generate(variant="openrouter"),
            generator.generate(variant="azure"),
            generator.generate(variant="error_prone")
        ]
        
        # Verify each config has expected structure
        for config in test_configs:
            assert 'api' in config
            assert 'max_workers' in config
            assert 'target_language' in config
    
    def test_api_response_mocking(self):
        """Example: Loading and generating API response mocks"""
        
        # Load predefined API responses
        openai_success = load_sample_api_response("openai", "success")
        assert openai_success['object'] == 'chat.completion'
        assert 'choices' in openai_success
        
        # Generate dynamic API responses
        api_generator = APIResponseGenerator(seed=123)
        
        responses = [
            api_generator.generate(api_type="openai", scenario="success"),
            api_generator.generate(api_type="openai", scenario="error"),
            api_generator.generate(api_type="whisper", scenario="success"),
            api_generator.generate(api_type="tts", scenario="success")
        ]
        
        # Verify response structures
        assert responses[0]['object'] == 'chat.completion'  # OpenAI success
        assert 'error' in responses[1]  # OpenAI error
        assert 'text' in responses[2]  # Whisper success
        assert 'audio_data' in responses[3]  # TTS success
    
    def test_media_file_creation(self):
        """Example: Creating temporary media files for testing"""
        
        # Create mock video file
        video_file = self.fixtures.create_mock_video_file(duration_seconds=5)
        assert video_file.exists()
        assert video_file.suffix == '.mp4'
        
        # Create mock audio file
        audio_file = self.fixtures.create_mock_audio_file(duration_seconds=3)
        assert audio_file.exists()
        assert audio_file.suffix == '.wav'
        
        # Create test directory structure
        test_dir = self.fixtures.create_test_directory_structure()
        assert (test_dir / 'input').exists()
        assert (test_dir / 'output').exists()
        assert (test_dir / 'temp' / 'log').exists()
        
        # Files will be automatically cleaned up
    
    def test_data_caching(self):
        """Example: Using the caching system for expensive operations"""
        
        cache = self.cache.get_cache()
        
        # Simulate expensive computation
        def expensive_computation(x: int) -> Dict[str, Any]:
            import time
            time.sleep(0.1)  # Simulate delay
            return {'result': x * x, 'computed_at': time.time()}
        
        # Cache the result with module scope
        key = "expensive_result_100"
        if not cache.get(key):
            result = expensive_computation(100)
            cache.set(key, result, DataScope.TEST_MODULE)
        
        # Retrieve cached result (should be fast)
        cached_result = cache.get(key)
        assert cached_result is not None
        assert cached_result['result'] == 10000
    
    def test_error_scenario_handling(self):
        """Example: Testing error scenarios with predefined data"""
        
        from tests.fixtures.generators import ErrorScenarioGenerator
        
        error_generator = ErrorScenarioGenerator(seed=456)
        
        # Generate different types of errors
        errors = [
            error_generator.generate(error_type="api_error"),
            error_generator.generate(error_type="file_error"),
            error_generator.generate(error_type="network_error"),
            error_generator.generate(error_type="validation_error")
        ]
        
        # Verify error structures
        assert errors[0]['error_code'] in [401, 429, 503, 400]  # API error
        assert 'file_path' in errors[1] or 'error_message' in errors[1]  # File error
        assert 'error_type' in errors[2]  # Network error
        assert 'field' in errors[3]  # Validation error
    
    def test_language_data_handling(self):
        """Example: Working with multilingual test data"""
        
        from tests.fixtures.generators import LanguageDataGenerator
        
        lang_generator = LanguageDataGenerator(seed=789)
        
        # Generate language-specific data
        english_sentences = lang_generator.generate(language="en", data_type="sentences")
        chinese_sentences = lang_generator.generate(language="zh", data_type="sentences")
        translation_pairs = lang_generator.generate(language="zh", data_type="translations")
        
        # Verify language data structure
        assert english_sentences['language'] == 'en'
        assert len(english_sentences['sentences']) > 0
        
        assert chinese_sentences['language'] == 'zh'
        assert len(chinese_sentences['sentences']) > 0
        
        assert translation_pairs['source_language'] == 'en'
        assert translation_pairs['target_language'] == 'zh'
        assert len(translation_pairs['pairs']) > 0
    
    def test_data_provider_integration(self):
        """Example: Using data providers directly"""
        
        # Get YAML provider for configurations
        config_provider = self.data_manager.get_provider(DataCategory.CONFIG)
        assert config_provider is not None
        
        # Save and retrieve custom configuration
        from tests.fixtures.base import DataMetadata, DataFormat
        
        custom_config = {
            'api': {'key': 'custom-test-key'},
            'max_workers': 8,
            'custom_field': 'test_value'
        }
        
        metadata = DataMetadata(
            name="custom_test_config",
            category=DataCategory.CONFIG,
            format=DataFormat.YAML,
            scope=DataScope.TEST_FUNCTION,
            description="Custom test configuration",
            tags=["custom", "test"]
        )
        
        # Save the data
        success = config_provider.save_data("custom_test_config", custom_config, metadata)
        assert success
        
        # Retrieve the data
        retrieved_config = config_provider.get_data("custom_test_config")
        assert retrieved_config == custom_config
        
        # Clean up
        config_provider.delete_data("custom_test_config")
    
    def test_managed_data_context(self):
        """Example: Using managed data context for automatic cleanup"""
        
        # Use context manager for automatic lifecycle management
        with self.data_manager.managed_data("temp_config", DataCategory.CONFIG, DataScope.TEMPORARY) as provider:
            
            # Create temporary configuration
            temp_config = {'test': 'temporary_data', 'should_cleanup': True}
            
            metadata = DataMetadata(
                name="temp_config",
                category=DataCategory.CONFIG,
                format=DataFormat.YAML,
                scope=DataScope.TEMPORARY,
                description="Temporary test configuration"
            )
            
            provider.save_data("temp_config", temp_config, metadata)
            
            # Use the data within the context
            retrieved = provider.get_data("temp_config")
            assert retrieved['test'] == 'temporary_data'
        
        # Data should be automatically cleaned up after context exit
        # (This is handled by the TEMPORARY scope)
    
    def test_comprehensive_workflow(self):
        """Example: Complete workflow combining all components"""
        
        # 1. Generate test configuration
        config_gen = ConfigGenerator(seed=999)
        test_config, config_metadata = config_gen.generate_with_metadata(
            "workflow_test_config",
            variant="openrouter"
        )
        
        # 2. Cache the configuration
        cache = self.cache.get_cache()
        cache.set("workflow_config", test_config, DataScope.TEST_FUNCTION)
        
        # 3. Create test media files
        video_file = self.fixtures.create_mock_video_file(10)
        audio_file = self.fixtures.create_mock_audio_file(5)
        
        # 4. Register files for cleanup
        self.cleaner.register_temp_file(video_file)
        self.cleaner.register_temp_file(audio_file)
        
        # 5. Generate API responses for the workflow
        api_gen = APIResponseGenerator(seed=999)
        transcription_response = api_gen.generate(api_type="whisper", scenario="success")
        translation_response = api_gen.generate(api_type="openai", scenario="success")
        
        # 6. Simulate workflow execution
        workflow_result = {
            'config': test_config,
            'input_video': str(video_file),
            'input_audio': str(audio_file),
            'transcription': transcription_response,
            'translation': translation_response,
            'status': 'completed'
        }
        
        # 7. Cache the workflow result
        cache.set("workflow_result", workflow_result, DataScope.TEST_FUNCTION)
        
        # 8. Verify everything worked
        cached_config = cache.get("workflow_config")
        cached_result = cache.get("workflow_result")
        
        assert cached_config == test_config
        assert cached_result['status'] == 'completed'
        assert video_file.exists()
        assert audio_file.exists()
        
        # Cleanup will happen automatically based on scopes
    
    @classmethod
    def teardown_class(cls):
        """Clean up after all tests"""
        # Cleanup will be handled automatically by the cleanup system
        # But we can also manually trigger cleanup if needed
        cls.cleaner.cleanup_by_scope(DataScope.TEST_MODULE)
        cls.fixtures.cleanup_all()

if __name__ == "__main__":
    """
    Run this file directly to see the test data management system in action
    """
    print("VideoLingo Test Data Management System - Example Usage")
    print("=" * 60)
    
    # Initialize the system
    print("1. Initializing test data system...")
    manager = initialize_test_data_system()
    fixtures = get_videolingo_fixtures(manager)
    
    # Load sample data
    print("2. Loading sample configurations...")
    configs = {
        'minimal': load_sample_config("minimal"),
        'openrouter': load_sample_config("openrouter"),
        'azure': load_sample_config("azure")
    }
    
    for name, config in configs.items():
        print(f"   - {name}: {config['api']['model']}")
    
    # Generate dynamic data
    print("3. Generating dynamic test data...")
    config_gen = ConfigGenerator(seed=42)
    api_gen = APIResponseGenerator(seed=42)
    
    generated_config = config_gen.generate(variant="full")
    generated_response = api_gen.generate(api_type="openai", scenario="success")
    
    print(f"   - Generated config workers: {generated_config['max_workers']}")
    print(f"   - Generated API response tokens: {generated_response['usage']['total_tokens']}")
    
    # Create test files
    print("4. Creating test media files...")
    video_file = fixtures.create_mock_video_file(5)
    audio_file = fixtures.create_mock_audio_file(3)
    test_dir = fixtures.create_test_directory_structure()
    
    print(f"   - Video file: {video_file}")
    print(f"   - Audio file: {audio_file}")
    print(f"   - Test directory: {test_dir}")
    
    # Test caching
    print("5. Testing caching system...")
    cache = get_cache_manager()
    scoped_cache = cache.get_cache()
    
    scoped_cache.set("example_data", {"cached": True, "value": 42}, DataScope.TEMPORARY)
    cached_data = scoped_cache.get("example_data")
    print(f"   - Cached data: {cached_data}")
    
    # Show stats
    print("6. System statistics...")
    cache_stats = cache.get_cache_stats()
    print(f"   - Cache entries: {cache_stats['memory']['size']}")
    
    cleaner = get_session_cleaner()
    cleaner_stats = cleaner.cleaner.get_stats()
    print(f"   - Tracked files: {cleaner_stats['tracked_files']}")
    print(f"   - Tracked directories: {cleaner_stats['tracked_dirs']}")
    
    print("\n✓ Test data management system is working correctly\!")
    print("Run 'pytest tests/fixtures/example_usage.py' to run the full test suite.")
EOF < /dev/null