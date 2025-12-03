"""
Test Fixtures Management System
Concrete implementations of data providers and fixture management for VideoLingo
"""

import json
import yaml
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import time
import io

from .base import (
    AbstractDataProvider, DataCategory, DataFormat, DataScope, 
    DataMetadata, TestDataError, DataNotFoundError, DataIntegrityError
)

class JSONDataProvider(AbstractDataProvider[Dict[str, Any]]):
    """Data provider for JSON test data"""
    
    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path)
        self.base_path = Path(base_path) / "json"
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_data(self, name: str, **kwargs) -> Dict[str, Any]:
        """Get JSON test data by name"""
        file_path = self.base_path / f"{name}.json"
        if not file_path.exists():
            raise DataNotFoundError(f"JSON data '{name}' not found at {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Verify integrity if metadata exists
            file_bytes = file_path.read_bytes()
            if not self.verify_integrity(name, file_bytes):
                raise DataIntegrityError(f"Data integrity check failed for '{name}'")
            
            return data
        except json.JSONDecodeError as e:
            raise TestDataError(f"Invalid JSON in '{name}': {e}")
    
    def save_data(self, name: str, data: Dict[str, Any], metadata: DataMetadata) -> bool:
        """Save JSON test data with metadata"""
        try:
            with self.lock:
                file_path = self.base_path / f"{name}.json"
                
                # Save JSON data
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Update metadata with checksum and size
                file_bytes = file_path.read_bytes()
                metadata.checksum = self.calculate_checksum(file_bytes)
                metadata.size_bytes = len(file_bytes)
                
                # Save metadata
                return self.save_metadata(name, metadata)
                
        except Exception as e:
            print(f"Error saving JSON data '{name}': {e}")
            return False
    
    def delete_data(self, name: str) -> bool:
        """Delete JSON test data"""
        try:
            with self.lock:
                file_path = self.base_path / f"{name}.json"
                metadata_path = self.base_path / f"{name}.meta.json"
                
                success = True
                if file_path.exists():
                    file_path.unlink()
                else:
                    success = False
                
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Remove from cache
                if name in self._metadata_cache:
                    del self._metadata_cache[name]
                
                return success
        except Exception:
            return False
    
    def list_data(self, category: Optional[DataCategory] = None, 
                  tags: Optional[List[str]] = None) -> List[str]:
        """List available JSON test data"""
        names = []
        for file_path in self.base_path.glob("*.json"):
            name = file_path.stem
            metadata = self.get_metadata(name)
            
            # Filter by category
            if category and metadata and metadata.category \!= category:
                continue
            
            # Filter by tags
            if tags and metadata:
                if not any(tag in metadata.tags for tag in tags):
                    continue
            
            names.append(name)
        
        return sorted(names)
    
    def exists(self, name: str) -> bool:
        """Check if JSON data exists"""
        return (self.base_path / f"{name}.json").exists()

class YAMLDataProvider(AbstractDataProvider[Dict[str, Any]]):
    """Data provider for YAML test data"""
    
    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path)
        self.base_path = Path(base_path) / "yaml"
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_data(self, name: str, **kwargs) -> Dict[str, Any]:
        """Get YAML test data by name"""
        file_path = self.base_path / f"{name}.yaml"
        if not file_path.exists():
            raise DataNotFoundError(f"YAML data '{name}' not found at {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            # Verify integrity if metadata exists
            file_bytes = file_path.read_bytes()
            if not self.verify_integrity(name, file_bytes):
                raise DataIntegrityError(f"Data integrity check failed for '{name}'")
            
            return data
        except yaml.YAMLError as e:
            raise TestDataError(f"Invalid YAML in '{name}': {e}")
    
    def save_data(self, name: str, data: Dict[str, Any], metadata: DataMetadata) -> bool:
        """Save YAML test data with metadata"""
        try:
            with self.lock:
                file_path = self.base_path / f"{name}.yaml"
                
                # Save YAML data
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                
                # Update metadata with checksum and size
                file_bytes = file_path.read_bytes()
                metadata.checksum = self.calculate_checksum(file_bytes)
                metadata.size_bytes = len(file_bytes)
                
                # Save metadata
                return self.save_metadata(name, metadata)
                
        except Exception as e:
            print(f"Error saving YAML data '{name}': {e}")
            return False
    
    def delete_data(self, name: str) -> bool:
        """Delete YAML test data"""
        try:
            with self.lock:
                file_path = self.base_path / f"{name}.yaml"
                metadata_path = self.base_path / f"{name}.meta.json"
                
                success = True
                if file_path.exists():
                    file_path.unlink()
                else:
                    success = False
                
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Remove from cache
                if name in self._metadata_cache:
                    del self._metadata_cache[name]
                
                return success
        except Exception:
            return False
    
    def list_data(self, category: Optional[DataCategory] = None, 
                  tags: Optional[List[str]] = None) -> List[str]:
        """List available YAML test data"""
        names = []
        for file_path in self.base_path.glob("*.yaml"):
            name = file_path.stem
            metadata = self.get_metadata(name)
            
            # Filter by category
            if category and metadata and metadata.category \!= category:
                continue
            
            # Filter by tags
            if tags and metadata:
                if not any(tag in metadata.tags for tag in tags):
                    continue
            
            names.append(name)
        
        return sorted(names)
    
    def exists(self, name: str) -> bool:
        """Check if YAML data exists"""
        return (self.base_path / f"{name}.yaml").exists()

class BinaryDataProvider(AbstractDataProvider[bytes]):
    """Data provider for binary test data (media files, etc.)"""
    
    def __init__(self, base_path: Union[str, Path]):
        super().__init__(base_path)
        self.base_path = Path(base_path) / "binary"
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_data(self, name: str, **kwargs) -> bytes:
        """Get binary test data by name"""
        # Find file with any extension
        matching_files = list(self.base_path.glob(f"{name}.*"))
        if not matching_files:
            raise DataNotFoundError(f"Binary data '{name}' not found")
        
        file_path = matching_files[0]  # Use first match
        
        try:
            data = file_path.read_bytes()
            
            # Verify integrity if metadata exists
            if not self.verify_integrity(name, data):
                raise DataIntegrityError(f"Data integrity check failed for '{name}'")
            
            return data
        except Exception as e:
            raise TestDataError(f"Error reading binary data '{name}': {e}")
    
    def save_data(self, name: str, data: bytes, metadata: DataMetadata) -> bool:
        """Save binary test data with metadata"""
        try:
            with self.lock:
                # Determine file extension from metadata tags or use .bin
                extension = ".bin"
                for tag in metadata.tags:
                    if tag.startswith("ext:"):
                        extension = tag[4:]  # Remove "ext:" prefix
                        if not extension.startswith("."):
                            extension = "." + extension
                        break
                
                file_path = self.base_path / f"{name}{extension}"
                
                # Save binary data
                file_path.write_bytes(data)
                
                # Update metadata with checksum and size
                metadata.checksum = self.calculate_checksum(data)
                metadata.size_bytes = len(data)
                
                # Save metadata
                return self.save_metadata(name, metadata)
                
        except Exception as e:
            print(f"Error saving binary data '{name}': {e}")
            return False
    
    def delete_data(self, name: str) -> bool:
        """Delete binary test data"""
        try:
            with self.lock:
                # Find and delete file with any extension
                matching_files = list(self.base_path.glob(f"{name}.*"))
                metadata_path = self.base_path / f"{name}.meta.json"
                
                success = False
                for file_path in matching_files:
                    if file_path.suffix \!= ".json":  # Don't delete metadata files
                        file_path.unlink()
                        success = True
                
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Remove from cache
                if name in self._metadata_cache:
                    del self._metadata_cache[name]
                
                return success
        except Exception:
            return False
    
    def list_data(self, category: Optional[DataCategory] = None, 
                  tags: Optional[List[str]] = None) -> List[str]:
        """List available binary test data"""
        names = set()
        for file_path in self.base_path.iterdir():
            if file_path.suffix \!= ".json":  # Skip metadata files
                name = file_path.stem
                metadata = self.get_metadata(name)
                
                # Filter by category
                if category and metadata and metadata.category \!= category:
                    continue
                
                # Filter by tags
                if tags and metadata:
                    if not any(tag in metadata.tags for tag in tags):
                        continue
                
                names.add(name)
        
        return sorted(names)
    
    def exists(self, name: str) -> bool:
        """Check if binary data exists"""
        matching_files = list(self.base_path.glob(f"{name}.*"))
        return any(f.suffix \!= ".json" for f in matching_files)

class FixtureRegistry:
    """Registry for managing test fixtures and their lifecycle"""
    
    def __init__(self):
        self._fixtures: Dict[str, Any] = {}
        self._cleanup_callbacks: Dict[str, List[Callable]] = {}
        self._temporary_paths: List[Path] = []
    
    def register_fixture(self, name: str, value: Any, 
                        cleanup_callback: Optional[Callable] = None) -> None:
        """Register a test fixture"""
        self._fixtures[name] = value
        if cleanup_callback:
            if name not in self._cleanup_callbacks:
                self._cleanup_callbacks[name] = []
            self._cleanup_callbacks[name].append(cleanup_callback)
    
    def get_fixture(self, name: str) -> Any:
        """Get a registered fixture"""
        if name not in self._fixtures:
            raise TestDataError(f"Fixture '{name}' not found")
        return self._fixtures[name]
    
    def cleanup_fixture(self, name: str) -> bool:
        """Cleanup a specific fixture"""
        if name in self._cleanup_callbacks:
            for callback in self._cleanup_callbacks[name]:
                try:
                    callback()
                except Exception as e:
                    print(f"Warning: Error in cleanup callback for '{name}': {e}")
            del self._cleanup_callbacks[name]
        
        if name in self._fixtures:
            del self._fixtures[name]
            return True
        return False
    
    def cleanup_all(self) -> int:
        """Cleanup all fixtures"""
        count = 0
        for name in list(self._fixtures.keys()):
            if self.cleanup_fixture(name):
                count += 1
        
        # Cleanup temporary paths
        for path in self._temporary_paths:
            try:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            except Exception as e:
                print(f"Warning: Could not cleanup temporary path {path}: {e}")
        
        self._temporary_paths.clear()
        return count
    
    def create_temp_file(self, content: Union[str, bytes] = "", 
                        suffix: str = ".tmp", prefix: str = "test_") -> Path:
        """Create a temporary file and register it for cleanup"""
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        temp_path = Path(temp_path)
        
        try:
            with os.fdopen(fd, 'wb' if isinstance(content, bytes) else 'w') as f:
                f.write(content)
        except Exception:
            os.close(fd)
            raise
        
        self._temporary_paths.append(temp_path)
        return temp_path
    
    def create_temp_dir(self, prefix: str = "test_") -> Path:
        """Create a temporary directory and register it for cleanup"""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self._temporary_paths.append(temp_dir)
        return temp_dir

class VideoLingoFixtures:
    """VideoLingo-specific test fixtures"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.registry = FixtureRegistry()
    
    def get_config_fixture(self, variant: str = "default") -> Dict[str, Any]:
        """Get configuration fixture variant"""
        provider = self.data_manager.get_provider(DataCategory.CONFIG)
        if not provider:
            raise TestDataError("No config provider registered")
        
        try:
            return provider.get_data(f"config_{variant}")
        except DataNotFoundError:
            # Return minimal default config if variant not found
            return self._get_minimal_config()
    
    def get_api_response_fixture(self, api_type: str, 
                                scenario: str = "success") -> Dict[str, Any]:
        """Get API response fixture"""
        provider = self.data_manager.get_provider(DataCategory.API_RESPONSE)
        if not provider:
            raise TestDataError("No API response provider registered")
        
        return provider.get_data(f"{api_type}_{scenario}")
    
    def get_media_sample_fixture(self, media_type: str, 
                               size: str = "small") -> bytes:
        """Get media sample fixture"""
        provider = self.data_manager.get_provider(DataCategory.MEDIA_SAMPLE)
        if not provider:
            raise TestDataError("No media sample provider registered")
        
        return provider.get_data(f"{media_type}_{size}")
    
    def create_mock_video_file(self, duration_seconds: int = 10) -> Path:
        """Create a mock video file for testing"""
        # Create a minimal valid video file structure
        # This is a placeholder - in real implementation you'd use proper video generation
        mock_video_content = b'MOCK_VIDEO_FILE' + b'\x00' * (1024 * duration_seconds)
        
        temp_file = self.registry.create_temp_file(
            content=mock_video_content,
            suffix='.mp4',
            prefix='test_video_'
        )
        
        return temp_file
    
    def create_mock_audio_file(self, duration_seconds: int = 5) -> Path:
        """Create a mock audio file for testing"""
        # Create a minimal valid audio file structure
        mock_audio_content = b'MOCK_AUDIO_FILE' + b'\x00' * (512 * duration_seconds)
        
        temp_file = self.registry.create_temp_file(
            content=mock_audio_content,
            suffix='.wav',
            prefix='test_audio_'
        )
        
        return temp_file
    
    def create_test_directory_structure(self) -> Path:
        """Create a complete test directory structure"""
        temp_dir = self.registry.create_temp_dir(prefix='videolingo_test_')
        
        # Create subdirectories matching VideoLingo's structure
        subdirs = [
            'input', 'output', 'temp', 'temp/log', 'temp/audio',
            'temp/audio/refers', 'temp/audio/segs', 'temp/audio/tmp'
        ]
        
        for subdir in subdirs:
            (temp_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        return temp_dir
    
    def _get_minimal_config(self) -> Dict[str, Any]:
        """Get minimal configuration for testing"""
        from core.constants import ConfigContract
        
        return {
            'api': {
                'key': 'test-api-key',
                'base_url': 'https://test.example.com',
                'model': 'test-model',
                'llm_support_json': True
            },
            'max_workers': 1,
            'target_language': '简体中文',
            'display_language': 'zh-CN',
            'youtube_resolution': '720',
            'demucs': False,
            'burn_subtitles': False,
            'ffmpeg_gpu': False,
            'whisper': {
                'model': 'base',
                'language': 'en',
                'runtime': 'local'
            },
            'tts_method': 'openai_tts',
            'subtitle': {
                'max_length': 75,
                'target_multiplier': 1.2
            },
            'video_storage': {
                'base_path': '',
                'input_dir': 'input',
                'temp_dir': 'temp', 
                'output_dir': 'output'
            }
        }
    
    def cleanup_all(self) -> int:
        """Cleanup all fixtures"""
        return self.registry.cleanup_all()

# Global fixture registry
_fixture_registry: Optional[FixtureRegistry] = None

def get_fixture_registry() -> FixtureRegistry:
    """Get global fixture registry"""
    global _fixture_registry
    if _fixture_registry is None:
        _fixture_registry = FixtureRegistry()
    return _fixture_registry

def cleanup_all_fixtures():
    """Cleanup all global fixtures"""
    global _fixture_registry
    if _fixture_registry:
        count = _fixture_registry.cleanup_all()
        _fixture_registry = None
        return count
    return 0
EOF < /dev/null