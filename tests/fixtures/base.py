"""
Test Data Management Base Classes
Abstract layer for test data management in VideoLingo
Provides unified interfaces for various data formats and test scenarios
"""

import abc
import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Generic, TypeVar
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import threading
from contextlib import contextmanager

T = TypeVar('T')

class DataFormat(Enum):
    """Supported data formats for test data"""
    JSON = "json"
    YAML = "yaml"
    TEXT = "txt"
    BINARY = "bin"
    MEDIA = "media"
    CONFIG = "config"
    XML = "xml"
    CSV = "csv"

class DataCategory(Enum):
    """Data categorization for test organization"""
    CONFIG = "config"
    API_RESPONSE = "api_response"
    MEDIA_SAMPLE = "media_sample"
    LANGUAGE_DATA = "language_data"
    ERROR_SCENARIO = "error_scenario"
    MOCK_DATA = "mock_data"
    CACHE_DATA = "cache_data"
    TEMP_DATA = "temp_data"

class DataScope(Enum):
    """Data scope for lifecycle management"""
    SESSION = "session"        # Persistent across test session
    TEST_MODULE = "test_module" # Module-level lifecycle
    TEST_FUNCTION = "test_function" # Function-level lifecycle
    TEMPORARY = "temporary"    # Cleanup immediately after use

@dataclass
class DataMetadata:
    """Metadata for test data items"""
    name: str
    category: DataCategory
    format: DataFormat
    scope: DataScope
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    size_bytes: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'name': self.name,
            'category': self.category.value,
            'format': self.format.value,
            'scope': self.scope.value,
            'version': self.version,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'checksum': self.checksum,
            'dependencies': self.dependencies,
            'size_bytes': self.size_bytes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataMetadata':
        """Create metadata from dictionary"""
        return cls(
            name=data['name'],
            category=DataCategory(data['category']),
            format=DataFormat(data['format']),
            scope=DataScope(data['scope']),
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            created_at=datetime.fromisoformat(data['created_at']),
            checksum=data.get('checksum'),
            dependencies=data.get('dependencies', []),
            size_bytes=data.get('size_bytes')
        )

class TestDataError(Exception):
    """Base exception for test data management errors"""
    pass

class DataNotFoundError(TestDataError):
    """Exception raised when test data is not found"""
    pass

class DataIntegrityError(TestDataError):
    """Exception raised when data integrity check fails"""
    pass

class DataGenerationError(TestDataError):
    """Exception raised when data generation fails"""
    pass

class AbstractDataProvider(abc.ABC, Generic[T]):
    """Abstract base class for test data providers"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.lock = threading.Lock()
        self._metadata_cache: Dict[str, DataMetadata] = {}
    
    @abc.abstractmethod
    def get_data(self, name: str, **kwargs) -> T:
        """Get test data by name"""
        pass
    
    @abc.abstractmethod
    def save_data(self, name: str, data: T, metadata: DataMetadata) -> bool:
        """Save test data with metadata"""
        pass
    
    @abc.abstractmethod
    def delete_data(self, name: str) -> bool:
        """Delete test data"""
        pass
    
    @abc.abstractmethod
    def list_data(self, category: Optional[DataCategory] = None, 
                  tags: Optional[List[str]] = None) -> List[str]:
        """List available test data"""
        pass
    
    @abc.abstractmethod
    def exists(self, name: str) -> bool:
        """Check if data exists"""
        pass
    
    def get_metadata(self, name: str) -> Optional[DataMetadata]:
        """Get metadata for test data"""
        if name in self._metadata_cache:
            return self._metadata_cache[name]
        
        metadata_path = self.base_path / f"{name}.meta.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
                metadata = DataMetadata.from_dict(metadata_dict)
                self._metadata_cache[name] = metadata
                return metadata
        return None
    
    def save_metadata(self, name: str, metadata: DataMetadata) -> bool:
        """Save metadata for test data"""
        try:
            with self.lock:
                metadata_path = self.base_path / f"{name}.meta.json"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                
                self._metadata_cache[name] = metadata
                return True
        except Exception:
            return False
    
    def calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum for data integrity"""
        return hashlib.sha256(data).hexdigest()
    
    def verify_integrity(self, name: str, data: bytes) -> bool:
        """Verify data integrity using checksum"""
        metadata = self.get_metadata(name)
        if not metadata or not metadata.checksum:
            return True
        return self.calculate_checksum(data) == metadata.checksum

class AbstractDataGenerator(abc.ABC, Generic[T]):
    """Abstract base class for test data generators"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
    
    @abc.abstractmethod
    def generate(self, **kwargs) -> T:
        """Generate test data"""
        pass
    
    @abc.abstractmethod
    def get_metadata_template(self) -> DataMetadata:
        """Get metadata template for generated data"""
        pass
    
    def generate_with_metadata(self, name: str, **kwargs) -> tuple[T, DataMetadata]:
        """Generate data with metadata"""
        data = self.generate(**kwargs)
        metadata = self.get_metadata_template()
        metadata.name = name
        return data, metadata

class AbstractDataCache(abc.ABC):
    """Abstract base class for test data caching"""
    
    @abc.abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get cached data"""
        pass
    
    @abc.abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached data with optional TTL"""
        pass
    
    @abc.abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cached data"""
        pass
    
    @abc.abstractmethod
    def clear(self) -> bool:
        """Clear all cached data"""
        pass
    
    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

class AbstractDataCleaner(abc.ABC):
    """Abstract base class for test data cleanup"""
    
    @abc.abstractmethod
    def cleanup_by_scope(self, scope: DataScope) -> int:
        """Cleanup data by scope, return number of items cleaned"""
        pass
    
    @abc.abstractmethod
    def cleanup_by_age(self, max_age_hours: int) -> int:
        """Cleanup data older than specified hours"""
        pass
    
    @abc.abstractmethod
    def cleanup_by_pattern(self, pattern: str) -> int:
        """Cleanup data matching pattern"""
        pass
    
    @abc.abstractmethod
    def cleanup_all(self) -> int:
        """Cleanup all test data"""
        pass

class DataManager:
    """Central manager for test data operations"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.providers: Dict[DataCategory, AbstractDataProvider] = {}
        self.generators: Dict[Type, AbstractDataGenerator] = {}
        self.cache: Optional[AbstractDataCache] = None
        self.cleaner: Optional[AbstractDataCleaner] = None
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def register_provider(self, category: DataCategory, 
                         provider: AbstractDataProvider) -> None:
        """Register a data provider for a category"""
        self.providers[category] = provider
    
    def register_generator(self, data_type: Type, 
                          generator: AbstractDataGenerator) -> None:
        """Register a data generator for a type"""
        self.generators[data_type] = generator
    
    def set_cache(self, cache: AbstractDataCache) -> None:
        """Set cache implementation"""
        self.cache = cache
    
    def set_cleaner(self, cleaner: AbstractDataCleaner) -> None:
        """Set cleaner implementation"""
        self.cleaner = cleaner
    
    def get_provider(self, category: DataCategory) -> Optional[AbstractDataProvider]:
        """Get provider for category"""
        return self.providers.get(category)
    
    def get_generator(self, data_type: Type) -> Optional[AbstractDataGenerator]:
        """Get generator for type"""
        return self.generators.get(data_type)
    
    @contextmanager
    def managed_data(self, name: str, category: DataCategory, 
                    scope: DataScope = DataScope.TEMPORARY):
        """Context manager for automatic data lifecycle management"""
        try:
            provider = self.get_provider(category)
            if not provider:
                raise TestDataError(f"No provider registered for category: {category}")
            
            yield provider
            
        finally:
            if scope == DataScope.TEMPORARY and provider and provider.exists(name):
                provider.delete_data(name)

    def cleanup_by_scope(self, scope: DataScope) -> int:
        """Cleanup data by scope across all providers"""
        total_cleaned = 0
        if self.cleaner:
            total_cleaned += self.cleaner.cleanup_by_scope(scope)
        return total_cleaned

# Global data manager instance
_data_manager: Optional[DataManager] = None

def get_data_manager(base_path: Optional[Union[str, Path]] = None) -> DataManager:
    """Get global data manager instance"""
    global _data_manager
    if _data_manager is None:
        if base_path is None:
            base_path = Path(__file__).parent / "test_data"
        _data_manager = DataManager(base_path)
    return _data_manager

def initialize_data_manager(base_path: Union[str, Path]) -> DataManager:
    """Initialize global data manager with specific path"""
    global _data_manager
    _data_manager = DataManager(base_path)
    return _data_manager
EOF < /dev/null