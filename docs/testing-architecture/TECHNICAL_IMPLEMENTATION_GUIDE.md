# VideoLingo 测试架构技术实施详细指南

## 📋 概述

本文档为VideoLingo项目测试架构重构提供详细的技术实施指导，包含具体的代码示例、配置文件、工具使用方法和故障排除方案。

## 🏗️ 基础设施实施

### 1. Mock管理系统实现

#### 1.1 BaseMock基类
```python
# tests/mocks/base_mock.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch, MagicMock
import logging
import time
import threading
from contextlib import contextmanager

class BaseMock(ABC):
    """统一的Mock基类，提供生命周期管理和标准接口"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_active = False
        self.call_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"mock.{name}")
        self._lock = threading.Lock()
        self.patches: List[patch] = []
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    @abstractmethod
    def start(self) -> None:
        """启动Mock"""
        if self.is_active:
            self.logger.warning(f"Mock {self.name} is already active")
            return
            
        self.is_active = True
        self.logger.info(f"Mock {self.name} started")
        
    @abstractmethod  
    def stop(self) -> None:
        """停止Mock"""
        if not self.is_active:
            self.logger.warning(f"Mock {self.name} is not active")
            return
            
        for patch_obj in self.patches:
            try:
                patch_obj.stop()
            except Exception as e:
                self.logger.error(f"Error stopping patch: {e}")
                
        self.patches.clear()
        self.is_active = False
        self.logger.info(f"Mock {self.name} stopped")
        
    def record_call(self, method: str, args: tuple = (), kwargs: dict = None) -> None:
        """记录调用历史"""
        with self._lock:
            self.call_history.append({
                'method': method,
                'args': args,
                'kwargs': kwargs or {},
                'timestamp': time.time()
            })
        self.logger.debug(f"Mock call recorded: {method}")
        
    def get_call_count(self, method: Optional[str] = None) -> int:
        """获取调用次数"""
        with self._lock:
            if method is None:
                return len(self.call_history)
            return len([call for call in self.call_history if call['method'] == method])
            
    def clear_history(self) -> None:
        """清除调用历史"""
        with self._lock:
            self.call_history.clear()
        self.logger.debug(f"Mock {self.name} call history cleared")
        
    @contextmanager
    def temporary_response(self, method: str, response: Any):
        """临时设置特定方法的响应"""
        original_response = getattr(self, f"_{method}_response", None)
        setattr(self, f"_{method}_response", response)
        try:
            yield
        finally:
            if original_response is not None:
                setattr(self, f"_{method}_response", original_response)
```

#### 1.2 API服务Mock实现
```python
# tests/mocks/api_service_mock.py
from typing import Dict, Any, Optional, List
from unittest.mock import patch, MagicMock, Mock
import json
import requests
from .base_mock import BaseMock

class APIServiceMock(BaseMock):
    """统一的API服务Mock管理"""
    
    def __init__(self):
        super().__init__("api_service")
        self.response_templates = self._load_response_templates()
        self.error_conditions = {}
        self.latency_simulation = {}
        
    def _load_response_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载响应模板"""
        return {
            'openai_chat_completion': {
                'id': 'chatcmpl-test',
                'object': 'chat.completion',
                'created': 1677652288,
                'model': 'gpt-3.5-turbo',
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': 'This is a mocked response from OpenAI API.'
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 20,
                    'total_tokens': 30
                }
            },
            'azure_tts_success': {
                'status': 'success',
                'audio_data': b'mocked_audio_data_bytes',
                'content_type': 'audio/wav'
            },
            'elevenlabs_tts_success': {
                'audio': b'mocked_elevenlabs_audio',
                'history_item_id': 'mock_history_id'
            },
            'whisperx_transcription': {
                'text': 'This is a mocked transcription result.',
                'segments': [{
                    'start': 0.0,
                    'end': 5.0,
                    'text': 'This is a mocked transcription result.',
                    'words': [
                        {'start': 0.0, 'end': 0.5, 'word': 'This'},
                        {'start': 0.5, 'end': 1.0, 'word': 'is'},
                        {'start': 1.0, 'end': 1.2, 'word': 'a'},
                        {'start': 1.2, 'end': 1.8, 'word': 'mocked'},
                        {'start': 1.8, 'end': 2.5, 'word': 'transcription'},
                        {'start': 2.5, 'end': 3.0, 'word': 'result.'}
                    ]
                }],
                'language': 'en'
            }
        }
        
    def start(self) -> None:
        """启动所有API Mock"""
        super().start()
        
        # Mock OpenAI API
        self._setup_openai_mock()
        
        # Mock Azure TTS API
        self._setup_azure_mock()
        
        # Mock ElevenLabs API
        self._setup_elevenlabs_mock()
        
        # Mock WhisperX API
        self._setup_whisperx_mock()
        
        # Mock requests.post for general HTTP calls
        self._setup_requests_mock()
        
    def _setup_openai_mock(self) -> None:
        """设置OpenAI API Mock"""
        openai_patch = patch('openai.OpenAI')
        mock_openai_client = openai_patch.start()
        
        # Mock chat completion
        mock_completion = MagicMock()
        mock_completion.create.return_value = MagicMock(
            **self.response_templates['openai_chat_completion']
        )
        mock_openai_client.return_value.chat.completions = mock_completion
        
        # 记录调用
        original_create = mock_completion.create
        def create_with_recording(*args, **kwargs):
            self.record_call('openai_chat_completion', args, kwargs)
            return original_create(*args, **kwargs)
        mock_completion.create = create_with_recording
        
        self.patches.append(openai_patch)
```

### 2. 测试数据管理系统

#### 2.1 Fixture管理器
```python
# tests/fixtures/__init__.py
import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import tempfile
import shutil
from contextlib import contextmanager

class TestDataManager:
    """测试数据管理器"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, Any] = {}
        self.temp_dirs: List[Path] = []
        
    def load_fixture(self, name: str, format: str = 'json') -> Union[Dict[str, Any], List, str]:
        """加载测试fixture"""
        cache_key = f"{name}.{format}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        fixture_path = self.data_dir / f"{name}.{format}"
        if not fixture_path.exists():
            raise FileNotFoundError(f"Fixture not found: {fixture_path}")
            
        with open(fixture_path, 'r', encoding='utf-8') as f:
            if format == 'json':
                data = json.load(f)
            elif format == 'yaml' or format == 'yml':
                data = yaml.safe_load(f)
            else:
                data = f.read()
                
        self._cache[cache_key] = data
        return data
        
    def get_sample_config(self, profile: str = 'default') -> Dict[str, Any]:
        """获取示例配置"""
        configs = {
            'default': {
                'api': {
                    'key': 'test_api_key_default',
                    'base_url': 'https://api.test.com',
                    'model': 'test-model'
                },
                'video': {
                    'max_duration': 600,
                    'quality': 'best',
                    'format': 'mp4'
                }
            }
        }
        
        return configs.get(profile, configs['default'])

# 全局测试数据管理器实例
test_data = TestDataManager()
```

### 3. pytest配置优化

#### 3.1 高性能pytest.ini配置
```ini
# tests/pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts = 
    --verbose
    --tb=short
    --strict-markers
    --maxfail=10
    --cov=core
    --cov-report=html:tests/reports/coverage_html
    --cov-report=xml:tests/reports/coverage.xml
    --cov-report=term-missing:skip-covered
    --cov-fail-under=60
    --durations=10

# 并发执行
addopts = -n auto

markers =
    unit: 单元测试
    component: 组件测试
    integration: 集成测试
    e2e: 端到端测试
    slow: 慢速测试
    fast: 快速测试

filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning

timeout = 300
timeout_method = thread

log_cli = false
log_cli_level = INFO
```

## 🔧 故障排除指南

### 常见问题解决方案

#### 1. 测试执行问题

**问题**: 测试无法运行，提示导入错误
```bash
ImportError: No module named 'core'
```

**解决方案**:
```python
# 在conftest.py中添加路径设置
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

#### 2. 覆盖率问题

**问题**: 覆盖率报告不准确或缺失
```bash
coverage report
# No data to report
```

**解决方案**:
```bash
# 检查覆盖率配置
pytest --cov=core --cov-report=term-missing tests/
```

### 性能优化技巧

#### 1. 并行测试执行
```bash
# 安装pytest-xdist
pip install pytest-xdist

# 自动检测CPU核心数
pytest -n auto
```

---

*本技术实施指南提供了VideoLingo项目测试架构重构的详细技术细节和最佳实践。*