# VideoLingo E2E Test Suite

## 概述 (Overview)

VideoLingo端到端（E2E）测试套件旨在验证整个视频处理流程的正确性和可靠性。这些测试覆盖了从视频输入到最终输出的完整工作流程，确保系统在各种场景下都能正常工作。

The VideoLingo End-to-End (E2E) test suite is designed to validate the correctness and reliability of the entire video processing pipeline. These tests cover the complete workflow from video input to final output, ensuring the system works correctly under various scenarios.

## 🚀 快速开始 (Quick Start)

### 运行完整E2E测试套件 (Run Complete E2E Test Suite)

```bash
# 运行所有E2E测试
python tests/e2e/test_e2e_suite_runner.py

# 或使用pytest运行
pytest tests/e2e/ -v --tb=short
```

### 运行特定测试场景 (Run Specific Test Scenarios)

```bash
# 完整流程测试
pytest tests/e2e/test_complete_video_pipeline.py -v

# 错误恢复测试
pytest tests/e2e/test_error_recovery_pipeline.py -v

# 配置变化测试
pytest tests/e2e/test_configuration_variants_pipeline.py -v
```

## 📋 测试场景 (Test Scenarios)

### 1. 完整视频处理流程 (Complete Video Pipeline)

**文件**: `test_complete_video_pipeline.py`

- **完整流程测试** (`test_full_pipeline_with_dubbing`)
  - 视频下载 → ASR转录 → 翻译 → TTS配音 → 最终输出
  - 验证端到端质量和性能指标
  - 测试时间：~45秒

- **纯字幕流程** (`test_subtitle_only_pipeline`)
  - 视频处理 → ASR → 翻译 → 字幕生成（无配音）
  - 验证字幕同步精度和翻译质量
  - 测试时间：~30秒

- **多语言处理** (`test_multi_language_pipeline`)
  - 多语言视频检测和处理
  - 跨语言翻译矩阵验证
  - 多输出格式生成
  - 测试时间：~60秒

### 2. 错误恢复机制 (Error Recovery Pipeline)

**文件**: `test_error_recovery_pipeline.py`

- **网络故障恢复** (`test_network_failure_recovery`)
  - 网络中断和API故障处理
  - 断点续传和重试机制
  - 测试时间：~25秒

- **资源耗尽恢复** (`test_resource_exhaustion_recovery`)
  - 内存、磁盘、CPU资源限制处理
  - 自适应资源管理
  - 测试时间：~20秒

- **数据损坏恢复** (`test_data_corruption_recovery`)
  - 文件损坏检测和修复
  - 数据完整性验证
  - 测试时间：~15秒

- **级联故障恢复** (`test_cascading_failure_recovery`)
  - 多系统同时故障处理
  - 复杂恢复场景验证
  - 测试时间：~35秒

### 3. 配置变化测试 (Configuration Variants)

**文件**: `test_configuration_variants_pipeline.py`

- **TTS后端变化** (`test_tts_backend_variations`)
  - Azure TTS, OpenAI TTS, Edge TTS, GPT-SoVITS, Fish TTS
  - 质量、速度、成本对比验证
  - 测试时间：~40秒

- **语言配置矩阵** (`test_language_configuration_matrix`)
  - 多种源语言-目标语言组合
  - 语言特定处理优化
  - 测试时间：~50秒

- **质量配置预设** (`test_quality_configuration_presets`)
  - 速度优化、平衡、质量优化配置
  - 性能与质量权衡验证
  - 测试时间：~25秒

- **高级配置场景** (`test_advanced_configuration_scenarios`)
  - 混合处理、企业工作流、AI优化
  - 复杂配置集成测试
  - 测试时间：~35秒

## 🏆 质量指标 (Quality Metrics)

### 性能指标 (Performance Metrics)
- **处理速度**: 字符/秒、视频分钟/小时
- **响应时间**: API调用延迟、端到端处理时间
- **资源利用率**: CPU、内存、磁盘使用效率

### 质量指标 (Quality Metrics)
- **ASR准确率**: >90% (目标: >95%)
- **翻译质量**: >85% (目标: >90%)
- **TTS自然度**: >85% (目标: >90%)
- **字幕同步精度**: >95% (目标: >98%)

### 可靠性指标 (Reliability Metrics)
- **系统可用性**: >99.5%
- **错误恢复率**: >95%
- **数据完整性**: 100%
- **故障隔离**: <5分钟

## 📊 测试报告 (Test Reports)

### 自动生成报告 (Automated Report Generation)

运行E2E测试套件后，会自动生成详细的测试报告：

```json
{
  "test_suite_info": {
    "name": "VideoLingo E2E Test Suite",
    "execution_date": "2024-01-15 14:30:00",
    "total_duration_formatted": "15m 30s"
  },
  "summary_statistics": {
    "total_scenarios": 11,
    "passed_scenarios": 11,
    "pass_rate_percentage": 100.0,
    "average_quality_score": 0.91
  },
  "priority_breakdown": {
    "critical": {"total": 1, "passed": 1},
    "high": {"total": 4, "passed": 4},
    "medium": {"total": 4, "passed": 4},
    "low": {"total": 2, "passed": 2}
  }
}
```

### 报告文件 (Report Files)
- **JSON报告**: `e2e_test_report.json` - 详细的机器可读报告
- **控制台输出**: 实时测试进度和结果摘要

## 🛠 配置说明 (Configuration)

### 环境要求 (Environment Requirements)

```bash
# 安装测试依赖
pip install pytest pytest-html pytest-cov

# 设置环境变量
export VIDEOLINGO_TEST_MODE=true
export VIDEOLINGO_CONFIG_PATH=./config.yaml
```

### 测试配置 (Test Configuration)

```yaml
# pytest.ini 或 pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests/e2e"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "e2e: end-to-end tests",
    "critical: critical functionality tests",
    "slow: tests that take longer to run"
]
```

## 🔧 自定义测试 (Custom Tests)

### 添加新的E2E测试场景 (Adding New E2E Test Scenarios)

1. **创建测试文件**:
```python
# tests/e2e/test_custom_scenario.py
class TestCustomScenario:
    def test_custom_workflow(self):
        # 实现自定义测试逻辑
        pass
```

2. **更新测试套件**:
```python
# 在 test_e2e_suite_runner.py 中添加新场景
test_scenarios['custom_scenario'] = {
    'description': '自定义测试场景',
    'test_file': 'test_custom_scenario.py::TestCustomScenario::test_custom_workflow',
    'priority': 'medium',
    'estimated_duration': 30
}
```

### 模拟和Mock策略 (Mocking and Simulation Strategies)

```python
# 外部服务模拟
@patch('external_service.api_call')
def test_with_mocked_service(mock_api):
    mock_api.return_value = expected_response
    # 测试逻辑
    
# 文件系统模拟
@patch('os.path.exists')
@patch('shutil.copy2')
def test_file_operations(mock_copy, mock_exists):
    # 测试文件操作
```

## 📈 持续集成 (Continuous Integration)

### GitHub Actions配置 (GitHub Actions Configuration)

```yaml
# .github/workflows/e2e-tests.yml
name: E2E Tests
on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-html
      
      - name: Run E2E Tests
        run: python tests/e2e/test_e2e_suite_runner.py
      
      - name: Upload test reports
        uses: actions/upload-artifact@v3
        with:
          name: e2e-test-reports
          path: tests/e2e/e2e_test_report.json
```

## 🚨 故障排除 (Troubleshooting)

### 常见问题 (Common Issues)

1. **测试超时**:
```bash
# 增加pytest超时设置
pytest tests/e2e/ --timeout=300
```

2. **资源不足**:
```bash
# 顺序执行避免资源竞争
pytest tests/e2e/ -x --maxfail=1
```

3. **网络问题**:
```bash
# 使用离线模式
export VIDEOLINGO_OFFLINE_MODE=true
pytest tests/e2e/
```

### 调试技巧 (Debugging Tips)

1. **详细输出**:
```bash
pytest tests/e2e/ -v -s --tb=long
```

2. **保存中间文件**:
```python
# 在测试中设置
os.environ['VIDEOLINGO_KEEP_TEMP_FILES'] = 'true'
```

3. **分步调试**:
```bash
# 只运行特定测试
pytest tests/e2e/test_complete_video_pipeline.py::TestCompleteVideoPipeline::test_full_pipeline_with_dubbing -v -s
```

## 📚 相关文档 (Related Documentation)

- [VideoLingo架构文档](../../ARCHITECTURE.md)
- [单元测试文档](../unit/README.md)
- [集成测试文档](../integration/README.md)
- [配置指南](../../docs/configuration.md)
- [故障排除指南](../../docs/troubleshooting.md)

## 🤝 贡献指南 (Contributing)

欢迎为E2E测试套件做出贡献！请遵循以下指南：

1. **添加新测试**: 确保新测试覆盖真实的用户场景
2. **保持独立性**: 每个测试应该独立运行，不依赖其他测试的状态
3. **性能考虑**: 合理设置测试超时和资源使用
4. **文档更新**: 添加新测试时同步更新文档

---

**版本**: 1.0.0  
**维护者**: VideoLingo团队  
**最后更新**: 2024-01-15