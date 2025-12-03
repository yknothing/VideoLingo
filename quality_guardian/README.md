# Quality Guardian Agent - 质量管理专家Agent

Claude Code全局质量管理系统，专注于文档、代码、测试的分离管理和质量保证。

## 🎯 核心特性

- **🔍 智能质量监控**: 实时监控所有代码、文档、测试的修改行为
- **📊 合规性审计**: 确保变更符合质量标准和审计要求  
- **📝 独立记录存储**: 质量数据完全独立于源代码，满足审计要求
- **⚖️ 风险评估分级**: 智能评估变更风险，提供分级处理建议
- **🤖 Agent协作**: 与其他Claude Code Agent无缝协作

## 🏗️ 架构设计

### 质量管理核心原则

```yaml
分离原则:
  - 文档与代码完全分离：文档变更不影响代码逻辑
  - 代码与测试严格分离：测试代码独立维护和版本控制
  - 过程数据独立存储：质量记录不污染源代码和文档
  - 高内聚低耦合：每个质量域独立管理
  - 审计合规：所有变更可追溯，满足审计要求
```

### 数据存储架构

```
.quality/                      # 质量数据根目录（独立于源码）
├── audit_logs/               # 审计日志（不可变）
│   ├── 2025-01/
│   │   ├── changes.jsonl     # 变更记录
│   │   └── metrics.jsonl     # 质量指标
├── reports/                  # 质量报告
├── config/                   # 质量标准配置
│   └── quality_standards.yaml
├── metrics/                  # 历史质量数据
└── compliance_reports/       # 合规报告
```

## 🚀 快速开始

### 安装依赖

```bash
cd VideoLingo
pip install pyyaml
```

### 基本使用

```python
from quality_guardian import QualityGuardian

# 初始化质量守护者
guardian = QualityGuardian()

# 监控文件变更
change_record = guardian.monitor_file_change("core/config.py")

# 生成质量报告
report = guardian.generate_quality_report(30)  # 30天趋势分析
```

### CLI使用

```bash
# 全面项目审计
python quality_guardian/cli.py audit --scope=project --depth=comprehensive

# 检查特定模块质量
python quality_guardian/cli.py check --module=core/utils/config.py

# 合规性检查
python quality_guardian/cli.py compliance --standard=enterprise

# 质量趋势分析
python quality_guardian/cli.py trend --period=30days --metrics=coverage,performance
```

## 📊 质量标准

### 覆盖率要求

| 模块类型 | 最低要求 | 目标值 | 优秀水平 |
|---------|----------|--------|----------|
| 单元测试 | 70% | 80% | 90%+ |
| 集成测试 | 60% | 70% | 80%+ |
| 端到端测试 | 50% | 60% | 70%+ |

### 风险级别定义

- **🟢 LOW**: 常规变更，自动通过
- **🟡 MEDIUM**: 中等风险，记录警告
- **🟠 HIGH**: 高风险变更，需要审查
- **🔴 CRITICAL**: 严重风险，阻止提交

### 质量检查清单

```yaml
代码质量:
  - [ ] 圈复杂度 ≤ 10
  - [ ] 函数长度 ≤ 50行
  - [ ] 类长度 ≤ 500行
  - [ ] 重复代码率 ≤ 5%

测试质量:
  - [ ] 测试命名规范
  - [ ] 断言充分有效
  - [ ] Mock使用合理
  - [ ] 执行时间 ≤ 1秒/测试

文档质量:
  - [ ] API文档完整
  - [ ] 配置说明齐全
  - [ ] 示例代码可运行
  - [ ] 版本信息准确
```

## 🔧 配置说明

### 质量标准配置文件

编辑 `.quality/config/quality_standards.yaml`:

```yaml
# 覆盖率要求
coverage_requirements:
  unit_tests:
    minimum: 70
    target: 80
    excellent: 90

# 风险评估
risk_assessment:
  high_risk_changes:
    coverage_drop: 10
    core_modules: 
      - "core/"
      - "src/main/"
    test_failure_rate: 5

# 告警配置  
alerting_config:
  critical_alerts:
    coverage_drop_threshold: 15
    test_failure_spike: 10
```

### VideoLingo项目特定配置

```yaml
project_specific:
  videolingo:
    core_modules:
      - "core/_1_ytdlp.py"
      - "core/_2_asr.py"
      - "core/utils/config_utils.py"
    critical_tests:
      - "tests/unit/test_config_utils.py"
      - "tests/integration/test_pipeline_flow.py"
```

## 🤖 Agent协作

### 与其他Agent集成

Quality Guardian与以下Agent协作：

- **Code Review Agent**: 代码审查时进行质量检查
- **Test Agent**: 集成测试覆盖率和结果数据
- **Documentation Agent**: API变更时同步文档检查
- **Performance Agent**: 性能回归检测

### 协作协议

```python
# 协作决策权限矩阵
权限分配:
  Quality Guardian (最终决策):
    - 阻止高风险变更提交
    - 要求强制质量审查
    - 设定质量标准和阈值
  
  其他Agent (建议权):
    - 提供专业领域建议
    - 执行具体质量检查
    - 生成专项报告
```

## 📈 监控与报告

### 质量指标监控

```python
# 实时质量监控
from quality_guardian.examples.integration_example import ProjectQualityManager

manager = ProjectQualityManager(project_path)

# 启动持续监控
manager.start_continuous_monitoring()

# 处理文件变更事件
manager.handle_file_change("core/config.py", "modified")
```

### 报告类型

1. **质量审计报告**: 全面的项目质量分析
2. **合规性报告**: 符合企业审计要求
3. **趋势分析报告**: 质量指标历史趋势
4. **风险评估报告**: 变更风险分析和建议

### 报告示例

```
📊 Quality Report - 2025-01-12

Overall Quality Score: 78.2/100
Code Coverage: 75.5%
Test Pass Rate: 96.8%
Build Success Rate: 94.7%

🚨 Issues Found:
- Coverage below minimum: 75.5%
- 3 high-risk changes in last 7 days

💡 Recommendations:
- 提高测试覆盖率：增加单元测试和集成测试
- 修复失败的测试用例，提高测试稳定性
```

## 🛡️ 安全与合规

### 审计特性

- **不可变记录**: 所有质量记录写入后不可修改
- **完整追踪**: 100%变更可追溯
- **独立存储**: 质量数据与源码完全分离
- **权限控制**: 基于角色的访问控制

### 合规标准

支持多种合规标准：

- **Enterprise**: 企业级合规要求
- **Basic**: 基础合规检查
- **Strict**: 严格合规标准

## 🔍 故障排除

### 常见问题

**Q: 质量数据存储在哪里？**
A: 所有质量数据存储在项目根目录的`.quality/`文件夹中，与源代码完全分离。

**Q: 如何调整质量标准？**
A: 编辑`.quality/config/quality_standards.yaml`文件，或使用CLI命令动态调整。

**Q: Agent如何处理高风险变更？**
A: 高风险变更会触发警告，记录详细信息，并可配置为阻止提交直到审查完成。

**Q: 如何与CI/CD集成？**
A: 在CI/CD pipeline中调用CLI命令进行质量检查，不通过则中断构建流程。

### 日志级别

```python
import logging

# 设置日志级别
logging.getLogger('QualityGuardian').setLevel(logging.DEBUG)
```

## 🤝 贡献指南

### 开发环境

1. 克隆项目
2. 安装依赖：`pip install -r requirements.txt`
3. 运行示例：`python quality_guardian/examples/integration_example.py`

### 测试

```bash
# 运行集成示例
python quality_guardian/examples/integration_example.py

# 测试CLI功能
python quality_guardian/cli.py audit --scope=project
```

## 📄 许可证

本项目采用与VideoLingo相同的许可证。

## 🆕 版本历史

### v1.0.0 (2025-01-12)
- ✨ 初始发布
- 🔍 基础质量监控功能
- 📊 质量报告生成
- 🤖 Agent协作机制
- 📋 合规性检查
- 🖥️ CLI工具支持

---

**Quality Guardian** - 让代码质量管理变得简单、可靠、可审计。

如需帮助，请查看 `examples/integration_example.py` 中的详细示例。