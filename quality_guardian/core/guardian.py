#!/usr/bin/env python3
"""
Quality Guardian Agent - 核心实现
质量管理专家Agent的主要实现文件
"""

import json
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import subprocess
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QualityGuardian")


class RiskLevel(Enum):
    """风险等级枚举"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ChangeType(Enum):
    """变更类型枚举"""

    CODE = "code"
    TEST = "test"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    BUILD = "build"


@dataclass
class QualityMetrics:
    """质量指标数据结构"""

    timestamp: str
    coverage_percentage: float
    test_pass_rate: float
    code_complexity: float
    documentation_completeness: float
    build_success_rate: float
    overall_quality_score: float


@dataclass
class ChangeRecord:
    """变更记录数据结构"""

    id: str
    timestamp: str
    change_type: ChangeType
    files_modified: List[str]
    author: str
    commit_hash: str
    risk_level: RiskLevel
    quality_impact: Dict[str, Any]
    metadata: Dict[str, Any]


class QualityGuardian:
    """质量管理专家Agent"""

    def __init__(self, project_root: Path = None):
        """初始化质量守护者"""
        self.project_root = project_root or Path.cwd()
        self.quality_root = self.project_root / ".quality"
        self.config_path = self.quality_root / "config" / "quality_standards.yaml"

        # 创建质量数据存储结构
        self._setup_quality_storage()

        # 加载质量标准配置
        self.quality_standards = self._load_quality_standards()

        # 初始化组件
        self.metrics_collector = QualityMetricsCollector(self.quality_root)
        self.risk_assessor = RiskAssessment(self.quality_standards)
        self.audit_logger = AuditLogger(self.quality_root)

    def _setup_quality_storage(self):
        """设置质量数据存储结构"""
        directories = [
            "audit_logs",
            "metrics",
            "reports",
            "config",
            "risk_assessments",
            "compliance_reports",
        ]

        for directory in directories:
            (self.quality_root / directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"Quality storage initialized at: {self.quality_root}")

    def _load_quality_standards(self) -> Dict[str, Any]:
        """加载质量标准配置"""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            # 创建默认质量标准
            default_standards = self._create_default_quality_standards()
            self._save_quality_standards(default_standards)
            return default_standards

    def _create_default_quality_standards(self) -> Dict[str, Any]:
        """创建默认质量标准"""
        return {
            "coverage_requirements": {
                "unit_tests": {"minimum": 70, "target": 80, "excellent": 90},
                "integration_tests": {"minimum": 60, "target": 70, "excellent": 80},
                "e2e_tests": {"minimum": 50, "target": 60, "excellent": 70},
            },
            "complexity_limits": {
                "cyclomatic_complexity": {"max": 10, "warning": 8},
                "function_length": {"max": 50, "warning": 30},
                "class_length": {"max": 500, "warning": 300},
            },
            "quality_thresholds": {
                "overall_quality_score": {"minimum": 60, "target": 75, "excellent": 85},
                "test_pass_rate": {"minimum": 95, "target": 98, "excellent": 99},
                "build_success_rate": {"minimum": 90, "target": 95, "excellent": 98},
            },
            "risk_assessment": {
                "high_risk_changes": {
                    "coverage_drop": 10,  # 覆盖率下降10%以上
                    "core_modules": ["core/", "src/main/"],
                    "test_failure_rate": 5,  # 测试失败率5%以上
                },
                "medium_risk_changes": {"coverage_drop": 5, "test_failure_rate": 2},
            },
        }

    def _save_quality_standards(self, standards: Dict[str, Any]):
        """保存质量标准配置"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(standards, f, default_flow_style=False, allow_unicode=True)

    def monitor_file_change(
        self, file_path: str, change_type: str = "modified"
    ) -> Optional[ChangeRecord]:
        """监控文件变更"""
        try:
            file_path_obj = Path(file_path)

            # 确定变更类型
            detected_change_type = self._detect_change_type(file_path_obj)

            # 收集变更信息
            change_info = self._collect_change_info(file_path_obj, change_type)

            # 评估风险
            risk_level, quality_impact = self.risk_assessor.assess_change_risk(
                file_path_obj, detected_change_type, change_info
            )

            # 创建变更记录
            change_record = ChangeRecord(
                id=self._generate_change_id(file_path, change_info),
                timestamp=datetime.now(timezone.utc).isoformat(),
                change_type=detected_change_type,
                files_modified=[file_path],
                author=change_info.get("author", "unknown"),
                commit_hash=change_info.get("commit_hash", ""),
                risk_level=risk_level,
                quality_impact=quality_impact,
                metadata=change_info,
            )

            # 记录到审计日志
            self.audit_logger.log_change(change_record)

            # 根据风险级别采取行动
            self._handle_risk_level(change_record)

            logger.info(f"File change monitored: {file_path} (Risk: {risk_level.value})")
            return change_record

        except Exception as e:
            logger.error(f"Error monitoring file change {file_path}: {e}")
            return None

    def _detect_change_type(self, file_path: Path) -> ChangeType:
        """检测变更类型"""
        file_name = file_path.name.lower()
        file_suffix = file_path.suffix.lower()

        # 测试文件
        if (
            file_name.startswith("test_")
            or file_name.endswith("_test.py")
            or "test" in file_path.parts
        ):
            return ChangeType.TEST

        # 文档文件
        if file_suffix in [".md", ".rst", ".txt"]:
            return ChangeType.DOCUMENTATION

        # 配置文件
        if file_suffix in [".yaml", ".yml", ".json", ".ini", ".toml", ".cfg"]:
            return ChangeType.CONFIGURATION

        # 构建文件
        if file_name in ["makefile", "dockerfile", "requirements.txt"] or file_suffix in [
            ".sh",
            ".bat",
        ]:
            return ChangeType.BUILD

        # 代码文件
        if file_suffix in [".py", ".js", ".ts", ".java", ".go", ".cpp", ".c"]:
            return ChangeType.CODE

        # 默认为代码类型
        return ChangeType.CODE

    def _collect_change_info(self, file_path: Path, change_type: str) -> Dict[str, Any]:
        """收集变更信息"""
        info = {
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "change_type": change_type,
            "file_extension": file_path.suffix,
        }

        # 尝试获取Git信息
        try:
            # 获取最新提交信息
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H|%an|%ae|%s", "--", str(file_path)],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0 and result.stdout.strip():
                commit_info = result.stdout.strip().split("|", 3)
                if len(commit_info) == 4:
                    info.update(
                        {
                            "commit_hash": commit_info[0],
                            "author": commit_info[1],
                            "author_email": commit_info[2],
                            "commit_message": commit_info[3],
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to get git info for {file_path}: {e}")

        return info

    def _generate_change_id(self, file_path: str, change_info: Dict) -> str:
        """生成变更ID"""
        content = f"{file_path}_{change_info.get('commit_hash', '')}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _handle_risk_level(self, change_record: ChangeRecord):
        """根据风险级别处理"""
        if change_record.risk_level == RiskLevel.CRITICAL:
            logger.critical(f"CRITICAL risk change detected: {change_record.files_modified}")
            self._send_critical_alert(change_record)
        elif change_record.risk_level == RiskLevel.HIGH:
            logger.warning(f"HIGH risk change detected: {change_record.files_modified}")
            self._send_warning_alert(change_record)
        elif change_record.risk_level == RiskLevel.MEDIUM:
            logger.info(f"MEDIUM risk change detected: {change_record.files_modified}")

    def _send_critical_alert(self, change_record: ChangeRecord):
        """发送严重警告"""
        # 这里可以集成实际的告警系统（邮件、Slack等）
        alert_message = f"""
        🚨 CRITICAL Quality Alert
        
        Change ID: {change_record.id}
        Files: {', '.join(change_record.files_modified)}
        Risk Level: {change_record.risk_level.value}
        Author: {change_record.author}
        
        Quality Impact: {json.dumps(change_record.quality_impact, indent=2)}
        
        Immediate action required!
        """
        logger.critical(alert_message)

    def _send_warning_alert(self, change_record: ChangeRecord):
        """发送警告"""
        alert_message = f"""
        ⚠️ Quality Warning
        
        Change ID: {change_record.id}
        Files: {', '.join(change_record.files_modified)}
        Risk Level: {change_record.risk_level.value}
        Author: {change_record.author}
        
        Please review the changes carefully.
        """
        logger.warning(alert_message)

    def generate_quality_report(self, period_days: int = 30) -> Dict[str, Any]:
        """生成质量报告"""
        try:
            # 收集指标数据
            current_metrics = self.metrics_collector.collect_current_metrics()

            # 分析趋势
            trend_analysis = self.metrics_collector.analyze_trends(period_days)

            # 风险评估汇总
            risk_summary = self.audit_logger.get_risk_summary(period_days)

            # 合规状态检查
            compliance_status = self._check_compliance_status(current_metrics)

            report = {
                "report_date": datetime.now(timezone.utc).isoformat(),
                "period_days": period_days,
                "current_metrics": asdict(current_metrics) if current_metrics else {},
                "trend_analysis": trend_analysis,
                "risk_summary": risk_summary,
                "compliance_status": compliance_status,
                "recommendations": self._generate_recommendations(current_metrics, trend_analysis),
            }

            # 保存报告
            report_path = (
                self.quality_root
                / "reports"
                / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"Quality report generated: {report_path}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return {}

    def _check_compliance_status(self, metrics: Optional[QualityMetrics]) -> Dict[str, Any]:
        """检查合规状态"""
        if not metrics:
            return {"status": "unknown", "issues": ["No metrics available"]}

        issues = []
        warnings = []

        # 检查覆盖率要求
        if (
            metrics.coverage_percentage
            < self.quality_standards["coverage_requirements"]["unit_tests"]["minimum"]
        ):
            issues.append(f"Coverage below minimum: {metrics.coverage_percentage:.1f}%")

        # 检查测试通过率
        if (
            metrics.test_pass_rate
            < self.quality_standards["quality_thresholds"]["test_pass_rate"]["minimum"]
        ):
            issues.append(f"Test pass rate below minimum: {metrics.test_pass_rate:.1f}%")

        # 检查整体质量分数
        if (
            metrics.overall_quality_score
            < self.quality_standards["quality_thresholds"]["overall_quality_score"]["minimum"]
        ):
            issues.append(
                f"Overall quality score below minimum: {metrics.overall_quality_score:.1f}"
            )

        if issues:
            status = "non_compliant"
        elif warnings:
            status = "warning"
        else:
            status = "compliant"

        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "last_checked": datetime.now(timezone.utc).isoformat(),
        }

    def _generate_recommendations(
        self, metrics: Optional[QualityMetrics], trends: Dict
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if not metrics:
            return ["无法生成建议：缺少质量指标数据"]

        # 基于当前指标的建议
        if metrics.coverage_percentage < 70:
            recommendations.append("提高测试覆盖率：增加单元测试和集成测试")

        if metrics.test_pass_rate < 95:
            recommendations.append("修复失败的测试用例，提高测试稳定性")

        if metrics.code_complexity > 8:
            recommendations.append("重构复杂代码，降低圈复杂度")

        if metrics.documentation_completeness < 80:
            recommendations.append("完善文档：更新API文档和用户指南")

        # 基于趋势的建议
        if trends.get("coverage_trend", 0) < 0:
            recommendations.append("覆盖率呈下降趋势，需要加强测试")

        if trends.get("quality_trend", 0) < 0:
            recommendations.append("质量分数下降，建议进行全面质量审查")

        return recommendations if recommendations else ["当前质量状况良好，请保持"]


class QualityMetricsCollector:
    """质量指标收集器"""

    def __init__(self, quality_root: Path):
        self.quality_root = quality_root
        self.metrics_path = quality_root / "metrics"

    def collect_current_metrics(self) -> Optional[QualityMetrics]:
        """收集当前质量指标"""
        try:
            # 这里需要实际实现指标收集逻辑
            # 目前返回模拟数据
            return QualityMetrics(
                timestamp=datetime.now(timezone.utc).isoformat(),
                coverage_percentage=75.5,
                test_pass_rate=96.8,
                code_complexity=6.2,
                documentation_completeness=82.1,
                build_success_rate=94.7,
                overall_quality_score=78.2,
            )
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return None

    def analyze_trends(self, period_days: int) -> Dict[str, float]:
        """分析质量趋势"""
        # 实际实现需要从历史数据分析趋势
        return {
            "coverage_trend": 2.1,  # 正值表示上升
            "quality_trend": 1.8,
            "performance_trend": -0.5,  # 负值表示下降
        }


class RiskAssessment:
    """风险评估器"""

    def __init__(self, quality_standards: Dict[str, Any]):
        self.quality_standards = quality_standards

    def assess_change_risk(
        self, file_path: Path, change_type: ChangeType, change_info: Dict[str, Any]
    ) -> Tuple[RiskLevel, Dict[str, Any]]:
        """评估变更风险"""
        risk_factors = []
        quality_impact = {}

        # 基于文件路径评估风险
        if any(
            core_path in str(file_path)
            for core_path in self.quality_standards["risk_assessment"]["high_risk_changes"][
                "core_modules"
            ]
        ):
            risk_factors.append("core_module_change")
            quality_impact["core_module_affected"] = True

        # 基于变更类型评估风险
        if change_type == ChangeType.CODE:
            risk_factors.append("code_change")
            quality_impact["requires_testing"] = True
        elif change_type == ChangeType.CONFIGURATION:
            risk_factors.append("config_change")
            quality_impact["requires_validation"] = True

        # 基于文件大小评估风险（大文件变更风险更高）
        file_size = change_info.get("file_size", 0)
        if file_size > 10000:  # 10KB以上
            risk_factors.append("large_file_change")
            quality_impact["large_change"] = True

        # 计算最终风险级别
        risk_level = self._calculate_risk_level(risk_factors)

        quality_impact.update(
            {
                "risk_factors": risk_factors,
                "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return risk_level, quality_impact

    def _calculate_risk_level(self, risk_factors: List[str]) -> RiskLevel:
        """计算风险级别"""
        high_risk_factors = ["core_module_change", "large_file_change"]
        medium_risk_factors = ["code_change", "config_change"]

        if any(factor in risk_factors for factor in high_risk_factors):
            if len(risk_factors) >= 3:
                return RiskLevel.CRITICAL
            else:
                return RiskLevel.HIGH
        elif any(factor in risk_factors for factor in medium_risk_factors):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, quality_root: Path):
        self.quality_root = quality_root
        self.audit_path = quality_root / "audit_logs"

    def log_change(self, change_record: ChangeRecord):
        """记录变更到审计日志"""
        try:
            # 按月分组存储
            month_dir = self.audit_path / datetime.now().strftime("%Y-%m")
            month_dir.mkdir(exist_ok=True)

            # 写入审计日志
            log_file = month_dir / "changes.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                # 转换枚举为字符串以支持JSON序列化
                record_dict = asdict(change_record)
                record_dict["change_type"] = change_record.change_type.value
                record_dict["risk_level"] = change_record.risk_level.value
                json.dump(record_dict, f, ensure_ascii=False)
                f.write("\n")

        except Exception as e:
            logger.error(f"Failed to log change: {e}")

    def get_risk_summary(self, period_days: int) -> Dict[str, Any]:
        """获取风险汇总"""
        # 实际实现需要从审计日志分析风险统计
        return {
            "total_changes": 156,
            "high_risk_changes": 8,
            "medium_risk_changes": 23,
            "low_risk_changes": 125,
            "risk_distribution": {"critical": 0, "high": 8, "medium": 23, "low": 125},
        }


if __name__ == "__main__":
    # 测试质量守护者
    guardian = QualityGuardian()

    # 模拟文件变更监控
    test_file = "core/test_example.py"
    change_record = guardian.monitor_file_change(test_file)

    if change_record:
        print(f"Change monitored: {change_record.id}")
        print(f"Risk level: {change_record.risk_level.value}")

    # 生成质量报告
    report = guardian.generate_quality_report(30)
    print(f"Quality report generated with {len(report)} sections")
