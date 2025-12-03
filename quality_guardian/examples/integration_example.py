#!/usr/bin/env python3
"""
Quality Guardian Integration Example
质量管理专家Agent集成示例

展示如何在实际项目中集成和使用Quality Guardian
"""

import os
import sys
from pathlib import Path
import time
import threading
from typing import Dict, Any

# 添加项目路径到系统路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from quality_guardian import QualityGuardian, RiskLevel


class ProjectQualityManager:
    """项目质量管理器

    演示如何在实际项目中集成Quality Guardian
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.guardian = QualityGuardian(project_path)
        self.monitoring_active = False
        self.monitoring_thread = None

    def start_continuous_monitoring(self):
        """启动持续质量监控"""
        if self.monitoring_active:
            print("⚠️ Quality monitoring is already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        print("🔍 Quality monitoring started")

    def stop_continuous_monitoring(self):
        """停止质量监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("⏹️ Quality monitoring stopped")

    def _monitoring_loop(self):
        """监控循环"""
        last_check_time = time.time()

        while self.monitoring_active:
            try:
                current_time = time.time()

                # 每5分钟进行一次质量检查
                if current_time - last_check_time > 300:  # 5分钟
                    self._periodic_quality_check()
                    last_check_time = current_time

                time.sleep(10)  # 每10秒检查一次

            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(30)  # 出错后等待30秒再重试

    def _periodic_quality_check(self):
        """定期质量检查"""
        print("📊 Performing periodic quality check...")

        try:
            report = self.guardian.generate_quality_report(7)  # 7天趋势

            if report:
                # 检查质量趋势
                self._analyze_quality_trends(report)

                # 检查合规状态
                self._check_compliance_alerts(report)

        except Exception as e:
            print(f"❌ Periodic quality check failed: {e}")

    def _analyze_quality_trends(self, report: Dict[str, Any]):
        """分析质量趋势"""
        current_metrics = report.get("current_metrics", {})
        trend_analysis = report.get("trend_analysis", {})

        if not current_metrics:
            return

        # 检查覆盖率趋势
        coverage_trend = trend_analysis.get("coverage_trend", 0)
        current_coverage = current_metrics.get("coverage_percentage", 0)

        if coverage_trend < -5:  # 下降超过5%
            print(
                f"⚠️ Coverage trending down: {current_coverage:.1f}% (trend: {coverage_trend:.1f})"
            )
        elif coverage_trend > 2:  # 上升超过2%
            print(f"✅ Coverage improving: {current_coverage:.1f}% (trend: +{coverage_trend:.1f})")

        # 检查质量分数趋势
        quality_trend = trend_analysis.get("quality_trend", 0)
        current_quality = current_metrics.get("overall_quality_score", 0)

        if quality_trend < -5:
            print(f"⚠️ Quality score declining: {current_quality:.1f} (trend: {quality_trend:.1f})")

    def _check_compliance_alerts(self, report: Dict[str, Any]):
        """检查合规性告警"""
        compliance_status = report.get("compliance_status", {})

        if compliance_status.get("status") == "non_compliant":
            issues = compliance_status.get("issues", [])
            print(f"🚨 COMPLIANCE ALERT: {len(issues)} issues found")
            for issue in issues:
                print(f"  - {issue}")

    def handle_file_change(self, file_path: str, change_type: str = "modified"):
        """处理文件变更事件

        这个方法可以被文件监控系统调用（如watchdog）
        """
        try:
            change_record = self.guardian.monitor_file_change(file_path, change_type)

            if change_record:
                self._handle_change_record(change_record)
                return change_record
            else:
                print(f"❌ Failed to process file change: {file_path}")
                return None

        except Exception as e:
            print(f"❌ Error handling file change {file_path}: {e}")
            return None

    def _handle_change_record(self, change_record):
        """处理变更记录"""
        risk_icons = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MEDIUM: "🟡",
            RiskLevel.HIGH: "🟠",
            RiskLevel.CRITICAL: "🔴",
        }

        risk_icon = risk_icons.get(change_record.risk_level, "⚪")

        print(f"{risk_icon} File changed: {change_record.files_modified[0]}")
        print(f"   Risk: {change_record.risk_level.value}")
        print(f"   Type: {change_record.change_type.value}")

        # 高风险变更需要特殊处理
        if change_record.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            self._handle_high_risk_change(change_record)

    def _handle_high_risk_change(self, change_record):
        """处理高风险变更"""
        print("🚨 HIGH RISK CHANGE DETECTED!")
        print("Recommended actions:")

        impact = change_record.quality_impact

        if impact.get("core_module_affected"):
            print("  - 🔍 Review core module changes carefully")
            print("  - 🧪 Run comprehensive test suite")

        if impact.get("requires_testing"):
            print("  - ✅ Ensure adequate test coverage")
            print("  - 🔄 Consider integration testing")

        if change_record.risk_level == RiskLevel.CRITICAL:
            print("  - ⚠️ Consider blocking deployment until review")

    def run_quality_audit(self, comprehensive: bool = True):
        """运行质量审计"""
        print("🔍 Starting quality audit...")

        try:
            # 生成详细报告
            report = self.guardian.generate_quality_report(30)

            if not report:
                print("❌ Failed to generate quality report")
                return

            # 显示当前状态
            self._display_quality_status(report)

            # 如果是全面审计，进行深度分析
            if comprehensive:
                self._comprehensive_audit()

        except Exception as e:
            print(f"❌ Quality audit failed: {e}")

    def _display_quality_status(self, report: Dict[str, Any]):
        """显示质量状态"""
        current_metrics = report.get("current_metrics", {})

        if not current_metrics:
            print("❌ No quality metrics available")
            return

        print("\n📊 Current Quality Status:")
        print(f"  Overall Score: {current_metrics.get('overall_quality_score', 0):.1f}/100")
        print(f"  Code Coverage: {current_metrics.get('coverage_percentage', 0):.1f}%")
        print(f"  Test Pass Rate: {current_metrics.get('test_pass_rate', 0):.1f}%")
        print(f"  Build Success: {current_metrics.get('build_success_rate', 0):.1f}%")
        print(f"  Code Complexity: {current_metrics.get('code_complexity', 0):.1f}")
        print(f"  Documentation: {current_metrics.get('documentation_completeness', 0):.1f}%")

        # 显示合规状态
        compliance = report.get("compliance_status", {})
        status = compliance.get("status", "unknown")

        status_icons = {"compliant": "✅", "warning": "⚠️", "non_compliant": "❌", "unknown": "❓"}

        print(f"\n{status_icons.get(status, '❓')} Compliance Status: {status}")

        # 显示问题
        if compliance.get("issues"):
            print("  Issues:")
            for issue in compliance["issues"]:
                print(f"    - {issue}")

        # 显示建议
        recommendations = report.get("recommendations", [])
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")

    def _comprehensive_audit(self):
        """全面审计"""
        print("\n🔬 Performing comprehensive analysis...")

        # 这里可以集成更多分析工具
        # 例如：静态代码分析、安全扫描、性能分析等

        analysis_results = {
            "code_quality": self._analyze_code_quality(),
            "security_scan": self._security_scan(),
            "performance_analysis": self._performance_analysis(),
            "dependency_audit": self._dependency_audit(),
        }

        for analysis_type, results in analysis_results.items():
            print(f"\n📋 {analysis_type.replace('_', ' ').title()}:")
            if isinstance(results, dict):
                for key, value in results.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {results}")

    def _analyze_code_quality(self) -> Dict[str, Any]:
        """分析代码质量（示例实现）"""
        return {
            "complexity_score": "Good (average: 6.2)",
            "maintainability": "High",
            "code_smells": "3 detected",
            "duplication": "2.1%",
        }

    def _security_scan(self) -> Dict[str, Any]:
        """安全扫描（示例实现）"""
        return {
            "vulnerabilities": "None critical",
            "secrets_detected": "0",
            "security_score": "85/100",
        }

    def _performance_analysis(self) -> Dict[str, Any]:
        """性能分析（示例实现）"""
        return {
            "test_performance": "Average 0.8s per test",
            "build_time": "2m 34s",
            "memory_usage": "Normal",
        }

    def _dependency_audit(self) -> Dict[str, Any]:
        """依赖审计（示例实现）"""
        return {
            "outdated_packages": "8",
            "security_updates": "2 recommended",
            "license_compliance": "OK",
        }


def demonstrate_integration():
    """演示集成使用"""
    print("🚀 Quality Guardian Integration Demo")
    print("=" * 50)

    # 使用当前项目作为示例
    project_path = Path.cwd()
    manager = ProjectQualityManager(project_path)

    print(f"📁 Project: {project_path}")

    # 1. 运行质量审计
    print("\n1. Running Quality Audit...")
    manager.run_quality_audit(comprehensive=True)

    # 2. 演示文件变更监控
    print("\n2. Simulating File Changes...")

    # 模拟不同类型的文件变更
    test_changes = [
        ("core/utils/config.py", "modified"),
        ("tests/test_new_feature.py", "created"),
        ("README.md", "modified"),
        ("config.yaml", "modified"),
    ]

    for file_path, change_type in test_changes:
        print(f"\n📝 Simulating {change_type}: {file_path}")
        change_record = manager.handle_file_change(file_path, change_type)

        if change_record:
            time.sleep(1)  # 模拟时间间隔

    # 3. 启动持续监控（演示）
    print("\n3. Starting Continuous Monitoring...")
    manager.start_continuous_monitoring()

    print("   Monitoring for 10 seconds...")
    time.sleep(10)

    manager.stop_continuous_monitoring()

    print("\n✅ Integration demo completed!")


def demonstrate_cli_usage():
    """演示CLI用法"""
    print("\n🖥️ CLI Usage Examples:")
    print("=" * 30)

    cli_examples = [
        "# 全面项目审计",
        "python quality_guardian/cli.py audit --scope=project --depth=comprehensive",
        "",
        "# 检查特定模块",
        "python quality_guardian/cli.py check --module=core/utils/config.py --type=code-quality",
        "",
        "# 合规性检查",
        "python quality_guardian/cli.py compliance --standard=enterprise --output=report",
        "",
        "# 趋势分析",
        "python quality_guardian/cli.py trend --period=30days --metrics=coverage,performance",
        "",
        "# 监控文件变更",
        "python quality_guardian/cli.py monitor --file=core/test.py",
    ]

    for example in cli_examples:
        if example.startswith("#"):
            print(f"\n💡 {example}")
        elif example:
            print(f"   {example}")
        else:
            print()


if __name__ == "__main__":
    # 运行演示
    try:
        demonstrate_integration()
        demonstrate_cli_usage()

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
