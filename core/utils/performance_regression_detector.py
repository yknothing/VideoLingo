"""
Performance Regression Detection and Alerting System

Provides automated detection of performance regressions with:
1. Statistical analysis of performance trends
2. Automated baseline updates
3. Real-time alerting for performance degradations
4. Performance health scoring
5. Predictive analysis for bottleneck identification

Integrates with VideoLingo's performance monitoring system.
"""

import json
import time
import statistics
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum

from core.utils.observability import log_event, inc_counter, observe_histogram


class RegressionSeverity(Enum):
    """Severity levels for performance regressions"""
    LOW = "low"              # 10-20% degradation
    MEDIUM = "medium"        # 20-50% degradation  
    HIGH = "high"            # 50-100% degradation
    CRITICAL = "critical"    # >100% degradation or system failure


class RegressionType(Enum):
    """Types of performance regressions"""
    DURATION = "duration"
    MEMORY = "memory"
    CPU = "cpu"
    API_LATENCY = "api_latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class PerformanceBaseline:
    """Performance baseline with statistical confidence intervals"""
    metric_name: str
    stage_name: str
    baseline_value: float
    p50_value: float
    p95_value: float
    p99_value: float
    std_deviation: float
    sample_count: int
    last_updated: str
    confidence_interval: Tuple[float, float]
    seasonal_adjustments: Dict[str, float] = None


@dataclass
class RegressionAlert:
    """Performance regression alert"""
    alert_id: str
    timestamp: str
    stage_name: str
    metric_name: str
    regression_type: RegressionType
    severity: RegressionSeverity
    current_value: float
    baseline_value: float
    degradation_percent: float
    confidence_score: float
    video_id: Optional[str] = None
    context: Dict[str, Any] = None
    resolved: bool = False
    resolution_notes: str = ""


@dataclass
class PerformanceHealthScore:
    """Overall performance health assessment"""
    overall_score: float  # 0-100
    stage_scores: Dict[str, float]
    regression_count: int
    critical_issues: List[str]
    recommendations: List[str]
    trend_direction: str  # improving, stable, degrading


class PerformanceRegressionDetector:
    """Advanced regression detection with statistical analysis"""
    
    def __init__(self, baseline_dir: str = None):
        self.baseline_dir = Path(baseline_dir or "reports/performance")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.active_alerts: List[RegressionAlert] = []
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # Thresholds for different severity levels
        self.severity_thresholds = {
            RegressionSeverity.LOW: 10.0,      # 10% degradation
            RegressionSeverity.MEDIUM: 20.0,   # 20% degradation
            RegressionSeverity.HIGH: 50.0,     # 50% degradation
            RegressionSeverity.CRITICAL: 100.0 # 100% degradation
        }
        
        self.load_baselines()
        self.load_historical_data()
    
    def load_baselines(self):
        """Load performance baselines from storage"""
        baseline_file = self.baseline_dir / "performance_baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                    for key, baseline_data in data.items():
                        self.baselines[key] = PerformanceBaseline(**baseline_data)
            except Exception as e:
                log_event("warning", f"Failed to load baselines: {e}")
    
    def save_baselines(self):
        """Save current baselines to storage"""
        baseline_file = self.baseline_dir / "performance_baselines.json"
        baselines_dict = {
            key: asdict(baseline) 
            for key, baseline in self.baselines.items()
        }
        with open(baseline_file, 'w') as f:
            json.dump(baselines_dict, f, indent=2)
    
    def load_historical_data(self):
        """Load historical performance data for trend analysis"""
        history_file = self.baseline_dir / "stage_performance_summary.jsonl"
        if not history_file.exists():
            return
        
        try:
            with open(history_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        stage_name = record.get('stage_name', 'unknown')
                        if stage_name not in self.historical_data:
                            self.historical_data[stage_name] = []
                        self.historical_data[stage_name].append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log_event("warning", f"Failed to load historical data: {e}")
    
    def update_baseline(self, stage_name: str, metric_name: str, 
                       values: List[float], confidence_level: float = 0.95) -> PerformanceBaseline:
        """Update performance baseline with statistical analysis"""
        if not values or len(values) < 3:
            return None
        
        # Calculate statistical measures
        mean_value = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        # Calculate percentiles
        sorted_values = sorted(values)
        n = len(sorted_values)
        p50 = sorted_values[int(n * 0.5)]
        p95 = sorted_values[int(n * 0.95)]
        p99 = sorted_values[int(n * 0.99)] if n > 20 else p95
        
        # Calculate confidence interval
        if len(values) > 1:
            # Using t-distribution for confidence interval
            import math
            t_value = 2.0  # Approximation for 95% confidence
            margin_error = t_value * (std_dev / math.sqrt(len(values)))
            confidence_interval = (mean_value - margin_error, mean_value + margin_error)
        else:
            confidence_interval = (mean_value, mean_value)
        
        baseline_key = f"{stage_name}_{metric_name}"
        
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            stage_name=stage_name,
            baseline_value=mean_value,
            p50_value=p50,
            p95_value=p95,
            p99_value=p99,
            std_deviation=std_dev,
            sample_count=len(values),
            last_updated=datetime.now().isoformat(),
            confidence_interval=confidence_interval
        )
        
        self.baselines[baseline_key] = baseline
        self.save_baselines()
        
        log_event("info", f"Baseline updated for {stage_name}.{metric_name}",
                 stage=stage_name, metric=metric_name, 
                 baseline_value=mean_value, sample_count=len(values))
        
        return baseline
    
    def detect_regression(self, stage_name: str, metric_name: str, 
                         current_value: float, video_id: str = None) -> Optional[RegressionAlert]:
        """Detect performance regression using statistical analysis"""
        baseline_key = f"{stage_name}_{metric_name}"
        
        if baseline_key not in self.baselines:
            # No baseline yet - try to create one from historical data
            if stage_name in self.historical_data:
                historical_values = [
                    record.get(metric_name) 
                    for record in self.historical_data[stage_name][-50:]  # Last 50 records
                    if record.get(metric_name) is not None
                ]
                if len(historical_values) >= 3:
                    self.update_baseline(stage_name, metric_name, historical_values)
            
            if baseline_key not in self.baselines:
                return None
        
        baseline = self.baselines[baseline_key]
        
        # Determine regression type
        regression_type = self._classify_regression_type(metric_name)
        
        # Calculate degradation percentage
        if baseline.baseline_value == 0:
            degradation_percent = float('inf') if current_value > 0 else 0
        else:
            if regression_type in [RegressionType.DURATION, RegressionType.MEMORY, 
                                  RegressionType.CPU, RegressionType.API_LATENCY]:
                # Higher is worse
                degradation_percent = ((current_value - baseline.baseline_value) / baseline.baseline_value) * 100
            else:
                # Lower is worse (e.g., throughput)
                degradation_percent = ((baseline.baseline_value - current_value) / baseline.baseline_value) * 100
        
        # Determine severity
        severity = self._determine_severity(degradation_percent)
        
        # Calculate confidence score based on statistical significance
        confidence_score = self._calculate_confidence(current_value, baseline)
        
        # Only alert if degradation is significant and confidence is high
        if degradation_percent > self.severity_thresholds[RegressionSeverity.LOW] and confidence_score > 0.8:
            alert = RegressionAlert(
                alert_id=f"{stage_name}_{metric_name}_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                stage_name=stage_name,
                metric_name=metric_name,
                regression_type=regression_type,
                severity=severity,
                current_value=current_value,
                baseline_value=baseline.baseline_value,
                degradation_percent=degradation_percent,
                confidence_score=confidence_score,
                video_id=video_id
            )
            
            self.active_alerts.append(alert)
            self._save_alert(alert)
            self._emit_alert_metrics(alert)
            
            return alert
        
        return None
    
    def _classify_regression_type(self, metric_name: str) -> RegressionType:
        """Classify the type of regression based on metric name"""
        metric_lower = metric_name.lower()
        
        if 'duration' in metric_lower or 'time' in metric_lower:
            return RegressionType.DURATION
        elif 'memory' in metric_lower or 'mem' in metric_lower:
            return RegressionType.MEMORY
        elif 'cpu' in metric_lower:
            return RegressionType.CPU
        elif 'api' in metric_lower and ('latency' in metric_lower or 'duration' in metric_lower):
            return RegressionType.API_LATENCY
        elif 'error' in metric_lower or 'fail' in metric_lower:
            return RegressionType.ERROR_RATE
        elif 'throughput' in metric_lower or 'rate' in metric_lower:
            return RegressionType.THROUGHPUT
        else:
            return RegressionType.DURATION  # Default
    
    def _determine_severity(self, degradation_percent: float) -> RegressionSeverity:
        """Determine severity based on degradation percentage"""
        if degradation_percent >= self.severity_thresholds[RegressionSeverity.CRITICAL]:
            return RegressionSeverity.CRITICAL
        elif degradation_percent >= self.severity_thresholds[RegressionSeverity.HIGH]:
            return RegressionSeverity.HIGH
        elif degradation_percent >= self.severity_thresholds[RegressionSeverity.MEDIUM]:
            return RegressionSeverity.MEDIUM
        else:
            return RegressionSeverity.LOW
    
    def _calculate_confidence(self, current_value: float, baseline: PerformanceBaseline) -> float:
        """Calculate statistical confidence in the regression detection"""
        if baseline.std_deviation == 0:
            return 1.0 if current_value != baseline.baseline_value else 0.0
        
        # Calculate z-score
        z_score = abs(current_value - baseline.baseline_value) / baseline.std_deviation
        
        # Convert z-score to confidence (simplified)
        if z_score >= 2.58:  # 99% confidence
            return 0.99
        elif z_score >= 1.96:  # 95% confidence
            return 0.95
        elif z_score >= 1.65:  # 90% confidence
            return 0.90
        elif z_score >= 1.28:  # 80% confidence
            return 0.80
        else:
            return max(0.5, z_score / 1.28 * 0.5)  # Linear scale for lower confidence
    
    def _save_alert(self, alert: RegressionAlert):
        """Save alert to storage"""
        alerts_file = self.baseline_dir / "regression_alerts.jsonl"
        with open(alerts_file, 'a') as f:
            f.write(json.dumps(asdict(alert)) + '\n')
    
    def _emit_alert_metrics(self, alert: RegressionAlert):
        """Emit metrics and logs for alert"""
        log_event(
            "warning" if alert.severity in [RegressionSeverity.LOW, RegressionSeverity.MEDIUM] else "error",
            f"PERFORMANCE REGRESSION DETECTED: {alert.regression_type.value} in {alert.stage_name}.{alert.metric_name}: "
            f"{alert.current_value:.2f} vs baseline {alert.baseline_value:.2f} "
            f"({alert.degradation_percent:+.1f}% degradation, {alert.confidence_score:.1%} confidence)",
            stage=alert.stage_name,
            metric=alert.metric_name,
            severity=alert.severity.value,
            video_id=alert.video_id
        )
        
        inc_counter("performance_regression_detected",
                   stage=alert.stage_name,
                   metric=alert.metric_name,
                   severity=alert.severity.value,
                   regression_type=alert.regression_type.value)
        
        observe_histogram("regression_degradation_percent", alert.degradation_percent,
                         stage=alert.stage_name, metric=alert.metric_name)
    
    def analyze_performance_trends(self, stage_name: str, days: int = 7) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if stage_name not in self.historical_data:
            return {"error": "No historical data available"}
        
        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_timestamp = cutoff_time.timestamp()
        
        recent_data = [
            record for record in self.historical_data[stage_name]
            if record.get('start_time', 0) >= cutoff_timestamp
        ]
        
        if len(recent_data) < 2:
            return {"error": "Insufficient recent data"}
        
        # Analyze trends for key metrics
        metrics_to_analyze = ['duration_seconds', 'memory_peak_mb', 'cpu_avg_percent']
        trends = {}
        
        for metric in metrics_to_analyze:
            values = [record.get(metric) for record in recent_data if record.get(metric) is not None]
            
            if len(values) >= 3:
                # Simple linear trend analysis
                n = len(values)
                x_values = list(range(n))
                
                # Calculate correlation coefficient
                mean_x = statistics.mean(x_values)
                mean_y = statistics.mean(values)
                
                numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, values))
                denominator_x = sum((x - mean_x) ** 2 for x in x_values)
                denominator_y = sum((y - mean_y) ** 2 for y in values)
                
                if denominator_x > 0 and denominator_y > 0:
                    correlation = numerator / math.sqrt(denominator_x * denominator_y)
                    
                    # Interpret trend
                    if correlation > 0.7:
                        trend = "degrading"
                    elif correlation < -0.7:
                        trend = "improving"
                    else:
                        trend = "stable"
                    
                    trends[metric] = {
                        'trend': trend,
                        'correlation': correlation,
                        'recent_mean': statistics.mean(values[-5:]) if len(values) >= 5 else statistics.mean(values),
                        'historical_mean': statistics.mean(values),
                        'sample_count': len(values)
                    }
        
        return {
            'stage_name': stage_name,
            'analysis_period_days': days,
            'total_samples': len(recent_data),
            'trends': trends,
            'overall_health': self._calculate_stage_health(trends)
        }
    
    def _calculate_stage_health(self, trends: Dict[str, Any]) -> str:
        """Calculate overall health score for a stage"""
        if not trends:
            return "unknown"
        
        degrading_count = sum(1 for trend in trends.values() if trend['trend'] == 'degrading')
        improving_count = sum(1 for trend in trends.values() if trend['trend'] == 'improving')
        
        if degrading_count > improving_count:
            return "degrading"
        elif improving_count > degrading_count:
            return "improving"
        else:
            return "stable"
    
    def calculate_performance_health_score(self) -> PerformanceHealthScore:
        """Calculate overall performance health score"""
        if not self.historical_data:
            return PerformanceHealthScore(
                overall_score=50.0,
                stage_scores={},
                regression_count=0,
                critical_issues=[],
                recommendations=["No historical data available for health assessment"],
                trend_direction="unknown"
            )
        
        stage_scores = {}
        critical_issues = []
        recommendations = []
        
        # Calculate score for each stage
        for stage_name in self.historical_data.keys():
            trends = self.analyze_performance_trends(stage_name, days=7)
            if 'trends' in trends:
                stage_health = self._calculate_stage_health(trends['trends'])
                
                # Convert health to score
                if stage_health == "improving":
                    score = 85.0
                elif stage_health == "stable":
                    score = 70.0
                elif stage_health == "degrading":
                    score = 40.0
                else:
                    score = 50.0
                
                # Adjust score based on active alerts
                stage_alerts = [alert for alert in self.active_alerts if alert.stage_name == stage_name and not alert.resolved]
                for alert in stage_alerts:
                    if alert.severity == RegressionSeverity.CRITICAL:
                        score -= 30
                        critical_issues.append(f"{stage_name}: {alert.regression_type.value} regression ({alert.degradation_percent:+.1f}%)")
                    elif alert.severity == RegressionSeverity.HIGH:
                        score -= 20
                    elif alert.severity == RegressionSeverity.MEDIUM:
                        score -= 10
                    else:
                        score -= 5
                
                score = max(0, min(100, score))
                stage_scores[stage_name] = score
        
        # Calculate overall score
        if stage_scores:
            overall_score = statistics.mean(stage_scores.values())
        else:
            overall_score = 50.0
        
        # Generate recommendations
        active_critical = sum(1 for alert in self.active_alerts if alert.severity == RegressionSeverity.CRITICAL and not alert.resolved)
        active_high = sum(1 for alert in self.active_alerts if alert.severity == RegressionSeverity.HIGH and not alert.resolved)
        
        if active_critical > 0:
            recommendations.append(f"URGENT: Address {active_critical} critical performance regression(s)")
        if active_high > 0:
            recommendations.append(f"High priority: Fix {active_high} high-severity performance issue(s)")
        
        if overall_score < 50:
            recommendations.append("Consider system resource optimization and performance tuning")
        elif overall_score < 70:
            recommendations.append("Monitor performance trends and update baselines")
        else:
            recommendations.append("Performance is within acceptable ranges")
        
        # Determine overall trend
        degrading_stages = sum(1 for score in stage_scores.values() if score < 50)
        improving_stages = sum(1 for score in stage_scores.values() if score > 80)
        
        if degrading_stages > improving_stages:
            trend_direction = "degrading"
        elif improving_stages > degrading_stages:
            trend_direction = "improving"
        else:
            trend_direction = "stable"
        
        return PerformanceHealthScore(
            overall_score=overall_score,
            stage_scores=stage_scores,
            regression_count=len([alert for alert in self.active_alerts if not alert.resolved]),
            critical_issues=critical_issues,
            recommendations=recommendations,
            trend_direction=trend_direction
        )
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Mark an alert as resolved"""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_notes = resolution_notes
                
                log_event("info", f"Performance regression alert resolved",
                         alert_id=alert_id, stage=alert.stage_name, 
                         metric=alert.metric_name, notes=resolution_notes)
                
                inc_counter("performance_regression_resolved",
                           stage=alert.stage_name, metric=alert.metric_name)
                break
    
    def get_active_alerts(self, severity_filter: RegressionSeverity = None) -> List[RegressionAlert]:
        """Get list of active alerts, optionally filtered by severity"""
        alerts = [alert for alert in self.active_alerts if not alert.resolved]
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        return sorted(alerts, key=lambda a: (a.severity.value, a.degradation_percent), reverse=True)


# Global instance for easy access
_global_regression_detector = None

def get_regression_detector() -> PerformanceRegressionDetector:
    """Get global regression detector instance"""
    global _global_regression_detector
    if _global_regression_detector is None:
        _global_regression_detector = PerformanceRegressionDetector()
    return _global_regression_detector
