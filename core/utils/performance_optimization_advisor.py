"""
Performance Optimization Advisory System

Provides intelligent performance optimization recommendations based on:
1. Real-time performance analysis
2. System resource utilization patterns
3. Historical performance trends
4. Best practices for video processing workflows
5. Hardware-specific optimizations

Generates actionable optimization suggestions for VideoLingo pipeline.
"""

import json
import statistics
import psutil
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, NamedTuple, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from core.utils.observability import log_event, inc_counter
from core.utils.performance_regression_detector import get_regression_detector, RegressionSeverity


class OptimizationCategory(Enum):
    """Categories of optimization recommendations"""
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    ARCHITECTURE = "architecture"
    WORKFLOW = "workflow"


class OptimizationPriority(Enum):
    """Priority levels for optimization recommendations"""
    CRITICAL = "critical"    # Immediate action required
    HIGH = "high"           # Should implement soon
    MEDIUM = "medium"       # Beneficial improvement
    LOW = "low"            # Nice to have


@dataclass
class OptimizationRecommendation:
    """A specific performance optimization recommendation"""
    id: str
    title: str
    description: str
    category: OptimizationCategory
    priority: OptimizationPriority
    impact_score: float  # 0-100, expected performance improvement
    implementation_effort: str  # "low", "medium", "high"
    affected_stages: List[str]
    current_metrics: Dict[str, Any]
    expected_improvement: Dict[str, Any]
    implementation_steps: List[str]
    prerequisites: List[str]
    risks: List[str]
    estimated_time_hours: float
    cost_benefit_ratio: float  # benefit / effort


@dataclass
class SystemResourceProfile:
    """System resource profile for optimization analysis"""
    cpu_count: int
    cpu_frequency_mhz: float
    total_memory_gb: float
    available_memory_gb: float
    disk_type: str  # "ssd", "hdd", "nvme"
    gpu_available: bool
    gpu_memory_gb: float
    network_bandwidth_mbps: Optional[float]
    platform_info: Dict[str, str]


@dataclass
class PerformanceBottleneck:
    """Identified performance bottleneck"""
    stage_name: str
    bottleneck_type: str  # "cpu", "memory", "io", "network", "api"
    severity_score: float  # 0-100
    impact_on_pipeline: float  # 0-100
    evidence: Dict[str, Any]
    suggested_fixes: List[str]


class PerformanceOptimizationAdvisor:
    """Intelligent performance optimization advisor"""
    
    def __init__(self, baseline_dir: str = None):
        self.baseline_dir = Path(baseline_dir or "reports/performance")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.system_profile = self._profile_system()
        self.regression_detector = get_regression_detector()
        
        # Load historical performance data
        self.load_performance_history()
        
        # Optimization knowledge base
        self.optimization_rules = self._load_optimization_rules()
    
    def _profile_system(self) -> SystemResourceProfile:
        """Profile the current system for optimization analysis"""
        # CPU information
        cpu_count = psutil.cpu_count(logical=False)
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_frequency = cpu_freq.current if cpu_freq else 2000.0
        except:
            cpu_frequency = 2000.0  # Default assumption
        
        # Memory information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # Disk type detection (simplified)
        disk_type = "ssd"  # Default assumption for modern systems
        try:
            # Attempt to detect disk type (platform-specific)
            if platform.system() == "Linux":
                import subprocess
                result = subprocess.run(['lsblk', '-d', '-o', 'name,rota'], 
                                      capture_output=True, text=True)
                if "1" in result.stdout:
                    disk_type = "hdd"
            elif platform.system() == "Darwin":  # macOS
                disk_type = "ssd"  # Most Macs use SSD
        except:
            pass
        
        # GPU detection
        gpu_available = False
        gpu_memory_gb = 0.0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_available = True
                gpu_memory_gb = gpus[0].memoryTotal / 1024  # Convert MB to GB
        except ImportError:
            pass
        
        # Platform information
        platform_info = {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
        profile = SystemResourceProfile(
            cpu_count=cpu_count,
            cpu_frequency_mhz=cpu_frequency,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            disk_type=disk_type,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            network_bandwidth_mbps=None,  # Would require network testing
            platform_info=platform_info
        )
        
        log_event("info", "System profiled for optimization analysis",
                 cpu_count=cpu_count, memory_gb=total_memory_gb, 
                 gpu_available=gpu_available)
        
        return profile
    
    def load_performance_history(self):
        """Load recent performance history for analysis"""
        history_file = self.baseline_dir / "stage_performance_summary.jsonl"
        self.performance_history = []
        
        if not history_file.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(days=7)  # Last 7 days
        cutoff_timestamp = cutoff_time.timestamp()
        
        try:
            with open(history_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get('start_time', 0) >= cutoff_timestamp:
                            self.performance_history.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log_event("warning", f"Failed to load performance history: {e}")
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules and patterns"""
        return {
            'memory_optimization': {
                'high_memory_usage_threshold': 0.8,  # 80% memory usage
                'memory_leak_detection_threshold': 1.5,  # 50% increase over time
                'gc_frequency_optimization': True
            },
            'cpu_optimization': {
                'high_cpu_usage_threshold': 0.85,  # 85% CPU usage
                'parallel_processing_threshold': 4,  # Enable if >4 cores
                'cpu_intensive_stages': ['_2_asr', '_10_gen_audio', '_4_2_translate']
            },
            'io_optimization': {
                'large_file_threshold_mb': 100,
                'batch_processing_threshold': 5,
                'ssd_optimization_enabled': True
            },
            'api_optimization': {
                'api_timeout_threshold': 30.0,  # seconds
                'retry_strategy_enabled': True,
                'concurrent_api_calls_max': 5
            }
        }
    
    def analyze_bottlenecks(self) -> List[PerformanceBottleneck]:
        """Identify performance bottlenecks from recent data"""
        bottlenecks = []
        
        if not self.performance_history:
            return bottlenecks
        
        # Group by stage for analysis
        stage_data = {}
        for record in self.performance_history:
            stage = record.get('stage_name', 'unknown')
            if stage not in stage_data:
                stage_data[stage] = []
            stage_data[stage].append(record)
        
        # Analyze each stage for bottlenecks
        for stage_name, records in stage_data.items():
            if len(records) < 3:  # Need minimum data points
                continue
            
            bottlenecks.extend(self._analyze_stage_bottlenecks(stage_name, records))
        
        return sorted(bottlenecks, key=lambda b: b.severity_score, reverse=True)
    
    def _analyze_stage_bottlenecks(self, stage_name: str, records: List[Dict[str, Any]]) -> List[PerformanceBottleneck]:
        """Analyze bottlenecks for a specific stage"""
        bottlenecks = []
        
        # Extract metrics
        durations = [r.get('duration_seconds', 0) for r in records if r.get('duration_seconds')]
        memory_peaks = [r.get('memory_peak_mb', 0) for r in records if r.get('memory_peak_mb')]
        cpu_averages = [r.get('cpu_avg_percent', 0) for r in records if r.get('cpu_avg_percent')]
        
        if not durations:
            return bottlenecks
        
        # Analyze duration bottlenecks
        if durations:
            avg_duration = statistics.mean(durations)
            max_duration = max(durations)
            
            # Check for duration outliers
            if max_duration > avg_duration * 2:
                severity = min(100, (max_duration / avg_duration - 1) * 50)
                bottlenecks.append(PerformanceBottleneck(
                    stage_name=stage_name,
                    bottleneck_type="duration",
                    severity_score=severity,
                    impact_on_pipeline=severity * 0.8,
                    evidence={
                        'avg_duration': avg_duration,
                        'max_duration': max_duration,
                        'duration_variance': statistics.stdev(durations) if len(durations) > 1 else 0
                    },
                    suggested_fixes=[
                        "Optimize algorithm efficiency",
                        "Consider parallel processing",
                        "Review I/O operations"
                    ]
                ))
        
        # Analyze memory bottlenecks
        if memory_peaks:
            avg_memory = statistics.mean(memory_peaks)
            max_memory = max(memory_peaks)
            
            # Check against system memory
            if max_memory > self.system_profile.available_memory_gb * 1024 * 0.7:  # 70% of available memory
                severity = min(100, (max_memory / (self.system_profile.available_memory_gb * 1024)) * 100)
                bottlenecks.append(PerformanceBottleneck(
                    stage_name=stage_name,
                    bottleneck_type="memory",
                    severity_score=severity,
                    impact_on_pipeline=severity * 0.9,
                    evidence={
                        'avg_memory_mb': avg_memory,
                        'max_memory_mb': max_memory,
                        'system_memory_gb': self.system_profile.available_memory_gb
                    },
                    suggested_fixes=[
                        "Implement memory streaming",
                        "Reduce batch sizes",
                        "Add memory cleanup",
                        "Consider chunking large files"
                    ]
                ))
        
        # Analyze CPU bottlenecks
        if cpu_averages:
            avg_cpu = statistics.mean(cpu_averages)
            
            if avg_cpu > 90:  # High CPU usage
                severity = min(100, avg_cpu)
                bottlenecks.append(PerformanceBottleneck(
                    stage_name=stage_name,
                    bottleneck_type="cpu",
                    severity_score=severity,
                    impact_on_pipeline=severity * 0.7,
                    evidence={
                        'avg_cpu_percent': avg_cpu,
                        'cpu_count': self.system_profile.cpu_count
                    },
                    suggested_fixes=[
                        "Enable multiprocessing" if self.system_profile.cpu_count > 2 else "Optimize CPU usage",
                        "Use vectorized operations",
                        "Consider GPU acceleration" if self.system_profile.gpu_available else "Profile CPU-intensive code"
                    ]
                ))
        
        return bottlenecks
    
    def generate_optimization_recommendations(self, bottlenecks: List[PerformanceBottleneck] = None) -> List[OptimizationRecommendation]:
        """Generate prioritized optimization recommendations"""
        if bottlenecks is None:
            bottlenecks = self.analyze_bottlenecks()
        
        recommendations = []
        
        # Generate recommendations based on bottlenecks
        for bottleneck in bottlenecks:
            recommendations.extend(self._generate_bottleneck_recommendations(bottleneck))
        
        # Generate general system recommendations
        recommendations.extend(self._generate_system_recommendations())
        
        # Generate configuration recommendations
        recommendations.extend(self._generate_configuration_recommendations())
        
        # Sort by priority and impact
        recommendations = sorted(recommendations, 
                               key=lambda r: (r.priority.value, -r.impact_score, -r.cost_benefit_ratio))
        
        return recommendations
    
    def _generate_bottleneck_recommendations(self, bottleneck: PerformanceBottleneck) -> List[OptimizationRecommendation]:
        """Generate recommendations for a specific bottleneck"""
        recommendations = []
        base_id = f"{bottleneck.stage_name}_{bottleneck.bottleneck_type}"
        
        if bottleneck.bottleneck_type == "memory":
            recommendations.append(OptimizationRecommendation(
                id=f"{base_id}_streaming",
                title=f"Implement memory streaming for {bottleneck.stage_name}",
                description="Process data in smaller chunks to reduce memory usage",
                category=OptimizationCategory.MEMORY,
                priority=OptimizationPriority.HIGH if bottleneck.severity_score > 70 else OptimizationPriority.MEDIUM,
                impact_score=min(80, bottleneck.severity_score),
                implementation_effort="medium",
                affected_stages=[bottleneck.stage_name],
                current_metrics=bottleneck.evidence,
                expected_improvement={"memory_reduction_percent": 40, "duration_increase_percent": 10},
                implementation_steps=[
                    "Identify large data structures in the stage",
                    "Implement chunked processing",
                    "Add progress tracking for chunks",
                    "Test with various chunk sizes"
                ],
                prerequisites=["Memory profiling tools", "Test data"],
                risks=["Slight increase in processing time", "Code complexity"],
                estimated_time_hours=8.0,
                cost_benefit_ratio=min(80, bottleneck.severity_score) / 8.0
            ))
        
        elif bottleneck.bottleneck_type == "cpu":
            if self.system_profile.cpu_count > 2:
                recommendations.append(OptimizationRecommendation(
                    id=f"{base_id}_parallel",
                    title=f"Enable parallel processing for {bottleneck.stage_name}",
                    description="Utilize multiple CPU cores to improve processing speed",
                    category=OptimizationCategory.CPU,
                    priority=OptimizationPriority.HIGH,
                    impact_score=min(90, self.system_profile.cpu_count * 15),
                    implementation_effort="high",
                    affected_stages=[bottleneck.stage_name],
                    current_metrics=bottleneck.evidence,
                    expected_improvement={
                        "duration_reduction_percent": min(60, self.system_profile.cpu_count * 15),
                        "cpu_efficiency_improvement": 30
                    },
                    implementation_steps=[
                        "Identify parallelizable operations",
                        "Implement multiprocessing or threading",
                        "Add proper synchronization",
                        "Optimize for CPU count"
                    ],
                    prerequisites=["Concurrent programming knowledge", "Testing framework"],
                    risks=["Thread safety issues", "Resource contention", "Debugging complexity"],
                    estimated_time_hours=16.0,
                    cost_benefit_ratio=min(90, self.system_profile.cpu_count * 15) / 16.0
                ))
        
        elif bottleneck.bottleneck_type == "duration":
            recommendations.append(OptimizationRecommendation(
                id=f"{base_id}_algorithm",
                title=f"Optimize algorithms in {bottleneck.stage_name}",
                description="Review and optimize core algorithms for better performance",
                category=OptimizationCategory.ARCHITECTURE,
                priority=OptimizationPriority.MEDIUM,
                impact_score=50,
                implementation_effort="high",
                affected_stages=[bottleneck.stage_name],
                current_metrics=bottleneck.evidence,
                expected_improvement={"duration_reduction_percent": 30, "resource_efficiency": 20},
                implementation_steps=[
                    "Profile code to identify hotspots",
                    "Research more efficient algorithms",
                    "Implement and benchmark alternatives",
                    "A/B test performance improvements"
                ],
                prerequisites=["Code profiling tools", "Algorithm knowledge", "Test suite"],
                risks=["Code complexity", "Potential bugs", "Maintenance overhead"],
                estimated_time_hours=24.0,
                cost_benefit_ratio=50 / 24.0
            ))
        
        return recommendations
    
    def _generate_system_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate system-level optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        memory_usage_ratio = (self.system_profile.total_memory_gb - self.system_profile.available_memory_gb) / self.system_profile.total_memory_gb
        
        if memory_usage_ratio > 0.8:
            recommendations.append(OptimizationRecommendation(
                id="system_memory_upgrade",
                title="Consider memory upgrade",
                description=f"Current memory usage is {memory_usage_ratio:.1%}. Consider upgrading RAM.",
                category=OptimizationCategory.ARCHITECTURE,
                priority=OptimizationPriority.HIGH,
                impact_score=70,
                implementation_effort="low",
                affected_stages=["all"],
                current_metrics={"memory_usage_ratio": memory_usage_ratio},
                expected_improvement={"overall_performance": 40, "stability": 50},
                implementation_steps=["Identify memory requirements", "Purchase and install RAM"],
                prerequisites=["Budget", "Hardware compatibility check"],
                risks=["Hardware compatibility", "Cost"],
                estimated_time_hours=2.0,
                cost_benefit_ratio=70 / 2.0
            ))
        
        # GPU recommendations
        if not self.system_profile.gpu_available:
            gpu_intensive_stages = ['_2_asr', '_10_gen_audio']
            if any(stage in [r.get('stage_name') for r in self.performance_history] for stage in gpu_intensive_stages):
                recommendations.append(OptimizationRecommendation(
                    id="system_gpu_acceleration",
                    title="Consider GPU acceleration",
                    description="Add GPU support for AI-intensive stages like ASR and TTS",
                    category=OptimizationCategory.ARCHITECTURE,
                    priority=OptimizationPriority.MEDIUM,
                    impact_score=85,
                    implementation_effort="high",
                    affected_stages=gpu_intensive_stages,
                    current_metrics={"gpu_available": False},
                    expected_improvement={"asr_speed_improvement": 300, "tts_speed_improvement": 200},
                    implementation_steps=[
                        "Research GPU compatibility",
                        "Install GPU and drivers",
                        "Update software for GPU support",
                        "Benchmark performance improvements"
                    ],
                    prerequisites=["GPU hardware", "CUDA/ROCm support", "Updated libraries"],
                    risks=["Hardware cost", "Driver compatibility", "Power requirements"],
                    estimated_time_hours=12.0,
                    cost_benefit_ratio=85 / 12.0
                ))
        
        return recommendations
    
    def _generate_configuration_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate configuration optimization recommendations"""
        recommendations = []
        
        # Batch processing recommendation
        if len(self.performance_history) > 10:  # Sufficient data to analyze
            avg_file_size = statistics.mean([
                r.get('additional_metrics', {}).get('file_size_mb', 0)
                for r in self.performance_history
                if r.get('additional_metrics', {}).get('file_size_mb', 0) > 0
            ]) if any(r.get('additional_metrics', {}).get('file_size_mb', 0) > 0 for r in self.performance_history) else 0
            
            if avg_file_size > 0:
                recommendations.append(OptimizationRecommendation(
                    id="config_batch_processing",
                    title="Optimize batch processing configuration",
                    description="Adjust batch sizes based on file size and system resources",
                    category=OptimizationCategory.CONFIGURATION,
                    priority=OptimizationPriority.LOW,
                    impact_score=30,
                    implementation_effort="low",
                    affected_stages=["all"],
                    current_metrics={"avg_file_size_mb": avg_file_size},
                    expected_improvement={"throughput_improvement": 15, "resource_efficiency": 10},
                    implementation_steps=[
                        "Analyze optimal batch sizes",
                        "Update configuration parameters",
                        "Test with various batch sizes",
                        "Monitor performance improvements"
                    ],
                    prerequisites=["Configuration access", "Test data"],
                    risks=["Suboptimal configuration", "Testing overhead"],
                    estimated_time_hours=4.0,
                    cost_benefit_ratio=30 / 4.0
                ))
        
        return recommendations
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        bottlenecks = self.analyze_bottlenecks()
        recommendations = self.generate_optimization_recommendations(bottlenecks)
        health_score = self.regression_detector.calculate_performance_health_score()
        
        # Calculate potential improvements
        total_impact = sum(r.impact_score for r in recommendations[:5])  # Top 5 recommendations
        total_effort = sum(r.estimated_time_hours for r in recommendations[:5])
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_profile': asdict(self.system_profile),
            'performance_health': asdict(health_score),
            'bottlenecks': [asdict(b) for b in bottlenecks],
            'recommendations': [asdict(r) for r in recommendations],
            'optimization_summary': {
                'total_recommendations': len(recommendations),
                'critical_recommendations': len([r for r in recommendations if r.priority == OptimizationPriority.CRITICAL]),
                'high_priority_recommendations': len([r for r in recommendations if r.priority == OptimizationPriority.HIGH]),
                'potential_performance_gain': total_impact / len(recommendations[:5]) if recommendations[:5] else 0,
                'estimated_implementation_time_hours': total_effort,
                'cost_benefit_analysis': total_impact / total_effort if total_effort > 0 else 0
            },
            'quick_wins': [
                asdict(r) for r in recommendations 
                if r.implementation_effort == "low" and r.impact_score > 30
            ][:3],
            'long_term_improvements': [
                asdict(r) for r in recommendations
                if r.impact_score > 60
            ][:3]
        }
        
        # Save report
        self._save_optimization_report(report)
        
        return report
    
    def _save_optimization_report(self, report: Dict[str, Any]):
        """Save optimization report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.baseline_dir / f"optimization_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as latest report
        latest_file = self.baseline_dir / "latest_optimization_report.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        log_event("info", "Optimization report generated",
                 total_recommendations=report['optimization_summary']['total_recommendations'],
                 critical_count=report['optimization_summary']['critical_recommendations'],
                 potential_gain=report['optimization_summary']['potential_performance_gain'])


# Global instance for easy access
_global_optimization_advisor = None

def get_optimization_advisor() -> PerformanceOptimizationAdvisor:
    """Get global optimization advisor instance"""
    global _global_optimization_advisor
    if _global_optimization_advisor is None:
        _global_optimization_advisor = PerformanceOptimizationAdvisor()
    return _global_optimization_advisor
