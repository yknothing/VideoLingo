"""
VideoLingo Performance Monitoring and Benchmarking System

Provides comprehensive performance monitoring for the 15-step processing pipeline:
1. Real-time performance metrics collection
2. Pipeline bottleneck detection  
3. Memory leak monitoring
4. API call performance tracking
5. User experience metrics (5-minute success path)
6. Performance regression detection
7. Optimization recommendations

Integration with existing observability system for seamless monitoring.
"""

import time
import json
import os
import threading
import statistics
import psutil
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
from functools import wraps
import warnings

# Import VideoLingo's observability system
from core.utils.observability import log_event, time_block, inc_counter, observe_histogram


@dataclass
class PipelineStageMetrics:
    """Performance metrics for a single pipeline stage"""
    stage_name: str
    stage_number: int
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    memory_start_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    cpu_avg_percent: Optional[float] = None
    cpu_peak_percent: Optional[float] = None
    api_calls_count: int = 0
    api_calls_duration: float = 0.0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    gpu_usage_percent: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelinePerformanceBaseline:
    """Performance baseline for pipeline stages"""
    stage_name: str
    expected_duration_p50: float  # seconds
    expected_duration_p95: float  # seconds
    memory_limit_mb: float
    cpu_limit_percent: float
    api_calls_limit: int
    io_limit_mb: float
    regression_threshold: float = 0.20  # 20% degradation triggers alert


@dataclass
class UserExperienceMetrics:
    """User experience focused metrics"""
    first_install_to_first_success_minutes: Optional[float] = None
    five_minute_success_rate: float = 0.0
    error_recovery_time_seconds: float = 0.0
    batch_processing_efficiency: float = 0.0
    user_wait_time_perception: str = "unknown"  # fast, acceptable, slow, too_slow


class PerformanceMonitor:
    """Real-time performance monitoring with minimal overhead"""
    
    def __init__(self, monitor_interval: float = 1.0):
        self.monitor_interval = monitor_interval
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_buffer = []
        self.start_io_counters = None
        self.start_time = None
        
        # GPU monitoring (optional)
        self.gpu_available = False
        try:
            import GPUtil
            self.gpu_available = True
            self.gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        except (ImportError, IndexError):
            pass
    
    def start_monitoring(self, stage_name: str = "", video_id: str = ""):
        """Start background monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.metrics_buffer = []
        self.start_time = time.time()
        
        try:
            self.start_io_counters = self.process.io_counters()
        except (AttributeError, psutil.AccessDenied):
            self.start_io_counters = None
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(stage_name, video_id),
            daemon=True
        )
        self.monitor_thread.start()
        
        log_event("debug", "Performance monitoring started", 
                 stage=stage_name, video_id=video_id)
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics"""
        if not self.monitoring:
            return {}
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Calculate aggregated metrics
        metrics = self._calculate_aggregated_metrics()
        
        log_event("debug", "Performance monitoring stopped", 
                 duration_seconds=metrics.get('duration_seconds', 0),
                 memory_peak_mb=metrics.get('memory_peak_mb', 0))
        
        return metrics
    
    def _monitoring_loop(self, stage_name: str, video_id: str):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                sample = self._collect_sample()
                self.metrics_buffer.append(sample)
                
                # Emit real-time metrics
                observe_histogram("memory_usage_mb", sample['memory_mb'], 
                                stage=stage_name, video_id=video_id)
                observe_histogram("cpu_usage_percent", sample['cpu_percent'],
                                stage=stage_name, video_id=video_id)
                
            except Exception as e:
                log_event("warning", f"Performance monitoring sample failed: {e}")
            
            time.sleep(self.monitor_interval)
    
    def _collect_sample(self) -> Dict[str, Any]:
        """Collect a single performance sample"""
        sample = {
            'timestamp': time.time(),
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'cpu_percent': self.process.cpu_percent(),
        }
        
        # GPU usage if available
        if self.gpu_available and self.gpu:
            try:
                self.gpu.load()  # Refresh GPU stats
                sample['gpu_percent'] = self.gpu.load * 100
                sample['gpu_memory_mb'] = self.gpu.memoryUsed
            except Exception:
                pass
        
        return sample
    
    def _calculate_aggregated_metrics(self) -> Dict[str, Any]:
        """Calculate aggregated metrics from samples"""
        if not self.metrics_buffer:
            return {}
        
        memory_values = [s['memory_mb'] for s in self.metrics_buffer]
        cpu_values = [s['cpu_percent'] for s in self.metrics_buffer]
        
        metrics = {
            'duration_seconds': time.time() - self.start_time if self.start_time else 0,
            'memory_peak_mb': max(memory_values),
            'memory_avg_mb': statistics.mean(memory_values),
            'cpu_peak_percent': max(cpu_values),
            'cpu_avg_percent': statistics.mean(cpu_values),
            'samples_count': len(self.metrics_buffer)
        }
        
        # I/O metrics
        if self.start_io_counters:
            try:
                current_io = self.process.io_counters()
                metrics['io_read_mb'] = (current_io.read_bytes - self.start_io_counters.read_bytes) / (1024 * 1024)
                metrics['io_write_mb'] = (current_io.write_bytes - self.start_io_counters.write_bytes) / (1024 * 1024)
            except (AttributeError, psutil.AccessDenied):
                metrics['io_read_mb'] = 0
                metrics['io_write_mb'] = 0
        
        # GPU metrics
        if self.gpu_available:
            gpu_values = [s.get('gpu_percent', 0) for s in self.metrics_buffer if 'gpu_percent' in s]
            if gpu_values:
                metrics['gpu_peak_percent'] = max(gpu_values)
                metrics['gpu_avg_percent'] = statistics.mean(gpu_values)
        
        return metrics


class PipelinePerformanceTracker:
    """Track performance across the entire 15-step pipeline"""
    
    # Default baselines for VideoLingo pipeline stages
    DEFAULT_BASELINES = {
        "_1_ytdlp": PipelinePerformanceBaseline(
            stage_name="_1_ytdlp", 
            expected_duration_p50=30.0, expected_duration_p95=120.0,
            memory_limit_mb=500, cpu_limit_percent=80, 
            api_calls_limit=10, io_limit_mb=500
        ),
        "_2_asr": PipelinePerformanceBaseline(
            stage_name="_2_asr",
            expected_duration_p50=60.0, expected_duration_p95=300.0,
            memory_limit_mb=2000, cpu_limit_percent=90,
            api_calls_limit=5, io_limit_mb=200
        ),
        "_3_1_split_nlp": PipelinePerformanceBaseline(
            stage_name="_3_1_split_nlp",
            expected_duration_p50=10.0, expected_duration_p95=30.0,
            memory_limit_mb=200, cpu_limit_percent=60,
            api_calls_limit=0, io_limit_mb=10
        ),
        "_4_2_translate": PipelinePerformanceBaseline(
            stage_name="_4_2_translate",
            expected_duration_p50=45.0, expected_duration_p95=180.0,
            memory_limit_mb=300, cpu_limit_percent=70,
            api_calls_limit=50, io_limit_mb=20
        ),
        "_10_gen_audio": PipelinePerformanceBaseline(
            stage_name="_10_gen_audio",
            expected_duration_p50=120.0, expected_duration_p95=600.0,
            memory_limit_mb=1500, cpu_limit_percent=85,
            api_calls_limit=100, io_limit_mb=1000
        )
    }
    
    def __init__(self, baseline_dir: str = None):
        self.baseline_dir = Path(baseline_dir or "reports/performance")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.baselines = self.DEFAULT_BASELINES.copy()
        self.load_custom_baselines()
        
        self.current_pipeline_start = None
        self.current_stage_metrics: Dict[str, PipelineStageMetrics] = {}
        self.pipeline_history: List[Dict[str, Any]] = []
        self.monitor = PerformanceMonitor()
        
        self.regression_alerts: List[Dict[str, Any]] = []
    
    def load_custom_baselines(self):
        """Load custom baselines from configuration"""
        baseline_file = self.baseline_dir / "pipeline_baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                    for stage_name, baseline_data in data.items():
                        self.baselines[stage_name] = PipelinePerformanceBaseline(**baseline_data)
            except Exception as e:
                log_event("warning", f"Failed to load custom baselines: {e}")
    
    def save_baselines(self):
        """Save current baselines"""
        baseline_file = self.baseline_dir / "pipeline_baselines.json"
        baselines_dict = {name: asdict(baseline) for name, baseline in self.baselines.items()}
        with open(baseline_file, 'w') as f:
            json.dump(baselines_dict, f, indent=2)
    
    @contextmanager
    def track_stage(self, stage_name: str, stage_number: int, video_id: str = ""):
        """Context manager to track a pipeline stage"""
        stage_metrics = PipelineStageMetrics(
            stage_name=stage_name,
            stage_number=stage_number,
            start_time=time.time(),
            memory_start_mb=psutil.Process().memory_info().rss / 1024 / 1024
        )
        
        # Start monitoring
        self.monitor.start_monitoring(stage_name, video_id)
        
        # Log stage start
        log_event("info", f"Pipeline stage started", 
                 stage=stage_name, video_id=video_id)
        inc_counter("pipeline_stage_started", stage=stage_name, video_id=video_id)
        
        try:
            with time_block(f"stage_{stage_name}", stage=stage_name, video_id=video_id):
                yield stage_metrics
            
            stage_metrics.success = True
            
        except Exception as e:
            stage_metrics.success = False
            stage_metrics.error_message = str(e)
            inc_counter("pipeline_stage_failed", stage=stage_name, video_id=video_id)
            log_event("error", f"Pipeline stage failed: {e}", stage=stage_name, video_id=video_id)
            raise
            
        finally:
            # Stop monitoring and collect metrics
            monitor_metrics = self.monitor.stop_monitoring()
            
            stage_metrics.end_time = time.time()
            stage_metrics.duration_seconds = stage_metrics.end_time - stage_metrics.start_time
            stage_metrics.memory_end_mb = psutil.Process().memory_info().rss / 1024 / 1024
            stage_metrics.memory_peak_mb = monitor_metrics.get('memory_peak_mb', stage_metrics.memory_end_mb)
            stage_metrics.cpu_avg_percent = monitor_metrics.get('cpu_avg_percent', 0)
            stage_metrics.cpu_peak_percent = monitor_metrics.get('cpu_peak_percent', 0)
            stage_metrics.io_read_mb = monitor_metrics.get('io_read_mb', 0)
            stage_metrics.io_write_mb = monitor_metrics.get('io_write_mb', 0)
            stage_metrics.gpu_usage_percent = monitor_metrics.get('gpu_avg_percent')
            
            # Store stage metrics
            self.current_stage_metrics[stage_name] = stage_metrics
            
            # Emit performance metrics
            observe_histogram("stage_duration_seconds", stage_metrics.duration_seconds,
                            stage=stage_name, video_id=video_id)
            observe_histogram("stage_memory_peak_mb", stage_metrics.memory_peak_mb,
                            stage=stage_name, video_id=video_id)
            
            # Check for regressions
            self._check_stage_regression(stage_metrics, video_id)
            
            # Save stage metrics
            self._save_stage_metrics(stage_metrics, video_id)
            
            if stage_metrics.success:
                inc_counter("pipeline_stage_completed", stage=stage_name, video_id=video_id)
                log_event("info", f"Pipeline stage completed successfully", 
                         stage=stage_name, video_id=video_id,
                         duration_seconds=stage_metrics.duration_seconds,
                         memory_peak_mb=stage_metrics.memory_peak_mb)
    
    def _check_stage_regression(self, metrics: PipelineStageMetrics, video_id: str):
        """Check for performance regression against baselines"""
        if metrics.stage_name not in self.baselines:
            return
        
        baseline = self.baselines[metrics.stage_name]
        alerts = []
        
        # Duration regression check
        if metrics.duration_seconds and metrics.duration_seconds > baseline.expected_duration_p95 * (1 + baseline.regression_threshold):
            alert = {
                'type': 'duration_regression',
                'stage': metrics.stage_name,
                'current': metrics.duration_seconds,
                'baseline_p95': baseline.expected_duration_p95,
                'exceed_percent': ((metrics.duration_seconds / baseline.expected_duration_p95) - 1) * 100
            }
            alerts.append(alert)
        
        # Memory regression check
        if metrics.memory_peak_mb and metrics.memory_peak_mb > baseline.memory_limit_mb:
            alert = {
                'type': 'memory_regression',
                'stage': metrics.stage_name,
                'current': metrics.memory_peak_mb,
                'baseline_limit': baseline.memory_limit_mb,
                'exceed_percent': ((metrics.memory_peak_mb / baseline.memory_limit_mb) - 1) * 100
            }
            alerts.append(alert)
        
        # CPU regression check
        if metrics.cpu_avg_percent and metrics.cpu_avg_percent > baseline.cpu_limit_percent:
            alert = {
                'type': 'cpu_regression',
                'stage': metrics.stage_name,
                'current': metrics.cpu_avg_percent,
                'baseline_limit': baseline.cpu_limit_percent,
                'exceed_percent': ((metrics.cpu_avg_percent / baseline.cpu_limit_percent) - 1) * 100
            }
            alerts.append(alert)
        
        # Log and store alerts
        for alert in alerts:
            self.regression_alerts.append({
                'timestamp': datetime.now().isoformat(),
                'video_id': video_id,
                **alert
            })
            
            log_event("warning", 
                     f"PERFORMANCE REGRESSION: {alert['type']} in {alert['stage']}: "
                     f"{alert['current']:.2f} exceeds baseline by {alert['exceed_percent']:.1f}%",
                     stage=metrics.stage_name, video_id=video_id)
            
            inc_counter("performance_regression_detected", 
                       stage=metrics.stage_name, 
                       regression_type=alert['type'],
                       video_id=video_id)
    
    def _save_stage_metrics(self, metrics: PipelineStageMetrics, video_id: str):
        """Save stage metrics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.baseline_dir / f"stage_{metrics.stage_name}_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Also append to summary
        summary_file = self.baseline_dir / "stage_performance_summary.jsonl"
        with open(summary_file, 'a') as f:
            record = asdict(metrics)
            record['video_id'] = video_id
            f.write(json.dumps(record) + '\n')
    
    def start_pipeline_tracking(self, video_id: str = "", total_stages: int = 15):
        """Start tracking a complete pipeline run"""
        self.current_pipeline_start = time.time()
        self.current_stage_metrics = {}
        
        log_event("info", "Pipeline tracking started", 
                 video_id=video_id, total_stages=total_stages)
        inc_counter("pipeline_started", video_id=video_id)
    
    def complete_pipeline_tracking(self, video_id: str = "", success: bool = True):
        """Complete pipeline tracking and generate report"""
        if not self.current_pipeline_start:
            log_event("warning", "Pipeline tracking completed without start")
            return None
        
        total_duration = time.time() - self.current_pipeline_start
        
        pipeline_summary = {
            'video_id': video_id,
            'start_time': self.current_pipeline_start,
            'total_duration_seconds': total_duration,
            'success': success,
            'stages_completed': len(self.current_stage_metrics),
            'stage_metrics': {name: asdict(metrics) for name, metrics in self.current_stage_metrics.items()},
            'regression_alerts': [alert for alert in self.regression_alerts if alert.get('video_id') == video_id]
        }
        
        # Calculate pipeline-level metrics
        if self.current_stage_metrics:
            durations = [m.duration_seconds for m in self.current_stage_metrics.values() if m.duration_seconds]
            memory_peaks = [m.memory_peak_mb for m in self.current_stage_metrics.values() if m.memory_peak_mb]
            
            pipeline_summary['metrics'] = {
                'longest_stage_duration': max(durations) if durations else 0,
                'total_processing_time': sum(durations) if durations else 0,
                'peak_memory_usage': max(memory_peaks) if memory_peaks else 0,
                'stages_with_regressions': len([m for m in self.current_stage_metrics.values() if not m.success])
            }
        
        # Save pipeline summary
        self._save_pipeline_summary(pipeline_summary)
        
        # Emit pipeline metrics
        observe_histogram("pipeline_total_duration", total_duration, video_id=video_id)
        inc_counter("pipeline_completed" if success else "pipeline_failed", video_id=video_id)
        
        log_event("info", f"Pipeline tracking completed",
                 video_id=video_id, 
                 duration_seconds=total_duration,
                 success=success,
                 stages_completed=len(self.current_stage_metrics))
        
        return pipeline_summary
    
    def _save_pipeline_summary(self, summary: Dict[str, Any]):
        """Save complete pipeline summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.baseline_dir / f"pipeline_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also append to master log
        master_log = self.baseline_dir / "pipeline_performance_log.jsonl"
        with open(master_log, 'a') as f:
            f.write(json.dumps(summary) + '\n')


# Performance monitoring decorators for easy integration
def monitor_performance(stage_name: str, stage_number: int = 0, video_id_param: str = None):
    """Decorator to monitor performance of a function"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract video_id if specified
            video_id = ""
            if video_id_param and video_id_param in kwargs:
                video_id = str(kwargs[video_id_param])
            
            # Get global tracker instance
            tracker = getattr(wrapper, '_tracker', None)
            if not tracker:
                tracker = PipelinePerformanceTracker()
                wrapper._tracker = tracker
            
            with tracker.track_stage(stage_name, stage_number, video_id):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def track_api_calls(api_name: str):
    """Decorator to track API call performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                # Log API call metrics
                observe_histogram("api_call_duration", duration, api_name=api_name, success=success)
                inc_counter("api_calls_total", api_name=api_name, success=success)
                
                if error:
                    log_event("warning", f"API call failed: {error}", 
                             api_name=api_name, duration_seconds=duration)
                else:
                    log_event("debug", f"API call completed",
                             api_name=api_name, duration_seconds=duration)
        
        return wrapper
    return decorator


# User Experience Metrics Tracker
class UserExperienceTracker:
    """Track user experience focused metrics"""
    
    def __init__(self):
        self.session_start = time.time()
        self.first_success_time = None
        self.error_count = 0
        self.recovery_times = []
    
    @contextmanager
    def track_user_journey(self, journey_name: str):
        """Track a user journey (e.g., first install to first success)"""
        start_time = time.time()
        
        try:
            yield self
            
            # Success path
            duration_minutes = (time.time() - start_time) / 60
            if journey_name == "first_install_to_first_success" and not self.first_success_time:
                self.first_success_time = duration_minutes
                observe_histogram("user_first_success_minutes", duration_minutes)
                
                # Classify user experience
                if duration_minutes <= 5:
                    experience = "excellent"
                elif duration_minutes <= 15:
                    experience = "good"
                elif duration_minutes <= 30:
                    experience = "acceptable"
                else:
                    experience = "needs_improvement"
                
                log_event("info", f"User first success: {duration_minutes:.1f} minutes ({experience})",
                         journey=journey_name, experience=experience)
        
        except Exception as e:
            self.error_count += 1
            inc_counter("user_errors", journey=journey_name, error_type=type(e).__name__)
            log_event("error", f"User journey failed: {e}", journey=journey_name)
            raise
    
    def track_error_recovery(self, error_type: str, recovery_action: str):
        """Track time to recover from errors"""
        start_time = time.time()
        
        @contextmanager
        def recovery_tracker():
            try:
                yield
                recovery_time = time.time() - start_time
                self.recovery_times.append(recovery_time)
                
                observe_histogram("error_recovery_seconds", recovery_time, 
                                error_type=error_type, recovery_action=recovery_action)
                log_event("info", f"Error recovery successful: {recovery_time:.1f}s",
                         error_type=error_type, recovery_action=recovery_action)
            except Exception as e:
                log_event("warning", f"Error recovery failed: {e}",
                         error_type=error_type, recovery_action=recovery_action)
                raise
        
        return recovery_tracker()


# Global instances for easy use
_global_performance_tracker = None
_global_ux_tracker = None

def get_performance_tracker() -> PipelinePerformanceTracker:
    """Get global performance tracker instance"""
    global _global_performance_tracker
    if _global_performance_tracker is None:
        _global_performance_tracker = PipelinePerformanceTracker()
    return _global_performance_tracker

def get_ux_tracker() -> UserExperienceTracker:
    """Get global UX tracker instance"""
    global _global_ux_tracker
    if _global_ux_tracker is None:
        _global_ux_tracker = UserExperienceTracker()
    return _global_ux_tracker
