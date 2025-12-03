"""
Performance Benchmarking System for Video Processing Application

This module provides comprehensive performance benchmarking capabilities including:
- Video processing speed benchmarks with baselines
- Memory usage monitoring and limits
- CPU utilization tracking
- I/O performance metrics
- Regression detection mechanism
- Alerts for performance degradation
"""

import time
import json
import os
import sys
import threading
import multiprocessing
import statistics
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps
import tempfile
import shutil

# Performance monitoring imports
import psutil
import numpy as np

# For video processing simulation
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV not installed. Video processing benchmarks will use mock data.")


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    test_name: str
    timestamp: str
    duration_seconds: float
    memory_peak_mb: float
    memory_average_mb: float
    cpu_peak_percent: float
    cpu_average_percent: float
    io_read_mb: float
    io_write_mb: float
    io_read_ops: int
    io_write_ops: int
    frames_processed: int = 0
    fps: float = 0.0
    throughput_mbps: float = 0.0
    success: bool = True
    error_message: str = ""
    additional_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    test_name: str
    duration_p50: float
    duration_p95: float
    memory_peak_limit: float
    cpu_average_limit: float
    min_fps: float
    min_throughput_mbps: float
    io_read_limit_mb: float
    io_write_limit_mb: float


class PerformanceMonitor:
    """Monitors system resources during benchmark execution"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.process = psutil.Process()
        self.monitoring = False
        self.thread = None
        self.metrics = {
            'memory_mb': [],
            'cpu_percent': [],
            'io_counters': []
        }
        self.start_io_counters = None
    
    def start(self):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.metrics = {'memory_mb': [], 'cpu_percent': [], 'io_counters': []}
        # I/O counters are not available on all platforms (e.g., macOS)
        try:
            self.start_io_counters = self.process.io_counters()
        except (AttributeError, psutil.AccessDenied):
            self.start_io_counters = None
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated metrics"""
        self.monitoring = False
        if self.thread:
            self.thread.join()
        
        # Try to get I/O counters if available
        try:
            end_io_counters = self.process.io_counters()
        except (AttributeError, psutil.AccessDenied):
            end_io_counters = None
        
        metrics = {}
        if self.metrics['memory_mb']:
            metrics['memory_peak_mb'] = max(self.metrics['memory_mb'])
            metrics['memory_average_mb'] = statistics.mean(self.metrics['memory_mb'])
        else:
            metrics['memory_peak_mb'] = 0
            metrics['memory_average_mb'] = 0
        
        if self.metrics['cpu_percent']:
            metrics['cpu_peak_percent'] = max(self.metrics['cpu_percent'])
            metrics['cpu_average_percent'] = statistics.mean(self.metrics['cpu_percent'])
        else:
            metrics['cpu_peak_percent'] = 0
            metrics['cpu_average_percent'] = 0
        
        if self.start_io_counters and end_io_counters:
            metrics['io_read_mb'] = (end_io_counters.read_bytes - 
                                    self.start_io_counters.read_bytes) / (1024 * 1024)
            metrics['io_write_mb'] = (end_io_counters.write_bytes - 
                                     self.start_io_counters.write_bytes) / (1024 * 1024)
            metrics['io_read_ops'] = (end_io_counters.read_count - 
                                      self.start_io_counters.read_count)
            metrics['io_write_ops'] = (end_io_counters.write_count - 
                                       self.start_io_counters.write_count)
        else:
            metrics['io_read_mb'] = 0
            metrics['io_write_mb'] = 0
            metrics['io_read_ops'] = 0
            metrics['io_write_ops'] = 0
        
        return metrics
    
    def _monitor(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Memory usage in MB
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.metrics['memory_mb'].append(memory_mb)
                
                # CPU percentage
                cpu_percent = self.process.cpu_percent()
                self.metrics['cpu_percent'].append(cpu_percent)
                
                # I/O counters (if available)
                try:
                    io_counters = self.process.io_counters()
                    self.metrics['io_counters'].append(io_counters)
                except (AttributeError, psutil.AccessDenied):
                    pass  # I/O counters not available on this platform
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            
            time.sleep(self.interval)


class PerformanceBenchmark:
    """Main benchmarking class with regression detection and alerting"""
    
    # Default performance baselines
    DEFAULT_BASELINES = {
        'video_processing_720p': PerformanceBaseline(
            test_name='video_processing_720p',
            duration_p50=1.0,  # 1 second for 1-second clip
            duration_p95=1.5,
            memory_peak_limit=500.0,  # MB
            cpu_average_limit=80.0,   # %
            min_fps=30.0,
            min_throughput_mbps=10.0,
            io_read_limit_mb=100.0,
            io_write_limit_mb=100.0
        ),
        'video_processing_1080p': PerformanceBaseline(
            test_name='video_processing_1080p',
            duration_p50=2.0,
            duration_p95=3.0,
            memory_peak_limit=1000.0,
            cpu_average_limit=85.0,
            min_fps=25.0,
            min_throughput_mbps=20.0,
            io_read_limit_mb=200.0,
            io_write_limit_mb=200.0
        ),
        'video_processing_4k': PerformanceBaseline(
            test_name='video_processing_4k',
            duration_p50=8.0,
            duration_p95=12.0,
            memory_peak_limit=3000.0,
            cpu_average_limit=90.0,
            min_fps=15.0,
            min_throughput_mbps=50.0,
            io_read_limit_mb=500.0,
            io_write_limit_mb=500.0
        ),
        'batch_processing': PerformanceBaseline(
            test_name='batch_processing',
            duration_p50=10.0,
            duration_p95=15.0,
            memory_peak_limit=2000.0,
            cpu_average_limit=85.0,
            min_fps=20.0,
            min_throughput_mbps=30.0,
            io_read_limit_mb=1000.0,
            io_write_limit_mb=1000.0
        ),
        'memory_stress_test': PerformanceBaseline(
            test_name='memory_stress_test',
            duration_p50=5.0,
            duration_p95=8.0,
            memory_peak_limit=4000.0,
            cpu_average_limit=70.0,
            min_fps=0.0,
            min_throughput_mbps=0.0,
            io_read_limit_mb=500.0,
            io_write_limit_mb=500.0
        )
    }
    
    def __init__(self, report_dir: str = "reports/performance"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.baselines = self.DEFAULT_BASELINES.copy()
        self.load_custom_baselines()
        self.alerts = []
        self.regression_threshold = 0.2  # 20% performance degradation triggers alert
    
    def load_custom_baselines(self):
        """Load custom baselines from configuration file"""
        baseline_file = self.report_dir / "baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    custom_baselines = json.load(f)
                    for name, data in custom_baselines.items():
                        self.baselines[name] = PerformanceBaseline(**data)
            except Exception as e:
                print(f"Warning: Could not load custom baselines: {e}")
    
    def save_baselines(self):
        """Save current baselines to configuration file"""
        baseline_file = self.report_dir / "baselines.json"
        baselines_dict = {
            name: asdict(baseline) 
            for name, baseline in self.baselines.items()
        }
        with open(baseline_file, 'w') as f:
            json.dump(baselines_dict, f, indent=2)
    
    @contextmanager
    def benchmark(self, test_name: str, monitor_interval: float = 0.1):
        """Context manager for benchmarking a code block"""
        monitor = PerformanceMonitor(interval=monitor_interval)
        start_time = time.perf_counter()
        
        monitor.start()
        metrics = PerformanceMetrics(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,
            memory_peak_mb=0,
            memory_average_mb=0,
            cpu_peak_percent=0,
            cpu_average_percent=0,
            io_read_mb=0,
            io_write_mb=0,
            io_read_ops=0,
            io_write_ops=0
        )
        
        try:
            yield metrics
            metrics.success = True
        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            raise
        finally:
            duration = time.perf_counter() - start_time
            monitor_metrics = monitor.stop()
            
            metrics.duration_seconds = duration
            metrics.memory_peak_mb = monitor_metrics['memory_peak_mb']
            metrics.memory_average_mb = monitor_metrics['memory_average_mb']
            metrics.cpu_peak_percent = monitor_metrics['cpu_peak_percent']
            metrics.cpu_average_percent = monitor_metrics['cpu_average_percent']
            metrics.io_read_mb = monitor_metrics['io_read_mb']
            metrics.io_write_mb = monitor_metrics['io_write_mb']
            metrics.io_read_ops = monitor_metrics['io_read_ops']
            metrics.io_write_ops = monitor_metrics['io_write_ops']
            
            # Check for regressions
            self.check_regression(metrics)
            
            # Save results
            self.save_metrics(metrics)
    
    def benchmark_function(self, func: Callable, test_name: str, *args, **kwargs) -> PerformanceMetrics:
        """Benchmark a specific function"""
        with self.benchmark(test_name) as metrics:
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                # Extract additional metrics from function result
                for key, value in result.items():
                    if key in ['frames_processed', 'fps', 'throughput_mbps']:
                        setattr(metrics, key, value)
                    else:
                        metrics.additional_metrics[key] = value
        return metrics
    
    def check_regression(self, metrics: PerformanceMetrics) -> List[str]:
        """Check for performance regression against baselines"""
        alerts = []
        
        if metrics.test_name not in self.baselines:
            return alerts
        
        baseline = self.baselines[metrics.test_name]
        
        # Check duration regression
        if metrics.duration_seconds > baseline.duration_p95 * (1 + self.regression_threshold):
            alert = f"PERFORMANCE REGRESSION: {metrics.test_name} duration {metrics.duration_seconds:.2f}s exceeds baseline P95 {baseline.duration_p95:.2f}s by >{self.regression_threshold*100}%"
            alerts.append(alert)
            self.alerts.append((datetime.now(), alert))
        
        # Check memory regression
        if metrics.memory_peak_mb > baseline.memory_peak_limit:
            alert = f"MEMORY REGRESSION: {metrics.test_name} peak memory {metrics.memory_peak_mb:.1f}MB exceeds limit {baseline.memory_peak_limit:.1f}MB"
            alerts.append(alert)
            self.alerts.append((datetime.now(), alert))
        
        # Check CPU regression
        if metrics.cpu_average_percent > baseline.cpu_average_limit:
            alert = f"CPU REGRESSION: {metrics.test_name} average CPU {metrics.cpu_average_percent:.1f}% exceeds limit {baseline.cpu_average_limit:.1f}%"
            alerts.append(alert)
            self.alerts.append((datetime.now(), alert))
        
        # Check FPS regression for video processing
        if baseline.min_fps > 0 and metrics.fps > 0 and metrics.fps < baseline.min_fps:
            alert = f"FPS REGRESSION: {metrics.test_name} FPS {metrics.fps:.1f} below minimum {baseline.min_fps:.1f}"
            alerts.append(alert)
            self.alerts.append((datetime.now(), alert))
        
        # Check I/O regression
        if metrics.io_read_mb > baseline.io_read_limit_mb:
            alert = f"I/O READ REGRESSION: {metrics.test_name} read {metrics.io_read_mb:.1f}MB exceeds limit {baseline.io_read_limit_mb:.1f}MB"
            alerts.append(alert)
            self.alerts.append((datetime.now(), alert))
        
        if metrics.io_write_mb > baseline.io_write_limit_mb:
            alert = f"I/O WRITE REGRESSION: {metrics.test_name} write {metrics.io_write_mb:.1f}MB exceeds limit {baseline.io_write_limit_mb:.1f}MB"
            alerts.append(alert)
            self.alerts.append((datetime.now(), alert))
        
        # Print alerts immediately
        for alert in alerts:
            print(f"\n⚠️  {alert}")
        
        return alerts
    
    def save_metrics(self, metrics: PerformanceMetrics):
        """Save metrics to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.report_dir / f"{metrics.test_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Also append to summary file
        summary_file = self.report_dir / "performance_summary.jsonl"
        with open(summary_file, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
    
    def generate_report(self, test_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate performance report from saved metrics"""
        summary_file = self.report_dir / "performance_summary.jsonl"
        if not summary_file.exists():
            return {"error": "No performance data available"}
        
        all_metrics = []
        with open(summary_file, 'r') as f:
            for line in f:
                try:
                    metrics = json.loads(line)
                    if test_names is None or metrics['test_name'] in test_names:
                        all_metrics.append(metrics)
                except json.JSONDecodeError:
                    continue
        
        if not all_metrics:
            return {"error": "No matching performance data found"}
        
        # Group by test name
        by_test = {}
        for metrics in all_metrics:
            test_name = metrics['test_name']
            if test_name not in by_test:
                by_test[test_name] = []
            by_test[test_name].append(metrics)
        
        # Calculate statistics
        report = {}
        for test_name, metrics_list in by_test.items():
            durations = [m['duration_seconds'] for m in metrics_list]
            memory_peaks = [m['memory_peak_mb'] for m in metrics_list]
            cpu_averages = [m['cpu_average_percent'] for m in metrics_list]
            
            report[test_name] = {
                'count': len(metrics_list),
                'latest_timestamp': metrics_list[-1]['timestamp'],
                'duration': {
                    'min': min(durations),
                    'max': max(durations),
                    'mean': statistics.mean(durations),
                    'median': statistics.median(durations),
                    'stdev': statistics.stdev(durations) if len(durations) > 1 else 0
                },
                'memory_peak_mb': {
                    'min': min(memory_peaks),
                    'max': max(memory_peaks),
                    'mean': statistics.mean(memory_peaks),
                    'median': statistics.median(memory_peaks)
                },
                'cpu_average_percent': {
                    'min': min(cpu_averages),
                    'max': max(cpu_averages),
                    'mean': statistics.mean(cpu_averages),
                    'median': statistics.median(cpu_averages)
                },
                'success_rate': sum(1 for m in metrics_list if m['success']) / len(metrics_list)
            }
            
            # Add baseline comparison if available
            if test_name in self.baselines:
                baseline = self.baselines[test_name]
                report[test_name]['baseline_comparison'] = {
                    'duration_vs_p50': (statistics.median(durations) / baseline.duration_p50 - 1) * 100,
                    'duration_vs_p95': (max(durations) / baseline.duration_p95 - 1) * 100,
                    'memory_vs_limit': (max(memory_peaks) / baseline.memory_peak_limit - 1) * 100,
                    'cpu_vs_limit': (max(cpu_averages) / baseline.cpu_average_limit - 1) * 100
                }
        
        # Add recent alerts
        report['recent_alerts'] = [
            {'timestamp': alert[0].isoformat(), 'message': alert[1]}
            for alert in self.alerts[-10:]  # Last 10 alerts
        ]
        
        # Save report
        report_file = self.report_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# Video Processing Benchmark Functions
def simulate_video_processing(resolution: str = "720p", duration_seconds: float = 1.0) -> Dict[str, Any]:
    """Simulate video processing workload"""
    resolutions = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "4k": (3840, 2160)
    }
    
    width, height = resolutions.get(resolution, (1280, 720))
    fps = 30
    total_frames = int(fps * duration_seconds)
    frame_size = width * height * 3  # RGB
    
    frames_processed = 0
    start_time = time.perf_counter()
    
    if HAS_CV2:
        # Use actual video processing with OpenCV
        for i in range(total_frames):
            # Create random frame
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Simulate processing operations
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Simulate encoding
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            frames_processed += 1
            
            # Add some CPU-intensive computation
            _ = np.sum(np.sqrt(np.abs(frame.astype(np.float32))))
    else:
        # Mock video processing without OpenCV
        for i in range(total_frames):
            # Simulate frame processing with numpy operations
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Simulate processing operations
            gray = np.mean(frame, axis=2).astype(np.uint8)
            
            # Simulate some computation
            for _ in range(10):
                _ = np.convolve(gray.flatten()[:1000], np.ones(5)/5, mode='valid')
            
            frames_processed += 1
            
            # Add delay to simulate processing time
            time.sleep(0.001 * (width * height / (1280 * 720)))
    
    end_time = time.perf_counter()
    actual_duration = end_time - start_time
    actual_fps = frames_processed / actual_duration if actual_duration > 0 else 0
    throughput_mbps = (frames_processed * frame_size / (1024 * 1024)) / actual_duration if actual_duration > 0 else 0
    
    return {
        'frames_processed': frames_processed,
        'fps': actual_fps,
        'throughput_mbps': throughput_mbps,
        'resolution': resolution,
        'frame_size_mb': frame_size / (1024 * 1024)
    }


def simulate_batch_processing(batch_size: int = 10) -> Dict[str, Any]:
    """Simulate batch video processing"""
    results = []
    total_frames = 0
    
    for i in range(batch_size):
        # Process videos of varying lengths
        duration = np.random.uniform(0.5, 2.0)
        result = simulate_video_processing("720p", duration)
        results.append(result)
        total_frames += result['frames_processed']
    
    return {
        'batch_size': batch_size,
        'frames_processed': total_frames,
        'fps': sum(r['fps'] for r in results) / len(results),
        'throughput_mbps': sum(r['throughput_mbps'] for r in results)
    }


def simulate_memory_stress(size_mb: int = 1000, iterations: int = 10) -> Dict[str, Any]:
    """Simulate memory-intensive operations"""
    arrays = []
    bytes_per_mb = 1024 * 1024
    
    for i in range(iterations):
        # Allocate memory
        array = np.random.random(size_mb * bytes_per_mb // 8)  # 8 bytes per float64
        arrays.append(array)
        
        # Perform operations
        result = np.sum(array) + np.mean(array) + np.std(array)
        
        # Simulate some processing
        time.sleep(0.1)
    
    # Clean up
    total_memory_mb = sum(arr.nbytes / bytes_per_mb for arr in arrays)
    arrays.clear()
    
    return {
        'total_memory_allocated_mb': total_memory_mb,
        'iterations': iterations,
        'size_per_iteration_mb': size_mb
    }


def simulate_io_intensive(file_size_mb: int = 100, operations: int = 10) -> Dict[str, Any]:
    """Simulate I/O intensive operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        bytes_written = 0
        bytes_read = 0
        
        for i in range(operations):
            # Write operation
            file_path = tmpdir / f"test_file_{i}.bin"
            data = np.random.bytes(file_size_mb * 1024 * 1024)
            with open(file_path, 'wb') as f:
                f.write(data)
            bytes_written += len(data)
            
            # Read operation
            with open(file_path, 'rb') as f:
                read_data = f.read()
            bytes_read += len(read_data)
            
            # Simulate processing
            time.sleep(0.05)
    
    return {
        'total_bytes_written': bytes_written,
        'total_bytes_read': bytes_read,
        'operations': operations,
        'file_size_mb': file_size_mb
    }


class PerformanceTestSuite:
    """Complete test suite for performance benchmarking"""
    
    def __init__(self, benchmark: Optional[PerformanceBenchmark] = None):
        self.benchmark = benchmark or PerformanceBenchmark()
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, PerformanceMetrics]:
        """Run all performance tests"""
        results = {}
        
        if verbose:
            print("\n" + "="*60)
            print("PERFORMANCE BENCHMARK SUITE")
            print("="*60)
        
        # Test 1: 720p Video Processing
        if verbose:
            print("\n▶ Running 720p video processing benchmark...")
        results['video_processing_720p'] = self.benchmark.benchmark_function(
            simulate_video_processing,
            'video_processing_720p',
            resolution='720p',
            duration_seconds=1.0
        )
        if verbose:
            self._print_metrics(results['video_processing_720p'])
        
        # Test 2: 1080p Video Processing
        if verbose:
            print("\n▶ Running 1080p video processing benchmark...")
        results['video_processing_1080p'] = self.benchmark.benchmark_function(
            simulate_video_processing,
            'video_processing_1080p',
            resolution='1080p',
            duration_seconds=1.0
        )
        if verbose:
            self._print_metrics(results['video_processing_1080p'])
        
        # Test 3: 4K Video Processing
        if verbose:
            print("\n▶ Running 4K video processing benchmark...")
        results['video_processing_4k'] = self.benchmark.benchmark_function(
            simulate_video_processing,
            'video_processing_4k',
            resolution='4k',
            duration_seconds=0.5
        )
        if verbose:
            self._print_metrics(results['video_processing_4k'])
        
        # Test 4: Batch Processing
        if verbose:
            print("\n▶ Running batch processing benchmark...")
        results['batch_processing'] = self.benchmark.benchmark_function(
            simulate_batch_processing,
            'batch_processing',
            batch_size=5
        )
        if verbose:
            self._print_metrics(results['batch_processing'])
        
        # Test 5: Memory Stress Test
        if verbose:
            print("\n▶ Running memory stress test...")
        results['memory_stress_test'] = self.benchmark.benchmark_function(
            simulate_memory_stress,
            'memory_stress_test',
            size_mb=500,
            iterations=5
        )
        if verbose:
            self._print_metrics(results['memory_stress_test'])
        
        # Test 6: I/O Performance Test
        if verbose:
            print("\n▶ Running I/O performance test...")
        results['io_performance_test'] = self.benchmark.benchmark_function(
            simulate_io_intensive,
            'io_performance_test',
            file_size_mb=50,
            operations=5
        )
        if verbose:
            self._print_metrics(results['io_performance_test'])
        
        # Generate and display report
        if verbose:
            print("\n" + "="*60)
            print("PERFORMANCE REPORT")
            print("="*60)
            report = self.benchmark.generate_report()
            self._print_report(report)
        
        return results
    
    def _print_metrics(self, metrics: PerformanceMetrics):
        """Pretty print performance metrics"""
        print(f"  ✓ Duration: {metrics.duration_seconds:.3f}s")
        print(f"  ✓ Memory (Peak/Avg): {metrics.memory_peak_mb:.1f}/{metrics.memory_average_mb:.1f} MB")
        print(f"  ✓ CPU (Peak/Avg): {metrics.cpu_peak_percent:.1f}/{metrics.cpu_average_percent:.1f}%")
        print(f"  ✓ I/O (Read/Write): {metrics.io_read_mb:.1f}/{metrics.io_write_mb:.1f} MB")
        if metrics.fps > 0:
            print(f"  ✓ FPS: {metrics.fps:.1f}")
        if metrics.throughput_mbps > 0:
            print(f"  ✓ Throughput: {metrics.throughput_mbps:.1f} MB/s")
    
    def _print_report(self, report: Dict[str, Any]):
        """Pretty print performance report"""
        for test_name, stats in report.items():
            if test_name == 'recent_alerts':
                continue
            
            print(f"\n{test_name}:")
            print(f"  Runs: {stats['count']}")
            print(f"  Duration: {stats['duration']['median']:.3f}s (median)")
            print(f"  Memory Peak: {stats['memory_peak_mb']['max']:.1f} MB (max)")
            print(f"  CPU Average: {stats['cpu_average_percent']['mean']:.1f}% (mean)")
            print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
            
            if 'baseline_comparison' in stats:
                bc = stats['baseline_comparison']
                print(f"  vs Baseline:")
                print(f"    Duration: {bc['duration_vs_p50']:+.1f}% (median vs P50)")
                print(f"    Memory: {bc['memory_vs_limit']:+.1f}% (max vs limit)")
                print(f"    CPU: {bc['cpu_vs_limit']:+.1f}% (max vs limit)")
        
        if report.get('recent_alerts'):
            print("\n⚠️  Recent Alerts:")
            for alert in report['recent_alerts'][-5:]:  # Show last 5 alerts
                print(f"  • {alert['message']}")


def update_baseline_from_current(benchmark: PerformanceBenchmark, test_name: str, percentile: int = 95):
    """Update baseline based on current performance data"""
    report = benchmark.generate_report([test_name])
    
    if test_name not in report:
        print(f"No data available for {test_name}")
        return
    
    stats = report[test_name]
    
    # Create new baseline based on current performance
    new_baseline = PerformanceBaseline(
        test_name=test_name,
        duration_p50=stats['duration']['median'],
        duration_p95=stats['duration']['max'],  # Simplified: use max as P95
        memory_peak_limit=stats['memory_peak_mb']['max'] * 1.2,  # 20% buffer
        cpu_average_limit=min(stats['cpu_average_percent']['max'] * 1.1, 95.0),  # 10% buffer, max 95%
        min_fps=benchmark.baselines[test_name].min_fps if test_name in benchmark.baselines else 0,
        min_throughput_mbps=benchmark.baselines[test_name].min_throughput_mbps if test_name in benchmark.baselines else 0,
        io_read_limit_mb=stats.get('io_read_mb', {}).get('max', 100) * 1.2,
        io_write_limit_mb=stats.get('io_write_mb', {}).get('max', 100) * 1.2
    )
    
    benchmark.baselines[test_name] = new_baseline
    benchmark.save_baselines()
    print(f"Updated baseline for {test_name}")


def main():
    """Main entry point for running benchmarks"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Benchmarking System')
    parser.add_argument('--test', choices=['all', '720p', '1080p', '4k', 'batch', 'memory', 'io'],
                       default='all', help='Which test to run')
    parser.add_argument('--report-only', action='store_true',
                       help='Generate report from existing data without running tests')
    parser.add_argument('--update-baseline', type=str,
                       help='Update baseline for specified test based on recent runs')
    parser.add_argument('--regression-threshold', type=float, default=0.2,
                       help='Performance regression threshold (default: 0.2 = 20%%)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    benchmark.regression_threshold = args.regression_threshold
    
    if args.update_baseline:
        update_baseline_from_current(benchmark, args.update_baseline)
        return
    
    if args.report_only:
        report = benchmark.generate_report()
        print(json.dumps(report, indent=2))
        return
    
    suite = PerformanceTestSuite(benchmark)
    
    if args.test == 'all':
        suite.run_all_tests(verbose=args.verbose)
    else:
        test_map = {
            '720p': lambda: benchmark.benchmark_function(simulate_video_processing, 'video_processing_720p', '720p', 1.0),
            '1080p': lambda: benchmark.benchmark_function(simulate_video_processing, 'video_processing_1080p', '1080p', 1.0),
            '4k': lambda: benchmark.benchmark_function(simulate_video_processing, 'video_processing_4k', '4k', 0.5),
            'batch': lambda: benchmark.benchmark_function(simulate_batch_processing, 'batch_processing', 5),
            'memory': lambda: benchmark.benchmark_function(simulate_memory_stress, 'memory_stress_test', 500, 5),
            'io': lambda: benchmark.benchmark_function(simulate_io_intensive, 'io_performance_test', 50, 5)
        }
        
        if args.test in test_map:
            print(f"\nRunning {args.test} benchmark...")
            metrics = test_map[args.test]()
            if args.verbose:
                suite._print_metrics(metrics)
    
    # Display any alerts
    if benchmark.alerts:
        print("\n" + "="*60)
        print("PERFORMANCE ALERTS")
        print("="*60)
        for timestamp, alert in benchmark.alerts:
            print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {alert}")


if __name__ == "__main__":
    main()
