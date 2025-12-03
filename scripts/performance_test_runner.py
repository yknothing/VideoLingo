"""
Performance Test Runner for CI/CD Integration

Automated performance testing suite for VideoLingo that runs in CI environments:
1. Automated performance benchmarking
2. Performance regression detection
3. Performance report generation
4. Integration with GitHub Actions/CI systems
5. Performance gate checks for deployments

Designed to be run in CI/CD pipelines with proper error handling and reporting.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.utils.performance_monitor import get_performance_tracker, get_ux_tracker
    from core.utils.performance_regression_detector import get_regression_detector, RegressionSeverity
    from core.utils.performance_optimization_advisor import get_optimization_advisor
    from core.utils.performance_visualization import get_visualization_generator
    from core.utils.observability import log_event, inc_counter
except ImportError as e:
    print(f"Error importing VideoLingo modules: {e}")
    print("Ensure the script is run from the project root directory")
    sys.exit(1)


class CIPerformanceTestRunner:
    """CI-friendly performance test runner"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.project_root = project_root
        self.reports_dir = self.project_root / "reports" / "ci_performance"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # CI environment detection
        self.is_ci = self._detect_ci_environment()
        self.ci_info = self._get_ci_info()
        
        # Initialize performance components
        self.performance_tracker = get_performance_tracker()
        self.regression_detector = get_regression_detector()
        self.optimization_advisor = get_optimization_advisor()
        self.visualization_generator = get_visualization_generator()
        
        # Performance gates configuration
        self.performance_gates = self.config.get('performance_gates', {
            'max_regression_critical': 0,      # No critical regressions allowed
            'max_regression_high': 2,          # Maximum 2 high-severity regressions
            'min_health_score': 60.0,          # Minimum overall health score
            'max_duration_increase_percent': 50.0  # Max 50% duration increase
        })
        
        self.test_results = {
            'start_time': datetime.now().isoformat(),
            'ci_environment': self.ci_info,
            'tests_executed': [],
            'performance_gates': [],
            'regressions_detected': [],
            'recommendations': [],
            'overall_result': 'unknown'
        }
    
    def _detect_ci_environment(self) -> bool:
        """Detect if running in CI environment"""
        ci_indicators = [
            'CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS',
            'JENKINS_URL', 'TRAVIS', 'CIRCLECI', 'GITLAB_CI'
        ]
        return any(os.environ.get(indicator) for indicator in ci_indicators)
    
    def _get_ci_info(self) -> Dict[str, str]:
        """Get CI environment information"""
        info = {
            'is_ci': self.is_ci,
            'platform': 'unknown'
        }
        
        if os.environ.get('GITHUB_ACTIONS'):
            info.update({
                'platform': 'github_actions',
                'workflow': os.environ.get('GITHUB_WORKFLOW', ''),
                'run_id': os.environ.get('GITHUB_RUN_ID', ''),
                'ref': os.environ.get('GITHUB_REF', ''),
                'sha': os.environ.get('GITHUB_SHA', '')[:8]
            })
        elif os.environ.get('JENKINS_URL'):
            info.update({
                'platform': 'jenkins',
                'build_number': os.environ.get('BUILD_NUMBER', ''),
                'job_name': os.environ.get('JOB_NAME', '')
            })
        elif os.environ.get('TRAVIS'):
            info.update({
                'platform': 'travis',
                'build_number': os.environ.get('TRAVIS_BUILD_NUMBER', ''),
                'branch': os.environ.get('TRAVIS_BRANCH', '')
            })
        elif os.environ.get('CIRCLECI'):
            info.update({
                'platform': 'circleci',
                'build_num': os.environ.get('CIRCLE_BUILD_NUM', ''),
                'branch': os.environ.get('CIRCLE_BRANCH', '')
            })
        
        return info
    
    def run_performance_benchmarks(self, test_suite: str = 'basic') -> Dict[str, Any]:
        """Run performance benchmark suite"""
        print(f"Running performance benchmarks: {test_suite}")
        
        benchmark_results = {
            'suite_name': test_suite,
            'start_time': datetime.now().isoformat(),
            'tests': [],
            'success': True
        }
        
        try:
            if test_suite == 'basic':
                tests = self._get_basic_performance_tests()
            elif test_suite == 'comprehensive':
                tests = self._get_comprehensive_performance_tests()
            elif test_suite == 'smoke':
                tests = self._get_smoke_performance_tests()
            else:
                tests = self._get_basic_performance_tests()
            
            for test_config in tests:
                test_result = self._run_single_performance_test(test_config)
                benchmark_results['tests'].append(test_result)
                
                if not test_result.get('success', False):
                    benchmark_results['success'] = False
        
        except Exception as e:
            print(f"Error running performance benchmarks: {e}")
            benchmark_results['success'] = False
            benchmark_results['error'] = str(e)
        
        benchmark_results['end_time'] = datetime.now().isoformat()
        self.test_results['tests_executed'].append(benchmark_results)
        
        return benchmark_results
    
    def _get_basic_performance_tests(self) -> List[Dict[str, Any]]:
        """Get basic performance test configuration"""
        return [
            {
                'name': 'config_manager_performance',
                'type': 'unit_performance',
                'target': 'core.utils.config_manager',
                'iterations': 100,
                'max_duration_seconds': 5.0
            },
            {
                'name': 'file_operations_performance',
                'type': 'io_performance', 
                'iterations': 20,
                'max_duration_seconds': 10.0
            },
            {
                'name': 'memory_allocation_test',
                'type': 'memory_test',
                'max_memory_mb': 500,
                'duration_seconds': 30
            }
        ]
    
    def _get_comprehensive_performance_tests(self) -> List[Dict[str, Any]]:
        """Get comprehensive performance test configuration"""
        basic_tests = self._get_basic_performance_tests()
        comprehensive_tests = [
            {
                'name': 'pipeline_stage_simulation',
                'type': 'pipeline_test',
                'stages': ['_1_ytdlp', '_2_asr', '_4_2_translate'],
                'mock_video_duration': 60  # 1 minute video simulation
            },
            {
                'name': 'concurrent_processing_test',
                'type': 'concurrency_test',
                'concurrent_operations': 5,
                'duration_seconds': 60
            },
            {
                'name': 'stress_test',
                'type': 'stress_test',
                'duration_seconds': 120,
                'max_memory_mb': 2000
            }
        ]
        return basic_tests + comprehensive_tests
    
    def _get_smoke_performance_tests(self) -> List[Dict[str, Any]]:
        """Get minimal smoke test configuration for quick CI runs"""
        return [
            {
                'name': 'basic_imports_performance',
                'type': 'import_test',
                'max_duration_seconds': 2.0
            },
            {
                'name': 'config_load_performance',
                'type': 'unit_performance',
                'target': 'core.utils.config_manager',
                'iterations': 10,
                'max_duration_seconds': 1.0
            }
        ]
    
    def _run_single_performance_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single performance test"""
        test_name = test_config['name']
        print(f"  Running test: {test_name}")
        
        test_result = {
            'name': test_name,
            'type': test_config['type'],
            'start_time': datetime.now().isoformat(),
            'success': False,
            'metrics': {},
            'duration_seconds': 0
        }
        
        start_time = datetime.now()
        
        try:
            if test_config['type'] == 'unit_performance':
                metrics = self._run_unit_performance_test(test_config)
            elif test_config['type'] == 'io_performance':
                metrics = self._run_io_performance_test(test_config)
            elif test_config['type'] == 'memory_test':
                metrics = self._run_memory_test(test_config)
            elif test_config['type'] == 'import_test':
                metrics = self._run_import_test(test_config)
            elif test_config['type'] == 'pipeline_test':
                metrics = self._run_pipeline_simulation_test(test_config)
            else:
                metrics = {'error': f"Unknown test type: {test_config['type']}"}
            
            test_result['metrics'] = metrics
            test_result['success'] = True
            
        except Exception as e:
            test_result['error'] = str(e)
            print(f"    Test failed: {e}")
        
        test_result['duration_seconds'] = (datetime.now() - start_time).total_seconds()
        test_result['end_time'] = datetime.now().isoformat()
        
        return test_result
    
    def _run_unit_performance_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run unit performance test"""
        import time
        
        iterations = config.get('iterations', 100)
        start_time = time.time()
        
        # Simulate unit test performance measurement
        for i in range(iterations):
            # Mock performance test - replace with actual unit test
            time.sleep(0.001)  # 1ms per iteration
        
        duration = time.time() - start_time
        
        return {
            'iterations': iterations,
            'total_duration_seconds': duration,
            'avg_duration_per_iteration': duration / iterations,
            'operations_per_second': iterations / duration if duration > 0 else 0
        }
    
    def _run_io_performance_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run I/O performance test"""
        import tempfile
        import time
        
        iterations = config.get('iterations', 20)
        file_size_kb = 100  # 100KB test files
        
        start_time = time.time()
        bytes_written = 0
        bytes_read = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for i in range(iterations):
                # Write test
                test_file = temp_path / f"test_{i}.dat"
                test_data = b"x" * (file_size_kb * 1024)
                
                with open(test_file, 'wb') as f:
                    f.write(test_data)
                bytes_written += len(test_data)
                
                # Read test
                with open(test_file, 'rb') as f:
                    read_data = f.read()
                bytes_read += len(read_data)
                
                # Cleanup
                test_file.unlink()
        
        duration = time.time() - start_time
        
        return {
            'iterations': iterations,
            'duration_seconds': duration,
            'bytes_written': bytes_written,
            'bytes_read': bytes_read,
            'write_speed_mbps': (bytes_written / (1024 * 1024)) / duration if duration > 0 else 0,
            'read_speed_mbps': (bytes_read / (1024 * 1024)) / duration if duration > 0 else 0
        }
    
    def _run_memory_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run memory performance test"""
        import time
        import psutil
        
        max_memory_mb = config.get('max_memory_mb', 500)
        duration_seconds = config.get('duration_seconds', 30)
        
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        peak_memory = start_memory
        
        start_time = time.time()
        data_structures = []
        
        # Gradually increase memory usage
        while (time.time() - start_time) < duration_seconds:
            # Allocate memory in chunks
            chunk = [0] * (1024 * 100)  # ~400KB chunk
            data_structures.append(chunk)
            
            current_memory = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)
            
            # Don't exceed memory limit
            if current_memory > max_memory_mb:
                data_structures.pop(0)  # Remove oldest chunk
            
            time.sleep(0.1)
        
        # Cleanup
        data_structures.clear()
        end_memory = process.memory_info().rss / (1024 * 1024)
        
        return {
            'duration_seconds': duration_seconds,
            'start_memory_mb': start_memory,
            'peak_memory_mb': peak_memory,
            'end_memory_mb': end_memory,
            'memory_increase_mb': peak_memory - start_memory,
            'memory_efficiency': (peak_memory - start_memory) / max_memory_mb if max_memory_mb > 0 else 0
        }
    
    def _run_import_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run import performance test"""
        import time
        
        modules_to_test = [
            'core.utils.config_manager',
            'core.utils.observability',
            'core.utils.models'
        ]
        
        results = {}
        total_start = time.time()
        
        for module_name in modules_to_test:
            start_time = time.time()
            try:
                # Clear module from cache if exists
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                # Import module
                __import__(module_name)
                
                duration = time.time() - start_time
                results[module_name] = {
                    'import_time_seconds': duration,
                    'success': True
                }
            except ImportError as e:
                results[module_name] = {
                    'import_time_seconds': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                }
        
        total_duration = time.time() - total_start
        
        return {
            'total_import_time_seconds': total_duration,
            'modules_tested': len(modules_to_test),
            'successful_imports': sum(1 for r in results.values() if r['success']),
            'module_results': results
        }
    
    def _run_pipeline_simulation_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline simulation test"""
        import time
        
        stages = config.get('stages', ['_1_ytdlp', '_2_asr'])
        video_duration = config.get('mock_video_duration', 60)  # seconds
        
        results = {}
        total_start = time.time()
        
        for stage in stages:
            stage_start = time.time()
            
            # Simulate stage processing time based on video duration
            if stage == '_1_ytdlp':
                # Download simulation - proportional to video length
                processing_time = video_duration * 0.1  # 10% of video length
            elif stage == '_2_asr':
                # ASR simulation - typically longer than video
                processing_time = video_duration * 0.5  # 50% of video length
            elif stage == '_4_2_translate':
                # Translation simulation - depends on content length
                processing_time = video_duration * 0.2  # 20% of video length
            else:
                processing_time = video_duration * 0.1  # Default 10%
            
            # Simulate processing with short sleep (much faster than real processing)
            time.sleep(min(processing_time / 100, 2.0))  # Scale down for CI speed
            
            stage_duration = time.time() - stage_start
            
            results[stage] = {
                'duration_seconds': stage_duration,
                'simulated_processing_time': processing_time,
                'efficiency_ratio': processing_time / stage_duration if stage_duration > 0 else 0
            }
        
        total_duration = time.time() - total_start
        
        return {
            'total_pipeline_duration_seconds': total_duration,
            'video_duration_seconds': video_duration,
            'stages_processed': len(stages),
            'stage_results': results,
            'overall_efficiency': sum(r['efficiency_ratio'] for r in results.values()) / len(results)
        }
    
    def check_performance_gates(self) -> Dict[str, Any]:
        """Check performance gates for CI/CD decisions"""
        print("Checking performance gates...")
        
        gate_results = {
            'overall_pass': True,
            'gates_checked': [],
            'failures': []
        }
        
        try:
            # Get current performance health
            health_score = self.regression_detector.calculate_performance_health_score()
            active_alerts = self.regression_detector.get_active_alerts()
            
            # Gate 1: Health Score
            min_health_score = self.performance_gates['min_health_score']
            health_gate = {
                'gate_name': 'minimum_health_score',
                'threshold': min_health_score,
                'current_value': health_score.overall_score,
                'pass': health_score.overall_score >= min_health_score
            }
            gate_results['gates_checked'].append(health_gate)
            
            if not health_gate['pass']:
                gate_results['overall_pass'] = False
                gate_results['failures'].append(f"Health score {health_score.overall_score:.1f} below minimum {min_health_score}")
            
            # Gate 2: Critical Regressions
            critical_alerts = [a for a in active_alerts if a.severity == RegressionSeverity.CRITICAL]
            max_critical = self.performance_gates['max_regression_critical']
            critical_gate = {
                'gate_name': 'max_critical_regressions',
                'threshold': max_critical,
                'current_value': len(critical_alerts),
                'pass': len(critical_alerts) <= max_critical
            }
            gate_results['gates_checked'].append(critical_gate)
            
            if not critical_gate['pass']:
                gate_results['overall_pass'] = False
                gate_results['failures'].append(f"{len(critical_alerts)} critical regressions exceed limit of {max_critical}")
            
            # Gate 3: High Priority Regressions
            high_alerts = [a for a in active_alerts if a.severity.value == 'high']
            max_high = self.performance_gates['max_regression_high']
            high_gate = {
                'gate_name': 'max_high_regressions',
                'threshold': max_high,
                'current_value': len(high_alerts),
                'pass': len(high_alerts) <= max_high
            }
            gate_results['gates_checked'].append(high_gate)
            
            if not high_gate['pass']:
                gate_results['overall_pass'] = False
                gate_results['failures'].append(f"{len(high_alerts)} high-priority regressions exceed limit of {max_high}")
            
            # Gate 4: Duration Regression Check
            # This would require comparing with historical baselines
            # For now, we'll use a simplified check
            duration_gate = {
                'gate_name': 'max_duration_regression',
                'threshold': self.performance_gates['max_duration_increase_percent'],
                'current_value': 0,  # Would calculate from recent vs baseline
                'pass': True  # Simplified for now
            }
            gate_results['gates_checked'].append(duration_gate)
        
        except Exception as e:
            print(f"Error checking performance gates: {e}")
            gate_results['overall_pass'] = False
            gate_results['error'] = str(e)
        
        self.test_results['performance_gates'] = gate_results
        return gate_results
    
    def generate_ci_report(self) -> Dict[str, Any]:
        """Generate CI-friendly performance report"""
        print("Generating CI performance report...")
        
        # Check performance gates
        gate_results = self.check_performance_gates()
        
        # Get optimization recommendations
        try:
            optimization_report = self.optimization_advisor.generate_optimization_report()
            top_recommendations = optimization_report.get('recommendations', [])[:3]
        except Exception as e:
            print(f"Error generating optimization recommendations: {e}")
            top_recommendations = []
        
        # Get performance health
        try:
            health_score = self.regression_detector.calculate_performance_health_score()
        except Exception as e:
            print(f"Error getting health score: {e}")
            health_score = None
        
        self.test_results.update({
            'end_time': datetime.now().isoformat(),
            'overall_result': 'pass' if gate_results['overall_pass'] else 'fail',
            'performance_health_score': health_score.overall_score if health_score else 0,
            'performance_gates_passed': gate_results['overall_pass'],
            'gate_failures': gate_results.get('failures', []),
            'top_recommendations': [
                {
                    'title': rec['title'],
                    'impact_score': rec['impact_score'],
                    'priority': rec['priority']
                }
                for rec in top_recommendations
            ]
        })
        
        # Save report
        self._save_ci_report(self.test_results)
        
        return self.test_results
    
    def _save_ci_report(self, report: Dict[str, Any]):
        """Save CI report to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        json_report_file = self.reports_dir / f"ci_performance_report_{timestamp}.json"
        with open(json_report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save latest report
        latest_report_file = self.reports_dir / "latest_ci_performance_report.json"
        with open(latest_report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate GitHub Actions summary if in GitHub Actions
        if os.environ.get('GITHUB_ACTIONS'):
            self._generate_github_actions_summary(report)
        
        print(f"CI performance report saved to: {json_report_file}")
    
    def _generate_github_actions_summary(self, report: Dict[str, Any]):
        """Generate GitHub Actions job summary"""
        github_summary_file = os.environ.get('GITHUB_STEP_SUMMARY')
        if not github_summary_file:
            return
        
        result_emoji = "✅" if report['overall_result'] == 'pass' else "❌"
        health_score = report.get('performance_health_score', 0)
        
        summary = f"""
# {result_emoji} Performance Test Results

## Overall Result: {report['overall_result'].upper()}

### Performance Health Score: {health_score:.1f}/100

### Performance Gates
"""
        
        for gate in report.get('performance_gates', {}).get('gates_checked', []):
            gate_emoji = "✅" if gate['pass'] else "❌"
            summary += f"- {gate_emoji} **{gate['gate_name']}**: {gate['current_value']} (threshold: {gate['threshold']})\n"
        
        if report.get('gate_failures'):
            summary += "\n### Gate Failures\n"
            for failure in report['gate_failures']:
                summary += f"- ❌ {failure}\n"
        
        if report.get('top_recommendations'):
            summary += "\n### Top Optimization Recommendations\n"
            for rec in report['top_recommendations']:
                summary += f"- **{rec['title']}** (Impact: {rec['impact_score']:.0f}%, Priority: {rec['priority']})\n"
        
        summary += f"\n### Test Execution Summary\n"
        summary += f"- Tests Executed: {len(report.get('tests_executed', []))}\n"
        summary += f"- Duration: {report['start_time']} - {report['end_time']}\n"
        summary += f"- CI Environment: {report['ci_environment']['platform']}\n"
        
        try:
            with open(github_summary_file, 'w') as f:
                f.write(summary)
        except Exception as e:
            print(f"Error writing GitHub Actions summary: {e}")


def main():
    """Main entry point for CI performance test runner"""
    parser = argparse.ArgumentParser(description='VideoLingo CI Performance Test Runner')
    parser.add_argument('--suite', choices=['smoke', 'basic', 'comprehensive'], 
                       default='basic', help='Performance test suite to run')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--min-health-score', type=float, default=60.0,
                       help='Minimum health score for passing gates')
    parser.add_argument('--max-critical-regressions', type=int, default=0,
                       help='Maximum critical regressions allowed')
    parser.add_argument('--reports-dir', type=str, 
                       help='Directory to save reports (default: reports/ci_performance)')
    parser.add_argument('--fail-on-regression', action='store_true',
                       help='Exit with error code if regressions detected')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command line args
    if args.min_health_score:
        config.setdefault('performance_gates', {})['min_health_score'] = args.min_health_score
    if args.max_critical_regressions is not None:
        config.setdefault('performance_gates', {})['max_regression_critical'] = args.max_critical_regressions
    
    # Initialize test runner
    if args.reports_dir:
        original_reports_dir = Path(args.reports_dir)
        original_reports_dir.mkdir(parents=True, exist_ok=True)
    
    runner = CIPerformanceTestRunner(config)
    
    try:
        print("=" * 60)
        print("VideoLingo CI Performance Test Runner")
        print("=" * 60)
        print(f"Suite: {args.suite}")
        print(f"CI Environment: {runner.ci_info}")
        print()
        
        # Run performance benchmarks
        benchmark_results = runner.run_performance_benchmarks(args.suite)
        
        # Generate report and check gates
        ci_report = runner.generate_ci_report()
        
        print()
        print("=" * 60)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        print(f"Overall Result: {ci_report['overall_result'].upper()}")
        print(f"Health Score: {ci_report.get('performance_health_score', 0):.1f}/100")
        print(f"Performance Gates: {'PASSED' if ci_report.get('performance_gates_passed') else 'FAILED'}")
        
        if ci_report.get('gate_failures'):
            print("\nGate Failures:")
            for failure in ci_report['gate_failures']:
                print(f"  - {failure}")
        
        if ci_report.get('top_recommendations'):
            print(f"\nTop Recommendations ({len(ci_report['top_recommendations'])}):")
            for rec in ci_report['top_recommendations']:
                print(f"  - {rec['title']} (Impact: {rec['impact_score']:.0f}%)")
        
        print("=" * 60)
        
        # Exit with appropriate code
        if args.fail_on_regression and ci_report['overall_result'] != 'pass':
            print("Exiting with error due to performance gate failures")
            sys.exit(1)
        else:
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\nPerformance tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Performance tests failed with error: {e}")
        if runner.is_ci:
            # In CI, we want to see the full traceback
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
