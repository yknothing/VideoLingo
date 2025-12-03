# VideoLingo Performance Monitoring System

## Overview

VideoLingo's performance monitoring system provides comprehensive performance analysis, regression detection, and optimization recommendations for the 15-step video processing pipeline. The system is designed to help developers and operators maintain optimal performance across all stages of video processing.

## Features

### 🎯 Core Capabilities
- **Real-time Performance Monitoring**: Track CPU, memory, GPU, and I/O usage
- **Pipeline Stage Analysis**: Monitor each of the 15 processing stages individually
- **Performance Regression Detection**: Automated detection with statistical analysis
- **Optimization Recommendations**: AI-driven suggestions for performance improvements
- **User Experience Metrics**: Track the "5-minute success" user journey
- **CI/CD Integration**: Automated performance testing in GitHub Actions

### 📊 Key Metrics Tracked
- **Duration Metrics**: Processing time per stage and overall pipeline
- **Memory Metrics**: Peak usage, average usage, memory leaks detection
- **CPU Metrics**: Utilization patterns, bottleneck identification
- **API Metrics**: Call latency, success rates, timeout analysis
- **I/O Metrics**: Read/write throughput, disk utilization
- **User Experience**: Time to first success, error recovery time

## Quick Start

### 1. Basic Pipeline Monitoring

```python
from core.utils.performance_integration import monitor_pipeline_stage

@monitor_pipeline_stage("_1_ytdlp", 1)
def download_video(url: str, video_id: str = "") -> str:
    # Your video download implementation
    return download_video_file(url)

@monitor_pipeline_stage("_2_asr", 2) 
def transcribe_audio(audio_file: str, video_id: str = "") -> dict:
    # Your ASR implementation
    return transcription_result
```

### 2. Complete Pipeline Monitoring

```python
from core.utils.performance_integration import (
    start_pipeline_monitoring,
    complete_pipeline_monitoring,
    track_user_journey
)

def process_video_with_monitoring(video_url: str):
    video_id = generate_video_id(video_url)
    
    # Start pipeline tracking
    start_pipeline_monitoring(video_id)
    
    try:
        with track_user_journey("video_processing"):
            # Process through all 15 stages
            result = process_complete_pipeline(video_url, video_id)
            
        # Mark as successful
        complete_pipeline_monitoring(video_id, success=True)
        return result
        
    except Exception as e:
        # Mark as failed
        complete_pipeline_monitoring(video_id, success=False)
        raise
```

### 3. API Call Monitoring

```python
from core.utils.performance_integration import monitor_api

@monitor_api("openai", "translate")
def call_translation_api(text: str) -> str:
    # Your API call implementation
    return translated_text

@monitor_api("whisper", "transcribe")
def call_whisper_api(audio_file: str) -> dict:
    # Your ASR API call
    return transcription
```

## Architecture

### System Components

```
Performance Monitoring System
├── core/utils/performance_monitor.py          # Real-time monitoring
├── core/utils/performance_regression_detector.py  # Regression analysis
├── core/utils/performance_optimization_advisor.py # Optimization suggestions
├── core/utils/performance_visualization.py    # Dashboard generation
├── core/utils/performance_integration.py      # Central integration
└── scripts/performance_test_runner.py         # CI test runner
```

### Data Flow

```
Pipeline Stage → Performance Monitor → Metrics Collection → Analysis
                                                          ↓
Dashboard ← Visualization ← Optimization Advisor ← Regression Detector
```

## Performance Baselines

The system maintains performance baselines for each pipeline stage:

### Default Baselines (for reference)

| Stage | Expected Duration (P50) | Memory Limit | CPU Limit |
|-------|------------------------|--------------|-----------|
| `_1_ytdlp` | 30s | 500MB | 80% |
| `_2_asr` | 60s | 2GB | 90% |
| `_3_1_split_nlp` | 10s | 200MB | 60% |
| `_4_2_translate` | 45s | 300MB | 70% |
| `_10_gen_audio` | 120s | 1.5GB | 85% |

### Customizing Baselines

```python
from core.utils.performance_regression_detector import get_regression_detector

detector = get_regression_detector()

# Update baseline for a specific stage
detector.update_baseline(
    stage_name="_2_asr",
    metric_name="duration_seconds", 
    values=[45.2, 48.1, 43.8, 46.5, 44.9]  # Recent measurements
)
```

## Regression Detection

### Severity Levels

- **🟢 LOW (10-20% degradation)**: Monitor but don't block
- **🟡 MEDIUM (20-50% degradation)**: Investigate soon  
- **🟠 HIGH (50-100% degradation)**: High priority fix needed
- **🔴 CRITICAL (>100% degradation)**: Immediate action required

### Automated Alerts

The system automatically detects regressions using statistical analysis:

```python
from core.utils.performance_regression_detector import get_regression_detector

detector = get_regression_detector()

# Check for regression
alert = detector.detect_regression(
    stage_name="_2_asr",
    metric_name="duration_seconds",
    current_value=95.0,  # Current measurement
    video_id="abc123"
)

if alert:
    print(f"Regression detected: {alert.degradation_percent:.1f}% degradation")
```

## Optimization Recommendations

### Getting Recommendations

```python
from core.utils.performance_integration import get_optimization_recommendations

recommendations = get_optimization_recommendations(limit=5)

for rec in recommendations:
    print(f"Title: {rec['title']}")
    print(f"Impact: {rec['impact_score']:.0f}%")
    print(f"Effort: {rec['implementation_effort']}")
    print(f"Category: {rec['category']}")
```

### Example Recommendations

1. **Memory Streaming for _2_asr** (Impact: 80%)
   - Implement chunked processing to reduce memory usage
   - Expected 40% memory reduction, 10% duration increase

2. **Enable Parallel Processing for _10_gen_audio** (Impact: 90%)
   - Utilize multiple CPU cores for audio generation
   - Expected 60% duration reduction

3. **API Timeout Optimization** (Impact: 50%)
   - Implement exponential backoff for API calls
   - Reduce timeout-related failures by 75%

## Performance Dashboard

### Generating Dashboards

```python
from core.utils.performance_integration import generate_performance_dashboard

dashboard_path = generate_performance_dashboard()
print(f"Dashboard saved to: {dashboard_path}")
```

### Dashboard Features

- **Real-time Performance Metrics**: Current system status
- **Performance Trends**: 30-day historical analysis  
- **Stage Performance**: Individual stage analysis
- **Resource Utilization**: CPU, Memory, GPU usage
- **Active Alerts**: Current performance issues
- **Optimization Recommendations**: Top 5 suggestions

### Sample Dashboard Sections

```html
Performance Health Score: 87.5/100 (Good)
├── Memory Efficiency: 92% 
├── CPU Utilization: 78%
├── Active Alerts: 2 (1 Medium, 1 Low)
└── Optimization Opportunities: 3 High-Impact
```

## CI/CD Integration

### GitHub Actions Workflow

The system includes automated performance testing in CI/CD:

```yaml
# .github/workflows/performance_monitoring.yml
name: Performance Monitoring
on: [push, pull_request, schedule]

jobs:
  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Performance Tests
        run: python scripts/performance_test_runner.py --suite basic
```

### Performance Gates

Configure performance gates to prevent regression:

```json
{
  "performance_gates": {
    "max_regression_critical": 0,
    "max_regression_high": 2, 
    "min_health_score": 60.0,
    "max_duration_increase_percent": 50.0
  }
}
```

### Running Performance Tests

```bash
# Smoke test (fast, for PR checks)
python scripts/performance_test_runner.py --suite smoke

# Basic test (comprehensive, for main branch)
python scripts/performance_test_runner.py --suite basic --fail-on-regression

# Comprehensive test (detailed, for nightly runs)
python scripts/performance_test_runner.py --suite comprehensive
```

## User Experience Monitoring

### 5-Minute Success Path

Track the critical user journey from first install to first successful video processing:

```python
from core.utils.performance_integration import track_user_journey

with track_user_journey("first_install_to_first_success"):
    # Complete user onboarding and first video processing
    result = complete_first_time_user_flow()
    
# Automatically tracks:
# - Duration from start to first success
# - Error recovery times
# - User experience classification (excellent/good/acceptable/needs_improvement)
```

### User Experience Metrics

- **First Success Time**: Time to complete first video processing
- **Error Recovery Time**: Time to recover from failures
- **Success Rate**: Percentage of successful completions
- **User Journey Classification**:
  - Excellent: <5 minutes
  - Good: 5-15 minutes  
  - Acceptable: 15-30 minutes
  - Needs Improvement: >30 minutes

## Configuration

### Environment Variables

```bash
# Enable/disable performance monitoring
export PERFORMANCE_MONITORING_ENABLED=true

# Set performance monitoring level
export PERFORMANCE_MONITORING_LEVEL=detailed  # basic, detailed, debug

# Configure baseline directory
export PERFORMANCE_BASELINE_DIR=./reports/performance
```

### Config File Options

```yaml
# config.yaml
performance_monitoring:
  enabled: true
  monitoring_level: detailed
  baseline_auto_update: true
  dashboard_auto_generate: true
  ci_integration: true
  
  # Regression detection settings
  regression_detection:
    enabled: true
    confidence_threshold: 0.8
    alert_thresholds:
      low: 10.0      # 10% degradation
      medium: 20.0   # 20% degradation  
      high: 50.0     # 50% degradation
      critical: 100.0 # 100% degradation
```

## Troubleshooting

### Common Issues

1. **Performance Monitoring Not Working**
   ```python
   from core.utils.performance_integration import is_performance_monitoring_enabled
   
   if not is_performance_monitoring_enabled():
       from core.utils.performance_integration import enable_performance_monitoring
       enable_performance_monitoring()
   ```

2. **Missing Performance Data**
   ```bash
   # Check if reports directory exists
   ls -la reports/performance/
   
   # Verify observability is configured
   python -c "from core.utils.observability import log_event; log_event('info', 'test')"
   ```

3. **Dashboard Generation Fails**
   ```python
   # Check if required dependencies are installed
   pip install plotly psutil
   
   # Verify data availability
   from core.utils.performance_visualization import get_visualization_generator
   generator = get_visualization_generator()
   generator.load_performance_data()
   ```

### Debug Mode

Enable debug logging for detailed monitoring information:

```python
from core.utils.observability import init_logging
import logging

# Set debug level
logging.getLogger().setLevel(logging.DEBUG)
init_logging()
```

## Best Practices

### 1. Monitoring Strategy

- **Always monitor critical path stages**: `_1_ytdlp`, `_2_asr`, `_4_2_translate`, `_10_gen_audio`
- **Monitor API calls separately**: Track external API performance independently
- **Use video_id consistently**: Enables per-video performance analysis
- **Track user journeys**: Monitor complete user workflows, not just individual operations

### 2. Performance Optimization

- **Start with high-impact recommendations**: Focus on >70% impact score suggestions
- **Implement quick wins first**: Low-effort, high-impact improvements
- **Monitor resource utilization**: Keep memory <80%, CPU <85% average
- **Batch similar operations**: Group API calls and file operations when possible

### 3. CI/CD Integration

- **Use appropriate test suites**: Smoke for PRs, basic for main branch, comprehensive for nightly
- **Set realistic performance gates**: Balance quality with development velocity
- **Review performance trends weekly**: Don't wait for regressions to be critical
- **Update baselines regularly**: Reflect legitimate performance improvements

### 4. Dashboard Usage

- **Check health score daily**: Maintain >70 for production readiness
- **Review trends weekly**: Identify gradual performance degradation
- **Act on critical alerts immediately**: >100% degradation needs immediate attention
- **Implement top 3 recommendations quarterly**: Continuous performance improvement

## API Reference

### Core Functions

```python
# Monitoring
monitor_stage(stage_name, stage_number, video_id)
monitor_api_call(api_name, operation)
track_user_journey(journey_name)

# Analysis
check_performance_health() -> Dict[str, Any]
get_optimization_recommendations(limit=5) -> List[Dict]
generate_performance_dashboard() -> str

# Pipeline Tracking
start_pipeline_monitoring(video_id)
complete_pipeline_monitoring(video_id, success=True)

# Control
enable_performance_monitoring()
disable_performance_monitoring()
is_performance_monitoring_enabled() -> bool
```

### Decorators

```python
@monitor_pipeline_stage(stage_name, stage_number)
@monitor_api(api_name, operation="")
@track_user_experience(journey_name)
```

## Contributing

### Adding New Metrics

1. **Define the metric** in `performance_monitor.py`
2. **Add baseline** in `performance_regression_detector.py`  
3. **Include in optimization analysis** in `performance_optimization_advisor.py`
4. **Update visualization** in `performance_visualization.py`
5. **Add to integration** in `performance_integration.py`

### Extending Optimization Rules

```python
# In performance_optimization_advisor.py
def _generate_custom_recommendations(self, bottleneck):
    if bottleneck.stage_name == "my_custom_stage":
        return [
            OptimizationRecommendation(
                title="Custom optimization for my stage",
                description="Detailed description of the optimization",
                category=OptimizationCategory.CPU,
                priority=OptimizationPriority.HIGH,
                impact_score=75.0,
                # ... other fields
            )
        ]
```

## License

This performance monitoring system is part of VideoLingo and follows the same license terms.

## Support

For questions or issues related to performance monitoring:

1. Check the troubleshooting section above
2. Review logs in `reports/performance/`
3. Run the system test: `python core/utils/performance_integration.py`
4. Open an issue with performance report attached

---

**Performance monitoring is essential for maintaining VideoLingo's quality and user experience. Use this system to keep your video processing pipeline running optimally! 🚀**
