"""
VideoLingo Performance Monitoring Integration

Central integration point for performance monitoring system.
Provides simple APIs for integrating performance monitoring into VideoLingo pipeline stages.

Usage Examples:
1. Monitor a pipeline stage:
   from core.utils.performance_integration import monitor_stage
   
   @monitor_stage("_1_ytdlp", 1)
   def download_video(url):
       # Your implementation
       pass

2. Track API calls:
   from core.utils.performance_integration import track_api
   
   @track_api("openai_api")
   def call_openai_api():
       # Your implementation
       pass

3. Monitor user experience:
   from core.utils.performance_integration import track_user_journey
   
   with track_user_journey("first_success"):
       # User journey code
       pass
"""

import functools
from typing import Callable, Any, Dict, Optional
from contextlib import contextmanager

# Import all performance monitoring components
from core.utils.performance_monitor import (
    get_performance_tracker, 
    get_ux_tracker,
    monitor_performance,
    track_api_calls
)
from core.utils.performance_regression_detector import get_regression_detector
from core.utils.performance_optimization_advisor import get_optimization_advisor
from core.utils.performance_visualization import get_visualization_generator
from core.utils.observability import log_event, inc_counter, observe_histogram


class PerformanceIntegration:
    """Central performance monitoring integration"""
    
    def __init__(self):
        self.tracker = get_performance_tracker()
        self.ux_tracker = get_ux_tracker()
        self.regression_detector = get_regression_detector()
        self.optimization_advisor = get_optimization_advisor()
        self.visualization_generator = get_visualization_generator()
        
        # Performance monitoring enabled flag
        self._enabled = True
    
    def enable(self):
        """Enable performance monitoring"""
        self._enabled = True
        log_event("info", "Performance monitoring enabled")
    
    def disable(self):
        """Disable performance monitoring"""
        self._enabled = False
        log_event("info", "Performance monitoring disabled")
    
    @property
    def enabled(self) -> bool:
        """Check if performance monitoring is enabled"""
        return self._enabled
    
    @contextmanager
    def monitor_stage(self, stage_name: str, stage_number: int, video_id: str = ""):
        """Monitor a pipeline stage"""
        if not self._enabled:
            yield
            return
            
        with self.tracker.track_stage(stage_name, stage_number, video_id) as stage_metrics:
            yield stage_metrics
    
    @contextmanager 
    def monitor_api_call(self, api_name: str, operation: str = ""):
        """Monitor an API call"""
        if not self._enabled:
            yield
            return
            
        operation_name = f"{api_name}_{operation}" if operation else api_name
        
        import time
        start_time = time.time()
        
        try:
            yield
            
            # Success metrics
            duration = time.time() - start_time
            observe_histogram("api_call_duration_seconds", duration, 
                            api_name=api_name, operation=operation, success=True)
            inc_counter("api_calls_total", api_name=api_name, operation=operation, success=True)
            
        except Exception as e:
            # Failure metrics
            duration = time.time() - start_time
            observe_histogram("api_call_duration_seconds", duration,
                            api_name=api_name, operation=operation, success=False)
            inc_counter("api_calls_total", api_name=api_name, operation=operation, success=False)
            inc_counter("api_call_errors_total", api_name=api_name, 
                       operation=operation, error_type=type(e).__name__)
            
            log_event("warning", f"API call failed: {e}", 
                     api_name=api_name, operation=operation, duration_seconds=duration)
            raise
    
    @contextmanager
    def track_user_journey(self, journey_name: str):
        """Track a user experience journey"""
        if not self._enabled:
            yield
            return
            
        with self.ux_tracker.track_user_journey(journey_name):
            yield
    
    def check_performance_health(self) -> Dict[str, Any]:
        """Check current performance health"""
        if not self._enabled:
            return {"performance_monitoring_disabled": True}
        
        health_score = self.regression_detector.calculate_performance_health_score()
        active_alerts = self.regression_detector.get_active_alerts()
        
        return {
            "overall_health_score": health_score.overall_score,
            "trend_direction": health_score.trend_direction,
            "active_alerts_count": len(active_alerts),
            "critical_alerts_count": len([a for a in active_alerts if a.severity.value == "critical"]),
            "stage_scores": health_score.stage_scores,
            "recommendations_count": len(health_score.recommendations)
        }
    
    def get_optimization_recommendations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top optimization recommendations"""
        if not self._enabled:
            return []
        
        try:
            optimization_report = self.optimization_advisor.generate_optimization_report()
            recommendations = optimization_report.get('recommendations', [])
            
            return [
                {
                    'title': rec['title'],
                    'description': rec['description'],
                    'impact_score': rec['impact_score'],
                    'priority': rec['priority'],
                    'category': rec['category'],
                    'implementation_effort': rec['implementation_effort'],
                    'affected_stages': rec['affected_stages']
                }
                for rec in recommendations[:limit]
            ]
        except Exception as e:
            log_event("warning", f"Failed to get optimization recommendations: {e}")
            return []
    
    def generate_performance_dashboard(self) -> Optional[str]:
        """Generate performance dashboard"""
        if not self._enabled:
            return None
        
        try:
            dashboard_file = self.visualization_generator.save_dashboard()
            return str(dashboard_file)
        except Exception as e:
            log_event("error", f"Failed to generate performance dashboard: {e}")
            return None
    
    def start_pipeline_monitoring(self, video_id: str = ""):
        """Start monitoring a complete pipeline run"""
        if self._enabled:
            self.tracker.start_pipeline_tracking(video_id)
    
    def complete_pipeline_monitoring(self, video_id: str = "", success: bool = True):
        """Complete pipeline monitoring"""
        if self._enabled:
            return self.tracker.complete_pipeline_tracking(video_id, success)
        return None


# Global instance
_performance_integration = PerformanceIntegration()

# Convenience functions
def enable_performance_monitoring():
    """Enable global performance monitoring"""
    _performance_integration.enable()

def disable_performance_monitoring():
    """Disable global performance monitoring"""
    _performance_integration.disable()

def is_performance_monitoring_enabled() -> bool:
    """Check if performance monitoring is enabled"""
    return _performance_integration.enabled

@contextmanager
def monitor_stage(stage_name: str, stage_number: int, video_id: str = ""):
    """Monitor a pipeline stage"""
    with _performance_integration.monitor_stage(stage_name, stage_number, video_id) as stage_metrics:
        yield stage_metrics

@contextmanager
def monitor_api_call(api_name: str, operation: str = ""):
    """Monitor an API call"""
    with _performance_integration.monitor_api_call(api_name, operation):
        yield

@contextmanager
def track_user_journey(journey_name: str):
    """Track a user experience journey"""
    with _performance_integration.track_user_journey(journey_name):
        yield

def check_performance_health() -> Dict[str, Any]:
    """Get current performance health status"""
    return _performance_integration.check_performance_health()

def get_optimization_recommendations(limit: int = 5) -> List[Dict[str, Any]]:
    """Get optimization recommendations"""
    return _performance_integration.get_optimization_recommendations(limit)

def generate_performance_dashboard() -> Optional[str]:
    """Generate and save performance dashboard"""
    return _performance_integration.generate_performance_dashboard()

def start_pipeline_monitoring(video_id: str = ""):
    """Start monitoring a pipeline run"""
    _performance_integration.start_pipeline_monitoring(video_id)

def complete_pipeline_monitoring(video_id: str = "", success: bool = True):
    """Complete monitoring a pipeline run"""
    return _performance_integration.complete_pipeline_monitoring(video_id, success)

# Decorators for easy integration
def monitor_pipeline_stage(stage_name: str, stage_number: int):
    """Decorator to monitor a pipeline stage function"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract video_id from args/kwargs
            video_id = kwargs.get('video_id', '')
            if not video_id and args:
                # Common pattern: video_id as first argument
                if isinstance(args[0], str):
                    video_id = args[0]
            
            with monitor_stage(stage_name, stage_number, video_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def monitor_api(api_name: str, operation: str = ""):
    """Decorator to monitor an API call function"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with monitor_api_call(api_name, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def track_user_experience(journey_name: str):
    """Decorator to track user experience journey"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with track_user_journey(journey_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage patterns
def example_pipeline_stage_integration():
    """Example of how to integrate performance monitoring into a pipeline stage"""
    
    @monitor_pipeline_stage("_1_ytdlp", 1)
    def download_video(url: str, video_id: str = "") -> str:
        """Download video with performance monitoring"""
        import time
        
        # Simulate download process
        log_event("info", "Starting video download", stage="_1_ytdlp", video_id=video_id)
        
        # Monitor specific operations
        with monitor_api_call("youtube_api", "get_info"):
            time.sleep(0.1)  # Simulate API call
        
        with monitor_api_call("youtube_api", "download"):
            time.sleep(1.0)  # Simulate download
        
        log_event("info", "Video download completed", stage="_1_ytdlp", video_id=video_id)
        return "downloaded_video.mp4"
    
    return download_video

def example_complete_pipeline_integration():
    """Example of complete pipeline monitoring"""
    
    def process_video(video_url: str) -> bool:
        """Complete video processing with performance monitoring"""
        
        # Generate video ID
        import hashlib
        video_id = hashlib.md5(video_url.encode()).hexdigest()[:8]
        
        # Start pipeline monitoring
        start_pipeline_monitoring(video_id)
        
        try:
            # Track user experience
            with track_user_journey("video_processing"):
                
                # Stage 1: Download
                with monitor_stage("_1_ytdlp", 1, video_id):
                    with monitor_api_call("youtube", "download"):
                        # Download implementation
                        pass
                
                # Stage 2: ASR
                with monitor_stage("_2_asr", 2, video_id):
                    with monitor_api_call("whisper", "transcribe"):
                        # ASR implementation
                        pass
                
                # Stage 3: Translation
                with monitor_stage("_4_2_translate", 4, video_id):
                    with monitor_api_call("openai", "translate"):
                        # Translation implementation
                        pass
                
                # Additional stages...
                
            # Complete successfully
            pipeline_summary = complete_pipeline_monitoring(video_id, success=True)
            return True
            
        except Exception as e:
            # Complete with failure
            complete_pipeline_monitoring(video_id, success=False)
            log_event("error", f"Pipeline failed: {e}", video_id=video_id)
            return False
    
    return process_video


if __name__ == "__main__":
    # Performance monitoring system test
    print("VideoLingo Performance Monitoring System")
    print("=" * 50)
    
    # Check system status
    health = check_performance_health()
    print(f"Performance Health Score: {health.get('overall_health_score', 0):.1f}/100")
    print(f"Trend: {health.get('trend_direction', 'unknown')}")
    print(f"Active Alerts: {health.get('active_alerts_count', 0)}")
    
    # Get recommendations
    recommendations = get_optimization_recommendations(3)
    if recommendations:
        print(f"\nTop {len(recommendations)} Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} (Impact: {rec['impact_score']:.0f}%)")
    
    # Generate dashboard
    dashboard_path = generate_performance_dashboard()
    if dashboard_path:
        print(f"\nPerformance dashboard generated: {dashboard_path}")
    
    print("\nPerformance monitoring system is ready!")
