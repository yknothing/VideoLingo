"""
Performance Visualization and Reporting System

Provides comprehensive performance visualization and reporting:
1. Performance dashboard generation
2. Trend analysis charts
3. Bottleneck identification visualizations
4. Health score dashboards
5. Interactive performance reports

Generates HTML reports and charts for VideoLingo performance monitoring.
"""

import json
import statistics
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from core.utils.observability import log_event
from core.utils.performance_regression_detector import get_regression_detector, RegressionSeverity
from core.utils.performance_optimization_advisor import get_optimization_advisor


class PerformanceVisualizationGenerator:
    """Generate performance visualizations and reports"""
    
    def __init__(self, baseline_dir: str = None):
        self.baseline_dir = Path(baseline_dir or "reports/performance")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.regression_detector = get_regression_detector()
        self.optimization_advisor = get_optimization_advisor()
        
        # Load performance data
        self.load_performance_data()
    
    def load_performance_data(self):
        """Load performance data for visualization"""
        self.performance_data = []
        history_file = self.baseline_dir / "stage_performance_summary.jsonl"
        
        if not history_file.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(days=30)  # Last 30 days
        cutoff_timestamp = cutoff_time.timestamp()
        
        try:
            with open(history_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get('start_time', 0) >= cutoff_timestamp:
                            self.performance_data.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log_event("warning", f"Failed to load performance data: {e}")
    
    def generate_html_dashboard(self) -> str:
        """Generate comprehensive HTML performance dashboard"""
        # Get current performance health
        health_score = self.regression_detector.calculate_performance_health_score()
        optimization_report = self.optimization_advisor.generate_optimization_report()
        active_alerts = self.regression_detector.get_active_alerts()
        
        # Generate charts data
        performance_trends = self._generate_performance_trends()
        stage_performance = self._generate_stage_performance_data()
        resource_utilization = self._generate_resource_utilization_data()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VideoLingo Performance Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #666;
            font-size: 1.1em;
            margin-bottom: 15px;
        }}
        .metric-change {{
            font-size: 0.9em;
            padding: 5px 10px;
            border-radius: 20px;
            display: inline-block;
        }}
        .positive {{ background: #d4edda; color: #155724; }}
        .negative {{ background: #f8d7da; color: #721c24; }}
        .neutral {{ background: #e2e3e5; color: #383d41; }}
        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .alerts-section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .alert {{
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .alert-critical {{ background: #f8d7da; border-color: #dc3545; }}
        .alert-high {{ background: #fff3cd; border-color: #ffc107; }}
        .alert-medium {{ background: #d1ecf1; border-color: #17a2b8; }}
        .alert-low {{ background: #d4edda; border-color: #28a745; }}
        .recommendations {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .recommendation {{
            padding: 20px;
            margin-bottom: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: #fafafa;
        }}
        .recommendation-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .recommendation-impact {{
            display: inline-block;
            padding: 4px 8px;
            background: #667eea;
            color: white;
            border-radius: 12px;
            font-size: 0.8em;
            margin-right: 10px;
        }}
        .health-score {{
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            margin: 20px 0;
        }}
        .health-excellent {{ color: #28a745; }}
        .health-good {{ color: #17a2b8; }}
        .health-warning {{ color: #ffc107; }}
        .health-critical {{ color: #dc3545; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>VideoLingo Performance Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p>Monitoring {len(self.performance_data)} recent performance records</p>
        </div>

        <!-- Performance Health Score -->
        <div class="metric-card">
            <div class="health-score {'health-excellent' if health_score.overall_score >= 80 else 'health-good' if health_score.overall_score >= 60 else 'health-warning' if health_score.overall_score >= 40 else 'health-critical'}">
                {health_score.overall_score:.1f}/100
            </div>
            <div class="metric-label">Overall Performance Health</div>
            <div class="metric-change {'positive' if health_score.trend_direction == 'improving' else 'negative' if health_score.trend_direction == 'degrading' else 'neutral'}">
                {health_score.trend_direction.title()}
            </div>
        </div>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            {self._generate_key_metrics_html(optimization_report)}
        </div>

        <!-- Performance Alerts -->
        {self._generate_alerts_html(active_alerts)}

        <!-- Performance Trends Chart -->
        <div class="chart-container">
            <h2>Performance Trends (Last 30 Days)</h2>
            <div id="performance-trends-chart"></div>
        </div>

        <!-- Stage Performance Chart -->
        <div class="chart-container">
            <h2>Stage Performance Analysis</h2>
            <div id="stage-performance-chart"></div>
        </div>

        <!-- Resource Utilization Chart -->
        <div class="chart-container">
            <h2>Resource Utilization</h2>
            <div id="resource-utilization-chart"></div>
        </div>

        <!-- Optimization Recommendations -->
        <div class="recommendations">
            <h2>Optimization Recommendations</h2>
            {self._generate_recommendations_html(optimization_report.get('recommendations', [])[:5])}
        </div>

        <div class="footer">
            <p>VideoLingo Performance Monitoring System</p>
            <p>For technical support, check the performance logs in {self.baseline_dir}</p>
        </div>
    </div>

    <script>
        // Performance Trends Chart
        {self._generate_trends_chart_js(performance_trends)}

        // Stage Performance Chart
        {self._generate_stage_chart_js(stage_performance)}

        // Resource Utilization Chart
        {self._generate_resource_chart_js(resource_utilization)}
    </script>
</body>
</html>
        """
        
        return html_content
    
    def _generate_key_metrics_html(self, optimization_report: Dict[str, Any]) -> str:
        """Generate HTML for key metrics cards"""
        system_profile = optimization_report.get('system_profile', {})
        summary = optimization_report.get('optimization_summary', {})
        
        # Calculate average processing time
        avg_duration = 0
        if self.performance_data:
            durations = [r.get('duration_seconds', 0) for r in self.performance_data if r.get('duration_seconds')]
            if durations:
                avg_duration = statistics.mean(durations)
        
        # Calculate memory efficiency
        memory_efficiency = 75  # Default
        if system_profile.get('available_memory_gb', 0) > 0:
            memory_efficiency = min(100, (system_profile['available_memory_gb'] / system_profile.get('total_memory_gb', 1)) * 100)
        
        return f"""
        <div class="metric-card">
            <div class="metric-value">{avg_duration:.1f}s</div>
            <div class="metric-label">Average Processing Time</div>
            <div class="metric-change neutral">Per Stage</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value">{memory_efficiency:.0f}%</div>
            <div class="metric-label">Memory Efficiency</div>
            <div class="metric-change {'positive' if memory_efficiency > 70 else 'warning' if memory_efficiency > 50 else 'negative'}">
                {system_profile.get('available_memory_gb', 0):.1f}GB Available
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value">{summary.get('total_recommendations', 0)}</div>
            <div class="metric-label">Optimization Opportunities</div>
            <div class="metric-change {'negative' if summary.get('critical_recommendations', 0) > 0 else 'positive'}">
                {summary.get('critical_recommendations', 0)} Critical
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value">{system_profile.get('cpu_count', 0)}</div>
            <div class="metric-label">CPU Cores Available</div>
            <div class="metric-change {'positive' if system_profile.get('gpu_available') else 'neutral'}">
                {'GPU Available' if system_profile.get('gpu_available') else 'CPU Only'}
            </div>
        </div>
        """
    
    def _generate_alerts_html(self, active_alerts: List) -> str:
        """Generate HTML for active alerts"""
        if not active_alerts:
            return """
            <div class="alerts-section">
                <h2>Performance Alerts</h2>
                <div class="alert alert-low">
                    <strong>All Clear!</strong> No active performance alerts detected.
                </div>
            </div>
            """
        
        alerts_html = """
        <div class="alerts-section">
            <h2>Performance Alerts</h2>
        """
        
        for alert in active_alerts[:10]:  # Show top 10 alerts
            severity_class = f"alert-{alert.severity.value}"
            alerts_html += f"""
            <div class="alert {severity_class}">
                <strong>{alert.severity.value.upper()}:</strong> 
                {alert.regression_type.value.title()} regression in {alert.stage_name}
                <br>
                <small>
                    Current: {alert.current_value:.2f} | 
                    Baseline: {alert.baseline_value:.2f} | 
                    Change: {alert.degradation_percent:+.1f}% |
                    Confidence: {alert.confidence_score:.1%}
                </small>
            </div>
            """
        
        alerts_html += "</div>"
        return alerts_html
    
    def _generate_recommendations_html(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate HTML for optimization recommendations"""
        if not recommendations:
            return "<p>No specific recommendations available at this time.</p>"
        
        html = ""
        for rec in recommendations:
            impact_color = "#28a745" if rec['impact_score'] > 70 else "#ffc107" if rec['impact_score'] > 40 else "#17a2b8"
            html += f"""
            <div class="recommendation">
                <div class="recommendation-title">{rec['title']}</div>
                <div style="margin-bottom: 10px;">
                    <span class="recommendation-impact" style="background-color: {impact_color};">
                        {rec['impact_score']:.0f}% Impact
                    </span>
                    <span style="color: #666; font-size: 0.9em;">
                        {rec['implementation_effort'].title()} Effort | 
                        {rec['estimated_time_hours']:.0f}h Estimated
                    </span>
                </div>
                <p style="color: #555; margin-bottom: 15px;">{rec['description']}</p>
                <div style="font-size: 0.9em; color: #666;">
                    <strong>Affects:</strong> {', '.join(rec['affected_stages'])}
                </div>
            </div>
            """
        
        return html
    
    def _generate_performance_trends(self) -> Dict[str, Any]:
        """Generate performance trends data for charting"""
        if not self.performance_data:
            return {'timestamps': [], 'durations': [], 'memory': [], 'cpu': []}
        
        # Sort by timestamp
        sorted_data = sorted(self.performance_data, key=lambda x: x.get('start_time', 0))
        
        trends = {
            'timestamps': [],
            'durations': [],
            'memory': [],
            'cpu': []
        }
        
        for record in sorted_data:
            if record.get('start_time'):
                trends['timestamps'].append(datetime.fromtimestamp(record['start_time']).isoformat())
                trends['durations'].append(record.get('duration_seconds', 0))
                trends['memory'].append(record.get('memory_peak_mb', 0))
                trends['cpu'].append(record.get('cpu_avg_percent', 0))
        
        return trends
    
    def _generate_stage_performance_data(self) -> Dict[str, Any]:
        """Generate stage performance data for charting"""
        if not self.performance_data:
            return {'stages': [], 'avg_durations': [], 'avg_memory': []}
        
        # Group by stage
        stage_data = {}
        for record in self.performance_data:
            stage = record.get('stage_name', 'unknown')
            if stage not in stage_data:
                stage_data[stage] = {'durations': [], 'memory': []}
            
            if record.get('duration_seconds'):
                stage_data[stage]['durations'].append(record['duration_seconds'])
            if record.get('memory_peak_mb'):
                stage_data[stage]['memory'].append(record['memory_peak_mb'])
        
        stages = []
        avg_durations = []
        avg_memory = []
        
        for stage, data in stage_data.items():
            stages.append(stage)
            avg_durations.append(statistics.mean(data['durations']) if data['durations'] else 0)
            avg_memory.append(statistics.mean(data['memory']) if data['memory'] else 0)
        
        return {
            'stages': stages,
            'avg_durations': avg_durations,
            'avg_memory': avg_memory
        }
    
    def _generate_resource_utilization_data(self) -> Dict[str, Any]:
        """Generate resource utilization data for charting"""
        system_profile = self.optimization_advisor.system_profile
        
        return {
            'labels': ['CPU Cores', 'Memory (GB)', 'GPU Memory (GB)'],
            'available': [system_profile.cpu_count, system_profile.total_memory_gb, system_profile.gpu_memory_gb],
            'used': [
                system_profile.cpu_count * 0.7,  # Estimated average usage
                system_profile.total_memory_gb - system_profile.available_memory_gb,
                system_profile.gpu_memory_gb * 0.3 if system_profile.gpu_available else 0
            ]
        }
    
    def _generate_trends_chart_js(self, trends: Dict[str, Any]) -> str:
        """Generate JavaScript for performance trends chart"""
        return f"""
        var trace1 = {{
            x: {json.dumps(trends['timestamps'])},
            y: {json.dumps(trends['durations'])},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Duration (s)',
            line: {{color: '#667eea'}}
        }};
        
        var trace2 = {{
            x: {json.dumps(trends['timestamps'])},
            y: {json.dumps(trends['memory'])},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Memory (MB)',
            yaxis: 'y2',
            line: {{color: '#f093fb'}}
        }};
        
        var layout = {{
            title: 'Performance Trends Over Time',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Duration (seconds)', side: 'left'}},
            yaxis2: {{
                title: 'Memory (MB)',
                side: 'right',
                overlaying: 'y'
            }},
            showlegend: true
        }};
        
        Plotly.newPlot('performance-trends-chart', [trace1, trace2], layout);
        """
    
    def _generate_stage_chart_js(self, stage_data: Dict[str, Any]) -> str:
        """Generate JavaScript for stage performance chart"""
        return f"""
        var trace1 = {{
            x: {json.dumps(stage_data['stages'])},
            y: {json.dumps(stage_data['avg_durations'])},
            type: 'bar',
            name: 'Avg Duration (s)',
            marker: {{color: '#667eea'}}
        }};
        
        var layout = {{
            title: 'Average Duration by Pipeline Stage',
            xaxis: {{title: 'Pipeline Stage'}},
            yaxis: {{title: 'Average Duration (seconds)'}},
            showlegend: false
        }};
        
        Plotly.newPlot('stage-performance-chart', [trace1], layout);
        """
    
    def _generate_resource_chart_js(self, resource_data: Dict[str, Any]) -> str:
        """Generate JavaScript for resource utilization chart"""
        return f"""
        var trace1 = {{
            x: {json.dumps(resource_data['labels'])},
            y: {json.dumps(resource_data['available'])},
            type: 'bar',
            name: 'Available',
            marker: {{color: '#28a745'}}
        }};
        
        var trace2 = {{
            x: {json.dumps(resource_data['labels'])},
            y: {json.dumps(resource_data['used'])},
            type: 'bar',
            name: 'Used',
            marker: {{color: '#dc3545'}}
        }};
        
        var layout = {{
            title: 'System Resource Utilization',
            barmode: 'group',
            xaxis: {{title: 'Resource Type'}},
            yaxis: {{title: 'Amount'}},
            showlegend: true
        }};
        
        Plotly.newPlot('resource-utilization-chart', [trace1, trace2], layout);
        """
    
    def save_dashboard(self, filename: str = None) -> Path:
        """Save dashboard to HTML file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_dashboard_{timestamp}.html"
        
        dashboard_html = self.generate_html_dashboard()
        dashboard_file = self.baseline_dir / filename
        
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        # Also save as latest dashboard
        latest_file = self.baseline_dir / "latest_performance_dashboard.html"
        with open(latest_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        log_event("info", f"Performance dashboard saved to {dashboard_file}")
        
        return dashboard_file
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate text-based summary report"""
        health_score = self.regression_detector.calculate_performance_health_score()
        optimization_report = self.optimization_advisor.generate_optimization_report()
        active_alerts = self.regression_detector.get_active_alerts()
        
        # Calculate summary statistics
        total_records = len(self.performance_data)
        avg_duration = 0
        avg_memory = 0
        
        if self.performance_data:
            durations = [r.get('duration_seconds', 0) for r in self.performance_data if r.get('duration_seconds')]
            memory_values = [r.get('memory_peak_mb', 0) for r in self.performance_data if r.get('memory_peak_mb')]
            
            if durations:
                avg_duration = statistics.mean(durations)
            if memory_values:
                avg_memory = statistics.mean(memory_values)
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'performance_health': {
                'overall_score': health_score.overall_score,
                'trend': health_score.trend_direction,
                'critical_issues_count': len(health_score.critical_issues)
            },
            'performance_statistics': {
                'total_performance_records': total_records,
                'average_stage_duration_seconds': avg_duration,
                'average_memory_usage_mb': avg_memory,
                'monitoring_period_days': 30
            },
            'alerts_summary': {
                'total_active_alerts': len(active_alerts),
                'critical_alerts': len([a for a in active_alerts if a.severity == RegressionSeverity.CRITICAL]),
                'high_priority_alerts': len([a for a in active_alerts if a.severity.value == 'high']),
                'recent_alerts': [
                    {
                        'stage': alert.stage_name,
                        'type': alert.regression_type.value,
                        'severity': alert.severity.value,
                        'degradation_percent': alert.degradation_percent
                    }
                    for alert in active_alerts[:5]
                ]
            },
            'optimization_opportunities': {
                'total_recommendations': optimization_report['optimization_summary']['total_recommendations'],
                'potential_performance_gain': optimization_report['optimization_summary']['potential_performance_gain'],
                'estimated_implementation_time': optimization_report['optimization_summary']['estimated_implementation_time_hours'],
                'top_recommendations': [
                    {
                        'title': rec['title'],
                        'impact_score': rec['impact_score'],
                        'effort': rec['implementation_effort'],
                        'category': rec['category']
                    }
                    for rec in optimization_report.get('recommendations', [])[:3]
                ]
            },
            'system_health': {
                'cpu_cores': optimization_report['system_profile']['cpu_count'],
                'total_memory_gb': optimization_report['system_profile']['total_memory_gb'],
                'available_memory_gb': optimization_report['system_profile']['available_memory_gb'],
                'gpu_available': optimization_report['system_profile']['gpu_available'],
                'disk_type': optimization_report['system_profile']['disk_type']
            }
        }
        
        return report


# Global instance for easy access
_global_visualization_generator = None

def get_visualization_generator() -> PerformanceVisualizationGenerator:
    """Get global visualization generator instance"""
    global _global_visualization_generator
    if _global_visualization_generator is None:
        _global_visualization_generator = PerformanceVisualizationGenerator()
    return _global_visualization_generator
