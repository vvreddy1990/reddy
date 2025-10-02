"""
Performance Monitoring Dashboard for AI/ML Operations

This module provides real-time performance metrics collection, trend analysis,
alerting, and automated performance tuning recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceAlert:
    """Performance alert information."""
    alert_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str
    metric: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    acknowledged: bool = False


@dataclass
class PerformanceTrend:
    """Performance trend analysis."""
    metric_name: str
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # 0-1, higher means stronger trend
    recent_average: float
    historical_average: float
    change_percent: float
    recommendation: str


class RealTimeMetricsCollector:
    """Collects real-time performance metrics."""
    
    def __init__(self, collection_interval: int = 5):
        self.collection_interval = collection_interval
        self.metrics_queue = queue.Queue()
        self.is_collecting = False
        self.collection_thread = None
        self.metrics_history = []
        self.max_history_size = 1000
        
    def start_collection(self):
        """Start real-time metrics collection."""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self.collection_thread.start()
        logger.info("Started real-time metrics collection")
    
    def stop_collection(self):
        """Stop real-time metrics collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped real-time metrics collection")
    
    def _collect_metrics(self):
        """Background thread for collecting metrics."""
        import psutil
        
        while self.is_collecting:
            try:
                # Collect system metrics
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                    'process_count': len(psutil.pids()),
                    'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
                }
                
                # Add to queue and history
                self.metrics_queue.put(metrics)
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the latest collected metrics."""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metrics for metrics in self.metrics_history
            if metrics['timestamp'] > cutoff_time
        ]


class PerformanceAlertManager:
    """Manages performance alerts and notifications."""
    
    def __init__(self):
        self.alerts: List[PerformanceAlert] = []
        self.alert_thresholds = {
            'cpu_percent': {'high': 80, 'critical': 95},
            'memory_percent': {'high': 85, 'critical': 95},
            'execution_time': {'high': 30, 'critical': 60},
            'cache_hit_rate': {'low': 30, 'critical': 10},
            'error_rate': {'medium': 5, 'high': 10, 'critical': 20}
        }
        self.alert_cooldown = {}  # Prevent spam alerts
        self.cooldown_period = 300  # 5 minutes
    
    def check_metrics_for_alerts(self, metrics: Dict[str, Any]) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts."""
        new_alerts = []
        current_time = datetime.now()
        
        for metric_name, value in metrics.items():
            if metric_name == 'timestamp':
                continue
            
            if metric_name not in self.alert_thresholds:
                continue
            
            thresholds = self.alert_thresholds[metric_name]
            alert_severity = None
            threshold_value = None
            
            # Check thresholds
            if 'critical' in thresholds and value >= thresholds['critical']:
                alert_severity = 'critical'
                threshold_value = thresholds['critical']
            elif 'high' in thresholds and value >= thresholds['high']:
                alert_severity = 'high'
                threshold_value = thresholds['high']
            elif 'medium' in thresholds and value >= thresholds['medium']:
                alert_severity = 'medium'
                threshold_value = thresholds['medium']
            elif 'low' in thresholds and value <= thresholds['low']:
                alert_severity = 'low'
                threshold_value = thresholds['low']
            
            # Generate alert if threshold exceeded and not in cooldown
            if alert_severity:
                alert_key = f"{metric_name}_{alert_severity}"
                
                if (alert_key not in self.alert_cooldown or 
                    (current_time - self.alert_cooldown[alert_key]).seconds > self.cooldown_period):
                    
                    alert = PerformanceAlert(
                        alert_id=f"{alert_key}_{int(current_time.timestamp())}",
                        severity=alert_severity,
                        component='system',
                        metric=metric_name,
                        current_value=value,
                        threshold=threshold_value,
                        message=self._generate_alert_message(metric_name, value, threshold_value, alert_severity),
                        timestamp=current_time
                    )
                    
                    new_alerts.append(alert)
                    self.alerts.append(alert)
                    self.alert_cooldown[alert_key] = current_time
        
        return new_alerts
    
    def _generate_alert_message(self, metric: str, value: float, threshold: float, severity: str) -> str:
        """Generate human-readable alert message."""
        messages = {
            'cpu_percent': f"CPU usage is {severity}: {value:.1f}% (threshold: {threshold}%)",
            'memory_percent': f"Memory usage is {severity}: {value:.1f}% (threshold: {threshold}%)",
            'execution_time': f"Execution time is {severity}: {value:.1f}s (threshold: {threshold}s)",
            'cache_hit_rate': f"Cache hit rate is {severity}: {value:.1f}% (threshold: {threshold}%)",
            'error_rate': f"Error rate is {severity}: {value:.1f}% (threshold: {threshold}%)"
        }
        
        return messages.get(metric, f"{metric} is {severity}: {value} (threshold: {threshold})")
    
    def get_active_alerts(self, severity_filter: Optional[str] = None) -> List[PerformanceAlert]:
        """Get active (unacknowledged) alerts."""
        active_alerts = [alert for alert in self.alerts if not alert.acknowledged]
        
        if severity_filter:
            active_alerts = [alert for alert in active_alerts if alert.severity == severity_filter]
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alerts by severity."""
        active_alerts = self.get_active_alerts()
        summary = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for alert in active_alerts:
            summary[alert.severity] += 1
        
        return summary


class PerformanceTrendAnalyzer:
    """Analyzes performance trends and provides insights."""
    
    def __init__(self):
        self.trend_window_hours = 24
        self.min_data_points = 10
    
    def analyze_trends(self, metrics_history: List[Dict[str, Any]]) -> List[PerformanceTrend]:
        """Analyze performance trends from historical data."""
        if len(metrics_history) < self.min_data_points:
            return []
        
        trends = []
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Analyze trends for key metrics
        metrics_to_analyze = ['cpu_percent', 'memory_percent', 'memory_available_mb']
        
        for metric in metrics_to_analyze:
            if metric not in df.columns:
                continue
            
            trend = self._analyze_metric_trend(df, metric)
            if trend:
                trends.append(trend)
        
        return trends
    
    def _analyze_metric_trend(self, df: pd.DataFrame, metric: str) -> Optional[PerformanceTrend]:
        """Analyze trend for a specific metric."""
        try:
            values = df[metric].dropna()
            if len(values) < self.min_data_points:
                return None
            
            # Calculate trend using linear regression
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]
            
            # Determine trend direction and strength
            trend_strength = abs(slope) / (values.std() + 1e-6)  # Normalize by standard deviation
            trend_strength = min(1.0, trend_strength)  # Cap at 1.0
            
            if abs(slope) < 0.01:
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'degrading' if metric in ['cpu_percent', 'memory_percent'] else 'improving'
            else:
                trend_direction = 'improving' if metric in ['cpu_percent', 'memory_percent'] else 'degrading'
            
            # Calculate averages
            recent_data = values.tail(min(10, len(values) // 4))  # Last 25% or 10 points
            recent_average = recent_data.mean()
            historical_average = values.mean()
            
            change_percent = ((recent_average - historical_average) / historical_average) * 100
            
            # Generate recommendation
            recommendation = self._generate_trend_recommendation(metric, trend_direction, change_percent)
            
            return PerformanceTrend(
                metric_name=metric,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                recent_average=recent_average,
                historical_average=historical_average,
                change_percent=change_percent,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend for {metric}: {e}")
            return None
    
    def _generate_trend_recommendation(self, metric: str, direction: str, change_percent: float) -> str:
        """Generate recommendation based on trend analysis."""
        recommendations = {
            ('cpu_percent', 'degrading'): "Consider optimizing CPU-intensive operations or increasing parallel processing limits",
            ('cpu_percent', 'improving'): "CPU usage is trending down - good optimization results",
            ('memory_percent', 'degrading'): "Memory usage is increasing - consider enabling memory optimization or increasing cache limits",
            ('memory_percent', 'improving'): "Memory usage is trending down - memory optimizations are working well",
            ('memory_available_mb', 'degrading'): "Available memory is decreasing - monitor for memory leaks",
            ('memory_available_mb', 'improving'): "Available memory is increasing - system is well optimized"
        }
        
        key = (metric, direction)
        base_recommendation = recommendations.get(key, f"{metric} is {direction}")
        
        if abs(change_percent) > 20:
            base_recommendation += f" (significant change: {change_percent:+.1f}%)"
        
        return base_recommendation


class AutoTuningEngine:
    """Provides automated performance tuning recommendations."""
    
    def __init__(self):
        self.tuning_rules = {
            'high_cpu_usage': {
                'condition': lambda metrics: metrics.get('cpu_percent', 0) > 80,
                'recommendation': 'Reduce parallel workers or enable CPU throttling',
                'auto_action': 'reduce_parallelism'
            },
            'high_memory_usage': {
                'condition': lambda metrics: metrics.get('memory_percent', 0) > 85,
                'recommendation': 'Enable memory optimization and reduce cache size',
                'auto_action': 'optimize_memory'
            },
            'low_cache_hit_rate': {
                'condition': lambda metrics: metrics.get('cache_hit_rate', 100) < 30,
                'recommendation': 'Increase cache size or adjust TTL settings',
                'auto_action': 'increase_cache_size'
            },
            'slow_execution': {
                'condition': lambda metrics: metrics.get('avg_execution_time', 0) > 30,
                'recommendation': 'Enable parallel processing or optimize algorithms',
                'auto_action': 'enable_parallelism'
            }
        }
        
        self.applied_tunings = []
    
    def analyze_and_recommend(self, metrics: Dict[str, Any], trends: List[PerformanceTrend]) -> List[Dict[str, Any]]:
        """Analyze metrics and trends to provide tuning recommendations."""
        recommendations = []
        
        # Check rule-based recommendations
        for rule_name, rule in self.tuning_rules.items():
            if rule['condition'](metrics):
                recommendations.append({
                    'type': 'rule_based',
                    'rule': rule_name,
                    'recommendation': rule['recommendation'],
                    'auto_action': rule['auto_action'],
                    'priority': self._calculate_priority(rule_name, metrics),
                    'confidence': 0.8
                })
        
        # Add trend-based recommendations
        for trend in trends:
            if trend.trend_direction == 'degrading' and trend.trend_strength > 0.5:
                recommendations.append({
                    'type': 'trend_based',
                    'metric': trend.metric_name,
                    'recommendation': trend.recommendation,
                    'auto_action': None,
                    'priority': trend.trend_strength,
                    'confidence': trend.trend_strength
                })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations
    
    def _calculate_priority(self, rule_name: str, metrics: Dict[str, Any]) -> float:
        """Calculate priority for a tuning recommendation."""
        priority_weights = {
            'high_cpu_usage': 0.9,
            'high_memory_usage': 0.95,
            'low_cache_hit_rate': 0.6,
            'slow_execution': 0.8
        }
        
        base_priority = priority_weights.get(rule_name, 0.5)
        
        # Adjust based on severity
        if rule_name == 'high_cpu_usage':
            cpu_percent = metrics.get('cpu_percent', 0)
            if cpu_percent > 95:
                base_priority = 1.0
            elif cpu_percent > 90:
                base_priority = 0.95
        
        elif rule_name == 'high_memory_usage':
            memory_percent = metrics.get('memory_percent', 0)
            if memory_percent > 95:
                base_priority = 1.0
            elif memory_percent > 90:
                base_priority = 0.98
        
        return base_priority
    
    def apply_auto_tuning(self, recommendation: Dict[str, Any], config_manager) -> bool:
        """Apply automatic tuning based on recommendation."""
        try:
            action = recommendation.get('auto_action')
            if not action:
                return False
            
            if action == 'reduce_parallelism':
                # Reduce parallel workers by 25%
                current_workers = config_manager.get_feature_config('performance_optimization').get('max_workers', 4)
                new_workers = max(1, int(current_workers * 0.75))
                config_manager.update_feature_config('performance_optimization', {'max_workers': new_workers})
                
            elif action == 'optimize_memory':
                # Enable memory optimization
                config_manager.update_feature_config('performance_optimization', {'enable_memory_optimization': True})
                
            elif action == 'increase_cache_size':
                # Increase cache size by 50%
                current_size = config_manager.get_feature_config('intelligent_caching').get('max_size_mb', 500)
                new_size = int(current_size * 1.5)
                config_manager.update_feature_config('intelligent_caching', {'max_size_mb': new_size})
                
            elif action == 'enable_parallelism':
                # Enable parallel processing
                config_manager.update_feature_config('performance_optimization', {'enable_parallel_processing': True})
            
            self.applied_tunings.append({
                'action': action,
                'recommendation': recommendation,
                'timestamp': datetime.now(),
                'success': True
            })
            
            logger.info(f"Applied auto-tuning: {action}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying auto-tuning {action}: {e}")
            self.applied_tunings.append({
                'action': action,
                'recommendation': recommendation,
                'timestamp': datetime.now(),
                'success': False,
                'error': str(e)
            })
            return False


class PerformanceDashboard:
    """Main performance monitoring dashboard."""
    
    def __init__(self):
        self.metrics_collector = RealTimeMetricsCollector()
        self.alert_manager = PerformanceAlertManager()
        self.trend_analyzer = PerformanceTrendAnalyzer()
        self.auto_tuning = AutoTuningEngine()
        
        # Dashboard state
        self.dashboard_data = {}
        self.last_update = datetime.now()
        
    def start_monitoring(self):
        """Start the performance monitoring system."""
        self.metrics_collector.start_collection()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the performance monitoring system."""
        self.metrics_collector.stop_collection()
        logger.info("Performance monitoring stopped")
    
    def update_dashboard_data(self, ai_ml_metrics: Optional[Dict[str, Any]] = None):
        """Update dashboard data with latest metrics."""
        try:
            # Get latest system metrics
            latest_metrics = self.metrics_collector.get_latest_metrics()
            if not latest_metrics:
                latest_metrics = {}
            
            # Add AI/ML specific metrics if provided
            if ai_ml_metrics:
                latest_metrics.update(ai_ml_metrics)
            
            # Check for alerts
            new_alerts = self.alert_manager.check_metrics_for_alerts(latest_metrics)
            
            # Get metrics history
            metrics_history = self.metrics_collector.get_metrics_history(hours=24)
            
            # Analyze trends
            trends = self.trend_analyzer.analyze_trends(metrics_history)
            
            # Get tuning recommendations
            recommendations = self.auto_tuning.analyze_and_recommend(latest_metrics, trends)
            
            # Update dashboard data
            self.dashboard_data = {
                'current_metrics': latest_metrics,
                'metrics_history': metrics_history,
                'alerts': {
                    'new_alerts': new_alerts,
                    'active_alerts': self.alert_manager.get_active_alerts(),
                    'alert_summary': self.alert_manager.get_alert_summary()
                },
                'trends': trends,
                'recommendations': recommendations,
                'last_update': datetime.now()
            }
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    def render_streamlit_dashboard(self):
        """Render the performance dashboard in Streamlit."""
        st.title("🚀 AI/ML Performance Monitoring Dashboard")
        
        # Auto-refresh
        if st.button("🔄 Refresh Dashboard"):
            self.update_dashboard_data()
        
        # Dashboard overview
        col1, col2, col3, col4 = st.columns(4)
        
        current_metrics = self.dashboard_data.get('current_metrics', {})
        alert_summary = self.dashboard_data.get('alerts', {}).get('alert_summary', {})
        
        with col1:
            cpu_percent = current_metrics.get('cpu_percent', 0)
            st.metric("CPU Usage", f"{cpu_percent:.1f}%", 
                     delta=f"{cpu_percent - 50:.1f}%" if cpu_percent > 50 else None)
        
        with col2:
            memory_percent = current_metrics.get('memory_percent', 0)
            st.metric("Memory Usage", f"{memory_percent:.1f}%",
                     delta=f"{memory_percent - 60:.1f}%" if memory_percent > 60 else None)
        
        with col3:
            cache_hit_rate = current_metrics.get('cache_hit_rate', 0)
            st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%",
                     delta=f"{cache_hit_rate - 70:.1f}%" if cache_hit_rate != 70 else None)
        
        with col4:
            total_alerts = sum(alert_summary.values())
            st.metric("Active Alerts", total_alerts,
                     delta=f"+{total_alerts}" if total_alerts > 0 else None)
        
        # Alerts section
        if alert_summary and sum(alert_summary.values()) > 0:
            st.subheader("🚨 Active Alerts")
            
            alert_cols = st.columns(4)
            colors = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🔵'}
            
            for i, (severity, count) in enumerate(alert_summary.items()):
                if count > 0:
                    with alert_cols[i]:
                        st.metric(f"{colors[severity]} {severity.title()}", count)
            
            # Show recent alerts
            active_alerts = self.dashboard_data.get('alerts', {}).get('active_alerts', [])
            if active_alerts:
                st.subheader("Recent Alerts")
                for alert in active_alerts[:5]:  # Show last 5 alerts
                    with st.expander(f"{colors[alert.severity]} {alert.message}"):
                        st.write(f"**Component:** {alert.component}")
                        st.write(f"**Metric:** {alert.metric}")
                        st.write(f"**Current Value:** {alert.current_value}")
                        st.write(f"**Threshold:** {alert.threshold}")
                        st.write(f"**Time:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        if st.button(f"Acknowledge", key=f"ack_{alert.alert_id}"):
                            self.alert_manager.acknowledge_alert(alert.alert_id)
                            st.success("Alert acknowledged")
                            st.experimental_rerun()
        
        # Performance trends
        st.subheader("📈 Performance Trends")
        
        metrics_history = self.dashboard_data.get('metrics_history', [])
        if metrics_history:
            df_history = pd.DataFrame(metrics_history)
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
            
            # Create performance charts
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Memory Available', 'Load Average'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # CPU Usage
            fig.add_trace(
                go.Scatter(x=df_history['timestamp'], y=df_history['cpu_percent'],
                          name='CPU %', line=dict(color='red')),
                row=1, col=1
            )
            
            # Memory Usage
            fig.add_trace(
                go.Scatter(x=df_history['timestamp'], y=df_history['memory_percent'],
                          name='Memory %', line=dict(color='blue')),
                row=1, col=2
            )
            
            # Memory Available
            if 'memory_available_mb' in df_history.columns:
                fig.add_trace(
                    go.Scatter(x=df_history['timestamp'], y=df_history['memory_available_mb'],
                              name='Available MB', line=dict(color='green')),
                    row=2, col=1
                )
            
            # Load Average
            if 'load_average' in df_history.columns:
                fig.add_trace(
                    go.Scatter(x=df_history['timestamp'], y=df_history['load_average'],
                              name='Load Avg', line=dict(color='orange')),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        trends = self.dashboard_data.get('trends', [])
        if trends:
            st.subheader("🔍 Trend Analysis")
            
            for trend in trends:
                trend_color = {
                    'improving': '🟢',
                    'degrading': '🔴',
                    'stable': '🟡'
                }[trend.trend_direction]
                
                with st.expander(f"{trend_color} {trend.metric_name.replace('_', ' ').title()} - {trend.trend_direction.title()}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Recent Average", f"{trend.recent_average:.2f}")
                    with col2:
                        st.metric("Historical Average", f"{trend.historical_average:.2f}")
                    with col3:
                        st.metric("Change", f"{trend.change_percent:+.1f}%")
                    
                    st.write(f"**Trend Strength:** {trend.trend_strength:.2f}")
                    st.write(f"**Recommendation:** {trend.recommendation}")
        
        # Performance recommendations
        recommendations = self.dashboard_data.get('recommendations', [])
        if recommendations:
            st.subheader("💡 Performance Recommendations")
            
            for i, rec in enumerate(recommendations[:5]):  # Show top 5 recommendations
                priority_color = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }
                
                priority = 'high' if rec['priority'] > 0.8 else 'medium' if rec['priority'] > 0.5 else 'low'
                
                with st.expander(f"{priority_color.get(priority, '🔵')} {rec['recommendation']}"):
                    st.write(f"**Type:** {rec['type'].replace('_', ' ').title()}")
                    st.write(f"**Priority:** {rec['priority']:.2f}")
                    st.write(f"**Confidence:** {rec['confidence']:.2f}")
                    
                    if rec.get('auto_action'):
                        if st.button(f"Apply Auto-Tuning", key=f"auto_{i}"):
                            # Note: Would need config_manager instance to actually apply
                            st.info("Auto-tuning would be applied here")
        
        # System information
        with st.expander("🖥️ System Information"):
            if current_metrics:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Current Metrics:**")
                    for key, value in current_metrics.items():
                        if key != 'timestamp':
                            if isinstance(value, float):
                                st.write(f"- {key.replace('_', ' ').title()}: {value:.2f}")
                            else:
                                st.write(f"- {key.replace('_', ' ').title()}: {value}")
                
                with col2:
                    st.write("**Last Update:**")
                    st.write(self.last_update.strftime('%Y-%m-%d %H:%M:%S'))
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data.copy()
    
    def export_performance_report(self, filepath: str):
        """Export performance report to file."""
        try:
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'dashboard_data': self.dashboard_data,
                'applied_tunings': self.auto_tuning.applied_tunings,
                'alert_history': [
                    {
                        'alert_id': alert.alert_id,
                        'severity': alert.severity,
                        'component': alert.component,
                        'metric': alert.metric,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'acknowledged': alert.acknowledged
                    }
                    for alert in self.alert_manager.alerts
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Performance report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")


# Convenience functions
def create_performance_dashboard() -> PerformanceDashboard:
    """Create a new performance dashboard instance."""
    return PerformanceDashboard()


def render_performance_dashboard_page():
    """Render the performance dashboard as a Streamlit page."""
    if 'performance_dashboard' not in st.session_state:
        st.session_state.performance_dashboard = create_performance_dashboard()
        st.session_state.performance_dashboard.start_monitoring()
    
    dashboard = st.session_state.performance_dashboard
    dashboard.update_dashboard_data()
    dashboard.render_streamlit_dashboard()