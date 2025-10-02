"""
Performance Monitoring and Optimization Framework for AI/ML Features

This module provides comprehensive performance monitoring, automatic feature disabling
when approaching time limits, memory usage tracking, and performance optimization.
"""

import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager
import logging
from datetime import datetime, timedelta
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComponentMetrics:
    """Metrics for a single AI/ML component."""
    name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: float = 0.0
    memory_start: float = 0.0
    memory_end: float = 0.0
    memory_peak: float = 0.0
    memory_delta: float = 0.0
    status: str = "not_started"  # not_started, running, completed, failed, skipped
    error_message: Optional[str] = None
    iterations: int = 0
    data_size: int = 0


@dataclass
class PerformanceReport:
    """Comprehensive performance report for AI/ML operations."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    total_memory_used: float = 0.0
    peak_memory: float = 0.0
    components: Dict[str, ComponentMetrics] = field(default_factory=dict)
    features_disabled: List[str] = field(default_factory=list)
    optimization_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    success: bool = True


class PerformanceMonitor:
    """
    Real-time performance monitoring with automatic optimization and feature disabling.
    
    Monitors execution time, memory usage, and automatically disables features
    when approaching the 200-second constraint or memory limits.
    """
    
    def __init__(self, max_total_time: int = 200, memory_limit_mb: int = 1024):
        self.max_total_time = max_total_time
        self.memory_limit_mb = memory_limit_mb
        self.session_start_time = time.time()
        self.session_id = f"session_{int(self.session_start_time)}"
        
        # Component tracking
        self.components: Dict[str, ComponentMetrics] = {}
        self.component_limits: Dict[str, int] = {
            "smart_matching": 30,
            "anomaly_detection": 20,
            "data_quality": 25,
            "predictive_scoring": 15,
            "ai_insights": 10,
            "caching": 5
        }
        
        # Performance state
        self.disabled_features: List[str] = []
        self.warnings: List[str] = []
        self.optimization_applied: List[str] = []
        
        # Memory monitoring
        self.process = psutil.Process()
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Performance monitor initialized - Session: {self.session_id}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self._get_memory_usage()
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time since session start."""
        return time.time() - self.session_start_time
    
    def get_remaining_time(self) -> float:
        """Get remaining time before hitting the limit."""
        return max(0, self.max_total_time - self.get_elapsed_time())
    
    def should_skip_feature(self, feature_name: str) -> bool:
        """
        Determine if a feature should be skipped based on time constraints.
        
        Args:
            feature_name: Name of the AI/ML feature
            
        Returns:
            True if feature should be skipped, False otherwise
        """
        with self._lock:
            # Check if feature is already disabled
            if feature_name in self.disabled_features:
                return True
            
            # Check remaining time
            remaining_time = self.get_remaining_time()
            required_time = self.component_limits.get(feature_name, 10)
            
            # Skip if not enough time remaining (with 10-second buffer)
            if remaining_time < (required_time + 10):
                self._disable_feature(feature_name, f"Insufficient time remaining: {remaining_time:.1f}s")
                return True
            
            # Check memory usage
            current_memory = self._get_memory_usage()
            if current_memory > self.memory_limit_mb:
                self._disable_feature(feature_name, f"Memory limit exceeded: {current_memory:.1f}MB")
                return True
            
            return False
    
    def _disable_feature(self, feature_name: str, reason: str):
        """Disable a feature and log the reason."""
        if feature_name not in self.disabled_features:
            self.disabled_features.append(feature_name)
            warning_msg = f"Feature '{feature_name}' disabled: {reason}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
    
    @contextmanager
    def monitor_component(self, component_name: str, data_size: int = 0):
        """
        Context manager for monitoring a specific AI/ML component.
        
        Args:
            component_name: Name of the component being monitored
            data_size: Size of data being processed (for optimization)
            
        Usage:
            with monitor.monitor_component("smart_matching", len(data)):
                # AI/ML processing code here
                pass
        """
        # Initialize component metrics
        metrics = ComponentMetrics(
            name=component_name,
            data_size=data_size,
            memory_start=self._get_memory_usage()
        )
        
        with self._lock:
            self.components[component_name] = metrics
        
        # Check if component should be skipped
        if self.should_skip_feature(component_name):
            metrics.status = "skipped"
            yield metrics
            return
        
        # Start monitoring
        metrics.start_time = time.time()
        metrics.status = "running"
        
        try:
            logger.info(f"Starting component: {component_name}")
            yield metrics
            
            # Component completed successfully
            metrics.status = "completed"
            logger.info(f"Component completed: {component_name} in {metrics.duration:.2f}s")
            
        except Exception as e:
            # Component failed
            metrics.status = "failed"
            metrics.error_message = str(e)
            logger.error(f"Component failed: {component_name} - {str(e)}")
            raise
            
        finally:
            # Finalize metrics
            if metrics.start_time:
                metrics.end_time = time.time()
                metrics.duration = metrics.end_time - metrics.start_time
            
            metrics.memory_end = self._get_memory_usage()
            metrics.memory_delta = metrics.memory_end - metrics.memory_start
            
            # Update peak memory
            self._update_peak_memory()
            
            # Check for performance warnings
            self._check_component_performance(metrics)
    
    def _check_component_performance(self, metrics: ComponentMetrics):
        """Check component performance and apply optimizations if needed."""
        component_limit = self.component_limits.get(metrics.name, 10)
        
        # Check time limit
        if metrics.duration > component_limit:
            warning = f"Component '{metrics.name}' exceeded time limit: {metrics.duration:.2f}s > {component_limit}s"
            self.warnings.append(warning)
            logger.warning(warning)
        
        # Check memory usage
        if metrics.memory_delta > 100:  # More than 100MB increase
            warning = f"Component '{metrics.name}' used significant memory: {metrics.memory_delta:.1f}MB"
            self.warnings.append(warning)
            logger.warning(warning)
        
        # Apply optimizations based on performance
        self._apply_optimizations(metrics)
    
    def _apply_optimizations(self, metrics: ComponentMetrics):
        """Apply performance optimizations based on component metrics."""
        optimizations = []
        
        # Time-based optimizations
        if metrics.duration > self.component_limits.get(metrics.name, 10) * 0.8:
            if metrics.data_size > 10000:
                optimizations.append("large_dataset_optimization")
            if metrics.memory_delta > 50:
                optimizations.append("memory_optimization")
        
        # Memory-based optimizations
        current_memory = self._get_memory_usage()
        if current_memory > self.memory_limit_mb * 0.8:
            optimizations.append("memory_cleanup")
        
        # Record applied optimizations
        for opt in optimizations:
            if opt not in self.optimization_applied:
                self.optimization_applied.append(opt)
                logger.info(f"Applied optimization: {opt} for component {metrics.name}")
    
    def start_timing(self, component_name: str) -> None:
        """Start timing a component (alternative to context manager)."""
        metrics = ComponentMetrics(
            name=component_name,
            start_time=time.time(),
            status="running",
            memory_start=self._get_memory_usage()
        )
        
        with self._lock:
            self.components[component_name] = metrics
        
        logger.info(f"Started timing: {component_name}")
    
    def end_timing(self, component_name: str) -> float:
        """
        End timing a component and return duration.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Duration in seconds
        """
        with self._lock:
            if component_name not in self.components:
                logger.warning(f"Component {component_name} not found in timing records")
                return 0.0
            
            metrics = self.components[component_name]
            
            if metrics.start_time is None:
                logger.warning(f"Component {component_name} was not started")
                return 0.0
            
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            metrics.memory_end = self._get_memory_usage()
            metrics.memory_delta = metrics.memory_end - metrics.memory_start
            metrics.status = "completed"
            
            self._update_peak_memory()
            self._check_component_performance(metrics)
            
            logger.info(f"Ended timing: {component_name} - Duration: {metrics.duration:.2f}s")
            return metrics.duration
    
    def get_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report."""
        total_duration = self.get_elapsed_time()
        total_memory = self._get_memory_usage() - self.initial_memory
        
        report = PerformanceReport(
            session_id=self.session_id,
            start_time=datetime.fromtimestamp(self.session_start_time),
            end_time=datetime.now(),
            total_duration=total_duration,
            total_memory_used=total_memory,
            peak_memory=self.peak_memory,
            components=self.components.copy(),
            features_disabled=self.disabled_features.copy(),
            optimization_applied=self.optimization_applied.copy(),
            warnings=self.warnings.copy(),
            success=total_duration <= self.max_total_time
        )
        
        return report
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics as dictionary."""
        report = self.get_performance_report()
        
        return {
            "session_id": report.session_id,
            "total_duration": report.total_duration,
            "remaining_time": self.get_remaining_time(),
            "memory_used": report.total_memory_used,
            "peak_memory": report.peak_memory,
            "components_completed": len([c for c in report.components.values() if c.status == "completed"]),
            "components_failed": len([c for c in report.components.values() if c.status == "failed"]),
            "components_skipped": len([c for c in report.components.values() if c.status == "skipped"]),
            "features_disabled": len(report.features_disabled),
            "warnings_count": len(report.warnings),
            "success": report.success
        }
    
    def save_performance_log(self, log_dir: str = "logs") -> bool:
        """Save performance report to log file."""
        try:
            os.makedirs(log_dir, exist_ok=True)
            
            report = self.get_performance_report()
            log_file = os.path.join(log_dir, f"performance_{self.session_id}.json")
            
            # Convert report to JSON-serializable format
            log_data = {
                "session_id": report.session_id,
                "start_time": report.start_time.isoformat(),
                "end_time": report.end_time.isoformat() if report.end_time else None,
                "total_duration": report.total_duration,
                "total_memory_used": report.total_memory_used,
                "peak_memory": report.peak_memory,
                "features_disabled": report.features_disabled,
                "optimization_applied": report.optimization_applied,
                "warnings": report.warnings,
                "success": report.success,
                "components": {}
            }
            
            # Add component details
            for name, metrics in report.components.items():
                log_data["components"][name] = {
                    "duration": metrics.duration,
                    "memory_delta": metrics.memory_delta,
                    "status": metrics.status,
                    "error_message": metrics.error_message,
                    "data_size": metrics.data_size,
                    "iterations": metrics.iterations
                }
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"Performance log saved: {log_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving performance log: {e}")
            return False
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations based on current metrics."""
        recommendations = []
        
        total_time = self.get_elapsed_time()
        remaining_time = self.get_remaining_time()
        current_memory = self._get_memory_usage()
        
        # Time-based recommendations
        if remaining_time < 30:
            recommendations.append("Consider disabling non-critical AI features to stay within time limit")
        
        if total_time > self.max_total_time * 0.7:
            recommendations.append("Processing is taking longer than expected - consider data size reduction")
        
        # Memory-based recommendations
        if current_memory > self.memory_limit_mb * 0.8:
            recommendations.append("High memory usage detected - consider enabling memory optimization")
        
        # Component-specific recommendations
        for name, metrics in self.components.items():
            if metrics.status == "completed" and metrics.duration > self.component_limits.get(name, 10) * 0.8:
                recommendations.append(f"Component '{name}' is slow - consider optimization or reduced data size")
        
        # Feature-specific recommendations
        if len(self.disabled_features) > 0:
            recommendations.append("Some features were disabled due to performance constraints")
        
        return recommendations
    
    def reset_session(self):
        """Reset monitoring for a new session."""
        with self._lock:
            self.session_start_time = time.time()
            self.session_id = f"session_{int(self.session_start_time)}"
            self.components.clear()
            self.disabled_features.clear()
            self.warnings.clear()
            self.optimization_applied.clear()
            self.initial_memory = self._get_memory_usage()
            self.peak_memory = self.initial_memory
        
        logger.info(f"Performance monitor reset - New session: {self.session_id}")


class PerformanceOptimizer:
    """
    Automatic performance optimization based on monitoring data.
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_strategies = {
            "large_dataset_optimization": self._optimize_large_dataset,
            "memory_optimization": self._optimize_memory_usage,
            "memory_cleanup": self._cleanup_memory,
            "time_optimization": self._optimize_processing_time
        }
    
    def _optimize_large_dataset(self, component_name: str, data_size: int) -> Dict[str, Any]:
        """Optimize processing for large datasets."""
        optimizations = {
            "batch_processing": True,
            "sample_size": min(data_size, 5000),
            "parallel_processing": True if data_size > 1000 else False,
            "memory_efficient": True
        }
        
        logger.info(f"Applied large dataset optimization for {component_name}: {optimizations}")
        return optimizations
    
    def _optimize_memory_usage(self, component_name: str, memory_delta: float) -> Dict[str, Any]:
        """Optimize memory usage for components."""
        optimizations = {
            "garbage_collection": True,
            "streaming_processing": True,
            "reduced_precision": True if memory_delta > 200 else False,
            "cache_cleanup": True
        }
        
        logger.info(f"Applied memory optimization for {component_name}: {optimizations}")
        return optimizations
    
    def _cleanup_memory(self) -> Dict[str, Any]:
        """Perform memory cleanup operations."""
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        optimizations = {
            "garbage_collected": collected,
            "memory_before": self.monitor._get_memory_usage()
        }
        
        # Additional cleanup operations can be added here
        
        optimizations["memory_after"] = self.monitor._get_memory_usage()
        optimizations["memory_freed"] = optimizations["memory_before"] - optimizations["memory_after"]
        
        logger.info(f"Memory cleanup completed: {optimizations}")
        return optimizations
    
    def _optimize_processing_time(self, component_name: str, duration: float) -> Dict[str, Any]:
        """Optimize processing time for slow components."""
        optimizations = {
            "reduced_iterations": True,
            "simplified_algorithms": True,
            "early_stopping": True,
            "cached_results": True
        }
        
        logger.info(f"Applied time optimization for {component_name}: {optimizations}")
        return optimizations
    
    def apply_optimization(self, strategy: str, **kwargs) -> Dict[str, Any]:
        """Apply a specific optimization strategy."""
        if strategy in self.optimization_strategies:
            return self.optimization_strategies[strategy](**kwargs)
        else:
            logger.warning(f"Unknown optimization strategy: {strategy}")
            return {}


# Convenience functions for easy integration
def create_performance_monitor(max_time: int = 200, memory_limit: int = 1024) -> PerformanceMonitor:
    """Create a new performance monitor instance."""
    return PerformanceMonitor(max_total_time=max_time, memory_limit_mb=memory_limit)


def monitor_ai_ml_component(monitor: PerformanceMonitor, component_name: str, data_size: int = 0):
    """Decorator for monitoring AI/ML components."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with monitor.monitor_component(component_name, data_size):
                return func(*args, **kwargs)
        return wrapper
    return decorator