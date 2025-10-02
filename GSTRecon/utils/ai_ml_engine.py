"""
AI/ML Engine Foundation for GST Reconciliation Enhancement

This module provides the main AIMLEngine class that coordinates all AI/ML features
with comprehensive error handling, graceful fallback mechanisms, and performance monitoring.
"""

import logging
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
import pandas as pd

from ai_ml_config import AIMLConfigManager, AIMLConfig
from performance_monitor import PerformanceMonitor, PerformanceReport
from intelligent_cache import IntelligentCache
from performance_optimizer import PerformanceOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancementResults:
    """Results from AI/ML enhancements with original data preservation."""
    original_results: Dict[str, Any]
    ai_enhancements: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    features_applied: List[str] = field(default_factory=list)
    features_skipped: List[str] = field(default_factory=list)
    success: bool = True


@dataclass
class AIMLError:
    """Structured error information for AI/ML operations."""
    component: str
    error_type: str
    message: str
    timestamp: datetime
    traceback: Optional[str] = None
    recoverable: bool = True


class AIMLErrorHandler:
    """Comprehensive error handling for AI/ML operations."""
    
    def __init__(self):
        self.errors: List[AIMLError] = []
        self.error_counts: Dict[str, int] = {}
        self.disabled_features: List[str] = []
        self.max_errors_per_component = 3
    
    def log_error(
        self,
        component: str,
        error: Exception,
        context: Dict[str, Any] = None,
        recoverable: bool = True
    ) -> AIMLError:
        """
        Log an error with context information.
        
        Args:
            component: Name of the AI/ML component that failed
            error: The exception that occurred
            context: Additional context information
            recoverable: Whether the error is recoverable
            
        Returns:
            AIMLError object with structured error information
        """
        error_obj = AIMLError(
            component=component,
            error_type=type(error).__name__,
            message=str(error),
            timestamp=datetime.now(),
            traceback=traceback.format_exc(),
            recoverable=recoverable
        )
        
        self.errors.append(error_obj)
        
        # Update error counts
        self.error_counts[component] = self.error_counts.get(component, 0) + 1
        
        # Log the error
        log_msg = f"AI/ML Error in {component}: {error_obj.message}"
        if context:
            log_msg += f" | Context: {context}"
        
        if recoverable:
            logger.warning(log_msg)
        else:
            logger.error(log_msg)
        
        # Check if component should be disabled
        if self.error_counts[component] >= self.max_errors_per_component:
            self._disable_component(component)
        
        return error_obj
    
    def _disable_component(self, component: str):
        """Disable a component due to repeated errors."""
        if component not in self.disabled_features:
            self.disabled_features.append(component)
            logger.error(f"Component '{component}' disabled due to repeated errors")
    
    def is_component_disabled(self, component: str) -> bool:
        """Check if a component is disabled due to errors."""
        return component in self.disabled_features
    
    def get_error_summary(self) -> List[str]:
        """Get a summary of all errors."""
        summary = []
        
        for component, count in self.error_counts.items():
            if count > 0:
                status = "DISABLED" if component in self.disabled_features else "ACTIVE"
                summary.append(f"{component}: {count} errors ({status})")
        
        return summary
    
    def reset_component_errors(self, component: str):
        """Reset error count for a specific component."""
        self.error_counts[component] = 0
        if component in self.disabled_features:
            self.disabled_features.remove(component)
        logger.info(f"Reset errors for component: {component}")


class AIMLEngine:
    """
    Main AI/ML engine that coordinates all enhancement features with
    comprehensive error handling and performance monitoring.
    """
    
    def __init__(self, config_file: str = "reconciliation_settings.json"):
        """
        Initialize the AI/ML engine.
        
        Args:
            config_file: Path to the configuration file
        """
        # Initialize core components
        self.config_manager = AIMLConfigManager(config_file)
        self.config = self.config_manager.config
        self.error_handler = AIMLErrorHandler()
        
        # Initialize performance monitoring
        performance_limits = self.config_manager.get_performance_limits()
        self.performance_monitor = PerformanceMonitor(
            max_total_time=performance_limits["total_time"],
            memory_limit_mb=performance_limits["memory_mb"]
        )
        
        # Initialize intelligent caching
        cache_config = self.config_manager.get_feature_config("intelligent_caching")
        self.cache = IntelligentCache(
            cache_dir=cache_config.get("cache_dir", ".cache/ai_ml"),
            max_size_mb=cache_config.get("max_size_mb", 500),
            default_ttl_hours=cache_config.get("ttl_hours", 24),
            compression_enabled=cache_config.get("compression", True),
            enable_prediction=cache_config.get("enable_prediction", True),
            enable_preloading=cache_config.get("enable_preloading", True)
        )
        
        # Initialize performance optimizer
        optimizer_config = self.config_manager.get_feature_config("performance_optimization")
        self.performance_optimizer = PerformanceOptimizer(
            max_memory_mb=optimizer_config.get("max_memory_mb", 1024),
            max_workers=optimizer_config.get("max_workers", None)
        )
        
        # Feature registry (will be populated by individual feature modules)
        self.feature_processors: Dict[str, Callable] = {}
        
        # Register built-in processors
        self._register_builtin_processors()
        
        logger.info("AI/ML Engine initialized successfully")
    
    def _register_builtin_processors(self):
        """Register built-in feature processors."""
        try:
            # Import and register predictive scoring
            from predictive_scoring import process_predictive_scoring
            self.register_feature_processor("predictive_scoring", process_predictive_scoring)
            
            # Import and register other processors as they become available
            try:
                from smart_matching import process_smart_matching
                self.register_feature_processor("smart_matching", process_smart_matching)
            except ImportError:
                logger.info("Smart matching processor not available")
            
            try:
                from anomaly_detector import process_anomaly_detection
                self.register_feature_processor("anomaly_detection", process_anomaly_detection)
            except ImportError:
                logger.info("Anomaly detection processor not available")
            
            try:
                from data_quality import process_data_quality
                self.register_feature_processor("data_quality", process_data_quality)
            except ImportError:
                logger.info("Data quality processor not available")
                
        except Exception as e:
            logger.warning(f"Error registering built-in processors: {e}")
    
    def register_feature_processor(self, feature_name: str, processor: Callable):
        """
        Register a processor function for an AI/ML feature.
        
        Args:
            feature_name: Name of the feature
            processor: Function that processes the feature
        """
        self.feature_processors[feature_name] = processor
        logger.info(f"Registered processor for feature: {feature_name}")
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a specific AI/ML feature is enabled and available.
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            True if feature is enabled and not disabled due to errors
        """
        return (
            self.config_manager.is_feature_enabled(feature_name) and
            not self.error_handler.is_component_disabled(feature_name) and
            not self.performance_monitor.should_skip_feature(feature_name)
        )
    
    def enhance_reconciliation(
        self,
        reconciliation_data: Dict[str, Any],
        enable_caching: bool = True
    ) -> EnhancementResults:
        """
        Apply AI/ML enhancements to reconciliation data.
        
        Args:
            reconciliation_data: Original reconciliation results
            enable_caching: Whether to use intelligent caching
            
        Returns:
            Enhanced results with AI/ML improvements
        """
        # Initialize results
        results = EnhancementResults(
            original_results=reconciliation_data.copy()
        )
        
        # Check if AI/ML is globally enabled
        if not self.config.ai_ml_enabled:
            logger.info("AI/ML features are globally disabled")
            return results
        
        # Reset performance monitor for new session
        self.performance_monitor.reset_session()
        
        try:
            logger.info("Starting AI/ML enhancement process")
            
            # Check cache if enabled
            if enable_caching and self.is_feature_enabled("intelligent_caching"):
                cached_result = self._check_cache(reconciliation_data)
                if cached_result is not None:
                    logger.info("Using cached AI/ML results")
                    return cached_result
            
            # Apply each enabled feature
            feature_order = [
                "data_quality",
                "smart_matching", 
                "anomaly_detection",
                "predictive_scoring",
                "ai_insights"
            ]
            
            for feature_name in feature_order:
                if self.is_feature_enabled(feature_name):
                    try:
                        self._apply_feature(feature_name, reconciliation_data, results)
                        results.features_applied.append(feature_name)
                    except Exception as e:
                        self._handle_feature_error(feature_name, e, results)
                        results.features_skipped.append(feature_name)
                else:
                    results.features_skipped.append(feature_name)
                    logger.info(f"Feature '{feature_name}' skipped (disabled or unavailable)")
            
            # Cache results if caching is enabled
            if enable_caching and self.is_feature_enabled("intelligent_caching"):
                self._cache_results(reconciliation_data, results)
            
            # Finalize results
            results.performance_metrics = self.performance_monitor.get_performance_metrics()
            results.error_log = [error.message for error in self.error_handler.errors]
            results.warnings = self.performance_monitor.warnings.copy()
            
            logger.info(f"AI/ML enhancement completed - Applied: {len(results.features_applied)}, Skipped: {len(results.features_skipped)}")
            
        except Exception as e:
            # Handle critical errors
            error_obj = self.error_handler.log_error("ai_ml_engine", e, recoverable=False)
            results.error_log.append(f"Critical error: {error_obj.message}")
            results.success = False
            logger.error(f"Critical error in AI/ML enhancement: {e}")
        
        return results
    
    def _apply_feature(
        self,
        feature_name: str,
        reconciliation_data: Dict[str, Any],
        results: EnhancementResults
    ):
        """Apply a specific AI/ML feature with performance optimization."""
        if feature_name not in self.feature_processors:
            logger.warning(f"No processor registered for feature: {feature_name}")
            return
        
        with self.performance_monitor.monitor_component(feature_name):
            # Get feature configuration
            feature_config = self.config_manager.get_feature_config(feature_name)
            
            # Apply the feature processor with performance optimization
            processor = self.feature_processors[feature_name]
            
            # Use performance optimizer for enhanced execution
            try:
                enhancement = self.performance_optimizer.optimize_operation(
                    operation_name=feature_name,
                    data=reconciliation_data,
                    operation_func=processor,
                    config=feature_config
                )
            except Exception as e:
                # Fallback to direct execution if optimization fails
                logger.warning(f"Performance optimization failed for {feature_name}, using direct execution: {e}")
                enhancement = processor(reconciliation_data, feature_config)
            
            # Store enhancement results
            results.ai_enhancements[feature_name] = enhancement
            
            logger.info(f"Successfully applied feature: {feature_name}")
    
    def _handle_feature_error(
        self,
        feature_name: str,
        error: Exception,
        results: EnhancementResults
    ):
        """Handle errors from individual features."""
        error_obj = self.error_handler.log_error(
            feature_name,
            error,
            context={"feature_config": self.config_manager.get_feature_config(feature_name)}
        )
        
        # Add to results error log
        results.error_log.append(f"{feature_name}: {error_obj.message}")
        
        # Check if fallback is enabled
        if self.config.fallback_enabled:
            logger.info(f"Graceful fallback applied for feature: {feature_name}")
        else:
            logger.warning(f"Feature {feature_name} failed without fallback")
    
    def _check_cache(self, reconciliation_data: Dict[str, Any]) -> Optional[EnhancementResults]:
        """Check if cached results exist for the given data."""
        try:
            with self.performance_monitor.monitor_component("caching"):
                cached_result = self.cache.get_cached_results(
                    self.cache._calculate_data_hash(reconciliation_data)
                )
                return cached_result
        except Exception as e:
            self.error_handler.log_error("caching", e)
            return None
    
    def _cache_results(self, reconciliation_data: Dict[str, Any], results: EnhancementResults):
        """Cache the enhancement results."""
        try:
            with self.performance_monitor.monitor_component("caching"):
                self.cache.cache_results(
                    self.cache._calculate_data_hash(reconciliation_data),
                    results,
                    ttl_hours=self.config.cache.cache_ttl_hours
                )
        except Exception as e:
            self.error_handler.log_error("caching", e)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        base_metrics = self.performance_monitor.get_performance_metrics()
        cache_stats = self.cache.get_cache_statistics()
        error_summary = self.error_handler.get_error_summary()
        
        return {
            "performance": base_metrics,
            "cache": cache_stats,
            "errors": error_summary,
            "features_status": {
                feature: self.is_feature_enabled(feature)
                for feature in self.config.enabled_features
            }
        }
    
    def get_feature_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status for all AI/ML features."""
        status = {}
        
        for feature in self.config.enabled_features:
            status[feature] = {
                "enabled": self.config_manager.is_feature_enabled(feature),
                "available": feature in self.feature_processors,
                "error_disabled": self.error_handler.is_component_disabled(feature),
                "performance_disabled": self.performance_monitor.should_skip_feature(feature),
                "overall_status": self.is_feature_enabled(feature),
                "error_count": self.error_handler.error_counts.get(feature, 0),
                "config": self.config_manager.get_feature_config(feature)
            }
        
        return status
    
    def reset_feature_errors(self, feature_name: str) -> bool:
        """
        Reset errors for a specific feature and re-enable it.
        
        Args:
            feature_name: Name of the feature to reset
            
        Returns:
            True if reset was successful
        """
        try:
            self.error_handler.reset_component_errors(feature_name)
            logger.info(f"Reset errors for feature: {feature_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting feature {feature_name}: {e}")
            return False
    
    def update_configuration(self, new_config: AIMLConfig) -> bool:
        """
        Update AI/ML configuration.
        
        Args:
            new_config: New configuration to apply
            
        Returns:
            True if update was successful
        """
        try:
            # Validate configuration
            is_valid, error_msg = self.config_manager.validate_config(new_config)
            if not is_valid:
                logger.error(f"Invalid configuration: {error_msg}")
                return False
            
            # Save configuration
            if self.config_manager.save_config(new_config):
                self.config = new_config
                logger.info("AI/ML configuration updated successfully")
                return True
            else:
                logger.error("Failed to save configuration")
                return False
                
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def cleanup_resources(self):
        """Clean up AI/ML engine resources."""
        try:
            # Clean up cache
            self.cache.cleanup_cache(force=True)
            
            # Clean up performance optimizer
            self.performance_optimizer.cleanup()
            
            # Save performance logs
            self.performance_monitor.save_performance_log()
            
            # Save configuration
            self.config_manager.save_config(self.config)
            
            logger.info("AI/ML engine resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
    
    def generate_diagnostics_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostics report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "ai_ml_enabled": self.config.ai_ml_enabled,
                "enabled_features": self.config.enabled_features,
                "fallback_enabled": self.config.fallback_enabled,
                "debug_mode": self.config.debug_mode
            },
            "performance": self.performance_monitor.get_performance_metrics(),
            "cache": self.cache.get_cache_statistics(),
            "optimization": self.performance_optimizer.get_optimization_report(),
            "errors": {
                "total_errors": len(self.error_handler.errors),
                "error_counts": self.error_handler.error_counts,
                "disabled_features": self.error_handler.disabled_features,
                "recent_errors": [
                    {
                        "component": error.component,
                        "message": error.message,
                        "timestamp": error.timestamp.isoformat(),
                        "recoverable": error.recoverable
                    }
                    for error in self.error_handler.errors[-5:]  # Last 5 errors
                ]
            },
            "features": self.get_feature_status(),
            "recommendations": self.performance_monitor.get_optimization_recommendations()
        }


# Convenience functions for easy integration
def create_ai_ml_engine(config_file: str = "reconciliation_settings.json") -> AIMLEngine:
    """Create a new AI/ML engine instance."""
    return AIMLEngine(config_file)


def enhance_gst_reconciliation(
    reconciliation_data: Dict[str, Any],
    engine: Optional[AIMLEngine] = None
) -> EnhancementResults:
    """
    Enhance GST reconciliation data with AI/ML features.
    
    Args:
        reconciliation_data: Original reconciliation results
        engine: AI/ML engine instance (creates new if None)
        
    Returns:
        Enhanced results with AI/ML improvements
    """
    if engine is None:
        engine = create_ai_ml_engine()
    
    return engine.enhance_reconciliation(reconciliation_data)