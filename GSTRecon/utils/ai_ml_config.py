"""
AI/ML Configuration Management System for GST Reconciliation Enhancement

This module provides configuration management for AI/ML features, integrating with
the existing reconciliation_settings.json file while maintaining backward compatibility.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for intelligent caching system."""
    enabled: bool = True
    cache_dir: str = ".cache/ai_ml"
    max_cache_size_mb: int = 500
    cache_ttl_hours: int = 24
    auto_cleanup: bool = True
    compression_enabled: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and limits."""
    max_total_time_seconds: int = 200
    max_component_time_seconds: Dict[str, int] = field(default_factory=lambda: {
        "smart_matching": 30,
        "anomaly_detection": 20,
        "data_quality": 25,
        "predictive_scoring": 15,
        "ai_insights": 10,
        "caching": 5
    })
    memory_limit_mb: int = 1024
    auto_disable_on_timeout: bool = True
    performance_logging: bool = True


@dataclass
class ModelConfig:
    """Configuration for AI/ML models."""
    model_paths: Dict[str, str] = field(default_factory=lambda: {
        "name_matching": "models/name_matcher.pkl",
        "anomaly_detection": "models/anomaly_detector.pkl",
        "predictive_scoring": "models/match_scorer.pkl"
    })
    use_pretrained: bool = True
    model_cache_enabled: bool = True
    fallback_to_rules: bool = True


@dataclass
class AIMLConfig:
    """Main AI/ML configuration dataclass with feature toggles and settings."""
    
    # Feature toggles
    enabled_features: List[str] = field(default_factory=lambda: [
        "smart_matching",
        "anomaly_detection", 
        "data_quality",
        "predictive_scoring",
        "intelligent_caching",
        "ai_insights"
    ])
    
    # Global AI/ML settings
    ai_ml_enabled: bool = True
    fallback_enabled: bool = True
    debug_mode: bool = False
    
    # Sub-configurations
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    
    # Feature-specific settings
    smart_matching_threshold: float = 0.85
    anomaly_sensitivity: float = 0.95
    data_quality_auto_fix: bool = True
    confidence_threshold: float = 0.7
    
    # Version and metadata
    config_version: str = "1.0"
    last_updated: Optional[str] = None


class AIMLConfigManager:
    """Manages AI/ML configuration with integration to existing settings system."""
    
    def __init__(self, settings_file: str = "reconciliation_settings.json"):
        self.settings_file = settings_file
        self.ai_ml_key = "ai_ml_config"
        self.default_config = AIMLConfig()
        self.config = self.load_config()
    
    def load_config(self) -> AIMLConfig:
        """Load AI/ML configuration from existing settings file or create default."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    all_settings = json.load(f)
                
                # Check if AI/ML config exists in settings
                if self.ai_ml_key in all_settings:
                    ai_ml_data = all_settings[self.ai_ml_key]
                    config = self._dict_to_config(ai_ml_data)
                    logger.info("AI/ML configuration loaded successfully")
                    return config
                else:
                    logger.info("No AI/ML configuration found, using defaults")
                    return self.default_config
            else:
                logger.info("Settings file not found, using default AI/ML configuration")
                return self.default_config
                
        except Exception as e:
            logger.error(f"Error loading AI/ML configuration: {e}")
            return self.default_config
    
    def save_config(self, config: AIMLConfig) -> bool:
        """Save AI/ML configuration to settings file, preserving existing settings."""
        try:
            # Load existing settings
            existing_settings = {}
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    existing_settings = json.load(f)
            
            # Update timestamp
            config.last_updated = datetime.now().isoformat()
            
            # Add AI/ML config to existing settings
            existing_settings[self.ai_ml_key] = self._config_to_dict(config)
            
            # Save updated settings
            with open(self.settings_file, 'w') as f:
                json.dump(existing_settings, f, indent=2)
            
            self.config = config
            logger.info("AI/ML configuration saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving AI/ML configuration: {e}")
            return False
    
    def validate_config(self, config: AIMLConfig) -> Tuple[bool, str]:
        """Validate AI/ML configuration parameters."""
        try:
            # Validate feature names
            valid_features = {
                "smart_matching", "anomaly_detection", "data_quality",
                "predictive_scoring", "intelligent_caching", "ai_insights"
            }
            
            for feature in config.enabled_features:
                if feature not in valid_features:
                    return False, f"Invalid feature name: {feature}"
            
            # Validate performance limits
            if config.performance.max_total_time_seconds <= 0:
                return False, "Max total time must be positive"
            
            if config.performance.max_total_time_seconds > 300:
                return False, "Max total time cannot exceed 300 seconds"
            
            # Validate thresholds
            if not (0.0 <= config.smart_matching_threshold <= 1.0):
                return False, "Smart matching threshold must be between 0.0 and 1.0"
            
            if not (0.0 <= config.anomaly_sensitivity <= 1.0):
                return False, "Anomaly sensitivity must be between 0.0 and 1.0"
            
            if not (0.0 <= config.confidence_threshold <= 1.0):
                return False, "Confidence threshold must be between 0.0 and 1.0"
            
            # Validate cache settings
            if config.cache.max_cache_size_mb <= 0:
                return False, "Cache size must be positive"
            
            if config.cache.cache_ttl_hours <= 0:
                return False, "Cache TTL must be positive"
            
            # Validate memory limits
            if config.performance.memory_limit_mb <= 0:
                return False, "Memory limit must be positive"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a specific AI/ML feature is enabled."""
        return (
            self.config.ai_ml_enabled and 
            feature_name in self.config.enabled_features
        )
    
    def get_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """Get configuration for a specific feature."""
        base_config = {
            "enabled": self.is_feature_enabled(feature_name),
            "fallback_enabled": self.config.fallback_enabled,
            "debug_mode": self.config.debug_mode
        }
        
        # Add feature-specific configurations
        if feature_name == "smart_matching":
            base_config.update({
                "threshold": self.config.smart_matching_threshold,
                "model_path": self.config.models.model_paths.get("name_matching"),
                "max_time": self.config.performance.max_component_time_seconds.get("smart_matching", 30)
            })
        elif feature_name == "anomaly_detection":
            base_config.update({
                "sensitivity": self.config.anomaly_sensitivity,
                "model_path": self.config.models.model_paths.get("anomaly_detection"),
                "max_time": self.config.performance.max_component_time_seconds.get("anomaly_detection", 20)
            })
        elif feature_name == "data_quality":
            base_config.update({
                "auto_fix": self.config.data_quality_auto_fix,
                "max_time": self.config.performance.max_component_time_seconds.get("data_quality", 25)
            })
        elif feature_name == "predictive_scoring":
            base_config.update({
                "confidence_threshold": self.config.confidence_threshold,
                "model_path": self.config.models.model_paths.get("predictive_scoring"),
                "max_time": self.config.performance.max_component_time_seconds.get("predictive_scoring", 15)
            })
        elif feature_name == "intelligent_caching":
            base_config.update({
                "cache_dir": self.config.cache.cache_dir,
                "max_size_mb": self.config.cache.max_cache_size_mb,
                "ttl_hours": self.config.cache.cache_ttl_hours,
                "compression": self.config.cache.compression_enabled,
                "max_time": self.config.performance.max_component_time_seconds.get("caching", 5)
            })
        elif feature_name == "ai_insights":
            base_config.update({
                "max_time": self.config.performance.max_component_time_seconds.get("ai_insights", 10)
            })
        
        return base_config
    
    def export_config(self) -> Dict[str, Any]:
        """Export configuration for backup or sharing."""
        return {
            "ai_ml_config": self._config_to_dict(self.config),
            "export_date": datetime.now().isoformat(),
            "config_version": self.config.config_version
        }
    
    def import_config(self, config_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Import configuration from exported data."""
        try:
            if "ai_ml_config" not in config_data:
                return False, "Invalid config format: missing ai_ml_config"
            
            imported_config = self._dict_to_config(config_data["ai_ml_config"])
            
            # Validate imported config
            is_valid, error_msg = self.validate_config(imported_config)
            if not is_valid:
                return False, f"Invalid imported config: {error_msg}"
            
            # Save imported config
            if self.save_config(imported_config):
                return True, "Configuration imported successfully"
            else:
                return False, "Failed to save imported configuration"
                
        except Exception as e:
            return False, f"Import error: {str(e)}"
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values."""
        try:
            default_config = AIMLConfig()
            return self.save_config(default_config)
        except Exception as e:
            logger.error(f"Error resetting to defaults: {e}")
            return False
    
    def _config_to_dict(self, config: AIMLConfig) -> Dict[str, Any]:
        """Convert AIMLConfig to dictionary for JSON serialization."""
        return asdict(config)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> AIMLConfig:
        """Convert dictionary to AIMLConfig object."""
        try:
            # Handle nested dataclasses
            if "performance" in data and isinstance(data["performance"], dict):
                data["performance"] = PerformanceConfig(**data["performance"])
            
            if "cache" in data and isinstance(data["cache"], dict):
                data["cache"] = CacheConfig(**data["cache"])
            
            if "models" in data and isinstance(data["models"], dict):
                data["models"] = ModelConfig(**data["models"])
            
            return AIMLConfig(**data)
            
        except Exception as e:
            logger.error(f"Error converting dict to config: {e}")
            return self.default_config
    
    def get_performance_limits(self) -> Dict[str, int]:
        """Get performance limits for all components."""
        return {
            "total_time": self.config.performance.max_total_time_seconds,
            "memory_mb": self.config.performance.memory_limit_mb,
            "component_times": self.config.performance.max_component_time_seconds.copy()
        }
    
    def update_feature_status(self, feature_name: str, enabled: bool) -> bool:
        """Enable or disable a specific feature."""
        try:
            if enabled and feature_name not in self.config.enabled_features:
                self.config.enabled_features.append(feature_name)
            elif not enabled and feature_name in self.config.enabled_features:
                self.config.enabled_features.remove(feature_name)
            
            return self.save_config(self.config)
            
        except Exception as e:
            logger.error(f"Error updating feature status: {e}")
            return False


def get_ai_ml_config() -> AIMLConfig:
    """Get current AI/ML configuration (convenience function)."""
    config_manager = AIMLConfigManager()
    return config_manager.config


def is_ai_ml_feature_enabled(feature_name: str) -> bool:
    """Check if AI/ML feature is enabled (convenience function)."""
    config_manager = AIMLConfigManager()
    return config_manager.is_feature_enabled(feature_name)