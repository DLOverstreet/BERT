"""
⚙️ Configuration Management for Political Tweet Intensity Analyzer

Centralized configuration settings for the application.
"""

import os
from pathlib import Path
from typing import Dict, Any

class AppConfig:
    """Application configuration settings"""
    
    # Application Information
    APP_NAME = "Political Tweet Intensity Analyzer"
    APP_VERSION = "1.3.0"
    APP_DESCRIPTION = "Analyze political sentiment and intensity in tweets using DistilBERT"
    
    # Model Configuration
    MODEL_NAME = "m-newhauser/distilbert-political-tweets"
    MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "transformers"
    
    # Data Storage Configuration
    DATA_DIR = Path("tweet_data")
    TWEETS_FILE = DATA_DIR / "analyzed_tweets.csv"
    ANALYTICS_CACHE_FILE = DATA_DIR / "analytics_cache.json"
    SETTINGS_FILE = DATA_DIR / "settings.json"
    
    # Default Application Settings
    DEFAULT_SETTINGS = {
        "data_retention_days": 365,
        "anonymize_after_days": 30,
        "cache_duration_minutes": 15,
        "max_tweets_display": 100,
        "privacy_mode": False,
        "export_format": "csv",
        "auto_cleanup": True,
        "show_confidence_warnings": True,
        "enable_analytics_caching": True
    }
    
    # Analysis Configuration
    ANALYSIS_CONFIG = {
        "min_text_length": 5,
        "max_text_length": 1000,
        "confidence_threshold": 0.6,  # Warn if model confidence below this
        "extreme_intensity_threshold": 80,
        "high_intensity_threshold": 60,
        "moderate_intensity_threshold": 30
    }
    
    # Baseline Statistics (from 2021 senator training data)
    BASELINE_STATS = {
        "mean_intensity": 0.65,
        "std_intensity": 0.20,
        "mean_confidence": 0.78,
        "training_year": 2021,
        "training_source": "US Senator Tweets"
    }
    
    # UI Configuration
    UI_CONFIG = {
        "page_title": "Political Tweet Intensity Analyzer",
        "page_icon": "🗳️",
        "layout": "wide",
        "sidebar_state": "expanded",
        "theme": "light",
        "primary_color": "#1f4788",
        "secondary_color": "#c41e3a",
        "neutral_color": "#6c757d"
    }
    
    # Color Schemes for Visualizations
    COLOR_SCHEMES = {
        "political": {
            "democratic": "#1f4788",
            "republican": "#c41e3a", 
            "neutral": "#6c757d"
        },
        "intensity": {
            "low": "#28a745",
            "moderate": "#ffc107",
            "high": "#fd7e14",
            "extreme": "#dc3545"
        },
        "gradient": ["#28a745", "#ffc107", "#fd7e14", "#dc3545"]
    }
    
    # Export Configuration
    EXPORT_CONFIG = {
        "csv_delimiter": ",",
        "csv_encoding": "utf-8",
        "json_indent": 2,
        "include_metadata": True,
        "timestamp_format": "%Y%m%d_%H%M%S"
    }
    
    # Performance Configuration
    PERFORMANCE_CONFIG = {
        "chunk_size": 100,  # For batch processing
        "max_cache_size_mb": 50,
        "analytics_refresh_interval": 900,  # 15 minutes in seconds
        "model_timeout": 30,  # seconds
        "max_concurrent_analyses": 5
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_name": "app.log",
        "max_file_size": "10MB",
        "backup_count": 3
    }

class DevelopmentConfig(AppConfig):
    """Development environment configuration"""
    
    LOGGING_CONFIG = {
        **AppConfig.LOGGING_CONFIG,
        "level": "DEBUG"
    }
    
    # Enable additional debugging features
    DEBUG_MODE = True
    SHOW_MODEL_DETAILS = True
    ENABLE_PERFORMANCE_MONITORING = True

class ProductionConfig(AppConfig):
    """Production environment configuration"""
    
    # More conservative settings for production
    DEFAULT_SETTINGS = {
        **AppConfig.DEFAULT_SETTINGS,
        "data_retention_days": 180,  # Shorter retention in production
        "privacy_mode": True,
        "auto_cleanup": True
    }
    
    DEBUG_MODE = False
    SHOW_MODEL_DETAILS = False
    ENABLE_PERFORMANCE_MONITORING = False

class ConfigManager:
    """Configuration manager for loading and managing settings"""
    
    def __init__(self, environment: str = "development"):
        """Initialize configuration manager
        
        Args:
            environment: "development" or "production"
        """
        self.environment = environment
        
        if environment == "production":
            self.config = ProductionConfig()
        else:
            self.config = DevelopmentConfig()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting
        
        Args:
            key: Setting key (supports dot notation like 'ui.theme')
            default: Default value if key not found
        """
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if hasattr(value, k.upper()):
                    value = getattr(value, k.upper())
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        except:
            return default
    
    def get_color_scheme(self, scheme_type: str = "political") -> Dict[str, str]:
        """Get color scheme for visualizations
        
        Args:
            scheme_type: "political", "intensity", or "gradient"
        """
        return self.config.COLOR_SCHEMES.get(scheme_type, {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "model_name": self.config.MODEL_NAME,
            "cache_dir": str(self.config.MODEL_CACHE_DIR),
            "baseline_stats": self.config.BASELINE_STATS,
            "analysis_config": self.config.ANALYSIS_CONFIG
        }
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration for Streamlit"""
        return self.config.UI_CONFIG
    
    def ensure_directories(self):
        """Ensure required directories exist"""
        self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep file in data directory
        gitkeep_file = self.config.DATA_DIR / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == "production"

# Global configuration instance
# Environment can be controlled via ENVIRONMENT environment variable
environment = os.getenv("ENVIRONMENT", "development").lower()
config_manager = ConfigManager(environment)

# Convenience functions for common operations
def get_config() -> ConfigManager:
    """Get the global configuration manager"""
    return config_manager

def get_colors(scheme: str = "political") -> Dict[str, str]:
    """Get color scheme"""
    return config_manager.get_color_scheme(scheme)

def get_model_config() -> Dict[str, Any]:
    """Get model configuration"""
    return config_manager.get_model_config()

def ensure_setup():
    """Ensure application is properly set up"""
    config_manager.ensure_directories()

# Environment-specific feature flags
FEATURES = {
    "enable_batch_processing": True,
    "enable_real_time_analysis": True,
    "enable_export_functionality": True,
    "enable_advanced_analytics": True,
    "enable_data_privacy_controls": True,
    "enable_performance_monitoring": config_manager.is_development(),
    "enable_debug_logging": config_manager.is_development()
}

# API Configuration (if needed for future extensions)
API_CONFIG = {
    "rate_limit_per_minute": 60,
    "max_request_size": "10MB",
    "timeout_seconds": 30,
    "enable_cors": True
}