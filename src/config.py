"""Configuration module for OpenDigger Repository Labeling POC.

This module contains all configuration settings, constants, and environment
variable management for the OpenDigger POC system.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for machine learning models."""
    
    flan_t5_model_name: str = "google/flan-t5-large"
    flan_t5_cache_dir: str = "./models"
    use_gpu: bool = True
    max_model_length: int = 512
    temperature: float = 0.3
    max_tokens: int = 100
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50


@dataclass(frozen=True)
class APIConfig:
    """Configuration for external APIs."""
    
    repo_list_url: str = "https://oss.open-digger.cn/repo_list.csv"
    metadata_url_template: str = "https://oss.open-digger.cn/{platform}/{org_or_user}/{repo}/meta.json"
    max_concurrent_requests: int = 10
    request_delay_seconds: float = 1.0
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass(frozen=True)
class ProcessingConfig:
    """Configuration for data processing."""
    
    batch_size: int = 50
    sample_size: int = 1000
    min_description_length: int = 10
    min_label_confidence: float = 0.5
    max_labels_per_repo: int = 10
    min_keyword_frequency: int = 2
    max_keywords_extracted: int = 20


@dataclass(frozen=True)
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Base directories
    data_dir: Path = field(default_factory=lambda: Path("data"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    
    # Data subdirectories
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    output_data_dir: Path = field(default_factory=lambda: Path("data/output"))
    
    # Specific file paths
    repo_list_file: Path = field(default_factory=lambda: Path("data/raw/repo_list.csv"))
    processed_repos_file: Path = field(default_factory=lambda: Path("data/processed/processed_repositories.json"))
    labeled_repos_file: Path = field(default_factory=lambda: Path("data/output/labeled_repositories.json"))
    summary_csv_file: Path = field(default_factory=lambda: Path("data/output/labeling_summary.csv"))


@dataclass(frozen=True)
class NLPConfig:
    """Configuration for NLP processing."""
    
    spacy_model: str = "en_core_web_sm"
    language_detection_threshold: float = 0.1
    min_word_count: int = 5
    generic_labels_to_filter: List[str] = field(default_factory=lambda: [
        "software", "code", "programming", "development", "project",
        "application", "system", "tool", "library", "framework"
    ])


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for logging."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class TechnicalDomains:
    """Technical domain categories and their associated keywords."""
    
    DOMAINS: Dict[str, List[str]] = {
        "web_development": [
            "web", "frontend", "backend", "fullstack", "html", "css", "javascript",
            "react", "vue", "angular", "nodejs", "express", "django", "flask"
        ],
        "mobile_development": [
            "mobile", "android", "ios", "react-native", "flutter", "swift",
            "kotlin", "xamarin", "cordova", "ionic"
        ],
        "data_science": [
            "data-science", "machine-learning", "deep-learning", "ai", "ml",
            "analytics", "statistics", "pandas", "numpy", "scikit-learn",
            "tensorflow", "pytorch", "jupyter"
        ],
        "devops": [
            "devops", "docker", "kubernetes", "ci-cd", "deployment", "infrastructure",
            "monitoring", "logging", "automation", "ansible", "terraform"
        ],
        "database": [
            "database", "sql", "nosql", "mongodb", "postgresql", "mysql",
            "redis", "elasticsearch", "orm", "migration"
        ],
        "security": [
            "security", "cybersecurity", "encryption", "authentication",
            "authorization", "vulnerability", "penetration-testing"
        ],
        "game_development": [
            "game", "gaming", "unity", "unreal", "godot", "pygame",
            "2d", "3d", "graphics", "rendering"
        ],
        "blockchain": [
            "blockchain", "cryptocurrency", "bitcoin", "ethereum", "smart-contracts",
            "defi", "nft", "web3", "solidity"
        ]
    }


class Config:
    """Main configuration class that aggregates all configuration settings."""
    
    def __init__(self) -> None:
        """Initialize configuration from environment variables."""
        self.model = self._load_model_config()
        self.api = self._load_api_config()
        self.processing = self._load_processing_config()
        self.paths = self._load_path_config()
        self.nlp = self._load_nlp_config()
        self.logging = self._load_logging_config()
        self.technical_domains = TechnicalDomains.DOMAINS
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from environment variables."""
        return ModelConfig(
            flan_t5_model_name=os.getenv("FLAN_T5_MODEL_NAME", "google/flan-t5-large"),
            flan_t5_cache_dir=os.getenv("FLAN_T5_CACHE_DIR", "./models"),
            use_gpu=os.getenv("USE_GPU", "true").lower() == "true",
            max_model_length=int(os.getenv("MAX_MODEL_LENGTH", "512")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "100")),
            do_sample=os.getenv("LLM_DO_SAMPLE", "true").lower() == "true",
            top_p=float(os.getenv("LLM_TOP_P", "0.9")),
            top_k=int(os.getenv("LLM_TOP_K", "50"))
        )
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment variables."""
        return APIConfig(
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
            request_delay_seconds=float(os.getenv("REQUEST_DELAY_SECONDS", "1.0")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "30"))
        )
    
    def _load_processing_config(self) -> ProcessingConfig:
        """Load processing configuration from environment variables."""
        return ProcessingConfig(
            batch_size=int(os.getenv("BATCH_SIZE", "50")),
            sample_size=int(os.getenv("SAMPLE_SIZE", "1000")),
            min_description_length=int(os.getenv("MIN_DESCRIPTION_LENGTH", "10")),
            min_label_confidence=float(os.getenv("MIN_LABEL_CONFIDENCE", "0.5")),
            max_labels_per_repo=int(os.getenv("MAX_LABELS_PER_REPO", "10")),
            min_keyword_frequency=int(os.getenv("MIN_KEYWORD_FREQUENCY", "2")),
            max_keywords_extracted=int(os.getenv("MAX_KEYWORDS_EXTRACTED", "20"))
        )
    
    def _load_path_config(self) -> PathConfig:
        """Load path configuration."""
        return PathConfig()
    
    def _load_nlp_config(self) -> NLPConfig:
        """Load NLP configuration from environment variables."""
        return NLPConfig(
            spacy_model=os.getenv("SPACY_MODEL", "en_core_web_sm"),
            language_detection_threshold=float(os.getenv("LANGUAGE_DETECTION_THRESHOLD", "0.1")),
            min_word_count=int(os.getenv("MIN_WORD_COUNT", "5"))
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment variables."""
        return LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.paths.data_dir,
            self.paths.logs_dir,
            self.paths.models_dir,
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.output_data_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise.
            
        Raises:
            ValueError: If critical configuration is invalid.
        """
        # Validate numeric ranges
        if not 0 < self.processing.min_label_confidence <= 1:
            raise ValueError("min_label_confidence must be between 0 and 1")
        
        if self.processing.max_labels_per_repo <= 0:
            raise ValueError("max_labels_per_repo must be positive")
        
        if self.api.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        
        # Validate paths
        if not self.paths.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.paths.data_dir}")
        
        return True
    
    def get_metadata_url(self, platform: str, org_or_user: str, repo: str) -> str:
        """Generate metadata URL for a specific repository.
        
        Args:
            platform: Repository platform (e.g., 'github')
            org_or_user: Organization or user name
            repo: Repository name
            
        Returns:
            Formatted metadata URL
        """
        return self.api.metadata_url_template.format(
            platform=platform,
            org_or_user=org_or_user,
            repo=repo
        )
    
    def to_dict(self) -> Dict[str, Union[str, int, float, bool, List[str]]]:
        """Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "model": {
                "flan_t5_model_name": self.model.flan_t5_model_name,
                "use_gpu": self.model.use_gpu,
                "max_model_length": self.model.max_model_length,
                "temperature": self.model.temperature
            },
            "processing": {
                "sample_size": self.processing.sample_size,
                "min_label_confidence": self.processing.min_label_confidence,
                "max_labels_per_repo": self.processing.max_labels_per_repo
            },
            "api": {
                "max_concurrent_requests": self.api.max_concurrent_requests,
                "request_delay_seconds": self.api.request_delay_seconds
            }
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Global configuration instance
    """
    return config


# Backward compatibility - maintain old interface
class LegacyConfig:
    """Legacy configuration interface for backward compatibility."""
    
    @property
    def SAMPLE_SIZE(self) -> int:
        return config.processing.sample_size
    
    @property
    def MIN_LABEL_CONFIDENCE(self) -> float:
        return config.processing.min_label_confidence
    
    @property
    def MAX_LABELS_PER_REPO(self) -> int:
        return config.processing.max_labels_per_repo
    
    @property
    def MAX_CONCURRENT_REQUESTS(self) -> int:
        return config.api.max_concurrent_requests
    
    @property
    def REQUEST_DELAY_SECONDS(self) -> float:
        return config.api.request_delay_seconds
    
    @property
    def TECHNICAL_DOMAINS(self) -> Dict[str, List[str]]:
        return config.technical_domains
    
    @property
    def SPACY_MODEL(self) -> str:
        return config.nlp.spacy_model
    
    @property
    def LOG_LEVEL(self) -> str:
        return config.logging.level
    
    @property
    def RAW_DATA_DIR(self) -> str:
        return str(config.paths.raw_data_dir)
    
    @property
    def OUTPUT_DATA_DIR(self) -> str:
        return str(config.paths.output_data_dir)
    
    @property
    def LOGS_DIR(self) -> str:
        return str(config.paths.logs_dir)
    
    def validate_config(self) -> bool:
        return config.validate()
    
    def get_metadata_url(self, platform: str, org_or_user: str, repo: str) -> str:
        return config.get_metadata_url(platform, org_or_user, repo)


# Legacy instance for backward compatibility
Config = LegacyConfig() 