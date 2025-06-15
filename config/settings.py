"""
Configuration settings for OpenDigger Repository Domain Labeling POC
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# OpenDigger URLs
OPENDIGGER_BASE_URL = "https://oss.open-digger.cn"
REPO_LIST_URL = f"{OPENDIGGER_BASE_URL}/repo_list.csv"

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# LLM Settings
LLM_CONFIG = {
    "openai": {
        "model": "gpt-3.5-turbo",
        "max_tokens": 500,
        "temperature": 0.3,
        "timeout": 30
    },
    "deepseek": {
        "model": "deepseek-chat",
        "max_tokens": 500,
        "temperature": 0.3,
        "timeout": 30
    }
}

# Processing Settings
PROCESSING_CONFIG = {
    "sample_size": 100,  # Number of repositories to process for POC
    "batch_size": 10,    # Batch size for concurrent processing
    "max_retries": 3,    # Maximum retries for failed requests
    "request_delay": 1,  # Delay between requests (seconds)
    "confidence_threshold": 0.6,  # Minimum confidence for labels
    "max_labels_per_repo": 10,    # Maximum labels per repository
}

# Text Processing Settings
TEXT_CONFIG = {
    "min_text_length": 50,      # Minimum text length to process
    "max_text_length": 5000,    # Maximum text length to send to LLM
    "languages": ["en"],        # Supported languages
    "remove_code_blocks": True, # Remove code blocks from text
    "normalize_whitespace": True, # Normalize whitespace
}

# NLP Settings
NLP_CONFIG = {
    "spacy_model": "en_core_web_sm",
    "extract_entities": True,
    "extract_keywords": True,
    "min_keyword_freq": 2,
    "max_keywords": 20,
}

# Label Categories (for validation and filtering)
TECHNICAL_DOMAINS = {
    "programming_languages": [
        "python", "javascript", "java", "c++", "c#", "go", "rust", "ruby", 
        "php", "swift", "kotlin", "typescript", "scala", "r", "matlab"
    ],
    "frameworks": [
        "react", "angular", "vue", "django", "flask", "spring", "express",
        "laravel", "rails", "tensorflow", "pytorch", "scikit-learn"
    ],
    "domains": [
        "machine-learning", "artificial-intelligence", "web-development",
        "mobile-development", "data-science", "devops", "cybersecurity",
        "blockchain", "iot", "cloud-computing", "database", "networking"
    ],
    "tools": [
        "docker", "kubernetes", "git", "jenkins", "ansible", "terraform",
        "prometheus", "grafana", "elasticsearch", "redis", "mongodb"
    ]
}

# Output Settings
OUTPUT_CONFIG = {
    "save_individual_files": True,
    "save_summary_csv": True,
    "save_review_sample": True,
    "review_sample_size": 10,
    "include_confidence_scores": True,
    "include_processing_metadata": True,
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "poc.log",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# Validation Settings
VALIDATION_CONFIG = {
    "required_fields": ["name", "platform", "owner", "repo"],
    "optional_fields": ["description", "readme", "topics", "language"],
    "min_description_length": 10,
    "exclude_forks": True,
    "exclude_archived": True,
} 