"""
Text processor module for OpenDigger Repository Labeling POC.
Handles text cleaning, preprocessing, and language detection.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

from .config import Config, get_config
from .utils import get_logger, clean_text

class TextProcessor:
    """Processes and cleans repository text content."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the text processor.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.stemmer = PorterStemmer()
        self._download_nltk_data()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.logger.warning("NLTK stopwords not available, using basic set")
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
            }
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            self.logger.info("Downloading NLTK stopwords")
            nltk.download('stopwords', quiet=True)
    
    def clean_and_normalize(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Use the utility function for basic cleaning
        cleaned = clean_text(text)
        
        # Additional processing for repository text
        # Remove URLs
        cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
        
        # Remove email addresses
        cleaned = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned)
        
        # Remove file extensions and paths
        cleaned = re.sub(r'\b\w+\.(js|py|java|cpp|c|h|html|css|php|rb|go|rs|ts|jsx|tsx|vue)\b', '', cleaned)
        
        # Remove version numbers
        cleaned = re.sub(r'\bv?\d+\.\d+(\.\d+)?(-\w+)?\b', '', cleaned)
        
        # Remove common code artifacts
        code_patterns = [
            r'\b(function|class|def|var|let|const|import|export|from|return)\b',
            r'\b(public|private|protected|static|final|abstract)\b',
            r'[{}()\[\];,]',
            r'[<>=/+\-*%&|^~!]+'
        ]
        
        for pattern in code_patterns:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        # Final cleanup
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (language_code, confidence_score)
        """
        # Simple heuristic-based language detection
        # For a more robust solution, consider using langdetect library
        
        if not text:
            return ('unknown', 0.0)
        
        # Count English indicators
        english_indicators = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that'
        ]
        
        words = text.lower().split()
        if not words:
            return ('unknown', 0.0)
        
        english_count = sum(1 for word in words if word in english_indicators)
        english_ratio = english_count / len(words)
        
        # Simple threshold-based classification
        if english_ratio > 0.1:
            return ('en', min(1.0, english_ratio * 2))
        else:
            return ('unknown', 1.0 - english_ratio)
    
    def extract_keywords(self, text: str, max_keywords: int = None) -> List[Tuple[str, int]]:
        """
        Extract keywords from text using frequency analysis.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to return
        
        Returns:
            List of (keyword, frequency) tuples
        """
        if not text:
            return []
        
        max_keywords = max_keywords or Config.MAX_KEYWORDS_EXTRACTED
        
        # Tokenize and clean
        try:
            words = word_tokenize(text.lower())
        except:
            # Fallback to simple split if NLTK fails
            words = text.lower().split()
        
        # Filter words
        filtered_words = []
        for word in words:
            # Keep only alphabetic words of reasonable length
            if (word.isalpha() and 
                len(word) >= 3 and 
                word not in self.stop_words and
                not self._is_generic_tech_term(word)):
                filtered_words.append(word)
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        
        # Filter by minimum frequency
        filtered_freq = {
            word: freq for word, freq in word_freq.items() 
            if freq >= Config.MIN_KEYWORD_FREQUENCY
        }
        
        # Return top keywords
        top_keywords = sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True)
        return top_keywords[:max_keywords]
    
    def _is_generic_tech_term(self, word: str) -> bool:
        """
        Check if a word is a generic technical term that should be filtered.
        
        Args:
            word: Word to check
        
        Returns:
            True if the word is generic
        """
        generic_terms = {
            'code', 'software', 'program', 'application', 'app', 'system',
            'tool', 'library', 'framework', 'package', 'module', 'component',
            'service', 'api', 'interface', 'client', 'server', 'database',
            'file', 'data', 'information', 'content', 'text', 'string',
            'number', 'value', 'object', 'item', 'element', 'node',
            'project', 'repository', 'repo', 'source', 'open', 'free',
            'simple', 'easy', 'fast', 'quick', 'small', 'light', 'basic',
            'advanced', 'powerful', 'flexible', 'extensible', 'scalable'
        }
        
        return word.lower() in generic_terms
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """
        Extract technical terms and domain-specific vocabulary.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of technical terms
        """
        if not text:
            return []
        
        technical_terms = []
        
        # Programming languages and technologies
        tech_patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|c\+\+|c#|php|ruby|go|rust|swift|kotlin|scala|r|matlab|perl|lua|dart|elixir|haskell|clojure|erlang)\b',
            
            # Web technologies
            r'\b(html|css|sass|scss|less|react|vue|angular|svelte|jquery|bootstrap|tailwind|webpack|babel|nodejs|express|django|flask|rails|laravel|spring)\b',
            
            # Databases
            r'\b(mysql|postgresql|mongodb|redis|elasticsearch|sqlite|oracle|sqlserver|cassandra|dynamodb|firebase|supabase)\b',
            
            # Cloud and DevOps
            r'\b(aws|azure|gcp|docker|kubernetes|jenkins|gitlab|github|terraform|ansible|vagrant|nginx|apache|linux|ubuntu|centos|debian)\b',
            
            # Mobile development
            r'\b(android|ios|react-native|flutter|xamarin|cordova|ionic|phonegap)\b',
            
            # Data science and ML
            r'\b(tensorflow|pytorch|scikit-learn|pandas|numpy|matplotlib|jupyter|anaconda|spark|hadoop|kafka|airflow)\b',
            
            # Game development
            r'\b(unity|unreal|godot|pygame|phaser|three\.js|webgl|opengl|directx)\b',
            
            # Blockchain
            r'\b(blockchain|bitcoin|ethereum|solidity|web3|defi|nft|smart-contract)\b'
        ]
        
        text_lower = text.lower()
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            technical_terms.extend(matches)
        
        # Remove duplicates and return
        return list(set(technical_terms))
    
    def process_repository_text(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all text content for a repository.
        
        Args:
            repo_data: Repository data dictionary
        
        Returns:
            Enhanced repository data with processed text
        """
        # Extract raw text content
        raw_text = self._extract_all_text(repo_data)
        
        config = get_config()
        if not raw_text or len(raw_text) < config.processing.min_description_length:
            self.logger.debug(f"Insufficient text content for {repo_data.get('full_name', 'unknown')}")
            return {
                **repo_data,
                'processed_text': {
                    'raw_text': raw_text,
                    'cleaned_text': '',
                    'language': 'unknown',
                    'language_confidence': 0.0,
                    'keywords': [],
                    'technical_terms': [],
                    'word_count': 0,
                    'is_processable': False
                }
            }
        
        # Clean and normalize text
        cleaned_text = self.clean_and_normalize(raw_text)
        
        # Detect language
        language, language_confidence = self.detect_language(cleaned_text)
        
        # Extract keywords and technical terms
        keywords = self.extract_keywords(cleaned_text)
        technical_terms = self.extract_technical_terms(cleaned_text)
        
        # Calculate word count
        word_count = len(cleaned_text.split()) if cleaned_text else 0
        
        # Determine if text is processable (more lenient criteria)
        is_processable = (
            (language == 'en' and language_confidence > 0.1) or  # English with low confidence
            (language == 'unknown' and word_count >= 5) or       # Unknown language but has content
            (word_count >= 10)                                    # Sufficient content regardless of language
        )
        
        processed_text = {
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'language': language,
            'language_confidence': language_confidence,
            'keywords': keywords,
            'technical_terms': technical_terms,
            'word_count': word_count,
            'is_processable': is_processable
        }
        
        return {
            **repo_data,
            'processed_text': processed_text
        }
    
    def _extract_all_text(self, repo_data: Dict[str, Any]) -> str:
        """
        Extract all available text content from repository data.
        
        Args:
            repo_data: Repository data dictionary
        
        Returns:
            Combined text content
        """
        text_parts = []
        
        # Repository name and organization
        if 'repo' in repo_data:
            text_parts.append(repo_data['repo'])
        if 'org_or_user' in repo_data:
            text_parts.append(repo_data['org_or_user'])
        
        # Check for text content in both metadata structure and direct structure
        data_sources = []
        
        # Add metadata if it exists
        if 'metadata' in repo_data:
            data_sources.append(repo_data['metadata'])
        
        # Also check the repo_data itself (for mock data structure)
        data_sources.append(repo_data)
        
        # Extract text from all data sources
        for data_source in data_sources:
            # Common text fields
            text_fields = [
                'description', 'readme', 'summary', 'about', 'homepage',
                'documentation', 'wiki', 'issues', 'pull_requests', 'text_content'
            ]
            
            for field in text_fields:
                if field in data_source and data_source[field]:
                    if isinstance(data_source[field], str):
                        text_parts.append(data_source[field])
                    elif isinstance(data_source[field], dict):
                        # Extract text from nested dictionaries
                        for key, value in data_source[field].items():
                            if isinstance(value, str):
                                text_parts.append(value)
            
            # Topics, tags, keywords
            list_fields = ['topics', 'tags', 'keywords', 'languages']
            for field in list_fields:
                if field in data_source and isinstance(data_source[field], list):
                    text_parts.extend([str(item) for item in data_source[field]])
            
            # Language information
            if 'language' in data_source and data_source['language']:
                text_parts.append(f"Primary language: {data_source['language']}")
        
        # Remove duplicates while preserving order
        unique_parts = []
        seen = set()
        for part in text_parts:
            if part and part not in seen:
                unique_parts.append(part)
                seen.add(part)
        
        return ' '.join(unique_parts)
    
    def batch_process_repositories(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process text content for multiple repositories.
        
        Args:
            repositories: List of repository data dictionaries
        
        Returns:
            List of repositories with processed text
        """
        self.logger.info(f"Processing text content for {len(repositories)} repositories")
        
        processed_repos = []
        processable_count = 0
        
        for repo in repositories:
            try:
                processed_repo = self.process_repository_text(repo)
                processed_repos.append(processed_repo)
                
                if processed_repo['processed_text']['is_processable']:
                    processable_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to process text for {repo.get('full_name', 'unknown')}: {e}")
                # Add repository with empty processed text
                processed_repos.append({
                    **repo,
                    'processed_text': {
                        'raw_text': '',
                        'cleaned_text': '',
                        'language': 'unknown',
                        'language_confidence': 0.0,
                        'keywords': [],
                        'technical_terms': [],
                        'word_count': 0,
                        'is_processable': False
                    }
                })
        
        self.logger.info(f"Successfully processed {len(processed_repos)} repositories")
        self.logger.info(f"{processable_count} repositories are suitable for label extraction")
        
        return processed_repos

def main():
    """Main function for testing the text processor."""
    logger = setup_logging('INFO', 'logs/text_processor.log')
    processor = TextProcessor(logger)
    
    # Test with sample data
    sample_repo = {
        'platform': 'github',
        'org_or_user': 'facebook',
        'repo': 'react',
        'full_name': 'github/facebook/react',
        'metadata': {
            'description': 'A declarative, efficient, and flexible JavaScript library for building user interfaces.',
            'topics': ['javascript', 'react', 'frontend', 'ui', 'library'],
            'language': 'JavaScript'
        }
    }
    
    processed = processor.process_repository_text(sample_repo)
    
    logger.info("Sample processing result:")
    logger.info(f"Language: {processed['processed_text']['language']}")
    logger.info(f"Keywords: {processed['processed_text']['keywords'][:5]}")
    logger.info(f"Technical terms: {processed['processed_text']['technical_terms']}")
    logger.info(f"Is processable: {processed['processed_text']['is_processable']}")

if __name__ == "__main__":
    main() 