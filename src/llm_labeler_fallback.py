"""
Fallback LLM labeler module for OpenDigger Repository Labeling POC.
Uses rule-based approaches when Flan-T5 model is not available.
"""

import asyncio
import json
import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import concurrent.futures

from .config import Config
from .utils import get_logger, retry_on_failure

class FallbackLabeler:
    """Extracts technical domain labels using rule-based approaches."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the fallback labeler.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'processing_time': 0,
            'error_types': {}
        }
        
        # Technical patterns for label extraction
        self.tech_patterns = {
            # Programming languages
            'languages': {
                'python': r'\b(?:python|py|django|flask|fastapi|pandas|numpy)\b',
                'javascript': r'\b(?:javascript|js|node|react|vue|angular|express)\b',
                'java': r'\b(?:java|spring|maven|gradle)\b',
                'typescript': r'\b(?:typescript|ts)\b',
                'go': r'\b(?:golang|go)\b',
                'rust': r'\b(?:rust|cargo)\b',
                'cpp': r'\b(?:c\+\+|cpp|cmake)\b',
                'c': r'\b(?:^c$|clang)\b',
                'php': r'\b(?:php|laravel|symfony)\b',
                'ruby': r'\b(?:ruby|rails)\b',
                'swift': r'\b(?:swift|ios)\b',
                'kotlin': r'\b(?:kotlin|android)\b',
                'scala': r'\b(?:scala|akka)\b',
                'r': r'\b(?:^r$|rstudio|shiny)\b',
                'matlab': r'\b(?:matlab|octave)\b',
                'shell': r'\b(?:bash|shell|zsh|fish)\b'
            },
            
            # Frameworks and libraries
            'frameworks': {
                'react': r'\b(?:react|jsx|next\.js|gatsby)\b',
                'vue': r'\b(?:vue|vuejs|nuxt)\b',
                'angular': r'\b(?:angular|angularjs)\b',
                'django': r'\b(?:django|drf)\b',
                'flask': r'\b(?:flask|jinja)\b',
                'express': r'\b(?:express|expressjs)\b',
                'spring': r'\b(?:spring|springboot)\b',
                'laravel': r'\b(?:laravel|eloquent)\b',
                'rails': r'\b(?:rails|activerecord)\b',
                'tensorflow': r'\b(?:tensorflow|tf|keras)\b',
                'pytorch': r'\b(?:pytorch|torch)\b',
                'scikit-learn': r'\b(?:sklearn|scikit-learn)\b',
                'opencv': r'\b(?:opencv|cv2)\b',
                'unity': r'\b(?:unity|unity3d)\b',
                'unreal': r'\b(?:unreal|ue4|ue5)\b'
            },
            
            # Domains and technologies
            'domains': {
                'web-development': r'\b(?:web|frontend|backend|fullstack|html|css|dom|browser)\b',
                'mobile-development': r'\b(?:mobile|android|ios|app|smartphone|tablet)\b',
                'machine-learning': r'\b(?:machine.learning|ml|ai|artificial.intelligence|neural|deep.learning)\b',
                'data-science': r'\b(?:data.science|analytics|statistics|visualization|pandas|numpy)\b',
                'devops': r'\b(?:devops|docker|kubernetes|ci/cd|deployment|infrastructure)\b',
                'database': r'\b(?:database|sql|nosql|mongodb|postgresql|mysql|redis)\b',
                'security': r'\b(?:security|cybersecurity|encryption|authentication|vulnerability)\b',
                'blockchain': r'\b(?:blockchain|cryptocurrency|bitcoin|ethereum|smart.contract)\b',
                'game-development': r'\b(?:game|gaming|gamedev|2d|3d|graphics|rendering)\b',
                'api': r'\b(?:api|rest|graphql|microservice|endpoint)\b',
                'cloud': r'\b(?:cloud|aws|azure|gcp|serverless|lambda)\b',
                'iot': r'\b(?:iot|internet.of.things|embedded|sensor|arduino|raspberry)\b'
            },
            
            # Tools and platforms
            'tools': {
                'git': r'\b(?:git|github|gitlab|version.control)\b',
                'docker': r'\b(?:docker|container|dockerfile)\b',
                'kubernetes': r'\b(?:kubernetes|k8s|helm)\b',
                'terraform': r'\b(?:terraform|infrastructure.as.code)\b',
                'jenkins': r'\b(?:jenkins|ci/cd|pipeline)\b',
                'webpack': r'\b(?:webpack|bundler|build.tool)\b',
                'npm': r'\b(?:npm|yarn|package.manager)\b',
                'maven': r'\b(?:maven|gradle|build.system)\b'
            }
        }
        
        self.logger.info("Fallback labeler initialized with rule-based patterns")
    
    def create_labeling_prompt(self, repo_data: Dict[str, Any]) -> str:
        """
        Create a prompt for rule-based label extraction.
        
        Args:
            repo_data: Repository data dictionary
        
        Returns:
            Combined text for analysis
        """
        # Extract relevant information
        repo_name = repo_data.get('repo', '')
        org_name = repo_data.get('org_or_user', '')
        
        # Get processed text content
        processed_text = repo_data.get('processed_text', {})
        description = processed_text.get('cleaned_text', '')
        technical_terms = processed_text.get('technical_terms', [])
        keywords = processed_text.get('keywords', [])
        
        # Combine all text for analysis
        text_parts = [repo_name, org_name, description]
        text_parts.extend(technical_terms)
        text_parts.extend([kw[0] if isinstance(kw, tuple) else str(kw) for kw in keywords])
        
        return ' '.join(str(part) for part in text_parts if part).lower()
    
    def _extract_labels_from_text(self, text: str) -> List[str]:
        """
        Extract labels using rule-based pattern matching.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of extracted labels
        """
        labels = []
        
        # Apply all patterns
        for category, patterns in self.tech_patterns.items():
            for label, pattern in patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    labels.append(label)
        
        # Add domain-specific labels based on combinations
        if any(lang in labels for lang in ['python', 'r', 'scala']):
            if any(term in text for term in ['data', 'analytics', 'science', 'ml', 'ai']):
                labels.append('data-science')
        
        if any(lang in labels for lang in ['javascript', 'typescript', 'react', 'vue', 'angular']):
            if any(term in text for term in ['web', 'frontend', 'ui', 'browser']):
                labels.append('frontend-development')
        
        if any(lang in labels for lang in ['java', 'python', 'go', 'rust']):
            if any(term in text for term in ['api', 'server', 'backend', 'microservice']):
                labels.append('backend-development')
        
        if any(lang in labels for lang in ['swift', 'kotlin', 'java']):
            if any(term in text for term in ['mobile', 'android', 'ios', 'app']):
                labels.append('mobile-development')
        
        # Remove duplicates and limit
        unique_labels = list(dict.fromkeys(labels))  # Preserves order
        return unique_labels[:Config.MAX_LABELS_PER_REPO]
    
    def _generate_labels(self, text: str) -> Optional[List[str]]:
        """
        Generate labels using rule-based approach.
        
        Args:
            text: Input text
        
        Returns:
            List of extracted labels or None if failed
        """
        try:
            start_time = time.time()
            
            # Extract labels using patterns
            labels = self._extract_labels_from_text(text)
            
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            
            if labels:
                self.stats['successful_requests'] += 1
                return labels
            else:
                self.stats['failed_requests'] += 1
                return None
                
        except Exception as e:
            error_type = type(e).__name__
            self.stats['error_types'][error_type] = self.stats['error_types'].get(error_type, 0) + 1
            self.logger.debug(f"Rule-based extraction error: {e}")
            self.stats['failed_requests'] += 1
            return None
    
    async def extract_labels_for_repository(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract labels for a single repository using rule-based approach.
        
        Args:
            repo_data: Repository data dictionary
        
        Returns:
            Repository data with extracted labels
        """
        self.stats['total_requests'] += 1
        
        try:
            # Create text for analysis
            text = self.create_labeling_prompt(repo_data)
            
            # Generate labels using thread pool to maintain async interface
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                labels = await loop.run_in_executor(executor, self._generate_labels, text)
            
            # Prepare result
            result = repo_data.copy()
            result['llm_labels'] = {
                'labels': labels or [],
                'confidence': 0.7 if labels else 0.0,  # Fixed confidence for rule-based
                'source': 'rule-based',
                'model': 'fallback-patterns',
                'timestamp': time.time()
            }
            
            if labels:
                self.logger.debug(f"Extracted {len(labels)} labels for {repo_data.get('repo', 'unknown')}")
            else:
                self.logger.debug(f"No labels extracted for {repo_data.get('repo', 'unknown')}")
            
            return result
            
        except Exception as e:
            error_type = type(e).__name__
            self.stats['error_types'][error_type] = self.stats['error_types'].get(error_type, 0) + 1
            self.logger.error(f"Error processing repository {repo_data.get('repo', 'unknown')}: {e}")
            
            # Return original data with empty labels
            result = repo_data.copy()
            result['llm_labels'] = {
                'labels': [],
                'confidence': 0.0,
                'source': 'rule-based',
                'model': 'fallback-patterns',
                'error': str(e),
                'timestamp': time.time()
            }
            return result
    
    async def extract_labels_batch(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract labels for a batch of repositories.
        
        Args:
            repositories: List of repository data dictionaries
        
        Returns:
            List of repositories with extracted labels
        """
        self.logger.info(f"Starting label extraction for {len(repositories)} repositories using rule-based approach")
        
        # Process repositories with progress bar
        results = []
        
        # Use semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
        
        async def process_with_semaphore(repo_data):
            async with semaphore:
                return await self.extract_labels_for_repository(repo_data)
        
        # Process all repositories
        tasks = [process_with_semaphore(repo) for repo in repositories]
        
        try:
            from tqdm.asyncio import tqdm
            results = await tqdm.gather(*tasks, desc="Extracting labels (rule-based)")
        except ImportError:
            # Fallback if tqdm.asyncio is not available
            results = await asyncio.gather(*tasks)
        
        # Log statistics
        self.logger.info(f"Label extraction completed. Success rate: {self.stats['successful_requests']}/{self.stats['total_requests']}")
        
        return results
    
    def analyze_label_patterns(self, repositories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in extracted labels.
        
        Args:
            repositories: List of repositories with labels
        
        Returns:
            Analysis results
        """
        label_counts = {}
        total_repos = len(repositories)
        repos_with_labels = 0
        
        for repo in repositories:
            llm_labels = repo.get('llm_labels', {})
            labels = llm_labels.get('labels', [])
            
            if labels:
                repos_with_labels += 1
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
        
        # Sort labels by frequency
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        
        analysis = {
            'total_repositories': total_repos,
            'repositories_with_labels': repos_with_labels,
            'coverage_percentage': (repos_with_labels / total_repos * 100) if total_repos > 0 else 0,
            'unique_labels': len(label_counts),
            'most_common_labels': sorted_labels[:20],
            'average_labels_per_repo': sum(len(repo.get('llm_labels', {}).get('labels', [])) for repo in repositories) / total_repos if total_repos > 0 else 0,
            'model_stats': self.stats.copy()
        }
        
        return analysis
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()

# Maintain backward compatibility
LLMLabeler = FallbackLabeler

async def main():
    """Test the fallback labeler."""
    # Test data
    test_repo = {
        'repo': 'react-app',
        'org_or_user': 'facebook',
        'platform': 'github',
        'processed_text': {
            'cleaned_text': 'A JavaScript library for building user interfaces with components and state management',
            'technical_terms': ['javascript', 'react', 'components', 'state', 'ui'],
            'keywords': [('javascript', 5), ('react', 4), ('components', 3), ('ui', 2)]
        }
    }
    
    # Initialize labeler
    labeler = FallbackLabeler()
    
    # Extract labels
    result = await labeler.extract_labels_for_repository(test_repo)
    
    print("Test Results:")
    print(f"Repository: {result['repo']}")
    print(f"Labels: {result['llm_labels']['labels']}")
    print(f"Confidence: {result['llm_labels']['confidence']}")
    print(f"Stats: {labeler.get_stats()}")

if __name__ == "__main__":
    asyncio.run(main()) 