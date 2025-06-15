"""
Hybrid LLM labeler module for OpenDigger Repository Labeling POC.
Tries Flan-T5 first, falls back to rule-based approach if network fails.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any

from .config import Config
from .utils import get_logger

class HybridLabeler:
    """Tries Flan-T5 first, falls back to rule-based approach."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the hybrid labeler.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.primary_labeler = None
        self.fallback_labeler = None
        self.using_fallback = False
        
        # Try to initialize Flan-T5 first
        self._initialize_labelers()
    
    def _initialize_labelers(self):
        """Initialize primary and fallback labelers."""
        try:
            # Try to import and initialize Flan-T5
            from .llm_labeler import FlanT5Labeler
            self.primary_labeler = FlanT5Labeler(self.logger)
            self.logger.info("Flan-T5 labeler initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Flan-T5 labeler: {e}")
            self.using_fallback = True
        
        # Always initialize fallback
        try:
            from .llm_labeler_fallback import FallbackLabeler
            self.fallback_labeler = FallbackLabeler(self.logger)
            if self.using_fallback:
                self.logger.info("Using rule-based fallback labeler")
        except Exception as e:
            self.logger.error(f"Failed to initialize fallback labeler: {e}")
            raise
    
    async def extract_labels_for_repository(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract labels for a single repository using hybrid approach.
        
        Args:
            repo_data: Repository data dictionary
        
        Returns:
            Repository data with extracted labels
        """
        # Try primary labeler first (if available and not already failed)
        if self.primary_labeler and not self.using_fallback:
            try:
                result = await self.primary_labeler.extract_labels_for_repository(repo_data)
                return result
            except Exception as e:
                self.logger.warning(f"Primary labeler failed, switching to fallback: {e}")
                self.using_fallback = True
        
        # Use fallback labeler
        if self.fallback_labeler:
            return await self.fallback_labeler.extract_labels_for_repository(repo_data)
        else:
            # Return empty result if both fail
            result = repo_data.copy()
            result['llm_labels'] = {
                'labels': [],
                'confidence': 0.0,
                'source': 'failed',
                'model': 'none',
                'error': 'Both primary and fallback labelers failed',
                'timestamp': time.time()
            }
            return result
    
    async def extract_labels_batch(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract labels for a batch of repositories using hybrid approach.
        
        Args:
            repositories: List of repository data dictionaries
        
        Returns:
            List of repositories with extracted labels
        """
        # Try primary labeler first (if available and not already failed)
        if self.primary_labeler and not self.using_fallback:
            try:
                self.logger.info("Attempting batch processing with Flan-T5...")
                results = await self.primary_labeler.extract_labels_batch(repositories)
                return results
            except Exception as e:
                self.logger.warning(f"Primary labeler batch processing failed, switching to fallback: {e}")
                self.using_fallback = True
        
        # Use fallback labeler
        if self.fallback_labeler:
            self.logger.info("Using rule-based fallback for batch processing...")
            return await self.fallback_labeler.extract_labels_batch(repositories)
        else:
            # Return empty results if both fail
            return [self._create_empty_result(repo) for repo in repositories]
    
    def _create_empty_result(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an empty result for failed processing."""
        result = repo_data.copy()
        result['llm_labels'] = {
            'labels': [],
            'confidence': 0.0,
            'source': 'failed',
            'model': 'none',
            'error': 'All labelers failed',
            'timestamp': time.time()
        }
        return result
    
    def analyze_label_patterns(self, repositories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in extracted labels.
        
        Args:
            repositories: List of repositories with labels
        
        Returns:
            Analysis results
        """
        # Use whichever labeler is active
        active_labeler = self.fallback_labeler if self.using_fallback else self.primary_labeler
        if active_labeler:
            return active_labeler.analyze_label_patterns(repositories)
        else:
            return {
                'total_repositories': len(repositories),
                'repositories_with_labels': 0,
                'coverage_percentage': 0,
                'unique_labels': 0,
                'most_common_labels': [],
                'average_labels_per_repo': 0,
                'model_stats': {}
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        active_labeler = self.fallback_labeler if self.using_fallback else self.primary_labeler
        if active_labeler:
            stats = active_labeler.get_stats()
            stats['using_fallback'] = self.using_fallback
            stats['labeler_type'] = 'rule-based' if self.using_fallback else 'flan-t5'
            return stats
        else:
            return {
                'using_fallback': True,
                'labeler_type': 'failed',
                'error': 'No labelers available'
            }

# Maintain backward compatibility
LLMLabeler = HybridLabeler

async def main():
    """Test the hybrid labeler."""
    import time
    
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
    labeler = HybridLabeler()
    
    # Extract labels
    result = await labeler.extract_labels_for_repository(test_repo)
    
    print("Test Results:")
    print(f"Repository: {result['repo']}")
    print(f"Labels: {result['llm_labels']['labels']}")
    print(f"Source: {result['llm_labels']['source']}")
    print(f"Confidence: {result['llm_labels']['confidence']}")
    print(f"Stats: {labeler.get_stats()}")

if __name__ == "__main__":
    asyncio.run(main()) 