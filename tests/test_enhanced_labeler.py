"""Tests for the Enhanced LLM Labeler module.

This module contains comprehensive tests for the enhanced labeling functionality,
including pattern matching, semantic clustering, and confidence scoring.
"""

import asyncio
import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.llm_labeler_enhanced import EnhancedLLMLabeler
from src.mock_data_fetcher import MockDataFetcher
from src.utils import get_logger


class TestEnhancedLLMLabeler:
    """Test suite for Enhanced LLM Labeler."""
    
    @pytest.fixture
    def labeler(self):
        """Create an Enhanced LLM Labeler instance for testing."""
        return EnhancedLLMLabeler()
    
    @pytest.fixture
    def sample_repo_data(self):
        """Sample repository data for testing."""
        return {
            "name": "tensorflow",
            "platform": "github",
            "org_or_user": "tensorflow",
            "full_name": "tensorflow/tensorflow",
            "description": "An Open Source Machine Learning Framework for Everyone",
            "language": "Python",
            "stars": 185000,
            "forks": 74000,
            "topics": ["machine-learning", "deep-learning", "neural-network", "tensorflow", "python", "ai"],
            "readme": "# TensorFlow\n\nTensorFlow is an end-to-end open source platform for machine learning.",
            "license": "Apache-2.0",
            "created_at": "2015-11-07T01:19:20Z",
            "updated_at": "2024-01-15T14:22:00Z"
        }
    
    def test_initialization(self, labeler):
        """Test labeler initialization."""
        assert labeler is not None
        assert hasattr(labeler, '_enhanced_patterns')
        assert hasattr(labeler, '_domain_taxonomy')
        assert hasattr(labeler, '_semantic_clusters')
        assert hasattr(labeler, '_context_rules')
    
    def test_pattern_creation(self, labeler):
        """Test enhanced pattern creation."""
        patterns = labeler._enhanced_patterns
        
        # Check main categories exist
        assert 'programming_languages' in patterns
        assert 'frameworks_libraries' in patterns
        assert 'technical_domains' in patterns
        
        # Check specific patterns
        assert 'python' in patterns['programming_languages']
        assert 'javascript' in patterns['programming_languages']
        assert 'react' in patterns['frameworks_libraries']
        assert 'machine-learning' in patterns['technical_domains']
    
    def test_taxonomy_creation(self, labeler):
        """Test domain taxonomy creation."""
        taxonomy = labeler._domain_taxonomy
        
        # Check hierarchical structure
        assert 'software-development' in taxonomy
        assert 'web-development' in taxonomy
        assert 'machine-learning' in taxonomy
        
        # Check parent-child relationships
        web_dev = taxonomy.get('web-development', {})
        assert web_dev.get('parent') == 'software-development'
        assert 'children' in web_dev
    
    def test_semantic_clusters(self, labeler):
        """Test semantic cluster creation."""
        clusters = labeler._semantic_clusters
        
        # Check cluster categories
        assert 'web_frontend' in clusters
        assert 'machine_learning' in clusters
        assert 'devops_tools' in clusters
        
        # Check cluster contents
        ml_cluster = clusters.get('machine_learning', [])
        assert 'tensorflow' in ml_cluster
        assert 'pytorch' in ml_cluster
    
    def test_text_extraction(self, labeler, sample_repo_data):
        """Test text content extraction."""
        text = labeler._extract_text_content(sample_repo_data)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert 'tensorflow' in text.lower()
        assert 'machine learning' in text.lower()
        assert 'python' in text.lower()
    
    def test_repository_analysis(self, labeler, sample_repo_data):
        """Test repository structure analysis."""
        analysis = labeler._analyze_repository_structure(sample_repo_data)
        
        assert isinstance(analysis, dict)
        assert 'is_active' in analysis
        assert 'size_category' in analysis
        assert 'popularity_score' in analysis
        
        # Check popularity score calculation
        popularity = analysis['popularity_score']
        assert 0.0 <= popularity <= 1.0
    
    def test_metadata_insights(self, labeler, sample_repo_data):
        """Test metadata insight extraction."""
        insights = labeler._extract_metadata_insights(sample_repo_data)
        
        assert isinstance(insights, dict)
        assert 'license_type' in insights
        assert 'creation_age' in insights
        assert 'community_engagement' in insights
        assert 'development_stage' in insights
    
    def test_confidence_calculation(self, labeler):
        """Test pattern confidence calculation."""
        from src.llm_labeler_enhanced import PatternConfig
        
        pattern_config = PatternConfig(
            patterns=[r'\b(?:python|tensorflow)\b'],
            weight=1.0,
            context_boost=['machine', 'learning'],
            confidence_base=0.8,
            category='test'
        )
        
        text = "This is a Python TensorFlow machine learning project"
        confidence = labeler._calculate_pattern_confidence(text, pattern_config)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.8  # Should be boosted by context
    
    @pytest.mark.asyncio
    async def test_label_extraction(self, labeler, sample_repo_data):
        """Test label extraction for a repository."""
        result = await labeler.extract_labels_for_repository(sample_repo_data)
        
        assert isinstance(result, dict)
        assert 'success' in result or 'labels' in result
        
        # If successful, check label structure
        if result.get('success', True):
            labels = result.get('labels', [])
            assert isinstance(labels, list)
            
            # Check label format
            for label in labels:
                if isinstance(label, dict):
                    assert 'label' in label
                    assert 'confidence' in label
                    assert 'source' in label
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, labeler):
        """Test batch label extraction."""
        # Create sample repositories
        repos = [
            {
                "name": "react",
                "description": "A JavaScript library for building user interfaces",
                "language": "JavaScript",
                "topics": ["javascript", "react", "frontend"]
            },
            {
                "name": "django",
                "description": "The Web framework for perfectionists with deadlines",
                "language": "Python",
                "topics": ["python", "web", "framework"]
            }
        ]
        
        results = await labeler.extract_labels_batch(repos)
        
        assert isinstance(results, list)
        assert len(results) == len(repos)
        
        # Check each result
        for result in results:
            assert isinstance(result, dict)
    
    def test_statistics_tracking(self, labeler):
        """Test statistics tracking functionality."""
        stats = labeler.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_requests' in stats
        assert 'successful_requests' in stats
        assert 'failed_requests' in stats
        assert 'success_rate' in stats


class TestIntegrationWithMockData:
    """Integration tests using mock data."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test full pipeline with mock data."""
        # Initialize components
        data_fetcher = MockDataFetcher()
        labeler = EnhancedLLMLabeler()
        
        async with data_fetcher:
            # Get repository list
            repo_list = await data_fetcher.fetch_repository_list()
            assert len(repo_list) > 0
            
            # Process first repository
            repo_id = repo_list[0]
            repo_data = await data_fetcher.fetch_repository_metadata(repo_id)
            assert repo_data is not None
            
            # Extract labels
            result = await labeler.extract_labels_for_repository(repo_data)
            assert result is not None
            
            # Verify result structure
            if result.get('success', True):
                labels = result.get('labels', [])
                print(f"Extracted {len(labels)} labels for {repo_data['name']}")
                
                # Print labels for manual verification
                for label in labels[:5]:  # Show first 5 labels
                    if isinstance(label, dict):
                        print(f"  - {label.get('label', 'N/A')}: {label.get('confidence', 0):.2f}")
    
    @pytest.mark.asyncio
    async def test_multiple_repositories(self):
        """Test processing multiple repositories."""
        data_fetcher = MockDataFetcher()
        labeler = EnhancedLLMLabeler()
        
        async with data_fetcher:
            repo_list = await data_fetcher.fetch_repository_list()
            
            # Process first 3 repositories
            test_repos = []
            for repo_id in repo_list[:3]:
                repo_data = await data_fetcher.fetch_repository_metadata(repo_id)
                if repo_data:
                    test_repos.append(repo_data)
            
            # Batch process
            results = await labeler.extract_labels_batch(test_repos)
            
            assert len(results) == len(test_repos)
            
            # Verify each result
            for i, result in enumerate(results):
                repo_name = test_repos[i]['name']
                print(f"\nRepository: {repo_name}")
                
                if result.get('success', True):
                    labels = result.get('labels', [])
                    print(f"  Labels found: {len(labels)}")
                    
                    # Show top labels
                    for label in labels[:3]:
                        if isinstance(label, dict):
                            print(f"    - {label.get('label', 'N/A')}: {label.get('confidence', 0):.2f}")
                else:
                    print(f"  Failed: {result.get('error', 'Unknown error')}")


def run_manual_tests():
    """Run manual tests for development and debugging."""
    print("Running Enhanced LLM Labeler Tests")
    print("=" * 50)
    
    async def test_sample_repositories():
        """Test with sample repositories."""
        labeler = EnhancedLLMLabeler()
        
        # Test repositories with different characteristics
        test_repos = [
            {
                "name": "tensorflow",
                "description": "An Open Source Machine Learning Framework for Everyone",
                "language": "Python",
                "topics": ["machine-learning", "deep-learning", "tensorflow", "python", "ai"],
                "readme": "TensorFlow is an end-to-end open source platform for machine learning."
            },
            {
                "name": "react",
                "description": "A declarative, efficient, and flexible JavaScript library for building user interfaces",
                "language": "JavaScript", 
                "topics": ["javascript", "react", "frontend", "ui", "library"],
                "readme": "React makes it easy to create interactive UIs."
            },
            {
                "name": "kubernetes",
                "description": "Production-Grade Container Scheduling and Management",
                "language": "Go",
                "topics": ["kubernetes", "container", "orchestration", "docker", "devops"],
                "readme": "Kubernetes is an open-source system for automating deployment."
            }
        ]
        
        for repo in test_repos:
            print(f"\nTesting repository: {repo['name']}")
            print(f"Description: {repo['description']}")
            print(f"Language: {repo['language']}")
            print(f"Topics: {repo['topics']}")
            
            result = await labeler.extract_labels_for_repository(repo)
            
            if result.get('success', True):
                labels = result.get('labels', [])
                print(f"Extracted {len(labels)} labels:")
                
                for label in labels:
                    if isinstance(label, dict):
                        print(f"  - {label.get('label', 'N/A')}: {label.get('confidence', 0):.2f} ({label.get('source', 'unknown')})")
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}")
        
        # Print final statistics
        stats = labeler.get_stats()
        print(f"\nFinal Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Average processing time: {stats['average_processing_time']:.3f}s")
    
    # Run the test
    asyncio.run(test_sample_repositories())


if __name__ == "__main__":
    # Run manual tests if executed directly
    run_manual_tests() 