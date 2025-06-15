"""
Mock data fetcher for OpenDigger Repository Labeling POC.

This module provides mock repository data for testing and development purposes
when the actual OpenDigger API is not available or for offline testing.
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta

from .config import get_config
from .utils import get_logger, format_timestamp


class MockDataFetcher:
    """Mock data fetcher that generates realistic repository data for testing."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the mock data fetcher.
        
        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.config = get_config()
        self.logger = logger or get_logger(__name__)
        self._mock_repositories = self._generate_mock_repositories()
    
    def _generate_mock_repositories(self) -> List[Dict[str, Any]]:
        """Generate a list of mock repository data.
        
        Returns:
            List of mock repository dictionaries
        """
        repositories = [
            {
                "name": "react",
                "platform": "github",
                "org_or_user": "facebook",
                "full_name": "facebook/react",
                "description": "A declarative, efficient, and flexible JavaScript library for building user interfaces.",
                "language": "JavaScript",
                "stars": 218000,
                "forks": 45000,
                "created_at": "2013-05-24T16:15:54Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "topics": ["javascript", "react", "frontend", "ui", "library"],
                "readme": "# React\n\nReact is a JavaScript library for building user interfaces. It lets you compose complex UIs from small and isolated pieces of code called components.",
                "license": "MIT",
                "homepage": "https://reactjs.org/",
                "size": 15420,
                "open_issues": 892,
                "watchers": 218000,
                "default_branch": "main",
                "archived": False,
                "disabled": False,
                "private": False,
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_pages": True,
                "has_downloads": True
            },
            {
                "name": "tensorflow",
                "platform": "github",
                "org_or_user": "tensorflow",
                "full_name": "tensorflow/tensorflow",
                "description": "An Open Source Machine Learning Framework for Everyone",
                "language": "C++",
                "stars": 185000,
                "forks": 74000,
                "created_at": "2015-11-07T01:19:20Z",
                "updated_at": "2024-01-15T14:22:00Z",
                "topics": ["machine-learning", "deep-learning", "neural-network", "tensorflow", "python", "ai"],
                "readme": "# TensorFlow\n\nTensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources.",
                "license": "Apache-2.0",
                "homepage": "https://www.tensorflow.org/",
                "size": 245680,
                "open_issues": 2156,
                "watchers": 185000,
                "default_branch": "master",
                "archived": False,
                "disabled": False,
                "private": False,
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_pages": True,
                "has_downloads": True
            },
            {
                "name": "vscode",
                "platform": "github",
                "org_or_user": "microsoft",
                "full_name": "microsoft/vscode",
                "description": "Visual Studio Code",
                "language": "TypeScript",
                "stars": 162000,
                "forks": 28500,
                "created_at": "2015-09-03T20:23:12Z",
                "updated_at": "2024-01-15T16:45:00Z",
                "topics": ["editor", "typescript", "electron", "vscode"],
                "readme": "# Visual Studio Code\n\nVisual Studio Code is a lightweight but powerful source code editor which runs on your desktop and is available for Windows, macOS and Linux.",
                "license": "MIT",
                "homepage": "https://code.visualstudio.com/",
                "size": 89234,
                "open_issues": 5432,
                "watchers": 162000,
                "default_branch": "main",
                "archived": False,
                "disabled": False,
                "private": False,
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_pages": True,
                "has_downloads": True
            },
            {
                "name": "kubernetes",
                "platform": "github",
                "org_or_user": "kubernetes",
                "full_name": "kubernetes/kubernetes",
                "description": "Production-Grade Container Scheduling and Management",
                "language": "Go",
                "stars": 109000,
                "forks": 39000,
                "created_at": "2014-06-06T22:56:04Z",
                "updated_at": "2024-01-15T12:18:00Z",
                "topics": ["kubernetes", "container", "orchestration", "docker", "devops", "cloud-native"],
                "readme": "# Kubernetes\n\nKubernetes is an open-source system for automating deployment, scaling, and management of containerized applications.",
                "license": "Apache-2.0",
                "homepage": "https://kubernetes.io/",
                "size": 156789,
                "open_issues": 2789,
                "watchers": 109000,
                "default_branch": "master",
                "archived": False,
                "disabled": False,
                "private": False,
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_pages": True,
                "has_downloads": True
            },
            {
                "name": "django",
                "platform": "github",
                "org_or_user": "django",
                "full_name": "django/django",
                "description": "The Web framework for perfectionists with deadlines.",
                "language": "Python",
                "stars": 78000,
                "forks": 31000,
                "created_at": "2012-04-28T02:47:18Z",
                "updated_at": "2024-01-15T09:33:00Z",
                "topics": ["python", "django", "web", "framework", "mvc"],
                "readme": "# Django\n\nDjango is a high-level Python Web framework that encourages rapid development and clean, pragmatic design.",
                "license": "BSD-3-Clause",
                "homepage": "https://www.djangoproject.com/",
                "size": 45123,
                "open_issues": 234,
                "watchers": 78000,
                "default_branch": "main",
                "archived": False,
                "disabled": False,
                "private": False,
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_pages": True,
                "has_downloads": True
            },
            {
                "name": "flutter",
                "platform": "github",
                "org_or_user": "flutter",
                "full_name": "flutter/flutter",
                "description": "Flutter makes it easy and fast to build beautiful apps for mobile and beyond",
                "language": "Dart",
                "stars": 164000,
                "forks": 27000,
                "created_at": "2015-03-06T22:54:58Z",
                "updated_at": "2024-01-15T11:27:00Z",
                "topics": ["flutter", "dart", "mobile", "cross-platform", "ui"],
                "readme": "# Flutter\n\nFlutter is Google's UI toolkit for building beautiful, natively compiled applications for mobile, web, and desktop from a single codebase.",
                "license": "BSD-3-Clause",
                "homepage": "https://flutter.dev/",
                "size": 98765,
                "open_issues": 12456,
                "watchers": 164000,
                "default_branch": "master",
                "archived": False,
                "disabled": False,
                "private": False,
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_pages": True,
                "has_downloads": True
            },
            {
                "name": "pytorch",
                "platform": "github",
                "org_or_user": "pytorch",
                "full_name": "pytorch/pytorch",
                "description": "Tensors and Dynamic neural networks in Python with strong GPU acceleration",
                "language": "Python",
                "stars": 81000,
                "forks": 21800,
                "created_at": "2016-08-13T17:41:24Z",
                "updated_at": "2024-01-15T13:55:00Z",
                "topics": ["pytorch", "machine-learning", "deep-learning", "neural-networks", "python", "gpu"],
                "readme": "# PyTorch\n\nPyTorch is a Python package that provides two high-level features: Tensor computation with strong GPU acceleration and Deep neural networks built on a tape-based autograd system.",
                "license": "BSD-3-Clause",
                "homepage": "https://pytorch.org/",
                "size": 187432,
                "open_issues": 13789,
                "watchers": 81000,
                "default_branch": "master",
                "archived": False,
                "disabled": False,
                "private": False,
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_pages": True,
                "has_downloads": True
            },
            {
                "name": "bitcoin",
                "platform": "github",
                "org_or_user": "bitcoin",
                "full_name": "bitcoin/bitcoin",
                "description": "Bitcoin Core integration/staging tree",
                "language": "C++",
                "stars": 77000,
                "forks": 35600,
                "created_at": "2010-12-19T15:16:43Z",
                "updated_at": "2024-01-15T08:42:00Z",
                "topics": ["bitcoin", "cryptocurrency", "blockchain", "p2p", "decentralized"],
                "readme": "# Bitcoin Core\n\nBitcoin Core is the reference implementation of the bitcoin system, currently developed by the Bitcoin Core project.",
                "license": "MIT",
                "homepage": "https://bitcoincore.org/",
                "size": 234567,
                "open_issues": 987,
                "watchers": 77000,
                "default_branch": "master",
                "archived": False,
                "disabled": False,
                "private": False,
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_pages": True,
                "has_downloads": True
            },
            {
                "name": "ansible",
                "platform": "github",
                "org_or_user": "ansible",
                "full_name": "ansible/ansible",
                "description": "Ansible is a radically simple IT automation platform",
                "language": "Python",
                "stars": 62000,
                "forks": 23800,
                "created_at": "2012-03-06T14:58:02Z",
                "updated_at": "2024-01-15T15:12:00Z",
                "topics": ["ansible", "automation", "devops", "infrastructure", "configuration-management"],
                "readme": "# Ansible\n\nAnsible is a radically simple IT automation platform that makes your applications and systems easier to deploy and maintain.",
                "license": "GPL-3.0",
                "homepage": "https://www.ansible.com/",
                "size": 67890,
                "open_issues": 1456,
                "watchers": 62000,
                "default_branch": "devel",
                "archived": False,
                "disabled": False,
                "private": False,
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_pages": True,
                "has_downloads": True
            },
            {
                "name": "elasticsearch",
                "platform": "github",
                "org_or_user": "elastic",
                "full_name": "elastic/elasticsearch",
                "description": "Free and Open, Distributed, RESTful Search Engine",
                "language": "Java",
                "stars": 69000,
                "forks": 24500,
                "created_at": "2010-02-08T13:20:56Z",
                "updated_at": "2024-01-15T17:08:00Z",
                "topics": ["elasticsearch", "search", "analytics", "java", "distributed", "lucene"],
                "readme": "# Elasticsearch\n\nElasticsearch is a distributed, RESTful search and analytics engine capable of addressing a growing number of use cases.",
                "license": "Apache-2.0",
                "homepage": "https://www.elastic.co/products/elasticsearch",
                "size": 145678,
                "open_issues": 4321,
                "watchers": 69000,
                "default_branch": "master",
                "archived": False,
                "disabled": False,
                "private": False,
                "has_issues": True,
                "has_projects": True,
                "has_wiki": True,
                "has_pages": True,
                "has_downloads": True
            }
        ]
        
        # Add some randomization to make data more realistic
        for repo in repositories:
            # Add random activity metrics
            repo["commits_count"] = random.randint(1000, 50000)
            repo["contributors_count"] = random.randint(10, 2000)
            repo["releases_count"] = random.randint(5, 500)
            
            # Add random metadata
            repo["has_discussions"] = random.choice([True, False])
            repo["has_security_policy"] = random.choice([True, False])
            repo["vulnerability_alerts"] = random.choice([True, False])
            
            # Add package manager info
            if repo["language"] == "JavaScript":
                repo["package_manager"] = "npm"
                repo["package_name"] = f"@{repo['org_or_user']}/{repo['name']}"
            elif repo["language"] == "Python":
                repo["package_manager"] = "pip"
                repo["package_name"] = repo["name"]
            elif repo["language"] == "Java":
                repo["package_manager"] = "maven"
                repo["package_name"] = f"{repo['org_or_user']}.{repo['name']}"
            
            # Add CI/CD info
            repo["has_ci"] = random.choice([True, False])
            if repo["has_ci"]:
                repo["ci_provider"] = random.choice(["github-actions", "travis-ci", "circleci", "jenkins"])
        
        return repositories
    
    async def fetch_repository_list(self) -> List[str]:
        """Fetch the list of repository identifiers.
        
        Returns:
            List of repository identifiers in format 'platform/org_or_user/repo'
        """
        self.logger.info("Fetching mock repository list")
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        repo_list = []
        for repo in self._mock_repositories:
            identifier = f"{repo['platform']}/{repo['org_or_user']}/{repo['name']}"
            repo_list.append(identifier)
        
        self.logger.info(f"Retrieved {len(repo_list)} mock repositories")
        return repo_list
    
    async def fetch_repository_metadata(self, repo_identifier: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a specific repository.
        
        Args:
            repo_identifier: Repository identifier in format 'platform/org_or_user/repo'
            
        Returns:
            Repository metadata dictionary or None if not found
        """
        try:
            # Parse repository identifier
            parts = repo_identifier.split('/')
            if len(parts) != 3:
                self.logger.warning(f"Invalid repository identifier format: {repo_identifier}")
                return None
            
            platform, org_or_user, repo_name = parts
            
            # Find matching repository
            for repo in self._mock_repositories:
                if (repo['platform'] == platform and 
                    repo['org_or_user'] == org_or_user and 
                    repo['name'] == repo_name):
                    
                    # Simulate network delay
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    
                    # Add some dynamic metadata
                    metadata = repo.copy()
                    metadata.update({
                        "fetched_at": format_timestamp(),
                        "api_version": "mock-v1.0",
                        "cache_status": "fresh"
                    })
                    
                    self.logger.debug(f"Retrieved metadata for {repo_identifier}")
                    return metadata
            
            self.logger.warning(f"Repository not found: {repo_identifier}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching metadata for {repo_identifier}: {e}")
            return None
    
    async def fetch_repositories_batch(
        self, 
        repo_identifiers: List[str],
        max_concurrent: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Fetch multiple repositories concurrently.
        
        Args:
            repo_identifiers: List of repository identifiers
            max_concurrent: Maximum concurrent requests (uses config default if None)
            
        Yields:
            Repository metadata dictionaries
        """
        max_concurrent = max_concurrent or self.config.api.max_concurrent_requests
        
        self.logger.info(f"Fetching {len(repo_identifiers)} repositories with max {max_concurrent} concurrent requests")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(repo_id: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.fetch_repository_metadata(repo_id)
        
        # Create tasks for all repositories
        tasks = [fetch_with_semaphore(repo_id) for repo_id in repo_identifiers]
        
        # Process results as they complete
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                if result:
                    yield result
            except Exception as e:
                self.logger.error(f"Error in batch fetch: {e}")
    
    async def get_repository_statistics(self) -> Dict[str, Any]:
        """Get statistics about available mock repositories.
        
        Returns:
            Dictionary containing repository statistics
        """
        self.logger.info("Calculating repository statistics")
        
        stats = {
            "total_repositories": len(self._mock_repositories),
            "languages": {},
            "platforms": {},
            "total_stars": 0,
            "total_forks": 0,
            "average_size": 0,
            "license_distribution": {},
            "topic_frequency": {},
            "creation_years": {}
        }
        
        total_size = 0
        
        for repo in self._mock_repositories:
            # Language distribution
            language = repo.get("language", "Unknown")
            stats["languages"][language] = stats["languages"].get(language, 0) + 1
            
            # Platform distribution
            platform = repo.get("platform", "Unknown")
            stats["platforms"][platform] = stats["platforms"].get(platform, 0) + 1
            
            # Aggregate metrics
            stats["total_stars"] += repo.get("stars", 0)
            stats["total_forks"] += repo.get("forks", 0)
            total_size += repo.get("size", 0)
            
            # License distribution
            license_name = repo.get("license", "Unknown")
            stats["license_distribution"][license_name] = stats["license_distribution"].get(license_name, 0) + 1
            
            # Topic frequency
            for topic in repo.get("topics", []):
                stats["topic_frequency"][topic] = stats["topic_frequency"].get(topic, 0) + 1
            
            # Creation year distribution
            created_at = repo.get("created_at", "")
            if created_at:
                try:
                    year = datetime.fromisoformat(created_at.replace('Z', '+00:00')).year
                    stats["creation_years"][str(year)] = stats["creation_years"].get(str(year), 0) + 1
                except ValueError:
                    pass
        
        # Calculate averages
        if stats["total_repositories"] > 0:
            stats["average_size"] = total_size / stats["total_repositories"]
            stats["average_stars"] = stats["total_stars"] / stats["total_repositories"]
            stats["average_forks"] = stats["total_forks"] / stats["total_repositories"]
        
        return stats
    
    def get_sample_repositories(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get a sample of repositories for testing.
        
        Args:
            count: Number of repositories to return
            
        Returns:
            List of sample repository dictionaries
        """
        sample_count = min(count, len(self._mock_repositories))
        return random.sample(self._mock_repositories, sample_count)
    
    def search_repositories(
        self, 
        query: str, 
        language: Optional[str] = None,
        min_stars: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search repositories by query, language, and minimum stars.
        
        Args:
            query: Search query (searches in name, description, topics)
            language: Filter by programming language
            min_stars: Minimum number of stars
            
        Returns:
            List of matching repositories
        """
        results = []
        query_lower = query.lower() if query else ""
        
        for repo in self._mock_repositories:
            # Language filter
            if language and repo.get("language", "").lower() != language.lower():
                continue
            
            # Stars filter
            if min_stars and repo.get("stars", 0) < min_stars:
                continue
            
            # Query filter
            if query_lower:
                searchable_text = " ".join([
                    repo.get("name", ""),
                    repo.get("description", ""),
                    " ".join(repo.get("topics", []))
                ]).lower()
                
                if query_lower not in searchable_text:
                    continue
            
            results.append(repo)
        
        # Sort by stars (descending)
        results.sort(key=lambda x: x.get("stars", 0), reverse=True)
        
        self.logger.info(f"Search for '{query}' returned {len(results)} results")
        return results
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.logger.debug("Mock data fetcher context entered")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.logger.debug("Mock data fetcher context exited")
        if exc_type:
            self.logger.error(f"Exception in mock data fetcher context: {exc_val}")
        return False

# For easy replacement
OpenDiggerDataFetcher = MockDataFetcher

async def main():
    """Test the mock data fetcher."""
    async with MockDataFetcher() as fetcher:
        # Download repository list
        df = fetcher.download_repo_list()
        
        # Filter repositories
        repos = fetcher.filter_repositories(df, 3)
        
        # Fetch metadata
        repositories = await fetcher.fetch_multiple_repositories(repos)
        
        print(f"Successfully processed {len(repositories)} repositories")
        for repo in repositories:
            print(f"- {repo['full_name']}: {repo['text_length']} chars")

if __name__ == "__main__":
    asyncio.run(main()) 