"""
Data fetcher module for OpenDigger Repository Labeling POC.
Handles downloading repository lists and fetching metadata asynchronously.
"""

import asyncio
import aiohttp
import pandas as pd
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm.asyncio import tqdm

from .config import Config, get_config
from .utils import get_logger

class OpenDiggerDataFetcher:
    """Fetches repository data from OpenDigger API with async support."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the data fetcher.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics tracking
        self.stats = {
            'repos_downloaded': 0,
            'repos_filtered': 0,
            'metadata_fetched': 0,
            'metadata_failed': 0,
            'total_fetch_time': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=Config.TIMEOUT_SECONDS),
            headers={'User-Agent': 'OpenDigger-POC/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def download_repo_list(self) -> pd.DataFrame:
        """
        Download and parse the repository list from OpenDigger.
        
        Returns:
            DataFrame containing repository information
        """
        self.logger.info("Downloading repository list from OpenDigger...")
        
        try:
            # Download CSV synchronously
            df = pd.read_csv(Config.REPO_LIST_URL)
            self.stats['repos_downloaded'] = len(df)
            self.logger.info(f"Downloaded {len(df)} repositories")
            
            # Save to local file
            output_path = Path(Config.RAW_DATA_DIR) / "repo_list.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Repository list saved to {output_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download repository list: {e}")
            raise
    
    def filter_repositories(self, df: pd.DataFrame, sample_size: int) -> List[Dict[str, Any]]:
        """
        Filter and sample repositories from the dataframe.
        
        Args:
            df: Repository dataframe
            sample_size: Number of repositories to sample
        
        Returns:
            List of repository dictionaries
        """
        self.logger.info(f"Filtering repositories (sample size: {sample_size})")
        
        # Sample repositories
        if len(df) > sample_size:
            sampled_df = df.sample(n=sample_size, random_state=42)
        else:
            sampled_df = df
        
        # Convert to list of dictionaries
        repositories = []
        for _, row in sampled_df.iterrows():
            # Parse repository identifier from first column
            repo_string = str(row.iloc[0])
            repo_info = self._parse_repo_identifier(repo_string)
            
            if repo_info:
                repositories.append(repo_info)
        
        self.stats['repos_filtered'] = len(repositories)
        self.logger.info(f"Filtered to {len(repositories)} repositories")
        
        return repositories
    
    def _parse_repo_identifier(self, repo_string: str) -> Optional[Dict[str, Any]]:
        """
        Parse repository identifier string into components.
        
        Args:
            repo_string: Repository identifier
        
        Returns:
            Dictionary with repository information
        """
        try:
            # Handle different formats
            if repo_string.startswith('http'):
                # Full URL format
                parts = repo_string.replace('https://', '').replace('http://', '').split('/')
            else:
                # Simple format
                parts = repo_string.split('/')
            
            if len(parts) >= 3:
                platform = parts[0].replace('.com', '').replace('.org', '')
                org_or_user = parts[1]
                repo = parts[2]
                
                return {
                    'platform': platform,
                    'org_or_user': org_or_user,
                    'repo': repo,
                    'full_name': f"{org_or_user}/{repo}",
                    'repo_string': repo_string
                }
        except Exception as e:
            self.logger.debug(f"Failed to parse repository identifier '{repo_string}': {e}")
        
        return None
    
    async def fetch_multiple_repositories(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fetch metadata for multiple repositories asynchronously.
        
        Args:
            repositories: List of repository info dictionaries
        
        Returns:
            List of repositories with metadata
        """
        self.logger.info(f"Fetching metadata for {len(repositories)} repositories")
        start_time = time.time()
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
        
        async def fetch_single_repo(repo_info):
            async with semaphore:
                result = await self._fetch_repository_metadata(repo_info)
                # Add delay to respect rate limits
                await asyncio.sleep(Config.REQUEST_DELAY_SECONDS)
                return result
        
        # Create tasks for all repositories
        tasks = [fetch_single_repo(repo) for repo in repositories]
        
        # Execute with progress bar
        results = await tqdm.gather(*tasks, desc="Fetching metadata")
        
        # Filter out failed fetches
        successful_repos = [repo for repo in results if repo is not None]
        
        self.stats['metadata_fetched'] = len(successful_repos)
        self.stats['metadata_failed'] = len(repositories) - len(successful_repos)
        self.stats['total_fetch_time'] = time.time() - start_time
        
        self.logger.info(f"Successfully fetched {len(successful_repos)} repositories, {self.stats['metadata_failed']} failed")
        
        return successful_repos
    
    async def _fetch_repository_metadata(self, repo_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata for a single repository.
        
        Args:
            repo_info: Repository information dictionary
        
        Returns:
            Repository with metadata or None if failed
        """
        if not self.session:
            self.logger.error("Session not initialized")
            return None
        
        platform = repo_info['platform']
        org_or_user = repo_info['org_or_user']
        repo = repo_info['repo']
        
        # Construct metadata URL
        url = Config.get_metadata_url(platform, org_or_user, repo)
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    metadata = await response.json()
                    
                    # Combine original repo info with metadata
                    result = {**repo_info, **metadata}
                    
                    # Extract text content
                    text_content = self._extract_text_content(result)
                    result['text_content'] = text_content
                    result['text_length'] = len(text_content)
                    result['metadata_url'] = url
                    
                    # Basic validation
                    if self._validate_repository(result):
                        return result
                    else:
                        self.logger.debug(f"Repository {repo_info['full_name']} failed validation")
                        return None
                else:
                    self.logger.debug(f"Failed to fetch {repo_info['full_name']}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.debug(f"Error fetching {repo_info['full_name']}: {e}")
            return None
    
    def _extract_text_content(self, metadata: Dict[str, Any]) -> str:
        """
        Extract and combine text content from repository metadata.
        
        Args:
            metadata: Repository metadata dictionary
        
        Returns:
            Combined text content
        """
        text_parts = []
        
        # Extract description
        if metadata.get('description'):
            text_parts.append(str(metadata['description']))
        
        # Extract README content (if available)
        if metadata.get('readme'):
            text_parts.append(str(metadata['readme']))
        
        # Extract topics/tags
        if metadata.get('topics'):
            if isinstance(metadata['topics'], list):
                text_parts.append(' '.join(str(topic) for topic in metadata['topics']))
            else:
                text_parts.append(str(metadata['topics']))
        
        # Extract language information
        if metadata.get('language'):
            text_parts.append(f"Primary language: {metadata['language']}")
        
        # Extract additional fields that might contain useful text
        text_fields = ['homepage', 'documentation', 'wiki']
        for field in text_fields:
            if metadata.get(field):
                text_parts.append(f"{field}: {metadata[field]}")
        
        return ' '.join(text_parts)
    
    def _validate_repository(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate repository metadata against criteria.
        
        Args:
            metadata: Repository metadata dictionary
        
        Returns:
            True if repository meets validation criteria
        """
        # Check minimum description length
        description = metadata.get('description', '')
        config = get_config()
        if len(str(description)) < config.processing.min_description_length:
            return False
        
        # Ensure we have some text content
        text_content = metadata.get('text_content', '')
        if len(text_content) < 20:  # Minimum meaningful content
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()

async def main():
    """Test the data fetcher."""
    async with OpenDiggerDataFetcher() as fetcher:
        # Download repository list
        df = fetcher.download_repo_list()
        
        # Filter repositories
        repos = fetcher.filter_repositories(df, 5)
        
        # Fetch metadata
        repositories = await fetcher.fetch_multiple_repositories(repos)
        
        print(f"Successfully processed {len(repositories)} repositories")
        for repo in repositories:
            print(f"- {repo['full_name']}: {repo['text_length']} chars")

if __name__ == "__main__":
    asyncio.run(main()) 