"""
LLM labeler module for OpenDigger Repository Labeling POC.
Handles label extraction using Google Flan-T5 model from Hugging Face.
"""

import asyncio
import json
import logging
import time
import re
import torch
from typing import Dict, List, Optional, Any, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm.asyncio import tqdm
import concurrent.futures
from threading import Lock

from .config import Config
from .utils import get_logger, retry_on_failure

class FlanT5Labeler:
    """Extracts technical domain labels using Google Flan-T5 model."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Flan-T5 labeler.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_lock = Lock()
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'model_load_time': 0,
            'inference_time': 0,
            'error_types': {}
        }
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the Flan-T5 model and tokenizer."""
        try:
            start_time = time.time()
            self.logger.info(f"Loading Flan-T5 model: {Config.FLAN_T5_MODEL_NAME}")
            
            # Determine device
            if Config.USE_GPU and torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.logger.info("Using GPU for inference")
            else:
                self.device = torch.device("cpu")
                self.logger.info("Using CPU for inference")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(
                Config.FLAN_T5_MODEL_NAME,
                cache_dir=Config.FLAN_T5_CACHE_DIR
            )
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(
                Config.FLAN_T5_MODEL_NAME,
                cache_dir=Config.FLAN_T5_CACHE_DIR,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start_time
            self.stats['model_load_time'] = load_time
            self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to load Flan-T5 model: {e}")
            raise
    
    def create_labeling_prompt(self, repo_data: Dict[str, Any]) -> str:
        """
        Create a prompt for Flan-T5 label extraction.
        
        Args:
            repo_data: Repository data dictionary
        
        Returns:
            Formatted prompt string optimized for Flan-T5
        """
        # Extract relevant information
        repo_name = repo_data.get('repo', 'Unknown')
        org_name = repo_data.get('org_or_user', 'Unknown')
        platform = repo_data.get('platform', 'Unknown')
        
        # Get processed text content
        processed_text = repo_data.get('processed_text', {})
        description = processed_text.get('cleaned_text', '')
        technical_terms = processed_text.get('technical_terms', [])
        keywords = processed_text.get('keywords', [])
        
        # Extract top keywords for context
        top_keywords = [kw[0] if isinstance(kw, tuple) else str(kw) for kw in keywords[:8]]
        
        # Create a concise prompt optimized for Flan-T5
        prompt = f"""Classify this software repository into technical domain labels.

Repository: {repo_name} by {org_name}
Description: {description[:200]}
Technical terms: {', '.join(technical_terms[:10])}
Keywords: {', '.join(top_keywords)}

Task: Generate 3-6 specific technical labels for this repository. Focus on:
- Programming languages and frameworks
- Application domain (web, mobile, data, etc.)
- Specific technologies used

Output format: Provide labels as a comma-separated list.
Labels:"""

        return prompt.strip()
    
    def _generate_labels(self, prompt: str) -> Optional[List[str]]:
        """
        Generate labels using Flan-T5 model.
        
        Args:
            prompt: Input prompt
        
        Returns:
            List of extracted labels or None if failed
        """
        if not self.model or not self.tokenizer:
            self.logger.error("Model not loaded")
            return None
        
        try:
            start_time = time.time()
            
            with self.model_lock:
                # Tokenize input
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=Config.MAX_MODEL_LENGTH,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=Config.LLM_MAX_TOKENS,
                        temperature=Config.LLM_TEMPERATURE,
                        do_sample=Config.LLM_DO_SAMPLE,
                        top_p=Config.LLM_TOP_P,
                        top_k=Config.LLM_TOP_K,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                # Decode response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            inference_time = time.time() - start_time
            self.stats['inference_time'] += inference_time
            
            # Parse labels from response
            labels = self._parse_labels_from_response(response)
            
            if labels:
                self.stats['successful_requests'] += 1
                return labels
            else:
                self.stats['failed_requests'] += 1
                return None
                
        except Exception as e:
            error_type = type(e).__name__
            self.stats['error_types'][error_type] = self.stats['error_types'].get(error_type, 0) + 1
            self.logger.debug(f"Flan-T5 inference error: {e}")
            self.stats['failed_requests'] += 1
            return None
    
    def _parse_labels_from_response(self, response: str) -> List[str]:
        """
        Parse labels from Flan-T5 response.
        
        Args:
            response: Raw model response
        
        Returns:
            List of parsed labels
        """
        try:
            # Clean the response
            response = response.strip()
            
            # Try to extract comma-separated labels
            if ',' in response:
                labels = [label.strip() for label in response.split(',')]
            else:
                # Try to extract labels from various formats
                labels = []
                
                # Look for bullet points or numbered lists
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        # Remove common prefixes
                        line = re.sub(r'^[-*•]\s*', '', line)
                        line = re.sub(r'^\d+\.\s*', '', line)
                        if line and len(line) < 50:  # Reasonable label length
                            labels.append(line)
                
                # If no structured format, try to extract from text
                if not labels:
                    # Look for potential labels (words/phrases)
                    words = response.split()
                    potential_labels = []
                    for word in words:
                        word = word.strip('.,!?;:"()[]{}')
                        if len(word) > 2 and len(word) < 30:
                            potential_labels.append(word)
                    labels = potential_labels[:6]  # Limit to 6 labels
            
            # Clean and filter labels
            cleaned_labels = []
            for label in labels:
                if isinstance(label, str):
                    label = label.strip().lower()
                    label = re.sub(r'[^\w\s-]', '', label)  # Remove special chars except hyphens
                    label = re.sub(r'\s+', '-', label)  # Replace spaces with hyphens
                    
                    if label and len(label) > 1 and len(label) < 30:
                        cleaned_labels.append(label)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_labels = []
            for label in cleaned_labels:
                if label not in seen:
                    seen.add(label)
                    unique_labels.append(label)
            
            return unique_labels[:Config.MAX_LABELS_PER_REPO]
            
        except Exception as e:
            self.logger.debug(f"Error parsing labels from response: {e}")
            return []
    
    async def extract_labels_for_repository(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract labels for a single repository using Flan-T5.
        
        Args:
            repo_data: Repository data dictionary
        
        Returns:
            Repository data with extracted labels
        """
        self.stats['total_requests'] += 1
        
        try:
            # Create prompt
            prompt = self.create_labeling_prompt(repo_data)
            
            # Generate labels using thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                labels = await loop.run_in_executor(executor, self._generate_labels, prompt)
            
            # Prepare result
            result = repo_data.copy()
            result['llm_labels'] = {
                'labels': labels or [],
                'confidence': 0.8 if labels else 0.0,  # Fixed confidence for local model
                'source': 'flan-t5',
                'model': Config.FLAN_T5_MODEL_NAME,
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
                'source': 'flan-t5',
                'model': Config.FLAN_T5_MODEL_NAME,
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
        self.logger.info(f"Starting label extraction for {len(repositories)} repositories using Flan-T5")
        
        # Process repositories with progress bar
        results = []
        
        # Use semaphore to limit concurrent processing (since we're using a single model)
        semaphore = asyncio.Semaphore(1)  # Process one at a time to avoid memory issues
        
        async def process_with_semaphore(repo_data):
            async with semaphore:
                return await self.extract_labels_for_repository(repo_data)
        
        # Process all repositories
        tasks = [process_with_semaphore(repo) for repo in repositories]
        results = await tqdm.gather(*tasks, desc="Extracting labels with Flan-T5")
        
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
LLMLabeler = FlanT5Labeler

async def main():
    """Test the Flan-T5 labeler."""
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
    labeler = FlanT5Labeler()
    
    # Extract labels
    result = await labeler.extract_labels_for_repository(test_repo)
    
    print("Test Results:")
    print(f"Repository: {result['repo']}")
    print(f"Labels: {result['llm_labels']['labels']}")
    print(f"Confidence: {result['llm_labels']['confidence']}")
    print(f"Stats: {labeler.get_stats()}")

if __name__ == "__main__":
    asyncio.run(main()) 