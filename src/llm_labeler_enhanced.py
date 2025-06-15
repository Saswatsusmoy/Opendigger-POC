"""
Enhanced LLM labeler module for OpenDigger Repository Labeling POC.
Uses advanced prompting techniques, semantic analysis, and sophisticated pattern matching.
"""

import asyncio
import json
import logging
import time
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import Counter, defaultdict
import concurrent.futures
import math

from .config import get_config
from .utils import get_logger, retry_on_failure

class EnhancedLLMLabeler:
    """Enhanced LLM labeler with advanced techniques for technical domain classification."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the enhanced LLM labeler.
        
        Args:
            logger: Optional logger instance
        """
        self.config = get_config()
        self.logger = logger or get_logger(__name__)
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'processing_time': 0,
            'error_types': {},
            'confidence_distribution': defaultdict(int),
            'label_sources': defaultdict(int)
        }
        
        # Enhanced technical patterns with weights and contexts
        self.enhanced_patterns = self._create_enhanced_patterns()
        
        # Domain taxonomy for hierarchical classification
        self.domain_taxonomy = self._create_domain_taxonomy()
        
        # Semantic similarity patterns
        self.semantic_clusters = self._create_semantic_clusters()
        
        # Context-aware rules
        self.context_rules = self._create_context_rules()
        
        self.logger.info("Enhanced LLM labeler initialized with advanced pattern matching")
    
    def _create_enhanced_patterns(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Create enhanced patterns with weights, contexts, and confidence scores."""
        return {
            'programming_languages': {
                'python': {
                    'patterns': [
                        r'\b(?:python|py|django|flask|fastapi|pandas|numpy|scipy|matplotlib|jupyter)\b',
                        r'\b(?:pip|conda|virtualenv|poetry|pyproject)\b'
                    ],
                    'weight': 1.0,
                    'context_boost': ['data', 'science', 'ml', 'ai', 'web', 'backend'],
                    'confidence_base': 0.9
                },
                'javascript': {
                    'patterns': [
                        r'\b(?:javascript|js|node|react|vue|angular|express|next|nuxt)\b',
                        r'\b(?:npm|yarn|webpack|babel|typescript|ts)\b'
                    ],
                    'weight': 1.0,
                    'context_boost': ['web', 'frontend', 'ui', 'browser', 'spa'],
                    'confidence_base': 0.9
                },
                'java': {
                    'patterns': [
                        r'\b(?:java|spring|maven|gradle|hibernate|junit)\b',
                        r'\b(?:jvm|openjdk|oracle|android|kotlin)\b',
                        r'\b(?:servlet|jsp|jsf|struts|wicket)\b'
                    ],
                    'weight': 1.0,
                    'context_boost': ['enterprise', 'backend', 'android', 'web'],
                    'confidence_base': 0.85
                },
                'typescript': {
                    'patterns': [
                        r'\b(?:typescript|ts|tsc|tsconfig|@types)\b',
                        r'\b(?:angular|nest|ionic|deno)\b'
                    ],
                    'weight': 0.9,
                    'context_boost': ['web', 'frontend', 'type-safe', 'large-scale'],
                    'confidence_base': 0.85
                }
            },
            
            'frameworks_libraries': {
                'react': {
                    'patterns': [
                        r'\b(?:react|jsx|tsx|next\.js|gatsby|create-react-app)\b',
                        r'\b(?:hooks|component|props|state|redux|context)\b',
                        r'\b(?:material-ui|chakra|styled-components)\b'
                    ],
                    'weight': 1.0,
                    'context_boost': ['frontend', 'ui', 'spa', 'component'],
                    'confidence_base': 0.9
                },
                'tensorflow': {
                    'patterns': [
                        r'\b(?:tensorflow|tf|keras|tensorboard|tfx)\b',
                        r'\b(?:neural|network|deep|learning|model|training)\b'
                    ],
                    'weight': 1.0,
                    'context_boost': ['ml', 'ai', 'deep-learning', 'neural'],
                    'confidence_base': 0.95
                },
                'docker': {
                    'patterns': [
                        r'\b(?:docker|dockerfile|container|image|registry)\b',
                        r'\b(?:compose|swarm|buildx|multi-stage)\b'
                    ],
                    'weight': 1.0,
                    'context_boost': ['devops', 'deployment', 'microservices'],
                    'confidence_base': 0.9
                }
            },
            
            'technical_domains': {
                'machine-learning': {
                    'patterns': [
                        r'\b(?:machine.learning|ml|artificial.intelligence|ai|deep.learning)\b',
                        r'\b(?:neural.network|cnn|rnn|lstm|transformer|bert|gpt)\b'
                    ],
                    'weight': 1.2,
                    'context_boost': ['data', 'model', 'training', 'prediction'],
                    'confidence_base': 0.9
                },
                'web-development': {
                    'patterns': [
                        r'\b(?:web|frontend|backend|fullstack|html|css|dom)\b',
                        r'\b(?:responsive|spa|pwa|ssr|ssg|jamstack)\b'
                    ],
                    'weight': 1.0,
                    'context_boost': ['browser', 'client', 'server', 'ui'],
                    'confidence_base': 0.8
                },
                'devops': {
                    'patterns': [
                        r'\b(?:devops|ci/cd|pipeline|deployment|infrastructure)\b',
                        r'\b(?:kubernetes|k8s|helm|terraform|ansible|jenkins)\b',
                        r'\b(?:monitoring|logging|observability|prometheus|grafana)\b',
                        r'\b(?:aws|azure|gcp|cloud|serverless|lambda)\b'
                    ],
                    'weight': 1.1,
                    'context_boost': ['automation', 'deployment', 'scaling'],
                    'confidence_base': 0.85
                }
            }
        }
    
    def _create_domain_taxonomy(self) -> Dict[str, Dict[str, Any]]:
        """Create hierarchical domain taxonomy for better classification."""
        return {
            'software-development': {
                'children': ['web-development', 'mobile-development', 'desktop-development'],
                'keywords': ['programming', 'coding', 'software', 'application'],
                'weight': 1.0
            },
            'web-development': {
                'parent': 'software-development',
                'children': ['frontend-development', 'backend-development', 'fullstack-development'],
                'keywords': ['web', 'html', 'css', 'javascript', 'browser'],
                'weight': 1.0
            },
            'data-science': {
                'children': ['machine-learning', 'data-analysis', 'data-visualization'],
                'keywords': ['data', 'analytics', 'statistics', 'visualization'],
                'weight': 1.1
            },
            'machine-learning': {
                'parent': 'data-science',
                'children': ['deep-learning', 'natural-language-processing', 'computer-vision'],
                'keywords': ['ml', 'ai', 'model', 'training', 'prediction'],
                'weight': 1.2
            },
            'devops': {
                'children': ['ci-cd', 'infrastructure', 'monitoring'],
                'keywords': ['deployment', 'automation', 'infrastructure', 'scaling'],
                'weight': 1.1
            }
        }
    
    def _create_semantic_clusters(self) -> Dict[str, List[str]]:
        """Create semantic clusters for related terms."""
        return {
            'web_frontend': [
                'react', 'vue', 'angular', 'svelte', 'frontend', 'ui', 'ux', 
                'html', 'css', 'javascript', 'typescript', 'spa', 'pwa'
            ],
            'web_backend': [
                'api', 'rest', 'graphql', 'server', 'backend', 'microservice',
                'database', 'orm', 'authentication', 'authorization'
            ],
            'ml_frameworks': [
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'xgboost',
                'lightgbm', 'catboost', 'huggingface', 'transformers'
            ],
            'data_tools': [
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
                'jupyter', 'notebook', 'analytics', 'visualization'
            ],
            'devops_tools': [
                'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins',
                'gitlab-ci', 'github-actions', 'aws', 'azure', 'gcp'
            ]
        }
    
    def _create_context_rules(self) -> List[Dict[str, Any]]:
        """Create context-aware rules for better classification."""
        return [
            {
                'condition': lambda text: any(term in text for term in ['react', 'vue', 'angular']),
                'action': 'boost',
                'labels': ['frontend-development', 'web-development', 'javascript'],
                'boost_factor': 1.3
            },
            {
                'condition': lambda text: any(term in text for term in ['tensorflow', 'pytorch', 'keras']),
                'action': 'boost',
                'labels': ['machine-learning', 'deep-learning', 'python'],
                'boost_factor': 1.4
            },
            {
                'condition': lambda text: any(term in text for term in ['docker', 'kubernetes', 'helm']),
                'action': 'boost',
                'labels': ['devops', 'containerization', 'orchestration'],
                'boost_factor': 1.2
            },
            {
                'condition': lambda text: 'mobile' in text and any(term in text for term in ['android', 'ios']),
                'action': 'boost',
                'labels': ['mobile-development'],
                'boost_factor': 1.5
            },
            {
                'condition': lambda text: 'game' in text and any(term in text for term in ['unity', 'unreal', '3d']),
                'action': 'boost',
                'labels': ['game-development', 'graphics'],
                'boost_factor': 1.3
            }
        ]
    
    def create_enhanced_prompt(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an enhanced prompt with structured information extraction.
        
        Args:
            repo_data: Repository data dictionary
        
        Returns:
            Structured prompt data for analysis
        """
        # Extract repository metadata
        repo_name = repo_data.get('repo', '')
        org_name = repo_data.get('org_or_user', '')
        description = repo_data.get('description', '')
        language = repo_data.get('language', '')
        topics = repo_data.get('topics', [])
        
        # Get processed text content
        processed_text = repo_data.get('processed_text', {})
        cleaned_text = processed_text.get('cleaned_text', '')
        technical_terms = processed_text.get('technical_terms', [])
        keywords = processed_text.get('keywords', [])
        
        # Create structured prompt data
        prompt_data = {
            'repository_name': repo_name,
            'organization': org_name,
            'primary_language': language,
            'description': description,
            'topics': topics,
            'technical_terms': technical_terms,
            'keywords': [kw[0] if isinstance(kw, tuple) else str(kw) for kw in keywords],
            'full_text': cleaned_text,
            'combined_text': ' '.join([
                repo_name, org_name, description, language,
                ' '.join(topics), ' '.join(technical_terms),
                ' '.join([kw[0] if isinstance(kw, tuple) else str(kw) for kw in keywords])
            ]).lower()
        }
        
        return prompt_data
    
    def _calculate_pattern_confidence(self, text: str, pattern_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a pattern match.
        
        Args:
            text: Text to analyze
            pattern_data: Pattern configuration
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = pattern_data.get('confidence_base', 0.7)
        weight = pattern_data.get('weight', 1.0)
        context_boost_terms = pattern_data.get('context_boost', [])
        
        # Count pattern matches
        match_count = 0
        for pattern in pattern_data['patterns']:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            match_count += matches
        
        # Calculate base score
        confidence = base_confidence * weight
        
        # Apply context boost
        context_matches = sum(1 for term in context_boost_terms if term in text)
        if context_matches > 0:
            context_boost = min(0.2, context_matches * 0.05)
            confidence += context_boost
        
        # Apply frequency boost
        if match_count > 1:
            frequency_boost = min(0.1, (match_count - 1) * 0.02)
            confidence += frequency_boost
        
        return min(1.0, confidence)
    
    def _extract_labels_with_confidence(self, prompt_data: Dict[str, Any]) -> List[Tuple[str, float, str]]:
        """
        Extract labels with confidence scores and sources.
        
        Args:
            prompt_data: Structured prompt data
        
        Returns:
            List of (label, confidence, source) tuples
        """
        text = prompt_data['combined_text']
        labels_with_confidence = []
        
        # Pattern-based extraction
        for category, patterns in self.enhanced_patterns.items():
            for label, pattern_data in patterns.items():
                # Check if any pattern matches
                has_match = False
                for pattern in pattern_data['patterns']:
                    if re.search(pattern, text, re.IGNORECASE):
                        has_match = True
                        break
                
                if has_match:
                    confidence = self._calculate_pattern_confidence(text, pattern_data)
                    labels_with_confidence.append((label, confidence, f'pattern_{category}'))
        
        # Semantic cluster analysis
        for cluster_name, cluster_terms in self.semantic_clusters.items():
            cluster_matches = sum(1 for term in cluster_terms if term in text)
            if cluster_matches >= 2:  # Require at least 2 matches
                cluster_confidence = min(0.8, 0.4 + (cluster_matches * 0.1))
                cluster_label = cluster_name.replace('_', '-')
                labels_with_confidence.append((cluster_label, cluster_confidence, 'semantic_cluster'))
        
        # Context rule application
        for rule in self.context_rules:
            if rule['condition'](text):
                boost_factor = rule.get('boost_factor', 1.2)
                for label in rule['labels']:
                    # Check if label already exists and boost it
                    existing_label = None
                    for i, (existing_label_name, conf, source) in enumerate(labels_with_confidence):
                        if existing_label_name == label:
                            existing_label = i
                            break
                    
                    if existing_label is not None:
                        # Boost existing label
                        old_conf = labels_with_confidence[existing_label][1]
                        new_conf = min(1.0, old_conf * boost_factor)
                        labels_with_confidence[existing_label] = (label, new_conf, 'context_boosted')
                    else:
                        # Add new label with moderate confidence
                        labels_with_confidence.append((label, 0.6 * boost_factor, 'context_rule'))
        
        # Domain taxonomy inference
        domain_scores = defaultdict(float)
        for label, confidence, source in labels_with_confidence:
            for domain, domain_data in self.domain_taxonomy.items():
                if label in domain_data.get('keywords', []) or label in domain_data.get('children', []):
                    domain_scores[domain] += confidence * domain_data.get('weight', 1.0)
        
        # Add high-confidence domain labels
        for domain, score in domain_scores.items():
            if score > 0.7:
                labels_with_confidence.append((domain, min(0.9, score), 'taxonomy_inference'))
        
        return labels_with_confidence
    
    def _post_process_labels(self, labels_with_confidence: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        """
        Post-process labels to remove duplicates, conflicts, and low-quality labels.
        
        Args:
            labels_with_confidence: List of (label, confidence, source) tuples
        
        Returns:
            Filtered and processed labels
        """
        # Group by label name
        label_groups = defaultdict(list)
        for label, confidence, source in labels_with_confidence:
            label_groups[label].append((confidence, source))
        
        # Merge duplicate labels (keep highest confidence)
        merged_labels = []
        for label, confidence_sources in label_groups.items():
            if confidence_sources:
                best_confidence, best_source = max(confidence_sources, key=lambda x: x[0])
                merged_labels.append((label, best_confidence, best_source))
        
        # Filter by minimum confidence
        filtered_labels = [
            (label, conf, source) for label, conf, source in merged_labels
            if conf >= self.config.processing.min_label_confidence
        ]
        
        # Sort by confidence and limit count
        sorted_labels = sorted(filtered_labels, key=lambda x: x[1], reverse=True)
        final_labels = sorted_labels[:self.config.processing.max_labels_per_repo]
        
        return final_labels
    
    def _generate_enhanced_labels(self, prompt_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate labels using enhanced techniques.
        
        Args:
            prompt_data: Structured prompt data
        
        Returns:
            Enhanced label results or None if failed
        """
        try:
            start_time = time.time()
            
            # Extract labels with confidence
            labels_with_confidence = self._extract_labels_with_confidence(prompt_data)
            
            # Post-process labels
            final_labels = self._post_process_labels(labels_with_confidence)
            
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            
            if final_labels:
                # Prepare result
                labels = [label for label, _, _ in final_labels]
                confidences = [conf for _, conf, _ in final_labels]
                sources = [source for _, _, source in final_labels]
                
                # Calculate overall confidence
                overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Update statistics
                self.stats['successful_requests'] += 1
                for source in sources:
                    self.stats['label_sources'][source] += 1
                
                confidence_bucket = int(overall_confidence * 10) / 10
                self.stats['confidence_distribution'][confidence_bucket] += 1
                
                return {
                    'labels': labels,
                    'confidences': confidences,
                    'sources': sources,
                    'overall_confidence': overall_confidence,
                    'processing_time': processing_time,
                    'method': 'enhanced_pattern_matching'
                }
            else:
                self.stats['failed_requests'] += 1
                return None
                
        except Exception as e:
            error_type = type(e).__name__
            self.stats['error_types'][error_type] = self.stats['error_types'].get(error_type, 0) + 1
            self.logger.debug(f"Enhanced label extraction error: {e}")
            self.stats['failed_requests'] += 1
            return None
    
    async def extract_labels_for_repository(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract labels for a single repository using enhanced techniques.
        
        Args:
            repo_data: Repository data dictionary
        
        Returns:
            Repository data with extracted labels
        """
        self.stats['total_requests'] += 1
        
        try:
            # Create enhanced prompt
            prompt_data = self.create_enhanced_prompt(repo_data)
            
            # Generate labels using thread pool to maintain async interface
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                label_results = await loop.run_in_executor(
                    executor, self._generate_enhanced_labels, prompt_data
                )
            
            # Prepare result
            result = repo_data.copy()
            
            if label_results:
                result['llm_labels'] = {
                    'labels': label_results['labels'],
                    'confidence': label_results['overall_confidence'],
                    'individual_confidences': label_results['confidences'],
                    'sources': label_results['sources'],
                    'source': 'enhanced_pattern_matching',
                    'model': 'enhanced_fallback',
                    'method': label_results['method'],
                    'processing_time': label_results['processing_time'],
                    'timestamp': time.time()
                }
                
                self.logger.debug(
                    f"Extracted {len(label_results['labels'])} labels for {repo_data.get('repo', 'unknown')} "
                    f"(confidence: {label_results['overall_confidence']:.2f})"
                )
            else:
                result['llm_labels'] = {
                    'labels': [],
                    'confidence': 0.0,
                    'individual_confidences': [],
                    'sources': [],
                    'source': 'enhanced_pattern_matching',
                    'model': 'enhanced_fallback',
                    'method': 'failed',
                    'timestamp': time.time()
                }
                
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
                'individual_confidences': [],
                'sources': [],
                'source': 'enhanced_pattern_matching',
                'model': 'enhanced_fallback',
                'error': str(e),
                'timestamp': time.time()
            }
            return result
    
    async def extract_labels_batch(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract labels for a batch of repositories using enhanced techniques.
        
        Args:
            repositories: List of repository data dictionaries
        
        Returns:
            List of repositories with extracted labels
        """
        self.logger.info(f"Starting enhanced label extraction for {len(repositories)} repositories")
        
        # Use semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.config.api.max_concurrent_requests)
        
        async def process_with_semaphore(repo_data):
            async with semaphore:
                return await self.extract_labels_for_repository(repo_data)
        
        # Process all repositories
        tasks = [process_with_semaphore(repo) for repo in repositories]
        
        try:
            from tqdm.asyncio import tqdm
            results = await tqdm.gather(*tasks, desc="Enhanced label extraction")
        except ImportError:
            # Fallback if tqdm.asyncio is not available
            results = await asyncio.gather(*tasks)
        
        # Log detailed statistics
        self.logger.info(f"Enhanced label extraction completed:")
        self.logger.info(f"  Success rate: {self.stats['successful_requests']}/{self.stats['total_requests']}")
        self.logger.info(f"  Average processing time: {self.stats['processing_time']/max(1, self.stats['total_requests']):.3f}s")
        
        # Log confidence distribution
        if self.stats['confidence_distribution']:
            self.logger.info("  Confidence distribution:")
            for conf_bucket, count in sorted(self.stats['confidence_distribution'].items()):
                self.logger.info(f"    {conf_bucket:.1f}-{conf_bucket+0.1:.1f}: {count} repositories")
        
        # Log label sources
        if self.stats['label_sources']:
            self.logger.info("  Label sources:")
            for source, count in sorted(self.stats['label_sources'].items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"    {source}: {count} labels")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed processing statistics."""
        return {
            **self.stats,
            'average_processing_time': self.stats['processing_time'] / max(1, self.stats['total_requests']),
            'success_rate': self.stats['successful_requests'] / max(1, self.stats['total_requests'])
        }

# Maintain backward compatibility
LLMLabeler = EnhancedLLMLabeler

async def main():
    """Test the enhanced LLM labeler."""
    # Test data
    test_repo = {
        'repo': 'tensorflow',
        'org_or_user': 'tensorflow',
        'platform': 'github',
        'description': 'An Open Source Machine Learning Framework for Everyone',
        'language': 'Python',
        'topics': ['machine-learning', 'deep-learning', 'neural-network', 'python', 'ai'],
        'processed_text': {
            'cleaned_text': 'An Open Source Machine Learning Framework for Everyone with TensorFlow Python APIs for deep learning and neural networks',
            'technical_terms': ['tensorflow', 'python', 'machine-learning', 'deep-learning', 'neural-network'],
            'keywords': [('machine', 3), ('learning', 3), ('tensorflow', 2), ('python', 2)]
        }
    }
    
    # Initialize enhanced labeler
    labeler = EnhancedLLMLabeler()
    
    # Extract labels
    result = await labeler.extract_labels_for_repository(test_repo)
    
    print("Enhanced LLM Labeler Test Results:")
    print(f"Repository: {result['repo']}")
    print(f"Labels: {result['llm_labels']['labels']}")
    print(f"Overall Confidence: {result['llm_labels']['confidence']:.2f}")
    print(f"Individual Confidences: {[f'{c:.2f}' for c in result['llm_labels']['individual_confidences']]}")
    print(f"Sources: {result['llm_labels']['sources']}")
    print(f"Processing Time: {result['llm_labels']['processing_time']:.3f}s")
    print(f"Stats: {labeler.get_stats()}")

if __name__ == "__main__":
    asyncio.run(main()) 