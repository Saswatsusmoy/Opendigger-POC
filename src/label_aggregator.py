"""
Label aggregator module for OpenDigger Repository Labeling POC.
Handles combining, filtering, and scoring labels from multiple sources (LLM + NLP).
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
import math

from .config import Config, get_config
from .utils import get_logger, calculate_confidence_score

class LabelAggregator:
    """Aggregates and processes labels from multiple sources."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the label aggregator.
        
        Args:
            logger: Optional logger instance
        """
        self.config = get_config()
        self.logger = logger or get_logger(__name__)
        
        # Statistics tracking
        self.stats = {
            'total_repositories': 0,
            'repositories_with_llm_labels': 0,
            'repositories_with_nlp_labels': 0,
            'repositories_with_both_sources': 0,
            'total_llm_labels': 0,
            'total_nlp_labels': 0,
            'total_final_labels': 0,
            'label_overlap_count': 0
        }
    
    def calculate_label_confidence(self, label: str, repo_data: Dict[str, Any], 
                                 sources: List[str]) -> float:
        """
        Calculate confidence score for a label based on multiple factors.
        
        Args:
            label: Label text
            repo_data: Repository data
            sources: List of sources that provided this label
        
        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.3
        
        # Source-based confidence
        source_boost = 0.0
        if 'llm' in sources:
            source_boost += 0.4
        if 'nlp' in sources:
            source_boost += 0.3
        if len(sources) > 1:
            source_boost += 0.2  # Bonus for multiple sources
        
        # Text-based confidence
        processed_text = repo_data.get('processed_text', {})
        text = processed_text.get('cleaned_text', '')
        
        text_confidence = calculate_confidence_score(
            label, text, self.config.technical_domains
        )
        
        # Technical term boost
        technical_terms = processed_text.get('technical_terms', [])
        tech_boost = 0.2 if label.lower() in [term.lower() for term in technical_terms] else 0.0
        
        # Keyword boost
        keywords = processed_text.get('keywords', [])
        keyword_texts = [kw[0] if isinstance(kw, tuple) else str(kw) for kw in keywords]
        keyword_boost = 0.1 if label.lower() in [kw.lower() for kw in keyword_texts] else 0.0
        
        # Domain relevance boost
        domain_boost = 0.0
        for domain, domain_keywords in self.config.technical_domains.items():
            if label.lower() in [kw.lower() for kw in domain_keywords]:
                domain_boost = 0.3
                break
        
        # Calculate final confidence
        final_confidence = min(1.0, 
            base_confidence + 
            source_boost + 
            text_confidence * 0.3 + 
            tech_boost + 
            keyword_boost + 
            domain_boost
        )
        
        return final_confidence
    
    def merge_repository_labels(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge labels from LLM and NLP sources for a single repository.
        
        Args:
            repo_data: Repository data with LLM and NLP labels
        
        Returns:
            Repository data with merged labels
        """
        # Extract labels from different sources
        llm_labels_data = repo_data.get('llm_labels', {})
        nlp_labels_data = repo_data.get('nlp_labels', {})
        
        llm_labels = llm_labels_data.get('labels', [])
        nlp_labels = nlp_labels_data.get('labels', [])
        
        # Track statistics
        if llm_labels:
            self.stats['repositories_with_llm_labels'] += 1
            self.stats['total_llm_labels'] += len(llm_labels)
        
        if nlp_labels:
            self.stats['repositories_with_nlp_labels'] += 1
            self.stats['total_nlp_labels'] += len(nlp_labels)
        
        if llm_labels and nlp_labels:
            self.stats['repositories_with_both_sources'] += 1
        
        # Create label source mapping
        label_sources = defaultdict(list)
        
        # Add LLM labels
        for label in llm_labels:
            if label and isinstance(label, str):
                label_sources[label.lower()].append('llm')
        
        # Add NLP labels
        for label in nlp_labels:
            if label and isinstance(label, str):
                label_sources[label.lower()].append('nlp')
        
        # Calculate overlap
        overlap_count = sum(1 for sources in label_sources.values() if len(sources) > 1)
        self.stats['label_overlap_count'] += overlap_count
        
        # Create final labels with confidence scores
        final_labels = []
        for label, sources in label_sources.items():
            confidence = self.calculate_label_confidence(label, repo_data, sources)
            
            final_labels.append({
                'label': label,
                'sources': sources,
                'confidence': confidence,
                'source_count': len(sources)
            })
        
        # Sort by confidence and limit
        final_labels.sort(key=lambda x: x['confidence'], reverse=True)
        final_labels = final_labels[:self.config.processing.max_labels_per_repo]
        
        # Filter by minimum confidence
        filtered_labels = [
            label for label in final_labels 
            if label['confidence'] >= self.config.processing.min_label_confidence
        ]
        
        # If no labels meet the threshold, keep the top ones with lower threshold
        if not filtered_labels and final_labels:
            lower_threshold = self.config.processing.min_label_confidence * 0.7
            filtered_labels = [
                label for label in final_labels 
                if label['confidence'] >= lower_threshold
            ][:3]  # Keep at least top 3
        
        self.stats['total_final_labels'] += len(filtered_labels)
        
        # Create aggregated labels structure
        aggregated_labels = {
            'labels': [label['label'] for label in filtered_labels],
            'detailed_labels': filtered_labels,
            'llm_source_count': len(llm_labels),
            'nlp_source_count': len(nlp_labels),
            'overlap_count': overlap_count,
            'final_count': len(filtered_labels),
            'avg_confidence': sum(label['confidence'] for label in filtered_labels) / len(filtered_labels) if filtered_labels else 0.0,
            'success': len(filtered_labels) > 0
        }
        
        return {
            **repo_data,
            'aggregated_labels': aggregated_labels
        }
    
    def aggregate_batch(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate labels for multiple repositories.
        
        Args:
            repositories: List of repository data dictionaries
        
        Returns:
            List of repositories with aggregated labels
        """
        self.logger.info(f"Aggregating labels for {len(repositories)} repositories")
        self.stats['total_repositories'] = len(repositories)
        
        aggregated_repos = []
        for repo in repositories:
            try:
                aggregated_repo = self.merge_repository_labels(repo)
                aggregated_repos.append(aggregated_repo)
            except Exception as e:
                self.logger.warning(f"Failed to aggregate labels for {repo.get('full_name', 'unknown')}: {e}")
                # Add repository with empty aggregated labels
                aggregated_repos.append({
                    **repo,
                    'aggregated_labels': {
                        'labels': [],
                        'detailed_labels': [],
                        'llm_source_count': 0,
                        'nlp_source_count': 0,
                        'overlap_count': 0,
                        'final_count': 0,
                        'avg_confidence': 0.0,
                        'success': False
                    }
                })
        
        # Log statistics
        self.logger.info(f"Label aggregation completed:")
        self.logger.info(f"  Total repositories: {self.stats['total_repositories']}")
        self.logger.info(f"  With LLM labels: {self.stats['repositories_with_llm_labels']}")
        self.logger.info(f"  With NLP labels: {self.stats['repositories_with_nlp_labels']}")
        self.logger.info(f"  With both sources: {self.stats['repositories_with_both_sources']}")
        self.logger.info(f"  Total LLM labels: {self.stats['total_llm_labels']}")
        self.logger.info(f"  Total NLP labels: {self.stats['total_nlp_labels']}")
        self.logger.info(f"  Total final labels: {self.stats['total_final_labels']}")
        self.logger.info(f"  Label overlaps: {self.stats['label_overlap_count']}")
        
        return aggregated_repos
    
    def analyze_label_quality(self, repositories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality and distribution of aggregated labels.
        
        Args:
            repositories: List of repositories with aggregated labels
        
        Returns:
            Quality analysis results
        """
        all_labels = []
        confidence_scores = []
        source_combinations = Counter()
        domain_distribution = defaultdict(int)
        
        successful_repos = 0
        
        for repo in repositories:
            aggregated = repo.get('aggregated_labels', {})
            if aggregated.get('success', False):
                successful_repos += 1
                
                detailed_labels = aggregated.get('detailed_labels', [])
                for label_info in detailed_labels:
                    label = label_info['label']
                    confidence = label_info['confidence']
                    sources = label_info['sources']
                    
                    all_labels.append(label)
                    confidence_scores.append(confidence)
                    
                    # Track source combinations
                    source_key = '+'.join(sorted(set(sources)))
                    source_combinations[source_key] += 1
                    
                    # Categorize by domain
                    categorized = False
                    for domain, keywords in self.config.technical_domains.items():
                        if label.lower() in [kw.lower() for kw in keywords]:
                            domain_distribution[domain] += 1
                            categorized = True
                            break
                    
                    if not categorized:
                        domain_distribution['other'] += 1
        
        # Calculate quality metrics
        label_freq = Counter(all_labels)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Confidence distribution
        confidence_ranges = {
            'high (0.8-1.0)': sum(1 for c in confidence_scores if c >= 0.8),
            'medium (0.5-0.8)': sum(1 for c in confidence_scores if 0.5 <= c < 0.8),
            'low (0.0-0.5)': sum(1 for c in confidence_scores if c < 0.5)
        }
        
        analysis = {
            'total_repositories': len(repositories),
            'successful_aggregations': successful_repos,
            'success_rate': successful_repos / len(repositories) if repositories else 0,
            'total_labels': len(all_labels),
            'unique_labels': len(set(all_labels)),
            'avg_labels_per_repo': len(all_labels) / successful_repos if successful_repos else 0,
            'avg_confidence': avg_confidence,
            'confidence_distribution': confidence_ranges,
            'most_common_labels': label_freq.most_common(20),
            'source_combinations': dict(source_combinations),
            'domain_distribution': dict(domain_distribution),
            'quality_score': self._calculate_quality_score(
                successful_repos / len(repositories) if repositories else 0,
                avg_confidence,
                len(set(all_labels)) / len(all_labels) if all_labels else 0
            )
        }
        
        return analysis
    
    def _calculate_quality_score(self, success_rate: float, avg_confidence: float, 
                               uniqueness_ratio: float) -> float:
        """
        Calculate an overall quality score for the labeling process.
        
        Args:
            success_rate: Proportion of successfully labeled repositories
            avg_confidence: Average confidence score
            uniqueness_ratio: Ratio of unique labels to total labels
        
        Returns:
            Quality score between 0 and 1
        """
        # Weighted combination of metrics
        quality_score = (
            success_rate * 0.4 +           # 40% weight on success rate
            avg_confidence * 0.4 +         # 40% weight on confidence
            uniqueness_ratio * 0.2         # 20% weight on label diversity
        )
        
        return min(1.0, quality_score)
    
    def create_label_recommendations(self, repositories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create recommendations for improving label quality.
        
        Args:
            repositories: List of repositories with aggregated labels
        
        Returns:
            Recommendations dictionary
        """
        analysis = self.analyze_label_quality(repositories)
        recommendations = []
        
        # Success rate recommendations
        if analysis['success_rate'] < 0.7:
            recommendations.append({
                'type': 'success_rate',
                'message': f"Success rate is {analysis['success_rate']:.1%}. Consider improving text preprocessing or adding more LLM/NLP sources.",
                'priority': 'high'
            })
        
        # Confidence recommendations
        if analysis['avg_confidence'] < 0.6:
            recommendations.append({
                'type': 'confidence',
                'message': f"Average confidence is {analysis['avg_confidence']:.2f}. Consider refining confidence calculation or filtering thresholds.",
                'priority': 'medium'
            })
        
        # Label diversity recommendations
        uniqueness_ratio = len(analysis['most_common_labels']) / analysis['total_labels'] if analysis['total_labels'] > 0 else 0
        if uniqueness_ratio < 0.3:
            recommendations.append({
                'type': 'diversity',
                'message': f"Label diversity is low ({uniqueness_ratio:.1%}). Consider expanding technical domain categories or improving NLP extraction.",
                'priority': 'medium'
            })
        
        # Source combination recommendations
        source_combos = analysis['source_combinations']
        if source_combos.get('llm+nlp', 0) < source_combos.get('llm', 0) * 0.3:
            recommendations.append({
                'type': 'source_overlap',
                'message': "Low overlap between LLM and NLP sources. Consider improving NLP patterns or LLM prompts for better agreement.",
                'priority': 'low'
            })
        
        return {
            'recommendations': recommendations,
            'quality_score': analysis['quality_score'],
            'priority_actions': [r for r in recommendations if r['priority'] == 'high']
        }

def main():
    """Main function for testing the label aggregator."""
    logger = setup_logging('INFO', 'logs/label_aggregator.log')
    aggregator = LabelAggregator(logger)
    
    # Test with sample data
    sample_repos = [
        {
            'platform': 'github',
            'org_or_user': 'facebook',
            'repo': 'react',
            'full_name': 'github/facebook/react',
            'processed_text': {
                'cleaned_text': 'A declarative, efficient, and flexible JavaScript library for building user interfaces.',
                'technical_terms': ['javascript', 'react', 'library'],
                'keywords': [('javascript', 3), ('library', 2), ('interfaces', 1)],
                'is_processable': True
            },
            'llm_labels': {
                'labels': ['javascript', 'react', 'frontend', 'ui-library'],
                'success': True
            },
            'nlp_labels': {
                'labels': ['javascript', 'library', 'react', 'web-development'],
                'success': True
            }
        },
        {
            'platform': 'github',
            'org_or_user': 'tensorflow',
            'repo': 'tensorflow',
            'full_name': 'github/tensorflow/tensorflow',
            'processed_text': {
                'cleaned_text': 'An open source machine learning framework for everyone.',
                'technical_terms': ['tensorflow', 'machine-learning', 'python'],
                'keywords': [('machine', 2), ('learning', 2), ('framework', 1)],
                'is_processable': True
            },
            'llm_labels': {
                'labels': ['machine-learning', 'tensorflow', 'deep-learning', 'python'],
                'success': True
            },
            'nlp_labels': {
                'labels': ['machine-learning', 'framework', 'tensorflow', 'ai'],
                'success': True
            }
        }
    ]
    
    # Aggregate labels
    aggregated_repos = aggregator.aggregate_batch(sample_repos)
    
    # Analyze quality
    analysis = aggregator.analyze_label_quality(aggregated_repos)
    recommendations = aggregator.create_label_recommendations(aggregated_repos)
    
    logger.info("Label aggregation completed")
    logger.info(f"Success rate: {analysis['success_rate']:.2%}")
    logger.info(f"Average confidence: {analysis['avg_confidence']:.2f}")
    logger.info(f"Quality score: {analysis['quality_score']:.2f}")
    
    # Show sample results
    for repo in aggregated_repos:
        labels = repo.get('aggregated_labels', {}).get('labels', [])
        confidence = repo.get('aggregated_labels', {}).get('avg_confidence', 0)
        logger.info(f"{repo['full_name']}: {labels} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main() 