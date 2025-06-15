"""
Output formatter module for OpenDigger Repository Labeling POC.
Handles formatting and saving results according to OpenDigger specifications.
"""

import json
import logging
import time
import csv
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd

from .config import Config, get_config
from .utils import get_logger, safe_json_save

class OutputFormatter:
    """Formats and saves labeling results according to OpenDigger specifications."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the output formatter.
        
        Args:
            logger: Optional logger instance
        """
        self.config = get_config()
        self.logger = logger or get_logger(__name__)
    
    def format_repository_metadata(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format repository data according to OpenDigger meta.json specifications.
        
        Args:
            repo_data: Repository data with aggregated labels
        
        Returns:
            Formatted metadata dictionary
        """
        # Extract basic repository information
        platform = repo_data.get('platform', 'unknown')
        org_or_user = repo_data.get('org_or_user', 'unknown')
        repo_name = repo_data.get('repo', 'unknown')
        
        # Get original metadata if available
        original_metadata = repo_data.get('metadata', {})
        
        # Get aggregated labels
        aggregated_labels = repo_data.get('aggregated_labels', {})
        final_labels = aggregated_labels.get('labels', [])
        detailed_labels = aggregated_labels.get('detailed_labels', [])
        
        # Get processing information
        processed_text = repo_data.get('processed_text', {})
        llm_labels = repo_data.get('llm_labels', {})
        nlp_labels = repo_data.get('nlp_labels', {})
        
        # Create OpenDigger-compatible metadata structure
        formatted_metadata = {
            # Basic repository information
            'platform': platform,
            'org_or_user': org_or_user,
            'repo': repo_name,
            'full_name': f"{platform}/{org_or_user}/{repo_name}",
            
            # Original metadata (preserve existing fields)
            **original_metadata,
            
            # Technical domain labels (main output)
            'technical_domains': {
                'labels': final_labels,
                'detailed_labels': [
                    {
                        'label': label['label'],
                        'confidence': round(label['confidence'], 3),
                        'sources': label['sources'],
                        'source_count': label['source_count']
                    }
                    for label in detailed_labels
                ],
                'extraction_metadata': {
                    'total_labels_found': aggregated_labels.get('llm_source_count', 0) + aggregated_labels.get('nlp_source_count', 0),
                    'final_label_count': aggregated_labels.get('final_count', 0),
                    'avg_confidence': round(aggregated_labels.get('avg_confidence', 0), 3),
                    'source_overlap_count': aggregated_labels.get('overlap_count', 0),
                    'extraction_success': aggregated_labels.get('success', False)
                }
            },
            
            # Processing information
            'processing_info': {
                'text_processing': {
                    'language_detected': processed_text.get('language', 'unknown'),
                    'language_confidence': round(processed_text.get('language_confidence', 0), 3),
                    'word_count': processed_text.get('word_count', 0),
                    'is_processable': processed_text.get('is_processable', False),
                    'technical_terms_found': len(processed_text.get('technical_terms', [])),
                    'keywords_extracted': len(processed_text.get('keywords', []))
                },
                'llm_extraction': {
                    'success': llm_labels.get('success', False),
                    'source': llm_labels.get('source', 'none'),
                    'labels_extracted': len(llm_labels.get('labels', [])),
                    'raw_labels_count': len(llm_labels.get('raw_labels', []))
                },
                'nlp_extraction': {
                    'success': nlp_labels.get('success', False),
                    'labels_extracted': len(nlp_labels.get('labels', [])),
                    'entities_found': len(nlp_labels.get('entities', [])),
                    'technical_terms_found': len(nlp_labels.get('technical_terms', [])),
                    'domain_keywords_found': len(nlp_labels.get('domain_keywords', []))
                },
                'processing_timestamp': time.time(),
                'poc_version': '1.0.0'
            }
        }
        
        return formatted_metadata
    
    def save_individual_repository(self, repo_data: Dict[str, Any], output_dir: str) -> str:
        """
        Save individual repository metadata to a JSON file.
        
        Args:
            repo_data: Repository data
            output_dir: Output directory path
        
        Returns:
            Path to saved file
        """
        # Format metadata
        formatted_metadata = self.format_repository_metadata(repo_data)
        
        # Create filename following OpenDigger convention
        platform = repo_data.get('platform', 'unknown')
        org_or_user = repo_data.get('org_or_user', 'unknown')
        repo_name = repo_data.get('repo', 'unknown')
        
        # Create directory structure: output_dir/platform/org_or_user/
        repo_dir = Path(output_dir) / platform / org_or_user
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as meta.json (following OpenDigger convention)
        output_file = repo_dir / f"{repo_name}_meta.json"
        
        safe_json_save(formatted_metadata, output_file)
        
        self.logger.debug(f"Saved repository metadata to {output_file}")
        return str(output_file)
    
    def create_summary_csv(self, repositories: List[Dict[str, Any]], output_file: str) -> None:
        """
        Create a summary CSV file with all processed repositories.
        
        Args:
            repositories: List of repository data
            output_file: Output CSV file path
        """
        self.logger.info(f"Creating summary CSV with {len(repositories)} repositories")
        
        # Prepare data for CSV
        csv_data = []
        
        for repo in repositories:
            # Basic information
            platform = repo.get('platform', 'unknown')
            org_or_user = repo.get('org_or_user', 'unknown')
            repo_name = repo.get('repo', 'unknown')
            full_name = f"{platform}/{org_or_user}/{repo_name}"
            
            # Get aggregated labels
            aggregated_labels = repo.get('aggregated_labels', {})
            final_labels = aggregated_labels.get('labels', [])
            
            # Get processing info
            processed_text = repo.get('processed_text', {})
            llm_labels = repo.get('llm_labels', {})
            nlp_labels = repo.get('nlp_labels', {})
            
            # Create row
            row = {
                'platform': platform,
                'org_or_user': org_or_user,
                'repo': repo_name,
                'full_name': full_name,
                'final_labels': '; '.join(final_labels),
                'label_count': len(final_labels),
                'avg_confidence': round(aggregated_labels.get('avg_confidence', 0), 3),
                'llm_success': llm_labels.get('success', False),
                'llm_source': llm_labels.get('source', 'none'),
                'llm_labels_count': len(llm_labels.get('labels', [])),
                'nlp_success': nlp_labels.get('success', False),
                'nlp_labels_count': len(nlp_labels.get('labels', [])),
                'text_language': processed_text.get('language', 'unknown'),
                'text_word_count': processed_text.get('word_count', 0),
                'is_processable': processed_text.get('is_processable', False),
                'processing_success': aggregated_labels.get('success', False),
                'source_overlap_count': aggregated_labels.get('overlap_count', 0)
            }
            
            csv_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Summary CSV saved to {output_file}")
    
    def create_analysis_report(self, repositories: List[Dict[str, Any]], 
                             analysis_results: Dict[str, Any], 
                             output_file: str) -> None:
        """
        Create a detailed analysis report.
        
        Args:
            repositories: List of repository data
            analysis_results: Analysis results from label aggregator
            output_file: Output JSON file path
        """
        self.logger.info("Creating detailed analysis report")
        
        # Calculate additional statistics
        total_repos = len(repositories)
        successful_repos = sum(1 for repo in repositories 
                             if repo.get('aggregated_labels', {}).get('success', False))
        
        # Domain distribution analysis
        domain_distribution = {}
        for repo in repositories:
            labels = repo.get('aggregated_labels', {}).get('labels', [])
            for label in labels:
                # Categorize by domain
                categorized = False
                for domain, keywords in self.config.technical_domains.items():
                    if label.lower() in [kw.lower() for kw in keywords]:
                        domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
                        categorized = True
                        break
                
                if not categorized:
                    domain_distribution['other'] = domain_distribution.get('other', 0) + 1
        
        # Processing pipeline statistics
        pipeline_stats = {
            'text_processing': {
                'total_repositories': total_repos,
                'processable_repositories': sum(1 for repo in repositories 
                                              if repo.get('processed_text', {}).get('is_processable', False)),
                'language_distribution': {},
                'avg_word_count': 0
            },
            'llm_processing': {
                'total_attempts': sum(1 for repo in repositories 
                                    if 'llm_labels' in repo),
                'successful_extractions': sum(1 for repo in repositories 
                                            if repo.get('llm_labels', {}).get('success', False)),
                'source_distribution': {},
                'avg_labels_per_repo': 0
            },
            'nlp_processing': {
                'total_attempts': sum(1 for repo in repositories 
                                    if 'nlp_labels' in repo),
                'successful_extractions': sum(1 for repo in repositories 
                                            if repo.get('nlp_labels', {}).get('success', False)),
                'avg_entities_per_repo': 0,
                'avg_technical_terms_per_repo': 0
            }
        }
        
        # Calculate language distribution
        language_counts = {}
        word_counts = []
        for repo in repositories:
            processed_text = repo.get('processed_text', {})
            language = processed_text.get('language', 'unknown')
            language_counts[language] = language_counts.get(language, 0) + 1
            
            word_count = processed_text.get('word_count', 0)
            if word_count > 0:
                word_counts.append(word_count)
        
        pipeline_stats['text_processing']['language_distribution'] = language_counts
        pipeline_stats['text_processing']['avg_word_count'] = round(
            sum(word_counts) / len(word_counts) if word_counts else 0, 1
        )
        
        # Calculate LLM source distribution
        llm_sources = {}
        llm_label_counts = []
        for repo in repositories:
            llm_labels = repo.get('llm_labels', {})
            if llm_labels.get('success', False):
                source = llm_labels.get('source', 'unknown')
                llm_sources[source] = llm_sources.get(source, 0) + 1
                llm_label_counts.append(len(llm_labels.get('labels', [])))
        
        pipeline_stats['llm_processing']['source_distribution'] = llm_sources
        pipeline_stats['llm_processing']['avg_labels_per_repo'] = round(
            sum(llm_label_counts) / len(llm_label_counts) if llm_label_counts else 0, 1
        )
        
        # Calculate NLP statistics
        entity_counts = []
        tech_term_counts = []
        for repo in repositories:
            nlp_labels = repo.get('nlp_labels', {})
            if nlp_labels.get('success', False):
                entity_counts.append(len(nlp_labels.get('entities', [])))
                tech_term_counts.append(len(nlp_labels.get('technical_terms', [])))
        
        pipeline_stats['nlp_processing']['avg_entities_per_repo'] = round(
            sum(entity_counts) / len(entity_counts) if entity_counts else 0, 1
        )
        pipeline_stats['nlp_processing']['avg_technical_terms_per_repo'] = round(
            sum(tech_term_counts) / len(tech_term_counts) if tech_term_counts else 0, 1
        )
        
        # Create comprehensive report
        report = {
            'metadata': {
                'report_generated': time.time(),
                'poc_version': '1.0.0',
                'total_repositories_processed': total_repos,
                'successful_labelings': successful_repos,
                'overall_success_rate': round(successful_repos / total_repos * 100, 1) if total_repos > 0 else 0
            },
            'label_analysis': analysis_results,
            'domain_distribution': domain_distribution,
            'pipeline_statistics': pipeline_stats,
            'configuration_used': {
                'sample_size': self.config.processing.sample_size,
                'max_labels_per_repo': self.config.processing.max_labels_per_repo,
                'min_label_confidence': self.config.processing.min_label_confidence,
                'llm_model': self.config.model.flan_t5_model_name,
                'spacy_model': self.config.nlp.spacy_model
            }
        }
        
        safe_json_save(report, output_file)
        self.logger.info(f"Analysis report saved to {output_file}")
    
    def create_manual_review_sample(self, repositories: List[Dict[str, Any]], 
                                  sample_size: int = 10, 
                                  output_file: str = None) -> List[Dict[str, Any]]:
        """
        Create a sample of repositories for manual review.
        
        Args:
            repositories: List of repository data
            sample_size: Number of repositories to sample
            output_file: Optional output file path
        
        Returns:
            List of sampled repositories
        """
        # Filter repositories with successful labeling
        successful_repos = [
            repo for repo in repositories 
            if repo.get('aggregated_labels', {}).get('success', False)
        ]
        
        if not successful_repos:
            self.logger.warning("No successfully labeled repositories found for manual review")
            return []
        
        # Sample repositories (try to get diverse examples)
        import random
        random.seed(42)  # For reproducible sampling
        
        sample_repos = random.sample(
            successful_repos, 
            min(sample_size, len(successful_repos))
        )
        
        # Create simplified format for manual review
        review_data = []
        for repo in sample_repos:
            aggregated_labels = repo.get('aggregated_labels', {})
            processed_text = repo.get('processed_text', {})
            
            review_item = {
                'repository': {
                    'full_name': repo.get('full_name', 'unknown'),
                    'platform': repo.get('platform', 'unknown'),
                    'description': processed_text.get('cleaned_text', '')[:200] + '...'
                },
                'extracted_labels': {
                    'final_labels': aggregated_labels.get('labels', []),
                    'avg_confidence': round(aggregated_labels.get('avg_confidence', 0), 3),
                    'detailed_labels': [
                        {
                            'label': label['label'],
                            'confidence': round(label['confidence'], 3),
                            'sources': label['sources']
                        }
                        for label in aggregated_labels.get('detailed_labels', [])
                    ]
                },
                'technical_context': {
                    'technical_terms': processed_text.get('technical_terms', []),
                    'top_keywords': [kw[0] if isinstance(kw, tuple) else str(kw) 
                                   for kw in processed_text.get('keywords', [])[:5]]
                },
                'review_questions': [
                    "Are the extracted labels accurate and relevant?",
                    "Are there any important technical domains missing?",
                    "Are any labels too generic or incorrect?",
                    "How would you rate the overall labeling quality (1-5)?"
                ]
            }
            
            review_data.append(review_item)
        
        if output_file:
            safe_json_save({
                'manual_review_sample': review_data,
                'sample_metadata': {
                    'total_repositories': len(repositories),
                    'successful_repositories': len(successful_repos),
                    'sample_size': len(review_data),
                    'sampling_date': time.time()
                }
            }, output_file)
            self.logger.info(f"Manual review sample saved to {output_file}")
        
        return review_data
    
    def save_all_outputs(self, repositories: List[Dict[str, Any]], 
                        analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save all output files for the POC.
        
        Args:
            repositories: List of repository data
            analysis_results: Analysis results
        
        Returns:
            Dictionary mapping output type to file path
        """
        self.logger.info("Saving all output files")
        
        output_files = {}
        
        # Save individual repository files
        individual_count = 0
        for repo in repositories:
            try:
                file_path = self.save_individual_repository(repo, str(self.config.paths.output_data_dir))
                individual_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to save individual file for {repo.get('full_name', 'unknown')}: {e}")
        
        self.logger.info(f"Saved {individual_count} individual repository files")
        
        # Save summary CSV
        try:
            summary_csv_file = str(self.config.paths.summary_csv_file)
            self.create_summary_csv(repositories, summary_csv_file)
            output_files['summary_csv'] = summary_csv_file
        except Exception as e:
            self.logger.error(f"Failed to create summary CSV: {e}")
        
        # Save labeled repositories JSON
        try:
            labeled_repos_file = str(self.config.paths.labeled_repos_file)
            safe_json_save({
                'repositories': repositories,
                'metadata': {
                    'total_count': len(repositories),
                    'generation_time': time.time(),
                    'poc_version': '1.0.0'
                }
            }, labeled_repos_file)
            output_files['labeled_repositories'] = labeled_repos_file
        except Exception as e:
            self.logger.error(f"Failed to save labeled repositories: {e}")
        
        # Save analysis report
        try:
            analysis_file = str(self.config.paths.output_data_dir / 'analysis_report.json')
            self.create_analysis_report(repositories, analysis_results, analysis_file)
            output_files['analysis_report'] = analysis_file
        except Exception as e:
            self.logger.error(f"Failed to create analysis report: {e}")
        
        # Save manual review sample
        try:
            review_file = str(self.config.paths.output_data_dir / 'manual_review_sample.json')
            self.create_manual_review_sample(repositories, 10, review_file)
            output_files['manual_review'] = review_file
        except Exception as e:
            self.logger.error(f"Failed to create manual review sample: {e}")
        
        self.logger.info(f"Output generation completed. Files saved: {list(output_files.keys())}")
        return output_files

def main():
    """Main function for testing the output formatter."""
    logger = setup_logging('INFO', 'logs/output_formatter.log')
    formatter = OutputFormatter(logger)
    
    # Test with sample data
    sample_repos = [
        {
            'platform': 'github',
            'org_or_user': 'facebook',
            'repo': 'react',
            'full_name': 'github/facebook/react',
            'metadata': {
                'description': 'A declarative, efficient, and flexible JavaScript library for building user interfaces.',
                'language': 'JavaScript',
                'stars': 200000
            },
            'processed_text': {
                'cleaned_text': 'A declarative, efficient, and flexible JavaScript library for building user interfaces.',
                'language': 'en',
                'language_confidence': 0.95,
                'word_count': 12,
                'is_processable': True,
                'technical_terms': ['javascript', 'react', 'library'],
                'keywords': [('javascript', 3), ('library', 2), ('interfaces', 1)]
            },
            'llm_labels': {
                'labels': ['javascript', 'react', 'frontend', 'ui-library'],
                'success': True,
                'source': 'openai'
            },
            'nlp_labels': {
                'labels': ['javascript', 'library', 'react', 'web-development'],
                'success': True,
                'entities': [],
                'technical_terms': [],
                'domain_keywords': []
            },
            'aggregated_labels': {
                'labels': ['javascript', 'react', 'frontend', 'library'],
                'detailed_labels': [
                    {'label': 'javascript', 'confidence': 0.9, 'sources': ['llm', 'nlp'], 'source_count': 2},
                    {'label': 'react', 'confidence': 0.85, 'sources': ['llm', 'nlp'], 'source_count': 2},
                    {'label': 'frontend', 'confidence': 0.8, 'sources': ['llm'], 'source_count': 1},
                    {'label': 'library', 'confidence': 0.75, 'sources': ['nlp'], 'source_count': 1}
                ],
                'final_count': 4,
                'avg_confidence': 0.825,
                'success': True,
                'overlap_count': 2
            }
        }
    ]
    
    # Test individual repository formatting
    formatted = formatter.format_repository_metadata(sample_repos[0])
    logger.info("Sample formatted metadata:")
    logger.info(f"Labels: {formatted['technical_domains']['labels']}")
    logger.info(f"Confidence: {formatted['technical_domains']['extraction_metadata']['avg_confidence']}")
    
    # Test summary CSV creation
    test_csv = 'test_summary.csv'
    formatter.create_summary_csv(sample_repos, test_csv)
    logger.info(f"Test CSV created: {test_csv}")

if __name__ == "__main__":
    main() 