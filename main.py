#!/usr/bin/env python3
"""Main entry point for OpenDigger Repository Labeling POC.

This script orchestrates the complete repository labeling pipeline, including
data fetching, text processing, label extraction, NLP enhancement, aggregation,
and output formatting.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_config
from src.utils import get_logger, setup_logging, ProgressTracker
from src.mock_data_fetcher import MockDataFetcher
from src.text_processor import TextProcessor
from src.llm_labeler_enhanced import EnhancedLLMLabeler
from src.nlp_enhancer import NLPEnhancer
from src.label_aggregator import LabelAggregator
from src.output_formatter import OutputFormatter


class OpenDiggerPipeline:
    """Main pipeline orchestrator for repository labeling."""
    
    def __init__(self):
        """Initialize the pipeline with all components."""
        self.config = get_config()
        self.logger = setup_logging(
            name="opendigger_pipeline",
            level=self.config.logging.level,
            log_file=self.config.paths.logs_dir / "pipeline.log"
        )
        
        # Initialize components
        self.data_fetcher = MockDataFetcher(logger=self.logger)
        self.text_processor = TextProcessor(logger=self.logger)
        self.llm_labeler = EnhancedLLMLabeler(logger=self.logger)
        self.nlp_enhancer = NLPEnhancer(logger=self.logger)
        self.label_aggregator = LabelAggregator(logger=self.logger)
        self.output_formatter = OutputFormatter(logger=self.logger)
        
        # Pipeline statistics
        self.stats = {
            'total_repositories': 0,
            'successfully_processed': 0,
            'failed_processing': 0,
            'total_labels_generated': 0,
            'processing_start_time': 0,
            'processing_end_time': 0,
            'component_stats': {}
        }
        
        self.logger.info("OpenDigger Pipeline initialized successfully")
    
    async def run_pipeline(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete labeling pipeline.
        
        Args:
            sample_size: Number of repositories to process. If None, uses config default.
            
        Returns:
            Pipeline execution results and statistics
        """
        self.logger.info("Starting OpenDigger Repository Labeling Pipeline")
        self.stats['processing_start_time'] = time.time()
        
        try:
            # Step 1: Fetch repository data
            repositories = await self._fetch_repositories(sample_size)
            if not repositories:
                raise ValueError("No repositories fetched")
            
            self.stats['total_repositories'] = len(repositories)
            
            # Step 2: Process repositories through the pipeline
            processed_results = await self._process_repositories(repositories)
            
            # Step 3: Generate output
            output_results = await self._generate_output(processed_results)
            
            # Step 4: Calculate final statistics
            self.stats['processing_end_time'] = time.time()
            final_stats = self._calculate_final_statistics()
            
            self.logger.info("Pipeline completed successfully")
            return {
                'success': True,
                'results': output_results,
                'statistics': final_stats
            }
        
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.stats['processing_end_time'] = time.time()
            return {
                'success': False,
                'error': str(e),
                'statistics': self._calculate_final_statistics()
            }
    
    async def _fetch_repositories(self, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch repository data from the data source.
        
        Args:
            sample_size: Number of repositories to fetch
            
        Returns:
            List of repository data dictionaries
        """
        self.logger.info("Step 1: Fetching repository data")
        
        sample_size = sample_size or self.config.processing.sample_size
        repositories = []
        
        async with self.data_fetcher:
            # Get repository list
            repo_list = await self.data_fetcher.fetch_repository_list()
            self.logger.info(f"Found {len(repo_list)} repositories in data source")
            
            # Limit to sample size
            repo_list = repo_list[:sample_size]
            self.logger.info(f"Processing {len(repo_list)} repositories")
            
            # Fetch metadata for each repository
            progress = ProgressTracker(len(repo_list), "Fetching repository metadata")
            
            async for repo_data in self.data_fetcher.fetch_repositories_batch(repo_list):
                if repo_data:
                    repositories.append(repo_data)
                progress.update()
            
            progress.finish()
        
        self.logger.info(f"Successfully fetched {len(repositories)} repositories")
        return repositories
    
    async def _process_repositories(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process repositories through the labeling pipeline.
        
        Args:
            repositories: List of repository data
            
        Returns:
            List of processed repositories with labels
        """
        self.logger.info("Step 2: Processing repositories through labeling pipeline")
        
        processed_results = []
        progress = ProgressTracker(len(repositories), "Processing repositories")
        
        for repo_data in repositories:
            try:
                # Step 2a: Text processing
                processed_repo = await self.text_processor.process_repository_text(repo_data)
                
                # Step 2b: LLM label extraction
                llm_result = await self.llm_labeler.extract_labels_for_repository(processed_repo)
                
                # Step 2c: NLP enhancement
                nlp_result = await self.nlp_enhancer.enhance_labels(llm_result)
                
                # Step 2d: Label aggregation
                final_result = self.label_aggregator.aggregate_labels([llm_result, nlp_result])
                
                processed_results.append(final_result)
                self.stats['successfully_processed'] += 1
                
                # Count labels
                if final_result.get('aggregated_labels', {}).get('labels'):
                    self.stats['total_labels_generated'] += len(final_result['aggregated_labels']['labels'])
            
            except Exception as e:
                self.logger.error(f"Error processing repository {repo_data.get('name', 'unknown')}: {e}")
                self.stats['failed_processing'] += 1
                
                # Add failed result
                processed_results.append({
                    **repo_data,
                    'processing_error': str(e),
                    'aggregated_labels': {'labels': [], 'confidence': 0.0}
                })
            
            progress.update()
        
        progress.finish()
        
        self.logger.info(
            f"Processing completed: {self.stats['successfully_processed']} successful, "
            f"{self.stats['failed_processing']} failed"
        )
        
        return processed_results
    
    async def _generate_output(self, processed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate formatted output from processed results.
        
        Args:
            processed_results: List of processed repositories
            
        Returns:
            Formatted output results
        """
        self.logger.info("Step 3: Generating formatted output")
        
        try:
            # Create analysis results (placeholder for now)
            analysis_results = {
                'total_repositories': len(processed_results),
                'successful_processing': len([r for r in processed_results if r.get('aggregated_labels', {}).get('success', False)]),
                'label_statistics': {}
            }
            
            # Save all outputs using the correct method
            output_files = self.output_formatter.save_all_outputs(processed_results, analysis_results)
            
            self.logger.info(f"Output generated successfully: {list(output_files.keys())}")
            return {
                'formatted_results': processed_results,
                'output_files': output_files
            }
        
        except Exception as e:
            self.logger.error(f"Error generating output: {e}")
            return {
                'formatted_results': None,
                'output_files': {},
                'error': str(e)
            }
    
    def _calculate_final_statistics(self) -> Dict[str, Any]:
        """Calculate final pipeline statistics.
        
        Returns:
            Dictionary containing comprehensive statistics
        """
        processing_time = self.stats['processing_end_time'] - self.stats['processing_start_time']
        
        # Collect component statistics
        component_stats = {
            'text_processor': getattr(self.text_processor, 'get_stats', lambda: {})(),
            'llm_labeler': self.llm_labeler.get_stats(),
            'nlp_enhancer': getattr(self.nlp_enhancer, 'get_stats', lambda: {})(),
            'label_aggregator': getattr(self.label_aggregator, 'get_stats', lambda: {})()
        }
        
        return {
            'pipeline_summary': {
                'total_repositories': self.stats['total_repositories'],
                'successfully_processed': self.stats['successfully_processed'],
                'failed_processing': self.stats['failed_processing'],
                'success_rate': (
                    self.stats['successfully_processed'] / max(1, self.stats['total_repositories']) * 100
                ),
                'total_processing_time': processing_time,
                'average_time_per_repo': (
                    processing_time / max(1, self.stats['total_repositories'])
                ),
                'total_labels_generated': self.stats['total_labels_generated'],
                'average_labels_per_repo': (
                    self.stats['total_labels_generated'] / max(1, self.stats['successfully_processed'])
                )
            },
            'component_statistics': component_stats,
            'configuration': self.config.to_dict()
        }
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of pipeline results.
        
        Args:
            results: Pipeline execution results
        """
        print("\n" + "="*60)
        print("OpenDigger Repository Labeling Pipeline Summary")
        print("="*60)
        
        if results['success']:
            stats = results['statistics']['pipeline_summary']
            
            print(f"✅ Pipeline Status: SUCCESS")
            print(f"📊 Repositories Processed: {stats['total_repositories']}")
            print(f"✅ Successfully Processed: {stats['successfully_processed']}")
            print(f"❌ Failed: {stats['failed_processing']}")
            print(f"📈 Success Rate: {stats['success_rate']:.1f}%")
            print(f"⏱️  Total Processing Time: {stats['total_processing_time']:.2f} seconds")
            print(f"⚡ Average Time per Repository: {stats['average_time_per_repo']:.2f} seconds")
            print(f"🏷️  Total Labels Generated: {stats['total_labels_generated']}")
            print(f"📊 Average Labels per Repository: {stats['average_labels_per_repo']:.1f}")
            
            # Component statistics
            print(f"\n📈 Component Performance:")
            comp_stats = results['statistics']['component_statistics']
            
            if 'llm_labeler' in comp_stats:
                llm_stats = comp_stats['llm_labeler']
                print(f"   LLM Labeler: {llm_stats.get('success_rate', 0):.1f}% success rate")
            
            # Output files
            if 'output_files' in results.get('results', {}):
                output_files = results['results']['output_files']
                print(f"\n📁 Output Files Generated:")
                for file_type, file_path in output_files.items():
                    print(f"   {file_type}: {file_path}")
        
        else:
            print(f"❌ Pipeline Status: FAILED")
            print(f"💥 Error: {results.get('error', 'Unknown error')}")
            
            if 'statistics' in results:
                stats = results['statistics']['pipeline_summary']
                print(f"📊 Repositories Attempted: {stats['total_repositories']}")
                print(f"⏱️  Processing Time: {stats['total_processing_time']:.2f} seconds")
        
        print("="*60)


async def main():
    """Main entry point for the application."""
    print("🚀 OpenDigger Repository Labeling POC")
    print("Automated Technical Domain Labeling System")
    print("-" * 50)
    
    try:
        # Initialize and run pipeline
        pipeline = OpenDiggerPipeline()
        results = await pipeline.run_pipeline()
        
        # Print summary
        pipeline.print_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if results['success'] else 1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the main pipeline
    asyncio.run(main()) 