"""
NLP enhancer module for OpenDigger Repository Labeling POC.
Handles traditional NLP-based label extraction using spaCy and NLTK.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import Counter
import re

try:
    import spacy
    from spacy.matcher import Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

from .config import Config
from .utils import get_logger

class NLPEnhancer:
    """Enhances label extraction using traditional NLP techniques."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the NLP enhancer.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.nlp = None
        self.matcher = None
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            self._initialize_spacy()
        else:
            self.logger.warning("spaCy not available, using NLTK-only approach")
        
        # Initialize NLTK components
        self._initialize_nltk()
        
        # Technical patterns for matching
        self.tech_patterns = self._create_technical_patterns()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'spacy_extractions': 0,
            'nltk_extractions': 0,
            'pattern_matches': 0,
            'entity_extractions': 0
        }
    
    def _initialize_spacy(self):
        """Initialize spaCy model and matcher."""
        try:
            self.nlp = spacy.load(Config.SPACY_MODEL)
            self.matcher = Matcher(self.nlp.vocab)
            self._add_technical_patterns()
            self.logger.info(f"spaCy model '{Config.SPACY_MODEL}' loaded successfully")
        except OSError:
            self.logger.warning(f"spaCy model '{Config.SPACY_MODEL}' not found. Install with: python -m spacy download {Config.SPACY_MODEL}")
            SPACY_AVAILABLE = False
    
    def _initialize_nltk(self):
        """Initialize NLTK components."""
        try:
            # Download required NLTK data
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            self.logger.info("Downloading NLTK POS tagger")
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            self.logger.info("Downloading NLTK named entity chunker")
            nltk.download('maxent_ne_chunker', quiet=True)
        
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            self.logger.info("Downloading NLTK words corpus")
            nltk.download('words', quiet=True)
    
    def _create_technical_patterns(self) -> Dict[str, List[str]]:
        """Create regex patterns for technical term extraction."""
        return {
            'programming_languages': [
                r'\b(python|java|javascript|typescript|c\+\+|c#|php|ruby|go|rust|swift|kotlin|scala|r|matlab|perl|lua|dart|elixir|haskell|clojure|erlang|objective-c)\b',
                r'\b(js|ts|cpp|py|rb|go|rs)\b'
            ],
            'web_frameworks': [
                r'\b(react|angular|vue|svelte|ember|backbone|jquery|bootstrap|tailwind|bulma)\b',
                r'\b(express|koa|fastify|hapi|django|flask|rails|laravel|spring|struts|play)\b',
                r'\b(nextjs|nuxtjs|gatsby|gridsome|sapper)\b'
            ],
            'databases': [
                r'\b(mysql|postgresql|sqlite|mongodb|redis|elasticsearch|cassandra|dynamodb|couchdb|neo4j|influxdb|clickhouse)\b',
                r'\b(sql|nosql|rdbms|orm|odm)\b'
            ],
            'cloud_devops': [
                r'\b(aws|azure|gcp|google-cloud|digitalocean|heroku|vercel|netlify)\b',
                r'\b(docker|kubernetes|k8s|helm|terraform|ansible|vagrant|jenkins|gitlab-ci|github-actions|circleci|travis-ci)\b',
                r'\b(nginx|apache|haproxy|cloudflare|cdn)\b'
            ],
            'mobile_development': [
                r'\b(android|ios|react-native|flutter|xamarin|cordova|ionic|phonegap|nativescript)\b',
                r'\b(swift|kotlin|objective-c|java-android)\b'
            ],
            'data_science_ml': [
                r'\b(tensorflow|pytorch|scikit-learn|keras|pandas|numpy|matplotlib|seaborn|plotly|jupyter|anaconda)\b',
                r'\b(machine-learning|deep-learning|neural-network|ai|ml|data-science|analytics|statistics)\b',
                r'\b(spark|hadoop|kafka|airflow|dask|ray)\b'
            ],
            'game_development': [
                r'\b(unity|unreal|godot|pygame|phaser|three\.js|babylon\.js|webgl|opengl|directx|vulkan)\b',
                r'\b(game-engine|2d|3d|graphics|rendering|shader)\b'
            ],
            'blockchain_crypto': [
                r'\b(blockchain|bitcoin|ethereum|solidity|web3|defi|nft|smart-contract|cryptocurrency|crypto)\b',
                r'\b(metamask|truffle|hardhat|ganache|ipfs)\b'
            ],
            'testing_tools': [
                r'\b(jest|mocha|chai|jasmine|karma|cypress|selenium|puppeteer|playwright|pytest|unittest|rspec)\b',
                r'\b(testing|test|unit-test|integration-test|e2e|tdd|bdd)\b'
            ],
            'build_tools': [
                r'\b(webpack|rollup|parcel|vite|gulp|grunt|babel|typescript|eslint|prettier|sass|less)\b',
                r'\b(npm|yarn|pip|composer|maven|gradle|cargo|bundler)\b'
            ]
        }
    
    def _add_technical_patterns(self):
        """Add technical patterns to spaCy matcher."""
        if not self.matcher:
            return
        
        # Add patterns for each technical category
        for category, patterns in self.tech_patterns.items():
            for i, pattern in enumerate(patterns):
                # Convert regex to spaCy pattern (simplified)
                # This is a basic conversion - more sophisticated pattern matching could be implemented
                pattern_name = f"{category}_{i}"
                try:
                    # Extract words from regex pattern
                    words = re.findall(r'\b([a-zA-Z][a-zA-Z0-9\-]*)\b', pattern)
                    if words:
                        spacy_pattern = [{"LOWER": word.lower()} for word in words[:3]]  # Limit to first 3 words
                        self.matcher.add(pattern_name, [spacy_pattern])
                except Exception as e:
                    self.logger.debug(f"Failed to add pattern {pattern_name}: {e}")
    
    def extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities using spaCy.
        
        Args:
            text: Input text
        
        Returns:
            List of entity dictionaries
        """
        if not self.nlp or not text:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            # Extract named entities
            for ent in doc.ents:
                # Focus on relevant entity types for technical domains
                if ent.label_ in ['ORG', 'PRODUCT', 'LANGUAGE', 'TECH', 'GPE', 'PERSON']:
                    entities.append({
                        'text': ent.text.lower(),
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.8  # spaCy doesn't provide confidence scores by default
                    })
            
            # Extract pattern matches
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                pattern_name = self.nlp.vocab.strings[match_id]
                entities.append({
                    'text': span.text.lower(),
                    'label': 'TECH_PATTERN',
                    'pattern': pattern_name,
                    'start': span.start_char,
                    'end': span.end_char,
                    'confidence': 0.7
                })
            
            self.stats['spacy_extractions'] += len(entities)
            return entities
            
        except Exception as e:
            self.logger.debug(f"spaCy entity extraction failed: {e}")
            return []
    
    def extract_entities_nltk(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities using NLTK.
        
        Args:
            text: Input text
        
        Returns:
            List of entity dictionaries
        """
        if not text:
            return []
        
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity recognition
            tree = ne_chunk(pos_tags)
            
            entities = []
            for subtree in tree:
                if isinstance(subtree, Tree):
                    entity_name = ' '.join([token for token, pos in subtree.leaves()])
                    entity_label = subtree.label()
                    
                    # Focus on relevant entity types
                    if entity_label in ['ORGANIZATION', 'PERSON', 'GPE']:
                        entities.append({
                            'text': entity_name.lower(),
                            'label': entity_label,
                            'confidence': 0.6
                        })
            
            self.stats['nltk_extractions'] += len(entities)
            return entities
            
        except Exception as e:
            self.logger.debug(f"NLTK entity extraction failed: {e}")
            return []
    
    def extract_technical_terms_regex(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract technical terms using regex patterns.
        
        Args:
            text: Input text
        
        Returns:
            List of technical term dictionaries
        """
        if not text:
            return []
        
        technical_terms = []
        text_lower = text.lower()
        
        for category, patterns in self.tech_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                    for match in matches:
                        term = match.group().strip()
                        if len(term) > 1:  # Skip single characters
                            technical_terms.append({
                                'text': term,
                                'category': category,
                                'label': 'TECH_TERM',
                                'start': match.start(),
                                'end': match.end(),
                                'confidence': 0.9
                            })
                except Exception as e:
                    self.logger.debug(f"Regex pattern matching failed for {pattern}: {e}")
        
        self.stats['pattern_matches'] += len(technical_terms)
        return technical_terms
    
    def extract_domain_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract domain-specific keywords using frequency analysis and domain matching.
        
        Args:
            text: Input text
        
        Returns:
            List of keyword dictionaries
        """
        if not text:
            return []
        
        # Tokenize and clean
        try:
            tokens = word_tokenize(text.lower())
        except:
            tokens = text.lower().split()
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            if (token.isalpha() and 
                len(token) >= 3 and 
                token not in stopwords.words('english')):
                filtered_tokens.append(token)
        
        # Count frequencies
        token_freq = Counter(filtered_tokens)
        
        # Extract keywords that appear in technical domains
        domain_keywords = []
        all_domain_terms = set()
        
        for domain, terms in Config.TECHNICAL_DOMAINS.items():
            all_domain_terms.update([term.lower() for term in terms])
        
        for token, freq in token_freq.items():
            if freq >= Config.MIN_KEYWORD_FREQUENCY:
                # Check if token matches any domain terms
                confidence = 0.3
                category = 'general'
                
                if token in all_domain_terms:
                    confidence = 0.8
                    # Find which domain it belongs to
                    for domain, terms in Config.TECHNICAL_DOMAINS.items():
                        if token in [term.lower() for term in terms]:
                            category = domain
                            break
                
                domain_keywords.append({
                    'text': token,
                    'frequency': freq,
                    'category': category,
                    'label': 'DOMAIN_KEYWORD',
                    'confidence': confidence
                })
        
        # Sort by frequency and confidence
        domain_keywords.sort(key=lambda x: (x['confidence'], x['frequency']), reverse=True)
        
        return domain_keywords[:Config.MAX_KEYWORDS_EXTRACTED]
    
    def enhance_repository_labels(self, repo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance repository with NLP-extracted labels.
        
        Args:
            repo_data: Repository data dictionary
        
        Returns:
            Repository data enhanced with NLP labels
        """
        self.stats['total_processed'] += 1
        
        # Get processed text
        processed_text = repo_data.get('processed_text', {})
        if not processed_text.get('is_processable', False):
            self.logger.debug(f"Skipping {repo_data.get('full_name', 'unknown')}: not processable")
            return {
                **repo_data,
                'nlp_labels': {
                    'labels': [],
                    'entities': [],
                    'technical_terms': [],
                    'domain_keywords': [],
                    'success': False,
                    'reason': 'not_processable'
                }
            }
        
        text = processed_text.get('cleaned_text', '')
        if not text:
            return {
                **repo_data,
                'nlp_labels': {
                    'labels': [],
                    'entities': [],
                    'technical_terms': [],
                    'domain_keywords': [],
                    'success': False,
                    'reason': 'no_text'
                }
            }
        
        # Extract different types of information
        spacy_entities = self.extract_entities_spacy(text) if SPACY_AVAILABLE else []
        nltk_entities = self.extract_entities_nltk(text)
        technical_terms = self.extract_technical_terms_regex(text)
        domain_keywords = self.extract_domain_keywords(text)
        
        # Combine all extractions into labels
        all_labels = []
        
        # Add entities as labels
        for entity in spacy_entities + nltk_entities:
            if entity['text'] and len(entity['text']) > 2:
                all_labels.append(entity['text'])
        
        # Add technical terms as labels
        for term in technical_terms:
            if term['text'] and len(term['text']) > 2:
                all_labels.append(term['text'])
        
        # Add high-confidence domain keywords as labels
        for keyword in domain_keywords:
            if keyword['confidence'] > 0.5 and keyword['text']:
                all_labels.append(keyword['text'])
        
        # Filter and clean labels
        filtered_labels = filter_labels(
            all_labels,
            Config.GENERIC_LABELS_TO_FILTER,
            Config.MAX_LABELS_PER_REPO
        )
        
        return {
            **repo_data,
            'nlp_labels': {
                'labels': filtered_labels,
                'entities': spacy_entities + nltk_entities,
                'technical_terms': technical_terms,
                'domain_keywords': domain_keywords,
                'success': True,
                'label_count': len(filtered_labels)
            }
        }
    
    def enhance_batch(self, repositories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance multiple repositories with NLP labels.
        
        Args:
            repositories: List of repository data dictionaries
        
        Returns:
            List of repositories with NLP labels
        """
        self.logger.info(f"Enhancing {len(repositories)} repositories with NLP labels")
        
        enhanced_repos = []
        for repo in repositories:
            try:
                enhanced_repo = self.enhance_repository_labels(repo)
                enhanced_repos.append(enhanced_repo)
            except Exception as e:
                self.logger.warning(f"Failed to enhance {repo.get('full_name', 'unknown')}: {e}")
                # Add repository with empty NLP labels
                enhanced_repos.append({
                    **repo,
                    'nlp_labels': {
                        'labels': [],
                        'entities': [],
                        'technical_terms': [],
                        'domain_keywords': [],
                        'success': False,
                        'reason': 'processing_error'
                    }
                })
        
        # Log statistics
        self.logger.info(f"NLP enhancement completed:")
        self.logger.info(f"  Total processed: {self.stats['total_processed']}")
        self.logger.info(f"  spaCy extractions: {self.stats['spacy_extractions']}")
        self.logger.info(f"  NLTK extractions: {self.stats['nltk_extractions']}")
        self.logger.info(f"  Pattern matches: {self.stats['pattern_matches']}")
        
        return enhanced_repos
    
    def analyze_nlp_results(self, repositories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze NLP enhancement results.
        
        Args:
            repositories: List of repositories with NLP labels
        
        Returns:
            Analysis results
        """
        all_labels = []
        all_entities = []
        all_technical_terms = []
        successful_enhancements = 0
        
        for repo in repositories:
            nlp_labels = repo.get('nlp_labels', {})
            if nlp_labels.get('success', False):
                successful_enhancements += 1
                all_labels.extend(nlp_labels.get('labels', []))
                all_entities.extend(nlp_labels.get('entities', []))
                all_technical_terms.extend(nlp_labels.get('technical_terms', []))
        
        # Analyze label frequencies
        label_freq = Counter(all_labels)
        
        # Analyze entity types
        entity_types = Counter([entity.get('label', 'unknown') for entity in all_entities])
        
        # Analyze technical term categories
        tech_categories = Counter([term.get('category', 'unknown') for term in all_technical_terms])
        
        analysis = {
            'total_repositories': len(repositories),
            'successful_enhancements': successful_enhancements,
            'success_rate': successful_enhancements / len(repositories) if repositories else 0,
            'total_labels': len(all_labels),
            'unique_labels': len(set(all_labels)),
            'avg_labels_per_repo': len(all_labels) / successful_enhancements if successful_enhancements else 0,
            'most_common_labels': label_freq.most_common(20),
            'entity_type_distribution': dict(entity_types),
            'technical_category_distribution': dict(tech_categories),
            'total_entities': len(all_entities),
            'total_technical_terms': len(all_technical_terms)
        }
        
        return analysis

def main():
    """Main function for testing the NLP enhancer."""
    logger = setup_logging('INFO', 'logs/nlp_enhancer.log')
    enhancer = NLPEnhancer(logger)
    
    # Test with sample data
    sample_repos = [
        {
            'platform': 'github',
            'org_or_user': 'facebook',
            'repo': 'react',
            'full_name': 'github/facebook/react',
            'processed_text': {
                'cleaned_text': 'A declarative, efficient, and flexible JavaScript library for building user interfaces using React components and JSX syntax.',
                'is_processable': True
            }
        },
        {
            'platform': 'github',
            'org_or_user': 'tensorflow',
            'repo': 'tensorflow',
            'full_name': 'github/tensorflow/tensorflow',
            'processed_text': {
                'cleaned_text': 'An open source machine learning framework for everyone. TensorFlow provides Python APIs for deep learning and neural networks.',
                'is_processable': True
            }
        }
    ]
    
    # Enhance repositories
    enhanced_repos = enhancer.enhance_batch(sample_repos)
    
    # Analyze results
    analysis = enhancer.analyze_nlp_results(enhanced_repos)
    
    logger.info("NLP enhancement completed")
    logger.info(f"Success rate: {analysis['success_rate']:.2%}")
    logger.info(f"Average labels per repo: {analysis['avg_labels_per_repo']:.1f}")
    
    # Show sample results
    for repo in enhanced_repos:
        labels = repo.get('nlp_labels', {}).get('labels', [])
        logger.info(f"{repo['full_name']}: {labels}")

if __name__ == "__main__":
    main() 