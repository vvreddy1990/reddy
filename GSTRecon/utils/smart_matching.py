"""
Smart Name Matching Module for GST Reconciliation AI/ML Enhancement

This module provides ML-enhanced company name matching capabilities using
sentence transformers and industry-specific normalization rules.
"""

import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from fuzzywuzzy import fuzz
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartNameMatcher:
    """
    ML-enhanced company name matching system with fallback to fuzzy matching.
    
    Features:
    - Sentence transformer embeddings for semantic similarity
    - Industry-specific name normalization
    - Disk caching for embeddings
    - Graceful fallback to existing fuzzy matching
    - Performance optimization for 200-second constraint
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = ".cache/embeddings"):
        """
        Initialize SmartNameMatcher with optional model and cache configuration.
        
        Args:
            model_name: Name of the sentence transformer model to use
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model will be loaded lazily
        self.model = None
        self.model_loaded = False
        self.model_load_failed = False
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_file = self.cache_dir / f"embeddings_{model_name.replace('/', '_')}.pkl"
        
        # Performance tracking
        self.performance_metrics = {
            'model_load_time': 0.0,
            'embedding_compute_time': 0.0,
            'similarity_compute_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_count': 0
        }
        
        # Audit trail for name transformations
        self.transformation_log = []
        
        # Load cached embeddings if available
        self._load_embedding_cache()
        
        # Initialize industry-specific normalization rules
        self._init_normalization_rules()
        
        logger.info(f"SmartNameMatcher initialized with model: {model_name}")
    
    def _init_normalization_rules(self):
        """Initialize comprehensive industry-specific normalization rules."""
        
        # Business type suffixes (comprehensive list)
        self.business_suffixes = [
            # Private Limited variations
            r'private limited', r'private ltd', r'pvt ltd', r'pvt\. ltd', r'pvt\.ltd',
            r'pvt limited', r'private limited company', r'pvt ltd\.', r'p ltd',
            
            # Limited variations
            r'limited', r'ltd', r'ltd\.', r'limited company', r'lmt', r'limted',
            
            # Public Limited variations
            r'public limited', r'public limited company', r'plc', r'public ltd',
            
            # Corporation variations
            r'corporation', r'corp', r'corp\.', r'incorporated', r'inc', r'inc\.',
            
            # Company variations
            r'company', r'co', r'co\.', r'coy', r'compny',
            
            # Partnership variations
            r'limited liability partnership', r'llp', r'partnership', r'partnership firm',
            r'limited liability company', r'llc', r'one person company', r'opc',
            
            # Proprietorship variations
            r'proprietorship', r'proprietor', r'sole proprietorship', r'individual',
            
            # Trust and Society variations
            r'trust', r'society', r'foundation', r'association', r'federation',
            
            # Government variations
            r'government', r'govt', r'municipal', r'corporation', r'board',
            r'authority', r'commission', r'department', r'ministry'
        ]
        
        # Industry-specific terms
        self.industry_terms = [
            # Manufacturing
            r'industries', r'industry', r'manufacturing', r'manufacturers?', r'mfg',
            r'production', r'producers?', r'mills?', r'works', r'factory',
            
            # Services
            r'services', r'service', r'consultancy', r'consulting', r'consultants?',
            r'solutions', r'solution', r'systems?', r'enterprises?', r'enterprise',
            
            # Technology
            r'technologies', r'technology', r'tech', r'software', r'systems?',
            r'infotech', r'information technology', r'it services', r'digital',
            
            # Trading and Commerce
            r'traders?', r'trading', r'exports?', r'exporters?', r'imports?', r'importers?',
            r'distributors?', r'distribution', r'suppliers?', r'supply', r'dealers?',
            r'marketing', r'sales', r'retail', r'wholesale',
            
            # Finance
            r'finance', r'financial', r'finserv', r'fintech', r'investments?',
            r'securities', r'capital', r'funding', r'credit', r'loans?',
            
            # Banking and Insurance
            r'bank', r'banking', r'insurance', r'assurance', r'mutual fund',
            r'asset management', r'wealth', r'portfolio',
            
            # Real Estate and Construction
            r'construction', r'builders?', r'building', r'real estate', r'properties',
            r'property', r'developers?', r'development', r'infrastructure',
            r'engineering', r'architects?', r'contractors?',
            
            # Healthcare and Pharma
            r'pharmaceuticals?', r'pharma', r'healthcare', r'health', r'medical',
            r'hospitals?', r'clinics?', r'diagnostics?', r'laboratories?', r'labs?',
            
            # Food and Beverages
            r'foods?', r'food products', r'beverages?', r'drinks?', r'dairy',
            r'nutrition', r'agro', r'agriculture', r'farming',
            
            # Textiles and Garments
            r'textiles?', r'textile mills', r'garments?', r'apparel', r'clothing',
            r'fabrics?', r'yarn', r'cotton', r'silk', r'wool',
            
            # Chemicals and Materials
            r'chemicals?', r'chemical industries', r'plastics?', r'polymers?',
            r'materials', r'steel', r'metals?', r'alloys?', r'minerals?',
            
            # Automotive
            r'automotive', r'automobiles?', r'auto', r'motors?', r'vehicles?',
            r'transport', r'transportation', r'logistics',
            
            # Energy and Utilities
            r'energy', r'power', r'electricity', r'solar', r'renewable',
            r'utilities', r'gas', r'petroleum', r'oil',
            
            # Packaging and Processing
            r'packaging', r'packers?', r'processors?', r'processing',
            r'bottling', r'canning', r'printing', r'publishing',
            
            # Group and Holding
            r'group', r'holding', r'holdings', r'conglomerate', r'ventures?',
            r'investments?', r'portfolio'
        ]
        
        # Geographic and common terms
        self.geographic_terms = [
            r'india', r'indian', r'bharat', r'hindustan',
            r'mumbai', r'delhi', r'bangalore', r'chennai', r'kolkata', r'hyderabad',
            r'pune', r'ahmedabad', r'surat', r'jaipur', r'lucknow', r'kanpur',
            r'nagpur', r'indore', r'thane', r'bhopal', r'visakhapatnam', r'pimpri',
            r'patna', r'vadodara', r'ghaziabad', r'ludhiana', r'agra', r'nashik',
            r'international', r'global', r'worldwide', r'universal'
        ]
        
        # Common connecting words
        self.connecting_words = [
            r'and', r'&', r'of', r'the', r'for', r'with', r'in', r'at', r'on',
            r'by', r'from', r'to', r'as', r'an', r'a'
        ]
        
        # Compile all patterns for efficiency
        self._compile_normalization_patterns()
    
    def _compile_normalization_patterns(self):
        """Compile regex patterns for efficient normalization."""
        
        # Combine all suffix patterns
        all_suffixes = (self.business_suffixes + self.industry_terms + 
                       self.geographic_terms + self.connecting_words)
        
        # Create comprehensive pattern
        self.suffix_pattern = re.compile(
            r'\b(?:' + '|'.join(all_suffixes) + r')\b', 
            re.IGNORECASE
        )
        
        # Pattern for special characters
        self.special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
        # Pattern for multiple spaces
        self.space_pattern = re.compile(r'\s+')
        
        # Pattern for common abbreviations
        self.abbreviation_patterns = {
            re.compile(r'\bpvt\b', re.IGNORECASE): 'private',
            re.compile(r'\bltd\b', re.IGNORECASE): 'limited',
            re.compile(r'\binc\b', re.IGNORECASE): 'incorporated',
            re.compile(r'\bcorp\b', re.IGNORECASE): 'corporation',
            re.compile(r'\bmfg\b', re.IGNORECASE): 'manufacturing',
            re.compile(r'\bco\b', re.IGNORECASE): 'company'
        }
    
    def _load_model(self) -> bool:
        """
        Lazily load the sentence transformer model with error handling.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self.model_loaded:
            return True
        
        if self.model_load_failed:
            return False
        
        try:
            import time
            start_time = time.time()
            
            # Try to import sentence transformers
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning("sentence-transformers not installed. Falling back to fuzzy matching.")
                self.model_load_failed = True
                return False
            
            # Load the model
            self.model = SentenceTransformer(self.model_name)
            self.model_loaded = True
            
            load_time = time.time() - start_time
            self.performance_metrics['model_load_time'] = load_time
            
            logger.info(f"Model {self.model_name} loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            self.model_load_failed = True
            return False
    
    def _load_embedding_cache(self):
        """Load cached embeddings from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {str(e)}")
            self.embedding_cache = {}
    
    def _save_embedding_cache(self):
        """Save embeddings to disk cache."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.debug(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def normalize_company_name(self, name: str, create_audit_trail: bool = True) -> str:
        """
        Normalize company name using comprehensive industry-specific rules.
        
        Args:
            name: Raw company name
            create_audit_trail: Whether to create audit trail for transformations
            
        Returns:
            str: Normalized company name
        """
        if not isinstance(name, str) or pd.isna(name):
            return ''
        
        original_name = name
        transformations = []
        
        # Step 1: Basic cleaning
        normalized = name.strip()
        if normalized != name:
            transformations.append(f"Trimmed whitespace")
        
        # Step 2: Convert to lowercase
        normalized = normalized.lower()
        transformations.append(f"Converted to lowercase")
        
        # Step 3: Expand common abbreviations first (before removing them)
        for pattern, replacement in self.abbreviation_patterns.items():
            old_normalized = normalized
            normalized = pattern.sub(replacement, normalized)
            if normalized != old_normalized:
                transformations.append(f"Expanded abbreviation: {pattern.pattern} -> {replacement}")
        
        # Step 4: Remove business suffixes and industry terms
        old_normalized = normalized
        normalized = self.suffix_pattern.sub('', normalized)
        if normalized != old_normalized:
            transformations.append(f"Removed business suffixes and industry terms")
        
        # Step 5: Remove special characters (keep only alphanumeric and spaces)
        old_normalized = normalized
        normalized = self.special_char_pattern.sub(' ', normalized)
        if normalized != old_normalized:
            transformations.append(f"Removed special characters")
        
        # Step 6: Normalize multiple spaces to single space
        old_normalized = normalized
        normalized = self.space_pattern.sub(' ', normalized)
        if normalized != old_normalized:
            transformations.append(f"Normalized multiple spaces")
        
        # Step 7: Final trim
        normalized = normalized.strip()
        
        # Create audit trail if requested
        if create_audit_trail and transformations:
            audit_entry = {
                'original_name': original_name,
                'normalized_name': normalized,
                'transformations': transformations,
                'timestamp': pd.Timestamp.now()
            }
            self.transformation_log.append(audit_entry)
        
        return normalized
    
    def get_industry_normalized_name(self, name: str) -> str:
        """
        Get industry-specific normalized name with detailed transformation tracking.
        
        Args:
            name: Company name to normalize
            
        Returns:
            str: Industry-normalized company name
        """
        return self.normalize_company_name(name, create_audit_trail=True)
    
    def get_transformation_log(self) -> List[Dict[str, Any]]:
        """
        Get the audit trail of all name transformations.
        
        Returns:
            List of transformation records
        """
        return self.transformation_log.copy()
    
    def clear_transformation_log(self):
        """Clear the transformation audit trail."""
        self.transformation_log.clear()
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of name transformations.
        
        Returns:
            Dictionary with transformation statistics
        """
        if not self.transformation_log:
            return {
                'total_transformations': 0,
                'unique_names_processed': 0,
                'common_transformations': {},
                'avg_transformations_per_name': 0
            }
        
        # Count transformation types
        transformation_counts = {}
        unique_names = set()
        
        for entry in self.transformation_log:
            unique_names.add(entry['original_name'])
            for transformation in entry['transformations']:
                transformation_counts[transformation] = transformation_counts.get(transformation, 0) + 1
        
        # Get most common transformations
        common_transformations = dict(sorted(transformation_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:10])
        
        return {
            'total_transformations': len(self.transformation_log),
            'unique_names_processed': len(unique_names),
            'common_transformations': common_transformations,
            'avg_transformations_per_name': len(self.transformation_log) / len(unique_names) if unique_names else 0
        }
    
    def compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Compute embedding for a single text with caching.
        
        Args:
            text: Text to compute embedding for
            
        Returns:
            np.ndarray or None: Embedding vector or None if failed
        """
        if not text or pd.isna(text):
            return None
        
        # Normalize the text first
        normalized_text = self.normalize_company_name(text)
        if not normalized_text:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(normalized_text)
        if cache_key in self.embedding_cache:
            self.performance_metrics['cache_hits'] += 1
            return self.embedding_cache[cache_key]
        
        # Load model if needed
        if not self._load_model():
            return None
        
        try:
            import time
            start_time = time.time()
            
            # Compute embedding
            embedding = self.model.encode([normalized_text])[0]
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            self.performance_metrics['cache_misses'] += 1
            
            compute_time = time.time() - start_time
            self.performance_metrics['embedding_compute_time'] += compute_time
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to compute embedding for '{text}': {str(e)}")
            return None
    
    def precompute_embeddings(self, names: List[str]) -> Dict[str, np.ndarray]:
        """
        Precompute embeddings for a list of company names.
        
        Args:
            names: List of company names
            
        Returns:
            Dict mapping original names to embeddings
        """
        embeddings = {}
        
        if not self._load_model():
            logger.warning("Model not available, skipping embedding precomputation")
            return embeddings
        
        # Filter out invalid names
        valid_names = [name for name in names if isinstance(name, str) and not pd.isna(name)]
        
        if not valid_names:
            return embeddings
        
        try:
            import time
            start_time = time.time()
            
            # Normalize all names
            normalized_names = [self.normalize_company_name(name) for name in valid_names]
            
            # Filter out empty normalized names
            valid_pairs = [(orig, norm) for orig, norm in zip(valid_names, normalized_names) if norm]
            
            if not valid_pairs:
                return embeddings
            
            original_names, normalized_names = zip(*valid_pairs)
            
            # Check which embeddings we already have cached
            uncached_indices = []
            uncached_names = []
            
            for i, norm_name in enumerate(normalized_names):
                cache_key = self._get_cache_key(norm_name)
                if cache_key in self.embedding_cache:
                    embeddings[original_names[i]] = self.embedding_cache[cache_key]
                    self.performance_metrics['cache_hits'] += 1
                else:
                    uncached_indices.append(i)
                    uncached_names.append(norm_name)
            
            # Compute embeddings for uncached names in batch
            if uncached_names:
                batch_embeddings = self.model.encode(uncached_names)
                
                for i, embedding in enumerate(batch_embeddings):
                    orig_idx = uncached_indices[i]
                    orig_name = original_names[orig_idx]
                    norm_name = uncached_names[i]
                    
                    # Store in results and cache
                    embeddings[orig_name] = embedding
                    cache_key = self._get_cache_key(norm_name)
                    self.embedding_cache[cache_key] = embedding
                    self.performance_metrics['cache_misses'] += 1
            
            compute_time = time.time() - start_time
            self.performance_metrics['embedding_compute_time'] += compute_time
            
            logger.info(f"Precomputed {len(embeddings)} embeddings in {compute_time:.2f}s")
            
            # Save cache periodically
            if len(self.embedding_cache) % 100 == 0:
                self._save_embedding_cache()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to precompute embeddings: {str(e)}")
            return embeddings
    
    def calculate_similarity_scores(self, names1: List[str], names2: List[str]) -> List[float]:
        """
        Calculate similarity scores between two lists of company names.
        
        Args:
            names1: First list of company names
            names2: Second list of company names
            
        Returns:
            List of similarity scores (0.0 to 1.0)
        """
        if len(names1) != len(names2):
            raise ValueError("Input lists must have the same length")
        
        scores = []
        
        # Try ML-enhanced scoring first
        if self._load_model():
            try:
                import time
                start_time = time.time()
                
                ml_scores = self._calculate_ml_similarity_scores(names1, names2)
                
                compute_time = time.time() - start_time
                self.performance_metrics['similarity_compute_time'] += compute_time
                
                if ml_scores:
                    return ml_scores
                    
            except Exception as e:
                logger.error(f"ML similarity calculation failed: {str(e)}")
        
        # Fallback to fuzzy matching
        self.performance_metrics['fallback_count'] += len(names1)
        return self._calculate_fuzzy_similarity_scores(names1, names2)
    
    def _calculate_ml_similarity_scores(self, names1: List[str], names2: List[str]) -> List[float]:
        """Calculate similarity scores using ML embeddings."""
        scores = []
        
        for name1, name2 in zip(names1, names2):
            # Get embeddings
            emb1 = self.compute_embedding(name1)
            emb2 = self.compute_embedding(name2)
            
            if emb1 is not None and emb2 is not None:
                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                # Convert to 0-1 range (cosine similarity is -1 to 1)
                similarity = (similarity + 1) / 2
                scores.append(float(similarity))
            else:
                # Fallback to fuzzy matching for this pair
                fuzzy_score = self._calculate_fuzzy_similarity_single(name1, name2)
                scores.append(fuzzy_score)
                self.performance_metrics['fallback_count'] += 1
        
        return scores
    
    def _calculate_fuzzy_similarity_scores(self, names1: List[str], names2: List[str]) -> List[float]:
        """Calculate similarity scores using fuzzy matching."""
        scores = []
        
        for name1, name2 in zip(names1, names2):
            score = self._calculate_fuzzy_similarity_single(name1, name2)
            scores.append(score)
        
        return scores
    
    def _calculate_fuzzy_similarity_single(self, name1: str, name2: str) -> float:
        """Calculate fuzzy similarity score for a single pair of names."""
        if not isinstance(name1, str) or not isinstance(name2, str):
            return 0.0
        
        if pd.isna(name1) or pd.isna(name2):
            return 0.0
        
        # Normalize names
        norm1 = self.normalize_company_name(name1)
        norm2 = self.normalize_company_name(name2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Use fuzzywuzzy ratio and convert to 0-1 scale
        return fuzz.ratio(norm1, norm2) / 100.0
    
    def enhance_similarity_scores(self, names1: List[str], names2: List[str]) -> List[float]:
        """
        Main method to enhance similarity scores with ML while maintaining compatibility.
        
        Args:
            names1: First list of company names
            names2: Second list of company names
            
        Returns:
            List of enhanced similarity scores (0.0 to 1.0)
        """
        return self.calculate_similarity_scores(names1, names2)
    
    def standardize_business_type(self, name: str) -> Tuple[str, str]:
        """
        Standardize business type and return both normalized name and business type.
        
        Args:
            name: Company name
            
        Returns:
            Tuple of (normalized_name, business_type)
        """
        if not isinstance(name, str) or pd.isna(name):
            return '', 'Unknown'
        
        name_lower = name.lower()
        
        # Define business type patterns and their standardized forms
        business_type_patterns = {
            'Private Limited': [
                r'\bprivate limited\b', r'\bpvt ltd\b', r'\bpvt\. ltd\b', 
                r'\bpvt limited\b', r'\bp ltd\b'
            ],
            'Limited': [
                r'\blimited\b', r'\bltd\b', r'\bltd\.\b'
            ],
            'Public Limited': [
                r'\bpublic limited\b', r'\bplc\b', r'\bpublic ltd\b'
            ],
            'Corporation': [
                r'\bcorporation\b', r'\bcorp\b', r'\bcorp\.\b', 
                r'\bincorporated\b', r'\binc\b', r'\binc\.\b'
            ],
            'Partnership': [
                r'\bpartnership\b', r'\bpartnership firm\b', r'\bllp\b',
                r'\blimited liability partnership\b'
            ],
            'Proprietorship': [
                r'\bproprietorship\b', r'\bproprietor\b', r'\bsole proprietorship\b'
            ],
            'Company': [
                r'\bcompany\b', r'\bco\b', r'\bco\.\b'
            ],
            'Trust': [
                r'\btrust\b', r'\bsociety\b', r'\bfoundation\b'
            ],
            'Government': [
                r'\bgovernment\b', r'\bgovt\b', r'\bmunicipal\b', 
                r'\bauthority\b', r'\bboard\b'
            ]
        }
        
        detected_type = 'Unknown'
        normalized_name = name
        
        # Find matching business type
        for business_type, patterns in business_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    detected_type = business_type
                    # Remove the business type from name
                    normalized_name = re.sub(pattern, '', name_lower, flags=re.IGNORECASE)
                    break
            if detected_type != 'Unknown':
                break
        
        # Clean up the normalized name
        normalized_name = self.normalize_company_name(normalized_name, create_audit_trail=False)
        
        return normalized_name, detected_type
    
    def batch_normalize_names(self, names: List[str]) -> List[Dict[str, Any]]:
        """
        Normalize a batch of company names with detailed results.
        
        Args:
            names: List of company names to normalize
            
        Returns:
            List of dictionaries with normalization results
        """
        results = []
        
        for name in names:
            if not isinstance(name, str) or pd.isna(name):
                results.append({
                    'original_name': name,
                    'normalized_name': '',
                    'business_type': 'Unknown',
                    'transformations_applied': [],
                    'confidence_score': 0.0
                })
                continue
            
            # Get initial transformation count
            initial_log_count = len(self.transformation_log)
            
            # Normalize the name
            normalized_name = self.normalize_company_name(name, create_audit_trail=True)
            
            # Get business type
            _, business_type = self.standardize_business_type(name)
            
            # Get transformations applied to this name
            transformations_applied = []
            if len(self.transformation_log) > initial_log_count:
                latest_entry = self.transformation_log[-1]
                transformations_applied = latest_entry['transformations']
            
            # Calculate confidence score based on transformations
            confidence_score = self._calculate_normalization_confidence(name, normalized_name, transformations_applied)
            
            results.append({
                'original_name': name,
                'normalized_name': normalized_name,
                'business_type': business_type,
                'transformations_applied': transformations_applied,
                'confidence_score': confidence_score
            })
        
        return results
    
    def _calculate_normalization_confidence(self, original: str, normalized: str, transformations: List[str]) -> float:
        """
        Calculate confidence score for name normalization.
        
        Args:
            original: Original name
            normalized: Normalized name
            transformations: List of transformations applied
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not original or not normalized:
            return 0.0
        
        # Base confidence
        confidence = 0.8
        
        # Adjust based on length preservation
        length_ratio = len(normalized) / len(original) if original else 0
        if length_ratio < 0.3:  # Too much removed
            confidence -= 0.2
        elif length_ratio > 0.7:  # Good preservation
            confidence += 0.1
        
        # Adjust based on number of transformations
        if len(transformations) > 5:  # Too many transformations might indicate issues
            confidence -= 0.1
        
        # Adjust based on whether core content remains
        if len(normalized.split()) == 0:  # No words left
            confidence = 0.1
        elif len(normalized.split()) == 1 and len(original.split()) > 3:  # Too much reduction
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        total_requests = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
        cache_hit_rate = (self.performance_metrics['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.performance_metrics,
            'cache_hit_rate_percent': cache_hit_rate,
            'model_available': self.model_loaded,
            'total_cached_embeddings': len(self.embedding_cache),
            'transformation_stats': self.get_transformation_summary()
        }
    
    def cleanup(self):
        """Clean up resources and save cache."""
        self._save_embedding_cache()
        logger.info("SmartNameMatcher cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cache is saved."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup