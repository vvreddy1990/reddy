"""
Intelligent Caching System for AI/ML Performance Optimization

This module provides hash-based data fingerprinting, cache storage and retrieval,
cache invalidation, cleanup functionality, and performance metrics monitoring.
"""

import os
import json
import pickle
import hashlib
import gzip
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import threading
import time
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    key: str
    data_hash: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_hours: int
    compressed: bool = False
    metadata: Dict[str, Any] = None
    similarity_score: float = 0.0
    access_pattern: List[datetime] = None
    prediction_score: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.access_pattern is None:
            self.access_pattern = []
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_hours <= 0:
            return False  # Never expires
        
        expiry_time = self.created_at + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.access_pattern.append(datetime.now())
        
        # Keep only last 100 access times for pattern analysis
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-100:]
    
    def get_access_frequency(self, hours: int = 24) -> float:
        """Calculate access frequency in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_accesses = [t for t in self.access_pattern if t > cutoff_time]
        return len(recent_accesses) / hours if hours > 0 else 0.0
    
    def get_recency_score(self) -> float:
        """Calculate recency score (0-1, higher is more recent)."""
        if not self.access_pattern:
            return 0.0
        
        hours_since_last_access = (datetime.now() - self.last_accessed).total_seconds() / 3600
        # Exponential decay with half-life of 24 hours
        return np.exp(-hours_since_last_access / 24)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    total_entries: int = 0
    total_size_mb: float = 0.0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    cleanup_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        return (self.hit_count / total_requests * 100) if total_requests > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 100.0 - self.hit_rate


class DataSimilarityDetector:
    """Detects similarity between datasets for cache reuse."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.similarity_threshold = 0.8
        
    def extract_features(self, data: Any) -> np.ndarray:
        """Extract features from data for similarity comparison."""
        try:
            features = []
            
            if isinstance(data, pd.DataFrame):
                # DataFrame features
                features.extend([
                    len(data),  # Number of rows
                    len(data.columns),  # Number of columns
                    data.memory_usage(deep=True).sum(),  # Memory usage
                    len(data.select_dtypes(include=[np.number]).columns),  # Numeric columns
                    len(data.select_dtypes(include=['object']).columns),  # String columns
                    data.isnull().sum().sum(),  # Total null values
                ])
                
                # Column name similarity (convert to text)
                column_text = ' '.join(data.columns.astype(str))
                features.append(hash(column_text) % 10000)  # Hash of column names
                
            elif isinstance(data, dict):
                # Dictionary features
                features.extend([
                    len(data),  # Number of keys
                    len(str(data)),  # String length
                    len([v for v in data.values() if isinstance(v, (int, float))]),  # Numeric values
                    len([v for v in data.values() if isinstance(v, str)]),  # String values
                ])
                
            elif isinstance(data, list):
                # List features
                features.extend([
                    len(data),  # Length
                    len(str(data)),  # String representation length
                    len([x for x in data if isinstance(x, (int, float))]),  # Numeric items
                    len([x for x in data if isinstance(x, str)]),  # String items
                ])
            
            else:
                # Generic features
                features.extend([
                    len(str(data)),  # String length
                    hash(str(type(data))) % 10000,  # Type hash
                ])
            
            # Pad or truncate to fixed size
            target_size = 10
            if len(features) < target_size:
                features.extend([0] * (target_size - len(features)))
            else:
                features = features[:target_size]
            
            return np.array(features, dtype=float)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros(10)  # Return zero vector on error
    
    def calculate_similarity(self, data1: Any, data2: Any) -> float:
        """Calculate similarity score between two datasets."""
        try:
            features1 = self.extract_features(data1)
            features2 = self.extract_features(data2)
            
            # Reshape for sklearn
            features1 = features1.reshape(1, -1)
            features2 = features2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(features1, features2)[0, 0]
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_entries(self, data: Any, cache_entries: Dict[str, CacheEntry], threshold: float = None) -> List[Tuple[str, float]]:
        """Find cache entries similar to the given data."""
        if threshold is None:
            threshold = self.similarity_threshold
        
        similar_entries = []
        data_features = self.extract_features(data)
        
        for key, entry in cache_entries.items():
            try:
                # Extract features from cached data metadata if available
                if 'features' in entry.metadata:
                    cached_features = np.array(entry.metadata['features'])
                else:
                    # Skip if no features stored
                    continue
                
                # Calculate similarity
                similarity = cosine_similarity(
                    data_features.reshape(1, -1),
                    cached_features.reshape(1, -1)
                )[0, 0]
                
                if similarity >= threshold:
                    similar_entries.append((key, similarity))
                    
            except Exception as e:
                logger.error(f"Error comparing with entry {key}: {e}")
                continue
        
        # Sort by similarity (highest first)
        similar_entries.sort(key=lambda x: x[1], reverse=True)
        return similar_entries


class CacheUsagePredictor:
    """Machine learning model for predicting cache usage patterns."""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'data_size', 'complexity_score', 'access_frequency', 'recency_score',
            'similarity_score', 'time_of_day', 'day_of_week', 'processing_time',
            'memory_usage', 'cache_hit_rate'
        ]
    
    def extract_prediction_features(self, data: Any, entry: CacheEntry = None, context: Dict[str, Any] = None) -> np.ndarray:
        """Extract features for cache usage prediction."""
        try:
            features = []
            now = datetime.now()
            
            # Data characteristics
            if isinstance(data, pd.DataFrame):
                data_size = len(data) * len(data.columns)
                complexity_score = min(1.0, data_size / 100000)  # Normalize complexity
                memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            elif isinstance(data, (dict, list)):
                data_size = len(data)
                complexity_score = min(1.0, data_size / 10000)
                memory_usage = len(str(data)) / 1024 / 1024  # Rough estimate
            else:
                data_size = len(str(data))
                complexity_score = 0.1
                memory_usage = data_size / 1024 / 1024
            
            features.extend([
                data_size,
                complexity_score,
                memory_usage
            ])
            
            # Entry-specific features
            if entry:
                features.extend([
                    entry.get_access_frequency(),
                    entry.get_recency_score(),
                    entry.similarity_score
                ])
            else:
                features.extend([0.0, 0.0, 0.0])  # Default values for new entries
            
            # Temporal features
            features.extend([
                now.hour / 24.0,  # Time of day (normalized)
                now.weekday() / 6.0,  # Day of week (normalized)
            ])
            
            # Context features
            if context:
                features.extend([
                    context.get('processing_time', 0.0),
                    context.get('cache_hit_rate', 0.0)
                ])
            else:
                features.extend([0.0, 0.0])
            
            return np.array(features, dtype=float)
            
        except Exception as e:
            logger.error(f"Error extracting prediction features: {e}")
            return np.zeros(len(self.feature_names))
    
    def train(self, training_data: List[Dict[str, Any]]):
        """Train the cache usage prediction model."""
        try:
            if len(training_data) < 10:
                logger.warning("Insufficient training data for cache prediction model")
                return
            
            X = []
            y = []
            
            for sample in training_data:
                features = self.extract_prediction_features(
                    sample['data'],
                    sample.get('entry'),
                    sample.get('context')
                )
                X.append(features)
                y.append(1 if sample['was_useful'] else 0)  # Binary classification
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info(f"Cache prediction model trained on {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Error training cache prediction model: {e}")
    
    def predict_usefulness(self, data: Any, entry: CacheEntry = None, context: Dict[str, Any] = None) -> float:
        """Predict the probability that caching this data will be useful."""
        try:
            if not self.is_trained:
                # Return heuristic-based prediction
                return self._heuristic_prediction(data, entry, context)
            
            features = self.extract_prediction_features(data, entry, context)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get probability of positive class (useful cache)
            probability = self.model.predict_proba(features_scaled)[0][1]
            return probability
            
        except Exception as e:
            logger.error(f"Error predicting cache usefulness: {e}")
            return 0.5  # Default moderate probability
    
    def _heuristic_prediction(self, data: Any, entry: CacheEntry = None, context: Dict[str, Any] = None) -> float:
        """Heuristic-based prediction when ML model is not trained."""
        score = 0.5  # Base score
        
        # Data size factor
        if isinstance(data, pd.DataFrame):
            size = len(data) * len(data.columns)
            if size > 10000:
                score += 0.2
            elif size > 1000:
                score += 0.1
        
        # Entry history factor
        if entry:
            if entry.access_count > 5:
                score += 0.2
            if entry.get_access_frequency() > 1.0:  # More than once per day
                score += 0.1
        
        # Context factor
        if context:
            processing_time = context.get('processing_time', 0)
            if processing_time > 10:  # Expensive operations benefit more from caching
                score += 0.2
        
        return min(1.0, score)
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        try:
            if self.is_trained:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names
                }
                joblib.dump(model_data, filepath)
                logger.info(f"Cache prediction model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = True
                logger.info(f"Cache prediction model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")


class IntelligentEvictionPolicy:
    """Intelligent cache eviction policy using multiple factors."""
    
    def __init__(self):
        self.weights = {
            'recency': 0.3,
            'frequency': 0.25,
            'size': 0.15,
            'prediction': 0.2,
            'similarity': 0.1
        }
    
    def calculate_eviction_score(self, entry: CacheEntry, predictor: CacheUsagePredictor = None) -> float:
        """Calculate eviction score (higher score = more likely to evict)."""
        try:
            score = 0.0
            
            # Recency factor (older entries have higher eviction score)
            recency_score = 1.0 - entry.get_recency_score()
            score += self.weights['recency'] * recency_score
            
            # Frequency factor (less frequently accessed entries have higher eviction score)
            frequency = entry.get_access_frequency()
            max_frequency = 10.0  # Normalize to reasonable range
            frequency_score = 1.0 - min(1.0, frequency / max_frequency)
            score += self.weights['frequency'] * frequency_score
            
            # Size factor (larger entries have higher eviction score)
            size_mb = entry.size_bytes / 1024 / 1024
            max_size = 100.0  # Normalize to reasonable range
            size_score = min(1.0, size_mb / max_size)
            score += self.weights['size'] * size_score
            
            # Prediction factor (entries predicted to be less useful have higher eviction score)
            if predictor and predictor.is_trained:
                usefulness = predictor.predict_usefulness(None, entry)
                prediction_score = 1.0 - usefulness
                score += self.weights['prediction'] * prediction_score
            
            # Similarity factor (entries with low similarity scores have higher eviction score)
            similarity_score = 1.0 - entry.similarity_score
            score += self.weights['similarity'] * similarity_score
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating eviction score: {e}")
            return 0.5  # Default moderate score
    
    def select_eviction_candidates(self, entries: Dict[str, CacheEntry], count: int, predictor: CacheUsagePredictor = None) -> List[str]:
        """Select entries for eviction based on intelligent scoring."""
        try:
            # Calculate eviction scores for all entries
            scored_entries = []
            for key, entry in entries.items():
                score = self.calculate_eviction_score(entry, predictor)
                scored_entries.append((key, score))
            
            # Sort by eviction score (highest first)
            scored_entries.sort(key=lambda x: x[1], reverse=True)
            
            # Return top candidates
            return [key for key, _ in scored_entries[:count]]
            
        except Exception as e:
            logger.error(f"Error selecting eviction candidates: {e}")
            return list(entries.keys())[:count]  # Fallback to simple selection


class CachePreloader:
    """Preloads cache with commonly used patterns."""
    
    def __init__(self):
        self.common_patterns = defaultdict(int)
        self.pattern_history = deque(maxlen=1000)  # Keep last 1000 access patterns
    
    def record_access_pattern(self, data_characteristics: Dict[str, Any]):
        """Record an access pattern for future preloading."""
        try:
            # Create a pattern signature
            pattern = {
                'data_type': data_characteristics.get('type', 'unknown'),
                'size_range': self._get_size_range(data_characteristics.get('size', 0)),
                'complexity': data_characteristics.get('complexity', 'low'),
                'time_of_day': datetime.now().hour // 4,  # 6-hour buckets
                'day_of_week': datetime.now().weekday()
            }
            
            pattern_key = json.dumps(pattern, sort_keys=True)
            self.common_patterns[pattern_key] += 1
            self.pattern_history.append((datetime.now(), pattern_key))
            
        except Exception as e:
            logger.error(f"Error recording access pattern: {e}")
    
    def _get_size_range(self, size: int) -> str:
        """Categorize data size into ranges."""
        if size < 100:
            return 'small'
        elif size < 10000:
            return 'medium'
        elif size < 1000000:
            return 'large'
        else:
            return 'xlarge'
    
    def get_preload_candidates(self, current_time: datetime = None) -> List[Dict[str, Any]]:
        """Get candidates for cache preloading based on patterns."""
        try:
            if current_time is None:
                current_time = datetime.now()
            
            candidates = []
            
            # Find patterns that are likely to be accessed soon
            current_hour_bucket = current_time.hour // 4
            current_day = current_time.weekday()
            
            for pattern_key, frequency in self.common_patterns.items():
                if frequency < 3:  # Only consider patterns seen at least 3 times
                    continue
                
                try:
                    pattern = json.loads(pattern_key)
                    
                    # Check if pattern matches current time context
                    if (pattern['time_of_day'] == current_hour_bucket or
                        pattern['day_of_week'] == current_day):
                        
                        candidates.append({
                            'pattern': pattern,
                            'frequency': frequency,
                            'priority': self._calculate_preload_priority(pattern, frequency, current_time)
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing pattern {pattern_key}: {e}")
                    continue
            
            # Sort by priority (highest first)
            candidates.sort(key=lambda x: x['priority'], reverse=True)
            
            return candidates[:10]  # Return top 10 candidates
            
        except Exception as e:
            logger.error(f"Error getting preload candidates: {e}")
            return []
    
    def _calculate_preload_priority(self, pattern: Dict[str, Any], frequency: int, current_time: datetime) -> float:
        """Calculate priority score for preloading a pattern."""
        try:
            score = frequency * 0.5  # Base score from frequency
            
            # Time-based boost
            if pattern['time_of_day'] == current_time.hour // 4:
                score += 2.0
            
            if pattern['day_of_week'] == current_time.weekday():
                score += 1.0
            
            # Complexity boost (more complex operations benefit more from preloading)
            if pattern['complexity'] == 'high':
                score += 1.5
            elif pattern['complexity'] == 'medium':
                score += 1.0
            
            # Size boost (larger data benefits more from caching)
            if pattern['size_range'] in ['large', 'xlarge']:
                score += 1.0
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating preload priority: {e}")
            return 0.0


class DataHasher:
    """Utility class for creating consistent data fingerprints."""
    
    @staticmethod
    def hash_dataframe(df: pd.DataFrame) -> str:
        """Create hash for pandas DataFrame."""
        try:
            # Create a consistent hash based on data content and structure
            content_hash = hashlib.md5()
            
            # Hash column names and types
            columns_str = str(sorted(df.columns.tolist()))
            content_hash.update(columns_str.encode())
            
            # Hash data types
            dtypes_str = str(df.dtypes.to_dict())
            content_hash.update(dtypes_str.encode())
            
            # Hash shape
            shape_str = str(df.shape)
            content_hash.update(shape_str.encode())
            
            # Hash a sample of the data for large datasets
            if len(df) > 1000:
                sample_df = df.sample(n=100, random_state=42).sort_index()
            else:
                sample_df = df.sort_index()
            
            # Convert to string and hash
            data_str = sample_df.to_string()
            content_hash.update(data_str.encode())
            
            return content_hash.hexdigest()
            
        except Exception as e:
            logger.error(f"Error hashing DataFrame: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    @staticmethod
    def hash_dict(data: Dict[str, Any]) -> str:
        """Create hash for dictionary data."""
        try:
            # Sort keys for consistent hashing
            sorted_data = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(sorted_data.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing dictionary: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    @staticmethod
    def hash_list(data: List[Any]) -> str:
        """Create hash for list data."""
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing list: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()
    
    @staticmethod
    def hash_mixed_data(data: Any) -> str:
        """Create hash for mixed data types."""
        try:
            if isinstance(data, pd.DataFrame):
                return DataHasher.hash_dataframe(data)
            elif isinstance(data, dict):
                return DataHasher.hash_dict(data)
            elif isinstance(data, list):
                return DataHasher.hash_list(data)
            elif isinstance(data, np.ndarray):
                return hashlib.md5(data.tobytes()).hexdigest()
            else:
                return hashlib.md5(str(data).encode()).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing mixed data: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()


class IntelligentCache:
    """
    Intelligent caching system with predictive algorithms, similarity detection,
    and advanced eviction policies for optimal performance.
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache/ai_ml",
        max_size_mb: int = 500,
        default_ttl_hours: int = 24,
        compression_enabled: bool = True,
        auto_cleanup: bool = True,
        enable_prediction: bool = True,
        enable_preloading: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.default_ttl_hours = default_ttl_hours
        self.compression_enabled = compression_enabled
        self.auto_cleanup = auto_cleanup
        self.enable_prediction = enable_prediction
        self.enable_preloading = enable_preloading
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.entries: Dict[str, CacheEntry] = {}
        self.metrics = CacheMetrics()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Predictive caching components
        self.similarity_detector = DataSimilarityDetector()
        self.usage_predictor = CacheUsagePredictor()
        self.eviction_policy = IntelligentEvictionPolicy()
        self.preloader = CachePreloader()
        
        # Model files
        self.model_file = self.cache_dir / "usage_predictor.joblib"
        
        # Load existing cache metadata
        self._load_metadata()
        
        # Load prediction model if available
        if self.enable_prediction:
            self.usage_predictor.load_model(str(self.model_file))
        
        # Perform initial cleanup if enabled
        if self.auto_cleanup:
            self._cleanup_expired_entries()
        
        # Start background preloading if enabled
        if self.enable_preloading:
            self._start_preloading_thread()
        
        logger.info(f"Intelligent cache initialized at {self.cache_dir} with predictive features")
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Reconstruct cache entries
                for key, entry_data in metadata.get('entries', {}).items():
                    # Handle access pattern (convert from ISO strings back to datetime objects)
                    access_pattern = []
                    if 'access_pattern' in entry_data:
                        for timestamp_str in entry_data['access_pattern']:
                            try:
                                access_pattern.append(datetime.fromisoformat(timestamp_str))
                            except:
                                pass  # Skip invalid timestamps
                    
                    entry = CacheEntry(
                        key=entry_data['key'],
                        data_hash=entry_data['data_hash'],
                        created_at=datetime.fromisoformat(entry_data['created_at']),
                        last_accessed=datetime.fromisoformat(entry_data['last_accessed']),
                        access_count=entry_data['access_count'],
                        size_bytes=entry_data['size_bytes'],
                        ttl_hours=entry_data['ttl_hours'],
                        compressed=entry_data.get('compressed', False),
                        metadata=entry_data.get('metadata', {}),
                        similarity_score=entry_data.get('similarity_score', 0.0),
                        access_pattern=access_pattern,
                        prediction_score=entry_data.get('prediction_score', 0.0)
                    )
                    self.entries[key] = entry
                
                # Load metrics
                if 'metrics' in metadata:
                    metrics_data = metadata['metrics']
                    self.metrics = CacheMetrics(**metrics_data)
                
                logger.info(f"Loaded {len(self.entries)} cache entries")
            
        except Exception as e:
            logger.error(f"Error loading cache metadata: {e}")
            self.entries = {}
            self.metrics = CacheMetrics()
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            metadata = {
                'entries': {},
                'metrics': asdict(self.metrics),
                'last_updated': datetime.now().isoformat()
            }
            
            # Serialize cache entries
            for key, entry in self.entries.items():
                # Convert access pattern to ISO strings
                access_pattern_str = []
                if hasattr(entry, 'access_pattern') and entry.access_pattern:
                    access_pattern_str = [t.isoformat() for t in entry.access_pattern[-50:]]  # Keep last 50
                
                metadata['entries'][key] = {
                    'key': entry.key,
                    'data_hash': entry.data_hash,
                    'created_at': entry.created_at.isoformat(),
                    'last_accessed': entry.last_accessed.isoformat(),
                    'access_count': entry.access_count,
                    'size_bytes': entry.size_bytes,
                    'ttl_hours': entry.ttl_hours,
                    'compressed': entry.compressed,
                    'metadata': entry.metadata,
                    'similarity_score': getattr(entry, 'similarity_score', 0.0),
                    'access_pattern': access_pattern_str,
                    'prediction_score': getattr(entry, 'prediction_score', 0.0)
                }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _get_cache_file_path(self, key: str, compressed: bool = False) -> Path:
        """Get file path for cache entry."""
        extension = ".pkl.gz" if compressed else ".pkl"
        return self.cache_dir / f"{key}{extension}"
    
    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate hash for data to be cached."""
        return DataHasher.hash_mixed_data(data)
    
    def _serialize_data(self, data: Any, compress: bool = False) -> bytes:
        """Serialize data for storage."""
        try:
            serialized = pickle.dumps(data)
            
            if compress:
                serialized = gzip.compress(serialized)
            
            return serialized
            
        except Exception as e:
            logger.error(f"Error serializing data: {e}")
            raise
    
    def _deserialize_data(self, data: bytes, compressed: bool = False) -> Any:
        """Deserialize data from storage."""
        try:
            if compressed:
                data = gzip.decompress(data)
            
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            raise
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries."""
        with self._lock:
            expired_keys = []
            
            for key, entry in self.entries.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
                self.metrics.cleanup_count += 1
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                self._save_metadata()
    
    def _remove_entry(self, key: str):
        """Remove a cache entry from disk and memory."""
        try:
            if key in self.entries:
                entry = self.entries[key]
                
                # Remove file
                cache_file = self._get_cache_file_path(key, entry.compressed)
                if cache_file.exists():
                    cache_file.unlink()
                
                # Remove from memory
                del self.entries[key]
                
                # Update metrics
                self.metrics.total_entries -= 1
                self.metrics.total_size_mb -= entry.size_bytes / 1024 / 1024
                
        except Exception as e:
            logger.error(f"Error removing cache entry {key}: {e}")
    
    def _enforce_size_limit(self):
        """Enforce cache size limit using intelligent eviction policy."""
        current_size_mb = sum(entry.size_bytes for entry in self.entries.values()) / 1024 / 1024
        
        if current_size_mb <= self.max_size_mb:
            return
        
        # Calculate how many entries to remove (remove 20% more than needed for buffer)
        target_size_mb = self.max_size_mb * 0.8
        size_to_remove_mb = current_size_mb - target_size_mb
        
        # Use intelligent eviction policy if available
        if hasattr(self, 'eviction_policy'):
            # Estimate number of entries to remove
            avg_entry_size_mb = current_size_mb / len(self.entries) if self.entries else 1
            estimated_entries_to_remove = int(size_to_remove_mb / avg_entry_size_mb) + 1
            
            # Get eviction candidates using intelligent policy
            candidates = self.eviction_policy.select_eviction_candidates(
                self.entries, 
                estimated_entries_to_remove,
                self.usage_predictor if self.enable_prediction else None
            )
        else:
            # Fallback to LRU
            sorted_entries = sorted(
                self.entries.items(),
                key=lambda x: x[1].last_accessed
            )
            candidates = [key for key, _ in sorted_entries]
        
        # Remove entries until under size limit
        removed_count = 0
        for key in candidates:
            if key not in self.entries:
                continue
                
            current_size_mb -= self.entries[key].size_bytes / 1024 / 1024
            self._remove_entry(key)
            self.metrics.eviction_count += 1
            removed_count += 1
            
            if current_size_mb <= target_size_mb:
                break
        
        if removed_count > 0:
            logger.info(f"Intelligently evicted {removed_count} cache entries to enforce size limit")
    
    def get_cached_results(self, data_hash: str) -> Optional[Any]:
        """
        Retrieve cached results by data hash.
        
        Args:
            data_hash: Hash of the input data
            
        Returns:
            Cached results if found and valid, None otherwise
        """
        with self._lock:
            # Check if entry exists
            if data_hash not in self.entries:
                self.metrics.miss_count += 1
                return None
            
            entry = self.entries[data_hash]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(data_hash)
                self.metrics.miss_count += 1
                return None
            
            try:
                # Load data from disk
                cache_file = self._get_cache_file_path(data_hash, entry.compressed)
                
                if not cache_file.exists():
                    # File missing, remove entry
                    self._remove_entry(data_hash)
                    self.metrics.miss_count += 1
                    return None
                
                with open(cache_file, 'rb') as f:
                    data = f.read()
                
                result = self._deserialize_data(data, entry.compressed)
                
                # Update access statistics
                entry.update_access()
                self.metrics.hit_count += 1
                
                logger.info(f"Cache hit for hash: {data_hash[:8]}...")
                return result
                
            except Exception as e:
                logger.error(f"Error retrieving cached data: {e}")
                self._remove_entry(data_hash)
                self.metrics.miss_count += 1
                return None
    
    def cache_results(
        self,
        data_hash: str,
        results: Any,
        ttl_hours: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache results with the given data hash.
        
        Args:
            data_hash: Hash of the input data
            results: Results to cache
            ttl_hours: Time to live in hours (uses default if None)
            metadata: Additional metadata for the cache entry
            
        Returns:
            True if successfully cached, False otherwise
        """
        with self._lock:
            try:
                ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
                
                # Serialize data
                serialized_data = self._serialize_data(results, self.compression_enabled)
                size_bytes = len(serialized_data)
                
                # Extract features for similarity detection
                features = None
                if hasattr(self, 'similarity_detector'):
                    try:
                        # We need the original data to extract features, but we only have the hash
                        # For now, store basic metadata that can help with similarity
                        enhanced_metadata = metadata or {}
                        enhanced_metadata.update({
                            'size_bytes': size_bytes,
                            'timestamp': datetime.now().isoformat(),
                            'compressed': self.compression_enabled
                        })
                    except Exception as e:
                        logger.error(f"Error extracting features: {e}")
                        enhanced_metadata = metadata or {}
                else:
                    enhanced_metadata = metadata or {}
                
                # Create cache entry
                entry = CacheEntry(
                    key=data_hash,
                    data_hash=data_hash,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=0,
                    size_bytes=size_bytes,
                    ttl_hours=ttl,
                    compressed=self.compression_enabled,
                    metadata=enhanced_metadata
                )
                
                # Save to disk
                cache_file = self._get_cache_file_path(data_hash, self.compression_enabled)
                with open(cache_file, 'wb') as f:
                    f.write(serialized_data)
                
                # Update memory structures
                if data_hash in self.entries:
                    # Update existing entry
                    old_entry = self.entries[data_hash]
                    self.metrics.total_size_mb -= old_entry.size_bytes / 1024 / 1024
                else:
                    # New entry
                    self.metrics.total_entries += 1
                
                self.entries[data_hash] = entry
                self.metrics.total_size_mb += size_bytes / 1024 / 1024
                
                # Enforce size limit
                self._enforce_size_limit()
                
                # Save metadata
                self._save_metadata()
                
                logger.info(f"Cached results for hash: {data_hash[:8]}... (Size: {size_bytes/1024:.1f}KB)")
                return True
                
            except Exception as e:
                logger.error(f"Error caching results: {e}")
                return False
    
    def get_or_compute(
        self,
        data: Any,
        compute_func: callable,
        ttl_hours: Optional[int] = None,
        force_recompute: bool = False,
        use_similarity: bool = True,
        **compute_kwargs
    ) -> Any:
        """
        Get cached results or compute and cache them with intelligent features.
        
        Args:
            data: Input data to hash
            compute_func: Function to compute results if not cached
            ttl_hours: Time to live for cached results
            force_recompute: Force recomputation even if cached
            use_similarity: Whether to use similarity detection for cache reuse
            **compute_kwargs: Additional arguments for compute function
            
        Returns:
            Computed or cached results
        """
        start_time = time.time()
        
        # Calculate data hash
        data_hash = self._calculate_data_hash(data)
        
        # Check exact cache match if not forcing recompute
        if not force_recompute:
            cached_result = self.get_cached_results(data_hash)
            if cached_result is not None:
                self.record_cache_usage(data, was_hit=True)
                return cached_result
        
        # Try similarity-based cache reuse if enabled
        if use_similarity and not force_recompute and hasattr(self, 'similarity_detector'):
            similar_result = self.find_similar_cached_data(data)
            if similar_result is not None:
                self.record_cache_usage(data, was_hit=True)
                return similar_result
        
        # Compute results
        try:
            logger.info(f"Computing results for hash: {data_hash[:8]}...")
            results = compute_func(data, **compute_kwargs)
            
            processing_time = time.time() - start_time
            
            # Predict if caching would be beneficial
            should_cache = True
            if hasattr(self, 'usage_predictor') and self.enable_prediction:
                cache_probability = self.predict_cache_usage({
                    'data': data,
                    'size': len(str(data)) if data else 0,
                    'processing_time': processing_time,
                    'complexity': 'high' if processing_time > 5 else 'medium' if processing_time > 1 else 'low'
                })
                
                # Only cache if probability is above threshold
                should_cache = cache_probability > 0.3
                logger.info(f"Cache probability: {cache_probability:.2f}, will cache: {should_cache}")
            
            # Cache the results if beneficial
            if should_cache:
                # Extract features for similarity detection
                metadata = {}
                if hasattr(self, 'similarity_detector'):
                    try:
                        features = self.similarity_detector.extract_features(data)
                        metadata['features'] = features.tolist()
                    except Exception as e:
                        logger.error(f"Error extracting features for caching: {e}")
                
                metadata.update({
                    'processing_time': processing_time,
                    'data_type': type(data).__name__,
                    'computed_at': datetime.now().isoformat()
                })
                
                self.cache_results(data_hash, results, ttl_hours, metadata)
            
            # Record usage for model training
            self.record_cache_usage(data, was_hit=False, processing_time=processing_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing results: {e}")
            raise
    
    def invalidate_cache(self, data_hash: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            data_hash: Specific hash to invalidate (None for all)
        """
        with self._lock:
            if data_hash:
                # Invalidate specific entry
                if data_hash in self.entries:
                    self._remove_entry(data_hash)
                    logger.info(f"Invalidated cache entry: {data_hash[:8]}...")
            else:
                # Invalidate all entries
                keys_to_remove = list(self.entries.keys())
                for key in keys_to_remove:
                    self._remove_entry(key)
                logger.info(f"Invalidated all {len(keys_to_remove)} cache entries")
            
            self._save_metadata()
    
    def cleanup_cache(self, force: bool = False):
        """
        Perform cache cleanup operations.
        
        Args:
            force: Force cleanup even if auto_cleanup is disabled
        """
        if not self.auto_cleanup and not force:
            return
        
        with self._lock:
            initial_count = len(self.entries)
            
            # Remove expired entries
            self._cleanup_expired_entries()
            
            # Enforce size limit
            self._enforce_size_limit()
            
            final_count = len(self.entries)
            removed_count = initial_count - final_count
            
            if removed_count > 0:
                logger.info(f"Cache cleanup completed: removed {removed_count} entries")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            current_size_mb = sum(entry.size_bytes for entry in self.entries.values()) / 1024 / 1024
            
            # Calculate age statistics
            now = datetime.now()
            ages = [(now - entry.created_at).total_seconds() / 3600 for entry in self.entries.values()]
            
            stats = {
                "total_entries": len(self.entries),
                "current_size_mb": current_size_mb,
                "max_size_mb": self.max_size_mb,
                "size_utilization_percent": (current_size_mb / self.max_size_mb * 100) if self.max_size_mb > 0 else 0,
                "hit_count": self.metrics.hit_count,
                "miss_count": self.metrics.miss_count,
                "hit_rate_percent": self.metrics.hit_rate,
                "eviction_count": self.metrics.eviction_count,
                "cleanup_count": self.metrics.cleanup_count,
                "average_entry_size_kb": (current_size_mb * 1024 / len(self.entries)) if self.entries else 0,
                "oldest_entry_hours": max(ages) if ages else 0,
                "newest_entry_hours": min(ages) if ages else 0,
                "compression_enabled": self.compression_enabled,
                "auto_cleanup_enabled": self.auto_cleanup
            }
            
            return stats
    
    def predict_cache_usage(self, data_characteristics: Dict[str, Any]) -> float:
        """
        Predict cache usage probability based on data characteristics.
        
        Args:
            data_characteristics: Dictionary with data size, type, etc.
            
        Returns:
            Probability (0-1) that this data will benefit from caching
        """
        try:
            if self.enable_prediction and self.usage_predictor.is_trained:
                # Use ML-based prediction
                return self.usage_predictor.predict_usefulness(
                    data_characteristics.get('data'),
                    context=data_characteristics
                )
            else:
                # Fallback to heuristic-based prediction
                score = 0.0
                
                # Data size factor (larger data benefits more from caching)
                data_size = data_characteristics.get('size', 0)
                if data_size > 10000:
                    score += 0.3
                elif data_size > 1000:
                    score += 0.2
                elif data_size > 100:
                    score += 0.1
                
                # Processing complexity factor
                complexity = data_characteristics.get('complexity', 'low')
                if complexity == 'high':
                    score += 0.4
                elif complexity == 'medium':
                    score += 0.2
                
                # Reuse likelihood factor
                reuse_likelihood = data_characteristics.get('reuse_likelihood', 'low')
                if reuse_likelihood == 'high':
                    score += 0.3
                elif reuse_likelihood == 'medium':
                    score += 0.2
                
                # Historical hit rate factor
                if self.metrics.hit_rate > 50:
                    score += 0.1
                
                return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error predicting cache usage: {e}")
            return 0.5  # Default moderate probability
    
    def find_similar_cached_data(self, data: Any, similarity_threshold: float = 0.8) -> Optional[Any]:
        """
        Find similar cached data that can be reused.
        
        Args:
            data: Input data to find similar cached results for
            similarity_threshold: Minimum similarity score required
            
        Returns:
            Cached results if similar data found, None otherwise
        """
        try:
            if not self.entries:
                return None
            
            # Find similar entries
            similar_entries = self.similarity_detector.find_similar_entries(
                data, self.entries, similarity_threshold
            )
            
            if not similar_entries:
                return None
            
            # Get the most similar entry that's not expired
            for entry_key, similarity_score in similar_entries:
                if entry_key in self.entries:
                    entry = self.entries[entry_key]
                    
                    if not entry.is_expired():
                        # Load and return the cached data
                        cached_data = self.get_cached_results(entry_key)
                        if cached_data is not None:
                            # Update similarity score for future eviction decisions
                            entry.similarity_score = similarity_score
                            logger.info(f"Found similar cached data with {similarity_score:.2f} similarity")
                            return cached_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar cached data: {e}")
            return None
    
    def _start_preloading_thread(self):
        """Start background thread for cache preloading."""
        def preload_worker():
            while True:
                try:
                    time.sleep(300)  # Check every 5 minutes
                    self._perform_preloading()
                except Exception as e:
                    logger.error(f"Error in preloading thread: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        preload_thread = threading.Thread(target=preload_worker, daemon=True)
        preload_thread.start()
        logger.info("Cache preloading thread started")
    
    def _perform_preloading(self):
        """Perform cache preloading based on usage patterns."""
        try:
            if not self.enable_preloading:
                return
            
            # Get preload candidates
            candidates = self.preloader.get_preload_candidates()
            
            if not candidates:
                return
            
            logger.info(f"Found {len(candidates)} preload candidates")
            
            # For now, just log the candidates (actual preloading would require
            # access to the data generation functions)
            for candidate in candidates[:3]:  # Log top 3
                logger.info(f"Preload candidate: {candidate['pattern']} (priority: {candidate['priority']:.2f})")
            
        except Exception as e:
            logger.error(f"Error performing preloading: {e}")
    
    def record_cache_usage(self, data: Any, was_hit: bool, processing_time: float = 0.0):
        """
        Record cache usage for training the prediction model.
        
        Args:
            data: The data that was accessed
            was_hit: Whether it was a cache hit
            processing_time: Time taken to process (if cache miss)
        """
        try:
            if not self.enable_prediction:
                return
            
            # Extract data characteristics
            data_characteristics = {
                'data': data,
                'size': len(str(data)) if data else 0,
                'type': type(data).__name__,
                'complexity': 'medium',  # Default complexity
                'processing_time': processing_time
            }
            
            # Record access pattern for preloading
            if self.enable_preloading:
                self.preloader.record_access_pattern(data_characteristics)
            
            # Store training data for model improvement
            if not hasattr(self, '_training_data'):
                self._training_data = []
            
            training_sample = {
                'data': data,
                'was_useful': was_hit,
                'context': {
                    'processing_time': processing_time,
                    'cache_hit_rate': self.metrics.hit_rate
                }
            }
            
            self._training_data.append(training_sample)
            
            # Retrain model periodically
            if len(self._training_data) >= 100 and len(self._training_data) % 50 == 0:
                self._retrain_prediction_model()
            
        except Exception as e:
            logger.error(f"Error recording cache usage: {e}")
    
    def _retrain_prediction_model(self):
        """Retrain the cache usage prediction model with new data."""
        try:
            if not self.enable_prediction or not hasattr(self, '_training_data'):
                return
            
            logger.info(f"Retraining cache prediction model with {len(self._training_data)} samples")
            
            # Train the model
            self.usage_predictor.train(self._training_data)
            
            # Save the updated model
            self.usage_predictor.save_model(str(self.model_file))
            
            # Keep only recent training data to prevent memory bloat
            if len(self._training_data) > 500:
                self._training_data = self._training_data[-300:]
            
        except Exception as e:
            logger.error(f"Error retraining prediction model: {e}")
    
    def get_cache_recommendations(self) -> Dict[str, Any]:
        """Get intelligent recommendations for cache optimization."""
        try:
            recommendations = {
                'actions': [],
                'insights': [],
                'performance_tips': []
            }
            
            # Analyze cache performance
            stats = self.get_cache_statistics()
            
            # Hit rate recommendations
            if stats['hit_rate_percent'] < 30:
                recommendations['actions'].append("Consider increasing cache size or TTL")
                recommendations['insights'].append("Low hit rate indicates cache is not being utilized effectively")
            elif stats['hit_rate_percent'] > 80:
                recommendations['insights'].append("Excellent cache performance - high hit rate")
            
            # Size utilization recommendations
            if stats['size_utilization_percent'] > 90:
                recommendations['actions'].append("Cache is nearly full - consider increasing max size")
            elif stats['size_utilization_percent'] < 20:
                recommendations['actions'].append("Cache is underutilized - could reduce max size")
            
            # Eviction recommendations
            if stats['eviction_count'] > stats['total_entries']:
                recommendations['actions'].append("High eviction rate - consider optimizing eviction policy")
            
            # Prediction model recommendations
            if self.enable_prediction and self.usage_predictor.is_trained:
                recommendations['insights'].append("ML-based cache prediction is active")
            elif self.enable_prediction:
                recommendations['actions'].append("Collect more usage data to train prediction model")
            
            # Performance tips
            if stats['average_entry_size_kb'] > 1000:
                recommendations['performance_tips'].append("Large cache entries detected - consider compression")
            
            if self.compression_enabled:
                recommendations['performance_tips'].append("Compression is enabled for space efficiency")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating cache recommendations: {e}")
            return {'actions': [], 'insights': [], 'performance_tips': []}


# Convenience functions
def create_intelligent_cache(
    cache_dir: str = ".cache/ai_ml",
    max_size_mb: int = 500,
    ttl_hours: int = 24
) -> IntelligentCache:
    """Create a new intelligent cache instance."""
    return IntelligentCache(
        cache_dir=cache_dir,
        max_size_mb=max_size_mb,
        default_ttl_hours=ttl_hours
    )


def hash_data(data: Any) -> str:
    """Create hash for any data type (convenience function)."""
    return DataHasher.hash_mixed_data(data)