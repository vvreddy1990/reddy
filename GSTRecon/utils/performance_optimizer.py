"""
Performance Optimization Module for AI/ML Operations

This module provides parallel processing, memory optimization, adaptive algorithm selection,
and performance profiling capabilities for the GST reconciliation system.
"""

import os
import gc
import psutil
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Callable, Optional, Tuple, Union
import time
import logging
from dataclasses import dataclass
from functools import wraps
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:
    """Performance profiling data for operations."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    data_size: int
    algorithm_used: str
    parallel_workers: int = 1
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class SystemResources:
    """Current system resource utilization."""
    cpu_percent: float
    memory_percent: float
    available_memory_mb: float
    cpu_count: int
    load_average: float = 0.0


class MemoryOptimizer:
    """Optimizes memory usage for large datasets and operations."""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.memory_threshold = 0.8  # Trigger optimization at 80% usage
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_available_memory(self) -> float:
        """Get available system memory in MB."""
        return psutil.virtual_memory().available / 1024 / 1024
    
    def should_optimize_memory(self) -> bool:
        """Check if memory optimization is needed."""
        current_usage = self.get_memory_usage()
        return current_usage > (self.max_memory_mb * self.memory_threshold)
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if df[col].dtype == 'int64':
                    if col_min >= -128 and col_max <= 127:
                        df[col] = df[col].astype('int8')
                    elif col_min >= -32768 and col_max <= 32767:
                        df[col] = df[col].astype('int16')
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        df[col] = df[col].astype('int32')
                
                elif df[col].dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize string columns
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].astype('category')
                    except:
                        pass  # Keep as object if conversion fails
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            reduction = ((original_memory - optimized_memory) / original_memory) * 100
            
            logger.info(f"DataFrame memory optimized: {original_memory:.1f}MB -> {optimized_memory:.1f}MB ({reduction:.1f}% reduction)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error optimizing DataFrame memory: {e}")
            return df
    
    def chunk_dataframe(self, df: pd.DataFrame, max_chunk_size: int = 10000) -> List[pd.DataFrame]:
        """Split DataFrame into memory-efficient chunks."""
        try:
            if len(df) <= max_chunk_size:
                return [df]
            
            chunks = []
            for i in range(0, len(df), max_chunk_size):
                chunk = df.iloc[i:i + max_chunk_size].copy()
                chunks.append(chunk)
            
            logger.info(f"DataFrame split into {len(chunks)} chunks of max size {max_chunk_size}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking DataFrame: {e}")
            return [df]
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        try:
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")


class ParallelProcessor:
    """Handles parallel processing for AI/ML operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.thread_pool = None
        self.process_pool = None
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up thread and process pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
    
    def parallel_map_threads(self, func: Callable, items: List[Any], max_workers: Optional[int] = None) -> List[Any]:
        """Execute function on items using thread pool."""
        try:
            workers = max_workers or self.max_workers
            
            if len(items) <= 1 or workers <= 1:
                return [func(item) for item in items]
            
            if not self.thread_pool:
                self.thread_pool = ThreadPoolExecutor(max_workers=workers)
            
            futures = [self.thread_pool.submit(func, item) for item in items]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel thread execution: {e}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel thread processing: {e}")
            return [func(item) for item in items]  # Fallback to sequential
    
    def parallel_map_processes(self, func: Callable, items: List[Any], max_workers: Optional[int] = None) -> List[Any]:
        """Execute function on items using process pool."""
        try:
            workers = max_workers or min(self.max_workers, multiprocessing.cpu_count())
            
            if len(items) <= 1 or workers <= 1:
                return [func(item) for item in items]
            
            # Use process pool for CPU-intensive tasks
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(func, item) for item in items]
                results = []
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=60)  # 60 second timeout
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in parallel process execution: {e}")
                        results.append(None)
                
                return results
            
        except Exception as e:
            logger.error(f"Error in parallel process processing: {e}")
            return [func(item) for item in items]  # Fallback to sequential
    
    def parallel_dataframe_operation(self, df: pd.DataFrame, func: Callable, chunk_size: int = 10000, use_processes: bool = False) -> pd.DataFrame:
        """Apply function to DataFrame chunks in parallel."""
        try:
            if len(df) <= chunk_size:
                return func(df)
            
            # Split DataFrame into chunks
            chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
            
            # Process chunks in parallel
            if use_processes:
                processed_chunks = self.parallel_map_processes(func, chunks)
            else:
                processed_chunks = self.parallel_map_threads(func, chunks)
            
            # Combine results
            valid_chunks = [chunk for chunk in processed_chunks if chunk is not None]
            
            if valid_chunks:
                result = pd.concat(valid_chunks, ignore_index=True)
                logger.info(f"Processed DataFrame in {len(chunks)} parallel chunks")
                return result
            else:
                logger.warning("No valid chunks returned from parallel processing")
                return func(df)  # Fallback to sequential
            
        except Exception as e:
            logger.error(f"Error in parallel DataFrame operation: {e}")
            return func(df)  # Fallback to sequential


class AdaptiveAlgorithmSelector:
    """Selects optimal algorithms based on data characteristics and system resources."""
    
    def __init__(self):
        self.performance_history = {}
        self.algorithm_profiles = {
            'fuzzy_matching': {
                'cpu_intensive': True,
                'memory_intensive': False,
                'optimal_data_size': (100, 10000),
                'parallel_friendly': True
            },
            'ml_similarity': {
                'cpu_intensive': True,
                'memory_intensive': True,
                'optimal_data_size': (1000, 100000),
                'parallel_friendly': False
            },
            'statistical_analysis': {
                'cpu_intensive': False,
                'memory_intensive': True,
                'optimal_data_size': (500, 50000),
                'parallel_friendly': True
            },
            'simple_matching': {
                'cpu_intensive': False,
                'memory_intensive': False,
                'optimal_data_size': (1, 1000),
                'parallel_friendly': True
            }
        }
    
    def get_system_resources(self) -> SystemResources:
        """Get current system resource utilization."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                available_memory_mb=memory.available / 1024 / 1024,
                cpu_count=psutil.cpu_count(),
                load_average=os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            )
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return SystemResources(
                cpu_percent=50.0,
                memory_percent=50.0,
                available_memory_mb=1024.0,
                cpu_count=4
            )
    
    def analyze_data_characteristics(self, data: Any) -> Dict[str, Any]:
        """Analyze data to determine optimal algorithm selection."""
        try:
            characteristics = {
                'size': 0,
                'complexity': 'low',
                'type': 'unknown',
                'memory_requirement': 'low'
            }
            
            if isinstance(data, pd.DataFrame):
                characteristics.update({
                    'size': len(data) * len(data.columns),
                    'type': 'dataframe',
                    'memory_requirement': 'high' if len(data) > 50000 else 'medium' if len(data) > 5000 else 'low',
                    'complexity': 'high' if len(data.columns) > 20 else 'medium' if len(data.columns) > 5 else 'low'
                })
            elif isinstance(data, (list, tuple)):
                characteristics.update({
                    'size': len(data),
                    'type': 'list',
                    'memory_requirement': 'high' if len(data) > 10000 else 'medium' if len(data) > 1000 else 'low',
                    'complexity': 'medium'
                })
            elif isinstance(data, dict):
                characteristics.update({
                    'size': len(data),
                    'type': 'dict',
                    'memory_requirement': 'medium' if len(data) > 1000 else 'low',
                    'complexity': 'low'
                })
            else:
                characteristics.update({
                    'size': len(str(data)) if data else 0,
                    'type': type(data).__name__,
                    'memory_requirement': 'low',
                    'complexity': 'low'
                })
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error analyzing data characteristics: {e}")
            return {'size': 0, 'complexity': 'low', 'type': 'unknown', 'memory_requirement': 'low'}
    
    def select_optimal_algorithm(self, operation_type: str, data: Any, available_algorithms: List[str]) -> str:
        """Select the optimal algorithm based on data and system characteristics."""
        try:
            data_chars = self.analyze_data_characteristics(data)
            system_resources = self.get_system_resources()
            
            # Score each available algorithm
            algorithm_scores = {}
            
            for algorithm in available_algorithms:
                if algorithm not in self.algorithm_profiles:
                    algorithm_scores[algorithm] = 0.5  # Default score
                    continue
                
                profile = self.algorithm_profiles[algorithm]
                score = 1.0
                
                # Data size compatibility
                optimal_min, optimal_max = profile['optimal_data_size']
                data_size = data_chars['size']
                
                if optimal_min <= data_size <= optimal_max:
                    score += 0.3
                elif data_size < optimal_min:
                    score -= 0.1
                else:  # data_size > optimal_max
                    score -= 0.2
                
                # System resource compatibility
                if profile['cpu_intensive'] and system_resources.cpu_percent > 80:
                    score -= 0.2
                
                if profile['memory_intensive'] and system_resources.memory_percent > 80:
                    score -= 0.3
                
                # Memory requirement vs available memory
                if (data_chars['memory_requirement'] == 'high' and 
                    system_resources.available_memory_mb < 500):
                    score -= 0.2
                
                # Historical performance
                history_key = f"{operation_type}_{algorithm}_{data_chars['type']}"
                if history_key in self.performance_history:
                    avg_performance = np.mean(self.performance_history[history_key])
                    # Normalize performance score (lower execution time = higher score)
                    perf_score = max(0, 1 - (avg_performance / 10))  # Assume 10s is poor performance
                    score += perf_score * 0.2
                
                algorithm_scores[algorithm] = max(0, score)
            
            # Select algorithm with highest score
            if algorithm_scores:
                best_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])[0]
                logger.info(f"Selected algorithm '{best_algorithm}' for {operation_type} (score: {algorithm_scores[best_algorithm]:.2f})")
                return best_algorithm
            else:
                return available_algorithms[0] if available_algorithms else 'default'
            
        except Exception as e:
            logger.error(f"Error selecting optimal algorithm: {e}")
            return available_algorithms[0] if available_algorithms else 'default'
    
    def record_performance(self, operation_type: str, algorithm: str, data_type: str, execution_time: float):
        """Record algorithm performance for future selection."""
        try:
            history_key = f"{operation_type}_{algorithm}_{data_type}"
            
            if history_key not in self.performance_history:
                self.performance_history[history_key] = []
            
            self.performance_history[history_key].append(execution_time)
            
            # Keep only last 20 measurements
            if len(self.performance_history[history_key]) > 20:
                self.performance_history[history_key] = self.performance_history[history_key][-20:]
            
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        try:
            recommendations = []
            system_resources = self.get_system_resources()
            
            # CPU recommendations
            if system_resources.cpu_percent > 90:
                recommendations.append("High CPU usage detected - consider reducing parallel workers")
            elif system_resources.cpu_percent < 30:
                recommendations.append("Low CPU usage - consider increasing parallelization")
            
            # Memory recommendations
            if system_resources.memory_percent > 85:
                recommendations.append("High memory usage - enable memory optimization and chunking")
            
            # Algorithm recommendations based on history
            if self.performance_history:
                slow_operations = []
                for key, times in self.performance_history.items():
                    avg_time = np.mean(times)
                    if avg_time > 5.0:  # Operations taking more than 5 seconds
                        slow_operations.append((key, avg_time))
                
                if slow_operations:
                    recommendations.append(f"Slow operations detected: {len(slow_operations)} operations averaging >5s")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating performance recommendations: {e}")
            return []


class PerformanceProfiler:
    """Profiles and monitors performance of operations."""
    
    def __init__(self, profile_dir: str = ".cache/performance"):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.profiles: List[PerformanceProfile] = []
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        
    def start_profiling(self, operation_name: str, data_size: int = 0, algorithm: str = "unknown") -> str:
        """Start profiling an operation."""
        try:
            profile_id = f"{operation_name}_{int(time.time() * 1000)}"
            
            self.active_profiles[profile_id] = {
                'operation_name': operation_name,
                'start_time': time.time(),
                'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,
                'start_cpu': psutil.cpu_percent(),
                'data_size': data_size,
                'algorithm': algorithm
            }
            
            return profile_id
            
        except Exception as e:
            logger.error(f"Error starting profiling: {e}")
            return ""
    
    def end_profiling(self, profile_id: str, parallel_workers: int = 1) -> Optional[PerformanceProfile]:
        """End profiling and create performance profile."""
        try:
            if profile_id not in self.active_profiles:
                logger.warning(f"Profile ID {profile_id} not found")
                return None
            
            profile_data = self.active_profiles[profile_id]
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            end_cpu = psutil.cpu_percent()
            
            profile = PerformanceProfile(
                operation_name=profile_data['operation_name'],
                execution_time=end_time - profile_data['start_time'],
                memory_usage_mb=end_memory - profile_data['start_memory'],
                cpu_usage_percent=(profile_data['start_cpu'] + end_cpu) / 2,
                data_size=profile_data['data_size'],
                algorithm_used=profile_data['algorithm'],
                parallel_workers=parallel_workers,
                timestamp=end_time
            )
            
            self.profiles.append(profile)
            del self.active_profiles[profile_id]
            
            # Save profile to disk periodically
            if len(self.profiles) % 10 == 0:
                self.save_profiles()
            
            return profile
            
        except Exception as e:
            logger.error(f"Error ending profiling: {e}")
            return None
    
    def profile_operation(self, operation_name: str, data_size: int = 0, algorithm: str = "unknown"):
        """Decorator for profiling operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                profile_id = self.start_profiling(operation_name, data_size, algorithm)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_profiling(profile_id)
            return wrapper
        return decorator
    
    def get_performance_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        try:
            relevant_profiles = self.profiles
            if operation_name:
                relevant_profiles = [p for p in self.profiles if p.operation_name == operation_name]
            
            if not relevant_profiles:
                return {}
            
            execution_times = [p.execution_time for p in relevant_profiles]
            memory_usages = [p.memory_usage_mb for p in relevant_profiles]
            
            summary = {
                'operation_count': len(relevant_profiles),
                'avg_execution_time': np.mean(execution_times),
                'min_execution_time': np.min(execution_times),
                'max_execution_time': np.max(execution_times),
                'avg_memory_usage': np.mean(memory_usages),
                'total_data_processed': sum(p.data_size for p in relevant_profiles),
                'algorithms_used': list(set(p.algorithm_used for p in relevant_profiles))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {}
    
    def save_profiles(self):
        """Save performance profiles to disk."""
        try:
            profile_file = self.profile_dir / "performance_profiles.json"
            
            profiles_data = []
            for profile in self.profiles:
                profiles_data.append({
                    'operation_name': profile.operation_name,
                    'execution_time': profile.execution_time,
                    'memory_usage_mb': profile.memory_usage_mb,
                    'cpu_usage_percent': profile.cpu_usage_percent,
                    'data_size': profile.data_size,
                    'algorithm_used': profile.algorithm_used,
                    'parallel_workers': profile.parallel_workers,
                    'timestamp': profile.timestamp
                })
            
            with open(profile_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            logger.info(f"Saved {len(profiles_data)} performance profiles")
            
        except Exception as e:
            logger.error(f"Error saving performance profiles: {e}")
    
    def load_profiles(self):
        """Load performance profiles from disk."""
        try:
            profile_file = self.profile_dir / "performance_profiles.json"
            
            if not profile_file.exists():
                return
            
            with open(profile_file, 'r') as f:
                profiles_data = json.load(f)
            
            self.profiles = []
            for data in profiles_data:
                profile = PerformanceProfile(
                    operation_name=data['operation_name'],
                    execution_time=data['execution_time'],
                    memory_usage_mb=data['memory_usage_mb'],
                    cpu_usage_percent=data['cpu_usage_percent'],
                    data_size=data['data_size'],
                    algorithm_used=data['algorithm_used'],
                    parallel_workers=data.get('parallel_workers', 1),
                    timestamp=data['timestamp']
                )
                self.profiles.append(profile)
            
            logger.info(f"Loaded {len(self.profiles)} performance profiles")
            
        except Exception as e:
            logger.error(f"Error loading performance profiles: {e}")


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, max_memory_mb: int = 1024, max_workers: Optional[int] = None):
        self.memory_optimizer = MemoryOptimizer(max_memory_mb)
        self.parallel_processor = ParallelProcessor(max_workers)
        self.algorithm_selector = AdaptiveAlgorithmSelector()
        self.profiler = PerformanceProfiler()
        
        # Load existing profiles
        self.profiler.load_profiles()
    
    def optimize_operation(self, operation_name: str, data: Any, operation_func: Callable, 
                          available_algorithms: List[str] = None, **kwargs) -> Any:
        """Optimize and execute an operation with all performance enhancements."""
        try:
            # Select optimal algorithm
            if available_algorithms:
                selected_algorithm = self.algorithm_selector.select_optimal_algorithm(
                    operation_name, data, available_algorithms
                )
                kwargs['algorithm'] = selected_algorithm
            else:
                selected_algorithm = kwargs.get('algorithm', 'default')
            
            # Optimize memory if needed
            if isinstance(data, pd.DataFrame) and self.memory_optimizer.should_optimize_memory():
                data = self.memory_optimizer.optimize_dataframe(data)
                self.memory_optimizer.force_garbage_collection()
            
            # Start profiling
            data_size = len(data) if hasattr(data, '__len__') else 0
            profile_id = self.profiler.start_profiling(operation_name, data_size, selected_algorithm)
            
            try:
                # Execute operation
                result = operation_func(data, **kwargs)
                
                # Record performance
                profile = self.profiler.end_profiling(profile_id)
                if profile:
                    self.algorithm_selector.record_performance(
                        operation_name, selected_algorithm, 
                        type(data).__name__, profile.execution_time
                    )
                
                return result
                
            except Exception as e:
                self.profiler.end_profiling(profile_id)
                raise e
            
        except Exception as e:
            logger.error(f"Error optimizing operation {operation_name}: {e}")
            # Fallback to direct execution
            return operation_func(data, **kwargs)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        try:
            report = {
                'performance_summary': self.profiler.get_performance_summary(),
                'memory_status': {
                    'current_usage_mb': self.memory_optimizer.get_memory_usage(),
                    'available_mb': self.memory_optimizer.get_available_memory(),
                    'optimization_needed': self.memory_optimizer.should_optimize_memory()
                },
                'system_resources': self.algorithm_selector.get_system_resources().__dict__,
                'recommendations': self.algorithm_selector.get_performance_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")
            return {}
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.parallel_processor.cleanup()
            self.profiler.save_profiles()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Convenience functions
def create_performance_optimizer(max_memory_mb: int = 1024, max_workers: Optional[int] = None) -> PerformanceOptimizer:
    """Create a new performance optimizer instance."""
    return PerformanceOptimizer(max_memory_mb, max_workers)


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage (convenience function)."""
    optimizer = MemoryOptimizer()
    return optimizer.optimize_dataframe(df)


def parallel_apply(func: Callable, items: List[Any], max_workers: Optional[int] = None, use_processes: bool = False) -> List[Any]:
    """Apply function to items in parallel (convenience function)."""
    with ParallelProcessor(max_workers) as processor:
        if use_processes:
            return processor.parallel_map_processes(func, items)
        else:
            return processor.parallel_map_threads(func, items)