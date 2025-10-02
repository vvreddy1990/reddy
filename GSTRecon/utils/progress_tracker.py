"""
Progress indicators and status tracking for merge operations
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Progress tracking for merge operations."""
    
    def __init__(self):
        self.current_operation = None
        self.progress = 0
        self.total_steps = 0
        self.current_step = 0
        self.status_message = ""
        self.start_time = None
        self.end_time = None
        self.estimated_completion = None
        self.is_running = False
        self.callbacks = []
        self.operation_history = []
    
    def start_operation(self, operation_name: str, total_steps: int, 
                       status_message: str = "Starting operation...") -> None:
        """Start tracking a new operation."""
        self.current_operation = operation_name
        self.total_steps = total_steps
        self.current_step = 0
        self.progress = 0
        self.status_message = status_message
        self.start_time = datetime.now()
        self.end_time = None
        self.estimated_completion = None
        self.is_running = True
        
        # Add to history
        self.operation_history.append({
            'operation': operation_name,
            'start_time': self.start_time.isoformat(),
            'total_steps': total_steps,
            'status': 'started'
        })
        
        logger.info(f"Started operation: {operation_name} ({total_steps} steps)")
        self._notify_callbacks()
    
    def update_progress(self, step: int, status_message: str = None) -> None:
        """Update progress for current operation."""
        if not self.is_running:
            return
        
        self.current_step = step
        self.progress = (step / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        if status_message:
            self.status_message = status_message
        
        # Calculate estimated completion time
        if self.start_time and step > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if step > 0:
                estimated_total = elapsed * (self.total_steps / step)
                remaining = estimated_total - elapsed
                self.estimated_completion = datetime.now().timestamp() + remaining
        
        logger.info(f"Progress: {self.progress:.1f}% - {self.status_message}")
        self._notify_callbacks()
    
    def complete_operation(self, status_message: str = "Operation completed") -> None:
        """Mark current operation as completed."""
        if not self.is_running:
            return
        
        self.progress = 100
        self.current_step = self.total_steps
        self.status_message = status_message
        self.end_time = datetime.now()
        self.is_running = False
        
        # Update history
        if self.operation_history:
            self.operation_history[-1].update({
                'end_time': self.end_time.isoformat(),
                'status': 'completed',
                'duration': (self.end_time - self.start_time).total_seconds()
            })
        
        logger.info(f"Completed operation: {self.current_operation}")
        self._notify_callbacks()
    
    def fail_operation(self, error_message: str) -> None:
        """Mark current operation as failed."""
        if not self.is_running:
            return
        
        self.status_message = f"Operation failed: {error_message}"
        self.end_time = datetime.now()
        self.is_running = False
        
        # Update history
        if self.operation_history:
            self.operation_history[-1].update({
                'end_time': self.end_time.isoformat(),
                'status': 'failed',
                'error': error_message,
                'duration': (self.end_time - self.start_time).total_seconds()
            })
        
        logger.error(f"Failed operation: {self.current_operation} - {error_message}")
        self._notify_callbacks()
    
    def add_callback(self, callback: Callable) -> None:
        """Add a callback function to be called on progress updates."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove a callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(self.get_status())
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status information."""
        status = {
            'operation': self.current_operation,
            'progress': round(self.progress, 1),
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'status_message': self.status_message,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'estimated_completion': self.estimated_completion,
            'elapsed_time': None,
            'remaining_time': None
        }
        
        # Calculate elapsed and remaining time
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            status['elapsed_time'] = round(elapsed, 1)
            
            if self.estimated_completion and self.is_running:
                remaining = self.estimated_completion - datetime.now().timestamp()
                status['remaining_time'] = round(max(0, remaining), 1)
        
        return status
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get history of all operations."""
        return self.operation_history.copy()
    
    def reset(self) -> None:
        """Reset the progress tracker."""
        self.current_operation = None
        self.progress = 0
        self.total_steps = 0
        self.current_step = 0
        self.status_message = ""
        self.start_time = None
        self.end_time = None
        self.estimated_completion = None
        self.is_running = False
        self.operation_history.clear()
        logger.info("Progress tracker reset")

class MergeProgressTracker(ProgressTracker):
    """Specialized progress tracker for merge operations."""
    
    def __init__(self):
        super().__init__()
        self.file_count = 0
        self.current_file = 0
        self.validation_progress = 0
        self.analysis_progress = 0
        self.merge_progress = 0
        self.post_processing_progress = 0
    
    def start_merge_operation(self, file_count: int) -> None:
        """Start tracking a merge operation."""
        self.file_count = file_count
        total_steps = file_count * 4 + 2  # 4 steps per file + validation + final processing
        
        self.start_operation(
            "Merge Ledger Files",
            total_steps,
            f"Starting merge of {file_count} files..."
        )
    
    def start_file_validation(self, file_index: int, filename: str) -> None:
        """Start validating a file."""
        self.current_file = file_index
        self.validation_progress = 0
        
        self.update_progress(
            file_index * 4,
            f"Validating file {file_index + 1}/{self.file_count}: {filename}"
        )
    
    def complete_file_validation(self, file_index: int, filename: str) -> None:
        """Complete file validation."""
        self.validation_progress = 100
        
        self.update_progress(
            file_index * 4 + 1,
            f"Completed validation of {filename}"
        )
    
    def start_file_analysis(self, file_index: int, filename: str) -> None:
        """Start analyzing a file."""
        self.analysis_progress = 0
        
        self.update_progress(
            file_index * 4 + 1,
            f"Analyzing file {file_index + 1}/{self.file_count}: {filename}"
        )
    
    def complete_file_analysis(self, file_index: int, filename: str) -> None:
        """Complete file analysis."""
        self.analysis_progress = 100
        
        self.update_progress(
            file_index * 4 + 2,
            f"Completed analysis of {filename}"
        )
    
    def start_file_merge(self, file_index: int, filename: str) -> None:
        """Start merging a file."""
        self.merge_progress = 0
        
        self.update_progress(
            file_index * 4 + 2,
            f"Merging file {file_index + 1}/{self.file_count}: {filename}"
        )
    
    def complete_file_merge(self, file_index: int, filename: str) -> None:
        """Complete file merge."""
        self.merge_progress = 100
        
        self.update_progress(
            file_index * 4 + 3,
            f"Completed merge of {filename}"
        )
    
    def start_post_processing(self) -> None:
        """Start post-processing."""
        self.post_processing_progress = 0
        
        self.update_progress(
            self.file_count * 4,
            "Applying post-processing (sign conversion, duplicates, etc.)"
        )
    
    def update_post_processing(self, progress: int, message: str) -> None:
        """Update post-processing progress."""
        self.post_processing_progress = progress
        
        self.update_progress(
            self.file_count * 4 + int(progress / 100),
            message
        )
    
    def complete_post_processing(self) -> None:
        """Complete post-processing."""
        self.post_processing_progress = 100
        
        self.update_progress(
            self.file_count * 4 + 1,
            "Post-processing completed"
        )
    
    def complete_merge_operation(self, total_records: int) -> None:
        """Complete the entire merge operation."""
        self.complete_operation(
            f"Successfully merged {self.file_count} files into {total_records} records"
        )
    
    def fail_merge_operation(self, error_message: str, file_index: int = None) -> None:
        """Fail the merge operation."""
        if file_index is not None:
            error_message = f"Failed at file {file_index + 1}/{self.file_count}: {error_message}"
        
        self.fail_operation(error_message)
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status including sub-progress."""
        status = self.get_status()
        
        status.update({
            'file_count': self.file_count,
            'current_file': self.current_file,
            'validation_progress': self.validation_progress,
            'analysis_progress': self.analysis_progress,
            'merge_progress': self.merge_progress,
            'post_processing_progress': self.post_processing_progress
        })
        
        return status

class ProgressIndicator:
    """Simple progress indicator for console output."""
    
    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, current: int, message: str = "") -> None:
        """Update progress indicator."""
        self.current = current
        progress = current / self.total
        filled = int(self.width * progress)
        bar = "=" * filled + "-" * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if current > 0:
            eta = elapsed * (self.total - current) / current
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        print(f"\r[{bar}] {progress*100:.1f}% ({current}/{self.total}) {eta_str} {message}", end="", flush=True)
    
    def complete(self, message: str = "Complete!") -> None:
        """Complete the progress indicator."""
        elapsed = time.time() - self.start_time
        print(f"\r[{'=' * self.width}] 100.0% ({self.total}/{self.total}) Time: {elapsed:.1f}s {message}")
    
    def fail(self, message: str = "Failed!") -> None:
        """Mark as failed."""
        elapsed = time.time() - self.start_time
        print(f"\r[{'=' * self.width}] FAILED Time: {elapsed:.1f}s {message}")
