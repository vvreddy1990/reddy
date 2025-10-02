"""
Enhanced error handling and user feedback system
"""

import traceback
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Enhanced error handling and user feedback system."""
    
    def __init__(self):
        self.error_log = []
        self.warning_log = []
        self.info_log = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_error(self, error_type: str, message: str, details: Optional[Dict] = None, 
                  file_context: Optional[str] = None, recoverable: bool = False) -> Dict[str, Any]:
        """Log an error with detailed context."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'ERROR',
            'error_type': error_type,
            'message': message,
            'details': details or {},
            'file_context': file_context,
            'recoverable': recoverable,
            'traceback': traceback.format_exc() if logger.isEnabledFor(logging.DEBUG) else None
        }
        
        self.error_log.append(error_entry)
        logger.error(f"[{error_type}] {message}")
        
        return error_entry
    
    def log_warning(self, warning_type: str, message: str, details: Optional[Dict] = None,
                   file_context: Optional[str] = None, suggestion: Optional[str] = None) -> Dict[str, Any]:
        """Log a warning with context and suggestions."""
        warning_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'WARNING',
            'warning_type': warning_type,
            'message': message,
            'details': details or {},
            'file_context': file_context,
            'suggestion': suggestion
        }
        
        self.warning_log.append(warning_entry)
        logger.warning(f"[{warning_type}] {message}")
        
        return warning_entry
    
    def log_info(self, info_type: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
        """Log informational messages."""
        info_entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'type': 'INFO',
            'info_type': info_type,
            'message': message,
            'details': details or {}
        }
        
        self.info_log.append(info_entry)
        logger.info(f"[{info_type}] {message}")
        
        return info_entry
    
    def handle_file_error(self, error: Exception, filename: str, operation: str) -> Dict[str, Any]:
        """Handle file-related errors with specific recovery suggestions."""
        error_type = type(error).__name__
        
        if "Permission denied" in str(error):
            return self.log_error(
                "FILE_PERMISSION_ERROR",
                f"Cannot access file '{filename}': Permission denied",
                {'filename': filename, 'operation': operation},
                file_context=filename,
                recoverable=True
            )
        
        elif "No such file" in str(error) or "FileNotFoundError" in error_type:
            return self.log_error(
                "FILE_NOT_FOUND_ERROR",
                f"File '{filename}' not found",
                {'filename': filename, 'operation': operation},
                file_context=filename,
                recoverable=True
            )
        
        elif "Invalid file format" in str(error) or "BadZipFile" in error_type:
            return self.log_error(
                "INVALID_FILE_FORMAT",
                f"File '{filename}' is not a valid Excel file",
                {'filename': filename, 'operation': operation},
                file_context=filename,
                recoverable=True
            )
        
        elif "EmptyDataError" in error_type:
            return self.log_error(
                "EMPTY_FILE_ERROR",
                f"File '{filename}' is empty or has no data",
                {'filename': filename, 'operation': operation},
                file_context=filename,
                recoverable=True
            )
        
        else:
            return self.log_error(
                "UNKNOWN_FILE_ERROR",
                f"Unexpected error processing file '{filename}': {str(error)}",
                {'filename': filename, 'operation': operation, 'error_type': error_type},
                file_context=filename,
                recoverable=False
            )
    
    def handle_data_error(self, error: Exception, column: str, operation: str) -> Dict[str, Any]:
        """Handle data processing errors."""
        error_type = type(error).__name__
        
        if "KeyError" in error_type:
            return self.log_error(
                "MISSING_COLUMN_ERROR",
                f"Required column '{column}' not found in data",
                {'column': column, 'operation': operation},
                recoverable=True
            )
        
        elif "ValueError" in error_type and "could not convert" in str(error):
            return self.log_error(
                "DATA_CONVERSION_ERROR",
                f"Cannot convert data in column '{column}' to expected format",
                {'column': column, 'operation': operation},
                recoverable=True
            )
        
        elif "TypeError" in error_type:
            return self.log_error(
                "DATA_TYPE_ERROR",
                f"Unexpected data type in column '{column}'",
                {'column': column, 'operation': operation},
                recoverable=True
            )
        
        else:
            return self.log_error(
                "UNKNOWN_DATA_ERROR",
                f"Unexpected error processing column '{column}': {str(error)}",
                {'column': column, 'operation': operation, 'error_type': error_type},
                recoverable=False
            )
    
    def handle_merge_error(self, error: Exception, files: List[str], operation: str) -> Dict[str, Any]:
        """Handle merge operation errors."""
        error_type = type(error).__name__
        
        if "MemoryError" in error_type:
            return self.log_error(
                "MEMORY_ERROR",
                f"Insufficient memory to merge {len(files)} files",
                {'file_count': len(files), 'operation': operation},
                recoverable=True
            )
        
        elif "ValueError" in error_type and "columns" in str(error):
            return self.log_error(
                "COLUMN_MISMATCH_ERROR",
                f"Column mismatch between files during merge",
                {'file_count': len(files), 'operation': operation},
                recoverable=True
            )
        
        else:
            return self.log_error(
                "UNKNOWN_MERGE_ERROR",
                f"Unexpected error during merge operation: {str(error)}",
                {'file_count': len(files), 'operation': operation, 'error_type': error_type},
                recoverable=False
            )
    
    def get_error_suggestions(self, error_entry: Dict[str, Any]) -> List[str]:
        """Get recovery suggestions for an error."""
        suggestions = []
        
        if error_entry['error_type'] == 'FILE_PERMISSION_ERROR':
            suggestions.extend([
                "Check if the file is open in another application",
                "Verify you have read permissions for the file",
                "Try copying the file to a different location"
            ])
        
        elif error_entry['error_type'] == 'FILE_NOT_FOUND_ERROR':
            suggestions.extend([
                "Verify the file path is correct",
                "Check if the file was moved or deleted",
                "Ensure the file exists in the specified location"
            ])
        
        elif error_entry['error_type'] == 'INVALID_FILE_FORMAT':
            suggestions.extend([
                "Ensure the file is a valid Excel file (.xlsx or .xls)",
                "Try opening the file in Excel and saving it again",
                "Check if the file is corrupted"
            ])
        
        elif error_entry['error_type'] == 'EMPTY_FILE_ERROR':
            suggestions.extend([
                "Check if the file contains data",
                "Verify the file has the expected worksheets",
                "Ensure the data is not in hidden rows/columns"
            ])
        
        elif error_entry['error_type'] == 'MISSING_COLUMN_ERROR':
            suggestions.extend([
                "Check if the column name is spelled correctly",
                "Verify the column exists in all files",
                "Use column mapping to specify correct column names"
            ])
        
        elif error_entry['error_type'] == 'DATA_CONVERSION_ERROR':
            suggestions.extend([
                "Check for non-numeric characters in numeric columns",
                "Verify date formats are consistent",
                "Remove special characters or currency symbols"
            ])
        
        elif error_entry['error_type'] == 'MEMORY_ERROR':
            suggestions.extend([
                "Try processing fewer files at once",
                "Close other applications to free up memory",
                "Consider using smaller file chunks"
            ])
        
        elif error_entry['error_type'] == 'COLUMN_MISMATCH_ERROR':
            suggestions.extend([
                "Ensure all files have the same column structure",
                "Use column mapping to standardize column names",
                "Check for extra spaces or special characters in column names"
            ])
        
        return suggestions
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of all logged messages for the session."""
        return {
            'session_id': self.session_id,
            'total_errors': len(self.error_log),
            'total_warnings': len(self.warning_log),
            'total_info': len(self.info_log),
            'recoverable_errors': len([e for e in self.error_log if e.get('recoverable', False)]),
            'critical_errors': len([e for e in self.error_log if not e.get('recoverable', False)]),
            'session_start': self.session_id,
            'session_end': datetime.now().isoformat()
        }
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate a comprehensive error report."""
        return {
            'summary': self.get_session_summary(),
            'errors': self.error_log,
            'warnings': self.warning_log,
            'info': self.info_log,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on logged issues."""
        recommendations = []
        
        error_types = [e['error_type'] for e in self.error_log]
        warning_types = [w['warning_type'] for w in self.warning_log]
        
        if 'FILE_PERMISSION_ERROR' in error_types:
            recommendations.append("Review file permissions and ensure files are not locked by other applications")
        
        if 'INVALID_FILE_FORMAT' in error_types:
            recommendations.append("Validate file formats before processing and convert to standard Excel format")
        
        if 'MISSING_COLUMN_ERROR' in error_types:
            recommendations.append("Implement column mapping validation to ensure required columns are present")
        
        if 'DATA_CONVERSION_ERROR' in error_types:
            recommendations.append("Add data cleaning and validation steps before processing")
        
        if 'MEMORY_ERROR' in error_types:
            recommendations.append("Implement batch processing for large files to manage memory usage")
        
        if len(self.error_log) > 5:
            recommendations.append("Consider implementing more robust error handling and data validation")
        
        if len(self.warning_log) > 10:
            recommendations.append("Review data quality and implement data cleaning procedures")
        
        return recommendations
    
    def clear_logs(self):
        """Clear all logged messages."""
        self.error_log.clear()
        self.warning_log.clear()
        self.info_log.clear()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

