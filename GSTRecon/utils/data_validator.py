"""
Enhanced data validation and quality checks for merge ledgers
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Enhanced data validation and quality checks."""
    
    def __init__(self):
        self.validation_results = {}
        self.quality_metrics = {}
        
    def validate_file(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Comprehensive file validation."""
        logger.info(f"Starting validation for file: {filename}")
        
        validation_results = {
            'filename': filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'validation_timestamp': datetime.now().isoformat(),
            'issues': [],
            'warnings': [],
            'quality_score': 0,
            'data_types': {},
            'missing_data': {},
            'gstin_validation': {},
            'date_validation': {},
            'numeric_validation': {},
            'duplicate_preview': {}
        }
        
        # 1. Data Type Validation
        validation_results['data_types'] = self._validate_data_types(df)
        
        # 2. Missing Data Analysis
        validation_results['missing_data'] = self._analyze_missing_data(df)
        
        # 3. GSTIN Format Validation
        validation_results['gstin_validation'] = self._validate_gstin_format(df)
        
        # 4. Date Validation
        validation_results['date_validation'] = self._validate_dates(df)
        
        # 5. Numeric Validation
        validation_results['numeric_validation'] = self._validate_numeric_data(df)
        
        # 6. Duplicate Detection Preview
        validation_results['duplicate_preview'] = self._detect_duplicates_preview(df)
        
        # 7. Calculate Quality Score
        validation_results['quality_score'] = self._calculate_quality_score(validation_results)
        
        # 8. Generate Issues and Warnings
        validation_results['issues'], validation_results['warnings'] = self._generate_issues_warnings(validation_results)
        
        logger.info(f"Validation completed for {filename}. Quality Score: {validation_results['quality_score']}")
        return validation_results
    
    def _validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data types of columns."""
        results = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                results[col] = {'type': 'empty', 'confidence': 0}
                continue
                
            # Check if it's numeric
            try:
                pd.to_numeric(col_data, errors='raise')
                results[col] = {'type': 'numeric', 'confidence': 1.0}
            except:
                # Check if it's date
                try:
                    pd.to_datetime(col_data, errors='raise')
                    results[col] = {'type': 'date', 'confidence': 1.0}
                except:
                    # Check if it's GSTIN-like
                    if 'gstin' in col.lower() or 'uin' in col.lower():
                        gstin_count = sum(1 for val in col_data if self._is_valid_gstin(str(val)))
                        results[col] = {
                            'type': 'gstin',
                            'confidence': gstin_count / len(col_data) if len(col_data) > 0 else 0
                        }
                    else:
                        results[col] = {'type': 'text', 'confidence': 1.0}
        
        return results
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        results = {}
        
        for col in df.columns:
            total_rows = len(df)
            missing_count = df[col].isna().sum()
            missing_percentage = (missing_count / total_rows) * 100 if total_rows > 0 else 0
            
            results[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_percentage, 2),
                'completeness_score': round(100 - missing_percentage, 2)
            }
        
        return results
    
    def _validate_gstin_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate GSTIN format."""
        results = {}
        
        gstin_columns = [col for col in df.columns if 'gstin' in col.lower() or 'uin' in col.lower()]
        
        for col in gstin_columns:
            col_data = df[col].dropna().astype(str)
            if len(col_data) == 0:
                results[col] = {'valid_count': 0, 'invalid_count': 0, 'validity_percentage': 0}
                continue
                
            valid_count = sum(1 for val in col_data if self._is_valid_gstin(val))
            invalid_count = len(col_data) - valid_count
            validity_percentage = (valid_count / len(col_data)) * 100 if len(col_data) > 0 else 0
            
            results[col] = {
                'valid_count': int(valid_count),
                'invalid_count': int(invalid_count),
                'validity_percentage': round(validity_percentage, 2)
            }
        
        return results
    
    def _is_valid_gstin(self, gstin: str) -> bool:
        """Check if GSTIN format is valid."""
        if not gstin or len(gstin) != 15:
            return False
        
        # GSTIN format: 2 digits (state) + 10 characters (PAN) + 1 character (entity) + 1 character (Z) + 1 character (checksum)
        pattern = r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$'
        return bool(re.match(pattern, gstin.upper()))
    
    def _validate_dates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate date columns."""
        results = {}
        
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        
        for col in date_columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                results[col] = {'valid_count': 0, 'invalid_count': 0, 'date_range': None}
                continue
                
            valid_dates = []
            invalid_count = 0
            
            for val in col_data:
                try:
                    parsed_date = pd.to_datetime(val)
                    valid_dates.append(parsed_date)
                except:
                    invalid_count += 1
            
            if valid_dates:
                min_date = min(valid_dates)
                max_date = max(valid_dates)
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            else:
                date_range = None
            
            results[col] = {
                'valid_count': len(valid_dates),
                'invalid_count': int(invalid_count),
                'date_range': date_range
            }
        
        return results
    
    def _validate_numeric_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate numeric columns."""
        results = {}
        
        numeric_columns = [col for col in df.columns if 'amount' in col.lower() or 'cgst' in col.lower() or 'sgst' in col.lower() or 'igst' in col.lower()]
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                results[col] = {'valid_count': 0, 'invalid_count': 0, 'statistics': {}}
                continue
                
            numeric_values = []
            invalid_count = 0
            
            for val in col_data:
                try:
                    # Handle common numeric formats
                    if isinstance(val, str):
                        # Remove currency symbols and commas
                        cleaned_val = re.sub(r'[₹,\s]', '', val)
                        numeric_val = float(cleaned_val)
                    else:
                        numeric_val = float(val)
                    
                    numeric_values.append(numeric_val)
                except:
                    invalid_count += 1
            
            if numeric_values:
                statistics = {
                    'min': round(min(numeric_values), 2),
                    'max': round(max(numeric_values), 2),
                    'mean': round(np.mean(numeric_values), 2),
                    'sum': round(sum(numeric_values), 2),
                    'negative_count': sum(1 for x in numeric_values if x < 0),
                    'zero_count': sum(1 for x in numeric_values if x == 0)
                }
            else:
                statistics = {}
            
            results[col] = {
                'valid_count': len(numeric_values),
                'invalid_count': int(invalid_count),
                'statistics': statistics
            }
        
        return results
    
    def _detect_duplicates_preview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Preview duplicate detection."""
        results = {}
        
        # Check for duplicates based on common key columns
        key_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['vch', 'voucher', 'invoice', 'bill']):
                key_columns.append(col)
        
        if not key_columns:
            results['message'] = 'No obvious key columns found for duplicate detection'
            return results
        
        # Check duplicates for each key column combination
        for col in key_columns:
            if col in df.columns:
                duplicates = df[df.duplicated(subset=[col], keep=False)]
                duplicate_count = len(duplicates)
                unique_count = len(df[col].dropna().unique())
                
                results[col] = {
                    'duplicate_rows': int(duplicate_count),
                    'unique_values': int(unique_count),
                    'duplicate_percentage': round((duplicate_count / len(df)) * 100, 2) if len(df) > 0 else 0
                }
        
        return results
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        score = 100.0
        
        # Deduct points for missing data
        missing_data = validation_results.get('missing_data', {})
        for col, data in missing_data.items():
            if data['missing_percentage'] > 50:
                score -= 20
            elif data['missing_percentage'] > 25:
                score -= 10
            elif data['missing_percentage'] > 10:
                score -= 5
        
        # Deduct points for invalid GSTINs
        gstin_validation = validation_results.get('gstin_validation', {})
        for col, data in gstin_validation.items():
            if data['validity_percentage'] < 50:
                score -= 15
            elif data['validity_percentage'] < 80:
                score -= 10
            elif data['validity_percentage'] < 95:
                score -= 5
        
        # Deduct points for invalid dates
        date_validation = validation_results.get('date_validation', {})
        for col, data in date_validation.items():
            if data['invalid_count'] > 0:
                score -= 5
        
        # Deduct points for invalid numeric data
        numeric_validation = validation_results.get('numeric_validation', {})
        for col, data in numeric_validation.items():
            if data['invalid_count'] > 0:
                score -= 5
        
        return max(0, round(score, 1))
    
    def _generate_issues_warnings(self, validation_results: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Generate issues and warnings based on validation results."""
        issues = []
        warnings = []
        
        # Check missing data
        missing_data = validation_results.get('missing_data', {})
        for col, data in missing_data.items():
            if data['missing_percentage'] > 50:
                issues.append(f"Column '{col}' has {data['missing_percentage']}% missing data")
            elif data['missing_percentage'] > 25:
                warnings.append(f"Column '{col}' has {data['missing_percentage']}% missing data")
        
        # Check GSTIN validation
        gstin_validation = validation_results.get('gstin_validation', {})
        for col, data in gstin_validation.items():
            if data['validity_percentage'] < 50:
                issues.append(f"Column '{col}' has {data['invalid_count']} invalid GSTINs")
            elif data['validity_percentage'] < 95:
                warnings.append(f"Column '{col}' has {data['invalid_count']} invalid GSTINs")
        
        # Check date validation
        date_validation = validation_results.get('date_validation', {})
        for col, data in date_validation.items():
            if data['invalid_count'] > 0:
                warnings.append(f"Column '{col}' has {data['invalid_count']} invalid dates")
        
        # Check numeric validation
        numeric_validation = validation_results.get('numeric_validation', {})
        for col, data in numeric_validation.items():
            if data['invalid_count'] > 0:
                warnings.append(f"Column '{col}' has {data['invalid_count']} invalid numeric values")
        
        return issues, warnings
    
    def get_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'filename': validation_results['filename'],
            'quality_score': validation_results['quality_score'],
            'total_issues': len(validation_results['issues']),
            'total_warnings': len(validation_results['warnings']),
            'total_rows': validation_results['total_rows'],
            'total_columns': validation_results['total_columns'],
            'validation_timestamp': validation_results['validation_timestamp']
        }

