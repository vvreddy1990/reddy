"""
Enhanced preview and analysis capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Enhanced data analysis and preview capabilities."""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_dataframe(self, df: pd.DataFrame, filename: str = "Unknown") -> Dict[str, Any]:
        """Comprehensive analysis of a DataFrame."""
        logger.info(f"Starting analysis for {filename}")
        
        analysis = {
            'filename': filename,
            'analysis_timestamp': datetime.now().isoformat(),
            'basic_stats': self._get_basic_stats(df),
            'column_analysis': self._analyze_columns(df),
            'data_quality': self._assess_data_quality(df),
            'patterns': self._detect_patterns(df),
            'recommendations': []
        }
        
        # Generate recommendations based on analysis
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        logger.info(f"Analysis completed for {filename}")
        return analysis
    
    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about the DataFrame."""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_rows': df.duplicated().sum(),
            'empty_rows': df.isnull().all(axis=1).sum(),
            'data_types': df.dtypes.value_counts().to_dict()
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze each column in detail."""
        column_analysis = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            analysis = {
                'data_type': str(df[col].dtype),
                'non_null_count': len(col_data),
                'null_count': df[col].isnull().sum(),
                'null_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2),
                'unique_count': df[col].nunique(),
                'duplicate_count': len(df[col]) - df[col].nunique(),
                'most_common_value': None,
                'most_common_frequency': 0,
                'statistics': {}
            }
            
            # Get most common value
            if len(col_data) > 0:
                value_counts = col_data.value_counts()
                if len(value_counts) > 0:
                    analysis['most_common_value'] = str(value_counts.index[0])
                    analysis['most_common_frequency'] = int(value_counts.iloc[0])
            
            # Get statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        analysis['statistics'] = {
                            'min': round(float(numeric_data.min()), 2),
                            'max': round(float(numeric_data.max()), 2),
                            'mean': round(float(numeric_data.mean()), 2),
                            'median': round(float(numeric_data.median()), 2),
                            'std': round(float(numeric_data.std()), 2),
                            'sum': round(float(numeric_data.sum()), 2),
                            'negative_count': int((numeric_data < 0).sum()),
                            'zero_count': int((numeric_data == 0).sum()),
                            'positive_count': int((numeric_data > 0).sum())
                        }
                except:
                    pass
            
            # Get statistics for date columns
            elif 'date' in col.lower():
                try:
                    date_data = pd.to_datetime(col_data, errors='coerce').dropna()
                    if len(date_data) > 0:
                        analysis['statistics'] = {
                            'earliest_date': date_data.min().strftime('%Y-%m-%d'),
                            'latest_date': date_data.max().strftime('%Y-%m-%d'),
                            'date_range_days': int((date_data.max() - date_data.min()).days),
                            'unique_dates': len(date_data.unique())
                        }
                except:
                    pass
            
            column_analysis[col] = analysis
        
        return column_analysis
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality_score = 100.0
        issues = []
        warnings = []
        
        # Check for missing data
        missing_percentages = df.isnull().sum() / len(df) * 100
        high_missing_cols = missing_percentages[missing_percentages > 50]
        
        if len(high_missing_cols) > 0:
            quality_score -= len(high_missing_cols) * 10
            issues.extend([f"Column '{col}' has {missing_percentages[col]:.1f}% missing data" 
                          for col in high_missing_cols.index])
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            if duplicate_percentage > 20:
                quality_score -= 15
                issues.append(f"{duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)")
            else:
                quality_score -= 5
                warnings.append(f"{duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)")
        
        # Check for empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            quality_score -= len(empty_cols) * 5
            warnings.extend([f"Column '{col}' is completely empty" for col in empty_cols])
        
        # Check for columns with all same values
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() == 1 and not df[col].isnull().all():
                constant_cols.append(col)
        
        if constant_cols:
            quality_score -= len(constant_cols) * 3
            warnings.extend([f"Column '{col}' has only one unique value" for col in constant_cols])
        
        return {
            'quality_score': max(0, round(quality_score, 1)),
            'issues': issues,
            'warnings': warnings,
            'missing_data_percentage': round(missing_percentages.mean(), 2),
            'duplicate_percentage': round((duplicate_count / len(df)) * 100, 2) if len(df) > 0 else 0
        }
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns in the data."""
        patterns = {
            'date_patterns': self._detect_date_patterns(df),
            'numeric_patterns': self._detect_numeric_patterns(df),
            'text_patterns': self._detect_text_patterns(df),
            'gstin_patterns': self._detect_gstin_patterns(df),
            'duplicate_patterns': self._detect_duplicate_patterns(df)
        }
        
        return patterns
    
    def _detect_date_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect date-related patterns."""
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        patterns = {}
        
        for col in date_cols:
            try:
                dates = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dates) > 0:
                    patterns[col] = {
                        'date_range': f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}",
                        'total_days': int((dates.max() - dates.min()).days),
                        'unique_dates': len(dates.unique()),
                        'weekend_dates': int(dates.dt.dayofweek.isin([5, 6]).sum()),
                        'future_dates': int((dates > datetime.now()).sum()),
                        'old_dates': int((dates < datetime(2020, 1, 1)).sum())
                    }
            except:
                pass
        
        return patterns
    
    def _detect_numeric_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect numeric patterns."""
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        patterns = {}
        
        for col in numeric_cols:
            try:
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(numeric_data) > 0:
                    patterns[col] = {
                        'has_negatives': bool((numeric_data < 0).any()),
                        'has_zeros': bool((numeric_data == 0).any()),
                        'has_decimals': bool((numeric_data % 1 != 0).any()),
                        'outliers': self._detect_outliers(numeric_data),
                        'distribution': self._get_distribution_info(numeric_data)
                    }
            except:
                pass
        
        return patterns
    
    def _detect_text_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect text patterns."""
        text_cols = [col for col in df.columns if df[col].dtype == 'object']
        patterns = {}
        
        for col in text_cols:
            text_data = df[col].dropna().astype(str)
            if len(text_data) > 0:
                patterns[col] = {
                    'avg_length': round(text_data.str.len().mean(), 2),
                    'max_length': int(text_data.str.len().max()),
                    'min_length': int(text_data.str.len().min()),
                    'has_special_chars': bool(text_data.str.contains(r'[^a-zA-Z0-9\s]').any()),
                    'has_numbers': bool(text_data.str.contains(r'\d').any()),
                    'has_uppercase': bool(text_data.str.contains(r'[A-Z]').any()),
                    'has_lowercase': bool(text_data.str.contains(r'[a-z]').any())
                }
        
        return patterns
    
    def _detect_gstin_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect GSTIN patterns."""
        gstin_cols = [col for col in df.columns if 'gstin' in col.lower() or 'uin' in col.lower()]
        patterns = {}
        
        for col in gstin_cols:
            gstin_data = df[col].dropna().astype(str)
            if len(gstin_data) > 0:
                # Check GSTIN format
                valid_gstins = gstin_data.str.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$')
                
                patterns[col] = {
                    'total_gstins': len(gstin_data),
                    'valid_gstins': int(valid_gstins.sum()),
                    'invalid_gstins': int((~valid_gstins).sum()),
                    'validity_percentage': round((valid_gstins.sum() / len(gstin_data)) * 100, 2),
                    'unique_gstins': len(gstin_data.unique()),
                    'duplicate_gstins': len(gstin_data) - len(gstin_data.unique())
                }
        
        return patterns
    
    def _detect_duplicate_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicate patterns."""
        patterns = {}
        
        # Check for duplicates in key columns
        key_columns = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['vch', 'voucher', 'invoice', 'bill', 'gstin'])]
        
        for col in key_columns:
            if col in df.columns:
                duplicates = df[df.duplicated(subset=[col], keep=False)]
                if len(duplicates) > 0:
                    patterns[col] = {
                        'duplicate_count': len(duplicates),
                        'duplicate_percentage': round((len(duplicates) / len(df)) * 100, 2),
                        'unique_values': len(df[col].unique()),
                        'most_duplicated_value': str(duplicates[col].value_counts().index[0]) if len(duplicates) > 0 else None,
                        'most_duplicated_count': int(duplicates[col].value_counts().iloc[0]) if len(duplicates) > 0 else 0
                    }
        
        return patterns
    
    def _detect_outliers(self, data: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        try:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            return {
                'count': len(outliers),
                'percentage': round((len(outliers) / len(data)) * 100, 2),
                'lower_bound': round(float(lower_bound), 2),
                'upper_bound': round(float(upper_bound), 2)
            }
        except:
            return {'count': 0, 'percentage': 0, 'lower_bound': 0, 'upper_bound': 0}
    
    def _get_distribution_info(self, data: pd.Series) -> Dict[str, Any]:
        """Get distribution information."""
        try:
            return {
                'skewness': round(float(data.skew()), 3),
                'kurtosis': round(float(data.kurtosis()), 3),
                'is_normal': abs(data.skew()) < 0.5 and abs(data.kurtosis()) < 0.5
            }
        except:
            return {'skewness': 0, 'kurtosis': 0, 'is_normal': False}
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Data quality recommendations
        if analysis['data_quality']['quality_score'] < 70:
            recommendations.append("Data quality is low. Consider data cleaning before processing.")
        
        if analysis['data_quality']['missing_data_percentage'] > 20:
            recommendations.append("High percentage of missing data. Review data collection process.")
        
        # Column-specific recommendations
        for col, col_analysis in analysis['column_analysis'].items():
            if col_analysis['null_percentage'] > 50:
                recommendations.append(f"Column '{col}' has {col_analysis['null_percentage']}% missing data. Consider removing or filling.")
            
            if col_analysis['duplicate_count'] > len(col_analysis) * 0.8:
                recommendations.append(f"Column '{col}' has many duplicate values. Check for data entry errors.")
        
        # Pattern-specific recommendations
        patterns = analysis['patterns']
        
        # GSTIN recommendations
        for col, gstin_pattern in patterns['gstin_patterns'].items():
            if gstin_pattern['validity_percentage'] < 80:
                recommendations.append(f"Column '{col}' has {gstin_pattern['invalid_gstins']} invalid GSTINs. Validate GSTIN format.")
        
        # Date recommendations
        for col, date_pattern in patterns['date_patterns'].items():
            if date_pattern.get('future_dates', 0) > 0:
                recommendations.append(f"Column '{col}' contains future dates. Verify date accuracy.")
            
            if date_pattern.get('old_dates', 0) > len(date_pattern) * 0.5:
                recommendations.append(f"Column '{col}' contains many old dates. Consider data relevance.")
        
        # Duplicate recommendations
        for col, dup_pattern in patterns['duplicate_patterns'].items():
            if dup_pattern['duplicate_percentage'] > 30:
                recommendations.append(f"Column '{col}' has {dup_pattern['duplicate_percentage']}% duplicates. Review for data quality issues.")
        
        return recommendations
    
    def get_preview_data(self, df: pd.DataFrame, rows: int = 10, 
                        columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get preview data with metadata."""
        if columns:
            preview_df = df[columns].head(rows)
        else:
            preview_df = df.head(rows)
        
        return {
            'data': preview_df.to_dict('records'),
            'columns': preview_df.columns.tolist(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'preview_rows': len(preview_df),
            'preview_columns': len(preview_df.columns),
            'has_more_rows': len(df) > rows,
            'has_more_columns': len(df.columns) > len(preview_df.columns) if columns else False
        }
    
    def get_column_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of all columns."""
        summary = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            summary[col] = {
                'type': str(df[col].dtype),
                'non_null_count': len(col_data),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'sample_values': col_data.head(5).tolist() if len(col_data) > 0 else []
            }
        
        return summary

