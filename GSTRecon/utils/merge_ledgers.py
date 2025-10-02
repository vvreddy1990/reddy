"""
Merge Books Ledgers Utility Module

This module provides functionality to merge multiple GST ledger Excel files
from different systems (Tally, SAP, Others) into a single dataset for reconciliation.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from .data_validator import DataValidator
from .error_handler import ErrorHandler
from .data_analyzer import DataAnalyzer
from .progress_tracker import MergeProgressTracker
from .gstin_validator import GSTINValidator
import io

# Configure logging
logger = logging.getLogger(__name__)

class LedgerMerger:
    """Handles merging of multiple GST ledger files."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.merged_data = None
        self.merge_settings = {}
        self.validator = DataValidator()
        self.validation_results = {}
        self.error_handler = ErrorHandler()
        self.analyzer = DataAnalyzer()
        self.analysis_results = {}
        self.progress_tracker = MergeProgressTracker()
        self.gstin_validator = GSTINValidator()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize predefined column mappings for different templates."""
        return {
            "Tally": {
                "standard_columns": [
                    "Date", "Particulars", "Vch Type", "Vch No.", "Vch Ref No.", 
                    "Vch Ref Date", "GSTIN/UIN", "PAN No."
                ],
                "tax_columns": [
                    "CGST-Input@6%", "SGST-Input@6%", "IGST-Input@18%",
                    "CGST-Input@9%", "SGST-Input@9%", "IGST-Input@12%",
                    "CGST-Input@18%", "SGST-Input@18%", "IGST-Input@28%"
                ]
            },
            "SAP": {
                "standard_columns": [
                    "Posting Date", "Document No.", "Doc Ref No.", "Doc Ref Date",
                    "Vendor GSTIN", "PAN Number", "Vendor Name", "Document Type"
                ],
                "tax_columns": [
                    "CGST Input 6%", "SGST Input 6%", "Integrated GST 18%",
                    "CGST Input 9%", "SGST Input 9%", "Integrated GST 12%",
                    "CGST Input 18%", "SGST Input 18%", "Integrated GST 28%"
                ]
            },
            "Others": {
                "standard_columns": [],
                "tax_columns": []
            }
        }
    
    def detect_template_from_filename(self, filenames: List[str]) -> str:
        """Suggest template based on filenames."""
        filename_text = " ".join(filenames).lower()
        
        if any(keyword in filename_text for keyword in ["tally", "6%", "9%", "18%", "28%"]):
            return "Tally"
        elif any(keyword in filename_text for keyword in ["sap", "erp", "enterprise"]):
            return "SAP"
        else:
            return "Others"
    
    def detect_columns(self, df: pd.DataFrame, template: str = "Others") -> Dict[str, List[str]]:
        """Auto-detect columns in the dataframe."""
        detected = {
            "standard_columns": [],
            "tax_columns": [],
            "additional_columns": []
        }
        
        if df is None or df.empty:
            return detected
        
        columns = df.columns.tolist()
        
        # Get template mappings if available
        if template in self.templates and template != "Others":
            template_standard = self.templates[template]["standard_columns"]
            template_tax = self.templates[template]["tax_columns"]
            
            # Find exact matches first
            for col in columns:
                if col in template_standard:
                    detected["standard_columns"].append(col)
                elif col in template_tax:
                    detected["tax_columns"].append(col)
                else:
                    detected["additional_columns"].append(col)
        else:
            # Auto-detect for "Others" template
            for col in columns:
                col_lower = col.lower()
                
                # Detect tax columns using improved regex to catch all variations
                if re.search(r'cgst|sgst|igst', col_lower):
                    detected["tax_columns"].append(col)
                # Detect common standard columns
                elif any(keyword in col_lower for keyword in [
                    "date", "particulars", "vch", "gstin", "pan", "document", 
                    "vendor", "invoice", "reference", "posting", "doc"
                ]):
                    detected["standard_columns"].append(col)
                else:
                    detected["additional_columns"].append(col)
        
        return detected
    
    def validate_files(self, uploaded_files: List[Any]) -> Tuple[List[Any], List[str]]:
        """Validate uploaded files and return valid files with warnings."""
        valid_files = []
        warnings = []
        
        for file in uploaded_files:
            try:
                # Check file extension
                if not (file.name.endswith('.xlsx') or file.name.endswith('.xls')):
                    warnings.append(f"File '{file.name}' is not an Excel file. Skipping.")
                    continue
                
                # Try to read the file
                df = pd.read_excel(file, nrows=1)  # Read only headers
                if df.empty:
                    warnings.append(f"File '{file.name}' appears to be empty. Skipping.")
                    continue
                
                valid_files.append(file)
                
            except Exception as e:
                warnings.append(f"Error reading file '{file.name}': {str(e)}. Skipping.")
                continue
        
        return valid_files, warnings
    
    def read_file_data(self, file: Any) -> pd.DataFrame:
        """Read data from uploaded file or dataframe with enhanced error handling."""
        try:
            if isinstance(file, pd.DataFrame):
                # If it's already a dataframe, just add source info
                df = file.copy()
                df['_source_file'] = 'dataframe_input'
                
                self.error_handler.log_info(
                    "DATAFRAME_READ_SUCCESS",
                    "Successfully processed DataFrame input",
                    {'rows': len(df), 'columns': len(df.columns)}
                )
                
                return df
            else:
                # Read from file with enhanced error handling
                filename = getattr(file, 'name', 'unknown_file')
                
                try:
                    df = pd.read_excel(file, engine='openpyxl')
                    df['_source_file'] = filename
                    
                    self.error_handler.log_info(
                        "FILE_READ_SUCCESS",
                        f"Successfully read file '{filename}'",
                        {'rows': len(df), 'columns': len(df.columns)}
                    )
                    
                    return df
                    
                except Exception as e:
                    # Handle specific Excel reading errors
                    error_entry = self.error_handler.handle_file_error(e, filename, "read_excel")
                    
                    # Try alternative reading methods
                    try:
                        # Reset file pointer if possible
                        if hasattr(file, 'seek'):
                            file.seek(0)
                        
                        # Try with different engine
                        df = pd.read_excel(file, engine='xlrd')
                        df['_source_file'] = filename
                        
                        self.error_handler.log_warning(
                            "FILE_READ_RECOVERY",
                            f"Successfully read file '{filename}' using alternative method",
                            {'rows': len(df), 'columns': len(df.columns)},
                            file_context=filename,
                            suggestion="Consider converting file to standard Excel format"
                        )
                        
                        return df
                        
                    except Exception as e2:
                        # Both methods failed
                        self.error_handler.handle_file_error(e2, filename, "read_excel_alternative")
                        raise e2
                        
        except Exception as e:
            filename = getattr(file, 'name', 'unknown_file')
            self.error_handler.handle_file_error(e, filename, "read_file_data")
            logger.error(f"Error reading file {filename}: {e}")
            return pd.DataFrame()
    
    def apply_merge_settings(self, df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
        """Apply merger settings to the dataframe."""
        result_df = df.copy()
        
        # Apply value sign conversion
        if settings.get('sign_conversion_columns'):
            for col in settings['sign_conversion_columns']:
                if col in result_df.columns:
                    conversion_type = settings.get('sign_conversion_type')
                    if conversion_type == 'positive_to_negative':
                        result_df[col] = -abs(pd.to_numeric(result_df[col], errors='coerce'))
                    elif conversion_type == 'negative_to_positive':
                        result_df[col] = abs(pd.to_numeric(result_df[col], errors='coerce'))
                    elif conversion_type == 'both_positive_to_negative_and_negative_to_positive':
                        # Convert both: positive to negative and negative to positive
                        numeric_values = pd.to_numeric(result_df[col], errors='coerce')
                        result_df[col] = -numeric_values  # This flips both positive and negative
        
        # Apply column renaming
        if settings.get('column_renames'):
            result_df = result_df.rename(columns=settings['column_renames'])
        
        # Apply missing value handling
        missing_value_strategy = settings.get('missing_value_strategy', '0')
        if missing_value_strategy == '0':
            # Fill tax columns with 0
            tax_cols = [col for col in result_df.columns if re.search(r'(c|s|i)gst', col.lower())]
            result_df[tax_cols] = result_df[tax_cols].fillna(0)
        elif missing_value_strategy == 'NaN':
            # Keep NaN values
            pass
        elif missing_value_strategy == 'Empty':
            # Fill with empty string
            result_df = result_df.fillna('')
        
        # Apply decimal precision
        decimal_precision = settings.get('decimal_precision', 2)
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        result_df[numeric_cols] = result_df[numeric_cols].round(decimal_precision)
        
        return result_df
    
    def merge_files(self, files: List[Any], template: str, column_mappings: Dict[str, List[str]], 
                   settings: Dict[str, Any]) -> pd.DataFrame:
        """Merge multiple files into a single dataset with progress tracking."""
        logger.info(f"Starting merge of {len(files)} files using template: {template}")
        
        # Start progress tracking
        self.progress_tracker.start_merge_operation(len(files))
        
        # Validate files comprehensively
        validation_results = self.validate_files_comprehensive(files)
        
        try:
            merged_dataframes = []
            
            for i, file in enumerate(files):
                filename = getattr(file, 'name', f'File_{i+1}')
                
                # Start file validation
                self.progress_tracker.start_file_validation(i, filename)
                df = self.read_file_data(file)
                if df.empty:
                    self.progress_tracker.complete_file_validation(i, filename)
                    continue
                
                # Complete validation
                self.progress_tracker.complete_file_validation(i, filename)
                
                # Start file analysis
                self.progress_tracker.start_file_analysis(i, filename)
                # Analysis is done in background, just mark as complete
                self.progress_tracker.complete_file_analysis(i, filename)
                
                # Start file merge
                self.progress_tracker.start_file_merge(i, filename)
                
                # Apply settings to individual file
                df = self.apply_merge_settings(df, settings)
                
                # Standardize columns based on mappings
                df = self._standardize_columns(df, column_mappings, template)
                
                merged_dataframes.append(df)
                
                # Complete file merge
                self.progress_tracker.complete_file_merge(i, filename)
            
            if not merged_dataframes:
                self.progress_tracker.fail_merge_operation("No valid files found")
                return pd.DataFrame()
            
            # Start post-processing
            self.progress_tracker.start_post_processing()
            
            # Concatenate all dataframes
            merged_df = pd.concat(merged_dataframes, ignore_index=True, sort=False)
            
            # Update progress
            self.progress_tracker.update_post_processing(50, "Concatenating dataframes...")
            
            # Apply post-merge processing
            merged_df = self._post_merge_processing(merged_df, settings)
            
            # Update progress
            self.progress_tracker.update_post_processing(100, "Post-processing completed")
            
            # Complete post-processing
            self.progress_tracker.complete_post_processing()
            
            # Store merged data
            self.merged_data = merged_df
            
            # Complete merge operation
            self.progress_tracker.complete_merge_operation(len(merged_df))
            
            logger.info(f"Successfully merged {len(files)} files into {len(merged_df)} records")
            return merged_df
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error during merge operation: {error_message}")
            self.progress_tracker.fail_merge_operation(error_message)
            raise
    
    def _standardize_columns(self, df: pd.DataFrame, column_mappings: Dict[str, List[str]], 
                           template: str) -> pd.DataFrame:
        """Standardize columns based on mappings."""
        result_df = df.copy()
        
        # Create standardized tax columns by finding ALL columns with tax keywords
        # This includes both mapped tax columns and any other columns with tax keywords
        all_columns = result_df.columns.tolist()
        
        # Find ALL CGST columns (case-insensitive) - improved regex to catch all variations
        cgst_cols = [col for col in all_columns if re.search(r'cgst', col.lower())]
        if cgst_cols:
            logger.info(f"Found CGST columns: {cgst_cols}")
        else:
            cgst_cols = []
        
        # Find ALL SGST columns (case-insensitive) - improved regex to catch all variations
        sgst_cols = [col for col in all_columns if re.search(r'sgst', col.lower())]
        if sgst_cols:
            logger.info(f"Found SGST columns: {sgst_cols}")
        else:
            sgst_cols = []
        
        # Find ALL IGST columns (case-insensitive) - improved regex to catch all variations
        igst_cols = [col for col in all_columns if re.search(r'igst', col.lower())]
        if igst_cols:
            logger.info(f"Found IGST columns: {igst_cols}")
        else:
            igst_cols = []
        
        # Create traceability column with all tax columns found
        all_tax_cols = cgst_cols + sgst_cols + igst_cols
        result_df['Original Ledger Name'] = result_df['_source_file'].apply(
            lambda x: self._create_traceability_info(x, all_tax_cols)
        )
        
        # Move Original Ledger Name to first column
        cols = result_df.columns.tolist()
        if 'Original Ledger Name' in cols:
            cols.remove('Original Ledger Name')
            cols.insert(0, 'Original Ledger Name')
            result_df = result_df[cols]
        
        return result_df
    
    def _create_traceability_info(self, filename: str, tax_columns: List[str]) -> str:
        """Create traceability information for the original ledger."""
        if not tax_columns:
            return f"From {filename}: No Tax Columns"
        
        # Group tax columns by type
        cgst_cols = [col for col in tax_columns if 'cgst' in col.lower()]
        sgst_cols = [col for col in tax_columns if 'sgst' in col.lower()]
        igst_cols = [col for col in tax_columns if 'igst' in col.lower()]
        
        tax_info = []
        if cgst_cols:
            tax_info.append(f"CGST: {', '.join(cgst_cols)}")
        if sgst_cols:
            tax_info.append(f"SGST: {', '.join(sgst_cols)}")
        if igst_cols:
            tax_info.append(f"IGST: {', '.join(igst_cols)}")
        
        return f"From {filename}: {' + '.join(tax_info)}"
    
    def _post_merge_processing(self, df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
        """Apply post-merge processing."""
        result_df = df.copy()
        
        # Add tax total columns at the end
        all_columns = result_df.columns.tolist()
        
        # Find ALL CGST columns and calculate total
        cgst_cols = [col for col in all_columns if re.search(r'cgst', col.lower())]
        if cgst_cols:
            result_df['Total CGST Amount'] = result_df[cgst_cols].apply(
                lambda row: pd.to_numeric(row, errors='coerce').sum(), axis=1
            )
            logger.info(f"Calculated Total CGST Amount from columns: {cgst_cols}")
        else:
            result_df['Total CGST Amount'] = 0
        
        # Find ALL SGST columns and calculate total
        sgst_cols = [col for col in all_columns if re.search(r'sgst', col.lower())]
        if sgst_cols:
            result_df['Total SGST Amount'] = result_df[sgst_cols].apply(
                lambda row: pd.to_numeric(row, errors='coerce').sum(), axis=1
            )
            logger.info(f"Calculated Total SGST Amount from columns: {sgst_cols}")
        else:
            result_df['Total SGST Amount'] = 0
        
        # Find ALL IGST columns and calculate total
        igst_cols = [col for col in all_columns if re.search(r'igst', col.lower())]
        if igst_cols:
            result_df['Total IGST Amount'] = result_df[igst_cols].apply(
                lambda row: pd.to_numeric(row, errors='coerce').sum(), axis=1
            )
            logger.info(f"Calculated Total IGST Amount from columns: {igst_cols}")
        else:
            result_df['Total IGST Amount'] = 0
        
        # Add GSTIN validation column
        if 'GSTIN/UIN' in result_df.columns:
            gstin_validation_results = self.gstin_validator.validate_gstin_column(result_df['GSTIN/UIN'])
            gstin_status = []
            gstin_details = []
            
            for result in gstin_validation_results:
                if result['status'] == 'Valid':
                    gstin_status.append('Valid')
                    details = result['details']
                    detail_str = f"State: {details.get('state_name', 'N/A')}, PAN: {details.get('pan_number', 'N/A')}, Type: {details.get('business_type', 'N/A')}"
                    gstin_details.append(detail_str)
                else:
                    gstin_status.append('Invalid')
                    error_messages = [error['message'] for error in result['errors']]
                    gstin_details.append('; '.join(error_messages))
            
            # Insert GSTIN validation columns after Original Ledger Name
            insert_pos = 1 if 'Original Ledger Name' in result_df.columns else 0
            result_df.insert(insert_pos, 'GST Number Validation Status', gstin_status)
            result_df.insert(insert_pos + 1, 'GST Validation Details', gstin_details)
        
        # Handle duplicates - only mark actual duplicates with occurrence numbers
        if settings.get('enable_duplicate_detection', False):
            key_columns = settings.get('deduplicate_key_columns', [])
            if key_columns:
                # Only use columns that exist in the dataframe
                existing_key_cols = [col for col in key_columns if col in result_df.columns]
                if existing_key_cols:
                    # Check if there are actually duplicates
                    duplicate_mask = result_df.duplicated(subset=existing_key_cols, keep=False)
                    has_duplicates = duplicate_mask.any()
                    
                    if has_duplicates:
                        # Initialize columns with empty values
                        result_df['Duplicate Unique ID'] = ''
                        result_df['Duplicate Status'] = ''
                        
                        # Only process rows that have duplicates
                        duplicate_rows = result_df[duplicate_mask].copy()
                        
                        # Group by key columns and assign occurrence numbers
                        duplicate_group_counter = 1
                        for group_key, group in duplicate_rows.groupby(existing_key_cols):
                            group_indices = group.index.tolist()
                            
                            # Assign simple sequential unique ID for this duplicate group
                            unique_id = duplicate_group_counter
                            
                            # Assign occurrence numbers
                            for i, idx in enumerate(group_indices):
                                result_df.loc[idx, 'Duplicate Unique ID'] = unique_id
                                result_df.loc[idx, 'Duplicate Status'] = f"{i+1}st Occurrence" if i == 0 else f"{i+1}nd Occurrence" if i == 1 else f"{i+1}rd Occurrence" if i == 2 else f"{i+1}th Occurrence"
                            
                            duplicate_group_counter += 1
                        
                        logger.info(f"Found {duplicate_mask.sum()} duplicate records out of {len(result_df)} total records")
                    else:
                        logger.info("No duplicates found, skipping duplicate detection columns")
        
        # Clean up temporary columns
        if '_source_file' in result_df.columns:
            result_df = result_df.drop('_source_file', axis=1)
        
        return result_df
    
    def validate_files_comprehensive(self, files: List[Any]) -> Dict[str, Any]:
        """Validate all files before merging."""
        logger.info(f"Starting validation for {len(files)} files")
        
        validation_results = {
            'files': {},
            'overall_quality_score': 0,
            'total_issues': 0,
            'total_warnings': 0,
            'validation_summary': {}
        }
        
        all_issues = []
        all_warnings = []
        quality_scores = []
        
        for i, file in enumerate(files):
            try:
                # Read file data for validation
                df = self.read_file_data(file)
                filename = getattr(file, 'name', f'File_{i+1}')
                
                # Validate the file
                file_validation = self.validator.validate_file(df, filename)
                validation_results['files'][filename] = file_validation
                
                # Collect issues and warnings
                all_issues.extend(file_validation['issues'])
                all_warnings.extend(file_validation['warnings'])
                quality_scores.append(file_validation['quality_score'])
                
                logger.info(f"Validated {filename}: Quality Score = {file_validation['quality_score']}")
                
            except Exception as e:
                logger.error(f"Error validating file {i+1}: {str(e)}")
                validation_results['files'][f'File_{i+1}'] = {
                    'error': str(e),
                    'quality_score': 0,
                    'issues': [f"Failed to validate file: {str(e)}"],
                    'warnings': []
                }
                all_issues.append(f"File {i+1}: {str(e)}")
        
        # Calculate overall metrics
        validation_results['overall_quality_score'] = round(np.mean(quality_scores), 1) if quality_scores else 0
        validation_results['total_issues'] = len(all_issues)
        validation_results['total_warnings'] = len(all_warnings)
        
        # Generate validation summary
        validation_results['validation_summary'] = {
            'total_files': len(files),
            'successfully_validated': len([f for f in validation_results['files'].values() if 'error' not in f]),
            'failed_validations': len([f for f in validation_results['files'].values() if 'error' in f]),
            'overall_quality_score': validation_results['overall_quality_score'],
            'total_issues': validation_results['total_issues'],
            'total_warnings': validation_results['total_warnings'],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        self.validation_results = validation_results
        logger.info(f"Validation completed. Overall Quality Score: {validation_results['overall_quality_score']}")
        
        return validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        if not self.validation_results:
            return {'message': 'No validation results available'}
        
        return self.validation_results.get('validation_summary', {})
    
    def get_file_validation_details(self, filename: str) -> Dict[str, Any]:
        """Get detailed validation results for a specific file."""
        if not self.validation_results or 'files' not in self.validation_results:
            return {'error': 'No validation results available'}
        
        return self.validation_results['files'].get(filename, {'error': 'File not found'})
    
    def analyze_files(self, files: List[Any]) -> Dict[str, Any]:
        """Analyze all files for patterns and insights."""
        logger.info(f"Starting analysis for {len(files)} files")
        
        analysis_results = {
            'files': {},
            'overall_analysis': {},
            'analysis_summary': {}
        }
        
        all_recommendations = []
        analysis_scores = []
        
        for i, file in enumerate(files):
            try:
                # Read file data for analysis
                df = self.read_file_data(file)
                filename = getattr(file, 'name', f'File_{i+1}')
                
                # Analyze the file
                file_analysis = self.analyzer.analyze_dataframe(df, filename)
                analysis_results['files'][filename] = file_analysis
                
                # Collect recommendations and scores
                all_recommendations.extend(file_analysis['recommendations'])
                analysis_scores.append(file_analysis['data_quality']['quality_score'])
                
                logger.info(f"Analyzed {filename}: Quality Score = {file_analysis['data_quality']['quality_score']}")
                
            except Exception as e:
                logger.error(f"Error analyzing file {i+1}: {str(e)}")
                analysis_results['files'][f'File_{i+1}'] = {
                    'error': str(e),
                    'data_quality': {'quality_score': 0},
                    'recommendations': [f"Failed to analyze file: {str(e)}"]
                }
                all_recommendations.append(f"File {i+1}: {str(e)}")
        
        # Calculate overall analysis
        analysis_results['overall_analysis'] = {
            'average_quality_score': round(np.mean(analysis_scores), 1) if analysis_scores else 0,
            'total_recommendations': len(all_recommendations),
            'unique_recommendations': len(set(all_recommendations)),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Generate analysis summary
        analysis_results['analysis_summary'] = {
            'total_files': len(files),
            'successfully_analyzed': len([f for f in analysis_results['files'].values() if 'error' not in f]),
            'failed_analyses': len([f for f in analysis_results['files'].values() if 'error' in f]),
            'average_quality_score': analysis_results['overall_analysis']['average_quality_score'],
            'total_recommendations': analysis_results['overall_analysis']['total_recommendations'],
            'analysis_timestamp': analysis_results['overall_analysis']['analysis_timestamp']
        }
        
        self.analysis_results = analysis_results
        logger.info(f"Analysis completed. Average Quality Score: {analysis_results['overall_analysis']['average_quality_score']}")
        
        return analysis_results
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get analysis summary."""
        if not self.analysis_results:
            return {'message': 'No analysis results available'}
        
        return self.analysis_results.get('analysis_summary', {})
    
    def get_file_analysis_details(self, filename: str) -> Dict[str, Any]:
        """Get detailed analysis results for a specific file."""
        if not self.analysis_results or 'files' not in self.analysis_results:
            return {'error': 'No analysis results available'}
        
        return self.analysis_results['files'].get(filename, {'error': 'File not found'})
    
    def get_progress_status(self) -> Dict[str, Any]:
        """Get current progress status."""
        return self.progress_tracker.get_detailed_status()
    
    def get_progress_history(self) -> List[Dict[str, Any]]:
        """Get progress history."""
        return self.progress_tracker.get_operation_history()
    
    def add_progress_callback(self, callback) -> None:
        """Add a progress callback function."""
        self.progress_tracker.add_callback(callback)
    
    def remove_progress_callback(self, callback) -> None:
        """Remove a progress callback function."""
        self.progress_tracker.remove_callback(callback)
    
    def reset_progress(self) -> None:
        """Reset progress tracker."""
        self.progress_tracker.reset()
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get comprehensive error report."""
        return self.error_handler.get_error_report()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return self.error_handler.get_session_summary()
    
    def get_error_suggestions(self, error_type: str) -> List[str]:
        """Get suggestions for resolving specific error types."""
        # Find the most recent error of this type
        for error in reversed(self.error_handler.error_log):
            if error['error_type'] == error_type:
                return self.error_handler.get_error_suggestions(error)
        return []
    
    def clear_error_logs(self):
        """Clear all error logs."""
        self.error_handler.clear_logs()
    
    def get_merge_summary(self, merged_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of the merged data."""
        if merged_df is None or merged_df.empty:
            return {}
        
        summary = {
            "total_records": len(merged_df),
            "total_columns": len(merged_df.columns),
            "total_cgst": merged_df.get('Total CGST Amount', pd.Series()).sum(),
            "total_sgst": merged_df.get('Total SGST Amount', pd.Series()).sum(),
            "total_igst": merged_df.get('Total IGST Amount', pd.Series()).sum(),
            "unique_gstins": merged_df.get('GSTIN/UIN', pd.Series()).nunique() if 'GSTIN/UIN' in merged_df.columns else 0,
            "date_range": self._get_date_range(merged_df),
            "duplicate_groups": 0,
            "data_quality_score": 0
        }
        
        # Duplicate groups
        if 'Duplicate Unique ID' in merged_df.columns:
            duplicate_ids = merged_df[merged_df['Duplicate Unique ID'] != '']['Duplicate Unique ID'].nunique()
            summary['duplicate_groups'] = int(duplicate_ids)
        
        # Data quality score (based on validation results)
        if self.validation_results:
            summary['data_quality_score'] = self.validation_results.get('overall_quality_score', 0)
        
        return summary
    
    def _get_date_range(self, df: pd.DataFrame) -> str:
        """Get date range from the dataframe."""
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if not date_columns:
            return "No date columns found"
        
        try:
            # Try to find the first valid date column
            for col in date_columns:
                dates = pd.to_datetime(df[col], errors='coerce')
                valid_dates = dates.dropna()
                if not valid_dates.empty:
                    min_date = valid_dates.min().strftime('%Y-%m-%d')
                    max_date = valid_dates.max().strftime('%Y-%m-%d')
                    return f"{min_date} to {max_date}"
        except Exception:
            pass
        
        return "Unable to determine date range"
    
    def export_merged_data(self, filename: str = None) -> bytes:
        """Export merged data to Excel format."""
        if self.merged_data is None or self.merged_data.empty:
            return b''
        
        if filename is None:
            filename = f"Merged_Books_Data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            self.merged_data.to_excel(writer, sheet_name='Merged Data', index=False)
        
        return output.getvalue()
