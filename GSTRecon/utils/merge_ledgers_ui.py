"""
Merge Books Ledgers UI Module

This module provides the Streamlit UI components for the merge functionality.
"""

import streamlit as st
import pandas as pd
import logging
import re
from typing import Dict, List, Any, Optional
from .merge_ledgers import LedgerMerger

# Configure logging
logger = logging.getLogger(__name__)

def render_merge_ledgers_page():
    """Render the main merge ledgers page."""
    st.markdown("## 📊 Merge Books Ledgers")
    st.markdown("Combine multiple GST ledger Excel files from different systems into a single dataset for reconciliation.")
    
    # Initialize session state for merge functionality
    if 'merge_ledger_files' not in st.session_state:
        st.session_state.merge_ledger_files = []
    if 'merge_template' not in st.session_state:
        st.session_state.merge_template = "Others"
    if 'merge_column_mappings' not in st.session_state:
        st.session_state.merge_column_mappings = {}
    if 'merge_settings' not in st.session_state:
        st.session_state.merge_settings = {}
    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None
    if 'merge_summary' not in st.session_state:
        st.session_state.merge_summary = {}
    
    # Initialize merger
    merger = LedgerMerger()
    
    # File Upload Section
    st.markdown("### 📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload multiple Excel files (.xlsx or .xls)",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        key="merge_file_uploader"
    )
    
    if uploaded_files:
        # Enhanced Data Validation & Quality Checks
        with st.spinner("🔍 Validating files and analyzing data quality..."):
            validation_results = merger.validate_files_comprehensive(uploaded_files)
            valid_files, warnings = merger.validate_files(uploaded_files)
        
        # Display Data Quality Dashboard
        st.markdown("### 📊 Data Quality Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Quality Score", f"{validation_results['overall_quality_score']}/100")
        with col2:
            st.metric("Files Validated", validation_results['validation_summary']['successfully_validated'])
        with col3:
            st.metric("Total Issues", validation_results['total_issues'])
        with col4:
            st.metric("Total Warnings", validation_results['total_warnings'])
        
        # Show Quality Score with color coding
        quality_score = validation_results['overall_quality_score']
        if quality_score >= 80:
            st.success(f"🟢 Excellent data quality! Score: {quality_score}/100")
        elif quality_score >= 60:
            st.warning(f"🟡 Good data quality with some issues. Score: {quality_score}/100")
        else:
            st.error(f"🔴 Poor data quality. Score: {quality_score}/100")
        
        # Display Issues and Warnings
        if validation_results['total_issues'] > 0 or validation_results['total_warnings'] > 0:
            with st.expander("⚠️ Data Quality Issues & Warnings", expanded=True):
                for filename, file_data in validation_results['files'].items():
                    if 'error' in file_data:
                        st.error(f"**{filename}**: {file_data['error']}")
                    else:
                        if file_data['issues']:
                            st.error(f"**{filename}** - Issues:")
                            for issue in file_data['issues']:
                                st.write(f"• {issue}")
                        
                        if file_data['warnings']:
                            st.warning(f"**{filename}** - Warnings:")
                            for warning in file_data['warnings']:
                                st.write(f"• {warning}")
        
        # Show warnings from basic validation
        if warnings:
            for warning in warnings:
                st.warning(warning)
        
        if valid_files:
            st.success(f"✅ {len(valid_files)} valid file(s) uploaded successfully!")
            
            # Enhanced File Analysis
            with st.expander("📈 Detailed File Analysis", expanded=False):
                for filename, file_data in validation_results['files'].items():
                    if 'error' not in file_data:
                        st.subheader(f"📄 {filename}")
                        
                        # Basic stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", file_data.get('total_rows', 0))
                        with col2:
                            st.metric("Columns", file_data.get('total_columns', 0))
                        with col3:
                            st.metric("Quality Score", f"{file_data.get('quality_score', 0)}/100")
                        
                        # Column analysis
                        if 'column_analysis' in file_data:
                            st.write("**Column Analysis:**")
                            for col_name, col_data in file_data['column_analysis'].items():
                                with st.expander(f"Column: {col_name}", expanded=False):
                                    st.write(f"**Type:** {col_data['data_type']}")
                                    st.write(f"**Non-null:** {col_data['non_null_count']}")
                                    st.write(f"**Null:** {col_data['null_count']} ({col_data['null_percentage']}%)")
                                    st.write(f"**Unique:** {col_data['unique_count']}")
                                    st.write(f"**Duplicates:** {col_data['duplicate_count']}")
                                    
                                    if col_data['statistics']:
                                        st.write("**Statistics:**")
                                        for stat_name, stat_value in col_data['statistics'].items():
                                            st.write(f"• {stat_name}: {stat_value}")
                        
                        # Pattern detection
                        if 'patterns' in file_data:
                            patterns = file_data['patterns']
                            
                            if patterns.get('gstin_patterns'):
                                st.write("**GSTIN Patterns:**")
                                for col_name, gstin_data in patterns['gstin_patterns'].items():
                                    st.write(f"• {col_name}: {gstin_data.get('validity_percentage', 0)}% valid ({gstin_data.get('valid_gstins', 0)}/{gstin_data.get('total_gstins', 0)})")
                            
                            if patterns.get('duplicate_patterns'):
                                st.write("**Duplicate Patterns:**")
                                for col_name, dup_data in patterns['duplicate_patterns'].items():
                                    st.write(f"• {col_name}: {dup_data.get('duplicate_percentage', 0)}% duplicates ({dup_data.get('duplicate_count', 0)} rows)")
                        
                        # Recommendations
                        if 'recommendations' in file_data and file_data['recommendations']:
                            st.write("**Recommendations:**")
                            for rec in file_data['recommendations']:
                                st.write(f"• {rec}")
            
            # Template Selection
            st.markdown("### 🎯 Template Selection")
            suggested_template = merger.detect_template_from_filename([f.name for f in valid_files])
            
            col1, col2 = st.columns([2, 1])
            with col1:
                template = st.radio(
                    "Select Template",
                    ["Tally", "SAP", "Others"],
                    index=["Tally", "SAP", "Others"].index(suggested_template),
                    help="Choose the template that best matches your data format"
                )
            with col2:
                st.info(f"💡 Suggested: **{suggested_template}**")
            
            st.session_state.merge_template = template
            st.session_state.merge_ledger_files = valid_files
            
            # Column Detection and Customization
            if st.button("🔍 Detect Columns", key="detect_columns_btn"):
                with st.spinner("Detecting columns..."):
                    # Use first file for column detection
                    sample_df = merger.read_file_data(valid_files[0])
                    detected_columns = merger.detect_columns(sample_df, template)
                    st.session_state.merge_column_mappings = detected_columns
                    st.success("Column detection completed!")
            
            # Show column mappings if available
            if st.session_state.merge_column_mappings:
                render_column_customization(merger, template)
            
            # Advanced Settings
            render_advanced_settings()
            
            # Merge Button with Progress Tracking
            if st.button("🔄 Merge Files", type="primary", key="merge_files_btn"):
                if not st.session_state.merge_column_mappings:
                    st.error("Please detect columns first!")
                else:
                    # Progress tracking setup
                    progress_container = st.container()
                    status_container = st.container()
                    
                    def progress_callback(status):
                        with progress_container:
                            progress_bar = st.progress(status['progress'] / 100)
                            st.write(f"**Progress:** {status['progress']:.1f}% - {status['status_message']}")
                            
                            if status['elapsed_time']:
                                st.write(f"**Elapsed Time:** {status['elapsed_time']:.1f}s")
                            if status['remaining_time']:
                                st.write(f"**Estimated Remaining:** {status['remaining_time']:.1f}s")
                    
                    # Add progress callback
                    merger.add_progress_callback(progress_callback)
                    
                    try:
                        with status_container:
                            st.info("🚀 Starting merge process with enhanced tracking...")
                        
                        merged_df = merger.merge_files(
                            valid_files,
                            template,
                            st.session_state.merge_column_mappings,
                            st.session_state.merge_settings
                        )
                        
                        if not merged_df.empty:
                            st.session_state.merged_data = merged_df
                            st.session_state.merge_summary = merger.get_merge_summary(merged_df)
                            
                            # Enhanced merge summary with all new features
                            st.markdown("### 📊 Merge Summary & Analysis")
                            summary = st.session_state.merge_summary
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Records", summary.get('total_records', 0))
                            with col2:
                                st.metric("Total Columns", summary.get('total_columns', 0))
                            with col3:
                                st.metric("Data Quality Score", f"{summary.get('data_quality_score', 0)}/100")
                            with col4:
                                st.metric("Duplicate Groups", summary.get('duplicate_groups', 0))
                            
                            # GSTIN Validation Summary
                            if 'GST Number Validation Status' in merged_df.columns:
                                gstin_status_counts = merged_df['GST Number Validation Status'].value_counts()
                                valid_gstins = gstin_status_counts.get('Valid', 0)
                                invalid_gstins = gstin_status_counts.get('Invalid', 0)
                                total_gstins = valid_gstins + invalid_gstins
                                
                                st.markdown("#### 🔍 GSTIN Validation Summary")
                                gstin_col1, gstin_col2, gstin_col3 = st.columns(3)
                                with gstin_col1:
                                    st.metric("Valid GSTINs", valid_gstins)
                                with gstin_col2:
                                    st.metric("Invalid GSTINs", invalid_gstins)
                                with gstin_col3:
                                    validity_percentage = round((valid_gstins / total_gstins * 100), 1) if total_gstins > 0 else 0
                                    st.metric("Validity %", f"{validity_percentage}%")
                                
                                # Show invalid GSTINs if any
                                if invalid_gstins > 0:
                                    invalid_gstins_df = merged_df[merged_df['GST Number Validation Status'] == 'Invalid'][
                                        ['GSTIN/UIN', 'GST Validation Details']
                                    ].head(10)
                                    with st.expander(f"⚠️ Invalid GSTINs ({invalid_gstins} total)", expanded=False):
                                        st.dataframe(invalid_gstins_df, use_container_width=True)
                                
                                # Tax totals
                                if any(key in summary for key in ['total_cgst', 'total_sgst', 'total_igst']):
                                    st.write("**Tax Totals:**")
                                    tax_col1, tax_col2, tax_col3 = st.columns(3)
                                    with tax_col1:
                                        st.metric("Total CGST", f"₹{summary.get('total_cgst', 0):,.2f}")
                                    with tax_col2:
                                        st.metric("Total SGST", f"₹{summary.get('total_sgst', 0):,.2f}")
                                    with tax_col3:
                                        st.metric("Total IGST", f"₹{summary.get('total_igst', 0):,.2f}")
                                
                                # Date range
                                if 'date_range' in summary and summary['date_range'] != 'N/A':
                                    st.write(f"**Date Range:** {summary['date_range']}")
                                
                                # Unique GSTINs
                                if 'unique_gstins' in summary:
                                    st.write(f"**Unique GSTINs:** {summary['unique_gstins']}")
                            
                            # Progress history
                            progress_history = merger.get_progress_history()
                            if progress_history:
                                with st.expander("📈 Progress History", expanded=False):
                                    for i, op in enumerate(progress_history):
                                        st.write(f"**{i+1}. {op['operation']}**")
                                        st.write(f"• Start: {op['start_time']}")
                                        if 'end_time' in op:
                                            st.write(f"• End: {op['end_time']}")
                                        st.write(f"• Status: {op['status']}")
                                        if 'duration' in op:
                                            st.write(f"• Duration: {op['duration']:.1f}s")
                                        if 'error' in op:
                                            st.error(f"• Error: {op['error']}")
                                        st.write("---")
                            
                            # Error report
                            error_summary = merger.get_error_summary()
                            if error_summary['total_errors'] > 0 or error_summary['total_warnings'] > 0:
                                with st.expander("⚠️ Error & Warning Report", expanded=False):
                                    st.write(f"**Total Errors:** {error_summary['total_errors']}")
                                    st.write(f"**Total Warnings:** {error_summary['total_warnings']}")
                                    st.write(f"**Recoverable Errors:** {error_summary['recoverable_errors']}")
                                    st.write(f"**Critical Errors:** {error_summary['critical_errors']}")
                                    
                                    if error_summary['total_errors'] > 0:
                                        st.error("⚠️ Some errors occurred during processing. Check the logs for details.")
                            
                            # Show preview of merged data
                            st.markdown("#### 📋 Merged Data Preview")
                            preview_df = merged_df.head(10)
                            st.dataframe(preview_df, use_container_width=True)
                            
                            # Show column information
                            st.markdown("#### 📊 Column Information")
                            col_info = []
                            for col in merged_df.columns:
                                col_type = str(merged_df[col].dtype)
                                non_null = merged_df[col].count()
                                null_count = len(merged_df) - non_null
                                col_info.append({
                                    'Column': col,
                                    'Type': col_type,
                                    'Non-Null': non_null,
                                    'Null': null_count,
                                    'Null %': round((null_count / len(merged_df)) * 100, 1)
                                })
                            
                            col_info_df = pd.DataFrame(col_info)
                            st.dataframe(col_info_df, use_container_width=True)
                            
                            st.success("🎉 Files merged successfully with enhanced analysis!")
                        else:
                            st.error("❌ No data could be merged. Please check your files and settings.")
                    except Exception as e:
                        logger.error(f"Error during merge: {e}")
                        st.error(f"❌ Error during merge: {str(e)}")
                        
                        # Show error details
                        error_summary = merger.get_error_summary()
                        if error_summary['total_errors'] > 0:
                            with st.expander("🔍 Error Details", expanded=True):
                                error_report = merger.get_error_report()
                                for error in error_report['errors']:
                                    st.error(f"**{error['error_type']}**: {error['message']}")
                                    if error.get('suggestion'):
                                        st.info(f"💡 Suggestion: {error['suggestion']}")
                    finally:
                        # Remove progress callback
                        merger.remove_progress_callback(progress_callback)
            
            # Show merged data and summary
            if st.session_state.merged_data is not None:
                render_merge_results(merger)
        
        else:
            st.error("❌ No valid files found. Please upload valid Excel files.")
    
    else:
        st.info("👆 Please upload Excel files to begin merging.")

def render_column_customization(merger: LedgerMerger, template: str):
    """Render column customization interface."""
    st.markdown("### 🔧 Column Customization")
    
    mappings = st.session_state.merge_column_mappings
    
    # Get all available columns from all files
    all_available_columns = []
    if st.session_state.merge_ledger_files:
        for file in st.session_state.merge_ledger_files:
            sample_df = merger.read_file_data(file)
            file_columns = sample_df.columns.tolist()
            all_available_columns.extend(file_columns)
        
        # Remove duplicates while preserving order
        all_available_columns = list(dict.fromkeys(all_available_columns))
    
    # Show all available columns with select all/deselect all
    st.markdown("#### All Available Columns")
    st.write(f"**Total columns found:** {len(all_available_columns)}")
    
    # Select all / Deselect all buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("Select All Columns", key="select_all_columns"):
            st.session_state.merge_column_mappings = {
                'standard_columns': all_available_columns.copy(),
                'tax_columns': [],
                'additional_columns': []
            }
            st.rerun()
    
    with col_btn2:
        if st.button("Deselect All Columns", key="deselect_all_columns"):
            st.session_state.merge_column_mappings = {
                'standard_columns': [],
                'tax_columns': [],
                'additional_columns': []
            }
            st.rerun()
    
    with col_btn3:
        if st.button("Auto-Detect Tax Columns", key="auto_detect_tax"):
            # Auto-detect tax columns
            tax_cols = [col for col in all_available_columns if re.search(r'cgst|sgst|igst', col.lower())]
            standard_cols = [col for col in all_available_columns if col not in tax_cols]
            st.session_state.merge_column_mappings = {
                'standard_columns': standard_cols,
                'tax_columns': tax_cols,
                'additional_columns': []
            }
            st.rerun()
    
    # Show columns in expandable sections
    with st.expander("View All Available Columns", expanded=True):
        for i, col in enumerate(all_available_columns, 1):
            # Check if it's a tax column
            is_tax = re.search(r'cgst|sgst|igst', col.lower())
            tax_indicator = " 🏷️ TAX" if is_tax else ""
            st.write(f"{i}. {col}{tax_indicator}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Standard Columns")
        standard_cols = st.multiselect(
            "Select Standard Columns",
            options=all_available_columns,
            default=mappings.get('standard_columns', []),
            key="standard_cols_select"
        )
        
        st.markdown("#### Tax Columns")
        tax_cols = st.multiselect(
            "Select Tax Columns",
            options=all_available_columns,
            default=mappings.get('tax_columns', []),
            key="tax_cols_select"
        )
    
    with col2:
        st.markdown("#### Additional Columns")
        additional_cols = st.multiselect(
            "Select Additional Columns",
            options=all_available_columns,
            default=mappings.get('additional_columns', []),
            key="additional_cols_select"
        )
        
        # Show template info
        if template in merger.templates:
            st.markdown("#### Template Information")
            template_info = merger.templates[template]
            st.write(f"**Standard Columns:** {', '.join(template_info['standard_columns'])}")
            st.write(f"**Tax Columns:** {', '.join(template_info['tax_columns'])}")
        
        # Show selection summary
        st.markdown("#### Selection Summary")
        st.write(f"**Standard Columns:** {len(standard_cols)}")
        st.write(f"**Tax Columns:** {len(tax_cols)}")
        st.write(f"**Additional Columns:** {len(additional_cols)}")
        st.write(f"**Total Selected:** {len(standard_cols) + len(tax_cols) + len(additional_cols)}")
    
    # Update mappings
    st.session_state.merge_column_mappings = {
        'standard_columns': standard_cols,
        'tax_columns': tax_cols,
        'additional_columns': additional_cols
    }
    
    # Quick actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Reset to Template", key="reset_template_btn"):
            if template in merger.templates:
                st.session_state.merge_column_mappings = merger.templates[template].copy()
                st.rerun()
    
    with col2:
        if st.button("🔄 Reset to Auto-Detection", key="reset_auto_btn"):
            # Re-detect columns
            sample_df = merger.read_file_data(st.session_state.merge_ledger_files[0])
            detected_columns = merger.detect_columns(sample_df, template)
            st.session_state.merge_column_mappings = detected_columns
            st.rerun()
    
    with col3:
        if st.button("✅ Use Current Selection", key="use_current_btn"):
            st.success("Column selection saved!")

def render_advanced_settings():
    """Render advanced merger settings."""
    with st.expander("⚙️ Advanced Settings", expanded=False):
        st.markdown("#### Value Sign Conversion")
        
        col1, col2 = st.columns(2)
        with col1:
            sign_conversion_type = st.selectbox(
                "Sign Conversion Type",
                ["No Conversion", "Positive to Negative", "Negative to Positive", "Both (Positive to Negative & Negative to Positive)"],
                key="sign_conversion_type"
            )
        
        with col2:
            if sign_conversion_type != "No Conversion":
                # Get all available columns from all files for sign conversion
                all_columns = []
                if st.session_state.merge_ledger_files:
                    from .merge_ledgers import LedgerMerger
                    merger = LedgerMerger()
                    for file in st.session_state.merge_ledger_files:
                        sample_df = merger.read_file_data(file)
                        file_columns = sample_df.columns.tolist()
                        all_columns.extend(file_columns)
                    
                    # Remove duplicates while preserving order
                    all_columns = list(dict.fromkeys(all_columns))
                
                sign_conversion_cols = st.multiselect(
                    "Apply to Columns",
                    options=all_columns,
                    key="sign_conversion_cols"
                )
            else:
                sign_conversion_cols = []
        
        st.markdown("#### Column Renaming")
        col_renames = {}
        # Get all available columns from all files for renaming
        all_columns = []
        if st.session_state.merge_ledger_files:
            from .merge_ledgers import LedgerMerger
            merger = LedgerMerger()
            for file in st.session_state.merge_ledger_files:
                sample_df = merger.read_file_data(file)
                file_columns = sample_df.columns.tolist()
                all_columns.extend(file_columns)
            
            # Remove duplicates while preserving order
            all_columns = list(dict.fromkeys(all_columns))
        
        rename_cols = st.multiselect(
            "Columns to Rename",
            options=all_columns,
            key="rename_cols_select"
        )
        
        for col in rename_cols:
            new_name = st.text_input(f"New name for '{col}'", key=f"rename_{col}")
            if new_name:
                col_renames[col] = new_name
        
        st.markdown("#### Duplicate Handling")
        enable_duplicate_detection = st.checkbox("Enable Duplicate Detection", key="enable_duplicate_detection_checkbox")
        if enable_duplicate_detection:
            # Get all available columns from all files for duplicate detection
            all_columns = []
            if st.session_state.merge_ledger_files:
                from .merge_ledgers import LedgerMerger
                merger = LedgerMerger()
                for file in st.session_state.merge_ledger_files:
                    sample_df = merger.read_file_data(file)
                    file_columns = sample_df.columns.tolist()
                    all_columns.extend(file_columns)
                
                # Remove duplicates while preserving order
                all_columns = list(dict.fromkeys(all_columns))
            
            deduplicate_key_cols = st.multiselect(
                "Key Columns for Duplicate Detection",
                options=all_columns,
                default=["Vch No.", "GSTIN/UIN"] if "Vch No." in all_columns else [],
                key="deduplicate_key_cols",
                help="Select columns to identify duplicates. Records with same values in these columns will be marked as duplicates."
            )
            st.info("💡 Duplicate detection will create 'Duplicate Status' and 'Duplicate Unique ID' columns instead of removing duplicates.")
        else:
            deduplicate_key_cols = []
        
        st.markdown("#### Missing Value Handling")
        missing_value_strategy = st.selectbox(
            "Missing Value Strategy",
            ["0", "NaN", "Empty"],
            index=0,
            key="missing_value_strategy"
        )
        
        st.markdown("#### Decimal Precision")
        decimal_precision = st.number_input(
            "Decimal Places for Tax Columns",
            min_value=0,
            max_value=6,
            value=2,
            key="decimal_precision"
        )
        
        # Update settings
        # Map sign conversion type to the correct value
        sign_conversion_mapping = {
            "No Conversion": None,
            "Positive to Negative": "positive_to_negative",
            "Negative to Positive": "negative_to_positive",
            "Both (Positive to Negative & Negative to Positive)": "both_positive_to_negative_and_negative_to_positive"
        }
        
        st.session_state.merge_settings = {
            'sign_conversion_type': sign_conversion_mapping.get(sign_conversion_type),
            'sign_conversion_columns': sign_conversion_cols,
            'column_renames': col_renames,
            'enable_duplicate_detection': enable_duplicate_detection,
            'deduplicate_key_columns': deduplicate_key_cols,
            'missing_value_strategy': missing_value_strategy,
            'decimal_precision': decimal_precision
        }

def render_merge_results(merger: LedgerMerger):
    """Render merge results and summary."""
    st.markdown("### 📊 Merge Results")
    
    merged_df = st.session_state.merged_data
    summary = st.session_state.merge_summary
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{summary.get('total_records', 0):,}")
    
    with col2:
        st.metric("Total CGST", f"₹{summary.get('total_cgst', 0):,.2f}")
    
    with col3:
        st.metric("Total SGST", f"₹{summary.get('total_sgst', 0):,.2f}")
    
    with col4:
        st.metric("Total IGST", f"₹{summary.get('total_igst', 0):,.2f}")
    
    # Additional info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Unique GSTINs", summary.get('unique_gstins', 0))
    
    with col2:
        st.metric("Date Range", summary.get('date_range', 'N/A'))
    
    # Data preview
    st.markdown("#### Data Preview (First 10 rows)")
    preview_df = merged_df.head(10)
    st.dataframe(preview_df, use_container_width=True)
    
    # Download options
    st.markdown("#### Download Options")
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel download
        excel_data = merger.export_merged_data()
        st.download_button(
            label="⬇️ Download Merged Data (Excel)",
            data=excel_data,
            file_name=f"Merged_Books_Data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_merged_excel"
        )
    
    with col2:
        # CSV download
        csv_data = merged_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Merged Data (CSV)",
            data=csv_data,
            file_name=f"Merged_Books_Data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_merged_csv"
        )
    
    # Integration with reconciliation
    st.markdown("#### Integration with Reconciliation")
    st.info("💡 The merged data is now available for use in the reconciliation process. You can upload this merged file in the main reconciliation section.")
    
    if st.button("🔄 Use for Reconciliation", type="secondary", key="use_for_reconciliation_btn"):
        # Store merged data for reconciliation
        st.session_state.merged_data_for_reconciliation = merged_df
        st.success("✅ Merged data is now available for reconciliation!")
        st.info("Go to the main reconciliation section and upload the merged data file, or use the data that's now stored in session state.")

def get_merged_data_for_reconciliation() -> Optional[pd.DataFrame]:
    """Get merged data for use in reconciliation."""
    return st.session_state.get('merged_data_for_reconciliation')
