import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from .helpers import format_indian_currency, get_gstr2a_due_date, get_return_days_lapsed, get_compliance_status
try:
    from .ai_insights_integration import AIInsightsReportingIntegration, render_ai_insights_for_reconciliation
except ImportError:
    # Fallback to simple integration
    from .ai_insights_integration_simple import render_ai_insights_for_reconciliation
import io
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import openpyxl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def get_unique_unmapped_gst_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a report of individual GSTR-2A Only records for GST numbers that have never been mapped with any Books data.
    
    Args:
        df: DataFrame containing reconciliation data with Status column
    
    Returns:
        DataFrame with individual GSTR-2A Only records for unmapped GST numbers
    """
    try:
        if df is None or df.empty:
            logger.warning("Input DataFrame is empty or None")
            return pd.DataFrame()
        
        # Check required columns
        required_columns = ['Source Name', 'Supplier GSTIN', 'Supplier Trade Name', 'Supplier Legal Name', 'Status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Clean and filter data
        df_clean = df.copy()
        
        # Remove rows with missing GSTIN
        df_clean = df_clean.dropna(subset=['Supplier GSTIN'])
        df_clean['Supplier GSTIN'] = df_clean['Supplier GSTIN'].astype(str).str.strip().str.upper()
        
        # Filter out invalid GSTINs (basic validation)
        df_clean = df_clean[df_clean['Supplier GSTIN'].str.len() == 15]
        
        if df_clean.empty:
            logger.warning("No valid GSTINs found in the data")
            return pd.DataFrame()
        
        # Split data by source
        books_df = df_clean[df_clean['Source Name'] == 'Books'].copy()
        gstr2a_df = df_clean[df_clean['Source Name'] == 'GSTR-2A'].copy()
        
        if books_df.empty:
            logger.warning("No Books records found")
            return pd.DataFrame()
        
        if gstr2a_df.empty:
            logger.warning("No GSTR-2A records found")
            return pd.DataFrame()
        
        # Get all GSTINs that appear in Books (these are mapped)
        books_gstins = set(books_df['Supplier GSTIN'].unique())
        
        # Get GSTR-2A Only records
        gstr2a_only_df = gstr2a_df[gstr2a_df['Status'] == 'GSTR-2A Only'].copy()
        
        if gstr2a_only_df.empty:
            logger.info("No GSTR-2A Only records found")
            return pd.DataFrame()
        
        # Filter for GSTR-2A Only records where GSTIN has never appeared in Books
        unmapped_gstr2a_df = gstr2a_only_df[
            ~gstr2a_only_df['Supplier GSTIN'].isin(books_gstins)
        ].copy()
        
        if unmapped_gstr2a_df.empty:
            logger.info("All GSTR-2A Only GST numbers have at least one mapping in Books")
            return pd.DataFrame()
        
        # Select relevant columns for the report
        report_columns = [
            'Supplier GSTIN', 'Supplier Trade Name', 'Supplier Legal Name',
            'Invoice Number', 'Invoice Date', 'Total Invoice Value',
            'Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount',
            'Status'
        ]
        
        # Filter columns that exist in the dataframe
        available_columns = [col for col in report_columns if col in unmapped_gstr2a_df.columns]
        report_df = unmapped_gstr2a_df[available_columns].copy()
        
        # Rename columns for better display
        column_mapping = {
            'Supplier GSTIN': 'GST Number',
            'Supplier Trade Name': 'Trade Name',
            'Supplier Legal Name': 'Legal Name',
            'Invoice Number': 'Invoice Number',
            'Invoice Date': 'Invoice Date',
            'Total Invoice Value': 'Invoice Value',
            'Total IGST Amount': 'IGST Amount',
            'Total CGST Amount': 'CGST Amount',
            'Total SGST Amount': 'SGST Amount',
            'Status': 'Status'
        }
        
        report_df = report_df.rename(columns=column_mapping)
        
        # Clean up names (remove NaN, fill empty strings)
        if 'Trade Name' in report_df.columns:
            report_df['Trade Name'] = report_df['Trade Name'].fillna('').astype(str).str.strip()
        if 'Legal Name' in report_df.columns:
            report_df['Legal Name'] = report_df['Legal Name'].fillna('').astype(str).str.strip()
        
        # Sort by GST Number and Invoice Date
        sort_columns = ['GST Number']
        if 'Invoice Date' in report_df.columns:
            sort_columns.append('Invoice Date')
        report_df = report_df.sort_values(sort_columns).reset_index(drop=True)
        
        logger.info(f"Generated unique unmapped GST report with {len(report_df)} individual records")
        return report_df
        
    except Exception as e:
        logger.error(f"Error generating unique unmapped GST report: {str(e)}")
        return pd.DataFrame()

def filter_gst_report(df: pd.DataFrame, search_term: str = "", sort_by: str = "GST Number", sort_ascending: bool = True) -> pd.DataFrame:
    """
    Filter and sort the GST report based on user input.
    
    Args:
        df: DataFrame containing the GST report
        search_term: Search term to filter by GST Number or Name
        sort_by: Column to sort by
        sort_ascending: Whether to sort in ascending order
    
    Returns:
        Filtered and sorted DataFrame
    """
    try:
        if df is None or df.empty:
            return df
        
        filtered_df = df.copy()
        
        # Apply search filter
        if search_term and search_term.strip():
            search_term = search_term.strip().upper()
            search_columns = []
            if 'GST Number' in filtered_df.columns:
                search_columns.append('GST Number')
            if 'Trade Name' in filtered_df.columns:
                search_columns.append('Trade Name')
            if 'Legal Name' in filtered_df.columns:
                search_columns.append('Legal Name')
            if 'Invoice Number' in filtered_df.columns:
                search_columns.append('Invoice Number')
            
            if search_columns:
                mask = pd.Series([False] * len(filtered_df))
                for col in search_columns:
                    mask |= filtered_df[col].str.contains(search_term, case=False, na=False)
                filtered_df = filtered_df[mask]
        
        # Apply sorting
        if sort_by in filtered_df.columns:
            # For numeric columns, ensure proper numeric sorting
            numeric_columns = ['Invoice Value', 'IGST Amount', 'CGST Amount', 'SGST Amount']
            if sort_by in numeric_columns:
                # Convert to numeric, handling any non-numeric values
                filtered_df[sort_by] = pd.to_numeric(filtered_df[sort_by], errors='coerce')
                filtered_df = filtered_df.sort_values(sort_by, ascending=sort_ascending, na_position='last').reset_index(drop=True)
            elif sort_by == 'Invoice Date':
                # For date columns, convert to datetime and sort
                filtered_df[sort_by] = pd.to_datetime(filtered_df[sort_by], errors='coerce')
                filtered_df = filtered_df.sort_values(sort_by, ascending=sort_ascending, na_position='last').reset_index(drop=True)
            else:
                # For text columns, sort alphabetically
                filtered_df = filtered_df.sort_values(sort_by, ascending=sort_ascending, na_position='last').reset_index(drop=True)
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error filtering GST report: {str(e)}")
        return df

def render_ai_insights_section_for_reports(
    reconciliation_results: Dict[str, any] = None,
    ai_enhancements: Dict[str, any] = None
) -> None:
    """
    Render AI insights section for reports without modifying core report functionality.
    
    Args:
        reconciliation_results: Results from reconciliation process
        ai_enhancements: AI/ML enhancement results
    """
    try:
        # Only render if we have data to analyze
        if reconciliation_results or ai_enhancements:
            # Use the integration layer to render insights
            success = render_ai_insights_for_reconciliation(
                reconciliation_results or {},
                ai_enhancements
            )
            
            if not success:
                # Fallback: show a simple message
                with st.expander("🤖 AI Insights", expanded=False):
                    st.info("AI insights are currently unavailable. Using standard reports.")
        
    except Exception as e:
        logger.error(f"Error rendering AI insights section: {e}")
        # Fail silently to not disrupt main reports


def render_unique_gst_report(df: pd.DataFrame) -> None:
    """
    Render the unique unmapped GST report as a consolidated table with expandable details, Excel-like features, and download.
    """
    st.markdown("## 🔍 Unique Unmapped GST Numbers (Summary & Details)")
    st.markdown("This report shows GST numbers from GSTR-2A Only that have **never been mapped** with any Books data. Select a row to see all individual records for that GST number.")
    
    # Add AI insights section for this report
    try:
        # Prepare data for AI analysis
        report_data = get_unique_unmapped_gst_report(df)
        if not report_data.empty:
            reconciliation_results = {
                "total_records": len(df),
                "unmapped_gst_count": len(report_data),
                "report_type": "unique_unmapped_gst"
            }
            render_ai_insights_section_for_reports(reconciliation_results)
    except Exception as e:
        logger.error(f"Error adding AI insights to unique GST report: {e}")

    summary_df, details_df = get_unmapped_gst_summary_and_details(df)
    if summary_df.empty:
        st.info("✅ No unmapped GST numbers found. All GSTR-2A Only records have corresponding GST numbers in Books data.")
        return

    # --- Summary/Insights Section ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Unique Unmapped GST Numbers", summary_df['GST Number'].nunique())
    with col2:
        st.metric("Total Unmapped Records", details_df.shape[0])
    with col3:
        st.metric("Total IGST Amount", format_indian_currency(summary_df['Total IGST Amount'].replace('₹','', regex=True).replace(',','', regex=True).astype(float).sum()))
    with col4:
        st.metric("Total CGST Amount", format_indian_currency(summary_df['Total CGST Amount'].replace('₹','', regex=True).replace(',','', regex=True).astype(float).sum()))
    st.markdown("---")

    # Format currency columns in summary
    for col in ['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(lambda x: format_indian_currency(x) if pd.notna(x) else "₹0")

    # AGGrid options for summary
    gb = GridOptionsBuilder.from_dataframe(summary_df)
    gb.configure_default_column(editable=False, groupable=True, filter=True, sortable=True, resizable=True)
    gb.configure_column('GST Number', pinned='left')
    gb.configure_column('Count', type=['numericColumn', 'numberColumnFilter', 'customNumericFormat'], precision=0)
    for col in ['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']:
        if col in summary_df.columns:
            gb.configure_column(col, type=['textColumn'], cellStyle={"textAlign": "right"})
    grid_options = gb.build()

    # Prepare details mapping for expansion
    details_map = {}
    for gst in summary_df['GST Number']:
        detail_rows = details_df[details_df['Supplier GSTIN'] == gst].copy()
        if not detail_rows.empty:
            for col in ['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']:
                if col in detail_rows.columns:
                    detail_rows[col] = detail_rows[col].apply(lambda x: format_indian_currency(x) if pd.notna(x) else "₹0")
            details_map[gst] = detail_rows

    # AGGrid main table
    st.markdown("### 📋 Summary Table (Select a row to see details)")
    grid_response = AgGrid(
        summary_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=True,
        height=400,
        width='100%'
    )
    selected = grid_response['selected_rows']
    selected_gst = selected[0]['GST Number'] if selected else None

    # Show details for selected GST
    if selected_gst and selected_gst in details_map:
        st.markdown(f"#### Details for GST Number: {selected_gst}")
        st.dataframe(details_map[selected_gst].reset_index(drop=True), use_container_width=True, hide_index=True)

    # Download Excel with both summary and details
    st.markdown("### 📥 Download Report (Excel)")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        details_df.to_excel(writer, sheet_name='Details', index=False)
    excel_data = output.getvalue()
    st.download_button(
        label="⬇️ Download Unique Unmapped GST Report (Excel)",
        data=excel_data,
        file_name=f"unique_unmapped_gst_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download the summary and details as Excel sheets"
    )
    st.info("Tip: Use the filter and sort icons in the table header for Excel-like features. Select a row to see invoice-level details below.")

# For main Excel download integration

def get_unmapped_gst_summary_for_excel(df: pd.DataFrame):
    summary_df, _ = get_unmapped_gst_summary_and_details(df)
    return summary_df

def get_report_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get summary statistics for the unique GST report.
    
    Args:
        df: DataFrame containing reconciliation data
    
    Returns:
        Dictionary with summary statistics
    """
    try:
        report_df = get_unique_unmapped_gst_report(df)
        
        summary = {
            'total_unmapped': len(report_df),
            'total_records': report_df['Count'].sum() if not report_df.empty else 0,
            'with_trade_name': len(report_df[report_df['Trade Name'] != '']) if not report_df.empty else 0,
            'with_legal_name': len(report_df[report_df['Legal Name'] != '']) if not report_df.empty else 0,
            'with_both_names': len(report_df[(report_df['Trade Name'] != '') & (report_df['Legal Name'] != '')]) if not report_df.empty else 0,
            'unique_state_codes': len(report_df['GST Number'].str[:2].unique()) if not report_df.empty else 0,
            'total_igst_amount': report_df['Total IGST Amount'].sum() if not report_df.empty else 0,
            'total_cgst_amount': report_df['Total CGST Amount'].sum() if not report_df.empty else 0,
            'total_sgst_amount': report_df['Total SGST Amount'].sum() if not report_df.empty else 0,
            'avg_records_per_gst': report_df['Count'].mean() if not report_df.empty else 0,
            'max_records_per_gst': report_df['Count'].max() if not report_df.empty else 0
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting report summary: {str(e)}")
        return {
            'total_unmapped': 0,
            'total_records': 0,
            'with_trade_name': 0,
            'with_legal_name': 0,
            'with_both_names': 0,
            'unique_state_codes': 0,
            'total_igst_amount': 0,
            'total_cgst_amount': 0,
            'total_sgst_amount': 0,
            'avg_records_per_gst': 0,
            'max_records_per_gst': 0
        }

def get_enhanced_insights(df: pd.DataFrame) -> Dict[str, any]:
    """
    Generate enhanced insights and analytics for the unique GST report.
    
    Args:
        df: DataFrame containing reconciliation data
    
    Returns:
        Dictionary containing various analytics and insights
    """
    try:
        report_df = get_unique_unmapped_gst_report(df)
        
        if report_df.empty:
            return {
                'summary_stats': {},
                'state_analysis': pd.DataFrame(),
                'tax_analysis': pd.DataFrame(),
                'frequency_analysis': pd.DataFrame(),
                'risk_analysis': pd.DataFrame()
            }
        
        # Summary Statistics
        unique_gst_count = report_df['GST Number'].nunique()
        total_records = len(report_df)
        
        # Calculate tax totals if columns exist
        total_igst = report_df['IGST Amount'].sum() if 'IGST Amount' in report_df.columns else 0
        total_cgst = report_df['CGST Amount'].sum() if 'CGST Amount' in report_df.columns else 0
        total_sgst = report_df['SGST Amount'].sum() if 'SGST Amount' in report_df.columns else 0
        total_tax = total_igst + total_cgst + total_sgst
        
        # Calculate name completeness
        gst_with_trade_name = len(report_df[report_df['Trade Name'] != '']) if 'Trade Name' in report_df.columns else 0
        gst_with_legal_name = len(report_df[report_df['Legal Name'] != '']) if 'Legal Name' in report_df.columns else 0
        gst_with_both_names = len(report_df[(report_df['Trade Name'] != '') & (report_df['Legal Name'] != '')]) if 'Trade Name' in report_df.columns and 'Legal Name' in report_df.columns else 0
        
        summary_stats = {
            'total_unmapped_gst': unique_gst_count,
            'total_records': total_records,
            'total_igst': total_igst,
            'total_cgst': total_cgst,
            'total_sgst': total_sgst,
            'total_tax': total_tax,
            'avg_records_per_gst': total_records / unique_gst_count if unique_gst_count > 0 else 0,
            'max_records_per_gst': report_df['GST Number'].value_counts().max() if unique_gst_count > 0 else 0,
            'min_records_per_gst': report_df['GST Number'].value_counts().min() if unique_gst_count > 0 else 0,
            'gst_with_trade_name': gst_with_trade_name,
            'gst_with_legal_name': gst_with_legal_name,
            'gst_with_both_names': gst_with_both_names
        }
        
        # State-wise Analysis
        state_analysis = report_df['GST Number'].str[:2].value_counts().reset_index()
        state_analysis.columns = ['State Code', 'Count']
        state_analysis['Percentage'] = (state_analysis['Count'] / total_records * 100).round(2)
        
        # Tax Analysis
        tax_analysis = pd.DataFrame({
            'Tax Type': ['IGST', 'CGST', 'SGST', 'Total'],
            'Amount': [total_igst, total_cgst, total_sgst, total_tax]
        })
        tax_analysis['Percentage'] = (tax_analysis['Amount'] / total_tax * 100).round(2) if total_tax > 0 else 0
        
        # Frequency Analysis (records per GST number)
        frequency_analysis = report_df['GST Number'].value_counts().value_counts().sort_index().reset_index()
        frequency_analysis.columns = ['Records per GST', 'Number of GSTs']
        frequency_analysis['Percentage'] = (frequency_analysis['Number of GSTs'] / unique_gst_count * 100).round(2)
        
        # Risk Analysis (based on frequency and amounts)
        risk_analysis = report_df.copy()
        
        # Calculate risk score based on available columns
        risk_score_components = []
        
        # Frequency component (normalized by max frequency)
        gst_frequency = report_df['GST Number'].value_counts()
        max_frequency = gst_frequency.max() if len(gst_frequency) > 0 else 1
        frequency_score = gst_frequency / max_frequency * 0.5
        
        # Tax amount components
        igst_score = 0
        cgst_score = 0
        if 'IGST Amount' in risk_analysis.columns:
            max_igst = risk_analysis['IGST Amount'].max() if risk_analysis['IGST Amount'].max() > 0 else 1
            igst_score = risk_analysis['IGST Amount'] / max_igst * 0.3
        
        if 'CGST Amount' in risk_analysis.columns:
            max_cgst = risk_analysis['CGST Amount'].max() if risk_analysis['CGST Amount'].max() > 0 else 1
            cgst_score = risk_analysis['CGST Amount'] / max_cgst * 0.2
        
        # Combine scores
        risk_analysis['Risk Score'] = (frequency_score + igst_score + cgst_score).round(3)
        
        risk_analysis['Risk Level'] = pd.cut(
            risk_analysis['Risk Score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        risk_summary = risk_analysis['Risk Level'].value_counts().reset_index()
        risk_summary.columns = ['Risk Level', 'Count']
        risk_summary['Percentage'] = (risk_summary['Count'] / len(risk_analysis) * 100).round(2)
        
        return {
            'summary_stats': summary_stats,
            'state_analysis': state_analysis,
            'tax_analysis': tax_analysis,
            'frequency_analysis': frequency_analysis,
            'risk_analysis': risk_analysis,
            'risk_summary': risk_summary
        }
        
    except Exception as e:
        logger.error(f"Error generating enhanced insights: {str(e)}")
        return {
            'summary_stats': {},
            'state_analysis': pd.DataFrame(),
            'tax_analysis': pd.DataFrame(),
            'frequency_analysis': pd.DataFrame(),
            'risk_analysis': pd.DataFrame()
        }

def render_enhanced_insights(df: pd.DataFrame) -> None:
    """
    Render enhanced insights and summary tables for the unique GST report.
    
    Args:
        df: DataFrame containing reconciliation data
    """
    try:
        insights = get_enhanced_insights(df)
        
        if not insights['summary_stats']:
            st.info("No data available for enhanced insights.")
            return
        
        st.markdown("---")
        st.markdown("### 📊 Enhanced Analytics & Insights")
        
        # Summary Statistics Cards
        col1, col2, col3, col4 = st.columns(4)
        stats = insights['summary_stats']
        
        with col1:
            st.metric("Total Tax Impact", format_indian_currency(stats['total_tax']))
        with col2:
            st.metric("Avg Records/GST", f"{stats['avg_records_per_gst']:.1f}")
        with col3:
            high_risk_count = len(insights['risk_analysis'][insights['risk_analysis']['Risk Level'] == 'High']) if 'risk_analysis' in insights else 0
            st.metric("High Risk GSTs", f"{high_risk_count}")
        with col4:
            st.metric("Complete Names", f"{stats['gst_with_both_names']}")
        
        # Detailed Analysis Tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏛️ State-wise Distribution")
            if not insights['state_analysis'].empty:
                st.dataframe(
                    insights['state_analysis'],
                    use_container_width=True,
                    hide_index=True
                )
            
            st.markdown("#### 📈 Tax Distribution")
            if not insights['tax_analysis'].empty:
                # Format tax amounts for display
                tax_display = insights['tax_analysis'].copy()
                tax_display['Amount'] = tax_display['Amount'].apply(lambda x: format_indian_currency(x))
                st.dataframe(
                    tax_display,
                    use_container_width=True,
                    hide_index=True
                )
        
        with col2:
            st.markdown("#### 🔢 Frequency Analysis")
            if not insights['frequency_analysis'].empty:
                st.dataframe(
                    insights['frequency_analysis'],
                    use_container_width=True,
                    hide_index=True
                )
            
            st.markdown("#### ⚠️ Risk Analysis")
            if 'risk_summary' in insights and not insights['risk_summary'].empty:
                st.dataframe(
                    insights['risk_summary'],
                    use_container_width=True,
                    hide_index=True
                )
        
        # Top Risk Suppliers
        if 'risk_analysis' in insights:
            st.markdown("#### 🚨 Top 10 High-Risk Suppliers")
            high_risk_suppliers = insights['risk_analysis'][insights['risk_analysis']['Risk Level'] == 'High'].nlargest(10, 'Risk Score')
            
            if not high_risk_suppliers.empty:
                # Select available columns for display
                display_columns = ['GST Number']
                if 'Trade Name' in high_risk_suppliers.columns:
                    display_columns.append('Trade Name')
                if 'IGST Amount' in high_risk_suppliers.columns:
                    display_columns.append('IGST Amount')
                if 'CGST Amount' in high_risk_suppliers.columns:
                    display_columns.append('CGST Amount')
                display_columns.append('Risk Score')
                
                display_risk = high_risk_suppliers[display_columns].copy()
                
                # Format currency columns
                if 'IGST Amount' in display_risk.columns:
                    display_risk['IGST Amount'] = display_risk['IGST Amount'].apply(lambda x: format_indian_currency(x))
                if 'CGST Amount' in display_risk.columns:
                    display_risk['CGST Amount'] = display_risk['CGST Amount'].apply(lambda x: format_indian_currency(x))
                display_risk['Risk Score'] = display_risk['Risk Score'].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(
                    display_risk,
                    use_container_width=True,
                    hide_index=True
                )
        
        # Recommendations
        st.markdown("#### 💡 Recommendations")
        
        recommendations = []
        
        if stats['total_tax'] > 1000000:  # More than 10 lakhs
            recommendations.append("🔴 **High Priority**: Total tax impact exceeds ₹10 lakhs. Immediate supplier verification required.")
        
        if stats['avg_records_per_gst'] > 5:
            recommendations.append("🟡 **Medium Priority**: High frequency suppliers detected. Consider bulk supplier onboarding.")
        
        if stats['gst_with_both_names'] < stats['total_records'] * 0.8:
            recommendations.append("🟡 **Data Quality**: Less than 80% suppliers have complete names. Improve data collection.")
        
        if 'risk_analysis' in insights:
            high_risk_count = len(insights['risk_analysis'][insights['risk_analysis']['Risk Level'] == 'High'])
            if high_risk_count > stats['total_records'] * 0.2:
                recommendations.append("🔴 **Risk Alert**: More than 20% suppliers are high-risk. Prioritize verification.")
        
        if not recommendations:
            recommendations.append("✅ **Good Standing**: All metrics are within acceptable ranges.")
        
        for rec in recommendations:
            st.markdown(rec)
        
    except Exception as e:
        logger.error(f"Error rendering enhanced insights: {str(e)}")
        st.error(f"❌ Error generating enhanced insights: {str(e)}")

def get_unmapped_gst_summary_and_details(df: pd.DataFrame):
    """
    Returns two DataFrames:
    - summary_df: one row per unmapped GST (never mapped in Books), with count and total IGST/CGST/SGST
    - details_df: all GSTR-2A Only records for those GSTs
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    required_columns = ['Source Name', 'Supplier GSTIN', 'Supplier Trade Name', 'Supplier Legal Name', 'Status']
    if any(col not in df.columns for col in required_columns):
        return pd.DataFrame(), pd.DataFrame()
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=['Supplier GSTIN'])
    df_clean['Supplier GSTIN'] = df_clean['Supplier GSTIN'].astype(str).str.strip().str.upper()
    df_clean = df_clean[df_clean['Supplier GSTIN'].str.len() == 15]
    books_df = df_clean[df_clean['Source Name'] == 'Books']
    gstr2a_df = df_clean[df_clean['Source Name'] == 'GSTR-2A']
    books_gstins = set(books_df['Supplier GSTIN'].unique())
    gstr2a_only_df = gstr2a_df[gstr2a_df['Status'] == 'GSTR-2A Only'].copy()
    unmapped_gstr2a_df = gstr2a_only_df[~gstr2a_only_df['Supplier GSTIN'].isin(books_gstins)].copy()
    if unmapped_gstr2a_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Details DataFrame (individual records)
    details_df = unmapped_gstr2a_df.copy()
    # Summary DataFrame (grouped)
    summary_df = (
        unmapped_gstr2a_df.groupby(['Supplier GSTIN', 'Supplier Trade Name', 'Supplier Legal Name'], dropna=False)
        .agg(
            Count=('Supplier GSTIN', 'size'),
            IGST_Amount=('Total IGST Amount', 'sum'),
            CGST_Amount=('Total CGST Amount', 'sum'),
            SGST_Amount=('Total SGST Amount', 'sum')
        )
        .reset_index()
        .rename(columns={
            'Supplier GSTIN': 'GST Number',
            'Supplier Trade Name': 'Trade Name',
            'Supplier Legal Name': 'Legal Name',
            'IGST_Amount': 'Total IGST Amount',
            'CGST_Amount': 'Total CGST Amount',
            'SGST_Amount': 'Total SGST Amount',
        })
    )
    return summary_df, details_df

@st.cache_data
def add_gstr2a_compliance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add Compliance Report and Return Days Lapsed columns to the main reconciliation table (vectorized)."""
    df = df.copy()
    if 'Invoice Date' not in df.columns or 'Filing Date' not in df.columns:
        return df
    mask = (df['Source Name'] == 'GSTR-2A')
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
    df['Filing Date'] = pd.to_datetime(df['Filing Date'], errors='coerce')
    due_dates = df.loc[mask, 'Invoice Date'].apply(get_gstr2a_due_date)
    df.loc[mask, 'Due Date'] = due_dates
    df['IGST_num'] = pd.to_numeric(df.get('IGST', df.get('Total IGST Amount', 0)), errors='coerce').fillna(0)
    df['CGST_num'] = pd.to_numeric(df.get('CGST', df.get('Total CGST Amount', 0)), errors='coerce').fillna(0)
    df['SGST_num'] = pd.to_numeric(df.get('SGST', df.get('Total SGST Amount', 0)), errors='coerce').fillna(0)
    df['Total Tax'] = df['IGST_num'] + df['CGST_num'] + df['SGST_num']
    filing = df.loc[mask, 'Filing Date']
    due = df.loc[mask, 'Due Date']
    days_late = (filing - due).dt.days.clip(lower=0)
    df.loc[mask, 'Return Days Lapsed'] = days_late
    invalid = filing.isna() | due.isna()
    late = (filing > due)
    df.loc[mask, 'Compliance Report'] = np.where(
        invalid, "Invalid Filing Date",
        np.where(late, "Risky", "Compliant")
    )
    df.drop(['IGST_num', 'CGST_num', 'SGST_num', 'Total Tax', 'Due Date'], axis=1, inplace=True, errors='ignore')
    return df

@st.cache_data
def get_gstr2a_compliance_summary(df: pd.DataFrame, fy_start=None, fy_end=None):
    """Return GSTIN-wise, month-wise compliance summary for GSTR-2A data (vectorized)."""
    df = df.copy()
    df = df[df['Source Name'] == 'GSTR-2A']
    if df.empty:
        return pd.DataFrame()
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
    df['Filing Date'] = pd.to_datetime(df['Filing Date'], errors='coerce')
    df['Due Date'] = df['Invoice Date'].apply(get_gstr2a_due_date)
    df['Month'] = df['Invoice Date'].dt.strftime('%b %Y')
    df['IGST_num'] = pd.to_numeric(df.get('IGST', df.get('Total IGST Amount', 0)), errors='coerce').fillna(0)
    df['CGST_num'] = pd.to_numeric(df.get('CGST', df.get('Total CGST Amount', 0)), errors='coerce').fillna(0)
    df['SGST_num'] = pd.to_numeric(df.get('SGST', df.get('Total SGST Amount', 0)), errors='coerce').fillna(0)
    df['Total Tax'] = df['IGST_num'] + df['CGST_num'] + df['SGST_num']
    filing = df['Filing Date']
    due = df['Due Date']
    days_late = (filing - due).dt.days.clip(lower=0)
    invalid = filing.isna() | due.isna()
    late = (filing > due)
    df['Compliance'] = np.where(
        invalid, "Invalid Filing Date",
        np.where(late, "Risky", "Compliant")
    )
    # Dynamically determine all months present in the data
    all_months = pd.date_range(
        df['Invoice Date'].min().replace(day=1),
        df['Invoice Date'].max().replace(day=1),
        freq='MS'
    ).strftime('%b %Y').tolist()
    def month_status(subdf):
        risky = subdf[subdf['Compliance'] == 'Risky']
        if not risky.empty:
            return "Risky"
        elif (subdf['Compliance'] == 'Compliant').all():
            return 'Compliant'
        else:
            return ''
    summary = (
        df.groupby(['Supplier GSTIN', 'Supplier Trade Name', 'Supplier Legal Name', 'Month'])
        .apply(month_status)
        .unstack(fill_value='')
        .reindex(columns=all_months, fill_value='')
        .reset_index()
    )
    risky_mask = summary[all_months].applymap(lambda x: x == 'Risky')
    compliant_mask = summary[all_months] == 'Compliant'
    summary['Total Risky Months'] = risky_mask.sum(axis=1)
    summary['Total Compliant Months'] = compliant_mask.sum(axis=1)
    summary['Total Return Months'] = (summary[all_months] != '').sum(axis=1)
    summary = summary[['Supplier GSTIN', 'Supplier Trade Name', 'Supplier Legal Name'] + all_months + ['Total Return Months', 'Total Risky Months', 'Total Compliant Months']]
    return summary
