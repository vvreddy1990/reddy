"""
Simple AI Insights Integration with Existing Reporting System

This module provides basic integration of AI insights with existing reports
without requiring all AI components to be available.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for Streamlit availability
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class SimpleAIInsightsIntegration:
    """
    Simple integration layer for AI insights with existing reporting system.
    
    This class provides basic AI insights functionality without requiring
    all AI components to be available.
    """
    
    def __init__(self, config_file: str = "reconciliation_settings.json"):
        """Initialize the simple AI insights integration."""
        self.config_file = config_file
        self.enabled = self._check_ai_insights_enabled()
        logger.info(f"Simple AI Insights Integration initialized (enabled: {self.enabled})")
    
    def _check_ai_insights_enabled(self) -> bool:
        """Check if AI insights are enabled in configuration."""
        try:
            import json
            import os
            
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('ai_ml_features', {}).get('ai_insights', {}).get('enabled', False)
            return False
        except Exception as e:
            logger.error(f"Error checking AI insights configuration: {e}")
            return False
    
    def render_ai_insights_section(
        self,
        reconciliation_results: Dict[str, Any],
        ai_enhancements: Dict[str, Any] = None,
        section_title: str = "🤖 AI Insights & Recommendations"
    ) -> bool:
        """
        Render AI insights section in Streamlit.
        
        Args:
            reconciliation_results: Original reconciliation results
            ai_enhancements: AI/ML enhancement results
            section_title: Title for the AI insights section
            
        Returns:
            bool: True if insights were successfully rendered, False otherwise
        """
        if not STREAMLIT_AVAILABLE:
            logger.warning("Streamlit not available, cannot render AI insights section")
            return False
        
        try:
            # Create expandable section for AI insights
            with st.expander(section_title, expanded=self.enabled):
                if not self.enabled:
                    st.info("""
                    🤖 **AI Insights & Recommendations**
                    
                    AI-powered insights are currently disabled. Enable them in Settings to get:
                    - Automated analysis of reconciliation patterns
                    - Data quality assessments  
                    - Risk identification and scoring
                    - Performance optimization suggestions
                    - Anomaly detection alerts
                    """)
                    return False
                
                # Generate basic insights from reconciliation results
                self._render_basic_insights(reconciliation_results, ai_enhancements)
                return True
                
        except Exception as e:
            logger.error(f"Error rendering AI insights section: {e}")
            if STREAMLIT_AVAILABLE:
                st.error("❌ Unable to generate AI insights. Using standard reports.")
            return False
    
    def _render_basic_insights(self, reconciliation_results: Dict[str, Any], ai_enhancements: Dict[str, Any] = None):
        """Render basic insights from reconciliation results."""
        if not STREAMLIT_AVAILABLE:
            return
        
        try:
            st.markdown("**AI-powered analysis of your reconciliation results**")
            
            # Basic metrics
            total_records = reconciliation_results.get('total_records', 0)
            matched_records = reconciliation_results.get('matched_records', 0)
            match_rate = reconciliation_results.get('match_rate', 0)
            
            if total_records > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", f"{total_records:,}")
                
                with col2:
                    st.metric("Matched Records", f"{matched_records:,}")
                
                with col3:
                    st.metric("Match Rate", f"{match_rate:.1f}%")
                
                # Generate basic insights
                st.markdown("### 💡 Basic Insights")
                
                if match_rate >= 90:
                    st.success("🟢 **Excellent Performance**: Your reconciliation has a high match rate of {:.1f}%. This indicates good data quality and matching processes.".format(match_rate))
                elif match_rate >= 75:
                    st.warning("🟡 **Good Performance**: Match rate of {:.1f}% is acceptable, but there's room for improvement. Consider reviewing unmatched records.".format(match_rate))
                else:
                    st.error("🔴 **Needs Attention**: Match rate of {:.1f}% is below optimal. Review data quality and matching criteria.".format(match_rate))
                
                # Basic recommendations
                st.markdown("### 🎯 Recommendations")
                
                unmatched_count = total_records - matched_records
                if unmatched_count > 0:
                    st.markdown(f"• **Review {unmatched_count:,} unmatched records** for potential data quality issues")
                
                if match_rate < 85:
                    st.markdown("• **Consider enabling advanced matching** in settings for better results")
                    st.markdown("• **Review supplier name variations** that might be causing match failures")
                
                if total_records > 1000:
                    st.markdown("• **Large dataset detected** - consider performance optimization settings")
                
                # AI enhancement insights if available
                if ai_enhancements:
                    st.markdown("### 🔬 AI Enhancement Results")
                    
                    if 'data_quality' in ai_enhancements:
                        dq = ai_enhancements['data_quality']
                        corrections = dq.get('corrections_made', 0)
                        if corrections > 0:
                            st.info(f"✨ **Data Quality**: {corrections} automatic corrections were applied to improve data quality")
                    
                    if 'smart_matching' in ai_enhancements:
                        sm = ai_enhancements['smart_matching']
                        enhanced_rate = sm.get('enhanced_match_rate', 0)
                        if enhanced_rate > match_rate:
                            improvement = enhanced_rate - match_rate
                            st.success(f"🚀 **Smart Matching**: Improved match rate by {improvement:.1f}% using AI algorithms")
                    
                    if 'anomaly_detection' in ai_enhancements:
                        ad = ai_enhancements['anomaly_detection']
                        anomalies = ad.get('total_anomalies', 0)
                        if anomalies > 0:
                            st.warning(f"⚠️ **Anomaly Detection**: {anomalies} potential anomalies detected requiring review")
            else:
                st.info("No reconciliation data available for analysis")
                
        except Exception as e:
            logger.error(f"Error rendering basic insights: {e}")
            if STREAMLIT_AVAILABLE:
                st.error("❌ Error displaying insights")
    
    def add_ai_insights_to_report(
        self,
        report_df: pd.DataFrame,
        reconciliation_results: Dict[str, Any],
        ai_enhancements: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Add basic AI insights as additional columns to existing report.
        
        Args:
            report_df: Original report DataFrame
            reconciliation_results: Reconciliation results
            ai_enhancements: AI enhancement results
            
        Returns:
            DataFrame with additional AI insights columns
        """
        if not self.enabled:
            return report_df
        
        try:
            enhanced_df = report_df.copy()
            
            # Add basic AI flags
            if ai_enhancements:
                # Add confidence scores if available
                if "predictive_scoring" in ai_enhancements:
                    confidence_scores = ai_enhancements["predictive_scoring"].get("confidence_scores", [])
                    if len(confidence_scores) == len(enhanced_df):
                        enhanced_df["AI_Confidence"] = [f"{score:.1f}%" for score in confidence_scores]
                
                # Add risk levels if available
                if "anomaly_detection" in ai_enhancements:
                    risk_scores = ai_enhancements["anomaly_detection"].get("risk_scores", {})
                    if risk_scores:
                        enhanced_df["AI_Risk_Level"] = enhanced_df.apply(
                            lambda row: self._get_risk_level(row, risk_scores), axis=1
                        )
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error adding AI insights to report: {e}")
            return report_df
    
    def _get_risk_level(self, row: pd.Series, risk_scores: Dict[str, float]) -> str:
        """Get risk level for a row."""
        try:
            gstin = row.get("Supplier GSTIN", "")
            risk_score = risk_scores.get(gstin, 0.0)
            
            if risk_score > 0.7:
                return "High"
            elif risk_score > 0.3:
                return "Medium"
            else:
                return "Low"
        except Exception:
            return "Unknown"


# Convenience functions
def render_ai_insights_for_reconciliation(
    reconciliation_results: Dict[str, Any],
    ai_enhancements: Dict[str, Any] = None,
    config_file: str = "reconciliation_settings.json"
) -> bool:
    """
    Convenience function to render AI insights for reconciliation results.
    """
    try:
        integration = SimpleAIInsightsIntegration(config_file)
        return integration.render_ai_insights_section(reconciliation_results, ai_enhancements)
    except Exception as e:
        logger.error(f"Error in convenience function: {e}")
        return False


def add_ai_insights_to_dataframe(
    df: pd.DataFrame,
    reconciliation_results: Dict[str, Any],
    ai_enhancements: Dict[str, Any] = None,
    config_file: str = "reconciliation_settings.json"
) -> pd.DataFrame:
    """
    Convenience function to add AI insights to a DataFrame.
    """
    try:
        integration = SimpleAIInsightsIntegration(config_file)
        return integration.add_ai_insights_to_report(df, reconciliation_results, ai_enhancements)
    except Exception as e:
        logger.error(f"Error adding AI insights to DataFrame: {e}")
        return df


def create_fallback_ai_insights_section(section_title: str = "🤖 AI Insights") -> None:
    """Create a fallback AI insights section."""
    if not STREAMLIT_AVAILABLE:
        return
        
    with st.expander(section_title, expanded=False):
        st.info("""
        🤖 **AI Insights & Recommendations**
        
        AI-powered insights provide automated analysis of your reconciliation results.
        
        **Features:**
        - Reconciliation pattern analysis
        - Data quality assessments
        - Performance recommendations
        - Basic anomaly detection
        
        **To Enable:**
        Go to Settings → AI Insights and enable the feature.
        """)