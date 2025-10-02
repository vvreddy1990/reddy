"""
AI Insights Integration with Existing Reporting System

This module provides seamless integration of AI-generated insights and recommendations
with the existing GST reconciliation reporting system without modifying core reports.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Streamlit and AI components, with fallbacks
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit not available, AI insights UI will be disabled")

try:
    from utils.ai_insights import AIInsightsGenerator
    AI_INSIGHTS_AVAILABLE = True
except ImportError:
    AI_INSIGHTS_AVAILABLE = False
    logger.warning("AI Insights not available")

try:
    from utils.ai_ml_config import AIMLConfigManager
    CONFIG_MANAGER_AVAILABLE = True
except ImportError:
    CONFIG_MANAGER_AVAILABLE = False
    logger.warning("AI ML Config Manager not available")

try:
    from utils.performance_monitor import PerformanceMonitor
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITOR_AVAILABLE = False
    logger.warning("Performance Monitor not available")


class AIInsightsReportingIntegration:
    """
    Integration layer for AI insights with existing reporting system.
    
    This class provides methods to seamlessly add AI insights to existing reports
    without modifying the core reporting functionality.
    """
    
    def __init__(self, config_file: str = "reconciliation_settings.json"):
        """
        Initialize the AI insights reporting integration.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.insights_generator = None
        self.config_manager = None
        self.performance_monitor = None
        self.enabled = False
        
        # Initialize components if available
        if CONFIG_MANAGER_AVAILABLE:
            try:
                self.config_manager = AIMLConfigManager(config_file)
                self.enabled = self._check_ai_insights_enabled()
            except Exception as e:
                logger.error(f"Failed to initialize config manager: {e}")
                self.enabled = False
        
        if PERFORMANCE_MONITOR_AVAILABLE:
            try:
                self.performance_monitor = PerformanceMonitor(max_total_time=200)
            except Exception as e:
                logger.error(f"Failed to initialize performance monitor: {e}")
        
        if self.enabled and AI_INSIGHTS_AVAILABLE:
            try:
                insights_config = self.config_manager.get_feature_config("ai_insights") if self.config_manager else {}
                self.insights_generator = AIInsightsGenerator(insights_config)
                logger.info("AI Insights Reporting Integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AI insights: {e}")
                self.enabled = False
                self.insights_generator = None
    
    def _check_ai_insights_enabled(self) -> bool:
        """Check if AI insights are enabled in configuration."""
        try:
            if not self.config_manager:
                return False
            return self.config_manager.is_feature_enabled("ai_insights")
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
        Render AI insights section in Streamlit without modifying existing reports.
        
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
            
        if not self.enabled or not self.insights_generator:
            # Show fallback message
            with st.expander(section_title, expanded=False):
                st.info("🤖 AI insights are currently disabled or unavailable. Enable them in Settings to get AI-powered analysis of your reconciliation results.")
            return False
        
        try:
            # Performance monitoring
            if self.performance_monitor:
                self.performance_monitor.start_timing("ai_insights_generation")
                
                # Check if we should skip due to time constraints
                if self.performance_monitor.should_skip_feature("ai_insights"):
                    st.info("⏱️ AI insights skipped due to performance constraints")
                    return False
            
            # Create expandable section for AI insights
            with st.expander(section_title, expanded=True):
                # User controls for AI insights
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown("**AI-powered analysis of your reconciliation results**")
                
                with col2:
                    refresh_insights = st.button("🔄 Refresh", key="refresh_ai_insights")
                
                with col3:
                    show_detailed = st.checkbox("Show Details", value=False, key="show_detailed_insights")
                
                # Generate insights
                if refresh_insights or "ai_insights_cache" not in st.session_state:
                    with st.spinner("Generating AI insights..."):
                        insights_summary = self.insights_generator.generate_reconciliation_summary(
                            reconciliation_results, ai_enhancements
                        )
                        st.session_state.ai_insights_cache = insights_summary
                else:
                    insights_summary = st.session_state.ai_insights_cache
                
                # Render insights
                self._render_insights_content(insights_summary, show_detailed)
                
                # Performance metrics
                if self.performance_monitor:
                    generation_time = self.performance_monitor.end_timing("ai_insights_generation")
                    if show_detailed:
                        st.caption(f"Generated in {generation_time:.2f}s")
                
                return True
                
        except Exception as e:
            logger.error(f"Error rendering AI insights section: {e}")
            if STREAMLIT_AVAILABLE:
                st.error("❌ Unable to generate AI insights. Using standard reports.")
            return False
    
    def _render_insights_content(self, insights_summary: Dict[str, Any], show_detailed: bool = False):
        """Render the actual insights content."""
        if not STREAMLIT_AVAILABLE:
            return
            
        try:
            # Overview section
            if "overview" in insights_summary:
                st.markdown("### 📊 Overview")
                st.info(insights_summary["overview"])
            
            # Key metrics
            if "key_metrics" in insights_summary and insights_summary["key_metrics"]:
                st.markdown("### 📈 Key Metrics")
                metrics = insights_summary["key_metrics"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", f"{metrics.get('total_records', 0):,}")
                with col2:
                    st.metric("Match Rate", f"{metrics.get('match_rate', 0):.1f}%")
                with col3:
                    st.metric("Processing Time", f"{metrics.get('processing_time', 0):.1f}s")
                with col4:
                    confidence = insights_summary.get("confidence_score", 0) * 100
                    st.metric("AI Confidence", f"{confidence:.0f}%")
            
            # Insights
            if "insights" in insights_summary and insights_summary["insights"]:
                st.markdown("### 💡 AI Insights")
                
                for insight in insights_summary["insights"]:
                    insight_dict = insight.to_dict() if hasattr(insight, 'to_dict') else insight
                    
                    # Color code by impact level
                    impact_colors = {
                        "high": "🔴",
                        "medium": "🟡", 
                        "low": "🟢"
                    }
                    
                    impact_icon = impact_colors.get(insight_dict.get("impact_level", "low"), "ℹ️")
                    
                    with st.container():
                        st.markdown(f"{impact_icon} **{insight_dict.get('title', 'Insight')}**")
                        st.markdown(insight_dict.get('description', 'No description available'))
                        
                        if show_detailed:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.caption(f"Confidence: {insight_dict.get('confidence', 'unknown').title()}")
                            with col2:
                                st.caption(f"Category: {insight_dict.get('category', 'general').title()}")
                        
                        st.markdown("---")
            
            # Recommendations
            if "recommendations" in insights_summary and insights_summary["recommendations"]:
                st.markdown("### 🎯 Recommendations")
                
                for rec in insights_summary["recommendations"]:
                    rec_dict = rec.to_dict() if hasattr(rec, 'to_dict') else rec
                    
                    # Priority indicators
                    priority_icons = {
                        "high": "🚨",
                        "medium": "⚠️",
                        "low": "💡"
                    }
                    
                    priority_icon = priority_icons.get(rec_dict.get("priority", "low"), "📝")
                    
                    with st.container():
                        st.markdown(f"{priority_icon} **{rec_dict.get('title', 'Recommendation')}**")
                        st.markdown(rec_dict.get('description', 'No description available'))
                        
                        # Action items
                        if "action_items" in rec_dict and rec_dict["action_items"]:
                            st.markdown("**Action Items:**")
                            for item in rec_dict["action_items"]:
                                st.markdown(f"• {item}")
                        
                        if show_detailed:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.caption(f"Priority: {rec_dict.get('priority', 'unknown').title()}")
                            with col2:
                                st.caption(f"Impact: {rec_dict.get('estimated_impact', 'unknown')}")
                        
                        st.markdown("---")
            
            # Fallback message if no insights
            if not insights_summary.get("insights") and not insights_summary.get("recommendations"):
                st.info("✅ No specific insights or recommendations at this time. Your reconciliation appears to be in good shape!")
                
        except Exception as e:
            logger.error(f"Error rendering insights content: {e}")
            if STREAMLIT_AVAILABLE:
                st.error("❌ Error displaying AI insights content")
    
    def add_ai_insights_to_report(
        self,
        report_df: pd.DataFrame,
        reconciliation_results: Dict[str, Any],
        ai_enhancements: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Add AI insights as additional columns to existing report without modifying core data.
        
        Args:
            report_df: Original report DataFrame
            reconciliation_results: Reconciliation results
            ai_enhancements: AI enhancement results
            
        Returns:
            DataFrame with additional AI insights columns
        """
        if not self.enabled or not self.insights_generator:
            return report_df
        
        try:
            enhanced_df = report_df.copy()
            
            # Add AI confidence scores if available
            if ai_enhancements and "predictive_scoring" in ai_enhancements:
                confidence_scores = ai_enhancements["predictive_scoring"].get("confidence_scores", [])
                if len(confidence_scores) == len(enhanced_df):
                    enhanced_df["AI_Confidence"] = confidence_scores
                    enhanced_df["AI_Confidence"] = enhanced_df["AI_Confidence"].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
                    )
            
            # Add risk scores if available
            if ai_enhancements and "anomaly_detection" in ai_enhancements:
                risk_scores = ai_enhancements["anomaly_detection"].get("risk_scores", {})
                if risk_scores:
                    enhanced_df["AI_Risk_Level"] = enhanced_df.apply(
                        lambda row: self._get_risk_level_for_row(row, risk_scores), axis=1
                    )
            
            # Add data quality flags
            if ai_enhancements and "data_quality" in ai_enhancements:
                quality_flags = ai_enhancements["data_quality"].get("quality_flags", [])
                if len(quality_flags) == len(enhanced_df):
                    enhanced_df["AI_Quality_Flag"] = quality_flags
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error adding AI insights to report: {e}")
            return report_df
    
    def _get_risk_level_for_row(self, row: pd.Series, risk_scores: Dict[str, float]) -> str:
        """Get risk level for a specific row."""
        try:
            # Try to match by GSTIN or supplier name
            gstin = row.get("Supplier GSTIN", "")
            supplier_name = row.get("Supplier Trade Name", "") or row.get("Supplier Legal Name", "")
            
            # Look up risk score
            risk_score = risk_scores.get(gstin) or risk_scores.get(supplier_name, 0.0)
            
            if risk_score > 0.7:
                return "High"
            elif risk_score > 0.3:
                return "Medium"
            else:
                return "Low"
                
        except Exception:
            return "Unknown"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for AI insights generation."""
        if self.performance_monitor:
            return self.performance_monitor.get_performance_report()
        return {}


# Convenience functions for easy integration
def render_ai_insights_for_reconciliation(
    reconciliation_results: Dict[str, Any],
    ai_enhancements: Dict[str, Any] = None,
    config_file: str = "reconciliation_settings.json"
) -> bool:
    """
    Convenience function to render AI insights for reconciliation results.
    
    Args:
        reconciliation_results: Reconciliation results
        ai_enhancements: AI enhancement results
        config_file: Configuration file path
        
    Returns:
        bool: True if insights were rendered successfully
    """
    try:
        integration = AIInsightsReportingIntegration(config_file)
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
    
    Args:
        df: Original DataFrame
        reconciliation_results: Reconciliation results
        ai_enhancements: AI enhancement results
        config_file: Configuration file path
        
    Returns:
        DataFrame with AI insights added
    """
    try:
        integration = AIInsightsReportingIntegration(config_file)
        return integration.add_ai_insights_to_report(df, reconciliation_results, ai_enhancements)
    except Exception as e:
        logger.error(f"Error adding AI insights to DataFrame: {e}")
        return df


def create_fallback_ai_insights_section(section_title: str = "🤖 AI Insights") -> None:
    """
    Create a fallback AI insights section when full AI functionality is not available.
    
    Args:
        section_title: Title for the section
    """
    if not STREAMLIT_AVAILABLE:
        return
        
    with st.expander(section_title, expanded=False):
        st.info("""
        🤖 **AI Insights & Recommendations**
        
        AI-powered insights are currently unavailable. This could be due to:
        - AI features are disabled in settings
        - Required AI components are not installed
        - Configuration issues
        
        **What AI Insights Provide:**
        - Automated analysis of reconciliation patterns
        - Data quality assessments
        - Risk identification and scoring
        - Performance optimization suggestions
        - Anomaly detection alerts
        
        **To Enable AI Insights:**
        1. Go to Settings → AI Insights
        2. Enable AI Insights & Recommendations
        3. Configure performance and display settings
        4. Save settings and refresh the page
        
        Your reconciliation will continue to work normally without AI insights.
        """)