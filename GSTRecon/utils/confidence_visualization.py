"""
Confidence Score Display and Visualization for GST Reconciliation

This module provides UI components and visualization functions for displaying
ML-based match confidence scores with color coding and interactive features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceVisualizer:
    """Handles visualization and display of confidence scores."""
    
    def __init__(self):
        self.confidence_colors = {
            "High": "#43a047",      # Green
            "Medium": "#fbc02d",    # Yellow/Orange
            "Low": "#e53935"        # Red
        }
        
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
    
    def get_confidence_level(self, score: float) -> str:
        """Get confidence level from score."""
        if score >= self.confidence_thresholds["high"]:
            return "High"
        elif score >= self.confidence_thresholds["medium"]:
            return "Medium"
        else:
            return "Low"
    
    def get_confidence_color(self, level: str) -> str:
        """Get color for confidence level."""
        return self.confidence_colors.get(level, "#757575")
    
    def format_confidence_badge(self, score: float, level: str = None) -> str:
        """Format confidence score as HTML badge."""
        if level is None:
            level = self.get_confidence_level(score)
        
        color = self.get_confidence_color(level)
        percentage = f"{score * 100:.1f}%"
        
        return f"""
        <span style="
            display: inline-block;
            padding: 0.25em 0.75em;
            border-radius: 12px;
            font-size: 0.9em;
            font-weight: 600;
            color: white;
            background-color: {color};
            margin: 2px;
        ">
            {percentage} ({level})
        </span>
        """
    
    def create_confidence_distribution_chart(
        self,
        confidence_scores: pd.Series,
        title: str = "Match Confidence Distribution"
    ) -> go.Figure:
        """Create histogram of confidence score distribution."""
        try:
            fig = go.Figure()
            
            # Create histogram
            fig.add_trace(go.Histogram(
                x=confidence_scores,
                nbinsx=20,
                name="Confidence Scores",
                marker_color="#1976d2",
                opacity=0.7
            ))
            
            # Add threshold lines
            for threshold_name, threshold_value in self.confidence_thresholds.items():
                color = self.get_confidence_color(threshold_name.title())
                fig.add_vline(
                    x=threshold_value,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"{threshold_name.title()} ({threshold_value})",
                    annotation_position="top"
                )
            
            fig.update_layout(
                title=title,
                xaxis_title="Confidence Score",
                yaxis_title="Number of Matches",
                showlegend=False,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confidence distribution chart: {e}")
            return go.Figure()
    
    def create_confidence_by_category_chart(
        self,
        matches_df: pd.DataFrame,
        confidence_scores: pd.Series,
        category_column: str = "Status"
    ) -> go.Figure:
        """Create confidence scores grouped by match category."""
        try:
            # Combine data
            df = matches_df.copy()
            df['confidence_score'] = confidence_scores
            df['confidence_level'] = df['confidence_score'].apply(self.get_confidence_level)
            
            # Group by category and confidence level
            grouped = df.groupby([category_column, 'confidence_level']).size().reset_index(name='count')
            
            # Create stacked bar chart
            fig = px.bar(
                grouped,
                x=category_column,
                y='count',
                color='confidence_level',
                color_discrete_map=self.confidence_colors,
                title="Confidence Levels by Match Category",
                labels={'count': 'Number of Matches', category_column: 'Match Category'}
            )
            
            fig.update_layout(height=400)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confidence by category chart: {e}")
            return go.Figure()
    
    def create_feature_importance_chart(
        self,
        feature_importance: Dict[str, float],
        title: str = "Feature Importance for Confidence Scoring"
    ) -> go.Figure:
        """Create horizontal bar chart of feature importance."""
        try:
            if not feature_importance:
                return go.Figure()
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features, importance = zip(*sorted_features)
            
            fig = go.Figure(go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker_color="#1976d2"
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(300, len(features) * 30)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance chart: {e}")
            return go.Figure()
    
    def create_confidence_trend_chart(
        self,
        matches_df: pd.DataFrame,
        confidence_scores: pd.Series,
        date_column: str = "Invoice Date"
    ) -> go.Figure:
        """Create confidence score trend over time."""
        try:
            # Combine data
            df = matches_df.copy()
            df['confidence_score'] = confidence_scores
            
            # Convert date column to datetime
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                df = df.dropna(subset=[date_column])
                
                # Group by date and calculate average confidence
                daily_confidence = df.groupby(df[date_column].dt.date)['confidence_score'].agg(['mean', 'count']).reset_index()
                daily_confidence.columns = ['date', 'avg_confidence', 'match_count']
                
                # Create line chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add confidence trend
                fig.add_trace(
                    go.Scatter(
                        x=daily_confidence['date'],
                        y=daily_confidence['avg_confidence'],
                        mode='lines+markers',
                        name='Average Confidence',
                        line=dict(color="#1976d2")
                    ),
                    secondary_y=False
                )
                
                # Add match count
                fig.add_trace(
                    go.Bar(
                        x=daily_confidence['date'],
                        y=daily_confidence['match_count'],
                        name='Match Count',
                        opacity=0.3,
                        marker_color="#43a047"
                    ),
                    secondary_y=True
                )
                
                # Update layout
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Average Confidence Score", secondary_y=False)
                fig.update_yaxes(title_text="Number of Matches", secondary_y=True)
                fig.update_layout(
                    title="Confidence Score Trend Over Time",
                    height=400
                )
                
                return fig
            else:
                return go.Figure()
                
        except Exception as e:
            logger.error(f"Error creating confidence trend chart: {e}")
            return go.Figure()


def render_confidence_score_section(
    matches_df: pd.DataFrame,
    predictive_results: Dict[str, Any],
    show_detailed: bool = True
) -> None:
    """
    Render the confidence score section in Streamlit.
    
    Args:
        matches_df: DataFrame with match results
        predictive_results: Results from predictive scoring
        show_detailed: Whether to show detailed analysis
    """
    try:
        if matches_df.empty or not predictive_results:
            st.info("No confidence score data available.")
            return
        
        confidence_scores = predictive_results.get('confidence_scores', pd.Series([]))
        detailed_scores = predictive_results.get('detailed_scores', [])
        summary = predictive_results.get('summary', {})
        model_info = predictive_results.get('model_info', {})
        
        if confidence_scores.empty:
            st.info("No confidence scores calculated.")
            return
        
        visualizer = ConfidenceVisualizer()
        
        # Header
        st.markdown("### 🎯 Match Confidence Analysis")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "High Confidence",
                summary.get('high_confidence', 0),
                delta=f"{summary.get('high_confidence', 0) / summary.get('total_matches', 1) * 100:.1f}%"
            )
        
        with col2:
            st.metric(
                "Medium Confidence",
                summary.get('medium_confidence', 0),
                delta=f"{summary.get('medium_confidence', 0) / summary.get('total_matches', 1) * 100:.1f}%"
            )
        
        with col3:
            st.metric(
                "Low Confidence",
                summary.get('low_confidence', 0),
                delta=f"{summary.get('low_confidence', 0) / summary.get('total_matches', 1) * 100:.1f}%"
            )
        
        with col4:
            st.metric(
                "Requires Review",
                summary.get('requires_review', 0),
                delta=f"{summary.get('requires_review', 0) / summary.get('total_matches', 1) * 100:.1f}%"
            )
        
        # Charts
        if show_detailed:
            chart_tabs = st.tabs(["Distribution", "By Category", "Feature Importance", "Trend"])
            
            with chart_tabs[0]:
                # Confidence distribution
                dist_chart = visualizer.create_confidence_distribution_chart(confidence_scores)
                st.plotly_chart(dist_chart, use_container_width=True)
            
            with chart_tabs[1]:
                # Confidence by category
                category_chart = visualizer.create_confidence_by_category_chart(matches_df, confidence_scores)
                st.plotly_chart(category_chart, use_container_width=True)
            
            with chart_tabs[2]:
                # Feature importance
                feature_importance = predictive_results.get('feature_importance', {})
                if feature_importance:
                    importance_chart = visualizer.create_feature_importance_chart(feature_importance)
                    st.plotly_chart(importance_chart, use_container_width=True)
                else:
                    st.info("Feature importance data not available.")
            
            with chart_tabs[3]:
                # Confidence trend
                trend_chart = visualizer.create_confidence_trend_chart(matches_df, confidence_scores)
                if trend_chart.data:
                    st.plotly_chart(trend_chart, use_container_width=True)
                else:
                    st.info("Date information not available for trend analysis.")
        
        # Low confidence matches table
        low_confidence_matches = predictive_results.get('low_confidence_matches', pd.DataFrame())
        if not low_confidence_matches.empty:
            st.markdown("#### 🔍 Low Confidence Matches (Requires Review)")
            
            # Add confidence level column for display
            low_confidence_matches['Confidence Level'] = low_confidence_matches['confidence_score'].apply(
                lambda x: visualizer.get_confidence_level(x)
            )
            
            # Format confidence score as percentage
            low_confidence_matches['Confidence %'] = (low_confidence_matches['confidence_score'] * 100).round(1)
            
            # Select relevant columns for display
            display_columns = [
                'Books_GSTIN', 'GSTR2A_GSTIN', 'Books_Legal_Name', 'GSTR2A_Legal_Name',
                'Books_Tax_Amount', 'GSTR2A_Tax_Amount', 'Confidence %', 'Confidence Level'
            ]
            
            # Filter columns that exist
            available_columns = [col for col in display_columns if col in low_confidence_matches.columns]
            
            if available_columns:
                st.dataframe(
                    low_confidence_matches[available_columns],
                    use_container_width=True
                )
            else:
                st.dataframe(low_confidence_matches, use_container_width=True)
            
            # Export option
            if st.button("📥 Export Low Confidence Matches"):
                csv = low_confidence_matches.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="low_confidence_matches.csv",
                    mime="text/csv"
                )
        
        # Model information
        if show_detailed and model_info:
            with st.expander("🤖 Model Information"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Model Details:**")
                    st.write(f"- Type: {model_info.get('model_type', 'Unknown')}")
                    st.write(f"- Trained: {'Yes' if model_info.get('is_trained', False) else 'No'}")
                    st.write(f"- scikit-learn: {'Available' if model_info.get('sklearn_available', False) else 'Not Available'}")
                    st.write(f"- XGBoost: {'Available' if model_info.get('xgboost_available', False) else 'Not Available'}")
                
                with col2:
                    metrics = model_info.get('metrics', {})
                    if metrics:
                        st.write("**Model Performance:**")
                        st.write(f"- Accuracy: {metrics.get('accuracy', 0):.3f}")
                        st.write(f"- Precision: {metrics.get('precision', 0):.3f}")
                        st.write(f"- Recall: {metrics.get('recall', 0):.3f}")
                        st.write(f"- F1 Score: {metrics.get('f1_score', 0):.3f}")
                        st.write(f"- Training Samples: {metrics.get('training_samples', 0)}")
    
    except Exception as e:
        logger.error(f"Error rendering confidence score section: {e}")
        st.error(f"Error displaying confidence scores: {e}")


def add_confidence_columns_to_dataframe(
    df: pd.DataFrame,
    confidence_scores: pd.Series,
    detailed_scores: List = None
) -> pd.DataFrame:
    """
    Add confidence score columns to the main dataframe for display.
    
    Args:
        df: Original dataframe
        confidence_scores: Series with confidence scores
        detailed_scores: List of detailed score objects
        
    Returns:
        DataFrame with added confidence columns
    """
    try:
        df_with_confidence = df.copy()
        
        # Add basic confidence score
        if len(confidence_scores) == len(df):
            df_with_confidence['Confidence Score'] = confidence_scores
            df_with_confidence['Confidence %'] = (confidence_scores * 100).round(1)
            
            # Add confidence level
            visualizer = ConfidenceVisualizer()
            df_with_confidence['Confidence Level'] = confidence_scores.apply(
                visualizer.get_confidence_level
            )
            
            # Add requires review flag
            df_with_confidence['Requires Review'] = confidence_scores < 0.6
        
        # Add detailed information if available
        if detailed_scores and len(detailed_scores) == len(df):
            df_with_confidence['Risk Factors'] = [
                '; '.join(score.risk_factors) if score.risk_factors else 'None'
                for score in detailed_scores
            ]
            
            df_with_confidence['Confidence Explanation'] = [
                score.explanation for score in detailed_scores
            ]
        
        return df_with_confidence
        
    except Exception as e:
        logger.error(f"Error adding confidence columns: {e}")
        return df


def create_confidence_filter_controls() -> Tuple[float, List[str], bool]:
    """
    Create Streamlit controls for filtering by confidence.
    
    Returns:
        Tuple of (min_confidence, selected_levels, show_review_only)
    """
    try:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Filter matches by minimum confidence score"
            )
        
        with col2:
            confidence_levels = st.multiselect(
                "Confidence Levels",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
                help="Select confidence levels to display"
            )
        
        with col3:
            show_review_only = st.checkbox(
                "Review Required Only",
                value=False,
                help="Show only matches that require manual review"
            )
        
        return min_confidence, confidence_levels, show_review_only
        
    except Exception as e:
        logger.error(f"Error creating confidence filter controls: {e}")
        return 0.0, ["High", "Medium", "Low"], False


def apply_confidence_filters(
    df: pd.DataFrame,
    confidence_scores: pd.Series,
    min_confidence: float,
    selected_levels: List[str],
    show_review_only: bool
) -> pd.DataFrame:
    """
    Apply confidence-based filters to dataframe.
    
    Args:
        df: DataFrame to filter
        confidence_scores: Confidence scores for filtering
        min_confidence: Minimum confidence threshold
        selected_levels: Selected confidence levels
        show_review_only: Whether to show only review-required matches
        
    Returns:
        Filtered DataFrame
    """
    try:
        if df.empty or confidence_scores.empty:
            return df
        
        # Create filter mask
        mask = pd.Series([True] * len(df), index=df.index)
        
        # Apply minimum confidence filter
        if len(confidence_scores) == len(df):
            mask &= confidence_scores >= min_confidence
        
        # Apply confidence level filter
        if selected_levels and len(confidence_scores) == len(df):
            visualizer = ConfidenceVisualizer()
            level_mask = confidence_scores.apply(visualizer.get_confidence_level).isin(selected_levels)
            mask &= level_mask
        
        # Apply review-only filter
        if show_review_only and len(confidence_scores) == len(df):
            mask &= confidence_scores < 0.6
        
        return df[mask]
        
    except Exception as e:
        logger.error(f"Error applying confidence filters: {e}")
        return df


def render_confidence_summary_cards(summary: Dict[str, Any]) -> None:
    """Render confidence summary as metric cards."""
    try:
        total_matches = summary.get('total_matches', 0)
        
        if total_matches == 0:
            st.info("No matches to analyze.")
            return
        
        # Calculate percentages
        high_pct = (summary.get('high_confidence', 0) / total_matches) * 100
        medium_pct = (summary.get('medium_confidence', 0) / total_matches) * 100
        low_pct = (summary.get('low_confidence', 0) / total_matches) * 100
        review_pct = (summary.get('requires_review', 0) / total_matches) * 100
        avg_confidence = summary.get('average_confidence', 0) * 100
        
        # Create cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #43a047;">{summary.get('high_confidence', 0)}</div>
                <div class="metric-label">High Confidence</div>
                <div style="font-size: 0.8em; color: #666;">{high_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #fbc02d;">{summary.get('medium_confidence', 0)}</div>
                <div class="metric-label">Medium Confidence</div>
                <div style="font-size: 0.8em; color: #666;">{medium_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #e53935;">{summary.get('low_confidence', 0)}</div>
                <div class="metric-label">Low Confidence</div>
                <div style="font-size: 0.8em; color: #666;">{low_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ff9800;">{summary.get('requires_review', 0)}</div>
                <div class="metric-label">Requires Review</div>
                <div style="font-size: 0.8em; color: #666;">{review_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #1976d2;">{avg_confidence:.1f}%</div>
                <div class="metric-label">Average Confidence</div>
                <div style="font-size: 0.8em; color: #666;">Overall Score</div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        logger.error(f"Error rendering confidence summary cards: {e}")


# CSS for confidence score styling
CONFIDENCE_CSS = """
<style>
.confidence-high {
    background-color: #43a047;
    color: white;
    padding: 0.25em 0.75em;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.9em;
}

.confidence-medium {
    background-color: #fbc02d;
    color: #222;
    padding: 0.25em 0.75em;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.9em;
}

.confidence-low {
    background-color: #e53935;
    color: white;
    padding: 0.25em 0.75em;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.9em;
}

.requires-review {
    background-color: #ff9800;
    color: white;
    padding: 0.25em 0.75em;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.9em;
}

.metric-card {
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    padding: 1.5rem;
    margin: 0.5rem 0;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
}

.metric-label {
    font-size: 1rem;
    color: #666;
}
</style>
"""