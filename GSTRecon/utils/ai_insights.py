"""
AI Insights Generator for GST Reconciliation Enhancement

This module provides natural language generation of insights and recommendations
based on reconciliation results, data quality analysis, and risk assessments.
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightCategory(Enum):
    """Categories for AI-generated insights."""
    QUALITY = "quality"
    RISK = "risk"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    OPTIMIZATION = "optimization"


class ConfidenceLevel(Enum):
    """Confidence levels for insights."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AIInsight:
    """Structured AI-generated insight with metadata."""
    category: InsightCategory
    title: str
    description: str
    confidence: ConfidenceLevel
    data_sources: List[str]
    impact_level: str  # "high", "medium", "low"
    timestamp: datetime = field(default_factory=datetime.now)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to dictionary format."""
        return {
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence.value,
            "data_sources": self.data_sources,
            "impact_level": self.impact_level,
            "timestamp": self.timestamp.isoformat(),
            "supporting_data": self.supporting_data
        }


@dataclass
class AIRecommendation:
    """Structured AI-generated recommendation."""
    title: str
    description: str
    action_items: List[str]
    priority: str  # "high", "medium", "low"
    category: InsightCategory
    confidence: ConfidenceLevel
    estimated_impact: str
    data_sources: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary format."""
        return {
            "title": self.title,
            "description": self.description,
            "action_items": self.action_items,
            "priority": self.priority,
            "category": self.category.value,
            "confidence": self.confidence.value,
            "estimated_impact": self.estimated_impact,
            "data_sources": self.data_sources,
            "timestamp": self.timestamp.isoformat()
        }


class InsightTemplateEngine:
    """Template-based insight generation system."""
    
    def __init__(self):
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load default insight templates."""
        return {
            "data_quality": {
                "high_quality": {
                    "template": "Data quality is excellent with {clean_percentage:.1f}% of records requiring no corrections.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "low"
                },
                "medium_quality": {
                    "template": "Data quality is good with {clean_percentage:.1f}% clean records. {correction_count} corrections were applied automatically.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "medium"
                },
                "low_quality": {
                    "template": "Data quality issues detected: {clean_percentage:.1f}% clean records. {correction_count} corrections applied, {manual_review_count} items need manual review.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "high"
                }
            },
            "matching_performance": {
                "excellent": {
                    "template": "Matching performance is excellent with {match_rate:.1f}% successful matches and {confidence_score:.1f}% average confidence.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "low"
                },
                "good": {
                    "template": "Good matching performance: {match_rate:.1f}% matches found. {low_confidence_count} matches flagged for review due to low confidence.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "medium"
                },
                "poor": {
                    "template": "Matching challenges detected: Only {match_rate:.1f}% successful matches. {unmatched_count} records remain unmatched, requiring manual review.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "high"
                }
            },
            "risk_assessment": {
                "low_risk": {
                    "template": "Risk assessment shows low overall risk with {high_risk_count} suppliers flagged for attention.",
                    "confidence": ConfidenceLevel.MEDIUM,
                    "impact": "low"
                },
                "medium_risk": {
                    "template": "Moderate risk detected: {medium_risk_count} suppliers require monitoring, {high_risk_count} need immediate attention.",
                    "confidence": ConfidenceLevel.MEDIUM,
                    "impact": "medium"
                },
                "high_risk": {
                    "template": "High risk situation: {high_risk_count} suppliers flagged as high-risk, {anomaly_count} anomalies detected requiring immediate investigation.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "high"
                }
            },
            "anomaly_detection": {
                "no_anomalies": {
                    "template": "No significant anomalies detected in the reconciliation data.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "low"
                },
                "minor_anomalies": {
                    "template": "{anomaly_count} minor anomalies detected in tax amounts and patterns. Review recommended for validation.",
                    "confidence": ConfidenceLevel.MEDIUM,
                    "impact": "medium"
                },
                "major_anomalies": {
                    "template": "{anomaly_count} significant anomalies detected including {outlier_count} statistical outliers. Immediate investigation required.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "high"
                }
            },
            "performance_optimization": {
                "optimal": {
                    "template": "System performance is optimal. Processing completed in {processing_time:.1f} seconds with {cache_hit_rate:.1f}% cache efficiency.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "low"
                },
                "good": {
                    "template": "Good performance: {processing_time:.1f}s processing time. Cache hit rate of {cache_hit_rate:.1f}% suggests room for optimization.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "medium"
                },
                "needs_optimization": {
                    "template": "Performance optimization needed: {processing_time:.1f}s processing time approaching limits. Consider enabling more aggressive caching.",
                    "confidence": ConfidenceLevel.HIGH,
                    "impact": "high"
                }
            }
        }
    
    def generate_insight(
        self,
        category: str,
        subcategory: str,
        data: Dict[str, Any],
        data_sources: List[str]
    ) -> Optional[AIInsight]:
        """Generate an insight using templates."""
        try:
            if category not in self.templates or subcategory not in self.templates[category]:
                return None
            
            template_info = self.templates[category][subcategory]
            
            # Format the template with data
            description = template_info["template"].format(**data)
            
            # Create insight
            insight = AIInsight(
                category=InsightCategory(category.split("_")[0] if "_" in category else "quality"),
                title=f"{category.replace('_', ' ').title()} Analysis",
                description=description,
                confidence=template_info["confidence"],
                data_sources=data_sources,
                impact_level=template_info["impact"],
                supporting_data=data
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Error generating insight for {category}/{subcategory}: {e}")
            return None


class AIInsightsGenerator:
    """
    Main class for generating AI-powered insights and recommendations
    from GST reconciliation data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the AI insights generator.
        
        Args:
            config: Configuration dictionary for insight generation
        """
        self.config = config or {}
        self.template_engine = InsightTemplateEngine()
        self.insights_cache: Dict[str, List[AIInsight]] = {}
        self.recommendations_cache: Dict[str, List[AIRecommendation]] = {}
        
        # Configuration parameters
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.7)
        self.max_insights_per_category = self.config.get("max_insights_per_category", 3)
        self.enable_detailed_analysis = self.config.get("enable_detailed_analysis", True)
        
        logger.info("AI Insights Generator initialized")
    
    def generate_reconciliation_summary(
        self,
        reconciliation_results: Dict[str, Any],
        ai_enhancements: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive reconciliation summary with AI insights.
        
        Args:
            reconciliation_results: Original reconciliation results
            ai_enhancements: AI/ML enhancement results
            
        Returns:
            Dictionary containing summary and insights
        """
        try:
            logger.info("Generating reconciliation summary with AI insights")
            
            # Initialize summary structure
            summary = {
                "overview": self._generate_overview(reconciliation_results),
                "insights": [],
                "recommendations": [],
                "key_metrics": self._extract_key_metrics(reconciliation_results),
                "confidence_score": 0.0,
                "generation_timestamp": datetime.now().isoformat()
            }
            
            # Generate insights from different data sources
            if ai_enhancements:
                summary["insights"].extend(self._generate_data_quality_insights(ai_enhancements))
                summary["insights"].extend(self._generate_matching_insights(ai_enhancements))
                summary["insights"].extend(self._generate_risk_insights(ai_enhancements))
                summary["insights"].extend(self._generate_anomaly_insights(ai_enhancements))
                summary["insights"].extend(self._generate_performance_insights(ai_enhancements))
            
            # Generate insights from base reconciliation data
            summary["insights"].extend(self._generate_base_reconciliation_insights(reconciliation_results))
            
            # Calculate overall confidence score
            summary["confidence_score"] = self._calculate_overall_confidence(summary["insights"])
            
            # Generate recommendations based on insights
            summary["recommendations"] = self._generate_recommendations_from_insights(summary["insights"])
            
            logger.info(f"Generated {len(summary['insights'])} insights and {len(summary['recommendations'])} recommendations")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating reconciliation summary: {e}")
            return {
                "overview": "Error generating AI insights",
                "insights": [],
                "recommendations": [],
                "key_metrics": {},
                "confidence_score": 0.0,
                "error": str(e),
                "generation_timestamp": datetime.now().isoformat()
            }
    
    def _generate_overview(self, reconciliation_results: Dict[str, Any]) -> str:
        """Generate a high-level overview of reconciliation results."""
        try:
            # Extract basic metrics
            total_records = reconciliation_results.get("total_records", 0)
            matched_records = reconciliation_results.get("matched_records", 0)
            unmatched_records = reconciliation_results.get("unmatched_records", 0)
            
            match_rate = (matched_records / total_records * 100) if total_records > 0 else 0
            
            # Generate overview based on performance
            if match_rate >= 90:
                overview = f"Excellent reconciliation results: {match_rate:.1f}% match rate with {matched_records:,} of {total_records:,} records successfully matched."
            elif match_rate >= 75:
                overview = f"Good reconciliation performance: {match_rate:.1f}% match rate. {unmatched_records:,} records require manual review."
            else:
                overview = f"Reconciliation challenges detected: {match_rate:.1f}% match rate. {unmatched_records:,} unmatched records need attention."
            
            return overview
            
        except Exception as e:
            logger.error(f"Error generating overview: {e}")
            return "Reconciliation completed with mixed results."
    
    def _extract_key_metrics(self, reconciliation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from reconciliation results."""
        try:
            return {
                "total_records": reconciliation_results.get("total_records", 0),
                "matched_records": reconciliation_results.get("matched_records", 0),
                "unmatched_records": reconciliation_results.get("unmatched_records", 0),
                "match_rate": reconciliation_results.get("match_rate", 0.0),
                "processing_time": reconciliation_results.get("processing_time", 0.0),
                "data_quality_score": reconciliation_results.get("data_quality_score", 0.0)
            }
        except Exception as e:
            logger.error(f"Error extracting key metrics: {e}")
            return {}
    
    def _generate_data_quality_insights(self, ai_enhancements: Dict[str, Any]) -> List[AIInsight]:
        """Generate insights about data quality."""
        insights = []
        
        try:
            data_quality = ai_enhancements.get("data_quality", {})
            if not data_quality:
                return insights
            
            # Calculate data quality metrics
            total_records = data_quality.get("total_processed", 0)
            corrections_made = data_quality.get("corrections_made", 0)
            manual_review_needed = data_quality.get("manual_review_needed", 0)
            
            if total_records > 0:
                clean_percentage = ((total_records - corrections_made) / total_records) * 100
                
                # Determine quality level and generate insight
                if clean_percentage >= 95:
                    subcategory = "high_quality"
                elif clean_percentage >= 80:
                    subcategory = "medium_quality"
                else:
                    subcategory = "low_quality"
                
                insight_data = {
                    "clean_percentage": clean_percentage,
                    "correction_count": corrections_made,
                    "manual_review_count": manual_review_needed
                }
                
                insight = self.template_engine.generate_insight(
                    "data_quality",
                    subcategory,
                    insight_data,
                    ["data_quality_engine"]
                )
                
                if insight:
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating data quality insights: {e}")
        
        return insights
    
    def _generate_matching_insights(self, ai_enhancements: Dict[str, Any]) -> List[AIInsight]:
        """Generate insights about matching performance."""
        insights = []
        
        try:
            smart_matching = ai_enhancements.get("smart_matching", {})
            predictive_scoring = ai_enhancements.get("predictive_scoring", {})
            
            if smart_matching or predictive_scoring:
                # Calculate matching metrics
                match_rate = smart_matching.get("enhanced_match_rate", 0.0)
                confidence_score = predictive_scoring.get("average_confidence", 0.0)
                low_confidence_count = predictive_scoring.get("low_confidence_matches", 0)
                unmatched_count = smart_matching.get("unmatched_count", 0)
                
                # Determine performance level
                if match_rate >= 90 and confidence_score >= 85:
                    subcategory = "excellent"
                elif match_rate >= 75 and confidence_score >= 70:
                    subcategory = "good"
                else:
                    subcategory = "poor"
                
                insight_data = {
                    "match_rate": match_rate,
                    "confidence_score": confidence_score,
                    "low_confidence_count": low_confidence_count,
                    "unmatched_count": unmatched_count
                }
                
                insight = self.template_engine.generate_insight(
                    "matching_performance",
                    subcategory,
                    insight_data,
                    ["smart_matching", "predictive_scoring"]
                )
                
                if insight:
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating matching insights: {e}")
        
        return insights
    
    def _generate_risk_insights(self, ai_enhancements: Dict[str, Any]) -> List[AIInsight]:
        """Generate insights about risk assessment."""
        insights = []
        
        try:
            anomaly_detection = ai_enhancements.get("anomaly_detection", {})
            if not anomaly_detection:
                return insights
            
            # Extract risk metrics
            risk_scores = anomaly_detection.get("risk_scores", {})
            high_risk_count = len([score for score in risk_scores.values() if score > 0.7])
            medium_risk_count = len([score for score in risk_scores.values() if 0.3 < score <= 0.7])
            
            # Determine risk level
            total_suppliers = len(risk_scores)
            if total_suppliers > 0:
                high_risk_ratio = high_risk_count / total_suppliers
                
                if high_risk_ratio <= 0.05:  # Less than 5% high risk
                    subcategory = "low_risk"
                elif high_risk_ratio <= 0.15:  # Less than 15% high risk
                    subcategory = "medium_risk"
                else:
                    subcategory = "high_risk"
                
                insight_data = {
                    "high_risk_count": high_risk_count,
                    "medium_risk_count": medium_risk_count,
                    "anomaly_count": anomaly_detection.get("total_anomalies", 0)
                }
                
                insight = self.template_engine.generate_insight(
                    "risk_assessment",
                    subcategory,
                    insight_data,
                    ["anomaly_detection"]
                )
                
                if insight:
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating risk insights: {e}")
        
        return insights
    
    def _generate_anomaly_insights(self, ai_enhancements: Dict[str, Any]) -> List[AIInsight]:
        """Generate insights about anomaly detection."""
        insights = []
        
        try:
            anomaly_detection = ai_enhancements.get("anomaly_detection", {})
            if not anomaly_detection:
                return insights
            
            anomaly_count = anomaly_detection.get("total_anomalies", 0)
            outlier_count = anomaly_detection.get("statistical_outliers", 0)
            
            # Determine anomaly severity
            if anomaly_count == 0:
                subcategory = "no_anomalies"
            elif anomaly_count <= 5 and outlier_count <= 2:
                subcategory = "minor_anomalies"
            else:
                subcategory = "major_anomalies"
            
            insight_data = {
                "anomaly_count": anomaly_count,
                "outlier_count": outlier_count
            }
            
            insight = self.template_engine.generate_insight(
                "anomaly_detection",
                subcategory,
                insight_data,
                ["anomaly_detection"]
            )
            
            if insight:
                insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating anomaly insights: {e}")
        
        return insights
    
    def _generate_performance_insights(self, ai_enhancements: Dict[str, Any]) -> List[AIInsight]:
        """Generate insights about system performance."""
        insights = []
        
        try:
            # Extract performance data from various sources
            performance_data = {}
            
            # Check if performance metrics are available
            if "performance_metrics" in ai_enhancements:
                perf_metrics = ai_enhancements["performance_metrics"]
                processing_time = perf_metrics.get("total_time", 0.0)
                cache_hit_rate = perf_metrics.get("cache_hit_rate", 0.0)
                
                # Determine performance level
                if processing_time <= 60 and cache_hit_rate >= 80:
                    subcategory = "optimal"
                elif processing_time <= 120 and cache_hit_rate >= 50:
                    subcategory = "good"
                else:
                    subcategory = "needs_optimization"
                
                insight_data = {
                    "processing_time": processing_time,
                    "cache_hit_rate": cache_hit_rate
                }
                
                insight = self.template_engine.generate_insight(
                    "performance_optimization",
                    subcategory,
                    insight_data,
                    ["performance_monitor"]
                )
                
                if insight:
                    insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")
        
        return insights
    
    def _generate_base_reconciliation_insights(self, reconciliation_results: Dict[str, Any]) -> List[AIInsight]:
        """Generate insights from base reconciliation data."""
        insights = []
        
        try:
            # Generate basic reconciliation insights
            total_records = reconciliation_results.get("total_records", 0)
            matched_records = reconciliation_results.get("matched_records", 0)
            
            if total_records > 0:
                match_rate = (matched_records / total_records) * 100
                
                insight = AIInsight(
                    category=InsightCategory.PERFORMANCE,
                    title="Reconciliation Completion",
                    description=f"Processed {total_records:,} records with {match_rate:.1f}% match rate",
                    confidence=ConfidenceLevel.HIGH,
                    data_sources=["reconciliation_engine"],
                    impact_level="medium" if match_rate < 80 else "low",
                    supporting_data={
                        "total_records": total_records,
                        "matched_records": matched_records,
                        "match_rate": match_rate
                    }
                )
                
                insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error generating base reconciliation insights: {e}")
        
        return insights
    
    def _calculate_overall_confidence(self, insights: List[AIInsight]) -> float:
        """Calculate overall confidence score for all insights."""
        if not insights:
            return 0.0
        
        confidence_values = {
            ConfidenceLevel.HIGH: 0.9,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.5
        }
        
        total_confidence = sum(confidence_values[insight.confidence] for insight in insights)
        return total_confidence / len(insights)
    
    def _generate_recommendations_from_insights(self, insights: List[AIInsight]) -> List[AIRecommendation]:
        """Generate actionable recommendations based on insights."""
        recommendations = []
        
        try:
            for insight in insights:
                if insight.impact_level == "high":
                    # Generate high-priority recommendations for high-impact insights
                    if insight.category == InsightCategory.QUALITY:
                        rec = self._create_data_quality_recommendation(insight)
                    elif insight.category == InsightCategory.RISK:
                        rec = self._create_risk_mitigation_recommendation(insight)
                    elif insight.category == InsightCategory.PERFORMANCE:
                        rec = self._create_performance_recommendation(insight)
                    else:
                        rec = self._create_general_recommendation(insight)
                    
                    if rec:
                        recommendations.append(rec)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _create_data_quality_recommendation(self, insight: AIInsight) -> Optional[AIRecommendation]:
        """Create data quality improvement recommendation."""
        try:
            return AIRecommendation(
                title="Improve Data Quality",
                description="Address data quality issues to improve reconciliation accuracy",
                action_items=[
                    "Review and correct flagged data quality issues",
                    "Implement additional data validation rules",
                    "Train users on proper data entry procedures"
                ],
                priority="high",
                category=InsightCategory.QUALITY,
                confidence=insight.confidence,
                estimated_impact="Reduce manual review time by 30-40%",
                data_sources=insight.data_sources
            )
        except Exception as e:
            logger.error(f"Error creating data quality recommendation: {e}")
            return None
    
    def _create_risk_mitigation_recommendation(self, insight: AIInsight) -> Optional[AIRecommendation]:
        """Create risk mitigation recommendation."""
        try:
            return AIRecommendation(
                title="Address Risk Factors",
                description="Investigate and mitigate identified risk factors",
                action_items=[
                    "Review high-risk suppliers immediately",
                    "Investigate detected anomalies",
                    "Implement additional compliance checks"
                ],
                priority="high",
                category=InsightCategory.RISK,
                confidence=insight.confidence,
                estimated_impact="Reduce compliance risk by 50-60%",
                data_sources=insight.data_sources
            )
        except Exception as e:
            logger.error(f"Error creating risk mitigation recommendation: {e}")
            return None
    
    def _create_performance_recommendation(self, insight: AIInsight) -> Optional[AIRecommendation]:
        """Create performance optimization recommendation."""
        try:
            return AIRecommendation(
                title="Optimize System Performance",
                description="Implement performance improvements for faster processing",
                action_items=[
                    "Enable intelligent caching features",
                    "Optimize data preprocessing steps",
                    "Consider upgrading system resources"
                ],
                priority="medium",
                category=InsightCategory.PERFORMANCE,
                confidence=insight.confidence,
                estimated_impact="Reduce processing time by 20-30%",
                data_sources=insight.data_sources
            )
        except Exception as e:
            logger.error(f"Error creating performance recommendation: {e}")
            return None
    
    def _create_general_recommendation(self, insight: AIInsight) -> Optional[AIRecommendation]:
        """Create general recommendation for other insight types."""
        try:
            return AIRecommendation(
                title="Review Findings",
                description=f"Review and act on {insight.category.value} findings",
                action_items=[
                    f"Investigate {insight.title.lower()} findings",
                    "Implement corrective actions as needed",
                    "Monitor progress and results"
                ],
                priority="medium",
                category=insight.category,
                confidence=insight.confidence,
                estimated_impact="Improve overall reconciliation quality",
                data_sources=insight.data_sources
            )
        except Exception as e:
            logger.error(f"Error creating general recommendation: {e}")
            return None


# Processor function for AI/ML engine integration
def process_ai_insights(reconciliation_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process AI insights for the AI/ML engine.
    
    Args:
        reconciliation_data: Reconciliation results and AI enhancements
        config: Configuration for AI insights generation
        
    Returns:
        Dictionary containing generated insights and recommendations
    """
    try:
        logger.info("Processing AI insights generation")
        
        # Extract reconciliation results and AI enhancements
        original_results = reconciliation_data.get("original_results", {})
        ai_enhancements = reconciliation_data.get("ai_enhancements", {})
        
        # Initialize insights generator
        generator = AIInsightsGenerator(config)
        
        # Generate comprehensive summary
        summary = generator.generate_reconciliation_summary(original_results, ai_enhancements)
        
        # Convert insights and recommendations to dictionaries
        insights_dict = [insight.to_dict() for insight in summary.get("insights", [])]
        recommendations_dict = [rec.to_dict() for rec in summary.get("recommendations", [])]
        
        result = {
            "summary": summary.get("overview", ""),
            "insights": insights_dict,
            "recommendations": recommendations_dict,
            "key_metrics": summary.get("key_metrics", {}),
            "confidence_score": summary.get("confidence_score", 0.0),
            "generation_timestamp": summary.get("generation_timestamp", ""),
            "total_insights": len(insights_dict),
            "total_recommendations": len(recommendations_dict)
        }
        
        logger.info(f"Generated {len(insights_dict)} insights and {len(recommendations_dict)} recommendations")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in AI insights processing: {e}")
        return {
            "summary": "Error generating AI insights",
            "insights": [],
            "recommendations": [],
            "key_metrics": {},
            "confidence_score": 0.0,
            "error": str(e),
            "generation_timestamp": datetime.now().isoformat()
        }


class RecommendationEngine:
    """
    Advanced recommendation engine that generates context-aware,
    actionable recommendations based on data patterns and analysis.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the recommendation engine.
        
        Args:
            config: Configuration for recommendation generation
        """
        self.config = config or {}
        self.recommendation_rules = self._load_recommendation_rules()
        self.feedback_data: List[Dict[str, Any]] = []
        self.recommendation_history: List[AIRecommendation] = []
        
        # Configuration parameters
        self.max_recommendations = self.config.get("max_recommendations", 10)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.6)
        self.enable_priority_ranking = self.config.get("enable_priority_ranking", True)
        
        logger.info("Recommendation Engine initialized")
    
    def _load_recommendation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load rule-based recommendation templates."""
        return {
            "data_quality_rules": {
                "high_gstin_errors": {
                    "condition": lambda data: data.get("gstin_error_rate", 0) > 0.1,
                    "recommendation": {
                        "title": "Improve GSTIN Data Quality",
                        "description": "High GSTIN error rate detected. Implement validation at data entry.",
                        "action_items": [
                            "Add real-time GSTIN validation to data entry forms",
                            "Train users on proper GSTIN format requirements",
                            "Implement automated GSTIN checksum validation",
                            "Review and correct existing GSTIN errors"
                        ],
                        "priority": "high",
                        "category": InsightCategory.QUALITY,
                        "estimated_impact": "Reduce GSTIN errors by 80-90%"
                    }
                },
                "company_name_inconsistencies": {
                    "condition": lambda data: data.get("name_normalization_rate", 0) > 0.2,
                    "recommendation": {
                        "title": "Standardize Company Name Formats",
                        "description": "Significant company name variations detected. Implement standardization.",
                        "action_items": [
                            "Create company name standardization guidelines",
                            "Implement automated name normalization rules",
                            "Maintain master supplier database with standard names",
                            "Train users on consistent naming conventions"
                        ],
                        "priority": "medium",
                        "category": InsightCategory.QUALITY,
                        "estimated_impact": "Improve matching accuracy by 15-25%"
                    }
                },
                "missing_data_fields": {
                    "condition": lambda data: data.get("missing_field_rate", 0) > 0.05,
                    "recommendation": {
                        "title": "Address Missing Data Fields",
                        "description": "Critical data fields are missing in multiple records.",
                        "action_items": [
                            "Identify and flag records with missing critical fields",
                            "Implement mandatory field validation",
                            "Create data completion workflows",
                            "Establish data quality monitoring dashboards"
                        ],
                        "priority": "high",
                        "category": InsightCategory.QUALITY,
                        "estimated_impact": "Improve data completeness by 90%+"
                    }
                }
            },
            "matching_performance_rules": {
                "low_match_rate": {
                    "condition": lambda data: data.get("match_rate", 0) < 0.7,
                    "recommendation": {
                        "title": "Enhance Matching Algorithms",
                        "description": "Low match rate indicates need for algorithm improvements.",
                        "action_items": [
                            "Enable AI-enhanced name matching features",
                            "Adjust fuzzy matching sensitivity parameters",
                            "Implement multi-field matching strategies",
                            "Review and update matching rules regularly"
                        ],
                        "priority": "high",
                        "category": InsightCategory.PERFORMANCE,
                        "estimated_impact": "Increase match rate by 20-30%"
                    }
                },
                "high_false_positives": {
                    "condition": lambda data: data.get("false_positive_rate", 0) > 0.1,
                    "recommendation": {
                        "title": "Reduce False Positive Matches",
                        "description": "High false positive rate requires matching refinement.",
                        "action_items": [
                            "Implement confidence-based match filtering",
                            "Add additional validation criteria for matches",
                            "Enable manual review for low-confidence matches",
                            "Fine-tune matching algorithm parameters"
                        ],
                        "priority": "medium",
                        "category": InsightCategory.QUALITY,
                        "estimated_impact": "Reduce false positives by 50-70%"
                    }
                },
                "slow_processing": {
                    "condition": lambda data: data.get("processing_time", 0) > 150,
                    "recommendation": {
                        "title": "Optimize Processing Performance",
                        "description": "Processing time approaching limits. Optimization needed.",
                        "action_items": [
                            "Enable intelligent caching for repeated operations",
                            "Implement parallel processing where possible",
                            "Optimize database queries and indexing",
                            "Consider hardware resource upgrades"
                        ],
                        "priority": "high",
                        "category": InsightCategory.PERFORMANCE,
                        "estimated_impact": "Reduce processing time by 30-50%"
                    }
                }
            },
            "risk_management_rules": {
                "high_risk_suppliers": {
                    "condition": lambda data: data.get("high_risk_supplier_count", 0) > 5,
                    "recommendation": {
                        "title": "Implement Enhanced Supplier Monitoring",
                        "description": "Multiple high-risk suppliers require immediate attention.",
                        "action_items": [
                            "Conduct detailed review of high-risk suppliers",
                            "Implement enhanced due diligence procedures",
                            "Set up automated risk monitoring alerts",
                            "Establish supplier risk mitigation protocols"
                        ],
                        "priority": "high",
                        "category": InsightCategory.RISK,
                        "estimated_impact": "Reduce compliance risk by 60-80%"
                    }
                },
                "anomaly_patterns": {
                    "condition": lambda data: data.get("recurring_anomalies", 0) > 3,
                    "recommendation": {
                        "title": "Investigate Recurring Anomaly Patterns",
                        "description": "Recurring anomalies suggest systematic issues.",
                        "action_items": [
                            "Analyze root causes of recurring anomalies",
                            "Implement preventive controls for identified patterns",
                            "Enhance anomaly detection sensitivity",
                            "Create anomaly investigation workflows"
                        ],
                        "priority": "high",
                        "category": InsightCategory.RISK,
                        "estimated_impact": "Prevent 70-90% of recurring issues"
                    }
                },
                "compliance_gaps": {
                    "condition": lambda data: data.get("compliance_score", 1.0) < 0.8,
                    "recommendation": {
                        "title": "Address Compliance Gaps",
                        "description": "Compliance score indicates areas needing attention.",
                        "action_items": [
                            "Review and update compliance procedures",
                            "Implement automated compliance checking",
                            "Provide compliance training to relevant staff",
                            "Establish regular compliance monitoring"
                        ],
                        "priority": "high",
                        "category": InsightCategory.COMPLIANCE,
                        "estimated_impact": "Improve compliance score to 95%+"
                    }
                }
            },
            "optimization_rules": {
                "cache_underutilization": {
                    "condition": lambda data: data.get("cache_hit_rate", 0) < 0.3,
                    "recommendation": {
                        "title": "Optimize Caching Strategy",
                        "description": "Low cache hit rate indicates optimization opportunities.",
                        "action_items": [
                            "Enable intelligent caching features",
                            "Adjust cache size and retention policies",
                            "Implement predictive cache warming",
                            "Monitor and tune cache performance regularly"
                        ],
                        "priority": "medium",
                        "category": InsightCategory.OPTIMIZATION,
                        "estimated_impact": "Reduce processing time by 20-40%"
                    }
                },
                "resource_inefficiency": {
                    "condition": lambda data: data.get("memory_usage_ratio", 0) > 0.8,
                    "recommendation": {
                        "title": "Optimize Resource Usage",
                        "description": "High resource usage indicates need for optimization.",
                        "action_items": [
                            "Implement memory usage optimization",
                            "Enable garbage collection tuning",
                            "Optimize data processing algorithms",
                            "Consider system resource upgrades"
                        ],
                        "priority": "medium",
                        "category": InsightCategory.OPTIMIZATION,
                        "estimated_impact": "Improve system stability and performance"
                    }
                }
            }
        }
    
    def generate_context_aware_recommendations(
        self,
        analysis_data: Dict[str, Any],
        insights: List[AIInsight],
        historical_data: Dict[str, Any] = None
    ) -> List[AIRecommendation]:
        """
        Generate context-aware recommendations based on comprehensive analysis.
        
        Args:
            analysis_data: Current analysis results
            insights: Generated insights
            historical_data: Historical performance data
            
        Returns:
            List of prioritized recommendations
        """
        try:
            logger.info("Generating context-aware recommendations")
            
            recommendations = []
            
            # Generate rule-based recommendations
            rule_recommendations = self._apply_recommendation_rules(analysis_data)
            recommendations.extend(rule_recommendations)
            
            # Generate insight-based recommendations
            insight_recommendations = self._generate_insight_based_recommendations(insights)
            recommendations.extend(insight_recommendations)
            
            # Generate pattern-based recommendations
            if historical_data:
                pattern_recommendations = self._generate_pattern_based_recommendations(
                    analysis_data, historical_data
                )
                recommendations.extend(pattern_recommendations)
            
            # Apply priority ranking and filtering
            if self.enable_priority_ranking:
                recommendations = self._rank_recommendations_by_priority(recommendations)
            
            # Filter by confidence threshold
            recommendations = [
                rec for rec in recommendations 
                if self._get_confidence_score(rec.confidence) >= self.min_confidence_threshold
            ]
            
            # Limit to maximum recommendations
            recommendations = recommendations[:self.max_recommendations]
            
            # Track recommendations for feedback
            self.recommendation_history.extend(recommendations)
            
            logger.info(f"Generated {len(recommendations)} context-aware recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating context-aware recommendations: {e}")
            return []
    
    def _apply_recommendation_rules(self, analysis_data: Dict[str, Any]) -> List[AIRecommendation]:
        """Apply rule-based recommendation generation."""
        recommendations = []
        
        try:
            for rule_category, rules in self.recommendation_rules.items():
                for rule_name, rule_config in rules.items():
                    try:
                        # Check if rule condition is met
                        if rule_config["condition"](analysis_data):
                            # Create recommendation from rule
                            rec_template = rule_config["recommendation"]
                            
                            recommendation = AIRecommendation(
                                title=rec_template["title"],
                                description=rec_template["description"],
                                action_items=rec_template["action_items"].copy(),
                                priority=rec_template["priority"],
                                category=rec_template["category"],
                                confidence=ConfidenceLevel.HIGH,  # Rule-based are high confidence
                                estimated_impact=rec_template["estimated_impact"],
                                data_sources=[rule_category, "rule_engine"]
                            )
                            
                            recommendations.append(recommendation)
                            
                    except Exception as e:
                        logger.warning(f"Error applying rule {rule_name}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error in rule-based recommendation generation: {e}")
        
        return recommendations
    
    def _generate_insight_based_recommendations(self, insights: List[AIInsight]) -> List[AIRecommendation]:
        """Generate recommendations based on insights."""
        recommendations = []
        
        try:
            for insight in insights:
                if insight.impact_level == "high":
                    # Generate specific recommendations for high-impact insights
                    rec = self._create_targeted_recommendation(insight)
                    if rec:
                        recommendations.append(rec)
            
        except Exception as e:
            logger.error(f"Error generating insight-based recommendations: {e}")
        
        return recommendations
    
    def _generate_pattern_based_recommendations(
        self,
        current_data: Dict[str, Any],
        historical_data: Dict[str, Any]
    ) -> List[AIRecommendation]:
        """Generate recommendations based on historical patterns."""
        recommendations = []
        
        try:
            # Analyze trends and patterns
            trends = self._analyze_performance_trends(current_data, historical_data)
            
            for trend_type, trend_data in trends.items():
                if trend_data.get("declining", False):
                    # Generate recommendation for declining trends
                    rec = self._create_trend_improvement_recommendation(trend_type, trend_data)
                    if rec:
                        recommendations.append(rec)
            
        except Exception as e:
            logger.error(f"Error generating pattern-based recommendations: {e}")
        
        return recommendations
    
    def _analyze_performance_trends(
        self,
        current_data: Dict[str, Any],
        historical_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze performance trends from historical data."""
        trends = {}
        
        try:
            # Analyze match rate trend
            current_match_rate = current_data.get("match_rate", 0)
            historical_match_rate = historical_data.get("average_match_rate", 0)
            
            if historical_match_rate > 0:
                match_rate_change = (current_match_rate - historical_match_rate) / historical_match_rate
                trends["match_rate"] = {
                    "current": current_match_rate,
                    "historical": historical_match_rate,
                    "change_percent": match_rate_change * 100,
                    "declining": match_rate_change < -0.05  # 5% decline threshold
                }
            
            # Analyze processing time trend
            current_time = current_data.get("processing_time", 0)
            historical_time = historical_data.get("average_processing_time", 0)
            
            if historical_time > 0:
                time_change = (current_time - historical_time) / historical_time
                trends["processing_time"] = {
                    "current": current_time,
                    "historical": historical_time,
                    "change_percent": time_change * 100,
                    "declining": time_change > 0.1  # 10% increase is declining performance
                }
            
            # Analyze data quality trend
            current_quality = current_data.get("data_quality_score", 0)
            historical_quality = historical_data.get("average_data_quality", 0)
            
            if historical_quality > 0:
                quality_change = (current_quality - historical_quality) / historical_quality
                trends["data_quality"] = {
                    "current": current_quality,
                    "historical": historical_quality,
                    "change_percent": quality_change * 100,
                    "declining": quality_change < -0.05  # 5% decline threshold
                }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
        
        return trends
    
    def _create_targeted_recommendation(self, insight: AIInsight) -> Optional[AIRecommendation]:
        """Create targeted recommendation based on specific insight."""
        try:
            if insight.category == InsightCategory.QUALITY:
                return AIRecommendation(
                    title="Address Data Quality Issues",
                    description=f"Based on analysis: {insight.description}",
                    action_items=[
                        "Review flagged data quality issues",
                        "Implement corrective measures",
                        "Monitor data quality improvements",
                        "Update data validation rules"
                    ],
                    priority="high" if insight.impact_level == "high" else "medium",
                    category=insight.category,
                    confidence=insight.confidence,
                    estimated_impact="Improve overall data quality by 20-40%",
                    data_sources=insight.data_sources
                )
            
            elif insight.category == InsightCategory.RISK:
                return AIRecommendation(
                    title="Mitigate Identified Risks",
                    description=f"Risk assessment findings: {insight.description}",
                    action_items=[
                        "Investigate high-risk items immediately",
                        "Implement additional controls",
                        "Monitor risk indicators closely",
                        "Update risk assessment procedures"
                    ],
                    priority="high",
                    category=insight.category,
                    confidence=insight.confidence,
                    estimated_impact="Reduce risk exposure by 50-70%",
                    data_sources=insight.data_sources
                )
            
            elif insight.category == InsightCategory.PERFORMANCE:
                return AIRecommendation(
                    title="Optimize System Performance",
                    description=f"Performance analysis: {insight.description}",
                    action_items=[
                        "Implement performance optimizations",
                        "Enable advanced caching features",
                        "Monitor system resource usage",
                        "Consider infrastructure upgrades"
                    ],
                    priority="medium",
                    category=insight.category,
                    confidence=insight.confidence,
                    estimated_impact="Improve processing speed by 25-45%",
                    data_sources=insight.data_sources
                )
            
        except Exception as e:
            logger.error(f"Error creating targeted recommendation: {e}")
            return None
    
    def _create_trend_improvement_recommendation(
        self,
        trend_type: str,
        trend_data: Dict[str, Any]
    ) -> Optional[AIRecommendation]:
        """Create recommendation for improving declining trends."""
        try:
            trend_templates = {
                "match_rate": {
                    "title": "Improve Declining Match Rate",
                    "description": f"Match rate has declined by {abs(trend_data['change_percent']):.1f}% from historical average",
                    "action_items": [
                        "Review and update matching algorithms",
                        "Analyze causes of match failures",
                        "Implement enhanced name matching",
                        "Provide user training on data quality"
                    ],
                    "impact": "Restore match rate to historical levels"
                },
                "processing_time": {
                    "title": "Address Performance Degradation",
                    "description": f"Processing time has increased by {trend_data['change_percent']:.1f}% from historical average",
                    "action_items": [
                        "Identify performance bottlenecks",
                        "Optimize resource-intensive operations",
                        "Enable performance monitoring",
                        "Consider system optimization"
                    ],
                    "impact": "Restore processing time to acceptable levels"
                },
                "data_quality": {
                    "title": "Address Data Quality Decline",
                    "description": f"Data quality has declined by {abs(trend_data['change_percent']):.1f}% from historical average",
                    "action_items": [
                        "Investigate data quality issues",
                        "Implement additional validation",
                        "Provide data entry training",
                        "Monitor data quality metrics"
                    ],
                    "impact": "Restore data quality to historical standards"
                }
            }
            
            if trend_type in trend_templates:
                template = trend_templates[trend_type]
                
                return AIRecommendation(
                    title=template["title"],
                    description=template["description"],
                    action_items=template["action_items"],
                    priority="high",
                    category=InsightCategory.PERFORMANCE,
                    confidence=ConfidenceLevel.MEDIUM,
                    estimated_impact=template["impact"],
                    data_sources=["trend_analysis", "historical_data"]
                )
            
        except Exception as e:
            logger.error(f"Error creating trend improvement recommendation: {e}")
            return None
    
    def _rank_recommendations_by_priority(self, recommendations: List[AIRecommendation]) -> List[AIRecommendation]:
        """Rank recommendations by priority and impact."""
        try:
            # Define priority weights
            priority_weights = {"high": 3, "medium": 2, "low": 1}
            confidence_weights = {
                ConfidenceLevel.HIGH: 3,
                ConfidenceLevel.MEDIUM: 2,
                ConfidenceLevel.LOW: 1
            }
            
            # Calculate composite scores
            def calculate_score(rec: AIRecommendation) -> float:
                priority_score = priority_weights.get(rec.priority, 1)
                confidence_score = confidence_weights.get(rec.confidence, 1)
                return priority_score * confidence_score
            
            # Sort by composite score (descending)
            return sorted(recommendations, key=calculate_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error ranking recommendations: {e}")
            return recommendations
    
    def _get_confidence_score(self, confidence: ConfidenceLevel) -> float:
        """Convert confidence level to numeric score."""
        confidence_scores = {
            ConfidenceLevel.HIGH: 0.9,
            ConfidenceLevel.MEDIUM: 0.7,
            ConfidenceLevel.LOW: 0.5
        }
        return confidence_scores.get(confidence, 0.5)
    
    def collect_feedback(
        self,
        recommendation_id: str,
        feedback_type: str,
        feedback_data: Dict[str, Any]
    ):
        """
        Collect feedback on recommendations for continuous improvement.
        
        Args:
            recommendation_id: Unique identifier for the recommendation
            feedback_type: Type of feedback ("implemented", "rejected", "modified")
            feedback_data: Additional feedback information
        """
        try:
            feedback_entry = {
                "recommendation_id": recommendation_id,
                "feedback_type": feedback_type,
                "feedback_data": feedback_data,
                "timestamp": datetime.now().isoformat(),
                "user_id": feedback_data.get("user_id", "unknown")
            }
            
            self.feedback_data.append(feedback_entry)
            
            logger.info(f"Collected feedback for recommendation {recommendation_id}: {feedback_type}")
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
    
    def get_recommendation_effectiveness(self) -> Dict[str, Any]:
        """Calculate recommendation effectiveness metrics."""
        try:
            if not self.feedback_data:
                return {"total_recommendations": 0, "feedback_received": 0}
            
            total_feedback = len(self.feedback_data)
            implemented_count = len([f for f in self.feedback_data if f["feedback_type"] == "implemented"])
            rejected_count = len([f for f in self.feedback_data if f["feedback_type"] == "rejected"])
            modified_count = len([f for f in self.feedback_data if f["feedback_type"] == "modified"])
            
            return {
                "total_recommendations": len(self.recommendation_history),
                "feedback_received": total_feedback,
                "implementation_rate": implemented_count / total_feedback if total_feedback > 0 else 0,
                "rejection_rate": rejected_count / total_feedback if total_feedback > 0 else 0,
                "modification_rate": modified_count / total_feedback if total_feedback > 0 else 0,
                "feedback_rate": total_feedback / len(self.recommendation_history) if self.recommendation_history else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating recommendation effectiveness: {e}")
            return {"error": str(e)}


# Enhanced processor function with recommendation engine
def process_ai_insights_with_recommendations(
    reconciliation_data: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhanced AI insights processing with advanced recommendation engine.
    
    Args:
        reconciliation_data: Reconciliation results and AI enhancements
        config: Configuration for AI insights and recommendations
        
    Returns:
        Dictionary containing insights, recommendations, and analytics
    """
    try:
        logger.info("Processing AI insights with advanced recommendations")
        
        # Extract data components
        original_results = reconciliation_data.get("original_results", {})
        ai_enhancements = reconciliation_data.get("ai_enhancements", {})
        historical_data = reconciliation_data.get("historical_data", {})
        
        # Initialize components
        insights_generator = AIInsightsGenerator(config.get("insights", {}))
        recommendation_engine = RecommendationEngine(config.get("recommendations", {}))
        
        # Generate insights
        summary = insights_generator.generate_reconciliation_summary(original_results, ai_enhancements)
        insights = summary.get("insights", [])
        
        # Prepare analysis data for recommendations
        analysis_data = {
            **summary.get("key_metrics", {}),
            "gstin_error_rate": ai_enhancements.get("data_quality", {}).get("gstin_error_rate", 0),
            "name_normalization_rate": ai_enhancements.get("data_quality", {}).get("name_normalization_rate", 0),
            "missing_field_rate": ai_enhancements.get("data_quality", {}).get("missing_field_rate", 0),
            "false_positive_rate": ai_enhancements.get("predictive_scoring", {}).get("false_positive_rate", 0),
            "high_risk_supplier_count": ai_enhancements.get("anomaly_detection", {}).get("high_risk_suppliers", 0),
            "recurring_anomalies": ai_enhancements.get("anomaly_detection", {}).get("recurring_anomalies", 0),
            "compliance_score": ai_enhancements.get("anomaly_detection", {}).get("compliance_score", 1.0),
            "cache_hit_rate": ai_enhancements.get("performance_metrics", {}).get("cache_hit_rate", 0),
            "memory_usage_ratio": ai_enhancements.get("performance_metrics", {}).get("memory_usage_ratio", 0)
        }
        
        # Generate context-aware recommendations
        recommendations = recommendation_engine.generate_context_aware_recommendations(
            analysis_data, insights, historical_data
        )
        
        # Convert to dictionaries
        insights_dict = [insight.to_dict() for insight in insights]
        recommendations_dict = [rec.to_dict() for rec in recommendations]
        
        # Calculate analytics
        effectiveness_metrics = recommendation_engine.get_recommendation_effectiveness()
        
        result = {
            "summary": summary.get("overview", ""),
            "insights": insights_dict,
            "recommendations": recommendations_dict,
            "key_metrics": summary.get("key_metrics", {}),
            "confidence_score": summary.get("confidence_score", 0.0),
            "analytics": {
                "total_insights": len(insights_dict),
                "total_recommendations": len(recommendations_dict),
                "high_priority_recommendations": len([r for r in recommendations_dict if r["priority"] == "high"]),
                "recommendation_effectiveness": effectiveness_metrics
            },
            "generation_timestamp": summary.get("generation_timestamp", ""),
            "processing_info": {
                "insights_generated": len(insights_dict),
                "recommendations_generated": len(recommendations_dict),
                "analysis_data_points": len(analysis_data),
                "historical_data_available": bool(historical_data)
            }
        }
        
        logger.info(f"Enhanced processing complete: {len(insights_dict)} insights, {len(recommendations_dict)} recommendations")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced AI insights processing: {e}")
        return {
            "summary": "Error generating enhanced AI insights",
            "insights": [],
            "recommendations": [],
            "key_metrics": {},
            "confidence_score": 0.0,
            "analytics": {"error": str(e)},
            "error": str(e),
            "generation_timestamp": datetime.now().isoformat()
        }