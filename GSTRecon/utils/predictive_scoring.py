"""
Predictive Match Scoring System for GST Reconciliation

This module implements ML-based confidence scoring for matches using lightweight models
to predict match quality and flag uncertain matches for manual review.
"""

import logging
import pickle
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. Predictive scoring will use rule-based fallback.")
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.info("XGBoost not available. Using Random Forest only.")
    XGBOOST_AVAILABLE = False


@dataclass
class MatchFeatures:
    """Features extracted from match data for ML prediction."""
    gstin_similarity: float = 0.0
    legal_name_similarity: float = 0.0
    trade_name_similarity: float = 0.0
    date_proximity_days: float = 0.0
    tax_amount_difference: float = 0.0
    tax_amount_ratio: float = 1.0
    invoice_number_similarity: float = 0.0
    supplier_frequency: int = 0
    historical_match_rate: float = 0.0
    data_quality_score: float = 1.0


@dataclass
class ConfidenceScore:
    """Match confidence score with explanation."""
    score: float
    confidence_level: str  # "High", "Medium", "Low"
    explanation: str
    feature_importance: Dict[str, float] = field(default_factory=dict)
    requires_review: bool = False
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    last_trained: datetime
    feature_importance: Dict[str, float] = field(default_factory=dict)


class FeatureExtractor:
    """Extract features from match data for ML prediction."""
    
    def __init__(self):
        self.supplier_stats = {}
        self.historical_data = []
    
    def extract_features(
        self,
        match_data: pd.DataFrame,
        books_data: pd.DataFrame = None,
        gstr2a_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Extract features from match data.
        
        Args:
            match_data: DataFrame with match results
            books_data: Original books data for context
            gstr2a_data: Original GSTR-2A data for context
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        for _, match in match_data.iterrows():
            features = self._extract_single_match_features(match, books_data, gstr2a_data)
            features_list.append(features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([
            {
                'gstin_similarity': f.gstin_similarity,
                'legal_name_similarity': f.legal_name_similarity,
                'trade_name_similarity': f.trade_name_similarity,
                'date_proximity_days': f.date_proximity_days,
                'tax_amount_difference': f.tax_amount_difference,
                'tax_amount_ratio': f.tax_amount_ratio,
                'invoice_number_similarity': f.invoice_number_similarity,
                'supplier_frequency': f.supplier_frequency,
                'historical_match_rate': f.historical_match_rate,
                'data_quality_score': f.data_quality_score
            }
            for f in features_list
        ])
        
        return feature_df
    
    def _extract_single_match_features(
        self,
        match: pd.Series,
        books_data: pd.DataFrame = None,
        gstr2a_data: pd.DataFrame = None
    ) -> MatchFeatures:
        """Extract features for a single match."""
        features = MatchFeatures()
        
        try:
            # GSTIN similarity
            if 'gstin_similarity' in match:
                features.gstin_similarity = float(match['gstin_similarity'])
            elif 'Books_GSTIN' in match and 'GSTR2A_GSTIN' in match:
                features.gstin_similarity = self._calculate_gstin_similarity(
                    match['Books_GSTIN'], match['GSTR2A_GSTIN']
                )
            
            # Name similarities
            if 'legal_name_similarity' in match:
                features.legal_name_similarity = float(match['legal_name_similarity'])
            elif 'Books_Legal_Name' in match and 'GSTR2A_Legal_Name' in match:
                features.legal_name_similarity = self._calculate_name_similarity(
                    match['Books_Legal_Name'], match['GSTR2A_Legal_Name']
                )
            
            if 'trade_name_similarity' in match:
                features.trade_name_similarity = float(match['trade_name_similarity'])
            elif 'Books_Trade_Name' in match and 'GSTR2A_Trade_Name' in match:
                features.trade_name_similarity = self._calculate_name_similarity(
                    match['Books_Trade_Name'], match['GSTR2A_Trade_Name']
                )
            
            # Date proximity
            if 'Books_Date' in match and 'GSTR2A_Date' in match:
                features.date_proximity_days = self._calculate_date_proximity(
                    match['Books_Date'], match['GSTR2A_Date']
                )
            
            # Tax amount features
            if 'Books_Tax_Amount' in match and 'GSTR2A_Tax_Amount' in match:
                books_amount = float(match['Books_Tax_Amount'])
                gstr2a_amount = float(match['GSTR2A_Tax_Amount'])
                
                features.tax_amount_difference = abs(books_amount - gstr2a_amount)
                if gstr2a_amount != 0:
                    features.tax_amount_ratio = books_amount / gstr2a_amount
                else:
                    features.tax_amount_ratio = 1.0 if books_amount == 0 else float('inf')
            
            # Invoice number similarity
            if 'Books_Invoice_Number' in match and 'GSTR2A_Invoice_Number' in match:
                features.invoice_number_similarity = self._calculate_invoice_similarity(
                    match['Books_Invoice_Number'], match['GSTR2A_Invoice_Number']
                )
            
            # Supplier frequency and historical data
            if 'Books_GSTIN' in match:
                gstin = match['Books_GSTIN']
                features.supplier_frequency = self._get_supplier_frequency(gstin)
                features.historical_match_rate = self._get_historical_match_rate(gstin)
            
            # Data quality score
            features.data_quality_score = self._calculate_data_quality_score(match)
            
        except Exception as e:
            logger.warning(f"Error extracting features for match: {e}")
        
        return features
    
    def _calculate_gstin_similarity(self, gstin1: str, gstin2: str) -> float:
        """Calculate GSTIN similarity score."""
        if pd.isna(gstin1) or pd.isna(gstin2):
            return 0.0
        
        gstin1 = str(gstin1).strip().upper()
        gstin2 = str(gstin2).strip().upper()
        
        if gstin1 == gstin2:
            return 1.0
        
        # Calculate character-level similarity
        if len(gstin1) == 15 and len(gstin2) == 15:
            matches = sum(c1 == c2 for c1, c2 in zip(gstin1, gstin2))
            return matches / 15.0
        
        return 0.0
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity using simple string matching."""
        if pd.isna(name1) or pd.isna(name2):
            return 0.0
        
        name1 = str(name1).strip().lower()
        name2 = str(name2).strip().lower()
        
        if name1 == name2:
            return 1.0
        
        # Simple Jaccard similarity on words
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_date_proximity(self, date1: Any, date2: Any) -> float:
        """Calculate date proximity in days."""
        try:
            if pd.isna(date1) or pd.isna(date2):
                return 365.0  # Large number for missing dates
            
            # Convert to datetime if needed
            if isinstance(date1, str):
                date1 = pd.to_datetime(date1)
            if isinstance(date2, str):
                date2 = pd.to_datetime(date2)
            
            diff = abs((date1 - date2).days)
            return float(diff)
            
        except Exception:
            return 365.0
    
    def _calculate_invoice_similarity(self, inv1: str, inv2: str) -> float:
        """Calculate invoice number similarity."""
        if pd.isna(inv1) or pd.isna(inv2):
            return 0.0
        
        inv1 = str(inv1).strip().upper()
        inv2 = str(inv2).strip().upper()
        
        if inv1 == inv2:
            return 1.0
        
        # Simple character similarity
        if len(inv1) > 0 and len(inv2) > 0:
            matches = sum(c1 == c2 for c1, c2 in zip(inv1, inv2))
            max_len = max(len(inv1), len(inv2))
            return matches / max_len
        
        return 0.0
    
    def _get_supplier_frequency(self, gstin: str) -> int:
        """Get supplier frequency from historical data."""
        return self.supplier_stats.get(gstin, {}).get('frequency', 1)
    
    def _get_historical_match_rate(self, gstin: str) -> float:
        """Get historical match rate for supplier."""
        return self.supplier_stats.get(gstin, {}).get('match_rate', 0.5)
    
    def _calculate_data_quality_score(self, match: pd.Series) -> float:
        """Calculate overall data quality score for the match."""
        score = 1.0
        
        # Penalize for missing critical fields
        critical_fields = ['Books_GSTIN', 'GSTR2A_GSTIN', 'Books_Tax_Amount', 'GSTR2A_Tax_Amount']
        missing_critical = sum(1 for field in critical_fields if pd.isna(match.get(field)))
        score -= (missing_critical * 0.2)
        
        # Penalize for missing optional fields
        optional_fields = ['Books_Legal_Name', 'GSTR2A_Legal_Name', 'Books_Date', 'GSTR2A_Date']
        missing_optional = sum(1 for field in optional_fields if pd.isna(match.get(field)))
        score -= (missing_optional * 0.1)
        
        return max(0.0, score)
    
    def update_supplier_stats(self, historical_data: pd.DataFrame):
        """Update supplier statistics from historical data."""
        try:
            if historical_data.empty:
                return
            
            # Calculate supplier frequency and match rates
            supplier_stats = {}
            
            for gstin in historical_data['Books_GSTIN'].dropna().unique():
                supplier_data = historical_data[historical_data['Books_GSTIN'] == gstin]
                
                supplier_stats[gstin] = {
                    'frequency': len(supplier_data),
                    'match_rate': supplier_data.get('is_match', pd.Series([0.5])).mean()
                }
            
            self.supplier_stats = supplier_stats
            logger.info(f"Updated statistics for {len(supplier_stats)} suppliers")
            
        except Exception as e:
            logger.error(f"Error updating supplier stats: {e}")


class PredictiveScorer:
    """
    ML-based match confidence scoring system using lightweight models.
    """
    
    def __init__(
        self,
        model_path: str = None,
        model_type: str = "random_forest",
        confidence_thresholds: Dict[str, float] = None
    ):
        """
        Initialize the predictive scorer.
        
        Args:
            model_path: Path to saved model files
            model_type: Type of ML model ("random_forest" or "xgboost")
            confidence_thresholds: Thresholds for confidence levels
        """
        self.model_path = model_path or ".cache/ai_ml/models"
        self.model_type = model_type
        self.confidence_thresholds = confidence_thresholds or {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.model_metrics = None
        
        # Feature names for consistency
        self.feature_names = [
            'gstin_similarity', 'legal_name_similarity', 'trade_name_similarity',
            'date_proximity_days', 'tax_amount_difference', 'tax_amount_ratio',
            'invoice_number_similarity', 'supplier_frequency', 'historical_match_rate',
            'data_quality_score'
        ]
        
        # Create model directory
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
        
        logger.info(f"PredictiveScorer initialized with {model_type} model")
    
    def calculate_match_confidence(
        self,
        matches: pd.DataFrame,
        books_data: pd.DataFrame = None,
        gstr2a_data: pd.DataFrame = None
    ) -> pd.Series:
        """
        Calculate confidence scores for matches.
        
        Args:
            matches: DataFrame with match results
            books_data: Original books data for context
            gstr2a_data: Original GSTR-2A data for context
            
        Returns:
            Series with confidence scores (0-1)
        """
        try:
            if matches.empty:
                return pd.Series([], dtype=float)
            
            # Extract features
            features_df = self.feature_extractor.extract_features(
                matches, books_data, gstr2a_data
            )
            
            # Use ML model if available and trained
            if self.is_trained and self.model is not None and SKLEARN_AVAILABLE:
                confidence_scores = self._predict_with_ml(features_df)
            else:
                # Fallback to rule-based scoring
                confidence_scores = self._predict_with_rules(features_df)
            
            return pd.Series(confidence_scores, index=matches.index)
            
        except Exception as e:
            logger.error(f"Error calculating match confidence: {e}")
            # Return default medium confidence for all matches
            return pd.Series([0.6] * len(matches), index=matches.index)
    
    def get_detailed_confidence_scores(
        self,
        matches: pd.DataFrame,
        books_data: pd.DataFrame = None,
        gstr2a_data: pd.DataFrame = None
    ) -> List[ConfidenceScore]:
        """
        Get detailed confidence scores with explanations.
        
        Args:
            matches: DataFrame with match results
            books_data: Original books data for context
            gstr2a_data: Original GSTR-2A data for context
            
        Returns:
            List of ConfidenceScore objects with detailed information
        """
        try:
            if matches.empty:
                return []
            
            # Get basic confidence scores
            confidence_scores = self.calculate_match_confidence(matches, books_data, gstr2a_data)
            
            # Extract features for explanations
            features_df = self.feature_extractor.extract_features(
                matches, books_data, gstr2a_data
            )
            
            detailed_scores = []
            
            for i, (_, match) in enumerate(matches.iterrows()):
                score = confidence_scores.iloc[i]
                features = features_df.iloc[i] if i < len(features_df) else None
                
                detailed_score = self._create_detailed_score(score, features, match)
                detailed_scores.append(detailed_score)
            
            return detailed_scores
            
        except Exception as e:
            logger.error(f"Error getting detailed confidence scores: {e}")
            return [
                ConfidenceScore(
                    score=0.6,
                    confidence_level="Medium",
                    explanation="Error calculating confidence score",
                    requires_review=True
                )
                for _ in range(len(matches))
            ]
    
    def _predict_with_ml(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict confidence scores using ML model."""
        try:
            # Ensure features are in correct order
            features_array = features_df[self.feature_names].values
            
            # Handle missing values
            features_array = np.nan_to_num(features_array, nan=0.0)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features_array = self.scaler.transform(features_array)
            
            # Predict probabilities
            if hasattr(self.model, 'predict_proba'):
                # For classification models, use probability of positive class
                probabilities = self.model.predict_proba(features_array)
                if probabilities.shape[1] > 1:
                    confidence_scores = probabilities[:, 1]  # Probability of match
                else:
                    confidence_scores = probabilities[:, 0]
            else:
                # For regression models, use direct prediction
                confidence_scores = self.model.predict(features_array)
                # Ensure scores are in [0, 1] range
                confidence_scores = np.clip(confidence_scores, 0.0, 1.0)
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            # Fallback to rule-based prediction
            return self._predict_with_rules(features_df)
    
    def _predict_with_rules(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict confidence scores using rule-based approach."""
        confidence_scores = []
        
        for _, features in features_df.iterrows():
            score = 0.3  # Lower base score for more variation
            
            # GSTIN similarity (high weight)
            gstin_sim = features.get('gstin_similarity', 0.0)
            score += 0.35 * gstin_sim
            
            # Name similarities (medium weight)
            legal_sim = features.get('legal_name_similarity', 0.0)
            trade_sim = features.get('trade_name_similarity', 0.0)
            name_sim = max(legal_sim, trade_sim)
            score += 0.25 * name_sim
            
            # Date proximity (medium weight)
            date_prox = features.get('date_proximity_days', 365.0)
            if date_prox <= 365.0:  # Only if we have valid date
                date_score = max(0.0, 1.0 - (date_prox / 90.0))  # Normalize to 90 days
                score += 0.15 * date_score
            
            # Tax amount ratio (medium weight)
            tax_ratio = features.get('tax_amount_ratio', 1.0)
            if tax_ratio > 0 and tax_ratio != float('inf'):
                # More sensitive to tax differences
                tax_score = max(0.0, 1.0 - min(2.0, abs(1.0 - tax_ratio)))
                score += 0.15 * tax_score
            
            # Invoice similarity (low weight)
            inv_sim = features.get('invoice_number_similarity', 0.0)
            score += 0.05 * inv_sim
            
            # Data quality (low weight)
            quality = features.get('data_quality_score', 1.0)
            score += 0.05 * quality
            
            # Ensure score is in valid range
            score = max(0.1, min(0.95, score))  # Avoid extreme values
            confidence_scores.append(score)
        
        return np.array(confidence_scores)
    
    def _create_detailed_score(
        self,
        score: float,
        features: pd.Series,
        match: pd.Series
    ) -> ConfidenceScore:
        """Create detailed confidence score with explanation."""
        # Determine confidence level
        if score >= self.confidence_thresholds["high"]:
            confidence_level = "High"
            requires_review = False
        elif score >= self.confidence_thresholds["medium"]:
            confidence_level = "Medium"
            requires_review = False
        else:
            confidence_level = "Low"
            requires_review = True
        
        # Generate explanation
        explanation_parts = []
        risk_factors = []
        
        if features is not None:
            # GSTIN similarity
            gstin_sim = features.get('gstin_similarity', 0.0)
            if gstin_sim >= 0.9:
                explanation_parts.append("Excellent GSTIN match")
            elif gstin_sim >= 0.7:
                explanation_parts.append("Good GSTIN similarity")
            else:
                explanation_parts.append("Poor GSTIN match")
                risk_factors.append("Low GSTIN similarity")
            
            # Name similarity
            legal_sim = features.get('legal_name_similarity', 0.0)
            trade_sim = features.get('trade_name_similarity', 0.0)
            name_sim = max(legal_sim, trade_sim)
            
            if name_sim >= 0.8:
                explanation_parts.append("Strong name match")
            elif name_sim >= 0.5:
                explanation_parts.append("Moderate name similarity")
            else:
                explanation_parts.append("Weak name match")
                risk_factors.append("Low name similarity")
            
            # Date proximity
            date_prox = features.get('date_proximity_days', 365.0)
            if date_prox <= 7:
                explanation_parts.append("Dates very close")
            elif date_prox <= 30:
                explanation_parts.append("Dates reasonably close")
            else:
                explanation_parts.append("Dates far apart")
                risk_factors.append("Large date difference")
            
            # Tax amount
            tax_ratio = features.get('tax_amount_ratio', 1.0)
            if 0.95 <= tax_ratio <= 1.05:
                explanation_parts.append("Tax amounts match well")
            elif 0.9 <= tax_ratio <= 1.1:
                explanation_parts.append("Tax amounts close")
            else:
                explanation_parts.append("Tax amounts differ significantly")
                risk_factors.append("Tax amount mismatch")
        
        explanation = "; ".join(explanation_parts) if explanation_parts else "Standard confidence calculation"
        
        # Get feature importance if model is available
        feature_importance = {}
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return ConfidenceScore(
            score=score,
            confidence_level=confidence_level,
            explanation=explanation,
            feature_importance=feature_importance,
            requires_review=requires_review,
            risk_factors=risk_factors
        )
    
    def train_on_feedback(
        self,
        feedback_data: pd.DataFrame,
        validation_split: float = 0.2
    ) -> ModelMetrics:
        """
        Train the ML model using historical match feedback.
        
        Args:
            feedback_data: DataFrame with match data and labels
            validation_split: Fraction of data to use for validation
            
        Returns:
            Model performance metrics
        """
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("scikit-learn not available. Cannot train ML model.")
                return None
            
            if feedback_data.empty or 'is_match' not in feedback_data.columns:
                logger.warning("No valid training data provided")
                return None
            
            logger.info(f"Training model with {len(feedback_data)} samples")
            
            # Extract features
            features_df = self.feature_extractor.extract_features(feedback_data)
            
            # Prepare training data
            X = features_df[self.feature_names].values
            y = feedback_data['is_match'].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_val_scaled)
            y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
            
            # Calculate metrics
            metrics = ModelMetrics(
                accuracy=accuracy_score(y_val, y_pred),
                precision=precision_score(y_val, y_pred, average='weighted'),
                recall=recall_score(y_val, y_pred, average='weighted'),
                f1_score=f1_score(y_val, y_pred, average='weighted'),
                training_samples=len(X_train),
                last_trained=datetime.now(),
                feature_importance=dict(zip(self.feature_names, self.model.feature_importances_))
            )
            
            self.model_metrics = metrics
            self.is_trained = True
            
            # Save model
            self._save_model()
            
            logger.info(f"Model trained successfully. Accuracy: {metrics.accuracy:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def get_confidence_level(self, score: float) -> str:
        """Get confidence level from score."""
        if score >= self.confidence_thresholds["high"]:
            return "High"
        elif score >= self.confidence_thresholds["medium"]:
            return "Medium"
        else:
            return "Low"
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}
    
    def _save_model(self):
        """Save trained model and scaler to disk."""
        try:
            model_file = os.path.join(self.model_path, f"predictive_model_{self.model_type}.pkl")
            scaler_file = os.path.join(self.model_path, "feature_scaler.pkl")
            metrics_file = os.path.join(self.model_path, "model_metrics.json")
            
            # Save model
            if self.model is not None:
                with open(model_file, 'wb') as f:
                    pickle.dump(self.model, f)
            
            # Save scaler
            if self.scaler is not None:
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            # Save metrics
            if self.model_metrics is not None:
                metrics_dict = {
                    'accuracy': self.model_metrics.accuracy,
                    'precision': self.model_metrics.precision,
                    'recall': self.model_metrics.recall,
                    'f1_score': self.model_metrics.f1_score,
                    'training_samples': self.model_metrics.training_samples,
                    'last_trained': self.model_metrics.last_trained.isoformat(),
                    'feature_importance': self.model_metrics.feature_importance,
                    'model_type': self.model_type
                }
                
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_dict, f, indent=2)
            
            logger.info(f"Model saved to {model_file}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load trained model and scaler from disk."""
        try:
            model_file = os.path.join(self.model_path, f"predictive_model_{self.model_type}.pkl")
            scaler_file = os.path.join(self.model_path, "feature_scaler.pkl")
            metrics_file = os.path.join(self.model_path, "model_metrics.json")
            
            # Load model
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info(f"Loaded model from {model_file}")
            
            # Load scaler
            if os.path.exists(scaler_file) and SKLEARN_AVAILABLE:
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load metrics
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_dict = json.load(f)
                
                self.model_metrics = ModelMetrics(
                    accuracy=metrics_dict['accuracy'],
                    precision=metrics_dict['precision'],
                    recall=metrics_dict['recall'],
                    f1_score=metrics_dict['f1_score'],
                    training_samples=metrics_dict['training_samples'],
                    last_trained=datetime.fromisoformat(metrics_dict['last_trained']),
                    feature_importance=metrics_dict.get('feature_importance', {})
                )
            
        except Exception as e:
            logger.info(f"Could not load existing model: {e}")
            self.is_trained = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        info = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'sklearn_available': SKLEARN_AVAILABLE,
            'xgboost_available': XGBOOST_AVAILABLE,
            'confidence_thresholds': self.confidence_thresholds,
            'feature_names': self.feature_names
        }
        
        if self.model_metrics is not None:
            info['metrics'] = {
                'accuracy': self.model_metrics.accuracy,
                'precision': self.model_metrics.precision,
                'recall': self.model_metrics.recall,
                'f1_score': self.model_metrics.f1_score,
                'training_samples': self.model_metrics.training_samples,
                'last_trained': self.model_metrics.last_trained.isoformat(),
                'feature_importance': self.model_metrics.feature_importance
            }
        
        return info


# Integration function for AI/ML engine
def create_predictive_scorer(config: Dict[str, Any]) -> PredictiveScorer:
    """
    Create a PredictiveScorer instance from configuration.
    
    Args:
        config: Configuration dictionary for predictive scoring
        
    Returns:
        Configured PredictiveScorer instance
    """
    return PredictiveScorer(
        model_path=config.get('model_path'),
        model_type=config.get('model_type', 'random_forest'),
        confidence_thresholds=config.get('confidence_thresholds')
    )


def process_predictive_scoring(
    reconciliation_data: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process predictive scoring for AI/ML engine integration.
    
    Args:
        reconciliation_data: Original reconciliation results
        config: Feature configuration
        
    Returns:
        Predictive scoring results
    """
    try:
        # Create scorer
        scorer = create_predictive_scorer(config)
        
        # Extract match data
        matches_df = reconciliation_data.get('matches', pd.DataFrame())
        books_data = reconciliation_data.get('books_data', pd.DataFrame())
        gstr2a_data = reconciliation_data.get('gstr2a_data', pd.DataFrame())
        
        if matches_df.empty:
            return {
                'confidence_scores': pd.Series([], dtype=float),
                'detailed_scores': [],
                'model_info': scorer.get_model_info(),
                'low_confidence_matches': pd.DataFrame(),
                'summary': {
                    'total_matches': 0,
                    'high_confidence': 0,
                    'medium_confidence': 0,
                    'low_confidence': 0,
                    'requires_review': 0
                }
            }
        
        # Calculate confidence scores
        confidence_scores = scorer.calculate_match_confidence(matches_df, books_data, gstr2a_data)
        detailed_scores = scorer.get_detailed_confidence_scores(matches_df, books_data, gstr2a_data)
        
        # Identify low confidence matches
        low_confidence_threshold = config.get('review_threshold', 0.6)
        low_confidence_mask = confidence_scores < low_confidence_threshold
        low_confidence_matches = matches_df[low_confidence_mask].copy()
        low_confidence_matches['confidence_score'] = confidence_scores[low_confidence_mask]
        
        # Generate summary
        high_count = sum(1 for score in detailed_scores if score.confidence_level == "High")
        medium_count = sum(1 for score in detailed_scores if score.confidence_level == "Medium")
        low_count = sum(1 for score in detailed_scores if score.confidence_level == "Low")
        review_count = sum(1 for score in detailed_scores if score.requires_review)
        
        summary = {
            'total_matches': len(matches_df),
            'high_confidence': high_count,
            'medium_confidence': medium_count,
            'low_confidence': low_count,
            'requires_review': review_count,
            'average_confidence': float(confidence_scores.mean()) if len(confidence_scores) > 0 else 0.0
        }
        
        return {
            'confidence_scores': confidence_scores,
            'detailed_scores': detailed_scores,
            'model_info': scorer.get_model_info(),
            'low_confidence_matches': low_confidence_matches,
            'summary': summary,
            'feature_importance': scorer.get_feature_importance()
        }
        
    except Exception as e:
        logger.error(f"Error in predictive scoring processing: {e}")
        return {
            'error': str(e),
            'confidence_scores': pd.Series([], dtype=float),
            'detailed_scores': [],
            'model_info': {},
            'low_confidence_matches': pd.DataFrame(),
            'summary': {
                'total_matches': 0,
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0,
                'requires_review': 0
            }
        }