"""
Enhanced GST Reconciliation with AI/ML Integration

This module provides an enhanced wrapper around the existing GSTReconciliation class
that integrates AI/ML features while maintaining 100% backward compatibility.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

from utils.reconciliation import GSTReconciliation
from utils.smart_matching import SmartNameMatcher
from utils.performance_monitor import PerformanceMonitor
from utils.ai_ml_config import AIMLConfigManager
from utils.anomaly_detector import AnomalyDetector, Anomaly, RiskScore
from utils.data_quality import DataQualityEngine, DataQualityReport

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGSTReconciliation(GSTReconciliation):
    """
    Enhanced GST Reconciliation with AI/ML capabilities.
    
    This class extends the original GSTReconciliation to add AI/ML features
    while maintaining complete backward compatibility.
    """
    
    def __init__(self, df, gstin_comments=None, ai_ml_config=None):
        """
        Initialize Enhanced GST Reconciliation.
        
        Args:
            df: Input DataFrame with reconciliation data
            gstin_comments: GSTIN comments for processing
            ai_ml_config: AI/ML configuration (optional)
        """
        # Initialize AI/ML components first
        self.ai_ml_enabled = ai_ml_config is not None
        self.ai_ml_config = ai_ml_config or {}
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor(max_total_time=200)
        
        # Initialize Smart Name Matcher if AI/ML is enabled
        self.smart_matcher = None
        if self.ai_ml_enabled and self.ai_ml_config.get('smart_matching', {}).get('enabled', False):
            try:
                with self.performance_monitor.monitor_component('smart_matcher_init'):
                    self.smart_matcher = SmartNameMatcher(
                        model_name=self.ai_ml_config.get('smart_matching', {}).get('model_name', 'all-MiniLM-L6-v2'),
                        cache_dir=self.ai_ml_config.get('smart_matching', {}).get('cache_dir', '.cache/embeddings')
                    )
                logger.info("SmartNameMatcher initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize SmartNameMatcher: {e}")
                self.smart_matcher = None
        
        # Initialize Anomaly Detector if AI/ML is enabled
        self.anomaly_detector = None
        if self.ai_ml_enabled and self.ai_ml_config.get('anomaly_detection', {}).get('enabled', False):
            try:
                with self.performance_monitor.monitor_component('anomaly_detector_init'):
                    anomaly_config = self.ai_ml_config.get('anomaly_detection', {})
                    self.anomaly_detector = AnomalyDetector(config=anomaly_config)
                logger.info("AnomalyDetector initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize AnomalyDetector: {e}")
                self.anomaly_detector = None
        
        # Initialize Data Quality Engine if AI/ML is enabled
        self.data_quality_engine = None
        if self.ai_ml_enabled and self.ai_ml_config.get('data_quality', {}).get('enabled', False):
            try:
                with self.performance_monitor.monitor_component('data_quality_init'):
                    data_quality_config = self.ai_ml_config.get('data_quality', {})
                    self.data_quality_engine = DataQualityEngine(config=data_quality_config)
                logger.info("DataQualityEngine initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize DataQualityEngine: {e}")
                self.data_quality_engine = None
        
        # AI/ML enhancement results
        self.ai_ml_results = {
            'smart_matching_applied': False,
            'enhanced_matches': 0,
            'anomaly_detection_applied': False,
            'detected_anomalies': [],
            'risk_assessments': [],
            'data_quality_applied': False,
            'data_quality_report': None,
            'performance_metrics': {},
            'errors': []
        }
        
        # Apply data quality preprocessing if enabled
        processed_df = df
        if self.data_quality_engine:
            processed_df = self._apply_data_quality_preprocessing(df)
        
        # Call parent constructor with processed data
        super().__init__(processed_df, gstin_comments)
        
        # Apply AI/ML enhancements if enabled
        if self.ai_ml_enabled:
            self._apply_ai_ml_enhancements()
    
    def _apply_data_quality_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality preprocessing to input data."""
        try:
            with self.performance_monitor.monitor_component('data_quality_preprocessing'):
                logger.info("Applying data quality preprocessing to input data")
                
                # Clean the dataframe
                cleaned_df, quality_report = self.data_quality_engine.clean_dataframe(df)
                
                # Store quality report
                self.ai_ml_results['data_quality_report'] = quality_report
                self.ai_ml_results['data_quality_applied'] = True
                
                logger.info(f"Data quality preprocessing complete: {quality_report.records_cleaned} records cleaned in {quality_report.processing_time:.2f}s")
                
                return cleaned_df
                
        except Exception as e:
            error_msg = f"Error in data quality preprocessing: {str(e)}"
            logger.error(error_msg)
            self.ai_ml_results['errors'].append(error_msg)
            return df  # Return original data if preprocessing fails
    
    def _apply_ai_ml_enhancements(self):
        """Apply AI/ML enhancements to the reconciliation results."""
        try:
            logger.info("Applying AI/ML enhancements to reconciliation results")
            
            # Enhance name matching if smart matcher is available
            if self.smart_matcher:
                self._enhance_name_matching()
            
            # Perform anomaly detection if detector is available
            if self.anomaly_detector:
                self._perform_anomaly_detection()
            
            # Store performance metrics
            self.ai_ml_results['performance_metrics'] = self.performance_monitor.get_performance_metrics()
            
            logger.info("AI/ML enhancements applied successfully")
            
        except Exception as e:
            error_msg = f"Error applying AI/ML enhancements: {str(e)}"
            logger.error(error_msg)
            self.ai_ml_results['errors'].append(error_msg)
    
    def _enhance_name_matching(self):
        """Enhance name matching using ML-based similarity scoring."""
        try:
            with self.performance_monitor.monitor_component('smart_matching'):
                logger.info("Enhancing name matching with ML-based similarity")
                
                # Get all unique company names for precomputation
                all_names = set()
                
                # Collect names from both datasets
                for df in [self.books_df, self.gstr2a_df]:
                    if 'Supplier Legal Name' in df.columns:
                        all_names.update(df['Supplier Legal Name'].dropna().astype(str).tolist())
                    if 'Supplier Trade Name' in df.columns:
                        all_names.update(df['Supplier Trade Name'].dropna().astype(str).tolist())
                
                # Precompute embeddings for all names
                if all_names:
                    logger.info(f"Precomputing embeddings for {len(all_names)} unique company names")
                    embeddings = self.smart_matcher.precompute_embeddings(list(all_names))
                    logger.info(f"Precomputed {len(embeddings)} embeddings")
                
                # Enhance existing matches with ML scores
                self._add_ml_similarity_scores()
                
                # Look for additional matches using ML similarity
                self._find_additional_ml_matches()
                
                self.ai_ml_results['smart_matching_applied'] = True
                
        except Exception as e:
            error_msg = f"Error in smart name matching: {str(e)}"
            logger.error(error_msg)
            self.ai_ml_results['errors'].append(error_msg)
    
    def _add_ml_similarity_scores(self):
        """Add ML similarity scores to existing matches."""
        if self.matched_df.empty:
            return
        
        try:
            # Group matched records by Match ID
            match_groups = self.matched_df.groupby('Match ID')
            
            enhanced_matches = []
            
            for match_id, group in match_groups:
                if len(group) != 2:  # Should have exactly 2 records (Books + GSTR-2A)
                    enhanced_matches.extend(group.to_dict('records'))
                    continue
                
                books_record = group[group['Source Name'] == 'Books'].iloc[0]
                gstr2a_record = group[group['Source Name'] == 'GSTR-2A'].iloc[0]
                
                # Calculate ML similarity scores
                ml_scores = self._calculate_ml_similarity_pair(books_record, gstr2a_record)
                
                # Add ML scores to both records
                books_dict = books_record.to_dict()
                gstr2a_dict = gstr2a_record.to_dict()
                
                for record_dict in [books_dict, gstr2a_dict]:
                    record_dict.update({
                        'ML_Legal_Name_Similarity': ml_scores['legal_name_similarity'],
                        'ML_Trade_Name_Similarity': ml_scores['trade_name_similarity'],
                        'ML_Overall_Similarity': ml_scores['overall_similarity'],
                        'ML_Confidence_Score': ml_scores['confidence_score']
                    })
                
                enhanced_matches.extend([books_dict, gstr2a_dict])
            
            # Update matched_df with enhanced records
            if enhanced_matches:
                self.matched_df = pd.DataFrame(enhanced_matches)
                logger.info(f"Enhanced {len(match_groups)} existing matches with ML similarity scores")
            
        except Exception as e:
            logger.error(f"Error adding ML similarity scores: {e}")
    
    def _calculate_ml_similarity_pair(self, books_record, gstr2a_record) -> Dict[str, float]:
        """Calculate ML similarity scores for a pair of records."""
        scores = {
            'legal_name_similarity': 0.0,
            'trade_name_similarity': 0.0,
            'overall_similarity': 0.0,
            'confidence_score': 0.0
        }
        
        try:
            # Legal name similarity
            books_legal = str(books_record.get('Supplier Legal Name', ''))
            gstr2a_legal = str(gstr2a_record.get('Supplier Legal Name', ''))
            
            if books_legal and gstr2a_legal:
                legal_scores = self.smart_matcher.calculate_similarity_scores([books_legal], [gstr2a_legal])
                scores['legal_name_similarity'] = legal_scores[0] if legal_scores else 0.0
            
            # Trade name similarity
            books_trade = str(books_record.get('Supplier Trade Name', ''))
            gstr2a_trade = str(gstr2a_record.get('Supplier Trade Name', ''))
            
            if books_trade and gstr2a_trade:
                trade_scores = self.smart_matcher.calculate_similarity_scores([books_trade], [gstr2a_trade])
                scores['trade_name_similarity'] = trade_scores[0] if trade_scores else 0.0
            
            # Overall similarity (weighted average)
            legal_weight = 0.7
            trade_weight = 0.3
            
            scores['overall_similarity'] = (
                scores['legal_name_similarity'] * legal_weight +
                scores['trade_name_similarity'] * trade_weight
            )
            
            # Confidence score based on data availability and similarity
            confidence = 0.5  # Base confidence
            
            if books_legal and gstr2a_legal:
                confidence += 0.3
            if books_trade and gstr2a_trade:
                confidence += 0.2
            
            # Adjust based on similarity scores
            if scores['overall_similarity'] > 0.8:
                confidence += 0.2
            elif scores['overall_similarity'] > 0.6:
                confidence += 0.1
            
            scores['confidence_score'] = min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating ML similarity: {e}")
        
        return scores
    
    def _find_additional_ml_matches(self):
        """Find additional matches using ML similarity that were missed by traditional matching."""
        try:
            # Get unprocessed records
            unprocessed_books = self.books_df[~self.books_df['processed']].copy()
            unprocessed_gstr2a = self.gstr2a_df[~self.gstr2a_df['processed']].copy()
            
            if unprocessed_books.empty or unprocessed_gstr2a.empty:
                return
            
            logger.info(f"Looking for additional ML matches among {len(unprocessed_books)} Books and {len(unprocessed_gstr2a)} GSTR-2A records")
            
            additional_matches = []
            ml_match_threshold = self.ai_ml_config.get('smart_matching', {}).get('match_threshold', 0.75)
            
            # For each unprocessed Books record, find best ML match in GSTR-2A
            for books_idx, books_row in unprocessed_books.iterrows():
                best_match = None
                best_score = 0.0
                
                for gstr2a_idx, gstr2a_row in unprocessed_gstr2a.iterrows():
                    # Skip if already processed
                    if self.gstr2a_df.at[gstr2a_idx, 'processed']:
                        continue
                    
                    # Calculate ML similarity
                    ml_scores = self._calculate_ml_similarity_pair(books_row, gstr2a_row)
                    
                    # Check if this is a potential match
                    if ml_scores['overall_similarity'] > ml_match_threshold and ml_scores['overall_similarity'] > best_score:
                        # Additional validation checks
                        if self._validate_ml_match(books_row, gstr2a_row, ml_scores):
                            best_match = (gstr2a_idx, gstr2a_row, ml_scores)
                            best_score = ml_scores['overall_similarity']
                
                # If we found a good match, create it
                if best_match:
                    gstr2a_idx, gstr2a_row, ml_scores = best_match
                    
                    # Mark as processed
                    self.books_df.at[books_idx, 'processed'] = True
                    self.gstr2a_df.at[gstr2a_idx, 'processed'] = True
                    
                    # Create match records
                    match_records = self._create_ml_match_records(books_row, gstr2a_row, ml_scores)
                    additional_matches.extend(match_records)
                    
                    self.ai_ml_results['enhanced_matches'] += 1
            
            # Add additional matches to matched_df
            if additional_matches:
                additional_df = pd.DataFrame(additional_matches)
                self.matched_df = pd.concat([self.matched_df, additional_df], ignore_index=True)
                logger.info(f"Found {len(additional_matches)//2} additional matches using ML similarity")
            
        except Exception as e:
            logger.error(f"Error finding additional ML matches: {e}")
    
    def _validate_ml_match(self, books_row, gstr2a_row, ml_scores) -> bool:
        """Validate that an ML match meets additional criteria."""
        try:
            # Check sign consistency (same as existing logic)
            books_total_tax = (books_row['Total IGST Amount'] + 
                             books_row['Total CGST Amount'] + 
                             books_row['Total SGST Amount'])
            gstr2a_total_tax = (gstr2a_row['Total IGST Amount'] + 
                              gstr2a_row['Total CGST Amount'] + 
                              gstr2a_row['Total SGST Amount'])
            
            books_sign = 1 if books_total_tax >= 0 else -1
            gstr2a_sign = 1 if gstr2a_total_tax >= 0 else -1
            
            if books_sign != gstr2a_sign:
                return False
            
            # Check if tax amounts are reasonably close (within 50% difference)
            if books_total_tax != 0 and gstr2a_total_tax != 0:
                tax_ratio = abs(books_total_tax - gstr2a_total_tax) / max(abs(books_total_tax), abs(gstr2a_total_tax))
                if tax_ratio > 0.5:  # More than 50% difference
                    return False
            
            # Check GSTIN similarity if available
            books_gstin = str(books_row.get('Supplier GSTIN', ''))
            gstr2a_gstin = str(gstr2a_row.get('Supplier GSTIN', ''))
            
            if books_gstin and gstr2a_gstin and books_gstin != 'nan' and gstr2a_gstin != 'nan':
                gstin_similarity = fuzz.ratio(books_gstin.lower(), gstr2a_gstin.lower()) / 100.0
                if gstin_similarity < 0.6:  # GSTIN should be reasonably similar
                    return False
            
            # Require high confidence for ML-only matches
            if ml_scores['confidence_score'] < 0.7:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating ML match: {e}")
            return False
    
    def _create_ml_match_records(self, books_row, gstr2a_row, ml_scores) -> List[Dict]:
        """Create match records for ML-discovered matches."""
        match_id = self.match_id_counter
        self.match_id_counter += 1
        
        # Calculate differences
        igst_diff = books_row['Total IGST Amount'] - gstr2a_row['Total IGST Amount']
        cgst_diff = books_row['Total CGST Amount'] - gstr2a_row['Total CGST Amount']
        sgst_diff = books_row['Total SGST Amount'] - gstr2a_row['Total SGST Amount']
        
        # Calculate date difference
        date_diff = np.nan
        if pd.notna(books_row['Invoice Date']) and pd.notna(gstr2a_row['Invoice Date']):
            date_diff = (books_row['Invoice Date'] - gstr2a_row['Invoice Date']).days
        
        # Determine match status
        total_tax_diff = abs(igst_diff) + abs(cgst_diff) + abs(sgst_diff)
        is_exact_match = total_tax_diff <= self.tax_tolerance
        
        status = 'ML Enhanced Match'
        if not is_exact_match:
            status = 'ML Partial Match'
        
        # Generate field status columns
        field_status = self._generate_field_status_columns(books_row, gstr2a_row)
        
        # Common data for both records
        common_data = {
            'Match ID': match_id,
            'Status': status,
            'Sub Status': f'ML-enhanced match with {ml_scores["overall_similarity"]:.1%} similarity',
            'Cross-Year': False,  # Will be updated if needed
            'Tax Diff Status': 'No Difference' if total_tax_diff <= self.tax_tolerance else 'Has Difference',
            'Date Status': 'Within Tolerance' if (pd.notna(date_diff) and abs(date_diff) <= self.date_tolerance.days) else 'Outside Tolerance' if pd.notna(date_diff) else 'N/A',
            'Tax Sign Status': 'Sign Match',
            'Narrative': f'ML-enhanced match with {ml_scores["overall_similarity"]:.1%} similarity',
            'Suggestions': f'High-confidence ML match (confidence: {ml_scores["confidence_score"]:.1%})',
            'ML_Legal_Name_Similarity': ml_scores['legal_name_similarity'],
            'ML_Trade_Name_Similarity': ml_scores['trade_name_similarity'],
            'ML_Overall_Similarity': ml_scores['overall_similarity'],
            'ML_Confidence_Score': ml_scores['confidence_score']
        }
        
        # Add field status columns
        common_data.update(field_status)
        
        # Books record
        books_match_data = books_row.to_dict()
        books_match_data.update(common_data)
        books_match_data.update({
            'Source Name': 'Books',
            'IGST Diff': igst_diff,
            'CGST Diff': cgst_diff,
            'SGST Diff': sgst_diff,
            'Date Diff': date_diff,
            'Value Sign': 'Positive' if (books_row['Total IGST Amount'] + books_row['Total CGST Amount'] + books_row['Total SGST Amount']) >= 0 else 'Negative'
        })
        
        # GSTR-2A record
        gstr2a_match_data = gstr2a_row.to_dict()
        gstr2a_match_data.update(common_data)
        gstr2a_match_data.update({
            'Source Name': 'GSTR-2A',
            'IGST Diff': -igst_diff,
            'CGST Diff': -cgst_diff,
            'SGST Diff': -sgst_diff,
            'Date Diff': -date_diff if pd.notna(date_diff) else np.nan,
            'Value Sign': 'Positive' if (gstr2a_row['Total IGST Amount'] + gstr2a_row['Total CGST Amount'] + gstr2a_row['Total SGST Amount']) >= 0 else 'Negative'
        })
        
        return [books_match_data, gstr2a_match_data]
    
    def _perform_anomaly_detection(self):
        """Perform comprehensive anomaly detection on reconciliation data."""
        try:
            with self.performance_monitor.monitor_component('anomaly_detection'):
                logger.info("Performing anomaly detection on reconciliation data")
                
                # Combine all data for anomaly detection
                combined_data = []
                
                # Add matched data
                if not self.matched_df.empty:
                    combined_data.append(self.matched_df)
                
                # Add unmatched Books data
                unmatched_books = self.books_df[~self.books_df['processed']].copy()
                if not unmatched_books.empty:
                    unmatched_books['Source Name'] = 'Books'
                    combined_data.append(unmatched_books)
                
                # Add unmatched GSTR-2A data
                unmatched_gstr2a = self.gstr2a_df[~self.gstr2a_df['processed']].copy()
                if not unmatched_gstr2a.empty:
                    unmatched_gstr2a['Source Name'] = 'GSTR-2A'
                    combined_data.append(unmatched_gstr2a)
                
                if not combined_data:
                    logger.warning("No data available for anomaly detection")
                    return
                
                # Combine all data
                analysis_df = pd.concat(combined_data, ignore_index=True, sort=False)
                
                # Perform anomaly detection
                anomalies, risk_scores = self.anomaly_detector.detect_anomalies(analysis_df)
                
                # Store results
                self.ai_ml_results['detected_anomalies'] = anomalies
                self.ai_ml_results['risk_assessments'] = risk_scores
                self.ai_ml_results['anomaly_detection_applied'] = True
                
                logger.info(f"Anomaly detection complete: {len(anomalies)} anomalies, {len(risk_scores)} risk assessments")
                
        except Exception as e:
            error_msg = f"Error in anomaly detection: {str(e)}"
            logger.error(error_msg)
            self.ai_ml_results['errors'].append(error_msg)
    
    def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """Get risk alerts for UI display."""
        alerts = []
        
        try:
            # Add anomaly alerts
            for anomaly in self.ai_ml_results.get('detected_anomalies', []):
                alert = {
                    'id': anomaly.id,
                    'type': 'anomaly',
                    'severity': anomaly.severity,
                    'title': f"{anomaly.type.title()} Anomaly Detected",
                    'description': anomaly.description,
                    'score': anomaly.score,
                    'affected_records': len(anomaly.affected_records),
                    'recommendations': anomaly.recommendations,
                    'details': anomaly.details,
                    'timestamp': anomaly.timestamp
                }
                alerts.append(alert)
            
            # Add high-risk supplier alerts
            for risk_score in self.ai_ml_results.get('risk_assessments', []):
                if risk_score.risk_level in ['high', 'medium']:
                    alert = {
                        'id': f"risk_{risk_score.entity_id}",
                        'type': 'supplier_risk',
                        'severity': risk_score.risk_level,
                        'title': f"{risk_score.risk_level.title()} Risk Supplier",
                        'description': f"Supplier {risk_score.entity_id} has {risk_score.risk_level} risk level",
                        'score': risk_score.risk_score / 100.0,  # Normalize to 0-1
                        'affected_records': 1,
                        'recommendations': risk_score.recommendations,
                        'details': {
                            'risk_factors': risk_score.risk_factors,
                            'explanation': risk_score.explanation,
                            'confidence': risk_score.confidence
                        },
                        'timestamp': datetime.now()
                    }
                    alerts.append(alert)
            
            # Sort by severity and score
            severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            alerts.sort(key=lambda x: (severity_order.get(x['severity'], 0), x['score']), reverse=True)
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {e}")
        
        return alerts
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        if not self.anomaly_detector:
            return {"enabled": False, "message": "Anomaly detection not enabled"}
        
        try:
            summary = self.anomaly_detector.get_anomaly_summary()
            risk_summary = self.anomaly_detector.get_risk_summary()
            
            return {
                "enabled": True,
                "anomalies": summary,
                "risks": risk_summary,
                "alerts_count": len(self.get_risk_alerts())
            }
        except Exception as e:
            logger.error(f"Error getting anomaly summary: {e}")
            return {"enabled": True, "error": str(e)}
    
    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get summary of data quality improvements."""
        if not self.data_quality_engine:
            return {"enabled": False, "message": "Data quality engine not enabled"}
        
        try:
            quality_report = self.ai_ml_results.get('data_quality_report')
            if not quality_report:
                return {"enabled": True, "message": "No data quality report available"}
            
            return {
                "enabled": True,
                "total_records": quality_report.total_records,
                "records_cleaned": quality_report.records_cleaned,
                "gstin_corrections": quality_report.gstin_corrections,
                "name_normalizations": quality_report.name_normalizations,
                "validation_errors": len(quality_report.validation_errors),
                "processing_time": quality_report.processing_time,
                "cleaning_summary": quality_report.cleaning_summary,
                "timestamp": quality_report.timestamp
            }
        except Exception as e:
            logger.error(f"Error getting data quality summary: {e}")
            return {"enabled": True, "error": str(e)}
    
    def get_ai_ml_summary(self) -> Dict[str, Any]:
        """Get summary of AI/ML enhancements applied."""
        summary = {
            'ai_ml_enabled': self.ai_ml_enabled,
            'smart_matching_applied': self.ai_ml_results.get('smart_matching_applied', False),
            'enhanced_matches': self.ai_ml_results.get('enhanced_matches', 0),
            'anomaly_detection_applied': self.ai_ml_results.get('anomaly_detection_applied', False),
            'detected_anomalies_count': len(self.ai_ml_results.get('detected_anomalies', [])),
            'risk_assessments_count': len(self.ai_ml_results.get('risk_assessments', [])),
            'data_quality_applied': self.ai_ml_results.get('data_quality_applied', False),
            'performance_metrics': self.ai_ml_results.get('performance_metrics', {}),
            'errors': self.ai_ml_results.get('errors', [])
        }
        
        # Add smart matcher metrics if available
        if self.smart_matcher:
            summary['smart_matcher_metrics'] = self.smart_matcher.get_performance_metrics()
        
        # Add anomaly detection summary if available
        if self.anomaly_detector:
            summary['anomaly_summary'] = self.get_anomaly_summary()
        
        # Add data quality summary if available
        if self.data_quality_engine:
            summary['data_quality_summary'] = self.get_data_quality_summary()
        
        return summary
    
    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get enhanced reconciliation summary including AI/ML metrics."""
        # Get original summary
        original_summary = {
            'raw_summary': self.raw_summary,
            'recon_summary': self.recon_summary,
            'integrity_checks': self.integrity_checks
        }
        
        # Add AI/ML summary
        ai_ml_summary = self.get_ai_ml_summary()
        
        return {
            **original_summary,
            'ai_ml_enhancements': ai_ml_summary
        }
    
    def cleanup_ai_ml_resources(self):
        """Clean up AI/ML resources."""
        try:
            if self.smart_matcher:
                self.smart_matcher.cleanup()
            
            if self.anomaly_detector:
                # Clear anomaly detection results to free memory
                self.ai_ml_results['detected_anomalies'] = []
                self.ai_ml_results['risk_assessments'] = []
            
            if hasattr(self, 'performance_monitor'):
                # Save performance metrics
                pass  # Performance monitor cleanup if needed
            
            logger.info("AI/ML resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up AI/ML resources: {e}")


def create_enhanced_reconciliation(df, gstin_comments=None, ai_ml_config=None):
    """
    Factory function to create enhanced reconciliation with AI/ML features.
    
    Args:
        df: Input DataFrame
        gstin_comments: GSTIN comments
        ai_ml_config: AI/ML configuration
        
    Returns:
        EnhancedGSTReconciliation instance
    """
    return EnhancedGSTReconciliation(df, gstin_comments, ai_ml_config)