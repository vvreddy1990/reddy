import streamlit as st
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReconciliationSettings:
    """Manages reconciliation settings for the GST application."""
    
    def __init__(self):
        self.settings_file = "reconciliation_settings.json"
        self.default_settings = {
            "tax_amount_tolerance": 10.0,
            "date_tolerance_days": 1,
            "name_preference": "Legal Name",
            "currency_format": "INR",
            "case_sensitive_names": False,
            "decimal_precision": 2,
            "enable_advanced_matching": True,
            "group_tax_tolerance": 50.0,
            "similarity_threshold": 80.0,
            "auto_apply_settings": True
        }
        self.settings = self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file or return defaults."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    settings = self.default_settings.copy()
                    settings.update(loaded_settings)
                    logger.info("Settings loaded successfully")
                    return settings
            else:
                logger.info("No settings file found, using defaults")
                return self.default_settings.copy()
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return self.default_settings.copy()
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            self.settings = settings
            logger.info("Settings saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
    
    def validate_settings(self, settings: Dict[str, Any]) -> tuple[bool, str]:
        """Validate settings and return (is_valid, error_message)."""
        try:
            # Validate tax amount tolerance
            if not isinstance(settings.get('tax_amount_tolerance'), (int, float)):
                return False, "Tax Amount Tolerance must be a number"
            if settings.get('tax_amount_tolerance') < 0:
                return False, "Tax Amount Tolerance must be non-negative"
            
            # Validate date tolerance
            if not isinstance(settings.get('date_tolerance_days'), (int, float)):
                return False, "Date Tolerance must be a number"
            if settings.get('date_tolerance_days') < 0:
                return False, "Date Tolerance must be non-negative"
            
            # Validate name preference
            valid_name_preferences = ["Trade Name", "Legal Name"]
            if settings.get('name_preference') not in valid_name_preferences:
                return False, "Name Preference must be 'Trade Name' or 'Legal Name'"
            
            # Validate currency format
            valid_currencies = ["INR", "USD", "EUR", "GBP"]
            if settings.get('currency_format') not in valid_currencies:
                return False, "Invalid currency format"
            
            # Validate decimal precision
            if not isinstance(settings.get('decimal_precision'), int):
                return False, "Decimal Precision must be an integer"
            if not (0 <= settings.get('decimal_precision') <= 6):
                return False, "Decimal Precision must be between 0 and 6"
            
            # Validate similarity threshold
            if not isinstance(settings.get('similarity_threshold'), (int, float)):
                return False, "Similarity Threshold must be a number"
            if not (0 <= settings.get('similarity_threshold') <= 100):
                return False, "Similarity Threshold must be between 0 and 100"
            
            # Validate group tax tolerance
            if not isinstance(settings.get('group_tax_tolerance'), (int, float)):
                return False, "Group Tax Tolerance must be a number"
            if settings.get('group_tax_tolerance') < 0:
                return False, "Group Tax Tolerance must be non-negative"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def apply_tax_diff_status(self, tax_diff: float) -> str:
        """Apply tax difference status based on settings."""
        tolerance = self.settings.get('tax_amount_tolerance', 10.0)
        if abs(tax_diff) <= tolerance:
            return "No Difference"
        else:
            return "Has Difference"
    
    def apply_date_status(self, date_diff_days: int) -> str:
        """Apply date status based on settings."""
        tolerance = self.settings.get('date_tolerance_days', 1)
        if abs(date_diff_days) <= tolerance:
            return "Within Tolerance"
        else:
            return "Outside Tolerance"
    
    def get_name_for_comparison(self, trade_name: str, legal_name: str) -> str:
        """Get the preferred name for comparison based on settings."""
        preference = self.settings.get('name_preference', 'Legal Name')
        if preference == "Trade Name":
            return trade_name if trade_name else legal_name
        else:
            return legal_name if legal_name else trade_name
    
    def format_currency(self, amount: float) -> str:
        """Format currency based on settings."""
        precision = self.settings.get('decimal_precision', 2)
        currency = self.settings.get('currency_format', 'INR')
        
        if currency == 'INR':
            return f"₹{amount:,.{precision}f}"
        elif currency == 'USD':
            return f"${amount:,.{precision}f}"
        elif currency == 'EUR':
            return f"€{amount:,.{precision}f}"
        elif currency == 'GBP':
            return f"£{amount:,.{precision}f}"
        else:
            return f"{amount:,.{precision}f}"
    
    def get_similarity_threshold(self) -> float:
        """Get similarity threshold for name matching."""
        return self.settings.get('similarity_threshold', 80.0) / 100.0
    
    def is_case_sensitive(self) -> bool:
        """Check if name matching should be case sensitive."""
        return self.settings.get('case_sensitive_names', False)
    
    def get_group_tax_tolerance(self) -> float:
        """Get group tax tolerance."""
        return self.settings.get('group_tax_tolerance', 50.0)
    
    def is_advanced_matching_enabled(self) -> bool:
        """Check if advanced matching is enabled."""
        return self.settings.get('enable_advanced_matching', True)
    
    def should_auto_apply(self) -> bool:
        """Check if settings should be auto-applied."""
        return self.settings.get('auto_apply_settings', True)

def render_settings_page() -> Dict[str, Any]:
    """Render the settings page and return updated settings."""
    st.markdown("## ⚙️ Reconciliation Settings")
    st.markdown("Configure settings that affect comments in the 'Tax Diff Status' and 'Date Status' columns.")
    
    # Initialize settings manager
    settings_manager = ReconciliationSettings()
    current_settings = settings_manager.settings.copy()
    
    # Create tabs for different setting categories
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Settings", "Advanced Settings", "AI Insights", "About"])
    
    with tab1:
        st.markdown("### Basic Reconciliation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tax Amount Tolerance
            tax_tolerance = st.number_input(
                "Tax Amount Tolerance (₹)",
                min_value=0.0,
                max_value=10000.0,
                value=float(current_settings.get('tax_amount_tolerance', 10.0)),
                step=1.0,
                help="If tax difference ≤ tolerance, 'Tax Diff Status' shows 'No Difference'. Otherwise shows 'Has Difference'."
            )
            
            # Date Tolerance
            date_tolerance = st.number_input(
                "Date Tolerance (days)",
                min_value=0,
                max_value=365,
                value=int(current_settings.get('date_tolerance_days', 1)),
                step=1,
                help="If date difference ≤ tolerance, 'Date Status' shows 'Within Tolerance'. Otherwise shows 'Outside Tolerance'."
            )
            
            # Name Preference
            name_preference = st.selectbox(
                "Name Preference for Reconciliation",
                options=["Legal Name", "Trade Name"],
                index=0 if current_settings.get('name_preference') == "Legal Name" else 1,
                help="Affects name-related reconciliation comments and matching logic."
            )
        
        with col2:
            # Currency Format
            currency_format = st.selectbox(
                "Currency Format",
                options=["INR", "USD", "EUR", "GBP"],
                index=["INR", "USD", "EUR", "GBP"].index(current_settings.get('currency_format', 'INR')),
                help="Format for displaying currency values in reports."
            )
            
            # Decimal Precision
            decimal_precision = st.slider(
                "Decimal Precision",
                min_value=0,
                max_value=6,
                value=int(current_settings.get('decimal_precision', 2)),
                step=1,
                help="Number of decimal places for currency and numeric values."
            )
            
            # Case Sensitivity
            case_sensitive = st.checkbox(
                "Case Sensitive Name Matching",
                value=current_settings.get('case_sensitive_names', False),
                help="Enable case-sensitive matching for company names."
            )
    
    with tab2:
        st.markdown("### Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Similarity Threshold
            similarity_threshold = st.slider(
                "Name Similarity Threshold (%)",
                min_value=0,
                max_value=100,
                value=int(current_settings.get('similarity_threshold', 80.0)),
                step=5,
                help="Minimum similarity percentage for name matching (0-100%)."
            )
            
            # Group Tax Tolerance
            group_tax_tolerance = st.number_input(
                "Group Tax Tolerance (₹)",
                min_value=0.0,
                max_value=10000.0,
                value=float(current_settings.get('group_tax_tolerance', 50.0)),
                step=5.0,
                help="Tolerance for tax amounts in group matching scenarios."
            )
        
        with col2:
            # Advanced Matching
            enable_advanced = st.checkbox(
                "Enable Advanced Matching",
                value=current_settings.get('enable_advanced_matching', True),
                help="Enable intelligent enhanced matching algorithms."
            )
            
            # Auto Apply Settings
            auto_apply = st.checkbox(
                "Auto-apply Settings",
                value=current_settings.get('auto_apply_settings', True),
                help="Automatically apply settings to reconciliation results."
            )
    
    with tab3:
        st.markdown("### 🤖 AI Insights & Recommendations")
        
        # AI Insights Settings
        try:
            # Try to import full AI insights integration
            from .ai_insights_integration import AIInsightsReportingIntegration
            ai_integration = AIInsightsReportingIntegration()
            ai_settings = ai_integration.render_ai_settings_controls()
            
            # Show performance metrics if available
            if ai_integration.enabled:
                st.markdown("---")
                st.markdown("#### Performance Metrics")
                
                try:
                    metrics = ai_integration.get_performance_metrics()
                    if metrics:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            total_time = metrics.get('total_time', 0)
                            st.metric("Total Processing Time", f"{total_time:.2f}s")
                        
                        with col2:
                            feature_count = len(metrics.get('component_times', {}))
                            st.metric("Active AI Features", feature_count)
                        
                        with col3:
                            avg_time = total_time / feature_count if feature_count > 0 else 0
                            st.metric("Avg Feature Time", f"{avg_time:.2f}s")
                        
                        # Show component breakdown
                        if metrics.get('component_times'):
                            st.markdown("**Component Performance:**")
                            for component, time_taken in metrics['component_times'].items():
                                st.text(f"• {component}: {time_taken:.2f}s")
                    else:
                        st.info("No performance metrics available yet. Run a reconciliation to see metrics.")
                        
                except Exception as e:
                    st.warning(f"Unable to load performance metrics: {e}")
            
        except ImportError:
            # Fallback to simple settings
            st.markdown("### 🤖 AI Insights Settings")
            
            # Basic enable/disable checkbox
            import json
            import os
            
            current_enabled = False
            try:
                if os.path.exists("reconciliation_settings.json"):
                    with open("reconciliation_settings.json", 'r') as f:
                        config = json.load(f)
                        current_enabled = config.get('ai_ml_features', {}).get('ai_insights', {}).get('enabled', False)
            except Exception:
                pass
            
            enabled = st.checkbox(
                "Enable AI Insights & Recommendations",
                value=current_enabled,
                help="Enable basic AI-powered insights and recommendations in reports"
            )
            
            if st.button("💾 Save AI Insights Settings"):
                try:
                    # Load existing config
                    config = {}
                    if os.path.exists("reconciliation_settings.json"):
                        with open("reconciliation_settings.json", 'r') as f:
                            config = json.load(f)
                    
                    # Update AI insights setting
                    if 'ai_ml_features' not in config:
                        config['ai_ml_features'] = {}
                    if 'ai_insights' not in config['ai_ml_features']:
                        config['ai_ml_features']['ai_insights'] = {}
                    
                    config['ai_ml_features']['ai_insights']['enabled'] = enabled
                    
                    # Save config
                    with open("reconciliation_settings.json", 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    st.success("✅ AI Insights settings saved successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error saving settings: {e}")
            
            st.info("Using basic AI insights integration. Install full AI components for advanced features.")
            
        except Exception as e:
            st.error(f"Error loading AI Insights settings: {e}")
    
    with tab4:
        st.markdown("### About Settings")
        st.markdown("""
        **Settings Overview:**
        
        - **Tax Amount Tolerance**: Controls when tax differences are considered significant
        - **Date Tolerance**: Controls when date differences are considered significant  
        - **Name Preference**: Determines which name field to prioritize in matching
        - **Currency Format**: How monetary values are displayed
        - **Decimal Precision**: Number of decimal places for numeric values
        - **Case Sensitivity**: Whether name matching considers letter case
        - **Similarity Threshold**: Minimum similarity for name matching
        - **Group Tax Tolerance**: Tolerance for group-based matching
        - **Advanced Matching**: Enable enhanced reconciliation algorithms
        - **Auto-apply**: Automatically apply settings to results
        
        **Note**: These settings only affect the display and comments in the reconciliation table, 
        not the core matching logic. Core reconciliation remains unchanged.
        """)
        
        # Show current settings summary
        st.markdown("### Current Settings Summary")
        summary_data = {
            "Setting": [
                "Tax Amount Tolerance",
                "Date Tolerance", 
                "Name Preference",
                "Currency Format",
                "Decimal Precision",
                "Case Sensitive Names",
                "Similarity Threshold",
                "Group Tax Tolerance",
                "Advanced Matching",
                "Auto-apply Settings"
            ],
            "Value": [
                f"₹{tax_tolerance:,.2f}",
                f"{date_tolerance} days",
                name_preference,
                currency_format,
                str(decimal_precision),
                "Yes" if case_sensitive else "No",
                f"{similarity_threshold}%",
                f"₹{group_tax_tolerance:,.2f}",
                "Enabled" if enable_advanced else "Disabled",
                "Enabled" if auto_apply else "Disabled"
            ]
        }
        st.dataframe(summary_data, use_container_width=True)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Settings", type="primary"):
            # Prepare updated settings
            updated_settings = {
                "tax_amount_tolerance": tax_tolerance,
                "date_tolerance_days": date_tolerance,
                "name_preference": name_preference,
                "currency_format": currency_format,
                "decimal_precision": decimal_precision,
                "case_sensitive_names": case_sensitive,
                "similarity_threshold": float(similarity_threshold),
                "group_tax_tolerance": group_tax_tolerance,
                "enable_advanced_matching": enable_advanced,
                "auto_apply_settings": auto_apply
            }
            
            # Validate settings
            is_valid, error_message = settings_manager.validate_settings(updated_settings)
            
            if is_valid:
                # Save settings
                if settings_manager.save_settings(updated_settings):
                    st.success("✅ Settings saved successfully!")
                    # Update session state
                    st.session_state.reconciliation_settings = updated_settings
                    return updated_settings
                else:
                    st.error("❌ Failed to save settings. Please try again.")
            else:
                st.error(f"❌ Invalid settings: {error_message}")
    
    with col2:
        if st.button("🔄 Reset to Defaults"):
            if settings_manager.save_settings(settings_manager.default_settings):
                st.success("✅ Settings reset to defaults!")
                st.session_state.reconciliation_settings = settings_manager.default_settings
                st.rerun()
            else:
                st.error("❌ Failed to reset settings.")
    
    with col3:
        if st.button("📋 Export Settings"):
            # Create settings export
            export_data = {
                "settings": current_settings,
                "export_date": datetime.now().isoformat(),
                "version": "1.0"
            }
            st.download_button(
                label="⬇️ Download Settings",
                data=json.dumps(export_data, indent=2),
                file_name=f"reconciliation_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col4:
        if st.button("📤 Import Settings"):
            uploaded_file = st.file_uploader(
                "Upload Settings File",
                type=['json'],
                key="settings_uploader"
            )
            if uploaded_file is not None:
                try:
                    import_data = json.load(uploaded_file)
                    if 'settings' in import_data:
                        imported_settings = import_data['settings']
                        is_valid, error_message = settings_manager.validate_settings(imported_settings)
                        if is_valid:
                            if settings_manager.save_settings(imported_settings):
                                st.success("✅ Settings imported successfully!")
                                st.session_state.reconciliation_settings = imported_settings
                                st.rerun()
                            else:
                                st.error("❌ Failed to import settings.")
                        else:
                            st.error(f"❌ Invalid imported settings: {error_message}")
                    else:
                        st.error("❌ Invalid settings file format.")
                except Exception as e:
                    st.error(f"❌ Error importing settings: {str(e)}")
    
    # Show example behavior
    st.markdown("---")
    with st.expander("📋 Example Behavior with Current Settings"):
        st.markdown("""
        **Tax Difference Examples:**
        - Tax difference of ₹8.00 → "No Difference" (≤ ₹{:.2f})
        - Tax difference of ₹12.00 → "Has Difference" (> ₹{:.2f})
        
        **Date Difference Examples:**
        - Date difference of 0 days → "Within Tolerance" (≤ {} days)
        - Date difference of 2 days → "Outside Tolerance" (> {} days)
        
        **Name Preference:**
        - Current preference: **{}**
        - This affects which name field is used for reconciliation comments
        """.format(
            tax_tolerance, tax_tolerance,
            date_tolerance, date_tolerance,
            name_preference
        ))
    
    return current_settings

def apply_settings_to_reconciliation(df: 'pd.DataFrame', settings: Dict[str, Any]) -> 'pd.DataFrame':
    """Apply reconciliation settings to the dataframe."""
    if df is None or df.empty:
        return df
    
    try:
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Apply tax difference status
        if 'IGST Diff' in result_df.columns and 'CGST Diff' in result_df.columns and 'SGST Diff' in result_df.columns:
            tax_tolerance = settings.get('tax_amount_tolerance', 10.0)
            
            # Calculate total tax difference
            result_df['Total Tax Diff'] = (
                result_df['IGST Diff'].fillna(0) + 
                result_df['CGST Diff'].fillna(0) + 
                result_df['SGST Diff'].fillna(0)
            )
            
            # Apply tax diff status - show N/A for Books Only and GSTR-2A Only
            def get_tax_diff_status_with_na(row):
                if row['Status'] in ['Books Only', 'GSTR-2A Only']:
                    return "N/A"
                else:
                    return "No Difference" if abs(row['Total Tax Diff']) <= tax_tolerance else "Has Difference"
            
            result_df['Tax Diff Status'] = result_df.apply(get_tax_diff_status_with_na, axis=1)
        
        # Apply date status - show N/A for Books Only and GSTR-2A Only
        if 'Date Diff' in result_df.columns:
            date_tolerance = settings.get('date_tolerance_days', 1)
            
            def get_date_status_with_na(row):
                if row['Status'] in ['Books Only', 'GSTR-2A Only']:
                    return "N/A"
                else:
                    return "Within Tolerance" if abs(row['Date Diff']) <= date_tolerance else "Outside Tolerance"
            
            result_df['Date Status'] = result_df.apply(get_date_status_with_na, axis=1)
        
        # Apply name preference to comments if applicable
        name_preference = settings.get('name_preference', 'Legal Name')
        if 'Narrative' in result_df.columns:
            # Update narratives to reflect name preference
            result_df['Narrative'] = result_df['Narrative'].astype(str).str.replace(
                'Trade Name', name_preference
            ).str.replace(
                'Legal Name', name_preference
            )
        
        logger.info("Settings applied to reconciliation results successfully")
        return result_df
        
    except Exception as e:
        logger.error(f"Error applying settings to reconciliation: {e}")
        return df

def get_current_settings() -> Dict[str, Any]:
    """Get current settings from session state or load from file."""
    if 'reconciliation_settings' not in st.session_state:
        settings_manager = ReconciliationSettings()
        st.session_state.reconciliation_settings = settings_manager.settings
    
    return st.session_state.reconciliation_settings
