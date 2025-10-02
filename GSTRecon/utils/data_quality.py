"""
Data Quality Engine for GST Reconciliation System

This module provides comprehensive data cleaning and standardization capabilities
for GST reconciliation data, including GSTIN validation, company name normalization,
and data quality reporting with audit trails.
"""

import re
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib


@dataclass
class ValidationError:
    """Represents a data validation error with context"""
    field: str
    original_value: str
    error_type: str
    message: str
    suggested_fix: Optional[str] = None


@dataclass
class DataCleaningResult:
    """Result of data cleaning operation with audit trail"""
    cleaned_value: str
    original_value: str
    changes_made: List[str]
    confidence: float
    validation_passed: bool


@dataclass
class DataQualityReport:
    """Comprehensive report of data quality operations"""
    total_records: int
    records_cleaned: int
    gstin_corrections: int
    name_normalizations: int
    validation_errors: List[ValidationError]
    cleaning_summary: Dict[str, int]
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class GSTINValidator:
    """GSTIN validation and cleaning utilities"""
    
    # GSTIN format: 2 digits (state) + 10 alphanumeric (PAN) + 1 digit (entity) + 1 alphanumeric (Z) + 1 alphanumeric (checksum)
    GSTIN_PATTERN = re.compile(r'^[0-9]{2}[A-Z0-9]{10}[0-9][A-Z0-9][A-Z0-9]$')
    
    # Common GSTIN formatting errors
    COMMON_ERRORS = {
        'lowercase': r'[a-z]',
        'spaces': r'\s',
        'special_chars': r'[^A-Z0-9]',
        'wrong_length': lambda x: len(x) != 15
    }
    
    @staticmethod
    def is_valid_gstin(gstin: str) -> bool:
        """Check if GSTIN is valid format"""
        if not gstin or len(gstin) != 15:
            return False
        return bool(GSTINValidator.GSTIN_PATTERN.match(gstin))
    
    @staticmethod
    def calculate_checksum(gstin_without_checksum: str) -> str:
        """Calculate GSTIN checksum digit"""
        if len(gstin_without_checksum) != 14:
            raise ValueError("GSTIN without checksum must be 14 characters")
        
        # GSTIN checksum calculation algorithm
        factor = 2
        sum_val = 0
        
        for char in reversed(gstin_without_checksum):
            if char.isdigit():
                digit = int(char)
            else:
                # A=10, B=11, ..., Z=35
                digit = ord(char) - ord('A') + 10
            
            prod = digit * factor
            sum_val += prod // 36 + prod % 36
            factor = 1 if factor == 2 else 2
        
        checksum = (36 - (sum_val % 36)) % 36
        return str(checksum) if checksum < 10 else chr(ord('A') + checksum - 10)
    
    @staticmethod
    def validate_checksum(gstin: str) -> bool:
        """Validate GSTIN checksum"""
        if len(gstin) != 15:
            return False
        
        try:
            calculated_checksum = GSTINValidator.calculate_checksum(gstin[:14])
            return calculated_checksum == gstin[14]
        except Exception:
            return False
    
    @staticmethod
    def clean_gstin(gstin: str) -> DataCleaningResult:
        """Clean and standardize GSTIN format"""
        if not gstin:
            return DataCleaningResult(
                cleaned_value="",
                original_value=gstin,
                changes_made=["Empty GSTIN"],
                confidence=0.0,
                validation_passed=False
            )
        
        original = str(gstin).strip()
        cleaned = original
        changes = []
        
        # Remove spaces and special characters
        if re.search(r'\s', cleaned):
            cleaned = re.sub(r'\s', '', cleaned)
            changes.append("Removed spaces")
        
        # Convert to uppercase
        if cleaned != cleaned.upper():
            cleaned = cleaned.upper()
            changes.append("Converted to uppercase")
        
        # Remove non-alphanumeric characters except valid GSTIN chars
        invalid_chars = re.sub(r'[A-Z0-9]', '', cleaned)
        if invalid_chars:
            cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
            changes.append("Removed invalid characters")
        
        # Check length
        if len(cleaned) != 15:
            if len(cleaned) < 15:
                changes.append(f"GSTIN too short: {len(cleaned)} characters")
            else:
                cleaned = cleaned[:15]
                changes.append(f"Truncated to 15 characters")
        
        # Validate format and checksum
        is_valid_format = GSTINValidator.is_valid_gstin(cleaned)
        is_valid_checksum = False
        
        if is_valid_format and len(cleaned) == 15:
            is_valid_checksum = GSTINValidator.validate_checksum(cleaned)
            if not is_valid_checksum:
                # Try to fix checksum
                try:
                    correct_checksum = GSTINValidator.calculate_checksum(cleaned[:14])
                    if cleaned[14] != correct_checksum:
                        cleaned = cleaned[:14] + correct_checksum
                        changes.append(f"Fixed checksum: {original[14]} -> {correct_checksum}")
                        is_valid_checksum = True
                except Exception:
                    changes.append("Could not calculate correct checksum")
        
        # Calculate confidence score
        confidence = 1.0
        if not is_valid_format:
            confidence -= 0.4
        if not is_valid_checksum and len(cleaned) == 15:
            confidence -= 0.2  # Less penalty if we can't validate checksum
        if len(changes) > 2:
            confidence -= 0.2
        
        # If no changes were needed and format is valid, high confidence
        if len(changes) == 0 and is_valid_format:
            confidence = 1.0
        
        confidence = max(0.0, confidence)
        
        return DataCleaningResult(
            cleaned_value=cleaned,
            original_value=original,
            changes_made=changes,
            confidence=confidence,
            validation_passed=is_valid_format and is_valid_checksum
        )


class CompanyNameNormalizer:
    """Company name normalization and standardization"""
    
    # Common business suffixes to standardize
    BUSINESS_SUFFIXES = {
        'private limited': 'PVT LTD',
        'pvt ltd': 'PVT LTD',
        'pvt. ltd.': 'PVT LTD',
        'private ltd': 'PVT LTD',
        'pvt limited': 'PVT LTD',
        'limited': 'LTD',
        'ltd': 'LTD',
        'ltd.': 'LTD',
        'llp': 'LLP',
        'l.l.p.': 'LLP',
        'l.l.p': 'LLP',
        'limited liability partnership': 'LLP',
        'company': 'CO',
        'co.': 'CO',
        'corporation': 'CORP',
        'corp': 'CORP',
        'corp.': 'CORP',
        'incorporated': 'INC',
        'inc': 'INC',
        'inc.': 'INC',
        'enterprises': 'ENTERPRISES',
        'enterprise': 'ENTERPRISES',
        'industries': 'INDUSTRIES',
        'industry': 'INDUSTRIES',
        'traders': 'TRADERS',
        'trader': 'TRADERS',
        'trading': 'TRADING',
        'exports': 'EXPORTS',
        'export': 'EXPORTS',
        'imports': 'IMPORTS',
        'import': 'IMPORTS'
    }
    
    # Words to remove from company names
    NOISE_WORDS = {
        'the', 'and', '&', 'of', 'in', 'at', 'by', 'for', 'with', 'on'
    }
    
    @staticmethod
    def normalize_name(name: str) -> DataCleaningResult:
        """Normalize company name with standardization rules"""
        if not name:
            return DataCleaningResult(
                cleaned_value="",
                original_value=name,
                changes_made=["Empty name"],
                confidence=0.0,
                validation_passed=False
            )
        
        original = str(name).strip()
        cleaned = original
        changes = []
        
        # Remove extra whitespace first
        if re.search(r'\s{2,}', cleaned):
            cleaned = re.sub(r'\s+', ' ', cleaned)
            changes.append("Normalized whitespace")
        
        # Remove special characters but keep essential ones
        special_chars = re.findall(r'[^\w\s&.-]', cleaned)
        if special_chars:
            cleaned = re.sub(r'[^\w\s&.-]', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            changes.append(f"Removed special characters: {''.join(set(special_chars))}")
        
        # Standardize business suffixes (before case conversion to preserve original case for matching)
        cleaned_lower = cleaned.lower()
        for suffix, standard in CompanyNameNormalizer.BUSINESS_SUFFIXES.items():
            # Create pattern that matches the suffix with optional trailing dots/spaces
            pattern = r'\s*\b' + re.escape(suffix) + r'\.?\s*$'
            if re.search(pattern, cleaned_lower):
                # Replace at the end of the name
                old_cleaned = cleaned
                cleaned = re.sub(pattern, ' ' + standard, cleaned, flags=re.IGNORECASE).strip()
                if cleaned != old_cleaned:
                    changes.append(f"Standardized suffix: {suffix} -> {standard}")
                    break
        
        # Convert to uppercase for processing (business names are typically uppercase)
        if cleaned != cleaned.upper():
            cleaned = cleaned.upper()
            changes.append("Converted to uppercase")
        
        # Remove noise words (but be careful not to remove important parts)
        words = cleaned.split()
        if len(words) > 3:  # Only remove noise words if name is long enough
            filtered_words = []
            for word in words:
                if word.lower() not in CompanyNameNormalizer.NOISE_WORDS or len(filtered_words) == 0:
                    filtered_words.append(word)
            
            if len(filtered_words) != len(words):
                cleaned = ' '.join(filtered_words)
                changes.append("Removed noise words")
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        # Calculate confidence
        confidence = 1.0
        if len(changes) > 3:
            confidence -= 0.2
        if len(cleaned) < 3:
            confidence -= 0.5
        
        confidence = max(0.0, confidence)
        
        return DataCleaningResult(
            cleaned_value=cleaned,
            original_value=original,
            changes_made=changes,
            confidence=confidence,
            validation_passed=len(cleaned) >= 3
        )


class DataQualityEngine:
    """Main data quality engine for GST reconciliation data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data quality engine with configuration"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration options
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.7)
        self.enable_gstin_checksum_fix = self.config.get('enable_gstin_checksum_fix', True)
        self.enable_name_normalization = self.config.get('enable_name_normalization', True)
        self.log_all_changes = self.config.get('log_all_changes', True)
        
        # Statistics tracking
        self.stats = {
            'gstin_cleaned': 0,
            'gstin_errors': 0,
            'names_normalized': 0,
            'validation_errors': 0
        }
    
    def clean_gstin_data(self, gstins: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Clean and standardize GSTIN data"""
        import time
        start_time = time.time()
        cleaned_gstins = []
        change_log = []
        
        for idx, gstin in gstins.items():
            result = GSTINValidator.clean_gstin(gstin)
            cleaned_gstins.append(result.cleaned_value)
            
            if result.changes_made:
                self.stats['gstin_cleaned'] += 1
                log_entry = f"Row {idx}: {result.original_value} -> {result.cleaned_value} ({', '.join(result.changes_made)})"
                change_log.append(log_entry)
                
                if self.log_all_changes:
                    self.logger.info(f"GSTIN cleaned: {log_entry}")
            
            if not result.validation_passed:
                self.stats['gstin_errors'] += 1
                if result.confidence < self.min_confidence_threshold:
                    self.logger.warning(f"Low confidence GSTIN cleaning at row {idx}: {result.original_value}")
        
        processing_time = time.time() - start_time
        self.logger.info(f"GSTIN cleaning completed in {processing_time:.2f}s. Cleaned: {self.stats['gstin_cleaned']}, Errors: {self.stats['gstin_errors']}")
        
        return pd.Series(cleaned_gstins, index=gstins.index), change_log
    
    def normalize_company_names(self, names: pd.Series) -> Tuple[pd.Series, List[str]]:
        """Normalize company names with standardization rules"""
        if not self.enable_name_normalization:
            return names, []
        
        import time
        start_time = time.time()
        normalized_names = []
        change_log = []
        
        for idx, name in names.items():
            result = CompanyNameNormalizer.normalize_name(name)
            normalized_names.append(result.cleaned_value)
            
            if result.changes_made and result.original_value != result.cleaned_value:
                self.stats['names_normalized'] += 1
                log_entry = f"Row {idx}: {result.original_value} -> {result.cleaned_value} ({', '.join(result.changes_made)})"
                change_log.append(log_entry)
                
                if self.log_all_changes:
                    self.logger.info(f"Name normalized: {log_entry}")
        
        processing_time = time.time() - start_time
        self.logger.info(f"Name normalization completed in {processing_time:.2f}s. Normalized: {self.stats['names_normalized']}")
        
        return pd.Series(normalized_names, index=names.index), change_log
    
    def validate_tax_amounts(self, df: pd.DataFrame) -> List[ValidationError]:
        """Validate tax amounts and flag errors"""
        validation_errors = []
        
        # Identify tax amount columns dynamically
        tax_keywords = ['tax', 'igst', 'cgst', 'sgst', 'cess', 'amount', 'value', 'taxable']
        available_columns = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in tax_keywords):
                # Check if column contains numeric data
                if pd.api.types.is_numeric_dtype(df[col]):
                    available_columns.append(col)
                else:
                    # Try to convert to numeric
                    try:
                        pd.to_numeric(df[col], errors='coerce')
                        available_columns.append(col)
                    except:
                        continue
        
        if not available_columns:
            self.logger.warning("No tax amount columns found for validation")
            return validation_errors
        
        for idx, row in df.iterrows():
            # Check for negative tax amounts
            for col in available_columns:
                if col in row and pd.notna(row[col]):
                    try:
                        value = float(row[col])
                        if value < 0:
                            validation_errors.append(ValidationError(
                                field=col,
                                original_value=str(row[col]),
                                error_type="negative_amount",
                                message=f"Negative tax amount in {col}",
                                suggested_fix="Review and correct negative amount"
                            ))
                    except (ValueError, TypeError):
                        validation_errors.append(ValidationError(
                            field=col,
                            original_value=str(row[col]),
                            error_type="invalid_format",
                            message=f"Invalid number format in {col}",
                            suggested_fix="Convert to valid number format"
                        ))
            
            # Check for unrealistic tax amounts (basic sanity check)
            # Look for taxable value columns with flexible naming
            taxable_cols = [col for col in available_columns if 'taxable' in col.lower() or 'value' in col.lower()]
            for col in taxable_cols:
                if col in row and pd.notna(row[col]):
                    try:
                        taxable_value = float(row[col])
                        if taxable_value > 10000000:  # 1 crore threshold
                            validation_errors.append(ValidationError(
                                field=col,
                                original_value=str(row[col]),
                                error_type="unusually_high",
                                message=f"Unusually high {col.lower()}",
                                suggested_fix="Verify amount accuracy"
                            ))
                    except (ValueError, TypeError):
                        pass
        
        self.stats['validation_errors'] = len(validation_errors)
        return validation_errors
    
    def generate_quality_report(self, total_records: int, processing_time: float, 
                              gstin_changes: List[str], name_changes: List[str], 
                              validation_errors: List[ValidationError]) -> DataQualityReport:
        """Generate comprehensive data quality report"""
        return DataQualityReport(
            total_records=total_records,
            records_cleaned=self.stats['gstin_cleaned'] + self.stats['names_normalized'],
            gstin_corrections=self.stats['gstin_cleaned'],
            name_normalizations=self.stats['names_normalized'],
            validation_errors=validation_errors,
            cleaning_summary={
                'gstin_cleaned': self.stats['gstin_cleaned'],
                'gstin_errors': self.stats['gstin_errors'],
                'names_normalized': self.stats['names_normalized'],
                'validation_errors': self.stats['validation_errors']
            },
            processing_time=processing_time
        )
    
    def clean_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Clean entire dataframe with comprehensive quality checks"""
        import time
        start_time = time.time()
        cleaned_df = df.copy()
        all_gstin_changes = []
        all_name_changes = []
        
        # Reset stats for this operation
        self.stats = {key: 0 for key in self.stats}
        
        # Clean GSTIN columns
        gstin_columns = [col for col in df.columns if 'gstin' in col.lower()]
        for col in gstin_columns:
            if col in cleaned_df.columns:
                cleaned_series, changes = self.clean_gstin_data(cleaned_df[col])
                cleaned_df[col] = cleaned_series
                all_gstin_changes.extend([f"{col}: {change}" for change in changes])
        
        # Normalize company name columns
        name_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['name', 'supplier', 'company'])]
        for col in name_columns:
            if col in cleaned_df.columns:
                normalized_series, changes = self.normalize_company_names(cleaned_df[col])
                cleaned_df[col] = normalized_series
                all_name_changes.extend([f"{col}: {change}" for change in changes])
        
        # Validate tax amounts
        validation_errors = self.validate_tax_amounts(cleaned_df)
        
        # Generate report
        processing_time = time.time() - start_time
        report = self.generate_quality_report(
            total_records=len(df),
            processing_time=processing_time,
            gstin_changes=all_gstin_changes,
            name_changes=all_name_changes,
            validation_errors=validation_errors
        )
        
        self.logger.info(f"Data quality processing completed: {report.records_cleaned} records cleaned in {processing_time:.2f}s")
        
        return cleaned_df, report