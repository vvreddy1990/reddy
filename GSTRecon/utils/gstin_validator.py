"""
GSTIN Validator Module

This module provides comprehensive GSTIN validation according to Indian GST rules.
Implements all validation conditions including checksum validation.
"""

import re
from typing import Dict, List, Any, Optional
from datetime import datetime

class GSTINValidator:
    """Comprehensive GSTIN validator with detailed error reporting."""
    
    def __init__(self):
        self.valid_state_codes = {
            "01": "Jammu & Kashmir",
            "02": "Himachal Pradesh", 
            "03": "Punjab",
            "04": "Chandigarh",
            "05": "Uttarakhand",
            "06": "Haryana",
            "07": "Delhi",
            "08": "Rajasthan",
            "09": "Uttar Pradesh",
            "10": "Bihar",
            "11": "Sikkim",
            "12": "Arunachal Pradesh",
            "13": "Nagaland",
            "14": "Manipur",
            "15": "Mizoram",
            "16": "Tripura",
            "17": "Meghalaya",
            "18": "Assam",
            "19": "West Bengal",
            "20": "Jharkhand",
            "21": "Odisha",
            "22": "Chhattisgarh",
            "23": "Madhya Pradesh",
            "24": "Gujarat",
            "25": "Dadra and Nagar Haveli and Daman and Diu",
            "27": "Maharashtra",
            "28": "Andhra Pradesh (pre-2014)",
            "29": "Karnataka",
            "30": "Goa",
            "31": "Lakshadweep",
            "32": "Kerala",
            "33": "Tamil Nadu",
            "34": "Puducherry",
            "35": "Andaman & Nicobar Islands",
            "36": "Telangana",
            "37": "Andhra Pradesh (post-2014)",
            "38": "Other Territory"
        }
        
        self.pan_business_types = {
            'A': 'Association of Persons (AOP)',
            'B': 'Body of Individuals (BOI)',
            'C': 'Company',
            'F': 'Firm',
            'G': 'Government',
            'H': 'HUF (Hindu Undivided Family)',
            'L': 'Local Authority',
            'J': 'Artificial Juridical Person',
            'P': 'Individual',
            'T': 'Trust'
        }
    
    def validate_gstin(self, gstin: str) -> Dict[str, Any]:
        """
        Validate a single GSTIN with comprehensive error reporting.
        
        Args:
            gstin: The GSTIN string to validate
            
        Returns:
            Dictionary with validation results and detailed information
        """
        if not gstin or not isinstance(gstin, str):
            return self._create_error_result(gstin, "LENGTH_ERROR", "GSTIN cannot be empty or non-string")
        
        gstin = gstin.strip()
        errors = []
        details = {}
        
        # 1. Length Validation
        if len(gstin) != 15:
            length_msg = f"Invalid GSTIN: Length {len(gstin)}, expected 15"
            if len(gstin) < 15:
                length_msg += f" - Missing {15 - len(gstin)} digit(s) at position(s) {len(gstin) + 1}-15"
            else:
                length_msg += f" - Extra {len(gstin) - 15} digit(s) at position(s) 16-{len(gstin)}"
            errors.append({
                "code": "LENGTH_ERROR",
                "message": length_msg
            })
            return self._create_error_result(gstin, "LENGTH_ERROR", length_msg)
        
        # 2. Case Validation
        if not gstin.isupper():
            # Find specific lowercase characters and their positions
            lowercase_positions = []
            for i, char in enumerate(gstin, 1):
                if char.islower():
                    lowercase_positions.append(f"position {i} ('{char}')")
            
            case_msg = f"Invalid GSTIN: Must be uppercase - Found lowercase at {', '.join(lowercase_positions)}"
            errors.append({
                "code": "CASE_ERROR", 
                "message": case_msg
            })
            return self._create_error_result(gstin, "CASE_ERROR", case_msg)
        
        # 3. State Code Validation
        state_code = gstin[:2]
        if state_code not in self.valid_state_codes:
            state_msg = f"Invalid state code at positions 1-2: '{state_code}' not in valid range (01-38, excluding 26)"
            errors.append({
                "code": "STATE_CODE_ERROR",
                "message": state_msg
            })
            return self._create_error_result(gstin, "STATE_CODE_ERROR", state_msg)
        
        details['state_code'] = state_code
        details['state_name'] = self.valid_state_codes[state_code]
        
        # 4. PAN Number Validation
        pan_number = gstin[2:12]
        pan_validation = self._validate_pan(pan_number)
        if not pan_validation['valid']:
            pan_msg = f"Invalid PAN format at positions 3-12: '{pan_number}' - {pan_validation['error']}"
            errors.append({
                "code": "PAN_FORMAT_ERROR",
                "message": pan_msg
            })
            return self._create_error_result(gstin, "PAN_FORMAT_ERROR", pan_msg)
        
        details['pan_number'] = pan_number
        details['business_type'] = pan_validation['business_type']
        
        # 5. Entity Code Validation
        entity_code = gstin[12]
        if not (entity_code.isdigit() or entity_code.isalpha()):
            entity_msg = f"Invalid entity code at position 13: '{entity_code}' - must be alphanumeric (1-9, A-Z)"
            errors.append({
                "code": "ENTITY_CODE_ERROR",
                "message": entity_msg
            })
            return self._create_error_result(gstin, "ENTITY_CODE_ERROR", entity_msg)
        
        details['entity_code'] = entity_code
        
        # 6. Constant Character Validation
        constant_char = gstin[13]
        if constant_char != 'Z':
            constant_msg = f"Invalid character at position 14: '{constant_char}', expected 'Z'"
            errors.append({
                "code": "CONSTANT_CHAR_ERROR",
                "message": constant_msg
            })
            return self._create_error_result(gstin, "CONSTANT_CHAR_ERROR", constant_msg)
        
        # 7. Checksum Validation
        checksum = gstin[14]
        calculated_checksum = self._calculate_checksum(gstin[:14])
        if checksum != calculated_checksum:
            checksum_msg = f"Checksum mismatch at position 15: Expected '{calculated_checksum}', found '{checksum}' - 15th digit has an issue"
            errors.append({
                "code": "CHECKSUM_ERROR",
                "message": checksum_msg
            })
            return self._create_error_result(gstin, "CHECKSUM_ERROR", checksum_msg)
        
        details['checksum'] = checksum
        
        # All validations passed
        return {
            "status": "Valid",
            "gstin": gstin,
            "errors": [],
            "details": details,
            "validation_timestamp": datetime.now().isoformat()
        }
    
    def _validate_pan(self, pan: str) -> Dict[str, Any]:
        """Validate PAN number format and business type."""
        if len(pan) != 10:
            return {"valid": False, "error": f"PAN must be exactly 10 characters, found {len(pan)}"}
        
        # Check positions 1-5 are letters
        if not re.match(r'^[A-Z]{5}', pan[:5]):
            # Find specific invalid characters
            invalid_chars = []
            for i, char in enumerate(pan[:5], 1):
                if not char.isalpha() or not char.isupper():
                    invalid_chars.append(f"position {i+2} ('{char}')")
            return {"valid": False, "error": f"First 5 characters (positions 3-7) must be letters - Invalid at {', '.join(invalid_chars)}"}
        
        # Check positions 6-9 are digits
        if not re.match(r'^[A-Z]{5}[0-9]{4}', pan):
            # Find specific invalid characters
            invalid_chars = []
            for i, char in enumerate(pan[5:9], 6):
                if not char.isdigit():
                    invalid_chars.append(f"position {i+2} ('{char}')")
            return {"valid": False, "error": f"Positions 6-9 (positions 8-11) must be digits - Invalid at {', '.join(invalid_chars)}"}
        
        # Check position 10 is letter
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', pan):
            last_char = pan[9]
            return {"valid": False, "error": f"Last character (position 12) must be a letter, found '{last_char}'"}
        
        # Check business type (4th character)
        business_type_char = pan[3]
        if business_type_char not in self.pan_business_types:
            return {"valid": False, "error": f"Invalid business type character at position 6: '{business_type_char}' - must be one of {list(self.pan_business_types.keys())}"}
        
        return {
            "valid": True,
            "business_type": self.pan_business_types[business_type_char]
        }
    
    def _calculate_checksum(self, first_14_chars: str) -> str:
        """Calculate GSTIN checksum using the official algorithm."""
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        factor = 1
        total = 0
        
        for char in first_14_chars:
            value = charset.index(char)
            product = value * factor
            factor = 3 - factor  # Alternates between 1 and 2
            total += (product // 36) + (product % 36)
        
        checksum_value = (36 - (total % 36)) % 36
        return charset[checksum_value]
    
    def _create_error_result(self, gstin: str, error_code: str, error_message: str) -> Dict[str, Any]:
        """Create error result structure."""
        return {
            "status": "Invalid",
            "gstin": gstin,
            "errors": [{"code": error_code, "message": error_message}],
            "details": {},
            "validation_timestamp": datetime.now().isoformat()
        }
    
    def validate_gstin_column(self, gstin_series) -> List[Dict[str, Any]]:
        """
        Validate a pandas Series of GSTINs.
        
        Args:
            gstin_series: Pandas Series containing GSTIN values
            
        Returns:
            List of validation results for each GSTIN
        """
        results = []
        
        for idx, gstin in gstin_series.items():
            if pd.isna(gstin) or gstin == '':
                results.append({
                    "status": "Invalid",
                    "gstin": str(gstin),
                    "errors": [{"code": "EMPTY_ERROR", "message": "GSTIN is empty or null"}],
                    "details": {},
                    "validation_timestamp": datetime.now().isoformat()
                })
            else:
                results.append(self.validate_gstin(str(gstin)))
        
        return results
    
    def get_validation_summary(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics from validation results."""
        total = len(validation_results)
        valid_count = sum(1 for result in validation_results if result['status'] == 'Valid')
        invalid_count = total - valid_count
        
        # Count error types
        error_counts = {}
        for result in validation_results:
            for error in result.get('errors', []):
                error_type = error['code']
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_gstins": total,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "validity_percentage": round((valid_count / total * 100), 2) if total > 0 else 0,
            "error_breakdown": error_counts,
            "summary_timestamp": datetime.now().isoformat()
        }

# Import pandas for the validate_gstin_column method
import pandas as pd
