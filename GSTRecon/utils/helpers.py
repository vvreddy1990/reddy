from typing import Union, Any
from datetime import datetime
import pandas as pd
from fuzzywuzzy import fuzz
import re

def format_currency(value: float) -> str:
    """Format currency value with Indian Rupee symbol"""
    return f"₹{value:,.2f}"

def format_percentage(value: float) -> str:
    """Format percentage value"""
    return f"{value:.1f}%"

def format_date(date) -> str:
    """Format date in DD/MM/YYYY format"""
    if pd.isna(date):
        return ""
    return date.strftime("%d/%m/%Y")

def clean_gstin(gstin: str) -> str:
    """Clean and standardize GSTIN number"""
    if pd.isna(gstin):
        return "UNKNOWN"
    return str(gstin).upper().strip()

def clean_invoice_number(invoice: str) -> str:
    """Clean and standardize invoice number"""
    if pd.isna(invoice):
        return "UNKNOWN"
    return str(invoice).upper().strip()

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings"""
    return fuzz.ratio(str(str1), str(str2)) / 100.0

def is_within_tolerance(value1: float, value2: float, tolerance: float = 1.0) -> bool:
    """Check if two values are within tolerance"""
    return abs(value1 - value2) <= tolerance

def is_within_date_range(date1, date2, days: int = 7) -> bool:
    """Check if two dates are within specified days"""
    if pd.isna(date1) or pd.isna(date2):
        return False
    return abs((date1 - date2).days) <= days

def calculate_date_difference(date1: Union[str, datetime], date2: Union[str, datetime]) -> int:
    """Calculate the difference in days between two dates"""
    if isinstance(date1, str):
        date1 = pd.to_datetime(date1)
    if isinstance(date2, str):
        date2 = pd.to_datetime(date2)
    return abs((date1 - date2).days)

def get_fiscal_year(date: Union[str, datetime]) -> str:
    """Get the fiscal year for a given date"""
    if isinstance(date, str):
        date = pd.to_datetime(date)
    year = date.year
    if date.month < 4:
        year -= 1
    return f"{year}-{year+1}"

def validate_tax_amounts(igst: float, cgst: float, sgst: float) -> bool:
    """Validate if tax amounts are reasonable"""
    return all(amount >= 0 for amount in [igst, cgst, sgst])

def calculate_tax_differences(
    books_igst: float,
    books_cgst: float,
    books_sgst: float,
    gstr2a_igst: float,
    gstr2a_cgst: float,
    gstr2a_sgst: float
) -> tuple[float, float, float]:
    """Calculate differences in tax amounts"""
    return (
        abs(books_igst - gstr2a_igst),
        abs(books_cgst - gstr2a_cgst),
        abs(books_sgst - gstr2a_sgst)
    )

def calculate_total_tax_difference(
    books_igst: float,
    books_cgst: float,
    books_sgst: float,
    gstr2a_igst: float,
    gstr2a_cgst: float,
    gstr2a_sgst: float
) -> float:
    """Calculate total tax difference across all tax types"""
    igst_diff = abs(books_igst - gstr2a_igst)
    cgst_diff = abs(books_cgst - gstr2a_cgst)
    sgst_diff = abs(books_sgst - gstr2a_sgst)
    return igst_diff + cgst_diff + sgst_diff

def get_tax_diff_status(tax_diff: float, tolerance: float = 10.0) -> str:
    """Get tax difference status based on tolerance"""
    if abs(tax_diff) <= tolerance:
        return "No Difference"
    else:
        return "Has Difference"

def get_tax_diff_status_with_status(tax_diff: float, status: str, tolerance: float = 10.0) -> str:
    """Get tax difference status based on tolerance and record status"""
    if status in ['Books Only', 'GSTR-2A Only']:
        return "N/A"
    else:
        return get_tax_diff_status(tax_diff, tolerance)

def get_date_status(date_diff_days: int, tolerance_days: int = 1) -> str:
    """Get date status based on tolerance"""
    if abs(date_diff_days) <= tolerance_days:
        return "Within Tolerance"
    else:
        return "Outside Tolerance"

def get_date_status_with_status(date_diff_days: int, status: str, tolerance_days: int = 1) -> str:
    """Get date status based on tolerance and record status"""
    if status in ['Books Only', 'GSTR-2A Only']:
        return "N/A"
    else:
        return get_date_status(date_diff_days, tolerance_days)

def format_currency_with_settings(amount: float, currency: str = "INR", precision: int = 2) -> str:
    """Format currency based on settings"""
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

def clean_company_name(name: str, case_sensitive: bool = False) -> str:
    """Clean company name for comparison"""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    
    cleaned = name.strip()
    if not case_sensitive:
        cleaned = cleaned.lower()
    
    return cleaned

def get_preferred_name(trade_name: str, legal_name: str, preference: str = "Legal Name") -> str:
    """Get the preferred name based on settings"""
    if preference == "Trade Name":
        return trade_name if trade_name else legal_name
    else:
        return legal_name if legal_name else trade_name

def validate_settings_input(value: Any, min_value: float = None, max_value: float = None, 
                           required_type: type = None) -> tuple[bool, str]:
    """Validate settings input values"""
    try:
        if required_type and not isinstance(value, required_type):
            return False, f"Value must be of type {required_type.__name__}"
        
        if min_value is not None and value < min_value:
            return False, f"Value must be >= {min_value}"
        
        if max_value is not None and value > max_value:
            return False, f"Value must be <= {max_value}"
        
        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def calculate_settings_impact(original_df: pd.DataFrame, updated_df: pd.DataFrame) -> dict:
    """Calculate the impact of settings changes on reconciliation results"""
    impact = {
        'tax_diff_changes': 0,
        'date_status_changes': 0,
        'total_changes': 0,
        'changed_rows': []
    }
    
    if original_df is None or updated_df is None:
        return impact
    
    # Count changes in Tax Diff Status
    if 'Tax Diff Status' in original_df.columns and 'Tax Diff Status' in updated_df.columns:
        tax_changes = (original_df['Tax Diff Status'] != updated_df['Tax Diff Status'])
        impact['tax_diff_changes'] = tax_changes.sum()
    
    # Count changes in Date Status
    if 'Date Status' in original_df.columns and 'Date Status' in updated_df.columns:
        date_changes = (original_df['Date Status'] != updated_df['Date Status'])
        impact['date_status_changes'] = date_changes.sum()
    
    # Total changes
    impact['total_changes'] = impact['tax_diff_changes'] + impact['date_status_changes']
    
    # Get changed rows
    if impact['total_changes'] > 0:
        changed_mask = (
            (original_df['Tax Diff Status'] != updated_df['Tax Diff Status']) |
            (original_df['Date Status'] != updated_df['Date Status'])
        )
        impact['changed_rows'] = updated_df[changed_mask].head(10).to_dict('records')
    
    return impact

def format_indian_currency(amount: float) -> str:
    """
    Format amount in Indian currency format with proper comma separation and lakhs/crores notation.
    
    Args:
        amount: Amount to format
    
    Returns:
        Formatted string with Indian currency notation
    """
    if pd.isna(amount) or amount == 0:
        return "₹0"
    
    # Convert to absolute value for formatting
    abs_amount = abs(amount)
    
    if abs_amount >= 10000000:  # 1 crore
        crores = abs_amount / 10000000
        if crores.is_integer():
            formatted = f"₹{int(crores):,} crore"
        else:
            formatted = f"₹{crores:.2f} crore"
    elif abs_amount >= 100000:  # 1 lakh
        lakhs = abs_amount / 100000
        if lakhs.is_integer():
            formatted = f"₹{int(lakhs):,} lakh"
        else:
            formatted = f"₹{lakhs:.2f} lakh"
    else:
        formatted = f"₹{abs_amount:,.2f}"
    
    # Add negative sign if original amount was negative
    if amount < 0:
        formatted = f"-{formatted}"
    
    return formatted

def get_gstr2a_due_date(invoice_date):
    """Return the due date (11th of next month) for a given invoice date."""
    if pd.isnull(invoice_date):
        return None
    if isinstance(invoice_date, str):
        invoice_date = pd.to_datetime(invoice_date, errors='coerce')
    if pd.isnull(invoice_date):
        return None
    year = invoice_date.year + (1 if invoice_date.month == 12 else 0)
    month = 1 if invoice_date.month == 12 else invoice_date.month + 1
    return datetime(year, month, 11)

def get_return_days_lapsed(filing_date, due_date):
    """Return days lapsed (positive if late, 0 if on/before due date, None if invalid)."""
    if pd.isnull(filing_date) or pd.isnull(due_date):
        return None
    if isinstance(filing_date, str):
        filing_date = pd.to_datetime(filing_date, errors='coerce')
    if pd.isnull(filing_date):
        return None
    days = (filing_date - due_date).days
    return days if days > 0 else 0

def get_compliance_status(filing_date, due_date, total_tax):
    """Return 'Compliant' if filing on/before due date, 'Risky (n days, ₹amount)' if late, 'Invalid Filing Date' if missing/invalid."""
    if pd.isnull(filing_date) or pd.isnull(due_date):
        return "Invalid Filing Date"
    if isinstance(filing_date, str):
        filing_date = pd.to_datetime(filing_date, errors='coerce')
    if pd.isnull(filing_date):
        return "Invalid Filing Date"
    if filing_date <= due_date:
        return "Compliant"
    days = (filing_date - due_date).days
    if days > 0:
        return f"Risky ({days} day{'s' if days != 1 else ''}, {format_indian_currency(total_tax)})"
    return "Compliant"

def extract_core_invoice_number(invoice: str) -> str:
    """Extract the core numeric part of an invoice number for fuzzy matching."""
    if pd.isna(invoice) or not isinstance(invoice, str):
        return ""
    # Remove common prefixes, years, and non-numeric parts, keep last numeric group
    # e.g., 'INV-2024-25/002' -> '002', '2024-25/002' -> '002', 'INV-002' -> '002'
    # Remove everything except numbers and split by non-digit
    parts = re.findall(r'\d+', invoice)
    if parts:
        return str(int(parts[-1]))  # Remove leading zeros
    return invoice.strip() 