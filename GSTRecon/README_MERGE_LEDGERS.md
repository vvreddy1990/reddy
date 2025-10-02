# Merge Books Ledgers Feature

## Overview

The "Merge Books Ledgers" feature allows users to combine multiple GST ledger Excel files from different systems (Tally, SAP, Others) into a single dataset for reconciliation. This is particularly useful when you have data from multiple sources or different tax rate categories that need to be consolidated.

## Features

### 1. Template Support
- **Tally**: Predefined mappings for Tally ERP columns
- **SAP**: Predefined mappings for SAP ERP columns  
- **Others**: Auto-detection for custom formats

### 2. Column Detection & Customization
- Automatic detection of standard and tax columns
- **Enhanced tax column detection**: Finds ALL columns with CGST, SGST, or IGST keywords anywhere in column names
- **Comprehensive column display**: Shows all available columns from all uploaded files
- **Smart selection tools**: Select All, Deselect All, and Auto-Detect Tax Columns buttons
- **Tax column indicators**: Visual indicators (🏷️ TAX) for easy identification
- Customizable column mappings with selection summary
- Support for additional columns

### 3. Advanced Settings
- Value sign conversion options:
  - Positive to Negative
  - Negative to Positive  
  - Both (Positive to Negative & Negative to Positive)
- Column renaming
- Smart duplicate detection (only creates status columns when duplicates are found):
  - `Duplicate Status` - Shows "1st Occurrence" or "Duplicate" (only when duplicates exist)
  - `Duplicate Unique ID` - Unique identifier for easy filtering (only when duplicates exist)
- Missing value strategies
- Decimal precision control

### 4. Merging Logic
- Combines multiple files into single dataset
- Creates standardized tax columns by finding ALL columns with tax keywords:
  - `Total CGST Amount` - Sum of ALL columns containing "CGST"
  - `Total SGST Amount` - Sum of ALL columns containing "SGST" 
  - `Total IGST Amount` - Sum of ALL columns containing "IGST"
- Adds traceability column showing source files and tax columns used
- Places `Original Ledger Name` as the first column for easy identification

### 5. Integration
- Seamless integration with existing reconciliation process
- Option to use merged data directly for reconciliation
- Export capabilities (Excel/CSV)

## Usage

1. **Navigate** to "Merge Books Ledgers" in the sidebar
2. **Upload** multiple Excel files (.xlsx or .xls)
3. **Select** appropriate template (Tally/SAP/Others)
4. **Detect** and customize column mappings
5. **Configure** advanced settings if needed
6. **Merge** files to create consolidated dataset
7. **Use** merged data for reconciliation

## File Requirements

- Excel files (.xlsx or .xls format)
- Files should contain GST-related data
- Column headers should be in first row
- Files are validated before processing

## Output

The merged dataset includes:
- `Original Ledger Name` as the first column (shows source file and tax columns)
- All original columns from source files
- Standardized tax amount columns (calculated from ALL tax columns with CGST/SGST/IGST keywords)
- Smart duplicate detection columns (`Duplicate Status`, `Duplicate Unique ID`) - only when duplicates are found
- Summary statistics

## Integration with Reconciliation

After merging, users can:
- Download the merged file for external use
- Use the "Use Merged Data" button in the main reconciliation section
- The merged data is automatically available in session state

## Technical Details

### Files Added
- `utils/merge_ledgers.py` - Core merging logic
- `utils/merge_ledgers_ui.py` - Streamlit UI components

### Files Modified
- `app.py` - Added menu item and integration logic

### Dependencies
- No new dependencies required
- Uses existing pandas, streamlit, and openpyxl libraries

## Error Handling

- File validation with detailed error messages
- Graceful handling of missing columns
- Warning messages for empty or invalid files
- Robust error recovery during merge process

## Performance

- Efficient column detection using regex patterns
- Memory-optimized file processing
- Progress indicators for long-running operations
- Handles large files with streaming approach
