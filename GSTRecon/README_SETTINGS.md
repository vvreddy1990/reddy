# GST Reconciliation Settings Feature

## Overview

The GST Reconciliation App now includes a comprehensive settings feature that allows users to configure various parameters that affect the display and comments in the reconciliation table, specifically the "Tax Diff Status" and "Date Status" columns. These settings do not impact the core reconciliation logic but provide flexibility in how differences are interpreted and displayed.

## Features

### 🎯 Core Settings

1. **Tax Amount Tolerance (₹)**
   - **Default**: ₹10.00
   - **Range**: ₹0.00 - ₹10,000.00
   - **Impact**: Controls when tax differences are considered significant
   - **Behavior**: 
     - If tax difference ≤ tolerance → "No Difference"
     - If tax difference > tolerance → "Has Difference"

2. **Date Tolerance (days)**
   - **Default**: 1 day
   - **Range**: 0 - 365 days
   - **Impact**: Controls when date differences are considered significant
   - **Behavior**:
     - If date difference ≤ tolerance → "Within Tolerance"
     - If date difference > tolerance → "Outside Tolerance"

3. **Name Preference**
   - **Options**: "Legal Name", "Trade Name"
   - **Default**: "Legal Name"
   - **Impact**: Affects name-related reconciliation comments and matching logic

### 🎨 Display Settings

4. **Currency Format**
   - **Options**: INR (₹), USD ($), EUR (€), GBP (£)
   - **Default**: INR
   - **Impact**: How monetary values are displayed in reports

5. **Decimal Precision**
   - **Range**: 0 - 6 decimal places
   - **Default**: 2
   - **Impact**: Number of decimal places for currency and numeric values

6. **Case Sensitive Name Matching**
   - **Default**: False
   - **Impact**: Whether name matching considers letter case

### 🔧 Advanced Settings

7. **Name Similarity Threshold (%)**
   - **Range**: 0% - 100%
   - **Default**: 80%
   - **Impact**: Minimum similarity percentage for name matching

8. **Group Tax Tolerance (₹)**
   - **Default**: ₹50.00
   - **Range**: ₹0.00 - ₹10,000.00
   - **Impact**: Tolerance for tax amounts in group matching scenarios

9. **Enable Advanced Matching**
   - **Default**: True
   - **Impact**: Enable intelligent enhanced matching algorithms

10. **Auto-apply Settings**
    - **Default**: True
    - **Impact**: Automatically apply settings to reconciliation results

## Usage

### Accessing Settings

1. Navigate to the **Settings** page using the sidebar navigation
2. The settings are organized into three tabs:
   - **Basic Settings**: Core reconciliation parameters
   - **Advanced Settings**: Advanced matching and display options
   - **About**: Documentation and current settings summary

### Configuring Settings

1. **Basic Settings Tab**:
   - Adjust tax amount tolerance using the number input
   - Set date tolerance using the number input
   - Choose name preference from the dropdown
   - Select currency format and decimal precision
   - Toggle case sensitivity for name matching

2. **Advanced Settings Tab**:
   - Adjust similarity threshold using the slider
   - Set group tax tolerance
   - Enable/disable advanced matching
   - Toggle auto-apply functionality

### Saving and Managing Settings

- **Save Settings**: Click the "💾 Save Settings" button to persist changes
- **Reset to Defaults**: Click "🔄 Reset to Defaults" to restore default values
- **Export Settings**: Click "📋 Export Settings" to download current settings as JSON
- **Import Settings**: Click "📤 Import Settings" to upload previously exported settings

### Applying Settings

Settings can be applied in two ways:

1. **Automatic Application** (if enabled):
   - Settings are automatically applied when running reconciliation
   - A notification appears confirming settings were applied

2. **Manual Application**:
   - Use the "🔄 Apply Settings to Results" button on the Settings page
   - This applies current settings to existing reconciliation results

### Viewing Settings Impact

Use the "📊 View Settings Impact" button to see how current settings would affect existing reconciliation results:

- Shows count of changes in Tax Diff Status
- Shows count of changes in Date Status
- Displays sample of rows that would be affected

## Example Behavior

### Tax Difference Examples

With Tax Amount Tolerance set to ₹1,000:

| Tax Difference | Status |
|----------------|--------|
| ₹800.00 | "No Difference" (≤ ₹1,000) |
| ₹1,200.00 | "Has Difference" (> ₹1,000) |
| ₹0.00 | "No Difference" (≤ ₹1,000) |

**Note**: For "Books Only" and "GSTR-2A Only" records, Tax Diff Status shows "N/A" since there's no comparison record.

### Date Difference Examples

With Date Tolerance set to 1 day:

| Date Difference | Status |
|----------------|--------|
| 0 days | "Within Tolerance" (≤ 1 day) |
| 1 day | "Within Tolerance" (≤ 1 day) |
| 2 days | "Outside Tolerance" (> 1 day) |
| -1 day | "Within Tolerance" (≤ 1 day) |

**Note**: For "Books Only" and "GSTR-2A Only" records, Date Status shows "N/A" since there's no comparison record.

### Status-Specific Behavior

| Record Status | Tax Diff Status | Date Status | Explanation |
|---------------|-----------------|-------------|-------------|
| Exact Match | Based on tolerance | Based on tolerance | Full comparison available |
| Partial Match | Based on tolerance | Based on tolerance | Partial comparison available |
| Books Only | N/A | N/A | No GSTR-2A record to compare |
| GSTR-2A Only | N/A | N/A | No Books record to compare |
| Group Match | Based on tolerance | Based on tolerance | Group comparison available |

## Technical Implementation

### File Structure

```