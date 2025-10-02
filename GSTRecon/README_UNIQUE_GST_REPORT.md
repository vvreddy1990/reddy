# Unique GST Report Feature

## Overview

The **Unique GST Report** is now integrated into the **Transactions tab** as a sub-tab, providing comprehensive analysis of GST numbers from GSTR-2A that have no corresponding entries in your Books data. This helps identify missing supplier records, potential data entry errors, and audit supplier master data completeness.

## 🎯 **Key Improvements**

### ✅ **Fixed Issues**
- **Sorting Fixed**: Proper numeric sorting for tax amounts (no more ₹ symbol interference)
- **Accurate Count**: Only counts truly unmapped GST numbers (excludes cases where GSTIN is missing in Books but supplier exists)
- **Integrated UI**: Now part of the Transactions tab for better workflow

### 🚀 **Enhanced Features**
- **Enhanced Analytics**: Comprehensive summary tables and insights
- **Risk Analysis**: Automatic risk scoring based on frequency and amounts
- **State-wise Analysis**: Geographic distribution of unmapped suppliers
- **Tax Distribution**: Detailed breakdown of tax amounts
- **Frequency Analysis**: Understanding supplier transaction patterns
- **Smart Recommendations**: AI-powered suggestions based on data analysis

## Features

### 🔍 **Core Functionality**
- Identifies GST numbers that appear only in GSTR-2A (not in Books)
- Displays associated Trade Names and Legal Names
- **Counts**: Shows how many times each GST number appears in GSTR-2A
- **Tax Amounts**: Displays total IGST, CGST, and SGST amounts for each unmapped GST
- Provides comprehensive summary statistics and insights

### 📊 **Interactive Table**
- **Columns**: GST Number, Trade Name, Legal Name, Count, Total IGST Amount, Total CGST Amount, Total SGST Amount
- **Search**: Filter by GST Number, Trade Name, or Legal Name (case-insensitive)
- **Smart Sorting**: Numeric columns sort by value (descending), text columns alphabetically
- **Serial Numbers**: Automatic row numbering for easy reference
- **Currency Formatting**: Tax amounts displayed with ₹ symbol and proper formatting

### 📥 **Export Options**
- **CSV Export**: Download filtered results as CSV
- **Excel Export**: Download filtered results as Excel (requires openpyxl package)
- **Timestamped Files**: Automatic file naming with date/time

### 📈 **Enhanced Analytics & Insights**
- **Summary Metrics**: Total unmapped GST numbers, total records, total tax amounts
- **Name Completeness Analysis**: Both names, only trade name, only legal name
- **Record Distribution**: Average, maximum, and minimum records per GST number
- **State Code Analysis**: Top 5 state codes from GST numbers
- **Tax Amount Analysis**: Total SGST amount and top 3 GST numbers by IGST amount
- **Real-time Filtering**: Dynamic results based on search criteria

### 🆕 **New Enhanced Analytics**
- **Risk Analysis**: Suppliers categorized as Low/Medium/High risk
- **State-wise Distribution**: Geographic breakdown with percentages
- **Tax Distribution**: IGST/CGST/SGST breakdown with percentages
- **Frequency Analysis**: How many suppliers have 1, 2, 3+ records
- **Top Risk Suppliers**: Top 10 high-risk suppliers requiring attention
- **Smart Recommendations**: AI-powered suggestions based on data patterns

## How to Use

### 1. **Access the Report**
- Navigate to the **Transactions** tab in the main application
- Click on the **"Unique Unmapped GST Report"** sub-tab

### 2. **Upload Data**
- Upload your Excel file containing Books and GSTR-2A data
- Ensure the file has the required columns:
  - `Source Name` (with values "Books" and "GSTR-2A")
  - `Supplier GSTIN` (15-digit GST numbers)
  - `Supplier Trade Name`
  - `Supplier Legal Name`
  - `Total IGST Amount` (or `IGST Amount` or `IGST`)
  - `Total CGST Amount` (or `CGST Amount` or `CGST`)
  - `Total SGST Amount` (or `SGST Amount` or `SGST`)

### 3. **View Results**
- The report automatically generates when data is available
- View summary statistics at the top (4 metrics: Total GST Numbers, Total Records, Total IGST, Total CGST)
- Use search and sort controls to filter results
- Export data using the download buttons

### 4. **Enhanced Analytics**
- Scroll down to see comprehensive analytics and insights
- Review risk analysis and recommendations
- Analyze state-wise and tax distributions
- Identify high-risk suppliers requiring attention

### 5. **Search and Filter**
- **Search Box**: Enter any text to search across all columns
- **Sort Dropdown**: Choose column to sort by (GST Number, Trade Name, Legal Name, Count, Total IGST Amount, Total CGST Amount, Total SGST Amount)
- **Real-time Results**: See filtered count and results immediately

## Technical Details

### **Data Processing**
- **GSTIN Validation**: Only 15-digit GST numbers are included
- **Data Cleaning**: Removes leading/trailing spaces from names
- **Case Normalization**: Converts GSTINs to uppercase
- **Duplicate Handling**: Shows unique GST numbers only
- **Tax Amount Aggregation**: Sums all tax amounts for each GST number across multiple records
- **Accurate Unmapping**: Only considers GST numbers truly missing from Books (excludes null/empty GSTINs)

### **Performance Optimizations**
- **Caching**: Uses `@st.cache_data` for efficient data processing
- **Lazy Loading**: Report generates only when needed
- **Error Handling**: Graceful handling of missing or invalid data

### **Error Handling**
- **Missing Columns**: Shows appropriate error messages
- **Empty Data**: Displays informative messages
- **Invalid GSTINs**: Automatically filters out invalid entries
- **Missing Data Sources**: Handles cases with only Books or only GSTR-2A data
- **Missing Tax Amounts**: Handles missing tax columns gracefully

## Use Cases

### **Audit & Compliance**
- Identify suppliers missing from your master data
- Verify data completeness for GST compliance
- Audit supplier registration status
- **Financial Impact**: Assess total tax amounts for unmapped suppliers
- **Risk Assessment**: Identify high-risk suppliers requiring immediate attention

### **Data Quality**
- Find potential data entry errors
- Identify duplicate or inconsistent supplier records
- Improve master data accuracy
- **Volume Analysis**: Understand how frequently unmapped suppliers appear
- **Geographic Analysis**: Identify state-wise supplier distribution

### **Business Intelligence**
- Analyze supplier distribution by state
- Track supplier onboarding completeness
- Monitor data quality metrics
- **Tax Analysis**: Identify suppliers with highest tax amounts requiring attention
- **Risk Management**: Proactive identification of high-risk suppliers

## Example Output

| S.No. | GST Number       | Trade Name       | Legal Name       | Count | Total IGST Amount | Total CGST Amount | Total SGST Amount |
|-------|------------------|------------------|------------------|-------|-------------------|-------------------|-------------------|
| 1     | 27AAABC1234C1Z5  | ABC Traders      | ABC Pvt Ltd      | 5     | ₹5,000.00         | ₹2,500.00         | ₹2,500.00         |
| 2     | 29XYZCD5678E1Z3  | XYZ Enterprises  | XYZ Corp         | 3     | ₹3,000.00         | ₹1,500.00         | ₹1,500.00         |
| 3     | 33IJKL9012L5M3   | IJK Limited      | IJK Corp         | 2     | ₹2,000.00         | ₹1,000.00         | ₹1,000.00         |

## Enhanced Analytics Example

### **Risk Analysis Summary**
| Risk Level | Count | Percentage |
|------------|-------|------------|
| Low        | 150   | 75.0%      |
| Medium     | 35    | 17.5%      |
| High       | 15    | 7.5%       |

### **State-wise Distribution**
| State Code | Count | Percentage |
|------------|-------|------------|
| 27         | 45    | 22.5%      |
| 29         | 38    | 19.0%      |
| 33         | 32    | 16.0%      |

### **Top Risk Suppliers**
| GST Number       | Trade Name    | Count | Total IGST | Risk Score |
|------------------|---------------|-------|------------|------------|
| 27AAABC1234C1Z5  | ABC Traders   | 15    | ₹50,000    | 0.892      |
| 29XYZCD5678E1Z3  | XYZ Corp      | 12    | ₹35,000    | 0.756      |

## File Structure

```
GSTReconciliationApp/
├── utils/
│   └── reports.py          # Core report functionality with enhanced analytics
├── app.py                  # Main application (updated with sub-tabs)
├── test_reports.py         # Test suite
└── README_UNIQUE_GST_REPORT.md  # This documentation
```

## Dependencies

### **Required Packages**
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations

### **Optional Packages**
- `openpyxl` - Excel export functionality

## Testing

Run the test suite to verify functionality:

```bash
python test_reports.py
```

The test suite covers:
- Basic functionality
- Error handling
- Data validation
- Filtering and sorting
- Tax amount calculations
- Multiple record handling
- Edge cases

## Integration

The Unique GST Report is fully integrated into the existing application:

- **Navigation**: Integrated into Transactions tab as sub-tab
- **Session State**: Uses existing data upload workflow
- **UI Consistency**: Matches existing design patterns
- **Error Handling**: Consistent with application standards
- **Workflow**: Seamless transition between main transactions and unmapped GST analysis

## Future Enhancements

Potential improvements for future versions:

1. **Advanced Filtering**: Date ranges, amount thresholds
2. **Bulk Actions**: Export selected records only
3. **Historical Tracking**: Track changes over time
4. **Integration**: Connect with external supplier databases
5. **Notifications**: Alert for new unmapped GST numbers
6. **Charts**: Visual representation of tax amounts and distributions
7. **Risk Scoring**: Automatically score suppliers based on frequency and amounts
8. **Automated Actions**: Direct integration with supplier onboarding workflows

## Support

For issues or questions about the Unique GST Report:

1. Check the error messages in the application
2. Verify your data format matches requirements
3. Run the test suite to validate functionality
4. Review the logs for detailed error information

---

**Note**: This feature is now seamlessly integrated into the main reconciliation workflow, providing enhanced capabilities for analyzing unmapped suppliers and their financial impact, with comprehensive risk assessment and actionable insights. 