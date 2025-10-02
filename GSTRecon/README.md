# GST Reconciliation Dashboard

A modern web application for GST reconciliation between books and GSTR-2A data. This application provides an intuitive interface for matching and analyzing GST transactions with advanced features like fuzzy matching, tax discrepancy analysis, and detailed reporting.

## Features

- **Modern Web Interface**: Built with Streamlit for a clean, responsive user experience
- **Advanced Matching**: Multiple levels of matching (exact, partial, fuzzy)
- **Tax Discrepancy Analysis**: Detailed analysis of IGST, CGST, and SGST differences
- **Interactive Visualizations**: Pie charts and metrics for quick insights
- **Export Capabilities**: Export results to Excel for further analysis
- **Data Validation**: Built-in validation for GSTIN, invoice numbers, and tax amounts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gst-reconciliation.git
cd gst-reconciliation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload your Excel file containing books and GSTR-2A data

4. View the reconciliation results and export as needed

## Input Data Format

The application expects an Excel file with the following columns:

- Source Name (BOOKS/GSTR-2A)
- Supplier GSTIN
- Supplier Legal Name
- Supplier Trade Name
- Invoice Date
- Books Date
- Invoice Number
- Total Taxable Value
- Total Tax Value
- Total IGST Amount
- Total CGST Amount
- Total SGST Amount
- Total Invoice Value

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 