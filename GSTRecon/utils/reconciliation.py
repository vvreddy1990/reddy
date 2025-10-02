import pandas as pd
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from utils.helpers import clean_gstin, extract_core_invoice_number
import re
import time

# Add this helper function at the top (after imports)
def _clean_trade_name(name):
    if not isinstance(name, str):
        return ''
    # Remove common company suffixes and keywords
    suffixes = [
        r'private limited', r'private ltd', r'pvt ltd', r'pvt. ltd', r'limited', r'ltd', r'public limited', r'plc', r'corporation', r'corp', r'co\.?', r'company', r'industries', r'enterprise', r'enterprises', r'group', r'holding', r'holdings', r'finance', r'finserv', r'fintech', r'bank', r'banking', r'of india', r'of', r'and', r'&', r'\bindia\b'
    ]
    pattern = re.compile(r'\b(?:' + '|'.join(suffixes) + r')\b', re.IGNORECASE)
    cleaned = pattern.sub('', name)
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', cleaned)  # Remove special chars
    return cleaned.strip().lower()

def _get_financial_year(date):
    if pd.isna(date):
        return None
    year = date.year
    if date.month < 4:
        return f"{year-1}-{year}"
    else:
        return f"{year}-{year+1}"

def _clean_company_name(name):
    if not isinstance(name, str):
        return ''
    # Remove common company suffixes and keywords (expanded list)
    suffixes = [
        r'private limited', r'private ltd', r'pvt ltd', r'pvt. ltd', r'limited', r'ltd', r'public limited', r'plc',
        r'corporation', r'corp', r'co\.?', r'company', r'industries', r'industry', r'enterprise', r'enterprises',
        r'group', r'holding', r'holdings', r'finance', r'finserv', r'fintech', r'bank', r'banking', r'of india',
        r'of', r'and', r'&', r'\bindia\b', r'consultancy', r'services', r'service', r'technologies', r'technology',
        r'solutions', r'solution', r'global', r'international', r'exports?', r'import', r'traders?', r'trading',
        r'logistics', r'chemicals?', r'foods?', r'beverages?', r'pharma', r'pharmaceuticals?', r'construction',
        r'builders?', r'engineering', r'automotive', r'automobiles?', r'textiles?', r'garments?', r'plastics?',
        r'manufacturers?', r'manufacturing', r'packaging', r'packers?', r'processors?', r'processors?', r'\binc\b',
        r'\bllp\b', r'\bllc\b', r'\bopc\b', r'\bpartnership\b', r'\bpartnership firm\b', r'\bproprietorship\b',
        r'\bproprietor\b', r'\bpartnership\b', r'\bpartnership firm\b', r'\bproprietorship\b', r'\bproprietor\b'
    ]
    pattern = re.compile(r'\b(?:' + '|'.join(suffixes) + r')\b', re.IGNORECASE)
    cleaned = pattern.sub('', name)
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', cleaned)  # Remove special chars
    return cleaned.strip().lower()

class GSTReconciliation:
    def __init__(self, df, gstin_comments=None):
        # Initialize progress tracking
        self.total_steps = 6  # Total number of steps in the process
        self.current_step = 1
        self.step_progress = 0.0
        
        # Store original data for comparison
        self.original_df = df.copy()
        
        # Store all original column names
        self.all_original_columns = df.columns.tolist()
        
        # Store GSTIN comments
        self.gstin_comments = gstin_comments or []
        
        # Validate required columns
        required_columns = [
            'Source Name', 'Supplier GSTIN', 'Supplier Legal Name', 'Supplier Trade Name',
            'Invoice Date', 'Books Date', 'Invoice Number', 'Total Taxable Value',
            'Total Tax Value', 'Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount',
            'Total Invoice Value'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Validate Source Name values
        invalid_sources = df[~df['Source Name'].isin(['Books', 'GSTR-2A'])]['Source Name'].unique()
        if len(invalid_sources) > 0:
            raise ValueError(f"Invalid Source Name values found: {', '.join(invalid_sources)}. Only 'Books' or 'GSTR-2A' are allowed.")
        
        # Split data into books and GSTR-2A records
        self.books_df = df[df['Source Name'] == 'Books'].copy()
        self.gstr2a_df = df[df['Source Name'] == 'GSTR-2A'].copy()
        
        # Debug: Print original counts
        print(f"Original Books count: {len(self.books_df)}")
        print(f"Original GSTR-2A count: {len(self.gstr2a_df)}")
        
        if len(self.books_df) == 0:
            raise ValueError("No Books records found in the data")
        if len(self.gstr2a_df) == 0:
            raise ValueError("No GSTR-2A records found in the data")
        
        # Initialize processed flag
        self.books_df['processed'] = False
        self.gstr2a_df['processed'] = False
        
        # Initialize counters
        self.matched_count = 0
        self.partial_count = 0
        self.group_count = 0
        self.sign_cancellation_count = 0
        self.matched_value = 0
        self.partial_value = 0
        self.group_value = 0
        self.sign_cancellation_value = 0
        self.match_id_counter = 1
        self.group_id_counter = 1
        
        # Initialize result DataFrames
        self.matched_df = pd.DataFrame()
        self.mismatch_df = pd.DataFrame()
        self.final_report_df = pd.DataFrame()
        self.books_report_df = pd.DataFrame()
        self.gstr2a_report_df = pd.DataFrame()
        
        # Set tolerance values
        self.tax_tolerance = 10.0  # ₹10 tolerance for tax amounts
        self.date_tolerance = pd.Timedelta(days=1)  # 1 day tolerance for dates
        self.group_tax_tolerance = 50.0  # ₹50 tolerance for group matches
        
        # Clean data and process reconciliation
        self._clean_data()
        self._process_reconciliation()
        
        # Generate summaries
        self.raw_summary = self._generate_raw_summary()
        self.recon_summary = self._generate_recon_summary()
        self.integrity_checks = self._perform_integrity_checks()
        
        # Debug: Print final counts
        print(f"Final Books count: {len(self.books_df)}")
        print(f"Final GSTR-2A count: {len(self.gstr2a_df)}")
        print(f"Matched count: {len(self.matched_df)}")
        print(f"Mismatch count: {len(self.mismatch_df)}")
    
    def _clean_data(self):
        """Clean and standardize data in both DataFrames."""
        try:
            # Standardize GSTIN format
            self.books_df['Supplier GSTIN'] = self.books_df['Supplier GSTIN'].fillna('').astype(str).str.strip().str.upper()
            self.gstr2a_df['Supplier GSTIN'] = self.gstr2a_df['Supplier GSTIN'].fillna('').astype(str).str.strip().str.upper()
            
            # Initialize Status column if not exists
            if 'Status' not in self.books_df.columns:
                self.books_df['Status'] = ''
            if 'Status' not in self.gstr2a_df.columns:
                self.gstr2a_df['Status'] = ''
            
            # Apply GSTIN comments
            self._apply_gstin_comments()
            
            # Standardize invoice numbers
            self.books_df['Invoice Number'] = self.books_df['Invoice Number'].fillna('').astype(str).str.strip()
            self.gstr2a_df['Invoice Number'] = self.gstr2a_df['Invoice Number'].fillna('').astype(str).str.strip()
            
            # Convert dates to datetime
            self.books_df['Invoice Date'] = pd.to_datetime(self.books_df['Invoice Date'], errors='coerce')
            self.books_df['Books Date'] = pd.to_datetime(self.books_df['Books Date'], errors='coerce')
            self.gstr2a_df['Invoice Date'] = pd.to_datetime(self.gstr2a_df['Invoice Date'], errors='coerce')
            
            # Convert numeric columns
            numeric_columns = [
                'Total Taxable Value', 'Total Tax Value', 'Total IGST Amount',
                'Total CGST Amount', 'Total SGST Amount', 'Total Invoice Value'
            ]
            
            for col in numeric_columns:
                self.books_df[col] = pd.to_numeric(self.books_df[col], errors='coerce').fillna(0)
                self.gstr2a_df[col] = pd.to_numeric(self.gstr2a_df[col], errors='coerce').fillna(0)
            
            # Update progress
            self.step_progress = 1.0
            self.current_step = 2
            self.step_progress = 0.0
            
        except Exception as e:
            raise Exception(f"Error cleaning data: {str(e)}")
    
    def _apply_gstin_comments(self):
        """Apply GSTIN comments to the Status column."""
        try:
            # Debug: Print initial row counts
            print("\nApplying GSTIN comments...")
            print(f"Initial Books count: {len(self.books_df)}")
            print(f"Initial GSTR-2A count: {len(self.gstr2a_df)}")
            
            # Process each GSTIN comment set
            for comment_set in self.gstin_comments:
                gstins = [clean_gstin(gstin) for gstin in comment_set['gstins']]
                comment = comment_set['comment']
                source = comment_set['source']
                
                # Apply to Books if source is Books or Both
                if source in ['Books', 'Both']:
                    books_mask = self.books_df['Supplier GSTIN'].isin(gstins)
                    self.books_df.loc[books_mask, 'Status'] = comment
                    print(f"Applied comment '{comment}' to {books_mask.sum()} rows in Books for GSTINs {gstins}")
                
                # Apply to GSTR-2A if source is GSTR-2A or Both
                if source in ['GSTR-2A', 'Both']:
                    gstr2a_mask = self.gstr2a_df['Supplier GSTIN'].isin(gstins)
                    self.gstr2a_df.loc[gstr2a_mask, 'Status'] = comment
                    print(f"Applied comment '{comment}' to {gstr2a_mask.sum()} rows in GSTR-2A for GSTINs {gstins}")
            
            # Debug: Print final row counts
            print(f"Final Books count: {len(self.books_df)}")
            print(f"Final GSTR-2A count: {len(self.gstr2a_df)}")
            
        except Exception as e:
            print(f"Error applying GSTIN comments: {str(e)}")
            raise
    
    def _process_reconciliation(self):
        """Process the reconciliation in steps."""
        try:
            # Debug: Print initial state
            print("\nInitial State:")
            print(f"Total Books records: {len(self.books_df)}")
            print(f"Total GSTR-2A records: {len(self.gstr2a_df)}")
            print(f"Total records: {len(self.original_df)}")
            
            # Step 1: Process exact matches
            self.current_step = 1
            self.step_progress = 0.0
            print("\nProcessing exact matches...")
            self._process_exact_matches()
            print(f"After exact matches - Books processed: {self.books_df['processed'].sum()}")
            print(f"After exact matches - GSTR-2A processed: {self.gstr2a_df['processed'].sum()}")
            
            # Step 2: Process tax-based matches
            self.current_step = 2
            self.step_progress = 0.0
            print("\nProcessing tax-based matches...")
            self._process_tax_based_matches()
            print(f"After tax-based matches - Books processed: {self.books_df['processed'].sum()}")
            print(f"After tax-based matches - GSTR-2A processed: {self.gstr2a_df['processed'].sum()}")
            
            # Step 3: Process group matches
            self.current_step = 3
            self.step_progress = 0.0
            print("\nProcessing group matches...")
            self._process_group_matches()
            print(f"After group matches - Books processed: {self.books_df['processed'].sum()}")
            print(f"After group matches - GSTR-2A processed: {self.gstr2a_df['processed'].sum()}")
            
            # Step 4: Process sign cancellations
            self.current_step = 4
            self.step_progress = 0.0
            print("\nProcessing sign cancellations...")
            self._process_sign_cancellations()
            print(f"After sign cancellations - Books processed: {self.books_df['processed'].sum()}")
            print(f"After sign cancellations - GSTR-2A processed: {self.gstr2a_df['processed'].sum()}")

            # Step 4.5: Process missing GSTIN matches
            print("\nProcessing missing GSTIN matches...")
            self._process_missing_gstin_matches()
            print(f"After missing GSTIN matches - Books processed: {self.books_df['processed'].sum()}")
            print(f"After missing GSTIN matches - GSTR-2A processed: {self.gstr2a_df['processed'].sum()}")

            # Step 4.6: Flag potential tax deviation matches for manual review
            print("\nFlagging potential matches with high tax deviation for manual review...")
            self._flag_potential_tax_deviation_matches()
            print(f"After potential match flagging - Books processed: {self.books_df['processed'].sum()}")
            print(f"After potential match flagging - GSTR-2A processed: {self.gstr2a_df['processed'].sum()}")

            # Step 5: Add remaining mismatches
            self.current_step = 5
            self.step_progress = 0.0
            print("\nProcessing remaining mismatches...")
            self._add_remaining_mismatches()
            print(f"After remaining mismatches - Books processed: {self.books_df['processed'].sum()}")
            print(f"After remaining mismatches - GSTR-2A processed: {self.gstr2a_df['processed'].sum()}")
            
            # Step 6: Generate final report
            self.current_step = 6
            self.step_progress = 0.0
            print("\nGenerating final report...")
            self._generate_final_report()
            
            # Set progress to 100%
            self.step_progress = 1.0
            
            # Debug: Print final state
            print("\nFinal State:")
            print(f"Total Books records: {len(self.books_df)}")
            print(f"Total GSTR-2A records: {len(self.gstr2a_df)}")
            print(f"Total matched records: {len(self.matched_df)}")
            print(f"Total mismatch records: {len(self.mismatch_df)}")
            print(f"Total processed records: {len(self.matched_df) + len(self.mismatch_df)}")
            
            # Verify all records are accounted for
            total_processed = len(self.matched_df) + len(self.mismatch_df)
            total_original = len(self.original_df)
            if total_processed != total_original:
                print(f"\nWARNING: Record count mismatch!")
                print(f"Original records: {total_original}")
                print(f"Processed records: {total_processed}")
                print(f"Difference: {total_original - total_processed}")
                
                # Find missing records
                print("\nChecking for missing records...")
                original_books = set(self.original_df[self.original_df['Source Name'] == 'Books'].index)
                processed_books = set(self.books_df.index)
                missing_books = original_books - processed_books
                if missing_books:
                    print(f"Missing Books records (indices): {missing_books}")
                    print("\nMissing Books records details:")
                    print(self.original_df.loc[missing_books])
            
        except Exception as e:
            raise Exception(f"Error during reconciliation process: {str(e)}")
    
    def _process_exact_matches(self):
        """
        Process exact and partial matches by grouping records by GSTIN and Invoice Number.
        This approach considers all possible pairings within a group to find the optimal matches,
        prioritizing same-financial-year matches with the lowest tax and date differences.
        """
        # Get all unprocessed records to form groups
        unprocessed_books = self.books_df[~self.books_df['processed']]
        unprocessed_gstr2a = self.gstr2a_df[~self.gstr2a_df['processed']]

        # Find common (GSTIN, Invoice Number) keys to form groups
        books_keys = set(unprocessed_books.set_index(['Supplier GSTIN', 'Invoice Number']).index)
        gstr2a_keys = set(unprocessed_gstr2a.set_index(['Supplier GSTIN', 'Invoice Number']).index)
        common_keys = books_keys.intersection(gstr2a_keys)

        total_keys = len(common_keys)
        processed_keys = 0

        # Iterate through each group of (GSTIN, Invoice Number)
        for key in common_keys:
            gstin, inv_num = key
            
            books_group = unprocessed_books[(unprocessed_books['Supplier GSTIN'] == gstin) & (unprocessed_books['Invoice Number'] == inv_num)]
            gstr2a_group = unprocessed_gstr2a[(unprocessed_gstr2a['Supplier GSTIN'] == gstin) & (unprocessed_gstr2a['Invoice Number'] == inv_num)]
            
            # Create a list of all possible pairings and their scores within the group
            pairings = []
            for book_idx, book_row in books_group.iterrows():
                for gstr2a_idx, gstr2a_row in gstr2a_group.iterrows():
                    # --- Scoring Logic ---
                    book_sign = 1 if (book_row['Total IGST Amount'] + book_row['Total CGST Amount'] + book_row['Total SGST Amount']) >= 0 else -1
                    gstr2a_sign = 1 if (gstr2a_row['Total IGST Amount'] + gstr2a_row['Total CGST Amount'] + gstr2a_row['Total SGST Amount']) >= 0 else -1
                    if book_sign != gstr2a_sign:
                        continue

                    book_fy = _get_financial_year(book_row['Invoice Date'])
                    gstr2a_fy = _get_financial_year(gstr2a_row['Invoice Date'])
                    
                    igst_diff_abs = abs(book_row['Total IGST Amount'] - gstr2a_row['Total IGST Amount'])
                    cgst_diff_abs = abs(book_row['Total CGST Amount'] - gstr2a_row['Total CGST Amount'])
                    sgst_diff_abs = abs(book_row['Total SGST Amount'] - gstr2a_row['Total SGST Amount'])
                    total_tax_diff = igst_diff_abs + cgst_diff_abs + sgst_diff_abs
                    
                    date_diff_days = abs((book_row['Invoice Date'] - gstr2a_row['Invoice Date']).days)

                    score = 0
                    if gstr2a_fy == book_fy:
                        score = total_tax_diff * 100 + date_diff_days
                    else:
                        score = total_tax_diff + 10000  # Penalty for cross-year

                    pairings.append({
                        'book_idx': book_idx, 'gstr2a_idx': gstr2a_idx, 'score': score,
                        'book_row': book_row, 'gstr2a_row': gstr2a_row,
                        'date_diff': (book_row['Invoice Date'] - gstr2a_row['Invoice Date']).days,
                        'igst_diff': book_row['Total IGST Amount'] - gstr2a_row['Total IGST Amount'],
                        'cgst_diff': book_row['Total CGST Amount'] - gstr2a_row['Total CGST Amount'],
                        'sgst_diff': book_row['Total SGST Amount'] - gstr2a_row['Total SGST Amount'],
                        'total_tax_diff': total_tax_diff,
                        'cross_year': gstr2a_fy != book_fy
                    })
            
            # Sort all possible pairings by the best score (lowest first)
            pairings.sort(key=lambda x: x['score'])
            
            # Greedily create matches using the best-scored pairs, ensuring no record is used twice
            used_book_indices = set()
            used_gstr2a_indices = set()

            for pairing in pairings:
                book_idx, gstr2a_idx = pairing['book_idx'], pairing['gstr2a_idx']
                
                if book_idx in used_book_indices or gstr2a_idx in used_gstr2a_indices:
                    continue

                # --- This is a valid match, process it ---
                used_book_indices.add(book_idx)
                used_gstr2a_indices.add(gstr2a_idx)
                
                self.books_df.at[book_idx, 'processed'] = True
                self.gstr2a_df.at[gstr2a_idx, 'processed'] = True

                # --- Create matched records (logic is similar to before) ---
                book_row, gstr2a_row = pairing['book_row'], pairing['gstr2a_row']
                igst_diff, cgst_diff, sgst_diff = pairing['igst_diff'], pairing['cgst_diff'], pairing['sgst_diff']
                cross_year, date_diff = pairing['cross_year'], pairing['date_diff']
                
                # Enhanced tax matching logic - check each tax type independently
                igst_match = abs(igst_diff) <= self.tax_tolerance
                cgst_match = abs(cgst_diff) <= self.tax_tolerance
                sgst_match = abs(sgst_diff) <= self.tax_tolerance
                
                # Check for tax head mismatches
                tax_head_mismatch = self._check_tax_head_mismatch(book_row, gstr2a_row)
                
                is_exact_match = igst_match and cgst_match and sgst_match and not tax_head_mismatch
                if not cross_year and pd.notna(date_diff):
                    is_exact_match = is_exact_match and abs(date_diff) <= self.date_tolerance.days
                
                match_id = self.match_id_counter
                self.match_id_counter += 1
                
                # Determine status with enhanced tax head checking
                status = 'Exact Match'
                sub_status_comment = self._generate_sub_status_comment(
                    book_row, gstr2a_row, igst_match, cgst_match, sgst_match, 
                    tax_head_mismatch, cross_year, is_exact_match
                )
                
                if not is_exact_match:
                    if tax_head_mismatch:
                        status = 'Partial Match'
                    else:
                        status = 'Partial Match'
                if cross_year:
                    status = 'Cross-Year Match' if is_exact_match else 'Partial Cross-Year Match'

                # Generate field status columns
                field_status = self._generate_field_status_columns(book_row, gstr2a_row)
                
                common_data = {
                    'Match ID': match_id, 'Status': status, 'Sub Status': sub_status_comment, 'Cross-Year': cross_year,
                    'Tax Diff Status': 'No Difference' if pairing['total_tax_diff'] <= self.tax_tolerance else 'Has Difference',
                    'Date Status': 'Within Tolerance' if (pd.notna(date_diff) and abs(date_diff) <= self.date_tolerance.days) else 'Outside Tolerance' if pd.notna(date_diff) else 'N/A',
                    'Tax Sign Status': 'Sign Match',
                    'Narrative': self._generate_narrative(book_row, gstr2a_row, is_exact_match),
                    'Suggestions': self._generate_suggestions(book_row, gstr2a_row, is_exact_match),
                }
                
                # Add field status columns
                common_data.update(field_status)

                book_match_data = book_row.to_dict()
                book_match_data.update(common_data)
                book_match_data.update({
                    'Source Name': 'Books', 'IGST Diff': igst_diff, 'CGST Diff': cgst_diff, 'SGST Diff': sgst_diff,
                    'Date Diff': date_diff,
                    'Value Sign': 'Positive' if (book_row['Total IGST Amount'] + book_row['Total CGST Amount'] + book_row['Total SGST Amount']) >= 0 else 'Negative',
                })

                gstr2a_match_data = gstr2a_row.to_dict()
                gstr2a_match_data.update(common_data)
                gstr2a_match_data.update({
                    'Source Name': 'GSTR-2A', 'IGST Diff': -igst_diff, 'CGST Diff': -cgst_diff, 'SGST Diff': -sgst_diff,
                    'Date Diff': -date_diff if pd.notna(date_diff) else np.nan,
                    'Value Sign': 'Positive' if (gstr2a_row['Total IGST Amount'] + gstr2a_row['Total CGST Amount'] + gstr2a_row['Total SGST Amount']) >= 0 else 'Negative',
                })
                
                # Add the matched records
                self.matched_df = pd.concat([self.matched_df, pd.DataFrame([book_match_data]), pd.DataFrame([gstr2a_match_data])], ignore_index=True)
                
                if is_exact_match: self.matched_count += 1
                else: self.partial_count += 1

            processed_keys += 1
            if total_keys > 0:
                self.step_progress = processed_keys / total_keys
    
    def _process_tax_based_matches(self):
        """Process matches based on tax amounts and dates with sign validation."""
        total_records = len(self.books_df[self.books_df['processed'] == False])
        processed = 0
        
        for idx, book_row in self.books_df[self.books_df['processed'] == False].iterrows():
            # Calculate total tax value to determine sign
            book_total_tax = book_row['Total IGST Amount'] + book_row['Total CGST Amount'] + book_row['Total SGST Amount']
            book_sign = 1 if book_total_tax >= 0 else -1
            
            # Prioritized matching: invoice-number-first, exact-before-tolerance
            def _collect_scored_matches(candidates_df):
                results = []
                for _, gstr2a_row in candidates_df.iterrows():
                    gstr2a_total_tax = gstr2a_row['Total IGST Amount'] + gstr2a_row['Total CGST Amount'] + gstr2a_row['Total SGST Amount']
                    gstr2a_sign = 1 if gstr2a_total_tax >= 0 else -1
                    if book_sign != gstr2a_sign:
                        continue
                    gstin_similarity = 0
                    if pd.notna(book_row['Supplier GSTIN']) and pd.notna(gstr2a_row['Supplier GSTIN']):
                        gstin_similarity = fuzz.ratio(
                            str(book_row['Supplier GSTIN']).lower(),
                            str(gstr2a_row['Supplier GSTIN']).lower()
                        )
                    legal_name_similarity = 0
                    if pd.notna(book_row['Supplier Legal Name']) and pd.notna(gstr2a_row['Supplier Legal Name']):
                        clean_book_legal = _clean_company_name(str(book_row['Supplier Legal Name']))
                        clean_gstr2a_legal = _clean_company_name(str(gstr2a_row['Supplier Legal Name']))
                        legal_name_similarity = fuzz.ratio(clean_book_legal, clean_gstr2a_legal)
                    trade_name_similarity = 0
                    if pd.notna(book_row['Supplier Trade Name']) and pd.notna(gstr2a_row['Supplier Trade Name']):
                        clean_book_trade = _clean_company_name(str(book_row['Supplier Trade Name']))
                        clean_gstr2a_trade = _clean_company_name(str(gstr2a_row['Supplier Trade Name']))
                        trade_name_similarity = fuzz.ratio(clean_book_trade, clean_gstr2a_trade)
                    if gstin_similarity < 80 and (legal_name_similarity < 70 and trade_name_similarity < 70):
                        continue
                    date_diff = np.nan
                    if pd.notna(book_row['Invoice Date']) and pd.notna(gstr2a_row['Invoice Date']):
                        date_diff = abs((book_row['Invoice Date'] - gstr2a_row['Invoice Date']).days)
                    score = 0
                    if pd.notna(date_diff) and date_diff <= self.date_tolerance.days:
                        score += 20
                    score += (gstin_similarity * 0.5)
                    score += (legal_name_similarity * 0.25)
                    score += (trade_name_similarity * 0.25)
                    results.append((gstr2a_row, score))
                return results

            def _pick_best(df):
                if df is None or len(df) == 0:
                    return None, None
                matches = _collect_scored_matches(df)
                if not matches:
                    return None, None
                matches.sort(key=lambda x: -x[1])
                return matches[0]

            def _amounts_equal(df):
                return (
                    (df['Total IGST Amount'] == book_row['Total IGST Amount']) &
                    (df['Total CGST Amount'] == book_row['Total CGST Amount']) &
                    (df['Total SGST Amount'] == book_row['Total SGST Amount'])
                )

            def _amounts_within_tol(df):
                return (
                    (abs(df['Total IGST Amount'] - book_row['Total IGST Amount']) <= self.tax_tolerance) &
                    (abs(df['Total CGST Amount'] - book_row['Total CGST Amount']) <= self.tax_tolerance) &
                    (abs(df['Total SGST Amount'] - book_row['Total SGST Amount']) <= self.tax_tolerance)
                )

            best_match = None
            best_score = None

            same_invoice_df = self.gstr2a_df[(self.gstr2a_df['processed'] == False) & (self.gstr2a_df['Invoice Number'] == book_row['Invoice Number'])]
            if not same_invoice_df.empty:
                exact_date_mask = pd.Series([False] * len(same_invoice_df), index=same_invoice_df.index)
                if pd.notna(book_row['Invoice Date']):
                    exact_date_mask = same_invoice_df['Invoice Date'] == book_row['Invoice Date']
                tier = same_invoice_df[_amounts_equal(same_invoice_df) & exact_date_mask]
                cand, score = _pick_best(tier)
                if cand is None:
                    tol_date_mask = pd.Series([True] * len(same_invoice_df), index=same_invoice_df.index)
                    if pd.notna(book_row['Invoice Date']):
                        tol_date_mask = abs((same_invoice_df['Invoice Date'] - book_row['Invoice Date']).dt.days) <= self.date_tolerance.days
                    tier = same_invoice_df[_amounts_equal(same_invoice_df) & tol_date_mask]
                    cand, score = _pick_best(tier)
                if cand is None:
                    exact_date_mask = pd.Series([False] * len(same_invoice_df), index=same_invoice_df.index)
                    if pd.notna(book_row['Invoice Date']):
                        exact_date_mask = same_invoice_df['Invoice Date'] == book_row['Invoice Date']
                    tier = same_invoice_df[_amounts_within_tol(same_invoice_df) & exact_date_mask]
                    cand, score = _pick_best(tier)
                if cand is None:
                    tol_date_mask = pd.Series([True] * len(same_invoice_df), index=same_invoice_df.index)
                    if pd.notna(book_row['Invoice Date']):
                        tol_date_mask = abs((same_invoice_df['Invoice Date'] - book_row['Invoice Date']).dt.days) <= self.date_tolerance.days
                    tier = same_invoice_df[_amounts_within_tol(same_invoice_df) & tol_date_mask]
                    cand, score = _pick_best(tier)
                if cand is not None:
                    best_match, best_score = cand, score

            if best_match is None and pd.notna(book_row['Invoice Date']):
                same_date_df = self.gstr2a_df[(self.gstr2a_df['processed'] == False) & (self.gstr2a_df['Invoice Date'] == book_row['Invoice Date'])]
                if not same_date_df.empty:
                    tier = same_date_df[_amounts_equal(same_date_df)]
                    cand, score = _pick_best(tier)
                    if cand is None:
                        tier = same_date_df[_amounts_within_tol(same_date_df)]
                        cand, score = _pick_best(tier)
                    if cand is not None:
                        best_match, best_score = cand, score

            if best_match is None:
                fallback_df = self.gstr2a_df[(self.gstr2a_df['processed'] == False) & _amounts_within_tol(self.gstr2a_df)]
                if len(fallback_df) > 0:
                    cand, score = _pick_best(fallback_df)
                    if cand is not None:
                        best_match, best_score = cand, score

            if best_match is not None:
                if best_score is not None and best_score < 60:
                    processed += 1
                    if total_records > 0:
                        self.step_progress = processed / total_records
                    continue
                self.books_df.at[idx, 'processed'] = True
                self.gstr2a_df.at[best_match.name, 'processed'] = True
                igst_diff = book_row['Total IGST Amount'] - best_match['Total IGST Amount']
                cgst_diff = book_row['Total CGST Amount'] - best_match['Total CGST Amount']
                sgst_diff = book_row['Total SGST Amount'] - best_match['Total SGST Amount']
                date_diff = np.nan
                if pd.notna(book_row['Invoice Date']) and pd.notna(best_match['Invoice Date']):
                    date_diff = (book_row['Invoice Date'] - best_match['Invoice Date']).days
                gstin_similarity = 0
                if pd.notna(book_row['Supplier GSTIN']) and pd.notna(best_match['Supplier GSTIN']):
                    gstin_similarity = fuzz.ratio(
                        str(book_row['Supplier GSTIN']).lower(),
                        str(best_match['Supplier GSTIN']).lower()
                    )
                legal_name_similarity = 0
                if pd.notna(book_row['Supplier Legal Name']) and pd.notna(best_match['Supplier Legal Name']):
                    legal_name_similarity = fuzz.ratio(
                        str(book_row['Supplier Legal Name']).lower(),
                        str(best_match['Supplier Legal Name']).lower()
                    )
                trade_name_similarity = 0
                if pd.notna(book_row['Supplier Trade Name']) and pd.notna(best_match['Supplier Trade Name']):
                    clean_book_name = _clean_trade_name(str(book_row['Supplier Trade Name']))
                    clean_gstr2a_name = _clean_trade_name(str(best_match['Supplier Trade Name']))
                    trade_name_similarity = fuzz.ratio(clean_book_name, clean_gstr2a_name)
                match_id = self.match_id_counter
                self.match_id_counter += 1
                book_match_data = book_row.to_dict()
                book_match_data.update({
                    'Match ID': match_id,
                    'Source Name': 'Books',
                    'IGST Diff': igst_diff,
                    'CGST Diff': cgst_diff,
                    'SGST Diff': sgst_diff,
                    'Date Diff': date_diff,
                    'Status': 'Partial Match',
                    'Tax Diff Status': 'No Difference' if abs(igst_diff + cgst_diff + sgst_diff) <= self.tax_tolerance else 'Has Difference',
                    'Date Status': 'Within Tolerance' if (pd.notna(date_diff) and abs(date_diff) <= self.date_tolerance.days) else 'Outside Tolerance' if pd.notna(date_diff) else 'N/A',
                    'GSTIN Score': gstin_similarity,
                    'Legal Name Score': legal_name_similarity,
                    'Trade Name Score': trade_name_similarity,
                    'Tax Sign Status': 'Sign Match',
                    'Value Sign': 'Positive' if book_sign > 0 else 'Negative',
                    'Narrative': self._generate_narrative(book_row, best_match, False),
                    'Suggestions': self._generate_suggestions(book_row, best_match, False)
                })
                gstr2a_match_data = best_match.to_dict()
                gstr2a_match_data.update({
                    'Match ID': match_id,
                    'Source Name': 'GSTR-2A',
                    'IGST Diff': -igst_diff,
                    'CGST Diff': -cgst_diff,
                    'SGST Diff': -sgst_diff,
                    'Date Diff': -date_diff if pd.notna(date_diff) else np.nan,
                    'Status': 'Partial Match',
                    'Tax Diff Status': 'No Difference' if abs(igst_diff + cgst_diff + sgst_diff) <= self.tax_tolerance else 'Has Difference',
                    'Date Status': 'Within Tolerance' if (pd.notna(date_diff) and abs(date_diff) <= self.date_tolerance.days) else 'Outside Tolerance' if pd.notna(date_diff) else 'N/A',
                    'GSTIN Score': gstin_similarity,
                    'Legal Name Score': legal_name_similarity,
                    'Trade Name Score': trade_name_similarity,
                    'Tax Sign Status': 'Sign Match',
                    'Value Sign': 'Positive' if book_sign > 0 else 'Negative',
                    'Narrative': self._generate_narrative(book_row, best_match, False),
                    'Suggestions': self._generate_suggestions(book_row, best_match, False)
                })
                self.matched_df = pd.concat([
                    self.matched_df,
                    pd.DataFrame([book_match_data]),
                    pd.DataFrame([gstr2a_match_data])
                ], ignore_index=True)
                self.partial_count += 1
                self.partial_value += book_row['Total Invoice Value']
            
            processed += 1
            self.step_progress = processed / total_records
    
    def _process_group_matches(self):
        """Process group matches based on invoice number and GSTIN."""
        # Get unprocessed records
        unprocessed_books = self.books_df[self.books_df['processed'] == False].copy()
        unprocessed_gstr2a = self.gstr2a_df[self.gstr2a_df['processed'] == False].copy()
        
        # Add fiscal year column for filtering
        unprocessed_books['Fiscal Year'] = unprocessed_books['Invoice Date'].apply(
            lambda x: f"{x.year}-{x.year+1}" if x.month < 4 else f"{x.year+1}-{x.year+2}"
        )
        unprocessed_gstr2a['Fiscal Year'] = unprocessed_gstr2a['Invoice Date'].apply(
            lambda x: f"{x.year}-{x.year+1}" if x.month < 4 else f"{x.year+1}-{x.year+2}"
        )
        
        # Group by GSTIN and Invoice Number
        books_groups = unprocessed_books.groupby(['Supplier GSTIN', 'Invoice Number', 'Fiscal Year'])
        gstr2a_groups = unprocessed_gstr2a.groupby(['Supplier GSTIN', 'Invoice Number', 'Fiscal Year'])
        
        # Process each group
        for (gstin, invoice, fiscal_year), books_group in books_groups:
            # Skip if no matching GSTR-2A group
            if (gstin, invoice, fiscal_year) not in gstr2a_groups.groups:
                continue
                
            gstr2a_group = gstr2a_groups.get_group((gstin, invoice, fiscal_year))
            
            # Calculate group totals
            books_total_igst = books_group['Total IGST Amount'].sum()
            books_total_cgst = books_group['Total CGST Amount'].sum()
            books_total_sgst = books_group['Total SGST Amount'].sum()
            books_total_tax = books_total_igst + books_total_cgst + books_total_sgst
            
            gstr2a_total_igst = gstr2a_group['Total IGST Amount'].sum()
            gstr2a_total_cgst = gstr2a_group['Total CGST Amount'].sum()
            gstr2a_total_sgst = gstr2a_group['Total SGST Amount'].sum()
            gstr2a_total_tax = gstr2a_total_igst + gstr2a_total_cgst + gstr2a_total_sgst
            
            # Check if totals match within tolerance
            total_tax_diff = abs(books_total_tax - gstr2a_total_tax)
            if total_tax_diff <= self.group_tax_tolerance:
                # Check for duplicate entries in books
                books_tax_values = books_group[['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']].values
                unique_tax_values = set(map(tuple, books_tax_values))
                
                if len(unique_tax_values) < len(books_group):
                    print(f"Warning: Duplicate tax values found for GSTIN {gstin}, Invoice {invoice}")
                    continue
                
                # Generate group ID
                group_id = f"G{self.group_id_counter}"
                self.group_id_counter += 1
                
                # Mark all records as processed
                for idx in books_group.index:
                    self.books_df.at[idx, 'processed'] = True
                for idx in gstr2a_group.index:
                    self.gstr2a_df.at[idx, 'processed'] = True
                
                # Add to matched records
                for _, book_row in books_group.iterrows():
                    book_match_data = book_row.to_dict()
                    book_match_data.update({
                        'Match ID': self.match_id_counter,
                        'Group ID': group_id,
                        'Source Name': 'Books',
                        'Status': 'Group Match',
                        'Reconciliation Level': 'Group',
                        'Narrative': f"Group match with total tax difference of ₹{total_tax_diff:.2f}",
                        'Suggestions': "Verify group totals match"
                    })
                    self.matched_df = pd.concat([
                        self.matched_df,
                        pd.DataFrame([book_match_data])
                    ], ignore_index=True)
                
                for _, gstr2a_row in gstr2a_group.iterrows():
                    gstr2a_match_data = gstr2a_row.to_dict()
                    gstr2a_match_data.update({
                        'Match ID': self.match_id_counter,
                        'Group ID': group_id,
                        'Source Name': 'GSTR-2A',
                        'Status': 'Group Match',
                        'Reconciliation Level': 'Group',
                        'Narrative': f"Group match with total tax difference of ₹{total_tax_diff:.2f}",
                        'Suggestions': "Verify group totals match"
                    })
                    self.matched_df = pd.concat([
                        self.matched_df,
                        pd.DataFrame([gstr2a_match_data])
                    ], ignore_index=True)
                
                self.match_id_counter += 1
                self.group_count += 1
                self.group_value += books_total_tax
                
                # Update progress
                self.step_progress = len(self.matched_df) / len(unprocessed_books)
    
    def _process_sign_cancellations(self):
        """Process sign cancellations within the same source."""
        # Process Books sign cancellations
        unprocessed_books = self.books_df[self.books_df['processed'] == False].copy()
        books_groups = unprocessed_books.groupby('Supplier GSTIN')
        
        for gstin, books_group in books_groups:
            # Get all tax values for this GSTIN
            tax_values = books_group[['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']].values
            total_tax_values = np.sum(tax_values, axis=1)
            
            # Find pairs of records with opposite signs
            for i in range(len(books_group)):
                for j in range(i + 1, len(books_group)):
                    if total_tax_values[i] == -total_tax_values[j]:
                        # Check if tax components match in absolute value
                        if (np.abs(tax_values[i]) == np.abs(tax_values[j])).all():
                            # Mark as processed
                            idx1 = books_group.index[i]
                            idx2 = books_group.index[j]
                            
                            # Skip if either record is already processed
                            if self.books_df.at[idx1, 'processed'] or self.books_df.at[idx2, 'processed']:
                                continue
                                
                            self.books_df.at[idx1, 'processed'] = True
                            self.books_df.at[idx2, 'processed'] = True
                            
                            # Add to matched records with a single match ID for the pair
                            match_id = self.match_id_counter
                            self.match_id_counter += 1
                            
                            # Add first record
                            book_row1 = self.books_df.loc[idx1]
                            book_match_data1 = book_row1.to_dict()
                            book_match_data1.update({
                                'Match ID': match_id,
                                'Source Name': 'Books',
                                'Status': 'Books Sign Cancellation',
                                'Sub Status': f"Sign cancellation within Books for GSTIN {gstin}",
                                'Tax Diff Status': 'Nullified',
                                'Narrative': f"Sign cancellation within Books for GSTIN {gstin}",
                                'Suggestions': "Verify sign cancellation is intentional",
                                'Tax Head Status': 'N/A',
                                'GSTIN Match Status': 'N/A',
                                'Invoice Match Status': 'N/A',
                                'Trade Name Match Status': 'N/A',
                                'Legal Name Match Status': 'N/A'
                            })
                            
                            # Add second record
                            book_row2 = self.books_df.loc[idx2]
                            book_match_data2 = book_row2.to_dict()
                            book_match_data2.update({
                                'Match ID': match_id,
                                'Source Name': 'Books',
                                'Status': 'Books Sign Cancellation',
                                'Sub Status': f"Sign cancellation within Books for GSTIN {gstin}",
                                'Tax Diff Status': 'Nullified',
                                'Narrative': f"Sign cancellation within Books for GSTIN {gstin}",
                                'Suggestions': "Verify sign cancellation is intentional",
                                'Tax Head Status': 'N/A',
                                'GSTIN Match Status': 'N/A',
                                'Invoice Match Status': 'N/A',
                                'Trade Name Match Status': 'N/A',
                                'Legal Name Match Status': 'N/A'
                            })
                            
                            # Add both records together
                            self.matched_df = pd.concat([
                                self.matched_df,
                                pd.DataFrame([book_match_data1, book_match_data2])
                            ], ignore_index=True)
                            
                            self.sign_cancellation_count += 1
                            self.sign_cancellation_value += abs(total_tax_values[i])
        
        # Process GSTR-2A sign cancellations
        unprocessed_gstr2a = self.gstr2a_df[self.gstr2a_df['processed'] == False].copy()
        gstr2a_groups = unprocessed_gstr2a.groupby('Supplier GSTIN')
        
        for gstin, gstr2a_group in gstr2a_groups:
            # Get all tax values for this GSTIN
            tax_values = gstr2a_group[['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']].values
            total_tax_values = np.sum(tax_values, axis=1)
            
            # Find pairs of records with opposite signs
            for i in range(len(gstr2a_group)):
                for j in range(i + 1, len(gstr2a_group)):
                    if total_tax_values[i] == -total_tax_values[j]:
                        # Check if tax components match in absolute value
                        if (np.abs(tax_values[i]) == np.abs(tax_values[j])).all():
                            # Mark as processed
                            idx1 = gstr2a_group.index[i]
                            idx2 = gstr2a_group.index[j]
                            
                            # Skip if either record is already processed
                            if self.gstr2a_df.at[idx1, 'processed'] or self.gstr2a_df.at[idx2, 'processed']:
                                continue
                                
                            self.gstr2a_df.at[idx1, 'processed'] = True
                            self.gstr2a_df.at[idx2, 'processed'] = True
                            
                            # Add to matched records with a single match ID for the pair
                            match_id = self.match_id_counter
                            self.match_id_counter += 1
                            
                            # Add first record
                            gstr2a_row1 = self.gstr2a_df.loc[idx1]
                            gstr2a_match_data1 = gstr2a_row1.to_dict()
                            gstr2a_match_data1.update({
                                'Match ID': match_id,
                                'Source Name': 'GSTR-2A',
                                'Status': 'GSTR-2A Sign Cancellation',
                                'Sub Status': f"Sign cancellation within GSTR-2A for GSTIN {gstin}",
                                'Tax Diff Status': 'Nullified',
                                'Narrative': f"Sign cancellation within GSTR-2A for GSTIN {gstin}",
                                'Suggestions': "Verify sign cancellation is intentional",
                                'Tax Head Status': 'N/A',
                                'GSTIN Match Status': 'N/A',
                                'Invoice Match Status': 'N/A',
                                'Trade Name Match Status': 'N/A',
                                'Legal Name Match Status': 'N/A'
                            })
                            
                            # Add second record
                            gstr2a_row2 = self.gstr2a_df.loc[idx2]
                            gstr2a_match_data2 = gstr2a_row2.to_dict()
                            gstr2a_match_data2.update({
                                'Match ID': match_id,
                                'Source Name': 'GSTR-2A',
                                'Status': 'GSTR-2A Sign Cancellation',
                                'Sub Status': f"Sign cancellation within GSTR-2A for GSTIN {gstin}",
                                'Tax Diff Status': 'Nullified',
                                'Narrative': f"Sign cancellation within GSTR-2A for GSTIN {gstin}",
                                'Suggestions': "Verify sign cancellation is intentional",
                                'Tax Head Status': 'N/A',
                                'GSTIN Match Status': 'N/A',
                                'Invoice Match Status': 'N/A',
                                'Trade Name Match Status': 'N/A',
                                'Legal Name Match Status': 'N/A'
                            })
                            
                            # Add both records together
                            self.matched_df = pd.concat([
                                self.matched_df,
                                pd.DataFrame([gstr2a_match_data1, gstr2a_match_data2])
                            ], ignore_index=True)
                            
                            self.sign_cancellation_count += 1
                            self.sign_cancellation_value += abs(total_tax_values[i])
        
        # Update progress
        total_unprocessed = len(unprocessed_books) + len(unprocessed_gstr2a)
        if total_unprocessed > 0:
            self.step_progress = (len(unprocessed_books) - len(self.books_df[self.books_df['processed'] == False]) +
                                len(unprocessed_gstr2a) - len(self.gstr2a_df[self.gstr2a_df['processed'] == False])) / total_unprocessed
    
    def _add_remaining_mismatches(self):
        """Add remaining unmatched records to mismatch DataFrame."""
        total_records = len(self.books_df) + len(self.gstr2a_df)
        processed = 0
        
        # Debug: Print counts before adding mismatches
        print(f"\nBefore adding mismatches:")
        print(f"Unprocessed Books: {len(self.books_df[self.books_df['processed'] == False])}")
        print(f"Unprocessed GSTR-2A: {len(self.gstr2a_df[self.gstr2a_df['processed'] == False])}")
        
        # Add remaining books records
        for idx, row in self.books_df[self.books_df['processed'] == False].iterrows():
            # Calculate total tax value to determine sign
            total_tax = row['Total IGST Amount'] + row['Total CGST Amount'] + row['Total SGST Amount']
            tax_sign = 1 if total_tax >= 0 else -1
            
            mismatch_data = row.to_dict()
            mismatch_data.update({
                'Match ID': self.match_id_counter,
                'Source Name': 'Books',
                'IGST Diff': 0,
                'CGST Diff': 0,
                'SGST Diff': 0,
                'Date Diff': np.nan,
                'Status': 'Books Only',
                'Sub Status': 'Record not found in GSTR-2A data',
                'Tax Diff Status': 'N/A',
                'Date Status': 'N/A',
                'Tax Sign Status': 'N/A',
                'Value Sign': 'Positive' if tax_sign > 0 else 'Negative',
                'Narrative': 'Record found only in Books',
                'Suggestions': self._generate_suggestions(row, row, False),
                'Tax Head Status': 'N/A',
                'GSTIN Match Status': 'N/A',
                'Invoice Match Status': 'N/A',
                'Trade Name Match Status': 'N/A',
                'Legal Name Match Status': 'N/A'
            })
            self.mismatch_df = pd.concat([self.mismatch_df, pd.DataFrame([mismatch_data])], ignore_index=True)
            self.match_id_counter += 1
            processed += 1
            self.step_progress = processed / total_records
        
        # Add remaining GSTR-2A records
        for idx, row in self.gstr2a_df[self.gstr2a_df['processed'] == False].iterrows():
            # Calculate total tax value to determine sign
            total_tax = row['Total IGST Amount'] + row['Total CGST Amount'] + row['Total SGST Amount']
            tax_sign = 1 if total_tax >= 0 else -1
            
            mismatch_data = row.to_dict()
            mismatch_data.update({
                'Match ID': self.match_id_counter,
                'Source Name': 'GSTR-2A',
                'IGST Diff': 0,
                'CGST Diff': 0,
                'SGST Diff': 0,
                'Date Diff': np.nan,
                'Status': 'GSTR-2A Only',
                'Sub Status': 'Record not found in Books data',
                'Tax Diff Status': 'N/A',
                'Date Status': 'N/A',
                'Tax Sign Status': 'N/A',
                'Value Sign': 'Positive' if tax_sign > 0 else 'Negative',
                'Narrative': 'Record found only in GSTR-2A',
                'Suggestions': self._generate_suggestions(row, row, False),
                'Tax Head Status': 'N/A',
                'GSTIN Match Status': 'N/A',
                'Invoice Match Status': 'N/A',
                'Trade Name Match Status': 'N/A',
                'Legal Name Match Status': 'N/A'
            })
            self.mismatch_df = pd.concat([self.mismatch_df, pd.DataFrame([mismatch_data])], ignore_index=True)
            self.match_id_counter += 1
            processed += 1
            self.step_progress = processed / total_records
        
        # Debug: Print final counts
        print(f"\nAfter adding mismatches:")
        print(f"Total Books records: {len(self.books_df)}")
        print(f"Total GSTR-2A records: {len(self.gstr2a_df)}")
        print(f"Total matched records: {len(self.matched_df)}")
        print(f"Total mismatch records: {len(self.mismatch_df)}")
        
        # Verify all records are accounted for
        total_processed = len(self.matched_df) + len(self.mismatch_df)
        total_original = len(self.original_df)
        if total_processed != total_original:
            print(f"\nWARNING: Record count mismatch!")
            print(f"Original records: {total_original}")
            print(f"Processed records: {total_processed}")
            print(f"Difference: {total_original - total_processed}")
            
            # Find missing records
            print("\nChecking for missing records...")
            original_books = set(self.original_df[self.original_df['Source Name'] == 'Books'].index)
            processed_books = set(self.books_df.index)
            missing_books = original_books - processed_books
            if missing_books:
                print(f"Missing Books records (indices): {missing_books}")
                print("\nMissing Books records details:")
                print(self.original_df.loc[missing_books])
    
    def _enhanced_matching(self):
        """Enhanced matching using multiple strategies with sign validation."""
        # Get unprocessed records
        unprocessed_books = self.books_df[self.books_df['processed'] == False]
        unprocessed_gstr2a = self.gstr2a_df[self.gstr2a_df['processed'] == False]
        
        # Find potential data entry errors where same GSTIN has swapped invoice numbers and taxes
        potential_swaps = self._find_potential_data_entry_swaps(unprocessed_books, unprocessed_gstr2a)
        
        # Process potential swaps
        for swap_info in potential_swaps:
            self._process_data_entry_swap(swap_info)
        
        # Find tax-based matches for same GSTIN with zero net difference
        tax_based_groups = self._find_tax_based_groups(unprocessed_books, unprocessed_gstr2a)
        
        # Process tax-based groups
        for group_info in tax_based_groups:
            self._process_tax_based_group(group_info)
        
        # Additional enhanced matching with strict rules
        self._enhanced_tax_based_matching_with_strict_rules(unprocessed_books, unprocessed_gstr2a)

    def _enhanced_tax_based_matching_with_strict_rules(self, books_df, gstr2a_df):
        """Enhanced tax-based matching with strict GSTIN and name similarity rules."""
        for idx, book_row in books_df[books_df['processed'] == False].iterrows():
            # Calculate total tax value to determine sign
            book_total_tax = book_row['Total IGST Amount'] + book_row['Total CGST Amount'] + book_row['Total SGST Amount']
            book_sign = 1 if book_total_tax >= 0 else -1
            
            # Find potential matches in GSTR-2A with relaxed tax tolerance
            gstr2a_matches = gstr2a_df[
                (gstr2a_df['processed'] == False) &
                (abs(gstr2a_df['Total Invoice Value'] - book_row['Total Invoice Value']) <= self.tax_tolerance * 3)
            ]
            
            if len(gstr2a_matches) > 0:
                # Filter matches by sign and find best match
                valid_matches = []
                for _, gstr2a_row in gstr2a_matches.iterrows():
                    gstr2a_total_tax = gstr2a_row['Total IGST Amount'] + gstr2a_row['Total CGST Amount'] + gstr2a_row['Total SGST Amount']
                    gstr2a_sign = 1 if gstr2a_total_tax >= 0 else -1
                    
                    if book_sign == gstr2a_sign:
                        # Calculate GSTIN similarity
                        gstin_similarity = 0
                        if pd.notna(book_row['Supplier GSTIN']) and pd.notna(gstr2a_row['Supplier GSTIN']):
                            gstin_similarity = fuzz.ratio(
                                str(book_row['Supplier GSTIN']).lower(),
                                str(gstr2a_row['Supplier GSTIN']).lower()
                            )
                        
                        # Clean and compare legal names
                        legal_name_similarity = 0
                        if pd.notna(book_row['Supplier Legal Name']) and pd.notna(gstr2a_row['Supplier Legal Name']):
                            clean_book_legal = _clean_company_name(str(book_row['Supplier Legal Name']))
                            clean_gstr2a_legal = _clean_company_name(str(gstr2a_row['Supplier Legal Name']))
                            legal_name_similarity = fuzz.ratio(clean_book_legal, clean_gstr2a_legal)
                        
                        # Clean and compare trade names
                        trade_name_similarity = 0
                        if pd.notna(book_row['Supplier Trade Name']) and pd.notna(gstr2a_row['Supplier Trade Name']):
                            clean_book_trade = _clean_company_name(str(book_row['Supplier Trade Name']))
                            clean_gstr2a_trade = _clean_company_name(str(gstr2a_row['Supplier Trade Name']))
                            trade_name_similarity = fuzz.ratio(clean_book_trade, clean_gstr2a_trade)
                        
                        # STRICT RULE: If GSTIN similarity is low (< 80%), then either legal name or trade name similarity must be high (>= 70%)
                        if gstin_similarity < 80:
                            if legal_name_similarity < 70 and trade_name_similarity < 70:
                                continue
                        
                        # Calculate date difference
                        date_diff = np.nan
                        if pd.notna(book_row['Invoice Date']) and pd.notna(gstr2a_row['Invoice Date']):
                            date_diff = abs((book_row['Invoice Date'] - gstr2a_row['Invoice Date']).days)
                        
                        # Calculate invoice number similarity
                        invoice_similarity = 0
                        if pd.notna(book_row['Invoice Number']) and pd.notna(gstr2a_row['Invoice Number']):
                            invoice_similarity = fuzz.ratio(
                                str(book_row['Invoice Number']).lower(),
                                str(gstr2a_row['Invoice Number']).lower()
                            )
                        
                        # Calculate total score with strict weighting
                        score = 0
                        if pd.notna(date_diff) and date_diff <= 7:  # 7 days tolerance for enhanced matching
                            score += 15
                        score += (gstin_similarity * 0.4)  # 40% weight for GSTIN
                        score += (legal_name_similarity * 0.2)  # 20% weight for legal name
                        score += (trade_name_similarity * 0.2)  # 20% weight for trade name
                        score += (invoice_similarity * 0.1)  # 10% weight for invoice number
                        
                        valid_matches.append((gstr2a_row, score))
                
                if valid_matches:
                    # Sort by score
                    valid_matches.sort(key=lambda x: -x[1])
                    best_match, best_score = valid_matches[0]
                    
                    # Additional validation: require minimum score threshold for enhanced matching
                    if best_score < 70:  # Higher threshold for enhanced matching
                        continue
                    
                    # Mark as processed
                    self.books_df.at[idx, 'processed'] = True
                    self.gstr2a_df.at[best_match.name, 'processed'] = True
                    
                    # Calculate tax differences
                    igst_diff = book_row['Total IGST Amount'] - best_match['Total IGST Amount']
                    cgst_diff = book_row['Total CGST Amount'] - best_match['Total CGST Amount']
                    sgst_diff = book_row['Total SGST Amount'] - best_match['Total SGST Amount']
                    
                    # Calculate date difference
                    date_diff = np.nan
                    if pd.notna(book_row['Invoice Date']) and pd.notna(best_match['Invoice Date']):
                        date_diff = (book_row['Invoice Date'] - best_match['Invoice Date']).days
                    
                    # Calculate similarities
                    gstin_similarity = 0
                    if pd.notna(book_row['Supplier GSTIN']) and pd.notna(best_match['Supplier GSTIN']):
                        gstin_similarity = fuzz.ratio(
                            str(book_row['Supplier GSTIN']).lower(),
                            str(best_match['Supplier GSTIN']).lower()
                        )
                    
                    legal_name_similarity = 0
                    if pd.notna(book_row['Supplier Legal Name']) and pd.notna(best_match['Supplier Legal Name']):
                        legal_name_similarity = fuzz.ratio(
                            str(book_row['Supplier Legal Name']).lower(),
                            str(best_match['Supplier Legal Name']).lower()
                        )
                    
                    trade_name_similarity = 0
                    if pd.notna(book_row['Supplier Trade Name']) and pd.notna(best_match['Supplier Trade Name']):
                        clean_book_name = _clean_trade_name(str(book_row['Supplier Trade Name']))
                        clean_gstr2a_name = _clean_trade_name(str(best_match['Supplier Trade Name']))
                        trade_name_similarity = fuzz.ratio(clean_book_name, clean_gstr2a_name)
                    
                    # Add to matched records
                    match_id = self.match_id_counter
                    self.match_id_counter += 1
                    
                    # Add Books record
                    book_match_data = book_row.to_dict()
                    book_match_data.update({
                        'Match ID': match_id,
                        'Source Name': 'Books',
                        'IGST Diff': igst_diff,
                        'CGST Diff': cgst_diff,
                        'SGST Diff': sgst_diff,
                        'Date Diff': date_diff,
                        'Status': 'Tax-Based Group Match',
                        'Tax Diff Status': 'No Difference' if abs(igst_diff + cgst_diff + sgst_diff) <= self.tax_tolerance * 3 else 'Has Difference',
                        'Date Status': 'Within Tolerance' if (pd.notna(date_diff) and abs(date_diff) <= 7) else 'Outside Tolerance' if pd.notna(date_diff) else 'N/A',
                        'GSTIN Score': gstin_similarity,
                        'Legal Name Score': legal_name_similarity,
                        'Trade Name Score': trade_name_similarity,
                        'Tax Sign Status': 'Sign Match',
                        'Value Sign': 'Positive' if book_sign > 0 else 'Negative',
                        'Narrative': f"Enhanced tax-based match with score {best_score:.1f}. GSTIN: {gstin_similarity}%, Legal Name: {legal_name_similarity}%, Trade Name: {trade_name_similarity}%",
                        'Suggestions': "Verify supplier details match expected values"
                    })
                    
                    # Add GSTR-2A record
                    gstr2a_match_data = best_match.to_dict()
                    gstr2a_match_data.update({
                        'Match ID': match_id,
                        'Source Name': 'GSTR-2A',
                        'IGST Diff': -igst_diff,
                        'CGST Diff': -cgst_diff,
                        'SGST Diff': -sgst_diff,
                        'Date Diff': -date_diff if pd.notna(date_diff) else np.nan,
                        'Status': 'Tax-Based Group Match',
                        'Tax Diff Status': 'No Difference' if abs(igst_diff + cgst_diff + sgst_diff) <= self.tax_tolerance * 3 else 'Has Difference',
                        'Date Status': 'Within Tolerance' if (pd.notna(date_diff) and abs(date_diff) <= 7) else 'Outside Tolerance' if pd.notna(date_diff) else 'N/A',
                        'GSTIN Score': gstin_similarity,
                        'Legal Name Score': legal_name_similarity,
                        'Trade Name Score': trade_name_similarity,
                        'Tax Sign Status': 'Sign Match',
                        'Value Sign': 'Positive' if book_sign > 0 else 'Negative',
                        'Narrative': f"Enhanced tax-based match with score {best_score:.1f}. GSTIN: {gstin_similarity}%, Legal Name: {legal_name_similarity}%, Trade Name: {trade_name_similarity}%",
                        'Suggestions': "Verify supplier details match expected values"
                    })
                    
                    self.matched_df = pd.concat([
                        self.matched_df,
                        pd.DataFrame([book_match_data]),
                        pd.DataFrame([gstr2a_match_data])
                    ], ignore_index=True)
        
    def _find_potential_data_entry_swaps(self, books_df, gstr2a_df):
        """Find potential data entry errors where invoice numbers and taxes are swapped."""
        swaps = []
        
        # Group by GSTIN
        books_by_gstin = books_df.groupby('Supplier GSTIN')
        gstr2a_by_gstin = gstr2a_df.groupby('Supplier GSTIN')
        
        for gstin in set(books_by_gstin.groups.keys()) & set(gstr2a_by_gstin.groups.keys()):
            books_group = books_by_gstin.get_group(gstin)
            gstr2a_group = gstr2a_by_gstin.get_group(gstin)
            
            # Check for potential swaps (2 records in each source with same GSTIN)
            if len(books_group) == 2 and len(gstr2a_group) == 2:
                books_records = books_group.to_dict('records')
                gstr2a_records = gstr2a_group.to_dict('records')
                
                # Check if swapping invoice numbers would create matches
                for i in range(2):
                    for j in range(2):
                        book1 = books_records[i]
                        book2 = books_records[1-i]  # Other book record
                        gstr2a1 = gstr2a_records[j]
                        gstr2a2 = gstr2a_records[1-j]  # Other GSTR-2A record
                        
                        # Check if swapping would create matches
                        if (book1['Invoice Number'] == gstr2a2['Invoice Number'] and 
                            book2['Invoice Number'] == gstr2a1['Invoice Number']):
                            
                            # Check if taxes match after swap
                            book1_tax_match = (
                                abs(book1['Total IGST Amount'] - gstr2a1['Total IGST Amount']) <= self.tax_tolerance and
                                abs(book1['Total CGST Amount'] - gstr2a1['Total CGST Amount']) <= self.tax_tolerance and
                                abs(book1['Total SGST Amount'] - gstr2a1['Total SGST Amount']) <= self.tax_tolerance
                            )
                            
                            book2_tax_match = (
                                abs(book2['Total IGST Amount'] - gstr2a2['Total IGST Amount']) <= self.tax_tolerance and
                                abs(book2['Total CGST Amount'] - gstr2a2['Total CGST Amount']) <= self.tax_tolerance and
                                abs(book2['Total SGST Amount'] - gstr2a2['Total SGST Amount']) <= self.tax_tolerance
                            )
                            
                            if book1_tax_match and book2_tax_match:
                                swaps.append({
                                    'gstin': gstin,
                                    'books_records': [book1, book2],
                                    'gstr2a_records': [gstr2a1, gstr2a2],
                                    'swap_type': 'invoice_swap',
                                    'suggestion': f"Swap invoice numbers: {book1['Invoice Number']} ↔ {book2['Invoice Number']}"
                                })
        
        return swaps

    def _process_data_entry_swap(self, swap_info):
        """Process a data entry swap by creating matched records with suggestions."""
        books_records = swap_info['books_records']
        gstr2a_records = swap_info['gstr2a_records']
        
        match_id = self.match_id_counter
        self.match_id_counter += 1
        
        # Process first pair
        book1 = books_records[0]
        gstr2a1 = gstr2a_records[0]
        
        # Get the actual DataFrame indices
        book1_idx = book1.get('name', None)
        gstr2a1_idx = gstr2a1.get('name', None)
        
        # Mark as processed if indices are available
        if book1_idx is not None:
            self.books_df.at[book1_idx, 'processed'] = True
        if gstr2a1_idx is not None:
            self.gstr2a_df.at[gstr2a1_idx, 'processed'] = True
        
        # Calculate differences
        igst_diff = book1['Total IGST Amount'] - gstr2a1['Total IGST Amount']
        cgst_diff = book1['Total CGST Amount'] - gstr2a1['Total CGST Amount']
        sgst_diff = book1['Total SGST Amount'] - gstr2a1['Total SGST Amount']
        
        # Add Books record
        book_match_data = book1.copy()
        book_match_data.update({
            'Match ID': match_id,
            'Source Name': 'Books',
            'IGST Diff': igst_diff,
            'CGST Diff': cgst_diff,
            'SGST Diff': sgst_diff,
            'Status': 'Data Entry Swap Match',
            'Tax Diff Status': 'No Difference',
            'Narrative': f"Data entry swap detected. {swap_info['suggestion']}",
            'Suggestions': f"Correct data entry: {swap_info['suggestion']}"
        })
        
        # Add GSTR-2A record
        gstr2a_match_data = gstr2a1.copy()
        gstr2a_match_data.update({
            'Match ID': match_id,
            'Source Name': 'GSTR-2A',
            'IGST Diff': -igst_diff,
            'CGST Diff': -cgst_diff,
            'SGST Diff': -sgst_diff,
            'Status': 'Data Entry Swap Match',
            'Tax Diff Status': 'No Difference',
            'Narrative': f"Data entry swap detected. {swap_info['suggestion']}",
            'Suggestions': f"Correct data entry: {swap_info['suggestion']}"
        })
        
        self.matched_df = pd.concat([
            self.matched_df,
            pd.DataFrame([book_match_data]),
            pd.DataFrame([gstr2a_match_data])
        ], ignore_index=True)
        
        # Process second pair
        book2 = books_records[1]
        gstr2a2 = gstr2a_records[1]
        
        # Get the actual DataFrame indices
        book2_idx = book2.get('name', None)
        gstr2a2_idx = gstr2a2.get('name', None)
        
        # Mark as processed if indices are available
        if book2_idx is not None:
            self.books_df.at[book2_idx, 'processed'] = True
        if gstr2a2_idx is not None:
            self.gstr2a_df.at[gstr2a2_idx, 'processed'] = True
        
        # Calculate differences
        igst_diff2 = book2['Total IGST Amount'] - gstr2a2['Total IGST Amount']
        cgst_diff2 = book2['Total CGST Amount'] - gstr2a2['Total CGST Amount']
        sgst_diff2 = book2['Total SGST Amount'] - gstr2a2['Total SGST Amount']
        
        # Generate field status columns for data entry swap
        field_status = self._generate_field_status_columns(book2, gstr2a2)
        
        # Add Books record
        book_match_data2 = book2.copy()
        book_match_data2.update({
            'Match ID': match_id + 1,
            'Source Name': 'Books',
            'IGST Diff': igst_diff2,
            'CGST Diff': cgst_diff2,
            'SGST Diff': sgst_diff2,
            'Status': 'Data Entry Swap Match',
            'Sub Status': f"Data entry swap detected: {swap_info['suggestion']}",
            'Tax Diff Status': 'No Difference',
            'Narrative': f"Data entry swap detected. {swap_info['suggestion']}",
            'Suggestions': f"Correct data entry: {swap_info['suggestion']}"
        })
        
        # Add field status columns
        book_match_data2.update(field_status)
        
        # Add GSTR-2A record
        gstr2a_match_data2 = gstr2a2.copy()
        gstr2a_match_data2.update({
            'Match ID': match_id + 1,
            'Source Name': 'GSTR-2A',
            'IGST Diff': -igst_diff2,
            'CGST Diff': -cgst_diff2,
            'SGST Diff': -sgst_diff2,
            'Status': 'Data Entry Swap Match',
            'Sub Status': f"Data entry swap detected: {swap_info['suggestion']}",
            'Tax Diff Status': 'No Difference',
            'Narrative': f"Data entry swap detected. {swap_info['suggestion']}",
            'Suggestions': f"Correct data entry: {swap_info['suggestion']}"
        })
        
        # Add field status columns
        gstr2a_match_data2.update(field_status)
        
        self.matched_df = pd.concat([
            self.matched_df,
            pd.DataFrame([book_match_data2]),
            pd.DataFrame([gstr2a_match_data2])
        ], ignore_index=True)
        
        self.match_id_counter += 1

    def _find_tax_based_groups(self, books_df, gstr2a_df):
        """Find groups where same GSTIN has different invoices but net tax difference is zero."""
        groups = []
        
        # Group by GSTIN
        books_by_gstin = books_df.groupby('Supplier GSTIN')
        gstr2a_by_gstin = gstr2a_df.groupby('Supplier GSTIN')
        
        for gstin in set(books_by_gstin.groups.keys()) & set(gstr2a_by_gstin.groups.keys()):
            books_group = books_by_gstin.get_group(gstin)
            gstr2a_group = gstr2a_by_gstin.get_group(gstin)
            
            # Calculate total taxes for each group
            books_total_igst = books_group['Total IGST Amount'].sum()
            books_total_cgst = books_group['Total CGST Amount'].sum()
            books_total_sgst = books_group['Total SGST Amount'].sum()
            
            gstr2a_total_igst = gstr2a_group['Total IGST Amount'].sum()
            gstr2a_total_cgst = gstr2a_group['Total CGST Amount'].sum()
            gstr2a_total_sgst = gstr2a_group['Total SGST Amount'].sum()
            
            # Check if net difference is within tolerance
            igst_diff = abs(books_total_igst - gstr2a_total_igst)
            cgst_diff = abs(books_total_cgst - gstr2a_total_cgst)
            sgst_diff = abs(books_total_sgst - gstr2a_total_sgst)
            
            if (igst_diff <= self.group_tax_tolerance and 
                cgst_diff <= self.group_tax_tolerance and 
                sgst_diff <= self.group_tax_tolerance):
                
                groups.append({
                    'gstin': gstin,
                    'books_group': books_group,
                    'gstr2a_group': gstr2a_group,
                    'igst_diff': books_total_igst - gstr2a_total_igst,
                    'cgst_diff': books_total_cgst - gstr2a_total_cgst,
                    'sgst_diff': books_total_sgst - gstr2a_total_sgst,
                    'group_type': 'tax_based'
                })
        
        return groups

    def _process_tax_based_group(self, group_info):
        """Process a tax-based group match."""
        books_group = group_info['books_group']
        gstr2a_group = group_info['gstr2a_group']
        
        group_id = f"TB{self.group_id_counter}"
        self.group_id_counter += 1
        
        # Mark all records as processed
        for idx in books_group.index:
            self.books_df.at[idx, 'processed'] = True
        for idx in gstr2a_group.index:
            self.gstr2a_df.at[idx, 'processed'] = True
        
        # Add all records to matched DataFrame
        for _, book_row in books_group.iterrows():
            # Generate field status columns (using first GSTR-2A record for comparison)
            first_gstr2a = gstr2a_group.iloc[0]
            field_status = self._generate_field_status_columns(book_row, first_gstr2a)
            
            book_match_data = book_row.to_dict()
            book_match_data.update({
                'Match ID': self.match_id_counter,
                'Group ID': group_id,
                'Source Name': 'Books',
                'Status': 'Tax-Based Group Match',
                'Sub Status': f"Tax-based group match for GSTIN {group_info['gstin']}",
                'IGST Diff': group_info['igst_diff'],
                'CGST Diff': group_info['cgst_diff'],
                'SGST Diff': group_info['sgst_diff'],
                'Tax Diff Status': 'No Difference',
                'Narrative': f"Tax-based group match for GSTIN {group_info['gstin']}. Net tax difference: ₹{abs(group_info['igst_diff'] + group_info['cgst_diff'] + group_info['sgst_diff']):.2f}",
                'Suggestions': "Verify individual invoice details match expected values"
            })
            
            # Add field status columns
            book_match_data.update(field_status)
            
            self.matched_df = pd.concat([
                self.matched_df,
                pd.DataFrame([book_match_data])
            ], ignore_index=True)
        
        for _, gstr2a_row in gstr2a_group.iterrows():
            # Generate field status columns (using first Books record for comparison)
            first_book = books_group.iloc[0]
            field_status = self._generate_field_status_columns(first_book, gstr2a_row)
            
            gstr2a_match_data = gstr2a_row.to_dict()
            gstr2a_match_data.update({
                'Match ID': self.match_id_counter,
                'Group ID': group_id,
                'Source Name': 'GSTR-2A',
                'Status': 'Tax-Based Group Match',
                'Sub Status': f"Tax-based group match for GSTIN {group_info['gstin']}",
                'IGST Diff': -group_info['igst_diff'],
                'CGST Diff': -group_info['cgst_diff'],
                'SGST Diff': -group_info['sgst_diff'],
                'Tax Diff Status': 'No Difference',
                'Narrative': f"Tax-based group match for GSTIN {group_info['gstin']}. Net tax difference: ₹{abs(group_info['igst_diff'] + group_info['cgst_diff'] + group_info['sgst_diff']):.2f}",
                'Suggestions': "Verify individual invoice details match expected values"
            })
            
            # Add field status columns
            gstr2a_match_data.update(field_status)
            
            self.matched_df = pd.concat([
                self.matched_df,
                pd.DataFrame([gstr2a_match_data])
            ], ignore_index=True)
        
        self.match_id_counter += 1

    def _generate_narrative(self, book_row, gstr2a_row, is_exact_match):
        """Generate a narrative for the match."""
        narrative_parts = []
        
        # Check tax differences
        igst_diff = book_row['Total IGST Amount'] - gstr2a_row['Total IGST Amount']
        cgst_diff = book_row['Total CGST Amount'] - gstr2a_row['Total CGST Amount']
        sgst_diff = book_row['Total SGST Amount'] - gstr2a_row['Total SGST Amount']
        total_tax_diff = igst_diff + cgst_diff + sgst_diff
        
        if abs(total_tax_diff) > self.tax_tolerance:
            narrative_parts.append(f"Tax difference of ₹{abs(total_tax_diff):.2f} found")
            if abs(igst_diff) > self.tax_tolerance:
                narrative_parts.append(f"IGST difference: ₹{abs(igst_diff):.2f}")
            if abs(cgst_diff) > self.tax_tolerance:
                narrative_parts.append(f"CGST difference: ₹{abs(cgst_diff):.2f}")
            if abs(sgst_diff) > self.tax_tolerance:
                narrative_parts.append(f"SGST difference: ₹{abs(sgst_diff):.2f}")
        
        # Check date differences
        calculated_date_diff = np.nan
        if pd.notna(book_row['Invoice Date']) and pd.notna(gstr2a_row['Invoice Date']):
            calculated_date_diff = abs((book_row['Invoice Date'] - gstr2a_row['Invoice Date']).days)
        
        if pd.notna(calculated_date_diff) and calculated_date_diff > self.date_tolerance.days:
            narrative_parts.append(f"Date difference of {calculated_date_diff} days found")
        
        # Check name differences
        name_similarity = fuzz.token_sort_ratio(book_row['Supplier Legal Name'], gstr2a_row['Supplier Legal Name'])
        trade_name_similarity = fuzz.token_sort_ratio(book_row['Supplier Trade Name'], gstr2a_row['Supplier Trade Name'])

        if name_similarity < 80 or trade_name_similarity < 80:
            narrative_parts.append(f"Name mismatch: Legal Name similarity {name_similarity}%, Trade Name similarity {trade_name_similarity}%")

        return "; ".join(narrative_parts) if narrative_parts else "No significant discrepancies found."

    def _generate_suggestions(self, book_row, gstr2a_row, is_exact_match):
        """Generate suggestions for the user based on discrepancies."""
        suggestions = []

        if is_exact_match:
            suggestions.append("Mark as reconciled.")
        else:
            # Check tax differences
            igst_diff = book_row['Total IGST Amount'] - gstr2a_row['Total IGST Amount']
            cgst_diff = book_row['Total CGST Amount'] - gstr2a_row['Total CGST Amount']
            sgst_diff = book_row['Total SGST Amount'] - gstr2a_row['Total SGST Amount']
            total_tax_diff = igst_diff + cgst_diff + sgst_diff

            if abs(total_tax_diff) > self.tax_tolerance:
                suggestions.append("Review tax amounts for discrepancy.")

            # Check date differences
            calculated_date_diff = np.nan
            if pd.notna(book_row['Invoice Date']) and pd.notna(gstr2a_row['Invoice Date']):
                calculated_date_diff = abs((book_row['Invoice Date'] - gstr2a_row['Invoice Date']).days)

            if pd.notna(calculated_date_diff) and calculated_date_diff > self.date_tolerance.days:
                suggestions.append("Verify invoice date for discrepancy.")

            # Check name differences
            name_similarity = fuzz.token_sort_ratio(book_row['Supplier Legal Name'], gstr2a_row['Supplier Legal Name'])
            trade_name_similarity = fuzz.token_sort_ratio(book_row['Supplier Trade Name'], gstr2a_row['Supplier Trade Name'])

            if name_similarity < 80 or trade_name_similarity < 80:
                suggestions.append("Check supplier name for variations.")
            
            if not suggestions:
                suggestions.append("Further manual review might be required for subtle differences.")

        return "; ".join(suggestions) if suggestions else "No specific suggestions."

    def _check_tax_head_mismatch(self, book_row, gstr2a_row):
        """
        Check if there's a tax head mismatch between Books and GSTR-2A.
        Returns True if there's a mismatch (e.g., IGST in Books vs CGST/SGST in GSTR-2A).
        """
        # Check if one has IGST and the other has CGST+SGST
        book_has_igst = abs(book_row['Total IGST Amount']) > 0
        book_has_cgst_sgst = abs(book_row['Total CGST Amount']) > 0 or abs(book_row['Total SGST Amount']) > 0
        
        gstr2a_has_igst = abs(gstr2a_row['Total IGST Amount']) > 0
        gstr2a_has_cgst_sgst = abs(gstr2a_row['Total CGST Amount']) > 0 or abs(gstr2a_row['Total SGST Amount']) > 0
        
        # Mismatch if one has IGST and the other has CGST+SGST
        if (book_has_igst and gstr2a_has_cgst_sgst) or (book_has_cgst_sgst and gstr2a_has_igst):
            return True
        
        return False

    def _generate_sub_status_comment(self, book_row, gstr2a_row, igst_match, cgst_match, sgst_match, 
                                   tax_head_mismatch, cross_year, is_exact_match):
        """
        Generate detailed sub-status comments based on the comparison results.
        """
        comments = []
        
        if is_exact_match:
            if cross_year:
                comments.append("All fields including tax heads and amounts fully matched across financial years.")
            else:
                comments.append("All fields including tax heads and amounts fully matched.")
        else:
            if tax_head_mismatch:
                book_has_igst = abs(book_row['Total IGST Amount']) > 0
                book_has_cgst_sgst = abs(book_row['Total CGST Amount']) > 0 or abs(book_row['Total SGST Amount']) > 0
                gstr2a_has_igst = abs(gstr2a_row['Total IGST Amount']) > 0
                gstr2a_has_cgst_sgst = abs(gstr2a_row['Total CGST Amount']) > 0 or abs(gstr2a_row['Total SGST Amount']) > 0
                
                if book_has_igst and gstr2a_has_cgst_sgst:
                    comments.append("Tax head mismatch: Books IGST vs. GSTR-2A CGST/SGST")
                elif book_has_cgst_sgst and gstr2a_has_igst:
                    comments.append("Tax head mismatch: Books CGST/SGST vs. GSTR-2A IGST")
            
            if not igst_match or not cgst_match or not sgst_match:
                tax_issues = []
                if not igst_match:
                    tax_issues.append("IGST amount difference")
                if not cgst_match:
                    tax_issues.append("CGST amount difference")
                if not sgst_match:
                    tax_issues.append("SGST amount difference")
                comments.append(f"Partial match due to: {', '.join(tax_issues)}")
            
            if cross_year:
                comments.append("Cross-year comparison")
        
        return "; ".join(comments) if comments else "Standard match"

    def _generate_group_sub_status(self, match_status, legal_name_similarity, trade_name_similarity, 
                                  invoice_similarity, igst_diff, cgst_diff, sgst_diff):
        """
        Generate sub-status comments for group matching scenarios.
        """
        comments = []
        
        if match_status == 'Partial Match':
            comments.append("Partial match based on name and invoice number similarity")
            if max(legal_name_similarity, trade_name_similarity) < 90:
                comments.append(f"Name similarity: {max(legal_name_similarity, trade_name_similarity)}%")
            if invoice_similarity < 90:
                comments.append(f"Invoice similarity: {invoice_similarity}%")
        elif match_status == 'Potential Match: Tax Deviation':
            comments.append("Potential match with high tax deviation requiring manual review")
            tax_issues = []
            if abs(igst_diff) > self.tax_tolerance:
                tax_issues.append(f"IGST difference: ₹{abs(igst_diff):.2f}")
            if abs(cgst_diff) > self.tax_tolerance:
                tax_issues.append(f"CGST difference: ₹{abs(cgst_diff):.2f}")
            if abs(sgst_diff) > self.tax_tolerance:
                tax_issues.append(f"SGST difference: ₹{abs(sgst_diff):.2f}")
            if tax_issues:
                comments.append("; ".join(tax_issues))
        
        return "; ".join(comments) if comments else "Group match"

    def _generate_field_status_columns(self, book_row, gstr2a_row):
        """
        Generate simple column values for field status comparisons.
        """
        return {
            'Tax Head Status': self._get_tax_head_status(book_row, gstr2a_row),
            'GSTIN Match Status': self._check_gstin_status(book_row, gstr2a_row),
            'Invoice Match Status': self._check_invoice_number_status(book_row, gstr2a_row),
            'Trade Name Match Status': self._check_trade_name_status(book_row, gstr2a_row),
            'Legal Name Match Status': self._check_legal_name_status(book_row, gstr2a_row)
        }

    def _get_tax_head_status(self, book_row, gstr2a_row):
        """Get simple tax head status."""
        book_has_igst = abs(book_row['Total IGST Amount']) > 0
        book_has_cgst_sgst = abs(book_row['Total CGST Amount']) > 0 or abs(book_row['Total SGST Amount']) > 0
        
        gstr2a_has_igst = abs(gstr2a_row['Total IGST Amount']) > 0
        gstr2a_has_cgst_sgst = abs(gstr2a_row['Total CGST Amount']) > 0 or abs(gstr2a_row['Total SGST Amount']) > 0
        
        if (book_has_igst and gstr2a_has_cgst_sgst) or (book_has_cgst_sgst and gstr2a_has_igst):
            return "Mismatched"
        return "Matched"

    def _check_gstin_status(self, book_row, gstr2a_row):
        """Check GSTIN matching status."""
        book_gstin = str(book_row['Supplier GSTIN']).strip().upper()
        gstr2a_gstin = str(gstr2a_row['Supplier GSTIN']).strip().upper()
        
        if book_gstin == gstr2a_gstin:
            return "Matched"
        elif book_gstin.replace('-', '') == gstr2a_gstin.replace('-', ''):
            return "Matched but with formatting difference"
        else:
            return "Unmatched"

    def _check_invoice_number_status(self, book_row, gstr2a_row):
        """Check Invoice Number matching status."""
        book_inv = str(book_row['Invoice Number']).strip()
        gstr2a_inv = str(gstr2a_row['Invoice Number']).strip()
        
        if book_inv == gstr2a_inv:
            return "Matched"
        elif book_inv.lower() == gstr2a_inv.lower():
            return "Matched but with case difference"
        else:
            # Check for partial match (common prefixes/suffixes)
            if len(book_inv) > 3 and len(gstr2a_inv) > 3:
                if book_inv[:3] == gstr2a_inv[:3] or book_inv[-3:] == gstr2a_inv[-3:]:
                    return "Partial Matched due to prefix/suffix variation"
            return "Unmatched"

    def _check_trade_name_status(self, book_row, gstr2a_row):
        """Check Trade Name matching status."""
        book_name = str(book_row['Supplier Trade Name']).strip()
        gstr2a_name = str(gstr2a_row['Supplier Trade Name']).strip()
        
        if book_name == gstr2a_name:
            return "Matched"
        
        # Use fuzzy matching for similarity
        from fuzzywuzzy import fuzz
        similarity = fuzz.token_sort_ratio(book_name, gstr2a_name)
        
        if similarity >= 90:
            return "Matched with minor variations"
        elif similarity >= 70:
            return "Partial Matched"
        else:
            return f"Unmatched: Books '{book_name}' vs. GSTR-2A '{gstr2a_name}'"

    def _check_legal_name_status(self, book_row, gstr2a_row):
        """Check Legal Name matching status."""
        book_name = str(book_row['Supplier Legal Name']).strip()
        gstr2a_name = str(gstr2a_row['Supplier Legal Name']).strip()
        
        if book_name == gstr2a_name:
            return "Matched"
        
        # Use fuzzy matching for similarity
        from fuzzywuzzy import fuzz
        similarity = fuzz.token_sort_ratio(book_name, gstr2a_name)
        
        if similarity >= 90:
            return "Matched with minor variations"
        elif similarity >= 70:
            return "Partial Matched"
        else:
            return f"Unmatched: Books '{book_name}' vs. GSTR-2A '{gstr2a_name}'"

    def _generate_final_report(self):
        """Generate the final report DataFrame by combining matched and mismatched records."""
        # Ensure all necessary columns are present in matched_df before concatenation
        # Define a comprehensive list of all expected columns in the final report
        all_expected_columns = [
            'Match ID', 'Group ID', 'Source Name', 'Supplier GSTIN', 'Supplier Legal Name', 'Supplier Trade Name',
            'Invoice Number', 'Invoice Date', 'Books Date', 'Total Taxable Value', 'Total Tax Value',
            'Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount', 'Total Invoice Value',
            'IGST Diff', 'CGST Diff', 'SGST Diff', 'Date Diff', 'Status', 'Sub Status', 'Tax Diff Status', 'Date Status',
            'Tax Sign Status', 'Value Sign', 'Narrative', 'Suggestions', 'Tax Head Status', 'GSTIN Match Status',
            'Invoice Match Status', 'Trade Name Match Status', 'Legal Name Match Status'
        ]

        # Add missing columns to self.matched_df with appropriate default values
        for col in all_expected_columns:
            if col not in self.matched_df.columns:
                if 'Diff' in col or 'Value' in col:
                    self.matched_df[col] = np.nan
                elif col == 'Sub Status':
                    self.matched_df[col] = 'Status not determined'
                elif col in ['Tax Head Status', 'GSTIN Match Status', 'Invoice Match Status', 'Trade Name Match Status', 'Legal Name Match Status']:
                    self.matched_df[col] = 'N/A'
                else:
                    self.matched_df[col] = ''

        # Add missing columns to self.mismatch_df with appropriate default values
        for col in all_expected_columns:
            if col not in self.mismatch_df.columns:
                if 'Diff' in col or 'Value' in col:
                    self.mismatch_df[col] = np.nan
                elif col == 'Sub Status':
                    self.mismatch_df[col] = 'Status not determined'
                elif col in ['Tax Head Status', 'GSTIN Match Status', 'Invoice Match Status', 'Trade Name Match Status', 'Legal Name Match Status']:
                    self.mismatch_df[col] = 'N/A'
                else:
                    self.mismatch_df[col] = ''
        
        self.final_report_df = pd.concat([
            self.matched_df,
            self.mismatch_df
        ], ignore_index=True).sort_values(by=['Match ID', 'Source Name']).reset_index(drop=True)
        
        # FORCE all new columns to have values - this is the final safety net
        new_columns = [
            'Sub Status', 'Tax Head Status', 'GSTIN Match Status', 
            'Invoice Match Status', 'Trade Name Match Status', 'Legal Name Match Status'
        ]
        
        for col in new_columns:
            if col in self.final_report_df.columns:
                # Replace any NaN or empty string values with meaningful defaults
                mask = self.final_report_df[col].isna() | (self.final_report_df[col] == '')
                if mask.any():
                    if col == 'Sub Status':
                        self.final_report_df.loc[mask, col] = 'Status determined by system'
                    else:
                        self.final_report_df.loc[mask, col] = 'N/A'
            else:
                # If column is completely missing, add it with default values
                if col == 'Sub Status':
                    self.final_report_df[col] = 'Status determined by system'
                else:
                    self.final_report_df[col] = 'N/A'

        # Update progress
        self.step_progress = 1.0

    def get_results(self):
        """Return a dictionary of reconciliation results and counts."""
        # Always regenerate final_report_df to ensure it's up to date
        self._generate_final_report()

        # Calculate counts from final_report_df statuses
        matched_count = len(self.final_report_df[self.final_report_df['Status'] == 'Exact Match'])
        partial_count = len(self.final_report_df[self.final_report_df['Status'] == 'Partial Match'])
        group_count = len(self.final_report_df[self.final_report_df['Status'] == 'Group Match'])
        data_entry_swap_count = len(self.final_report_df[self.final_report_df['Status'] == 'Data Entry Swap Match'])
        tax_based_group_count = len(self.final_report_df[self.final_report_df['Status'] == 'Tax-Based Group Match'])
        books_only_count = len(self.final_report_df[self.final_report_df['Status'] == 'Books Only'])
        gstr2a_only_count = len(self.final_report_df[self.final_report_df['Status'] == 'GSTR-2A Only'])
        sign_cancellation_count = len(self.final_report_df[self.final_report_df['Status'].str.contains('Sign Cancellation', na=False)])
        
        # Calculate total records processed (excluding sign cancellations as they're internal)
        total_records_processed = len(self.final_report_df) // 2 if len(self.final_report_df) > 0 else 0
        
        # Re-calculate value summaries from final_report_df
        matched_value = self.final_report_df[self.final_report_df['Status'] == 'Exact Match']['Total Invoice Value'].sum() / 2
        partial_value = self.final_report_df[self.final_report_df['Status'] == 'Partial Match']['Total Invoice Value'].sum() / 2
        group_value = self.final_report_df[self.final_report_df['Status'] == 'Group Match']['Total Invoice Value'].sum() / 2
        data_entry_swap_value = self.final_report_df[self.final_report_df['Status'] == 'Data Entry Swap Match']['Total Invoice Value'].sum() / 2
        tax_based_group_value = self.final_report_df[self.final_report_df['Status'] == 'Tax-Based Group Match']['Total Invoice Value'].sum() / 2
        books_only_value = self.final_report_df[self.final_report_df['Status'] == 'Books Only']['Total Invoice Value'].sum()
        gstr2a_only_value = self.final_report_df[self.final_report_df['Status'] == 'GSTR-2A Only']['Total Invoice Value'].sum()
        sign_cancellation_value = self.final_report_df[self.final_report_df['Status'].str.contains('Sign Cancellation', na=False)]['Total Invoice Value'].sum() / 2
        
        total_books_value = self.books_df['Total Invoice Value'].sum()
        total_gstr2a_value = self.gstr2a_df['Total Invoice Value'].sum()

        # Calculate percentages based on total records (count and value)
        matched_percentage = (matched_count / total_records_processed * 100) if total_records_processed > 0 else 0
        partial_percentage = (partial_count / total_records_processed * 100) if total_records_processed > 0 else 0
        group_percentage = (group_count / total_records_processed * 100) if total_records_processed > 0 else 0
        data_entry_swap_percentage = (data_entry_swap_count / total_records_processed * 100) if total_records_processed > 0 else 0
        tax_based_group_percentage = (tax_based_group_count / total_records_processed * 100) if total_records_processed > 0 else 0
        books_only_percentage = (books_only_count / total_records_processed * 100) if total_records_processed > 0 else 0
        gstr2a_only_percentage = (gstr2a_only_count / total_records_processed * 100) if total_records_processed > 0 else 0

        matched_value_percentage = (matched_value / total_books_value * 100) if total_books_value > 0 else 0
        partial_value_percentage = (partial_value / total_books_value * 100) if total_books_value > 0 else 0
        group_value_percentage = (group_value / total_books_value * 100) if total_books_value > 0 else 0
        data_entry_swap_value_percentage = (data_entry_swap_value / total_books_value * 100) if total_books_value > 0 else 0
        tax_based_group_value_percentage = (tax_based_group_value / total_books_value * 100) if total_books_value > 0 else 0
        books_only_value_percentage = (books_only_value / total_books_value * 100) if total_books_value > 0 else 0
        gstr2a_only_value_percentage = (gstr2a_only_value / total_gstr2a_value * 100) if total_gstr2a_value > 0 else 0
        
        # Calculate overall tax differences from final_report_df
        igst_diff = self.final_report_df['IGST Diff'].sum()
        cgst_diff = self.final_report_df['CGST Diff'].sum()
        sgst_diff = self.final_report_df['SGST Diff'].sum()

        total_igst_books = self.books_df['Total IGST Amount'].sum()
        total_cgst_books = self.books_df['Total CGST Amount'].sum()
        total_sgst_books = self.books_df['Total SGST Amount'].sum()

        igst_diff_percentage = (abs(igst_diff) / total_igst_books * 100) if total_igst_books > 0 else 0
        cgst_diff_percentage = (abs(cgst_diff) / total_cgst_books * 100) if total_cgst_books > 0 else 0
        sgst_diff_percentage = (abs(sgst_diff) / total_sgst_books * 100) if total_sgst_books > 0 else 0

        # Calculate counts and values for raw and reconciliation summaries directly from internal dfs
        raw_books_count = len(self.books_df)
        raw_gstr2a_count = len(self.gstr2a_df)
        raw_books_igst = self.books_df['Total IGST Amount'].sum()
        raw_books_cgst = self.books_df['Total CGST Amount'].sum()
        raw_books_sgst = self.books_df['Total SGST Amount'].sum()
        raw_books_value = self.books_df['Total Invoice Value'].sum()
        raw_gstr2a_igst = self.gstr2a_df['Total IGST Amount'].sum()
        raw_gstr2a_cgst = self.gstr2a_df['Total CGST Amount'].sum()
        raw_gstr2a_sgst = self.gstr2a_df['Total SGST Amount'].sum()
        raw_gstr2a_value = self.gstr2a_df['Total Invoice Value'].sum()

        # Generate all summaries using the updated final_report_df
        status_summary = self._generate_status_summary()
        raw_summary = self._generate_raw_summary()
        recon_summary = self._generate_recon_summary()
        integrity_checks = self._perform_integrity_checks()
        tax_summary = self._generate_tax_summary()
        exact_matches_summary = self._generate_exact_matches_summary()
        partial_matches_summary = self._generate_partial_matches_summary()
        group_matches_summary = self._generate_group_matches_summary()
        cross_year_matches_summary = self._generate_cross_year_matches_summary()

        return {
            'current_step': self.current_step,
            'step_progress': self.step_progress,
            'matched_count': matched_count,
            'partial_count': partial_count,
            'group_count': group_count,
            'data_entry_swap_count': data_entry_swap_count,
            'tax_based_group_count': tax_based_group_count,
            'books_only_count': books_only_count,
            'gstr2a_only_count': gstr2a_only_count,
            'sign_cancellation_count': sign_cancellation_count,
            'matched_value': matched_value,
            'partial_value': partial_value,
            'group_value': group_value,
            'data_entry_swap_value': data_entry_swap_value,
            'tax_based_group_value': tax_based_group_value,
            'books_only_value': books_only_value,
            'gstr2a_only_value': gstr2a_only_value,
            'sign_cancellation_value': sign_cancellation_value,
            'matched_percentage': matched_percentage,
            'partial_percentage': partial_percentage,
            'group_percentage': group_percentage,
            'data_entry_swap_percentage': data_entry_swap_percentage,
            'tax_based_group_percentage': tax_based_group_percentage,
            'books_only_percentage': books_only_percentage,
            'gstr2a_only_percentage': gstr2a_only_percentage,
            'matched_value_percentage': matched_value_percentage,
            'partial_value_percentage': partial_value_percentage,
            'group_value_percentage': group_value_percentage,
            'data_entry_swap_value_percentage': data_entry_swap_value_percentage,
            'tax_based_group_value_percentage': tax_based_group_value_percentage,
            'books_only_value_percentage': books_only_value_percentage,
            'gstr2a_only_value_percentage': gstr2a_only_value_percentage,
            'igst_diff': igst_diff,
            'cgst_diff': cgst_diff,
            'sgst_diff': sgst_diff,
            'igst_diff_percentage': igst_diff_percentage,
            'cgst_diff_percentage': cgst_diff_percentage,
            'sgst_diff_percentage': sgst_diff_percentage,
            'raw_books_count': raw_books_count,
            'raw_gstr2a_count': raw_gstr2a_count,
            'raw_books_igst': raw_books_igst,
            'raw_books_cgst': raw_books_cgst,
            'raw_books_sgst': raw_books_sgst,
            'raw_books_value': raw_books_value,
            'raw_gstr2a_igst': raw_gstr2a_igst,
            'raw_gstr2a_cgst': raw_gstr2a_cgst,
            'raw_gstr2a_sgst': raw_gstr2a_sgst,
            'raw_gstr2a_value': raw_gstr2a_value,
            'matched_df': self.matched_df,
            'mismatch_df': self.mismatch_df,
            'final_report': self.final_report_df,  # Add final report to results
            'raw_summary': raw_summary,
            'recon_summary': recon_summary,
            'integrity_checks': integrity_checks,
            'status_summary': status_summary,
            'tax_summary': tax_summary,
            'exact_matches_summary': exact_matches_summary,
            'partial_matches_summary': partial_matches_summary,
            'group_matches_summary': group_matches_summary,
            'cross_year_matches_summary': cross_year_matches_summary
        }

    def _generate_raw_summary(self):
        """Generate a summary of raw data for Books and GSTR-2A."""
        data = []

        # Books Summary
        books_count = len(self.books_df)
        books_igst = self.books_df['Total IGST Amount'].sum()
        books_cgst = self.books_df['Total CGST Amount'].sum()
        books_sgst = self.books_df['Total SGST Amount'].sum()
        books_value = self.books_df['Total Invoice Value'].sum()
        data.append({
            'Source': 'Books',
            'Record Count': books_count,
            'Total IGST': books_igst,
            'Total CGST': books_cgst,
            'Total SGST': books_sgst,
            'Total Value': books_value
        })

        # GSTR-2A Summary
        gstr2a_count = len(self.gstr2a_df)
        gstr2a_igst = self.gstr2a_df['Total IGST Amount'].sum()
        gstr2a_cgst = self.gstr2a_df['Total CGST Amount'].sum()
        gstr2a_sgst = self.gstr2a_df['Total SGST Amount'].sum()
        gstr2a_value = self.gstr2a_df['Total Invoice Value'].sum()
        data.append({
            'Source': 'GSTR-2A',
            'Record Count': gstr2a_count,
            'Total IGST': gstr2a_igst,
            'Total CGST': gstr2a_cgst,
            'Total SGST': gstr2a_sgst,
            'Total Value': gstr2a_value
        })
        
        # Total Row
        total_raw_count = books_count + gstr2a_count
        total_raw_igst = books_igst + gstr2a_igst
        total_raw_cgst = books_cgst + gstr2a_cgst
        total_raw_sgst = books_sgst + gstr2a_sgst
        total_raw_value = books_value + gstr2a_value
        data.append({
            'Source': 'Total',
            'Record Count': total_raw_count,
            'Total IGST': total_raw_igst,
            'Total CGST': total_raw_cgst,
            'Total SGST': total_raw_sgst,
            'Total Value': total_raw_value
        })

        return pd.DataFrame(data)

    def _generate_recon_summary(self):
        """Generate a summary of reconciliation results."""
        data = []

        # Use final_report_df for all calculations
        final_books = self.final_report_df[self.final_report_df['Source Name'] == 'Books']
        final_gstr2a = self.final_report_df[self.final_report_df['Source Name'] == 'GSTR-2A']

        # Exact Matches
        exact_matches_books = final_books[final_books['Status'] == 'Exact Match']
        exact_matches_gstr2a = final_gstr2a[final_gstr2a['Status'] == 'Exact Match']
        data.append({
            'Match Type': 'Exact Matches',
            'Books Count': len(exact_matches_books),
            'GSTR-2A Count': len(exact_matches_gstr2a),
            'Total IGST': exact_matches_books['Total IGST Amount'].sum(),
            'Total CGST': exact_matches_books['Total CGST Amount'].sum(),
            'Total SGST': exact_matches_books['Total SGST Amount'].sum(),
            'Total Value': exact_matches_books['Total Invoice Value'].sum()
        })

        # Partial Matches
        partial_matches_books = final_books[final_books['Status'] == 'Partial Match']
        partial_matches_gstr2a = final_gstr2a[final_gstr2a['Status'] == 'Partial Match']
        data.append({
            'Match Type': 'Partial Matches',
            'Books Count': len(partial_matches_books),
            'GSTR-2A Count': len(partial_matches_gstr2a),
            'Total IGST': partial_matches_books['Total IGST Amount'].sum(),
            'Total CGST': partial_matches_books['Total CGST Amount'].sum(),
            'Total SGST': partial_matches_books['Total SGST Amount'].sum(),
            'Total Value': partial_matches_books['Total Invoice Value'].sum()
        })

        # Group Matches
        group_matches_books = final_books[final_books['Status'] == 'Group Match']
        group_matches_gstr2a = final_gstr2a[final_gstr2a['Status'] == 'Group Match']
        data.append({
            'Match Type': 'Group Matches',
            'Books Count': len(group_matches_books),
            'GSTR-2A Count': len(group_matches_gstr2a),
            'Total IGST': group_matches_books['Total IGST Amount'].sum(),
            'Total CGST': group_matches_books['Total CGST Amount'].sum(),
            'Total SGST': group_matches_books['Total SGST Amount'].sum(),
            'Total Value': group_matches_books['Total Invoice Value'].sum()
        })

        # Data Entry Swap Matches
        data_entry_swap_books = final_books[final_books['Status'] == 'Data Entry Swap Match']
        data_entry_swap_gstr2a = final_gstr2a[final_gstr2a['Status'] == 'Data Entry Swap Match']
        data.append({
            'Match Type': 'Data Entry Swap Matches',
            'Books Count': len(data_entry_swap_books),
            'GSTR-2A Count': len(data_entry_swap_gstr2a),
            'Total IGST': data_entry_swap_books['Total IGST Amount'].sum(),
            'Total CGST': data_entry_swap_books['Total CGST Amount'].sum(),
            'Total SGST': data_entry_swap_books['Total SGST Amount'].sum(),
            'Total Value': data_entry_swap_books['Total Invoice Value'].sum()
        })

        # Tax-Based Group Matches
        tax_based_group_books = final_books[final_books['Status'] == 'Tax-Based Group Match']
        tax_based_group_gstr2a = final_gstr2a[final_gstr2a['Status'] == 'Tax-Based Group Match']
        data.append({
            'Match Type': 'Tax-Based Group Matches',
            'Books Count': len(tax_based_group_books),
            'GSTR-2A Count': len(tax_based_group_gstr2a),
            'Total IGST': tax_based_group_books['Total IGST Amount'].sum(),
            'Total CGST': tax_based_group_books['Total CGST Amount'].sum(),
            'Total SGST': tax_based_group_books['Total SGST Amount'].sum(),
            'Total Value': tax_based_group_books['Total Invoice Value'].sum()
        })

        # Books Only
        books_only_df = final_books[final_books['Status'] == 'Books Only']
        data.append({
            'Match Type': 'Books Only',
            'Books Count': len(books_only_df),
            'GSTR-2A Count': 0,
            'Total IGST': books_only_df['Total IGST Amount'].sum(),
            'Total CGST': books_only_df['Total CGST Amount'].sum(),
            'Total SGST': books_only_df['Total SGST Amount'].sum(),
            'Total Value': books_only_df['Total Invoice Value'].sum()
        })

        # GSTR-2A Only
        gstr2a_only_df = final_gstr2a[final_gstr2a['Status'] == 'GSTR-2A Only']
        data.append({
            'Match Type': 'GSTR-2A Only',
            'Books Count': 0,
            'GSTR-2A Count': len(gstr2a_only_df),
            'Total IGST': gstr2a_only_df['Total IGST Amount'].sum(),
            'Total CGST': gstr2a_only_df['Total CGST Amount'].sum(),
            'Total SGST': gstr2a_only_df['Total SGST Amount'].sum(),
            'Total Value': gstr2a_only_df['Total Invoice Value'].sum()
        })

        # Sign Cancellations (if any)
        sign_cancellation_books = final_books[final_books['Status'].str.contains('Sign Cancellation', na=False)]
        sign_cancellation_gstr2a = final_gstr2a[final_gstr2a['Status'].str.contains('Sign Cancellation', na=False)]
        if len(sign_cancellation_books) > 0 or len(sign_cancellation_gstr2a) > 0:
            data.append({
                'Match Type': 'Sign Cancellations',
                'Books Count': len(sign_cancellation_books),
                'GSTR-2A Count': len(sign_cancellation_gstr2a),
                'Total IGST': sign_cancellation_books['Total IGST Amount'].sum() + sign_cancellation_gstr2a['Total IGST Amount'].sum(),
                'Total CGST': sign_cancellation_books['Total CGST Amount'].sum() + sign_cancellation_gstr2a['Total CGST Amount'].sum(),
                'Total SGST': sign_cancellation_books['Total SGST Amount'].sum() + sign_cancellation_gstr2a['Total SGST Amount'].sum(),
                'Total Value': sign_cancellation_books['Total Invoice Value'].sum() + sign_cancellation_gstr2a['Total Invoice Value'].sum()
            })

        # Calculate totals from individual status counts to ensure accuracy
        total_books_count_recon = sum([
            len(exact_matches_books),
            len(partial_matches_books),
            len(group_matches_books),
            len(data_entry_swap_books),
            len(tax_based_group_books),
            len(books_only_df),
            len(sign_cancellation_books)
        ])
        
        total_gstr2a_count_recon = sum([
            len(exact_matches_gstr2a),
            len(partial_matches_gstr2a),
            len(group_matches_gstr2a),
            len(data_entry_swap_gstr2a),
            len(tax_based_group_gstr2a),
            len(gstr2a_only_df),
            len(sign_cancellation_gstr2a)
        ])

        total_igst_recon = final_books['Total IGST Amount'].sum()
        total_cgst_recon = final_books['Total CGST Amount'].sum()
        total_sgst_recon = final_books['Total SGST Amount'].sum()
        total_value_recon = final_books['Total Invoice Value'].sum()

        data.append({
            'Match Type': 'Total',
            'Books Count': total_books_count_recon,
            'GSTR-2A Count': total_gstr2a_count_recon,
            'Total IGST': total_igst_recon,
            'Total CGST': total_cgst_recon,
            'Total SGST': total_sgst_recon,
            'Total Value': total_value_recon
        })

        return pd.DataFrame(data)

    def _perform_integrity_checks(self):
        """Perform data integrity checks and summarize issues."""
        integrity_issues = []

        # Check for duplicate Invoice Numbers within Books
        duplicate_books_invoices = self.books_df[self.books_df.duplicated(subset=['Invoice Number'], keep=False)]
        if not duplicate_books_invoices.empty:
            issue_details = """Duplicate Invoice Numbers found in Books:\n"""
            for gstin, inv_nums in duplicate_books_invoices.groupby('Supplier GSTIN')['Invoice Number'].apply(lambda x: x.tolist()).items():
                issue_details += f"""  GSTIN {gstin}: {set(inv_nums)}\n"""
            integrity_issues.append({
                'Check': 'Duplicate Invoice Numbers (Books)',
                'Status': 'Warning',
                'Details': issue_details.strip()
            })

        # Check for duplicate Invoice Numbers within GSTR-2A
        duplicate_gstr2a_invoices = self.gstr2a_df[self.gstr2a_df.duplicated(subset=['Invoice Number'], keep=False)]
        if not duplicate_gstr2a_invoices.empty:
            issue_details = """Duplicate Invoice Numbers found in GSTR-2A:\n"""
            for gstin, inv_nums in duplicate_gstr2a_invoices.groupby('Supplier GSTIN')['Invoice Number'].apply(lambda x: x.tolist()).items():
                issue_details += f"""  GSTIN {gstin}: {set(inv_nums)}\n"""
            integrity_issues.append({
                'Check': 'Duplicate Invoice Numbers (GSTR-2A)',
                'Status': 'Warning',
                'Details': issue_details.strip()
            })

        # Check for missing GSTINs
        missing_gstin_books = self.books_df[self.books_df['Supplier GSTIN'].isna() | (self.books_df['Supplier GSTIN'] == '')]
        if not missing_gstin_books.empty:
            integrity_issues.append({
                'Check': 'Missing Supplier GSTIN (Books)',
                'Status': 'Error',
                'Details': f"{len(missing_gstin_books)} records in Books have missing GSTINs."
            })

        missing_gstin_gstr2a = self.gstr2a_df[self.gstr2a_df['Supplier GSTIN'].isna() | (self.gstr2a_df['Supplier GSTIN'] == '')]
        if not missing_gstin_gstr2a.empty:
            integrity_issues.append({
                'Check': 'Missing Supplier GSTIN (GSTR-2A)',
                'Status': 'Error',
                'Details': f"{len(missing_gstin_gstr2a)} records in GSTR-2A have missing GSTINs."
            })

        # Check for missing Invoice Dates
        missing_invoice_date_books = self.books_df[self.books_df['Invoice Date'].isna()]
        if not missing_invoice_date_books.empty:
            integrity_issues.append({
                'Check': 'Missing Invoice Date (Books)',
                'Status': 'Error',
                'Details': f"{len(missing_invoice_date_books)} records in Books have missing Invoice Dates."
            })

        missing_invoice_date_gstr2a = self.gstr2a_df[self.gstr2a_df['Invoice Date'].isna()]
        if not missing_invoice_date_gstr2a.empty:
            integrity_issues.append({
                'Check': 'Missing Invoice Date (GSTR-2A)',
                'Status': 'Error',
                'Details': f"{len(missing_invoice_date_gstr2a)} records in GSTR-2A have missing Invoice Dates."
            })

        # Check for missing Books Date (only relevant for Books records)
        missing_books_date = self.books_df[self.books_df['Books Date'].isna()]
        if not missing_books_date.empty:
            integrity_issues.append({
                'Check': 'Missing Books Date (Books)',
                'Status': 'Error',
                'Details': f"{len(missing_books_date)} records in Books have missing Books Dates."
            })

        # Check for negative amounts in numeric columns (optional, but good practice)
        numeric_cols_to_check = [
            'Total Taxable Value', 'Total Tax Value', 'Total IGST Amount',
            'Total CGST Amount', 'Total SGST Amount', 'Total Invoice Value'
        ]
        for col in numeric_cols_to_check:
            negative_amounts_books = self.books_df[self.books_df[col] < 0]
            if not negative_amounts_books.empty:
                integrity_issues.append({
                    'Check': f'Negative {col} (Books)',
                    'Status': 'Warning',
                    'Details': f'{len(negative_amounts_books)} records in Books have negative {col}.'
                })
            negative_amounts_gstr2a = self.gstr2a_df[self.gstr2a_df[col] < 0]
            if not negative_amounts_gstr2a.empty:
                integrity_issues.append({
                    'Check': f'Negative {col} (GSTR-2A)',
                    'Status': 'Warning',
                    'Details': f'{len(negative_amounts_gstr2a)} records in GSTR-2A have negative {col}.'
                })

        if not integrity_issues:
            integrity_issues.append({
                'Check': 'All checks passed',
                'Status': 'Success',
                'Details': 'No data integrity issues found.'
            })

        return pd.DataFrame(integrity_issues)

    def get_summaries(self):
        """Get all summary information."""
        # Always regenerate final_report_df to ensure it's up to date
        self._generate_final_report()
        
        # Generate all summaries using the updated final_report_df
        return {
            'raw_summary': self._generate_raw_summary(),
            'recon_summary': self._generate_recon_summary(),
            'integrity_checks': self._perform_integrity_checks(),
            'tax_summary': self._generate_tax_summary(),
            'exact_matches_summary': self._generate_exact_matches_summary(),
            'partial_matches_summary': self._generate_partial_matches_summary(),
            'group_matches_summary': self._generate_group_matches_summary(),
            'cross_year_matches_summary': self._generate_cross_year_matches_summary(),
            'status_summary': self._generate_status_summary()
        }

    def _generate_tax_summary(self):
        """Generate a summary table of tax amounts by match type."""
        # Initialize summary dictionary with tax types as rows
        summary = {
            'Tax Type': ['IGST', 'CGST', 'SGST'],
            'Exact Match': [0, 0, 0],
            'Partial Match': [0, 0, 0],
            'Group Match': [0, 0, 0],
            'Data Entry Swap Match': [0, 0, 0],
            'Tax-Based Group Match': [0, 0, 0],
            'Books Only': [0, 0, 0],
            'GSTR-2A Only': [0, 0, 0]
        }
        
        # Process all records from final_report_df
        for _, row in self.final_report_df.iterrows():
            match_type = row['Status']
            if match_type in ['Exact Match', 'Partial Match', 'Group Match', 'Data Entry Swap Match', 'Tax-Based Group Match', 'Books Only', 'GSTR-2A Only']:
                # Find the index for the tax type
                tax_type_idx = 0  # Default to IGST
                if row['Total CGST Amount'] > 0:
                    tax_type_idx = 1
                elif row['Total SGST Amount'] > 0:
                    tax_type_idx = 2
                
                # Add the amounts based on match type
                if match_type == 'Exact Match':
                    summary['Exact Match'][tax_type_idx] += row['Total IGST Amount'] if tax_type_idx == 0 else (row['Total CGST Amount'] if tax_type_idx == 1 else row['Total SGST Amount'])
                elif match_type == 'Partial Match':
                    summary['Partial Match'][tax_type_idx] += row['Total IGST Amount'] if tax_type_idx == 0 else (row['Total CGST Amount'] if tax_type_idx == 1 else row['Total SGST Amount'])
                elif match_type == 'Group Match':
                    summary['Group Match'][tax_type_idx] += row['Total IGST Amount'] if tax_type_idx == 0 else (row['Total CGST Amount'] if tax_type_idx == 1 else row['Total SGST Amount'])
                elif match_type == 'Data Entry Swap Match':
                    summary['Data Entry Swap Match'][tax_type_idx] += row['Total IGST Amount'] if tax_type_idx == 0 else (row['Total CGST Amount'] if tax_type_idx == 1 else row['Total SGST Amount'])
                elif match_type == 'Tax-Based Group Match':
                    summary['Tax-Based Group Match'][tax_type_idx] += row['Total IGST Amount'] if tax_type_idx == 0 else (row['Total CGST Amount'] if tax_type_idx == 1 else row['Total SGST Amount'])
                elif match_type == 'Books Only':
                    summary['Books Only'][tax_type_idx] += row['Total IGST Amount'] if tax_type_idx == 0 else (row['Total CGST Amount'] if tax_type_idx == 1 else row['Total SGST Amount'])
                elif match_type == 'GSTR-2A Only':
                    summary['GSTR-2A Only'][tax_type_idx] += row['Total IGST Amount'] if tax_type_idx == 0 else (row['Total CGST Amount'] if tax_type_idx == 1 else row['Total SGST Amount'])
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary)
        
        # Add total column
        summary_df['Total'] = summary_df.iloc[:, 1:].sum(axis=1)
        
        # Debug: Print tax summary
        print("\nTax Summary:")
        print(summary_df)
        
        return summary_df

    def _generate_exact_matches_summary(self):
        """Generates a detailed summary table for exact matches."""
        exact_matches_df = self.final_report_df[self.final_report_df['Status'] == 'Exact Match']

        books_exact = exact_matches_df[exact_matches_df['Source Name'] == 'Books']
        gstr2a_exact = exact_matches_df[exact_matches_df['Source Name'] == 'GSTR-2A']

        data = []

        # Books row
        data.append({
            'Source': 'Books',
            'IGST': books_exact['Total IGST Amount'].sum(),
            'CGST': books_exact['Total CGST Amount'].sum(),
            'SGST': books_exact['Total SGST Amount'].sum()
        })

        # GSTR-2A row
        data.append({
            'Source': 'GSTR-2A',
            'IGST': gstr2a_exact['Total IGST Amount'].sum(),
            'CGST': gstr2a_exact['Total CGST Amount'].sum(),
            'SGST': gstr2a_exact['Total SGST Amount'].sum()
        })

        # Net Off (Difference) row
        net_off_igst = books_exact['Total IGST Amount'].sum() - gstr2a_exact['Total IGST Amount'].sum()
        net_off_cgst = books_exact['Total CGST Amount'].sum() - gstr2a_exact['Total CGST Amount'].sum()
        net_off_sgst = books_exact['Total SGST Amount'].sum() - gstr2a_exact['Total SGST Amount'].sum()

        data.append({
            'Source': 'Net Off',
            'IGST': net_off_igst,
            'CGST': net_off_cgst,
            'SGST': net_off_sgst
        })

        return pd.DataFrame(data)

    def _generate_partial_matches_summary(self):
        """Generates a detailed summary table for partial matches."""
        partial_matches_df = self.final_report_df[self.final_report_df['Status'] == 'Partial Match']

        books_partial = partial_matches_df[partial_matches_df['Source Name'] == 'Books']
        gstr2a_partial = partial_matches_df[partial_matches_df['Source Name'] == 'GSTR-2A']

        data = []

        # Books row
        data.append({
            'Source': 'Books',
            'IGST': books_partial['Total IGST Amount'].sum(),
            'CGST': books_partial['Total CGST Amount'].sum(),
            'SGST': books_partial['Total SGST Amount'].sum()
        })

        # GSTR-2A row
        data.append({
            'Source': 'GSTR-2A',
            'IGST': gstr2a_partial['Total IGST Amount'].sum(),
            'CGST': gstr2a_partial['Total CGST Amount'].sum(),
            'SGST': gstr2a_partial['Total SGST Amount'].sum()
        })

        # Net Off (Difference) row
        net_off_igst = books_partial['Total IGST Amount'].sum() - gstr2a_partial['Total IGST Amount'].sum()
        net_off_cgst = books_partial['Total CGST Amount'].sum() - gstr2a_partial['Total CGST Amount'].sum()
        net_off_sgst = books_partial['Total SGST Amount'].sum() - gstr2a_partial['Total SGST Amount'].sum()

        data.append({
            'Source': 'Net Off',
            'IGST': net_off_igst,
            'CGST': net_off_cgst,
            'SGST': net_off_sgst
        })

        return pd.DataFrame(data)

    def _generate_group_matches_summary(self):
        """Generates a detailed summary table for group matches."""
        # Use final_report_df instead of just matched_df to include all group matches
        group_matches_df = self.final_report_df[self.final_report_df['Status'] == 'Group Match']

        books_group = group_matches_df[group_matches_df['Source Name'] == 'Books']
        gstr2a_group = group_matches_df[group_matches_df['Source Name'] == 'GSTR-2A']

        data = []

        # Books row
        data.append({
            'Source': 'Books',
            'IGST': books_group['Total IGST Amount'].sum(),
            'CGST': books_group['Total CGST Amount'].sum(),
            'SGST': books_group['Total SGST Amount'].sum()
        })

        # GSTR-2A row
        data.append({
            'Source': 'GSTR-2A',
            'IGST': gstr2a_group['Total IGST Amount'].sum(),
            'CGST': gstr2a_group['Total CGST Amount'].sum(),
            'SGST': gstr2a_group['Total SGST Amount'].sum()
        })

        # Net Off (Difference) row
        net_off_igst = books_group['Total IGST Amount'].sum() - gstr2a_group['Total IGST Amount'].sum()
        net_off_cgst = books_group['Total CGST Amount'].sum() - gstr2a_group['Total CGST Amount'].sum()
        net_off_sgst = books_group['Total SGST Amount'].sum() - gstr2a_group['Total SGST Amount'].sum()

        data.append({
            'Source': 'Net Off',
            'IGST': net_off_igst,
            'CGST': net_off_cgst,
            'SGST': net_off_sgst
        })

        return pd.DataFrame(data)

    def _generate_cross_year_matches_summary(self):
        """Generates a detailed summary table for cross-year matches."""
        cross_year_statuses = ['Cross-Year Match', 'Partial Cross-Year Match']
        cross_year_matches_df = self.final_report_df[self.final_report_df['Status'].isin(cross_year_statuses)]

        books_cross = cross_year_matches_df[cross_year_matches_df['Source Name'] == 'Books']
        gstr2a_cross = cross_year_matches_df[cross_year_matches_df['Source Name'] == 'GSTR-2A']

        data = []

        # Books row
        data.append({
            'Source': 'Books',
            'IGST': books_cross['Total IGST Amount'].sum(),
            'CGST': books_cross['Total CGST Amount'].sum(),
            'SGST': books_cross['Total SGST Amount'].sum()
        })

        # GSTR-2A row
        data.append({
            'Source': 'GSTR-2A',
            'IGST': gstr2a_cross['Total IGST Amount'].sum(),
            'CGST': gstr2a_cross['Total CGST Amount'].sum(),
            'SGST': gstr2a_cross['Total SGST Amount'].sum()
        })

        # Net Off (Difference) row
        net_off_igst = books_cross['Total IGST Amount'].sum() - gstr2a_cross['Total IGST Amount'].sum()
        net_off_cgst = books_cross['Total CGST Amount'].sum() - gstr2a_cross['Total CGST Amount'].sum()
        net_off_sgst = books_cross['Total SGST Amount'].sum() - gstr2a_cross['Total SGST Amount'].sum()

        data.append({
            'Source': 'Net Off',
            'IGST': net_off_igst,
            'CGST': net_off_cgst,
            'SGST': net_off_sgst
        })

        return pd.DataFrame(data)

    def _generate_status_summary(self):
        """Generate a comprehensive status-wise summary of all reconciliation results."""
        # Use final_report_df for all status calculations to ensure consistency
        if self.final_report_df.empty:
            return pd.DataFrame()
        
        # Get all unique statuses from final_report_df
        all_statuses = self.final_report_df['Status'].unique()
        
        data = []
        
        for status in sorted(all_statuses):
            if pd.isna(status) or status == '':
                continue
                
            # Get records with this status
            status_records = self.final_report_df[self.final_report_df['Status'] == status]
            
            # Count records by source
            books_count = len(status_records[status_records['Source Name'] == 'Books'])
            gstr2a_count = len(status_records[status_records['Source Name'] == 'GSTR-2A'])
            total_count = len(status_records)
            
            # Calculate values by source
            books_value = status_records[status_records['Source Name'] == 'Books']['Total Invoice Value'].sum()
            gstr2a_value = status_records[status_records['Source Name'] == 'GSTR-2A']['Total Invoice Value'].sum()
            total_value = status_records['Total Invoice Value'].sum()
            
            # Calculate tax amounts
            total_igst = status_records['Total IGST Amount'].sum()
            total_cgst = status_records['Total CGST Amount'].sum()
            total_sgst = status_records['Total SGST Amount'].sum()
            
            data.append({
                'Status': status,
                'Matched Records': books_count,  # Books records are considered "matched" in the context
                'Mismatch Records': gstr2a_count,  # GSTR-2A records are considered "mismatch" in the context
                'Total Records': total_count,
                'Matched Value': books_value,
                'Mismatch Value': gstr2a_value,
                'Total Value': total_value,
                'Total IGST': total_igst,
                'Total CGST': total_cgst,
                'Total SGST': total_sgst
            })
        
        # Add summary row
        total_records = sum(row['Total Records'] for row in data)
        total_value = sum(row['Total Value'] for row in data)
        total_igst = sum(row['Total IGST'] for row in data)
        total_cgst = sum(row['Total CGST'] for row in data)
        total_sgst = sum(row['Total SGST'] for row in data)
        
        data.append({
            'Status': 'TOTAL',
            'Matched Records': sum(row['Matched Records'] for row in data),
            'Mismatch Records': sum(row['Mismatch Records'] for row in data),
            'Total Records': total_records,
            'Matched Value': sum(row['Matched Value'] for row in data),
            'Mismatch Value': sum(row['Mismatch Value'] for row in data),
            'Total Value': total_value,
            'Total IGST': total_igst,
            'Total CGST': total_cgst,
            'Total SGST': total_sgst
        })
        
        return pd.DataFrame(data)

    def analyze_matching_patterns(self):
        """Analyze the data and suggest intelligent matching patterns based on the situation."""
        analysis = {}
        
        # Analyze Books Only records
        books_only = self.mismatch_df[(self.mismatch_df['Source Name'] == 'Books') & (self.mismatch_df['Status'] == 'Books Only')]
        gstr2a_only = self.mismatch_df[(self.mismatch_df['Source Name'] == 'GSTR-2A') & (self.mismatch_df['Status'] == 'GSTR-2A Only')]
        partials_with_diff = self.matched_df[
            (self.matched_df['Status'] == 'Partial Match') & 
            (self.matched_df['Tax Diff Status'] == 'Has Difference')
        ]
        
        # Pattern 1: GSTIN + Invoice Number grouping
        books_gstin_inv_groups = books_only.groupby(['Supplier GSTIN', 'Invoice Number']).size()
        gstr2a_gstin_inv_groups = gstr2a_only.groupby(['Supplier GSTIN', 'Invoice Number']).size()
        
        # Pattern 2: Tax amount matching
        books_tax_ranges = {
            'small': books_only[books_only['Total Invoice Value'] <= 10000],
            'medium': books_only[(books_only['Total Invoice Value'] > 10000) & (books_only['Total Invoice Value'] <= 100000)],
            'large': books_only[books_only['Total Invoice Value'] > 100000]
        }
        
        gstr2a_tax_ranges = {
            'small': gstr2a_only[gstr2a_only['Total Invoice Value'] <= 10000],
            'medium': gstr2a_only[(gstr2a_only['Total Invoice Value'] > 10000) & (gstr2a_only['Total Invoice Value'] <= 100000)],
            'large': gstr2a_only[gstr2a_only['Total Invoice Value'] > 100000]
        }
        
        # Pattern 3: Date-based matching
        books_date_groups = books_only.groupby(books_only['Invoice Date'].dt.to_period('M')).size()
        gstr2a_date_groups = gstr2a_only.groupby(gstr2a_only['Invoice Date'].dt.to_period('M')).size()
        
        # Pattern 4: Supplier name similarity
        books_suppliers = books_only['Supplier Trade Name'].value_counts()
        gstr2a_suppliers = gstr2a_only['Supplier Trade Name'].value_counts()
        
        # Generate recommendations
        recommendations = []
        
        # Recommendation 1: GSTIN + Invoice Number grouping
        potential_gstin_inv_matches = set(books_gstin_inv_groups.index) & set(gstr2a_gstin_inv_groups.index)
        if len(potential_gstin_inv_matches) > 0:
            recommendations.append({
                'pattern': 'GSTIN + Invoice Number Grouping',
                'description': f'Found {len(potential_gstin_inv_matches)} potential group matches by GSTIN and Invoice Number',
                'priority': 'High',
                'estimated_matches': len(potential_gstin_inv_matches)
            })
        
        # Recommendation 2: Tax amount range matching
        for range_name in ['small', 'medium', 'large']:
            books_count = len(books_tax_ranges[range_name])
            gstr2a_count = len(gstr2a_tax_ranges[range_name])
            if books_count > 0 and gstr2a_count > 0:
                recommendations.append({
                    'pattern': f'Tax Amount Range Matching ({range_name})',
                    'description': f'{books_count} Books vs {gstr2a_count} GSTR-2A records in {range_name} range',
                    'priority': 'Medium',
                    'estimated_matches': min(books_count, gstr2a_count)
                })
        
        # Recommendation 3: Date-based matching
        common_months = set(books_date_groups.index) & set(gstr2a_date_groups.index)
        if len(common_months) > 0:
            recommendations.append({
                'pattern': 'Date-based Grouping',
                'description': f'Found {len(common_months)} months with records in both sources',
                'priority': 'Medium',
                'estimated_matches': len(common_months)
            })
        
        # Recommendation 4: Supplier name similarity
        common_suppliers = set(books_suppliers.index) & set(gstr2a_suppliers.index)
        if len(common_suppliers) > 0:
            recommendations.append({
                'pattern': 'Supplier Name Similarity',
                'description': f'Found {len(common_suppliers)} suppliers with records in both sources',
                'priority': 'Low',
                'estimated_matches': len(common_suppliers)
            })
        
        analysis = {
            'books_only_count': len(books_only),
            'gstr2a_only_count': len(gstr2a_only),
            'partials_with_diff_count': len(partials_with_diff),
            'potential_gstin_inv_groups': len(potential_gstin_inv_matches),
            'recommendations': recommendations,
            'tax_distribution': {
                'books': {k: len(v) for k, v in books_tax_ranges.items()},
                'gstr2a': {k: len(v) for k, v in gstr2a_tax_ranges.items()}
            }
        }
        
        return analysis

    def run_intelligent_enhanced_matching(self):
        """Run enhanced matching with intelligent pattern analysis."""
        print("Analyzing matching patterns...")
        analysis = self.analyze_matching_patterns()
        
        print(f"\nEnhanced Matching Analysis:")
        print(f"  Books Only records: {analysis['books_only_count']}")
        print(f"  GSTR-2A Only records: {analysis['gstr2a_only_count']}")
        print(f"  Partial matches with tax differences: {analysis['partials_with_diff_count']}")
        print(f"  Potential GSTIN+Invoice groups: {analysis['potential_gstin_inv_groups']}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec['pattern']} ({rec['priority']} priority)")
            print(f"     {rec['description']}")
            print(f"     Estimated matches: {rec['estimated_matches']}")
        
        # Run the enhanced group matching
        print(f"\nExecuting enhanced group matching...")
        self.run_enhanced_group_matching()
        
        return analysis

    def run_enhanced_group_matching(self):
        """Enhanced group/duplication-aware matching for Books Only, GSTR-2A Only, and partials with tax differences beyond tolerance."""
        start_time = time.time()
        books_only = self.mismatch_df[(self.mismatch_df['Source Name'] == 'Books') & (self.mismatch_df['Status'] == 'Books Only')].copy()
        gstr2a_only = self.mismatch_df[(self.mismatch_df['Source Name'] == 'GSTR-2A') & (self.mismatch_df['Status'] == 'GSTR-2A Only')].copy()
        partials = self.matched_df[(self.matched_df['Status'] == 'Partial Match') & (self.matched_df['Tax Diff Status'] == 'Has Difference')].copy()
        print(f"Enhanced reconciliation targets:")
        print(f"  Books Only records: {len(books_only)}")
        print(f"  GSTR-2A Only records: {len(gstr2a_only)}")
        print(f"  Partial matches with tax differences: {len(partials)}")
        books_groups = books_only.groupby(['Supplier GSTIN', 'Invoice Number'])
        gstr2a_groups = gstr2a_only.groupby(['Supplier GSTIN', 'Invoice Number'])
        partials_groups = partials.groupby(['Supplier GSTIN', 'Invoice Number'])
        def extract_int_id(val):
            try:
                return int(str(val))
            except:
                return 0
        max_match_id = 0
        max_group_id = 0
        if 'Match ID' in self.matched_df.columns:
            max_match_id = max([extract_int_id(x) for x in self.matched_df['Match ID'].dropna().unique()] + [0])
        if 'Match ID' in self.mismatch_df.columns:
            max_match_id = max(max_match_id, max([extract_int_id(x) for x in self.mismatch_df['Match ID'].dropna().unique()] + [0]))
        if 'Group ID' in self.matched_df.columns:
            max_group_id = max([extract_int_id(x) for x in self.matched_df['Group ID'].dropna().unique()] + [0])
        if 'Group ID' in self.mismatch_df.columns:
            max_group_id = max(max_group_id, max([extract_int_id(x) for x in self.mismatch_df['Group ID'].dropna().unique()] + [0]))
        next_id = max(max_match_id, max_group_id) + 1
        before_count = len(self.matched_df) + len(self.mismatch_df)
        enhanced_matches_found = 0
        
        # Reset group counters for enhanced matching
        enhanced_group_count = 0
        enhanced_group_value = 0
        
        # --- Optimization: accumulate updates in lists ---
        mismatch_updates = []
        matched_updates = []
        for key in set(books_groups.groups.keys()).union(gstr2a_groups.groups.keys()).union(partials_groups.groups.keys()):
            books_group = books_groups.get_group(key) if key in books_groups.groups else pd.DataFrame()
            gstr2a_group = gstr2a_groups.get_group(key) if key in gstr2a_groups.groups else pd.DataFrame()
            partials_group = partials_groups.get_group(key) if key in partials_groups.groups else pd.DataFrame()
            partials_books_count = 0
            partials_gstr2a_count = 0
            if 'Source Name' in partials_group.columns:
                partials_books_count = len(partials_group[partials_group['Source Name'] == 'Books'])
                partials_gstr2a_count = len(partials_group[partials_group['Source Name'] == 'GSTR-2A'])
            if (len(books_group) + partials_books_count <= 1 and len(gstr2a_group) + partials_gstr2a_count <= 1):
                continue
            books_tax = books_group[['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']].sum() if not books_group.empty else pd.Series([0,0,0], index=['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount'])
            gstr2a_tax = gstr2a_group[['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']].sum() if not gstr2a_group.empty else pd.Series([0,0,0], index=['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount'])
            if 'Source Name' in partials_group.columns:
                partials_books = partials_group[partials_group['Source Name'] == 'Books']
                partials_gstr2a = partials_group[partials_group['Source Name'] == 'GSTR-2A']
            else:
                partials_books = pd.DataFrame()
                partials_gstr2a = pd.DataFrame()
            partials_books_tax = partials_books[['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']].sum() if not partials_books.empty else pd.Series([0,0,0], index=['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount'])
            partials_gstr2a_tax = partials_gstr2a[['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']].sum() if not partials_gstr2a.empty else pd.Series([0,0,0], index=['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount'])
            total_books_tax = books_tax + partials_books_tax
            total_gstr2a_tax = gstr2a_tax + partials_gstr2a_tax
            if (abs(total_books_tax['Total IGST Amount'] - total_gstr2a_tax['Total IGST Amount']) <= self.group_tax_tolerance and abs(total_books_tax['Total CGST Amount'] - total_gstr2a_tax['Total CGST Amount']) <= self.group_tax_tolerance and abs(total_books_tax['Total SGST Amount'] - total_gstr2a_tax['Total SGST Amount']) <= self.group_tax_tolerance):
                group_id = str(next_id)
                match_id = str(next_id)
                next_id += 1
                enhanced_matches_found += 1
                enhanced_group_count += 1
                enhanced_group_value += total_books_tax['Total IGST Amount'] + total_books_tax['Total CGST Amount'] + total_books_tax['Total SGST Amount']
                
                igst_diff = total_books_tax['Total IGST Amount'] - total_gstr2a_tax['Total IGST Amount']
                cgst_diff = total_books_tax['Total CGST Amount'] - total_gstr2a_tax['Total CGST Amount']
                sgst_diff = total_books_tax['Total SGST Amount'] - total_gstr2a_tax['Total SGST Amount']
                if abs(igst_diff) <= self.group_tax_tolerance and abs(cgst_diff) <= self.group_tax_tolerance and abs(sgst_diff) <= self.group_tax_tolerance:
                    tax_diff_status = 'No Difference'
                else:
                    tax_diff_status = 'Has Difference'
                narration = f"Enhanced group match. IGST Diff: {igst_diff:.2f}, CGST Diff: {cgst_diff:.2f}, SGST Diff: {sgst_diff:.2f}"
                suggestions = 'Matched as group (enhanced) - tax differences resolved'
                # Generate field status columns for enhanced group matching
                # Use first records from each group for comparison
                first_book = books_group.iloc[0] if len(books_group) > 0 else None
                first_gstr2a = gstr2a_group.iloc[0] if len(gstr2a_group) > 0 else None
                
                if first_book is not None and first_gstr2a is not None:
                    field_status = self._generate_field_status_columns(first_book, first_gstr2a)
                else:
                    # Default field status if we can't compare
                    field_status = {
                        'Tax Head Status': 'N/A',
                        'GSTIN Match Status': 'N/A',
                        'Invoice Match Status': 'N/A',
                        'Trade Name Match Status': 'N/A',
                        'Legal Name Match Status': 'N/A'
                    }
                
                # Create enhanced update data with new columns
                enhanced_update_data = {
                    'Status': 'Group Match', 
                    'Group ID': group_id, 
                    'Match ID': int(match_id), 
                    'IGST Diff': float(igst_diff), 
                    'CGST Diff': float(cgst_diff), 
                    'SGST Diff': float(sgst_diff), 
                    'Tax Diff Status': tax_diff_status, 
                    'Narrative': narration, 
                    'Suggestions': suggestions,
                    'Sub Status': f'Enhanced group match for GSTIN {key}'
                }
                
                # Add field status columns
                enhanced_update_data.update(field_status)
                
                # --- Batch update: accumulate dicts ---
                for idx in books_group.index:
                    mismatch_updates.append({'idx': idx, **enhanced_update_data})
                for idx in gstr2a_group.index:
                    mismatch_updates.append({'idx': idx, **enhanced_update_data})
                for idx in partials_books.index:
                    matched_updates.append({'idx': idx, **enhanced_update_data})
                for idx in partials_gstr2a.index:
                    matched_updates.append({'idx': idx, **enhanced_update_data})
        # --- Apply all updates in batch ---
        if mismatch_updates:
            mismatch_df_updates = pd.DataFrame(mismatch_updates).set_index('idx')
            for col in mismatch_df_updates.columns:
                if col != 'idx' and col in self.mismatch_df.columns:
                    self.mismatch_df.loc[mismatch_df_updates.index, col] = mismatch_df_updates[col]
        if matched_updates:
            matched_df_updates = pd.DataFrame(matched_updates).set_index('idx')
            for col in matched_df_updates.columns:
                if col != 'idx' and col in self.matched_df.columns:
                    self.matched_df.loc[matched_df_updates.index, col] = matched_df_updates[col]
        
        # Move records from mismatch_df to matched_df for enhanced group matches
        enhanced_group_records = self.mismatch_df[self.mismatch_df['Status'] == 'Group Match'].copy()
        if not enhanced_group_records.empty:
            # Add these records to matched_df
            self.matched_df = pd.concat([self.matched_df, enhanced_group_records], ignore_index=True)
            # Remove these records from mismatch_df
            self.mismatch_df = self.mismatch_df[self.mismatch_df['Status'] != 'Group Match'].copy()
            print(f"Moved {len(enhanced_group_records)} enhanced group match records from mismatch_df to matched_df")
        
        # --- Standardize Tax Diff Status ---
        for df in [self.matched_df, self.mismatch_df]:
            if 'Tax Diff Status' in df.columns:
                df['Tax Diff Status'] = df['Tax Diff Status'].replace({'No Diff': 'No Difference', 'No Diffrence': 'No Difference'})
        # --- Flag possible duplicates (unchanged logic) ---
        for group in [books_groups, gstr2a_groups, partials_groups]:
            for key, g in group:
                if len(g) > 1:
                    unique_taxes = g[['Total IGST Amount', 'Total CGST Amount', 'Total SGST Amount']].drop_duplicates()
                    if len(unique_taxes) == 1:
                        for idx in g.index:
                            if 'Possible Duplicate' in self.mismatch_df.columns and idx in self.mismatch_df.index:
                                self.mismatch_df.at[idx, 'Possible Duplicate'] = True
                            if 'Possible Duplicate' in self.matched_df.columns and idx in self.matched_df.index:
                                self.matched_df.at[idx, 'Possible Duplicate'] = True
        
        # Update group counters for enhanced matching
        self.group_count += enhanced_group_count
        self.group_value += enhanced_group_value
        
        # --- Update final report and summaries ---
        self._generate_final_report()
        self.recon_summary = self._generate_recon_summary()
        self.raw_summary = self._generate_raw_summary()
        self.integrity_checks = self._perform_integrity_checks()
        after_count = len(self.matched_df) + len(self.mismatch_df)
        print(f"Enhanced reconciliation: before row count = {before_count}, after row count = {after_count}")
        print(f"Enhanced matches found: {enhanced_matches_found}")
        print(f"Enhanced group count: {enhanced_group_count}")
        print(f"Enhanced group value: {enhanced_group_value:.2f}")
        print(f"run_enhanced_group_matching completed in {time.time() - start_time:.2f} seconds")
        print("\nStatus Summary after Enhanced Group Matching:")
        status_summary = self._generate_status_summary()
        print(status_summary.to_string(index=False))
        group_matches = self.final_report_df[self.final_report_df['Status'] == 'Group Match']
        if not group_matches.empty:
            print(f"\nGroup Matches found: {len(group_matches)} records")
            unique_groups = group_matches['Group ID'].nunique()
            print(f"Unique Groups: {unique_groups}")
        else:
            print("\nNo Group Matches found")

    def _process_missing_gstin_matches(self):
        """Match Books records with missing GSTIN to GSTR-2A using name, invoice, and tax similarity."""
        unprocessed_books = self.books_df[(self.books_df['processed'] == False) & ((self.books_df['Supplier GSTIN'].isna()) | (self.books_df['Supplier GSTIN'].str.strip() == ''))]
        unprocessed_gstr2a = self.gstr2a_df[self.gstr2a_df['processed'] == False]
        for idx, book_row in unprocessed_books.iterrows():
            book_sign = 1 if (book_row['Total IGST Amount'] + book_row['Total CGST Amount'] + book_row['Total SGST Amount']) >= 0 else -1
            # Get book_row financial year
            book_fy = None
            if pd.notna(book_row['Invoice Date']):
                book_fy = _get_financial_year(book_row['Invoice Date'])
            # Find GSTR-2A candidates with same sign and tax within tolerance
            candidates = unprocessed_gstr2a[
                (unprocessed_gstr2a['processed'] == False) &
                (abs(unprocessed_gstr2a['Total IGST Amount'] - book_row['Total IGST Amount']) <= self.tax_tolerance) &
                (abs(unprocessed_gstr2a['Total CGST Amount'] - book_row['Total CGST Amount']) <= self.tax_tolerance) &
                (abs(unprocessed_gstr2a['Total SGST Amount'] - book_row['Total SGST Amount']) <= self.tax_tolerance)
            ]
            # Add financial year column to candidates
            candidates = candidates.copy()
            if 'Invoice Date' in candidates.columns:
                candidates['Financial Year'] = candidates['Invoice Date'].apply(lambda x: _get_financial_year(x) if pd.notna(x) else None)
            # Try matching within same financial year first
            best_score = 0
            best_match = None
            for restrict_fy in [True, False]:
                for gidx, gstr2a_row in candidates.iterrows():
                    if restrict_fy and book_fy is not None:
                        gstr2a_fy = gstr2a_row.get('Financial Year', None)
                        if gstr2a_fy != book_fy:
                            continue
                    gstr2a_sign = 1 if (gstr2a_row['Total IGST Amount'] + gstr2a_row['Total CGST Amount'] + gstr2a_row['Total SGST Amount']) >= 0 else -1
                    if book_sign != gstr2a_sign:
                        continue
                    # Name similarity
                    legal_name_similarity = 0
                    if pd.notna(book_row['Supplier Legal Name']) and pd.notna(gstr2a_row['Supplier Legal Name']):
                        clean_book_legal = _clean_company_name(str(book_row['Supplier Legal Name']))
                        clean_gstr2a_legal = _clean_company_name(str(gstr2a_row['Supplier Legal Name']))
                        legal_name_similarity = fuzz.ratio(clean_book_legal, clean_gstr2a_legal)
                    trade_name_similarity = 0
                    if pd.notna(book_row['Supplier Trade Name']) and pd.notna(gstr2a_row['Supplier Trade Name']):
                        clean_book_trade = _clean_company_name(str(book_row['Supplier Trade Name']))
                        clean_gstr2a_trade = _clean_company_name(str(gstr2a_row['Supplier Trade Name']))
                        trade_name_similarity = fuzz.ratio(clean_book_trade, clean_gstr2a_trade)
                    # Invoice number similarity (normalized)
                    invoice_similarity = 0
                    if pd.notna(book_row['Invoice Number']) and pd.notna(gstr2a_row['Invoice Number']):
                        norm_book_inv = extract_core_invoice_number(str(book_row['Invoice Number']))
                        norm_gstr2a_inv = extract_core_invoice_number(str(gstr2a_row['Invoice Number']))
                        invoice_similarity = fuzz.ratio(norm_book_inv, norm_gstr2a_inv)
                    # Calculate tax differences
                    igst_diff = book_row['Total IGST Amount'] - gstr2a_row['Total IGST Amount']
                    cgst_diff = book_row['Total CGST Amount'] - gstr2a_row['Total CGST Amount']
                    sgst_diff = book_row['Total SGST Amount'] - gstr2a_row['Total SGST Amount']
                    total_tax_diff = abs(igst_diff) + abs(cgst_diff) + abs(sgst_diff)
                    # At least one name similarity >= 70 and normalized invoice similarity >= 60
                    if (max(legal_name_similarity, trade_name_similarity) >= 70) and (invoice_similarity >= 60):
                        # If tax diff is within tolerance, it's a Partial Match
                        if total_tax_diff <= self.tax_tolerance:
                            score = (max(legal_name_similarity, trade_name_similarity) * 0.5) + (invoice_similarity * 0.5)
                            if score > best_score:
                                best_score = score
                                best_match = (gidx, gstr2a_row, legal_name_similarity, trade_name_similarity, invoice_similarity, 'Partial Match', igst_diff, cgst_diff, sgst_diff)
                        # If tax diff is above tolerance, it's a Potential Match: Tax Deviation (no date restriction)
                        else:
                            score = (max(legal_name_similarity, trade_name_similarity) * 0.5) + (invoice_similarity * 0.5)
                            if score > best_score:
                                best_score = score
                                best_match = (gidx, gstr2a_row, legal_name_similarity, trade_name_similarity, invoice_similarity, 'Potential Match: Tax Deviation', igst_diff, cgst_diff, sgst_diff)
                if best_match:
                    break
            if best_match:
                gidx, gstr2a_row, legal_name_similarity, trade_name_similarity, invoice_similarity, match_status, igst_diff, cgst_diff, sgst_diff = best_match
                self.books_df.at[idx, 'processed'] = True
                self.gstr2a_df.at[gidx, 'processed'] = True
                date_diff = np.nan
                if pd.notna(book_row['Invoice Date']) and pd.notna(gstr2a_row['Invoice Date']):
                    date_diff = (book_row['Invoice Date'] - gstr2a_row['Invoice Date']).days
                match_id = self.match_id_counter
                self.match_id_counter += 1
                # Generate sub status for group matching
                group_sub_status = self._generate_group_sub_status(
                    match_status, legal_name_similarity, trade_name_similarity, 
                    invoice_similarity, igst_diff, cgst_diff, sgst_diff
                )
                
                # Generate field status columns
                field_status = self._generate_field_status_columns(book_row, gstr2a_row)
                
                # Add Books record
                book_match_data = book_row.to_dict()
                book_match_data.update({
                    'Match ID': match_id,
                    'Source Name': 'Books',
                    'IGST Diff': igst_diff,
                    'CGST Diff': cgst_diff,
                    'SGST Diff': sgst_diff,
                    'Date Diff': date_diff,
                    'Status': match_status,
                    'Sub Status': group_sub_status,
                    'Tax Diff Status': 'No Difference' if abs(igst_diff + cgst_diff + sgst_diff) <= self.tax_tolerance else 'High Deviation',
                    'Date Status': 'N/A' if match_status == 'Potential Match: Tax Deviation' else ('Within Tolerance' if (pd.notna(date_diff) and abs(date_diff) <= self.date_tolerance.days) else 'Outside Tolerance' if pd.notna(date_diff) else 'N/A'),
                    'GSTIN Score': 0,
                    'Legal Name Score': legal_name_similarity,
                    'Trade Name Score': trade_name_similarity,
                    'Invoice Number Score': invoice_similarity,
                    'Tax Sign Status': 'Sign Match',
                    'Value Sign': 'Positive' if book_sign > 0 else 'Negative',
                    'Narrative': self._generate_narrative(book_row, gstr2a_row, False) if match_status == 'Partial Match' else 'Potential match for manual review due to tax deviation.',
                    'Suggestions': self._generate_suggestions(book_row, gstr2a_row, False) if match_status == 'Partial Match' else 'Manual review required: High tax difference but high similarity.'
                })
                
                # Add field status columns
                book_match_data.update(field_status)
                # Add GSTR-2A record
                gstr2a_match_data = gstr2a_row.to_dict()
                gstr2a_match_data.update({
                    'Match ID': match_id,
                    'Source Name': 'GSTR-2A',
                    'IGST Diff': -igst_diff,
                    'CGST Diff': -cgst_diff,
                    'SGST Diff': -sgst_diff,
                    'Date Diff': -date_diff if pd.notna(date_diff) else np.nan,
                    'Status': match_status,
                    'Sub Status': group_sub_status,
                    'Tax Diff Status': 'No Difference' if abs(igst_diff + cgst_diff + sgst_diff) <= self.tax_tolerance else 'High Deviation',
                    'Date Status': 'N/A' if match_status == 'Potential Match: Tax Deviation' else ('Within Tolerance' if (pd.notna(date_diff) and abs(date_diff) <= self.date_tolerance.days) else 'Outside Tolerance' if pd.notna(date_diff) else 'N/A'),
                    'GSTIN Score': 0,
                    'Legal Name Score': legal_name_similarity,
                    'Trade Name Score': trade_name_similarity,
                    'Invoice Number Score': invoice_similarity,
                    'Tax Sign Status': 'Sign Match',
                    'Value Sign': 'Positive' if book_sign > 0 else 'Negative',
                    'Narrative': self._generate_narrative(book_row, gstr2a_row, False) if match_status == 'Partial Match' else 'Potential match for manual review due to tax deviation.',
                    'Suggestions': self._generate_suggestions(book_row, gstr2a_row, False) if match_status == 'Partial Match' else 'Manual review required: High tax difference but high similarity.'
                })
                
                # Add field status columns
                gstr2a_match_data.update(field_status)
                self.matched_df = pd.concat([
                    self.matched_df,
                    pd.DataFrame([book_match_data]),
                    pd.DataFrame([gstr2a_match_data])
                ], ignore_index=True)

    def _flag_potential_tax_deviation_matches(self):
        """Flag high-similarity, high-tax-difference pairs for manual review as 'Potential Match: Tax Deviation'."""
        potential_tax_tolerance = 1000  # You can adjust this threshold as needed
        unprocessed_books = self.books_df[self.books_df['processed'] == False]
        unprocessed_gstr2a = self.gstr2a_df[self.gstr2a_df['processed'] == False]
        # Precompute normalized invoice numbers for fast filtering
        def norm_inv(inv):
            return str(inv).lstrip('0')[:3] if pd.notna(inv) else ''
        books_inv_map = unprocessed_books.copy()
        books_inv_map['norm_inv'] = books_inv_map['Invoice Number'].apply(norm_inv)
        gstr2a_inv_map = unprocessed_gstr2a.copy()
        gstr2a_inv_map['norm_inv'] = gstr2a_inv_map['Invoice Number'].apply(norm_inv)
        # Index GSTR-2A by (Invoice Date, norm_inv) for fast lookup
        gstr2a_groups = gstr2a_inv_map.groupby(['Invoice Date', 'norm_inv'])
        for bidx, book_row in books_inv_map.iterrows():
            book_sign = 1 if (book_row['Total IGST Amount'] + book_row['Total CGST Amount'] + book_row['Total SGST Amount']) >= 0 else -1
            norm_inv_b = book_row['norm_inv']
            inv_date_b = book_row['Invoice Date']
            # Only consider GSTR-2A with same date and similar invoice number
            if (inv_date_b, norm_inv_b) not in gstr2a_groups.groups:
                continue
            for gidx in gstr2a_groups.get_group((inv_date_b, norm_inv_b)).index:
                gstr2a_row = gstr2a_inv_map.loc[gidx]
                gstr2a_sign = 1 if (gstr2a_row['Total IGST Amount'] + gstr2a_row['Total CGST Amount'] + gstr2a_row['Total SGST Amount']) >= 0 else -1
                if book_sign != gstr2a_sign:
                    continue
                # Name similarity
                legal_name_similarity = 0
                if pd.notna(book_row['Supplier Legal Name']) and pd.notna(gstr2a_row['Supplier Legal Name']):
                    clean_book_legal = _clean_company_name(str(book_row['Supplier Legal Name']))
                    clean_gstr2a_legal = _clean_company_name(str(gstr2a_row['Supplier Legal Name']))
                    legal_name_similarity = fuzz.ratio(clean_book_legal, clean_gstr2a_legal)
                trade_name_similarity = 0
                if pd.notna(book_row['Supplier Trade Name']) and pd.notna(gstr2a_row['Supplier Trade Name']):
                    clean_book_trade = _clean_company_name(str(book_row['Supplier Trade Name']))
                    clean_gstr2a_trade = _clean_company_name(str(gstr2a_row['Supplier Trade Name']))
                    trade_name_similarity = fuzz.ratio(clean_book_trade, clean_gstr2a_trade)
                # Invoice number similarity
                invoice_similarity = 0
                if pd.notna(book_row['Invoice Number']) and pd.notna(gstr2a_row['Invoice Number']):
                    invoice_similarity = fuzz.ratio(str(book_row['Invoice Number']).lower(), str(gstr2a_row['Invoice Number']).lower())
                # At least one name similarity >= 70 and invoice similarity >= 70
                if (max(legal_name_similarity, trade_name_similarity) >= 70) and (invoice_similarity >= 70):
                    # Tax difference above strict tolerance but below potential threshold
                    igst_diff = book_row['Total IGST Amount'] - gstr2a_row['Total IGST Amount']
                    cgst_diff = book_row['Total CGST Amount'] - gstr2a_row['Total CGST Amount']
                    sgst_diff = book_row['Total SGST Amount'] - gstr2a_row['Total SGST Amount']
                    total_tax_diff = abs(igst_diff) + abs(cgst_diff) + abs(sgst_diff)
                    if total_tax_diff > self.tax_tolerance and total_tax_diff <= potential_tax_tolerance:
                        date_diff = np.nan
                        if pd.notna(book_row['Invoice Date']) and pd.notna(gstr2a_row['Invoice Date']):
                            date_diff = (book_row['Invoice Date'] - gstr2a_row['Invoice Date']).days
                        match_id = self.match_id_counter
                        self.match_id_counter += 1
                        # Generate field status columns
                        field_status = self._generate_field_status_columns(book_row, gstr2a_row)
                        
                        # Add Books record
                        book_match_data = book_row.to_dict()
                        book_match_data.update({
                            'Match ID': match_id,
                            'Source Name': 'Books',
                            'IGST Diff': igst_diff,
                            'CGST Diff': cgst_diff,
                            'SGST Diff': sgst_diff,
                            'Date Diff': date_diff,
                            'Status': 'Potential Match: Tax Deviation',
                            'Sub Status': f'Potential match with high tax deviation (₹{total_tax_diff:.2f}) but high similarity',
                            'Tax Diff Status': 'High Deviation',
                            'Date Status': 'Within Tolerance' if (pd.notna(date_diff) and abs(date_diff) <= self.date_tolerance.days) else 'Outside Tolerance' if pd.notna(date_diff) else 'N/A',
                            'GSTIN Score': 0,
                            'Legal Name Score': legal_name_similarity,
                            'Trade Name Score': trade_name_similarity,
                            'Invoice Number Score': invoice_similarity,
                            'Tax Sign Status': 'Sign Match',
                            'Value Sign': 'Positive' if book_sign > 0 else 'Negative',
                            'Narrative': 'Potential match for manual review due to tax deviation.',
                            'Suggestions': 'Manual review required: High tax difference but high similarity.'
                        })
                        
                        # Add field status columns
                        book_match_data.update(field_status)
                        # Add GSTR-2A record
                        gstr2a_match_data = gstr2a_row.to_dict()
                        gstr2a_match_data.update({
                            'Match ID': match_id,
                            'Source Name': 'GSTR-2A',
                            'IGST Diff': -igst_diff,
                            'CGST Diff': -cgst_diff,
                            'SGST Diff': -sgst_diff,
                            'Date Diff': -date_diff if pd.notna(date_diff) else np.nan,
                            'Status': 'Potential Match: Tax Deviation',
                            'Sub Status': f'Potential match with high tax deviation (₹{total_tax_diff:.2f}) but high similarity',
                            'Tax Diff Status': 'High Deviation',
                            'Date Status': 'Within Tolerance' if (pd.notna(date_diff) and abs(date_diff) <= self.date_tolerance.days) else 'Outside Tolerance' if pd.notna(date_diff) else 'N/A',
                            'GSTIN Score': 0,
                            'Legal Name Score': legal_name_similarity,
                            'Trade Name Score': trade_name_similarity,
                            'Invoice Number Score': invoice_similarity,
                            'Tax Sign Status': 'Sign Match',
                            'Value Sign': 'Positive' if book_sign > 0 else 'Negative',
                            'Narrative': 'Potential match for manual review due to tax deviation.',
                            'Suggestions': 'Manual review required: High tax difference but high similarity.'
                        })
                        
                        # Add field status columns
                        gstr2a_match_data.update(field_status)
                        self.matched_df = pd.concat([
                            self.matched_df,
                            pd.DataFrame([book_match_data]),
                            pd.DataFrame([gstr2a_match_data])
                        ], ignore_index=True)
                        # Mark as processed so not counted as unmatched
                        self.books_df.at[bidx, 'processed'] = True
                        self.gstr2a_df.at[gidx, 'processed'] = True
