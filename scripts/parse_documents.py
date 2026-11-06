#!/usr/bin/env python3
"""
Parse all PDFs in a directory into a single CSV file.
Adapted for Final Pipeline.
"""

import csv
import sys
import os
from pathlib import Path

# Import the reusable parser relative to this entry point.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdf_parser import extract_text_from_pdf, extract_sentences, extract_metadata

def is_valid_sentence(sentence):
    """
    Check if a sentence is valid and not garbage/nonsensical.
    Heuristics:
    1. Minimum length (e.g., 5 characters).
    2. Alphanumeric ratio (at least 50% alphanumeric).
    """
    if not sentence or len(sentence) < 5:
        return False
    
    # Count alphanumeric characters
    alnum_count = sum(c.isalnum() for c in sentence)
    if len(sentence) > 0 and (alnum_count / len(sentence)) < 0.5:
        return False
        
    return True

def parse_all_pdfs_to_single_csv(pdf_dir, output_csv):
    """Parse all PDFs in a directory into a single CSV file."""
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        print(f"Error: Directory '{pdf_dir}' not found")
        return False

    # Find all PDF files
    pdf_files = sorted(list(pdf_path.glob("**/*.pdf"))) # Recursive search to find files in subfolders if needed

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return False

    print(f"Found {len(pdf_files)} PDF files to process")
    print("-" * 70)

    # Open output CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(['Global #', 'Doc #', 'Sentence #', 'Firm name', 'Year', 'Quarter', 'File title', 'Doc published date', 'Context'])

        processed_doc_count = 0
        global_counter = 1  # Global sentence counter across all documents

        # Process each PDF
        for pdf_file in pdf_files:
            # Skip Non-EXAMPLE files
            if "Non-EXAMPLE" in str(pdf_file):
                continue

            # print(f"Processing: {pdf_file.name}") # Reduced verbosity

            try:
                # Extract text from PDF
                text = extract_text_from_pdf(pdf_file)
                if text is None:
                    # print(f"  ✗ Failed to extract text")
                    continue

                # Extract metadata
                metadata = extract_metadata(pdf_file, text)
                firm_name = metadata.get('bank', 'Unknown')
                year = metadata.get('year', 'Unknown')
                quarter = metadata.get('quarter', 'Unknown')
                date = metadata.get('date', 'Unknown')
                file_title = metadata.get('filename', pdf_file.stem)

                # Extract sentences
                raw_sentences = extract_sentences(text)
                
                # Filter sentences
                sentences = [s for s in raw_sentences if is_valid_sentence(s)]

                if not sentences:
                    # print(f"  ⚠ No valid sentences extracted (Original: {len(raw_sentences)})")
                    continue

                processed_doc_count += 1
                print(f"Processing [{processed_doc_count}]: {pdf_file.name}")

                # Write each sentence to CSV
                for sentence_num, sentence in enumerate(sentences, 1):
                    writer.writerow([
                        global_counter,  # Global #
                        processed_doc_count, # Doc #
                        sentence_num,    # Sentence #
                        firm_name,       # Firm name
                        year,            # Year
                        quarter,         # Quarter
                        file_title,      # File title
                        date,            # Doc published date
                        sentence         # Context
                    ])
                    global_counter += 1

                print(f"  ✓ Extracted {len(sentences)} sentences | Date: {date} | Firm: {firm_name}")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

    print("-" * 70)
    print(f"✓ Successfully created {output_csv}")
    print(f"  Total documents processed: {processed_doc_count}")
    print(f"  Total sentences extracted: {global_counter - 1}")

    return True

def main():
    # Default paths
    # Assuming running from "Final Pipeline" or root
    if Path("PDFs").exists():
        pdf_dir = "PDFs"
    elif Path("../PyPDF/data/PDFs").exists():
        pdf_dir = "../PyPDF/data/PDFs"
    else:
        pdf_dir = "PDFs" # Fallback
        
    output_csv = "parsed_data.csv"

    # Allow command line arguments
    if len(sys.argv) > 1:
        pdf_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]

    print(f"Input Directory: {pdf_dir}")
    print(f"Output CSV: {output_csv}")

    success = parse_all_pdfs_to_single_csv(pdf_dir, output_csv)

    if not success:
        print("\n✗ Processing failed!")
        sys.exit(1)
    else:
        print(f"\n✓ All PDFs parsed successfully!")

if __name__ == "__main__":
    main()
