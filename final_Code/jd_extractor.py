import pdfplumber
import fitz  # The PyMuPDF library
import sys
import os
import argparse
import shutil
import json 
def organize_pdfs_by_domain(json_file_path, source_folder, output_folder):
    """
    Organizes PDF files into subdirectories based on a JSON file.

    This function reads a JSON object where keys are category names (domains)
    and values are lists of PDF filenames. It then creates a directory for each
    domain in the specified output folder and copies the corresponding PDFs from
    the source folder into these new directories.

    Args:
        json_file_path (str): The full path to the JSON file containing the domain clusters.
        source_folder (str): The full path to the directory where all the source PDFs are stored.
        output_folder (str): The path to the directory where the organized folders and
                             PDFs will be created.
    """
    # --- 1. Validate Paths and Read JSON ---
    if not os.path.isfile(json_file_path):
        print(f"Error: JSON file not found at '{json_file_path}'", file=sys.stderr)
        return

    if not os.path.isdir(source_folder):
        print(f"Error: Source folder not found at '{source_folder}'", file=sys.stderr)
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            domain_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. Please check the file format.", file=sys.stderr)
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the JSON file: {e}", file=sys.stderr)
        return

    # --- 2. Create Base Output Directory ---
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output will be saved in: '{os.path.abspath(output_folder)}'")

    # --- 3. Iterate Through Domains and Copy Files ---
    files_copied = 0
    files_missing = 0

    for domain, filenames in domain_data.items():
        safe_domain_name = "".join(c if c.isalnum() else '_' for c in domain)
        domain_folder_path = os.path.join(output_folder, safe_domain_name)
        os.makedirs(domain_folder_path, exist_ok=True)
        print(f"\nProcessing domain: '{domain}'...")

        if not isinstance(filenames, list):
            print(f"  - Warning: Expected a list of filenames for domain '{domain}', but found {type(filenames)}. Skipping.")
            continue

        for filename in filenames:
            source_file_path = os.path.join(source_folder, filename)
            destination_file_path = os.path.join(domain_folder_path, filename)

            if os.path.exists(source_file_path):
                try:
                    shutil.copy2(source_file_path, destination_file_path)
                    files_copied += 1
                except Exception as e:
                    print(f"  - Error copying '{filename}': {e}", file=sys.stderr)
            else:
                print(f"  - Warning: File not found in source folder and was skipped: '{filename}'")
                files_missing += 1

    # --- 4. Final Report ---
    print("\n---------------------------------")
    print("         Process Complete        ")
    print("---------------------------------")
    print(f"Successfully copied: {files_copied} files.")
    print(f"Missing from source: {files_missing} files.")
    print("---------------------------------")

def extract_text_with_layout(pdf_path: str) -> str:
    """
    Extracts text from a single PDF file, preserving the original layout.
    Falls back to a secondary method if the primary one fails.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = "\n".join(
                page.extract_text(layout=True) or "" for page in pdf.pages
            )
        return full_text
    except Exception as e:
        print(f"Warning: pdfplumber failed for {os.path.basename(pdf_path)} ({e}). Falling back to PyMuPDF.", file=sys.stderr)
        try:
            with fitz.open(pdf_path) as doc:
                full_text = "\n".join(page.get_text("text") or "" for page in doc)
            return full_text
        except Exception as e_fallback:
            print(f"Error: PyMuPDF fallback also failed for {os.path.basename(pdf_path)} ({e_fallback}).", file=sys.stderr)
            return ""

def process_single_pdf(pdf_path: str, output_folder: str):
    """Processes a single PDF file and saves its content to a .txt file."""
    filename = os.path.basename(pdf_path)
    print(f"\nProcessing '{filename}'...")
    
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(output_folder, txt_filename)

    extracted_content = extract_text_with_layout(pdf_path)

    if extracted_content:
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_content)
        print(f"Successfully converted and saved to '{txt_filename}'")
    else:
        print(f"Failed to extract text from '{filename}'. No output file created.")

def process_pdfs_in_folder(input_folder: str, output_folder: str):
    """Processes all PDF files in an input folder."""
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{input_folder}'.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")
    for filename in pdf_files:
        pdf_path = os.path.join(input_folder, filename)
        process_single_pdf(pdf_path, output_folder)

# --- COMMAND-LINE INTERFACE SETUP ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Batch convert PDF files to TXT files or test a single file."
    )
    
    # Use a mutually exclusive group to ensure only one input method is used.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_folder", "-i",
        type=str,
        help="The path to the folder containing the PDF files to convert."
    )
    group.add_argument(
        "--test", "-t",
        type=str,
        dest='test_file', # Store the value in an attribute named 'test_file'
        help="The path to a single PDF file to test."
    )

    parser.add_argument(
        "--output_folder", "-o",
        type=str,
        required=True,
        help="The path to the folder where the output .txt files will be saved."
    )

    args = parser.parse_args()

    # Create the output folder; this is needed in both modes.
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Output will be saved to: {args.output_folder}")

    # Decide which function to call based on the provided arguments.
    if args.input_folder:
        process_pdfs_in_folder(args.input_folder, args.output_folder)
    elif args.test_file:
        if not os.path.isfile(args.test_file):
             print(f"Error: The test file was not found at '{args.test_file}'", file=sys.stderr)
        elif not args.test_file.lower().endswith(".pdf"):
             print(f"Error: The test file '{args.test_file}' is not a PDF.", file=sys.stderr)
        else:
            process_single_pdf(args.test_file, args.output_folder)


