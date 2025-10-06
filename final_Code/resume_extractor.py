#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass

try:
    import pdfplumber
    import fitz  # PyMuPDF
    from tqdm import tqdm
except ImportError:
    print("Required packages not found. Please run: pip install pdfplumber PyMuPDF tqdm")
    sys.exit(1)

@contextmanager
def suppress_stderr():
    """Suppress warnings from pdfplumber during table detection"""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_extraction.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ContentElement:
    """Represents a content element (text or table) with its position"""
    content_type: str  # 'text' or 'table'
    content: str
    y_position: float  # Top position for ordering
    bbox: tuple = None  # Bounding box (x0, y0, x1, y1)

class ImprovedHybridExtractor:
    """Improved hybrid PDF extractor with intelligent table detection and positioning"""
    
    def __init__(self):
        self.min_text_length = 100
        self.min_word_count = 20
    
    def is_legitimate_table(self, table) -> bool:
        """
        Check if detected table is actually legitimate (minimum 2 rows × 2 columns)
        """
        try:
            table_data = table.extract()
            
            if not table_data or len(table_data) < 2:
                return False
            
            # Check column consistency
            valid_rows = [row for row in table_data if row is not None]
            if len(valid_rows) < 2:
                return False
            
            # Count columns in each row
            column_counts = []
            for row in valid_rows:
                if row:
                    non_none_cells = [cell for cell in row if cell is not None]
                    column_counts.append(len(non_none_cells))
            
            if not column_counts or max(column_counts) < 2:
                return False
            
            # Check that most rows have at least 2 columns
            rows_with_min_cols = sum(1 for count in column_counts if count >= 2)
            if rows_with_min_cols < 2:
                return False
            
            # Check data density (30% of cells should have content)
            total_cells = sum(column_counts)
            filled_cells = 0
            
            for row in valid_rows:
                if row:
                    for cell in row:
                        if cell is not None and str(cell).strip():
                            filled_cells += 1
            
            if total_cells == 0:
                return False
            
            data_density = filled_cells / total_cells
            if data_density < 0.3:
                return False
            
            # Check table dimensions
            bbox = table.bbox
            if bbox:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Reject very small tables (likely noise)
                if width < 100 or height < 50:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def detect_legitimate_tables(self, pdf_path: str) -> bool:
        """
        The decision-maker: Check if PDF contains any legitimate tables
        """
        try:
            with suppress_stderr():
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        detected_tables = page.find_tables()
                        
                        if detected_tables:
                            for table in detected_tables:
                                if self.is_legitimate_table(table):
                                    logger.info(f"  -> Decision: Legitimate table detected in '{os.path.basename(pdf_path)}'")
                                    return True
            
            logger.info(f"  -> Decision: No legitimate tables in '{os.path.basename(pdf_path)}'")
            return False
            
        except Exception as e:
            logger.warning(f"  [Error] Could not check tables in {pdf_path}: {e}")
            return False  # Default to 'no tables' on error
    
    def format_table_properly(self, table_data) -> str:
        """Format table data into a readable table structure"""
        if not table_data:
            return ""
        
        # Find maximum width for each column
        col_widths = []
        for row in table_data:
            for j, cell in enumerate(row or []):
                cell_text = str(cell).strip() if cell else ""
                while len(col_widths) <= j:
                    col_widths.append(0)
                col_widths[j] = max(col_widths[j], len(cell_text))
        
        # Format each row
        formatted_rows = []
        for row in table_data:
            if row:
                formatted_cells = []
                for j, cell in enumerate(row):
                    cell_text = str(cell).strip() if cell else ""
                    width = col_widths[j] if j < len(col_widths) else 0
                    formatted_cells.append(cell_text.ljust(width))
                
                formatted_row = "  ".join(formatted_cells).rstrip()
                if formatted_row.strip():
                    formatted_rows.append(formatted_row)
        
        # Add a separator line after header (if exists)
        if len(formatted_rows) > 1:
            separator = "  ".join("-" * width for width in col_widths)
            formatted_rows.insert(1, separator)
        
        return "\n".join(formatted_rows)
    
    def extract_page_with_positioned_tables(self, page) -> List[ContentElement]:
        """
        Extract page content with tables in their proper positions
        Returns a list of ContentElements sorted by position
        """
        content_elements = []
        
        # Get all tables and their positions
        tables = page.find_tables()
        table_regions = []
        
        for table in tables:
            try:
                if not self.is_legitimate_table(table):
                    continue
                    
                table_data = table.extract()
                formatted_table = self.format_table_properly(table_data)
                
                if formatted_table and table.bbox:
                    table_element = ContentElement(
                        content_type='table',
                        content=formatted_table,
                        y_position=table.bbox[1],  # y0 (top)
                        bbox=table.bbox
                    )
                    content_elements.append(table_element)
                    table_regions.append(table.bbox)
                    
            except Exception as e:
                logger.warning(f"Failed to process table: {e}")
        
        # Get text blocks
        try:
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:  # Skip non-text blocks
                    continue
                
                block_rect = fitz.Rect(block["bbox"])
                
                # Check if this block overlaps with any table
                is_table_block = False
                for table_bbox in table_regions:
                    table_rect = fitz.Rect(table_bbox)
                    if block_rect.intersects(table_rect):
                        is_table_block = True
                        break
                
                if not is_table_block:
                    # Extract text from this block
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                        block_text += "\n"
                    
                    if block_text.strip():
                        text_element = ContentElement(
                            content_type='text',
                            content=block_text.strip(),
                            y_position=block["bbox"][1],  # y0 (top)
                            bbox=block["bbox"]
                        )
                        content_elements.append(text_element)
                        
        except Exception as e:
            logger.warning(f"Failed to extract text blocks: {e}")
            # Fallback: just get all text
            page_text = page.get_text()
            if page_text.strip():
                content_elements.append(ContentElement(
                    content_type='text',
                    content=page_text,
                    y_position=0,
                    bbox=None
                ))
        
        # Sort by vertical position (top to bottom)
        content_elements.sort(key=lambda x: x.y_position)
        
        return content_elements
    
    def extract_with_pymupdf(self, pdf_path: str) -> str:
        """
        Extractor #1: Used when legitimate tables are detected
        Places tables in their proper positions within the text flow
        """
        logger.info("  -> Action: Using PyMuPDF for positioned table extraction")
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for i, page in enumerate(doc):
                if i > 0:
                    full_text += f"\n--- Page {i + 1} ---\n\n"
                
                # Extract content with positioned tables
                content_elements = self.extract_page_with_positioned_tables(page)
                
                # Combine all elements in order
                for element in content_elements:
                    if element.content_type == 'table':
                        full_text += f"\n[TABLE]\n{element.content}\n[/TABLE]\n\n"
                    else:
                        full_text += f"{element.content}\n\n"
            
            doc.close()
            return full_text
            
        except Exception as e:
            logger.error(f"  [Error] PyMuPDF positioned extraction failed: {e}")
            # Fallback to simple extraction
            try:
                doc = fitz.open(pdf_path)
                full_text = ""
                for i, page in enumerate(doc):
                    if i > 0:
                        full_text += f"\n--- Page {i + 1} ---\n\n"
                    full_text += page.get_text() + "\n"
                doc.close()
                return full_text
            except:
                return ""
    
    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """
        Extractor #2: Used when no legitimate tables are detected
        Excellent at preserving general layout
        """
        logger.info("  -> Action: Using pdfplumber for layout extraction")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                
                for i, page in enumerate(pdf.pages):
                    if i > 0:
                        full_text += f"\n--- Page {i + 1} ---\n\n"
                    
                    # Preserve layout for regular text
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        full_text += page_text
                
                return full_text
                
        except Exception as e:
            logger.error(f"  [Error] pdfplumber extraction failed: {e}")
            return ""
    
    def extract_with_pymupdf_simple(self, pdf_path: str) -> str:
        """
        Simple PyMuPDF extraction as fallback
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for i, page in enumerate(doc):
                if i > 0:
                    full_text += f"\n--- Page {i + 1} ---\n\n"
                full_text += page.get_text() + "\n"
            doc.close()
            return full_text
        except Exception as e:
            logger.error(f"  [Error] PyMuPDF simple extraction failed: {e}")
            return ""
    
    def is_good_quality(self, text: str) -> bool:
        """Check if extracted text is of good quality"""
        if not text or len(text.strip()) < self.min_text_length:
            return False
        
        word_count = len(text.split())
        if word_count < self.min_word_count:
            return False
        
        # Check for excessive garbled characters
        garbled_indicators = ['�', 'CID', '\x00', '\ufffd']
        garbled_count = sum(text.count(indicator) for indicator in garbled_indicators)
        
        if len(text) > 0 and (garbled_count / len(text)) > 0.1:
            return False
        
        return True
    
    def extract_pdf_text(self, pdf_path: str) -> Tuple[str, bool, str]:
        """
        Main extraction logic using hybrid approach with smart fallback
        """
        logger.info(f"Processing: {os.path.basename(pdf_path)}")
        
        try:
            # THE CORE DECISION LOGIC
            has_tables = self.detect_legitimate_tables(pdf_path)
            
            if has_tables:
                # RULE: Legitimate tables detected -> Use PyMuPDF with positioned tables
                extracted_text = self.extract_with_pymupdf(pdf_path)
                method = "PyMuPDF-PositionedTables"
            else:
                # RULE: No legitimate tables -> Use pdfplumber  
                extracted_text = self.extract_with_pdfplumber(pdf_path)
                method = "pdfplumber-Layout"
                
                # SMART FALLBACK: If pdfplumber gives poor results, try PyMuPDF
                if not self.is_good_quality(extracted_text):
                    logger.warning(f"pdfplumber gave poor quality ({len(extracted_text)} chars), trying PyMuPDF fallback")
                    pymupdf_text = self.extract_with_pymupdf_simple(pdf_path)
                    if self.is_good_quality(pymupdf_text) or len(pymupdf_text) > len(extracted_text):
                        extracted_text = pymupdf_text
                        method = "PyMuPDF-Fallback"
            
            # Check quality
            success = self.is_good_quality(extracted_text)

            if not extracted_text:
                # Last resort: Try simple PyMuPDF if everything else failed
                logger.warning("No text extracted, trying final PyMuPDF fallback")
                extracted_text = self.extract_with_pymupdf_simple(pdf_path)
                method = "PyMuPDF-LastResort"
                success = self.is_good_quality(extracted_text)
                
                if not extracted_text:
                    return "ERROR: No text extracted", False, f"{method}-Failed"
            
            return extracted_text, success, method
            
        except Exception as e:
            error_msg = f"ERROR: Extraction failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, False, "Error"

def process_single_pdf(pdf_path: str, output_folder: str, flagged_folder: str) -> Tuple[str, bool, str, str]:
    """Process a single PDF file"""
    file_name = os.path.basename(pdf_path)
    file_base = os.path.splitext(file_name)[0]
    output_file = os.path.join(output_folder, f"{file_base}.txt")
    
    extractor = ImprovedHybridExtractor()
    
    try:
        # Extract text using hybrid approach
        extracted_text, is_good_quality, method_used = extractor.extract_pdf_text(pdf_path)
        
        # Always write the output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        # Flag if poor quality
        if not is_good_quality:
            flagged_pdf_path = os.path.join(flagged_folder, file_name)
            shutil.copy2(pdf_path, flagged_pdf_path)
            logger.info(f"Flagged {file_name} for poor quality")
            return file_name, False, f"Poor quality: {len(extracted_text)} chars", method_used
        
        return file_name, True, f"Success: {len(extracted_text)} chars", method_used
        
    except Exception as e:
        error_msg = f"Critical error: {str(e)}"
        logger.error(f"Failed to process {file_name}: {error_msg}")
        
        # Flag and write error
        try:
            flagged_pdf_path = os.path.join(flagged_folder, file_name)
            shutil.copy2(pdf_path, flagged_pdf_path)
        except:
            pass
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"EXTRACTION FAILED: {error_msg}")
        
        return file_name, False, error_msg, "Error"

def main():
    parser = argparse.ArgumentParser(description='Improved Hybrid PDF Text Extraction')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing PDF files')
    parser.add_argument('--output', '-o', required=True, help='Output folder for text files')
    parser.add_argument('--test', help='Test on a single PDF file')
    args = parser.parse_args()
    
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    flagged_folder = output_folder / "flagged"
    
    # Create directories
    output_folder.mkdir(parents=True, exist_ok=True)
    flagged_folder.mkdir(parents=True, exist_ok=True)
    
    # Test mode
    if args.test:
        test_file = Path(args.test)
        if not test_file.exists():
            logger.error(f"Test file not found: {test_file}")
            return
        
        logger.info(f"Testing: {test_file}")
        file_name, success, message, method_used = process_single_pdf(
            str(test_file), str(output_folder), str(flagged_folder)
        )
        
        print(f"\n{'='*50}")
        print(f"TEST RESULT")
        print(f"{'='*50}")
        print(f"File: {file_name}")
        print(f"Status: {'SUCCESS' if success else 'FLAGGED'}")
        print(f"Method: {method_used}")
        print(f"Details: {message}")
        
        output_file = output_folder / f"{test_file.stem}.txt"
        print(f"Output: {output_file}")
        
        # Show preview
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                print(f"\nFirst 20 lines:")
                for i, line in enumerate(lines[:20], 1):
                    print(f"{i:2d}: {line}")
        
        return
    
    # Process all PDFs
    pdf_files = list(input_folder.glob('*.pdf')) + list(input_folder.glob('*.PDF'))
    
    if not pdf_files:
        logger.error(f"No PDF files found in {input_folder}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    print(f"Output: {output_folder}")
    print(f"Flagged: {flagged_folder}")
    
    results = []
    method_stats = {}
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        file_name, success, message, method_used = process_single_pdf(
            str(pdf_file), str(output_folder), str(flagged_folder)
        )
        results.append((file_name, success, message, method_used))
        method_stats[method_used] = method_stats.get(method_used, 0) + 1
    
    # Summary
    successful = sum(1 for _, success, _, _ in results if success)
    flagged = len(results) - successful
    
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Flagged: {flagged}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    # Method statistics
    print(f"\nExtraction Method Statistics:")
    print(f"{'='*30}")
    for method, count in sorted(method_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"{method:>20}: {count:>3} files ({percentage:>5.1f}%)")
    
    # Show flagged files
    flagged_files = [(name, msg) for name, success, msg, _ in results if not success]
    if flagged_files:
        print(f"\nFlagged files:")
        for name, msg in flagged_files[:10]:  # Show first 10
            print(f"  - {name}: {msg}")
        if len(flagged_files) > 10:
            print(f"  ... and {len(flagged_files) - 10} more")

if __name__ == "__main__":
    main()