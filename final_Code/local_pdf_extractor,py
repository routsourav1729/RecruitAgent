#!/usr/bin/env python3
"""
Local PDF Extractor (Cloud-Ready)

Two extraction modes:
1. Simple (for JDs): Fast pdfplumber → PyMuPDF fallback
2. Advanced (for CVs): Table detection, positioning, quality checks

Returns text in-memory without writing to disk.
Designed for Streamlit Cloud deployment.

Usage from master:
    from stages.local_pdf_extractor import extract_jd, extract_cvs
    
    # Extract JD (simple mode)
    jd_text = extract_jd(jd_file.read(), "jd.pdf")
    
    # Extract CVs (advanced mode with table handling)
    cv_files = [(f.name, f.read()) for f in uploaded_cvs]
    cv_texts = extract_cvs(cv_files)
"""

import io
from typing import Dict, Tuple, List
from dataclasses import dataclass
import pdfplumber
import fitz  # PyMuPDF


@dataclass
class ContentElement:
    """Represents a content element (text or table) with its position"""
    content_type: str  # 'text' or 'table'
    content: str
    y_position: float  # Top position for ordering
    bbox: tuple = None  # Bounding box (x0, y0, x1, y1)


class SimplePDFExtractor:
    """
    Simple extraction for JDs.
    Fast, no table detection.
    """
    
    @staticmethod
    def extract_with_pdfplumber(pdf_bytes: bytes) -> str:
        """Primary method: pdfplumber with layout."""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                full_text = ""
                for i, page in enumerate(pdf.pages):
                    if i > 0:
                        full_text += f"\n--- Page {i + 1} ---\n\n"
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        full_text += page_text
                return full_text
        except Exception as e:
            raise Exception(f"pdfplumber failed: {str(e)}")
    
    @staticmethod
    def extract_with_pymupdf(pdf_bytes: bytes) -> str:
        """Fallback method: PyMuPDF."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = ""
            for i, page in enumerate(doc):
                if i > 0:
                    full_text += f"\n--- Page {i + 1} ---\n\n"
                full_text += page.get_text() + "\n"
            doc.close()
            return full_text
        except Exception as e:
            raise Exception(f"PyMuPDF failed: {str(e)}")
    
    def extract(self, pdf_bytes: bytes, filename: str) -> Tuple[str, bool, str]:
        """Extract with automatic fallback."""
        try:
            text = self.extract_with_pdfplumber(pdf_bytes)
            return text, True, "pdfplumber"
        except:
            try:
                text = self.extract_with_pymupdf(pdf_bytes)
                return text, True, "pymupdf-fallback"
            except Exception as e:
                return f"ERROR: {str(e)}", False, "failed"


class AdvancedPDFExtractor:
    """
    Advanced extraction for CVs.
    Includes table detection, positioning, and quality checks.
    """
    
    def __init__(self):
        self.min_text_length = 100
        self.min_word_count = 20
    
    def is_legitimate_table(self, table) -> bool:
        """Check if detected table is real (min 2x2 with data)."""
        try:
            table_data = table.extract()
            if not table_data or len(table_data) < 2:
                return False
            
            valid_rows = [row for row in table_data if row is not None]
            if len(valid_rows) < 2:
                return False
            
            # Count columns per row
            column_counts = []
            for row in valid_rows:
                if row:
                    non_none = [cell for cell in row if cell is not None]
                    column_counts.append(len(non_none))
            
            if not column_counts or max(column_counts) < 2:
                return False
            
            # Check density (30% of cells have content)
            total_cells = sum(column_counts)
            filled_cells = sum(
                1 for row in valid_rows if row
                for cell in row
                if cell is not None and str(cell).strip()
            )
            
            if total_cells == 0 or (filled_cells / total_cells) < 0.3:
                return False
            
            # Check dimensions
            bbox = table.bbox
            if bbox:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width < 100 or height < 50:
                    return False
            
            return True
        except:
            return False
    
    def detect_legitimate_tables(self, pdf_bytes: bytes) -> bool:
        """Check if PDF has legitimate tables."""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    tables = page.find_tables()
                    if tables:
                        for table in tables:
                            if self.is_legitimate_table(table):
                                return True
            return False
        except:
            return False
    
    def format_table(self, table_data) -> str:
        """Format table data with proper alignment."""
        if not table_data:
            return ""
        
        # Find max width per column
        col_widths = []
        for row in table_data:
            for j, cell in enumerate(row or []):
                cell_text = str(cell).strip() if cell else ""
                while len(col_widths) <= j:
                    col_widths.append(0)
                col_widths[j] = max(col_widths[j], len(cell_text))
        
        # Format rows
        formatted = []
        for row in table_data:
            if row:
                cells = []
                for j, cell in enumerate(row):
                    text = str(cell).strip() if cell else ""
                    width = col_widths[j] if j < len(col_widths) else 0
                    cells.append(text.ljust(width))
                
                row_text = "  ".join(cells).rstrip()
                if row_text.strip():
                    formatted.append(row_text)
        
        # Add separator after header
        if len(formatted) > 1:
            separator = "  ".join("-" * w for w in col_widths)
            formatted.insert(1, separator)
        
        return "\n".join(formatted)
    
    def extract_with_positioned_tables(self, pdf_bytes: bytes) -> str:
        """Extract with tables in proper positions."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = ""
            
            for page_num, page in enumerate(doc):
                if page_num > 0:
                    full_text += f"\n--- Page {page_num + 1} ---\n\n"
                
                # Get tables from pdfplumber
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    plumber_page = pdf.pages[page_num]
                    tables = plumber_page.find_tables()
                    
                    content_elements = []
                    table_regions = []
                    
                    # Process tables
                    for table in tables:
                        if self.is_legitimate_table(table):
                            table_data = table.extract()
                            formatted = self.format_table(table_data)
                            
                            if formatted and table.bbox:
                                content_elements.append(ContentElement(
                                    content_type='table',
                                    content=formatted,
                                    y_position=table.bbox[1],
                                    bbox=table.bbox
                                ))
                                table_regions.append(table.bbox)
                    
                    # Get text blocks (avoiding table regions)
                    blocks = page.get_text("dict")["blocks"]
                    for block in blocks:
                        if "lines" not in block:
                            continue
                        
                        block_rect = fitz.Rect(block["bbox"])
                        
                        # Skip if overlaps with table
                        is_table_block = any(
                            block_rect.intersects(fitz.Rect(tb))
                            for tb in table_regions
                        )
                        
                        if not is_table_block:
                            block_text = ""
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    block_text += span["text"]
                                block_text += "\n"
                            
                            if block_text.strip():
                                content_elements.append(ContentElement(
                                    content_type='text',
                                    content=block_text.strip(),
                                    y_position=block["bbox"][1],
                                    bbox=block["bbox"]
                                ))
                    
                    # Sort by position and combine
                    content_elements.sort(key=lambda x: x.y_position)
                    for element in content_elements:
                        if element.content_type == 'table':
                            full_text += f"\n[TABLE]\n{element.content}\n[/TABLE]\n\n"
                        else:
                            full_text += f"{element.content}\n\n"
            
            doc.close()
            return full_text
        except Exception as e:
            raise Exception(f"Positioned extraction failed: {str(e)}")
    
    def extract_simple(self, pdf_bytes: bytes) -> str:
        """Simple extraction when no tables detected."""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                full_text = ""
                for i, page in enumerate(pdf.pages):
                    if i > 0:
                        full_text += f"\n--- Page {i + 1} ---\n\n"
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        full_text += page_text
                return full_text
        except Exception as e:
            # Fallback to PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = ""
            for i, page in enumerate(doc):
                if i > 0:
                    full_text += f"\n--- Page {i + 1} ---\n\n"
                full_text += page.get_text() + "\n"
            doc.close()
            return full_text
    
    def is_good_quality(self, text: str) -> bool:
        """Check text quality."""
        if not text or len(text.strip()) < self.min_text_length:
            return False
        if len(text.split()) < self.min_word_count:
            return False
        
        # Check for garbled chars
        garbled = ['�', 'CID', '\x00', '\ufffd']
        garbled_count = sum(text.count(g) for g in garbled)
        if len(text) > 0 and (garbled_count / len(text)) > 0.1:
            return False
        
        return True
    
    def extract(self, pdf_bytes: bytes, filename: str) -> Tuple[str, bool, str]:
        """Main extraction with intelligent routing."""
        try:
            # Detect tables
            has_tables = self.detect_legitimate_tables(pdf_bytes)
            
            if has_tables:
                text = self.extract_with_positioned_tables(pdf_bytes)
                method = "advanced-tables"
            else:
                text = self.extract_simple(pdf_bytes)
                method = "advanced-simple"
            
            success = self.is_good_quality(text)
            return text, success, method
            
        except Exception as e:
            return f"ERROR: {str(e)}", False, "failed"


# Convenience functions for master

def extract_jd(jd_bytes: bytes, filename: str = "jd.pdf") -> Dict:
    """Extract JD using simple method."""
    extractor = SimplePDFExtractor()
    text, success, method = extractor.extract(jd_bytes, filename)
    return {
        "filename": filename,
        "text": text,
        "success": success,
        "method": method
    }


def extract_cvs(
    cv_files: List[Tuple[str, bytes]],
    progress_callback=None
) -> Dict[str, Dict]:
    """Extract CVs using advanced method with table handling."""
    extractor = AdvancedPDFExtractor()
    results = {}
    total = len(cv_files)
    
    for idx, (filename, pdf_bytes) in enumerate(cv_files, 1):
        if progress_callback:
            progress_callback(idx, total, filename)
        
        text, success, method = extractor.extract(pdf_bytes, filename)
        results[filename] = {
            "filename": filename,
            "text": text,
            "success": success,
            "method": method,
            "length": len(text)
        }
    
    return results


def extract_cvs_from_zip(
    zip_bytes: bytes,
    progress_callback=None
) -> Dict[str, Dict]:
    """Extract CVs from ZIP using advanced method."""
    import zipfile
    from pathlib import Path
    
    pdf_files = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.is_dir():
                    continue
                if not file_info.filename.lower().endswith('.pdf'):
                    continue
                
                pdf_bytes = zip_ref.read(file_info.filename)
                filename = Path(file_info.filename).name
                pdf_files.append((filename, pdf_bytes))
        
        if not pdf_files:
            raise ValueError("No PDFs in ZIP")
        
        return extract_cvs(pdf_files, progress_callback)
    except Exception as e:
        return {"error": {"error": str(e), "success": False}}