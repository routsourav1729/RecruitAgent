import google.generativeai as genai
from pathlib import Path
import time
from typing import Dict, List, Optional
import zipfile
import io

class GeminiExtractor:
    """
    Extracts text from PDFs using Gemini API.
    Better for scanned/poor quality PDFs.
    """
    
    def __init__(self, model: genai.GenerativeModel, delay: float = 6.0):
        """
        Args:
            model: Pre-configured Gemini model instance
            delay: Delay between API calls (rate limiting)
        """
        self.model = model
        self.delay = delay
        
    def extract_single_pdf(self, pdf_bytes: bytes, filename: str) -> Dict[str, str]:
        """
        Extract text from a single PDF using Gemini.
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename (for reference)
            
        Returns:
            Dict with 'filename', 'text', 'status', 'error'
        """
        try:
            # Upload PDF to Gemini
            file = genai.upload_file(
                io.BytesIO(pdf_bytes),
                mime_type="application/pdf",
                display_name=filename
            )
            
            # Wait for processing
            time.sleep(2)
            
            # Create extraction prompt
            prompt = """Extract ALL text from this PDF document.

Instructions:
- Preserve the original structure and formatting
- Extract text from tables in a readable format
- Include headers, sections, and bullet points
- Do NOT summarize or interpret - extract verbatim text only
- If the document is scanned, use OCR to extract text

Output only the raw extracted text, nothing else."""

            # Generate response
            response = self.model.generate_content([prompt, file])
            
            # Clean up uploaded file
            genai.delete_file(file.name)
            
            # Extract text
            extracted_text = response.text
            
            return {
                "filename": filename,
                "text": extracted_text,
                "status": "success",
                "error": None,
                "length": len(extracted_text)
            }
            
        except Exception as e:
            return {
                "filename": filename,
                "text": "",
                "status": "error",
                "error": str(e),
                "length": 0
            }
    
    def extract_multiple_pdfs(
        self, 
        pdf_files: List[tuple], 
        progress_callback=None
    ) -> Dict[str, Dict]:
        """
        Extract text from multiple PDFs.
        
        Args:
            pdf_files: List of (filename, bytes) tuples
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            Dict mapping filename to extraction result
        """
        results = {}
        total = len(pdf_files)
        
        for idx, (filename, pdf_bytes) in enumerate(pdf_files, 1):
            if progress_callback:
                progress_callback(idx, total, filename)
            
            result = self.extract_single_pdf(pdf_bytes, filename)
            results[filename] = result
            
            # Rate limiting (except for last file)
            if idx < total:
                time.sleep(self.delay)
        
        return results
    
    def extract_from_zip(
        self, 
        zip_bytes: bytes,
        progress_callback=None
    ) -> Dict[str, Dict]:
        """
        Extract PDFs from a ZIP file and process them.
        
        Args:
            zip_bytes: ZIP file content as bytes
            progress_callback: Optional callback function
            
        Returns:
            Dict mapping filename to extraction result
        """
        pdf_files = []
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    # Skip directories and non-PDF files
                    if file_info.is_dir():
                        continue
                    if not file_info.filename.lower().endswith('.pdf'):
                        continue
                    
                    # Extract PDF
                    pdf_bytes = zip_ref.read(file_info.filename)
                    filename = Path(file_info.filename).name
                    pdf_files.append((filename, pdf_bytes))
            
            if not pdf_files:
                raise ValueError("No PDF files found in ZIP")
            
            # Process all PDFs
            return self.extract_multiple_pdfs(pdf_files, progress_callback)
            
        except Exception as e:
            return {
                "error": {
                    "filename": "zip_extraction",
                    "text": "",
                    "status": "error",
                    "error": f"Failed to process ZIP: {str(e)}",
                    "length": 0
                }
            }


def extract_with_gemini(
    model: genai.GenerativeModel,
    jd_file: Optional[bytes] = None,
    cv_files: Optional[List[tuple]] = None,
    cv_zip: Optional[bytes] = None,
    progress_callback=None
) -> tuple:
    """
    Convenience function for extraction in master pipeline.
    
    Args:
        model: Gemini model instance
        jd_file: JD PDF bytes (optional)
        cv_files: List of (filename, bytes) tuples (optional)
        cv_zip: ZIP file bytes containing CVs (optional)
        progress_callback: Progress callback function
    
    Returns:
        (jd_result, cv_results) tuple
    """
    extractor = GeminiExtractor(model=model)
    
    jd_result = None
    cv_results = {}
    
    # Extract JD
    if jd_file:
        if progress_callback:
            progress_callback(0, 1, "Extracting Job Description...")
        jd_result = extractor.extract_single_pdf(jd_file, "job_description.pdf")
    
    # Extract CVs
    if cv_zip:
        if progress_callback:
            progress_callback(0, 1, "Extracting CVs from ZIP...")
        cv_results = extractor.extract_from_zip(cv_zip, progress_callback)
    elif cv_files:
        cv_results = extractor.extract_multiple_pdfs(cv_files, progress_callback)
    
    return jd_result, cv_results