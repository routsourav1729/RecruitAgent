#!/usr/bin/env python3
"""
Gemini Parser (Stages 1 & 2) - Text to Structured JSON

Converts raw text (from CVs or JDs) into structured JSON using Gemini API.
Can also parse PDFs directly for the "Accurate" mode.

Designed to be called by master orchestrator.

Usage from master:
    from stages.gemini_parser import GeminiParser, parse_jd, parse_cvs
    
    # Fast mode - from extracted text
    jd_json = parse_jd(model, jd_text, "jd.pdf")
    
    # Accurate mode - direct from PDF
    jd_json = parse_jd_from_pdf(model, jd_bytes, "jd.pdf")
"""

import json
import re
import time
import google.generativeai as genai
from typing import Dict, Optional
from pathlib import Path


class GeminiParser:
    """
    Parses raw text into structured JSON using Gemini API.
    Handles both CV and JD parsing based on prompt template.
    """
    
    # Domain and seniority options for prompt injection
    DOMAINS = [
        "Sales & Business Development", 
        "Credit & Risk Management",
        "Finance & Accounts (F&A)", 
        "Operations", 
        "Technology (IT)",
        "Human Resources (HR)", 
        "Legal & Compliance",
        "Marketing & Communications", 
        "Administration & Facilities", 
        "Other"
    ]
    
    SENIORITY_LEVELS = [
        "Entry-Level / Officer", 
        "Team Lead / Supervisor", 
        "Manager",
        "Senior Manager / Lead", 
        "Head of Department / VP", 
        "Executive / C-Suite"
    ]
    
    def __init__(
        self, 
        model,  # genai.GenerativeModel instance
        prompt_file: str,
        max_retries: int = 2,
        delay: float = 6.0
    ):
        """
        Initialize parser with Gemini model and prompt template.
        
        Args:
            model: Pre-configured Gemini model instance (from master)
            prompt_file: Path to prompt template file
            max_retries: Number of retry attempts for failed parsing
            delay: Delay between retries (rate limiting)
        """
        self.model = model
        self.max_retries = max_retries
        self.delay = delay
        
        # Load and parse prompt template
        self.prompt_template = self._load_prompt_template(prompt_file)
        
        print(f"GeminiParser initialized with prompt: {Path(prompt_file).name}")
    
    def _load_prompt_template(self, prompt_file: str) -> str:
        """Load prompt template from file and clean formatting."""
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Remove Qwen-specific markers if they exist
            template = template.replace("[USER_PROMPT_BELOW]", "")
            
            # Clean up any model-specific tags
            template = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', template, flags=re.DOTALL)
            
            return template.strip()
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template not found: {prompt_file}")
    
    def _create_prompt(self, text_content: str, filename: str) -> str:
        """
        Create final prompt by filling template placeholders.
        
        Args:
            text_content: Raw text from PDF extraction
            filename: Original filename for reference
            
        Returns:
            Complete prompt ready for Gemini
        """
        # Format domain and seniority lists
        domains_str = ', '.join(f'"{d}"' for d in self.DOMAINS)
        seniority_str = ', '.join(f'"{s}"' for s in self.SENIORITY_LEVELS)
        
        # Fill template placeholders
        prompt = self.prompt_template.format(
            FILENAME=filename,
            TEXT_CONTENT=text_content,
            DOMAINS=domains_str,
            SENIORITY_LEVELS=seniority_str
        )
        
        return prompt
    
    def _extract_json(self, response_text: str) -> Dict:
        """
        Extract JSON from Gemini's response.
        Handles code blocks and attempts repair if needed.
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Parsed JSON dict or error dict
        """
        # Try to find JSON in code blocks first
        json_match = re.search(
            r'```(?:json)?\s*(\{.*?\})\s*```', 
            response_text, 
            re.DOTALL
        )
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: find first { to last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                return {
                    "error": "No JSON object found in response",
                    "context": response_text[:200]
                }
            
            json_str = response_text[start_idx:end_idx + 1]
        
        # Try parsing
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Attempt repair
            try:
                repaired = self._repair_json(json_str)
                return json.loads(repaired)
            except json.JSONDecodeError as e:
                return {
                    "error": f"JSON parsing failed: {str(e)}",
                    "context": json_str[:200] + "..."
                }
    
    def _repair_json(self, json_str: str) -> str:
        """Apply regex fixes to common JSON errors."""
        # Fix trailing commas
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
        
        # Fix unescaped backslashes
        json_str = re.sub(r'\\(?![/bfnrt"\\])', '/', json_str)
        
        # Fix unquoted keys (best effort)
        json_str = re.sub(
            r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', 
            r'\1"\2"\3', 
            json_str
        )
        
        return json_str
    
    def _validate_result(self, result: Dict, is_cv: bool = True) -> bool:
        """
        Check if parsed result has minimum required fields.
        
        Args:
            result: Parsed JSON dict
            is_cv: True for CV parsing, False for JD parsing
            
        Returns:
            True if valid, False otherwise
        """
        if "error" in result:
            return False
        
        if not result.get("source_file"):
            return False
        
        # CV-specific validation
        if is_cv:
            return bool(
                result.get("candidate_name") and 
                result.get("gating_profile")
            )
        
        # JD-specific validation
        else:
            return bool(
                result.get("role_classification") and 
                result.get("key_responsibilities")
            )
    
    def process_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        is_cv: bool = True
    ) -> Dict:
        """
        Process PDF directly with Gemini (no extraction step needed).
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename
            is_cv: True for CV, False for JD
            
        Returns:
            Structured JSON dict or error dict
        """
        import io
        print(f"Parsing PDF directly: {filename}")
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                print(f"  Retry {attempt}/{self.max_retries}...")
                time.sleep(self.delay)
            
            try:
                # Upload PDF to Gemini
                file = genai.upload_file(
                    io.BytesIO(pdf_bytes),
                    mime_type="application/pdf",
                    display_name=filename
                )
                
                time.sleep(2)  # Wait for processing
                
                # Create prompt for PDF parsing
                prompt = self._create_prompt_for_pdf(filename)
                
                # Call Gemini API with PDF
                response = self.model.generate_content(
                    [prompt, file],
                    generation_config={
                        "temperature": 0.1 if attempt == 0 else 0.2,
                        "max_output_tokens": 4096,
                    }
                )
                
                # Clean up uploaded file
                genai.delete_file(file.name)
                
                # Extract JSON from response
                result = self._extract_json(response.text)
                
                # Validate result
                if self._validate_result(result, is_cv=is_cv):
                    if attempt > 0:
                        print(f"  ✓ Success after {attempt} retries")
                    return result
                
                error_msg = result.get("error", "Invalid structure")
                print(f"  ✗ Attempt {attempt} failed: {error_msg}")
                
                if attempt == self.max_retries:
                    return {
                        "error": f"Failed after {self.max_retries} retries",
                        "last_error": error_msg,
                        "filename": filename
                    }
            
            except Exception as e:
                print(f"  ✗ Exception: {str(e)}")
                if attempt == self.max_retries:
                    return {
                        "error": f"Processing exception: {str(e)}",
                        "filename": filename
                    }
        
        return {"error": "Unexpected failure", "filename": filename}
    
    def _create_prompt_for_pdf(self, filename: str) -> str:
        """
        Create prompt for direct PDF parsing.
        
        Args:
            filename: Original filename for reference
            
        Returns:
            Prompt string
        """
        # Format domain and seniority lists
        domains_str = ', '.join(f'"{d}"' for d in self.DOMAINS)
        seniority_str = ', '.join(f'"{s}"' for s in self.SENIORITY_LEVELS)
        
        # Use the template but replace TEXT_CONTENT placeholder with instruction
        prompt = self.prompt_template.replace(
            "{TEXT_CONTENT}",
            "[Extract all text from the provided PDF document and parse it]"
        )
        prompt = prompt.format(
            FILENAME=filename,
            DOMAINS=domains_str,
            SENIORITY_LEVELS=seniority_str
        )
        
        return prompt
    
    def process_text(
        self, 
        text_content: str, 
        filename: str,
        is_cv: bool = True
    ) -> Dict:
        """
        Process a single text document and return structured JSON.
        
        Args:
            text_content: Raw text from extraction stage
            filename: Original filename
            is_cv: True for CV, False for JD
            
        Returns:
            Structured JSON dict or error dict
        """
        print(f"Parsing: {filename} ({len(text_content)} chars)")
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                print(f"  Retry {attempt}/{self.max_retries}...")
                time.sleep(self.delay)
            
            try:
                # Create prompt
                prompt = self._create_prompt(text_content, filename)
                
                # Call Gemini API
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1 if attempt == 0 else 0.2,
                        "max_output_tokens": 4096,
                    }
                )
                
                # Extract JSON from response
                result = self._extract_json(response.text)
                
                # Validate result
                if self._validate_result(result, is_cv=is_cv):
                    if attempt > 0:
                        print(f"  ✓ Success after {attempt} retries")
                    return result
                
                # Invalid result - log and retry
                error_msg = result.get("error", "Invalid structure")
                print(f"  ✗ Attempt {attempt} failed: {error_msg}")
                
                if attempt == self.max_retries:
                    return {
                        "error": f"Failed after {self.max_retries} retries",
                        "last_error": error_msg,
                        "filename": filename
                    }
            
            except Exception as e:
                print(f"  ✗ Exception: {str(e)}")
                if attempt == self.max_retries:
                    return {
                        "error": f"Processing exception: {str(e)}",
                        "filename": filename
                    }
        
        return {"error": "Unexpected failure", "filename": filename}
    
    def process_multiple(
        self,
        text_dict: Dict[str, str],
        is_cv: bool = True,
        progress_callback=None
    ) -> Dict[str, Dict]:
        """
        Process multiple text documents in batch.
        
        Args:
            text_dict: Dict mapping filename -> text content
            is_cv: True for CVs, False for JDs
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            Dict mapping filename -> JSON result
        """
        results = {}
        total = len(text_dict)
        
        for idx, (filename, text_content) in enumerate(text_dict.items(), 1):
            if progress_callback:
                progress_callback(idx, total, filename)
            
            result = self.process_text(text_content, filename, is_cv=is_cv)
            results[filename] = result
            
            # Rate limiting between requests
            if idx < total:
                time.sleep(self.delay)
        
        return results


# Convenience functions for master orchestrator

def parse_jd(
    model,
    jd_text: str,
    jd_filename: str = "job_description.pdf",
    prompt_file: str = "prompts/stage_1_2.txt"
) -> Dict:
    """
    Parse a single Job Description from extracted text.
    
    Args:
        model: Gemini model instance
        jd_text: Extracted JD text
        jd_filename: Original filename
        prompt_file: Path to JD parsing prompt
        
    Returns:
        Structured JD JSON
    """
    parser = GeminiParser(model=model, prompt_file=prompt_file)
    return parser.process_text(jd_text, jd_filename, is_cv=False)


def parse_jd_from_pdf(
    model,
    jd_bytes: bytes,
    jd_filename: str = "job_description.pdf",
    prompt_file: str = "prompts/stage_1_2.txt"
) -> Dict:
    """
    Parse a single Job Description directly from PDF.
    
    Args:
        model: Gemini model instance
        jd_bytes: PDF file as bytes
        jd_filename: Original filename
        prompt_file: Path to JD parsing prompt
        
    Returns:
        Structured JD JSON
    """
    parser = GeminiParser(model=model, prompt_file=prompt_file)
    return parser.process_pdf(jd_bytes, jd_filename, is_cv=False)


def parse_cvs(
    model,
    cv_texts: Dict[str, str],
    prompt_file: str = "prompts/stage_1_1.txt",
    progress_callback=None
) -> Dict[str, Dict]:
    """
    Parse multiple CVs from extracted text in batch.
    
    Args:
        model: Gemini model instance
        cv_texts: Dict of {filename: text_content}
        prompt_file: Path to CV parsing prompt
        progress_callback: Optional progress function
        
    Returns:
        Dict of {filename: structured_json}
    """
    parser = GeminiParser(model=model, prompt_file=prompt_file)
    return parser.process_multiple(
        cv_texts, 
        is_cv=True, 
        progress_callback=progress_callback
    )


def parse_cvs_from_pdfs(
    model,
    cv_pdfs: Dict[str, bytes],
    prompt_file: str = "prompts/stage_1_1.txt",
    progress_callback=None
) -> Dict[str, Dict]:
    """
    Parse multiple CVs directly from PDFs in batch.
    
    Args:
        model: Gemini model instance
        cv_pdfs: Dict of {filename: pdf_bytes}
        prompt_file: Path to CV parsing prompt
        progress_callback: Optional progress function
        
    Returns:
        Dict of {filename: structured_json}
    """
    parser = GeminiParser(model=model, prompt_file=prompt_file)
    results = {}
    total = len(cv_pdfs)
    
    for idx, (filename, pdf_bytes) in enumerate(cv_pdfs.items(), 1):
        if progress_callback:
            progress_callback(idx, total, filename)
        
        result = parser.process_pdf(pdf_bytes, filename, is_cv=True)
        results[filename] = result
        
        # Rate limiting between requests
        if idx < total:
            time.sleep(parser.delay)
    
    return results


def parse_all(
    model,
    jd_text: str,
    cv_texts: Dict[str, str],
    jd_filename: str = "job_description.pdf",
    progress_callback=None
) -> tuple:
    """
    Convenience function to parse JD + all CVs.
    
    Args:
        model: Gemini model instance
        jd_text: JD text content
        cv_texts: Dict of CV texts
        jd_filename: JD filename
        progress_callback: Progress callback
        
    Returns:
        (jd_json, cv_jsons_dict) tuple
    """
    # Parse JD
    if progress_callback:
        progress_callback(0, 1, "Parsing Job Description...")
    jd_json = parse_jd(model, jd_text, jd_filename)
    
    # Parse CVs
    cv_jsons = parse_cvs(model, cv_texts, progress_callback=progress_callback)
    
    return jd_json, cv_jsons