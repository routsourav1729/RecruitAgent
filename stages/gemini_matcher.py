#!/usr/bin/env python3
"""
Gemini Matcher (Stage 3) - JD to CV Matching & Scoring

Compares a single parsed Job Description (JD) JSON against multiple parsed
Resume (CV) JSONs, generating a scored and ranked analysis for each candidate.

Designed to be called by master orchestrator.

Usage from master:
    from stages.gemini_matcher import match_candidates
    
    # Match all CVs against JD
    ranked_results = match_candidates(
        model=gemini_model,
        jd_json=jd_json,
        cv_jsons=cv_jsons_dict,
        seniority_level="balanced"  # or "senior"
    )
"""

import json
import re
import time
from typing import Dict, Any, List
from pathlib import Path


class GeminiMatcher:
    """
    Matches candidates (CVs) against a job description (JD) using Gemini API.
    Produces scored analysis with detailed breakdown.
    """
    
    def __init__(
        self,
        model,  # genai.GenerativeModel instance
        prompt_file: str,
        max_retries: int = 2,
        delay: float = 6.0
    ):
        """
        Initialize matcher with Gemini model and prompt template.
        
        Args:
            model: Pre-configured Gemini model instance (from master)
            prompt_file: Path to matching prompt template
            max_retries: Number of retry attempts for failed matching
            delay: Delay between retries (rate limiting)
        """
        self.model = model
        self.max_retries = max_retries
        self.delay = delay
        
        # Load and parse prompt template
        self.prompt_template = self._load_prompt_template(prompt_file)
        
        print(f"GeminiMatcher initialized with prompt: {Path(prompt_file).name}")
    
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
    
    def _create_prompt(self, jd_json: Dict[str, Any], cv_json: Dict[str, Any]) -> str:
        """
        Create final prompt by filling template with JD and CV JSONs.
        
        Args:
            jd_json: Parsed job description JSON
            cv_json: Parsed candidate resume JSON
            
        Returns:
            Complete prompt ready for Gemini
        """
        # Convert JSONs to formatted strings
        jd_str = json.dumps(jd_json, indent=2)
        cv_str = json.dumps(cv_json, indent=2)
        
        # Fill template placeholders
        prompt = self.prompt_template.format(
            JD_JSON=jd_str,
            CV_JSON=cv_str
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
    
    def _validate_result(self, result: Dict) -> bool:
        """
        Check if matching result has required fields.
        
        Args:
            result: Parsed matching JSON dict
            
        Returns:
            True if valid, False otherwise
        """
        if "error" in result:
            return False
        
        # Check for essential fields
        required_fields = ["candidate_source_file", "final_score", "score_breakdown"]
        
        for field in required_fields:
            if field not in result:
                return False
        
        # Validate score is a number
        if not isinstance(result.get("final_score"), (int, float)):
            return False
        
        return True
    
    def match_single(
        self,
        jd_json: Dict[str, Any],
        cv_json: Dict[str, Any]
    ) -> Dict:
        """
        Match a single CV against a JD and return scored analysis.
        
        Args:
            jd_json: Parsed job description
            cv_json: Parsed candidate resume
            
        Returns:
            Matching result with score breakdown
        """
        candidate_file = cv_json.get("source_file", "unknown_cv")
        print(f"\n{'='*50}")
        print(f"Matching: {candidate_file}")
        print(f"{'='*50}")
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                print(f"  Retry {attempt}/{self.max_retries}...")
                time.sleep(self.delay)
            
            try:
                # Create prompt
                prompt = self._create_prompt(jd_json, cv_json)
                prompt_length = len(prompt)
                print(f"  Prompt length: {prompt_length} chars (~{prompt_length//4} tokens)")
                
                # Call Gemini API with timeout handling
                print(f"  Sending request to Gemini API... (this may take 30-60s for large prompts)")
                start_time = time.time()
                
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.1 if attempt == 0 else 0.2,
                            "max_output_tokens": 4096,
                        },
                        request_options={"timeout": 120}  # 2 minute timeout
                    )
                    elapsed = time.time() - start_time
                    print(f"  ✓ Received response in {elapsed:.1f}s")
                except Exception as api_error:
                    elapsed = time.time() - start_time
                    print(f"  ✗ API call failed after {elapsed:.1f}s: {str(api_error)}")
                    raise
                
                # Extract JSON from response
                print(f"  Extracting JSON from response...")
                result = self._extract_json(response.text)
                print(f"  JSON extracted: {list(result.keys())}")
                
                # Validate result
                print(f"  Validating result...")
                is_valid = self._validate_result(result)
                print(f"  Validation result: {is_valid}")
                
                if is_valid:
                    if attempt > 0:
                        print(f"  ✓ Success after {attempt} retries")
                    print(f"  ✓ Match successful for {candidate_file}")
                    return result
                
                # Invalid result - log and retry
                error_msg = result.get("error", "Invalid structure or missing fields")
                print(f"  ✗ Attempt {attempt} failed: {error_msg}")
                print(f"  Available keys in result: {list(result.keys())}")
                
                if attempt == self.max_retries:
                    return {
                        "error": f"Failed after {self.max_retries} retries",
                        "last_error": error_msg,
                        "candidate_source_file": candidate_file
                    }
            
            except Exception as e:
                print(f"  ✗ Exception: {str(e)}")
                if attempt == self.max_retries:
                    return {
                        "error": f"Processing exception: {str(e)}",
                        "candidate_source_file": candidate_file
                    }
        
        return {
            "error": "Unexpected failure", 
            "candidate_source_file": candidate_file
        }
    
    def match_multiple(
        self,
        jd_json: Dict[str, Any],
        cv_jsons: Dict[str, Dict],
        progress_callback=None
    ) -> List[Dict]:
        """
        Match multiple CVs against a single JD.
        
        Args:
            jd_json: Parsed job description
            cv_jsons: Dict of {filename: cv_json}
            progress_callback: Optional callback(current, total, filename)
            
        Returns:
            List of matching results (unsorted)
        """
        results = []
        total = len(cv_jsons)
        
        for idx, (filename, cv_json) in enumerate(cv_jsons.items(), 1):
            if progress_callback:
                progress_callback(idx, total, filename)
            
            # Skip CVs that failed parsing
            if isinstance(cv_json, dict) and "error" in cv_json:
                print(f"Skipping {filename} - parsing error")
                results.append({
                    "error": f"Skipped: {cv_json.get('error')}",
                    "candidate_source_file": filename,
                    "final_score": 0
                })
                continue
            
            # Match CV against JD
            result = self.match_single(jd_json, cv_json)
            results.append(result)
            
            # Rate limiting between requests
            if idx < total:
                time.sleep(self.delay)
        
        return results


def match_candidates(
    model,
    jd_json: Dict[str, Any],
    cv_jsons: Dict[str, Dict],
    seniority_level: str = "balanced",
    progress_callback=None
) -> Dict:
    """
    Match all candidates against a job description and return ranked results.
    
    Args:
        model: Gemini model instance
        jd_json: Parsed job description JSON
        cv_jsons: Dict of {filename: cv_json}
        seniority_level: "balanced" for junior/manager, "senior" for above manager
        progress_callback: Optional progress function
        
    Returns:
        Complete matching report with ranked candidates
    """
    # Select appropriate prompt based on seniority
    if seniority_level.lower() in ["senior", "above_manager", "executive"]:
        prompt_file = "prompts/stage_3_senior.txt"
    else:
        prompt_file = "prompts/stage_3_junior.txt"
    
    # Initialize matcher
    matcher = GeminiMatcher(model=model, prompt_file=prompt_file)
    
    # Run matching
    if progress_callback:
        progress_callback(0, len(cv_jsons), "Starting candidate matching...")
    
    results = matcher.match_multiple(jd_json, cv_jsons, progress_callback)
    
    # Sort by score (descending)
    results.sort(
        key=lambda x: x.get("final_score", 0) if isinstance(x.get("final_score"), (int, float)) else 0,
        reverse=True
    )
    
    # Prepare final report structure
    report = {
        "job_description_source": jd_json.get("source_file", "unknown_jd"),
        "job_title": jd_json.get("role_classification", {}).get("job_title", "N/A"),
        "seniority_framework": seniority_level,
        "total_candidates_processed": len(cv_jsons),
        "ranked_candidates": results
    }
    
    return report


def get_shortlist(
    matching_report: Dict,
    min_score: int = None,
    top_n: int = None,
    exclude_errors: bool = True
) -> List[Dict]:
    """
    Filter matching results to get shortlist.
    
    Args:
        matching_report: Output from match_candidates()
        min_score: Minimum score threshold
        top_n: Maximum number of candidates
        exclude_errors: Skip candidates with errors
        
    Returns:
        Filtered list of candidates
    """
    candidates = matching_report.get("ranked_candidates", [])
    
    # Filter out errors if requested
    if exclude_errors:
        candidates = [c for c in candidates if "error" not in c]
    
    # Apply score filter
    if min_score is not None:
        candidates = [
            c for c in candidates 
            if isinstance(c.get("final_score"), (int, float)) and c["final_score"] >= min_score
        ]
    
    # Apply top N filter
    if top_n is not None:
        candidates = candidates[:top_n]
    
    return candidates