#!/usr/bin/env python3
"""
Qwen Matcher (Stage 3) - Unified & Refactored

Compares a single parsed Job Description (JD) JSON against multiple parsed
Resume (CV) JSONs, generating a scored and ranked analysis for each candidate.

This script is designed to be called by a master orchestrator but can also run standalone.

Standalone Usage:
  # For a standard role analysis
  python qwen_matcher.py \
    --jd_file /path/to/parsed_jds/jd_1.json \
    --cv_file /path/to/parsed_cvs/all_cvs.json \
    --prompt_file /path/to/prompts/matching_prompt_standard.txt \
    --output_file /path/to/reports/report_for_jd_1.json \
    --gpus 0

  # For a senior role analysis
  python qwen_matcher.py \
    --jd_file /path/to/parsed_jds/jd_senior.json \
    --cv_file /path/to/parsed_cvs/all_cvs.json \
    --prompt_file /path/to/prompts/matching_prompt_senior.txt \
    --output_file /path/to/reports/report_for_jd_senior.json \
    --gpus 0
"""

import os
import json
import argparse
from typing import Dict, Any, List
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re

# --- Environment Setup ---
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["HF_HUB_OFFLINE"] = "1" 
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- Model Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
MODEL_CACHE_DIR = "/nas/backup/users/student/pg/pg23/souravrout/models"

class QwenMatcher:
    """
    A matcher that uses a pre-loaded Qwen model to compare a JD and a CV
    based on a provided prompt template, producing a JSON analysis.
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, raw_prompt_template: str, max_retries: int = 2):
        """
        Initializes the matcher with a pre-loaded model, tokenizer, and prompt.

        Args:
            model: The loaded Hugging Face CausalLM model instance.
            tokenizer: The loaded Hugging Face Tokenizer instance.
            raw_prompt_template: The raw string template loaded from an external file.
            max_retries: Maximum number of retries when JSON parsing fails.
        """
        print("QwenMatcher initialized.")
        self.model = model
        self.tokenizer = tokenizer
        self.max_retries = max_retries
        self.device = self.model.device

        # The code handles the model-specific chat format.
        prompt_parts = raw_prompt_template.split("[USER_PROMPT_BELOW]")
        self.system_prompt_template = prompt_parts[0].strip()
        self.user_prompt_template = prompt_parts[1].strip() if len(prompt_parts) > 1 else self.system_prompt_template

    def create_prompt(self, jd_json: Dict[str, Any], cv_json: Dict[str, Any]) -> str:
        """Creates the full, model-specific chat prompt by injecting the JD and CV JSON."""
        
        # Convert the JSON objects to formatted strings
        jd_str = json.dumps(jd_json, indent=2)
        cv_str = json.dumps(cv_json, indent=2)

        # Fill placeholders in the user prompt part
        user_content = self.user_prompt_template.format(
            JD_JSON=jd_str,
            CV_JSON=cv_str
        )
        
        # Assemble the final prompt in the Qwen chat format
        final_prompt = f"<|im_start|>system\n{self.system_prompt_template}<|im_end|>\n"
        final_prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        final_prompt += "<|im_start|>assistant\n```json\n"
        
        return final_prompt

    def _repair_json(self, json_str: str) -> str:
        """Applies a series of regex fixes to a broken JSON string."""
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
        return json_str

    def extract_json(self, text: str) -> Dict:
        """Extracts a JSON object from a model's response, prioritizing fenced code blocks."""
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                return {"error": "No JSON object found in response.", "context": text[:200]}
            json_str = text[start_idx:end_idx]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                repaired_json_str = self._repair_json(json_str)
                return json.loads(repaired_json_str)
            except json.JSONDecodeError as e:
                return {"error": f"Failed to parse JSON after repair: {str(e)}", "context": json_str[:200] + "..."}

    def process_match(self, jd_json: Dict[str, Any], cv_json: Dict[str, Any]) -> Dict:
        """
        Processes a single CV-JD pair and returns the JSON analysis.
        """
        candidate_file = cv_json.get("source_file", "unknown_cv.txt")
        print(f"Matching JD against '{candidate_file}'...")
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                print(f"Retry attempt {attempt}/{self.max_retries} for '{candidate_file}'...")
            
            prompt = self.create_prompt(jd_json, cv_json)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            try:
                with torch.no_grad():
                    temp = 0.1 if attempt == 0 else 0.15 + (attempt * 0.05)
                    outputs = self.model.generate(
                        inputs["input_ids"], max_new_tokens=4096, temperature=temp,
                        top_p=0.95, do_sample=True, pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                result = self.extract_json(response)
                
                # Semantic validation
                if "error" not in result and result.get("final_score") is not None:
                    return result
                
                print(f"Attempt {attempt} for '{candidate_file}' failed: {result.get('error', 'Invalid or empty response')}")
                if attempt == self.max_retries:
                    return {"error": f"Failed to get valid analysis for '{candidate_file}' after all retries.", "candidate_source_file": candidate_file}

            except Exception as e:
                print(f"A critical exception occurred during matching for '{candidate_file}': {str(e)}")
                if attempt == self.max_retries:
                    return {"error": f"Processing failed due to exception: {str(e)}", "candidate_source_file": candidate_file}
        
        return {"error": "Processing failed after all retries.", "candidate_source_file": candidate_file}

def load_model_and_tokenizer(gpus: str):
    """Loads the Qwen model and tokenizer with specified configurations."""
    print(f"--- Loading Model: {MODEL_NAME} ---")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR, local_files_only=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=quantization_config, device_map="auto",
            trust_remote_code=True, low_cpu_mem_usage=True, cache_dir=MODEL_CACHE_DIR,
            local_files_only=True, attn_implementation="flash_attention_2"
        )
        print("--- Model and Tokenizer loaded successfully! ---")
        return model, tokenizer
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to load model/tokenizer. Details: {e}\n")
        exit()

def run_matching_process(matcher: QwenMatcher, jd_filepath: str, cv_filepath: str, output_filepath: str):
    """Loads data, runs the matching for all candidates, and saves the report."""
    # Load the single JD JSON
    try:
        with open(jd_filepath, 'r', encoding='utf-8') as f:
            # The JD file might contain a top-level key (the filename), so we handle that.
            jd_data = json.load(f)
            jd_json = next(iter(jd_data.values())) if isinstance(jd_data, dict) and len(jd_data) == 1 else jd_data
    except (IOError, json.JSONDecodeError, StopIteration) as e:
        print(f"FATAL ERROR: Could not read or parse JD file: {jd_filepath}. Error: {e}")
        return

    # Load the CVs JSON which contains multiple CVs
    try:
        with open(cv_filepath, 'r', encoding='utf-8') as f:
            all_cvs = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"FATAL ERROR: Could not read or parse CVs file: {cv_filepath}. Error: {e}")
        return
        
    print(f"Loaded JD '{jd_json.get('source_file')}' and {len(all_cvs)} candidates from '{cv_filepath}'.")

    final_results = []
    for cv_filename, cv_json in tqdm(all_cvs.items(), desc="Matching Candidates"):
        if isinstance(cv_json, dict) and "error" not in cv_json:
            analysis_result = matcher.process_match(jd_json, cv_json)
            final_results.append(analysis_result)
        else:
            print(f"Skipping '{cv_filename}' due to previous parsing error.")
            final_results.append({"error": f"Skipped due to parsing error: {cv_json.get('error', 'Unknown')}", "candidate_source_file": cv_filename})

    # Sort results by final_score in descending order for an actionable report
    # Handle cases where score might be missing due to an error
    final_results.sort(key=lambda x: x.get("final_score", 0) if isinstance(x.get("final_score"), int) else 0, reverse=True)

    # Prepare the final report structure
    report = {
        "job_description_source": jd_json.get('source_file', 'unknown_jd.txt'),
        "job_title": jd_json.get('role_classification', {}).get('job_title', 'N/A'),
        "total_candidates_processed": len(all_cvs),
        "ranked_candidates": final_results
    }

    # Save the final ranked report
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nMatching complete. Ranked report saved to {output_filepath}")

def main():
    parser = argparse.ArgumentParser(description="Qwen Matcher for comparing a JD against multiple CVs.")
    parser.add_argument("--jd_file","-j", required=True, help="Path to the single parsed JD JSON file.")
    parser.add_argument("--cv_file", "-c", required=True, help="Path to the JSON file containing all parsed CVs.")
    parser.add_argument("--prompt_file", "-p", required=True, help="Path to the matching prompt template (.txt).")
    parser.add_argument("--output_file", "-o", required=True, help="Path to save the final ranked report JSON file.")
    parser.add_argument("--gpus","-g", help="Comma-separated list of GPU IDs to use.")
    parser.add_argument("--max_retries","-m", type=int, default=2, help="Maximum number of retries for failed analysis.")
    args = parser.parse_args()

    # --- Load Prompt ---
    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            raw_prompt_template = f.read()
    except FileNotFoundError:
        print(f"FATAL ERROR: Prompt file not found at '{args.prompt_file}'")
        exit()

    # --- Load Model (Simulating Orchestrator) ---
    model, tokenizer = load_model_and_tokenizer(gpus=args.gpus)

    # --- Initialize Matcher ---
    matcher_instance = QwenMatcher(
        model=model, tokenizer=tokenizer,
        raw_prompt_template=raw_prompt_template,
        max_retries=args.max_retries
    )
    
    # --- Run Matching Process ---
    run_matching_process(
        matcher=matcher_instance,
        jd_filepath=args.jd_file,
        cv_filepath=args.cv_file,
        output_filepath=args.output_file
    )

if __name__ == "__main__":
    main()
