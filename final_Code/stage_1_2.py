#!/usr/bin/env python3
"""
Qwen Parser (Stages 1 & 2) - Unified & Refactored

A generic script to parse structured JSON from raw text files (CVs or JDs)
using a Qwen model and an external prompt template.

This script is designed to be called by a master orchestrator but can also run standalone.

Standalone Usage:
  # For CVs
  python qwen_parser.py \
    --input_dir /path/to/resumes/ \
    --output_dir /path/to/cv_output/ \
    --prompt_file /path/to/prompts/cv_prompt.txt \
    --output_filename all_cvs.json \
    --gpus 0

  # For JDs
  python qwen_parser.py \
    --input_dir /path/to/jds/ \
    --output_dir /path/to/jd_output/ \
    --prompt_file /path/to/prompts/jd_prompt.txt \
    --output_filename all_jds.json \
    --gpus 0

  # To repair a previous run
  python qwen_parser.py --repair --input_dir ... --output_dir ... --prompt_file ...
"""

import os
import json
import argparse
import glob
import re
from typing import Dict, List, Any
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Environment Setup ---
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# Set to "1" to use local files only, helpful in offline environments
os.environ["HF_HUB_OFFLINE"] = "1" 
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- Model Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
# IMPORTANT: Set this to the directory where your model is cached locally
MODEL_CACHE_DIR = "/nas/backup/users/student/pg/pg23/souravrout/models"

class QwenParser:
    """
    A generic parser that uses a pre-loaded Qwen model to extract structured
    JSON from text based on a provided prompt template.
    """
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, raw_prompt_template: str, max_retries: int = 2):
        """
        Initializes the parser with a pre-loaded model, tokenizer, and prompt.

        Args:
            model: The loaded Hugging Face CausalLM model instance.
            tokenizer: The loaded Hugging Face Tokenizer instance.
            raw_prompt_template: The raw string template loaded from an external file.
            max_retries: Maximum number of retries when JSON parsing fails.
        """
        print("QwenParser initialized.")
        self.model = model
        self.tokenizer = tokenizer
        self.max_retries = max_retries
        self.device = self.model.device

        # --- REFACTORED PROMPT HANDLING ---
        # The code now handles the model-specific chat format.
        prompt_parts = raw_prompt_template.split("[USER_PROMPT_BELOW]")
        self.system_prompt_template = prompt_parts[0].strip()
        if len(prompt_parts) > 1:
            self.user_prompt_template = prompt_parts[1].strip()
        else:
            # Fallback for prompts with no system/user split (like the original JD prompt)
            self.user_prompt_template = self.system_prompt_template
            self.system_prompt_template = "You are a helpful AI assistant." # Default system prompt

        print("System prompt template loaded.")
        print("User prompt template loaded.")
        # ------------------------------------

        # These can be injected into the prompt template
        self.DOMAINS = [
            "Sales & Business Development", "Credit & Risk Management",
            "Finance & Accounts (F&A)", "Operations", "Technology (IT)",
            "Human Resources (HR)", "Legal & Compliance",
            "Marketing & Communications", "Administration & Facilities", "Other"
        ]
        self.SENIORITY_LEVELS = [
            "Entry-Level / Officer", "Team Lead / Supervisor", "Manager",
            "Senior Manager / Lead", "Head of Department / VP", "Executive / C-Suite"
        ]

    def create_prompt(self, text_content: str, filename: str) -> str:
        """Creates the full, model-specific chat prompt by filling the templates."""
        
        # Fill placeholders in the user prompt part
        user_content = self.user_prompt_template.format(
            FILENAME=filename,
            TEXT_CONTENT=text_content,
            DOMAINS=', '.join(f'"{d}"' for d in self.DOMAINS),
            SENIORITY_LEVELS=', '.join(f'"{s}"' for s in self.SENIORITY_LEVELS)
        )
        
        # Assemble the final prompt in the Qwen chat format
        final_prompt = f"<|im_start|>system\n{self.system_prompt_template}<|im_end|>\n"
        final_prompt += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        final_prompt += "<|im_start|>assistant\n```json\n"
        
        return final_prompt

    def _repair_json(self, json_str: str) -> str:
        """Applies a series of regex fixes to a broken JSON string."""
        # Fix unescaped backslashes that are not part of a valid escape sequence
        json_str = re.sub(r'\\(?![/bfnrt"\\])', '/', json_str)
        # Fix trailing commas before closing brackets or braces
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
        # Fix keys without quotes (best effort)
        json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', json_str)
        return json_str

    def extract_json(self, text: str, attempt_repair: bool = True) -> Dict:
        """
        Extracts a JSON object from a model's response, prioritizing fenced code blocks.
        """
        # Most reliable method: find the JSON within ```json ... ```
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback for cases where the model forgets the fence
            assistant_start = text.rfind("<|im_start|>assistant")
            if assistant_start != -1:
                text = text[assistant_start:] # Reduce search space
            
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                return {"error": "No JSON object found in response.", "context": text[:200]}
            json_str = text[start_idx:end_idx]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if attempt_repair:
                print("Initial JSON parsing failed. Attempting to repair...")
                try:
                    repaired_json_str = self._repair_json(json_str)
                    return json.loads(repaired_json_str)
                except json.JSONDecodeError as e2:
                    return {"error": f"Failed to parse JSON after repair: {str(e2)}", "context": json_str[:200] + "..."}
            return {"error": f"Failed to parse JSON: {str(e)}", "context": json_str[:200] + "..."}

    def process_text(self, text_content: str, filename: str) -> Dict:
        """
        Processes a single text content (from a CV or JD) and returns structured JSON.
        """
        print(f"Processing text from '{filename}' with length: {len(text_content)} chars")
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                print(f"Retry attempt {attempt}/{self.max_retries} for '{filename}'...")
            
            prompt = self.create_prompt(text_content, filename)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            try:
                with torch.no_grad():
                    # Slightly increase temperature on retries to encourage different output
                    temp = 0.1 if attempt == 0 else 0.15 + (attempt * 0.05)
                    
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=4096,
                        temperature=temp,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode only the newly generated tokens
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                result = self.extract_json(response, attempt_repair=True)
                
                # Semantic validation: check if the model just returned a placeholder
                if "error" not in result:
                    # A simple check to see if any key field is empty or a placeholder
                    if not result.get("source_file") or not (result.get("candidate_name") or result.get("role_classification")):
                         result = {"error": "Logical error: Model returned an empty or placeholder template."}
                    else:
                        if attempt > 0:
                            print(f"Successfully parsed JSON for '{filename}' after {attempt} retries.")
                        return result # Success

                # If we are here, it means there was an error
                print(f"Attempt {attempt} for '{filename}' failed: {result.get('error', 'Unknown error')}")
                if attempt == self.max_retries:
                    print(f"Failed to get valid data for '{filename}' after all retries.")
                    return result

            except Exception as e:
                print(f"A critical exception occurred during generation for '{filename}': {str(e)}")
                if attempt == self.max_retries:
                    return {"error": f"Processing failed due to exception: {str(e)}"}
        
        return {"error": "Processing failed after all retries."} # Should not be reached

def load_model_and_tokenizer(gpus: str):
    """Loads the Qwen model and tokenizer with specified configurations."""
    print(f"--- Loading Model: {MODEL_NAME} ---")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    device_map = "auto"
    
    try:
        print(f"Attempting to load tokenizer from cache: {MODEL_CACHE_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Attempting to load model from cache: {MODEL_CACHE_DIR}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            cache_dir=MODEL_CACHE_DIR,
            local_files_only=True,
            attn_implementation="flash_attention_2"
        )
        print("--- Model and Tokenizer loaded successfully! ---")
        return model, tokenizer
    except Exception as e:
        print("\n" + "="*50)
        print("FATAL ERROR: Failed to load the model or tokenizer.")
        print(f"Error details: {e}")
        print("Please check that model files exist in the cache directory and dependencies are installed.")
        print(f"Cache directory: '{MODEL_CACHE_DIR}'")
        print("="*50 + "\n")
        exit()

def load_existing_results(output_path: str) -> Dict:
    """Loads existing results from a JSON file to support repair mode."""
    if not os.path.exists(output_path):
        return {}
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load or parse existing results file '{output_path}'. Starting fresh. Error: {e}")
        return {}

def process_directory(parser_instance: QwenParser, input_dir: str, output_dir: str, output_filename: str, repair_mode: bool = False):
    """
    Processes all .txt files in a directory using the provided parser instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    all_results = {}
    files_to_process = []

    if repair_mode:
        print(f"** REPAIR MODE **: Analyzing {output_path} for errors.")
        existing_results = load_existing_results(output_path)
        all_results = existing_results.copy()
        error_files = {fname for fname, res in existing_results.items() if isinstance(res, dict) and "error" in res}
        
        if not error_files:
            print("No files with errors found. Nothing to repair.")
            return
            
        print(f"Found {len(error_files)} files to repair: {', '.join(error_files)}")
        all_txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
        files_to_process = [f for f in all_txt_files if os.path.basename(f) in error_files]
    else:
        print(f"Searching for .txt files in: {input_dir}")
        files_to_process = glob.glob(os.path.join(input_dir, "*.txt"))

    if not files_to_process:
        print("No files to process found.")
        if not repair_mode: # Only write empty file if it's not a repair run
             with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)
        return

    print(f"Found {len(files_to_process)} files to process.")
    for filepath in tqdm(files_to_process, desc="Processing files"):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
            
            filename = os.path.basename(filepath)
            result = parser_instance.process_text(text_content, filename)
            all_results[filename] = result
            
            status = "Success" if "error" not in result else f"Error: {result.get('error')}"
            print(f"Completed processing: {filename} - Status: {status}")

        except Exception as e:
            print(f"Critical error processing file {filepath}: {str(e)}")
            all_results[os.path.basename(filepath)] = {"error": f"File processing error: {str(e)}"}

    print(f"\nProcessing complete. Saving all results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Unified Qwen Parser for structured data extraction from text.")
    parser.add_argument("--input_dir", "--i", required=True, help="Directory containing input .txt files (CVs or JDs).")
    parser.add_argument("--output_dir", "--o", required=True, help="Directory to save the final JSON output file.")
    parser.add_argument("--prompt_file", "--p", required=True, help="Path to the text file containing the prompt template.")
    parser.add_argument("--output_filename", "--f", required=True, help="Name for the output JSON file (e.g., all_cvs.json).")
    parser.add_argument("--gpus", "--g", default="0", help="Comma-separated list of GPU IDs to use.")
    parser.add_argument("--max_retries", "--m", type=int, default=2, help="Maximum number of retries for failed processing.")
    parser.add_argument("--repair","-r", action="store_true", help="Enable repair mode to re-process files that failed previously.")
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

    # --- Initialize Parser ---
    parser_instance = QwenParser(
        model=model,
        tokenizer=tokenizer,
        raw_prompt_template=raw_prompt_template, # Pass the raw template
        max_retries=args.max_retries
    )
    
    # --- Process Directory ---
    process_directory(
        parser_instance=parser_instance,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        repair_mode=args.repair
    )

if __name__ == "__main__":
    main()
