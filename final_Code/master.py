#!/usr/bin/env python3
"""
Master Orchestrator - Streamlit UI

This script serves as the main user interface and controller for the end-to-end
AI recruitment pipeline. It handles multiple PDF uploads for CVs, 
orchestrates the four main stages of processing, and displays the final 
results interactively.
"""

import os
import subprocess
import json
from datetime import datetime
import tempfile
import shutil
import pandas as pd
import streamlit as st
import torch
import zipfile

# --- Configuration & Path Setup ---
# Base directory where all the code and prompts are located.
# We assume this script is placed inside the 'final_Code' directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "../prompts") 
OUTPUTS_DIR = os.path.join(BASE_DIR, "../run_outputs")

# --- Import Custom Modules ---
# UPDATED: Using the correct filenames from your project structure.
try:
    # Assuming 'stage_1_2.py' contains the QwenParser class and its helper functions
    from stage_1_2 import QwenParser, process_directory as run_parser_process, load_model_and_tokenizer
    # Assuming 'stage_3.py' contains the QwenMatcher class and its helper functions
    from stage_3 import QwenMatcher, run_matching_process
    # Assuming 'report_generator.py' contains the report generation function
    from report_generator import generate_consolidated_report
except ImportError as e:
    st.error(f"Failed to import a required module: {e}. Please ensure all required .py files (stage_1_2.py, stage_3.py, etc.) are in the same directory.")
    st.stop()


# --- Constants for Script and Prompt Names ---
# UPDATED: Reflects the actual filenames from your context.
JD_EXTRACTOR_SCRIPT = os.path.join(BASE_DIR, "jd_extractor.py")
CV_EXTRACTOR_SCRIPT = os.path.join(BASE_DIR, "resume_extractor.py")

PROMPT_CV_PARSE = os.path.join(PROMPTS_DIR, "stage_1_1.txt") # For CVs
PROMPT_JD_PARSE = os.path.join(PROMPTS_DIR, "stage_1_2.txt") # For JDs
PROMPT_MATCH_JUNIOR = os.path.join(PROMPTS_DIR, "stage_3_junior.txt") # For Manager & Below
PROMPT_MATCH_SENIOR = os.path.join(PROMPTS_DIR, "stage_3_senior.txt") # For Above Manager


# --- Helper Functions ---

def run_command(command: list, status_text):
    """A helper function to run external scripts and update the UI."""
    # Create a user-friendly command string for display
    display_command = ' '.join([os.path.basename(c) if os.path.exists(c) else c for c in command])
    status_text.text(f"Running: {display_command}...")
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        st.error(f"A subprocess failed: {display_command}")
        st.expander("Error Details").code(f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}")
        st.stop()
    except FileNotFoundError:
        st.error(f"Script not found at '{command[1]}'. Please check the path.")
        st.stop()

def display_scoring_criteria(seniority_choice: str):
    """Displays the scoring criteria in the Streamlit UI based on the user's choice."""
    st.subheader("Scoring Framework Used")
    if seniority_choice == "Above Manager Level":
        st.markdown("""
        - **Framework:** Senior Leadership
        - **Functional Gate:** Pass (50 pts) / Fail
        - **Strategic Experience & Leadership:** 35 pts
        - **Skills & Credentials Match:** 10 pts
        - **Education Fit:** 5 pts
        """)
    else:
        st.markdown("""
        - **Framework:** Standard / Manager Level
        - **Functional Gate:** Pass (45 pts) / Fail
        - **Experience Depth & Leadership:** 30 pts
        - **Key Skills Match:** 15 pts
        - **Education & Credentials:** 10 pts
        """)

# --- Main Pipeline Orchestration ---

def execute_pipeline(jd_file, cv_files, gpu_id: str, seniority_choice: str):
    """
    Executes the entire end-to-end recruitment pipeline and updates the UI.
    """
    # --- Create Dynamic Directory Structure for this run ---
    run_folder = os.path.join(OUTPUTS_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Define paths for all stages
    s0_txt_output_jd = os.path.join(run_folder, "stage0_txt", "jd")
    s0_txt_output_cv = os.path.join(run_folder, "stage0_txt", "cv")
    s1_json_output_jd = os.path.join(run_folder, "stage1_json", "jd")
    s1_json_output_cv = os.path.join(run_folder, "stage1_json", "cv")
    s3_matching_output = os.path.join(run_folder, "stage3_matching")
    s4_report_output = os.path.join(run_folder, "stage4_report")

    # Create all directories
    for path in [s0_txt_output_jd, s0_txt_output_cv, s1_json_output_jd, s1_json_output_cv, s3_matching_output, s4_report_output]:
        os.makedirs(path, exist_ok=True)

    # --- Save uploaded files to a temporary location for processing ---
    temp_jd_path = os.path.join(s0_txt_output_jd, jd_file.name)
    with open(temp_jd_path, "wb") as f:
        f.write(jd_file.getbuffer())

    # --- Handle the uploaded CV PDFs ---
    temp_cv_folder = os.path.join(run_folder, "temp_cv_uploads")
    os.makedirs(temp_cv_folder, exist_ok=True)
    for cv_file in cv_files:
        with open(os.path.join(temp_cv_folder, cv_file.name), "wb") as f:
            f.write(cv_file.getbuffer())
    st.info(f"Successfully saved {len(cv_files)} CVs for processing.")


    # --- UI Progress Updates ---
    progress_bar = st.progress(0)
    status_text = st.empty()

    # STAGE 0: PDF to TEXT Extraction
    status_text.info("Stage 0: Converting PDFs to Text...")
    run_command(["python", JD_EXTRACTOR_SCRIPT, "-t", temp_jd_path, "-o", s0_txt_output_jd], status_text)
    run_command(["python", CV_EXTRACTOR_SCRIPT, "-i", temp_cv_folder, "-o", s0_txt_output_cv], status_text)
    progress_bar.progress(10)

    # MODEL LIFECYCLE: Load Model (Once)
    status_text.info(f"Loading AI Model onto GPU: {gpu_id} (this may take a moment)...")
    model, tokenizer = load_model_and_tokenizer(gpus=gpu_id)
    progress_bar.progress(25)

    # STAGE 1 & 2: TEXT to JSON Parsing
    status_text.info("Stage 1 & 2: Parsing Job Description and Resumes...")
    with open(PROMPT_JD_PARSE, 'r', encoding='utf-8') as f:
        jd_prompt_template = f.read()
    jd_parser_agent = QwenParser(model, tokenizer, jd_prompt_template)
    run_parser_process(jd_parser_agent, s0_txt_output_jd, s1_json_output_jd, "jd.json")
    progress_bar.progress(40)

    with open(PROMPT_CV_PARSE, 'r', encoding='utf-8') as f:
        cv_prompt_template = f.read()
    cv_parser_agent = QwenParser(model, tokenizer, cv_prompt_template)
    run_parser_process(cv_parser_agent, s0_txt_output_cv, s1_json_output_cv, "cvs.json")
    progress_bar.progress(60)

    # STAGE 3: MATCHING
    status_text.info("Stage 3: Matching Candidates to Job Description...")
    prompt_to_use = PROMPT_MATCH_SENIOR if seniority_choice == "Above Manager Level" else PROMPT_MATCH_JUNIOR
    
    with open(prompt_to_use, 'r', encoding='utf-8') as f:
        matching_prompt_template = f.read()

    matcher_agent = QwenMatcher(model, tokenizer, matching_prompt_template)
    final_report_path = os.path.join(s3_matching_output, "matching_report.json")
    
    run_matching_process(
        matcher=matcher_agent,
        jd_filepath=os.path.join(s1_json_output_jd, "jd.json"),
        cv_filepath=os.path.join(s1_json_output_cv, "cvs.json"),
        output_filepath=final_report_path
    )
    progress_bar.progress(90)

    # MODEL LIFECYCLE: Unload Model
    status_text.info("Finalizing... Unloading AI Model.")
    del model, tokenizer
    torch.cuda.empty_cache()
    
    status_text.success("AI Analysis Complete!")
    progress_bar.progress(100)

    return final_report_path, seniority_choice, s4_report_output


# --- Streamlit App Main Function ---

def main():
    st.set_page_config(layout="wide", page_title="AI Recruitment Pipeline")
    st.title("🤖 AI Recruitment Pipeline Orchestrator")

    # --- UI Configuration Section ---
    st.header("1. Configure Pipeline")
    
    col1, col2 = st.columns(2)
    with col1:
        seniority_choice = st.radio(
            "Select Role Level for Matching",
            ("Manager & Below", "Above Manager Level"),
            help="This determines which scoring framework the AI will use for Stage 3."
        )
        gpu_id = st.selectbox("Select GPU for AI Model", options=[str(i) for i in range(torch.cuda.device_count())], help="Choose which GPU to run the AI model on.")

    with col2:
        st.markdown("**Set Reporting Filters**")
        use_score_filter = st.checkbox("Filter by Score", value=True)
        score_cutoff = st.slider("Minimum Score", 0, 100, 80, disabled=not use_score_filter)
        
        use_top_n_filter = st.checkbox("Filter by Top N")
        top_n = st.number_input("Number of top candidates", min_value=1, value=5, step=1, disabled=not use_top_n_filter)

    # --- File Uploader Section ---
    st.header("2. Upload Files")
    jd_file = st.file_uploader("Upload Job Description (JD) PDF", type="pdf")
    cv_files = st.file_uploader("Upload all Candidate CVs (PDFs)", type="pdf", accept_multiple_files=True)


    # --- Run Analysis Button ---
    if st.button("Run Analysis", disabled=(not jd_file or not cv_files)):
        with st.spinner("Pipeline is running... This will take several minutes."):
            st.session_state.report_path, st.session_state.seniority_choice, st.session_state.report_output_dir = execute_pipeline(jd_file, cv_files, gpu_id, seniority_choice)
            
            # Store the filter settings at the time of running
            st.session_state.final_top_n = top_n if use_top_n_filter else None
            st.session_state.final_score_above = score_cutoff if use_score_filter else None

    # --- STAGE 4: Reporting & UI Interaction ---
    if 'report_path' in st.session_state and os.path.exists(st.session_state.report_path):
        st.header("3. Analysis Results")

        try:
            with open(st.session_state.report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st.error("Could not load the final report file.")
            st.stop()
            
        display_scoring_criteria(st.session_state.seniority_choice)
        
        # --- Generate and Download Report ---
        st.subheader("Download Report")
        output_excel_path = os.path.join(st.session_state.report_output_dir, "Consolidated_Report.xlsx")
        
        # Generate the report using the filter settings saved from the run
        generate_consolidated_report(
            input_json_path=st.session_state.report_path,
            output_excel_path=output_excel_path,
            top_n=st.session_state.final_top_n,
            score_above=st.session_state.final_score_above
        )

        with open(output_excel_path, "rb") as f:
            st.download_button(
                label="📥 Download Excel Report",
                data=f,
                file_name="recruitment_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # --- Display Filtered Shortlist ---
        st.subheader("Candidate Shortlist")
        all_candidates = report_data.get("ranked_candidates", [])
        
        # Create a clean list of passed candidates first
        passed_candidates = []
        for cand in all_candidates:
            if "error" not in cand and cand.get("fit_flag") != "Reject - Functional Mismatch":
                passed_candidates.append(cand)

        if not passed_candidates:
            st.warning("No candidates passed the functional gate or all resulted in errors.")
        else:
            passed_df = pd.DataFrame(passed_candidates)
            
            # Apply the saved filters to the passed candidates
            display_df = passed_df
            if st.session_state.final_score_above is not None:
                display_df = display_df[display_df['final_score'] >= st.session_state.final_score_above]
            if st.session_state.final_top_n is not None:
                display_df = display_df.sort_values(by='final_score', ascending=False).head(st.session_state.final_top_n)

            if display_df.empty:
                st.warning("No candidates match the selected filter criteria.")
            else:
                # CORRECTED: This helper function robustly handles pros/cons
                def format_list_or_string(data):
                    if isinstance(data, list):
                        return ", ".join(data)
                    if isinstance(data, str):
                        return data
                    return "N/A"

                for index, row in display_df.iterrows():
                    with st.expander(f"**{row.get('candidate_source_file', 'N/A')}** - Score: {int(row.get('final_score', 0))}"):
                        qual_analysis = row.get('qualitative_analysis', {})
                        
                        st.markdown(f"**Summary:** {qual_analysis.get('summary', 'N/A')}")
                        st.markdown(f"**Pros:** {format_list_or_string(qual_analysis.get('pros'))}")
                        st.markdown(f"**Cons:** {format_list_or_string(qual_analysis.get('cons'))}")
                        st.json(row.get('score_breakdown', {}))

if __name__ == "__main__":
    main()

