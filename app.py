#!/usr/bin/env python3
"""
AI Recruitment Agent - Master Orchestrator
Cloud-Ready Streamlit Application

Deploys on: Streamlit Cloud
Architecture: 100% in-memory, API-based, no local dependencies
"""

import streamlit as st
import google.generativeai as genai
import json
import time
from datetime import datetime

# Import all stages
from stages.local_pdf_extractor import extract_jd, extract_cvs, extract_cvs_from_zip
from stages.gemini_extractor import extract_with_gemini
from stages.gemini_parser import parse_jd, parse_cvs
from stages.gemini_matcher import match_candidates
from stages.report_generator import generate_excel_report, generate_summary_stats

# Page config
st.set_page_config(
    page_title="AI Recruitment Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .stage-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 0
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = None
if 'cv_texts' not in st.session_state:
    st.session_state.cv_texts = {}
if 'jd_json' not in st.session_state:
    st.session_state.jd_json = None
if 'cv_jsons' not in st.session_state:
    st.session_state.cv_jsons = {}
if 'matching_report' not in st.session_state:
    st.session_state.matching_report = None


def initialize_gemini():
    """Initialize Gemini API from secrets."""
    try:
        api_key = st.secrets["api"]["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {e}")
        st.info("Please add your GEMINI_API_KEY to Streamlit secrets.")
        st.stop()


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Recruitment Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Intelligent CV-JD Matching with Advanced Analytics**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Model Display
        st.info("**Model:** Gemini 2.0 Flash")
        
        # Seniority selection
        seniority = st.radio(
            "Role Seniority Level",
            ["Manager & Below", "Above Manager"],
            help="Determines scoring framework"
        )
        
        st.markdown("---")
        
        # Report filters
        st.subheader("📊 Report Filters")
        score_filter = st.slider("Minimum Score", 0, 100, 70)
        top_n = st.number_input("Top N Candidates", 1, 50, 10)
        
        st.markdown("---")
        st.caption("Powered by Gemini 2.0 Flash")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "⚙️ Process", "📊 Results", "🔍 Debug"])
    
    # TAB 1: UPLOAD
    with tab1:
        st.header("Step 1: Upload Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Job Description")
            jd_file = st.file_uploader(
                "Upload JD PDF",
                type=['pdf'],
                key='jd_upload',
                help="Upload the job description PDF"
            )
            if jd_file:
                st.success(f"✓ {jd_file.name} uploaded ({jd_file.size // 1024} KB)")
        
        with col2:
            st.subheader("Candidate Resumes")
            cv_option = st.radio(
                "Upload Format",
                ["Multiple PDFs", "ZIP File"],
                horizontal=True
            )
            
            if cv_option == "Multiple PDFs":
                cv_files = st.file_uploader(
                    "Upload CV PDFs",
                    type=['pdf'],
                    accept_multiple_files=True,
                    key='cv_upload'
                )
                if cv_files:
                    st.success(f"✓ {len(cv_files)} CVs uploaded")
            else:
                cv_zip = st.file_uploader(
                    "Upload ZIP containing CVs",
                    type=['zip'],
                    key='zip_upload'
                )
                if cv_zip:
                    st.success(f"✓ ZIP uploaded ({cv_zip.size // 1024} KB)")
        
        st.markdown("---")
        
        # Extraction method
        st.subheader("Extraction Method")
        extraction_method = st.radio(
            "Choose extraction approach",
            ["⚡ Fast (Local Processing)", "🎯 Accurate (Gemini AI)"],
            horizontal=True,
            help="Fast: Instant, good for clean PDFs | Accurate: AI-powered, better for scanned/complex PDFs"
        )
        
        # Process button
        can_process = jd_file and ((cv_option == "Multiple PDFs" and cv_files) or (cv_option == "ZIP File" and cv_zip))
        
        if st.button("🚀 Start Processing", disabled=not can_process, type="primary", use_container_width=True):
            st.session_state.stage = 1
            st.session_state.jd_file = jd_file
            st.session_state.cv_option = cv_option
            st.session_state.cv_files = cv_files if cv_option == "Multiple PDFs" else None
            st.session_state.cv_zip = cv_zip if cv_option == "ZIP File" else None
            st.session_state.extraction_method = extraction_method
            st.session_state.seniority = seniority
            st.rerun()
    
    # TAB 2: PROCESS
    with tab2:
        if st.session_state.stage >= 1:
            st.header("Processing Pipeline")
            
            # Initialize Gemini
            model = initialize_gemini()
            
            # Progress tracking
            progress_container = st.container()
            
            with progress_container:
                # STAGE 0: Extraction
                with st.expander("📄 Stage 0: PDF Text Extraction", expanded=True):
                    stage0_status = st.empty()
                    stage0_progress = st.progress(0)
                    
                    if st.session_state.extraction_method == "⚡ Fast (Local Processing)":
                        stage0_status.info("Using local extraction (fast)...")
                        
                        # Extract JD
                        jd_result = extract_jd(
                            st.session_state.jd_file.read(),
                            st.session_state.jd_file.name
                        )
                        stage0_progress.progress(30)
                        
                        # Extract CVs
                        if st.session_state.cv_option == "ZIP File":
                            cv_results = extract_cvs_from_zip(
                                st.session_state.cv_zip.read(),
                                progress_callback=lambda i, t, f: stage0_status.text(f"Extracting {i}/{t}: {f}")
                            )
                        else:
                            cv_file_tuples = [(f.name, f.read()) for f in st.session_state.cv_files]
                            cv_results = extract_cvs(
                                cv_file_tuples,
                                progress_callback=lambda i, t, f: stage0_status.text(f"Extracting {i}/{t}: {f}")
                            )
                        stage0_progress.progress(100)
                        
                    else:  # Gemini extraction
                        stage0_status.info("Using Gemini AI extraction (accurate)...")
                        
                        jd_bytes = st.session_state.jd_file.read()
                        
                        if st.session_state.cv_option == "ZIP File":
                            jd_result, cv_results = extract_with_gemini(
                                model=model,
                                jd_file=jd_bytes,
                                cv_zip=st.session_state.cv_zip.read(),
                                progress_callback=lambda i, t, f: stage0_status.text(f"Extracting {i}/{t}: {f}")
                            )
                        else:
                            cv_file_tuples = [(f.name, f.read()) for f in st.session_state.cv_files]
                            jd_result, cv_results = extract_with_gemini(
                                model=model,
                                jd_file=jd_bytes,
                                cv_files=cv_file_tuples,
                                progress_callback=lambda i, t, f: stage0_status.text(f"Extracting {i}/{t}: {f}")
                            )
                        stage0_progress.progress(100)
                    
                    # Store in session
                    st.session_state.jd_text = jd_result['text']
                    st.session_state.cv_texts = {k: v['text'] for k, v in cv_results.items() if 'text' in v}
                    
                    stage0_status.success(f"✓ Extracted 1 JD + {len(st.session_state.cv_texts)} CVs")
                
                # STAGE 1-2: Parsing
                with st.expander("🧠 Stage 1-2: Intelligent Parsing", expanded=True):
                    stage12_status = st.empty()
                    stage12_progress = st.progress(0)
                    
                    stage12_status.info("Parsing Job Description...")
                    st.session_state.jd_json = parse_jd(
                        model=model,
                        jd_text=st.session_state.jd_text,
                        jd_filename=st.session_state.jd_file.name
                    )
                    stage12_progress.progress(30)
                    
                    stage12_status.info(f"Parsing {len(st.session_state.cv_texts)} CVs...")
                    st.session_state.cv_jsons = parse_cvs(
                        model=model,
                        cv_texts=st.session_state.cv_texts,
                        progress_callback=lambda i, t, f: stage12_status.text(f"Parsing {i}/{t}: {f}")
                    )
                    stage12_progress.progress(100)
                    
                    stage12_status.success("✓ Parsing complete")
                
                # STAGE 3: Matching
                with st.expander("🎯 Stage 3: AI Matching & Scoring", expanded=True):
                    stage3_status = st.empty()
                    stage3_progress = st.progress(0)
                    
                    seniority_key = "senior" if st.session_state.seniority == "Above Manager" else "balanced"
                    
                    stage3_status.info("Matching candidates against JD...")
                    st.session_state.matching_report = match_candidates(
                        model=model,
                        jd_json=st.session_state.jd_json,
                        cv_jsons=st.session_state.cv_jsons,
                        seniority_level=seniority_key,
                        progress_callback=lambda i, t, f: (
                            stage3_status.text(f"Analyzing {i}/{t}: {f}"),
                            stage3_progress.progress(int(i/t*100))
                        )
                    )
                    
                    stage3_status.success("✓ Matching complete")
                    stage3_progress.progress(100)
                
                # Success
                st.balloons()
                st.success("🎉 All stages complete! View results in the Results tab.")
                st.session_state.stage = 2
        
        else:
            st.info("Upload documents in the Upload tab to begin processing.")
    
    # TAB 3: RESULTS
    with tab3:
        if st.session_state.stage >= 2:
            st.header("📊 Analysis Results")
            
            # Summary stats
            stats = generate_summary_stats(st.session_state.matching_report)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Candidates", stats['total'])
            with col2:
                st.metric("Passed Gate", stats['passed'])
            with col3:
                st.metric("Average Score", f"{stats['avg_score']:.1f}")
            with col4:
                st.metric("Top Score", stats['top_score'])
            
            st.markdown("---")
            
            # Download report
            st.subheader("📥 Download Report")
            
            excel_bytes = generate_excel_report(
                matching_report=st.session_state.matching_report,
                top_n=top_n,
                score_above=score_filter
            )
            
            job_title = st.session_state.matching_report.get('job_title', 'recruitment')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_bytes,
                file_name=f"recruitment_report_{job_title}_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Candidate shortlist
            st.subheader("🏆 Top Candidates")
            
            candidates = st.session_state.matching_report.get('ranked_candidates', [])
            passed = [c for c in candidates if 'error' not in c and c.get('fit_flag') != 'Reject - Functional Mismatch']
            
            # Apply filters
            filtered = [c for c in passed if c.get('final_score', 0) >= score_filter]
            filtered.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            filtered = filtered[:top_n]
            
            if not filtered:
                st.warning("No candidates match the selected filters.")
            else:
                for idx, cand in enumerate(filtered, 1):
                    qual = cand.get('qualitative_analysis', {})
                    score = cand.get('final_score', 0)
                    
                    with st.expander(f"#{idx} | {cand.get('candidate_source_file', 'Unknown')} | Score: {score}"):
                        st.markdown(f"**Summary:** {qual.get('summary', 'N/A')}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**✅ Strengths:**")
                            pros = qual.get('pros', [])
                            if isinstance(pros, list):
                                for pro in pros:
                                    st.markdown(f"• {pro}")
                            else:
                                st.markdown(f"• {pros}")
                        
                        with col_b:
                            st.markdown("**⚠️ Concerns:**")
                            cons = qual.get('cons', [])
                            if isinstance(cons, list):
                                for con in cons:
                                    st.markdown(f"• {con}")
                            else:
                                st.markdown(f"• {cons}")
                        
                        st.markdown("**Score Breakdown:**")
                        st.json(cand.get('score_breakdown', {}))
        else:
            st.info("Complete processing to view results.")
    
    # TAB 4: DEBUG
    with tab4:
        st.header("🔍 Debug & Inspection")
        
        if st.session_state.stage >= 2:
            # Sample extraction
            st.subheader("Sample Extracted Text")
            sample_cv = list(st.session_state.cv_texts.keys())[0] if st.session_state.cv_texts else None
            
            if sample_cv:
                with st.expander("📄 JD Extracted Text (first 500 chars)"):
                    st.text(st.session_state.jd_text[:500] + "...")
                
                with st.expander(f"📄 Sample CV: {sample_cv} (first 500 chars)"):
                    st.text(st.session_state.cv_texts[sample_cv][:500] + "...")
            
            st.markdown("---")
            
            # Sample JSON
            st.subheader("Sample Parsed JSON")
            
            with st.expander("📋 JD JSON"):
                st.json(st.session_state.jd_json)
            
            if sample_cv and sample_cv in st.session_state.cv_jsons:
                with st.expander(f"📋 Sample CV JSON: {sample_cv}"):
                    st.json(st.session_state.cv_jsons[sample_cv])
            
            st.markdown("---")
            
            # Full matching report
            st.subheader("Complete Matching Report")
            with st.expander("View Full Report JSON"):
                st.json(st.session_state.matching_report)
        else:
            st.info("Process documents to view debug information.")


if __name__ == "__main__":
    main()