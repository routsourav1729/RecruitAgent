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
import os
from pathlib import Path
from datetime import datetime

# Import all stages
from stages.local_pdf_extractor import extract_jd, extract_cvs, extract_cvs_from_zip
from stages.gemini_parser import parse_jd, parse_cvs, parse_jd_from_pdf, parse_cvs_from_pdfs
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
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background: #fff3cd;
        border: 1px solid #ffc107;
        color: #856404;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize ALL session state variables at startup
def init_session_state():
    """Initialize all session state variables to prevent KeyError"""
    defaults = {
        'stage': 0,
        'jd_text': None,
        'cv_texts': {},
        'jd_json': None,
        'cv_jsons': {},
        'matching_report': None,
        'jd_file_content': None,
        'jd_file_name': None,
        'cv_files_content': None,
        'cv_zip_content': None,
        'extraction_method': None,
        'seniority': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()


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


def load_demo_files():
    """Load demo files from demo_data folder."""
    demo_path = Path("demo_data")
    
    if not demo_path.exists():
        st.error("Demo data folder not found. Please add demo_data/ to your project.")
        return None, None
    
    # Load JD
    jd_path = demo_path / "jd"
    jd_files = list(jd_path.glob("*.pdf"))
    
    if not jd_files:
        st.error("No demo JD found in demo_data/jd/")
        return None, None
    
    jd_file = jd_files[0]  # Take first JD
    
    # Load CVs
    cv_path = demo_path / "cvs"
    cv_files = list(cv_path.glob("*.pdf"))
    
    if not cv_files:
        st.error("No demo CVs found in demo_data/cvs/")
        return None, None
    
    return jd_file, cv_files


def handle_api_error(error_msg: str):
    """Display user-friendly API error messages"""
    if "429" in error_msg or "quota" in error_msg.lower():
        st.error("🚫 **API Quota Exceeded**")
        st.markdown("""
        <div class="warning-box">
        <strong>You've reached your daily API limit for Gemini.</strong><br><br>
        
        <strong>Options:</strong>
        <ul>
            <li>⏰ Wait 24 hours for quota reset</li>
            <li>💳 Upgrade to paid tier at <a href="https://ai.google.dev/pricing" target="_blank">Google AI Studio</a></li>
            <li>⚡ Use local extraction only (faster, no API calls for extraction)</li>
        </ul>
        
        <strong>Tip:</strong> The free tier allows 50 requests per day. Each CV processing uses 1-2 requests.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"❌ API Error: {error_msg}")
        st.info("Please check your API key and try again.")


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
        
        # Report filters with clear explanation
        st.subheader("📊 Report Filters")
        st.caption("Filter results by score (0-100) **and/or** number of candidates")
        
        score_filter = st.slider(
            "Minimum Score (out of 100)", 
            0, 100, 70,
            help="Only show candidates with score above this threshold"
        )
        
        top_n = st.number_input(
            "Maximum Number of Candidates", 
            1, 50, 10,
            help="Show at most this many top candidates"
        )
        
        st.markdown("---")
        st.caption("Powered by Gemini 2.0 Flash")
    
    # STAGE 0: UPLOAD
    if st.session_state.stage == 0:
        st.header("📤 Step 1: Upload Documents")
        
        # Demo vs Custom mode
        st.markdown('<div class="info-box">💡 <strong>Tip:</strong> Try Demo mode first to see how it works, then switch to Custom for your own files.</div>', unsafe_allow_html=True)
        
        upload_mode = st.radio(
            "Select Mode",
            ["🎬 Demo", "📁 Custom"],
            horizontal=True,
            help="Demo: Use sample files | Custom: Upload your own files"
        )
        
        if upload_mode == "🎬 Demo":
            st.markdown("---")
            st.subheader("Demo Mode")
            st.info("📋 Demo files will be loaded from `demo_data/` folder")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Demo Contents:**")
                demo_jd, demo_cvs = load_demo_files()
                if demo_jd and demo_cvs:
                    st.write(f"- Job Description: `{demo_jd.name}`")
                    st.write(f"- Resumes: {len(demo_cvs)} file(s)")
                    for cv in demo_cvs:
                        st.write(f"  - `{cv.name}`")
            
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("🚀 Run Demo", type="primary", use_container_width=True):
                    if demo_jd and demo_cvs:
                        # Load demo files into session state
                        with open(demo_jd, 'rb') as f:
                            st.session_state.jd_file_content = f.read()
                            st.session_state.jd_file_name = demo_jd.name
                        
                        st.session_state.cv_files_content = []
                        for cv in demo_cvs:
                            with open(cv, 'rb') as f:
                                st.session_state.cv_files_content.append((cv.name, f.read()))
                        
                        st.session_state.cv_zip_content = None  # Not using ZIP in demo
                        st.session_state.extraction_method = "⚡ Fast (Local Processing)"
                        st.session_state.seniority = seniority
                        st.session_state.stage = 1
                        st.rerun()
                    else:
                        st.error("Failed to load demo files.")
        
        else:  # Custom mode
            st.markdown("---")
            st.subheader("Custom Upload")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Job Description")
                jd_file = st.file_uploader(
                    "Upload JD PDF",
                    type=['pdf'],
                    key='jd_upload',
                    help="Upload the job description PDF file"
                )
                if jd_file:
                    st.success(f"✓ {jd_file.name} ({jd_file.size // 1024} KB)")
            
            with col2:
                st.markdown("##### Candidate Resumes")
                cv_files_or_zip = st.file_uploader(
                    "Upload CVs (Multiple PDFs or one ZIP file)",
                    type=['pdf', 'zip'],
                    accept_multiple_files=True,
                    key='cv_upload',
                    help="Upload multiple PDF files or a single ZIP file containing PDFs"
                )
                
                if cv_files_or_zip:
                    # Check if it's a ZIP or multiple PDFs
                    if len(cv_files_or_zip) == 1 and cv_files_or_zip[0].name.endswith('.zip'):
                        st.success(f"✓ ZIP file uploaded: {cv_files_or_zip[0].name} ({cv_files_or_zip[0].size // 1024} KB)")
                    else:
                        st.success(f"✓ {len(cv_files_or_zip)} PDF(s) uploaded")
            
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
            can_process = jd_file and cv_files_or_zip
            
            if st.button("🚀 Start Processing", disabled=not can_process, type="primary", use_container_width=True):
                # Store files in session state
                st.session_state.jd_file_content = jd_file.read()
                st.session_state.jd_file_name = jd_file.name
                
                # Handle ZIP or multiple PDFs
                if len(cv_files_or_zip) == 1 and cv_files_or_zip[0].name.endswith('.zip'):
                    st.session_state.cv_zip_content = cv_files_or_zip[0].read()
                    st.session_state.cv_files_content = None
                else:
                    st.session_state.cv_files_content = [(f.name, f.read()) for f in cv_files_or_zip]
                    st.session_state.cv_zip_content = None
                
                st.session_state.extraction_method = extraction_method
                st.session_state.seniority = seniority
                st.session_state.stage = 1
                st.rerun()
    
    # STAGE 1: PROCESSING
    elif st.session_state.stage == 1:
        st.header("⚙️ Processing Pipeline")
        
        # Initialize Gemini
        try:
            model = initialize_gemini()
        except Exception as e:
            st.error(f"Failed to initialize Gemini: {e}")
            if st.button("🔄 Go Back to Upload"):
                st.session_state.stage = 0
                st.rerun()
            st.stop()
        
        # Progress tracking
        progress_container = st.container()
        
        try:
            with progress_container:
                
                if st.session_state.extraction_method == "⚡ Fast (Local Processing)":
                    # FAST OPTION: Stage 0 → Extract text, then Stage 1-2 → Parse text
                    
                    # STAGE 0: Extraction
                    with st.expander("📄 Stage 0: PDF Text Extraction", expanded=True):
                        stage0_status = st.empty()
                        stage0_progress = st.progress(0)
                        
                        stage0_status.info("Using local extraction (fast)...")
                        
                        # Extract JD
                        jd_result = extract_jd(
                            st.session_state.jd_file_content,
                            st.session_state.jd_file_name
                        )
                        stage0_progress.progress(30)
                        
                        # Extract CVs
                        if st.session_state.cv_zip_content:
                            cv_results = extract_cvs_from_zip(
                                st.session_state.cv_zip_content,
                                progress_callback=lambda i, t, f: stage0_status.text(f"Extracting {i}/{t}: {f}")
                            )
                        else:
                            cv_results = extract_cvs(
                                st.session_state.cv_files_content,
                                progress_callback=lambda i, t, f: stage0_status.text(f"Extracting {i}/{t}: {f}")
                            )
                        stage0_progress.progress(100)
                        
                        # Store extracted text
                        st.session_state.jd_text = jd_result['text']
                        st.session_state.cv_texts = {k: v['text'] for k, v in cv_results.items() if 'text' in v}
                        
                        stage0_status.success(f"✓ Extracted 1 JD + {len(st.session_state.cv_texts)} CVs")
                    
                    # STAGE 1-2: Parsing from TEXT
                    with st.expander("🧠 Stage 1-2: Intelligent Parsing", expanded=True):
                        stage12_status = st.empty()
                        stage12_progress = st.progress(0)
                        
                        stage12_status.info("Parsing Job Description from text...")
                        st.session_state.jd_json = parse_jd(
                            model=model,
                            jd_text=st.session_state.jd_text,
                            jd_filename=st.session_state.jd_file_name
                        )
                        stage12_progress.progress(30)
                        
                        stage12_status.info(f"Parsing {len(st.session_state.cv_texts)} CVs from text...")
                        st.session_state.cv_jsons = parse_cvs(
                            model=model,
                            cv_texts=st.session_state.cv_texts,
                            progress_callback=lambda i, t, f: stage12_status.text(f"Parsing {i}/{t}: {f}")
                        )
                        stage12_progress.progress(100)
                        
                        stage12_status.success("✓ Parsing complete")
                
                else:
                    # ACCURATE OPTION: Parse PDFs directly (no extraction stage)
                    
                    with st.expander("🎯 Stage 1-2: Direct PDF Parsing with Gemini", expanded=True):
                        stage12_status = st.empty()
                        stage12_progress = st.progress(0)
                        
                        stage12_status.info("Parsing Job Description PDF directly...")
                        st.session_state.jd_json = parse_jd_from_pdf(
                            model=model,
                            jd_bytes=st.session_state.jd_file_content,
                            jd_filename=st.session_state.jd_file_name
                        )
                        stage12_progress.progress(30)
                        
                        # Prepare CV PDFs
                        cv_pdfs = {}
                        if st.session_state.cv_zip_content:
                            import zipfile
                            import io
                            from pathlib import Path
                            
                            stage12_status.info("Extracting PDFs from ZIP...")
                            with zipfile.ZipFile(io.BytesIO(st.session_state.cv_zip_content), 'r') as zip_ref:
                                for file_info in zip_ref.filelist:
                                    if file_info.is_dir():
                                        continue
                                    if not file_info.filename.lower().endswith('.pdf'):
                                        continue
                                    
                                    pdf_bytes = zip_ref.read(file_info.filename)
                                    filename = Path(file_info.filename).name
                                    cv_pdfs[filename] = pdf_bytes
                        else:
                            for fname, fbytes in st.session_state.cv_files_content:
                                cv_pdfs[fname] = fbytes
                        
                        stage12_status.info(f"Parsing {len(cv_pdfs)} CV PDFs directly...")
                        st.session_state.cv_jsons = parse_cvs_from_pdfs(
                            model=model,
                            cv_pdfs=cv_pdfs,
                            progress_callback=lambda i, t, f: stage12_status.text(f"Parsing PDF {i}/{t}: {f}")
                        )
                        stage12_progress.progress(100)
                        
                        # For debug purposes, set empty text placeholders
                        st.session_state.jd_text = "[Parsed directly from PDF - no intermediate text]"
                        st.session_state.cv_texts = {k: "[Parsed directly from PDF]" for k in cv_pdfs.keys()}
                        
                        stage12_status.success("✓ Direct PDF parsing complete")
                
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
                st.success("🎉 All stages complete! Preparing results...")
                time.sleep(2)  # Brief pause for user to see success
                st.session_state.stage = 2
                st.rerun()
        
        except Exception as e:
            error_msg = str(e)
            st.error("❌ Processing failed")
            handle_api_error(error_msg)
            
            if st.button("🔄 Go Back to Upload"):
                st.session_state.stage = 0
                st.rerun()
            st.stop()
    
    # STAGE 2: RESULTS
    elif st.session_state.stage == 2:
        st.header("📊 Analysis Results")
        
        # Download Report Section
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
        
        # Summary stats
        st.subheader("📈 Summary Statistics")
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
        
        # Candidate shortlist
        st.subheader("🏆 Top Candidates")
        
        candidates = st.session_state.matching_report.get('ranked_candidates', [])
        passed = [c for c in candidates if 'error' not in c and c.get('fit_flag') != 'Reject - Functional Mismatch']
        
        # Apply filters
        filtered = [c for c in passed if c.get('final_score', 0) >= score_filter]
        filtered.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        filtered = filtered[:top_n]
        
        if not filtered:
            st.warning("No candidates match the selected filters. Try adjusting the filters in the sidebar.")
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
                    
                    st.markdown("---")
                    st.markdown("**📊 Score Breakdown:**")
                    breakdown = cand.get('score_breakdown', {})
                    for category, details in breakdown.items():
                        if isinstance(details, dict):
                            cat_name = category.replace('_', ' ').title()
                            cat_score = details.get('score', 'N/A')
                            cat_just = details.get('justification', 'N/A')
                            st.markdown(f"**{cat_name}:** {cat_score}")
                            st.caption(cat_just)
        
        st.markdown("---")
        
        # Optional: View parsed data
        with st.expander("🔍 View Parsed Data (Debug)"):
            tab_jd, tab_cv, tab_report = st.tabs(["JD JSON", "Sample CV JSON", "Full Report"])
            
            with tab_jd:
                st.json(st.session_state.jd_json)
            
            with tab_cv:
                sample_cv = list(st.session_state.cv_jsons.keys())[0] if st.session_state.cv_jsons else None
                if sample_cv:
                    st.json(st.session_state.cv_jsons[sample_cv])
                else:
                    st.info("No CV data available")
            
            with tab_report:
                st.json(st.session_state.matching_report)
        
        st.markdown("---")
        
        # Start Over button
        if st.button("🔄 Start Over", type="secondary", use_container_width=True):
            # Reset session state
            init_session_state()
            st.session_state.stage = 0
            st.rerun()


if __name__ == "__main__":
    main()