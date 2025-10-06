#!/usr/bin/env python3
"""
Report Generator (Stage 4) - Cloud-Ready

Generates Excel report from matching results.
Returns BytesIO for Streamlit download (no disk writes).

Usage from master:
    from stages.report_generator import generate_excel_report
    
    # Generate report
    excel_bytes = generate_excel_report(
        matching_report=st.session_state.matching_report,
        top_n=10,
        score_above=75
    )
    
    # Download in Streamlit
    st.download_button(
        "Download Report",
        data=excel_bytes,
        file_name="recruitment_report.xlsx"
    )
"""

import io
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def _safe_list_to_string(data) -> str:
    """Convert pros/cons to string, handling both list and string inputs."""
    if isinstance(data, list):
        return " | ".join(str(item) for item in data if item)
    elif isinstance(data, str):
        return data
    return "N/A"


def _safe_list_to_multiline(data) -> str:
    """Convert pros/cons to multiline string for detailed view."""
    if isinstance(data, list):
        return "\n".join(f"• {item}" for item in data if item)
    elif isinstance(data, str):
        return f"• {data}"
    return "N/A"


def _detect_score_categories(candidates: List[Dict]) -> List[str]:
    """
    Dynamically detect all score breakdown categories from data.
    Returns unique category names found across all candidates.
    """
    categories = set()
    
    for candidate in candidates:
        if "error" in candidate:
            continue
        
        breakdown = candidate.get("score_breakdown", {})
        if isinstance(breakdown, dict):
            categories.update(breakdown.keys())
    
    # Sort for consistent order (functional_relevance should be first)
    priority = ["functional_relevance"]
    sorted_cats = []
    
    for cat in priority:
        if cat in categories:
            sorted_cats.append(cat)
            categories.remove(cat)
    
    # Add rest alphabetically
    sorted_cats.extend(sorted(categories))
    
    return sorted_cats


def _filter_candidates(
    candidates: List[Dict],
    min_score: Optional[int] = None,
    top_n: Optional[int] = None,
    exclude_errors: bool = True
) -> List[Dict]:
    """
    Filter candidates based on criteria.
    """
    filtered = []
    
    for cand in candidates:
        # Exclude errors if requested
        if exclude_errors and "error" in cand:
            continue
        
        # Check min score
        score = cand.get("final_score", 0)
        if min_score is not None and isinstance(score, (int, float)):
            if score < min_score:
                continue
        
        filtered.append(cand)
    
    # Sort by score (descending)
    filtered.sort(
        key=lambda x: x.get("final_score", 0) if isinstance(x.get("final_score"), (int, float)) else 0,
        reverse=True
    )
    
    # Apply top N filter
    if top_n is not None:
        filtered = filtered[:top_n]
    
    return filtered


def generate_excel_report(
    matching_report: Dict,
    top_n: Optional[int] = None,
    score_above: Optional[int] = None
) -> bytes:
    """
    Generate Excel report from matching results.
    
    Args:
        matching_report: Output from Stage 3 (gemini_matcher)
        top_n: Show only top N candidates (optional)
        score_above: Show only candidates above this score (optional)
        
    Returns:
        Excel file as bytes (BytesIO) for download
    """
    
    # Extract data from report
    job_title = matching_report.get("job_title", "N/A")
    job_source = matching_report.get("job_description_source", "N/A")
    seniority = matching_report.get("seniority_framework", "N/A")
    total_processed = matching_report.get("total_candidates_processed", 0)
    all_candidates = matching_report.get("ranked_candidates", [])
    
    # Separate passed and rejected candidates
    passed_candidates = []
    rejected_candidates = []
    
    for cand in all_candidates:
        if "error" in cand:
            rejected_candidates.append({
                "Candidate File": cand.get("candidate_source_file", "Unknown"),
                "Reason": cand.get("error", "Processing error")
            })
        elif cand.get("fit_flag") == "Reject - Functional Mismatch":
            rejected_candidates.append({
                "Candidate File": cand.get("candidate_source_file", "Unknown"),
                "Reason": "Functional mismatch - rejected at gate"
            })
        else:
            passed_candidates.append(cand)
    
    # Calculate statistics
    stats = {
        "Job Title": job_title,
        "Job Description": job_source,
        "Seniority Framework": seniority.title(),
        "Total Candidates": total_processed,
        "Passed Functional Gate": len(passed_candidates),
        "Rejected/Errors": len(rejected_candidates)
    }
    
    if passed_candidates:
        scores = [c.get('final_score', 0) for c in passed_candidates 
                  if isinstance(c.get('final_score'), (int, float))]
        if scores:
            stats["Average Score"] = round(np.mean(scores), 1)
            stats["Median Score"] = int(np.median(scores))
            stats["Highest Score"] = int(np.max(scores))
            stats["Lowest Score"] = int(np.min(scores))
    
    # Apply filters for shortlist
    shortlist = _filter_candidates(
        passed_candidates,
        min_score=score_above,
        top_n=top_n,
        exclude_errors=True
    )
    
    # === SHEET 1: Dashboard with Shortlist ===
    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    
    shortlist_data = []
    for i, cand in enumerate(shortlist, 1):
        qual = cand.get("qualitative_analysis", {})
        shortlist_data.append({
            "Rank": i,
            "Candidate": cand.get("candidate_source_file", "Unknown"),
            "Score": cand.get("final_score", 0),
            "Summary": qual.get("summary", "N/A"),
            "Key Strengths": _safe_list_to_string(qual.get("pros", [])),
            "Key Concerns": _safe_list_to_string(qual.get("cons", []))
        })
    
    shortlist_df = pd.DataFrame(shortlist_data)
    
    # === SHEET 2: All Passed Candidates (Detailed) ===
    detailed_data = []
    score_categories = _detect_score_categories(passed_candidates)
    
    for cand in passed_candidates:
        qual = cand.get("qualitative_analysis", {})
        breakdown = cand.get("score_breakdown", {})
        
        row = {
            "Candidate": cand.get("candidate_source_file", "Unknown"),
            "Final Score": cand.get("final_score", 0),
            "Summary": qual.get("summary", "N/A"),
            "Strengths": _safe_list_to_multiline(qual.get("pros", [])),
            "Concerns": _safe_list_to_multiline(qual.get("cons", []))
        }
        
        # Add score breakdown columns dynamically
        for category in score_categories:
            cat_name = category.replace('_', ' ').title()
            cat_data = breakdown.get(category, {})
            
            if isinstance(cat_data, dict):
                row[f"{cat_name} - Score"] = cat_data.get("score", "N/A")
                row[f"{cat_name} - Justification"] = cat_data.get("justification", "N/A")
            else:
                row[f"{cat_name} - Score"] = "N/A"
                row[f"{cat_name} - Justification"] = "N/A"
        
        detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # === SHEET 3: Rejected Candidates ===
    rejected_df = pd.DataFrame(rejected_candidates)
    
    # === Write to Excel (in-memory) ===
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 14,
            'fg_color': '#D9E1F2',
            'border': 1
        })
        
        # === SHEET 1: Dashboard ===
        worksheet = workbook.add_worksheet('Dashboard')
        
        # Write title
        worksheet.write('A1', 'Recruitment Report - Overview', title_format)
        worksheet.set_row(0, 20)
        
        # Write statistics
        worksheet.write('A3', 'Statistics', header_format)
        worksheet.write('B3', 'Value', header_format)
        
        for idx, (metric, value) in enumerate(stats.items(), start=4):
            worksheet.write(f'A{idx}', metric)
            worksheet.write(f'B{idx}', value)
        
        # Write shortlist
        shortlist_start = len(stats) + 6
        worksheet.write(f'A{shortlist_start}', 'Top Candidates Shortlist', title_format)
        
        if not shortlist_df.empty:
            shortlist_df.to_excel(
                writer,
                sheet_name='Dashboard',
                startrow=shortlist_start,
                index=False
            )
            
            # Format shortlist headers
            for col_num, value in enumerate(shortlist_df.columns):
                worksheet.write(shortlist_start, col_num, value, header_format)
        
        # Set column widths
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:C', 15)
        worksheet.set_column('D:D', 50)
        worksheet.set_column('E:F', 40)
        
        # === SHEET 2: Detailed View ===
        if not detailed_df.empty:
            detailed_df.to_excel(writer, sheet_name='All Candidates', index=False)
            worksheet_detailed = writer.sheets['All Candidates']
            
            # Format headers
            for col_num, value in enumerate(detailed_df.columns):
                worksheet_detailed.write(0, col_num, value, header_format)
            
            # Set column widths
            worksheet_detailed.set_column('A:A', 25)  # Candidate
            worksheet_detailed.set_column('B:B', 12)  # Score
            worksheet_detailed.set_column('C:C', 50)  # Summary
            worksheet_detailed.set_column('D:E', 40)  # Strengths/Concerns
            
            # Score columns - narrower
            for col_num in range(5, len(detailed_df.columns)):
                col_name = detailed_df.columns[col_num]
                if 'Score' in col_name:
                    worksheet_detailed.set_column(col_num, col_num, 12)
                else:
                    worksheet_detailed.set_column(col_num, col_num, 50)
        
        # === SHEET 3: Rejected ===
        if not rejected_df.empty:
            rejected_df.to_excel(writer, sheet_name='Rejected', index=False)
            worksheet_rejected = writer.sheets['Rejected']
            
            # Format headers
            for col_num, value in enumerate(rejected_df.columns):
                worksheet_rejected.write(0, col_num, value, header_format)
            
            worksheet_rejected.set_column('A:A', 30)
            worksheet_rejected.set_column('B:B', 60)
    
    # Return bytes
    output.seek(0)
    return output.getvalue()


def generate_summary_stats(matching_report: Dict) -> Dict:
    """
    Generate summary statistics for display in UI.
    
    Returns:
        Dict with statistics for quick display
    """
    candidates = matching_report.get("ranked_candidates", [])
    
    passed = [c for c in candidates if "error" not in c and c.get("fit_flag") != "Reject - Functional Mismatch"]
    rejected = len(candidates) - len(passed)
    
    scores = [c.get('final_score', 0) for c in passed if isinstance(c.get('final_score'), (int, float))]
    
    return {
        "total": len(candidates),
        "passed": len(passed),
        "rejected": rejected,
        "avg_score": round(np.mean(scores), 1) if scores else 0,
        "top_score": max(scores) if scores else 0
    }