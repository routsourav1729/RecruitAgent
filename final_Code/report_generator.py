import json
import pandas as pd
import argparse
import numpy as np

def generate_consolidated_report(input_json_path: str, output_excel_path: str, top_n: int = None, score_above: int = None):
    """
    Loads the final matching JSON, processes it, and generates a multi-sheet
    consolidated Excel report for HR. This version is more robust against
    missing data from the AI model.
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"FATAL ERROR: Could not read or parse input JSON file '{input_json_path}'. Error: {e}")
        return

    job_title = data.get("job_title", "N/A")
    ranked_candidates = data.get("ranked_candidates", [])

    passed_candidates = []
    rejected_candidates = []

    for cand in ranked_candidates:
        # Check for errors or the rejection flag to separate candidates
        if "error" in cand or cand.get("fit_flag") == "Reject - Functional Mismatch":
            rejected_candidates.append({
                "Candidate File": cand.get("candidate_source_file", "Unknown File"),
                "Reason for Rejection": cand.get("error", cand.get("score_breakdown", {}).get("functional_relevance", {}).get("justification", "Functional mismatch or processing error."))
            })
        else:
            passed_candidates.append(cand)

    # --- Prepare Data for Excel Sheets ---
    
    # 1. Dashboard Sheet Data
    stats = {
        "Job Title": job_title,
        "Total Candidates Processed": data.get("total_candidates_processed", 0),
        "Passed Functional Gate": len(passed_candidates),
        "Rejected or Errored": len(rejected_candidates),
    }
    if passed_candidates:
        scores = [c.get('final_score', 0) for c in passed_candidates]
        stats["Average Score (Passed)"] = round(np.mean(scores), 2)
        stats["Median Score (Passed)"] = int(np.median(scores))
        stats["Highest Score"] = int(np.max(scores))
    
    # Apply filters for the shortlist
    shortlist_candidates = passed_candidates
    if score_above is not None:
        shortlist_candidates = [c for c in shortlist_candidates if c.get('final_score', 0) >= score_above]
    if top_n is not None:
        # Ensure sorting before taking the top N
        shortlist_candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        shortlist_candidates = shortlist_candidates[:top_n]

    dashboard_shortlist_data = [{
        "Rank": i + 1,
        "Candidate File": c.get("candidate_source_file"),
        "Final Score": c.get("final_score"),
        "Summary": c.get("qualitative_analysis", {}).get("summary", ""),
        "Key Pros": " | ".join(c.get("qualitative_analysis", {}).get("pros", [])),
        "Key Cons": " | ".join(c.get("qualitative_analysis", {}).get("cons", []))
    } for i, c in enumerate(shortlist_candidates)]

    # 2. Detailed Sheet Data (Robust Version)
    detailed_data = []
    # Define all possible columns to ensure consistency
    score_categories = [
        "functional_relevance", "experience_depth_and_leadership", 
        "strategic_experience_and_leadership", "key_skills_match", 
        "key_skills_and_credentials", "education_and_credentials", "education_fit"
    ]

    for c in passed_candidates:
        row = {
            "Candidate File": c.get("candidate_source_file"),
            "Final Score": c.get("final_score"),
            "Summary": c.get("qualitative_analysis", {}).get("summary", ""),
            "Pros": "\n".join(c.get("qualitative_analysis", {}).get("pros", [])),
            "Cons": "\n".join(c.get("qualitative_analysis", {}).get("cons", []))
        }
        
        score_breakdown = c.get("score_breakdown", {})
        
        # Safely get score and justification for each category
        for category in score_categories:
            cat_name = category.replace('_', ' ').title()
            details = score_breakdown.get(category, {}) # Get the sub-dictionary safely
            
            row[f"{cat_name} Score"] = details.get("score", "N/A")
            row[f"{cat_name} Justification"] = details.get("justification", "N/A")
            
        detailed_data.append(row)

    # --- Create DataFrames ---
    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    dashboard_shortlist_df = pd.DataFrame(dashboard_shortlist_data)
    detailed_df = pd.DataFrame(detailed_data)
    rejected_df = pd.DataFrame(rejected_candidates)

    # --- Write to Excel ---
    print(f"Generating Excel report at: {output_excel_path}")
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        # Dashboard Sheet
        stats_df.to_excel(writer, sheet_name='Dashboard', index=False, startrow=1)
        if not dashboard_shortlist_df.empty:
            dashboard_shortlist_df.to_excel(writer, sheet_name='Dashboard', index=False, startrow=len(stats_df) + 4)
        
        workbook  = writer.book
        worksheet = writer.sheets['Dashboard']
        header_format = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
        
        worksheet.write('A1', 'Overall Statistics', header_format)
        worksheet.set_column('A:B', 30)
        if not dashboard_shortlist_df.empty:
            worksheet.write(f'A{len(stats_df) + 4}', 'Candidate Shortlist', header_format)
            for i, col in enumerate(dashboard_shortlist_df.columns):
                worksheet.write(len(stats_df) + 4, i, col, header_format)
            worksheet.set_column('C:C', 15)
            worksheet.set_column('D:F', 50)

        # Detailed Sheet
        if not detailed_df.empty:
            detailed_df.to_excel(writer, sheet_name='All Passed Candidates', index=False)
            worksheet_detailed = writer.sheets['All Passed Candidates']
            for i, col in enumerate(detailed_df.columns):
                worksheet_detailed.write(0, i, col, header_format)
            worksheet_detailed.set_column('A:E', 30)
            worksheet_detailed.set_column('F:Z', 50)

        # Rejected Sheet
        if not rejected_df.empty:
            rejected_df.to_excel(writer, sheet_name='Rejected Candidates', index=False)
            worksheet_rejected = writer.sheets['Rejected Candidates']
            for i, col in enumerate(rejected_df.columns):
                worksheet_rejected.write(0, i, col, header_format)
            worksheet_rejected.set_column('A:B', 50)

    print("--- Report generation complete! ---")

def main():
    parser = argparse.ArgumentParser(description="Generate a consolidated Excel report from the matcher's JSON output.")
    parser.add_argument("--input_json", required=True, help="Path to the matching_report.json file.")
    parser.add_argument("--output_file", required=True, help="Path to save the final .xlsx report.")
    parser.add_argument("--top_n", type=int, help="Filter dashboard to show only the top N candidates.")
    parser.add_argument("--score_above", type=int, help="Filter dashboard to show candidates with a score above this value.")
    args = parser.parse_args()

    generate_consolidated_report(args.input_json, args.output_file, args.top_n, args.score_above)

if __name__ == "__main__":
    main()
