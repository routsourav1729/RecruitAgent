import os
import json
import shutil

def organize_pdfs_by_domain(json_file_path, source_folder, output_folder):
    """
    Organizes PDF files into subdirectories based on a JSON file.

    This function reads a JSON object where keys are category names (domains)
    and values are lists of PDF filenames. It then creates a directory for each
    domain in the specified output folder and copies the corresponding PDFs from
    the source folder into these new directories.

    Args:
        json_file_path (str): The full path to the JSON file containing the domain clusters.
        source_folder (str): The full path to the directory where all the source PDFs are stored.
        output_folder (str): The full path to the directory where the organized folders and
                             PDFs will be created.
    """
    # --- 1. Validate Paths and Read JSON ---
    if not os.path.isfile(json_file_path):
        print(f"Error: JSON file not found at '{json_file_path}'")
        return

    if not os.path.isdir(source_folder):
        print(f"Error: Source folder not found at '{source_folder}'")
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            domain_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. Please check the file format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the JSON file: {e}")
        return

    # --- 2. Create Base Output Directory ---
    # os.makedirs with exist_ok=True will not raise an error if the directory already exists.
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output will be saved in: '{os.path.abspath(output_folder)}'")

    # --- 3. Iterate Through Domains and Copy Files ---
    files_copied = 0
    files_missing = 0

    # The .items() method allows us to loop through the keys (domains) and values (filenames) at the same time.
    for domain, filenames in domain_data.items():
        # Create a safe, OS-agnostic path for the new domain-specific folder.
        domain_folder_path = os.path.join(output_folder, domain)
        os.makedirs(domain_folder_path, exist_ok=True)
        print(f"\nProcessing domain: '{domain}'...")

        if not isinstance(filenames, list):
            print(f"  - Warning: Expected a list of filenames for domain '{domain}', but found {type(filenames)}. Skipping.")
            continue

        for filename in filenames:
            source_file_path = os.path.join(source_folder, filename)
            destination_file_path = os.path.join(domain_folder_path, filename)

            # Check if the source PDF actually exists before trying to copy it.
            if os.path.exists(source_file_path):
                try:
                    # shutil.copy2 preserves more metadata (like timestamps) than shutil.copy.
                    shutil.copy2(source_file_path, destination_file_path)
                    files_copied += 1
                except Exception as e:
                    print(f"  - Error copying '{filename}': {e}")
            else:
                print(f"  - Warning: File not found in source folder and was skipped: '{filename}'")
                files_missing += 1

    # --- 4. Final Report ---
    print("\n---------------------------------")
    print("         Process Complete        ")
    print("---------------------------------")
    print(f"Successfully copied: {files_copied} files.")
    print(f"Missing from source: {files_missing} files.")
    print("---------------------------------")


if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Please update these paths before running the script.

    # 1. Path to your JSON file.
    # Assumes the JSON file is in the same directory as the script.
    # If it's elsewhere, provide the full path e.g., "C:/Users/YourUser/Documents/clustered_data.json"
    JSON_FILE = '/users/student/pg/pg23/souravrout/ALL_FILES/TIH/RESUME/final_testing/utils/clustered_jd.json'

    # 2. Path to the folder containing ALL of your source PDF files.
    # Example for Windows: "C:/Users/YourUser/Documents/All_Resumes"
    # Example for Mac/Linux: "/home/YourUser/Documents/All_Resumes"
    # SOURCE_PDF_FOLDER = '/users/student/pg/pg23/souravrout/ALL_FILES/TIH/RESUME/JD_CV/data/input/resumes/resume_500'
    SOURCE_PDF_FOLDER = '/users/student/pg/pg23/souravrout/ALL_FILES/TIH/RESUME/JD_CV/data/input/job_descriptions/jd_315'

    # 3. Path to the folder where you want the organized output to be created.
    # The script will create this folder if it doesn't exist.
    # Example for Windows: "C:/Users/YourUser/Documents/Organized_Resumes"
    # Example for Mac/Linux: "/home/YourUser/Documents/Organized_Resumes"
    OUTPUT_FOLDER = '/users/student/pg/pg23/souravrout/ALL_FILES/TIH/RESUME/final_testing/data/input/jd'
    # --- End of Configuration ---

    # Get the directory of the current script to resolve relative paths.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, JSON_FILE)

    # Execute the main function with your configured paths.
    organize_pdfs_by_domain(json_path, SOURCE_PDF_FOLDER, OUTPUT_FOLDER)
