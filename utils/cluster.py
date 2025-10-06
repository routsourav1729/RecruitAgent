import json
from collections import defaultdict
import os

def cluster_resumes_by_domain(input_file_path, output_file_path):
    """
    Reads a JSON file containing resume data, clusters the resumes by their 'domain',
    and saves the result to a new JSON file.

    The output JSON will have domains as keys and a list of source file names
    as values.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_file_path (str): The path to save the output JSON file.
    """
    # Use defaultdict to automatically handle the creation of new domain lists.
    # If a key is accessed for the first time, it will be initialized with an empty list.
    domain_clusters = defaultdict(list)

    try:
        # Open and read the input JSON file.
        # 'utf-8' encoding is specified for better compatibility.
        with open(input_file_path, 'r', encoding='utf-8') as f:
            resume_data = json.load(f)

        # Ensure the 'resumes' key exists and is a list before proceeding.
        if 'resumes' not in resume_data or not isinstance(resume_data['resumes'], list):
            print("Error: The input JSON file does not have the expected format. A 'resumes' key with a list of objects is required.")
            return

        # Iterate over each resume object in the 'resumes' list.
        for resume in resume_data['resumes']:
            # Safely access nested keys using .get() to avoid errors if a key is missing.
            # .get('gating_profile', {}) returns an empty dict if 'gating_profile' is not found.
            gating_profile = resume.get('gating_profile', {})
            domain = gating_profile.get('domain')
            source_file = resume.get('source_file')

            # Only add the resume to a cluster if both 'domain' and 'source_file' were found.
            if domain and source_file:
                domain_clusters[domain].append(source_file)
            else:
                # If a resume is missing critical info, add it to an 'Unknown' category.
                # This helps in debugging the source data.
                domain_clusters['Unknown_Domain_Or_Source_File'].append(source_file or 'Unknown_Source_File')


        # Write the resulting dictionary to the output file.
        # 'indent=4' makes the JSON output human-readable.
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(domain_clusters, f, indent=4)

        print(f"Successfully processed the file.")
        print(f"Clustered data has been saved to: {output_file_path}")

    except FileNotFoundError:
        print(f"Error: The input file was not found at '{input_file_path}'")
    except json.JSONDecodeError:
        print(f"Error: The file '{input_file_path}' is not a valid JSON file. Please check its content.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    # Set the name of your input file containing the resume data.
    input_filename = '/users/student/pg/pg23/souravrout/ALL_FILES/TIH/RESUME/groundtruth/json_gt/cv_ground_truth.json'
    # Set the desired name for the output file.
    output_filename = '/users/student/pg/pg23/souravrout/ALL_FILES/TIH/RESUME/clustered_by_domain.json'
    # --- End of Configuration ---

    # Get the directory of the current script to build full file paths.
    # This makes the script runnable from any location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_filename)
    output_path = os.path.join(script_dir, output_filename)

    # Execute the main function.
    cluster_resumes_by_domain(input_path, output_path)

