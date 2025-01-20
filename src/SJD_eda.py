import pandas as pd
import csv
from docx import Document
import re

def process_word_document(doc_path):
    # Open the Word document
    doc = Document(doc_path)
    cases = {}  # Dictionary to store all cases
    current_case = ""  # Variable to keep track of the current case being processed
    current_content = []  # List to store content of the current case

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        
        # Check if the paragraph is a new case identifier (e.g., NC_12345)
        if re.match(r'NC_\d+\+?$', text):
            # If we were processing a case, save it before moving to the new one
            if current_case:
                full_content = '\n'.join(current_content).strip()
                main_content, extended_content = split_content(full_content)
                cases[current_case] = {
                    'content': main_content,
                    'extended': extended_content
                }
            # Start a new case, removing '+' if present
            current_case = text.rstrip('+')
            current_content = []  # Reset content for the new case
        else:
            # If not a case identifier, add to current case's content
            current_content.append(text)

    # Add the last case (since the loop ends before processing it)
    if current_case:
        print("Processing last case")  # Debug print
        print(current_content)  # Debug print to see content of last case
        full_content = '\n'.join(current_content).strip()
        main_content, extended_content = split_content(full_content)
        cases[current_case] = {
            'content': main_content,
            'extended': extended_content
        }

    return cases

def split_content(full_content):
    # Markers that indicate the start of extended information
    extended_markers = ["[INFORMACIÓN AMPLIADA]", "[INFORMACIÓN AMPLIADA]:", "INFORMACIÓN AMPLIADA", "INFORMACIÓN AMPLIADA:"]
    for marker in extended_markers:
        if marker in full_content:
            # Split the content at the marker
            parts = full_content.split(marker)
            return parts[0].strip(), parts[1].strip()
    # If no marker is found, return the full content as main and None as extended
    return full_content, None

def create_csv(cases, metadata_path, output_path):
    # Read the metadata CSV file
    metadata = pd.read_csv(metadata_path, encoding='latin1', sep=';')
    
    # Debug prints to understand the structure of the metadata
    print("Column names in the metadata file:")
    print(metadata.columns)
    
    # Define column names for case identifier and diagnosis
    case_column = 'HC_anonimizada'
    diagnosis_column = 'Diagnóstico'
    
    # More debug prints
    print(f"Case column: {case_column}")
    print(f"Diagnosis column: {diagnosis_column}")
    print("\nFirst few rows of metadata:")
    print(metadata.head())
    
    print("\nCase numbers from Word document:")
    print(list(cases.keys())[:5])  # Print first 5 case numbers for debugging
    
    # Create and write to the output CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        # Write header row
        writer.writerow(['Caso', 'Descripción', 'Descripción Ampliada', 'Diagnóstico'])
        
        for case_num, content in cases.items():
            # Convert NC_ to HC_ for matching with metadata
            metadata_case_num = case_num.replace('NC_', 'HC_')
            # Find the corresponding row in metadata
            diagnosis_row = metadata.loc[metadata[case_column] == metadata_case_num]
            if not diagnosis_row.empty:
                diagnosis = diagnosis_row[diagnosis_column].values[0]
            else:
                print(f"Warning: No matching diagnosis found for case {case_num} (looking for {metadata_case_num})")
                diagnosis = "N/A"
            # Write the case data to CSV
            writer.writerow([case_num, content['content'], content['extended'], diagnosis])

    print(f"\nOutput file created: {output_path}")
    print("Please check the output file and the printed information to identify any mismatches.")

# Main execution
word_doc_path = 'SJD_cases/DxGPT_casos_NotesClíniques_final.docx'
metadata_path = 'SJD_cases/DxGPT_casos_metadata.csv'
output_path = 'SJD_cases/cases_with_diagnosis.csv'

# Process the Word document
cases = process_word_document(word_doc_path)
# Create the CSV file with cases and diagnoses
create_csv(cases, metadata_path, output_path)
