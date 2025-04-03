import os
import pandas as pd
from tqdm import tqdm
import json

def load_metadata(metadata_path):
    """
    Load the MIMIC-CXR metadata CSV file.
    This includes patient IDs, study IDs, and image paths.
    """
    df = pd.read_csv(metadata_path)
    df = df[df['ViewPosition'].isin(['PA', 'AP'])]  # Keep only frontal views
    df = df.dropna(subset=['StudyInstanceUID', 'DicomPath'])
    return df


def load_reports(report_path):
    """
    Load reports from a JSONL file, where each line is a dict:
    {"study_id": ..., "report": ...}
    """
    reports = {}
    with open(report_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            reports[entry['study_id']] = entry['report']
    return reports


def match_image_report(df_meta, reports):
    """
    Align reports and metadata by study_id and return a DataFrame.
    """
    matched = []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        study_id = str(row['StudyInstanceUID'])
        if study_id in reports:
            matched.append({
                'subject_id': row['SubjectID'],
                'study_id': study_id,
                'image_path': row['DicomPath'],
                'report': reports[study_id]
            })

    return pd.DataFrame(matched)


def preprocess_and_save(meta_csv, reports_json, output_csv):
    """
    Main preprocessing pipeline.
    """
    print("Loading metadata...")
    df_meta = load_metadata(meta_csv)

    print("Loading reports...")
    reports = load_reports(reports_json)

    print("Matching images with reports...")
    df_matched = match_image_report(df_meta, reports)

    print(f"Saving {len(df_matched)} matched pairs to {output_csv}")
    df_matched.to_csv(output_csv, index=False)


if __name__ == "__main__":
    # Example paths (to be replaced by actual paths)
    metadata_csv = "data/mimic-cxr-2.0.0-metadata.csv"
    reports_jsonl = "data/mimic-cxr-reports.jsonl"
    output_file = "data/matched_image_report_pairs.csv"

    preprocess_and_save(metadata_csv, reports_jsonl, output_file)
