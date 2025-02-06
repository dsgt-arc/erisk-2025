import pandas as pd
import glob
from pathlib import Path
import unicodedata, re, itertools, sys
import os

# Control character removal setup
all_chars = (chr(i) for i in range(sys.maxunicode))
categories = {'Cc'}
control_chars = ''.join(map(chr, itertools.chain(range(0x00, 0x20), range(0x7f, 0xa0))))
control_char_re = re.compile('[%s]' % re.escape(control_chars))

def remove_control_chars(s):
    return control_char_re.sub('', s)

def process_trec_file(file_path, output_folder):
    """Process a single .trec file and save as a Parquet file."""
    try:
        with open(file_path, 'r', encoding="utf8") as f:
            xml = f.read()
            xml = xml.replace('&<', '&amp;<')  # Fixing possible XML errors
            xml = remove_control_chars(xml)
            xml = '<ROOT>' + xml + '</ROOT>'  # Adding a root tag
            
            # Parse XML and convert to DataFrame
            df = pd.read_xml(xml)

            # Optimize memory usage
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype('category')  # Convert text columns to categorical

            # Save to Parquet
            filename = Path(file_path).stem
            output_path = os.path.join(output_folder, f"{filename}.parquet")
            df.to_parquet(output_path, index=False)

            print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def trec2parquet():
    """Processes all .trec files efficiently to Parquet format."""
    path = r'.\task1-symptom-ranking\erisk25-t1-dataset'  # Input folder path
    output_folder = r'.\test-parquet'  # Output folder for Parquet files
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    all_files = glob.glob(path + "/*.trec")

    for file_ in all_files:
        process_trec_file(file_, output_folder)

if __name__ == "__main__":
    trec2parquet()
