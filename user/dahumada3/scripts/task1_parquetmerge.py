import pandas as pd
import glob
import os

def merge_parquet_files(input_folder, output_file):
    """Merges multiple Parquet files into one without running out of memory."""
    
    print(f"Looking for Parquet files in: {input_folder}")
    parquet_files = glob.glob(os.path.join(input_folder, "*.parquet"))
    
    if not parquet_files:
        print("No Parquet files found. Exiting script.")
        return

    print(f"Found {len(parquet_files)} Parquet files. Starting merge...")

    try:
        # Read the first file to create the base structure
        first_file = parquet_files.pop(0)
        print(f"Processing first file: {first_file}")
        
        df = pd.read_parquet(first_file)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
        
        df.to_parquet(output_file, index=False, engine="fastparquet")
        print(f"First file saved as {output_file}")

        # Append remaining files one by one
        for file in parquet_files:
            print(f"Merging {file}...")
            df = pd.read_parquet(file)

            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype('category')

            df.to_parquet(output_file, index=False, compression="snappy", engine="fastparquet", append=True)

        print(f"Merge complete! Merged file saved as: {output_file}")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    input_folder = r'.\test-parquet'  # Adjust this path
    output_file = r'.\merged_output.parquet'
    
    merge_parquet_files(input_folder, output_file)
