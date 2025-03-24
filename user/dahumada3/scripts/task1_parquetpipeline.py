import pandas as pd
import pyarrow.parquet as pq
import glob
import os
import re
import itertools
from pathlib import Path
from tqdm import tqdm


class ParquetPipeline:
    def __init__(
        self,
        trec_input_dir,
        intermediate_dir,
        merged_file,
        final_output_dir,
        partition_size_mb=256,
    ):
        self.trec_input_dir = trec_input_dir
        self.intermediate_dir = intermediate_dir
        self.merged_file = merged_file
        self.final_output_dir = final_output_dir
        self.partition_size_mb = partition_size_mb
        self._prepare_dirs()

        # Precompile control char cleaner
        control_chars = "".join(
            map(chr, itertools.chain(range(0x00, 0x20), range(0x7F, 0xA0)))
        )
        self.control_char_re = re.compile("[%s]" % re.escape(control_chars))

    def _prepare_dirs(self):
        os.makedirs(self.intermediate_dir, exist_ok=True)
        os.makedirs(self.final_output_dir, exist_ok=True)

    def remove_control_chars(self, s):
        return self.control_char_re.sub("", s)

    def convert_trec_to_parquet(self):
        all_files = glob.glob(os.path.join(self.trec_input_dir, "*.trec"))
        print(f"Found {len(all_files)} TREC files to process...")
        for file_path in all_files:
            try:
                with open(file_path, "r", encoding="utf8") as f:
                    xml = f.read()
                    xml = xml.replace("&<", "&amp;<")
                    xml = self.remove_control_chars(xml)
                    xml = "<ROOT>" + xml + "</ROOT>"

                    df = pd.read_xml(xml)
                    for col in df.select_dtypes(include=["object"]).columns:
                        df[col] = df[col].astype("category")

                    filename = Path(file_path).stem
                    output_path = os.path.join(
                        self.intermediate_dir, f"{filename}.parquet"
                    )
                    df.to_parquet(output_path, index=False)

                    print(f"✓ Saved: {output_path}")
            except Exception as e:
                print(f"✗ Error processing {file_path}: {e}")

    def merge_parquet_files(self):
        print(f"Merging Parquet files from: {self.intermediate_dir}")
        parquet_files = glob.glob(os.path.join(self.intermediate_dir, "*.parquet"))

        if not parquet_files:
            print("No Parquet files found for merging.")
            return

        df_list = []
        for file in tqdm(parquet_files, desc="Merging", unit="file"):
            df = pd.read_parquet(file)
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype("category")
            df_list.append(df)

        merged_df = pd.concat(df_list, ignore_index=True)
        merged_df.to_parquet(self.merged_file, index=False, compression="snappy")
        print(f"✓ Merged file saved: {self.merged_file}")

    def split_parquet_file(self):
        print(f"Splitting merged Parquet: {self.merged_file}")
        table = pq.read_table(self.merged_file)
        total_rows = table.num_rows
        total_memory = table.nbytes
        rows_per_partition = int(
            (self.partition_size_mb * 1024 * 1024) / (total_memory / total_rows)
        )
        rows_per_partition = max(1, rows_per_partition)

        start_idx = 0
        file_index = 1

        with tqdm(total=total_rows, desc="Splitting", unit="rows") as pbar:
            while start_idx < total_rows:
                end_idx = min(start_idx + rows_per_partition, total_rows)
                partition_table = table.slice(start_idx, end_idx - start_idx)
                output_file = os.path.join(
                    self.final_output_dir, f"split_output_{file_index}.parquet"
                )
                pq.write_table(partition_table, output_file, compression="snappy")
                print(f"✓ Saved: {output_file} ({start_idx}-{end_idx})")
                pbar.update(end_idx - start_idx)
                start_idx = end_idx
                file_index += 1

        print(f"✓ Split complete. Output folder: {self.final_output_dir}")

    def run_all(self):
        print("=== STARTING PARQUET PIPELINE ===")
        self.convert_trec_to_parquet()
        self.merge_parquet_files()
        self.split_parquet_file()
        print("=== PIPELINE COMPLETE ===")


# Example usage
if __name__ == "__main__":
    pipeline = ParquetPipeline(
        trec_input_dir="/storage/home/hcoda1/6/dahumada3/erisk_shared/raw/training_data/2023/new_data",
        intermediate_dir="/storage/home/hcoda1/6/dahumada3/erisk_shared/parquet/training_data/2023",
        merged_file="/storage/home/hcoda1/6/dahumada3/erisk_shared/parquet/training_data/2023/merged_output_2023.parquet",
        final_output_dir="/storage/home/hcoda1/6/dahumada3/erisk_shared/parquet/training_data/2023/partitions",
        partition_size_mb=256,
    )
    pipeline.run_all()
