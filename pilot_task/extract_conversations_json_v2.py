import json
import argparse
import os
import glob
import sys

def extract_conversation(turns, llm_name):
    """
    Given a list of turn dicts, return a list of
    {"role": ..., "message": ...} entries with:
      - persona messages (input_message) as role llm_name
      - assistant messages (output_message) as role "user"
      - skipping any input_message == "start" (but including its greeting)
    """
    # Pair with original index, sort by turn_number if present else by index
    indexed = list(enumerate(turns))
    indexed.sort(key=lambda pair: pair[1].get('turn_number', pair[0]))
    sorted_turns = [t for _, t in indexed]

    conv = []
    for turn in sorted_turns:
        in_msg = turn.get('input_message', '').strip()
        out_msg = turn.get('output_message', '').strip()

        # Skip the literal "start" prompt, but include its response
        if in_msg.lower() == "start":
            if out_msg:
                conv.append({"role": "user", "message": out_msg})
            continue

        if in_msg:
            conv.append({"role": llm_name, "message": in_msg})
        if out_msg:
            conv.append({"role": "user", "message": out_msg})

    return conv

def gather_files(path, verbose=False):
    """
    Return a list of JSON file paths given either a directory or a single file.
    """
    if os.path.isdir(path):
        pattern = os.path.join(path, '*.json')
        files = sorted(glob.glob(pattern))
        if verbose:
            print(f"Input is directory. Found {len(files)} .json files.")
        return files
    elif os.path.isfile(path) and path.lower().endswith('.json'):
        if verbose:
            print(f"Input is single file: {path}")
        return [path]
    else:
        raise FileNotFoundError(f"No such directory or JSON file: {path}")

def main():
    parser = argparse.ArgumentParser(
        description="Merge one or more conversation JSON files into a single JSON array."
    )
    parser.add_argument(
        "input_path",
        help="Path to a folder of JSONs or a single JSON file."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        required=True,
        help="Path to the merged output JSON file."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose progress."
    )

    args = parser.parse_args()
    v = args.verbose

    try:
        files = gather_files(args.input_path, verbose=v)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("No JSON files to process.", file=sys.stderr)
        sys.exit(1)

    merged = []

    for filepath in files:
        filename = os.path.basename(filepath)
        llm_name = os.path.splitext(filename)[0].capitalize()
        if v:
            print(f"\n--- Processing '{filename}' as LLM='{llm_name}' ---")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                turns = json.load(f)
            if v:
                print(f"Loaded {len(turns)} turns")
        except Exception as e:
            print(f"Warning: failed to load {filename}: {e}", file=sys.stderr)
            continue

        conv = extract_conversation(turns, llm_name)
        if v:
            print(f"Extracted {len(conv)} messages")

        merged.append({
            "LLM": llm_name,
            "conversation": conv
        })

    # Always write output, even if merged is empty
    try:
        with open(args.output_file, 'w', encoding='utf-8') as out_f:
            json.dump(merged, out_f, ensure_ascii=False, indent=2)
        print(f"\n✅ Wrote {len(merged)} transcript(s) to '{args.output_file}'")
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
