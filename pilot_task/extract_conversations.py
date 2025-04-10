#!/usr/bin/env python3
import re
import json
import argparse
import os
from typing import List, Dict

def extract_conversation_from_text(text: str) -> Dict:
    # 1) LLM name
    m = re.search(r'^# Conversation with\s+([^\s(]+)', text, re.MULTILINE)
    llm_name = m.group(1) if m else "LLM"

    # 2) Split into turns
    parts = re.split(r'^## Turn\s+(\d+)', text, flags=re.MULTILINE)

    conversation = []
    for idx in range(1, len(parts), 2):
        turn_no = parts[idx]
        chunk   = parts[idx+1]

        # Extract Simulator block
        sim_match = re.search(r'### Simulator\s*(.*?)\s*(?=### Evaluator|$)', chunk, re.DOTALL)
        sim_text  = sim_match.group(1).strip() if sim_match else ""
        # Extract Evaluator block
        eval_match = re.search(r'### Evaluator\s*(.*?)(?=###|$)', chunk, re.DOTALL)
        eval_text  = eval_match.group(1).strip() if eval_match else ""

        # Remove the placeholder if present
        if sim_text.startswith("[conversation start]"):
            sim_text = ""

        # Turn 1: only the Evaluator
        if turn_no == "1":
            if eval_text:
                conversation.append({
                    "role": "user",
                    "message": eval_text
                })
        else:
            # Subsequent turns: Simulator (LLM) first, then Evaluator
            if sim_text:
                conversation.append({
                    "role": llm_name,
                    "message": sim_text
                })
            if eval_text:
                conversation.append({
                    "role": "user",
                    "message": eval_text
                })

    return {
        "LLM": llm_name,
        "conversation": conversation
    }

def process_directory(input_dir: str) -> List[Dict]:
    result = []
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(".md"):
            continue
        path = os.path.join(input_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        conv = extract_conversation_from_text(text)
        result.append(conv)
    return result

'''
Usage in terminal:
        python ./extract_all_conversations.py path/to/md_folder -o conversations.json
'''
def main():
    parser = argparse.ArgumentParser(
        description="Extract simulator/evaluator dialogues from all Markdown files in a directory into one JSON."
    )
    parser.add_argument("input_dir", help="Path to the directory containing .md files")
    parser.add_argument("-o", "--output", default="all_conversations.json",
                        help="Path to the output JSON file")
    args = parser.parse_args()

    all_convs = process_directory(args.input_dir)
    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(all_convs, out_f, ensure_ascii=False, indent=4)

    print(f"✅  Extracted {len(all_convs)} conversations to {args.output}")

if __name__ == "__main__":
    main()
