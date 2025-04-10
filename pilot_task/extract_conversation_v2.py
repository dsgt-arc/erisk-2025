#!/usr/bin/env python3
import re
import json
import argparse

def extract_conversation(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

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

        # Clean up the "[conversation start]" placeholder
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

    return [{
        "LLM": llm_name,
        "conversation": conversation
    }]

def main():
    parser = argparse.ArgumentParser(
        description="Extract simulator/evaluator dialogue from a Markdown file into JSON."
    )
    parser.add_argument("input_md", help="Path to the input Markdown file (e.g. alex.md)")
    parser.add_argument("-o", "--output", default="conversation.json",
                        help="Path to the output JSON file")
    args = parser.parse_args()

    data = extract_conversation(args.input_md)
    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=4)

    print(f"✅  Extracted conversation written to {args.output}")

if __name__ == "__main__":
    main()