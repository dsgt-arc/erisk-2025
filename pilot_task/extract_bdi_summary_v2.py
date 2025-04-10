#!/usr/bin/env python3
"""
Usage:
    chmod +x extract_bdi_summary.py
    ./extract_bdi_summary.py path/to/md_folder -o bdi_summary.json

This script extracts the final (last) Total BDI Score and up to four Key Symptoms
from all Markdown files in a specified directory, normalizing them to the
BDI-II questionnaire categories, and outputs a JSON summary.
"""

import os
import re
import json
import sys
import argparse
from typing import List, Dict

# Canonical BDI-II symptom list
CANONICAL_SYMPTOMS = [
    "Sadness",
    "Pessimism",
    "Sense of Failure",
    "Lack of Satisfaction",
    "Guilty Feelings",
    "Punishment Feelings",
    "Self-Dislike",
    "Self-Criticalness",
    "Suicidal Thoughts",
    "Crying",
    "Agitation",
    "Loss of Interest",
    "Indecisiveness",
    "Worthlessness",
    "Loss of Energy",
    "Changes in Sleep",
    "Irritability",
    "Changes in Appetite",
    "Concentration Problems",
    "Tiredness or Fatigue",
    "Loss of Interest in Sex"
]

# Mapping of common variants to canonical names
SYMPTOM_MAP = {
    "Anhedonia": "Lack of Satisfaction",
    "Fatigue": "Tiredness or Fatigue",
    "Indecisiveness": "Indecisiveness",
    "Self-Criticalness/Dislike": "Self-Criticalness",
    "Self-Criticalness": "Self-Criticalness",
    "Self-Dislike": "Self-Dislike",
    "Pessimism": "Pessimism",
    "Hopelessness": "Pessimism",
    "Sleep Disturbance": "Changes in Sleep",
    "Sleep Changes": "Changes in Sleep",
    "Concentration Problems": "Concentration Problems",
    "Worthlessness": "Worthlessness",
    "Guilt": "Guilty Feelings",
    "Guilty Feelings": "Guilty Feelings",
    "Irritability": "Irritability",
    "Loss of Interest": "Loss of Interest",
    "Loss of Energy": "Loss of Energy",
    "Tiredness or Fatigue": "Tiredness or Fatigue",
    "Loss of Interest in Sex": "Loss of Interest in Sex",
    "Crying": "Crying",
    "Agitation": "Agitation",
    "Sense of Failure": "Sense of Failure",
    "Punishment Feelings": "Punishment Feelings",
    "Changes in Appetite": "Changes in Appetite"
}

def normalize_symptom(raw: str) -> str:
    """Map a raw symptom string to its canonical BDI-II name."""
    raw = raw.strip().rstrip('|').strip()
    # Direct mapping
    if raw in SYMPTOM_MAP:
        return SYMPTOM_MAP[raw]
    # If already canonical
    if raw in CANONICAL_SYMPTOMS:
        return raw
    # Try partial match
    for key, canon in SYMPTOM_MAP.items():
        if key.lower() in raw.lower():
            return canon
    # Fallback: ignore unknown
    return None

def extract_bdi_from_text(text: str) -> Dict:
    # 1) Extract LLM name
    m = re.search(r'^# Conversation with\s+([^\s(]+)', text, re.MULTILINE)
    llm = m.group(1) if m else "LLM"

    # 2) Last Total BDI Score
    scores = re.findall(r'Total BDI Score\s*\|\s*(\d+)', text)
    bdi_score = int(scores[-1]) if scores else None

    # 3) Last Key Symptoms line
    keys = re.findall(r'Key Symptoms\s*\|\s*(.+)', text)
    raw_keys = keys[-1] if keys else ""
    # Split and normalize
    normalized = []
    for part in re.split(r',\s*', raw_keys):
        norm = normalize_symptom(part)
        if norm and norm not in normalized:
            normalized.append(norm)
        if len(normalized) >= 4:
            break

    return {
        "LLM": llm,
        "bdi-score": bdi_score,
        "key-symptoms": normalized
    }

def process_directory(input_dir: str) -> List[Dict]:
    results = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".md"):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        entry = extract_bdi_from_text(text)
        if entry["bdi-score"] is not None:
            results.append(entry)
        else:
            print(f"⚠️ Warning: no BDI score found in {fname}", file=sys.stderr)
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Extract final BDI scores and up to 4 normalized key symptoms from Markdown files."
    )
    parser.add_argument("input_dir", help="Directory containing .md files")
    parser.add_argument("-o", "--output", default="bdi_summary.json",
                        help="Output JSON file")
    args = parser.parse_args()

    summary = process_directory(args.input_dir)
    with open(args.output, 'w', encoding='utf-8') as out_f:
        json.dump(summary, out_f, ensure_ascii=False, indent=4)

    print(f"✅ Extracted BDI summary for {len(summary)} files to {args.output}")

if __name__ == "__main__":
    main()

