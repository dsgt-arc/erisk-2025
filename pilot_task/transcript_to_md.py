#!/usr/bin/env python3
"""
Converts JSON transcripts from eRisk 2025 BDI conversations to
formatted markdown files for easier analysis and review.
Places markdown files alongside the original JSON files.
"""

import json
import os
import glob
import argparse


def get_simulator_name(json_path):
    """Extract the simulator name from the JSON file path."""
    return os.path.basename(json_path).split(".")[0]


def get_model_name(json_path):
    """Extract the model name from the JSON file path."""
    parts = json_path.split("/")
    if len(parts) > 2:
        return parts[-2]
    return "unknown-model"


def create_markdown_table(data, headers):
    """Create a markdown table from a list of dictionaries."""
    if not data:
        return ""

    # Create header row
    md_table = f"| {' | '.join(headers)} |\n"
    # Create separator row
    md_table += f"| {' | '.join(['---'] * len(headers))} |\n"

    # Create data rows
    for row in data:
        md_table += f"| {' | '.join([str(row.get(h, '')) for h in headers])} |\n"

    return md_table


def create_bdi_scores_table(bdi_scores):
    """Create a markdown table for BDI scores."""
    rows = []
    for question, data in bdi_scores.items():
        rows.append(
            {"Question": question, "Score": data["score"], "Reason": data["reason"]}
        )

    return create_markdown_table(rows, ["Question", "Score", "Reason"])


def create_assessment_meta_table(evaluation):
    """Create a markdown table for assessment metadata."""
    meta = {
        "Assessment Turn": evaluation["assessment_turn"],
        "Assessment State": evaluation["assessment_state"],
        "Assessment Complete": evaluation["assessment_complete"],
        "Total BDI Score": evaluation["total_bdi_score"],
        "Classification": evaluation["classification_suggestion"],
        "Confidence": evaluation["confidence"],
        "Key Symptoms": ", ".join(evaluation["key_symptoms"]),
    }

    rows = []
    for key, value in meta.items():
        rows.append({"Attribute": key, "Value": value})

    return create_markdown_table(rows, ["Attribute", "Value"])


def convert_transcript_to_md(json_path):
    """Convert a JSON transcript to a markdown file in the same directory."""

    # Extract simulator name and model name from path
    simulator_name = get_simulator_name(json_path)
    model_name = get_model_name(json_path)

    # Read JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create output file path (same directory, just change extension)
    output_path = os.path.splitext(json_path)[0] + ".md"

    # Generate markdown content
    md_content = f"# Conversation with {simulator_name.capitalize()} ({model_name})\n\n"

    for turn in data:
        turn_number = turn["turn_number"]
        md_content += f"## Turn {turn_number}\n\n"

        # Add conversation messages
        md_content += "### Simulator\n\n"
        if turn["input_message"] != "start":
            md_content += f"{turn['input_message']}\n\n"
        else:
            md_content += "[conversation start]\n\n"

        md_content += "### Evaluator\n\n"
        md_content += f"{turn['output_message']}\n\n"

        md_content += "### Reasoning\n\n"
        md_content += f"{turn['next_step_reasoning']}\n\n"

        # Add BDI scores table
        md_content += "### BDI Scores\n\n"
        md_content += create_bdi_scores_table(turn["evaluation"]["bdi_scores"])
        md_content += "\n\n"

        # Add assessment metadata table
        md_content += "### Assessment Metadata\n\n"
        md_content += create_assessment_meta_table(turn["evaluation"])
        md_content += "\n\n"

        # Add separation between turns
        md_content += "---\n\n"

    # Write to file
    with open(output_path, "w") as f:
        f.write(md_content)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert eRisk 2025 BDI JSON transcripts to markdown."
    )
    parser.add_argument(
        "--input-dir",
        default="pilot_task/transcripts",
        help="Directory containing JSON transcripts (default: pilot_task/transcripts)",
    )
    args = parser.parse_args()

    # Find all JSON files in the input directory and its subdirectories
    json_paths = glob.glob(f"{args.input_dir}/**/*.json", recursive=True)

    if not json_paths:
        print(f"No JSON files found in {args.input_dir}")
        return

    # Convert each JSON file to markdown (in the same directory)
    for json_path in json_paths:
        output_path = convert_transcript_to_md(json_path)
        print(f"Converted {json_path} to {output_path}")


if __name__ == "__main__":
    main()
