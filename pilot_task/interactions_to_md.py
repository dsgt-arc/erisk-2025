#!/usr/bin/env python3
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert transcript JSON to Markdown.")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to output Markdown file")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)

    with open(args.output, 'w', encoding='utf-8') as out:
        for persona in transcripts:
            name = persona.get("LLM", "Unknown")
            out.write(f"# {name}\n\n")
            for turn in persona.get("conversation", []):
                role = turn.get("role", "").capitalize()
                # Preserve line breaks in Markdown
                msg = turn.get("message", "").replace("\n", "  \n")
                out.write(f"**{role}:** {msg}\n\n")

if __name__ == "__main__":
    main()
