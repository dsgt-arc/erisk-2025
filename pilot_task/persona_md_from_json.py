import json
import os
import argparse
import re

def sanitize_filename(name):
    # Remove unsafe characters
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

def main():
    parser = argparse.ArgumentParser(description="Convert transcript JSON to separate Markdown files per persona.")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to write output Markdown files")
    args = parser.parse_args()

    # Load JSON
    with open(args.input, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate a markdown file for each persona
    for persona in transcripts:
        name = persona.get("LLM", "Unknown")
        safe_name = sanitize_filename(name)
        out_path = os.path.join(args.output_dir, f"{safe_name}.md")
        with open(out_path, 'w', encoding='utf-8') as out:
            out.write(f"# {name}\n\n")
            for turn in persona.get("conversation", []):
                role = turn.get("role", "").capitalize()
                message = turn.get("message", "").strip()
                # Write role line and message with paragraph spacing
                out.write(f"**{role}:**\n\n")
                # Preserve original paragraphs
                for para in message.split('\\n\\n'):
                    out.write(f"{para.strip()}\n\n")
        print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()