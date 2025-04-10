#!/usr/bin/env python3
import json
import argparse
import sys

def verify_conversations(data):
    errors = []
    for idx, entry in enumerate(data):
        llm = entry.get('LLM')
        conv = entry.get('conversation')
        # Basic sanity checks
        if not isinstance(llm, str):
            errors.append(f"Entry {idx}: missing or invalid LLM name: {llm!r}")
            continue
        if not isinstance(conv, list):
            errors.append(f"Entry {idx} ('{llm}'): 'conversation' is not a list")
            continue
        if len(conv) == 0:
            errors.append(f"Entry {idx} ('{llm}'): conversation is empty")
            continue

        # Check first message is from user
        first_role = conv[0].get('role')
        if first_role != 'user':
            errors.append(
                f"Entry {idx} ('{llm}'): first role is {first_role!r}, expected 'user'"
            )

        # Check alternation
        for i, msg in enumerate(conv):
            expected = 'user' if i % 2 == 0 else llm
            actual = msg.get('role')
            if actual != expected:
                errors.append(
                    f"Entry {idx} ('{llm}'): message #{i} role is {actual!r}, expected {expected!r}"
                )
    return errors

def main():
    parser = argparse.ArgumentParser(
        description="Verify that each conversation alternates user → LLM → user → …"
    )
    parser.add_argument("input_json", help="Path to the JSON file to verify")
    args = parser.parse_args()

    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read or parse JSON: {e}", file=sys.stderr)
        sys.exit(2)

    errors = verify_conversations(data)
    if errors:
        print("❌ Validation failed with the following errors:")
        for err in errors:
            print("  - " + err)
        sys.exit(1)
    else:
        print("✅ All conversations validated successfully: roles alternate user ↔ LLM.")
        sys.exit(0)

if __name__ == "__main__":
    main()
