# eRisk 2025 Pilot Task: Conversational Depression Detection via LLMs

This directory contains resources for the eRisk 2025 Pilot Task focused on conversational depression detection via Large Language Models (LLMs). For full details on the task, visit the [official eRisk pilot task page](https://erisk.irlab.org/pilotTaskLLMs.html).

## Directory Structure

- `/prompt`: Contains system prompts and schemas used for the conversational assessment
  - `/v2`: Updated version of the prompt system with improved assessment approach
    - `prompt.md`: The system prompt for the conversational assessment
    - `schema.json`: JSON schema for structured output from the LLM

- `/transcripts`: Contains JSON transcripts of conversations between the evaluator and the simulated persona
  - Each subdirectory represents a different model used for evaluation
  - Each JSON file contains a conversation with a particular simulated persona

## Approach

The approach used in this project involves LLMs in two roles:

1. **Simulator Role**: An LLM simulates a human with various levels of depression symptoms
2. **Evaluator Role**: An LLM assesses the simulated human for signs of depression through natural conversation

The evaluator uses a prompt with structured output that includes:
- Turn-by-turn assessment of BDI-II (Beck Depression Inventory) indicators
- Natural, empathetic conversation that avoids direct questions about depression
- Progressive assessment through phases (Initializing → Gathering → Consolidating → Concluding → Finalized)
- Confidence-based classification into severity categories (Control, Mild, Borderline, Moderate, Severe, Extreme)

Each turn produces a detailed evaluation with scores for 21 BDI dimensions, providing transparency into the assessment process and supporting research into conversational depression detection.

## Usage

### Generating Markdown from Transcripts

For easier analysis of the JSON transcripts, use the `transcript_to_md.py` script to generate markdown files:

```bash
# From project root
python pilot_task/transcript_to_md.py

# Or specify custom input/output directories
python pilot_task/transcript_to_md.py --input-dir custom/input/path
```

The script will:
- Process all JSON files in the transcripts directory and subdirectories
- Generate markdown files alongside JSON files with the same basename
- Format each turn with proper headings for simulator messages, evaluator responses, reasoning, and assessment data
- Create tables for BDI scores and assessment metadata

### Adding New Transcripts

Place new transcript JSON files in the appropriate subdirectory under `/transcripts`, organized by model name. Then run the markdown generation script to create readable versions of these transcripts.