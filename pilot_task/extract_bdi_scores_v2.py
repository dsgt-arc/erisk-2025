import json
import argparse
import os
import glob
import sys
import difflib

# Updated canonical BDI symptom list
CANONICAL_SYMPTOMS = [
    "Sadness",
    "Pessimism",
    "Past Failure",
    "Loss of Pleasure",
    "Guilty Feelings",
    "Punishment Feelings",
    "Self-Dislike",
    "Self-Criticalness",
    "Suicidal Thoughts or Wishes",
    "Crying",
    "Agitation",
    "Loss of Interest",
    "Indecisiveness",
    "Worthlessness",
    "Loss of Energy",
    "Changes in Sleeping Pattern",
    "Irritability",
    "Changes in Appetite",
    "Concentration Difficulty",
    "Tiredness or Fatigue",
    "Loss of Interest in Sex"
]

# Map raw labels to the new canonical forms
SYMPTOM_MAP = {
    "Sadness": "Sadness",
    "Pessimism": "Pessimism",
    "Pessimism/Hopelessness": "Pessimism",
    "Hopelessness": "Pessimism",
    "Sense of Failure": "Past Failure",
    "Past Failure": "Past Failure",
    "Failure": "Past Failure",
    "Anhedonia": "Loss of Pleasure",
    "Anhedonia/Disconnection": "Loss of Pleasure",
    "Loss of Pleasure": "Loss of Pleasure",
    "Lack of Satisfaction": "Loss of Pleasure",
    "Emotional Suppression": "Loss of Pleasure",
    "Loss of Interest": "Loss of Interest",
    "Loss of Interest Social": "Loss of Interest",
    "Loss Of Social Interest": "Loss of Interest",
    "Social Withdrawal": "Loss of Interest",
    "Social Disinterest": "Loss of Interest",
    "Guilty Feelings": "Guilty Feelings",
    "Guilt": "Guilty Feelings",
    "Punishment Feelings": "Punishment Feelings",
    "Self-Dislike": "Self-Dislike",
    "Self Dislike": "Self-Dislike",
    "Self-Criticism": "Self-Criticalness",
    "Self-Criticalness": "Self-Criticalness",
    "Self-Criticalness/Worthlessness" : "Self-Criticalness",
    "Suicidal Thoughts": "Suicidal Thoughts or Wishes",
    "Suicidal Thoughts or Wishes": "Suicidal Thoughts or Wishes",
    "Crying": "Crying",
    "Agitation": "Agitation",
    "Indecisiveness": "Indecisiveness",
    "Worthlessness": "Worthlessness",
    "Depersonalization": "Loss of Pleasure",
    "Loss of Energy": "Loss of Energy",
    "Fatigue/Loss Of Energy": "Loss of Energy",
    "Mild Fatigue": "Tiredness or Fatigue",
    "Fatigue": "Tiredness or Fatigue",
    "Tiredness or Fatigue": "Tiredness or Fatigue",
    "Changes in Sleep": "Changes in Sleeping Pattern",
    "Changes in Sleeping Pattern": "Changes in Sleeping Pattern",
    "Sleep Disturbance": "Changes in Sleeping Pattern",
    "Sleep Changes": "Changes in Sleeping Pattern",
    "Sleep Problems": "Changes in Sleeping Pattern",    
    "Mild Sleep Disturbance": "Changes in Sleeping Pattern",
    "Changes in Appetite": "Changes in Appetite",
    "Appetite/Weight Changes": "Changes in Appetite",
    "Appetite Changes": "Changes in Appetite",
    "Concentration Problems": "Concentration Difficulty",
    "Concentration Difficulty": "Concentration Difficulty",
    "Diminished Concentration": "Concentration Difficulty",
    "Irritability": "Irritability",
    "Loss of Interest in Sex": "Loss of Interest in Sex"
}

def canonicalize(symptom):
    """
    Map a raw symptom label to its canonical form.
    1) Exact, case‐insensitive match against SYMPTOM_MAP keys
    2) Fuzzy match against SYMPTOM_MAP keys
    3) Fuzzy match against CANONICAL_SYMPTOMS
    Returns None if no good match found.
    """
    sym = symptom.strip().lower()

    # 1) exact match
    for raw, canon in SYMPTOM_MAP.items():
        if raw.lower() == sym:
            return canon

    # 2) fuzzy against raw‐variant keys
    raw_keys = list(SYMPTOM_MAP.keys())
    close = difflib.get_close_matches(symptom, raw_keys, n=1, cutoff=0.75)
    if close:
        return SYMPTOM_MAP[close[0]]

    # 3) fuzzy against canonical list
    close2 = difflib.get_close_matches(symptom, CANONICAL_SYMPTOMS, n=1, cutoff=0.75)
    if close2:
        return close2[0]

    return None

def gather_files(path):
    """
    Given a path to a directory or single JSON file,
    return a list of JSON file paths.
    """
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, '*.json')))
    elif os.path.isfile(path) and path.lower().endswith('.json'):
        return [path]
    else:
        raise FileNotFoundError(f"No JSON files found at: {path}")

def extract_final_evaluation(turns):
    """
    Given a list of turn dicts, return the evaluation dict
    from the last turn (sorted by turn_number if present).
    """
    indexed = list(enumerate(turns))
    indexed.sort(key=lambda pair: pair[1].get('turn_number', pair[0]))
    _, final_turn = indexed[-1]
    return final_turn.get('evaluation', {})

def main():
    parser = argparse.ArgumentParser(
        description="Extract BDI scores and canonical key symptoms from conversation JSONs."
    )
    parser.add_argument(
        "input_path",
        help="Path to folder of JSONs or a single JSON file."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        required=True,
        help="Path to the output JSON file."
    )
    parser.add_argument(
        "-m", "--max",
        dest="max_symptoms",
        type=int,
        default=4,
        help="Maximum number of key symptoms to include per entry."
    )

    args = parser.parse_args()

    try:
        files = gather_files(args.input_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    results = []

    for filepath in files:
        filename = os.path.basename(filepath)
        llm_name = os.path.splitext(filename)[0].capitalize()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                turns = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load {filename}: {e}", file=sys.stderr)
            continue

        eval_dict = extract_final_evaluation(turns)
        score = eval_dict.get('total_bdi_score')
        raw_symptoms = eval_dict.get('key_symptoms', [])

        # canonicalize and dedupe
        seen = set()
        canon_list = []
        for s in raw_symptoms:
            c = canonicalize(s)
            if c and c not in seen:
                seen.add(c)
                canon_list.append(c)
            if len(canon_list) >= args.max_symptoms:
                break

        results.append({
            "LLM": llm_name,
            "bdi-score": score,
            "key-symptoms": canon_list
        })

    # write out
    with open(args.output_file, 'w', encoding='utf-8') as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} entries to {args.output_file}")

if __name__ == "__main__":
    main()
