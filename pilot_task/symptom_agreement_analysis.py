import json
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import numpy as np

def load_model_predictions():
    """Load predictions from all model directories"""
    # Define the correct model directory mappings with original names
    model_dirs = {
        'Claude-3.7-sonnet': 'Claude-3.7-sonnet',
        'gpt4o': 'gpt4o',
        'gemini-2.0-flash': 'gemini-2.0-flash',
        'gemini-2.5-pro-exp-03-25': 'gemini-2.5-pro-exp-03-25'
    }
    
    rows = []
    base_path = Path("pilot_task/transcripts")
    
    # Process each model directory
    for model_key, model_dir in model_dirs.items():
        model_path = base_path / model_dir
        if not model_path.exists():
            print(f"Warning: Directory {model_path} does not exist")
            continue
            
        print(f"\nProcessing {model_dir}...")
        json_files = list(model_path.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in {model_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    for datum in data:
                        if "evaluation" in datum:
                            evaluation = datum["evaluation"]
                            if "bdi_scores" in evaluation:
                                bdi_scores = evaluation["bdi_scores"]
                                # Convert BDI scores to the format used in transcript-eda
                                bdi_scores_list = [
                                    {
                                        "question_number": int(re.search(r"q(\d+)_", k).group(1)),
                                        "question_name": k.split("_", 1)[1],
                                        "score": int(v["score"]),
                                        "reason": v["reason"],
                                    }
                                    for k, v in bdi_scores.items()
                                ]
                                
                                d = {
                                    "model": model_key,
                                    "agent": json_file.stem.lower(),
                                    "assessment_turn": int(evaluation["assessment_turn"]),
                                    "assessment_state": evaluation["assessment_state"],
                                    "assessment_complete": bool(evaluation["assessment_complete"]),
                                    "total_bdi_score": int(evaluation["total_bdi_score"]),
                                    "classification_suggestion": evaluation["classification_suggestion"],
                                    "confidence": float(evaluation["confidence"]),
                                    "bdi_scores": bdi_scores_list
                                }
                                rows.append(d)
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
    
    print(f"\nTotal predictions loaded: {len(rows)}")
    return rows

def analyze_symptom_agreement():
    # Load all predictions
    rows = load_model_predictions()
    if not rows:
        print("No predictions were loaded. Please check the data paths and file structure.")
        return
        
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(rows)
    
    # Get only the final assessment for each model-agent pair
    final_assessments = df.sort_values('assessment_turn').groupby(['model', 'agent']).last().reset_index()
    print(f"\nFound {len(final_assessments)} final assessments")
    
    # Create a dictionary to store results for each agent and question
    agent_question_results = defaultdict(lambda: defaultdict(list))
    
    # Get all unique agents
    all_agents = set(final_assessments['agent'].unique())
    print(f"\nFound {len(all_agents)} unique agents")
    
    # For each agent, collect scores from all models for each question
    for agent in all_agents:
        agent_scores = final_assessments[final_assessments['agent'] == agent]
        
        # Only analyze if we have predictions from all four models
        if len(agent_scores) == 4:
            for _, row in agent_scores.iterrows():
                model = row['model']
                for score_info in row['bdi_scores']:
                    if isinstance(score_info, dict):
                        q_num = score_info['question_number']
                        q_name = score_info['question_name']
                        score = score_info['score']
                        agent_question_results[agent][f'q{q_num:02d}_{q_name}'].append((model, score))
    
    # Create tables for each agent
    for agent, questions in agent_question_results.items():
        print(f"\n\nSymptom Scores for Agent: {agent}")
        print("=" * 80)
        
        # Prepare table data
        table_data = []
        
        for question, model_scores in sorted(questions.items()):
            if len(model_scores) == 4:  # All four models have scores
                # Sort scores by model name for consistent comparison
                model_scores.sort(key=lambda x: x[0])
                scores = [score for _, score in model_scores]
                
                # Calculate standard deviation of scores
                std_dev = np.std(scores)
                mean_score = np.mean(scores)
                
                # Add to table data
                table_data.append({
                    'Question': question,
                    'Claude-3.7-sonnet': scores[0],
                    'gpt4o': scores[1],
                    'gemini-2.0-flash': scores[2],
                    'gemini-2.5-pro-exp-03-25': scores[3],
                    'Mean': f"{mean_score:.2f}",
                    'Std Dev': f"{std_dev:.2f}"
                })
        
        # Create and display DataFrame
        df_agent = pd.DataFrame(table_data)
        print(df_agent.to_string(index=False))
    
    # Create overall agreement summary
    print("\n\nOverall Agreement Summary Across All Agents")
    print("=" * 80)
    
    # Calculate statistics for each question across all agents
    question_stats = defaultdict(lambda: {"scores": [], "std_devs": []})
    
    for agent, questions in agent_question_results.items():
        for question, model_scores in questions.items():
            if len(model_scores) == 4:
                scores = [score for _, score in sorted(model_scores, key=lambda x: x[0])]
                std_dev = np.std(scores)
                question_stats[question]["scores"].extend(scores)
                question_stats[question]["std_devs"].append(std_dev)
    
    # Prepare summary table
    summary_data = []
    for question, stats in sorted(question_stats.items()):
        if stats["scores"]:
            mean_std_dev = np.mean(stats["std_devs"])
            total_cases = len(stats["std_devs"])
            summary_data.append({
                'Question': question,
                'Total Cases': total_cases,
                'Mean Std Dev': f"{mean_std_dev:.2f}",
                'Min Std Dev': f"{min(stats['std_devs']):.2f}",
                'Max Std Dev': f"{max(stats['std_devs']):.2f}"
            })
    
    # Create and display summary DataFrame
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # Create visualization of standard deviation rates
    plt.figure(figsize=(12, 6))
    
    # Convert Mean Std Dev to float for plotting
    df_summary['Mean Std Dev Float'] = df_summary['Mean Std Dev'].astype(float)
    
    # Sort by Mean Std Dev for better visualization
    df_summary_sorted = df_summary.sort_values('Mean Std Dev Float')
    
    # Create bar plot
    plt.bar(df_summary_sorted['Question'], df_summary_sorted['Mean Std Dev Float'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Model Agreement Rates Across Symptoms (Lower Std Dev = Higher Agreement)')
    plt.ylabel('Mean Standard Deviation')
    plt.xlabel('Symptoms')
    
    # Add a horizontal line at std dev = 0.5 to show a reference point
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Reference Line (Std Dev = 0.5)')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_agreement_rates.png')
    plt.close()
    
    # Also create a heatmap of standard deviations
    plt.figure(figsize=(15, 8))
    
    # Create a pivot table for the heatmap
    heatmap_data = []
    for agent, questions in agent_question_results.items():
        for question, model_scores in questions.items():
            if len(model_scores) == 4:
                scores = [score for _, score in sorted(model_scores, key=lambda x: x[0])]
                std_dev = np.std(scores)
                heatmap_data.append({
                    'Agent': agent,
                    'Question': question,
                    'Std Dev': std_dev
                })
    
    df_heatmap = pd.DataFrame(heatmap_data)
    pivot_table = df_heatmap.pivot(index='Agent', columns='Question', values='Std Dev')
    
    # Create heatmap
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Standard Deviation'})
    plt.title('Model Agreement Heatmap Across Agents and Symptoms')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig('model_agreement_heatmap.png')
    plt.close()

if __name__ == "__main__":
    analyze_symptom_agreement()
