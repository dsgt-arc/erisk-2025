import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
from scipy import stats

def extract_agent_profile(json_file):
    """Extract agent profile from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    # Get the final assessment
    final_assessment = None
    for turn in data:  # data is a list of turns
        if 'evaluation' in turn and turn['evaluation'].get('assessment_complete', False):
            final_assessment = turn['evaluation']
    
    if not final_assessment:
        return None
        
    # Convert classification to numeric value
    classification_map = {
        'Control': 1,
        'Mild': 2,
        'Borderline': 3,
        'Moderate': 4,
        'Severe': 5,
        'Extreme': 6,
        'Uncertain': 0
    }
    
    classification = final_assessment.get('classification_suggestion', 'Uncertain')
    classification_numeric = classification_map.get(classification, 0)
        
    return {
        'agent': os.path.basename(json_file).replace('.json', ''),
        'Final BDI Score': final_assessment.get('total_bdi_score', 0),
        'classification_suggestion': classification,
        'classification_numeric': classification_numeric
    }

def create_agent_summary_table():
    """Create a summary table of agent profiles."""
    # Define model directories
    model_dirs = [
        'pilot_task/transcripts/Claude-3.7-sonnet',
        'pilot_task/transcripts/gpt4o',
        'pilot_task/transcripts/gemini-2.0-flash',
        'pilot_task/transcripts/gemini-2.5-pro-exp-03-25'
    ]
    
    # Load all agent profiles
    profiles = []
    for model_dir in model_dirs:
        json_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
        for json_file in json_files:
            profile = extract_agent_profile(os.path.join(model_dir, json_file))
            if profile:
                profiles.append(profile)
    
    # Convert to DataFrame
    df = pd.DataFrame(profiles)
    
    # Create summary statistics by classification
    summary = df.groupby('classification_suggestion').agg({
        'Final BDI Score': ['count', 'mean', 'std', 'min', 'max'],
        'classification_numeric': 'first'
    }).round(2)
    
    # Sort by classification_numeric
    summary = summary.sort_values(('classification_numeric', 'first'))
    
    # Print summary statistics
    print("\nSummary Statistics by Classification:")
    print(summary)
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['classification_numeric'], df['Final BDI Score'])
    r_squared = r_value ** 2
    
    # Create scatter plot with fitted line
    plt.figure(figsize=(10, 6))
    plt.scatter(df['classification_numeric'], df['Final BDI Score'], alpha=0.5, label='Data points')
    
    # Add fitted line
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    y = slope * x + intercept
    plt.plot(x, y, 'r-', label=f'Fitted line (R² = {r_squared:.3f})')
    
    plt.title('Relationship between Classification and BDI Score')
    plt.xlabel('Classification (Numeric)')
    plt.ylabel('Final BDI Score')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], 
               ['Uncertain', 'Control', 'Mild', 'Borderline', 'Moderate', 'Severe', 'Extreme'],
               rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pilot_task/classification_bdi_fitted_line.png')
    
    # Print regression statistics
    print(f"\nRegression Statistics:")
    print(f"Slope: {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")
    print(f"R-squared: {r_squared:.3f}")
    print(f"P-value: {p_value:.3f}")
    
    # Create box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='classification_suggestion', y='Final BDI Score',
                order=['Control', 'Mild', 'Borderline', 'Moderate', 'Severe', 'Extreme', 'Uncertain'])
    plt.title('BDI Score Distribution by Classification')
    plt.xlabel('Classification')
    plt.ylabel('Final BDI Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('pilot_task/classification_bdi_boxplot.png')
    
    # Create scatter plot with jitter
    plt.figure(figsize=(12, 6))
    sns.stripplot(data=df, x='classification_suggestion', y='Final BDI Score', 
                 jitter=True, alpha=0.5,
                 order=['Control', 'Mild', 'Borderline', 'Moderate', 'Severe', 'Extreme', 'Uncertain'])
    plt.title('BDI Scores by Classification')
    plt.xlabel('Classification')
    plt.ylabel('Final BDI Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('pilot_task/classification_bdi_scatter.png')
    
    # Calculate correlation between classification and BDI score
    correlation = df['classification_numeric'].corr(df['Final BDI Score'])
    print(f"\nCorrelation between classification and BDI score: {correlation:.3f}")
    
    # Save results to CSV
    df.to_csv('pilot_task/agent_profiles.csv', index=False)
    print("\nTable has been saved to 'pilot_task/agent_profiles.csv'")

if __name__ == "__main__":
    create_agent_summary_table()