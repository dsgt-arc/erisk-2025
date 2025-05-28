import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Data from the analysis
models = ['claude-3.7-sonnet', 'gpt4o', 'gemini-2.0-flash', 'gemini-2.5-pro-exp-03-25']
overall_scores = [41.35, 41.3, 41.29, 40.95]
detection_rates = [100.0, 100.0, 100.0, 100.0]
response_scores = [0.02, 0.02, 0.02, 0.01]
total_turns = [138, 130, 121, 135]

# Symptom recognition scores
symptom_data = {
    'Model': models,
    'Mood': [0.044, 0.036, 0.035, 0.040],
    'Energy': [0.055, 0.047, 0.051, 0.054],
    'Interest': [0.004, 0.006, 0.006, 0.004],
    'Cognitive': [0.016, 0.004, 0.019, 0.015],
    'Physical': [0.010, 0.004, 0.006, 0.007],
    'Social': [0.031, 0.022, 0.026, 0.014]
}

# Severity classification data
severity_data = {
    'claude-3.7-sonnet': {'Mild': 87, 'Moderate': 6, 'Severe': 10},
    'gpt4o': {'Mild': 54, 'Moderate': 9, 'Severe': 8},
    'gemini-2.0-flash': {'Mild': 63, 'Moderate': 8, 'Severe': 3},
    'gemini-2.5-pro-exp-03-25': {'Mild': 76, 'Moderate': 7, 'Severe': 9}
}

# 1. Overall Performance Comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(models, overall_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Overall Performance Scores by Model', fontsize=16, fontweight='bold')
plt.ylabel('Performance Score (%)')
plt.ylim(40.5, 41.5)
plt.xticks(rotation=45, ha='right')

# Add value labels on bars
for bar, score in zip(bars, overall_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('01_overall_performance_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Symptom Recognition Heatmap
plt.figure(figsize=(10, 6))
symptom_df = pd.DataFrame(symptom_data)
symptom_df_plot = symptom_df.set_index('Model')
sns.heatmap(symptom_df_plot.T, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Symptom Recognition Scores Heatmap', fontsize=16, fontweight='bold')
plt.xlabel('Models')
plt.ylabel('Symptoms')
plt.tight_layout()
plt.savefig('02_symptom_recognition_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Model Performance Summary Table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
detection_table_data = [
    ['Model', 'Detection Rate', 'Total Turns', 'Response Quality'],
    ['claude-3.7-sonnet', '100.0%', '138', '0.02'],
    ['gpt4o', '100.0%', '130', '0.02'],
    ['gemini-2.0-flash', '100.0%', '121', '0.02'],
    ['gemini-2.5-pro-exp-03-25', '100.0%', '135', '0.01']
]
table = ax.table(cellText=detection_table_data[1:], colLabels=detection_table_data[0],
                 cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
ax.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig('03_model_performance_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Severity Classification Stacked Bar Chart
plt.figure(figsize=(10, 6))
mild_counts = [severity_data[model]['Mild'] for model in models]
moderate_counts = [severity_data[model]['Moderate'] for model in models]
severe_counts = [severity_data[model]['Severe'] for model in models]

width = 0.6
x_pos = np.arange(len(models))

p1 = plt.bar(x_pos, mild_counts, width, label='Mild', color='#90EE90')
p2 = plt.bar(x_pos, moderate_counts, width, bottom=mild_counts, label='Moderate', color='#FFD700')
p3 = plt.bar(x_pos, severe_counts, width, 
             bottom=[mild_counts[i] + moderate_counts[i] for i in range(len(models))], 
             label='Severe', color='#FF6B6B')

plt.title('Depression Severity Classification by Model', fontsize=16, fontweight='bold')
plt.xlabel('Models')
plt.ylabel('Number of Cases')
plt.xticks(x_pos, models, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('04_severity_classification.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Symptom Recognition Radar Chart
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
symptom_categories = ['Mood', 'Energy', 'Interest', 'Cognitive', 'Physical', 'Social']
angles = np.linspace(0, 2 * np.pi, len(symptom_categories), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # Complete the circle

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, model in enumerate(models):
    values = [symptom_data[cat][i] for cat in symptom_categories]
    values += [values[0]]  # Complete the circle
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(symptom_categories)
ax.set_title('Symptom Recognition Performance Radar', fontsize=16, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig('05_symptom_radar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Multi-Metric Model Comparison
plt.figure(figsize=(12, 6))
ranking_data = pd.DataFrame({
    'Model': models,
    'Overall Score': overall_scores,
    'Response Quality': [x * 100 for x in response_scores],  # Scale up for visibility
    'Avg Symptom Score': [np.mean([symptom_data[cat][i] for cat in symptom_categories[:-1]]) * 100 
                          for i in range(len(models))]
})

x = np.arange(len(models))
width = 0.25

bars1 = plt.bar(x - width, ranking_data['Overall Score'], width, label='Overall Score', alpha=0.8)
bars2 = plt.bar(x, ranking_data['Response Quality'], width, label='Response Quality (×100)', alpha=0.8)
bars3 = plt.bar(x + width, ranking_data['Avg Symptom Score'], width, label='Avg Symptom Score (×100)', alpha=0.8)

plt.title('Multi-Metric Model Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.xticks(x, models, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('06_multi_metric_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Coverage Analysis Pie Chart
plt.figure(figsize=(8, 8))
coverage_data = [12, 12, 12, 12]  # All models analyzed all 12 people
plt.pie(coverage_data, labels=models, autopct='%1.0f%%', startangle=90)
plt.title('Coverage Analysis - People Analyzed per Model', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('07_coverage_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Detailed Performance Metrics Table
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Create detailed metrics table
detailed_metrics = [
    ['Metric', 'claude-3.7-sonnet', 'gpt4o', 'gemini-2.0-flash', 'gemini-2.5-pro-exp-03-25'],
    ['Overall Score', '41.35%', '41.30%', '41.29%', '40.95%'],
    ['Total Turns', '138', '130', '121', '135'],
    ['Depression Detection', '100.0%', '100.0%', '100.0%', '100.0%'],
    ['Response Quality', '0.02', '0.02', '0.02', '0.01'],
    ['Mood Recognition', '0.044', '0.036', '0.035', '0.040'],
    ['Energy Recognition', '0.055', '0.047', '0.051', '0.054'],
    ['Interest Recognition', '0.004', '0.006', '0.006', '0.004'],
    ['Cognitive Recognition', '0.016', '0.004', '0.019', '0.015'],
    ['Physical Recognition', '0.010', '0.004', '0.006', '0.007'],
    ['Social Recognition', '0.031', '0.022', '0.026', '0.014']
]

table2 = ax.table(cellText=detailed_metrics[1:], colLabels=detailed_metrics[0],
                  cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 1.5)
ax.set_title('Detailed Performance Metrics', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('08_detailed_metrics_table.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate summary statistics table
print("=== SUMMARY STATISTICS TABLE ===")
summary_df = pd.DataFrame({
    'Model': models,
    'Overall_Score': overall_scores,
    'Total_Conversations': [12, 12, 12, 12],
    'Total_Turns': total_turns,
    'Detection_Rate': detection_rates,
    'Response_Quality': response_scores,
    'Mild_Cases': [severity_data[model]['Mild'] for model in models],
    'Moderate_Cases': [severity_data[model]['Moderate'] for model in models],
    'Severe_Cases': [severity_data[model]['Severe'] for model in models]
})

print(summary_df.to_string(index=False))

# Generate symptom recognition table
print("\n=== SYMPTOM RECOGNITION SCORES TABLE ===")
symptom_df = pd.DataFrame(symptom_data)
print(symptom_df.to_string(index=False))

# Calculate and display rankings
print("\n=== MODEL RANKINGS ===")
rankings_df = pd.DataFrame({
    'Rank': [1, 2, 3, 4],
    'Model': ['claude-3.7-sonnet', 'gpt4o', 'gemini-2.0-flash', 'gemini-2.5-pro-exp-03-25'],
    'Overall_Score': [41.35, 41.30, 41.29, 40.95],
    'Performance_Gap': [0.00, -0.05, -0.06, -0.40]
})
print(rankings_df.to_string(index=False))

print("\n=== INDIVIDUAL CHARTS GENERATED ===")
print("8 separate chart files have been saved:")
print("01_overall_performance_scores.png")
print("02_symptom_recognition_heatmap.png")
print("03_model_performance_summary.png")
print("04_severity_classification.png")
print("05_symptom_radar_chart.png")
print("06_multi_metric_comparison.png")
print("07_coverage_analysis.png")
print("08_detailed_metrics_table.png")
print("\nAll tables and statistics printed above for easy copy-paste into your report.")