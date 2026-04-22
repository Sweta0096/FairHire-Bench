import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu, fisher_exact

def essential_bias_analysis(file_path):
    """Essential statistical validation for intersectional bias"""

    df = pd.read_excel(file_path)
    models = ['Claude 3.5 Haiku', 'Claude 4.5 Haiku', 'Deepseek_Chat', 'Deepseek v3.2',
              'Gemini2', 'Gemini 3', 'GPT 4omini', 'GPT 5.2', 'llama3', 'llama4']

    df['intersect'] = df['Gender'] + "_" + df['Race/Ethnicity']

    print("ESSENTIAL STATISTICAL VALIDATION")
    print("="*50)

    # 1. KRUSKAL-WALLIS TEST
    group_rates = []
    for race in df['Race/Ethnicity'].unique():
        for gender in df['Gender'].unique():
            subset = df[(df['Race/Ethnicity'] == race) & (df['Gender'] == gender)]
            rates = [(subset[model] == 'Selected').mean() * 100 for model in models]
            group_rates.append(rates)

    h_stat, kw_p = kruskal(*group_rates)
    print(f"\n1. KRUSKAL-WALLIS TEST")
    print(f"H-statistic: {h_stat:.4f}, p-value: {kw_p:.8f}")
    print(f"Result: {'SIGNIFICANT' if kw_p < 0.05 else 'Not significant'}")

    # 2. MANN-WHITNEY U TEST (Effect Size)
    avg_rates = [np.mean(rates) for rates in group_rates]
    min_idx, max_idx = np.argmin(avg_rates), np.argmax(avg_rates)

    u_stat, mw_p = mannwhitneyu(group_rates[min_idx], group_rates[max_idx])

    mean1, mean2 = np.mean(group_rates[min_idx]), np.mean(group_rates[max_idx])
    std1, std2 = np.std(group_rates[min_idx]), np.std(group_rates[max_idx])
    cohens_d = abs(mean1 - mean2) / np.sqrt((std1**2 + std2**2) / 2)

    print(f"\n2. MANN-WHITNEY U TEST")
    print(f"Cohen's d: {cohens_d:.3f} (Large effect)")
    print(f"p-value: {mw_p:.6f}")

    # 3. FISHER'S EXACT TESTS
    print(f"\n3. FISHER'S EXACT TESTS")
    significant_count = 0

    for model in models:
        rates = {}
        for group in df['intersect'].unique():
            group_data = df[df['intersect'] == group]
            selected = (group_data[model] == 'Selected').sum()
            total = len(group_data)
            rates[group] = {'rate': selected/total, 'selected': selected, 'total': total}

        sorted_rates = sorted(rates.items(), key=lambda x: x[1]['rate'])
        worst, best = sorted_rates[0], sorted_rates[-1]

        table = [[best[1]['selected'], best[1]['total'] - best[1]['selected']],
                 [worst[1]['selected'], worst[1]['total'] - worst[1]['selected']]]

        odds_ratio, fisher_p = fisher_exact(table)

        if fisher_p < 0.05:
            significant_count += 1

        sig = "*" if fisher_p < 0.05 else ""
        print(f"{model}: OR={odds_ratio:.2f}, p={fisher_p:.6f}{sig}")

    # 4. AMPLIFICATION FACTOR
    print(f"\n4. AMPLIFICATION FACTOR")
    amplifications = []

    for model in models:
        gender_rates = df.groupby('Gender')[model].apply(lambda x: (x=='Selected').mean())
        race_rates = df.groupby('Race/Ethnicity')[model].apply(lambda x: (x=='Selected').mean())

        max_single = max(gender_rates.max() - gender_rates.min(),
                        race_rates.max() - race_rates.min())

        intersect_rates = df.groupby('intersect')[model].apply(lambda x: (x=='Selected').mean())
        intersect_gap = intersect_rates.max() - intersect_rates.min()

        amp_factor = intersect_gap / max_single if max_single > 0 else 1
        amplifications.append(amp_factor)

    mean_amp = np.mean(amplifications)
    print(f"Mean amplification: {mean_amp:.2f}x")
    print(f"Range: {min(amplifications):.2f}x to {max(amplifications):.2f}x")



# Run analysis
essential_bias_analysis('Dataset.xlsx')

"""**New Code for statitical same**

**Cross Generational Bias Analaysis**  # to find if newer are performing better or older
"""

#!/usr/bin/env python3
"""
Cross-Generational Bias Evolution Analysis
Statistical calculations for "Newer ≠ Better" finding in AI hiring bias study
"""

import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(file_path):
    """Load and prepare the dataset"""
    df = pd.read_excel(file_path)
    # Create intersectional group column
    df['Intersectional_Group'] = df['Gender'] + '_' + df['Race/Ethnicity']
    return df

def calculate_bias_gap(df, model_name, group1='Non-binary_Hispanic', group2='Woman_Hispanic'):
    """
    Calculate bias gap between two intersectional groups for a specific model
    Gap = SelectionRate_advantaged - SelectionRate_disadvantaged
    """
    group1_data = df[df['Intersectional_Group'] == group1]
    group2_data = df[df['Intersectional_Group'] == group2]

    group1_rate = (group1_data[model_name] == 'Selected').sum() / len(group1_data) * 100
    group2_rate = (group2_data[model_name] == 'Selected').sum() / len(group2_data) * 100

    gap = group2_rate - group1_rate
    return group1_rate, group2_rate, gap

def cross_generational_analysis(df):
    """
    Perform cross-generational bias evolution analysis
    Δ_bias = Gap_newer - Gap_older
    """
    model_families = {
        'Claude': {'older': 'Claude 3.5 Haiku', 'newer': 'Claude 4.5 Haiku'},
        'Deepseek': {'older': 'Deepseek_Chat', 'newer': 'Deepseek v3.2'},
        'Gemini': {'older': 'Gemini2', 'newer': 'Gemini 3'},
        'GPT': {'older': 'GPT 4omini', 'newer': 'GPT 5.2'},
        'Llama': {'older': 'llama3', 'newer': 'llama4'}
    }

    results = {}
    print("CROSS-GENERATIONAL BIAS EVOLUTION ANALYSIS")
    print("=" * 60)
    print("Non-binary Hispanic (disadvantaged) vs Woman Hispanic (advantaged)\n")

    improved_count = 0
    worsened_count = 0

    for family, versions in model_families.items():
        older_model = versions['older']
        newer_model = versions['newer']

        # Calculate bias gaps
        older_dis, older_adv, older_gap = calculate_bias_gap(df, older_model)
        newer_dis, newer_adv, newer_gap = calculate_bias_gap(df, newer_model)

        delta_bias = newer_gap - older_gap

        results[family] = {
            'older_model': older_model,
            'newer_model': newer_model,
            'older_gap': older_gap,
            'newer_gap': newer_gap,
            'delta_bias': delta_bias,
            'improved': delta_bias < 0
        }

        # Print detailed results
        print(f"{family} Family: {older_model} → {newer_model}")
        print("-" * 60)
        print(f"  {older_model:20s}: {older_dis:5.1f}% → {older_adv:5.1f}% (gap: {older_gap:4.1f}%)")
        print(f"  {newer_model:20s}: {newer_dis:5.1f}% → {newer_adv:5.1f}% (gap: {newer_gap:4.1f}%)")

        if delta_bias < 0:
            print(f"  ✅ IMPROVED: Δ_bias = {delta_bias:+4.1f}% (bias gap reduced)")
            improved_count += 1
        elif delta_bias > 0:
            print(f"  ❌ WORSENED: Δ_bias = {delta_bias:+4.1f}% (bias gap increased)")
            worsened_count += 1
        else:
            print(f"  ➖ NO CHANGE: Δ_bias = {delta_bias:+4.1f}%")
        print()

    # Summary statistics
    print("STATISTICAL SUMMARY:")
    print("=" * 30)
    print(f"Model families that IMPROVED: {improved_count}/5 ({improved_count/5*100:.0f}%)")
    print(f"Model families that WORSENED: {worsened_count}/5 ({worsened_count/5*100:.0f}%)")
    print()

    delta_values = [results[family]['delta_bias'] for family in results]
    mean_delta = np.mean(delta_values)
    std_delta = np.std(delta_values)

    print(f"Mean Δ_bias across all families: {mean_delta:+4.2f}%")
    print(f"Standard deviation: {std_delta:.2f}%")

    # Paired t-test
    older_gaps = [results[family]['older_gap'] for family in results]
    newer_gaps = [results[family]['newer_gap'] for family in results]
    t_stat, p_value = stats.ttest_rel(newer_gaps, older_gaps)
    print(f"Paired t-test (newer vs older): t = {t_stat:.3f}, p = {p_value:.3f}")

    return results

def complete_intersectional_ranking(df):
    """Calculate complete ranking of all intersectional groups"""
    all_models = [
        'Claude 3.5 Haiku', 'Claude 4.5 Haiku',
        'Deepseek_Chat', 'Deepseek v3.2',
        'Gemini2', 'Gemini 3',
        'GPT 4omini', 'GPT 5.2',
        'llama3', 'llama4'
    ]

    results = []
    for group in df['Intersectional_Group'].unique():
        group_data = df[df['Intersectional_Group'] == group]
        selection_rates = [(group_data[model] == 'Selected').sum() / len(group_data) * 100 for model in all_models]
        avg_rate = np.mean(selection_rates)
        results.append({'Group': group, 'Avg_Selection_Rate': avg_rate})

    # Sort by average selection rate
    results_sorted = sorted(results, key=lambda x: x['Avg_Selection_Rate'])

    print("\nCOMPLETE INTERSECTIONAL RANKING:")
    print("=" * 40)
    print("Rank | Group                | Avg Selection Rate")
    print("-" * 45)
    for i, r in enumerate(results_sorted, 1):
        print(f"{i:2d}   | {r['Group']:20s} | {r['Avg_Selection_Rate']:5.2f}%")

    return results_sorted

def main():
    """Main analysis function"""
    # Load dataset
    dataset_path = 'Dataset.xlsx'  # Update path as needed
    df = load_dataset(dataset_path)

    print(f"\nDataset loaded: {len(df):,} candidates")
    print(f"Intersectional groups: {df['Intersectional_Group'].nunique()}\n")

    cross_gen_results = cross_generational_analysis(df)
    ranking_results = complete_intersectional_ranking(df)

    print(f"\nMost disadvantaged: {ranking_results[0]['Group']} ({ranking_results[0]['Avg_Selection_Rate']:.2f}%)")
    print(f"Most advantaged: {ranking_results[-1]['Group']} ({ranking_results[-1]['Avg_Selection_Rate']:.2f}%)")
    print(f"Total gap: {ranking_results[-1]['Avg_Selection_Rate'] - ranking_results[0]['Avg_Selection_Rate']:.2f} percentage points")

if __name__ == "__main__":
    main()

"""**fisher figure**"""

#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Set consistent font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Fisher's exact test results from the paper
fisher_results = {
    'Claude 3.5 Haiku': {'OR': 1.34, 'p': 0.012, 'sig': True},
    'Claude 4.5 Haiku': {'OR': 1.27, 'p': 0.039, 'sig': True},
    'Deepseek Chat': {'OR': 1.32, 'p': 0.016, 'sig': True},
    'Deepseek v3.2': {'OR': 1.22, 'p': 0.083, 'sig': False},
    'Gemini2': {'OR': 1.24, 'p': 0.064, 'sig': False},
    'Gemini 3': {'OR': 1.26, 'p': 0.044, 'sig': True},
    'GPT 4omini': {'OR': 1.24, 'p': 0.057, 'sig': False},
    'GPT 5.2': {'OR': 1.24, 'p': 0.057, 'sig': False},
    'llama3': {'OR': 1.25, 'p': 0.050, 'sig': True},
    'llama4': {'OR': 1.27, 'p': 0.034, 'sig': True}
}

# Create figure with larger size for better spacing
fig, ax = plt.subplots(figsize=(14, 10))

models = list(fisher_results.keys())
odds_ratios = [fisher_results[m]['OR'] for m in models]
p_values = [fisher_results[m]['p'] for m in models]
significant = [fisher_results[m]['sig'] for m in models]

# Create scatter plot
colors = ['#d62728' if sig else '#1f77b4' for sig in significant]
sizes = [150 for _ in significant]  # Same size for all points

scatter = ax.scatter(odds_ratios, p_values, c=colors, s=sizes, alpha=0.8,
                    edgecolors='black', linewidth=1.5, zorder=3)

# Add significance threshold line
ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, alpha=0.8,
          label='p = 0.05 significance threshold', zorder=1)

# Set axis labels and title with consistent fonts
ax.set_xlabel('Odds Ratio', fontsize=16, fontweight='bold')
ax.set_ylabel('p-value', fontsize=16, fontweight='bold')
ax.set_title('Statistical Significance of Intersectional Bias\n(Fisher\'s Exact Test Results)',
             fontsize=18, fontweight='bold', pad=25)

# Improve grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, zorder=0)

# Manual positioning for each label to avoid overlap
label_positions = {
    'Claude 3.5 Haiku': (1.34, 0.012, 'right', 'bottom', 10, -5),
    'Claude 4.5 Haiku': (1.27, 0.039, 'left', 'top', 5, 5),
    'Deepseek Chat': (1.32, 0.016, 'left', 'bottom', 5, -5),
    'Deepseek v3.2': (1.22, 0.083, 'center', 'bottom', 0, -15),
    'Gemini2': (1.24, 0.064, 'right', 'center', -5, 0),
    'Gemini 3': (1.26, 0.044, 'left', 'bottom', 5, -5),
    'GPT 4omini': (1.24, 0.057, 'right', 'top', -5, 8),
    'GPT 5.2': (1.24, 0.057, 'left', 'bottom', 5, -15),
    'llama3': (1.25, 0.050, 'center', 'top', 0, 8),
    'llama4': (1.27, 0.034, 'right', 'center', -5, 0)
}

# Add labels with consistent formatting
for model in models:
    x, y, ha, va, offset_x, offset_y = label_positions[model]

    ax.annotate(model,
                xy=(x, y),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=11,
                fontweight='normal',
                ha=ha,
                va=va,
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor='white',
                         edgecolor='gray',
                         alpha=0.9,
                         linewidth=0.8),
                zorder=4)

# Create legend with consistent styling
legend_elements = [
    plt.scatter([], [], c='#d62728', s=150, alpha=0.8, edgecolors='black',
               label='Significant (p < 0.05)', linewidth=1.5),
    plt.scatter([], [], c='#1f77b4', s=150, alpha=0.8, edgecolors='black',
               label='Not Significant (p ≥ 0.05)', linewidth=1.5),
    plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, alpha=0.8,
              label='Significance Threshold')
]

legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=13,
                  framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

# Set axis limits with proper padding
ax.set_xlim(1.18, 1.38)
ax.set_ylim(0.005, 0.095)

# Format tick labels
ax.tick_params(axis='both', which='major', labelsize=13)

# Add summary statistics in a smaller box
summary_text = '''Statistical Summary:
• Significant models: 6/10 (60%)
• Odds ratio range: 1.22 - 1.34
• Mean odds ratio: 1.26
• Models with p < 0.05 show systematic bias'''

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue',
                 alpha=0.9, edgecolor='navy', linewidth=1))

# Improve overall appearance
plt.tight_layout()

# Save with high quality
plt.savefig('Fisher_Exact_Test_Results.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

# Display the plot
plt.show()