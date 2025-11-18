"""
Simple visualization utility.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def visualize_all_results(output_dir: str = 'outputs'):
    """
    Create all visualizations from result files.

    Args:
        output_dir: Directory containing result JSON files
    """
    output_path = Path(output_dir)

    print("\nCreating visualizations...")

    # Load results
    results = {}
    files = {
        'hourly': 'task1_hourly_allocation_result.json',
        'weekly': 'task1_weekly_allocation_result.json',
        'multi': 'task2_multi_campaign_result.json',
        'clustering': 'optional1_clustering_result.json'
    }

    for key, filename in files.items():
        filepath = output_path / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[key] = json.load(f)
            print(f"[OK] Loaded {filename}")

    # Create visualizations
    if 'hourly' in results:
        plot_hourly_allocation(results['hourly'], output_path)

    if 'weekly' in results:
        plot_weekly_allocation(results['weekly'], output_path)

    if 'multi' in results:
        plot_multi_campaign(results['multi'], output_path)

    if 'clustering' in results:
        plot_clustering(results['clustering'], output_path)

    print("\n[DONE] All visualizations created!")
    print(f"Saved to: {output_path}/")


def plot_hourly_allocation(result: dict, output_dir: Path):
    """Plot hourly allocation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task 1: Hourly Budget Allocation', fontsize=16, fontweight='bold')

    # Extract data
    allocation = {int(k): v for k, v in result['hourly_allocation'].items()}
    hours = list(range(24))
    budgets = [allocation[h] for h in hours]

    # 1. Bar chart
    ax1 = axes[0, 0]
    ax1.bar(hours, budgets, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Budget ($)')
    ax1.set_title('Hourly Budget Distribution')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Cumulative
    ax2 = axes[0, 1]
    cumulative = np.cumsum(budgets)
    ax2.plot(hours, cumulative, marker='o', linewidth=2, color='darkgreen')
    ax2.fill_between(hours, cumulative, alpha=0.3, color='green')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Cumulative Budget ($)')
    ax2.set_title('Cumulative Allocation')
    ax2.grid(alpha=0.3)

    # 3. Metrics
    ax3 = axes[1, 0]
    ax3.axis('off')
    metrics_text = f"""
    Budget: ${result['daily_budget']:.2f}
    Allocated: ${result['total_allocated']:.2f}

    Expected Revenue: ${result['expected_revenue']:.2f}
    Expected ROAS: {result['expected_roas']:.2f}x

    Smoothness: {result['smoothness_score']:.6f}
    Peak Hours: {result['peak_hours']}
    """
    ax3.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Time periods
    ax4 = axes[1, 1]
    periods = {
        'Night (0-5)': sum(budgets[0:6]),
        'Morning (6-11)': sum(budgets[6:12]),
        'Afternoon (12-17)': sum(budgets[12:18]),
        'Evening (18-23)': sum(budgets[18:24])
    }
    ax4.pie(periods.values(), labels=periods.keys(), autopct='%1.1f%%', startangle=90)
    ax4.set_title('Budget by Time Period')

    plt.tight_layout()
    plt.savefig(output_dir / 'viz_hourly_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Created viz_hourly_allocation.png")


def plot_weekly_allocation(result: dict, output_dir: Path):
    """Plot weekly allocation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task 1: Weekly Budget Strategy', fontsize=16, fontweight='bold')

    weekly = result['weekly_budget_strategy']
    days = list(weekly.keys())
    budgets = [weekly[d]['daily_budget'] for d in days]
    revenues = [weekly[d]['expected_revenue'] for d in days]
    roas = [weekly[d]['expected_roas'] for d in days]

    x = np.arange(len(days))

    # 1. Daily budgets
    ax1 = axes[0, 0]
    ax1.bar(x, budgets, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Budget ($)')
    ax1.set_title('Daily Budget Allocation')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in days], rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Expected revenues
    ax2 = axes[0, 1]
    ax2.bar(x, revenues, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Revenue ($)')
    ax2.set_title('Expected Revenue by Day')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.capitalize() for d in days], rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    # 3. ROAS trend
    ax3 = axes[1, 0]
    ax3.plot(x, roas, marker='o', linewidth=2, color='darkred')
    ax3.fill_between(x, roas, alpha=0.3, color='red')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('ROAS')
    ax3.set_title('ROAS by Day')
    ax3.set_xticks(x)
    ax3.set_xticklabels([d.capitalize() for d in days], rotation=45)
    ax3.grid(alpha=0.3)

    # 4. Budget vs Revenue
    ax4 = axes[1, 1]
    width = 0.35
    ax4.bar(x - width/2, budgets, width, label='Budget', color='steelblue', alpha=0.7)
    ax4.bar(x + width/2, revenues, width, label='Revenue', color='green', alpha=0.7)
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Amount ($)')
    ax4.set_title('Budget vs Revenue')
    ax4.set_xticks(x)
    ax4.set_xticklabels([d.capitalize() for d in days], rotation=45)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'viz_weekly_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Created viz_weekly_allocation.png")


def plot_multi_campaign(result: dict, output_dir: Path):
    """Plot multi-campaign optimization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Task 2: Multi-Campaign Optimization', fontsize=16, fontweight='bold')

    allocation = result['allocation']
    campaigns = list(allocation.keys())
    budgets = list(allocation.values())

    # Sort by budget
    sorted_pairs = sorted(zip(campaigns, budgets), key=lambda x: x[1], reverse=True)
    campaigns, budgets = zip(*sorted_pairs)

    # 1. Allocation bars
    ax1 = axes[0, 0]
    ax1.barh(campaigns, budgets, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Budget ($)')
    ax1.set_title('Budget Allocation by Campaign')
    ax1.grid(axis='x', alpha=0.3)

    # 2. Pie chart (top 5 + others)
    ax2 = axes[0, 1]
    if len(campaigns) > 6:
        pie_labels = list(campaigns[:5]) + ['Others']
        pie_values = list(budgets[:5]) + [sum(budgets[5:])]
    else:
        pie_labels = campaigns
        pie_values = budgets
    ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Budget Distribution')

    # 3. Metrics
    ax3 = axes[1, 0]
    ax3.axis('off')
    outcomes = result['expected_outcomes']
    metrics_text = f"""
    Total Budget: ${result['total_budget']:.2f}
    Allocated: ${result['total_allocated']:.2f}

    Expected Revenue: ${outcomes['total_revenue']:.2f}
    Overall ROAS: {outcomes['overall_roas']:.2f}x
    Average ACoS: {outcomes['average_acos']*100:.2f}%
    """
    ax3.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 4. Recommendations
    ax4 = axes[1, 1]
    ax4.axis('off')
    recs = result['adjustment_recommendations']
    rec_text = f"""
    Recommendations:

    Increase Budget:
    {', '.join(recs['increase_budget'][:3])}

    Decrease Budget:
    {', '.join(recs['decrease_budget'][:3])}
    """
    ax4.text(0.1, 0.5, rec_text, fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'viz_multi_campaign.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Created viz_multi_campaign.png")


def plot_clustering(result: dict, output_dir: Path):
    """Plot clustering results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optional 1: Campaign Clustering', fontsize=16, fontweight='bold')

    campaigns = result['campaigns']
    stats = result['cluster_statistics']

    # 1. PCA scatter
    ax1 = axes[0, 0]
    for cluster_id in set(c['cluster_id'] for c in campaigns):
        cluster_data = [c for c in campaigns if c['cluster_id'] == cluster_id]
        x = [c['pca_x'] for c in cluster_data]
        y = [c['pca_y'] for c in cluster_data]
        ax1.scatter(x, y, label=f'Cluster {cluster_id}', s=100, alpha=0.6)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Campaign Clusters (PCA)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Cluster sizes
    ax2 = axes[0, 1]
    cluster_ids = [int(k) for k in stats.keys()]
    sizes = [stats[str(k)]['size'] for k in cluster_ids]
    ax2.bar(cluster_ids, sizes, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Campaigns')
    ax2.set_title('Cluster Sizes')
    ax2.set_xticks(cluster_ids)
    ax2.grid(axis='y', alpha=0.3)

    # 3. ROAS vs ACoS
    ax3 = axes[1, 0]
    roas_vals = [stats[str(k)]['avg_roas'] for k in cluster_ids]
    acos_vals = [stats[str(k)]['avg_acos'] for k in cluster_ids]
    ax3.scatter(acos_vals, roas_vals, s=300, alpha=0.6, c=cluster_ids, cmap='Set2', edgecolors='black')
    ax3.set_xlabel('Average ACoS')
    ax3.set_ylabel('Average ROAS')
    ax3.set_title('Cluster Characteristics')
    ax3.grid(alpha=0.3)

    # 4. Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    quality = result['quality_metrics']
    summary_text = f"""
    Clustering Results:

    Method: {result['method']}
    Clusters: {result['n_clusters']}

    Silhouette: {quality['silhouette_score']:.4f}
    Davies-Bouldin: {quality['davies_bouldin_score']:.4f}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'viz_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Created viz_clustering.png")
