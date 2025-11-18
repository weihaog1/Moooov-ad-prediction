"""
Visualization script - creates all charts from results.

Run this after run_optimization.py to create visualizations!
"""

from utils.visualizer import visualize_all_results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    visualize_all_results('outputs')

    print("\n" + "="*70)
    print("[DONE]")
    print("="*70)
    print("\nCheck outputs/ folder for PNG files:")
    print("   - viz_hourly_allocation.png")
    print("   - viz_weekly_allocation.png")
    print("   - viz_multi_campaign.png")
    print("   - viz_clustering.png")
    print("="*70)
