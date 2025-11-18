"""
Main script to run budget optimization.

This is the entry point - just run this file and everything happens automatically!
"""

from automation.optimizer import BudgetOptimizer


def main():
    """
    Main function - runs all optimizations.

    This demonstrates the "automation module" feature:
    One line of code runs everything!
    """

    # Initialize optimizer with data file
    optimizer = BudgetOptimizer('data/data_for_ads.xlsx')

    # Option 1: Run everything automatically (EASIEST!)
    print("\nRunning all optimizations automatically...\n")
    results = optimizer.run_all_optimizations()

    # Option 2: Run individual optimizations (if you prefer)
    # Uncomment these to run individually:

    # # Task 1: Hourly allocation
    # hourly_result = optimizer.optimize_hourly(
    #     campaign_name='Campaign_A',
    #     daily_budget=20.0,
    #     day_of_week=1  # Monday
    # )

    # # Task 1: Weekly allocation
    # weekly_result = optimizer.optimize_weekly(
    #     campaign_name='Campaign_A',
    #     weekly_budget_limit=150.0,
    #     min_daily_budget=15.0,
    #     max_daily_budget=30.0
    # )

    # # Task 2: Multi-campaign optimization
    # multi_result = optimizer.optimize_multi_campaign(
    #     total_budget=100.0,
    #     optimization_target="maximize_revenue"
    # )

    # # Optional 1: Clustering
    # clustering_result = optimizer.analyze_clustering(n_clusters=3)

    # # Optional 2: ML Prediction
    # ml_result = optimizer.train_predictor()

    print("\n" + "="*70)
    print("[DONE] ALL COMPLETE!")
    print("="*70)
    print("\nCheck outputs/ folder for results:")
    print("   - task1_hourly_allocation_result.json")
    print("   - task1_weekly_allocation_result.json")
    print("   - task2_multi_campaign_result.json")
    print("   - optional1_clustering_result.json")
    print("   - revenue_model.joblib (if ML trained successfully)")
    print("\nNext step: Run visualize.py to create charts!")
    print("="*70)


if __name__ == "__main__":
    main()
