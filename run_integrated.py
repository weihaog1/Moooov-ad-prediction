"""
Run Integrated Budget Optimization Pipeline

This script demonstrates the connected system where:
1. Revenue predictions inform allocation weights
2. Campaign clustering drives budget strategy
3. All components work together in a unified pipeline

Usage:
    python run_integrated.py
"""

from automation.optimizer import BudgetOptimizer


def main():
    """Run the integrated optimization pipeline."""

    print("\n" + "="*70)
    print("INTEGRATED BUDGET OPTIMIZATION PIPELINE")
    print("="*70)
    print("""
This demonstrates the improved system architecture:

BEFORE (Disconnected):
    Excel → Predictor → JSON (unused)
    Excel → Clusterer → JSON (unused)
    Excel → Allocator → Allocation

AFTER (Connected):
    Excel → Predictor ─┐
                       ├──→ Integrated Allocator → Allocation
    Excel → Clusterer ─┘

Key improvements:
1. Predictions used as allocation weights (forward-looking)
2. Clustering determines budget strategy (constraints)
3. Single pipeline produces informed allocations
    """)

    # Initialize optimizer
    optimizer = BudgetOptimizer('data/data_for_ads.xlsx')

    # Option 1: Run the full integrated pipeline
    print("\n" + "="*70)
    print("OPTION 1: FULL INTEGRATED PIPELINE")
    print("="*70)

    results = optimizer.run_integrated_optimizations(
        campaign_name='Campaign_A',
        daily_budget=20.0,
        weekly_budget=150.0,
        day_of_week=1
    )

    # Option 2: Compare legacy vs integrated
    print("\n" + "="*70)
    print("OPTION 2: LEGACY VS INTEGRATED COMPARISON")
    print("="*70)

    comparison = optimizer.compare_legacy_vs_integrated(
        campaign_name='Campaign_A',
        daily_budget=20.0,
        day_of_week=1
    )

    # Summary
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)

    print("\nOutput files generated:")
    print("  outputs/")
    print("  ├── integrated_clustering_result.json    # Cluster assignments & strategies")
    print("  ├── task1_integrated_hourly_result.json  # Hourly allocation with predictions")
    print("  ├── task1_integrated_weekly_result.json  # Weekly allocation with predictions")
    print("  ├── comparison_legacy_vs_integrated_result.json  # Side-by-side comparison")
    print("  └── revenue_model.joblib                 # Trained prediction model")

    print("\nKey differences from legacy approach:")
    if 'integrated' in comparison:
        integrated = comparison['integrated']
        print(f"  - Weight source: {integrated.get('weight_source', 'N/A')}")
        if 'strategy' in integrated:
            print(f"  - Strategy applied: {integrated['strategy'].get('type', 'N/A')}")
            print(f"  - Strategy description: {integrated['strategy'].get('description', 'N/A')}")

    print("\n" + "="*70)


def run_step_by_step():
    """
    Alternative: Run pipeline step by step for more control.

    Use this if you want to:
    - Inspect intermediate results
    - Customize parameters at each stage
    - Run specific components only
    """

    optimizer = BudgetOptimizer('data/data_for_ads.xlsx')

    # Step 1: Initialize pipeline with custom settings
    print("\n[Step 1] Initializing pipeline...")
    init_result = optimizer.initialize_integrated_pipeline(
        n_clusters=3,           # Number of campaign groups
        train_predictor=True,   # Train ML model
        save_components=True    # Save intermediate results
    )

    print(f"\nClusters created: {init_result['clustering']['n_clusters']}")
    print(f"Predictor R2: {init_result['predictor_training']['test_metrics']['r2']:.4f}")

    # Step 2: Run hourly optimization for specific campaign/day
    print("\n[Step 2] Running hourly optimization...")
    hourly_result = optimizer.optimize_hourly_integrated(
        campaign_name='Campaign_A',
        daily_budget=20.0,
        day_of_week=1,
        use_prediction=True,    # Use ML predictions for weights
        use_strategy=True       # Apply cluster-based strategy
    )

    print(f"\nAllocation complete:")
    print(f"  Total: ${hourly_result['total_allocated']}")
    print(f"  Weight source: {hourly_result['weight_source']}")
    print(f"  Strategy: {hourly_result.get('strategy', {}).get('type', 'none')}")

    # Step 3: Run weekly optimization
    print("\n[Step 3] Running weekly optimization...")
    weekly_result = optimizer.optimize_weekly_integrated(
        campaign_name='Campaign_A',
        weekly_budget_limit=150.0,
        min_daily_budget=15.0,
        max_daily_budget=30.0
    )

    print(f"\nWeekly allocation complete:")
    print(f"  Total: ${weekly_result['weekly_total']}")

    return optimizer, hourly_result, weekly_result


if __name__ == "__main__":
    main()

    # Uncomment to run step-by-step instead:
    # run_step_by_step()
