"""
Run Advanced Budget Optimization Pipeline

Demonstrates all advanced features:
1. Cold Start - New campaign initialization
2. Response Curves - Diminishing returns modeling
3. E&E (Bandit) - Exploration vs Exploitation
4. Realtime Controller - Dynamic adjustment with +-20% flexibility

Usage:
    python run_advanced.py
"""

from automation.optimizer import BudgetOptimizer
from utils.data_simulator import CampaignDataSimulator
from algorithms.realtime_controller import RealtimeBudgetController


def main():
    """Run the full advanced pipeline demonstration."""

    print("\n" + "="*70)
    print("ADVANCED BUDGET OPTIMIZATION SYSTEM")
    print("="*70)
    print("""
This demonstrates the improved system addressing:

1. MODULE CONNECTIVITY (Completed in Phase 1)
   - Prediction -> Allocation weights
   - Clustering -> Strategy constraints

2. COLD START STRATEGY
   - Similar campaign transfer
   - Category baselines
   - Exploration-weighted initialization

3. DIMINISHING RETURNS (Response Curves)
   - Logarithmic / Square root / Hill function fitting
   - Marginal equilibrium allocation

4. EXPLORATION & EXPLOITATION (Bandit)
   - Thompson Sampling
   - UCB (Upper Confidence Bound)

5. REALTIME DYNAMIC ADJUSTMENT
   - Budget pacing
   - Performance-based reallocation
   - +-20% daily budget flexibility
    """)

    # Initialize optimizer with real data
    optimizer = BudgetOptimizer('data/data_for_ads.xlsx')

    # Run the advanced pipeline
    print("\n" + "="*70)
    print("RUNNING ADVANCED PIPELINE")
    print("="*70)

    results = optimizer.run_advanced_pipeline(
        campaign_name='Campaign_A',
        daily_budget=20.0,
        weekly_budget=150.0,
        day_of_week=1,
        product_category='electronics',
        enable_cold_start=True,
        enable_response_curves=True,
        enable_bandit=True
    )

    # Print summary
    print_summary(results)

    return results


def demo_realtime_simulation():
    """
    Demonstrate realtime controller with simulated events.
    """
    print("\n" + "="*70)
    print("REALTIME CONTROLLER SIMULATION")
    print("="*70)

    # Create simulator
    simulator = CampaignDataSimulator(seed=42)

    # Planned allocation (example)
    planned = {h: 20.0 / 24 for h in range(24)}  # Equal distribution
    planned.update({20: 1.5, 21: 1.5, 22: 1.2})  # Boost evening

    # Create controller
    controller = RealtimeBudgetController(
        planned_allocation=planned,
        daily_budget=20.0,
        baseline_roas={h: 3.0 for h in range(24)}
    )

    print(f"\nInitial daily budget: ${controller.base_daily_budget:.2f}")
    print(f"Budget flexibility: ${controller.min_daily_budget:.2f} - ${controller.max_daily_budget:.2f}")

    # Simulate high-performance scenario
    print("\n--- Simulating HIGH PERFORMANCE scenario ---")
    events = simulator.simulate_realtime_events(
        planned_allocation=planned,
        performance_scenario='high_performance'
    )

    result = controller.simulate_day(events, check_every_n_events=10)

    print(f"\nSimulation Results:")
    print(f"   Total spent: ${result['summary']['total_spent']:.2f}")
    print(f"   Total revenue: ${result['summary']['total_revenue']:.2f}")
    print(f"   Overall ROAS: {result['summary']['overall_roas']:.2f}x")
    print(f"   Daily budget change: {result['summary']['daily_budget_change_pct']:.1f}%")
    print(f"   Number of adjustments: {result['summary']['n_adjustments']}")

    if result['adjustments']:
        print("\n   Adjustments made:")
        for adj in result['adjustments'][:5]:  # Show first 5
            print(f"      - Hour {adj['hour']}: {adj['type']} - {adj['reason'][:50]}...")

    return result


def demo_cold_start():
    """
    Demonstrate cold start for a new campaign.
    """
    print("\n" + "="*70)
    print("COLD START DEMONSTRATION")
    print("="*70)

    optimizer = BudgetOptimizer('data/data_for_ads.xlsx')

    # Simulate a new campaign with known category
    print("\n--- New Campaign with Category Info ---")
    result1 = optimizer.optimize_with_cold_start(
        campaign_name='New_Campaign_X',
        daily_budget=25.0,
        day_of_week=1,
        product_category='electronics',
        audience_type='retargeting'
    )

    print(f"\nStrategy used: {result1['cold_start_strategy']}")
    print(f"Confidence: {result1['confidence']:.2f}")
    print(f"Exploration ratio: {result1['exploration_ratio']:.0%}")

    # Simulate completely unknown campaign
    print("\n--- Completely New Campaign (No Info) ---")
    result2 = optimizer.optimize_with_cold_start(
        campaign_name='Unknown_Campaign',
        daily_budget=25.0,
        day_of_week=1,
        product_category=None,
        audience_type=None
    )

    print(f"\nStrategy used: {result2['cold_start_strategy']}")
    print(f"Confidence: {result2['confidence']:.2f}")
    print(f"Exploration ratio: {result2['exploration_ratio']:.0%}")

    return result1, result2


def demo_response_curves():
    """
    Demonstrate response curve fitting and allocation.
    """
    print("\n" + "="*70)
    print("RESPONSE CURVE DEMONSTRATION")
    print("="*70)

    optimizer = BudgetOptimizer('data/data_for_ads.xlsx')

    result = optimizer.optimize_with_response_curves(
        total_budget=100.0,
        campaigns=None  # All campaigns
    )

    print("\n--- Response Curve Results ---")
    print(f"Optimization method: {result['optimization_method']}")
    print(f"Converged: {result['converged']}")
    print(f"Total expected revenue: ${result['total_expected_revenue']:.2f}")

    print("\n--- Allocation by Campaign ---")
    for campaign in list(result['allocation'].keys())[:5]:  # Show first 5
        alloc = result['allocation'][campaign]
        exp_rev = result['expected_revenue'].get(campaign, 0)
        curve = result['curve_results'].get(campaign, {})
        print(f"   {campaign}: ${alloc:.2f} -> ${exp_rev:.2f} ({curve.get('curve_type', 'N/A')})")

    return result


def demo_bandit():
    """
    Demonstrate E&E bandit allocation.
    """
    print("\n" + "="*70)
    print("E&E (BANDIT) DEMONSTRATION")
    print("="*70)

    optimizer = BudgetOptimizer('data/data_for_ads.xlsx')

    # Thompson Sampling
    print("\n--- Thompson Sampling ---")
    ts_result = optimizer.optimize_with_bandit(
        total_budget=100.0,
        method='thompson',
        exploration_ratio=0.15
    )

    print(f"Exploration ratio: {ts_result['exploration_ratio']:.0%}")
    print("\nAllocation:")
    for campaign in list(ts_result['allocation'].keys())[:5]:
        alloc = ts_result['allocation'][campaign]
        print(f"   {campaign}: ${alloc:.2f}")

    # UCB
    print("\n--- UCB (Upper Confidence Bound) ---")
    ucb_result = optimizer.optimize_with_bandit(
        total_budget=100.0,
        method='ucb',
        exploration_ratio=0.15
    )

    print(f"Exploration ratio: {ucb_result['exploration_ratio']:.0%}")
    print("\nAllocation:")
    for campaign in list(ucb_result['allocation'].keys())[:5]:
        alloc = ucb_result['allocation'][campaign]
        print(f"   {campaign}: ${alloc:.2f}")

    return ts_result, ucb_result


def print_summary(results):
    """Print summary of all results."""
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)

    print("\nOutput files generated:")
    print("  outputs/")
    print("  |-- cold_start_allocation_result.json")
    print("  |-- response_curve_allocation_result.json")
    print("  |-- bandit_allocation_result.json")
    print("  |-- advanced_pipeline_result.json")

    if 'cold_start' in results and 'error' not in results['cold_start']:
        cs = results['cold_start']
        print(f"\nCold Start:")
        print(f"  Strategy: {cs.get('cold_start_strategy', 'N/A')}")
        print(f"  Confidence: {cs.get('confidence', 0):.2f}")

    if 'response_curves' in results and 'error' not in results['response_curves']:
        rc = results['response_curves']
        print(f"\nResponse Curves:")
        print(f"  Method: {rc.get('optimization_method', 'N/A')}")
        print(f"  Expected Revenue: ${rc.get('total_expected_revenue', 0):.2f}")

    if 'bandit' in results and 'error' not in results['bandit']:
        b = results['bandit']
        print(f"\nE&E Bandit:")
        print(f"  Method: {b.get('method', 'N/A')}")
        print(f"  Exploration: {b.get('exploration_ratio', 0):.0%}")

    if 'realtime_controller' in results and 'error' not in results['realtime_controller']:
        rt = results['realtime_controller']
        print(f"\nRealtime Controller:")
        print(f"  Status: {rt.get('status', 'N/A')}")
        print(f"  Flexibility: {rt.get('budget_flexibility', 'N/A')}")

    print("\n" + "="*70)


if __name__ == "__main__":
    # Run main pipeline
    main()

    # Uncomment to run individual demos:
    # demo_cold_start()
    # demo_response_curves()
    # demo_bandit()
    # demo_realtime_simulation()
