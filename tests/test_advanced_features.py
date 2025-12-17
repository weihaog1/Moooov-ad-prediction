"""
Test Suite for Advanced Budget Optimization Features

Tests:
- Data Simulator
- Cold Start Handler
- Response Curve Modeling
- Bandit Algorithms (Thompson Sampling, UCB)
- Realtime Controller

Run with: python -m pytest tests/test_advanced_features.py -v
Or: python tests/test_advanced_features.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime


class TestDataSimulator:
    """Test data simulation functionality."""

    def test_generate_campaigns(self):
        """Test campaign data generation."""
        from utils.data_simulator import CampaignDataSimulator

        simulator = CampaignDataSimulator(seed=42)
        df = simulator.generate_campaigns(
            n_campaigns=5,
            n_days=7,
            include_new_campaigns=1
        )

        # Check structure
        assert len(df) > 0
        assert 'campaign_name' in df.columns
        assert 'hour' in df.columns
        assert 'revenue' in df.columns
        assert 'spend' in df.columns
        assert 'roas' in df.columns

        # Check data validity
        assert df['hour'].min() >= 0
        assert df['hour'].max() <= 23
        assert df['spend'].min() >= 0
        assert df['revenue'].min() >= 0

        print("[PASS] test_generate_campaigns")

    def test_diminishing_returns_data(self):
        """Test diminishing returns simulation."""
        from utils.data_simulator import CampaignDataSimulator

        simulator = CampaignDataSimulator(seed=42)
        df = simulator.simulate_diminishing_returns_data(
            campaign_name='Test',
            curve_type='log'
        )

        assert len(df) > 0
        assert 'spend' in df.columns
        assert 'revenue' in df.columns

        # Check diminishing returns pattern (revenue growth slows)
        low_budget = df[df['spend'] <= 20]['revenue'].mean()
        high_budget = df[df['spend'] >= 100]['revenue'].mean()

        # Revenue should increase but not proportionally
        assert high_budget > low_budget
        roas_low = low_budget / 10  # Approximate ROAS at low budget
        roas_high = high_budget / 150  # Approximate ROAS at high budget
        assert roas_low > roas_high  # Diminishing returns

        print("[PASS] test_diminishing_returns_data")

    def test_realtime_events(self):
        """Test realtime event simulation."""
        from utils.data_simulator import CampaignDataSimulator

        simulator = CampaignDataSimulator(seed=42)
        planned = {h: 1.0 for h in range(24)}

        events = simulator.simulate_realtime_events(
            planned_allocation=planned,
            performance_scenario='high_performance'
        )

        assert len(events) > 0
        assert all('hour' in e for e in events)
        assert all('spend' in e for e in events)
        assert all('revenue' in e for e in events)

        print("[PASS] test_realtime_events")


class TestColdStart:
    """Test cold start handler."""

    def test_category_baseline(self):
        """Test category baseline weights."""
        from algorithms.cold_start import ColdStartHandler

        handler = ColdStartHandler()

        # Test electronics baseline
        weights = handler.get_category_baseline('electronics', day_of_week=1)

        assert len(weights) == 24
        assert abs(weights.sum() - 1.0) < 0.01  # Should sum to 1

        # Peak hours should have higher weights
        peak_weight = np.mean([weights[h] for h in [12, 13, 14, 20, 21, 22]])
        off_peak_weight = np.mean([weights[h] for h in [0, 1, 2, 3, 4, 5]])
        assert peak_weight > off_peak_weight

        print("[PASS] test_category_baseline")

    def test_exploration_uniform(self):
        """Test exploration-weighted uniform distribution."""
        from algorithms.cold_start import ColdStartHandler

        handler = ColdStartHandler()
        weights = handler.get_exploration_weighted_uniform(day_of_week=1)

        assert len(weights) == 24
        assert abs(weights.sum() - 1.0) < 0.01

        # Evening hours should have slight boost
        evening_weight = np.mean([weights[h] for h in [18, 19, 20, 21, 22]])
        morning_weight = np.mean([weights[h] for h in [6, 7, 8, 9, 10]])
        assert evening_weight >= morning_weight

        print("[PASS] test_exploration_uniform")

    def test_get_initial_weights(self):
        """Test initial weights selection with historical data."""
        from algorithms.cold_start import ColdStartHandler
        from utils.data_simulator import generate_test_dataset

        df = generate_test_dataset(n_campaigns=10, n_days=14)
        handler = ColdStartHandler(historical_data=df)

        result = handler.get_initial_weights(
            campaign_name='New_Campaign',
            day_of_week=1,
            product_category='electronics'
        )

        assert 'weights' in result
        assert 'strategy' in result
        assert 'confidence' in result
        assert 'exploration_ratio' in result
        assert len(result['weights']) == 24

        print("[PASS] test_get_initial_weights")


class TestResponseCurve:
    """Test response curve modeling."""

    def test_curve_fitting(self):
        """Test response curve fitting."""
        from algorithms.response_curve import ResponseCurveModeler, CurveType
        from utils.data_simulator import CampaignDataSimulator

        simulator = CampaignDataSimulator(seed=42)
        df = simulator.simulate_diminishing_returns_data(
            campaign_name='Test_Campaign',
            curve_type='log'
        )

        modeler = ResponseCurveModeler(min_data_points=5)
        result = modeler.fit(df, 'Test_Campaign')

        assert result.curve_type in CurveType
        assert result.r_squared >= 0
        assert result.params is not None

        print("[PASS] test_curve_fitting")

    def test_predict_revenue(self):
        """Test revenue prediction."""
        from algorithms.response_curve import ResponseCurveModeler
        from utils.data_simulator import CampaignDataSimulator

        simulator = CampaignDataSimulator(seed=42)
        df = simulator.simulate_diminishing_returns_data(
            campaign_name='Test',
            curve_type='log'
        )

        modeler = ResponseCurveModeler(min_data_points=5)
        modeler.fit(df, 'Test')

        # Predict at different budget levels
        pred_10 = modeler.predict_revenue('Test', 10)
        pred_100 = modeler.predict_revenue('Test', 100)

        assert pred_10 > 0
        assert pred_100 > pred_10  # More budget = more revenue
        assert pred_100 < pred_10 * 10  # But not linear (diminishing)

        print("[PASS] test_predict_revenue")

    def test_marginal_return(self):
        """Test marginal return calculation."""
        from algorithms.response_curve import ResponseCurveModeler
        from utils.data_simulator import CampaignDataSimulator

        simulator = CampaignDataSimulator(seed=42)
        df = simulator.simulate_diminishing_returns_data(
            campaign_name='Test',
            curve_type='log'
        )

        modeler = ResponseCurveModeler(min_data_points=5)
        modeler.fit(df, 'Test')

        mr_10 = modeler.get_marginal_return('Test', 10)
        mr_100 = modeler.get_marginal_return('Test', 100)

        assert mr_10 > 0
        assert mr_100 > 0
        assert mr_10 > mr_100  # Diminishing marginal returns

        print("[PASS] test_marginal_return")


class TestBandit:
    """Test bandit algorithms."""

    def test_thompson_sampling(self):
        """Test Thompson Sampling allocator."""
        from algorithms.bandit import ThompsonSamplingAllocator

        campaigns = ['A', 'B', 'C']
        allocator = ThompsonSamplingAllocator(campaigns, target_roas=1.0)

        # Get allocation
        allocation = allocator.sample_allocation(total_budget=100, min_per_campaign=5)

        assert len(allocation) == 3
        assert all(c in allocation for c in campaigns)
        assert all(v >= 5 for v in allocation.values())  # Respects minimum

        # Update with results
        allocator.update('A', spent=30, revenue=90)  # Good performance
        allocator.update('B', spent=30, revenue=20)  # Poor performance

        # Get expected values
        expected = allocator.get_expected_values()
        assert expected['A'] > expected['B']  # A should be higher after good result

        print("[PASS] test_thompson_sampling")

    def test_ucb(self):
        """Test UCB allocator."""
        from algorithms.bandit import UCBAllocator

        campaigns = ['A', 'B', 'C']
        allocator = UCBAllocator(campaigns, exploration_weight=2.0)

        # Initial allocation (all unexplored)
        allocation = allocator.select_allocation(total_budget=100, min_per_campaign=5)

        assert len(allocation) == 3
        assert all(v > 0 for v in allocation.values())

        # Update and check UCB scores change
        allocator.update('A', spent=30, revenue=90)

        ucb_a = allocator.compute_ucb('A')
        ucb_b = allocator.compute_ucb('B')

        # B should have higher UCB (unexplored = infinite)
        assert ucb_b > ucb_a or ucb_b == float('inf')

        print("[PASS] test_ucb")

    def test_hourly_bandit(self):
        """Test hourly bandit allocation."""
        from algorithms.bandit import HourlyBandit

        bandit = HourlyBandit(target_roas=1.0)

        allocation = bandit.sample_hourly_allocation(
            daily_budget=24.0,
            min_per_hour=0.5
        )

        assert len(allocation) == 24
        assert all(v >= 0.5 for v in allocation.values())

        # Update some hours
        bandit.update_hour(hour=20, spent=1.0, revenue=3.0)  # Good
        bandit.update_hour(hour=3, spent=1.0, revenue=0.5)   # Poor

        expected = bandit.get_expected_values()
        assert expected[20] > expected[3]

        print("[PASS] test_hourly_bandit")


class TestRealtimeController:
    """Test realtime controller."""

    def test_controller_initialization(self):
        """Test controller initialization."""
        from algorithms.realtime_controller import RealtimeBudgetController

        planned = {h: 1.0 for h in range(24)}
        controller = RealtimeBudgetController(
            planned_allocation=planned,
            daily_budget=24.0
        )

        assert controller.base_daily_budget == 24.0
        assert controller.max_daily_budget == 24.0 * 1.2  # +20%
        assert controller.min_daily_budget == 24.0 * 0.8  # -20%

        print("[PASS] test_controller_initialization")

    def test_pacing_check(self):
        """Test pacing adjustment trigger."""
        from algorithms.realtime_controller import RealtimeBudgetController, AdjustmentType

        planned = {h: 1.0 for h in range(24)}
        controller = RealtimeBudgetController(
            planned_allocation=planned,
            daily_budget=24.0
        )

        # Simulate fast spending: 90% budget spent but only 20% time elapsed
        controller.spent[10] = 0.9  # 90% of hourly budget

        adjustments = controller.check_and_adjust(current_hour=10, current_minute=12)

        # Should trigger pacing
        pacing_adjustments = [a for a in adjustments if a.type == AdjustmentType.PACING]
        assert len(pacing_adjustments) > 0

        print("[PASS] test_pacing_check")

    def test_performance_boost(self):
        """Test performance boost adjustment."""
        from algorithms.realtime_controller import RealtimeBudgetController, AdjustmentType

        planned = {h: 1.0 for h in range(24)}
        baseline_roas = {h: 3.0 for h in range(24)}

        controller = RealtimeBudgetController(
            planned_allocation=planned,
            daily_budget=24.0,
            baseline_roas=baseline_roas
        )

        # Simulate high performance: ROAS much higher than baseline
        controller.spent[10] = 0.5
        controller.revenue[10] = 3.0  # ROAS = 6.0 (vs baseline 3.0)

        adjustments = controller.check_and_adjust(current_hour=10, current_minute=30)

        # Should trigger boost
        boost_adjustments = [a for a in adjustments if a.type == AdjustmentType.PERFORMANCE_BOOST]
        assert len(boost_adjustments) > 0

        print("[PASS] test_performance_boost")

    def test_simulate_day(self):
        """Test full day simulation."""
        from algorithms.realtime_controller import RealtimeBudgetController
        from utils.data_simulator import CampaignDataSimulator

        simulator = CampaignDataSimulator(seed=42)
        planned = {h: 1.0 for h in range(24)}

        events = simulator.simulate_realtime_events(
            planned_allocation=planned,
            performance_scenario='normal'
        )

        controller = RealtimeBudgetController(
            planned_allocation=planned,
            daily_budget=24.0
        )

        result = controller.simulate_day(events, check_every_n_events=10)

        assert 'summary' in result
        assert 'hourly_performance' in result
        assert 'final_allocation' in result
        assert result['summary']['total_spent'] > 0

        print("[PASS] test_simulate_day")


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test the full advanced pipeline."""
        from automation.optimizer import BudgetOptimizer
        from utils.data_simulator import generate_test_dataset

        # Generate test data
        df = generate_test_dataset(n_campaigns=10, n_days=14, seed=42)
        df.to_excel('data/test_data.xlsx', index=False)

        try:
            optimizer = BudgetOptimizer('data/test_data.xlsx')

            # Test cold start
            cold_start_result = optimizer.optimize_with_cold_start(
                campaign_name='New_Campaign',
                daily_budget=20.0,
                day_of_week=1,
                product_category='electronics',
                save_result=False
            )
            assert 'hourly_allocation' in cold_start_result
            assert cold_start_result['cold_start_strategy'] is not None

            # Test response curves
            response_result = optimizer.optimize_with_response_curves(
                total_budget=100.0,
                save_result=False
            )
            assert 'allocation' in response_result
            assert 'curve_results' in response_result

            # Test bandit
            bandit_result = optimizer.optimize_with_bandit(
                total_budget=100.0,
                method='thompson',
                save_result=False
            )
            assert 'allocation' in bandit_result
            assert bandit_result['method'] == 'thompson_sampling'

            print("[PASS] test_full_pipeline")

        finally:
            # Cleanup
            import os
            if os.path.exists('data/test_data.xlsx'):
                os.remove('data/test_data.xlsx')


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING ADVANCED FEATURES TEST SUITE")
    print("="*70 + "\n")

    test_classes = [
        TestDataSimulator,
        TestColdStart,
        TestResponseCurve,
        TestBandit,
        TestRealtimeController,
        TestIntegration
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    total_passed += 1
                except Exception as e:
                    total_failed += 1
                    failures.append((test_class.__name__, method_name, str(e)))
                    print(f"[FAIL] {method_name}: {e}")

    print("\n" + "="*70)
    print(f"TEST RESULTS: {total_passed} passed, {total_failed} failed")
    print("="*70)

    if failures:
        print("\nFailures:")
        for cls, method, error in failures:
            print(f"  - {cls}.{method}: {error}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
