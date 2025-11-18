"""
Task 1: Hourly Budget Allocation Algorithm

Distributes daily budget across 24 hours using convex optimization.
Balances performance (ROAS) with smooth allocation.
"""

import numpy as np
import cvxpy as cp
import pandas as pd
from typing import Dict, Tuple


class HourlyBudgetAllocator:
    """
    Optimizes hourly budget allocation for a single campaign.

    Uses CVXPY to solve:
        maximize: sum(weight[h] * budget[h]) - smoothness_penalty
        subject to: sum(budget) = daily_budget, budget >= min, budget <= max
    """

    def __init__(
        self,
        smoothness_factor: float = 0.3,
        min_hourly_budget: float = 0.1,
        max_hourly_ratio: float = 0.15
    ):
        """
        Initialize allocator.

        Args:
            smoothness_factor: Weight for smoothness penalty (0-1)
            min_hourly_budget: Minimum budget per hour
            max_hourly_ratio: Maximum hourly budget as ratio of daily budget
        """
        self.smoothness_factor = smoothness_factor
        self.min_hourly_budget = min_hourly_budget
        self.max_hourly_ratio = max_hourly_ratio

    def optimize(
        self,
        daily_budget: float,
        hourly_metrics: pd.DataFrame
    ) -> Dict:
        """
        Optimize hourly budget allocation.

        Args:
            daily_budget: Total budget for the day
            hourly_metrics: DataFrame with columns ['hour', 'roas', ...]

        Returns:
            Dictionary with allocation results
        """
        print(f"\nOptimizing hourly allocation (budget: ${daily_budget})")

        # Extract ROAS values for each hour
        roas_values = hourly_metrics.set_index('hour')['roas'].to_dict()
        roas_array = np.array([roas_values.get(h, 0) for h in range(24)])

        # Normalize ROAS to create weights
        roas_sum = roas_array.sum()
        if roas_sum > 0:
            weights = roas_array / roas_sum
        else:
            weights = np.ones(24) / 24

        # Solve optimization problem
        budgets = cp.Variable(24, nonneg=True)

        # Objective: maximize weighted allocation - smoothness penalty
        revenue_objective = weights @ budgets
        smoothness_penalty = cp.sum_squares(cp.diff(budgets))
        objective = cp.Maximize(revenue_objective - self.smoothness_factor * smoothness_penalty)

        # Constraints
        constraints = [
            cp.sum(budgets) == daily_budget,
            budgets >= self.min_hourly_budget,
            budgets <= daily_budget * self.max_hourly_ratio
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"[WARNING] Solver status: {problem.status}, using fallback")
            return self._fallback_allocation(daily_budget, weights)

        # Extract solution
        allocation = {h: max(0, float(budgets.value[h])) for h in range(24)}

        # Calculate metrics
        total_allocated = sum(allocation.values())
        expected_revenue = sum(allocation[h] * roas_array[h] for h in range(24))
        expected_roas = expected_revenue / total_allocated if total_allocated > 0 else 0
        smoothness = self._calculate_smoothness(list(allocation.values()))

        # Identify peak hours
        peak_hours = self._identify_peak_hours(roas_array, top_n=6)

        result = {
            'daily_budget': daily_budget,
            'hourly_allocation': allocation,
            'total_allocated': round(total_allocated, 2),
            'expected_revenue': round(expected_revenue, 2),
            'expected_roas': round(expected_roas, 2),
            'smoothness_score': round(smoothness, 6),
            'peak_hours': peak_hours,
            'solver_status': problem.status
        }

        print(f"[OK] Allocated: ${result['total_allocated']:.2f}")
        print(f"[OK] Expected Revenue: ${result['expected_revenue']:.2f}")
        print(f"[OK] Expected ROAS: {result['expected_roas']:.2f}x")

        return result

    def optimize_weekly(
        self,
        weekly_budget_limit: float,
        campaign_df: pd.DataFrame,
        min_daily_budget: float,
        max_daily_budget: float
    ) -> Dict:
        """
        Optimize budget allocation for entire week.

        Args:
            weekly_budget_limit: Total budget for the week
            campaign_df: Full campaign data
            min_daily_budget: Minimum budget per day
            max_daily_budget: Maximum budget per day

        Returns:
            Weekly allocation strategy
        """
        print(f"\nOptimizing weekly allocation (budget: ${weekly_budget_limit})")

        # Calculate daily performance scores
        daily_scores = {}
        for dow in range(1, 8):
            dow_data = campaign_df[campaign_df['day_of_week'] == dow]
            if dow_data.empty:
                daily_scores[dow] = 1.0
            else:
                avg_revenue = dow_data['revenue'].mean()
                avg_roas = dow_data['roas'].mean()
                daily_scores[dow] = max(avg_revenue * avg_roas, 0.1)

        # Allocate daily budgets proportional to scores
        total_score = sum(daily_scores.values())
        daily_budgets = {}
        for dow, score in daily_scores.items():
            budget = (score / total_score) * weekly_budget_limit
            budget = max(min_daily_budget, min(max_daily_budget, budget))
            daily_budgets[dow] = budget

        # Adjust to meet weekly limit
        total_allocated = sum(daily_budgets.values())
        if total_allocated > weekly_budget_limit:
            scale = weekly_budget_limit / total_allocated
            daily_budgets = {dow: b * scale for dow, b in daily_budgets.items()}

        # Optimize hourly for each day
        dow_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        weekly_strategy = {}

        for dow in range(1, 8):
            dow_data = campaign_df[campaign_df['day_of_week'] == dow]

            if dow_data.empty:
                continue

            # Get hourly metrics for this day
            from utils.data_loader import get_hourly_metrics
            hourly_metrics = get_hourly_metrics(dow_data)

            # Optimize
            day_result = self.optimize(daily_budgets[dow], hourly_metrics)

            weekly_strategy[dow_names[dow-1]] = {
                'daily_budget': daily_budgets[dow],
                'hourly_allocation': day_result['hourly_allocation'],
                'total': day_result['total_allocated'],
                'expected_revenue': day_result['expected_revenue'],
                'expected_roas': day_result['expected_roas']
            }

        total_weekly = sum(daily_budgets.values())
        total_revenue = sum(d['expected_revenue'] for d in weekly_strategy.values())

        return {
            'weekly_budget_strategy': weekly_strategy,
            'weekly_total': round(total_weekly, 2),
            'total_expected_revenue': round(total_revenue, 2),
            'overall_roas': round(total_revenue / total_weekly, 2) if total_weekly > 0 else 0
        }

    def _fallback_allocation(self, daily_budget: float, weights: np.ndarray) -> Dict:
        """Fallback to proportional allocation if optimization fails."""
        allocation = {h: float(daily_budget * weights[h]) for h in range(24)}
        return {
            'daily_budget': daily_budget,
            'hourly_allocation': allocation,
            'total_allocated': daily_budget,
            'solver_status': 'fallback'
        }

    def _calculate_smoothness(self, budgets: list) -> float:
        """Calculate smoothness score."""
        if len(budgets) < 2:
            return 0.0
        diffs = np.diff(budgets)
        return float(np.var(diffs))

    def _identify_peak_hours(self, roas_array: np.ndarray, top_n: int = 6) -> list:
        """Identify top performing hours."""
        hour_roas = [(h, roas_array[h]) for h in range(24)]
        sorted_hours = sorted(hour_roas, key=lambda x: x[1], reverse=True)
        return sorted([h for h, _ in sorted_hours[:top_n]])
