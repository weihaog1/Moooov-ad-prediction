"""
Task 2: Multi-Campaign Budget Optimization

Allocates total budget across multiple campaigns to maximize performance.
"""

import numpy as np
import cvxpy as cp
import pandas as pd
from typing import Dict, List


class MultiCampaignOptimizer:
    """
    Optimizes budget allocation across multiple campaigns.

    Uses linear programming to:
        maximize: sum(allocation[i] * expected_roas[i])
        subject to: sum(allocation) <= total_budget, bounds on each campaign
    """

    def optimize(
        self,
        total_budget: float,
        campaign_data: pd.DataFrame,
        optimization_target: str = "maximize_revenue",
        min_allocation_ratio: float = 0.05,
        max_allocation_ratio: float = 0.40
    ) -> Dict:
        """
        Optimize budget allocation across campaigns.

        Args:
            total_budget: Total available budget
            campaign_data: DataFrame with all campaign performance data
            optimization_target: "maximize_revenue", "minimize_acos", or "maximize_roas"
            min_allocation_ratio: Minimum allocation per campaign (as ratio of total)
            max_allocation_ratio: Maximum allocation per campaign (as ratio of total)

        Returns:
            Optimization results with allocations and metrics
        """
        print(f"\nOptimizing multi-campaign allocation (budget: ${total_budget})")
        print(f"   Target: {optimization_target}")

        # Calculate campaign-level metrics
        campaign_metrics = campaign_data.groupby('campaign_name').agg({
            'revenue': 'sum',
            'spend': 'sum',
            'orders': 'sum',
            'roas': 'mean',
            'acos': 'mean',
        }).reset_index()

        campaigns = campaign_metrics['campaign_name'].tolist()
        n_campaigns = len(campaigns)

        print(f"   Campaigns: {n_campaigns}")

        # Extract metrics
        expected_roas = campaign_metrics['roas'].values
        expected_acos = campaign_metrics['acos'].values

        # Set budget bounds
        min_allocation = total_budget * min_allocation_ratio
        max_allocation = total_budget * max_allocation_ratio

        # Decision variables
        allocations = cp.Variable(n_campaigns, nonneg=True)

        # Objective function
        if optimization_target == "maximize_revenue":
            objective = cp.Maximize(expected_roas @ allocations)
        elif optimization_target == "minimize_acos":
            objective = cp.Minimize(cp.sum(cp.multiply(expected_acos, allocations)) / cp.sum(allocations))
        else:  # maximize_roas
            objective = cp.Maximize(expected_roas @ allocations / cp.sum(allocations))

        # Constraints
        constraints = [
            cp.sum(allocations) <= total_budget,
            allocations >= min_allocation,
            allocations <= max_allocation
        ]

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"[WARNING] Solver status: {problem.status}, using fallback")
            allocation_dict = self._proportional_allocation(
                total_budget, expected_roas, campaigns, max_allocation
            )
        else:
            allocation_dict = {
                campaigns[i]: round(float(allocations.value[i]), 2)
                for i in range(n_campaigns)
            }

        # Calculate outcomes
        total_allocated = sum(allocation_dict.values())

        expected_revenue = sum(
            allocation_dict[campaigns[i]] * expected_roas[i]
            for i in range(n_campaigns)
        )

        weighted_acos = sum(
            allocation_dict[campaigns[i]] * expected_acos[i]
            for i in range(n_campaigns)
        )

        avg_acos = weighted_acos / total_allocated if total_allocated > 0 else 0
        overall_roas = expected_revenue / total_allocated if total_allocated > 0 else 0

        # Categorize campaigns
        rationale = self._categorize_campaigns(campaign_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            allocation_dict, campaign_metrics
        )

        result = {
            'total_budget': total_budget,
            'optimization_target': optimization_target,
            'allocation': allocation_dict,
            'total_allocated': round(total_allocated, 2),
            'allocation_rationale': rationale,
            'expected_outcomes': {
                'total_revenue': round(expected_revenue, 2),
                'overall_roas': round(overall_roas, 2),
                'average_acos': round(avg_acos, 4)
            },
            'adjustment_recommendations': recommendations,
            'solver_status': problem.status if problem.status else 'fallback'
        }

        print(f"[OK] Allocated: ${result['total_allocated']:.2f}")
        print(f"[OK] Expected Revenue: ${result['expected_outcomes']['total_revenue']:.2f}")
        print(f"[OK] Overall ROAS: {result['expected_outcomes']['overall_roas']:.2f}x")

        return result

    def _proportional_allocation(
        self,
        total_budget: float,
        expected_roas: np.ndarray,
        campaigns: List[str],
        max_allocation: float
    ) -> Dict[str, float]:
        """Fallback: allocate proportional to ROAS."""
        weights = expected_roas / (expected_roas.sum() + 1e-6)
        allocations = total_budget * weights
        allocations = np.minimum(allocations, max_allocation)

        # Redistribute if some hit max
        total_alloc = allocations.sum()
        if total_alloc < total_budget:
            remaining = total_budget - total_alloc
            can_increase = max_allocation - allocations
            if can_increase.sum() > 0:
                increase_weights = can_increase / can_increase.sum()
                allocations += remaining * increase_weights
                allocations = np.minimum(allocations, max_allocation)

        return {campaigns[i]: round(float(allocations[i]), 2) for i in range(len(campaigns))}

    def _categorize_campaigns(self, campaign_metrics: pd.DataFrame) -> Dict[str, str]:
        """Categorize campaigns by performance."""
        categories = {}

        roas_q75 = campaign_metrics['roas'].quantile(0.75)
        roas_q25 = campaign_metrics['roas'].quantile(0.25)
        acos_median = campaign_metrics['acos'].median()

        for _, row in campaign_metrics.iterrows():
            campaign = row['campaign_name']
            roas = row['roas']
            acos = row['acos']

            if roas >= roas_q75 and acos <= acos_median:
                categories[campaign] = "high_roas_stable"
            elif roas >= roas_q75:
                categories[campaign] = "high_roas"
            elif roas <= roas_q25 and acos >= acos_median:
                categories[campaign] = "low_efficiency_high_acos"
            elif acos >= acos_median:
                categories[campaign] = "high_acos"
            else:
                categories[campaign] = "moderate_performance"

        return categories

    def _generate_recommendations(
        self,
        allocation: Dict[str, float],
        campaign_metrics: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """Generate budget adjustment recommendations."""
        # High ROAS campaigns
        high_roas = campaign_metrics.nlargest(3, 'roas')['campaign_name'].tolist()

        # High ACoS campaigns
        high_acos = campaign_metrics.nlargest(3, 'acos')['campaign_name'].tolist()

        # Low allocation but high performance
        sorted_alloc = sorted(allocation.items(), key=lambda x: x[1])
        low_allocated = [name for name, _ in sorted_alloc[:3]]
        monitor = [c for c in low_allocated if c in high_roas]

        return {
            'increase_budget': high_roas,
            'decrease_budget': high_acos,
            'monitor_closely': monitor
        }
