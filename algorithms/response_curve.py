"""
Response Curve Modeling for Diminishing Returns

Models the non-linear relationship between budget and revenue
to handle diminishing returns in advertising.

Supported curve types:
- Logarithmic: R = a × log(1 + b × B)
- Square Root: R = a × √B + c
- Hill Function: R = a × B^n / (k^n + B^n)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class CurveType(Enum):
    """Supported response curve types."""
    LOG = "logarithmic"
    SQRT = "square_root"
    HILL = "hill"
    LINEAR = "linear"


@dataclass
class CurveFitResult:
    """Result of curve fitting."""
    curve_type: CurveType
    params: Dict[str, float]
    r_squared: float
    rmse: float
    is_reliable: bool
    saturation_point: Optional[float]


class ResponseCurveModeler:
    """
    Models budget-revenue relationship with diminishing returns.

    Fits multiple curve types and selects the best fit.
    """

    # Curve function definitions
    @staticmethod
    def _log_func(B, a, b):
        """Logarithmic: R = a × log(1 + b × B)"""
        return a * np.log(1 + b * np.maximum(B, 0))

    @staticmethod
    def _sqrt_func(B, a, c):
        """Square root: R = a × √B + c"""
        return a * np.sqrt(np.maximum(B, 0)) + c

    @staticmethod
    def _hill_func(B, a, n, k):
        """Hill function: R = a × B^n / (k^n + B^n)"""
        B = np.maximum(B, 0)
        return a * (B ** n) / (k ** n + B ** n + 1e-10)

    @staticmethod
    def _linear_func(B, a, c):
        """Linear: R = a × B + c"""
        return a * B + c

    # Derivative functions for marginal return calculation
    @staticmethod
    def _log_derivative(B, a, b):
        """Derivative of log function."""
        return a * b / (1 + b * B + 1e-10)

    @staticmethod
    def _sqrt_derivative(B, a, c):
        """Derivative of sqrt function."""
        return a / (2 * np.sqrt(B + 1e-10))

    @staticmethod
    def _hill_derivative(B, a, n, k):
        """Derivative of Hill function."""
        B = np.maximum(B, 1e-10)
        numerator = a * n * k ** n * B ** (n - 1)
        denominator = (k ** n + B ** n) ** 2 + 1e-10
        return numerator / denominator

    @staticmethod
    def _linear_derivative(B, a, c):
        """Derivative of linear function (constant)."""
        return a

    def __init__(
        self,
        min_data_points: int = 10,
        r_squared_threshold: float = 0.3
    ):
        """
        Initialize modeler.

        Args:
            min_data_points: Minimum data points required for fitting
            r_squared_threshold: Minimum R² to consider fit reliable
        """
        self.min_data_points = min_data_points
        self.r_squared_threshold = r_squared_threshold

        # Cache fitted curves
        self.fitted_curves: Dict[str, CurveFitResult] = {}

        # Map curve types to functions
        self._curve_functions = {
            CurveType.LOG: (self._log_func, ['a', 'b']),
            CurveType.SQRT: (self._sqrt_func, ['a', 'c']),
            CurveType.HILL: (self._hill_func, ['a', 'n', 'k']),
            CurveType.LINEAR: (self._linear_func, ['a', 'c'])
        }

        self._derivative_functions = {
            CurveType.LOG: self._log_derivative,
            CurveType.SQRT: self._sqrt_derivative,
            CurveType.HILL: self._hill_derivative,
            CurveType.LINEAR: self._linear_derivative
        }

    def fit(
        self,
        campaign_data: pd.DataFrame,
        campaign_name: str
    ) -> CurveFitResult:
        """
        Fit response curve for a campaign.

        Args:
            campaign_data: Campaign performance data
            campaign_name: Target campaign

        Returns:
            CurveFitResult with best fit
        """
        # Filter for campaign
        camp_data = campaign_data[campaign_data['campaign_name'] == campaign_name]

        if len(camp_data) < self.min_data_points:
            return self._create_default_result(campaign_name, "Insufficient data")

        # Prepare budget-revenue data
        budgets, revenues = self._prepare_data(camp_data)

        if len(budgets) < self.min_data_points:
            return self._create_default_result(campaign_name, "Insufficient unique budget levels")

        # Fit all curves
        fit_results = {}
        for curve_type in CurveType:
            try:
                result = self._fit_curve(budgets, revenues, curve_type)
                if result is not None:
                    fit_results[curve_type] = result
            except Exception as e:
                continue

        if not fit_results:
            return self._create_default_result(campaign_name, "All curve fits failed")

        # Select best curve
        best_type, best_result = self._select_best_curve(fit_results)

        # Calculate saturation point
        saturation = self._find_saturation_point(best_type, best_result['params'])

        # Create final result
        is_reliable = (
            best_result['r_squared'] >= self.r_squared_threshold and
            best_type != CurveType.LINEAR
        )

        final_result = CurveFitResult(
            curve_type=best_type,
            params=best_result['params'],
            r_squared=best_result['r_squared'],
            rmse=best_result['rmse'],
            is_reliable=is_reliable,
            saturation_point=saturation
        )

        # Cache result
        self.fitted_curves[campaign_name] = final_result

        return final_result

    def _prepare_data(
        self,
        campaign_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare budget-revenue data by aggregating into budget bins.
        """
        df = campaign_data.copy()

        # Aggregate by spend level
        df['spend_bin'] = pd.cut(df['spend'], bins=20, labels=False)

        aggregated = df.groupby('spend_bin').agg({
            'spend': 'mean',
            'revenue': 'mean'
        }).dropna()

        if len(aggregated) < 3:
            # Not enough bins, use raw data
            return df['spend'].values, df['revenue'].values

        return aggregated['spend'].values, aggregated['revenue'].values

    def _fit_curve(
        self,
        budgets: np.ndarray,
        revenues: np.ndarray,
        curve_type: CurveType
    ) -> Optional[Dict]:
        """
        Fit a specific curve type to the data.
        """
        func, param_names = self._curve_functions[curve_type]

        # Initial parameter guesses
        max_rev = revenues.max()
        max_budget = budgets.max()

        if curve_type == CurveType.LOG:
            p0 = [max_rev, 0.1]
            bounds = ([0, 0.001], [max_rev * 10, 10])
        elif curve_type == CurveType.SQRT:
            p0 = [max_rev / np.sqrt(max_budget + 1), 0]
            bounds = ([0, -max_rev], [max_rev * 10, max_rev])
        elif curve_type == CurveType.HILL:
            p0 = [max_rev * 2, 1.5, max_budget / 2]
            bounds = ([0, 0.1, 1], [max_rev * 10, 5, max_budget * 2])
        else:  # LINEAR
            p0 = [max_rev / max_budget, 0]
            bounds = ([0, -max_rev], [max_rev, max_rev])

        try:
            popt, _ = curve_fit(
                func, budgets, revenues,
                p0=p0, bounds=bounds,
                maxfev=5000
            )
        except Exception:
            return None

        # Calculate fit metrics
        predictions = func(budgets, *popt)
        ss_res = np.sum((revenues - predictions) ** 2)
        ss_tot = np.sum((revenues - revenues.mean()) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-10))
        rmse = np.sqrt(np.mean((revenues - predictions) ** 2))

        return {
            'params': dict(zip(param_names, popt)),
            'r_squared': float(r_squared),
            'rmse': float(rmse)
        }

    def _select_best_curve(
        self,
        fit_results: Dict[CurveType, Dict]
    ) -> Tuple[CurveType, Dict]:
        """
        Select best curve based on R² and preference for non-linear.

        Prefers non-linear curves if they have reasonable fit.
        """
        # Sort by R²
        ranked = sorted(
            fit_results.items(),
            key=lambda x: x[1]['r_squared'],
            reverse=True
        )

        best_type, best_result = ranked[0]

        # If linear is best but non-linear is close, prefer non-linear
        if best_type == CurveType.LINEAR and len(ranked) > 1:
            for curve_type, result in ranked[1:]:
                if result['r_squared'] > best_result['r_squared'] - 0.1:
                    return curve_type, result

        return best_type, best_result

    def _find_saturation_point(
        self,
        curve_type: CurveType,
        params: Dict[str, float],
        marginal_threshold: float = 0.5
    ) -> Optional[float]:
        """
        Find budget level where marginal return falls below threshold.

        Args:
            curve_type: Type of fitted curve
            params: Curve parameters
            marginal_threshold: Stop when MR < this value

        Returns:
            Budget level at saturation, or None if not found
        """
        derivative_func = self._derivative_functions[curve_type]

        # Search for saturation point
        for budget in np.linspace(1, 1000, 1000):
            try:
                mr = derivative_func(budget, *params.values())
                if mr < marginal_threshold:
                    return float(budget)
            except Exception:
                continue

        return None

    def _create_default_result(
        self,
        campaign_name: str,
        reason: str
    ) -> CurveFitResult:
        """Create default linear result when fitting fails."""
        result = CurveFitResult(
            curve_type=CurveType.LINEAR,
            params={'a': 1.0, 'c': 0.0},
            r_squared=0.0,
            rmse=float('inf'),
            is_reliable=False,
            saturation_point=None
        )
        self.fitted_curves[campaign_name] = result
        return result

    def predict_revenue(
        self,
        campaign_name: str,
        budget: float
    ) -> float:
        """
        Predict revenue for a given budget level.

        Args:
            campaign_name: Campaign to predict for
            budget: Budget amount

        Returns:
            Predicted revenue
        """
        if campaign_name not in self.fitted_curves:
            # Default: assume ROAS of 3
            return budget * 3.0

        result = self.fitted_curves[campaign_name]
        func, _ = self._curve_functions[result.curve_type]

        return float(func(budget, *result.params.values()))

    def get_marginal_return(
        self,
        campaign_name: str,
        budget: float
    ) -> float:
        """
        Calculate marginal return (derivative) at a budget level.

        Args:
            campaign_name: Campaign name
            budget: Budget level

        Returns:
            Marginal return (additional revenue per additional dollar)
        """
        if campaign_name not in self.fitted_curves:
            return 3.0  # Default ROAS

        result = self.fitted_curves[campaign_name]
        derivative_func = self._derivative_functions[result.curve_type]

        return float(derivative_func(budget, *result.params.values()))

    def should_use_realtime_fallback(self, campaign_name: str) -> bool:
        """
        Determine if realtime adjustment should be used instead.

        Returns True if curve fit is unreliable.
        """
        if campaign_name not in self.fitted_curves:
            return True

        result = self.fitted_curves[campaign_name]
        return not result.is_reliable


class NonlinearBudgetOptimizer:
    """
    Optimizes budget allocation considering diminishing returns.

    Uses marginal equilibrium principle: at optimum, all campaigns
    have equal marginal returns.
    """

    def __init__(
        self,
        response_modeler: ResponseCurveModeler,
        max_iterations: int = 100,
        convergence_threshold: float = 0.01
    ):
        """
        Initialize optimizer.

        Args:
            response_modeler: Fitted response curve modeler
            max_iterations: Maximum optimization iterations
            convergence_threshold: Stop when MR difference < this
        """
        self.response_modeler = response_modeler
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def optimize(
        self,
        total_budget: float,
        campaigns: List[str],
        min_per_campaign: Optional[float] = None,
        max_per_campaign: Optional[float] = None
    ) -> Dict:
        """
        Allocate budget across campaigns using marginal equilibrium.

        Args:
            total_budget: Total budget to allocate
            campaigns: List of campaign names
            min_per_campaign: Minimum budget per campaign
            max_per_campaign: Maximum budget per campaign

        Returns:
            {
                'allocation': Dict[str, float],
                'expected_revenue': Dict[str, float],
                'marginal_returns': Dict[str, float],
                'iterations': int,
                'converged': bool
            }
        """
        n = len(campaigns)

        if min_per_campaign is None:
            min_per_campaign = total_budget * 0.05
        if max_per_campaign is None:
            max_per_campaign = total_budget * 0.40

        bounds = {c: (min_per_campaign, max_per_campaign) for c in campaigns}

        # Check if we have reliable curves
        reliable_campaigns = [
            c for c in campaigns
            if not self.response_modeler.should_use_realtime_fallback(c)
        ]

        if len(reliable_campaigns) < len(campaigns) / 2:
            # Fall back to proportional allocation
            return self._proportional_allocation(total_budget, campaigns, bounds)

        # Use marginal equilibrium
        allocation, iterations, converged = self._marginal_equilibrium(
            total_budget, campaigns, bounds
        )

        # Calculate expected outcomes
        expected_revenue = {
            c: self.response_modeler.predict_revenue(c, allocation[c])
            for c in campaigns
        }

        marginal_returns = {
            c: self.response_modeler.get_marginal_return(c, allocation[c])
            for c in campaigns
        }

        return {
            'allocation': allocation,
            'expected_revenue': expected_revenue,
            'total_expected_revenue': sum(expected_revenue.values()),
            'marginal_returns': marginal_returns,
            'iterations': iterations,
            'converged': converged,
            'method': 'marginal_equilibrium'
        }

    def _marginal_equilibrium(
        self,
        total_budget: float,
        campaigns: List[str],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Tuple[Dict[str, float], int, bool]:
        """
        Iterative marginal equilibrium allocation.

        Transfers budget from low MR campaigns to high MR campaigns
        until equilibrium is reached.
        """
        # Initialize with equal allocation
        allocation = {c: total_budget / len(campaigns) for c in campaigns}

        # Ensure bounds are respected
        for c in campaigns:
            allocation[c] = np.clip(allocation[c], bounds[c][0], bounds[c][1])

        converged = False

        for iteration in range(self.max_iterations):
            # Compute marginal returns
            marginals = {
                c: self.response_modeler.get_marginal_return(c, allocation[c])
                for c in campaigns
            }

            # Find highest and lowest MR campaigns
            max_mr_campaign = max(marginals, key=marginals.get)
            min_mr_campaign = min(marginals, key=marginals.get)

            mr_diff = marginals[max_mr_campaign] - marginals[min_mr_campaign]

            # Check convergence
            if mr_diff < self.convergence_threshold:
                converged = True
                break

            # Transfer budget
            transfer_amount = min(
                allocation[min_mr_campaign] - bounds[min_mr_campaign][0],
                bounds[max_mr_campaign][1] - allocation[max_mr_campaign],
                total_budget * 0.05  # Max 5% transfer per iteration
            )

            if transfer_amount <= 0:
                break

            allocation[max_mr_campaign] += transfer_amount
            allocation[min_mr_campaign] -= transfer_amount

        return allocation, iteration + 1, converged

    def _proportional_allocation(
        self,
        total_budget: float,
        campaigns: List[str],
        bounds: Dict[str, Tuple[float, float]]
    ) -> Dict:
        """Fallback: allocate proportionally to predicted revenue."""
        # Use predicted revenue at mid-budget as weight
        mid_budget = total_budget / len(campaigns)

        weights = {
            c: self.response_modeler.predict_revenue(c, mid_budget)
            for c in campaigns
        }

        total_weight = sum(weights.values())

        allocation = {}
        for c in campaigns:
            alloc = (weights[c] / total_weight) * total_budget
            alloc = np.clip(alloc, bounds[c][0], bounds[c][1])
            allocation[c] = round(alloc, 2)

        expected_revenue = {
            c: self.response_modeler.predict_revenue(c, allocation[c])
            for c in campaigns
        }

        return {
            'allocation': allocation,
            'expected_revenue': expected_revenue,
            'total_expected_revenue': sum(expected_revenue.values()),
            'marginal_returns': {},
            'iterations': 0,
            'converged': False,
            'method': 'proportional_fallback'
        }
