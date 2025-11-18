"""
Simple metrics calculation utilities.
"""

import numpy as np


def calculate_smoothness(budgets: list) -> float:
    """
    Calculate smoothness score (lower is better).
    Measures variance of adjacent hour differences.
    """
    if len(budgets) < 2:
        return 0.0

    diffs = np.diff(budgets)
    return float(np.var(diffs))


def calculate_roas(revenue: float, spend: float) -> float:
    """Calculate Return on Ad Spend."""
    if spend == 0:
        return 0.0
    return revenue / spend


def calculate_acos(spend: float, revenue: float) -> float:
    """Calculate Advertising Cost of Sale."""
    if revenue == 0:
        return 0.0
    return spend / revenue


def calculate_expected_revenue(budgets: dict, hourly_roas: dict) -> float:
    """Calculate expected revenue based on budget allocation and historical ROAS."""
    total_revenue = 0.0
    for hour, budget in budgets.items():
        roas = hourly_roas.get(hour, 0.0)
        total_revenue += budget * roas
    return total_revenue
