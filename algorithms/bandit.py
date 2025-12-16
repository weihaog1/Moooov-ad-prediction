"""
Bandit Algorithms for Exploration & Exploitation

Implements multi-armed bandit algorithms for budget allocation:
- Thompson Sampling: Bayesian approach with Beta posteriors
- UCB (Upper Confidence Bound): Optimism in the face of uncertainty
- Hourly Bandit: E&E for within-campaign hourly allocation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class BanditMethod(Enum):
    """Supported bandit methods."""
    THOMPSON_SAMPLING = "thompson_sampling"
    UCB = "ucb"


@dataclass
class BanditState:
    """State for a single arm (campaign/hour)."""
    alpha: float = 1.0  # Beta prior: successes
    beta: float = 1.0   # Beta prior: failures
    total_spent: float = 0.0
    total_revenue: float = 0.0
    n_observations: int = 0


class ThompsonSamplingAllocator:
    """
    Thompson Sampling for budget allocation E&E.

    Maintains Beta distribution posteriors for each campaign,
    samples from distributions to make allocation decisions.
    """

    def __init__(
        self,
        campaigns: List[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        target_roas: float = 1.0
    ):
        """
        Initialize with Beta priors for each campaign.

        Args:
            campaigns: List of campaign names
            prior_alpha: Prior successes (higher = more optimistic)
            prior_beta: Prior failures (higher = more pessimistic)
            target_roas: ROAS threshold for success (default: break-even)
        """
        self.campaigns = campaigns
        self.target_roas = target_roas

        # Beta distribution parameters for each campaign
        self.states: Dict[str, BanditState] = {
            c: BanditState(alpha=prior_alpha, beta=prior_beta)
            for c in campaigns
        }

        # History for analysis
        self.history: List[Dict] = []

    def sample_allocation(
        self,
        total_budget: float,
        min_per_campaign: float = 0,
        exploration_ratio: float = 0.0
    ) -> Dict[str, float]:
        """
        Sample from posteriors to determine allocation.

        Args:
            total_budget: Total budget to allocate
            min_per_campaign: Minimum allocation per campaign
            exploration_ratio: Additional exploration budget ratio

        Returns:
            Dict of campaign -> allocation
        """
        # Sample from Beta distributions
        samples = {}
        for campaign, state in self.states.items():
            theta = np.random.beta(state.alpha, state.beta)
            samples[campaign] = theta

        # Split budget: exploitation vs exploration
        exploit_budget = total_budget * (1 - exploration_ratio)
        explore_budget = total_budget * exploration_ratio

        # Exploitation: allocate proportional to samples
        total_sample = sum(samples.values())
        allocations = {}

        for campaign, theta in samples.items():
            # Exploitation allocation
            exploit_alloc = (theta / total_sample) * exploit_budget if total_sample > 0 else exploit_budget / len(self.campaigns)

            # Exploration allocation (uniform)
            explore_alloc = explore_budget / len(self.campaigns)

            alloc = exploit_alloc + explore_alloc
            alloc = max(alloc, min_per_campaign)
            allocations[campaign] = round(alloc, 2)

        # Record for analysis
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'samples': samples.copy(),
            'allocations': allocations.copy(),
            'exploration_ratio': exploration_ratio
        })

        return allocations

    def update(
        self,
        campaign: str,
        spent: float,
        revenue: float
    ):
        """
        Update posterior based on observed results.

        Args:
            campaign: Campaign name
            spent: Amount spent
            revenue: Revenue generated
        """
        if campaign not in self.states:
            return

        state = self.states[campaign]
        state.total_spent += spent
        state.total_revenue += revenue
        state.n_observations += 1

        # Convert to success/failure
        actual_roas = revenue / spent if spent > 0 else 0

        if actual_roas >= self.target_roas:
            state.alpha += 1
        else:
            state.beta += 1

    def get_expected_values(self) -> Dict[str, float]:
        """
        Get expected success probability for each campaign.

        Returns mean of Beta distribution.
        """
        return {
            c: state.alpha / (state.alpha + state.beta)
            for c, state in self.states.items()
        }

    def get_uncertainty(self) -> Dict[str, float]:
        """
        Get uncertainty (variance) for each campaign.

        Higher uncertainty = needs more exploration.
        """
        uncertainties = {}
        for campaign, state in self.states.items():
            a, b = state.alpha, state.beta
            variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
            uncertainties[campaign] = variance
        return uncertainties

    def get_exploitation_ranking(self) -> List[Tuple[str, float]]:
        """
        Get campaigns ranked by expected performance.
        """
        expected = self.get_expected_values()
        return sorted(expected.items(), key=lambda x: x[1], reverse=True)

    def get_exploration_ranking(self) -> List[Tuple[str, float]]:
        """
        Get campaigns ranked by uncertainty (need for exploration).
        """
        uncertainty = self.get_uncertainty()
        return sorted(uncertainty.items(), key=lambda x: x[1], reverse=True)

    def get_statistics(self) -> Dict:
        """Get summary statistics."""
        return {
            'campaigns': {
                c: {
                    'alpha': state.alpha,
                    'beta': state.beta,
                    'expected_value': state.alpha / (state.alpha + state.beta),
                    'total_spent': state.total_spent,
                    'total_revenue': state.total_revenue,
                    'n_observations': state.n_observations,
                    'observed_roas': state.total_revenue / state.total_spent if state.total_spent > 0 else 0
                }
                for c, state in self.states.items()
            },
            'total_observations': sum(s.n_observations for s in self.states.values()),
            'history_length': len(self.history)
        }


class UCBAllocator:
    """
    UCB (Upper Confidence Bound) for budget allocation E&E.

    UCB score = estimated_value + exploration_bonus

    Exploration bonus decreases with more observations.
    """

    def __init__(
        self,
        campaigns: List[str],
        exploration_weight: float = 2.0
    ):
        """
        Initialize UCB allocator.

        Args:
            campaigns: List of campaign names
            exploration_weight: Controls exploration (typically 1-3)
        """
        self.campaigns = campaigns
        self.c = exploration_weight

        self.states: Dict[str, BanditState] = {
            c: BanditState() for c in campaigns
        }

        self.total_rounds = 0
        self.history: List[Dict] = []

    def compute_ucb(self, campaign: str) -> float:
        """
        Compute UCB score for a campaign.

        UCB = mean_return + c × sqrt(log(N) / n)

        where:
        - mean_return = total_revenue / total_spent
        - N = total rounds
        - n = observations for this campaign
        """
        state = self.states[campaign]

        if state.n_observations == 0:
            return float('inf')  # Unexplored = highest priority

        # Exploitation term: mean return
        mean_return = state.total_revenue / state.total_spent if state.total_spent > 0 else 0

        # Exploration term: uncertainty bonus
        exploration_bonus = self.c * np.sqrt(
            np.log(self.total_rounds + 1) / (state.n_observations + 1)
        )

        return mean_return + exploration_bonus

    def select_allocation(
        self,
        total_budget: float,
        min_per_campaign: float = 0
    ) -> Dict[str, float]:
        """
        Allocate budget based on UCB scores.

        Args:
            total_budget: Total budget to allocate
            min_per_campaign: Minimum per campaign

        Returns:
            Dict of campaign -> allocation
        """
        ucb_scores = {c: self.compute_ucb(c) for c in self.campaigns}

        # Handle unexplored campaigns (infinite UCB)
        unexplored = [c for c, s in ucb_scores.items() if s == float('inf')]

        if unexplored:
            # Allocate exploration budget to unexplored
            explore_ratio = min(0.3, len(unexplored) / len(self.campaigns))
            explore_budget = total_budget * explore_ratio
            exploit_budget = total_budget * (1 - explore_ratio)

            allocations = {}

            # Unexplored get equal exploration budget
            for c in unexplored:
                allocations[c] = explore_budget / len(unexplored)

            # Explored get proportional exploitation budget
            explored = [c for c in self.campaigns if c not in unexplored]
            if explored:
                explored_scores = {c: ucb_scores[c] for c in explored}
                total_score = sum(explored_scores.values())
                for c in explored:
                    allocations[c] = (explored_scores[c] / total_score) * exploit_budget
        else:
            # All explored: allocate proportional to UCB
            total_score = sum(ucb_scores.values())
            allocations = {
                c: (s / total_score) * total_budget
                for c, s in ucb_scores.items()
            }

        # Enforce minimums
        for c in allocations:
            allocations[c] = max(allocations[c], min_per_campaign)
            allocations[c] = round(allocations[c], 2)

        self.total_rounds += 1

        # Record history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'ucb_scores': ucb_scores.copy(),
            'allocations': allocations.copy(),
            'round': self.total_rounds
        })

        return allocations

    def update(
        self,
        campaign: str,
        spent: float,
        revenue: float
    ):
        """Update campaign statistics with new observation."""
        if campaign not in self.states:
            return

        state = self.states[campaign]
        state.total_spent += spent
        state.total_revenue += revenue
        state.n_observations += 1

    def get_statistics(self) -> Dict:
        """Get summary statistics."""
        return {
            'campaigns': {
                c: {
                    'ucb_score': self.compute_ucb(c),
                    'total_spent': state.total_spent,
                    'total_revenue': state.total_revenue,
                    'n_observations': state.n_observations,
                    'observed_roas': state.total_revenue / state.total_spent if state.total_spent > 0 else 0
                }
                for c, state in self.states.items()
            },
            'total_rounds': self.total_rounds,
            'exploration_weight': self.c
        }


class HourlyBandit:
    """
    Bandit for hourly allocation within a single campaign.

    Uses Thompson Sampling to balance exploiting known good hours
    with exploring uncertain hours.
    """

    def __init__(
        self,
        target_roas: float = 1.0
    ):
        """
        Initialize hourly bandit.

        Args:
            target_roas: ROAS threshold for success
        """
        self.target_roas = target_roas

        # Beta posteriors for each hour (0-23)
        self.states: Dict[int, BanditState] = {
            h: BanditState() for h in range(24)
        }

        self.history: List[Dict] = []

    def sample_hourly_allocation(
        self,
        daily_budget: float,
        min_per_hour: float = 0.1,
        exploration_ratio: float = 0.1
    ) -> Dict[int, float]:
        """
        Sample allocation across 24 hours.

        Args:
            daily_budget: Total daily budget
            min_per_hour: Minimum budget per hour
            exploration_ratio: Ratio of budget for exploration

        Returns:
            Dict of hour -> allocation
        """
        # Sample from Beta distributions
        samples = {}
        for hour, state in self.states.items():
            samples[hour] = np.random.beta(state.alpha, state.beta)

        # Allocate
        exploit_budget = daily_budget * (1 - exploration_ratio)
        explore_budget = daily_budget * exploration_ratio

        total_sample = sum(samples.values())
        allocations = {}

        for hour, theta in samples.items():
            exploit_alloc = (theta / total_sample) * exploit_budget if total_sample > 0 else exploit_budget / 24
            explore_alloc = explore_budget / 24

            alloc = exploit_alloc + explore_alloc
            alloc = max(alloc, min_per_hour)
            allocations[hour] = round(alloc, 2)

        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'samples': samples.copy(),
            'allocations': allocations.copy()
        })

        return allocations

    def update_hour(
        self,
        hour: int,
        spent: float,
        revenue: float
    ):
        """Update posterior for a specific hour."""
        if hour not in self.states:
            return

        state = self.states[hour]
        state.total_spent += spent
        state.total_revenue += revenue
        state.n_observations += 1

        roas = revenue / spent if spent > 0 else 0

        if roas >= self.target_roas:
            state.alpha += 1
        else:
            state.beta += 1

    def get_expected_values(self) -> Dict[int, float]:
        """Get expected value for each hour."""
        return {
            h: state.alpha / (state.alpha + state.beta)
            for h, state in self.states.items()
        }

    def get_peak_hours(self, top_n: int = 6) -> List[int]:
        """Get top performing hours."""
        expected = self.get_expected_values()
        sorted_hours = sorted(expected.items(), key=lambda x: x[1], reverse=True)
        return [h for h, _ in sorted_hours[:top_n]]


class EEBudgetAllocator:
    """
    Combined E&E Budget Allocator.

    Supports both campaign-level and hourly E&E with configurable methods.
    """

    def __init__(
        self,
        campaigns: List[str],
        method: BanditMethod = BanditMethod.THOMPSON_SAMPLING,
        exploration_ratio: float = 0.15,
        target_roas: float = 1.0
    ):
        """
        Initialize E&E allocator.

        Args:
            campaigns: List of campaign names
            method: Bandit method to use
            exploration_ratio: Ratio of budget for exploration
            target_roas: ROAS threshold for success
        """
        self.method = method
        self.exploration_ratio = exploration_ratio

        if method == BanditMethod.THOMPSON_SAMPLING:
            self.campaign_bandit = ThompsonSamplingAllocator(
                campaigns, target_roas=target_roas
            )
        else:
            self.campaign_bandit = UCBAllocator(campaigns)

        # Hourly bandits for each campaign
        self.hourly_bandits: Dict[str, HourlyBandit] = {
            c: HourlyBandit(target_roas=target_roas)
            for c in campaigns
        }

    def allocate_budget(
        self,
        total_budget: float,
        min_per_campaign: float = None
    ) -> Dict:
        """
        Allocate budget across campaigns.

        Args:
            total_budget: Total budget
            min_per_campaign: Minimum per campaign

        Returns:
            {
                'campaign_allocation': Dict[str, float],
                'exploration_ratio': float,
                'method': str
            }
        """
        if min_per_campaign is None:
            min_per_campaign = total_budget * 0.05

        if self.method == BanditMethod.THOMPSON_SAMPLING:
            allocation = self.campaign_bandit.sample_allocation(
                total_budget, min_per_campaign, self.exploration_ratio
            )
        else:
            allocation = self.campaign_bandit.select_allocation(
                total_budget, min_per_campaign
            )

        return {
            'campaign_allocation': allocation,
            'exploration_ratio': self.exploration_ratio,
            'method': self.method.value
        }

    def allocate_hourly(
        self,
        campaign_name: str,
        daily_budget: float,
        min_per_hour: float = 0.1
    ) -> Dict[int, float]:
        """
        Allocate budget across hours for a campaign.

        Args:
            campaign_name: Campaign to allocate for
            daily_budget: Daily budget
            min_per_hour: Minimum per hour

        Returns:
            Dict of hour -> allocation
        """
        if campaign_name not in self.hourly_bandits:
            # Unknown campaign: uniform
            return {h: daily_budget / 24 for h in range(24)}

        return self.hourly_bandits[campaign_name].sample_hourly_allocation(
            daily_budget, min_per_hour, self.exploration_ratio
        )

    def update_campaign(
        self,
        campaign: str,
        spent: float,
        revenue: float
    ):
        """Update campaign-level bandit."""
        self.campaign_bandit.update(campaign, spent, revenue)

    def update_hourly(
        self,
        campaign: str,
        hour: int,
        spent: float,
        revenue: float
    ):
        """Update hourly bandit for a campaign."""
        if campaign in self.hourly_bandits:
            self.hourly_bandits[campaign].update_hour(hour, spent, revenue)

    def get_statistics(self) -> Dict:
        """Get combined statistics."""
        return {
            'method': self.method.value,
            'exploration_ratio': self.exploration_ratio,
            'campaign_stats': self.campaign_bandit.get_statistics(),
            'hourly_stats': {
                c: bandit.get_expected_values()
                for c, bandit in self.hourly_bandits.items()
            }
        }
