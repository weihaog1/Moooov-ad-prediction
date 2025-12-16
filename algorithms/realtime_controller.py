"""
Real-Time Budget Controller

Provides dynamic budget adjustment during campaign execution:
- Budget pacing (smooth consumption)
- Performance-based reallocation
- Daily budget flexibility (±20%)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class AdjustmentType(Enum):
    """Types of budget adjustments."""
    PACING = "pacing"
    PERFORMANCE_BOOST = "performance_boost"
    PERFORMANCE_REDUCE = "performance_reduce"
    REALLOCATION = "reallocation"


@dataclass
class SpendEvent:
    """A single spending event."""
    hour: int
    minute: int
    spend: float
    revenue: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Adjustment:
    """A budget adjustment action."""
    type: AdjustmentType
    hour: int
    action: str
    reason: str
    amount: float
    details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class RealtimeBudgetController:
    """
    Real-time budget monitoring and adjustment engine.

    Features:
    - Budget pacing: smooth consumption throughout hours
    - Performance monitoring: track ROAS vs baseline
    - Dynamic reallocation: shift budget between hours
    - Daily budget flexibility: ±20% adjustment
    """

    # Adjustment thresholds
    PACING_THRESHOLD = 0.8        # Trigger pacing if >80% spent before time
    ROAS_HIGH_THRESHOLD = 1.3     # ROAS 30% above baseline = boost
    ROAS_LOW_THRESHOLD = 0.7      # ROAS 30% below baseline = reduce
    BUDGET_FLEX_RATIO = 0.20      # ±20% daily budget flexibility

    def __init__(
        self,
        planned_allocation: Dict[int, float],
        daily_budget: float,
        baseline_roas: Optional[Dict[int, float]] = None,
        check_interval_minutes: int = 15
    ):
        """
        Initialize controller.

        Args:
            planned_allocation: Planned hourly budget {hour: budget}
            daily_budget: Base daily budget
            baseline_roas: Expected ROAS per hour {hour: roas}
            check_interval_minutes: How often to check for adjustments
        """
        self.planned = planned_allocation.copy()
        self.current_allocation = planned_allocation.copy()

        self.base_daily_budget = daily_budget
        self.current_daily_budget = daily_budget
        self.max_daily_budget = daily_budget * (1 + self.BUDGET_FLEX_RATIO)
        self.min_daily_budget = daily_budget * (1 - self.BUDGET_FLEX_RATIO)

        # Default baseline ROAS if not provided
        if baseline_roas is None:
            baseline_roas = {h: 3.0 for h in range(24)}
        self.baseline_roas = baseline_roas

        self.check_interval = check_interval_minutes

        # Real-time tracking
        self.spent: Dict[int, float] = defaultdict(float)
        self.revenue: Dict[int, float] = defaultdict(float)
        self.events: List[SpendEvent] = []
        self.adjustments: List[Adjustment] = []

    def record_event(
        self,
        hour: int,
        minute: int,
        spend: float,
        revenue: float
    ):
        """
        Record a spending event.

        Args:
            hour: Hour of event (0-23)
            minute: Minute of event (0-59)
            spend: Amount spent
            revenue: Revenue generated
        """
        event = SpendEvent(
            hour=hour,
            minute=minute,
            spend=spend,
            revenue=revenue
        )
        self.events.append(event)
        self.spent[hour] += spend
        self.revenue[hour] += revenue

    def check_and_adjust(
        self,
        current_hour: int,
        current_minute: int
    ) -> List[Adjustment]:
        """
        Check if adjustments are needed and compute adjustment plan.

        Args:
            current_hour: Current hour (0-23)
            current_minute: Current minute (0-59)

        Returns:
            List of adjustments to apply
        """
        adjustments = []

        # Check pacing
        pacing_adj = self._check_pacing(current_hour, current_minute)
        if pacing_adj:
            adjustments.append(pacing_adj)

        # Check performance
        perf_adj = self._check_performance(current_hour)
        if perf_adj:
            adjustments.append(perf_adj)

        # Apply adjustments
        for adj in adjustments:
            self._apply_adjustment(adj)

        return adjustments

    def _check_pacing(
        self,
        hour: int,
        minute: int
    ) -> Optional[Adjustment]:
        """
        Check if pacing adjustment is needed.

        Triggers when budget consumption is too fast relative to time.
        """
        if hour not in self.current_allocation:
            return None

        planned = self.current_allocation[hour]
        if planned <= 0:
            return None

        time_ratio = minute / 60
        budget_ratio = self.spent[hour] / planned

        # Too fast: >80% spent but <50% time elapsed
        if budget_ratio > self.PACING_THRESHOLD and time_ratio < 0.5:
            return Adjustment(
                type=AdjustmentType.PACING,
                hour=hour,
                action='slow_down',
                reason=f'Budget {budget_ratio:.0%} spent but only {time_ratio:.0%} time elapsed',
                amount=0,
                details={
                    'budget_ratio': budget_ratio,
                    'time_ratio': time_ratio,
                    'recommendation': 'Reduce bid or pause until next interval'
                }
            )

        return None

    def _check_performance(
        self,
        hour: int
    ) -> Optional[Adjustment]:
        """
        Check if performance-based adjustment is needed.

        Compares current ROAS to baseline and triggers boost/reduce.
        """
        if self.spent[hour] <= 0:
            return None

        current_roas = self.revenue[hour] / self.spent[hour]
        baseline = self.baseline_roas.get(hour, 3.0)

        if baseline <= 0:
            return None

        ratio = current_roas / baseline

        if ratio > self.ROAS_HIGH_THRESHOLD:
            return self._create_boost_adjustment(hour, current_roas, baseline)
        elif ratio < self.ROAS_LOW_THRESHOLD:
            return self._create_reduce_adjustment(hour, current_roas, baseline)

        return None

    def _create_boost_adjustment(
        self,
        hour: int,
        current_roas: float,
        baseline_roas: float
    ) -> Adjustment:
        """
        Create adjustment for high-performing hour.

        Actions:
        1. Borrow from underperforming hours
        2. Increase daily budget if possible
        """
        remaining = max(0, self.current_allocation[hour] - self.spent[hour])

        # Find donor hours (underperforming)
        donor_hours = self._find_donor_hours(exclude=hour)
        borrowable = sum(
            max(0, self.current_allocation[h] - self.spent[h]) * 0.5
            for h in donor_hours
        )

        # Check daily budget increase possibility
        daily_increase = 0
        if self.current_daily_budget < self.max_daily_budget:
            available = self.max_daily_budget - self.current_daily_budget
            daily_increase = min(available, self.base_daily_budget * 0.10)

        total_boost = remaining + borrowable + daily_increase

        return Adjustment(
            type=AdjustmentType.PERFORMANCE_BOOST,
            hour=hour,
            action='increase_budget',
            reason=f'ROAS {current_roas:.2f}x exceeds baseline {baseline_roas:.2f}x',
            amount=total_boost,
            details={
                'current_roas': current_roas,
                'baseline_roas': baseline_roas,
                'ratio': current_roas / baseline_roas,
                'borrowed_from': donor_hours,
                'borrow_amount': borrowable,
                'daily_increase': daily_increase,
                'new_daily_budget': self.current_daily_budget + daily_increase
            }
        )

    def _create_reduce_adjustment(
        self,
        hour: int,
        current_roas: float,
        baseline_roas: float
    ) -> Adjustment:
        """
        Create adjustment for underperforming hour.

        Actions:
        1. Reduce remaining budget
        2. Reallocate to better hours
        """
        remaining = max(0, self.current_allocation[hour] - self.spent[hour])
        reduction = remaining * 0.5

        # Find recipient hours (high performers)
        recipient_hours = self._find_high_performing_hours(exclude=hour)

        return Adjustment(
            type=AdjustmentType.PERFORMANCE_REDUCE,
            hour=hour,
            action='decrease_budget',
            reason=f'ROAS {current_roas:.2f}x below baseline {baseline_roas:.2f}x',
            amount=reduction,
            details={
                'current_roas': current_roas,
                'baseline_roas': baseline_roas,
                'ratio': current_roas / baseline_roas,
                'reduction': reduction,
                'reallocate_to': recipient_hours
            }
        )

    def _find_donor_hours(self, exclude: int) -> List[int]:
        """Find underperforming hours that can donate budget."""
        donors = []
        for hour in range(24):
            if hour == exclude:
                continue
            if self.spent[hour] <= 0:
                continue

            current_roas = self.revenue[hour] / self.spent[hour]
            baseline = self.baseline_roas.get(hour, 3.0)

            if current_roas < baseline * 0.9:
                donors.append(hour)

        return donors

    def _find_high_performing_hours(self, exclude: int) -> List[int]:
        """Find high-performing hours to receive budget."""
        recipients = []
        for hour in range(24):
            if hour == exclude:
                continue
            if self.spent[hour] <= 0:
                continue

            current_roas = self.revenue[hour] / self.spent[hour]
            baseline = self.baseline_roas.get(hour, 3.0)

            if current_roas > baseline * 1.1:
                recipients.append(hour)

        # If no high performers, return future hours
        if not recipients:
            current_events = [e.hour for e in self.events[-10:]] if self.events else []
            current_hour = max(current_events) if current_events else 12
            recipients = [h for h in range(current_hour + 1, 24) if h != exclude]

        return recipients[:3]  # Max 3 recipients

    def _apply_adjustment(self, adjustment: Adjustment):
        """Apply an adjustment to the allocation."""
        self.adjustments.append(adjustment)

        if adjustment.type == AdjustmentType.PERFORMANCE_BOOST:
            details = adjustment.details

            # Update daily budget
            if 'daily_increase' in details and details['daily_increase'] > 0:
                self.current_daily_budget += details['daily_increase']

            # Increase current hour allocation
            boost_amount = adjustment.amount
            self.current_allocation[adjustment.hour] += boost_amount * 0.5

            # Reduce donor hours
            for donor in details.get('borrowed_from', []):
                if donor in self.current_allocation:
                    reduce = min(
                        self.current_allocation[donor] * 0.3,
                        boost_amount / len(details.get('borrowed_from', [1]))
                    )
                    self.current_allocation[donor] = max(0, self.current_allocation[donor] - reduce)

        elif adjustment.type == AdjustmentType.PERFORMANCE_REDUCE:
            details = adjustment.details

            # Reduce current hour
            reduction = details.get('reduction', 0)
            self.current_allocation[adjustment.hour] = max(
                0,
                self.current_allocation[adjustment.hour] - reduction
            )

            # Redistribute to recipients
            recipients = details.get('reallocate_to', [])
            if recipients:
                per_recipient = reduction / len(recipients)
                for r in recipients:
                    if r in self.current_allocation:
                        self.current_allocation[r] += per_recipient

    def get_current_allocation(self) -> Dict[int, float]:
        """Get current (adjusted) allocation."""
        return self.current_allocation.copy()

    def get_remaining_budget(self) -> Dict[int, float]:
        """Get remaining budget per hour."""
        return {
            h: max(0, self.current_allocation.get(h, 0) - self.spent.get(h, 0))
            for h in range(24)
        }

    def get_hourly_performance(self) -> Dict[int, Dict]:
        """Get performance metrics per hour."""
        perf = {}
        for hour in range(24):
            spent = self.spent.get(hour, 0)
            revenue = self.revenue.get(hour, 0)
            baseline = self.baseline_roas.get(hour, 3.0)

            perf[hour] = {
                'spent': spent,
                'revenue': revenue,
                'roas': revenue / spent if spent > 0 else 0,
                'baseline_roas': baseline,
                'vs_baseline': (revenue / spent / baseline - 1) * 100 if spent > 0 and baseline > 0 else 0,
                'budget_utilization': spent / self.current_allocation.get(hour, 1) if self.current_allocation.get(hour, 0) > 0 else 0
            }

        return perf

    def get_summary(self) -> Dict:
        """Get summary of controller state and adjustments."""
        total_spent = sum(self.spent.values())
        total_revenue = sum(self.revenue.values())

        return {
            'base_daily_budget': self.base_daily_budget,
            'current_daily_budget': self.current_daily_budget,
            'daily_budget_change_pct': (self.current_daily_budget / self.base_daily_budget - 1) * 100,
            'total_spent': total_spent,
            'total_revenue': total_revenue,
            'overall_roas': total_revenue / total_spent if total_spent > 0 else 0,
            'n_events': len(self.events),
            'n_adjustments': len(self.adjustments),
            'adjustments_by_type': {
                t.value: len([a for a in self.adjustments if a.type == t])
                for t in AdjustmentType
            },
            'budget_utilization': total_spent / self.base_daily_budget if self.base_daily_budget > 0 else 0
        }

    def simulate_day(
        self,
        events: List[Dict],
        check_every_n_events: int = 10
    ) -> Dict:
        """
        Simulate a full day with provided events.

        Args:
            events: List of event dicts with 'hour', 'minute', 'spend', 'revenue'
            check_every_n_events: Check for adjustments every N events

        Returns:
            Simulation results
        """
        for i, event in enumerate(events):
            self.record_event(
                hour=event['hour'],
                minute=event['minute'],
                spend=event['spend'],
                revenue=event['revenue']
            )

            # Periodic check
            if (i + 1) % check_every_n_events == 0:
                self.check_and_adjust(event['hour'], event['minute'])

        return {
            'summary': self.get_summary(),
            'hourly_performance': self.get_hourly_performance(),
            'final_allocation': self.get_current_allocation(),
            'adjustments': [
                {
                    'type': a.type.value,
                    'hour': a.hour,
                    'action': a.action,
                    'reason': a.reason,
                    'amount': a.amount
                }
                for a in self.adjustments
            ]
        }


class MultiCampaignRealtimeController:
    """
    Manages realtime controllers for multiple campaigns.
    """

    def __init__(
        self,
        campaign_allocations: Dict[str, Dict[int, float]],
        campaign_budgets: Dict[str, float],
        baseline_roas: Optional[Dict[str, Dict[int, float]]] = None
    ):
        """
        Initialize multi-campaign controller.

        Args:
            campaign_allocations: {campaign: {hour: budget}}
            campaign_budgets: {campaign: daily_budget}
            baseline_roas: {campaign: {hour: roas}}
        """
        self.controllers: Dict[str, RealtimeBudgetController] = {}

        for campaign, allocation in campaign_allocations.items():
            budget = campaign_budgets.get(campaign, sum(allocation.values()))
            baseline = baseline_roas.get(campaign) if baseline_roas else None

            self.controllers[campaign] = RealtimeBudgetController(
                planned_allocation=allocation,
                daily_budget=budget,
                baseline_roas=baseline
            )

    def record_event(
        self,
        campaign: str,
        hour: int,
        minute: int,
        spend: float,
        revenue: float
    ):
        """Record event for a specific campaign."""
        if campaign in self.controllers:
            self.controllers[campaign].record_event(hour, minute, spend, revenue)

    def check_all(
        self,
        current_hour: int,
        current_minute: int
    ) -> Dict[str, List[Adjustment]]:
        """Check and adjust all campaigns."""
        adjustments = {}
        for campaign, controller in self.controllers.items():
            adj = controller.check_and_adjust(current_hour, current_minute)
            if adj:
                adjustments[campaign] = adj
        return adjustments

    def get_summary(self) -> Dict:
        """Get summary for all campaigns."""
        return {
            campaign: controller.get_summary()
            for campaign, controller in self.controllers.items()
        }
