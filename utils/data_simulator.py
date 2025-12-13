"""
Data Simulator for Campaign Data Generation

Generates synthetic campaign data for testing:
- Cold start scenarios (new campaigns)
- Diminishing returns patterns
- Various performance profiles
- Edge cases and anomalies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random


class CampaignDataSimulator:
    """
    Simulates realistic campaign data for testing and validation.
    """

    # Product category configurations
    CATEGORY_CONFIGS = {
        'electronics': {
            'peak_hours': [12, 13, 14, 20, 21, 22],
            'base_roas': 3.5,
            'base_ctr': 0.025,
            'base_cvr': 0.08,
            'volatility': 0.3
        },
        'fashion': {
            'peak_hours': [18, 19, 20, 21, 22],
            'base_roas': 4.0,
            'base_ctr': 0.035,
            'base_cvr': 0.06,
            'volatility': 0.4
        },
        'home': {
            'peak_hours': [9, 10, 11, 19, 20],
            'base_roas': 3.0,
            'base_ctr': 0.02,
            'base_cvr': 0.10,
            'volatility': 0.2
        },
        'beauty': {
            'peak_hours': [10, 11, 12, 19, 20, 21],
            'base_roas': 4.5,
            'base_ctr': 0.04,
            'base_cvr': 0.07,
            'volatility': 0.35
        },
        'sports': {
            'peak_hours': [6, 7, 8, 17, 18, 19],
            'base_roas': 3.2,
            'base_ctr': 0.028,
            'base_cvr': 0.09,
            'volatility': 0.25
        }
    }

    # Audience type multipliers
    AUDIENCE_MULTIPLIERS = {
        'broad': {'roas': 0.7, 'volume': 1.5},
        'interest': {'roas': 1.0, 'volume': 1.0},
        'retargeting': {'roas': 1.8, 'volume': 0.4},
        'lookalike': {'roas': 1.2, 'volume': 0.8}
    }

    # Day of week patterns (1=Monday, 7=Sunday)
    DOW_PATTERNS = {
        1: 0.9,   # Monday - lower
        2: 0.95,  # Tuesday
        3: 1.0,   # Wednesday - baseline
        4: 1.0,   # Thursday
        5: 1.1,   # Friday - higher
        6: 1.2,   # Saturday - peak
        7: 1.15   # Sunday
    }

    def __init__(self, seed: int = 42):
        """
        Initialize simulator with random seed for reproducibility.

        Args:
            seed: Random seed
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_campaigns(
        self,
        n_campaigns: int = 20,
        n_days: int = 30,
        start_date: str = '2025-10-01',
        include_new_campaigns: int = 3
    ) -> pd.DataFrame:
        """
        Generate complete campaign dataset.

        Args:
            n_campaigns: Total number of campaigns
            n_days: Number of days of data
            start_date: Start date for data
            include_new_campaigns: Number of campaigns with limited history (for cold start testing)

        Returns:
            DataFrame with campaign data
        """
        all_data = []
        categories = list(self.CATEGORY_CONFIGS.keys())
        audiences = list(self.AUDIENCE_MULTIPLIERS.keys())

        start = pd.to_datetime(start_date)

        for i in range(n_campaigns):
            campaign_name = f"Campaign_{chr(65 + i)}"  # Campaign_A, Campaign_B, ...
            category = categories[i % len(categories)]
            audience = audiences[i % len(audiences)]

            # Determine campaign age (for cold start testing)
            if i < include_new_campaigns:
                # New campaign: only last few days of data
                campaign_days = min(3, n_days)
                campaign_start = n_days - campaign_days
            else:
                campaign_days = n_days
                campaign_start = 0

            # Generate performance tier (some campaigns perform better than others)
            performance_tier = np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2])
            tier_multiplier = {'high': 1.3, 'medium': 1.0, 'low': 0.7}[performance_tier]

            for day_offset in range(campaign_start, n_days):
                date = start + timedelta(days=day_offset)
                dow = date.weekday() + 1  # 1-7

                for hour in range(24):
                    row = self._generate_hourly_record(
                        campaign_name=campaign_name,
                        date=date,
                        hour=hour,
                        day_of_week=dow,
                        category=category,
                        audience=audience,
                        tier_multiplier=tier_multiplier,
                        campaign_age_days=day_offset - campaign_start
                    )
                    row['product_category'] = category
                    row['audience_type'] = audience
                    row['performance_tier'] = performance_tier
                    all_data.append(row)

        df = pd.DataFrame(all_data)

        # Add derived metrics
        df['ctr'] = df['clicks'] / df['impressions'].replace(0, 1)
        df['cvr'] = df['orders'] / df['clicks'].replace(0, 1)
        df['cpc'] = df['spend'] / df['clicks'].replace(0, 1)
        df['acos'] = df['spend'] / df['revenue'].replace(0, 1)
        df['roas'] = df['revenue'] / df['spend'].replace(0, 1)

        # Cap extreme values
        df['acos'] = df['acos'].clip(0, 2)
        df['roas'] = df['roas'].clip(0, 20)
        df['ctr'] = df['ctr'].clip(0, 0.2)
        df['cvr'] = df['cvr'].clip(0, 0.5)

        return df

    def _generate_hourly_record(
        self,
        campaign_name: str,
        date: datetime,
        hour: int,
        day_of_week: int,
        category: str,
        audience: str,
        tier_multiplier: float,
        campaign_age_days: int
    ) -> Dict:
        """Generate a single hourly record."""

        config = self.CATEGORY_CONFIGS[category]
        aud_mult = self.AUDIENCE_MULTIPLIERS[audience]
        dow_mult = self.DOW_PATTERNS[day_of_week]

        # Hour multiplier (peak hours get more traffic)
        if hour in config['peak_hours']:
            hour_mult = 1.5 + np.random.uniform(0, 0.3)
        elif hour in [0, 1, 2, 3, 4, 5]:
            hour_mult = 0.3 + np.random.uniform(0, 0.1)
        else:
            hour_mult = 0.8 + np.random.uniform(0, 0.3)

        # Base metrics with randomness
        volatility = config['volatility']

        base_impressions = 1000 * aud_mult['volume'] * dow_mult * hour_mult
        impressions = int(base_impressions * (1 + np.random.uniform(-volatility, volatility)))
        impressions = max(10, impressions)

        ctr = config['base_ctr'] * (1 + np.random.uniform(-volatility, volatility))
        clicks = int(impressions * ctr)
        clicks = max(0, clicks)

        cvr = config['base_cvr'] * tier_multiplier * (1 + np.random.uniform(-volatility, volatility))
        orders = int(clicks * cvr)
        orders = max(0, orders)

        # Spend with diminishing returns simulation
        base_spend = clicks * (0.5 + np.random.uniform(0, 0.3))  # CPC varies
        spend = round(base_spend, 2)

        # Revenue with ROAS variation
        base_roas = config['base_roas'] * aud_mult['roas'] * tier_multiplier
        # Add diminishing returns effect for high spend
        if spend > 20:
            diminishing_factor = 1 - 0.1 * np.log(spend / 20)
            base_roas *= max(0.5, diminishing_factor)

        roas = base_roas * (1 + np.random.uniform(-volatility, volatility))
        revenue = round(spend * roas, 2)

        return {
            'campaign_name': campaign_name,
            'date': date,
            'hour': hour,
            'day_of_week': day_of_week,
            'impressions': impressions,
            'clicks': clicks,
            'orders': orders,
            'spend': spend,
            'revenue': revenue,
            'campaign_age_days': campaign_age_days
        }

    def simulate_diminishing_returns_data(
        self,
        campaign_name: str = 'Test_Campaign',
        budget_levels: List[float] = None,
        curve_type: str = 'log',
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """
        Generate data specifically showing diminishing returns pattern.

        Args:
            campaign_name: Campaign identifier
            budget_levels: List of budget levels to simulate
            curve_type: 'log', 'sqrt', or 'hill'
            noise_level: Amount of noise to add

        Returns:
            DataFrame with budget-revenue pairs
        """
        if budget_levels is None:
            budget_levels = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]

        data = []
        for budget in budget_levels:
            # Generate multiple observations per budget level
            for _ in range(5):
                if curve_type == 'log':
                    base_revenue = 50 * np.log(1 + 0.1 * budget)
                elif curve_type == 'sqrt':
                    base_revenue = 20 * np.sqrt(budget)
                else:  # hill
                    base_revenue = 200 * (budget ** 1.5) / (50 ** 1.5 + budget ** 1.5)

                noise = np.random.normal(0, noise_level * base_revenue)
                revenue = max(0, base_revenue + noise)

                data.append({
                    'campaign_name': campaign_name,
                    'spend': budget,
                    'revenue': round(revenue, 2),
                    'roas': round(revenue / budget, 2) if budget > 0 else 0
                })

        return pd.DataFrame(data)

    def simulate_realtime_events(
        self,
        planned_allocation: Dict[int, float],
        performance_scenario: str = 'normal'
    ) -> List[Dict]:
        """
        Simulate real-time spending events for testing realtime controller.

        Args:
            planned_allocation: Planned hourly budget allocation
            performance_scenario: 'normal', 'high_performance', 'low_performance', 'volatile'

        Returns:
            List of spend events with timestamps
        """
        events = []

        scenario_configs = {
            'normal': {'roas_mult': 1.0, 'spend_rate': 1.0, 'volatility': 0.2},
            'high_performance': {'roas_mult': 1.5, 'spend_rate': 1.3, 'volatility': 0.15},
            'low_performance': {'roas_mult': 0.6, 'spend_rate': 0.8, 'volatility': 0.25},
            'volatile': {'roas_mult': 1.0, 'spend_rate': 1.0, 'volatility': 0.5}
        }

        config = scenario_configs.get(performance_scenario, scenario_configs['normal'])

        for hour, budget in planned_allocation.items():
            # Generate events throughout the hour
            n_events = np.random.randint(5, 15)
            remaining_budget = budget

            for event_idx in range(n_events):
                if remaining_budget <= 0:
                    break

                # Spend amount
                spend_rate = config['spend_rate'] * (1 + np.random.uniform(-0.3, 0.3))
                event_spend = min(remaining_budget, budget / n_events * spend_rate)

                # Revenue
                base_roas = 3.0 * config['roas_mult']
                volatility = config['volatility']
                event_roas = base_roas * (1 + np.random.uniform(-volatility, volatility))
                event_revenue = event_spend * event_roas

                minute = int((event_idx / n_events) * 60)

                events.append({
                    'hour': hour,
                    'minute': minute,
                    'spend': round(event_spend, 2),
                    'revenue': round(event_revenue, 2),
                    'roas': round(event_roas, 2)
                })

                remaining_budget -= event_spend

        return events

    def inject_anomalies(
        self,
        data: pd.DataFrame,
        anomaly_rate: float = 0.05
    ) -> pd.DataFrame:
        """
        Add realistic anomalies to data for robustness testing.

        Args:
            data: Original DataFrame
            anomaly_rate: Proportion of rows to make anomalous

        Returns:
            DataFrame with anomalies
        """
        df = data.copy()
        n_anomalies = int(len(df) * anomaly_rate)
        anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)

        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'drop', 'zero', 'outlier'])

            if anomaly_type == 'spike':
                df.loc[idx, 'revenue'] *= np.random.uniform(3, 10)
                df.loc[idx, 'orders'] *= np.random.randint(2, 5)
            elif anomaly_type == 'drop':
                df.loc[idx, 'revenue'] *= np.random.uniform(0.1, 0.3)
                df.loc[idx, 'orders'] = 0
            elif anomaly_type == 'zero':
                df.loc[idx, ['clicks', 'orders', 'revenue']] = 0
            else:  # outlier
                df.loc[idx, 'spend'] *= np.random.uniform(5, 20)

        # Recalculate derived metrics
        df['roas'] = df['revenue'] / df['spend'].replace(0, 1)
        df['acos'] = df['spend'] / df['revenue'].replace(0, 1)

        return df


def generate_test_dataset(
    n_campaigns: int = 15,
    n_days: int = 14,
    include_new: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Convenience function to generate a test dataset.

    Args:
        n_campaigns: Number of campaigns
        n_days: Days of history
        include_new: Number of new campaigns (for cold start testing)
        seed: Random seed

    Returns:
        Generated DataFrame
    """
    simulator = CampaignDataSimulator(seed=seed)
    return simulator.generate_campaigns(
        n_campaigns=n_campaigns,
        n_days=n_days,
        include_new_campaigns=include_new
    )
