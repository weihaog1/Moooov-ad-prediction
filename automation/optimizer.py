"""
Optional Feature 3: Automated Budget Optimization Module

Encapsulates all optimization algorithms into a simple automation module.
No API needed - just a Python class you can import and use!
"""

import pandas as pd
from pathlib import Path
import json
from typing import Dict, Optional
from datetime import datetime

from algorithms.hourly_allocation import HourlyBudgetAllocator
from algorithms.multi_campaign import MultiCampaignOptimizer
from algorithms.clustering import CampaignClusterer
from models.revenue_predictor import RevenuePredictor
from utils.data_loader import load_campaign_data, filter_campaign, get_hourly_metrics


class BudgetOptimizer:
    """
    Automated Budget Optimization Module.

    A simple, all-in-one interface for budget optimization.
    No database, no API - just load data and optimize!

    Example:
        >>> optimizer = BudgetOptimizer('data/data_for_ads.xlsx')
        >>> result = optimizer.optimize_hourly('Campaign_A', daily_budget=20, day_of_week=1)
        >>> print(result['expected_roas'])  # 4.59
    """

    def __init__(self, data_file: str = 'data/data_for_ads.xlsx'):
        """
        Initialize optimizer with data file.

        Args:
            data_file: Path to Excel file with campaign data
        """
        self.data_file = data_file
        self.data = None

        # Initialize algorithms
        self.hourly_allocator = HourlyBudgetAllocator()
        self.multi_optimizer = MultiCampaignOptimizer()
        self.clusterer = CampaignClusterer()
        self.predictor = RevenuePredictor()

        print("="*60)
        print("Budget Optimizer Initialized")
        print("="*60)

    def load_data(self) -> pd.DataFrame:
        """Load campaign data from Excel file."""
        if self.data is None:
            self.data = load_campaign_data(self.data_file)
        return self.data

    def optimize_hourly(
        self,
        campaign_name: str,
        daily_budget: float,
        day_of_week: int,
        save_result: bool = True
    ) -> Dict:
        """
        Optimize hourly budget allocation for a single campaign.

        Args:
            campaign_name: Name of campaign
            daily_budget: Total budget for the day
            day_of_week: Day of week (1=Monday, 7=Sunday)
            save_result: Whether to save result to JSON

        Returns:
            Optimization results
        """
        print("\n" + "="*60)
        print("TASK 1: HOURLY BUDGET ALLOCATION")
        print("="*60)

        # Load data
        df = self.load_data()

        # Filter for campaign and day
        campaign_df = filter_campaign(df, campaign_name)
        dow_df = campaign_df[campaign_df['day_of_week'] == day_of_week]

        # Get hourly metrics
        hourly_metrics = get_hourly_metrics(dow_df)

        # Optimize
        result = self.hourly_allocator.optimize(daily_budget, hourly_metrics)
        result['campaign_name'] = campaign_name
        result['day_of_week'] = day_of_week

        # Save
        if save_result:
            self._save_result(result, 'task1_hourly_allocation')

        return result

    def optimize_weekly(
        self,
        campaign_name: str,
        weekly_budget_limit: float,
        min_daily_budget: float,
        max_daily_budget: float,
        save_result: bool = True
    ) -> Dict:
        """
        Optimize weekly budget allocation for a single campaign.

        Args:
            campaign_name: Name of campaign
            weekly_budget_limit: Total budget for the week
            min_daily_budget: Minimum per day
            max_daily_budget: Maximum per day
            save_result: Whether to save result

        Returns:
            Weekly optimization results
        """
        print("\n" + "="*60)
        print("TASK 1: WEEKLY BUDGET ALLOCATION")
        print("="*60)

        # Load data
        df = self.load_data()
        campaign_df = filter_campaign(df, campaign_name)

        # Optimize
        result = self.hourly_allocator.optimize_weekly(
            weekly_budget_limit, campaign_df, min_daily_budget, max_daily_budget
        )
        result['campaign_name'] = campaign_name

        # Save
        if save_result:
            self._save_result(result, 'task1_weekly_allocation')

        return result

    def optimize_multi_campaign(
        self,
        total_budget: float,
        campaign_names: Optional[list] = None,
        optimization_target: str = "maximize_revenue",
        save_result: bool = True
    ) -> Dict:
        """
        Optimize budget allocation across multiple campaigns.

        Args:
            total_budget: Total budget to allocate
            campaign_names: List of campaigns (None = all campaigns)
            optimization_target: Optimization goal
            save_result: Whether to save result

        Returns:
            Multi-campaign optimization results
        """
        print("\n" + "="*60)
        print("TASK 2: MULTI-CAMPAIGN OPTIMIZATION")
        print("="*60)

        # Load data
        df = self.load_data()

        # Filter campaigns if specified
        if campaign_names:
            df = df[df['campaign_name'].isin(campaign_names)]

        # Optimize
        result = self.multi_optimizer.optimize(
            total_budget, df, optimization_target
        )

        # Save
        if save_result:
            self._save_result(result, 'task2_multi_campaign')

        return result

    def analyze_clustering(
        self,
        n_clusters: int = 3,
        save_result: bool = True
    ) -> Dict:
        """
        Perform clustering analysis on campaigns.

        Args:
            n_clusters: Number of clusters
            save_result: Whether to save result

        Returns:
            Clustering results
        """
        print("\n" + "="*60)
        print("OPTIONAL 1: CAMPAIGN CLUSTERING")
        print("="*60)

        # Load data
        df = self.load_data()

        # Cluster
        result = self.clusterer.cluster(df, n_clusters)

        # Save
        if save_result:
            self._save_result(result, 'optional1_clustering')

        return result

    def train_predictor(
        self,
        save_model: bool = True
    ) -> Dict:
        """
        Train revenue prediction model.

        Args:
            save_model: Whether to save trained model

        Returns:
            Training results
        """
        print("\n" + "="*60)
        print("OPTIONAL 2: REVENUE PREDICTION MODEL")
        print("="*60)

        # Load data
        df = self.load_data()

        # Train
        result = self.predictor.train(df)

        # Save model
        if save_model and 'error' not in result:
            self.predictor.save_model()

        return result

    def run_all_optimizations(self):
        """
        Run all optimization tasks automatically.

        This is the "automation" feature - one command does everything!
        """
        print("\n" + "="*70)
        print("RUNNING ALL OPTIMIZATIONS AUTOMATICALLY")
        print("="*70)

        results = {}

        # Task 1: Hourly allocation
        try:
            results['task1_hourly'] = self.optimize_hourly(
                campaign_name='Campaign_A',
                daily_budget=20.0,
                day_of_week=1
            )
        except Exception as e:
            print(f"[ERROR] Task 1 Hourly failed: {e}")

        # Task 1: Weekly allocation
        try:
            results['task1_weekly'] = self.optimize_weekly(
                campaign_name='Campaign_A',
                weekly_budget_limit=150.0,
                min_daily_budget=15.0,
                max_daily_budget=30.0
            )
        except Exception as e:
            print(f"[ERROR] Task 1 Weekly failed: {e}")

        # Task 2: Multi-campaign
        try:
            results['task2_multi'] = self.optimize_multi_campaign(
                total_budget=100.0
            )
        except Exception as e:
            print(f"[ERROR] Task 2 failed: {e}")

        # Optional 1: Clustering
        try:
            results['optional1_clustering'] = self.analyze_clustering()
        except Exception as e:
            print(f"[ERROR] Clustering failed: {e}")

        # Optional 2: ML Prediction
        try:
            results['optional2_prediction'] = self.train_predictor()
        except Exception as e:
            print(f"[ERROR] ML Prediction failed: {e}")

        print("\n" + "="*70)
        print("[DONE] AUTOMATION COMPLETE!")
        print("="*70)
        print(f"Results saved to: outputs/")

        return results

    def _save_result(self, result: Dict, filename_prefix: str):
        """Save result to JSON file."""
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / f"{filename_prefix}_result.json"

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"[SAVED] Result saved: {filepath}")
