"""
Optional Feature 3: Automated Budget Optimization Module

Encapsulates all optimization algorithms into a simple automation module.
No API needed - just a Python class you can import and use!

Enhanced with integrated pipeline connecting:
- Revenue prediction → Budget allocation weights
- Campaign clustering → Strategy-based constraints
"""

import pandas as pd
from pathlib import Path
import json
from typing import Dict, Optional, List
from datetime import datetime

from algorithms.hourly_allocation import HourlyBudgetAllocator
from algorithms.multi_campaign import MultiCampaignOptimizer
from algorithms.clustering import CampaignClusterer
from algorithms.integrated_allocator import (
    IntegratedHourlyAllocator,
    ClusterStrategyMapper,
    create_integrated_pipeline
)
from models.revenue_predictor import RevenuePredictor
from utils.data_loader import load_campaign_data, filter_campaign, get_hourly_metrics


class BudgetOptimizer:
    """
    Automated Budget Optimization Module.

    A simple, all-in-one interface for budget optimization.
    No database, no API - just load data and optimize!

    Supports two modes:
    1. Legacy mode: Independent modules (original behavior)
    2. Integrated mode: Connected pipeline (prediction → clustering → allocation)

    Example (Legacy):
        >>> optimizer = BudgetOptimizer('data/data_for_ads.xlsx')
        >>> result = optimizer.optimize_hourly('Campaign_A', daily_budget=20, day_of_week=1)

    Example (Integrated):
        >>> optimizer = BudgetOptimizer('data/data_for_ads.xlsx')
        >>> optimizer.initialize_integrated_pipeline()
        >>> result = optimizer.optimize_hourly_integrated('Campaign_A', daily_budget=20, day_of_week=1)
    """

    def __init__(self, data_file: str = 'data/data_for_ads.xlsx'):
        """
        Initialize optimizer with data file.

        Args:
            data_file: Path to Excel file with campaign data
        """
        self.data_file = data_file
        self.data = None

        # Initialize legacy algorithms
        self.hourly_allocator = HourlyBudgetAllocator()
        self.multi_optimizer = MultiCampaignOptimizer()
        self.clusterer = CampaignClusterer()
        self.predictor = RevenuePredictor()

        # Integrated pipeline components (initialized lazily)
        self.integrated_allocator: Optional[IntegratedHourlyAllocator] = None
        self.clustering_result: Optional[Dict] = None
        self.pipeline_initialized = False

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

    # =========================================================================
    # INTEGRATED PIPELINE METHODS
    # =========================================================================

    def initialize_integrated_pipeline(
        self,
        n_clusters: int = 3,
        train_predictor: bool = True,
        save_components: bool = True
    ) -> Dict:
        """
        Initialize the integrated pipeline with all components connected.

        This runs:
        1. Campaign clustering to identify performance groups
        2. Revenue predictor training for forward-looking weights
        3. Creates integrated allocator combining both

        Args:
            n_clusters: Number of clusters for campaign grouping
            train_predictor: Whether to train the ML predictor
            save_components: Whether to save clustering/model results

        Returns:
            Initialization results including clustering and training metrics
        """
        print("\n" + "="*70)
        print("INITIALIZING INTEGRATED PIPELINE")
        print("="*70)

        results = {}
        df = self.load_data()

        # Step 1: Run clustering
        print("\n[Step 1/3] Running campaign clustering...")
        self.clustering_result = self.clusterer.cluster(df, n_clusters)
        results['clustering'] = self.clustering_result

        if save_components:
            self._save_result(self.clustering_result, 'integrated_clustering')

        # Step 2: Train predictor (optional but recommended)
        predictor_for_allocator = None
        if train_predictor:
            print("\n[Step 2/3] Training revenue predictor...")
            train_result = self.predictor.train(df)
            results['predictor_training'] = train_result

            if 'error' not in train_result:
                predictor_for_allocator = self.predictor
                if save_components:
                    self.predictor.save_model()
            else:
                print(f"[WARNING] Predictor training failed: {train_result.get('message')}")
                print("   Pipeline will use historical weights instead.")
        else:
            print("\n[Step 2/3] Skipping predictor training (use_prediction=False)")

        # Step 3: Create integrated allocator
        print("\n[Step 3/3] Creating integrated allocator...")
        self.integrated_allocator = IntegratedHourlyAllocator(
            predictor=predictor_for_allocator,
            clustering_result=self.clustering_result
        )

        self.pipeline_initialized = True

        # Summary
        print("\n" + "-"*50)
        print("PIPELINE INITIALIZED SUCCESSFULLY")
        print("-"*50)
        print(f"   Clusters: {n_clusters}")
        print(f"   Predictor: {'Trained' if predictor_for_allocator else 'Not available'}")
        print(f"   Campaigns mapped to strategies:")

        strategy_mapper = self.integrated_allocator.strategy_mapper
        for campaign, strategy in strategy_mapper.campaign_strategies.items():
            print(f"      - {campaign}: {strategy.value}")

        print("-"*50)

        return results

    def optimize_hourly_integrated(
        self,
        campaign_name: str,
        daily_budget: float,
        day_of_week: int,
        use_prediction: bool = True,
        use_strategy: bool = True,
        save_result: bool = True
    ) -> Dict:
        """
        Optimize hourly allocation using the integrated pipeline.

        Uses:
        - ML predictions for allocation weights (instead of historical averages)
        - Cluster-based strategy for constraints (instead of fixed parameters)

        Args:
            campaign_name: Target campaign
            daily_budget: Total budget for the day
            day_of_week: Day of week (1=Monday, 7=Sunday)
            use_prediction: Use ML predictions for weights
            use_strategy: Apply cluster-based strategy constraints
            save_result: Save result to JSON

        Returns:
            Optimization results with strategy metadata
        """
        print("\n" + "="*60)
        print("TASK 1: INTEGRATED HOURLY BUDGET ALLOCATION")
        print("="*60)

        # Auto-initialize pipeline if needed
        if not self.pipeline_initialized:
            print("[INFO] Pipeline not initialized. Running initialization...")
            self.initialize_integrated_pipeline()

        # Load data
        df = self.load_data()

        # Run integrated optimization
        result = self.integrated_allocator.optimize(
            daily_budget=daily_budget,
            campaign_data=df,
            campaign_name=campaign_name,
            day_of_week=day_of_week,
            use_prediction=use_prediction,
            use_strategy=use_strategy
        )

        # Save
        if save_result:
            self._save_result(result, 'task1_integrated_hourly')

        return result

    def optimize_weekly_integrated(
        self,
        campaign_name: str,
        weekly_budget_limit: float,
        min_daily_budget: float,
        max_daily_budget: float,
        use_prediction: bool = True,
        use_strategy: bool = True,
        save_result: bool = True
    ) -> Dict:
        """
        Optimize weekly allocation using the integrated pipeline.

        Args:
            campaign_name: Target campaign
            weekly_budget_limit: Total budget for the week
            min_daily_budget: Minimum per day
            max_daily_budget: Maximum per day
            use_prediction: Use ML predictions
            use_strategy: Apply cluster-based strategy
            save_result: Save result to JSON

        Returns:
            Weekly optimization results
        """
        print("\n" + "="*60)
        print("TASK 1: INTEGRATED WEEKLY BUDGET ALLOCATION")
        print("="*60)

        # Auto-initialize pipeline if needed
        if not self.pipeline_initialized:
            print("[INFO] Pipeline not initialized. Running initialization...")
            self.initialize_integrated_pipeline()

        df = self.load_data()

        # Calculate daily performance scores using predictions if available
        daily_budgets = self._compute_weekly_daily_budgets(
            df, campaign_name, weekly_budget_limit,
            min_daily_budget, max_daily_budget,
            use_prediction
        )

        # Optimize hourly for each day
        dow_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        weekly_strategy = {}

        for dow in range(1, 8):
            if daily_budgets[dow] < min_daily_budget:
                continue

            day_result = self.integrated_allocator.optimize(
                daily_budget=daily_budgets[dow],
                campaign_data=df,
                campaign_name=campaign_name,
                day_of_week=dow,
                use_prediction=use_prediction,
                use_strategy=use_strategy
            )

            weekly_strategy[dow_names[dow-1]] = {
                'daily_budget': daily_budgets[dow],
                'hourly_allocation': day_result['hourly_allocation'],
                'total': day_result['total_allocated'],
                'weight_source': day_result.get('weight_source', 'unknown'),
                'strategy': day_result.get('strategy', {})
            }

        total_weekly = sum(daily_budgets.values())

        result = {
            'campaign_name': campaign_name,
            'weekly_budget_strategy': weekly_strategy,
            'daily_budgets': daily_budgets,
            'weekly_total': round(total_weekly, 2),
            'pipeline_mode': 'integrated',
            'use_prediction': use_prediction,
            'use_strategy': use_strategy
        }

        if save_result:
            self._save_result(result, 'task1_integrated_weekly')

        return result

    def _compute_weekly_daily_budgets(
        self,
        df: pd.DataFrame,
        campaign_name: str,
        weekly_budget: float,
        min_daily: float,
        max_daily: float,
        use_prediction: bool
    ) -> Dict[int, float]:
        """Compute daily budgets for the week based on performance."""

        campaign_df = filter_campaign(df, campaign_name)

        daily_scores = {}
        for dow in range(1, 8):
            dow_data = campaign_df[campaign_df['day_of_week'] == dow]

            if dow_data.empty:
                daily_scores[dow] = 1.0
                continue

            if use_prediction and self.integrated_allocator:
                # Use predictions for scoring
                pred_rev, uncertainty = self.integrated_allocator.prediction_integrator.predict_hourly_revenue(
                    df, campaign_name, dow
                )
                # Score = sum of predicted revenue, penalized by uncertainty
                score = pred_rev.sum() * (1 - 0.3 * uncertainty.mean())
                daily_scores[dow] = max(score, 0.1)
            else:
                # Historical scoring
                avg_revenue = dow_data['revenue'].mean()
                avg_roas = dow_data['roas'].mean()
                daily_scores[dow] = max(avg_revenue * avg_roas, 0.1)

        # Allocate proportionally
        total_score = sum(daily_scores.values())
        daily_budgets = {}

        for dow, score in daily_scores.items():
            budget = (score / total_score) * weekly_budget
            budget = max(min_daily, min(max_daily, budget))
            daily_budgets[dow] = round(budget, 2)

        # Adjust to meet weekly limit
        total = sum(daily_budgets.values())
        if total > weekly_budget:
            scale = weekly_budget / total
            daily_budgets = {dow: round(b * scale, 2) for dow, b in daily_budgets.items()}

        return daily_budgets

    def run_integrated_optimizations(
        self,
        campaign_name: str = 'Campaign_A',
        daily_budget: float = 20.0,
        weekly_budget: float = 150.0,
        day_of_week: int = 1
    ) -> Dict:
        """
        Run all optimizations using the integrated pipeline.

        This demonstrates the full connected system:
        1. Initialize pipeline (clustering + prediction)
        2. Run hourly allocation with predictions & strategy
        3. Run weekly allocation with predictions & strategy

        Args:
            campaign_name: Target campaign
            daily_budget: Budget for hourly optimization
            weekly_budget: Budget for weekly optimization
            day_of_week: Day for hourly optimization

        Returns:
            All optimization results
        """
        print("\n" + "="*70)
        print("RUNNING INTEGRATED OPTIMIZATION PIPELINE")
        print("="*70)

        results = {}

        # Step 1: Initialize pipeline
        print("\n" + "="*50)
        print("PHASE 1: PIPELINE INITIALIZATION")
        print("="*50)
        init_result = self.initialize_integrated_pipeline()
        results['initialization'] = init_result

        # Step 2: Hourly optimization (integrated)
        print("\n" + "="*50)
        print("PHASE 2: INTEGRATED HOURLY ALLOCATION")
        print("="*50)
        try:
            hourly_result = self.optimize_hourly_integrated(
                campaign_name=campaign_name,
                daily_budget=daily_budget,
                day_of_week=day_of_week
            )
            results['hourly_integrated'] = hourly_result
        except Exception as e:
            print(f"[ERROR] Hourly integrated failed: {e}")
            results['hourly_integrated'] = {'error': str(e)}

        # Step 3: Weekly optimization (integrated)
        print("\n" + "="*50)
        print("PHASE 3: INTEGRATED WEEKLY ALLOCATION")
        print("="*50)
        try:
            weekly_result = self.optimize_weekly_integrated(
                campaign_name=campaign_name,
                weekly_budget_limit=weekly_budget,
                min_daily_budget=15.0,
                max_daily_budget=30.0
            )
            results['weekly_integrated'] = weekly_result
        except Exception as e:
            print(f"[ERROR] Weekly integrated failed: {e}")
            results['weekly_integrated'] = {'error': str(e)}

        # Summary
        print("\n" + "="*70)
        print("INTEGRATED PIPELINE COMPLETE")
        print("="*70)
        print("\nResults saved to outputs/:")
        print("   - integrated_clustering_result.json")
        print("   - task1_integrated_hourly_result.json")
        print("   - task1_integrated_weekly_result.json")
        print("   - revenue_model.joblib")
        print("="*70)

        return results

    def compare_legacy_vs_integrated(
        self,
        campaign_name: str = 'Campaign_A',
        daily_budget: float = 20.0,
        day_of_week: int = 1
    ) -> Dict:
        """
        Compare legacy (disconnected) vs integrated pipeline results.

        Useful for validating the integrated approach against the baseline.

        Args:
            campaign_name: Target campaign
            daily_budget: Budget for comparison
            day_of_week: Day for comparison

        Returns:
            Comparison results showing both approaches
        """
        print("\n" + "="*70)
        print("COMPARING LEGACY VS INTEGRATED APPROACHES")
        print("="*70)

        results = {}

        # Run legacy
        print("\n[1/2] Running LEGACY optimization...")
        legacy_result = self.optimize_hourly(
            campaign_name=campaign_name,
            daily_budget=daily_budget,
            day_of_week=day_of_week,
            save_result=False
        )
        results['legacy'] = legacy_result

        # Run integrated
        print("\n[2/2] Running INTEGRATED optimization...")
        if not self.pipeline_initialized:
            self.initialize_integrated_pipeline()

        integrated_result = self.optimize_hourly_integrated(
            campaign_name=campaign_name,
            daily_budget=daily_budget,
            day_of_week=day_of_week,
            save_result=False
        )
        results['integrated'] = integrated_result

        # Comparison
        print("\n" + "-"*50)
        print("COMPARISON SUMMARY")
        print("-"*50)

        legacy_alloc = legacy_result.get('hourly_allocation', {})
        integrated_alloc = integrated_result.get('hourly_allocation', {})

        # Find differences
        max_diff_hour = 0
        max_diff = 0
        for h in range(24):
            diff = abs(legacy_alloc.get(h, 0) - integrated_alloc.get(h, 0))
            if diff > max_diff:
                max_diff = diff
                max_diff_hour = h

        print(f"   Legacy source: historical ROAS")
        print(f"   Integrated source: {integrated_result.get('weight_source', 'unknown')}")
        print(f"   Strategy applied: {integrated_result.get('strategy', {}).get('type', 'none')}")
        print(f"   Max hourly difference: ${max_diff:.2f} at hour {max_diff_hour}")

        # Peak hours comparison
        legacy_peaks = legacy_result.get('peak_hours', [])
        integrated_peaks = integrated_result.get('peak_hours', [])
        print(f"   Legacy peak hours: {legacy_peaks}")
        print(f"   Integrated peak hours: {integrated_peaks}")

        print("-"*50)

        # Save comparison
        self._save_result(results, 'comparison_legacy_vs_integrated')

        return results

    # =========================================================================
    # ADVANCED FEATURES
    # =========================================================================

    def optimize_with_cold_start(
        self,
        campaign_name: str,
        daily_budget: float,
        day_of_week: int,
        product_category: str = None,
        audience_type: str = None,
        save_result: bool = True
    ) -> Dict:
        """
        Optimize allocation for new campaigns using cold start strategy.

        Uses 3-level fallback:
        1. Similar campaign transfer
        2. Category baseline
        3. Exploration-weighted uniform

        Args:
            campaign_name: Target campaign
            daily_budget: Daily budget
            day_of_week: Day of week (1=Monday, 7=Sunday)
            product_category: Product category if known
            audience_type: Audience type if known
            save_result: Whether to save result

        Returns:
            Optimization result with cold start metadata
        """
        from algorithms.cold_start import ColdStartHandler

        print("\n" + "="*60)
        print("COLD START OPTIMIZATION")
        print("="*60)

        df = self.load_data()

        # Initialize cold start handler
        handler = ColdStartHandler(historical_data=df)

        # Get initial weights
        cold_start_result = handler.get_initial_weights(
            campaign_name=campaign_name,
            day_of_week=day_of_week,
            product_category=product_category,
            audience_type=audience_type
        )

        print(f"   Campaign: {campaign_name}")
        print(f"   Strategy: {cold_start_result['strategy'].value}")
        print(f"   Confidence: {cold_start_result['confidence']:.2f}")
        print(f"   Exploration ratio: {cold_start_result['exploration_ratio']:.0%}")

        # Use weights for allocation
        weights = cold_start_result['weights']

        # Simple proportional allocation based on weights
        allocation = {h: round(daily_budget * weights[h], 2) for h in range(24)}

        result = {
            'campaign_name': campaign_name,
            'daily_budget': daily_budget,
            'day_of_week': day_of_week,
            'hourly_allocation': allocation,
            'cold_start_strategy': cold_start_result['strategy'].value,
            'confidence': cold_start_result['confidence'],
            'exploration_ratio': cold_start_result['exploration_ratio'],
            'details': cold_start_result['details']
        }

        if save_result:
            self._save_result(result, 'cold_start_allocation')

        return result

    def optimize_with_response_curves(
        self,
        total_budget: float,
        campaigns: List[str] = None,
        save_result: bool = True
    ) -> Dict:
        """
        Multi-campaign allocation with diminishing returns modeling.

        Fits response curves and uses marginal equilibrium for allocation.

        Args:
            total_budget: Total budget to allocate
            campaigns: List of campaigns (None = all)
            save_result: Whether to save result

        Returns:
            Optimization result with response curve info
        """
        from algorithms.response_curve import ResponseCurveModeler, NonlinearBudgetOptimizer

        print("\n" + "="*60)
        print("RESPONSE CURVE OPTIMIZATION")
        print("="*60)

        df = self.load_data()

        if campaigns is None:
            campaigns = df['campaign_name'].unique().tolist()

        # Fit response curves
        modeler = ResponseCurveModeler()
        curve_results = {}

        print("\n[Step 1] Fitting response curves...")
        for campaign in campaigns:
            result = modeler.fit(df, campaign)
            curve_results[campaign] = {
                'curve_type': result.curve_type.value,
                'r_squared': result.r_squared,
                'is_reliable': result.is_reliable,
                'saturation_point': result.saturation_point
            }
            status = "OK" if result.is_reliable else "Unreliable"
            print(f"   {campaign}: {result.curve_type.value} (R^2={result.r_squared:.3f}) [{status}]")

        # Optimize allocation
        print("\n[Step 2] Optimizing allocation...")
        optimizer = NonlinearBudgetOptimizer(modeler)
        allocation_result = optimizer.optimize(total_budget, campaigns)

        print(f"\n   Method: {allocation_result['method']}")
        print(f"   Converged: {allocation_result['converged']}")
        print(f"   Expected Revenue: ${allocation_result['total_expected_revenue']:.2f}")

        result = {
            'total_budget': total_budget,
            'campaigns': campaigns,
            'curve_results': curve_results,
            'allocation': allocation_result['allocation'],
            'expected_revenue': allocation_result['expected_revenue'],
            'total_expected_revenue': allocation_result['total_expected_revenue'],
            'marginal_returns': allocation_result['marginal_returns'],
            'optimization_method': allocation_result['method'],
            'converged': allocation_result['converged']
        }

        if save_result:
            self._save_result(result, 'response_curve_allocation')

        return result

    def optimize_with_bandit(
        self,
        total_budget: float,
        campaigns: List[str] = None,
        method: str = 'thompson',
        exploration_ratio: float = 0.15,
        save_result: bool = True
    ) -> Dict:
        """
        Allocate budget using E&E (Exploration & Exploitation) mechanism.

        Args:
            total_budget: Total budget
            campaigns: List of campaigns (None = all)
            method: 'thompson' or 'ucb'
            exploration_ratio: Ratio for exploration (0-1)
            save_result: Whether to save result

        Returns:
            Allocation with E&E metadata
        """
        from algorithms.bandit import EEBudgetAllocator, BanditMethod

        print("\n" + "="*60)
        print("E&E (BANDIT) OPTIMIZATION")
        print("="*60)

        df = self.load_data()

        if campaigns is None:
            campaigns = df['campaign_name'].unique().tolist()

        bandit_method = BanditMethod.THOMPSON_SAMPLING if method == 'thompson' else BanditMethod.UCB

        # Initialize allocator
        allocator = EEBudgetAllocator(
            campaigns=campaigns,
            method=bandit_method,
            exploration_ratio=exploration_ratio
        )

        # Warm up with historical data
        print("\n[Step 1] Warming up with historical data...")
        for campaign in campaigns:
            camp_data = df[df['campaign_name'] == campaign]
            if len(camp_data) > 0:
                total_spent = camp_data['spend'].sum()
                total_revenue = camp_data['revenue'].sum()
                allocator.update_campaign(campaign, total_spent, total_revenue)

        # Get allocation
        print("\n[Step 2] Computing E&E allocation...")
        allocation_result = allocator.allocate_budget(total_budget)

        print(f"\n   Method: {allocation_result['method']}")
        print(f"   Exploration ratio: {allocation_result['exploration_ratio']:.0%}")

        # Get statistics
        stats = allocator.get_statistics()

        result = {
            'total_budget': total_budget,
            'campaigns': campaigns,
            'method': allocation_result['method'],
            'exploration_ratio': allocation_result['exploration_ratio'],
            'allocation': allocation_result['campaign_allocation'],
            'statistics': stats['campaign_stats']
        }

        if save_result:
            self._save_result(result, 'bandit_allocation')

        return result

    def create_realtime_controller(
        self,
        campaign_name: str,
        daily_budget: float,
        day_of_week: int
    ):
        """
        Create a realtime controller for dynamic budget adjustment.

        Args:
            campaign_name: Target campaign
            daily_budget: Daily budget
            day_of_week: Day of week

        Returns:
            RealtimeBudgetController instance
        """
        from algorithms.realtime_controller import RealtimeBudgetController

        df = self.load_data()

        # Get planned allocation (use integrated if available)
        if self.pipeline_initialized:
            result = self.integrated_allocator.optimize(
                daily_budget=daily_budget,
                campaign_data=df,
                campaign_name=campaign_name,
                day_of_week=day_of_week,
                use_prediction=True,
                use_strategy=True
            )
            planned = result['hourly_allocation']
        else:
            # Simple proportional
            planned = {h: daily_budget / 24 for h in range(24)}

        # Get baseline ROAS from historical data
        camp_data = df[
            (df['campaign_name'] == campaign_name) &
            (df['day_of_week'] == day_of_week)
        ]

        baseline_roas = {}
        for hour in range(24):
            hour_data = camp_data[camp_data['hour'] == hour]
            if len(hour_data) > 0:
                baseline_roas[hour] = hour_data['roas'].mean()
            else:
                baseline_roas[hour] = 3.0

        controller = RealtimeBudgetController(
            planned_allocation=planned,
            daily_budget=daily_budget,
            baseline_roas=baseline_roas
        )

        return controller

    def run_advanced_pipeline(
        self,
        campaign_name: str = 'Campaign_A',
        daily_budget: float = 20.0,
        weekly_budget: float = 150.0,
        day_of_week: int = 1,
        product_category: str = None,
        enable_cold_start: bool = True,
        enable_response_curves: bool = True,
        enable_bandit: bool = True
    ) -> Dict:
        """
        Run the complete advanced optimization pipeline.

        Combines all advanced features:
        1. Cold start for new campaigns
        2. Response curves for diminishing returns
        3. Bandit for E&E
        4. Realtime controller setup

        Args:
            campaign_name: Target campaign
            daily_budget: Daily budget for hourly optimization
            weekly_budget: Budget for multi-campaign optimization
            day_of_week: Day of week
            product_category: Product category (for cold start)
            enable_cold_start: Enable cold start features
            enable_response_curves: Enable response curve modeling
            enable_bandit: Enable E&E mechanism

        Returns:
            Combined results from all features
        """
        print("\n" + "="*70)
        print("RUNNING ADVANCED OPTIMIZATION PIPELINE")
        print("="*70)

        results = {}
        df = self.load_data()
        all_campaigns = df['campaign_name'].unique().tolist()

        # 1. Cold Start (if enabled)
        if enable_cold_start:
            print("\n" + "="*50)
            print("PHASE 1: COLD START ANALYSIS")
            print("="*50)
            try:
                cold_start_result = self.optimize_with_cold_start(
                    campaign_name=campaign_name,
                    daily_budget=daily_budget,
                    day_of_week=day_of_week,
                    product_category=product_category
                )
                results['cold_start'] = cold_start_result
            except Exception as e:
                print(f"[ERROR] Cold start failed: {e}")
                results['cold_start'] = {'error': str(e)}

        # 2. Response Curves (if enabled)
        if enable_response_curves:
            print("\n" + "="*50)
            print("PHASE 2: RESPONSE CURVE MODELING")
            print("="*50)
            try:
                response_result = self.optimize_with_response_curves(
                    total_budget=weekly_budget,
                    campaigns=all_campaigns
                )
                results['response_curves'] = response_result
            except Exception as e:
                print(f"[ERROR] Response curves failed: {e}")
                results['response_curves'] = {'error': str(e)}

        # 3. E&E Bandit (if enabled)
        if enable_bandit:
            print("\n" + "="*50)
            print("PHASE 3: E&E (BANDIT) ALLOCATION")
            print("="*50)
            try:
                bandit_result = self.optimize_with_bandit(
                    total_budget=weekly_budget,
                    campaigns=all_campaigns,
                    method='thompson'
                )
                results['bandit'] = bandit_result
            except Exception as e:
                print(f"[ERROR] Bandit failed: {e}")
                results['bandit'] = {'error': str(e)}

        # 4. Create realtime controller
        print("\n" + "="*50)
        print("PHASE 4: REALTIME CONTROLLER SETUP")
        print("="*50)
        try:
            controller = self.create_realtime_controller(
                campaign_name=campaign_name,
                daily_budget=daily_budget,
                day_of_week=day_of_week
            )
            results['realtime_controller'] = {
                'status': 'created',
                'campaign': campaign_name,
                'daily_budget': daily_budget,
                'budget_flexibility': '±20%',
                'max_daily_budget': controller.max_daily_budget,
                'min_daily_budget': controller.min_daily_budget
            }
            print(f"   Controller created for {campaign_name}")
            print(f"   Budget flexibility: ${controller.min_daily_budget:.2f} - ${controller.max_daily_budget:.2f}")
        except Exception as e:
            print(f"[ERROR] Controller setup failed: {e}")
            results['realtime_controller'] = {'error': str(e)}

        # Summary
        print("\n" + "="*70)
        print("ADVANCED PIPELINE COMPLETE")
        print("="*70)

        self._save_result(results, 'advanced_pipeline')

        return results
