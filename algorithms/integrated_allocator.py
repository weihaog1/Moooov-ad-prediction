"""
Integrated Budget Allocator

Connects revenue prediction, campaign clustering, and budget allocation
into a unified decision pipeline.

Data Flow:
    Raw Data → Predictor (future revenue) → Allocator (weights)
            → Clusterer (strategy)       → Allocator (constraints)
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, Optional, Tuple
from enum import Enum


class CampaignStrategy(Enum):
    """Budget strategy based on cluster characteristics."""
    HIGH_PERFORMER = "high_performer"      # High ROAS, low ACoS - prioritize but cap
    POTENTIAL = "potential"                 # Moderate metrics - explore with budget
    LOW_PERFORMER = "low_performer"         # Low ROAS, high ACoS - minimize budget
    NEW_CAMPAIGN = "new_campaign"           # No history - exploration budget


class ClusterStrategyMapper:
    """
    Maps clustering results to budget allocation strategies.

    Translates cluster characteristics into actionable budget constraints.
    """

    # Strategy parameters: (min_ratio, max_ratio, priority_weight)
    STRATEGY_PARAMS = {
        CampaignStrategy.HIGH_PERFORMER: {
            'min_ratio': 0.10,      # At least 10% of budget
            'max_ratio': 0.35,      # Cap at 35% to prevent over-concentration
            'priority_weight': 1.2,  # 20% boost in allocation priority
            'description': 'High efficiency - prioritize but cap to prevent diminishing returns'
        },
        CampaignStrategy.POTENTIAL: {
            'min_ratio': 0.08,
            'max_ratio': 0.25,
            'priority_weight': 1.0,
            'description': 'Growth potential - allocate exploration budget'
        },
        CampaignStrategy.LOW_PERFORMER: {
            'min_ratio': 0.02,      # Minimal budget
            'max_ratio': 0.10,      # Strict cap
            'priority_weight': 0.6,  # Reduced priority
            'description': 'Low efficiency - minimize or pause'
        },
        CampaignStrategy.NEW_CAMPAIGN: {
            'min_ratio': 0.05,
            'max_ratio': 0.15,
            'priority_weight': 0.9,
            'description': 'New campaign - exploration budget for data collection'
        }
    }

    def __init__(self):
        self.campaign_strategies: Dict[str, CampaignStrategy] = {}
        self.cluster_mapping: Dict[int, CampaignStrategy] = {}

    def map_clusters_to_strategies(self, clustering_result: Dict) -> Dict[str, CampaignStrategy]:
        """
        Convert clustering output to campaign strategies.

        Args:
            clustering_result: Output from CampaignClusterer.cluster()

        Returns:
            Mapping of campaign_name -> CampaignStrategy
        """
        cluster_stats = clustering_result.get('cluster_statistics', {})
        campaigns = clustering_result.get('campaigns', [])

        # Determine strategy for each cluster based on metrics
        for cluster_id, stats in cluster_stats.items():
            cluster_id = int(cluster_id)
            avg_roas = stats.get('avg_roas', 0)
            avg_acos = stats.get('avg_acos', 1)

            if avg_roas > 4.0 and avg_acos < 0.25:
                strategy = CampaignStrategy.HIGH_PERFORMER
            elif avg_roas > 2.5 and avg_acos < 0.35:
                strategy = CampaignStrategy.POTENTIAL
            else:
                strategy = CampaignStrategy.LOW_PERFORMER

            self.cluster_mapping[cluster_id] = strategy

        # Map each campaign to its strategy
        for camp in campaigns:
            campaign_name = camp['campaign_name']
            cluster_id = camp['cluster_id']
            self.campaign_strategies[campaign_name] = self.cluster_mapping.get(
                cluster_id, CampaignStrategy.POTENTIAL
            )

        return self.campaign_strategies

    def get_strategy_params(self, campaign_name: str) -> Dict:
        """Get budget parameters for a campaign based on its strategy."""
        strategy = self.campaign_strategies.get(
            campaign_name, CampaignStrategy.NEW_CAMPAIGN
        )
        return {
            'strategy': strategy,
            **self.STRATEGY_PARAMS[strategy]
        }

    def get_hourly_strategy_adjustments(
        self,
        campaign_name: str,
        base_weights: np.ndarray
    ) -> np.ndarray:
        """
        Adjust hourly weights based on campaign strategy.

        Args:
            campaign_name: Campaign to get adjustments for
            base_weights: Base hourly weights (24,)

        Returns:
            Adjusted weights incorporating strategy
        """
        params = self.get_strategy_params(campaign_name)
        priority_weight = params['priority_weight']

        # Apply priority weight to base weights
        adjusted = base_weights * priority_weight

        # Normalize to sum to 1
        if adjusted.sum() > 0:
            adjusted = adjusted / adjusted.sum()

        return adjusted


class PredictionIntegrator:
    """
    Integrates revenue predictions into budget allocation.

    Uses predicted future revenue instead of historical averages
    for allocation weights.
    """

    def __init__(self, predictor=None):
        """
        Args:
            predictor: Trained RevenuePredictor instance
        """
        self.predictor = predictor
        self.prediction_cache: Dict = {}

    def set_predictor(self, predictor):
        """Set or update the predictor."""
        self.predictor = predictor
        self.prediction_cache.clear()

    def predict_hourly_revenue(
        self,
        campaign_data: pd.DataFrame,
        campaign_name: str,
        day_of_week: int,
        base_metrics: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict revenue for each hour of a specific day.

        Args:
            campaign_data: Historical campaign data
            campaign_name: Target campaign
            day_of_week: Target day (1=Monday, 7=Sunday)
            base_metrics: Optional baseline metrics to use for prediction

        Returns:
            Tuple of (predicted_revenue[24], prediction_uncertainty[24])
        """
        if self.predictor is None or self.predictor.model is None:
            # Fallback to historical average if no predictor
            return self._fallback_to_historical(campaign_data, campaign_name, day_of_week)

        # Prepare prediction features for each hour
        predictions = []
        uncertainties = []

        # Get average metrics for the campaign as baseline
        camp_data = campaign_data[campaign_data['campaign_name'] == campaign_name]
        if base_metrics is None and len(camp_data) > 0:
            base_metrics = camp_data.mean(numeric_only=True)

        for hour in range(24):
            # Create feature row for prediction
            features = self._create_prediction_features(
                hour, day_of_week, campaign_name, campaign_data, base_metrics
            )

            if features is not None:
                pred = self.predictor.model.predict(features)[0]
                predictions.append(max(0, pred))  # Ensure non-negative

                # Estimate uncertainty using prediction variance
                # (simplified - could use quantile regression for better estimates)
                uncertainties.append(self._estimate_uncertainty(
                    campaign_data, campaign_name, hour, day_of_week
                ))
            else:
                predictions.append(0)
                uncertainties.append(1.0)

        return np.array(predictions), np.array(uncertainties)

    def _create_prediction_features(
        self,
        hour: int,
        day_of_week: int,
        campaign_name: str,
        campaign_data: pd.DataFrame,
        base_metrics: pd.Series
    ) -> Optional[pd.DataFrame]:
        """Create feature DataFrame for a single prediction."""

        # Get campaign encoding
        campaign_map = {
            name: idx for idx, name
            in enumerate(campaign_data['campaign_name'].unique())
        }

        if campaign_name not in campaign_map:
            return None

        # Build feature dict matching predictor's expected features
        features = {
            'hour': hour,
            'day_of_week': day_of_week,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'dow_sin': np.sin(2 * np.pi * day_of_week / 7),
            'dow_cos': np.cos(2 * np.pi * day_of_week / 7),
            'is_weekend': 1 if day_of_week in [6, 7] else 0,
            'is_business_hours': 1 if 9 <= hour <= 17 else 0,
            'campaign_id': campaign_map[campaign_name],
            # Add date features (use median values from campaign data)
            'day_of_month': 15,  # Middle of month as default
            'month': 10  # Default month
        }

        # Try to get actual date info from campaign data
        if 'date' in campaign_data.columns:
            camp_dates = campaign_data[campaign_data['campaign_name'] == campaign_name]['date']
            if len(camp_dates) > 0:
                latest_date = pd.to_datetime(camp_dates).max()
                features['day_of_month'] = latest_date.day
                features['month'] = latest_date.month

        # Add performance metrics if available
        if base_metrics is not None:
            for col in ['impressions', 'clicks', 'spend', 'ctr', 'cpc']:
                if col in base_metrics.index:
                    features[col] = base_metrics[col]

        # Filter to only features the model expects
        available_features = [f for f in self.predictor.feature_names if f in features]

        return pd.DataFrame([{f: features[f] for f in available_features}])

    def _estimate_uncertainty(
        self,
        campaign_data: pd.DataFrame,
        campaign_name: str,
        hour: int,
        day_of_week: int
    ) -> float:
        """Estimate prediction uncertainty based on historical variance."""

        mask = (
            (campaign_data['campaign_name'] == campaign_name) &
            (campaign_data['hour'] == hour) &
            (campaign_data['day_of_week'] == day_of_week)
        )

        subset = campaign_data[mask]['revenue']

        if len(subset) < 2:
            return 1.0  # High uncertainty for sparse data

        # Coefficient of variation as uncertainty measure
        mean_rev = subset.mean()
        std_rev = subset.std()

        if mean_rev > 0:
            cv = std_rev / mean_rev
            return min(cv, 2.0)  # Cap at 2.0

        return 1.0

    def _fallback_to_historical(
        self,
        campaign_data: pd.DataFrame,
        campaign_name: str,
        day_of_week: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback to historical averages when predictor unavailable."""

        mask = (
            (campaign_data['campaign_name'] == campaign_name) &
            (campaign_data['day_of_week'] == day_of_week)
        )

        subset = campaign_data[mask]

        revenues = []
        uncertainties = []

        for hour in range(24):
            hour_data = subset[subset['hour'] == hour]['revenue']

            if len(hour_data) > 0:
                revenues.append(hour_data.mean())
                uncertainties.append(
                    hour_data.std() / hour_data.mean() if hour_data.mean() > 0 else 1.0
                )
            else:
                revenues.append(0)
                uncertainties.append(1.0)

        return np.array(revenues), np.array(uncertainties)

    def get_allocation_weights(
        self,
        predicted_revenue: np.ndarray,
        uncertainty: np.ndarray,
        uncertainty_penalty: float = 0.3
    ) -> np.ndarray:
        """
        Convert predictions to allocation weights.

        Incorporates uncertainty to be more conservative on uncertain predictions.

        Args:
            predicted_revenue: Predicted revenue per hour (24,)
            uncertainty: Uncertainty estimate per hour (24,)
            uncertainty_penalty: How much to penalize uncertain predictions (0-1)

        Returns:
            Allocation weights (24,) summing to 1
        """
        # Adjust predictions by uncertainty
        # Higher uncertainty -> lower effective weight
        adjusted = predicted_revenue * (1 - uncertainty_penalty * np.clip(uncertainty, 0, 1))

        # Ensure non-negative
        adjusted = np.maximum(adjusted, 0)

        # Normalize to weights
        total = adjusted.sum()
        if total > 0:
            weights = adjusted / total
        else:
            weights = np.ones(24) / 24

        return weights


class IntegratedHourlyAllocator:
    """
    Unified hourly budget allocator integrating:
    - Revenue prediction for forward-looking weights
    - Clustering for strategy-based constraints
    - CVXPY optimization for final allocation

    This replaces the standalone HourlyBudgetAllocator with a connected system.
    """

    def __init__(
        self,
        predictor=None,
        clustering_result: Optional[Dict] = None,
        smoothness_factor: float = 0.3,
        min_hourly_budget: float = 0.1,
        max_hourly_ratio: float = 0.15
    ):
        """
        Initialize integrated allocator.

        Args:
            predictor: Trained RevenuePredictor (optional)
            clustering_result: Output from CampaignClusterer (optional)
            smoothness_factor: Weight for smoothness penalty in optimization
            min_hourly_budget: Minimum budget per hour
            max_hourly_ratio: Maximum hourly budget as ratio of daily budget
        """
        self.prediction_integrator = PredictionIntegrator(predictor)
        self.strategy_mapper = ClusterStrategyMapper()

        self.smoothness_factor = smoothness_factor
        self.min_hourly_budget = min_hourly_budget
        self.max_hourly_ratio = max_hourly_ratio

        # Apply clustering if provided
        if clustering_result is not None:
            self.update_clustering(clustering_result)

    def set_predictor(self, predictor):
        """Update the revenue predictor."""
        self.prediction_integrator.set_predictor(predictor)

    def update_clustering(self, clustering_result: Dict):
        """Update with new clustering results."""
        self.strategy_mapper.map_clusters_to_strategies(clustering_result)

    def optimize(
        self,
        daily_budget: float,
        campaign_data: pd.DataFrame,
        campaign_name: str,
        day_of_week: int,
        use_prediction: bool = True,
        use_strategy: bool = True
    ) -> Dict:
        """
        Optimize hourly budget allocation using integrated approach.

        Args:
            daily_budget: Total budget for the day
            campaign_data: Historical campaign data
            campaign_name: Target campaign
            day_of_week: Target day (1=Monday, 7=Sunday)
            use_prediction: Whether to use ML predictions for weights
            use_strategy: Whether to apply cluster-based strategy adjustments

        Returns:
            Optimization results with allocation and metadata
        """
        print(f"\n[Integrated] Optimizing hourly allocation")
        print(f"   Campaign: {campaign_name}")
        print(f"   Budget: ${daily_budget}")
        print(f"   Day: {day_of_week}")
        print(f"   Using prediction: {use_prediction}")
        print(f"   Using strategy: {use_strategy}")

        # Step 1: Get base weights from prediction or history
        if use_prediction:
            predicted_revenue, uncertainty = self.prediction_integrator.predict_hourly_revenue(
                campaign_data, campaign_name, day_of_week
            )
            weights = self.prediction_integrator.get_allocation_weights(
                predicted_revenue, uncertainty
            )
            weight_source = "prediction"
        else:
            # Fallback to historical ROAS
            weights = self._get_historical_weights(campaign_data, campaign_name, day_of_week)
            predicted_revenue = weights * daily_budget  # Rough estimate
            uncertainty = np.ones(24) * 0.5
            weight_source = "historical"

        # Step 2: Apply strategy adjustments from clustering
        strategy_info = None
        if use_strategy and campaign_name in self.strategy_mapper.campaign_strategies:
            weights = self.strategy_mapper.get_hourly_strategy_adjustments(
                campaign_name, weights
            )
            strategy_info = self.strategy_mapper.get_strategy_params(campaign_name)
            print(f"   Strategy: {strategy_info['strategy'].value}")

        # Step 3: Solve optimization problem
        result = self._solve_optimization(daily_budget, weights, strategy_info)

        # Add metadata
        result['campaign_name'] = campaign_name
        result['day_of_week'] = day_of_week
        result['weight_source'] = weight_source
        result['predicted_revenue'] = {h: float(predicted_revenue[h]) for h in range(24)}
        result['prediction_uncertainty'] = {h: float(uncertainty[h]) for h in range(24)}

        if strategy_info:
            result['strategy'] = {
                'type': strategy_info['strategy'].value,
                'description': strategy_info['description'],
                'priority_weight': strategy_info['priority_weight']
            }

        return result

    def _get_historical_weights(
        self,
        campaign_data: pd.DataFrame,
        campaign_name: str,
        day_of_week: int
    ) -> np.ndarray:
        """Get weights from historical ROAS."""

        mask = (
            (campaign_data['campaign_name'] == campaign_name) &
            (campaign_data['day_of_week'] == day_of_week)
        )

        subset = campaign_data[mask]

        roas_values = []
        for hour in range(24):
            hour_data = subset[subset['hour'] == hour]['roas']
            roas_values.append(hour_data.mean() if len(hour_data) > 0 else 0)

        roas_array = np.array(roas_values)

        # Normalize
        total = roas_array.sum()
        if total > 0:
            return roas_array / total
        return np.ones(24) / 24

    def _solve_optimization(
        self,
        daily_budget: float,
        weights: np.ndarray,
        strategy_info: Optional[Dict] = None
    ) -> Dict:
        """Solve CVXPY optimization problem."""

        # Decision variables
        budgets = cp.Variable(24, nonneg=True)

        # Objective: maximize weighted allocation - smoothness penalty
        revenue_objective = weights @ budgets
        smoothness_penalty = cp.sum_squares(cp.diff(budgets))
        objective = cp.Maximize(
            revenue_objective - self.smoothness_factor * smoothness_penalty
        )

        # Base constraints
        min_budget = self.min_hourly_budget
        max_budget = daily_budget * self.max_hourly_ratio

        # Adjust constraints based on strategy
        if strategy_info:
            strategy = strategy_info['strategy']
            if strategy == CampaignStrategy.LOW_PERFORMER:
                # Tighter constraints for low performers
                max_budget = daily_budget * 0.08
            elif strategy == CampaignStrategy.HIGH_PERFORMER:
                # Allow slightly higher allocation
                max_budget = daily_budget * 0.18

        constraints = [
            cp.sum(budgets) == daily_budget,
            budgets >= min_budget,
            budgets <= max_budget
        ]

        # Solve
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception as e:
            print(f"[WARNING] ECOS failed: {e}, trying SCS")
            problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"[WARNING] Solver status: {problem.status}, using fallback")
            return self._fallback_allocation(daily_budget, weights)

        # Extract solution
        allocation = {h: max(0, float(budgets.value[h])) for h in range(24)}

        # Calculate metrics
        total_allocated = sum(allocation.values())
        expected_revenue = sum(allocation[h] * weights[h] * daily_budget for h in range(24))
        smoothness = self._calculate_smoothness(list(allocation.values()))
        peak_hours = self._identify_peak_hours(weights)

        return {
            'daily_budget': daily_budget,
            'hourly_allocation': allocation,
            'total_allocated': round(total_allocated, 2),
            'expected_weighted_score': round(float(revenue_objective.value), 4),
            'smoothness_score': round(smoothness, 6),
            'peak_hours': peak_hours,
            'solver_status': problem.status
        }

    def _fallback_allocation(self, daily_budget: float, weights: np.ndarray) -> Dict:
        """Fallback to proportional allocation."""
        allocation = {h: float(daily_budget * weights[h]) for h in range(24)}
        return {
            'daily_budget': daily_budget,
            'hourly_allocation': allocation,
            'total_allocated': daily_budget,
            'solver_status': 'fallback'
        }

    def _calculate_smoothness(self, budgets: list) -> float:
        """Calculate smoothness score (lower is smoother)."""
        if len(budgets) < 2:
            return 0.0
        diffs = np.diff(budgets)
        return float(np.var(diffs))

    def _identify_peak_hours(self, weights: np.ndarray, top_n: int = 6) -> list:
        """Identify top performing hours."""
        hour_weights = [(h, weights[h]) for h in range(24)]
        sorted_hours = sorted(hour_weights, key=lambda x: x[1], reverse=True)
        return sorted([h for h, _ in sorted_hours[:top_n]])


def create_integrated_pipeline(
    campaign_data: pd.DataFrame,
    predictor=None,
    n_clusters: int = 3
) -> Tuple[IntegratedHourlyAllocator, Dict]:
    """
    Factory function to create a fully integrated allocation pipeline.

    Args:
        campaign_data: Historical campaign data
        predictor: Optional trained RevenuePredictor
        n_clusters: Number of clusters for campaign grouping

    Returns:
        Tuple of (IntegratedHourlyAllocator, clustering_result)
    """
    from algorithms.clustering import CampaignClusterer

    # Run clustering
    clusterer = CampaignClusterer()
    clustering_result = clusterer.cluster(campaign_data, n_clusters)

    # Create integrated allocator
    allocator = IntegratedHourlyAllocator(
        predictor=predictor,
        clustering_result=clustering_result
    )

    return allocator, clustering_result
