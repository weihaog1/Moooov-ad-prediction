"""
Cold Start Handler for New Campaign Initialization

Provides allocation weights for new campaigns with no historical data
using a 3-level fallback strategy:
1. Similar campaign transfer
2. Category baseline
3. Exploration-weighted uniform
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class ColdStartStrategy(Enum):
    """Strategy used for cold start initialization."""
    SIMILAR_TRANSFER = "similar_transfer"
    CATEGORY_BASELINE = "category_baseline"
    EXPLORATION_UNIFORM = "exploration_uniform"


class ColdStartHandler:
    """
    Handles budget allocation for new campaigns with no historical data.

    Uses a 3-level fallback strategy:
    1. Transfer weights from similar campaigns (same category/audience)
    2. Use category-specific hourly baselines
    3. Exploration-weighted uniform distribution
    """

    # Category-specific hourly patterns (normalized weights for 24 hours)
    CATEGORY_BASELINES = {
        'electronics': {
            'peak_hours': [12, 13, 14, 20, 21, 22],
            'off_peak_hours': [0, 1, 2, 3, 4, 5],
            'description': 'Double peak: lunch and evening'
        },
        'fashion': {
            'peak_hours': [18, 19, 20, 21, 22],
            'off_peak_hours': [0, 1, 2, 3, 4, 5, 6],
            'description': 'Evening heavy'
        },
        'home': {
            'peak_hours': [9, 10, 11, 19, 20],
            'off_peak_hours': [0, 1, 2, 3, 4],
            'description': 'Morning and evening'
        },
        'beauty': {
            'peak_hours': [10, 11, 12, 19, 20, 21],
            'off_peak_hours': [0, 1, 2, 3, 4, 5],
            'description': 'Late morning and evening'
        },
        'sports': {
            'peak_hours': [6, 7, 8, 17, 18, 19],
            'off_peak_hours': [0, 1, 2, 3, 23],
            'description': 'Early morning and after work'
        },
        'default': {
            'peak_hours': [10, 11, 12, 19, 20, 21],
            'off_peak_hours': [0, 1, 2, 3, 4, 5],
            'description': 'Standard e-commerce pattern'
        }
    }

    # Day of week adjustments
    DOW_ADJUSTMENTS = {
        1: 0.9,   # Monday
        2: 0.95,  # Tuesday
        3: 1.0,   # Wednesday
        4: 1.0,   # Thursday
        5: 1.1,   # Friday
        6: 1.2,   # Saturday
        7: 1.15   # Sunday
    }

    def __init__(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        similarity_threshold: float = 0.7,
        min_similar_campaigns: int = 2
    ):
        """
        Initialize cold start handler.

        Args:
            historical_data: Historical campaign data for similarity matching
            similarity_threshold: Minimum similarity score to consider campaigns similar
            min_similar_campaigns: Minimum similar campaigns needed for transfer
        """
        self.historical_data = historical_data
        self.similarity_threshold = similarity_threshold
        self.min_similar_campaigns = min_similar_campaigns

        # Cache for campaign features (for similarity computation)
        self._campaign_features: Optional[pd.DataFrame] = None
        self._feature_scaler: Optional[StandardScaler] = None

        if historical_data is not None:
            self._precompute_campaign_features()

    def _precompute_campaign_features(self):
        """Precompute campaign features for similarity matching."""
        if self.historical_data is None:
            return

        df = self.historical_data

        # Aggregate campaign-level features
        features = df.groupby('campaign_name').agg({
            'impressions': 'mean',
            'clicks': 'mean',
            'spend': 'mean',
            'revenue': 'mean',
            'roas': 'mean',
            'ctr': 'mean',
            'cvr': 'mean'
        }).reset_index()

        # Add category and audience if available
        if 'product_category' in df.columns:
            cat_map = df.groupby('campaign_name')['product_category'].first()
            features['product_category'] = features['campaign_name'].map(cat_map)

        if 'audience_type' in df.columns:
            aud_map = df.groupby('campaign_name')['audience_type'].first()
            features['audience_type'] = features['campaign_name'].map(aud_map)

        self._campaign_features = features

        # Scale numeric features for similarity computation
        numeric_cols = ['impressions', 'clicks', 'spend', 'revenue', 'roas', 'ctr', 'cvr']
        available_cols = [c for c in numeric_cols if c in features.columns]

        if available_cols:
            self._feature_scaler = StandardScaler()
            self._feature_scaler.fit(features[available_cols].fillna(0))

    def get_initial_weights(
        self,
        campaign_name: str,
        day_of_week: int = 1,
        product_category: Optional[str] = None,
        audience_type: Optional[str] = None
    ) -> Dict:
        """
        Get initial allocation weights for a new campaign.

        Args:
            campaign_name: Name of the new campaign
            day_of_week: Target day of week (1=Monday, 7=Sunday)
            product_category: Product category if known
            audience_type: Audience type if known

        Returns:
            {
                'weights': np.ndarray of shape (24,),
                'strategy': ColdStartStrategy,
                'confidence': float (0-1),
                'exploration_ratio': float,
                'details': dict with strategy-specific info
            }
        """
        # Try Level 1: Similar campaign transfer
        if self.historical_data is not None:
            similar_result = self._try_similar_transfer(
                product_category, audience_type, day_of_week
            )
            if similar_result is not None:
                return similar_result

        # Try Level 2: Category baseline
        if product_category is not None:
            baseline_result = self._get_category_baseline_result(
                product_category, day_of_week
            )
            return baseline_result

        # Level 3: Exploration-weighted uniform
        return self._get_exploration_uniform_result(day_of_week)

    def _try_similar_transfer(
        self,
        product_category: Optional[str],
        audience_type: Optional[str],
        day_of_week: int
    ) -> Optional[Dict]:
        """
        Try to transfer weights from similar campaigns.

        Returns None if not enough similar campaigns found.
        """
        similar_campaigns = self.find_similar_campaigns(
            product_category, audience_type
        )

        if len(similar_campaigns) < self.min_similar_campaigns:
            return None

        weights = self._transfer_weights(similar_campaigns, day_of_week)
        similarity_scores = [s for _, s in similar_campaigns]
        avg_similarity = np.mean(similarity_scores)

        return {
            'weights': weights,
            'strategy': ColdStartStrategy.SIMILAR_TRANSFER,
            'confidence': min(0.9, avg_similarity),
            'exploration_ratio': 0.15,  # 15% exploration for known similar patterns
            'details': {
                'similar_campaigns': [c for c, _ in similar_campaigns],
                'similarity_scores': similarity_scores,
                'avg_similarity': avg_similarity
            }
        }

    def find_similar_campaigns(
        self,
        product_category: Optional[str],
        audience_type: Optional[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar historical campaigns.

        Args:
            product_category: Target category
            audience_type: Target audience type
            top_k: Maximum number of similar campaigns to return

        Returns:
            List of (campaign_name, similarity_score) tuples
        """
        if self._campaign_features is None:
            return []

        features = self._campaign_features
        candidates = features.copy()

        # Filter by category if specified
        if product_category and 'product_category' in candidates.columns:
            category_match = candidates['product_category'] == product_category
            if category_match.sum() >= self.min_similar_campaigns:
                candidates = candidates[category_match]

        # Filter by audience type if specified
        if audience_type and 'audience_type' in candidates.columns:
            audience_match = candidates['audience_type'] == audience_type
            if audience_match.sum() >= self.min_similar_campaigns:
                candidates = candidates[audience_match]

        if len(candidates) == 0:
            return []

        # Compute similarity based on performance metrics
        numeric_cols = ['roas', 'ctr', 'cvr', 'spend', 'revenue']
        available_cols = [c for c in numeric_cols if c in candidates.columns]

        if not available_cols:
            # If no metrics, just return top campaigns by revenue
            top_campaigns = candidates.nlargest(top_k, 'revenue' if 'revenue' in candidates.columns else candidates.columns[1])
            return [(row['campaign_name'], 0.5) for _, row in top_campaigns.iterrows()]

        # For new campaign, use average metrics as reference
        reference = candidates[available_cols].mean().values.reshape(1, -1)
        candidate_metrics = candidates[available_cols].fillna(0).values

        # Compute cosine similarity
        similarities = cosine_similarity(reference, candidate_metrics)[0]

        # Create ranked list
        results = []
        for idx, (_, row) in enumerate(candidates.iterrows()):
            if similarities[idx] >= self.similarity_threshold:
                results.append((row['campaign_name'], float(similarities[idx])))

        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _transfer_weights(
        self,
        similar_campaigns: List[Tuple[str, float]],
        day_of_week: int
    ) -> np.ndarray:
        """
        Transfer and aggregate hourly weights from similar campaigns.

        Weights are aggregated using similarity-weighted average.
        """
        if self.historical_data is None:
            return np.ones(24) / 24

        weights_sum = np.zeros(24)
        similarity_sum = 0

        for campaign_name, similarity in similar_campaigns:
            # Get hourly performance for this campaign on this day
            mask = (
                (self.historical_data['campaign_name'] == campaign_name) &
                (self.historical_data['day_of_week'] == day_of_week)
            )
            campaign_data = self.historical_data[mask]

            if len(campaign_data) == 0:
                continue

            # Compute hourly weights based on ROAS
            hourly_roas = campaign_data.groupby('hour')['roas'].mean()

            # Fill missing hours with 0
            hourly_weights = np.zeros(24)
            for hour, roas in hourly_roas.items():
                hourly_weights[int(hour)] = roas

            # Normalize
            if hourly_weights.sum() > 0:
                hourly_weights = hourly_weights / hourly_weights.sum()

            # Add to weighted sum
            weights_sum += hourly_weights * similarity
            similarity_sum += similarity

        if similarity_sum > 0:
            final_weights = weights_sum / similarity_sum
        else:
            final_weights = np.ones(24) / 24

        # Ensure normalization
        if final_weights.sum() > 0:
            final_weights = final_weights / final_weights.sum()

        return final_weights

    def _get_category_baseline_result(
        self,
        product_category: str,
        day_of_week: int
    ) -> Dict:
        """Get category baseline with day-of-week adjustment."""
        weights = self.get_category_baseline(product_category, day_of_week)

        category_info = self.CATEGORY_BASELINES.get(
            product_category,
            self.CATEGORY_BASELINES['default']
        )

        return {
            'weights': weights,
            'strategy': ColdStartStrategy.CATEGORY_BASELINE,
            'confidence': 0.5,
            'exploration_ratio': 0.25,  # 25% exploration for category-based
            'details': {
                'category': product_category,
                'peak_hours': category_info['peak_hours'],
                'description': category_info['description'],
                'dow_adjustment': self.DOW_ADJUSTMENTS.get(day_of_week, 1.0)
            }
        }

    def get_category_baseline(
        self,
        product_category: str,
        day_of_week: int = 1
    ) -> np.ndarray:
        """
        Get category-specific hourly baseline weights.

        Args:
            product_category: Product category
            day_of_week: Day of week for adjustment

        Returns:
            Normalized hourly weights (24,)
        """
        category = product_category.lower() if product_category else 'default'
        if category not in self.CATEGORY_BASELINES:
            category = 'default'

        config = self.CATEGORY_BASELINES[category]
        weights = np.ones(24) * 0.5  # Base weight

        # Boost peak hours
        for hour in config['peak_hours']:
            weights[hour] = 1.5

        # Reduce off-peak hours
        for hour in config['off_peak_hours']:
            weights[hour] = 0.2

        # Apply day-of-week adjustment
        dow_mult = self.DOW_ADJUSTMENTS.get(day_of_week, 1.0)
        weights = weights * dow_mult

        # Normalize
        weights = weights / weights.sum()

        return weights

    def _get_exploration_uniform_result(self, day_of_week: int) -> Dict:
        """Get exploration-weighted uniform distribution."""
        weights = self.get_exploration_weighted_uniform(day_of_week)

        return {
            'weights': weights,
            'strategy': ColdStartStrategy.EXPLORATION_UNIFORM,
            'confidence': 0.2,
            'exploration_ratio': 0.40,  # 40% exploration for unknown campaigns
            'details': {
                'strategy': 'Uniform with exploration bonus on typical peak hours',
                'dow_adjustment': self.DOW_ADJUSTMENTS.get(day_of_week, 1.0)
            }
        }

    def get_exploration_weighted_uniform(
        self,
        day_of_week: int = 1,
        exploration_bonus: float = 0.3
    ) -> np.ndarray:
        """
        Get uniform distribution with exploration bonus on typical peak hours.

        Args:
            day_of_week: Day of week
            exploration_bonus: Extra weight for typical peak hours (0-1)

        Returns:
            Normalized hourly weights (24,)
        """
        # Start with uniform
        weights = np.ones(24)

        # Add slight bonus to typical peak hours (evening 18-22)
        typical_peaks = [18, 19, 20, 21, 22]
        for hour in typical_peaks:
            weights[hour] += exploration_bonus

        # Apply day-of-week adjustment
        dow_mult = self.DOW_ADJUSTMENTS.get(day_of_week, 1.0)
        weights = weights * dow_mult

        # Normalize
        weights = weights / weights.sum()

        return weights

    def update_with_data(self, new_data: pd.DataFrame):
        """
        Update handler with new historical data.

        Call this when new campaign data becomes available
        to improve similarity matching.
        """
        if self.historical_data is None:
            self.historical_data = new_data
        else:
            self.historical_data = pd.concat([self.historical_data, new_data])

        self._precompute_campaign_features()

    def get_recommended_exploration_ratio(self, confidence: float) -> float:
        """
        Get recommended exploration ratio based on confidence.

        Lower confidence = more exploration needed.
        """
        # Linear mapping: confidence 1.0 -> 10% exploration, 0.0 -> 50% exploration
        return 0.5 - (confidence * 0.4)
