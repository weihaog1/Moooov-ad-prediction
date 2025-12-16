"""
Budget Allocation Algorithms

This package contains optimization algorithms for ad budget allocation:
- hourly_allocation: Single campaign hourly budget distribution
- multi_campaign: Cross-campaign budget optimization
- clustering: Campaign performance clustering
- integrated_allocator: Connected pipeline integrating prediction + clustering + allocation
"""

from algorithms.hourly_allocation import HourlyBudgetAllocator
from algorithms.multi_campaign import MultiCampaignOptimizer
from algorithms.clustering import CampaignClusterer
from algorithms.integrated_allocator import (
    IntegratedHourlyAllocator,
    ClusterStrategyMapper,
    PredictionIntegrator,
    CampaignStrategy,
    create_integrated_pipeline
)

__all__ = [
    'HourlyBudgetAllocator',
    'MultiCampaignOptimizer',
    'CampaignClusterer',
    'IntegratedHourlyAllocator',
    'ClusterStrategyMapper',
    'PredictionIntegrator',
    'CampaignStrategy',
    'create_integrated_pipeline'
]
