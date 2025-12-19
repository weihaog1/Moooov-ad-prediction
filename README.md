# Ad Budget Allocation

A budget optimization system for advertising campaigns using convex optimization, machine learning, and bandit algorithms.

## Features

### Core Tasks
1. **Task 1**: Hourly budget allocation using CVXPY optimization
2. **Task 2**: Multi-campaign budget optimization

### Optional Features
1. **Campaign Clustering**: Groups campaigns by performance patterns
2. **XGBoost Revenue Prediction**: ML model for predicting hourly revenue
3. **Automation Module**: BudgetOptimizer class for automated optimization

### Advanced Features
1. **Integrated Allocator**: Connects prediction, clustering, and allocation modules
2. **Cold Start Handler**: 3-level fallback for new campaigns (similar transfer / category baseline / exploration)
3. **Response Curve Modeling**: Diminishing returns optimization with marginal equilibrium
4. **E&E (Bandit) Algorithms**: Thompson Sampling and UCB for exploration-exploitation balance
5. **Realtime Controller**: Dynamic budget adjustment with +/-20% daily flexibility

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Optimization

```bash
# Basic optimization
python run_optimization.py

# Integrated pipeline (prediction + clustering + allocation)
python run_integrated.py

# Advanced features demo (cold start, response curves, bandit, realtime)
python run_advanced.py
```

### 3. Generate Visualizations

```bash
python visualize.py
```

### 4. Run Tests

```bash
python tests/test_advanced_features.py
```

## Project Structure

```
├── data/
│   └── data_for_ads.xlsx          # Input data
├── algorithms/
│   ├── hourly_allocation.py       # Task 1: Hourly CVXPY optimization
│   ├── multi_campaign.py          # Task 2: Multi-campaign LP
│   ├── clustering.py              # K-Means clustering
│   ├── integrated_allocator.py    # Module integration
│   ├── cold_start.py              # Cold start handler
│   ├── response_curve.py          # Diminishing returns modeling
│   ├── bandit.py                  # Thompson Sampling & UCB
│   └── realtime_controller.py     # Dynamic adjustment
├── models/
│   └── revenue_predictor.py       # XGBoost predictor
├── automation/
│   └── optimizer.py               # BudgetOptimizer orchestrator
├── utils/
│   ├── data_loader.py
│   ├── data_simulator.py          # Synthetic data generation
│   ├── metrics.py
│   └── visualizer.py
├── tests/
│   └── test_advanced_features.py  # Test suite (17 tests)
├── outputs/                       # Results saved here
├── run_optimization.py            # Basic optimization script
├── run_integrated.py              # Integrated pipeline script
├── run_advanced.py                # Advanced features demo
└── visualize.py                   # Visualization script
```

## Key Algorithms

### Basic

**Hourly Allocation (CVXPY)** - Distributes daily budget across 24 hours to maximize weighted revenue with smoothness penalty.

**Multi-Campaign (Linear Programming)** - Allocates total budget across campaigns to maximize overall ROAS.

**Clustering (K-Means)** - Groups campaigns based on performance metrics (ROAS, ACoS, CTR, CVR).

**Prediction (XGBoost)** - Predicts hourly revenue using time features and performance metrics.

### Advanced

**Cold Start** - 3-level fallback strategy:
- Similar Transfer (confidence 0.90): Transfer patterns from similar campaigns
- Category Baseline (confidence 0.70): Use category-specific patterns
- Exploration Uniform (confidence 0.50): Conservative exploration distribution

**Response Curves** - Fits diminishing returns curves (log, sqrt, hill) and optimizes via marginal equilibrium allocation.

**Bandit (E&E)** - Thompson Sampling and UCB for balancing exploration vs exploitation.

**Realtime Controller** - Dynamic adjustment with:
- Pacing control (triggers at >80% spend, <50% time)
- Performance-based reallocation (+/-30% ROAS threshold)
- +/-20% daily budget flexibility

## Usage Examples

```python
from automation.optimizer import BudgetOptimizer

optimizer = BudgetOptimizer('data/data_for_ads.xlsx')

# Basic optimization
results = optimizer.run_all_optimizations()

# Cold start for new campaign
cold_start_result = optimizer.optimize_with_cold_start(
    campaign_name='New_Campaign',
    daily_budget=20.0,
    day_of_week=1,
    product_category='electronics'
)

# Response curve optimization
response_result = optimizer.optimize_with_response_curves(
    total_budget=100.0
)

# Bandit allocation
bandit_result = optimizer.optimize_with_bandit(
    total_budget=100.0,
    method='thompson'  # or 'ucb'
)
```

## Output Files

Results saved to `outputs/`:
- `task1_hourly_allocation_result.json`
- `task1_weekly_allocation_result.json`
- `task2_multi_campaign_result.json`
- `optional1_clustering_result.json`
- `cold_start_allocation_result.json`
- `response_curve_allocation_result.json`
- `bandit_allocation_result.json`
- `revenue_model.joblib`

Visualizations:
- `viz_hourly_allocation.png`
- `viz_weekly_allocation.png`
- `viz_multi_campaign.png`
- `viz_clustering.png`
