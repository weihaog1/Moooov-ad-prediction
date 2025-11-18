# Ad Budget Allocation - Simplified Version

A clean, database-free implementation of the ad budget optimization system for Amazon advertising campaigns.

## Features

Implements 100% of assignment requirements:

### Core Tasks
1. **Task 1**: Hourly budget allocation using CVXPY optimization
2. **Task 2**: Multi-campaign budget optimization

### Optional Features
1. **Campaign Clustering**: Groups campaigns by performance patterns
2. **XGBoost Revenue Prediction**: ML model for predicting hourly revenue
3. **Automation Module**: BudgetOptimizer class for automated optimization

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Optimization

```bash
python run_optimization.py
```

This automatically runs all optimizations and saves results to `outputs/` folder.

### 3. Generate Visualizations

```bash
python visualize.py
```

Creates 4 high-quality charts in `outputs/` folder.

## Project Structure

```
simple/
├── data/
│   └── data_for_ads.xlsx          # Test data
├── algorithms/
│   ├── hourly_allocation.py       # Task 1
│   ├── multi_campaign.py          # Task 2
│   └── clustering.py              # Optional 1
├── models/
│   └── revenue_predictor.py       # Optional 2
├── automation/
│   └── optimizer.py               # Optional 3
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   └── visualizer.py
├── outputs/                       # Results saved here
├── run_optimization.py            # Main script
├── visualize.py                   # Visualization script
└── requirements.txt
```

## Key Algorithms

### 1. Hourly Allocation (CVXPY)

Distributes daily budget across 24 hours to maximize weighted revenue while maintaining smooth allocation.

### 2. Multi-Campaign (Linear Programming)

Allocates total budget across campaigns to maximize overall ROAS.

### 3. Clustering (K-Means)

Groups campaigns based on performance metrics (ROAS, ACoS, CTR, CVR, etc.).

### 4. Prediction (XGBoost)

Predicts hourly revenue using time features and performance metrics.

## Using the Automation Module

```python
from automation.optimizer import BudgetOptimizer

# Initialize
optimizer = BudgetOptimizer('data/data_for_ads.xlsx')

# Run everything
results = optimizer.run_all_optimizations()

# Or run individual tasks
hourly_result = optimizer.optimize_hourly(
    campaign_name='Campaign_A',
    daily_budget=20.0,
    day_of_week=1
)

multi_result = optimizer.optimize_multi_campaign(
    total_budget=100.0
)
```

## Results

All results are saved as JSON files in the `outputs/` folder:
- `task1_hourly_allocation_result.json`
- `task1_weekly_allocation_result.json`
- `task2_multi_campaign_result.json`
- `optional1_clustering_result.json`
- `revenue_model.joblib` (trained ML model)

Visualizations are saved as PNG files:
- `viz_hourly_allocation.png`
- `viz_weekly_allocation.png`
- `viz_multi_campaign.png`
- `viz_clustering.png`

## Requirements

- Python 3.8+
- 11 lightweight dependencies (see requirements.txt)
- No database or Docker required

## License

MIT
