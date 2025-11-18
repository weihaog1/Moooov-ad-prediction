"""
Simple data loader - works directly with Excel files.
No database needed!
"""

import pandas as pd
from pathlib import Path


def load_campaign_data(file_path: str = 'data/data_for_ads.xlsx') -> pd.DataFrame:
    """
    Load campaign performance data from Excel.

    Args:
        file_path: Path to Excel file

    Returns:
        DataFrame with all campaign data
    """
    print(f"Loading data from {file_path}...")

    df = pd.read_excel(file_path)

    # Convert date column
    df['date'] = pd.to_datetime(df['date'])

    # Add day_of_week if not present (1=Monday, 7=Sunday)
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date'].dt.dayofweek + 1

    # Fill NaN values with 0
    df['acos'] = df['acos'].fillna(0)
    df['roas'] = df['roas'].fillna(0)

    print(f"[OK] Loaded {len(df)} records")
    print(f"[OK] Campaigns: {df['campaign_name'].unique().tolist()}")
    print(f"[OK] Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def filter_campaign(df: pd.DataFrame, campaign_name: str) -> pd.DataFrame:
    """Filter data for a specific campaign."""
    return df[df['campaign_name'] == campaign_name].copy()


def filter_day_of_week(df: pd.DataFrame, day_of_week: int) -> pd.DataFrame:
    """Filter data for a specific day of week (1=Monday, 7=Sunday)."""
    return df[df['day_of_week'] == day_of_week].copy()


def get_hourly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average metrics for each hour.

    Args:
        df: Campaign performance data

    Returns:
        DataFrame with hourly aggregated metrics
    """
    hourly = df.groupby('hour').agg({
        'impressions': 'mean',
        'clicks': 'mean',
        'spend': 'mean',
        'orders': 'mean',
        'revenue': 'mean',
        'roas': 'mean',
        'acos': 'mean',
        'ctr': 'mean',
        'cvr': 'mean',
        'cpc': 'mean',
    }).reset_index()

    # Ensure all 24 hours are present
    all_hours = pd.DataFrame({'hour': range(24)})
    hourly = all_hours.merge(hourly, on='hour', how='left').fillna(0)

    return hourly
