"""
Optional Feature 1: Campaign Clustering Analysis

Groups campaigns by performance characteristics using K-Means.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict


class CampaignClusterer:
    """
    Performs clustering analysis on campaigns to identify patterns.
    """

    def cluster(
        self,
        campaign_data: pd.DataFrame,
        n_clusters: int = 3,
        method: str = "kmeans"
    ) -> Dict:
        """
        Perform clustering analysis.

        Args:
            campaign_data: Full campaign performance data
            n_clusters: Number of clusters to create
            method: Clustering method (only "kmeans" implemented)

        Returns:
            Clustering results with statistics and characteristics
        """
        print(f"\nPerforming clustering analysis ({method}, k={n_clusters})")

        # Aggregate campaign metrics
        campaign_metrics = campaign_data.groupby('campaign_name').agg({
            'impressions': ['mean', 'std'],
            'clicks': ['mean', 'std'],
            'spend': ['mean', 'sum'],
            'orders': ['mean', 'sum'],
            'revenue': ['mean', 'sum'],
            'ctr': 'mean',
            'cvr': 'mean',
            'cpc': 'mean',
            'acos': 'mean',
            'roas': 'mean',
        })

        campaign_metrics.columns = ['_'.join(col).strip('_') for col in campaign_metrics.columns]
        campaign_metrics = campaign_metrics.reset_index()

        # Select features
        feature_cols = [
            'roas_mean', 'acos_mean', 'ctr_mean', 'cvr_mean', 'cpc_mean',
            'revenue_mean', 'spend_mean'
        ]

        features = campaign_metrics[feature_cols].fillna(0).values

        # Normalize
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Cluster
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_features)

        # PCA for visualization
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(scaled_features)

        # Calculate statistics
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_campaigns = campaign_metrics[mask]

            cluster_stats[cluster_id] = {
                'size': int(mask.sum()),
                'campaigns': cluster_campaigns['campaign_name'].tolist(),
                'avg_roas': float(cluster_campaigns['roas_mean'].mean()),
                'avg_acos': float(cluster_campaigns['acos_mean'].mean()),
                'avg_revenue': float(cluster_campaigns['revenue_mean'].mean()),
                'total_revenue': float(cluster_campaigns['revenue_sum'].sum())
            }

        # Characterize clusters
        characteristics = {}
        for cluster_id in range(n_clusters):
            stats = cluster_stats[cluster_id]
            roas = stats['avg_roas']
            acos = stats['avg_acos']

            if roas > 4.0 and acos < 0.25:
                char = "High Performance - Excellent ROAS & ACoS"
            elif roas > 3.0 and acos < 0.30:
                char = "Good Performance - Strong ROAS"
            elif roas < 2.0 or acos > 0.35:
                char = "Low Performance - Needs Optimization"
            else:
                char = "Moderate Performance - Standard Metrics"

            characteristics[cluster_id] = char

        # Quality metrics
        quality = {
            'silhouette_score': float(silhouette_score(scaled_features, labels)),
            'davies_bouldin_score': float(davies_bouldin_score(scaled_features, labels))
        }

        # Campaign-cluster mapping
        campaigns_list = []
        for idx, row in campaign_metrics.iterrows():
            campaigns_list.append({
                'campaign_name': row['campaign_name'],
                'cluster_id': int(labels[idx]),
                'roas': float(row['roas_mean']),
                'acos': float(row['acos_mean']),
                'pca_x': float(pca_coords[idx, 0]),
                'pca_y': float(pca_coords[idx, 1])
            })

        result = {
            'method': method,
            'n_clusters': n_clusters,
            'campaigns': campaigns_list,
            'cluster_statistics': cluster_stats,
            'cluster_characteristics': characteristics,
            'quality_metrics': quality,
            'features_used': feature_cols
        }

        print(f"[OK] Identified {n_clusters} clusters")
        print(f"[OK] Silhouette Score: {quality['silhouette_score']:.4f}")

        return result
