#!/usr/bin/env python3
"""
Validate Feature Engineering Results
Check data quality, distributions, and potential issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'dataset'


def find_latest_file(directory, pattern):
    """Find the most recent file matching pattern"""
    files = list(Path(directory).glob(pattern))
    return str(max(files, key=os.path.getctime)) if files else None


def validate_dataset():
    """Validate the ML dataset"""
    print("="*70)
    print("FEATURE DATASET VALIDATION")
    print("="*70)
    
    # Load latest ML dataset
    import os
    ml_file = find_latest_file(OUTPUT_DIR, 'ml_dataset_*.csv')
    
    if not ml_file:
        print("❌ No ML dataset found!")
        return
    
    print(f"\nLoading: {ml_file}")
    df = pd.read_csv(ml_file)
    
    print(f"\n{'='*70}")
    print("DATASET OVERVIEW")
    print(f"{'='*70}")
    print(f"Releases: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Target distribution
    print(f"\n{'='*70}")
    print("TARGET VARIABLE DISTRIBUTION")
    print(f"{'='*70}")
    print(f"Degraded releases: {df['is_degraded'].sum()} ({df['is_degraded'].mean()*100:.1f}%)")
    print(f"Stable releases: {(~df['is_degraded'].astype(bool)).sum()} ({(~df['is_degraded'].astype(bool)).mean()*100:.1f}%)")
    print(f"Class balance ratio: 1:{(~df['is_degraded'].astype(bool)).sum() / max(df['is_degraded'].sum(), 1):.2f}")
    
    # Missing values
    print(f"\n{'='*70}")
    print("MISSING VALUES CHECK")
    print(f"{'='*70}")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found")
    else:
        print(f"Columns with missing values:")
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # Infinite values
    print(f"\n{'='*70}")
    print("INFINITE VALUES CHECK")
    print(f"{'='*70}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if len(inf_counts) == 0:
        print("✓ No infinite values found")
    else:
        print(f"Columns with infinite values:")
        for col, count in inf_counts.items():
            print(f"  {col}: {count}")
    
    # Feature statistics
    print(f"\n{'='*70}")
    print("FEATURE CATEGORIES")
    print(f"{'='*70}")
    
    velocity_features = ['weighted_velocity', 'velocity_change_pct', 'issue_count', 
                        'bugs', 'improvements', 'new_features', 'velocity_per_day']
    churn_features = ['total_commits', 'total_churn', 'churn_concentration', 
                     'developer_churn_inequality', 'refactoring_ratio', 
                     'file_hotspots', 'avg_commit_size', 'churn_volatility']
    network_features = ['num_nodes', 'num_edges', 'density', 'avg_clustering',
                       'degree_centralization', 'network_jaccard', 'turnover_rate']
    
    print(f"Velocity features: {len([c for c in velocity_features if c in df.columns])}")
    print(f"Churn features: {len([c for c in churn_features if c in df.columns])}")
    print(f"Network features: {len([c for c in network_features if c in df.columns])}")
    
    # Key feature distributions
    print(f"\n{'='*70}")
    print("KEY FEATURE STATISTICS")
    print(f"{'='*70}")
    
    key_features = {
        'weighted_velocity': 'Weighted Velocity',
        'velocity_change_pct': 'Velocity Change %',
        'total_churn': 'Total Churn',
        'churn_concentration': 'Churn Concentration (Gini)',
        'density': 'Network Density',
        'degree_centralization': 'Degree Centralization',
        'turnover_rate': 'Developer Turnover Rate',
    }
    
    for col, label in key_features.items():
        if col in df.columns:
            print(f"\n{label}:")
            print(f"  Mean: {df[col].mean():.3f}")
            print(f"  Std: {df[col].std():.3f}")
            print(f"  Min: {df[col].min():.3f}")
            print(f"  25%: {df[col].quantile(0.25):.3f}")
            print(f"  50%: {df[col].quantile(0.50):.3f}")
            print(f"  75%: {df[col].quantile(0.75):.3f}")
            print(f"  Max: {df[col].max():.3f}")
    
    # Correlation with target
    print(f"\n{'='*70}")
    print("TOP CORRELATIONS WITH TARGET (is_degraded)")
    print(f"{'='*70}")
    
    correlations = df[numeric_cols].corrwith(df['is_degraded']).sort_values(ascending=False)
    correlations = correlations[correlations.index != 'is_degraded']
    
    print("\nTop 10 positive correlations:")
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"  {i:2d}. {feature:40s} {corr:+.3f}")
    
    print("\nTop 10 negative correlations:")
    for i, (feature, corr) in enumerate(correlations.tail(10).items(), 1):
        print(f"  {i:2d}. {feature:40s} {corr:+.3f}")
    
    # Data quality warnings
    print(f"\n{'='*70}")
    print("DATA QUALITY WARNINGS")
    print(f"{'='*70}")
    
    warnings_found = []
    
    # Check for zero variance features
    zero_var = df[numeric_cols].std() == 0
    if zero_var.sum() > 0:
        warnings_found.append(f"⚠ {zero_var.sum()} features have zero variance")
        for col in zero_var[zero_var].index:
            print(f"  - {col}")
    
    # Check for highly correlated features (>0.95)
    corr_matrix = df[numeric_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(col, row, corr_matrix.loc[row, col]) 
                 for col in upper_tri.columns 
                 for row in upper_tri.index 
                 if upper_tri.loc[row, col] > 0.95]
    
    if len(high_corr) > 0:
        warnings_found.append(f"⚠ {len(high_corr)} pairs of features are highly correlated (>0.95)")
        for col1, col2, corr in high_corr[:5]:  # Show first 5
            print(f"  - {col1} <-> {col2}: {corr:.3f}")
    
    # Check for imbalanced target
    if df['is_degraded'].mean() < 0.2 or df['is_degraded'].mean() > 0.8:
        warnings_found.append(f"⚠ Target variable is imbalanced ({df['is_degraded'].mean()*100:.1f}% degraded)")
    
    if len(warnings_found) == 0:
        print("✓ No major data quality issues detected")
    
    # Save summary
    print(f"\n{'='*70}")
    print("SUMMARY SAVED")
    print(f"{'='*70}")
    
    summary = {
        'total_releases': len(df),
        'total_features': len(df.columns),
        'degraded_releases': int(df['is_degraded'].sum()),
        'stable_releases': int((~df['is_degraded'].astype(bool)).sum()),
        'missing_values': int(missing.sum()),
        'has_infinite': len(inf_counts) > 0,
        'zero_variance_features': int(zero_var.sum()),
        'highly_correlated_pairs': len(high_corr),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = f'{OUTPUT_DIR}/validation_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Validation summary saved to {summary_file}")
    
    return df


if __name__ == '__main__':
    import os
    df = validate_dataset()
    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")
