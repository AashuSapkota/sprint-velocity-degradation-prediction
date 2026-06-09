#!/usr/bin/env python3
"""
Data Preparation for Machine Learning
Handles feature scaling, time-series cross-validation, and class balancing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Will install these if needed
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
except ImportError:
    print("⚠️  Required libraries not installed. Please run:")
    print("   pip install scikit-learn imbalanced-learn")
    import sys
    sys.exit(1)

OUTPUT_DIR = 'dataset'
PREPARED_DIR = 'dataset/prepared'
os.makedirs(PREPARED_DIR, exist_ok=True)


def find_latest_file(directory, pattern):
    """Find the most recent file matching pattern"""
    files = list(Path(directory).glob(pattern))
    return str(max(files, key=os.path.getctime)) if files else None


def load_ml_dataset():
    """Load the latest ML dataset"""
    print("="*70)
    print("LOADING ML DATASET")
    print("="*70)
    
    ml_file = find_latest_file(OUTPUT_DIR, 'ml_dataset_*.csv')
    
    if not ml_file:
        raise FileNotFoundError("No ML dataset found. Run feature_engineering.py first.")
    
    print(f"\nLoading: {ml_file}")
    df = pd.read_csv(ml_file)
    
    print(f"✓ Loaded {len(df)} releases with {len(df.columns)} columns")
    
    return df


def prepare_features_and_target(df):
    """
    Separate features from target and metadata
    Returns: X (features), y (target), metadata, feature_names
    """
    print("\n" + "="*70)
    print("PREPARING FEATURES AND TARGET")
    print("="*70)
    
    # Identify columns
    metadata_cols = ['release', 'start_date', 'end_date']
    target_col = 'is_degraded'
    
    # Columns to exclude from features (dates, metadata, intermediate calculations)
    exclude_cols = metadata_cols + [target_col, 'velocity_change', 'prev_weighted_velocity']
    
    # Also exclude any remaining date columns
    for col in df.columns:
        if 'date' in col.lower() or df[col].dtype == 'object':
            if col not in exclude_cols and col not in metadata_cols:
                # Check if it's actually a date string
                try:
                    pd.to_datetime(df[col].iloc[0])
                    exclude_cols.append(col)
                except:
                    pass
    
    # Feature columns (only numeric)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Separate data
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    metadata = df[metadata_cols].copy()
    
    # Ensure all features are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"\n⚠️  Removing non-numeric columns: {non_numeric}")
        X = X.select_dtypes(include=[np.number])
        feature_cols = X.columns.tolist()
    
    # Handle missing values (fill with median)
    if X.isnull().sum().sum() > 0:
        print("\n⚠️  Handling missing values...")
        missing_cols = X.columns[X.isnull().any()].tolist()
        for col in missing_cols:
            median_val = X[col].median()
            n_missing = X[col].isnull().sum()
            X[col].fillna(median_val, inplace=True)
            print(f"  - Filled {n_missing} missing values in '{col}' with median: {median_val:.3f}")
    
    print(f"\n✓ Features: {X.shape[1]} columns")
    print(f"✓ Samples: {X.shape[0]} releases")
    print(f"✓ Target distribution: {y.sum()} degraded ({y.mean()*100:.1f}%), {(~y.astype(bool)).sum()} stable ({(~y.astype(bool)).mean()*100:.1f}%)")
    
    # Print feature categories
    print(f"\nFeature categories:")
    velocity_features = [c for c in feature_cols if any(x in c.lower() for x in ['velocity', 'issue', 'bug', 'improvement', 'feature', 'task'])]
    churn_features = [c for c in feature_cols if any(x in c.lower() for x in ['churn', 'commit', 'refactor', 'hotspot', 'developer'])]
    network_features = [c for c in feature_cols if any(x in c.lower() for x in ['node', 'edge', 'density', 'cluster', 'central', 'diameter', 'component', 'jaccard', 'turnover'])]
    
    print(f"  - Velocity-related: {len(velocity_features)}")
    print(f"  - Churn-related: {len(churn_features)}")
    print(f"  - Network-related: {len(network_features)}")
    print(f"  - Other: {len(feature_cols) - len(velocity_features) - len(churn_features) - len(network_features)}")
    
    return X, y, metadata, feature_cols


def scale_features(X_train, X_val, X_test, method='standard'):
    """
    Scale features using specified method
    
    Parameters:
    - method: 'standard' (StandardScaler) or 'robust' (RobustScaler)
    
    Returns: scaled arrays and fitted scaler
    """
    print("\n" + "="*70)
    print(f"FEATURE SCALING ({method.upper()})")
    print("="*70)
    
    if method == 'standard':
        scaler = StandardScaler()
        print("\nUsing StandardScaler (z-score normalization)")
        print("  Formula: (X - mean) / std")
    elif method == 'robust':
        scaler = RobustScaler()
        print("\nUsing RobustScaler (robust to outliers)")
        print("  Formula: (X - median) / IQR")
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit on training data only
    scaler.fit(X_train)
    
    # Transform all sets
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"\n✓ Scaled {X_train_scaled.shape[1]} features")
    print(f"  Training set: {X_train_scaled.shape[0]} samples")
    print(f"  Validation set: {X_val_scaled.shape[0]} samples")
    print(f"  Test set: {X_test_scaled.shape[0]} samples")
    
    # Print scaling statistics for a few key features
    print(f"\nExample scaling (first 5 features):")
    for i, col in enumerate(X_train.columns[:5]):
        original_mean = X_train[col].mean()
        scaled_mean = X_train_scaled[col].mean()
        scaled_std = X_train_scaled[col].std()
        print(f"  {col[:40]:40s} | Original mean: {original_mean:8.2f} | Scaled mean: {scaled_mean:6.3f} ± {scaled_std:.3f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def create_time_series_splits(X, y, metadata, n_splits=5, test_size=0.15, val_size=0.15):
    """
    Create time-series cross-validation splits
    
    Parameters:
    - n_splits: Number of CV folds
    - test_size: Proportion for final test set (most recent data)
    - val_size: Proportion for validation set
    
    Returns: train/val/test splits and CV fold indices
    """
    print("\n" + "="*70)
    print("TIME-SERIES CROSS-VALIDATION SPLITS")
    print("="*70)
    
    n_samples = len(X)
    
    # Calculate split sizes
    test_samples = int(n_samples * test_size)
    val_samples = int(n_samples * val_size)
    train_samples = n_samples - test_samples - val_samples
    
    print(f"\nTotal samples: {n_samples}")
    print(f"  Training: {train_samples} ({train_samples/n_samples*100:.1f}%)")
    print(f"  Validation: {val_samples} ({val_samples/n_samples*100:.1f}%)")
    print(f"  Test: {test_samples} ({test_samples/n_samples*100:.1f}%)")
    
    # Split indices (temporal order preserved)
    train_idx = range(0, train_samples)
    val_idx = range(train_samples, train_samples + val_samples)
    test_idx = range(train_samples + val_samples, n_samples)
    
    # Create splits
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_test = X.iloc[test_idx].copy()
    
    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()
    y_test = y.iloc[test_idx].copy()
    
    metadata_train = metadata.iloc[train_idx].copy()
    metadata_val = metadata.iloc[val_idx].copy()
    metadata_test = metadata.iloc[test_idx].copy()
    
    # Print date ranges
    print(f"\nTemporal split details:")
    print(f"  Training:   {metadata_train['start_date'].min()} to {metadata_train['end_date'].max()}")
    print(f"  Validation: {metadata_val['start_date'].min()} to {metadata_val['end_date'].max()}")
    print(f"  Test:       {metadata_test['start_date'].min()} to {metadata_test['end_date'].max()}")
    
    # Check class distributions
    print(f"\nClass distribution per split:")
    print(f"  Training:   {y_train.sum()}/{len(y_train)} degraded ({y_train.mean()*100:.1f}%)")
    print(f"  Validation: {y_val.sum()}/{len(y_val)} degraded ({y_val.mean()*100:.1f}%)")
    print(f"  Test:       {y_test.sum()}/{len(y_test)} degraded ({y_test.mean()*100:.1f}%)")
    
    # Create time-series CV folds for training set
    print(f"\n" + "="*70)
    print(f"CREATING {n_splits}-FOLD TIME-SERIES CV")
    print("="*70)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_folds = []
    
    print(f"\nExpanding window approach:")
    for fold_idx, (train_cv_idx, val_cv_idx) in enumerate(tscv.split(X_train), 1):
        cv_folds.append((train_cv_idx, val_cv_idx))
        
        # Calculate temporal range
        train_cv_releases = metadata_train.iloc[train_cv_idx]['release'].tolist()
        val_cv_releases = metadata_train.iloc[val_cv_idx]['release'].tolist()
        
        print(f"  Fold {fold_idx}: Train={len(train_cv_idx):3d} samples | Val={len(val_cv_idx):3d} samples")
        print(f"           Train releases: {train_cv_releases[0]} to {train_cv_releases[-1]}")
        print(f"           Val releases:   {val_cv_releases[0]} to {val_cv_releases[-1]}")
    
    splits = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'metadata_train': metadata_train,
        'metadata_val': metadata_val,
        'metadata_test': metadata_test,
        'cv_folds': cv_folds,
    }
    
    return splits


def handle_class_imbalance(X_train, y_train, method='none'):
    """
    Handle class imbalance using various techniques
    
    Parameters:
    - method: 'none', 'oversample' (SMOTE), 'undersample', 'combine' (SMOTE+Tomek)
    
    Returns: resampled X_train, y_train
    """
    print("\n" + "="*70)
    print(f"CLASS IMBALANCE HANDLING ({method.upper()})")
    print("="*70)
    
    original_distribution = y_train.value_counts().to_dict()
    print(f"\nOriginal distribution:")
    print(f"  Class 0 (stable):   {original_distribution.get(0, 0)} samples")
    print(f"  Class 1 (degraded): {original_distribution.get(1, 0)} samples")
    print(f"  Ratio: 1:{original_distribution.get(0, 0)/max(original_distribution.get(1, 1), 1):.2f}")
    
    if method == 'none':
        print("\n✓ No resampling applied")
        return X_train, y_train
    
    elif method == 'oversample':
        print("\nApplying SMOTE (Synthetic Minority Over-sampling)")
        sampler = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
        
    elif method == 'undersample':
        print("\nApplying Random Under-sampling")
        sampler = RandomUnderSampler(random_state=42)
        
    elif method == 'combine':
        print("\nApplying SMOTE + Tomek Links (combined)")
        sampler = SMOTETomek(random_state=42)
        
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Resample
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    # Convert back to DataFrame/Series
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)
    
    new_distribution = y_resampled.value_counts().to_dict()
    print(f"\nResampled distribution:")
    print(f"  Class 0 (stable):   {new_distribution.get(0, 0)} samples ({new_distribution.get(0, 0)-original_distribution.get(0, 0):+d})")
    print(f"  Class 1 (degraded): {new_distribution.get(1, 0)} samples ({new_distribution.get(1, 0)-original_distribution.get(1, 0):+d})")
    print(f"  Ratio: 1:{new_distribution.get(0, 0)/max(new_distribution.get(1, 1), 1):.2f}")
    
    return X_resampled, y_resampled


def save_prepared_data(splits, scaler, feature_names, config):
    """Save all prepared datasets and artifacts"""
    print("\n" + "="*70)
    print("SAVING PREPARED DATA")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save train/val/test splits
    datasets = {
        'X_train': splits['X_train'],
        'X_val': splits['X_val'],
        'X_test': splits['X_test'],
        'y_train': splits['y_train'],
        'y_val': splits['y_val'],
        'y_test': splits['y_test'],
    }
    
    saved_files = []
    for name, data in datasets.items():
        filename = f'{PREPARED_DIR}/{name}_{timestamp}.csv'
        data.to_csv(filename, index=False)
        saved_files.append(filename)
        print(f"✓ {name:12s}: {filename} ({data.shape[0]} × {data.shape[1] if hasattr(data, 'shape') and len(data.shape) > 1 else 1})")
    
    # Save metadata
    for name in ['metadata_train', 'metadata_val', 'metadata_test']:
        filename = f'{PREPARED_DIR}/{name}_{timestamp}.csv'
        splits[name].to_csv(filename, index=False)
        saved_files.append(filename)
        print(f"✓ {name:12s}: {filename}")
    
    # Save scaler
    scaler_file = f'{PREPARED_DIR}/scaler_{timestamp}.pkl'
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    saved_files.append(scaler_file)
    print(f"✓ scaler:       {scaler_file}")
    
    # Save CV folds
    cv_file = f'{PREPARED_DIR}/cv_folds_{timestamp}.pkl'
    with open(cv_file, 'wb') as f:
        pickle.dump(splits['cv_folds'], f)
    saved_files.append(cv_file)
    print(f"✓ cv_folds:     {cv_file}")
    
    # Save configuration
    config_data = {
        'timestamp': timestamp,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'n_train': len(splits['X_train']),
        'n_val': len(splits['X_val']),
        'n_test': len(splits['X_test']),
        'train_degraded_pct': float(splits['y_train'].mean() * 100),
        'val_degraded_pct': float(splits['y_val'].mean() * 100),
        'test_degraded_pct': float(splits['y_test'].mean() * 100),
        'scaling_method': config['scaling_method'],
        'balancing_method': config['balancing_method'],
        'n_cv_folds': config['n_cv_folds'],
    }
    
    config_file = f'{PREPARED_DIR}/config_{timestamp}.pkl'
    with open(config_file, 'wb') as f:
        pickle.dump(config_data, f)
    saved_files.append(config_file)
    print(f"✓ config:       {config_file}")
    
    # Save summary as CSV
    summary_df = pd.DataFrame([{
        'timestamp': timestamp,
        'n_features': len(feature_names),
        'n_train': len(splits['X_train']),
        'n_val': len(splits['X_val']),
        'n_test': len(splits['X_test']),
        'train_degraded_pct': splits['y_train'].mean() * 100,
        'val_degraded_pct': splits['y_val'].mean() * 100,
        'test_degraded_pct': splits['y_test'].mean() * 100,
        'scaling_method': config['scaling_method'],
        'balancing_method': config['balancing_method'],
    }])
    
    summary_file = f'{PREPARED_DIR}/preparation_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    saved_files.append(summary_file)
    print(f"✓ summary:      {summary_file}")
    
    return saved_files, timestamp


def main():
    """Main data preparation pipeline"""
    print("\n" + "="*70)
    print("DATA PREPARATION FOR MACHINE LEARNING")
    print("Sprint Velocity Degradation Prediction")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    config = {
        'scaling_method': 'standard',      # 'standard' or 'robust'
        'balancing_method': 'none',        # 'none', 'oversample', 'undersample', 'combine'
        'n_cv_folds': 5,                   # Number of time-series CV folds
        'test_size': 0.15,                 # Proportion for test set
        'val_size': 0.15,                  # Proportion for validation set
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # Step 1: Load data
        df = load_ml_dataset()
        
        # Step 2: Prepare features and target
        X, y, metadata, feature_names = prepare_features_and_target(df)
        
        # Step 3: Create time-series splits
        splits = create_time_series_splits(
            X, y, metadata,
            n_splits=config['n_cv_folds'],
            test_size=config['test_size'],
            val_size=config['val_size']
        )
        
        # Step 4: Handle class imbalance (on training set only)
        # Note: Applying to raw data before scaling for better synthetic sample generation
        if config['balancing_method'] != 'none':
            splits['X_train'], splits['y_train'] = handle_class_imbalance(
                splits['X_train'],
                splits['y_train'],
                method=config['balancing_method']
            )
        
        # Step 5: Scale features (fit on training, transform all)
        splits['X_train'], splits['X_val'], splits['X_test'], scaler = scale_features(
            splits['X_train'],
            splits['X_val'],
            splits['X_test'],
            method=config['scaling_method']
        )
        
        # Step 6: Save prepared data
        saved_files, timestamp = save_prepared_data(splits, scaler, feature_names, config)
        
        # Final summary
        print("\n" + "="*70)
        print("DATA PREPARATION COMPLETE")
        print("="*70)
        print(f"\n✓ Successfully prepared data for ML modeling")
        print(f"✓ Saved {len(saved_files)} files to {PREPARED_DIR}/")
        print(f"\nDataset ready for modeling:")
        print(f"  Training:   {len(splits['X_train']):3d} samples ({splits['y_train'].mean()*100:.1f}% degraded)")
        print(f"  Validation: {len(splits['X_val']):3d} samples ({splits['y_val'].mean()*100:.1f}% degraded)")
        print(f"  Test:       {len(splits['X_test']):3d} samples ({splits['y_test'].mean()*100:.1f}% degraded)")
        print(f"  Features:   {len(feature_names)}")
        print(f"  CV Folds:   {config['n_cv_folds']}")
        
        print(f"\nTo use in modeling:")
        print(f"  import pickle")
        print(f"  import pandas as pd")
        print(f"  ")
        print(f"  X_train = pd.read_csv('{PREPARED_DIR}/X_train_{timestamp}.csv')")
        print(f"  y_train = pd.read_csv('{PREPARED_DIR}/y_train_{timestamp}.csv').squeeze()")
        print(f"  ")
        print(f"  with open('{PREPARED_DIR}/scaler_{timestamp}.pkl', 'rb') as f:")
        print(f"      scaler = pickle.load(f)")
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
