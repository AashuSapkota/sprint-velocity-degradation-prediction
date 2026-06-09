#!/usr/bin/env python3
"""
Create Alternative Balanced Dataset Versions
Generates datasets with different class balancing strategies
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add parent directory to path
sys.path.append('.')
from prepare_data import (
    load_ml_dataset, prepare_features_and_target,
    create_time_series_splits, handle_class_imbalance,
    scale_features, save_prepared_data
)

PREPARED_DIR = 'dataset/prepared'


def create_balanced_versions():
    """Create multiple versions with different balancing strategies"""
    print("="*70)
    print("CREATING BALANCED DATASET VERSIONS")
    print("="*70)
    
    # Load data
    df = load_ml_dataset()
    X, y, metadata, feature_names = prepare_features_and_target(df)
    
    # Base configuration
    base_config = {
        'scaling_method': 'standard',
        'n_cv_folds': 5,
        'test_size': 0.15,
        'val_size': 0.15,
    }
    
    # Create splits (common for all versions)
    splits = create_time_series_splits(
        X, y, metadata,
        n_splits=base_config['n_cv_folds'],
        test_size=base_config['test_size'],
        val_size=base_config['val_size']
    )
    
    # Store original training data
    X_train_original = splits['X_train'].copy()
    y_train_original = splits['y_train'].copy()
    
    # Balancing strategies to try
    strategies = ['oversample', 'undersample', 'combine']
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Processing: {strategy.upper()}")
        print(f"{'='*70}")
        
        # Reset to original training data
        splits['X_train'] = X_train_original.copy()
        splits['y_train'] = y_train_original.copy()
        
        # Apply balancing
        config = base_config.copy()
        config['balancing_method'] = strategy
        
        splits['X_train'], splits['y_train'] = handle_class_imbalance(
            splits['X_train'],
            splits['y_train'],
            method=strategy
        )
        
        # Scale features
        splits['X_train'], splits['X_val'], splits['X_test'], scaler = scale_features(
            splits['X_train'],
            splits['X_val'],
            splits['X_test'],
            method='standard'
        )
        
        # Save
        saved_files, timestamp = save_prepared_data(splits, scaler, feature_names, config)
        
        print(f"\n✓ Saved {strategy} version with timestamp: {timestamp}")


if __name__ == '__main__':
    try:
        create_balanced_versions()
        print("\n" + "="*70)
        print("ALL VERSIONS CREATED SUCCESSFULLY")
        print("="*70)
        print("\nAvailable versions:")
        print("  1. none        - Original class distribution (45.6% degraded)")
        print("  2. oversample  - SMOTE oversampling (balanced)")
        print("  3. undersample - Random undersampling (balanced)")
        print("  4. combine     - SMOTE + Tomek Links (balanced)")
        print("\nRecommendation: Start with 'none' for baseline, try others if needed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
