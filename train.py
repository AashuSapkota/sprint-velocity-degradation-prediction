#!/usr/bin/env python3
"""
Predictive Modelling for Sprint Velocity Degradation
Implements:
  - 3 baseline models (naive, persistence, moving-average)
  - 4 ML algorithms × 3 feature variants = 12 models
  - Time-series cross-validation on each model
  - Evaluation: F1, ROC-AUC, MCC, precision, recall, confusion matrix
Addresses O4, RQ4, H3
"""

import copy
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

PREPARED_DIR = 'dataset/prepared'
RESULTS_DIR  = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Feature group definitions ──────────────────────────────────────────────
CHURN_FEATURES = [
    'total_commits', 'total_churn', 'churn_concentration',
    'developer_churn_inequality', 'refactoring_ratio', 'file_hotspots',
    'avg_commit_size', 'churn_volatility', 'unique_developers',
    'avg_files_per_commit', 'commits_per_day', 'churn_per_developer',
]

NETWORK_FEATURES = [
    'num_nodes', 'num_edges', 'density', 'avg_clustering', 'diameter',
    'num_components', 'giant_component_size', 'avg_degree', 'max_betweenness',
    'degree_centralization', 'avg_closeness', 'network_jaccard', 'nodes_added',
    'nodes_removed', 'density_change', 'centralization_change', 'turnover_rate',
    'edges_per_node',
]


# ── Data loading ───────────────────────────────────────────────────────────
def load_prepared_data():
    files = list(Path(PREPARED_DIR).glob('X_train_*.csv'))
    if not files:
        raise FileNotFoundError("No prepared data found. Run prepare_data.py first.")
    ts = sorted(f.stem.split('_', 2)[2] for f in files)[-1]
    print(f"Loading prepared data (timestamp: {ts})")

    X_train = pd.read_csv(f'{PREPARED_DIR}/X_train_{ts}.csv')
    X_val   = pd.read_csv(f'{PREPARED_DIR}/X_val_{ts}.csv')
    X_test  = pd.read_csv(f'{PREPARED_DIR}/X_test_{ts}.csv')
    y_train = pd.read_csv(f'{PREPARED_DIR}/y_train_{ts}.csv').squeeze()
    y_val   = pd.read_csv(f'{PREPARED_DIR}/y_val_{ts}.csv').squeeze()
    y_test  = pd.read_csv(f'{PREPARED_DIR}/y_test_{ts}.csv').squeeze()

    # Impute any residual NaNs using training-set medians (fit on train, apply to all)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_val   = X_val.fillna(train_medians)
    X_test  = X_test.fillna(train_medians)

    with open(f'{PREPARED_DIR}/cv_folds_{ts}.pkl', 'rb') as f:
        cv_folds = pickle.load(f)

    meta_ts = sorted(
        f.stem.split('_', 2)[2]
        for f in Path(PREPARED_DIR).glob('metadata_train_*.csv')
    )[-1]
    metadata_train = pd.read_csv(f'{PREPARED_DIR}/metadata_train_{meta_ts}.csv')
    metadata_val   = pd.read_csv(f'{PREPARED_DIR}/metadata_val_{meta_ts}.csv')
    metadata_test  = pd.read_csv(f'{PREPARED_DIR}/metadata_test_{meta_ts}.csv')

    print(f"  Train : {X_train.shape}  Degraded: {y_train.sum()}/{len(y_train)}")
    print(f"  Val   : {X_val.shape}    Degraded: {y_val.sum()}/{len(y_val)}")
    print(f"  Test  : {X_test.shape}   Degraded: {y_test.sum()}/{len(y_test)}")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            cv_folds,
            metadata_train, metadata_val, metadata_test)


def load_raw_dataset():
    files = list(Path('dataset').glob('ml_dataset_*.csv'))
    if not files:
        raise FileNotFoundError("No ml_dataset found. Run feature_engineering.py first.")
    return pd.read_csv(str(max(files, key=os.path.getctime)))


# ── Shared evaluation helper ───────────────────────────────────────────────
def evaluate(y_true, y_pred, y_prob=None):
    metrics = {
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'mcc':       matthews_corrcoef(y_true, y_pred),
        'roc_auc':   roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan'),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    return metrics


# ── Baselines ──────────────────────────────────────────────────────────────
def run_baselines(y_train, y_val, y_test,
                  metadata_val, metadata_test, ml_df):
    print("\n" + "=" * 70)
    print("BASELINE MODELS")
    print("=" * 70)
    results = []

    # 1. Naive majority-class classifier
    majority = int(y_train.mode()[0])
    for split_name, y_true in [('val', y_val), ('test', y_test)]:
        y_pred = np.full(len(y_true), majority)
        m = evaluate(y_true, y_pred)
        results.append({'model': 'naive', 'split': split_name, **m})
        print(f"  Naive (majority={majority}) [{split_name}] "
              f"F1={m['f1']:.3f}  MCC={m['mcc']:.3f}  AUC=N/A")

    # 2. Persistence: predict degraded if PREVIOUS release was degraded
    # 3. Moving average: predict degraded if velocity is below 3-release mean
    if ml_df is not None:
        seq = (ml_df[['release', 'start_date', 'weighted_velocity', 'is_degraded']]
               .sort_values('start_date')
               .reset_index(drop=True))

        seq['persist_pred'] = seq['is_degraded'].shift(1).fillna(0).astype(int)

        rolling_mean = seq['weighted_velocity'].shift(1).rolling(3, min_periods=1).mean()
        seq['ma3_pred'] = (seq['weighted_velocity'] < rolling_mean).astype(int)

        def preds_for_split(meta_df, pred_col):
            return np.array([
                int(seq.loc[seq['release'] == r, pred_col].values[0])
                if r in seq['release'].values else 0
                for r in meta_df['release']
            ])

        for bname, pcol in [('persistence', 'persist_pred'), ('moving_avg', 'ma3_pred')]:
            for split_name, y_true, meta_df in [
                ('val',  y_val,  metadata_val),
                ('test', y_test, metadata_test),
            ]:
                y_pred = preds_for_split(meta_df, pcol)
                m = evaluate(y_true, y_pred)
                results.append({'model': bname, 'split': split_name, **m})
                print(f"  {bname.capitalize():12s} [{split_name}] "
                      f"F1={m['f1']:.3f}  MCC={m['mcc']:.3f}")

    return pd.DataFrame(results)


# ── ML Models ──────────────────────────────────────────────────────────────
def build_models():
    return {
        'logistic_regression': LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, class_weight='balanced'),
        'random_forest': RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight='balanced'),
        'svm': SVC(
            kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
        'xgboost': XGBClassifier(
            n_estimators=200, random_state=42, eval_metric='logloss',
            scale_pos_weight=1, verbosity=0, use_label_encoder=False),
    }


# velocity_change_pct is excluded: is_degraded = (velocity_change_pct < -10),
# so including it in features is direct data leakage.
LEAKY_FEATURES = ['velocity_change_pct']


def feature_subsets(all_cols):
    safe_cols = [c for c in all_cols if c not in LEAKY_FEATURES]
    avail_churn   = [f for f in CHURN_FEATURES   if f in safe_cols]
    avail_network = [f for f in NETWORK_FEATURES if f in safe_cols]
    return {
        'A_churn':    avail_churn,
        'B_network':  avail_network,
        'C_combined': safe_cols,
    }


def cv_f1(model, X_tr, y_tr, folds):
    scores = []
    for tr_idx, vl_idx in folds:
        m = copy.deepcopy(model)
        m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        y_p = m.predict(X_tr.iloc[vl_idx])
        scores.append(f1_score(y_tr.iloc[vl_idx], y_p, zero_division=0))
    return float(np.mean(scores)), float(np.std(scores))


def train_all_models(X_train, X_val, X_test,
                     y_train, y_val, y_test, cv_folds):
    print("\n" + "=" * 70)
    print("ML MODELS  (4 algorithms × 3 feature variants)")
    print("=" * 70)

    subsets        = feature_subsets(X_train.columns)
    base_models    = build_models()
    rows           = []
    trained_models = {}

    for variant, feat_cols in subsets.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {variant}  ({len(feat_cols)} features)")
        print(f"{'─' * 60}")

        Xtr  = X_train[feat_cols]
        Xval = X_val[feat_cols]
        Xte  = X_test[feat_cols]

        for mname, base_model in base_models.items():
            model = copy.deepcopy(base_model)

            mean_cv, std_cv = cv_f1(model, Xtr, y_train, cv_folds)

            model.fit(Xtr, y_train)

            prob_val  = model.predict_proba(Xval)[:, 1]
            prob_test = model.predict_proba(Xte)[:, 1]
            pred_val  = model.predict(Xval)
            pred_test = model.predict(Xte)

            vm = evaluate(y_val,  pred_val,  prob_val)
            tm = evaluate(y_test, pred_test, prob_test)

            key = f'{mname}__{variant}'
            trained_models[key] = model

            rows.append({
                'model': mname, 'variant': variant,
                'n_features': len(feat_cols),
                'cv_f1_mean': mean_cv, 'cv_f1_std': std_cv,
                **{f'val_{k}':  v for k, v in vm.items()},
                **{f'test_{k}': v for k, v in tm.items()},
            })

            print(f"  {mname:25s} | CV F1={mean_cv:.3f}±{std_cv:.3f} "
                  f"| Val  F1={vm['f1']:.3f} AUC={vm['roc_auc']:.3f} "
                  f"| Test F1={tm['f1']:.3f} MCC={tm['mcc']:.3f}")

    return pd.DataFrame(rows), trained_models, subsets


# ── Summary table ──────────────────────────────────────────────────────────
def print_summary(model_df, baseline_df):
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — Test Set")
    print("=" * 70)

    print("\nBaselines:")
    bl_test = baseline_df[baseline_df['split'] == 'test'][
        ['model', 'f1', 'precision', 'recall', 'mcc', 'roc_auc']
    ].copy()
    print(bl_test.to_string(index=False))

    print("\nML models (sorted by test F1):")
    cols = ['model', 'variant', 'n_features',
            'cv_f1_mean', 'test_f1', 'test_roc_auc', 'test_mcc',
            'test_precision', 'test_recall']
    print(model_df.sort_values('test_f1', ascending=False)[cols].to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("PREDICTIVE MODELLING — Sprint Velocity Degradation")
    print("=" * 70)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     cv_folds,
     meta_train, meta_val, meta_test) = load_prepared_data()

    ml_df = load_raw_dataset()

    baseline_df = run_baselines(
        y_train, y_val, y_test,
        meta_val, meta_test, ml_df,
    )

    model_df, trained_models, subsets = train_all_models(
        X_train, X_val, X_test,
        y_train, y_val, y_test, cv_folds,
    )

    print_summary(model_df, baseline_df)

    # Save
    baseline_df.to_csv(f'{RESULTS_DIR}/baseline_results_{ts}.csv', index=False)
    model_df.to_csv(f'{RESULTS_DIR}/model_results_{ts}.csv',   index=False)

    with open(f'{RESULTS_DIR}/trained_models_{ts}.pkl', 'wb') as f:
        pickle.dump(trained_models, f)

    print(f"\n✓ baseline_results  → {RESULTS_DIR}/baseline_results_{ts}.csv")
    print(f"✓ model_results     → {RESULTS_DIR}/model_results_{ts}.csv")
    print(f"✓ trained_models    → {RESULTS_DIR}/trained_models_{ts}.pkl")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext step: run analyze.py for SHAP, hypothesis tests, and sensitivity analysis.")


if __name__ == '__main__':
    main()
