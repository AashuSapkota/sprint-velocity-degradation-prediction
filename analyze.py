#!/usr/bin/env python3
"""
Post-training analysis for Sprint Velocity Degradation study
  - Hypothesis testing H1 & H2 (Spearman correlations)
  - H3: combined ML vs baselines
  - H4 / RQ3: variant comparison (churn vs network vs combined)
  - SHAP feature importance (best combined model)
  - Sensitivity analysis (velocity thresholds, period stratification)
Addresses RQ1, RQ2, RQ3, RQ4, H1–H4
"""

import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch
from scipy import stats

warnings.filterwarnings('ignore')

PREPARED_DIR = 'dataset/prepared'
RESULTS_DIR  = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

CHURN_FEATURES = [
    'total_commits', 'total_churn', 'churn_concentration',
    'developer_churn_inequality', 'refactoring_ratio', 'file_hotspots',
    'avg_commit_size', 'churn_volatility', 'unique_developers',
    'avg_files_per_commit', 'commits_per_day', 'churn_per_developer',
]
NETWORK_STATIC = [
    'num_nodes', 'num_edges', 'density', 'avg_clustering', 'diameter',
    'num_components', 'giant_component_size', 'avg_degree',
    'max_betweenness', 'degree_centralization', 'avg_closeness',
]
NETWORK_CHANGE = [
    'network_jaccard', 'nodes_added', 'nodes_removed',
    'density_change', 'centralization_change', 'turnover_rate', 'edges_per_node',
]


# ── Loaders ────────────────────────────────────────────────────────────────
def load_latest(pattern, directory=RESULTS_DIR):
    files = list(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {directory}")
    return str(max(files, key=os.path.getctime))


def load_results():
    path = load_latest('model_results_*.csv')
    print(f"Model results : {path}")
    return pd.read_csv(path)


def load_baselines():
    path = load_latest('baseline_results_*.csv')
    print(f"Baseline results: {path}")
    return pd.read_csv(path)


def load_trained_models():
    path = load_latest('trained_models_*.pkl')
    print(f"Trained models: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_prepared_data():
    files = list(Path(PREPARED_DIR).glob('X_train_*.csv'))
    ts = sorted(f.stem.split('_', 2)[2] for f in files)[-1]
    X_train = pd.read_csv(f'{PREPARED_DIR}/X_train_{ts}.csv')
    X_test  = pd.read_csv(f'{PREPARED_DIR}/X_test_{ts}.csv')
    y_train = pd.read_csv(f'{PREPARED_DIR}/y_train_{ts}.csv').squeeze()
    y_test  = pd.read_csv(f'{PREPARED_DIR}/y_test_{ts}.csv').squeeze()
    return X_train, X_test, y_train, y_test


def load_raw_dataset():
    files = list(Path('dataset').glob('ml_dataset_*.csv'))
    return pd.read_csv(str(max(files, key=os.path.getctime)))


# ── H1 & H2: Correlation hypothesis tests (RQ1, RQ2) ─────────────────────
def hypothesis_h1_h2(df_raw, ts):
    print("\n" + "=" * 70)
    print("H1 & H2 — SPEARMAN CORRELATIONS  (RQ1 & RQ2)")
    print("=" * 70)

    target = df_raw['is_degraded']
    groups = {
        'H1 — Network centralization & density': [
            'degree_centralization', 'density', 'density_change',
            'centralization_change', 'max_betweenness', 'avg_clustering',
        ],
        'H2 — Churn concentration & inequality': [
            'churn_concentration', 'developer_churn_inequality',
            'churn_volatility', 'total_churn', 'refactoring_ratio',
        ],
    }

    rows = []
    for hyp_label, features in groups.items():
        print(f"\n{hyp_label}:")
        for feat in features:
            if feat not in df_raw.columns:
                continue
            r, p = stats.spearmanr(df_raw[feat].fillna(0), target)
            sig = ('***' if p < 0.001
                   else '**'  if p < 0.01
                   else '*'   if p < 0.05
                   else '')
            rows.append({'hypothesis': hyp_label, 'feature': feat,
                         'spearman_r': r, 'p_value': p})
            print(f"  {feat:35s}  r={r:+.3f}  p={p:.4f}  {sig}")

    corr_df = pd.DataFrame(rows)

    # Verdict
    h1_rows = corr_df[corr_df['hypothesis'].str.startswith('H1')]
    h1_pass = (h1_rows['spearman_r'].abs() > 0.30).any() and (h1_rows['p_value'] < 0.05).any()
    print(f"\nH1 verdict: {'SUPPORTED' if h1_pass else 'NOT SUPPORTED'}"
          f"  (target: |r| > 0.30 for network centralisation/density)")

    h2_cc = corr_df[corr_df['feature'] == 'churn_concentration']
    if not h2_cc.empty:
        h2_pass = (h2_cc['spearman_r'].abs() > 0.25).values[0]
        h2_p    = h2_cc['p_value'].values[0]
        print(f"H2 verdict: {'SUPPORTED' if h2_pass else 'NOT SUPPORTED'}"
              f"  (churn_concentration |r|={h2_cc['spearman_r'].abs().values[0]:.3f}, p={h2_p:.4f})")

    out = f'{RESULTS_DIR}/hypothesis_correlations_{ts}.csv'
    corr_df.to_csv(out, index=False)
    print(f"\n✓ Saved → {out}")
    return corr_df


# ── H3: Combined ML vs baselines ──────────────────────────────────────────
def hypothesis_h3(model_df, baseline_df):
    print("\n" + "=" * 70)
    print("H3 — COMBINED ML MODELS vs BASELINES")
    print("=" * 70)

    bl_test = baseline_df[baseline_df['split'] == 'test']
    best_bl_f1    = bl_test['f1'].max()
    best_bl_model = bl_test.loc[bl_test['f1'].idxmax(), 'model']

    comb = model_df[model_df['variant'] == 'C_combined']
    best_ml_f1    = comb['test_f1'].max()
    best_ml_model = comb.loc[comb['test_f1'].idxmax(), 'model']

    print(f"\n  Best baseline (test F1): {best_bl_f1:.3f}  [{best_bl_model}]")
    print(f"  Best combined ML (test F1): {best_ml_f1:.3f}  [{best_ml_model}]")

    h3_pass = best_ml_f1 >= 0.75
    print(f"\n  H3 verdict: {'SUPPORTED' if h3_pass else 'NOT SUPPORTED'}")
    print(f"    Target: F1 ≥ 0.75 for combined model (achieved: {best_ml_f1:.3f})")
    print(f"    Baseline target ~0.55 (observed: {best_bl_f1:.3f})")


# ── H4 / RQ3: Variant A vs B vs C ─────────────────────────────────────────
def hypothesis_h4_rq3(model_df):
    print("\n" + "=" * 70)
    print("H4 / RQ3 — CHURN vs NETWORK vs COMBINED VARIANTS")
    print("=" * 70)
    print(f"\n  {'Model':25s} {'Variant A (Churn)':>18} {'Variant B (Network)':>20} {'Variant C (Combined)':>22}  H4-pass?")
    print(f"  {'─'*25} {'─'*18} {'─'*20} {'─'*22}  {'─'*8}")

    rows = []
    for mname in model_df['model'].unique():
        sub = model_df[model_df['model'] == mname].set_index('variant')
        try:
            a = sub.loc['A_churn',    'test_f1']
            b = sub.loc['B_network',  'test_f1']
            c = sub.loc['C_combined', 'test_f1']
        except KeyError:
            continue

        gap_a = (c - a) / max(c, 1e-9) * 100
        gap_b = (c - b) / max(c, 1e-9) * 100
        h4_pass = gap_a >= 10 and gap_b >= 10   # neither within 10% of combined

        print(f"  {mname:25s} {a:.3f} (Δ{gap_a:+5.1f}%)   {b:.3f} (Δ{gap_b:+5.1f}%)   {c:.3f}  "
              f"{'YES' if h4_pass else 'NO'}")
        rows.append({'model': mname, 'A_f1': a, 'B_f1': b, 'C_f1': c,
                     'gap_A_pct': gap_a, 'gap_B_pct': gap_b, 'h4_pass': h4_pass})

    overall = all(r['h4_pass'] for r in rows)
    print(f"\n  H4 overall: {'SUPPORTED' if overall else 'MIXED / NOT FULLY SUPPORTED'}")
    print("  (H4 requires: neither A nor B within 10% of C, for all models)")
    return pd.DataFrame(rows)


# ── SHAP feature importance ────────────────────────────────────────────────
def shap_analysis(trained_models, X_train, X_test, model_df, ts):
    print("\n" + "=" * 70)
    print("SHAP FEATURE IMPORTANCE")
    print("=" * 70)

    # Use Random Forest combined — most compatible with TreeExplainer
    model_key = 'random_forest__C_combined'
    if model_key not in trained_models:
        # Fallback: best non-SVM combined model
        comb = model_df[model_df['variant'] == 'C_combined']
        comb = comb[~comb['model'].str.contains('svm')]
        model_key = f"{comb.loc[comb['test_f1'].idxmax(), 'model']}__C_combined"

    row = model_df[(model_df['model'] == model_key.split('__')[0]) &
                   (model_df['variant'] == 'C_combined')]
    f1  = row['test_f1'].values[0] if not row.empty else 'N/A'
    print(f"\n  SHAP model: {model_key}  (Test F1={f1:.3f})")

    if model_key not in trained_models:
        print(f"  ⚠  Model key '{model_key}' not in trained_models — skipping SHAP.")
        return None

    model = trained_models[model_key]

    # Use only features the model was trained on (feature_names_in_)
    if hasattr(model, 'feature_names_in_'):
        feat_cols = list(model.feature_names_in_)
    else:
        feat_cols = list(X_train.columns)

    X_pool = pd.concat(
        [X_train[feat_cols], X_test[feat_cols]], ignore_index=True
    )

    print("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_pool)
    # Handle different SHAP output shapes:
    #   list of arrays → [class0, class1] → take class 1
    #   3-D array      → (samples, features, classes) → take [:, :, 1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    def feature_type(f):
        if f in CHURN_FEATURES:   return 'churn'
        if f in NETWORK_CHANGE:   return 'network_change'
        if f in NETWORK_STATIC:   return 'network_static'
        return 'velocity/context'

    shap_df = pd.DataFrame({
        'feature':        X_pool.columns,
        'mean_abs_shap':  np.abs(shap_vals).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)
    shap_df['type'] = shap_df['feature'].apply(feature_type)

    print(f"\n  Top 15 features by mean |SHAP value|:")
    print(shap_df.head(15).to_string(index=False))

    # H4 — network change vs network static
    nc_mean = shap_df[shap_df['type'] == 'network_change']['mean_abs_shap'].mean()
    ns_mean = shap_df[shap_df['type'] == 'network_static']['mean_abs_shap'].mean()
    print(f"\n  Network change SHAP (mean): {nc_mean:.4f}")
    print(f"  Network static SHAP (mean): {ns_mean:.4f}")
    print(f"  H4 (change > static): {'SUPPORTED' if nc_mean > ns_mean else 'NOT SUPPORTED'}")

    # ── Bar plot ───────────────────────────────────────────────────────────
    colour_map = {
        'churn':            '#e74c3c',
        'network_change':   '#2980b9',
        'network_static':   '#27ae60',
        'velocity/context': '#f39c12',
    }
    top15 = shap_df.head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        top15['feature'][::-1],
        top15['mean_abs_shap'][::-1],
        color=[colour_map[t] for t in top15['type'][::-1]],
    )
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Feature Importance (SHAP) — {model_key}')

    legend_handles = [
        Patch(facecolor=c, label=t.replace('_', ' ').title())
        for t, c in colour_map.items()
        if t in top15['type'].values
    ]
    ax.legend(handles=legend_handles, loc='lower right')
    plt.tight_layout()
    plot_path = f'{RESULTS_DIR}/shap_importance_{ts}.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ── Group-level bar chart ──────────────────────────────────────────────
    group_shap = (shap_df.groupby('type')['mean_abs_shap']
                  .mean()
                  .sort_values(ascending=True))
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colours = [colour_map.get(t, '#aaa') for t in group_shap.index]
    ax2.barh(group_shap.index, group_shap.values, color=colours)
    ax2.set_xlabel('Mean |SHAP value| (group average)')
    ax2.set_title('SHAP Importance by Feature Group')
    plt.tight_layout()
    group_plot_path = f'{RESULTS_DIR}/shap_by_group_{ts}.png'
    fig2.savefig(group_plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    shap_df.to_csv(f'{RESULTS_DIR}/shap_importance_{ts}.csv', index=False)
    print(f"\n  ✓ Saved SHAP table → {RESULTS_DIR}/shap_importance_{ts}.csv")
    print(f"  ✓ Saved SHAP plot  → {plot_path}")
    print(f"  ✓ Saved group plot → {group_plot_path}")
    return shap_df


# ── Sensitivity analysis ───────────────────────────────────────────────────
def sensitivity_analysis(df_raw, ts):
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)

    df = df_raw.sort_values('start_date').copy()

    # 1. Alternative degradation thresholds
    # velocity_change_pct is in % units (e.g., -10 means -10%)
    print("\n1. Alternative velocity-degradation thresholds:")
    thresh_rows = []
    for t in [-5, -10, -15, -20, -25]:
        col = f'degraded_{abs(t)}pct'
        df[col] = (df['velocity_change_pct'] < t).astype(int)
        pct = df[col].mean() * 100
        n   = df[col].sum()
        print(f"   Threshold {t:5d}% : {n:3d} degraded releases ({pct:.1f}%)")
        thresh_rows.append({'threshold_pct': abs(t), 'n_degraded': n, 'pct_degraded': pct})
    pd.DataFrame(thresh_rows).to_csv(f'{RESULTS_DIR}/sensitivity_thresholds_{ts}.csv', index=False)

    # 2. Period-stratified degradation rates
    print("\n2. Period-stratified degradation rates:")
    df['year'] = pd.to_datetime(df['start_date'], utc=True, errors='coerce').dt.year

    def era(y):
        if y <= 2015: return 'Early (2011–2015)'
        if y <= 2020: return 'Mid (2016–2020)'
        return 'Recent (2021–2026)'

    df['period'] = df['year'].apply(era)
    period_rows = []
    for period, grp in df.groupby('period', sort=False):
        dr = grp['is_degraded'].mean() * 100
        print(f"   {period}: n={len(grp):3d}  degraded={dr:.1f}%")
        period_rows.append({'period': period, 'n': len(grp), 'pct_degraded': dr})
    pd.DataFrame(period_rows).to_csv(f'{RESULTS_DIR}/sensitivity_periods_{ts}.csv', index=False)

    # 3. Velocity operationalisation: raw issue count vs weighted
    print("\n3. Velocity operationalisation comparison:")
    for op_name, col in [
        ('Weighted velocity (used)',  'weighted_velocity'),
        ('Raw issue count',           'issue_count'),
        ('Velocity per day',          'velocity_per_day'),
    ]:
        if col not in df.columns:
            continue
        r, p = stats.spearmanr(df[col].fillna(0), df['is_degraded'])
        print(f"   {op_name:30s}  r={r:+.3f}  p={p:.4f}")

    print(f"\n  ✓ Sensitivity results saved to {RESULTS_DIR}/")


# ── Variant performance plot ───────────────────────────────────────────────
def plot_variant_comparison(model_df, ts):
    variants = ['A_churn', 'B_network', 'C_combined']
    models   = sorted(model_df['model'].unique())
    x        = np.arange(len(models))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, var in enumerate(variants):
        f1s = [
            model_df.loc[(model_df['model'] == m) & (model_df['variant'] == var), 'test_f1'].values[0]
            if ((model_df['model'] == m) & (model_df['variant'] == var)).any() else 0
            for m in models
        ]
        ax.bar(x + i * width, f1s, width, label=var)

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Test F1-score')
    ax.set_title('Test F1 by Algorithm and Feature Variant')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.axhline(0.75, color='red', linestyle='--', linewidth=0.8, label='H3 target (0.75)')
    ax.legend()
    plt.tight_layout()
    path = f'{RESULTS_DIR}/variant_comparison_{ts}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Variant comparison plot → {path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("ANALYSIS — SHAP, HYPOTHESES, SENSITIVITY")
    print("=" * 70)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    model_df      = load_results()
    baseline_df   = load_baselines()
    trained_models = load_trained_models()
    X_train, X_test, y_train, y_test = load_prepared_data()
    df_raw        = load_raw_dataset()

    # Hypothesis H1 & H2
    hypothesis_h1_h2(df_raw, ts)

    # Hypothesis H3
    hypothesis_h3(model_df, baseline_df)

    # Hypothesis H4 / RQ3
    h4_df = hypothesis_h4_rq3(model_df)
    h4_df.to_csv(f'{RESULTS_DIR}/h4_variant_comparison_{ts}.csv', index=False)

    # SHAP analysis
    shap_analysis(trained_models, X_train, X_test, model_df, ts)

    # Sensitivity analysis
    sensitivity_analysis(df_raw, ts)

    # Variant comparison plot
    print("\n" + "=" * 70)
    print("PLOTS")
    print("=" * 70)
    plot_variant_comparison(model_df, ts)

    print(f"\n✓ All analysis complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✓ All outputs saved to {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
