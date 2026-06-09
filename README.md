# Sprint Velocity Degradation Prediction

**Title:** Predicting Sprint Velocity Degradation in Agile Teams Using Developer Interaction Networks and Code Churn Metrics

**Repository:** Apache Kafka Issue & Commit Data Collection

## Project Status

🎯 **Phase 1: COMPLETE** ✅ (March 9, 2026)
- ✅ Data Collection (3 scripts, 10 datasets)
- ✅ Feature Engineering (47 features, 111 releases)
- ✅ Data Validation (balanced, high quality)

🚀 **Phase 2: IN PROGRESS** - Predictive Modeling

See [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) for detailed completion report.

## Overview

This project collects and analyzes data from the Apache Kafka project to predict sprint velocity degradation using:
- Code churn metrics from GitHub commits (10 features)
- Developer interaction networks & co-commit patterns (17 features)
- File coupling networks (architectural dependencies)
- Jira issue tracking data & velocity metrics (7 features)

## Project Structure

```
project/
├── fetch_jira_issues.py        # Collect Jira issues
├── fetch_github_issues.py      # Collect GitHub commits & networks
├── link_jira_github.py         # Link Jira & GitHub data
├── feature_engineering.py      # ✨ NEW: Calculate all ML features
├── validate_features.py        # ✨ NEW: Validate feature quality
├── dataset/                    # All output data (auto-created)
├── checkpoints/                # Progress checkpoints (auto-created)
├── README.md                   # This file
└── PHASE1_SUMMARY.md           # ✨ NEW: Phase 1 completion report
```

## Quick Start

### Prerequisites

```bash
pip install pandas numpy jira pydriller networkx
```

### Data Collection Workflow

**Step 1: Collect Jira Issues**
```bash
python fetch_jira_issues.py
```
- Fetches all KAFKA issues from Apache Jira
- Output: `dataset/kafka_issues_YYYYMMDD_HHMMSS.csv`

**Step 2: Collect GitHub Commits**
```bash
python fetch_github_issues.py
```
- Analyzes Apache Kafka Git repository
- Extracts commits with churn metrics
- Builds developer collaboration networks
- Builds file coupling networks
- Output: 5 CSV files in `dataset/`

**Step 3: Link Jira & GitHub**
```bash
python link_jira_github.py
```
- Links commits to Jira issues
- Aggregates metrics by issue and release
- Creates final dataset
- Output: 4 CSV files in `dataset/`

**Step 4: Feature Engineering ✨ NEW**
```bash
python feature_engineering.py
```
- Extracts 111 release cycles from Kafka history
- Calculates velocity degradation labels (44.1% degraded)
- Computes 10 advanced code churn features
- Builds temporal co-commit networks (7-day windows)
- Calculates 17 developer network features
- Creates ML-ready dataset: 111 releases × 47 features
- Output: `ml_dataset_*.csv` + 3 feature subsets
- Runtime: ~7 minutes

## Output Datasets

### From `fetch_jira_issues.py`

**kafka_issues_*.csv**
- All Kafka Jira issues with metadata
- Columns: key, created, resolved, status, resolution, priority, type, fix_versions, assignee, reporter
- Use: Velocity tracking, release grouping

### From `fetch_github_issues.py`

**commits_*.csv**
- All commits with churn metrics
- Columns: hash, author_name, author_email, date, timestamp, msg, insertions, deletions, churn, num_files, jira_keys, has_jira_ref
- Use: Code churn analysis, Jira linking

**file_authors_*.csv**
- File-author relationships
- Columns: commit_hash, file, author, date, additions, deletions
- Use: Build co-commit networks

**file_cochanges_*.csv**
- Files that changed together
- Columns: commit_hash, files, num_files, author, date
- Use: File coupling analysis

**developer_pairs_*.csv**
- Developer collaboration pairs
- Columns: developer1, developer2, shared_files
- Use: Pre-computed co-commit network edges

**file_coupling_*.csv**
- File coupling relationships
- Columns: file1, file2, cochange_count
- Use: Pre-computed file coupling network

### From `link_jira_github.py`

**commit_issue_links_*.csv**
- Detailed commit-issue mappings
- Use: Commit-level analysis

**issue_metrics_*.csv**
- Aggregated metrics per issue
- Use: Issue-level analysis

**release_metrics_*.csv**
- Aggregated metrics per release
- Use: Sprint/release velocity analysis

### From `feature_engineering.py` ✨ **ML-READY DATASETS**

**ml_dataset_*.csv** ⭐ **PRIMARY DATASET FOR MODELING**
- Complete ML-ready dataset with 47 features
- 111 releases (samples) from Apache Kafka history (2011-2026)
- Target variable: `is_degraded` (binary: 1=degraded, 0=stable)
- Class distribution: 44.1% degraded, 55.9% stable (well-balanced)

**Feature Categories:**
1. **Velocity Features (7):**
   - weighted_velocity, velocity_change_pct, issue_count
   - bugs, improvements, new_features, tasks
   - velocity_per_day

2. **Code Churn Features (10):**
   - total_commits, total_churn
   - churn_concentration (Gini coefficient)
   - developer_churn_inequality (Gini coefficient)
   - refactoring_ratio (deletion/insertion ratio)
   - file_hotspots (90th percentile)
   - avg_commit_size, churn_volatility
   - unique_developers, avg_files_per_commit

3. **Developer Network Features (17):**
   - Structural: num_nodes, num_edges, density, avg_clustering
   - Diameter, num_components, giant_component_size
   - Centrality: avg_degree, max_betweenness, degree_centralization
   - avg_closeness
   - Temporal: network_jaccard, nodes_added, nodes_removed
   - density_change, centralization_change, turnover_rate

4. **Derived Features (3):**
   - commits_per_day, churn_per_developer, edges_per_node

**velocity_features_*.csv**
- Velocity metrics and degradation labels per release
- Use: Baseline modeling, velocity trend analysis

**churn_features_*.csv**
- Advanced code churn metrics per release
- Use: Technical dimension analysis (Model A)

**network_features_*.csv**
- Developer network metrics per release
- Use: Social dimension analysis (Model B)

### From `validate_features.py`

**validation_summary.csv**
- Data quality report
- Feature statistics and correlations

**final_dataset_*.csv** (Legacy)
- Old format from link_jira_github.py
- Use: ml_dataset_*.csv instead for modeling

## Key Metrics (Phase 1 Complete)

### Velocity Metrics ✅
- `weighted_velocity`: Issues × (type_weight × priority_weight)
- `velocity_change_pct`: % change from previous release
- `is_degraded`: Binary label (1 if change < -10%, else 0)
- `velocity_per_day`: Normalized by release duration
- Mean velocity: 113.7 | 49 degraded (44.1%) | 62 stable (55.9%)

### Code Churn Metrics ✅
- `total_churn`: Total lines changed (insertions + deletions)
- `churn_concentration`: Gini coefficient (0.757 mean = localized)
- `developer_churn_inequality`: Gini of commits/dev (0.701 mean)
- `refactoring_ratio`: Proportion with deletion/insertion > 0.8 (35.2%)
- `churn_volatility`: Std dev of daily churn
- `file_hotspots`: Files in 90th percentile of changes
- `avg_commit_size`: Mean lines changed per commit

### Developer Network Metrics ✅
- **Structural:** density (0.030 mean), clustering, diameter, components
- **Centrality:** degree (708 centralization = high inequality), betweenness, closeness
- **Temporal:** turnover_rate (24.3% mean), network_jaccard, density_change
- **Construction:** 7-day temporal co-commit windows

### Temporal Metrics (Legacy - from issue_metrics)
- `resolution_time_days`: Days from creation to resolution
- `development_duration_days`: Days from first to last commit
- `days_to_first_commit`: Days from issue creation to first commit

## Research Questions (Data Ready)

**Phase 1 Complete - Features Engineered:**

1. **RQ1:** To what extent do code churn metrics correlate with sprint velocity degradation?
   - ✅ 10 churn features calculated
   - Key correlation: `total_churn` vs `is_degraded`

2. **RQ2:** Can developer collaboration network characteristics predict velocity changes?
   - ✅ 17 network features calculated
   - Key findings: density (0.030), centralization (0.708), turnover (24.3%)

3. **RQ3:** What is the relative importance of social vs technical metrics?
   - ✅ Feature variants ready: Model A (churn), Model B (network), Model C (combined)
   - Top correlations: velocity_change (-0.636), prev_velocity (+0.363)

4. **RQ4:** How accurately can ML models predict degradation vs baselines?
   - ✅ Dataset ready: 111 samples, 47 features, balanced classes
   - 🚀 Next: Implement baseline + ML models

**Phase 2 Tasks - Predictive Modeling:**
- [ ] Baseline models (Naive, Previous Velocity, Moving Average)
- [ ] ML models (Logistic Regression, Random Forest, XGBoost, SVM)
- [ ] Time-series cross-validation
- [ ] SHAP feature importance analysis

## Data Quality

### Data Collection (Phase 0)
- **Jira Issues:** 18,823 total (9,552 with fix_versions)
- **GitHub Commits:** 16,887 total
- **Commit Coverage:** ~64% reference Jira issues (10,881 links)
- **Date Range:** 2011-07-19 to 2026-01-26
- **File Types:** Java, Scala, Python, JavaScript, Kotlin
- **Developer Pairs:** 101,681 collaboration edges
- **File Authors:** 104,647 file-author relationships

### Feature Engineering (Phase 1) ✅
- **Releases Analyzed:** 111 (filtered: ≥3 issues per release)
- **Features Generated:** 47 total
- **Missing Values:** 1 (0.9% - commits_per_day)
- **Infinite Values:** 0
- **Zero Variance:** 0 features
- **Class Balance:** 1:1.27 (excellent for ML)
- **Highly Correlated Pairs:** 5 (will handle in modeling)

### Top Correlations with Target (is_degraded)
**Positive (degradation predictors):**
- prev_weighted_velocity: +0.363
- nodes_removed: +0.326

**Negative (stability predictors):**
- velocity_change: -0.636
- bugs: -0.461
- weighted_velocity: -0.448
- issue_count: -0.428

## Pipeline Execution Times

- **fetch_jira_issues.py**: ~5-10 minutes
- **fetch_github_issues.py**: ~2-3 hours (full Kafka history)
- **link_jira_github.py**: ~5-10 minutes
- **feature_engineering.py**: ~7 minutes ✨
- **validate_features.py**: <1 minute

## Notes

- All timestamps are in UTC
- Some issues have multiple `fix_versions` (creates multiple rows in release metrics)
- Commits may reference multiple Jira issues
- Backdated commits (commit before issue creation) are flagged
- Network construction uses **7-day temporal windows** for co-commit relationships
- Velocity degradation threshold: **-10%** (configurable)
- Gini coefficient: 0 = perfect equality, 1 = perfect inequality

## Quick Reference

### For Data Collection (Already Complete)
```bash
python fetch_jira_issues.py      # Step 1: Get Jira issues
python fetch_github_issues.py    # Step 2: Mine Git history
python link_jira_github.py       # Step 3: Link data sources
```

### For Feature Engineering (Phase 1 Complete) ✅
```bash
python feature_engineering.py    # Generate 47 ML features
python validate_features.py      # Check data quality
```

### For Modeling (Phase 2 - Next Steps)
```bash
# To be implemented:
python baseline_models.py        # Naive, Previous Velocity, Moving Avg
python ml_models.py              # Logistic, RF, XGBoost, SVM
python evaluate_models.py        # Cross-validation, metrics, SHAP
```

### Key Files to Use
- **For ML modeling:** `dataset/ml_dataset_YYYYMMDD_HHMMSS.csv`
- **For exploration:** `PHASE1_SUMMARY.md`
- **For validation:** `dataset/validation_summary.csv`

## Troubleshooting

**Q: Feature engineering is slow?**
- The network calculation processes 111 releases with temporal windows
- Expected runtime: ~7 minutes
- Most time spent on network metrics (optimized with sliding window)

**Q: Missing values in commits_per_day?**
- Occurs for 1 release with 0-duration (same day release)
- Safe to fill with median or 0

**Q: High correlation between features?**
- 5 pairs with correlation > 0.95 (e.g., total_churn ↔ total_commits)
- Will use feature selection or regularization in modeling

## License

This project analyzes publicly available data from:
- Apache Kafka (Apache License 2.0)
- Apache Jira (Public issues)

Data collection scripts are provided for research purposes.
