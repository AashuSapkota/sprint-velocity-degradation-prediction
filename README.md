# Sprint Velocity Degradation Prediction

**Title:** Predicting Sprint Velocity Degradation in Agile Teams Using Developer Interaction Networks and Code Churn Metrics

**Repository:** Apache Kafka Issue & Commit Data Collection

## Overview

This project collects and analyzes data from the Apache Kafka project to predict sprint velocity degradation using:
- Code churn metrics from GitHub commits
- Developer interaction networks (co-commit patterns)
- File coupling networks (architectural dependencies)
- Jira issue tracking data (velocity metrics)

## Project Structure

```
project/
├── fetch_jira_issues.py      # Collect Jira issues
├── fetch_github_issues.py    # Collect GitHub commits & networks
├── link_jira_github.py       # Link Jira & GitHub data
├── dataset/                  # All output data (auto-created)
├── checkpoints/              # Progress checkpoints (auto-created)
└── README.md                 # This file
```

## Quick Start

### Prerequisites

```bash
pip install pandas numpy jira pydriller
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

**final_dataset_*.csv** ⭐ **PRIMARY DATASET**
- Complete dataset with all features
- Columns: 32 features including:
  - Commit metrics: num_commits, total_churn, avg_churn_per_commit
  - Developer metrics: num_developers, multi_developer
  - Temporal metrics: resolution_time_days, development_duration_days
  - Issue metadata: type, priority, status, fix_versions
  - Time features: year, quarter, month
  - Binary flags: is_resolved, is_bug, is_high_priority
- Use: **START HERE** for analysis

## Key Metrics

### Code Churn Metrics
- `total_churn`: Total lines changed (insertions + deletions)
- `insertions`: Lines added
- `deletions`: Lines removed
- `num_commits`: Number of commits
- `avg_churn_per_commit`: Average churn per commit

### Developer Metrics
- `num_developers`: Unique developers per issue
- `multi_developer`: Flag for multi-developer issues
- Developer network centrality (from network analysis)

### Temporal Metrics
- `resolution_time_days`: Days from creation to resolution
- `development_duration_days`: Days from first to last commit
- `days_to_first_commit`: Days from issue creation to first commit

### Velocity Metrics (Derived)
- Issues completed per sprint
- Average resolution time per sprint
- Total churn per sprint
- Velocity change (%)
- Degradation flag

## Research Questions

1. **RQ1:** Can code churn metrics predict sprint velocity degradation?
2. **RQ2:** Do developer interaction networks correlate with velocity changes?
3. **RQ3:** How do file coupling patterns relate to development velocity?
4. **RQ4:** What combination of metrics best predicts velocity degradation?

## Data Quality

- **Jira Coverage:** ~18,000+ Kafka issues collected
- **Commit Coverage:** ~50-60% of commits reference Jira issues
- **Date Range:** 2011 - Present
- **File Types:** Java, Scala, Python, JavaScript, Kotlin

## Notes

- All timestamps are in UTC
- Some issues have multiple `fix_versions` (creates multiple rows in release metrics)
- Commits may reference multiple Jira issues
- Backdated commits (commit before issue creation) are flagged
- Files in `duplicate/` directory are not modified

## License

This project analyzes publicly available data from:
- Apache Kafka (Apache License 2.0)
- Apache Jira (Public issues)

Data collection scripts are provided for research purposes.
