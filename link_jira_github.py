#!/usr/bin/env python3
"""
Link Jira Issues with GitHub Commits
Creates comprehensive dataset for Sprint Velocity Degradation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path

OUTPUT_DIR = 'dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DIR = 'dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_latest_file(directory, pattern):
    """Find the most recent file matching pattern"""
    files = list(Path(directory).glob(pattern))
    return str(max(files, key=os.path.getctime)) if files else None


def load_all_data(jira_file=None, commits_file=None, file_authors_file=None, 
                  developer_pairs_file=None, auto_detect=True):
    """Load all data files with auto-detection"""
    print("="*70)
    print("LOADING DATA FILES")
    print("="*70)
    
    if auto_detect:
        print("\nAuto-detecting data files...")
        
        if not jira_file:
            jira_file = find_latest_file(OUTPUT_DIR, 'kafka_issues_*.csv')
            if jira_file:
                print(f"  Found Jira file: {jira_file}")
        
        if not commits_file:
            commits_file = find_latest_file(OUTPUT_DIR, 'commits_*.csv')
            if commits_file:
                print(f"  Found commits file: {commits_file}")
        
        if not file_authors_file:
            file_authors_file = find_latest_file(OUTPUT_DIR, 'file_authors_*.csv')
            if file_authors_file:
                print(f"  Found file_authors file: {file_authors_file}")
        
        if not developer_pairs_file:
            developer_pairs_file = find_latest_file(OUTPUT_DIR, 'developer_pairs_*.csv')
            if developer_pairs_file:
                print(f"  Found developer_pairs file: {developer_pairs_file}")
    
    # Validate required files
    if not jira_file or not os.path.exists(jira_file):
        raise FileNotFoundError(f"Jira issues file not found: {jira_file}")
    if not commits_file or not os.path.exists(commits_file):
        raise FileNotFoundError(f"Commits file not found: {commits_file}")
    
    # Load Jira issues
    print(f"\nLoading Jira issues from: {jira_file}")
    df_issues = pd.read_csv(jira_file)
    df_issues['created'] = pd.to_datetime(df_issues['created'], utc=True)
    df_issues['resolved'] = pd.to_datetime(df_issues['resolved'], utc=True)
    print(f"  ✓ Loaded {len(df_issues):,} issues")
    
    # Load commits
    print(f"\nLoading commits from: {commits_file}")
    df_commits = pd.read_csv(commits_file)
    df_commits['date'] = pd.to_datetime(df_commits['date'], utc=True)
    print(f"  ✓ Loaded {len(df_commits):,} commits")
    
    # Load optional files
    df_file_authors = None
    if file_authors_file and os.path.exists(file_authors_file):
        print(f"\nLoading file authors from: {file_authors_file}")
        df_file_authors = pd.read_csv(file_authors_file)
        df_file_authors['date'] = pd.to_datetime(df_file_authors['date'], utc=True)
        print(f"  ✓ Loaded {len(df_file_authors):,} file-author records")
    
    df_developer_pairs = None
    if developer_pairs_file and os.path.exists(developer_pairs_file):
        print(f"\nLoading developer pairs from: {developer_pairs_file}")
        df_developer_pairs = pd.read_csv(developer_pairs_file)
        print(f"  ✓ Loaded {len(df_developer_pairs):,} developer pairs")
    
    return df_issues, df_commits, df_file_authors, df_developer_pairs

# =============================================================================
# JIRA-COMMIT LINKING
# =============================================================================

def link_commits_to_issues(df_commits, df_issues):
    """
    Link commits to Jira issues based on commit messages
    
    Returns: DataFrame with commit-issue links including all metrics
    """
    print("\n" + "="*70)
    print("LINKING COMMITS TO JIRA ISSUES")
    print("="*70)
    
    # Expand commits with multiple Jira keys
    commit_issue_links = []
    
    print("\nProcessing commit-issue references...")
    for idx, commit in df_commits.iterrows():
        if idx % 5000 == 0 and idx > 0:
            print(f"  Processed {idx:,} commits...")
        
        if pd.notna(commit['jira_keys']) and commit['jira_keys']:
            jira_keys = commit['jira_keys'].split(',')
            for key in jira_keys:
                commit_issue_links.append({
                    'commit_hash': commit['hash'],
                    'jira_key': key.strip(),
                    'commit_date': commit['date'],
                    'commit_timestamp': commit['timestamp'],
                    'author_name': commit['author_name'],
                    'author_email': commit['author_email'],
                    'insertions': commit['insertions'],
                    'deletions': commit['deletions'],
                    'churn': commit['churn'],
                    'num_files': commit['num_files'],
                })
    
    df_links = pd.DataFrame(commit_issue_links)
    
    if df_links.empty:
        print("\n⚠ WARNING: No commit-issue links found!")
        print("  This means commits don't reference Jira issues in their messages")
        return df_links
    
    print(f"\n✓ Created {len(df_links):,} commit-issue links")
    print(f"  Unique commits: {df_links['commit_hash'].nunique():,}")
    print(f"  Unique issues referenced: {df_links['jira_key'].nunique():,}")
    
    # Merge with issue data to enrich
    print("\nEnriching with Jira metadata...")
    df_links = df_links.merge(
        df_issues[['key', 'created', 'resolved', 'status', 'resolution', 
                   'priority', 'type', 'fix_versions', 'assignee', 'reporter']],
        left_on='jira_key',
        right_on='key',
        how='left'
    )
    
    # Calculate temporal metrics
    df_links['issue_created'] = pd.to_datetime(df_links['created'], utc=True)
    df_links['issue_resolved'] = pd.to_datetime(df_links['resolved'], utc=True)
    df_links['commit_date'] = pd.to_datetime(df_links['commit_date'], utc=True)
    
    # Days from issue creation to commit
    df_links['days_after_issue_created'] = (
        df_links['commit_date'] - df_links['issue_created']
    ).dt.total_seconds() / 86400
    
    # Days from commit to issue resolution
    df_links['days_before_issue_resolved'] = (
        df_links['issue_resolved'] - df_links['commit_date']
    ).dt.total_seconds() / 86400
    
    # Flag commits made before issue was created (backdated references)
    df_links['commit_before_issue'] = df_links['days_after_issue_created'] < 0
    
    # Drop redundant columns
    df_links = df_links.drop(['key', 'created', 'resolved'], axis=1)
    
    # Summary statistics
    print(f"\n✓ Enrichment complete")
    print(f"  Issues with matched metadata: {df_links['issue_created'].notna().sum():,}")
    print(f"  Commits before issue creation: {df_links['commit_before_issue'].sum():,}")
    
    return df_links

# =============================================================================
# COVERAGE ANALYSIS
# =============================================================================

def analyze_coverage(df_commits, df_issues, df_links):
    """Analyze how well issues are covered by commits"""
    print("\n" + "="*70)
    print("COVERAGE ANALYSIS")
    print("="*70)
    
    # Issue coverage
    issues_with_commits = set(df_links['jira_key'].dropna().unique())
    total_issues = len(df_issues)
    resolved_issues = df_issues['resolved'].notna().sum()
    
    print(f"\nIssue Coverage:")
    print(f"  Total issues: {total_issues:,}")
    print(f"  Resolved issues: {resolved_issues:,}")
    print(f"  Issues with commits: {len(issues_with_commits):,}")
    print(f"  Coverage of all issues: {len(issues_with_commits)/total_issues*100:.1f}%")
    print(f"  Coverage of resolved issues: {len(issues_with_commits)/max(resolved_issues,1)*100:.1f}%")
    
    # Commit coverage
    commits_with_jira = df_commits['has_jira_ref'].sum()
    total_commits = len(df_commits)
    
    print(f"\nCommit Coverage:")
    print(f"  Total commits: {total_commits:,}")
    print(f"  Commits with Jira refs: {commits_with_jira:,}")
    print(f"  Coverage: {commits_with_jira/total_commits*100:.1f}%")
    
    # Commits per issue
    if not df_links.empty:
        commits_per_issue = df_links.groupby('jira_key')['commit_hash'].nunique()
        
        print(f"\nCommits per Issue:")
        print(f"  Mean: {commits_per_issue.mean():.2f}")
        print(f"  Median: {commits_per_issue.median():.0f}")
        print(f"  Max: {commits_per_issue.max():.0f}")
        print(f"  75th percentile: {commits_per_issue.quantile(0.75):.0f}")
        
        # Issues by type with commits
        print(f"\nIssues with commits by type:")
        type_counts = df_links.groupby('type')['jira_key'].nunique().sort_values(ascending=False)
        for issue_type, count in type_counts.items():
            total_of_type = len(df_issues[df_issues['type'] == issue_type])
            print(f"  {issue_type}: {count:,} / {total_of_type:,} ({count/max(total_of_type,1)*100:.1f}%)")

# =============================================================================
# ISSUE-LEVEL AGGREGATION
# =============================================================================

def aggregate_issue_metrics(df_links):
    """
    Aggregate commit-level metrics to issue level
    
    Returns: DataFrame with one row per issue containing aggregated metrics
    """
    print("\n" + "="*70)
    print("AGGREGATING METRICS BY ISSUE")
    print("="*70)
    
    if df_links.empty:
        print("⚠ No links to aggregate")
        return pd.DataFrame()
    
    print("\nCalculating per-issue metrics...")
    
    # Group by issue
    issue_metrics = df_links.groupby('jira_key').agg({
        # Commit counts
        'commit_hash': 'nunique',
        
        # Churn metrics
        'insertions': 'sum',
        'deletions': 'sum',
        'churn': 'sum',
        'num_files': 'sum',
        
        # Developer metrics
        'author_name': 'nunique',
        
        # Temporal metrics
        'commit_date': ['min', 'max'],
        'days_after_issue_created': 'min',
        'days_before_issue_resolved': 'min',
        
        # Issue metadata (take first since they're the same)
        'issue_created': 'first',
        'issue_resolved': 'first',
        'status': 'first',
        'resolution': 'first',
        'priority': 'first',
        'type': 'first',
        'fix_versions': 'first',
        'assignee': 'first',
        'reporter': 'first',
    }).reset_index()
    
    # Flatten column names
    issue_metrics.columns = [
        'jira_key', 'num_commits', 'total_insertions', 'total_deletions',
        'total_churn', 'total_files_changed', 'num_developers',
        'first_commit_date', 'last_commit_date',
        'days_to_first_commit', 'days_from_last_commit_to_resolution',
        'created', 'resolved', 'status', 'resolution', 'priority', 
        'type', 'fix_versions', 'assignee', 'reporter'
    ]
    
    # Calculate development duration
    issue_metrics['development_duration_days'] = (
        pd.to_datetime(issue_metrics['last_commit_date'], utc=True) - 
        pd.to_datetime(issue_metrics['first_commit_date'], utc=True)
    ).dt.total_seconds() / 86400
    
    # Calculate resolution time
    issue_metrics['resolution_time_days'] = (
        pd.to_datetime(issue_metrics['resolved'], utc=True) - 
        pd.to_datetime(issue_metrics['created'], utc=True)
    ).dt.total_seconds() / 86400
    
    # Churn per commit
    issue_metrics['avg_churn_per_commit'] = (
        issue_metrics['total_churn'] / issue_metrics['num_commits']
    )
    
    # Files per commit
    issue_metrics['avg_files_per_commit'] = (
        issue_metrics['total_files_changed'] / issue_metrics['num_commits']
    )
    
    print(f"\n✓ Aggregated metrics for {len(issue_metrics):,} issues")
    print(f"\nMetric ranges:")
    print(f"  Commits per issue: {issue_metrics['num_commits'].min():.0f} - {issue_metrics['num_commits'].max():.0f}")
    print(f"  Total churn per issue: {issue_metrics['total_churn'].min():.0f} - {issue_metrics['total_churn'].max():.0f}")
    print(f"  Developers per issue: {issue_metrics['num_developers'].min():.0f} - {issue_metrics['num_developers'].max():.0f}")
    print(f"  Development duration: {issue_metrics['development_duration_days'].min():.1f} - {issue_metrics['development_duration_days'].max():.1f} days")
    
    return issue_metrics

# =============================================================================
# DEVELOPER NETWORK METRICS
# =============================================================================

def calculate_developer_network_metrics(df_file_authors, df_developer_pairs, 
                                        df_issue_metrics):
    """
    Calculate developer network metrics per issue
    
    Returns: DataFrame with network metrics for each issue
    """
    print("\n" + "="*70)
    print("CALCULATING DEVELOPER NETWORK METRICS")
    print("="*70)
    
    if df_file_authors is None or df_developer_pairs is None:
        print("⚠ Network data not available")
        return None
    
    print("\nThis feature requires additional implementation...")
    print("Network metrics to be calculated:")
    print("  - Developer centrality per issue")
    print("  - Team cohesion metrics")
    print("  - Collaboration patterns")
    
    # Placeholder for now
    return None

# =============================================================================
# RELEASE GROUPING
# =============================================================================

def group_by_releases(df_issue_metrics):
    """
    Group issues by release (fix_versions)
    
    Returns: DataFrame with per-release aggregated metrics
    """
    print("\n" + "="*70)
    print("GROUPING BY RELEASES")
    print("="*70)
    
    if df_issue_metrics.empty:
        print("⚠ No issue metrics to group")
        return pd.DataFrame()
    
    # Filter issues with fix_versions
    versioned = df_issue_metrics[df_issue_metrics['fix_versions'].notna()].copy()
    
    print(f"\nIssues with fix_versions: {len(versioned):,} / {len(df_issue_metrics):,}")
    
    if versioned.empty:
        print("⚠ No issues have fix_versions - cannot group by release")
        return pd.DataFrame()
    
    # For issues with multiple versions, create separate rows
    expanded_rows = []
    for _, row in versioned.iterrows():
        versions = [v.strip() for v in row['fix_versions'].split(',')]
        for version in versions:
            row_copy = row.copy()
            row_copy['release'] = version
            expanded_rows.append(row_copy)
    
    df_expanded = pd.DataFrame(expanded_rows)
    
    print(f"Total issue-release pairs: {len(df_expanded):,}")
    print(f"Unique releases: {df_expanded['release'].nunique()}")
    
    # Aggregate by release
    release_metrics = df_expanded.groupby('release').agg({
        'jira_key': 'count',
        'num_commits': 'sum',
        'total_churn': 'sum',
        'num_developers': 'sum',
        'resolution_time_days': 'mean',
        'development_duration_days': 'mean',
        'created': 'min',
        'resolved': 'max',
    }).reset_index()
    
    release_metrics.columns = [
        'release', 'num_issues', 'total_commits', 'total_churn',
        'total_developers', 'avg_resolution_time', 'avg_development_duration',
        'release_start', 'release_end'
    ]
    
    # Calculate release duration
    release_metrics['release_duration_days'] = (
        pd.to_datetime(release_metrics['release_end'], utc=True) - 
        pd.to_datetime(release_metrics['release_start'], utc=True)
    ).dt.total_seconds() / 86400
    
    # Velocity metrics
    release_metrics['issues_per_day'] = (
        release_metrics['num_issues'] / release_metrics['release_duration_days']
    )
    
    release_metrics['commits_per_day'] = (
        release_metrics['total_commits'] / release_metrics['release_duration_days']
    )
    
    # Sort by release start
    release_metrics = release_metrics.sort_values('release_start')
    
    print(f"\n✓ Aggregated metrics for {len(release_metrics)} releases")
    print(f"\nRelease ranges:")
    print(f"  Issues per release: {release_metrics['num_issues'].min():.0f} - {release_metrics['num_issues'].max():.0f}")
    print(f"  Velocity range: {release_metrics['issues_per_day'].min():.2f} - {release_metrics['issues_per_day'].max():.2f} issues/day")
    
    return release_metrics

# =============================================================================
# FINAL DATASET CREATION
# =============================================================================

def create_final_dataset(df_issue_metrics, df_commits, df_links):
    """
    Create the final comprehensive dataset for analysis
    
    Combines:
    - Issue-level metrics
    - Historical trend data
    - Release groupings
    """
    print("\n" + "="*70)
    print("CREATING FINAL DATASET")
    print("="*70)
    
    # Add temporal features
    df_final = df_issue_metrics.copy()
    
    # Extract year, quarter, month
    created_utc = pd.to_datetime(df_final['created'], utc=True)
    df_final['year'] = created_utc.dt.year
    df_final['quarter'] = created_utc.dt.quarter
    df_final['month'] = created_utc.dt.month
    
    # Add issue age (days from creation to now)
    df_final['issue_age_days'] = (
        pd.Timestamp.now(tz='UTC') - created_utc
    ).dt.total_seconds() / 86400
    
    # Binary flags
    df_final['is_resolved'] = df_final['resolved'].notna()
    df_final['is_bug'] = df_final['type'] == 'Bug'
    df_final['is_high_priority'] = df_final['priority'].isin(['Critical', 'Blocker', 'Major'])
    df_final['multi_developer'] = df_final['num_developers'] > 1
    
    # Sort by creation date
    df_final = df_final.sort_values('created')
    
    print(f"\n✓ Final dataset created with {len(df_final):,} issues")
    print(f"  Features: {len(df_final.columns)} columns")
    print(f"  Date range: {df_final['created'].min()} to {df_final['created'].max()}")
    
    return df_final

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("JIRA-GITHUB LINKAGE - FINAL DATASET CREATION")
    print("="*70)
    print("\nPurpose: Create comprehensive dataset for Sprint Velocity")
    print()
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("\nUsage:")
        print("  python link_jira_github_final.py [jira_csv] [commits_csv]")
        print("\nOr let the script auto-detect files:")
        print("  python link_jira_github_final.py")
        print("\nThe script will automatically find the latest files in:")
        print("  - Current directory for kafka_issues_*.csv")
        print("  - dataset/ for commits_*.csv and other data")
        return
    
    # Load data (auto-detect or use provided paths)
    jira_file = sys.argv[1] if len(sys.argv) > 1 else None
    commits_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        df_issues, df_commits, df_file_authors, df_developer_pairs = load_all_data(
            jira_file=jira_file,
            commits_file=commits_file,
            auto_detect=True
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you have:")
        print("  1. kafka_issues_*.csv in dataset/ directory")
        print("  2. commits_*.csv in dataset/ directory")
        print("\nRun with --help for more information")
        return
    
    # Link commits to issues
    df_links = link_commits_to_issues(df_commits, df_issues)
    
    # Coverage analysis
    analyze_coverage(df_commits, df_issues, df_links)
    
    # Aggregate to issue level
    df_issue_metrics = aggregate_issue_metrics(df_links)
    
    # Group by releases
    df_release_metrics = group_by_releases(df_issue_metrics)
    
    # Create final dataset
    df_final = create_final_dataset(df_issue_metrics, df_commits, df_links)
    
    # Save outputs
    print("\n" + "="*70)
    print("SAVING OUTPUT FILES")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    files_saved = []
    
    if not df_links.empty:
        filename = os.path.join(OUTPUT_DIR, f'commit_issue_links_{timestamp}.csv')
        df_links.to_csv(filename, index=False)
        print(f"\n✓ Commit-Issue Links: {filename}")
        print(f"  {len(df_links):,} rows (commit-level detail)")
        files_saved.append(filename)
    
    if not df_issue_metrics.empty:
        filename = os.path.join(OUTPUT_DIR, f'issue_metrics_{timestamp}.csv')
        df_issue_metrics.to_csv(filename, index=False)
        print(f"\n✓ Issue Metrics: {filename}")
        print(f"  {len(df_issue_metrics):,} rows (issue-level aggregation)")
        files_saved.append(filename)
    
    if not df_release_metrics.empty:
        filename = os.path.join(OUTPUT_DIR, f'release_metrics_{timestamp}.csv')
        df_release_metrics.to_csv(filename, index=False)
        print(f"\n✓ Release Metrics: {filename}")
        print(f"  {len(df_release_metrics):,} rows (per-release aggregation)")
        files_saved.append(filename)
    
    if not df_final.empty:
        filename = os.path.join(OUTPUT_DIR, f'final_dataset_{timestamp}.csv')
        df_final.to_csv(filename, index=False)
        print(f"\n✓ FINAL DATASET: {filename}")
        print(f"  {len(df_final):,} rows × {len(df_final.columns)} columns")
        print(f"  Ready for analysis!")
        files_saved.append(filename)
    
    # Final summary
    print("\n" + "="*70)
    print("DATASET CREATION COMPLETE ✓")
    print("="*70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print(f"📊 Files created: {len(files_saved)}")
    print(f"\n🎯 PRIMARY DATASET:")
    print(f"   final_dataset_{timestamp}.csv")
    print(f"   → {len(df_final):,} issues × {len(df_final.columns)} features")
    print(f"\n📈 NEXT STEPS:")
    print("   1. Load final_dataset_*.csv")
    print("   2. Define release/sprint boundaries")
    print("   3. Calculate velocity metrics per sprint")
    print("   4. Build predictive models for velocity degradation")
    print("\n" + "="*70)
    
    # Quick stats of datasets
    if not df_final.empty:
        print("\nQUICK STATS:")
        print(f"  - Date range: {df_final['created'].min().date()} to {df_final['created'].max().date()}")
        print(f"  - Resolved issues: {df_final['is_resolved'].sum():,} ({df_final['is_resolved'].mean()*100:.1f}%)")
        print(f"  - Bugs: {df_final['is_bug'].sum():,} ({df_final['is_bug'].mean()*100:.1f}%)")
        print(f"  - Multi-developer issues: {df_final['multi_developer'].sum():,} ({df_final['multi_developer'].mean()*100:.1f}%)")
        print(f"  - Avg commits per issue: {df_final['num_commits'].mean():.2f}")
        print(f"  - Avg churn per issue: {df_final['total_churn'].mean():.0f} lines")
        print(f"  - Avg resolution time: {df_final['resolution_time_days'].mean():.1f} days")
    
    print("\n✅ All done! Dataset is ready for analysis.")

if __name__ == '__main__':
    main()