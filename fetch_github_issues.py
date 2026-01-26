#!/usr/bin/env python3
"""
GitHub Data Collection for Sprint Velocity Degradation
Collects commits, file-author relationships, and co-change patterns
"""

from pydriller import Repository
import pandas as pd
import re
from datetime import datetime
from collections import defaultdict
import os

REPO_URL = 'https://github.com/apache/kafka'
OUTPUT_DIR = 'dataset'
CHECKPOINT_DIR = 'checkpoints'
CODE_EXTENSIONS = ('.java', '.scala', '.py', '.js', '.kt')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def extract_jira_keys(text):
    """Extract KAFKA-XXXX from commit message"""
    return list(set(re.findall(r'KAFKA-\d+', text or '')))


def save_checkpoint(df, filename):
    """Save progress checkpoint"""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    df.to_csv(filepath, index=False)
    return filepath


def collect_commits(max_commits=None, checkpoint_interval=500):
    """Collect commits with churn metrics, author info, and file changes"""
    print("="*70)
    print("COLLECTING COMMIT DATA")
    print("="*70)
    print(f"Repository: {REPO_URL}")
    print(f"This will take 2-3 hours for full Kafka history\n")
    
    commits_data = []
    file_author_mapping = []
    commit_files_mapping = []
    commit_count = 0
    
    for commit in Repository(REPO_URL).traverse_commits():
        commit_count += 1
        jira_keys = extract_jira_keys(commit.msg)
        
        commits_data.append({
            'hash': commit.hash,
            'author_name': commit.author.name,
            'author_email': commit.author.email,
            'date': commit.author_date.isoformat(),
            'timestamp': commit.author_date.timestamp(),
            'msg': commit.msg[:300],
            'insertions': commit.insertions,
            'deletions': commit.deletions,
            'churn': commit.insertions + commit.deletions,
            'num_files': len(commit.modified_files),
            'jira_keys': ','.join(jira_keys) if jira_keys else None,
            'has_jira_ref': len(jira_keys) > 0,
        })
        
        modified_files = []
        for mf in commit.modified_files:
            if mf.filename.endswith(CODE_EXTENSIONS):
                modified_files.append(mf.filename)
                file_author_mapping.append({
                    'commit_hash': commit.hash,
                    'file': mf.filename,
                    'author': commit.author.name,
                    'date': commit.author_date.isoformat(),
                    'additions': mf.added_lines,
                    'deletions': mf.deleted_lines,
                })
        
        if len(modified_files) > 1:
            commit_files_mapping.append({
                'commit_hash': commit.hash,
                'files': '|'.join(sorted(modified_files)),
                'num_files': len(modified_files),
                'author': commit.author.name,
                'date': commit.author_date.isoformat(),
            })
        
        if commit_count % 100 == 0:
            print(f"  Processed {commit_count} commits...")
        
        if commit_count % checkpoint_interval == 0:
            save_checkpoint(pd.DataFrame(commits_data), 'commits_checkpoint.csv')
            print(f"  ✓ Checkpoint saved at {commit_count} commits")
        
        if max_commits and commit_count >= max_commits:
            break
    
    df_commits = pd.DataFrame(commits_data)
    df_file_authors = pd.DataFrame(file_author_mapping)
    df_file_cochanges = pd.DataFrame(commit_files_mapping)
    
    print(f"\n{'='*70}")
    print("COLLECTION SUMMARY")
    print(f"{'='*70}")
    print(f"Total commits: {len(df_commits):,}")
    print(f"Date range: {df_commits['date'].min()} to {df_commits['date'].max()}")
    print(f"Unique authors: {df_commits['author_name'].nunique()}")
    print(f"Commits with Jira refs: {df_commits['has_jira_ref'].sum():,} ({df_commits['has_jira_ref'].mean()*100:.1f}%)")
    print(f"Total churn: {df_commits['churn'].sum():,} lines")
    print(f"File-author relationships: {len(df_file_authors):,}")
    print(f"Multi-file commits: {len(df_file_cochanges):,}")
    
    return df_commits, df_file_authors, df_file_cochanges


def build_developer_pairs(df_file_authors):
    """Build co-commit network from file-author data"""
    print(f"\n{'='*70}")
    print("BUILDING DEVELOPER PAIRS")
    print(f"{'='*70}")
    
    file_authors = df_file_authors.groupby('file')['author'].apply(list).to_dict()
    developer_pairs = defaultdict(int)
    
    for authors in file_authors.values():
        unique_authors = list(set(authors))
        for i in range(len(unique_authors)):
            for j in range(i + 1, len(unique_authors)):
                pair = tuple(sorted([unique_authors[i], unique_authors[j]]))
                developer_pairs[pair] += 1
    
    pairs_data = [{
        'developer1': pair[0],
        'developer2': pair[1],
        'shared_files': count,
    } for pair, count in developer_pairs.items()]
    
    df_pairs = pd.DataFrame(pairs_data).sort_values('shared_files', ascending=False)
    
    print(f"Developer pairs: {len(df_pairs):,}")
    if not df_pairs.empty:
        print(f"Top pair: {df_pairs.iloc[0]['developer1']} <-> {df_pairs.iloc[0]['developer2']}")
        print(f"  Shared files: {df_pairs.iloc[0]['shared_files']}")
    
    return df_pairs


def build_file_coupling(df_file_cochanges):
    """Build file coupling network from co-change data"""
    print(f"\n{'='*70}")
    print("BUILDING FILE COUPLING")
    print(f"{'='*70}")
    
    file_pairs = defaultdict(int)
    
    for _, row in df_file_cochanges.iterrows():
        files = row['files'].split('|')
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                pair = tuple(sorted([files[i], files[j]]))
                file_pairs[pair] += 1
    
    pairs_data = [{
        'file1': pair[0],
        'file2': pair[1],
        'cochange_count': count,
    } for pair, count in file_pairs.items()]
    
    df_cochanges = pd.DataFrame(pairs_data).sort_values('cochange_count', ascending=False)
    
    print(f"File pairs: {len(df_cochanges):,}")
    if not df_cochanges.empty:
        print(f"Top pair: {df_cochanges.iloc[0]['file1']} <-> {df_cochanges.iloc[0]['file2']}")
        print(f"  Co-changed: {df_cochanges.iloc[0]['cochange_count']} times")
    
    return df_cochanges


def save_datasets(df_commits, df_file_authors, df_file_cochanges, df_developer_pairs, df_coupling_pairs):
    """Save all datasets to CSV files"""
    print(f"\n{'='*70}")
    print("SAVING DATASETS")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    files_saved = []
    
    datasets = [
        (df_commits, 'commits', "Commit data with churn metrics"),
        (df_file_authors, 'file_authors', "File-author relationships"),
        (df_file_cochanges, 'file_cochanges', "File co-change patterns"),
        (df_developer_pairs, 'developer_pairs', "Developer collaboration pairs"),
        (df_coupling_pairs, 'file_coupling', "File coupling pairs"),
    ]
    
    for df, name, description in datasets:
        filename = os.path.join(OUTPUT_DIR, f'{name}_{timestamp}.csv')
        df.to_csv(filename, index=False)
        print(f"✓ {description}: {filename}")
        print(f"  {len(df):,} rows")
        files_saved.append(filename)
    
    return files_saved


def main():
    """Main execution"""
    print("="*70)
    print("DATA COLLECTION - GITHUB COMMITS")
    print("="*70)
    print("\nCollecting:")
    print("  ✓ Commits (churn metrics)")
    print("  ✓ File-author relationships")
    print("  ✓ File co-changes")

    
    df_commits, df_file_authors, df_file_cochanges = collect_commits(max_commits=None)
    df_developer_pairs = build_developer_pairs(df_file_authors)
    df_coupling_pairs = build_file_coupling(df_file_cochanges)
    files_saved = save_datasets(df_commits, df_file_authors, df_file_cochanges, 
                                df_developer_pairs, df_coupling_pairs)
    
    print(f"\n{'='*70}")
    print("COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Files created: {len(files_saved)}")
    print("\n✓ All commits with churn metrics")
    print("✓ Developer collaboration network data")
    print("✓ File coupling network data")
    print("✓ Jira linking keys")
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
