#!/usr/bin/env python3
"""
Jira Data Collection for Sprint Velocity Degradation
Fetches all Apache Kafka issues from Jira
"""

from jira import JIRA
import pandas as pd
from datetime import datetime
import os

JIRA_SERVER = 'https://issues.apache.org/jira'
PROJECT_KEY = 'KAFKA'
OUTPUT_DIR = 'dataset'
CHUNK_SIZE = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_issues(max_results=None):
    """Pull all Kafka issues from Jira"""
    print(f"Connecting to {JIRA_SERVER}...")
    jira = JIRA(server=JIRA_SERVER)
    
    issues = []
    start = 0
    
    print(f"Fetching {PROJECT_KEY} issues...")
    
    while True:
        jql = f'project = {PROJECT_KEY} ORDER BY created ASC'
        try:
            chunk = jira.search_issues(
                jql,
                startAt=start,
                maxResults=CHUNK_SIZE,
                fields='key,created,resolutiondate,status,resolution,priority,issuetype,fixVersions,assignee,reporter'
            )
            
            if not chunk:
                break
            
            issues.extend(chunk)
            start += CHUNK_SIZE
            print(f"Fetched {len(issues)} issues...")
            
            if max_results and len(issues) >= max_results:
                break
        except Exception as e:
            print(f"Error at position {start}: {e}")
            break
    
    print(f"Total issues fetched: {len(issues)}")
    return issues


def convert_to_dataframe(issues):
    """Convert Jira issues to pandas DataFrame"""
    print("\nConverting to DataFrame...")
    data = []
    
    for issue in issues:
        data.append({
            'key': issue.key,
            'created': issue.fields.created,
            'resolved': getattr(issue.fields, 'resolutiondate', None),
            'status': issue.fields.status.name if hasattr(issue.fields, 'status') else None,
            'resolution': issue.fields.resolution.name if hasattr(issue.fields, 'resolution') and issue.fields.resolution else None,
            'priority': issue.fields.priority.name if hasattr(issue.fields, 'priority') and issue.fields.priority else None,
            'type': issue.fields.issuetype.name if hasattr(issue.fields, 'issuetype') else None,
            'fix_versions': ', '.join([v.name for v in issue.fields.fixVersions]) if hasattr(issue.fields, 'fixVersions') and issue.fields.fixVersions else None,
            'assignee': issue.fields.assignee.name if hasattr(issue.fields, 'assignee') and issue.fields.assignee else None,
            'reporter': issue.fields.reporter.name if hasattr(issue.fields, 'reporter') and issue.fields.reporter else None,
        })
    
    df = pd.DataFrame(data)
    df['created'] = pd.to_datetime(df['created'])
    df['resolved'] = pd.to_datetime(df['resolved'])
    
    return df


def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total issues: {len(df):,}")
    print(f"\nIssues by status:")
    print(df['status'].value_counts())
    print(f"\nIssues by type:")
    print(df['type'].value_counts())
    print(f"\nIssues by priority:")
    print(df['priority'].value_counts())
    print(f"\nDate range: {df['created'].min()} to {df['created'].max()}")


def main():
    """Main execution"""
    print("="*70)
    print("DATA COLLECTION - JIRA ISSUES")
    print("="*70)
    print()
    
    issues = fetch_issues()
    
    if not issues:
        print("No issues found")
        return
    
    df = convert_to_dataframe(issues)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(OUTPUT_DIR, f'kafka_issues_{timestamp}.csv')
    
    print(f"\nSaving to {filename}...")
    df.to_csv(filename, index=False)
    
    print(f"✓ Successfully saved {len(df):,} issues")
    
    print_summary(df)
    
    print("\n" + "="*70)
    print("COLLECTION COMPLETE")
    print("="*70)
    print(f"Output file: {filename}")
    print("\nNext: Collect GitHub data using fetch_github_issues.py")
    print("="*70)


if __name__ == '__main__':
    main()