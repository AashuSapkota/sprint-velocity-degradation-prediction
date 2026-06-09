#!/usr/bin/env python3
"""
Feature Engineering for Sprint Velocity Degradation Prediction
Implements all features from the research proposal
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_latest_file(directory, pattern):
    """Find the most recent file matching pattern"""
    files = list(Path(directory).glob(pattern))
    return str(max(files, key=os.path.getctime)) if files else None


def load_datasets():
    """Load all required datasets"""
    print("="*70)
    print("LOADING DATASETS")
    print("="*70)
    
    # Find latest files
    files = {
        'issues': find_latest_file(OUTPUT_DIR, 'kafka_issues_*.csv'),
        'commits': find_latest_file(OUTPUT_DIR, 'commits_*.csv'),
        'file_authors': find_latest_file(OUTPUT_DIR, 'file_authors_*.csv'),
        'developer_pairs': find_latest_file(OUTPUT_DIR, 'developer_pairs_*.csv'),
        'commit_issue_links': find_latest_file(OUTPUT_DIR, 'commit_issue_links_*.csv'),
        'issue_metrics': find_latest_file(OUTPUT_DIR, 'issue_metrics_*.csv'),
    }
    
    # Check for missing files
    missing = [k for k, v in files.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    
    # Load data
    print("\nLoading files...")
    df_issues = pd.read_csv(files['issues'])
    df_commits = pd.read_csv(files['commits'])
    df_file_authors = pd.read_csv(files['file_authors'])
    df_developer_pairs = pd.read_csv(files['developer_pairs'])
    df_commit_links = pd.read_csv(files['commit_issue_links'])
    df_issue_metrics = pd.read_csv(files['issue_metrics'])
    
    # Parse dates
    df_issues['created'] = pd.to_datetime(df_issues['created'], utc=True)
    df_issues['resolved'] = pd.to_datetime(df_issues['resolved'], utc=True)
    df_commits['date'] = pd.to_datetime(df_commits['date'], utc=True)
    df_file_authors['date'] = pd.to_datetime(df_file_authors['date'], utc=True)
    df_commit_links['commit_date'] = pd.to_datetime(df_commit_links['commit_date'], utc=True)
    df_issue_metrics['created'] = pd.to_datetime(df_issue_metrics['created'], utc=True)
    df_issue_metrics['resolved'] = pd.to_datetime(df_issue_metrics['resolved'], utc=True)
    
    print(f"✓ Loaded {len(df_issues):,} issues")
    print(f"✓ Loaded {len(df_commits):,} commits")
    print(f"✓ Loaded {len(df_file_authors):,} file-author records")
    print(f"✓ Loaded {len(df_developer_pairs):,} developer pairs")
    print(f"✓ Loaded {len(df_commit_links):,} commit-issue links")
    print(f"✓ Loaded {len(df_issue_metrics):,} issue metrics")
    
    return df_issues, df_commits, df_file_authors, df_developer_pairs, df_commit_links, df_issue_metrics


# =============================================================================
# RELEASE CYCLE DEFINITION & VELOCITY CALCULATION
# =============================================================================

def extract_release_boundaries(df_issues):
    """
    Extract release cycles from issue fix_versions
    Returns DataFrame with release name, start_date, end_date
    """
    print("\n" + "="*70)
    print("EXTRACTING RELEASE BOUNDARIES")
    print("="*70)
    
    # Get all issues with fix_versions
    versioned = df_issues[df_issues['fix_versions'].notna()].copy()
    print(f"\nIssues with fix_versions: {len(versioned):,} / {len(df_issues):,}")
    
    # Expand issues with multiple versions
    release_issues = []
    for _, row in versioned.iterrows():
        versions = [v.strip() for v in row['fix_versions'].split(',') if v.strip()]
        for version in versions:
            release_issues.append({
                'release': version,
                'issue_key': row['key'],
                'created': row['created'],
                'resolved': row['resolved'],
                'type': row['type'],
                'priority': row['priority'],
            })
    
    df_release_issues = pd.DataFrame(release_issues)
    
    # Calculate release boundaries based on issue resolution dates
    release_bounds = df_release_issues.groupby('release').agg({
        'issue_key': 'count',
        'resolved': ['min', 'max'],
        'created': 'min'
    }).reset_index()
    
    release_bounds.columns = ['release', 'num_issues', 'start_date', 'end_date', 'earliest_issue_created']
    
    # Sort by end date
    release_bounds = release_bounds.sort_values('end_date')
    release_bounds = release_bounds[release_bounds['num_issues'] >= 3]  # Filter out tiny releases
    
    # Calculate release duration
    release_bounds['duration_days'] = (
        release_bounds['end_date'] - release_bounds['start_date']
    ).dt.total_seconds() / 86400
    
    print(f"\n✓ Identified {len(release_bounds)} releases")
    print(f"  Date range: {release_bounds['start_date'].min()} to {release_bounds['end_date'].max()}")
    print(f"  Issues per release: {release_bounds['num_issues'].median():.0f} (median)")
    
    return release_bounds, df_release_issues


def calculate_velocity_metrics(release_bounds, df_release_issues):
    """
    Calculate velocity per release with weighted issue counts
    """
    print("\n" + "="*70)
    print("CALCULATING VELOCITY METRICS")
    print("="*70)
    
    # Define issue weights based on type and priority
    type_weights = {
        'Bug': 1.0,
        'Improvement': 1.5,
        'New Feature': 2.0,
        'Task': 1.0,
        'Sub-task': 0.5,
        'Test': 0.5,
    }
    
    priority_weights = {
        'Blocker': 2.0,
        'Critical': 1.5,
        'Major': 1.0,
        'Minor': 0.5,
        'Trivial': 0.3,
    }
    
    # Calculate weighted velocity per release
    velocity_data = []
    
    for release in release_bounds['release']:
        release_issues = df_release_issues[df_release_issues['release'] == release]
        
        # Count by type
        type_counts = release_issues['type'].value_counts().to_dict()
        
        # Calculate weighted velocity
        total_weight = 0
        for _, issue in release_issues.iterrows():
            type_weight = type_weights.get(issue['type'], 1.0)
            priority_weight = priority_weights.get(issue['priority'], 1.0)
            total_weight += type_weight * priority_weight
        
        velocity_data.append({
            'release': release,
            'issue_count': len(release_issues),
            'weighted_velocity': total_weight,
            'bugs': type_counts.get('Bug', 0),
            'improvements': type_counts.get('Improvement', 0),
            'new_features': type_counts.get('New Feature', 0),
            'tasks': type_counts.get('Task', 0),
        })
    
    df_velocity = pd.DataFrame(velocity_data)
    
    # Merge with release bounds
    df_velocity = release_bounds.merge(df_velocity, on='release')
    
    # Calculate velocity degradation
    df_velocity = df_velocity.sort_values('end_date').reset_index(drop=True)
    df_velocity['prev_weighted_velocity'] = df_velocity['weighted_velocity'].shift(1)
    df_velocity['velocity_change'] = df_velocity['weighted_velocity'] - df_velocity['prev_weighted_velocity']
    df_velocity['velocity_change_pct'] = (
        df_velocity['velocity_change'] / df_velocity['prev_weighted_velocity'] * 100
    )
    
    # Binary classification: degradation if velocity drops by >10%
    df_velocity['is_degraded'] = (df_velocity['velocity_change_pct'] < -10).astype(int)
    
    # Add velocity per day (normalized by duration)
    df_velocity['velocity_per_day'] = df_velocity['weighted_velocity'] / df_velocity['duration_days']
    
    print(f"\n✓ Calculated velocity for {len(df_velocity)} releases")
    print(f"  Degraded releases: {df_velocity['is_degraded'].sum()} ({df_velocity['is_degraded'].mean()*100:.1f}%)")
    print(f"  Stable/improving: {(1-df_velocity['is_degraded']).sum()}")
    print(f"\nVelocity statistics:")
    print(f"  Mean velocity: {df_velocity['weighted_velocity'].mean():.1f}")
    print(f"  Mean change: {df_velocity['velocity_change_pct'].mean():.1f}%")
    
    return df_velocity


# =============================================================================
# ADVANCED CODE CHURN FEATURES
# =============================================================================

def calculate_gini_coefficient(values):
    """Calculate Gini coefficient for inequality measurement"""
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n


def calculate_churn_features(df_velocity, df_commits, df_commit_links):
    """
    Calculate advanced code churn features per release
    """
    print("\n" + "="*70)
    print("CALCULATING ADVANCED CHURN FEATURES")
    print("="*70)
    
    churn_features = []
    
    for idx, release in df_velocity.iterrows():
        release_name = release['release']
        start_date = release['start_date']
        end_date = release['end_date']
        
        # Get commits for this release (based on linked issues)
        release_issues = df_commit_links[
            df_commit_links['jira_key'].str.startswith('KAFKA-')
        ]
        
        # Alternative: use commit dates within release window
        release_commits = df_commits[
            (df_commits['date'] >= start_date) & 
            (df_commits['date'] <= end_date)
        ]
        
        if len(release_commits) == 0:
            churn_features.append({
                'release': release_name,
                'total_commits': 0,
                'total_churn': 0,
                'churn_concentration': 0,
                'developer_churn_inequality': 0,
                'refactoring_ratio': 0,
                'file_hotspots': 0,
                'avg_commit_size': 0,
                'churn_volatility': 0,
            })
            continue
        
        # Total churn
        total_churn = release_commits['churn'].sum()
        
        # Churn concentration (Gini coefficient of churn per commit)
        churn_per_commit = release_commits['churn'].values
        churn_concentration = calculate_gini_coefficient(churn_per_commit)
        
        # Developer churn inequality (Gini coefficient of commits per developer)
        commits_per_dev = release_commits.groupby('author_name').size().values
        developer_inequality = calculate_gini_coefficient(commits_per_dev)
        
        # Refactoring ratio: commits with deletion/insertion > 0.8
        release_commits_copy = release_commits.copy()
        release_commits_copy['del_ins_ratio'] = np.where(
            release_commits_copy['insertions'] > 0,
            release_commits_copy['deletions'] / release_commits_copy['insertions'],
            0
        )
        refactoring_ratio = (release_commits_copy['del_ins_ratio'] > 0.8).mean()
        
        # File hotspots: number of files in 90th percentile of changes
        # This requires file-level data - approximate using commits with >90th percentile churn
        churn_90th = release_commits['churn'].quantile(0.90)
        file_hotspots = (release_commits['churn'] > churn_90th).sum()
        
        # Average commit size
        avg_commit_size = release_commits['churn'].mean()
        
        # Churn volatility: std dev of daily churn
        daily_churn = release_commits.groupby(release_commits['date'].dt.date)['churn'].sum()
        churn_volatility = daily_churn.std() if len(daily_churn) > 1 else 0
        
        churn_features.append({
            'release': release_name,
            'total_commits': len(release_commits),
            'total_churn': total_churn,
            'churn_concentration': churn_concentration,
            'developer_churn_inequality': developer_inequality,
            'refactoring_ratio': refactoring_ratio,
            'file_hotspots': file_hotspots,
            'avg_commit_size': avg_commit_size,
            'churn_volatility': churn_volatility,
            'unique_developers': release_commits['author_name'].nunique(),
            'avg_files_per_commit': release_commits['num_files'].mean(),
        })
    
    df_churn = pd.DataFrame(churn_features)
    
    print(f"\n✓ Calculated churn features for {len(df_churn)} releases")
    print(f"\nChurn feature statistics:")
    print(f"  Mean churn concentration (Gini): {df_churn['churn_concentration'].mean():.3f}")
    print(f"  Mean developer inequality: {df_churn['developer_churn_inequality'].mean():.3f}")
    print(f"  Mean refactoring ratio: {df_churn['refactoring_ratio'].mean():.3f}")
    
    return df_churn


# =============================================================================
# DEVELOPER NETWORK FEATURES
# =============================================================================

def build_temporal_cocommit_network(df_file_authors, start_date, end_date, window_days=7):
    """
    Build co-commit network for a specific time window
    Uses temporal window: developers who modified same file within window_days
    Optimized version using vectorized operations
    """
    # Filter commits in time range
    period_commits = df_file_authors[
        (df_file_authors['date'] >= start_date) &
        (df_file_authors['date'] <= end_date)
    ].copy()
    
    if len(period_commits) == 0:
        return nx.Graph()
    
    # Build co-commit network
    G = nx.Graph()
    
    # Add all developers as nodes
    developers = period_commits['author'].unique()
    G.add_nodes_from(developers)
    
    # Optimized approach: group by file and use time windows
    edge_weights = defaultdict(int)
    
    for file, group in period_commits.groupby('file'):
        # Convert to sorted list for efficient window search
        file_commits = group.sort_values('date').reset_index(drop=True)
        dates = file_commits['date'].values
        authors = file_commits['author'].values
        
        # Use sliding window approach
        n = len(file_commits)
        for i in range(n):
            dev1 = authors[i]
            date1 = dates[i]
            
            # Only look ahead within the window
            for j in range(i + 1, n):
                dev2 = authors[j]
                date2 = dates[j]
                
                # Calculate time difference in days
                time_diff = (date2 - date1) / np.timedelta64(1, 'D')
                
                # Stop if beyond window
                if time_diff > window_days:
                    break
                
                # Add edge if different developers
                if dev1 != dev2:
                    pair = tuple(sorted([dev1, dev2]))
                    edge_weights[pair] += 1
    
    # Add edges to graph
    for (dev1, dev2), weight in edge_weights.items():
        G.add_edge(dev1, dev2, weight=weight)
    
    return G


def calculate_network_metrics(G):
    """Calculate comprehensive network metrics for a graph"""
    if len(G) == 0:
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'density': 0,
            'avg_clustering': 0,
            'diameter': 0,
            'num_components': 0,
            'giant_component_size': 0,
            'avg_degree': 0,
            'max_betweenness': 0,
            'degree_centralization': 0,
            'avg_closeness': 0,
        }
    
    # Basic structure
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Clustering
    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0
    
    # Diameter (only for connected graphs)
    try:
        if nx.is_connected(G):
            diameter = nx.diameter(G)
        else:
            # Use largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            diameter = nx.diameter(subgraph) if len(subgraph) > 1 else 0
    except:
        diameter = 0
    
    # Components
    num_components = nx.number_connected_components(G)
    components = list(nx.connected_components(G))
    giant_component_size = len(max(components, key=len)) / num_nodes if components else 0
    
    # Centrality measures
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values())) if degrees else 0
    
    # Betweenness centrality (expensive, sample if large graph)
    try:
        if num_nodes > 100:
            # Sample for large graphs
            k = min(100, num_nodes)
            betweenness = nx.betweenness_centrality(G, k=k)
        else:
            betweenness = nx.betweenness_centrality(G)
        max_betweenness = max(betweenness.values()) if betweenness else 0
    except:
        max_betweenness = 0
    
    # Degree centralization (Gini coefficient of degrees)
    degree_values = np.array(list(degrees.values()))
    degree_centralization = calculate_gini_coefficient(degree_values)
    
    # Closeness centrality
    try:
        if nx.is_connected(G):
            closeness = nx.closeness_centrality(G)
            avg_closeness = np.mean(list(closeness.values()))
        else:
            avg_closeness = 0
    except:
        avg_closeness = 0
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_clustering': avg_clustering,
        'diameter': diameter,
        'num_components': num_components,
        'giant_component_size': giant_component_size,
        'avg_degree': avg_degree,
        'max_betweenness': max_betweenness,
        'degree_centralization': degree_centralization,
        'avg_closeness': avg_closeness,
    }


def calculate_network_change(G_prev, G_curr):
    """Calculate change metrics between two networks"""
    if len(G_prev) == 0 or len(G_curr) == 0:
        return {
            'network_jaccard': 0,
            'nodes_added': len(G_curr),
            'nodes_removed': 0,
            'density_change': 0,
            'centralization_change': 0,
            'turnover_rate': 0,
        }
    
    # Edge Jaccard similarity
    edges_prev = set(G_prev.edges())
    edges_curr = set(G_curr.edges())
    
    if len(edges_prev) == 0 and len(edges_curr) == 0:
        network_jaccard = 1.0
    else:
        intersection = len(edges_prev & edges_curr)
        union = len(edges_prev | edges_curr)
        network_jaccard = intersection / union if union > 0 else 0
    
    # Node changes
    nodes_prev = set(G_prev.nodes())
    nodes_curr = set(G_curr.nodes())
    nodes_added = len(nodes_curr - nodes_prev)
    nodes_removed = len(nodes_prev - nodes_curr)
    
    # Density change
    density_change = nx.density(G_curr) - nx.density(G_prev)
    
    # Centralization change
    degrees_prev = np.array(list(dict(G_prev.degree()).values()))
    degrees_curr = np.array(list(dict(G_curr.degree()).values()))
    centralization_prev = calculate_gini_coefficient(degrees_prev)
    centralization_curr = calculate_gini_coefficient(degrees_curr)
    centralization_change = centralization_curr - centralization_prev
    
    # Turnover rate (proportion of new developers)
    turnover_rate = nodes_added / len(nodes_curr) if len(nodes_curr) > 0 else 0
    
    return {
        'network_jaccard': network_jaccard,
        'nodes_added': nodes_added,
        'nodes_removed': nodes_removed,
        'density_change': density_change,
        'centralization_change': centralization_change,
        'turnover_rate': turnover_rate,
    }


def calculate_network_features(df_velocity, df_file_authors):
    """
    Calculate developer network features per release
    """
    print("\n" + "="*70)
    print("CALCULATING DEVELOPER NETWORK FEATURES")
    print("="*70)
    print("This may take several minutes for large networks...")
    
    network_features = []
    prev_graph = None
    
    for idx, release in df_velocity.iterrows():
        release_name = release['release']
        start_date = release['start_date']
        end_date = release['end_date']
        
        print(f"\n  Processing {release_name}...")
        
        # Build temporal co-commit network
        G = build_temporal_cocommit_network(df_file_authors, start_date, end_date, window_days=7)
        
        # Calculate network metrics
        metrics = calculate_network_metrics(G)
        metrics['release'] = release_name
        
        # Calculate network change if we have previous network
        if prev_graph is not None:
            change_metrics = calculate_network_change(prev_graph, G)
            metrics.update(change_metrics)
        else:
            # First release - no change metrics
            metrics.update({
                'network_jaccard': 0,
                'nodes_added': metrics['num_nodes'],
                'nodes_removed': 0,
                'density_change': 0,
                'centralization_change': 0,
                'turnover_rate': 0,
            })
        
        network_features.append(metrics)
        prev_graph = G
    
    df_network = pd.DataFrame(network_features)
    
    print(f"\n✓ Calculated network features for {len(df_network)} releases")
    print(f"\nNetwork feature statistics:")
    print(f"  Mean network density: {df_network['density'].mean():.3f}")
    print(f"  Mean degree centralization: {df_network['degree_centralization'].mean():.3f}")
    print(f"  Mean turnover rate: {df_network['turnover_rate'].mean():.3f}")
    
    return df_network


# =============================================================================
# FINAL DATASET CONSTRUCTION
# =============================================================================

def build_final_dataset(df_velocity, df_churn, df_network):
    """
    Merge all features into final ML-ready dataset
    """
    print("\n" + "="*70)
    print("BUILDING FINAL ML-READY DATASET")
    print("="*70)
    
    # Merge all features
    df_final = df_velocity.copy()
    df_final = df_final.merge(df_churn, on='release', how='left')
    df_final = df_final.merge(df_network, on='release', how='left')
    
    # Fill missing values
    df_final = df_final.fillna(0)
    
    # Add additional derived features
    df_final['commits_per_day'] = df_final['total_commits'] / df_final['duration_days']
    df_final['churn_per_developer'] = np.where(
        df_final['unique_developers'] > 0,
        df_final['total_churn'] / df_final['unique_developers'],
        0
    )
    df_final['edges_per_node'] = np.where(
        df_final['num_nodes'] > 0,
        df_final['num_edges'] / df_final['num_nodes'],
        0
    )
    
    # Reorder columns for clarity
    id_cols = ['release', 'start_date', 'end_date', 'duration_days']
    target_cols = ['weighted_velocity', 'velocity_change_pct', 'is_degraded']
    velocity_cols = ['issue_count', 'bugs', 'improvements', 'new_features', 'tasks', 
                     'velocity_per_day', 'prev_weighted_velocity']
    churn_cols = [c for c in df_final.columns if c in df_churn.columns and c != 'release']
    network_cols = [c for c in df_final.columns if c in df_network.columns and c != 'release']
    derived_cols = ['commits_per_day', 'churn_per_developer', 'edges_per_node']
    
    ordered_cols = id_cols + target_cols + velocity_cols + churn_cols + network_cols + derived_cols
    remaining_cols = [c for c in df_final.columns if c not in ordered_cols]
    df_final = df_final[ordered_cols + remaining_cols]
    
    print(f"\n✓ Built final dataset")
    print(f"  Releases: {len(df_final)}")
    print(f"  Features: {len(df_final.columns)}")
    print(f"  Target variable: is_degraded")
    print(f"  Degraded samples: {df_final['is_degraded'].sum()} ({df_final['is_degraded'].mean()*100:.1f}%)")
    
    # Print feature summary
    print(f"\nFeature categories:")
    print(f"  Velocity features: {len(velocity_cols)}")
    print(f"  Churn features: {len(churn_cols)}")
    print(f"  Network features: {len(network_cols)}")
    print(f"  Derived features: {len(derived_cols)}")
    
    return df_final


def save_features(df_final, df_velocity, df_churn, df_network):
    """Save all feature datasets"""
    print("\n" + "="*70)
    print("SAVING FEATURE DATASETS")
    print("="*70)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    datasets = [
        (df_final, 'ml_dataset', "Final ML-ready dataset with all features"),
        (df_velocity, 'velocity_features', "Velocity and degradation labels"),
        (df_churn, 'churn_features', "Advanced code churn features"),
        (df_network, 'network_features', "Developer network features"),
    ]
    
    saved_files = []
    for df, name, description in datasets:
        filename = os.path.join(OUTPUT_DIR, f'{name}_{timestamp}.csv')
        df.to_csv(filename, index=False)
        saved_files.append(filename)
        print(f"✓ {description}")
        print(f"  → {filename}")
    
    return saved_files


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main feature engineering pipeline"""
    print("\n" + "="*70)
    print("FEATURE ENGINEERING PIPELINE")
    print("Sprint Velocity Degradation Prediction")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Load datasets
        df_issues, df_commits, df_file_authors, df_developer_pairs, df_commit_links, df_issue_metrics = load_datasets()
        
        # Phase 1: Release boundaries and velocity
        release_bounds, df_release_issues = extract_release_boundaries(df_issues)
        df_velocity = calculate_velocity_metrics(release_bounds, df_release_issues)
        
        # Phase 2: Advanced churn features
        df_churn = calculate_churn_features(df_velocity, df_commits, df_commit_links)
        
        # Phase 3: Developer network features
        df_network = calculate_network_features(df_velocity, df_file_authors)
        
        # Phase 4: Build final dataset
        df_final = build_final_dataset(df_velocity, df_churn, df_network)
        
        # Save all datasets
        saved_files = save_features(df_final, df_velocity, df_churn, df_network)
        
        # Final summary
        print("\n" + "="*70)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*70)
        print(f"\n✓ Successfully created {len(saved_files)} dataset files")
        print(f"✓ Final dataset: {len(df_final)} releases × {len(df_final.columns)} features")
        print(f"✓ Ready for machine learning modeling")
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display sample
        print("\n" + "="*70)
        print("SAMPLE OF FINAL DATASET (First 5 releases)")
        print("="*70)
        print(df_final[['release', 'weighted_velocity', 'velocity_change_pct', 
                        'is_degraded', 'total_churn', 'density', 'turnover_rate']].head())
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
