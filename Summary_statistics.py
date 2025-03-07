import pandas as pd
from scipy.stats import pearsonr
import sys
from contextlib import contextmanager

# Create a custom context manager to handle output
@contextmanager
def output_to_file_and_console(filename):
    class FileAndConsole:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.file = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.file.write(message)
            
        def flush(self):
            self.terminal.flush()
            self.file.flush()
    
    original_stdout = sys.stdout
    sys.stdout = FileAndConsole(filename)
    try:
        yield
    finally:
        sys.stdout.file.close()
        sys.stdout = original_stdout

# Main analysis code wrapped in the context manager
with output_to_file_and_console('Summary_statistics.txt'):
    # Load the dataset with outlier flags
    df = pd.read_csv('Outlier_detection_results')

    # Ensure columns are numeric
    df['avg_runtime_ms'] = pd.to_numeric(df['avg_runtime_ms'], errors='coerce')
    df['avg_nodes_expanded'] = pd.to_numeric(df['avg_nodes_expanded'], errors='coerce')
    df['density'] = pd.to_numeric(df['density'], errors='coerce')

    # Step 1: Overall Summary Statistics
    overall_summary = df.groupby('heuristic').agg({
        'avg_runtime_ms': ['mean', 'std', 'median', 'min', 'max'],
        'avg_nodes_expanded': ['mean', 'std', 'median', 'min', 'max']
    }).reset_index()
    print("Overall Summary Statistics:\n", overall_summary)

    # Step 2: Summary Statistics by Grouping (Dataset Type, Density Level, Heuristic)
    # Ensure dataset_type and density_level are defined (as in previous code)
    df['dataset_type'] = df['map'].apply(lambda x: 'mazes' if 'maze' in x.lower() else 'rooms')
    def assign_density_level(map_name):
        if 'maze512-32' in map_name or '32room' in map_name:
            return 'low'
        elif 'maze512-16' in map_name or '16room' in map_name:
            return 'medium'
        elif 'maze512-8' in map_name or '8room' in map_name:
            return 'high'
        else:
            return 'unknown'
    df['density_level'] = df['map'].apply(assign_density_level)

    grouped_summary = df.groupby(['dataset_type', 'density_level', 'heuristic']).agg({
        'avg_runtime_ms': ['mean', 'std', 'median'],
        'avg_nodes_expanded': ['mean', 'std', 'median'],
        'map': 'count'  # Count of scenarios
    }).reset_index()

    # Flatten column names
    grouped_summary.columns = ['dataset_type', 'density_level', 'heuristic',
                               'mean_runtime_ms', 'std_runtime_ms', 'median_runtime_ms',
                               'mean_nodes_expanded', 'std_nodes_expanded', 'median_nodes_expanded',
                               'scenario_count']
    print("\nGrouped Summary Statistics:\n", grouped_summary)

    # Calculate percentage difference between heuristics within each group
    # Pivot to compare Manhattan vs. Euclidean
    pivot_runtime = grouped_summary.pivot_table(index=['dataset_type', 'density_level'],
                                                columns='heuristic',
                                                values='mean_runtime_ms').reset_index()
    pivot_runtime['percent_diff'] = ((pivot_runtime['manhattan'] - pivot_runtime['euclidean']) / pivot_runtime['euclidean']) * 100

    pivot_nodes = grouped_summary.pivot_table(index=['dataset_type', 'density_level'],
                                              columns='heuristic',
                                              values='mean_nodes_expanded').reset_index()
    pivot_nodes['percent_diff'] = ((pivot_nodes['manhattan'] - pivot_nodes['euclidean']) / pivot_nodes['euclidean']) * 100

    print("\nPercentage Difference in Runtime:\n", pivot_runtime)
    print("\nPercentage Difference in Nodes Expanded:\n", pivot_nodes)

    # Step 3: Summary Statistics by Map (Optional)
    map_summary = df.groupby(['map', 'heuristic']).agg({
        'avg_runtime_ms': ['mean', 'std'],
        'avg_nodes_expanded': ['mean', 'std']
    }).reset_index()
    map_summary.columns = ['map', 'heuristic', 'mean_runtime_ms', 'std_runtime_ms', 'mean_nodes_expanded', 'std_nodes_expanded']
    print("\nSummary by Map:\n", map_summary)

    # Step 4: Outlier Impact Statistics
    # With outliers
    grouped_with_outliers = df.groupby(['dataset_type', 'density_level', 'heuristic']).agg({
        'avg_runtime_ms': 'mean',
        'avg_nodes_expanded': 'mean'
    }).reset_index()

    # Without outliers
    no_outliers = df[(df['runtime_outlier'] == False) & (df['nodes_outlier'] == False)]
    grouped_no_outliers = no_outliers.groupby(['dataset_type', 'density_level', 'heuristic']).agg({
        'avg_runtime_ms': 'mean',
        'avg_nodes_expanded': 'mean'
    }).reset_index()

    # Merge and calculate percentage change
    outlier_impact = grouped_with_outliers.merge(grouped_no_outliers, on=['dataset_type', 'density_level', 'heuristic'], suffixes=('_with', '_without'))
    outlier_impact['runtime_percent_change'] = ((outlier_impact['avg_runtime_ms_with'] - outlier_impact['avg_runtime_ms_without']) / outlier_impact['avg_runtime_ms_with']) * 100
    outlier_impact['nodes_percent_change'] = ((outlier_impact['avg_nodes_expanded_with'] - outlier_impact['avg_nodes_expanded_without']) / outlier_impact['avg_nodes_expanded_with']) * 100
    print("\nOutlier Impact:\n", outlier_impact)

    # Step 5: Correlation Statistics
    # Remove any NaN values for correlation
    corr_data = df[['density', 'avg_runtime_ms', 'avg_nodes_expanded']].dropna()

    # Pearson's correlation
    pearson_density_runtime, _ = pearsonr(corr_data['density'], corr_data['avg_runtime_ms'])
    pearson_density_nodes, _ = pearsonr(corr_data['density'], corr_data['avg_nodes_expanded'])
    pearson_runtime_nodes, _ = pearsonr(corr_data['avg_runtime_ms'], corr_data['avg_nodes_expanded'])

    print("\nCorrelation Statistics:")
    print(f"Pearson: Density vs Runtime: {pearson_density_runtime:.3f}")
    print(f"Pearson: Density vs Nodes Expanded: {pearson_density_nodes:.3f}")
    print(f"Pearson: Runtime vs Nodes Expanded: {pearson_runtime_nodes:.3f}")

    print("\nFinalizing output...")

    # Save summary for use in EE
    grouped_summary.to_csv('Summary_statistics.txt', index=False)
    print("\nSummary statistics saved as 'Summary_statistics.txt'")