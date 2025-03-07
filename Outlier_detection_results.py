import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('results.csv')

# Ensure 'density' is numeric (if not already)
df['density'] = pd.to_numeric(df['density'], errors='coerce')

# Step 1: Define a function to detect outliers using the IQR method
def detect_outliers(group, column):
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Flag outliers
    outliers = (group[column] < lower_bound) | (group[column] > upper_bound)
    return outliers

# Step 2: Group data by dataset type, density level, and heuristic
# First, create a 'dataset_type' column to distinguish mazes and rooms
df['dataset_type'] = df['map'].apply(lambda x: 'mazes' if 'maze' in x.lower() else 'rooms')

# Create a 'density_level' column based on map names or density values
# Assuming map names indicate density (e.g., maze512-8-* for high, 32room_* for low)
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

# Step 3: Apply outlier detection within each group for both metrics
df['runtime_outlier'] = False
df['nodes_outlier'] = False

# Group by dataset_type, density_level, and heuristic
grouped = df.groupby(['dataset_type', 'density_level', 'heuristic'])

# Detect outliers for runtime and nodes expanded
for name, group in grouped:
    # Detect outliers for avg_runtime_ms
    runtime_outliers = detect_outliers(group, 'avg_runtime_ms')
    df.loc[group.index, 'runtime_outlier'] = runtime_outliers
    
    # Detect outliers for avg_nodes_expanded
    nodes_outliers = detect_outliers(group, 'avg_nodes_expanded')
    df.loc[group.index, 'nodes_outlier'] = nodes_outliers

# Step 4: Summarize the number of outliers detected
print("Number of runtime outliers:", df['runtime_outlier'].sum())
print("Number of nodes expanded outliers:", df['nodes_outlier'].sum())

# Step 5: Optionally, inspect some outliers
runtime_outliers = df[df['runtime_outlier'] == True][['map', 'heuristic', 'density_level', 'avg_runtime_ms']]
nodes_outliers = df[df['nodes_outlier'] == True][['map', 'heuristic', 'density_level', 'avg_nodes_expanded']]
print("\nSample runtime outliers:\n", runtime_outliers.head())
print("\nSample nodes expanded outliers:\n", nodes_outliers.head())

# Step 6: Save the updated dataset with outlier flags
df.to_csv('results_with_outliers.csv', index=False)
print("Updated dataset saved as 'results_with_outliers.csv'")