import heapq
import math
import time
from typing import List, Tuple, Dict, Set
import os
import csv
from collections import deque

# Custom Node class for A* with tie-breaking
class CustomNode:
    def __init__(self, x: int, y: int, g: float = 0, h: float = 0, parent=None, order: int = 0):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.order = order  # For FIFO tie-breaking

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        return self.order < other.order

# A* Implementation with custom heuristics and metrics tracking
class CustomAStar:
    def __init__(self, grid: List[List[int]], heuristic_type: str = "manhattan"):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0
        self.heuristic_type = heuristic_type
        self.nodes_expanded = 0
        self.node_order_counter = 0

    def manhattan_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def euclidean_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        return math.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)

    def get_heuristic(self, node: Tuple[int, int], goal: Tuple[int, int]) -> float:
        if self.heuristic_type == "manhattan":
            return self.manhattan_distance(node, goal)
        elif self.heuristic_type == "euclidean":
            return self.euclidean_distance(node, goal)
        else:
            raise ValueError("Unknown heuristic type")

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # No diagonal movement
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.height and 0 <= ny < self.width and self.grid[nx][ny] == 1:
                neighbors.append((nx, ny))
        return neighbors

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], int]:
        self.nodes_expanded = 0
        self.node_order_counter = 0
        
        open_set: List[CustomNode] = []
        closed_set: Set[Tuple[int, int]] = set()
        g_scores: Dict[Tuple[int, int], float] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        start_node = CustomNode(start[0], start[1], 0, self.get_heuristic(start, goal), None, self.node_order_counter)
        heapq.heappush(open_set, start_node)

        while open_set:
            current_node = heapq.heappop(open_set)
            current_pos = (current_node.x, current_node.y)

            if current_pos == goal:
                path = self.reconstruct_path(came_from, goal)
                return path, self.nodes_expanded

            closed_set.add(current_pos)
            self.nodes_expanded += 1

            for neighbor in self.get_neighbors(current_pos):
                if neighbor in closed_set:
                    continue

                tentative_g = g_scores[current_pos] + 1

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current_pos
                    g_scores[neighbor] = tentative_g
                    h_score = self.get_heuristic(neighbor, goal)
                    self.node_order_counter += 1
                    neighbor_node = CustomNode(neighbor[0], neighbor[1], tentative_g, h_score, current_pos, self.node_order_counter)
                    heapq.heappush(open_set, neighbor_node)

        return [], self.nodes_expanded  # No path found

    def reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = []
        current = goal
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]

# Load a map file
def load_map(file_path: str) -> List[List[int]]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (type, height, width, map)
    grid = []
    expected_width = 512  # Since maps are supposed to be 512x512
    for line in lines[4:]:  # Map data starts after header
        row = []
        line = line.strip()
        for char in line:
            if char == '@':
                row.append(0)  # Obstacle
            elif char == '.':
                row.append(1)  # Free space
        # Pad or trim row to ensure consistent width
        if len(row) < expected_width:
            row.extend([0] * (expected_width - len(row)))  # Pad with obstacles
        elif len(row) > expected_width:
            row = row[:expected_width]  # Trim to expected width
        if row:  # Only append non-empty rows
            grid.append(row)
    
    # Ensure we have 512 rows
    height = len(grid)
    if height < 512:
        grid.extend([[0] * expected_width for _ in range(512 - height)])
    elif height > 512:
        grid = grid[:512]
    
    print(f"Loaded map {file_path}: {len(grid)}x{len(grid[0])}")
    return grid

# Load scenarios from a .scen file
def load_scenarios(file_path: str) -> List[Tuple[int, int, int, int, int]]:
    scenarios = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header line ("version 1.0")
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 9:
            bucket = int(parts[0])  # First column is "Bucket"
            start_x = int(parts[4])
            start_y = int(parts[5])
            goal_x = int(parts[6])
            goal_y = int(parts[7])
            scenarios.append((bucket, start_x, start_y, goal_x, goal_y))
    return scenarios

# Calculate obstacle density
def calculate_obstacle_density(grid: List[List[int]]) -> float:
    total_cells = len(grid) * len(grid[0])
    obstacle_cells = sum(row.count(0) for row in grid)
    return obstacle_cells / total_cells

# Main experiment runner
def run_experiment():
    # Define map and scenario directories
    base_dir = "benchmark data"
    map_dirs = ["maze-map", "room-map"]
    scen_dirs = ["maze-scen", "room-scen"]
    
    # Results storage
    results = []
    target_successful_runs = 50  # We want 50 successful runs per map per heuristic
    num_repetitions = 10

    for map_dir, scen_dir in zip(map_dirs, scen_dirs):
        map_dir_path = os.path.join(base_dir, map_dir)
        scen_dir_path = os.path.join(base_dir, scen_dir)
        
        # Process each map file
        for map_file in sorted(os.listdir(map_dir_path)):
            if not map_file.endswith(".map"):
                continue
                
            map_path = os.path.join(map_dir_path, map_file)
            scen_file = map_file + ".scen"
            scen_path = os.path.join(scen_dir_path, scen_file)
            
            if not os.path.exists(scen_path):
                print(f"Scenario file not found for {map_file}")
                continue
            
            # Load map and scenarios
            try:
                grid = load_map(map_path)
            except Exception as e:
                print(f"Error loading map {map_file}: {e}")
                continue
                
            height = len(grid)
            width = len(grid[0])
            scenarios = load_scenarios(scen_path)
            density = calculate_obstacle_density(grid)
            
            print(f"Processing {map_file} with density {density:.2f}")
            
            # Process scenarios until we get 50 successful runs for each heuristic
            successful_runs = {"manhattan": 0, "euclidean": 0}
            scen_idx = 0
            
            while (scen_idx < len(scenarios) and
                   (successful_runs["manhattan"] < target_successful_runs or
                    successful_runs["euclidean"] < target_successful_runs)):
                bucket, start_x, start_y, goal_x, goal_y = scenarios[scen_idx]
                start = (start_x, start_y)
                goal = (goal_x, goal_y)
                
                # Check bounds before accessing grid
                if (start_x < 0 or start_x >= height or
                    start_y < 0 or start_y >= width or
                    goal_x < 0 or goal_x >= height or
                    goal_y < 0 or goal_y >= width):
                    print(f"Skipping scenario {scen_idx} in {map_file}: Start ({start_x},{start_y}) or Goal ({goal_x},{goal_y}) out of bounds ({height}x{width})")
                    scen_idx += 1
                    continue
                
                # Check if start or goal is on an obstacle
                if grid[start_x][start_y] == 0 or grid[goal_x][goal_y] == 0:
                    print(f"Skipping scenario {scen_idx} in {map_file}: Start or Goal on obstacle")
                    scen_idx += 1
                    continue
                
                # Run for both heuristics if needed
                for heuristic in ["manhattan", "euclidean"]:
                    if successful_runs[heuristic] >= target_successful_runs:
                        continue
                    
                    # Initialize metrics
                    total_runtime = 0
                    total_nodes_expanded = 0
                    success = True
                    
                    # Run repetitions
                    for _ in range(num_repetitions):
                        a_star = CustomAStar(grid, heuristic_type=heuristic)
                        start_time = time.time()
                        path, nodes_expanded = a_star.find_path(start, goal)
                        end_time = time.time()
                        
                        runtime = (end_time - start_time) * 1000  # Convert to milliseconds
                        total_runtime += runtime
                        total_nodes_expanded += nodes_expanded
                        
                        if not path:  # No path found
                            success = False
                            break
                    
                    if success:
                        # Compute averages
                        avg_runtime = total_runtime / num_repetitions
                        avg_nodes_expanded = total_nodes_expanded / num_repetitions
                        
                        # Increment successful runs for this heuristic
                        successful_runs[heuristic] += 1
                        
                        # Store result
                        results.append({
                            "map": map_file,
                            "bucket": bucket,
                            "scenario": scen_idx,
                            "start_x": start_x,
                            "start_y": start_y,
                            "heuristic": heuristic,
                            "density": density,
                            "avg_runtime_ms": avg_runtime,
                            "avg_nodes_expanded": avg_nodes_expanded,
                            "success": success
                        })
                    else:
                        print(f"Scenario {scen_idx} in {map_file} failed with {heuristic}: No path found")
                
                scen_idx += 1
            
            if successful_runs["manhattan"] < target_successful_runs or successful_runs["euclidean"] < target_successful_runs:
                print(f"Warning: Not enough valid scenarios for {map_file}. Manhattan: {successful_runs['manhattan']}, Euclidean: {successful_runs['euclidean']}")
            else:
                print(f"Completed {map_file}: {successful_runs['manhattan']} successful Manhattan runs, {successful_runs['euclidean']} successful Euclidean runs")
    
    # Save results to CSV
    output_file = "results.csv"
    with open(output_file, 'w', newline='') as f:
        fieldnames = ["map", "bucket", "scenario", "start_x", "start_y", "heuristic", "density", "avg_runtime_ms", "avg_nodes_expanded", "success"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    run_experiment()