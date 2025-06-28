import matplotlib.pyplot as plt
import torch
import os
import pickle
import networkx as nx
from scipy.stats import entropy
from community import community_louvain
import time
import numpy as np


def load_trajectories(file_path):
    with open(file_path, "rb") as f:
        trajectories = pickle.load(f)
    return trajectories

trajectories = load_trajectories("./trajectories.pkl")

trajectories = [trajectories[0]] # JJ just taking first trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_rqa_measures(recurrence_matrix, case_name, min_diagonal=2, min_vertical=2, 
                           save_plots=True, plot_dir="plots"):
    """
    Calculate Recurrence Quantification Analysis (RQA) measures from a recurrence matrix.
    
    Parameters:
    -----------
    recurrence_matrix : numpy.ndarray or torch.Tensor
        The recurrence matrix where recurrence_matrix[i,j] = 1 if states i and j are recurrent
    case_name : str
        Name identifier for saving results and plots
    min_diagonal : int, optional
        Minimum length of diagonal lines to consider
    min_vertical : int, optional
        Minimum length of vertical lines to consider
    save_plots : bool, optional
        Whether to save visualization plots
    plot_dir : str, optional
        Directory to save plots in
    
    Returns:
    --------
    dict
        Dictionary containing all calculated RQA measures
    """
    # Ensure recurrence matrix is a PyTorch tensor
    if not isinstance(recurrence_matrix, torch.Tensor):
        recurrence_matrix = torch.tensor(recurrence_matrix, dtype=torch.float32)
    
    # Create plots directory if it doesn't exist
    if save_plots and not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    rqa_measures = {}
    
    print(f"Calculating RQA measures for case: {case_name}")
    
    # Recurrence Rate (RR)
    rr = calculate_recurrence_rate(recurrence_matrix)
    rqa_measures['recurrence_rate'] = rr.item() if hasattr(rr, 'item') else rr
    print(f"Recurrence Rate: {rqa_measures['recurrence_rate']:.4f}")
    
    # Diagonal line structures
    diag_results = analyze_diagonal_lines(recurrence_matrix, min_length=min_diagonal)
    rqa_measures.update(diag_results)
    print(f"Determinism: {diag_results['determinism']:.4f}")
    print(f"Average Diagonal Line Length: {diag_results['avg_diagonal_length']:.4f}")
    print(f"Diagonal Line Entropy: {diag_results['diagonal_entropy']:.4f}")
    
    # Vertical line structures
    vert_results = analyze_vertical_lines(recurrence_matrix, min_length=min_vertical)
    rqa_measures.update(vert_results)
    print(f"Laminarity: {vert_results['laminarity']:.4f}")
    print(f"Trapping Time: {vert_results['trapping_time']:.4f}")
    
    # Fractal dimensions
    try:
        fractal_dim = estimate_fractal_dimension(recurrence_matrix)
        rqa_measures['fractal_dimension'] = fractal_dim
        print(f"Estimated Fractal Dimension: {fractal_dim:.4f}")
    except Exception as e:
        print(f"Could not calculate fractal dimension: {e}")
        rqa_measures['fractal_dimension'] = None
    
    # Save visualization
    if save_plots:
        visualize_rqa(recurrence_matrix, rqa_measures, case_name, plot_dir)
    
    # Save measures to file
    save_rqa_results(rqa_measures, case_name)
    
    return rqa_measures

def calculate_recurrence_rate(recurrence_matrix):
    """Calculate the recurrence rate (density of recurrence points)."""
    if isinstance(recurrence_matrix, torch.Tensor):
        return torch.sum(recurrence_matrix) / (recurrence_matrix.shape[0] * recurrence_matrix.shape[1])
    else:
        return np.sum(recurrence_matrix) / (recurrence_matrix.shape[0] * recurrence_matrix.shape[1])

def analyze_diagonal_lines(recurrence_matrix, min_length=2):
    """
    Analyze diagonal line structures in the recurrence matrix.
    
    Returns:
    --------
    dict
        Dictionary containing determinism, average diagonal line length, and diagonal line entropy
    """
    results = {}
    
    # Convert to numpy for easier diagonal extraction if it's a torch tensor
    if isinstance(recurrence_matrix, torch.Tensor):
        rm_numpy = recurrence_matrix.cpu().numpy()
    else:
        rm_numpy = recurrence_matrix
    
    N = rm_numpy.shape[0]
    
    # Find all diagonal lines and their lengths
    diag_lengths = []
    
    # Check diagonals above and including the main diagonal
    for offset in range(0, N): ## GG maybe just go until floor(N/2) as it is symmetric adn we font care about central diagonal
        diag = np.diag(rm_numpy, k=offset)
        if len(diag) > 0:
            # Find lengths of consecutive 1's
            lengths = find_consecutive_ones(diag)
            diag_lengths.extend([length for length in lengths if length >= min_length])
    
    # Check diagonals below the main diagonal (except the main diagonal which is already counted)
    for offset in range(1, N):
        diag = np.diag(rm_numpy, k=-offset)
        if len(diag) > 0:
            # Find lengths of consecutive 1's
            lengths = find_consecutive_ones(diag)
            diag_lengths.extend([length for length in lengths if length >= min_length])
    
    # Calculate measures
    total_points = np.sum(rm_numpy)
    
    if total_points > 0 and len(diag_lengths) > 0:
        # Determinism: percentage of recurrence points forming diagonal lines
        points_in_diagonals = sum(diag_lengths)
        results['determinism'] = points_in_diagonals / total_points
        
        # Average diagonal line length
        results['avg_diagonal_length'] = np.mean(diag_lengths)
        
        # Longest diagonal line
        results['max_diagonal_length'] = np.max(diag_lengths)
        
        # Entropy of diagonal line lengths
        hist, _ = np.histogram(diag_lengths, bins=range(min(diag_lengths), max(diag_lengths) + 2))
        hist = hist / np.sum(hist)  # Normalize
        results['diagonal_entropy'] = entropy(hist)
        
        # Store the full distribution
        results['diagonal_length_dist'] = diag_lengths
    else:
        results['determinism'] = 0.0
        results['avg_diagonal_length'] = 0.0
        results['max_diagonal_length'] = 0.0
        results['diagonal_entropy'] = 0.0
        results['diagonal_length_dist'] = []
    
    return results

def analyze_vertical_lines(recurrence_matrix, min_length=2):
    """
    Analyze vertical line structures in the recurrence matrix.
    
    Returns:
    --------
    dict
        Dictionary containing laminarity and trapping time
    """
    results = {}
    
    # Convert to numpy if it's a torch tensor
    if isinstance(recurrence_matrix, torch.Tensor):
        rm_numpy = recurrence_matrix.cpu().numpy()
    else:
        rm_numpy = recurrence_matrix
    
    N = rm_numpy.shape[0]
    
    # Find all vertical lines and their lengths
    vert_lengths = []
    
    # Check each column for vertical lines
    for col in range(N):
        column = rm_numpy[:, col]
        # Find lengths of consecutive 1's
        lengths = find_consecutive_ones(column)
        vert_lengths.extend([length for length in lengths if length >= min_length])
    
    # Calculate measures
    total_points = np.sum(rm_numpy)
    
    if total_points > 0 and len(vert_lengths) > 0:
        # Laminarity: percentage of recurrence points forming vertical lines
        points_in_verticals = sum(vert_lengths)
        results['laminarity'] = points_in_verticals / total_points
        
        # Trapping time: average vertical line length
        results['trapping_time'] = np.mean(vert_lengths)
        
        # Longest vertical line
        results['max_vertical_length'] = np.max(vert_lengths)
        
        # Store the full distribution
        results['vertical_length_dist'] = vert_lengths
    else:
        results['laminarity'] = 0.0
        results['trapping_time'] = 0.0
        results['max_vertical_length'] = 0.0
        results['vertical_length_dist'] = []
    
    return results

def find_consecutive_ones(arr):
    """Find lengths of consecutive 1's in a binary array."""
    # Convert to int array if needed
    if isinstance(arr, (list, np.ndarray)):
        arr = np.asarray(arr, dtype=int)
    
    # Find where transitions occur
    transitions = np.diff(np.hstack(([0], arr, [0])))
    # Start indices of consecutive 1's
    starts = np.where(transitions == 1)[0]
    # End indices of consecutive 1's
    ends = np.where(transitions == -1)[0]
    # Lengths of consecutive 1's
    lengths = ends - starts
    
    return lengths

def estimate_fractal_dimension(recurrence_matrix, max_boxes=20):
    """
    Estimate the fractal dimension of the recurrence plot using box-counting method.
    
    This is a simplified implementation and provides an estimate.
    
    Parameters:
    -----------
    recurrence_matrix : torch.Tensor or numpy.ndarray
        The recurrence matrix
    max_boxes : int, optional
        Maximum number of different box sizes to use
        
    Returns:
    --------
    float
        Estimated fractal dimension
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(recurrence_matrix, torch.Tensor):
        rm_numpy = recurrence_matrix.cpu().numpy()
    else:
        rm_numpy = recurrence_matrix
    
    N = rm_numpy.shape[0]
    
    # Prepare box sizes (powers of 2)
    box_sizes = []
    size = 1
    while size <= N/2 and len(box_sizes) < max_boxes:
        box_sizes.append(size)
        size *= 2
    
    # Count boxes for each size
    counts = []
    for size in box_sizes:
        count = 0
        # Skip small sizes that would result in too many boxes
        if N // size > 1000:  
            # Approximate for computational efficiency
            samples = 1000
            step = max(1, N // samples)
            for i in range(0, N - size + 1, step):
                for j in range(0, N - size + 1, step):
                    if np.any(rm_numpy[i:i+size, j:j+size]):
                        count += 1
            # Scale count based on sampling
            count = count * ((N - size + 1) / step) ** 2 / ((N - size + 1) ** 2)
        else:
            # Full calculation for manageable sizes
            for i in range(0, N - size + 1):
                for j in range(0, N - size + 1):
                    if np.any(rm_numpy[i:i+size, j:j+size]):
                        count += 1
        counts.append(count)
    
    # Convert to log scale
    log_sizes = np.log(box_sizes)
    log_counts = np.log(counts)
    
    # Linear regression to find the slope
    if len(log_sizes) > 1 and len(log_counts) > 1:
        # Use only valid data points
        valid_indices = ~np.isnan(log_counts) & ~np.isinf(log_counts)
        if np.sum(valid_indices) > 1:
            slope, _ = np.polyfit(log_sizes[valid_indices], log_counts[valid_indices], 1)
            # The negative of the slope gives the fractal dimension
            return -slope
    
    return None

def visualize_rqa(recurrence_matrix, rqa_measures, case_name, plot_dir="plots"):
    """
    Visualize the recurrence matrix and RQA measures.
    
    Parameters:
    -----------
    recurrence_matrix : torch.Tensor or numpy.ndarray
        The recurrence matrix
    rqa_measures : dict
        Dictionary of RQA measures
    case_name : str
        Case identifier for saving files
    plot_dir : str, optional
        Directory to save plots
    """
    if isinstance(recurrence_matrix, torch.Tensor):
        rm_numpy = recurrence_matrix.cpu().numpy()
    else:
        rm_numpy = recurrence_matrix
    
    plt.figure(figsize=(15, 12))
    
    # Plot recurrence matrix
    plt.subplot(2, 2, 1)
    plt.imshow(rm_numpy, cmap='binary', origin='lower', aspect='equal')
    plt.colorbar(label='Recurrence')
    plt.title(f'Recurrence Plot - {case_name}')
    plt.xlabel('Time index')
    plt.ylabel('Time index')
    
    # Plot diagonal line length distribution
    plt.subplot(2, 2, 2)
    if 'diagonal_length_dist' in rqa_measures and len(rqa_measures['diagonal_length_dist']) > 0:
        plt.hist(rqa_measures['diagonal_length_dist'], bins=30, alpha=0.7, color='blue')
        plt.axvline(rqa_measures['avg_diagonal_length'], color='red', linestyle='--', 
                   label=f'Avg: {rqa_measures["avg_diagonal_length"]:.2f}')
        plt.legend()
    plt.title('Diagonal Line Length Distribution')
    plt.xlabel('Line Length')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Plot vertical line length distribution
    plt.subplot(2, 2, 3)
    if 'vertical_length_dist' in rqa_measures and len(rqa_measures['vertical_length_dist']) > 0:
        plt.hist(rqa_measures['vertical_length_dist'], bins=30, alpha=0.7, color='green')
        plt.axvline(rqa_measures['trapping_time'], color='red', linestyle='--', 
                   label=f'Avg: {rqa_measures["trapping_time"]:.2f}')
        plt.legend()
    plt.title('Vertical Line Length Distribution')
    plt.xlabel('Line Length')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    # Plot RQA measures as a table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    measures_text = "\n".join([
        f"RQA Measures Summary:",
        f"Recurrence Rate: {rqa_measures.get('recurrence_rate', 'N/A'):.4f}",
        f"Determinism: {rqa_measures.get('determinism', 'N/A'):.4f}",
        f"Laminarity: {rqa_measures.get('laminarity', 'N/A'):.4f}",
        f"Avg Diagonal Length: {rqa_measures.get('avg_diagonal_length', 'N/A'):.4f}",
        f"Max Diagonal Length: {rqa_measures.get('max_diagonal_length', 'N/A'):.1f}",
        f"Diagonal Entropy: {rqa_measures.get('diagonal_entropy', 'N/A'):.4f}",
        f"Trapping Time: {rqa_measures.get('trapping_time', 'N/A'):.4f}",
        f"Fractal Dimension: {rqa_measures.get('fractal_dimension', 'N/A')}"
    ])
    plt.text(0.1, 0.9, measures_text, va='top', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'rqa_analysis_{case_name}.png'), dpi=300, bbox_inches='tight')
    print(f"RQA visualization saved as 'rqa_analysis_{case_name}.png'")
    plt.close()

def save_rqa_results(rqa_measures, case_name, output_dir="results"):
    """
    Save RQA measures to a text file.
    
    Parameters:
    -----------
    rqa_measures : dict
        Dictionary of RQA measures
    case_name : str
        Case identifier for saving files
    output_dir : str, optional
        Directory to save results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f'rqa_measures_{case_name}.txt')
    
    with open(output_file, 'w') as f:
        f.write(f"RQA Measures for {case_name}\n")
        f.write("=" * 50 + "\n\n")
        
        for measure, value in rqa_measures.items():
            if measure not in ['diagonal_length_dist', 'vertical_length_dist']:
                if isinstance(value, (int, float)):
                    f.write(f"{measure}: {value:.6f}\n")
                else:
                    f.write(f"{measure}: {value}\n")
        
        # Save distributions as histograms
        if 'diagonal_length_dist' in rqa_measures and len(rqa_measures['diagonal_length_dist']) > 0:
            f.write("\nDiagonal length distribution summary:\n")
            f.write(f"  Min: {min(rqa_measures['diagonal_length_dist'])}\n")
            f.write(f"  Max: {max(rqa_measures['diagonal_length_dist'])}\n")
            f.write(f"  Mean: {np.mean(rqa_measures['diagonal_length_dist']):.4f}\n")
            f.write(f"  Median: {np.median(rqa_measures['diagonal_length_dist']):.4f}\n")
            
        if 'vertical_length_dist' in rqa_measures and len(rqa_measures['vertical_length_dist']) > 0:
            f.write("\nVertical length distribution summary:\n")
            f.write(f"  Min: {min(rqa_measures['vertical_length_dist'])}\n")
            f.write(f"  Max: {max(rqa_measures['vertical_length_dist'])}\n")
            f.write(f"  Mean: {np.mean(rqa_measures['vertical_length_dist']):.4f}\n")
            f.write(f"  Median: {np.median(rqa_measures['vertical_length_dist']):.4f}\n")
    
    print(f"RQA measures saved to {output_file}")

def visualize_recurrence_pattern(recurrence_matrix, case_name, zoom_region=None, plot_dir="plots"):
    """
    Create a more detailed visualization of the recurrence pattern.
    
    Parameters:
    -----------
    recurrence_matrix : torch.Tensor or numpy.ndarray
        The recurrence matrix
    case_name : str
        Case identifier for saving files
    zoom_region : tuple, optional
        Region to zoom in (x_start, x_end, y_start, y_end)
    plot_dir : str, optional
        Directory to save plots
    """
    if isinstance(recurrence_matrix, torch.Tensor):
        rm_numpy = recurrence_matrix.cpu().numpy()
    else:
        rm_numpy = recurrence_matrix
    
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plt.figure(figsize=(12, 10))
    
    # Full recurrence plot
    plt.subplot(2, 1, 1)
    plt.imshow(rm_numpy, cmap='binary', origin='lower', aspect='equal')
    plt.colorbar(label='Recurrence')
    plt.title(f'Recurrence Plot - {case_name}')
    plt.xlabel('Time index')
    plt.ylabel('Time index')
    
    # Zoomed region if specified
    plt.subplot(2, 1, 2)
    if zoom_region is not None:
        x_start, x_end, y_start, y_end = zoom_region
        zoom = rm_numpy[y_start:y_end, x_start:x_end]
        plt.imshow(zoom, cmap='binary', origin='lower', aspect='equal')
        plt.colorbar(label='Recurrence')
        plt.title(f'Zoomed Recurrence Pattern ({x_start}:{x_end}, {y_start}:{y_end})')
    else:
        # If no zoom region specified, show diagonal lines
        N = rm_numpy.shape[0]
        # Extract main diagonal ±10% of matrix size
        diag_width = max(2, int(N * 0.1))
        diag_region = np.zeros_like(rm_numpy)
        
        for i in range(-diag_width, diag_width+1):
            if i >= -N and i < N:  # Ensure diagonal is within bounds
                diag = np.diag(rm_numpy, k=i)
                # Put the diagonal back in the matrix for visualization
                for j in range(len(diag)):
                    row, col = j, j + i
                    if 0 <= row < N and 0 <= col < N:
                        diag_region[row, col] = rm_numpy[row, col]
        
        plt.imshow(diag_region, cmap='binary', origin='lower', aspect='equal')
        plt.colorbar(label='Recurrence')
        plt.title('Diagonal Line Structures')
    
    plt.xlabel('Time index')
    plt.ylabel('Time index')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'recurrence_pattern_{case_name}.png'), dpi=300, bbox_inches='tight')
    print(f"Recurrence pattern visualization saved as 'recurrence_pattern_{case_name}.png'")
    plt.close()


    








def create_recurrence_network(recurrence, save_path="recurrence_network.graphml"):
    """
    Generate a recurrence network from a recurrence matrix and analyze its properties.
    
    Parameters:
    -----------
    recurrence : numpy.ndarray or torch.Tensor
        The recurrence matrix (adjacency matrix) where recurrence[i,j] = 1 if nodes i and j are connected
    save_path : str, optional
        Path to save the network in GraphML format
        
    Returns:
    --------
    G : networkx.Graph
        The generated recurrence network
    metrics : dict
        Dictionary containing various network metrics
    """
    if not isinstance(recurrence, torch.Tensor):
        recurrence = torch.tensor(recurrence, dtype=torch.float32)
    
    # Create a NetworkX graph from the recurrence matrix
    G = nx.from_numpy_array(recurrence.cpu().numpy())
    
    # Save the network in GraphML format
    nx.write_graphml(G, save_path) # JJ hopefully this helps you 
    print(f"Network saved to {save_path}")
    
    metrics = {}
    
    print("Calculating network metrics...")
    
    # Basic network properties
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Node degree distribution
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    metrics['degree_sequence'] = degree_sequence
    metrics['avg_degree'] = sum(degree_sequence) / len(degree_sequence)
    
    # Clustering coefficient
    try:
        start_time = time.time()
        metrics['clustering_coefficient'] = nx.average_clustering(G)
        metrics['clustering_coefficient_by_node'] = nx.clustering(G)
        print(f"Clustering coefficient calculated in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Could not calculate clustering coefficient: {e}")
    
    # Path length metrics
    if nx.is_connected(G):
        start_time = time.time()
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        metrics['diameter'] = nx.diameter(G)
        print(f"Path metrics calculated in {time.time() - start_time:.2f} seconds")
    else:
        print("Graph is not connected. Using largest connected component for path metrics.")
        largest_cc = max(nx.connected_components(G), key=len)
        largest_subgraph = G.subgraph(largest_cc).copy()
        
        try:
            metrics['avg_path_length'] = nx.average_shortest_path_length(largest_subgraph)
            metrics['diameter'] = nx.diameter(largest_subgraph)
        except Exception as e:
            print(f"Could not calculate path metrics: {e}")
            metrics['avg_path_length'] = None
            metrics['diameter'] = None
    
    # Centrality measures
    try:
        start_time = time.time()
        metrics['degree_centrality'] = nx.degree_centrality(G)
        print(f"Degree centrality calculated in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Could not calculate degree centrality: {e}")
    
    try:
        start_time = time.time()
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
        print(f"Betweenness centrality calculated in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Could not calculate betweenness centrality: {e}")
    
    try:
        start_time = time.time()
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
        print(f"Eigenvector centrality calculated in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Could not calculate eigenvector centrality: {e}")
    
    try:
        start_time = time.time()
        metrics['pagerank'] = nx.pagerank(G)
        print(f"PageRank calculated in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Could not calculate PageRank: {e}")
    
    # Community detection / Modularity
    try:
        start_time = time.time()
        partition = community_louvain.best_partition(G)
        metrics['communities'] = partition
        modularity = community_louvain.modularity(partition, G)
        metrics['modularity'] = modularity
        print(f"Community detection completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Could not perform community detection: {e}")
    
    # Network entropy (using degree distribution)
    try:
        degree_counts = np.bincount(degree_sequence)
        degree_prob = degree_counts / degree_counts.sum()
        metrics['network_entropy'] = entropy(degree_prob)
    except Exception as e:
        print(f"Could not calculate network entropy: {e}")
    
    # Motif analysis is computationally expensive for larger networks
    if G.number_of_nodes() <= 100:
        try:
            start_time = time.time()
            # Count triangles as a simple motif
            metrics['triangle_count'] = sum(nx.triangles(G).values()) // 3
            print(f"Motif analysis completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Could not perform motif analysis: {e}")
    else:
        print("Skipping motif analysis due to network size")
    
    return G, metrics

def visualize_recurrence_network(G, metrics, show_labels=False):
    """
    Visualize the recurrence network and display key metrics.
    
    Parameters:
    -----------
    G : networkx.Graph
        The recurrence network to visualize
    metrics : dict
        Dictionary containing network metrics
    show_labels : bool, optional
        Whether to show node labels
    """
    plt.figure(figsize=(18, 10))
    
    # Network visualization
    plt.subplot(2, 3, 1)
    # Use different layout algorithms based on network size
    if G.number_of_nodes() < 200:
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Color nodes by degree
    node_degrees = dict(G.degree())
    node_colors = [node_degrees[n] for n in G.nodes()]
    
    # Draw the network
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, cmap=plt.cm.viridis)
    plt.colorbar(nodes, label='Node Degree')
    
    if show_labels and G.number_of_nodes() < 50:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Recurrence Network')
    plt.axis('off')
    
    # Degree distribution
    plt.subplot(2, 3, 2)
    degrees = metrics['degree_sequence']
    plt.hist(degrees, bins=range(max(degrees)+2), alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title(f'Degree Distribution\nAvg Degree: {metrics["avg_degree"]:.2f}')
    plt.grid(alpha=0.3)
    
    # Centrality comparison
    plt.subplot(2, 3, 3)
    if 'degree_centrality' in metrics and 'betweenness_centrality' in metrics:
        dc_values = list(metrics['degree_centrality'].values())
        bc_values = list(metrics['betweenness_centrality'].values())
        plt.scatter(dc_values, bc_values, alpha=0.6)
        plt.xlabel('Degree Centrality')
        plt.ylabel('Betweenness Centrality')
        plt.title('Centrality Comparison')
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Centrality data not available', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Centrality Comparison')
    
    # Community visualization if available
    plt.subplot(2, 3, 4)
    if 'communities' in metrics:
        partition = metrics['communities']
        # Color nodes by community
        cmap = plt.cm.get_cmap('tab20', max(partition.values()) + 1)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        for com in set(partition.values()):
            list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            nx.draw_networkx_nodes(G, pos, list_nodes, node_size=50, 
                                  node_color=[cmap(com)] * len(list_nodes))
        plt.title(f'Communities\nModularity: {metrics.get("modularity", "N/A")}')
    else:
        nx.draw(G, pos, node_size=50, alpha=0.7)
        plt.title('Network Structure')
    plt.axis('off')
    
    # Print summary metrics
    plt.subplot(2, 3, 5)
    plt.axis('off')
    metrics_text = (
        f"Network Summary:\n"
        f"Nodes: {metrics['num_nodes']}\n"
        f"Edges: {metrics['num_edges']}\n"
        f"Density: {metrics['density']:.4f}\n"
        f"Clustering Coef: {metrics.get('clustering_coefficient', 'N/A')}\n"
        f"Avg Path Length: {metrics.get('avg_path_length', 'N/A')}\n"
        f"Diameter: {metrics.get('diameter', 'N/A')}\n"
        f"Network Entropy: {metrics.get('network_entropy', 'N/A'):.4f}\n"
        f"Triangles: {metrics.get('triangle_count', 'N/A')}"
    )
    plt.text(0.1, 0.9, metrics_text, va='top', fontsize=12)
    
    # Eigenvector centrality visualization if available
    plt.subplot(2, 3, 6)
    if 'eigenvector_centrality' in metrics:
        ec_values = list(metrics['eigenvector_centrality'].values())
        node_sizes = [v * 1000 for v in ec_values]
        nx.draw(G, pos, node_size=node_sizes, alpha=0.7, 
                node_color=list(metrics['eigenvector_centrality'].values()),
                cmap=plt.cm.plasma)
        plt.title('Eigenvector Centrality')
    elif 'pagerank' in metrics:
        pr_values = list(metrics['pagerank'].values())
        node_sizes = [v * 1000 for v in pr_values]
        nx.draw(G, pos, node_size=node_sizes, alpha=0.7, 
                node_color=list(metrics['pagerank'].values()),
                cmap=plt.cm.plasma)
        plt.title('PageRank')
    else:
        plt.text(0.5, 0.5, 'Centrality data not available', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Node Importance')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('recurrence_network_analysis.png', dpi=300, bbox_inches='tight')
    print("Network visualization saved as 'recurrence_network_analysis.png'")
    plt.show()

def identify_attractor_states(G):
    """
    Identify potential attractor states in the network based on connectivity patterns.
    
    Parameters:
    -----------
    G : networkx.Graph
        The recurrence network
        
    Returns:
    --------
    dict
        Dictionary with results about attractor and transient states
    """
    results = {}
    
    # Find strongly connected components (potential attractors)
    try:
        # Convert to directed graph for SCC analysis
        DG = nx.DiGraph(G)
        sccs = list(nx.strongly_connected_components(DG))
        results['num_attractors'] = len(sccs)
        results['attractor_sizes'] = [len(c) for c in sccs]
        
        # Nodes with high clustering coefficient and degree are likely attractor states
        clustering = nx.clustering(G)
        degree = dict(G.degree())
        
        # Combine metrics to identify attractor states
        attractor_score = {node: clustering.get(node, 0) * degree.get(node, 0) 
                          for node in G.nodes()}
        
        # Get top 10% of nodes by attractor score
        threshold = np.percentile(list(attractor_score.values()), 90)
        results['attractor_states'] = [node for node, score in attractor_score.items() 
                                      if score > threshold]
        results['transient_states'] = [node for node, score in attractor_score.items() 
                                      if score <= threshold]
    except Exception as e:
        print(f"Could not identify attractor states: {e}")
        results['error'] = str(e)
    
    return results



def recurrence_plot_with_threshold(trajectories, output_dir="./"):
    
    for traj_idx, traj in enumerate(trajectories):
        
        ### DISTANCE WITH ITSELF (typical recurrence plot)

        # cosine 
        traj_normalized = traj / torch.norm(traj, dim=1, keepdim=True)
        cosine_sim = traj_normalized @ traj_normalized.T  # (n, n)
        distances = 1 - cosine_sim

        distances = distances / torch.max(distances) # XX could this normalization be the responsible of some weird plots


        for threshold in [0.3]: # JJ you could change this if you want. from my experience it should be between 0.3 and 0.4 for cosine 

            recurrence = distances < threshold # JJ this is used as the adjacency matrix. 
            # Is this ok? would it make sense to use the distance matrix for the network?

            # Set diagonal to zero to avoid self-loops 
            recurrence.fill_diagonal_(0) # JJ I think this is necessary??
            
            # Create and analyze the recurrence network
            G, metrics = create_recurrence_network(recurrence)
            
            # Identify attractor states
            attractor_results = identify_attractor_states(G)
            print("\nAttractor Analysis:")
            for key, value in attractor_results.items():
                if isinstance(value, list) and len(value) > 10:
                    print(f"{key}: {len(value)} items")
                else:
                    print(f"{key}: {value}")
            
            # Visualize the network and its properties
            visualize_recurrence_network(G, metrics)

            # Calculate RQA measures
            rqa_measures = calculate_rqa_measures(recurrence, case_name="first")

            # Extra visualization of recurrence patterns
            visualize_recurrence_pattern(recurrence, case_name="first")



PLOTS_DIR = "."
recurrence_plot_with_threshold(trajectories, output_dir=PLOTS_DIR)



############################################################





















