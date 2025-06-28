import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
from community import community_louvain
import time
import leidenalg
import igraph as ig
from collections import defaultdict

def create_recurrence_network(recurrence, similarity_matrix=None, save_path="recurrence_network.graphml"):
    """
    Generate a recurrence network from a recurrence matrix and analyze its properties.
    
    Parameters:
    -----------
    recurrence : numpy.ndarray 
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
    
    # Create a NetworkX graph from the recurrence matrix
    G = nx.from_numpy_array(recurrence)
    
    # Handle isolated nodes - identify them but keep them in the graph
    isolated_nodes = list(nx.isolates(G))
    print(f"Found {len(isolated_nodes)} isolated nodes out of {G.number_of_nodes()} total nodes")
    
    # Initialize metrics dictionary
    metrics = {}
    metrics['isolated_nodes'] = isolated_nodes
    
    # Calculate network metrics
    print("Calculating network metrics...")
    
    # Create a subgraph excluding isolated nodes for analysis
    G_filtered = G.copy()
    G_filtered.remove_nodes_from(isolated_nodes)
    print(f"Working with connected subgraph of {G_filtered.number_of_nodes()} nodes and {G_filtered.number_of_edges()} edges")
    
    # Basic network properties
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_connected_nodes'] = G_filtered.number_of_nodes()
    metrics['num_edges'] = G_filtered.number_of_edges()
    metrics['density'] = nx.density(G_filtered)
    
    # Calculate recurrence rate (equivalent to network density)
    metrics['recurrence_rate'] = metrics['density']
    
    # If we have the original similarity matrix, store additional metrics
    if similarity_matrix is not None:
        # Use only the connected nodes for these calculations
        connected_nodes = list(G_filtered.nodes())
        if len(connected_nodes) > 1:
            sub_matrix = similarity_matrix[np.ix_(connected_nodes, connected_nodes)]
            metrics['avg_similarity'] = np.mean(sub_matrix)
            metrics['max_similarity'] = np.max(sub_matrix)
            metrics['min_similarity'] = np.min(sub_matrix[sub_matrix > 0])  # Only non-zero values
    
    # Node degree distribution
    degree_sequence = sorted([d for n, d in G_filtered.degree()], reverse=True)
    metrics['degree_sequence'] = degree_sequence
    metrics['avg_degree'] = sum(degree_sequence) / len(degree_sequence) if degree_sequence else 0
    
    # Clustering coefficient
    try:
        start_time = time.time()
        metrics['clustering_coefficient'] = nx.average_clustering(G_filtered)
        metrics['clustering_coefficient_by_node'] = nx.clustering(G_filtered)
        print(f"Clustering coefficient calculated in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Could not calculate clustering coefficient: {e}")
    
    # Path length metrics
    if nx.is_connected(G_filtered):
        start_time = time.time()
        metrics['avg_path_length'] = nx.average_shortest_path_length(G_filtered)
        metrics['diameter'] = nx.diameter(G_filtered)
        print(f"Path metrics calculated in {time.time() - start_time:.2f} seconds")
    else:
        print("Filtered graph is still not connected. Using largest connected component for path metrics.")
        largest_cc = max(nx.connected_components(G_filtered), key=len)
        largest_subgraph = G_filtered.subgraph(largest_cc).copy()
        
        try:
            metrics['avg_path_length'] = nx.average_shortest_path_length(largest_subgraph)
            metrics['diameter'] = nx.diameter(largest_subgraph)
        except Exception as e:
            print(f"Could not calculate path metrics: {e}")
            metrics['avg_path_length'] = None
            metrics['diameter'] = None
    
    # Community detection with multiple parameter combinations
    try:
        start_time = time.time()
        
        # Store results from multiple community detection algorithms
        community_results = {}
        
        # Create igraph version of the filtered graph for community detection
        ig_graph = ig.Graph.from_networkx(G_filtered)
        
        # Try Leiden with different resolution parameters
        resolution_values = [0.01]
        
        print("\nTrying Leiden algorithm with different resolution parameters:")
        for resolution in resolution_values:
            try:
                # Run Leiden algorithm with current resolution
                leiden_partition = leidenalg.find_partition(
                    ig_graph, 
                    leidenalg.ModularityVertexPartition if resolution == 1.0 else leidenalg.CPMVertexPartition, 
                    resolution_parameter=resolution,
                    seed=42
                )
                
                # Convert result and map back to original node indices
                leiden_result = {}
                node_map = {i: node for i, node in enumerate(G_filtered.nodes())}
                
                for i, cluster in enumerate(leiden_partition.membership):
                    leiden_result[node_map[i]] = cluster
                
                # Calculate standard modularity using NetworkX for proper comparison
                partition_for_nx = {node: cluster for node, cluster in zip(G_filtered.nodes(), leiden_partition.membership)}
                standard_modularity = community_louvain.modularity(partition_for_nx, G_filtered)
                
                # Calculate quality score from igraph
                quality_score = leiden_partition.quality()
                
                # Calculate meaningful communities (size > 1)
                community_sizes = {}
                for community_id in set(leiden_partition.membership):
                    size = leiden_partition.membership.count(community_id)
                    community_sizes[community_id] = size
                meaningful_communities = sum(1 for size in community_sizes.values() if size > 1)
                
                # Instead of adding isolated nodes as separate communities, merge small communities
                # Only process isolated nodes if we have meaningful community structure
                if meaningful_communities > 0:
                    # Get largest community ID from current partition
                    max_community_id = max(leiden_partition.membership) if leiden_partition.membership else -1
                    
                    # Add isolated nodes by merging with existing communities when possible
                    for i, node in enumerate(isolated_nodes):
                        # Try to connect to existing communities based on potential connections
                        potential_connections = []
                        for neighbor in range(G.number_of_nodes()):
                            if neighbor not in isolated_nodes and recurrence[node, neighbor] > 0.5:  # Use a lower threshold
                                if neighbor in leiden_result:
                                    potential_connections.append(leiden_result[neighbor])
                        
                        if potential_connections:
                            # Assign to most frequent community among connections
                            from collections import Counter
                            most_common = Counter(potential_connections).most_common(1)[0][0]
                            leiden_result[node] = most_common
                        else:
                            # If no connections found, create new singleton community
                            leiden_result[node] = max_community_id + 1 + i
                else:
                    # If no meaningful communities, just add isolated nodes as singletons
                    max_community_id = max(leiden_partition.membership) + 1 if leiden_partition.membership else 0
                    for i, node in enumerate(isolated_nodes):
                        leiden_result[node] = max_community_id + i
                
                # Recalculate the number of communities and meaningful communities
                final_communities = set(leiden_result.values())
                final_community_sizes = {}
                for comm_id in final_communities:
                    size = sum(1 for v in leiden_result.values() if v == comm_id)
                    final_community_sizes[comm_id] = size
                
                final_meaningful = sum(1 for size in final_community_sizes.values() if size > 1)
                
                # Store results
                method_name = f"leiden_res{resolution}"
                community_results[method_name] = {
                    'partition': leiden_result,
                    'modularity': standard_modularity,
                    'quality': quality_score,
                    'num_communities': len(final_communities),
                    'num_meaningful_communities': final_meaningful,
                    'resolution': resolution,
                    'isolated_communities': sum(1 for size in final_community_sizes.values() if size == 1)
                }
                
                print(f"Leiden (res={resolution}) detected {community_results[method_name]['num_communities']} communities "
                  f"({community_results[method_name]['num_meaningful_communities']} meaningful, "
                  f"{community_results[method_name]['isolated_communities']} singletons)"
                  f" - Modularity: {standard_modularity:.4f}, Quality: {quality_score:.4f}")
            except Exception as e:
                print(f"Leiden community detection with resolution {resolution} failed: {e}")
        
        # Try hierarchical community detection
        try:
            print("\nPerforming hierarchical community detection:")
            # First level - detect main communities
            base_partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.CPMVertexPartition,
                resolution_parameter=0.2,
                seed=42
            )
            
            # Convert first level result
            first_level_result = {}
            node_map = {i: node for i, node in enumerate(G_filtered.nodes())}
            
            for i, cluster in enumerate(base_partition.membership):
                first_level_result[node_map[i]] = cluster
            
            # Add isolated nodes as singleton communities at first level
            max_community_id = max(base_partition.membership) + 1 if base_partition.membership else 0
            for i, node in enumerate(isolated_nodes):
                first_level_result[node] = max_community_id + i
            
            # Group nodes by community
            community_nodes = defaultdict(list)
            for node, community in enumerate(base_partition.membership):
                community_nodes[community].append(node)
            
            # Second level - detect subcommunities within each first-level community
            hierarchical_result = first_level_result.copy()
            subcommunity_offset = max(first_level_result.values()) + 1
            subcommunity_count = 0
            
            # Process only communities with enough nodes
            min_size_for_subdivision = 10
            for community_id, members in community_nodes.items():
                if len(members) >= min_size_for_subdivision:
                    # Create subgraph for this community
                    subgraph = ig_graph.subgraph(members)
                    
                    # Apply Leiden to the subgraph
                    sub_partition = leidenalg.find_partition(
                        subgraph,
                        leidenalg.CPMVertexPartition,
                        resolution_parameter=0.5,
                        seed=42
                    )
                    
                    # If meaningful subcommunities found
                    if len(set(sub_partition.membership)) > 1:
                        # Map back to original nodes and assign new hierarchical IDs
                        for i, sub_id in enumerate(sub_partition.membership):
                            orig_node = node_map[members[i]]
                            hierarchical_result[orig_node] = subcommunity_offset + subcommunity_count + sub_id
                        
                        subcommunity_count += len(set(sub_partition.membership))
            
            # Calculate hierarchical modularity (approximate)
            # Use the filtered graph for modularity calculation
            G_filtered_copy = G_filtered.copy()
            hierarchical_partition = {node: comm for node, comm in hierarchical_result.items() 
                                     if node not in isolated_nodes}
            hierarchical_modularity = community_louvain.modularity(hierarchical_partition, G_filtered_copy)
            
            # Store hierarchical results
            community_results['hierarchical'] = {
                'partition': hierarchical_result,
                'modularity': hierarchical_modularity,
                'num_communities': len(set(hierarchical_result.values())),
                'num_first_level': len(set(first_level_result.values())),
                'num_subcommunities': subcommunity_count,
                'isolated_communities': len(isolated_nodes)
            }
            
            print(f"Hierarchical detected {community_results['hierarchical']['num_first_level'] - len(isolated_nodes)} main communities "
                  f"divided into {subcommunity_count} subcommunities "
                  f"+ {len(isolated_nodes)} isolated nodes, with modularity {hierarchical_modularity:.4f}")
        except Exception as e:
            print(f"Hierarchical community detection failed: {e}")
        
        # Custom scoring function to select best algorithm based on multiple factors
        def community_score(alg):
            result = community_results[alg]
            if 'num_meaningful_communities' not in result:
                return -1
            
            # Create a score that prioritizes:
            # 1. Number of meaningful communities (more is better)
            # 2. Proportion of nodes in meaningful communities (higher is better)
            # 3. Modularity (higher is better)
            # 4. Fewer singleton communities (lower is better)
            
            meaningful_ratio = result['num_meaningful_communities'] / result['num_communities'] if result['num_communities'] > 0 else 0
            singleton_penalty = result['isolated_communities'] / metrics['num_nodes'] if metrics['num_nodes'] > 0 else 0
            
            # Balanced score with weights
            score = (0.4 * meaningful_ratio + 
                     0.4 * result['modularity'] + 
                     0.2 * (1 - singleton_penalty))
            
            return score
            
        # Select best algorithm based on custom score
        best_algorithm = max(
            [alg for alg in community_results.keys()],
            key=community_score
        )
        
        # Store all results for more detailed analysis
        metrics['community_detection'] = community_results
        metrics['best_community_algorithm'] = best_algorithm
        
        # Update the primary community result with the best algorithm
        metrics['communities'] = community_results[best_algorithm]['partition']
        metrics['modularity'] = community_results[best_algorithm]['modularity']
            
        print(f"\nBest community detection algorithm: {best_algorithm} "
              f"(modularity: {community_results[best_algorithm]['modularity']:.4f})")
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
    
    # Additional analysis for communities
    try:
        if 'communities' in metrics:
            # Get community assignments
            partition = metrics['communities']
            
            # Analyze community size distribution
            community_sizes = defaultdict(int)
            for node, community in partition.items():
                community_sizes[community] += 1
            
            metrics['community_sizes'] = dict(community_sizes)
            metrics['avg_community_size'] = np.mean(list(community_sizes.values()))
            metrics['max_community_size'] = max(community_sizes.values())
            metrics['min_community_size'] = min(community_sizes.values())
            
            print(f"\nCommunity size analysis:")
            print(f"Average community size: {metrics['avg_community_size']:.2f}")
            print(f"Largest community size: {metrics['max_community_size']}")
            print(f"Smallest community size: {metrics['min_community_size']}")
            
            # Identify community membership for reporting
            community_members = defaultdict(list)
            for node_idx, community_idx in partition.items():
                community_members[community_idx].append(node_idx)
            
            metrics['community_members'] = dict(community_members)
            
            # Print some example communities
            print("\nSample communities (showing 5 largest):")
            largest_communities = sorted(community_members.items(), 
                                        key=lambda x: len(x[1]), 
                                        reverse=True)[:5]
            for comm_id, members in largest_communities:
                if len(members) > 1:  # Only show non-singleton communities
                    print(f"Community {comm_id}: {len(members)} nodes - Indices: {members[:5]}...")
    except Exception as e:
        print(f"Could not analyze communities: {e}")
    
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
    # Create a subgraph excluding isolated nodes for visualization
    G_viz = G.copy()
    if 'isolated_nodes' in metrics:
        G_viz.remove_nodes_from(metrics['isolated_nodes'])
    
    plt.figure(figsize=(18, 12))
    
    # Network visualization
    plt.subplot(2, 3, 1)
    # Use different layout algorithms based on network size
    if G_viz.number_of_nodes() < 200:
        pos = nx.spring_layout(G_viz, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G_viz)
    
    # Color nodes by degree
    node_degrees = dict(G_viz.degree())
    node_colors = [node_degrees[n] for n in G_viz.nodes()]
    
    # Draw the network
    nx.draw_networkx_edges(G_viz, pos, alpha=0.2)
    nodes = nx.draw_networkx_nodes(G_viz, pos, node_size=50, node_color=node_colors, cmap=plt.cm.viridis)
    plt.colorbar(nodes, label='Node Degree')
    
    if show_labels and G_viz.number_of_nodes() < 50:
        nx.draw_networkx_labels(G_viz, pos, font_size=8)
    
    plt.title('Recurrence Network\n(isolated nodes removed)')
    plt.axis('off')
    
    # Degree distribution
    plt.subplot(2, 3, 2)
    degrees = metrics['degree_sequence']
    plt.hist(degrees, bins=range(max(degrees)+2), alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title(f'Degree Distribution\nAvg Degree: {metrics["avg_degree"]:.2f}')
    plt.grid(alpha=0.3)
    
    # Community size distribution
    plt.subplot(2, 3, 3)
    if 'community_sizes' in metrics:
        # Filter out isolated nodes (singleton communities)
        filtered_sizes = [size for size in metrics['community_sizes'].values() if size > 1]
        plt.hist(filtered_sizes, bins=20, alpha=0.7)
        plt.xlabel('Community Size')
        plt.ylabel('Frequency')
        plt.title(f'Community Size Distribution\nAvg Size: {metrics["avg_community_size"]:.2f}')
        plt.grid(alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Community data not available', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Community Size Distribution')
    
    # Community visualization if available
    plt.subplot(2, 3, 4)
    if 'communities' in metrics:
        partition = metrics['communities']
        # Filter out isolated nodes
        partition_filtered = {node: comm for node, comm in partition.items() 
                             if node not in metrics.get('isolated_nodes', [])}
        
        # Color nodes by community
        cmap = plt.cm.get_cmap('tab20', len(set(partition_filtered.values())))
        nx.draw_networkx_edges(G_viz, pos, alpha=0.2)
        
        for i, com in enumerate(sorted(set(partition_filtered.values()))):
            list_nodes = [nodes for nodes in partition_filtered.keys() 
                         if partition_filtered[nodes] == com]
            nx.draw_networkx_nodes(G_viz, pos, list_nodes, node_size=50, 
                                  node_color=[cmap(i % 20)] * len(list_nodes))
        plt.title(f'Communities\nModularity: {metrics.get("modularity", "N/A"):.4f}')
    else:
        nx.draw(G_viz, pos, node_size=50, alpha=0.7)
        plt.title('Network Structure')
    plt.axis('off')
    
    # Print summary metrics
    plt.subplot(2, 3, 5)
    plt.axis('off')
    metrics_text = (
        f"Network Summary:\n"
        f"Total Nodes: {metrics['num_nodes']}\n"
        f"Connected Nodes: {metrics['num_connected_nodes']}\n"
        f"Isolated Nodes: {len(metrics.get('isolated_nodes', []))}\n"
        f"Edges: {metrics['num_edges']}\n"
        f"Density: {metrics['density']:.4f}\n"
        f"Clustering Coef: {metrics.get('clustering_coefficient', 'N/A'):.4f}\n"
        f"Avg Path Length: {metrics.get('avg_path_length', 'N/A')}\n"
        f"Communities: {len(set(metrics['communities'].values()))}\n"
        f"Best Algorithm: {metrics.get('best_community_algorithm', 'N/A')}"
    )
    plt.text(0.1, 0.9, metrics_text, va='top', fontsize=12)
    
    # Hierarchical community visualization if available
    plt.subplot(2, 3, 6)
    if 'community_detection' in metrics and 'hierarchical' in metrics['community_detection']:
        hierarchical = metrics['community_detection']['hierarchical']
        partition = hierarchical['partition']
        
        # Filter out isolated nodes
        partition_filtered = {node: comm for node, comm in partition.items() 
                             if node not in metrics.get('isolated_nodes', [])}
        
        # Color nodes by hierarchical community
        cmap = plt.cm.get_cmap('nipy_spectral', len(set(partition_filtered.values())))
        nx.draw_networkx_edges(G_viz, pos, alpha=0.2)
        
        for i, com in enumerate(sorted(set(partition_filtered.values()))):
            list_nodes = [nodes for nodes in partition_filtered.keys() 
                         if partition_filtered[nodes] == com]
            nx.draw_networkx_nodes(G_viz, pos, list_nodes, node_size=50, 
                                  node_color=[cmap(i % cmap.N)] * len(list_nodes))
        
        plt.title(f'Hierarchical Communities\n'
                 f'L1: {hierarchical["num_first_level"]-len(metrics.get("isolated_nodes", []))} '
                 f'L2: {hierarchical["num_subcommunities"]}')
    else:
        plt.text(0.5, 0.5, 'Hierarchical data not available', 
                 horizontalalignment='center', verticalalignment='center')
        plt.title('Hierarchical Communities')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('recurrence_network_analysis.png', dpi=300, bbox_inches='tight')
    print("Network visualization saved as 'recurrence_network_analysis.png'")
    plt.show()

def analyze_community_membership(metrics, recurrence_matrix=None):
    """
    Analyze and report on community membership structure.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing network metrics including community information
    recurrence_matrix : numpy.ndarray, optional
        The original recurrence matrix for additional analyses
    """
    if 'communities' not in metrics:
        print("No community information available for analysis")
        return
    
    partition = metrics['communities']
    
    # Create a mapping from communities to nodes
    communities = defaultdict(list)
    for node, community in partition.items():
        communities[community].append(node)
    
    # Sort communities by size
    sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Save community membership to file
    with open("community_membership.txt", "w") as f:
        f.write(f"Total communities: {len(communities)}\n")
        f.write(f"Total nodes: {sum(len(nodes) for nodes in communities.values())}\n\n")
        
        f.write("Community sizes:\n")
        for comm_id, nodes in sorted_communities:
            f.write(f"Community {comm_id}: {len(nodes)} nodes\n")
        
        f.write("\n\nDetailed community membership:\n")
        for comm_id, nodes in sorted_communities:
            f.write(f"\nCommunity {comm_id} ({len(nodes)} nodes):\n")
            f.write(f"Node indices: {nodes}\n")
            
            # Calculate additional community properties if recurrence matrix available
            if recurrence_matrix is not None and len(nodes) > 1:
                # Extract the submatrix for this community
                sub_matrix = recurrence_matrix[np.ix_(nodes, nodes)]
                
                # Calculate community density (internal edge density)
                internal_edges = np.sum(sub_matrix)
                max_possible = len(nodes) * (len(nodes) - 1)
                density = internal_edges / max_possible if max_possible > 0 else 0
                
                f.write(f"Internal density: {density:.4f}\n")
                
                # Calculate community connectedness to rest of network
                all_nodes = set(range(recurrence_matrix.shape[0]))
                external_nodes = list(all_nodes - set(nodes))
                
                if external_nodes:
                    external_matrix = recurrence_matrix[np.ix_(nodes, external_nodes)]
                    external_edges = np.sum(external_matrix)
                    max_external = len(nodes) * len(external_nodes)
                    external_density = external_edges / max_external if max_external > 0 else 0
                    
                    f.write(f"External connectivity: {external_density:.4f}\n")
                    f.write(f"Isolation ratio: {(density / external_density) if external_density > 0 else float('inf'):.4f}\n")
    
    print(f"\nCommunity membership saved to 'community_membership.txt'")
    
    # Print summary statistics
    print(f"\nCommunity membership summary:")
    print(f"Total communities: {len(communities)}")
    print(f"Largest community: {len(sorted_communities[0][1])} nodes")
    print(f"Smallest community: {len(sorted_communities[-1][1])} nodes")
    
    # Community size distribution
    sizes = [len(nodes) for _, nodes in communities.items()]
    print(f"Average community size: {np.mean(sizes):.2f}")
    print(f"Median community size: {np.median(sizes):.2f}")
    
    # Count singleton communities (isolated nodes)
    singletons = sum(1 for size in sizes if size == 1)
    print(f"Singleton communities: {singletons} ({singletons/len(communities)*100:.1f}%)")
    
    # More meaningful communities (size > 1)
    meaningful = [size for size in sizes if size > 1]
    if meaningful:
        print(f"Meaningful communities (size > 1): {len(meaningful)}")
        print(f"Average meaningful community size: {np.mean(meaningful):.2f}")
        
        # Calculate modularity contribution per community
        if recurrence_matrix is not None:
            print("\nAnalyzing modularity contribution per community...")
            G = nx.from_numpy_array(recurrence_matrix)
            modularity_contrib = {}
            
            for comm_id, nodes in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True):
                if len(nodes) > 1:  # Only meaningful communities
                    # Calculate this community's contribution to modularity
                    subgraph = G.subgraph(nodes)
                    internal_edges = subgraph.number_of_edges()
                    total_degree = sum(dict(G.degree(nodes)).values())
                    expected_edges = (total_degree ** 2) / (2 * G.number_of_edges()) if G.number_of_edges() > 0 else 0
                    
                    contrib = (internal_edges / G.number_of_edges()) - (expected_edges / G.number_of_edges()) if G.number_of_edges() > 0 else 0
                    modularity_contrib[comm_id] = contrib
            
            # Print top contributors
            top_contributors = sorted(modularity_contrib.items(), key=lambda x: x[1], reverse=True)[:5]
            print("Top modularity contributing communities:")
            for comm_id, contrib in top_contributors:
                print(f"Community {comm_id} (size {len(communities[comm_id])}): contribution = {contrib:.4f}")

# Example usage:
def run_network_analysis(cosine_sim_matrix, threshold=0.8):
    """
    Run complete network analysis on the cosine similarity matrix
    
    Parameters:
    -----------
    cosine_sim_matrix : numpy.ndarray
        The cosine similarity matrix between vector embeddings
    threshold : float, optional
        Threshold for converting similarity to binary recurrence
    """
    # Keep original similarity matrix for reference
    similarity_matrix = cosine_sim_matrix.copy()
    
    # Create binary recurrence matrix
    recurrence = cosine_sim_matrix > threshold
    np.fill_diagonal(recurrence, 0)  # Remove self-loops
    
    print(f"Using threshold {threshold} for recurrence matrix")
    
    # Create and analyze network
    G, metrics = create_recurrence_network(recurrence, similarity_matrix)
    
    # Visualize network
    visualize_recurrence_network(G, metrics)
    
    # Analyze community membership
    analyze_community_membership(metrics, recurrence)
    
    return G, metrics

# Usage:
G, metrics = run_network_analysis(cosine_sim_last_last.numpy())