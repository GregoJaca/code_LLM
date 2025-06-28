import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
from itertools import combinations
from community import community_louvain
# from sklearn.metrics import silhouette_score
# import matplotlib.colors as mcolors
import matplotlib.cm as cm
# from scipy.cluster.hierarchy import dendrogram, linkage
# from scipy.spatial.distance import pdist

def create_network_from_recurrence(recurrence_matrix, tokens=None):
    """
    Create a complex network from a recurrence matrix.
    
    Args:
        recurrence_matrix: NxN binary recurrence matrix
        tokens: List of N tokens corresponding to each node
        
    Returns:
        networkx Graph object
    """
    # Ensure diagonal elements are 0
    rec_matrix = recurrence_matrix.copy()
    np.fill_diagonal(rec_matrix, 0)
    
    # Create graph
    G = nx.from_numpy_array(rec_matrix)
    
    # Add token information if provided
    if tokens is not None:
        if len(tokens) != recurrence_matrix.shape[0]:
            raise ValueError(f"Number of tokens ({len(tokens)}) does not match matrix dimensions ({recurrence_matrix.shape[0]})")
        
        # Add token as node attribute
        for i, token in enumerate(tokens):
            G.nodes[i]['token'] = token
    
    return G

def detect_communities(G, resolution=1.0):
    """
    Detect communities in the network using Louvain algorithm.
    
    Args:
        G: networkx Graph
        resolution: Resolution parameter for Louvain algorithm
        
    Returns:
        Dictionary mapping node indices to community indices
    """
    return community_louvain.best_partition(G, resolution=resolution)

def analyze_network(G, community_mapping=None, tokens=None):
    """
    Perform comprehensive network analysis.
    
    Args:
        G: networkx Graph
        community_mapping: Dictionary mapping nodes to communities
        tokens: List of tokens corresponding to nodes
        
    Returns:
        Dictionary of network metrics
    """
    metrics = {}
    
    # Basic network properties
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Check if network is connected
    if nx.is_connected(G):
        metrics['connected'] = True
        metrics['diameter'] = nx.diameter(G)
        metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
    else:
        metrics['connected'] = False
        # Calculate for largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc)
        metrics['largest_component_size'] = len(largest_cc)
        metrics['largest_component_diameter'] = nx.diameter(G_largest)
        metrics['largest_component_avg_path'] = nx.average_shortest_path_length(G_largest)
    
    # Node-level metrics
    degrees = dict(G.degree())
    metrics['avg_degree'] = sum(degrees.values()) / len(degrees)
    metrics['max_degree'] = max(degrees.values())
    metrics['min_degree'] = min(degrees.values())
    
    # Centrality measures
    degree_centrality = nx.degree_centrality(G)
    metrics['avg_degree_centrality'] = sum(degree_centrality.values()) / len(degree_centrality)
    
    betweenness_centrality = nx.betweenness_centrality(G)
    metrics['avg_betweenness'] = sum(betweenness_centrality.values()) / len(betweenness_centrality)
    
    closeness_centrality = nx.closeness_centrality(G)
    metrics['avg_closeness'] = sum(closeness_centrality.values()) / len(closeness_centrality)
    
    # Clustering
    metrics['avg_clustering'] = nx.average_clustering(G)
    
    # Community metrics
    if community_mapping is not None:
        communities = defaultdict(list)
        for node, community in community_mapping.items():
            communities[community].append(node)
        
        metrics['num_communities'] = len(communities)
        community_sizes = [len(nodes) for nodes in communities.values()]
        metrics['avg_community_size'] = sum(community_sizes) / len(community_sizes)
        metrics['max_community_size'] = max(community_sizes)
        metrics['min_community_size'] = min(community_sizes)
        
        # Calculate modularity
        metrics['modularity'] = community_louvain.modularity(community_mapping, G)
        
        # Community token analysis if tokens are provided
        if tokens is not None:
            community_tokens = {}
            for comm_id, nodes in communities.items():
                node_tokens = [tokens[node] for node in nodes]
                token_count = Counter(node_tokens)
                common_tokens = token_count.most_common(5)
                community_tokens[comm_id] = common_tokens
            
            metrics['community_tokens'] = community_tokens
    
    return metrics, communities

def visualize_network(G, community_mapping=None, tokens=None, layout=None, figsize=(12, 12)):
    """
    Visualize the network with communities.
    
    Args:
        G: networkx Graph
        community_mapping: Dictionary mapping nodes to communities
        tokens: List of tokens corresponding to nodes
        layout: Pre-computed layout (optional)
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate layout if not provided
    if layout is None:
        layout = nx.spring_layout(G, seed=42)
    
    # Prepare node colors based on communities
    if community_mapping is not None:
        unique_communities = sorted(set(community_mapping.values()))
        color_map = cm.rainbow(np.linspace(0, 1, len(unique_communities)))
        node_colors = [color_map[community_mapping[node]] for node in G.nodes()]
    else:
        node_colors = 'skyblue'
    
    # Draw the network
    nx.draw_networkx_nodes(G, layout, node_color=node_colors, node_size=50, alpha=0.8)
    nx.draw_networkx_edges(G, layout, width=0.5, alpha=0.5)
    
    # Add a colorbar for community colors
    if community_mapping is not None:
        sm = plt.cm.ScalarMappable(cmap=cm.rainbow, 
                                   norm=plt.Normalize(vmin=0, vmax=len(unique_communities)-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Community ID')
    
    plt.title('Network Visualization with Communities')
    plt.axis('off')
    
    return fig, layout

def analyze_community_transitions(G, community_mapping, tokens=None):
    """
    Analyze transitions between communities in the temporal sequence.
    
    Args:
        G: networkx Graph
        community_mapping: Dictionary mapping nodes to communities
        tokens: List of tokens (optional)
        
    Returns:
        Transition analysis results
    """
    # Create community sequence
    community_sequence = [community_mapping[i] for i in range(len(community_mapping))]
    
    # Analyze transitions between communities
    transitions = []
    for i in range(len(community_sequence) - 1):
        if community_sequence[i] != community_sequence[i + 1]:
            transitions.append((community_sequence[i], community_sequence[i + 1]))
    
    # Count transitions
    transition_counts = Counter(transitions)
    
    # Find common patterns
    patterns = {}
    for length in range(2, min(11, len(community_sequence) // 2)):
        for i in range(len(community_sequence) - length + 1):
            pattern = tuple(community_sequence[i:i+length])
            if pattern not in patterns:
                # Count occurrences of this pattern
                count = 0
                for j in range(len(community_sequence) - length + 1):
                    if tuple(community_sequence[j:j+length]) == pattern:
                        count += 1
                
                if count > 1:  # Only include patterns that occur multiple times
                    patterns[pattern] = count
    
    # Sort patterns by frequency
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate transition probability matrix
    unique_communities = sorted(set(community_mapping.values()))
    n_communities = len(unique_communities)
    transition_matrix = np.zeros((n_communities, n_communities))
    
    for i in range(len(community_sequence) - 1):
        from_idx = unique_communities.index(community_sequence[i])
        to_idx = unique_communities.index(community_sequence[i + 1])
        transition_matrix[from_idx, to_idx] += 1
    
    # Convert to probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, 
                                  out=np.zeros_like(transition_matrix), 
                                  where=row_sums!=0)
    
    # Calculate community persistence
    persistence = {}
    current_community = community_sequence[0]
    length = 1
    
    for comm in community_sequence[1:]:
        if comm == current_community:
            length += 1
        else:
            if current_community not in persistence:
                persistence[current_community] = []
            persistence[current_community].append(length)
            current_community = comm
            length = 1
    
    # Add the last segment
    if current_community not in persistence:
        persistence[current_community] = []
    persistence[current_community].append(length)
    
    # Calculate average persistence
    avg_persistence = {comm: sum(lengths)/len(lengths) for comm, lengths in persistence.items()}
    
    return {
        'transition_counts': transition_counts,
        'common_patterns': sorted_patterns[:20],  # Top 20 patterns
        'transition_matrix': transition_matrix,
        'unique_communities': unique_communities,
        'persistence': persistence,
        'avg_persistence': avg_persistence
    }

def visualize_community_transitions(transition_results, figsize=(15, 15)):
    """
    Visualize community transitions.
    
    Args:
        transition_results: Results from analyze_community_transitions
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # 1. Plot transition heatmap
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    sns.heatmap(transition_results['transition_matrix'], 
                annot=True, cmap='viridis', fmt='.2f',
                xticklabels=transition_results['unique_communities'],
                yticklabels=transition_results['unique_communities'],
                ax=ax1)
    ax1.set_title('Community Transition Probabilities')
    ax1.set_xlabel('To Community')
    ax1.set_ylabel('From Community')
    
    # 2. Plot community sequence
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    community_sequence = [transition_results['unique_communities'].index(i) 
                         for i in transition_results['unique_communities']]
    ax2.plot(community_sequence, marker='o', linestyle='-', markersize=4)
    ax2.set_title('Community Sequence (First 100 steps)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Community ID')
    ax2.set_yticks(range(len(transition_results['unique_communities'])))
    ax2.set_yticklabels(transition_results['unique_communities'])
    ax2.grid(True, alpha=0.3)
    
    # 3. Plot community persistence
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    persistence_data = []
    for comm, lengths in transition_results['persistence'].items():
        for length in lengths:
            persistence_data.append({'Community': comm, 'Persistence Length': length})
    
    df = pd.DataFrame(persistence_data)
    sns.boxplot(x='Community', y='Persistence Length', data=df, ax=ax3)
    ax3.set_title('Community Persistence Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Plot common patterns
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    patterns = [' → '.join(str(c) for c in pattern[0]) for pattern in transition_results['common_patterns'][:10]]
    counts = [pattern[1] for pattern in transition_results['common_patterns'][:10]]
    
    if patterns:  # Check if there are patterns to plot
        ax4.barh(range(len(patterns)), counts, align='center')
        ax4.set_yticks(range(len(patterns)))
        ax4.set_yticklabels(patterns)
        ax4.set_title('Top Community Transition Patterns')
        ax4.set_xlabel('Frequency')
    else:
        ax4.text(0.5, 0.5, 'No recurring patterns found', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    return fig

def analyze_community_token_relationships(communities, tokens):
    """
    Analyze relationships between communities and their tokens.
    
    Args:
        communities: Dictionary mapping community ID to list of node indices
        tokens: List of tokens
        
    Returns:
        Analysis results
    """
    # Token frequency per community
    community_token_freq = {}
    for comm_id, nodes in communities.items():
        token_counter = Counter([tokens[node] for node in nodes])
        community_token_freq[comm_id] = token_counter
    
    # Token uniqueness (how exclusively a token belongs to a community)
    token_community_count = defaultdict(int)
    for comm_id, token_counter in community_token_freq.items():
        for token in token_counter:
            token_community_count[token] += 1
    
    # Tokens that appear in only one community
    unique_tokens = {token: next(comm_id for comm_id, counter in community_token_freq.items() 
                               if token in counter)
                    for token, count in token_community_count.items() if count == 1}
    
    # Calculate token overlap between communities
    community_similarity = {}
    for comm1, comm2 in combinations(communities.keys(), 2):
        tokens1 = set(tokens[node] for node in communities[comm1])
        tokens2 = set(tokens[node] for node in communities[comm2])
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        # Jaccard similarity
        similarity = len(intersection) / len(union) if union else 0
        community_similarity[(comm1, comm2)] = similarity
    
    # Find representative tokens for each community (using TF-IDF concept)
    community_representative_tokens = {}
    
    # Total number of communities
    n_communities = len(communities)
    
    # Calculate IDF for each token
    token_idf = {}
    for token, comm_count in token_community_count.items():
        token_idf[token] = np.log(n_communities / comm_count)
    
    # Calculate TF-IDF
    for comm_id, token_counter in community_token_freq.items():
        total_tokens = sum(token_counter.values())
        
        # TF-IDF score for each token
        token_scores = {}
        for token, count in token_counter.items():
            tf = count / total_tokens
            idf = token_idf[token]
            token_scores[token] = tf * idf
        
        # Get top tokens by TF-IDF score
        representative_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        community_representative_tokens[comm_id] = representative_tokens
    
    return {
        'community_token_freq': community_token_freq,
        'token_community_count': token_community_count,
        'unique_tokens': unique_tokens,
        'community_similarity': community_similarity,
        'representative_tokens': community_representative_tokens
    }

def visualize_community_token_relationships(token_analysis_results, communities, figsize=(15, 15)):
    """
    Visualize relationships between communities and tokens.
    
    Args:
        token_analysis_results: Results from analyze_community_token_relationships
        communities: Dictionary mapping community ID to list of node indices
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # 1. Plot community sizes and token diversity
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    comm_ids = sorted(communities.keys())
    comm_sizes = [len(communities[comm_id]) for comm_id in comm_ids]
    token_diversity = [len(token_analysis_results['community_token_freq'][comm_id]) 
                      for comm_id in comm_ids]
    
    ax1_twin = ax1.twinx()
    ax1.bar(comm_ids, comm_sizes, alpha=0.7, label='Community Size')
    ax1_twin.plot(comm_ids, token_diversity, 'ro-', label='Token Diversity')
    
    ax1.set_xlabel('Community ID')
    ax1.set_ylabel('Community Size (nodes)')
    ax1_twin.set_ylabel('Token Diversity')
    ax1.set_title('Community Size vs Token Diversity')
    ax1.grid(True, alpha=0.3)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # 2. Plot community similarity heatmap
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    
    # Prepare similarity matrix
    similarity_matrix = np.zeros((len(comm_ids), len(comm_ids)))
    for (comm1, comm2), sim in token_analysis_results['community_similarity'].items():
        i = comm_ids.index(comm1)
        j = comm_ids.index(comm2)
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim  # Mirror the matrix
    
    # Set diagonal to 1 (self-similarity)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis', 
                xticklabels=comm_ids, yticklabels=comm_ids, ax=ax2)
    ax2.set_title('Community Token Similarity (Jaccard)')
    
    # 3. Plot representative tokens for top communities
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    
    # Select top communities by size
    top_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    top_comm_ids = [comm_id for comm_id, _ in top_communities]
    
    # Prepare data for plotting
    rep_tokens_data = []
    for comm_id in top_comm_ids:
        for token, score in token_analysis_results['representative_tokens'][comm_id][:5]:  # Top 5 tokens
            rep_tokens_data.append({
                'Community': f'Community {comm_id}',
                'Token': token,
                'Score': score
            })
    
    if rep_tokens_data:
        rep_tokens_df = pd.DataFrame(rep_tokens_data)
        sns.barplot(x='Token', y='Score', hue='Community', data=rep_tokens_df, ax=ax3)
        ax3.set_title('Representative Tokens for Top Communities')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    else:
        ax3.text(0.5, 0.5, 'No representative tokens data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    return fig

def main(recurrence_matrix, tokens=None):
    """
    Main function for network analysis from recurrence matrix.
    
    Args:
        recurrence_matrix: NxN binary recurrence matrix
        tokens: List of tokens corresponding to each node
    """
    print("Creating network from recurrence matrix...")
    G = create_network_from_recurrence(recurrence_matrix, tokens)
    
    print(f"Network created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Detect communities
    print("Detecting communities...")
    community_mapping = detect_communities(G)
    print(f"Found {len(set(community_mapping.values()))} communities")
    
    # Analyze network
    print("Analyzing network...")
    metrics, communities = analyze_network(G, community_mapping, tokens)
    
    # Print network metrics
    print("\n=== Network Metrics ===")
    for key, value in metrics.items():
        if key != 'community_tokens':  # Skip printing token details
            print(f"{key}: {value}")
    
    # Print community details
    print("\n=== Community Details ===")
    for comm_id, nodes in communities.items():
        print(f"Community {comm_id}: {len(nodes)} nodes")
        
        # Print some representative tokens if available
        if tokens is not None and 'community_tokens' in metrics:
            if comm_id in metrics['community_tokens']:
                print("  Common tokens:", end=" ")
                for token, count in metrics['community_tokens'][comm_id]:
                    print(f"{token}({count})", end=" ")
                print()
    
    # Visualize the network
    print("\nVisualizing network...")
    fig_network, layout = visualize_network(G, community_mapping, tokens)
    
    # Analyze community transitions
    print("\nAnalyzing community transitions...")
    transition_results = analyze_community_transitions(G, community_mapping, tokens)
    
    # Print transition results
    print("\n=== Community Transitions ===")
    print(f"Number of unique communities: {len(transition_results['unique_communities'])}")
    
    print("\nTop transition patterns:")
    for pattern, count in transition_results['common_patterns'][:5]:
        pattern_str = ' → '.join(str(c) for c in pattern)
        print(f"  {pattern_str}: {count} occurrences")
    
    print("\nAverage persistence by community:")
    for comm, avg in transition_results['avg_persistence'].items():
        print(f"  Community {comm}: {avg:.2f} steps")
    
    # Visualize transitions
    print("\nVisualizing community transitions...")
    fig_transitions = visualize_community_transitions(transition_results)
    
    # Token analysis
    if tokens is not None:
        print("\nAnalyzing token relationships with communities...")
        token_analysis = analyze_community_token_relationships(communities, tokens)
        
        # Print token analysis results
        print("\n=== Token-Community Relationships ===")
        print(f"Number of tokens: {len(set(tokens))}")
        
        print("\nToken distribution across communities:")
        token_comm_counts = Counter(token_analysis['token_community_count'].values())
        for count, freq in sorted(token_comm_counts.items()):
            print(f"  Tokens in {count} communities: {freq}")
        
        print("\nCommunity token similarity:")
        similarity_values = list(token_analysis['community_similarity'].values())
        print(f"  Average similarity: {sum(similarity_values)/len(similarity_values):.4f}")
        print(f"  Max similarity: {max(similarity_values):.4f}")
        print(f"  Min similarity: {min(similarity_values):.4f}")
        
        print("\nRepresentative tokens by community:")
        for comm_id, rep_tokens in token_analysis['representative_tokens'].items():
            print(f"  Community {comm_id}:", end=" ")
            for token, score in rep_tokens[:3]:  # Show top 3
                print(f"{token}({score:.3f})", end=" ")
            print()
        
        # Visualize token relationships
        print("\nVisualizing token-community relationships...")
        fig_tokens = visualize_community_token_relationships(token_analysis, communities)
        
    # Save network to GraphML
    output_file = "recurrence_network.graphml"
    print(f"\nSaving network to {output_file}...")
    nx.write_graphml(G, output_file)
    
    # Show all figures
    plt.show()
    
    return {
        'graph': G,
        'metrics': metrics,
        'communities': communities,
        'community_mapping': community_mapping,
        'transitions': transition_results,
        'token_analysis': token_analysis if tokens is not None else None
    }

# Example usage
if __name__ == "__main__":

    
    recurrence_matrix = cosine_sim_last_last.numpy()
    
    sample_tokens = tokens
    
    # Run the analysis
    results = main(recurrence_matrix, sample_tokens)