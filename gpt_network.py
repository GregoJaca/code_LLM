import os
import json
import numpy as np
import networkx as nx
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
from collections import defaultdict
from community import community_louvain

def analyze_recurrence_network(recurrence_matrix: np.ndarray, resolution: float, out_dir: str):
    # Prepare output
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    # Build graph, ignore isolated
    G = nx.from_numpy_array(recurrence_matrix)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    # Basic metrics
    results['num_nodes'] = G.number_of_nodes()
    results['num_edges'] = G.number_of_edges()
    results['density'] = nx.density(G)
    results['connected'] = nx.is_connected(G)
    results['num_isolates'] = len(isolates)

    # Degree distribution
    degrees = np.array([d for _, d in G.degree()])
    def summarize(arr): return dict(min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()), median=float(np.median(arr)), std=float(arr.std()))
    results['degree_distribution'] = summarize(degrees)

    # Centralities and metrics
    metrics = {}
    metrics['degree_centrality'] = summarize(np.array(list(nx.degree_centrality(G).values())))
    metrics['clustering_coefficient'] = summarize(np.array(list(nx.clustering(G).values())))
    metrics['closeness_centrality'] = summarize(np.array(list(nx.closeness_centrality(G).values())))
    metrics['betweenness_centrality'] = summarize(np.array(list(nx.betweenness_centrality(G).values())))
    metrics['eigenvector_centrality'] = summarize(np.array(list(nx.eigenvector_centrality(G).values())))
    metrics['pagerank'] = summarize(np.array(list(nx.pagerank(G).values())))
    metrics['local_degree_anomaly'] = summarize(np.array([abs(G.degree(n) - np.mean(degrees)) for n in G]))
    metrics['edge_centrality'] = summarize(np.array(list(nx.edge_betweenness_centrality(G).values())))
    metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
    # metrics['matching_index'] = summarize(np.array(list(nx.matching_index(G).values())))
    try:
        metrics['path_length'] = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        metrics['path_length'] = None
    try:
        metrics['network_diameter'] = nx.diameter(G)
    except nx.NetworkXError:
        metrics['network_diameter'] = None
    results.update(metrics)

    # Other: entropy, triangles
    adj = nx.to_numpy_array(G)

    p = adj.flatten().astype(float)
    total = p.sum()
    if total > 0:
        p /= total
        # only keep p>0
        nonzero = p > 0
        results['network_entropy'] = float(-np.sum(p[nonzero] * np.log2(p[nonzero])))
    else:
        results['network_entropy'] = 0.0

    results['triangle_count'] = int(sum(nx.triangles(G).values()) / 3)

    # Attractor analysis placeholder
    # results.update({'num_attractors': None, 'attractor_sizes': None, 'attractor_states': None, 'transient_states': None})


    try:
        # Convert to directed graph for SCC analysis
        DG = nx.DiGraph(G)
        sccs = list(nx.strongly_connected_components(DG))
        metrics['num_attractors'] = len(sccs)
        metrics['attractor_sizes'] = summarize(np.array([len(c) for c in sccs]))
        
        
        # Nodes with high clustering coefficient and degree are likely attractor states
        clustering = nx.clustering(G)
        degree = dict(G.degree())
        
        # Combine metrics to identify attractor states
        attractor_score = {node: clustering.get(node, 0) * degree.get(node, 0) 
                          for node in G.nodes()}
        
        # Get top 10% of nodes by attractor score
        threshold = np.percentile(list(attractor_score.values()), 90)
        metrics['attractor_states'] = summarize(np.array([node for node, score in attractor_score.items() 
                                      if score > threshold]))
        metrics['transient_states'] = summarize(np.array([node for node, score in attractor_score.items() 
                                      if score <= threshold]))
    except Exception as e:
        print(f"Could not identify attractor states: {e}")
        metrics['error'] = str(e)

    results.update(metrics)



    # Community detection with leiden
    ig_graph = ig.Graph.Adjacency((nx.to_numpy_array(G) > 0).tolist())
    partition = leidenalg.find_partition(ig_graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution)
    membership = partition.membership
    # Map igraph indices to original node labels
    mapping = list(G.nodes())
    comms = defaultdict(list)
    for idx, comm in enumerate(membership):
        comms[comm].append(mapping[idx])
    




    # After computing `comms` and `membership` as in the main code, build a color map:
    import matplotlib.cm as cm
    # only color meaningful communities (>1); assign others gray
    meaningful_comms = [c for c, members in comms.items() if len(members)>1]
    # map each community to a color from a qualitative colormap
    cmap = cm.get_cmap('tab20', len(meaningful_comms))
    comm_color = {c: cmap(i) for i, c in enumerate(meaningful_comms)}
    # default color for singleton or small communities
    default_color = (0.8,0.8,0.8,1.0)

    # node_colors list aligned with G.nodes()
    node_colors = []
    for n in G.nodes():
        # find community index for this node
        for c, members in comms.items():
            if n in members:
                node_colors.append(comm_color.get(c, default_color))
                break




    # Community stats
    sizes = np.array([len(c) for c in comms.values()])
    total = len(sizes)
    meaningful = sizes[sizes>1]

    # for calculating modularity
    partition_for_nx = {node: cluster for node, cluster in zip(G.nodes(), partition.membership)}
    standard_modularity = community_louvain.modularity(partition_for_nx, G)

    results['communities'] = {
        'quality': partition.quality(),
        'modularity': standard_modularity,
        'total': int(total),
        'largest': int(sizes.max()),
        'average_size': float(sizes.mean()),
        'median_size': float(np.median(sizes)),
        'num_individual': int((sizes==1).sum()),
        'num_meaningful': int((sizes>1).sum()),
        'average_meaningful_size': float(meaningful.mean()) if len(meaningful)>0 else 0,
        'median_meaningful_size': float(np.median(meaningful)) if len(meaningful)>0 else 0,
        # 'partition': {str(comm): members for comm, members in comms.items()} # GGXX
    }

    # Plot function
    def save_plot(fig, name):
        fig.savefig(os.path.join(out_dir, name), bbox_inches='tight')
        plt.close(fig)
        plt.show()


    # Plots
    layouts = {'spring': nx.spring_layout, 'fruchterman': nx.fruchterman_reingold_layout, 'kamada_kawai': nx.kamada_kawai_layout}
    for lname, func in layouts.items():
        fig = plt.figure()
        pos = func(G)
        nx.draw(
            G,
            pos=pos,
            node_color=node_colors,
            node_size=20,
            edge_color='gray'
        )
        save_plot(fig, f'network_{lname}_colored.png')



    # Degree distribution plot
    fig = plt.figure()
    plt.hist(degrees, bins=50)
    plt.title('Degree Distribution')
    save_plot(fig, 'degree_distribution.png')

    # Centrality comparison: degree vs eigenvector
    fig = plt.figure()
    dc = list(nx.degree_centrality(G).values())
    ec = list(nx.eigenvector_centrality(G).values())
    plt.scatter(dc, ec)
    plt.xlabel('Degree Centrality')
    plt.ylabel('Eigenvector Centrality')
    plt.title('Centrality Comparison')
    save_plot(fig, 'centrality_comparison.png')

    # PageRank plot
    fig = plt.figure()
    plt.hist(list(nx.pagerank(G).values()), bins=50)
    plt.title('PageRank Distribution')
    save_plot(fig, 'pagerank.png')

    # Eigenvector centrality plot
    fig = plt.figure()
    plt.hist(ec, bins=50)
    plt.title('Eigenvector Centrality Distribution')
    save_plot(fig, 'eigenvector_centrality.png')

    # Community sizes plot
    fig = plt.figure()
    plt.hist(sizes, bins=50)
    plt.title('Community Sizes')
    save_plot(fig, 'community_sizes.png')

    # Number of tokens per community
    fig = plt.figure()
    plt.bar(range(len(sizes)), sizes)
    plt.title('Nodes per Community')
    save_plot(fig, 'tokens_per_community.png')

    # Save results
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
