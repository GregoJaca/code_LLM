import numpy as np
from clustering import SemanticPipeline

def run_minimal():
    np.random.seed(0)
    n_tokens = 200
    dim = 16
    tokens = [f"tok_{i}" for i in range(n_tokens)]
    embeddings = np.random.randn(n_tokens, dim).astype(np.float32)

    pipeline = SemanticPipeline()
    results = pipeline.run_analysis(
        all_tokens=tokens,
        all_embeddings=embeddings,
        n_initial_hidden_states=50,
        initial_clustering_method='kmeans',
        n_clusters=4,
        window_type='moving',
        window_size=80,
        stride=80,
        visualization_method='pca'
    )

    print('Minimal pipeline run complete.')
    # print('Initial clusters:', results['initial_cluster_result'].n_clusters)
    print('Windows:', len(results['window_results']))

if __name__ == '__main__':
    run_minimal()
