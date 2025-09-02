
"""Investigating hidden state representations of the semantic meaning in LLMs
"""


# Section:Algorithms and functions

# új improved
import numpy as np
import torch
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULTS = {
    ### Most interesting to play around with ###

    'n_initial_hidden_states': 512, # this determine the centroids of the clusters, which then remain fixed
    'initial_clustering_method': ['kmeans'],  # 'kmeans', 'hierarchical'
    'n_clusters': 10,

    # In order to study the "time evolution" I make a window for which hidden state vectors are used the clustering. then in the pdf plots you can see how it evolves. 
    'window_type': 'moving',  # 'expanding' (adds new), 'moving' (discards the old adds new)
    'window_size': 128,
    'stride': 64, # step sizeof window   

    'pdf_plot_methods': ['kde', 'timeline'],  # 'kde', 'timeline'
    'pdf_point_mode': 'raw_points',  # 'raw_points', 'kde', 'both'
    'metrics_to_plot': ['cosine', 'variance', 'radius'],  # 'cosine', 'variance', 'volume', 'radius'

    ### Least interesting to play around with ###

    'visualization_method': 'tsne',  # 'tsne'
    'n_components_tsne': 2,  # 2 or 3
    'perplexity': 30.0,  # typically between 5 and 50
    'sample_top': 12,
    'inline': True,
    'variance_threshold_structural': 0.1,
    'isolation_threshold_structural': 0.8,
    'k_range_semantic': range(2, 15),
    'min_variance_explanation_pca': 0.9,  # 0 to 1
    'figsize_visualizer': (12, 8),

    # Hierarchical clustering part
    'linkage_method_hierarchical': 'ward',  # 'ward', 'complete', 'average', 'single'
    'max_display_tokens_dendrogram': 50,
    # Maximum number of leaves to show when truncating dendrograms (used as 'p' parameter)
    'max_dendrogram_leaves': 30,
    'metric_dendrogram': 'euclidean',  # 'euclidean', 'cityblock', 'cosine', etc.
    'truncate_mode_dendrogram': 'lastp',  # 'lastp', 'level', None
    'comparison_methods_dendrogram': ['ward', 'complete', 'average'],  # 'ward', 'complete', 'average'
    'show_comparison_hierarchical': False,

    # File paths
    'tokens_path': 'C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs_temperature_0-6/childhood_personality_development/tokens.json',
    'embeddings_path': 'C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs_temperature_0-6/childhood_personality_development/hidden_states_first.pt'
}

@dataclass
class ClusterResult:
    labels: np.ndarray
    centroids: Optional[np.ndarray]
    metrics: Dict[str, float]
    algorithm: str
    n_clusters: int

class DimensionalityReducer(ABC):
    @abstractmethod
    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        pass

class TSNEReducer(DimensionalityReducer):
    def __init__(self, n_components: int = DEFAULTS.get('n_components_tsne', 2), perplexity: float = DEFAULTS.get('perplexity', 30.0)):
        self.n_components = n_components
        self.perplexity = perplexity

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity,
                   random_state=42, n_iter=1000)
        return tsne.fit_transform(X)



class ClusteringAlgorithm(ABC):
    @abstractmethod
    def cluster(self, X: np.ndarray, **kwargs) -> ClusterResult:
        pass

class KMeansClusterer(ClusteringAlgorithm):
    def __init__(self, n_clusters: int = DEFAULTS.get('n_clusters', 8)):
        self.n_clusters = n_clusters

    def cluster(self, X: np.ndarray, **kwargs) -> ClusterResult:
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        metrics = self._compute_metrics(X, labels)

        return ClusterResult(
            labels=labels,
            centroids=kmeans.cluster_centers_,
            metrics=metrics,
            algorithm="KMeans",
            n_clusters=self.n_clusters
        )

    def _compute_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if len(np.unique(labels)) < 2:
            return {"silhouette": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": float('inf')}

        return {
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels)
        }



class HierarchicalClusterer(ClusteringAlgorithm):
    def __init__(self, n_clusters: int = DEFAULTS.get('n_clusters', 8), linkage_method: str = DEFAULTS.get('linkage_method_hierarchical', 'ward')):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method

    def cluster(self, X: np.ndarray, **kwargs) -> ClusterResult:
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             linkage=self.linkage_method)
        labels = hierarchical.fit_predict(X)

        metrics = self._compute_metrics(X, labels)

        return ClusterResult(
            labels=labels,
            centroids=None,
            metrics=metrics,
            algorithm="Hierarchical",
            n_clusters=self.n_clusters
        )

    def _compute_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if len(np.unique(labels)) < 2:
            return {"silhouette": 0.0, "calinski_harabasz": 0.0, "davies_bouldin": float('inf')}

        return {
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels)
        }

class StructuralFilter:
    def __init__(self, variance_threshold: float = DEFAULTS.get('variance_threshold_structural', 0.1), isolation_threshold: float = DEFAULTS.get('isolation_threshold_structural', 0.8)):
        self.variance_threshold = variance_threshold
        self.isolation_threshold = isolation_threshold

    def identify_structural_clusters(self, embeddings: np.ndarray, tokens: List[str],
                                   cluster_labels: np.ndarray) -> List[int]:
        structural_clusters = []
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            if label == -1:  # noise cluster
                continue

            cluster_mask = cluster_labels == label
            cluster_embeddings = embeddings[cluster_mask]
            cluster_tokens = [tokens[i] for i in np.where(cluster_mask)[0]]

            if self._is_structural_cluster(cluster_embeddings, cluster_tokens):
                structural_clusters.append(label)

        return structural_clusters


    def _is_structural_cluster(self, embeddings: np.ndarray, tokens: List[str]) -> bool:
        if len(embeddings) < 3:
            return True

        # Check for pure structural token clusters first
        if self._is_pure_structural_cluster(tokens):
            return True

        # For mixed content, apply existing logic
        embedding_variance = np.mean(np.var(embeddings, axis=0))
        structural_ratio = self._detect_structural_patterns(tokens)

        if structural_ratio > 0.8:
            return True

        if embedding_variance < self.variance_threshold and structural_ratio > 0.6:
            return True

        if self._has_semantic_diversity(tokens):
            return False

        unique_tokens = set(tokens)
        if len(unique_tokens) == 1:
            return True

        repetition_ratio = 1 - (len(unique_tokens) / len(tokens))
        return repetition_ratio > 0.9 and structural_ratio > 0.5

    def _is_pure_structural_cluster(self, tokens: List[str]) -> bool:
        """Check if cluster contains exclusively structural tokens"""
        structural_chars = set('*.,;:-\n "\'()[]{}')

        for token in tokens:
            # Check if token contains any non-structural characters
            if any(c not in structural_chars for c in token):
                return False

        return True

    def _has_semantic_diversity(self, tokens: List[str]) -> bool:
        """Check if tokens represent semantic concepts rather than pure structure"""
        content_tokens = [t.strip() for t in tokens if t.strip()]
        if not content_tokens:
            return False

        semantic_indicators = 0
        for token in content_tokens:
            clean_token = token.lower().strip()

            # Skip pure formatting/punctuation tokens
            if all(c in '*.,;:-\n "\'()[]{}' for c in token):
                continue

            if len(clean_token) < 2:
                continue

            if clean_token.isalpha() and len(clean_token) > 2:
                semantic_indicators += 1


        return semantic_indicators > len(content_tokens) * 0.4

    def _detect_structural_patterns(self, tokens: List[str]) -> float:
        structural_count = 0

        for token in tokens:
            if self._is_structural_token(token):
                structural_count += 1

        return structural_count / len(tokens) if tokens else 0.0

    def _is_structural_token(self, token: str) -> bool: # GGG
        # Punctuation and formatting
        if len(token) == 1 and not token.isalnum():
            return True

        structural_words = {'.', ',', '-', '--', '\n', ':', '*', '**', '"', '\n\n', ' ', ':', '.\n', '**\n\n', '*\n\n', '**\n' '<\uff5cbegin\u2581of\u2581sentence\uff5c>'}
        if token.lower() in structural_words:
            return True

        # Numeric patterns
        if token.isdigit() or token.replace('.', '').replace(',', '').isdigit():
            return True

        return False

class SemanticAnalyzer:
    def __init__(self):
        self.embeddings = None
        self.tokens = None
        self.scaler = StandardScaler()
        self.structural_filter = StructuralFilter()

    def load_data(self, embeddings: np.ndarray):
        self.embeddings = self.scaler.fit_transform(embeddings)

    def find_optimal_clusters(self, clustering_algorithms: List[ClusteringAlgorithm],
                            k_range: range = DEFAULTS.get('k_range_semantic', range(2, 15))) -> Dict[str, ClusterResult]:
        results = {}

        for algorithm in clustering_algorithms:
            if isinstance(algorithm, (KMeansClusterer, HierarchicalClusterer)):
                best_result = None
                best_score = -1

                for k in k_range:
                    algorithm.n_clusters = k
                    result = algorithm.cluster(self.embeddings)

                    if result.metrics['silhouette'] > best_score:
                        best_score = result.metrics['silhouette']
                        best_result = result

                results[algorithm.__class__.__name__] = best_result
            else:
                results[algorithm.__class__.__name__] = algorithm.cluster(self.embeddings)

        return results

    def filter_semantic_clusters(self, cluster_result: ClusterResult) -> Tuple[np.ndarray, List[int]]:
        structural_clusters = self.structural_filter.identify_structural_clusters(
            self.embeddings, self.tokens, cluster_result.labels
        )

        # Create filtered labels, setting structural clusters to -1
        filtered_labels = cluster_result.labels.copy()
        for struct_cluster in structural_clusters:
            filtered_labels[filtered_labels == struct_cluster] = -1

        return filtered_labels, structural_clusters

    def refine_semantic_clusters(self, filtered_labels: np.ndarray,
                               algorithm: ClusteringAlgorithm) -> ClusterResult:
        # Extract semantic tokens and embeddings
        semantic_mask = filtered_labels != -1
        if np.sum(semantic_mask) < 10:  # Not enough semantic tokens
            return ClusterResult(filtered_labels, None, {}, "Refined", 0)

        semantic_embeddings = self.embeddings[semantic_mask]

        # Re-cluster semantic embeddings
        refined_result = algorithm.cluster(semantic_embeddings)

        # Map back to original indices
        final_labels = np.full(len(self.tokens), -1)
        semantic_indices = np.where(semantic_mask)[0]
        final_labels[semantic_indices] = refined_result.labels

        return ClusterResult(
            labels=final_labels,
            centroids=refined_result.centroids,
            metrics=refined_result.metrics,
            algorithm=f"Refined_{refined_result.algorithm}",
            n_clusters=refined_result.n_clusters
        )

    def analyze_clusters(self, cluster_result: ClusterResult) -> Dict[str, Any]:
        analysis = {
            'cluster_sizes': {},
            'cluster_tokens': {},
            'cluster_centroids': cluster_result.centroids,
            'metrics': cluster_result.metrics,
            'total_tokens': len(self.tokens),
            'n_clusters': cluster_result.n_clusters
        }

        unique_labels = np.unique(cluster_result.labels)

        for label in unique_labels:
            mask = cluster_result.labels == label
            cluster_tokens = [self.tokens[i] for i in np.where(mask)[0]]

            analysis['cluster_sizes'][int(label)] = len(cluster_tokens)
            analysis['cluster_tokens'][int(label)] = cluster_tokens[:20]  # First 20 tokens

        return analysis

    def initial_cluster(self, embeddings: np.ndarray, algorithm: ClusteringAlgorithm) -> ClusterResult:
        # Use the scaler fitted in `load_data` to transform (do not refit here).
        # This keeps the scaler consistent across the pipeline.
        try:
            scaled_embeddings = self.scaler.transform(embeddings)
        except Exception:
            # If scaler hasn't been fitted (defensive), fit on these embeddings.
            scaled_embeddings = self.scaler.fit_transform(embeddings)

        result = algorithm.cluster(scaled_embeddings)
        return result

    def assign_to_clusters(self, embeddings: np.ndarray, centroids: np.ndarray, algorithm_name: str) -> np.ndarray:
        scaled_embeddings = self.scaler.transform(embeddings) # Use transform, not fit_transform

        if algorithm_name == "KMeans":
            # Assign each point to the closest centroid
            distances = np.sqrt(((scaled_embeddings - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            return labels
        elif algorithm_name == "Hierarchical":
            # For hierarchical, we need to find the closest centroid from the initial clustering
            # This is a simplified assignment based on Euclidean distance to the fixed centroids
            # A more sophisticated approach might involve re-running hierarchical on the new data
            # and then mapping clusters, but for fixed centroids, this is the most direct way.
            distances = np.sqrt(((scaled_embeddings - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            return labels
        else:
            raise ValueError(f"Unsupported algorithm for assignment: {algorithm_name}")

    def calculate_cosine_similarity_distribution(self, embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> Dict[int, Dict[str, Any]]:
        metrics = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == -1: # Skip noise if any
                continue

            cluster_embeddings = embeddings[labels == label]
            centroid = centroids[label] # Assuming centroids are indexed by label

            # Calculate cosine similarity
            # Ensure embeddings and centroid are L2-normalized for dot product to be cosine similarity
            cluster_embeddings_norm = cluster_embeddings / np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
            centroid_norm = centroid / np.linalg.norm(centroid)

            cos_sims = np.dot(cluster_embeddings_norm, centroid_norm)

            metrics[int(label)] = {
                "mean_cos_sim": np.mean(cos_sims),
                "std_cos_sim": np.std(cos_sims),
                "min_cos_sim": np.min(cos_sims),
                "max_cos_sim": np.max(cos_sims),
                "cos_sim_distribution": cos_sims.tolist() # Store for plotting distribution
            }
        return metrics

    def calculate_cluster_size_metrics(self, embeddings: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                                       include_centroid_distances: bool = True,
                                       include_density: bool = True,
                                       include_intrinsic_dimensionality: bool = True) -> Dict[int, Dict[str, Any]]:
        size_metrics = {}
        unique_labels = np.unique(labels)
        MINIMUM_VARIANCE_EXPLANATION = DEFAULTS.get('min_variance_explanation_pca', 0.9)

        for label in unique_labels:
            if label == -1: # Skip noise if any
                continue

            cluster_embeddings = embeddings[labels == label]
            centroid = centroids[label]

            metrics_for_cluster = {
                "num_points": cluster_embeddings.shape[0]
            }

            if cluster_embeddings.shape[0] == 0:
                size_metrics[int(label)] = metrics_for_cluster
                continue

            # Radius: max distance from centroid
            distances_from_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            metrics_for_cluster["radius"] = np.max(distances_from_centroid)
            # Store per-point distances for downstream PDF plotting (useful as "variance" proxy)
            metrics_for_cluster["point_distances"] = distances_from_centroid.tolist()

            # Per-point variance across feature dims (another per-point signal)
            try:
                point_var = np.var(cluster_embeddings, axis=1)
                metrics_for_cluster["point_variances"] = point_var.tolist()
            except Exception:
                metrics_for_cluster["point_variances"] = []

            # Volume: compute using SVD on centered data (proper volume estimator)
            # For data matrix X (n_samples x d), centered: Xc = X - mean
            # SVD: Xc = U S V^T, singular values s_i relate to covariance eigenvalues
            # eigenvals = (s_i^2) / (n-1) -> sqrt(eigenvals) = s_i / sqrt(n-1)
            # Volume ~ product_i sqrt(eigenvals_i) = product_i (s_i / sqrt(n-1))
            if cluster_embeddings.shape[0] > 1:
                try:
                    Xc = cluster_embeddings - np.mean(cluster_embeddings, axis=0)
                    # Compute thin SVD
                    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
                    n_eff = max(1, cluster_embeddings.shape[0] - 1)
                    # scaled singular values correspond to sqrt(eigenvalues of covariance
                    scaled = s / np.sqrt(n_eff)
                    # if any scaled singulars are <= 0, volume is zero
                    if np.any(scaled <= 0):
                        metrics_for_cluster["volume"] = 0.0
                    else:
                        # compute product in log-space for numerical stability
                        log_vol = float(np.sum(np.log(scaled)))
                        metrics_for_cluster["volume"] = float(np.exp(log_vol))
                except Exception:
                    metrics_for_cluster["volume"] = 0.0
            else:
                metrics_for_cluster["volume"] = 0.0 # Single point has no volume

            # 2. Centroid-to-Point Distances (Mean/Median)
            if include_centroid_distances:
                metrics_for_cluster["mean_centroid_distance"] = np.mean(distances_from_centroid)
                metrics_for_cluster["median_centroid_distance"] = np.median(distances_from_centroid)

            # 3. Density
            if include_density:
                if cluster_embeddings.shape[0] > 1:
                    pairwise_distances = pdist(cluster_embeddings, metric='euclidean')
                    avg_pairwise_distance = np.mean(pairwise_distances)
                    if avg_pairwise_distance > 0:
                        metrics_for_cluster["density"] = cluster_embeddings.shape[0] / avg_pairwise_distance
                    else:
                        metrics_for_cluster["density"] = float('inf') # All points are identical
                else:
                    metrics_for_cluster["density"] = 0.0 # Single point has no density

            # 5. Intrinsic Dimensionality
            if include_intrinsic_dimensionality and cluster_embeddings.shape[0] > 1 and cluster_embeddings.shape[1] > 1:
                # Normalize before PCA (already scaled by self.scaler, but PCA works better with zero mean)
                # We can use the already scaled embeddings, PCA will center them internally.
                pca = PCA()
                pca.fit(cluster_embeddings)
                explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
                intrinsic_dim = np.where(explained_variance_ratio_cumsum >= MINIMUM_VARIANCE_EXPLANATION)[0]
                if len(intrinsic_dim) > 0:
                    metrics_for_cluster["intrinsic_dimensionality"] = intrinsic_dim[0] + 1 # +1 because it's 0-indexed
                else:
                    metrics_for_cluster["intrinsic_dimensionality"] = cluster_embeddings.shape[1] # All dimensions needed
            else:
                metrics_for_cluster["intrinsic_dimensionality"] = 0 # Not applicable or single point

            size_metrics[int(label)] = metrics_for_cluster
        return size_metrics

class Visualizer:
    def __init__(self, figsize: Tuple[int, int] = DEFAULTS.get('figsize_visualizer', (12, 8))):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')

    def plot_2d_clusters(self, embeddings: np.ndarray, labels: np.ndarray,
                        tokens: List[str], reducer: DimensionalityReducer,
                        title: str = "Cluster Visualization") -> go.Figure:

        # Reduce dimensionality
        coords_2d = reducer.fit_transform(embeddings)

        # Create color palette
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Set3[:len(unique_labels)]

        fig = go.Figure()

        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_coords = coords_2d[mask]
            cluster_tokens = [tokens[j] for j in np.where(mask)[0]]

            color = colors[i % len(colors)] if label != -1 else 'gray'
            name = f'Cluster {label}' if label != -1 else 'Noise'

            fig.add_trace(go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode='markers',
                marker=dict(color=color, size=6, opacity=0.7),
                text=cluster_tokens,
                name=name,
                hovertemplate='<b>%{text}</b><br>Cluster: ' + name + '<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title=f'{reducer.__class__.__name__} 1',
            yaxis_title=f'{reducer.__class__.__name__} 2',
            showlegend=True,
            width=800,
            height=600
        )

        return fig

    def plot_cluster_metrics(self, results: Dict[str, ClusterResult]) -> go.Figure:
        algorithms = list(results.keys())
        silhouette_scores = [results[alg].metrics.get('silhouette', 0) for alg in algorithms]
        calinski_scores = [results[alg].metrics.get('calinski_harabasz', 0) for alg in algorithms]
        davies_scores = [results[alg].metrics.get('davies_bouldin', 0) for alg in algorithms]

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )

        fig.add_trace(go.Bar(x=algorithms, y=silhouette_scores, name='Silhouette'), row=1, col=1)
        fig.add_trace(go.Bar(x=algorithms, y=calinski_scores, name='Calinski-Harabasz'), row=1, col=2)
        fig.add_trace(go.Bar(x=algorithms, y=davies_scores, name='Davies-Bouldin'), row=1, col=3)

        fig.update_layout(title="Clustering Algorithm Comparison", showlegend=False)
        return fig

    def plot_cluster_distribution(self, cluster_result: ClusterResult) -> go.Figure:
        unique_labels, counts = np.unique(cluster_result.labels, return_counts=True)

        # Sort by count
        sorted_indices = np.argsort(counts)[::-1]
        sorted_labels = unique_labels[sorted_indices]
        sorted_counts = counts[sorted_indices]

        # Create labels for display
        display_labels = [f'Cluster {label}' if label != -1 else 'Noise'
                         for label in sorted_labels]

        fig = go.Figure(data=[go.Bar(x=display_labels, y=sorted_counts)])
        fig.update_layout(
            title="Cluster Size Distribution",
            xaxis_title="Clusters",
            yaxis_title="Number of Tokens"
        )

        return fig

    def plot_window_metric_pdfs(self, window_results: List[Dict[str, Any]], metric: str = 'cosine', cluster_id: Optional[int] = None,
                                 bandwidth: Optional[float] = None, title: Optional[str] = None,
                                 pdf_point_mode: str = 'both') -> go.Figure:
        """Plot evolution of a per-window metric (PDF) for each window step for a specific cluster.

        metric options: 'cosine', 'variance', 'volume', 'radius'
        If cluster_id is None, create one figure per cluster by returning a single combined figure (all traces).
        """
        fig = go.Figure()

        # pdf_point_mode controls how raw datapoints and KDE are shown:
        #   'raw_points' -> only show raw datapoints (no KDE lines)
        #   'kde'        -> only show KDE lines (no raw datapoint markers)
        #   'both'       -> show both KDE lines and raw datapoint markers

        show_raw = pdf_point_mode in ('raw_points', 'both')
        show_kde = pdf_point_mode in ('kde', 'both')

        # Iterate windows and add KDE traces for the selected metric for the given cluster
        # Use a darker qualitative palette so traces are easy to see
        # and ensure different windows get distinct colors.
        try:
            colors = px.colors.qualitative.Dark24
        except Exception:
            # Fallback to a commonly available palette
            colors = px.colors.qualitative.Set1
        for w in window_results:
            window_idx = w['window_num']
            # Determine the per-cluster metric values
            if metric == 'cosine':
                # extract mean_cos_sim per cluster and use underlying distribution if present
                metric_dict = w['cosine_similarity_metrics']
                values = []
                if cluster_id is None:
                    # sum across clusters for this window (not typical) - skip
                    continue
                else:
                    if int(cluster_id) in metric_dict:
                        values = metric_dict[int(cluster_id)].get('cos_sim_distribution', [])
                    else:
                        values = []
            elif metric == 'variance':
                metric_dict = w['cluster_size_metrics']
                if int(cluster_id) in metric_dict:
                    values = metric_dict[int(cluster_id)].get('point_variances', [])
                else:
                    values = []
            elif metric == 'volume':
                metric_dict = w['cluster_size_metrics']
                if int(cluster_id) in metric_dict:
                    # Volume is a single value per cluster; replicate it to make a degenerate distribution
                    vol = metric_dict[int(cluster_id)].get('volume', 0.0)
                    values = [vol] * max(3, metric_dict[int(cluster_id)].get('num_points', 1))
                else:
                    values = []
            elif metric == 'radius':
                metric_dict = w['cluster_size_metrics']
                if int(cluster_id) in metric_dict:
                    # replicate per-point distance to form a distribution
                    values = metric_dict[int(cluster_id)].get('point_distances', [])
                else:
                    values = []
            else:
                raise ValueError('Unknown metric for PDF plotting')

            if len(values) == 0:
                continue

            color = colors[window_idx % len(colors)] if colors else None

            # Compute histogram-based density once for stable y-scaling (bins chosen adaptively)
            try:
                n_bins = min(50, max(5, int(len(values) // 2)))
                hist, bin_edges = np.histogram(values, bins=n_bins, density=True)
                xs_hist = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ys_hist = hist
            except Exception:
                xs_hist = None
                ys_hist = None

            # Plot histogram density as primary curve when KDE lines are requested
            if show_kde and ys_hist is not None:
                fig.add_trace(go.Scatter(
                    x=xs_hist,
                    y=ys_hist,
                    mode='lines',
                    line=dict(width=1.5, color=color),
                    name=f'window_{window_idx}',
                    opacity=0.75,
                    hoverinfo='skip'
                ))

                # Overlay a smoothed KDE rescaled to histogram peak for visual smoothness
                try:
                    kde = gaussian_kde(values)
                    xs_kde = np.linspace(min(values), max(values), 200)
                    ys_kde = kde(xs_kde)
                    if ys_kde.max() > 0 and ys_hist.max() > 0:
                        ys_kde = ys_kde * (ys_hist.max() / ys_kde.max())
                    fig.add_trace(go.Scatter(
                        x=xs_kde,
                        y=ys_kde,
                        mode='lines',
                        line=dict(width=1, color=color, dash='dot'),
                        name=f'window_{window_idx}_kde',
                        opacity=0.45,
                        hoverinfo='skip'
                    ))
                except Exception:
                    pass

            # Plot raw datapoints for this window when requested.
            # Draw a connected line through the raw points (sorted by x) and
            # highlight points with markers. Use transparency so multiple windows
            # can be overplotted and the time evolution remains visible.
            if show_raw:
                vals_np = np.array(values)

                if ys_hist is not None and xs_hist is not None and len(vals_np) > 0:
                    # Map each value to its histogram bin density
                    bin_idx = np.searchsorted(bin_edges, vals_np, side='right') - 1
                    bin_idx = np.clip(bin_idx, 0, len(ys_hist) - 1)
                    marker_y = (ys_hist[bin_idx] * 0.9).astype(float)

                    # Sort by x so the connecting line is monotonic and readable
                    sort_idx = np.argsort(vals_np)
                    xs_sorted = vals_np[sort_idx]
                    ys_sorted = marker_y[sort_idx]

                    fig.add_trace(go.Scatter(
                        x=xs_sorted,
                        y=ys_sorted,
                        mode='lines+markers',
                        line=dict(width=2, color=color),
                        marker=dict(symbol='circle', size=6, opacity=0.9, color=color, line=dict(width=0)),
                        opacity=0.7,
                        name=f'window_{window_idx}_points',
                        hovertemplate='value: %{x}<extra></extra>'
                    ))
                else:
                    # Fallback: place markers/line at a small constant y for visibility
                    if len(vals_np) > 0:
                        xs_sorted = np.sort(vals_np)
                        ys_const = [0.01] * len(xs_sorted)
                        fig.add_trace(go.Scatter(
                            x=xs_sorted,
                            y=ys_const,
                            mode='lines+markers',
                            line=dict(width=1.5, color=color),
                            marker=dict(symbol='circle', size=6, opacity=0.85, color=color, line=dict(width=0)),
                            opacity=0.65,
                            name=f'window_{window_idx}_points',
                            hovertemplate='value: %{x}<extra></extra>'
                        ))

        fig.update_layout(
            title=title or f'Window-wise PDF evolution (metric={metric}, cluster={cluster_id})',
            xaxis_title=metric,
            yaxis_title='Density',
            showlegend=True,
            width=800,
            height=500
        )

        return fig

    def plot_window_metric_timeline(self, window_results: List[Dict[str, Any]], metric: str = 'cosine', cluster_id: Optional[int] = None,
                                    title: Optional[str] = None, alpha: float = 0.35, marker_size: int = 6) -> go.Figure:
        """Plot metric (x) vs window step (y) as a scatter for each datapoint in the cluster.

        This creates a vertical timeline-style plot where window index is on the y axis and
        the metric value is on the x axis. Overlap intensity shows density per window.
        """
        xs_all = []
        ys_all = []

        fig = go.Figure()

        for w in window_results:
            window_idx = w['window_num']

            if metric == 'cosine':
                metric_dict = w['cosine_similarity_metrics']
                vals = metric_dict.get(int(cluster_id), {}).get('cos_sim_distribution', []) if int(cluster_id) in metric_dict else []
            elif metric == 'variance':
                metric_dict = w['cluster_size_metrics']
                vals = metric_dict.get(int(cluster_id), {}).get('point_variances', []) if int(cluster_id) in metric_dict else []
            elif metric == 'volume':
                metric_dict = w['cluster_size_metrics']
                if int(cluster_id) in metric_dict:
                    vol = metric_dict[int(cluster_id)].get('volume', 0.0)
                    n = max(3, metric_dict[int(cluster_id)].get('num_points', 1))
                    vals = [vol] * n
                else:
                    vals = []
            elif metric == 'radius':
                metric_dict = w['cluster_size_metrics']
                vals = metric_dict.get(int(cluster_id), {}).get('point_distances', []) if int(cluster_id) in metric_dict else []
            else:
                raise ValueError('Unknown metric for timeline plotting')

            if not vals:
                continue

            ys = [window_idx] * len(vals)
            xs = vals

            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='markers',
                marker=dict(size=marker_size, opacity=alpha, color='rgba(31,119,180,0.6)', line=dict(width=0)),
                hoverinfo='x+y',
                name=f'window_{window_idx}'
            ))

        fig.update_layout(
            title=title or f'Window timeline (metric={metric}, cluster={cluster_id})',
            xaxis_title=metric,
            yaxis_title='Window step',
            showlegend=False,
            width=800,
            height=500
        )

        return fig

class SemanticPipeline:
    def __init__(self):
        self.analyzer = SemanticAnalyzer()
        self.visualizer = Visualizer()

        # Available algorithms
        self.clustering_algorithms = {
            'kmeans': KMeansClusterer,
            'hierarchical': HierarchicalClusterer
        }

        self.dimensionality_reducers = {
            'tsne': TSNEReducer
        }

    def run_analysis(self, all_tokens: List[str], all_embeddings: np.ndarray,
                    n_initial_hidden_states: int,
                    initial_clustering_method: str = DEFAULTS.get('initial_clustering_method', 'kmeans'),
                    n_clusters: int = DEFAULTS.get('n_clusters', 8),
                    window_type: str = DEFAULTS.get('window_type', 'expanding'), # 'expanding' or 'moving'
                    window_size: int = DEFAULTS.get('window_size', 100),
                    stride: int = DEFAULTS.get('stride', 50),
                    visualization_method: str = DEFAULTS.get('visualization_method', 'tsne')) -> Dict[str, Any]:

        # Load all data (embeddings are already scaled in analyzer.load_data)
        self.analyzer.load_data(all_embeddings)
        self.analyzer.tokens = all_tokens # Store tokens in analyzer for structural filtering

        # 1. Initial Clustering to get fixed centroids
        print(f"Performing initial clustering on first {n_initial_hidden_states} hidden states...")
        initial_embeddings = all_embeddings[:n_initial_hidden_states]
        initial_clusterer = self.clustering_algorithms[initial_clustering_method](n_clusters=n_clusters)
        initial_cluster_result = self.analyzer.initial_cluster(initial_embeddings, initial_clusterer)
        fixed_centroids = initial_cluster_result.centroids
        initial_labels = initial_cluster_result.labels

        if fixed_centroids is None:
            raise ValueError("Initial clustering did not produce centroids. Cannot proceed with fixed centroids.")

        print(f"Initial clustering complete. Found {initial_cluster_result.n_clusters} clusters.")

        # Store initial clustering results
        results = {
            'initial_cluster_result': initial_cluster_result,
            'fixed_centroids': fixed_centroids,
            'window_results': []
        }

        # 2. Sliding Window Analysis
        print("Starting sliding window analysis...")
        total_embeddings_len = all_embeddings.shape[0]
        current_start_idx = 0
        window_num = 0

        while True:
            window_end_idx = current_start_idx + window_size

            if window_type == 'expanding':
                window_embeddings = all_embeddings[:window_end_idx]
                window_tokens = all_tokens[:window_end_idx]
            elif window_type == 'moving':
                if window_end_idx > total_embeddings_len:
                    break # Reached end of data
                window_embeddings = all_embeddings[current_start_idx:window_end_idx]
                window_tokens = all_tokens[current_start_idx:window_end_idx]
            else:
                raise ValueError("Invalid window_type. Must be 'expanding' or 'moving'.")

            if window_embeddings.shape[0] == 0:
                break # No more data in window

            print(f"Processing window {window_num}: indices {current_start_idx}-{window_end_idx-1}")

            # Assign current window embeddings to fixed centroids
            # Use the algorithm name returned by the initial clustering result (not the clusterer instance)
            window_labels = self.analyzer.assign_to_clusters(
                window_embeddings, fixed_centroids, initial_cluster_result.algorithm
            )

            # Filter structural clusters (using the analyzer's stored tokens and embeddings for context)
            # Note: Structural filtering is applied to the *current window's* labels and embeddings
            # It's important that the structural filter operates on the actual tokens/embeddings of the window
            # For this, we temporarily set the analyzer's tokens/embeddings to the current window's data
            original_analyzer_embeddings = self.analyzer.embeddings
            original_analyzer_tokens = self.analyzer.tokens

            self.analyzer.embeddings = window_embeddings # Temporarily set for structural filter
            self.analyzer.tokens = window_tokens # Temporarily set for structural filter

            filtered_labels, structural_clusters = self.analyzer.filter_semantic_clusters(
                ClusterResult(labels=window_labels, centroids=None, metrics={}, algorithm="", n_clusters=0)
            )

            # Restore original analyzer state
            self.analyzer.embeddings = original_analyzer_embeddings
            self.analyzer.tokens = original_analyzer_tokens

            # Calculate cosine similarity distribution
            cos_sim_metrics = self.analyzer.calculate_cosine_similarity_distribution(
                window_embeddings, filtered_labels, fixed_centroids
            )

            # Calculate cluster size metrics
            cluster_size_metrics = self.analyzer.calculate_cluster_size_metrics(
                window_embeddings, filtered_labels, fixed_centroids,
                include_centroid_distances=True, # Example: enable this metric
                include_density=True, # Example: enable this metric
                include_intrinsic_dimensionality=True # Example: enable this metric
            )

            window_results = {
                'window_num': window_num,
                'start_idx': current_start_idx,
                'end_idx': window_end_idx - 1,
                'labels': window_labels,
                'filtered_labels': filtered_labels,
                'structural_clusters': structural_clusters,
                'cosine_similarity_metrics': cos_sim_metrics,
                'cluster_size_metrics': cluster_size_metrics
            }
            results['window_results'].append(window_results)

            if window_type == 'expanding':
                if window_end_idx >= total_embeddings_len:
                    break # Reached end of data
                current_start_idx += stride # For expanding, stride moves the "end"
            elif window_type == 'moving':
                current_start_idx += stride
                if current_start_idx >= total_embeddings_len:
                    break # Moved past end of data

            window_num += 1

        print("Sliding window analysis complete.")
        return results

class DendrogramVisualizer:
    def __init__(self, max_display_tokens: int = DEFAULTS.get('max_display_tokens_dendrogram', 50)):
        self.max_display_tokens = max_display_tokens

    def load_data(self, tokens: List[str], embeddings: np.ndarray) -> tuple[List[str], np.ndarray]:
        return tokens, embeddings

    def create_dendrogram(self, embeddings: np.ndarray, tokens: List[str],
                         linkage_method: str = DEFAULTS.get('linkage_method_hierarchical', 'ward'),
                         metric: str = DEFAULTS.get('metric_dendrogram', 'euclidean'),
                         truncate_mode: Optional[str] = DEFAULTS.get('truncate_mode_dendrogram', 'lastp')) -> go.Figure:

        n_samples = min(len(tokens), self.max_display_tokens)
        sample_indices = np.random.choice(len(tokens), n_samples, replace=False)
        sample_embeddings = embeddings[sample_indices]
        sample_tokens = [tokens[i] for i in sample_indices]

        distances = pdist(sample_embeddings, metric=metric)
        linkage_matrix = linkage(distances, method=linkage_method)

        # Create dendrogram structure
        dend_data = dendrogram(
            linkage_matrix,
            labels=sample_tokens,
            no_plot=True,
            truncate_mode=truncate_mode,
            p=min(DEFAULTS.get('max_dendrogram_leaves', 30), n_samples) if truncate_mode else None
        )

        return self._build_plotly_dendrogram(dend_data, linkage_method, metric)

    def _build_plotly_dendrogram(self, dend_data: dict,
                                linkage_method: str, metric: str) -> go.Figure:
        fig = go.Figure()

        # Add dendrogram branches
        for xs, ys in zip(dend_data['icoord'], dend_data['dcoord']):
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(color='#2E86AB', width=1.5),
                hoverinfo='skip',
                showlegend=False
            ))

        # Add leaf labels
        leaf_positions = [(dend_data['icoord'][i][1] + dend_data['icoord'][i][2]) / 2
                         for i in range(len(dend_data['icoord']))][:len(dend_data['ivl'])]

        fig.add_trace(go.Scatter(
            x=leaf_positions,
            y=[0] * len(dend_data['ivl']),
            mode='text',
            text=dend_data['ivl'],
            textposition='bottom center',
            textfont=dict(size=9),
            hovertemplate='<b>%{text}</b><extra></extra>',
            showlegend=False
        ))

        fig.update_layout(
            title=f'Hierarchical Clustering Dendrogram<br><sub>{linkage_method.title()} linkage, {metric} distance</sub>',
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title='Distance',
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            plot_bgcolor='white',
            height=700,
            margin=dict(b=120, t=80),
            font=dict(size=11)
        )

        return fig

    def create_comparison_plot(self, embeddings: np.ndarray, tokens: List[str],
                              methods: List[str] = DEFAULTS.get('comparison_methods_dendrogram', ['ward', 'complete', 'average'])) -> go.Figure:
        n_methods = len(methods)
        fig = make_subplots(
            rows=n_methods, cols=1,
            subplot_titles=[f'{method.title()} Linkage' for method in methods],
            vertical_spacing=0.08
        )

        n_samples = min(len(tokens), self.max_display_tokens)
        sample_indices = np.random.choice(len(tokens), n_samples, replace=False)
        sample_embeddings = embeddings[sample_indices]
        sample_tokens = [tokens[i] for i in sample_indices]

        for i, method in enumerate(methods):
            metric = 'euclidean'
            distances = pdist(sample_embeddings, metric=metric)
            linkage_matrix = linkage(distances, method=method)

            dend_data = dendrogram(linkage_matrix, labels=sample_tokens, no_plot=True,
                                   truncate_mode=None)

            # Add branches for this subplot
            for xs, ys in zip(dend_data['icoord'], dend_data['dcoord']):
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode='lines',
                    line=dict(color=px.colors.qualitative.Set2[i], width=1.2),
                    hoverinfo='skip',
                    showlegend=False
                ), row=i+1, col=1)

        fig.update_layout(
            title='Hierarchical Clustering Method Comparison',
            height=300 * n_methods,
            showlegend=False
        )

        for i in range(n_methods):
            fig.update_xaxes(showticklabels=False, row=i+1, col=1)
            fig.update_yaxes(title_text='Distance', row=i+1, col=1)

        return fig



"""
this code clusters and visualizes the hidden_states. It tries to find the "structural clusters" which correspond to tokens like

{'**', '**\n', '**\n\n', ' **', ':**'} (there is some hard-coding used here but for the most part we used ml algorithms)

which dont carry semantic meaning and are uninteresting to us. then it excludes them from the analysis.

finally it is able to compare between different clustering methods (KMeans, Hierarchical) to determine which is best based on some standard "clustering quality measures" like silhouette, calinski_harabasz, davies_bouldin

we also look at things like the cluster size distribution: if there is a cluster with 90% of the tokens and then a lot of clusters with a single token, then it is a bad clustering technique for our use case.

like this we investigate clustering and find some nice methods which are semantically meaningful.

for visualization we used tsne. we used libraries that implemented them and it was straightforward to try out
"""

### Section: Visualization and running nicely and report making


# pipeline for evaluating and displaying results
def usage(all_tokens: List[str], all_embeddings: np.ndarray,
          n_initial_hidden_states: int,
          initial_clustering_method: str = DEFAULTS.get('initial_clustering_method', 'kmeans'),
          n_clusters: int = DEFAULTS.get('n_clusters', 8),
          window_type: str = DEFAULTS.get('window_type', 'expanding'),
          window_size: int = DEFAULTS.get('window_size', 100),
          stride: int = DEFAULTS.get('stride', 50),
          visualization_method: str = DEFAULTS.get('visualization_method', 'tsne')):
    pipeline = SemanticPipeline()

    results = pipeline.run_analysis(
        all_tokens=all_tokens,
        all_embeddings=all_embeddings,
        n_initial_hidden_states=n_initial_hidden_states,
        initial_clustering_method=initial_clustering_method,
        n_clusters=n_clusters,
        window_type=window_type,
        window_size=window_size,
        stride=stride,
        visualization_method=visualization_method
    )

    # Display results
    print(f"Initial clustering result: {results['initial_cluster_result'].algorithm} with {results['initial_cluster_result'].n_clusters} clusters.")
    print(f"Number of windows processed: {len(results['window_results'])}")

    # Print detailed results for each window
    for i, window_res in enumerate(results['window_results']):
        print(f"\n--- Window {i} (indices {window_res['start_idx']}-{window_res['end_idx']}) ---")
        print(f"  Structural clusters identified: {window_res['structural_clusters']}")
        print("  Cosine Similarity Metrics per Cluster:")
        for cluster_id, metrics in window_res['cosine_similarity_metrics'].items():
            print(f"    Cluster {cluster_id}: Mean Cos Sim = {metrics['mean_cos_sim']:.4f}, Std Dev = {metrics['std_cos_sim']:.4f}")
        print("  Cluster Size Metrics per Cluster:")
        for cluster_id, metrics in window_res['cluster_size_metrics'].items():
            print(f"    Cluster {cluster_id}: Num Points = {metrics['num_points']}, Radius = {metrics['radius']:.4f}, Volume = {metrics['volume']:.4f}")
            if 'mean_centroid_distance' in metrics: print(f"      Mean Centroid Dist = {metrics['mean_centroid_distance']:.4f}, Median Centroid Dist = {metrics['median_centroid_distance']:.4f}")
            if 'density' in metrics: print(f"      Density = {metrics['density']:.4f}")
            if 'intrinsic_dimensionality' in metrics: print(f"      Intrinsic Dim = {metrics['intrinsic_dimensionality']}")

def visualize_hierarchical_clustering(all_tokens: List[str], all_embeddings: np.ndarray,
                                    linkage_method: str = DEFAULTS.get('linkage_method_hierarchical', 'ward'),
                                    show_comparison: bool = DEFAULTS.get('show_comparison_hierarchical', False)):
    viz = DendrogramVisualizer(max_display_tokens=60)
    tokens, embeddings = viz.load_data(all_tokens, all_embeddings)

    print(f"Loaded {len(tokens)} tokens with {embeddings.shape[1]}D embeddings")

    fig = viz.create_dendrogram(embeddings, tokens, linkage_method=linkage_method)
    fig.show()

    if show_comparison:
        comparison_fig = viz.create_comparison_plot(embeddings, tokens)
        comparison_fig.show()


def initial_tsne_and_summaries(all_tokens: List[str], all_embeddings: np.ndarray,
                               N: int = DEFAULTS.get('n_initial_hidden_states', 1024), 
                               n_clusters: int = DEFAULTS.get('n_clusters', 8),
                               perplexity: float = DEFAULTS.get('perplexity', 30.0), 
                               sample_top: int = DEFAULTS.get('sample_top', 12),
                               inline: bool = DEFAULTS.get('inline', True)) -> Tuple[go.Figure, Dict[int, Dict[str, Any]], ClusterResult]:
    """Run initial KMeans on first N embeddings, produce TSNE figure and concise summaries.

    Returns (fig, summaries, cluster_result). summaries[label] = {'count': int, 'unique_tokens': [..]}
    If inline=True the figure will be shown (suitable for notebooks).
    """
    initial_embeddings = all_embeddings[:N]
    initial_tokens = all_tokens[:N]

    pipeline = SemanticPipeline()
    pipeline.analyzer.load_data(initial_embeddings)

    clusterer = KMeansClusterer(n_clusters=n_clusters)
    cluster_result = pipeline.analyzer.initial_cluster(initial_embeddings, clusterer)

    reducer = TSNEReducer(n_components=2, perplexity=perplexity)
    fig = pipeline.visualizer.plot_2d_clusters(initial_embeddings, cluster_result.labels, initial_tokens, reducer,
                                               title=f"Initial KMeans (n={n_clusters}) TSNE")

    # Do not auto-show here; caller will display the figure once to avoid duplicate outputs in notebooks

    summaries: Dict[int, Dict[str, Any]] = {}
    unique_labels = np.unique(cluster_result.labels)
    for label in unique_labels:
        mask = cluster_result.labels == label
        cluster_tokens_full = [initial_tokens[i] for i in np.where(mask)[0]]
        count = len(cluster_tokens_full)
        unique_tokens = list(dict.fromkeys(cluster_tokens_full))
        summaries[int(label)] = {
            'count': int(count),
            'unique_tokens': unique_tokens[:sample_top]
        }

    # Print concise human-friendly summary (no repetition)
    for label, info in summaries.items():
        sample_display = ', '.join(repr(t) for t in info['unique_tokens'])
        print(f"Cluster {int(label)} ({info['count']} tokens):\n  Sample tokens: {{{sample_display}}}\n")

    return fig, summaries, cluster_result




def usage_example(all_tokens: List[str], all_embeddings: np.ndarray, params: Optional[Dict[str, Any]] = None):
    """Run a standard example using DEFAULTS; prints where to change parameters.

    Returns the figure and summaries.
    """
    # params may be None; the user can set DEFAULTS in the usage block below
    if params is None:
        params = DEFAULTS

    N = params.get('n_initial_hidden_states', 1024)
    n_clusters = params.get('n_clusters', 8)
    perplexity = params.get('perplexity', 30.0)
    sample_top = params.get('sample_top', 12)
    inline = params.get('inline', True)

    # Support list of initial methods
    methods = params.get('initial_clustering_method', 'kmeans')
    if isinstance(methods, str):
        methods = [methods]

    outputs = {}

    for method in methods:
        print(f"Running usage_example for initial_clustering_method={method}")
        fig, summaries, cluster_result = initial_tsne_and_summaries(
            all_tokens, all_embeddings,
            N=N, n_clusters=n_clusters, perplexity=perplexity,
            sample_top=sample_top, inline=False
        )

        # Show the TSNE plot once per method
        if inline:
            fig.show()

        # Run sliding-window full pipeline for the chosen method to collect window metrics for PDF plotting
        pipeline = SemanticPipeline()
        results = pipeline.run_analysis(
            all_tokens=all_tokens,
            all_embeddings=all_embeddings,
            n_initial_hidden_states=N,
            initial_clustering_method=method,
            n_clusters=n_clusters,
            window_type=params.get('window_type', 'expanding'),
            window_size=params.get('window_size', 512),
            stride=params.get('stride', 512),
            visualization_method=params.get('visualization_method', 'tsne')
        )

        # For each cluster, create one figure per metric and per selected plot method
        cluster_ids = list(range(cluster_result.n_clusters))
        metrics_to_plot = params.get('metrics_to_plot', ['cosine', 'variance', 'volume', 'radius'])
        plot_methods = params.get('pdf_plot_methods', ['kde'])
        
        pdf_point_mode = params.get('pdf_point_mode', 'both')

        pdf_figs = {}
        for cid in cluster_ids:
            for metric in metrics_to_plot:
                for pmethod in plot_methods:
                    title = f"Method={method} | Cluster={cid} | Metric={metric} | Plot={pmethod}"
                    if pmethod == 'kde':
                        pdf_fig = pipeline.visualizer.plot_window_metric_pdfs(results['window_results'], metric=metric, cluster_id=cid, title=title, pdf_point_mode=pdf_point_mode)
                        if inline:
                            pdf_fig.show()
                    elif pmethod == 'timeline':
                        pdf_fig = pipeline.visualizer.plot_window_metric_timeline(results['window_results'], metric=metric, cluster_id=cid, title=title)
                        if inline:
                            pdf_fig.show()
                    else:
                        raise ValueError(f'Unknown pdf plot method: {pmethod}')

                    pdf_figs[(cid, metric, pmethod)] = pdf_fig

        outputs[method] = {
            'tsne_fig': fig,
            'summaries': summaries,
            'cluster_result': cluster_result,
            'window_results': results['window_results'],
            'pdf_figs': pdf_figs
        }

    # Additional example: hierarchical clustering + dendrogram visualization
    try:
        print('\nRunning hierarchical/dendrogram example...')
        viz = DendrogramVisualizer(max_display_tokens=DEFAULTS.get('max_display_tokens_dendrogram', 50))
        tokens_sample, embeddings_sample = viz.load_data(all_tokens, all_embeddings)

        # Single dendrogram using the default linkage
        dend_fig = viz.create_dendrogram(embeddings_sample, tokens_sample, linkage_method=DEFAULTS.get('linkage_method_hierarchical', 'ward'))
        outputs['dendrogram'] = dend_fig
        if inline:
            dend_fig.show()

        # Comparison plot across all configured methods (uses existing create_comparison_plot)
        methods = DEFAULTS.get('comparison_methods_dendrogram', ['ward', 'complete', 'average'])
        comp_fig = viz.create_comparison_plot(embeddings_sample, tokens_sample, methods=methods)
        outputs['dendrogram_comparison'] = comp_fig
        if inline:
            comp_fig.show()
    except Exception as e:
        print('Dendrogram example failed:', e)

    print("\nParameters used (edit DEFAULTS at top of file to change):")
    for k in ['n_initial_hidden_states', 'initial_clustering_method', 'n_clusters', 'window_type', 'window_size', 'stride', 'visualization_method', 'perplexity', 'sample_top']:
        print(f"  {k}: {params.get(k)}")

    return outputs





### Section: Usage Example
# DEFAULTS for interactive use / notebook. Move or edit these values when running experiments.


if __name__ == '__main__':
    # Minimal script-mode demo: load tokens & embeddings and run the usage_example once.
    with open(DEFAULTS.get('tokens_path'), 'r') as f:
        all_tokens = json.load(f)

    embeddings_tensor = torch.load(DEFAULTS.get('embeddings_path'), map_location='cpu')

    # Handle different tensor shapes
    if embeddings_tensor.dim() == 3:
        all_embeddings = embeddings_tensor.view(-1, embeddings_tensor.size(-1)).numpy()
    elif embeddings_tensor.dim() == 2:
        all_embeddings = embeddings_tensor.T.numpy()
    else:
        raise ValueError(f"Unexpected embedding tensor shape: {embeddings_tensor.shape}")

    # Trim to matching lengths if necessary
    if len(all_tokens) != all_embeddings.shape[0]:
        min_len = min(len(all_tokens), all_embeddings.shape[0])
        all_tokens = all_tokens[:min_len]
        all_embeddings = all_embeddings[:min_len]

    # Run the unified usage example (it will show TSNE and pdf/timeline plots once per method)
    usage_example(all_tokens, all_embeddings, params=DEFAULTS)
