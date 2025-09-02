from dataclasses_json import config
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import torch
from collections import defaultdict
from src.config import Experiment, Analysis, Default

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
def compute_pca_eigenvectors(X, k=None):
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    # Return eigenvectors of covariance matrix
    if k is not None:
        Vh = Vh[:k]
    return Vh  # shape: (k or D, D)

def cosine_sim_matrix(A, B):
    A_norm = A / A.norm(dim=1, keepdim=True)
    B_norm = B / B.norm(dim=1, keepdim=True)
    return A_norm @ B_norm.T  # shape: (len(A), len(B))

# =============================================================================
# MAIN
# =============================================================================

for rrr in Experiment.RADII:
    for ttt in range(len(Experiment.TEMPS)):
        TEMPERATURE = Experiment.TEMPS[ttt]
        RADIUS_INITIAL_CONDITIONS = rrr
        RESULTS_DIR = f"{Default.RESULTS_DIR}/run_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/"
        PLOTS_DIR = f"{Default.RESULTS_DIR}_plots/run_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/"
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # =============================================================================
        # LOAD DATA
        # =============================================================================


        # the hidden states corresponding to different initial conditions at a given state form a sort of blob.
        # we look at the hypervolume and the largest axes.
        # you can think of it as a 3d ellipsoid and we look at how it deforms, but in higher dimensions.
        hypervolumes = torch.load(os.path.join(RESULTS_DIR, "hypervolume.pt"), weights_only=True)
        axis_lengths = torch.load(os.path.join(RESULTS_DIR, "axis_lengths.pt"), weights_only=True)
        # k is a parameter that I pick when I run all the cases. Normally 3 or 4

        ### These below are all lists and every element of the list corresponds to a different initial condtion

        # List of [seq_len, hidden_dim] tensors for each generation. Each tensor contains the hidden state for every generated token at the last layer.
        trajectories = torch.load(os.path.join(RESULTS_DIR, "trajectories.pt"), weights_only=True)
        # minimum_trajectory_len = min( [len(t) for t in trajectories] )

        # =============================================================================
        # PLOT and SAVE
        # =============================================================================

        ### Plot: Hypervolume and Axis Lengths
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(hypervolumes.cpu())
        plt.title("Hypervolumes")
        plt.xlabel("Time step")
        plt.ylabel("Volume")

        plt.subplot(1, 2, 2)
        axis_lengths_np = np.array(axis_lengths.cpu())
        if axis_lengths_np.ndim == 2:
            for i in range(axis_lengths_np.shape[1]):
                plt.plot(axis_lengths_np[:, i], label=f"Axis {i+1}")
            plt.title("Axis Lengths")
            plt.xlabel("Time step")
            plt.ylabel("Length")
            plt.legend()
        else:
            plt.title("Axis Lengths - Format Issue")

        if Analysis.SAVE_PLOTS:
            plt.savefig(os.path.join(PLOTS_DIR, "hypervolume_and_axes.png"))
            plt.clf()
        else:
            plt.show()


        # --- Local Dimensionality via SVD --- (Basically a PCA on a sliding window)
        # measuring the number of dimensions needed to account for Analysis.MINIMUM_VARIANCE_EXPLANATION for each trajectory
        local_dims = []
        for traj_idx, traj in enumerate(trajectories):
            dim_over_time = []
            for i in range(0, traj.size(0) - Analysis.SLIDING_WINDOW_SIZE + 1, Analysis.SLIDING_WINDOW_DISPLACEMENT):
                window = traj[i:i + Analysis.SLIDING_WINDOW_SIZE]  # shape: (W, D)
                window = window - window.mean(dim=0, keepdim=True)
                
                # SVD
                U, S, Vh = torch.linalg.svd(window, full_matrices=False)  # S: (min(W, D),)

                # Variance explained by each component
                var_explained = S**2 / (Analysis.SLIDING_WINDOW_SIZE - 1)
                cumulative_variance = torch.cumsum(var_explained, dim=0)
                total_variance = cumulative_variance[-1]
                cumulative_ratio = cumulative_variance / total_variance

                # Smallest number of components explaining desired variance
                num_components = torch.searchsorted(cumulative_ratio, Analysis.MINIMUM_VARIANCE_EXPLANATION) + 1
                dim_over_time.append(num_components.item())

            local_dims.append(dim_over_time)

        # Plotting local dimensionality
        plt.figure(figsize=(8, 5))

        # If you want to plot the dimension for each individual trajectory GG
        # for idx, dims in enumerate(local_dims):
        #     plt.plot(np.arange(len(dims)) * Analysis.SLIDING_WINDOW_DISPLACEMENT, dims, label=f"Trajectory {idx}")

        # If you want to average the dimension needed for different trajectories
        time_buckets = defaultdict(list)
        for traj in local_dims:
            for t, val in enumerate(traj):
                time_buckets[t].append(val)
        local_dims_avg = [np.mean(time_buckets[t]) for t in sorted(time_buckets)]

        plt.plot(np.arange(len(local_dims_avg)) * Analysis.SLIDING_WINDOW_DISPLACEMENT, local_dims_avg, label=f"Average for all trajectories")

        plt.title(f"Local Dimensionality (≥{Analysis.MINIMUM_VARIANCE_EXPLANATION * 100:.0f}% Variance)")
        plt.xlabel("Time step (center of window)")
        plt.ylabel("# of Components")
        plt.legend(fontsize="small", loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        if Analysis.SAVE_PLOTS:
            plt.savefig(os.path.join(PLOTS_DIR, "local_dimensionality.png"))
            plt.clf()
        else:
            plt.show()



        # --- Rank Eigenvectors of pairs of trajectories ---
        for i, j in Analysis.PAIRS_TO_PLOT:
            t1, t2 = trajectories[i], trajectories[j]

            # Clip to same length
            min_len = min(t1.shape[0], t2.shape[0])
            t1, t2 = t1[:min_len], t2[:min_len]

            # Compute PCA eigenvectors
            v1 = compute_pca_eigenvectors(t1)
            v2 = compute_pca_eigenvectors(t2)

            # Plot cross-rank
            # Compute cosine similarity between eigenvectors
            sim_matrix = cosine_sim_matrix(v1, v2)  # [D1, D2]
            closest_ranks = []

            for ii in range(sim_matrix.shape[0]):
                # Rank of trajectory j's vectors by similarity to i-th of traj1
                _, indices = torch.sort(sim_matrix[ii], descending=True)
                # Index of the most similar vector
                closest_idx = indices[0].item()
                closest_ranks.append(closest_idx + 1)  # 1-based indexing

            plt.figure(figsize=(6, 6))
            plt.scatter(range(1, len(closest_ranks) + 1), closest_ranks, marker='o')
            plt.plot([1, len(closest_ranks)], [1, len(closest_ranks)], 'k--', label="Perfect Match")
            plt.xlabel(f"Eigenvector Rank (Trajectory {i})")
            plt.ylabel(f"Rank of Closest Match (Trajectory {j})")
            plt.title(f"Trajectory {i} vs {j}\n Eigenvector ranking using Cosine similarity")
            plt.grid(True)
            plt.legend()

            if Analysis.SAVE_PLOTS:
                plt.savefig(os.path.join(PLOTS_DIR, f"rank_eigen_pca_{i}_{j}.png"))
                plt.clf()
            else:
                plt.show()
