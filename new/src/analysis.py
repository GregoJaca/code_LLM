import torch
from scipy import stats
import torch.nn.functional as F

def calculate_hypervolume_and_axes(trajectories, n_axes=1):
    N, seq_len, D = trajectories.shape
    device = trajectories.device

    X_centered = trajectories - trajectories.mean(dim=0, keepdim=True)
    X_centered = X_centered.permute(1, 0, 2)
    G = torch.bmm(X_centered, X_centered.transpose(1, 2))

    hypervolumes = torch.zeros(seq_len, device=device)
    axes_lengths = torch.zeros(seq_len, n_axes, device=device)
    denom = torch.sqrt(torch.tensor(max(N - 1, 1), dtype=torch.float32, device=device))

    for t in range(seq_len):
        eigvals = torch.linalg.eigvalsh(G[t])
        svals = torch.flip(eigvals, dims=[0]).sqrt()
        lengths = svals / denom if N > 1 else torch.zeros_like(svals)
        valid_lengths = lengths[:n_axes]
        axes_lengths[t, :len(valid_lengths)] = valid_lengths
        hypervolumes[t] = torch.prod(valid_lengths[valid_lengths > 0])
    
    return hypervolumes, axes_lengths

