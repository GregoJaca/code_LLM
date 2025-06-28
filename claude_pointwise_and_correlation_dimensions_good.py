import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import json
import torch

def calculate_correlation_dimension(distance_matrix: np.ndarray, 
                                   num_thresholds: int = 20,
                                   min_percent: float = 0.05,
                                   max_percent: float = 0.95,
                                   first_n_exclude: int = 3,
                                   last_n_exclude: int = 3):
    """
    Calculate correlation dimension from a Distance matrix.
    
    Args:
        distance_matrix: NxN matrix of pairwise cosine similarities between points
        num_thresholds: Number of threshold values to use
        min_percent: Minimum Distance threshold
        max_percent: Maximum Distance threshold
        first_n_exclude: Number of points to exclude from beginning when fitting
        last_n_exclude: Number of points to exclude from end when fitting
    
    Returns:
        Correlation dimension value
    """
    N = distance_matrix.shape[0]
    assert distance_matrix.shape == (N, N), "Distance matrix must be square"
    
    thresholds = np.logspace(np.log10(min_percent), np.log10(max_percent), num_thresholds)
    
    correlation_sums = []
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for eps in thresholds:
        recurrence_matrix = distance_matrix <= eps # 
        
        # Calculate correlation sum (sum of all recurrence points divided by N²)
        correlation_sum = (torch.sum(recurrence_matrix)-N) / (N * (N-1)) # -N is to not count the diagonal (so dont count a point with itself) 
        correlation_sums.append(correlation_sum)
    
    # Plot correlation sum vs threshold
    axes[0].plot(thresholds, correlation_sums, 'go-', linewidth=2)
    axes[0].set_xlabel('Distance Threshold')
    axes[0].set_ylabel('Correlation Sum')
    axes[0].set_title('Correlation Dimension Analysis')
    
    # Add padding to y-axis
    y_min_corr = min(correlation_sums) * 0.9
    y_max_corr = max(correlation_sums) * 1.1
    axes[0].set_ylim(y_min_corr, y_max_corr)
    
    # Plot correlation sum vs threshold (log-log)
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle log(0) cases
        log_thresholds = np.log(thresholds)
        log_correlation_sums = np.log(correlation_sums)
    
    # Remove any inf or nan values for the plot
    valid_indices = np.isfinite(log_thresholds) & np.isfinite(log_correlation_sums)
    plot_thresholds = thresholds[valid_indices]
    plot_log_corr_sums = log_correlation_sums[valid_indices]
    
    axes[1].loglog(plot_thresholds, np.exp(plot_log_corr_sums), 'go-', linewidth=2)
    axes[1].set_xlabel('log(Distance Threshold)')
    axes[1].set_ylabel('log(Correlation Sum)')
    axes[1].set_title('Correlation Dimension (log-log scale)')
    
    # Add padding to y-axis for log-log plot
    log_y_min_corr = min(np.exp(plot_log_corr_sums)) * 0.9
    log_y_max_corr = max(np.exp(plot_log_corr_sums)) * 1.1
    axes[1].set_ylim(log_y_min_corr, log_y_max_corr)
    
    # Calculate correlation dimension from slope of log-log plot
    # Select only the middle range for fitting (exclude specified points)
    fit_indices = slice(first_n_exclude, len(thresholds) - last_n_exclude)
    fit_log_thresholds = log_thresholds[fit_indices]
    fit_log_correlation_sums = log_correlation_sums[fit_indices]
    
    # Filter out any inf or nan values for the regression
    valid_fit_indices = np.isfinite(fit_log_thresholds) & np.isfinite(fit_log_correlation_sums)
    if np.sum(valid_fit_indices) < 2:
        print("Warning: Not enough valid points for linear regression!")
        correlation_slope = float('nan')
        return correlation_slope
    else:
        fit_log_thresholds_clean = fit_log_thresholds[valid_fit_indices]
        fit_log_correlation_sums_clean = fit_log_correlation_sums[valid_fit_indices]
        
        correlation_slope, correlation_intercept, r_value, p_value, std_err = linregress(
            fit_log_thresholds_clean, fit_log_correlation_sums_clean)
    
    with open('thresholds.npy', 'wb') as f:
        np.save(f, thresholds)
    with open('correlation_sums.npy', 'wb') as f:
        np.save(f, correlation_sums)


    results = {
        "correlation_dimension": correlation_slope,
        "r_value": r_value,
        "p_value": p_value,
        "std_err": std_err,
        "intercept": correlation_intercept
    }
    with open('dimension_results_L2.json', 'w') as f:
        json.dump(results, f, indent=4)

        
    # Add regression line to correlation log-log plot
    corr_reg_line = correlation_slope * log_thresholds + correlation_intercept
    axes[1].plot(thresholds, np.exp(corr_reg_line), 'b--', 
                label=f'Dimension = {correlation_slope:.3f}')
    
    # Highlight the points used for fitting
    valid_fit_plot_indices = fit_indices.start <= np.arange(len(thresholds)) 
    valid_fit_plot_indices &= np.arange(len(thresholds)) < fit_indices.stop
    valid_fit_plot_indices &= valid_indices
    
    if np.any(valid_fit_plot_indices):
        axes[1].plot(thresholds[valid_fit_plot_indices], 
                    np.exp(log_correlation_sums[valid_fit_plot_indices]), 
                    'bo', markersize=8, alpha=0.6, label='Points used for fitting')
    
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("embedding_matrix_correltion_dimension_L2")
    
    return correlation_slope


if __name__ == "__main__":

    distance_matrix = torch.load("./input_embeddings_l2.pt")

    
    correlation_dim = calculate_correlation_dimension(
        distance_matrix,
        num_thresholds=30,
        min_percent=1,
        max_percent=2,
        first_n_exclude=5,
        last_n_exclude=2
    )
    