import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from scipy import stats

def analyze_zero_distributions(recurrence_matrix):
    """
    Analyze the distribution of 0s between 1s in horizontal and diagonal directions.
    
    Args:
        recurrence_matrix: NxN binary recurrence matrix with 0s and 1s
        
    Returns:
        Dictionary containing horizontal and diagonal zero sequences
    """
    N = recurrence_matrix.shape[0]
    
    # Horizontal zero sequences
    horizontal_zeros = defaultdict(list)
    
    # For each row
    for i in range(N):
        row = recurrence_matrix[i]
        zeros_lengths = []
        current_zeros = 0
        
        for j in range(N):
            if row[j] == 0:
                current_zeros += 1
            else:  # Found a 1
                if current_zeros > 0:
                    zeros_lengths.append(current_zeros)
                    current_zeros = 0
        
        # Handle the case where the row ends with zeros
        if current_zeros > 0:
            zeros_lengths.append(current_zeros)
        
        horizontal_zeros[i] = zeros_lengths
    
    # Diagonal zero sequences
    diagonal_zeros = defaultdict(list)
    
    # For each diagonal (main and above)
    for k in range(N):
        diag = np.diag(recurrence_matrix, k)
        zeros_lengths = []
        current_zeros = 0
        
        for val in diag:
            if val == 0:
                current_zeros += 1
            else:  # Found a 1
                if current_zeros > 0:
                    zeros_lengths.append(current_zeros)
                    current_zeros = 0
        
        # Handle the case where the diagonal ends with zeros
        if current_zeros > 0:
            zeros_lengths.append(current_zeros)
        
        diagonal_zeros[k] = zeros_lengths
    
    # For each diagonal below the main diagonal
    for k in range(1, N):
        diag = np.diag(recurrence_matrix, -k)
        zeros_lengths = []
        current_zeros = 0
        
        for val in diag:
            if val == 0:
                current_zeros += 1
            else:  # Found a 1
                if current_zeros > 0:
                    zeros_lengths.append(current_zeros)
                    current_zeros = 0
        
        # Handle the case where the diagonal ends with zeros
        if current_zeros > 0:
            zeros_lengths.append(current_zeros)
        
        diagonal_zeros[-k] = zeros_lengths
    
    return {
        'horizontal': horizontal_zeros,
        'diagonal': diagonal_zeros
    }

def plot_zero_distributions(zero_distributions, figsize=(15, 15)):
    """
    Plot the distributions of 0s between 1s in horizontal and diagonal directions.
    
    Args:
        zero_distributions: Dictionary containing horizontal and diagonal zero sequences
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure
    """
    horizontal_zeros = zero_distributions['horizontal']
    diagonal_zeros = zero_distributions['diagonal']
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # 1. Horizontal zero sequences - overall distribution
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    all_horizontal = [length for lengths in horizontal_zeros.values() for length in lengths]
    
    if all_horizontal:
        sns.histplot(all_horizontal, kde=False, ax=ax1)
        ax1.set_title('Distribution of Horizontal Zero Sequences')
        ax1.set_xlabel('Zero Sequence Length')
        ax1.set_ylabel('Frequency')
        
        # Add some statistics
        mean_length = np.mean(all_horizontal)
        median_length = np.median(all_horizontal)
        ax1.axvline(mean_length, color='r', linestyle='--', label=f'Mean: {mean_length:.2f}')
        ax1.axvline(median_length, color='g', linestyle='--', label=f'Median: {median_length:.2f}')
        ax1.legend()
        ax1.set_xlim([0, 5*max(mean_length, median_length)])

    else:
        ax1.text(0.5, 0.5, 'No horizontal zero sequences found', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 2. Diagonal zero sequences - overall distribution
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    all_diagonal = [length for lengths in diagonal_zeros.values() for length in lengths]
    
    if all_diagonal:
        sns.histplot(all_diagonal, kde=False, ax=ax2)
        ax2.set_title('Distribution of Diagonal Zero Sequences')
        ax2.set_xlabel('Zero Sequence Length')
        ax2.set_ylabel('Frequency')
        
        # Add some statistics
        mean_length = np.mean(all_diagonal)
        median_length = np.median(all_diagonal)
        ax2.axvline(mean_length, color='r', linestyle='--', label=f'Mean: {mean_length:.2f}')
        ax2.axvline(median_length, color='g', linestyle='--', label=f'Median: {median_length:.2f}')
        ax2.set_xlim([0, 5*max(mean_length, median_length)])
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No diagonal zero sequences found', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 3. Horizontal zero sequences - heatmap by row
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    
    # Prepare data for heatmap - average length per row
    row_indices = sorted(horizontal_zeros.keys())
    avg_lengths = [np.mean(horizontal_zeros[i]) if horizontal_zeros[i] else 0 for i in row_indices]
    
    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(row_indices), 1))
    for i, avg_length in enumerate(avg_lengths):
        heatmap_data[i, 0] = avg_length
    
    # Plot heatmap
    sns.heatmap(heatmap_data, cmap='viridis', ax=ax3)
    ax3.set_title('Average Horizontal Zero Sequence Length by Row')
    ax3.set_ylabel('Row Index')
    ax3.set_xlabel('Average Length')
    
    # 4. Diagonal zero sequences - average length by diagonal index
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    
    # Prepare data
    diag_indices = sorted(diagonal_zeros.keys())
    diag_avg_lengths = [np.mean(diagonal_zeros[i]) if diagonal_zeros[i] else 0 for i in diag_indices]
    
    # Plot
    ax4.bar(diag_indices, diag_avg_lengths)
    ax4.set_title('Average Diagonal Zero Sequence Length by Diagonal Index')
    ax4.set_xlabel('Diagonal Index')
    ax4.set_ylabel('Average Length')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_temporal_patterns(recurrence_matrix):
    """
    Analyze temporal patterns in the recurrence matrix.
    
    Args:
        recurrence_matrix: NxN binary recurrence matrix
        
    Returns:
        Dictionary of temporal analysis results
    """
    N = recurrence_matrix.shape[0]
    
    # Calculate recurrence rate over time
    recurrence_rate = np.sum(recurrence_matrix, axis=1) / N
    
    # Calculate determinism (ratio of points forming diagonal lines)
    min_line_length = 2
    diagonal_lengths = []
    
    # For each diagonal (main and above)
    for k in range(N):
        diag = np.diag(recurrence_matrix, k)
        current_length = 0
        
        for val in diag:
            if val == 1:
                current_length += 1
            else:  # Found a 0
                if current_length >= min_line_length:
                    diagonal_lengths.append(current_length)
                current_length = 0
        
        # Handle the case where the diagonal ends with ones
        if current_length >= min_line_length:
            diagonal_lengths.append(current_length)
    
    # For each diagonal below the main diagonal
    for k in range(1, N):
        diag = np.diag(recurrence_matrix, -k)
        current_length = 0
        
        for val in diag:
            if val == 1:
                current_length += 1
            else:  # Found a 0
                if current_length >= min_line_length:
                    diagonal_lengths.append(current_length)
                current_length = 0
        
        # Handle the case where the diagonal ends with ones
        if current_length >= min_line_length:
            diagonal_lengths.append(current_length)
    
    # Calculate determinism if diagonal lines exist
    total_ones = np.sum(recurrence_matrix)
    diag_ones = sum(diagonal_lengths)
    determinism = diag_ones / total_ones if total_ones > 0 else 0
    
    # Calculate average diagonal line length
    avg_diag_length = np.mean(diagonal_lengths) if diagonal_lengths else 0
    
    # Calculate laminarity (ratio of points forming vertical lines)
    vertical_lengths = []
    
    # For each column
    for j in range(N):
        col = recurrence_matrix[:, j]
        current_length = 0
        
        for val in col:
            if val == 1:
                current_length += 1
            else:  # Found a 0
                if current_length >= min_line_length:
                    vertical_lengths.append(current_length)
                current_length = 0
        
        # Handle the case where the column ends with ones
        if current_length >= min_line_length:
            vertical_lengths.append(current_length)
    
    # Calculate laminarity if vertical lines exist
    vert_ones = sum(vertical_lengths)
    laminarity = vert_ones / total_ones if total_ones > 0 else 0
    
    # Calculate average vertical line length
    avg_vert_length = np.mean(vertical_lengths) if vertical_lengths else 0
    
    # Calculate entropy of diagonal line lengths
    if diagonal_lengths:
        diag_lengths_counter = np.bincount(diagonal_lengths)[min_line_length:]
        diag_lengths_prob = diag_lengths_counter / sum(diag_lengths_counter)
        entropy = -np.sum(diag_lengths_prob * np.log(diag_lengths_prob + 1e-10))
    else:
        entropy = 0
    
    return {
        'recurrence_rate': recurrence_rate,
        'determinism': determinism,
        'avg_diag_length': avg_diag_length,
        'laminarity': laminarity,
        'avg_vert_length': avg_vert_length,
        'entropy': entropy,
        'diagonal_lengths': diagonal_lengths,
        'vertical_lengths': vertical_lengths
    }

def plot_temporal_patterns(temporal_results, figsize=(15, 15)):
    """
    Plot temporal patterns analysis results.
    
    Args:
        temporal_results: Results from analyze_temporal_patterns
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # 1. Recurrence rate over time
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.plot(temporal_results['recurrence_rate'], 'b-')
    ax1.set_title('Recurrence Rate over Time')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Recurrence Rate')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_rate = np.mean(temporal_results['recurrence_rate'])
    ax1.axhline(mean_rate, color='r', linestyle='--', 
                label=f'Mean: {mean_rate:.3f}')
    ax1.legend()
    
    # 2. Diagonal line length distribution
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    if temporal_results['diagonal_lengths']:
        sns.histplot(temporal_results['diagonal_lengths'], kde=False, ax=ax2)
        ax2.set_title('Diagonal Line Length Distribution')
        ax2.set_xlabel('Line Length')
        ax2.set_ylabel('Frequency')
        
        # Add determinism value
        ax2.text(0.05, 0.95, f"Determinism: {temporal_results['determinism']:.3f}", 
                 transform=ax2.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Add average line length
        ax2.axvline(temporal_results['avg_diag_length'], color='g', linestyle='--', 
                    label=f'Avg: {temporal_results["avg_diag_length"]:.2f}')
        ax2.set_xlim([0, 10*temporal_results['avg_diag_length']])
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No diagonal lines found', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 3. Vertical line length distribution
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    if temporal_results['vertical_lengths']:
        sns.histplot(temporal_results['vertical_lengths'], kde=False, ax=ax3)
        ax3.set_title('Vertical Line Length Distribution')
        ax3.set_xlabel('Line Length')
        ax3.set_ylabel('Frequency')
        
        # Add laminarity value
        ax3.text(0.05, 0.95, f"Laminarity: {temporal_results['laminarity']:.3f}", 
                 transform=ax3.transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Add average line length
        ax3.axvline(temporal_results['avg_vert_length'], color='g', linestyle='--', 
                    label=f'Avg: {temporal_results["avg_vert_length"]:.2f}')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No vertical lines found', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 4. Summary metrics
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    metrics = [
        ('Determinism', temporal_results['determinism']),
        ('Laminarity', temporal_results['laminarity']),
        ('Avg. Diag. Length', temporal_results['avg_diag_length']),
        ('Avg. Vert. Length', temporal_results['avg_vert_length']),
        ('Entropy', temporal_results['entropy'])
    ]
    
    # Plot metrics as a bar chart
    ax4.bar(range(len(metrics)), [m[1] for m in metrics])
    ax4.set_xticks(range(len(metrics)))
    ax4.set_xticklabels([m[0] for m in metrics], rotation=45)
    ax4.set_title('Recurrence Quantification Analysis Metrics')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_zero_patterns_log_log(recurrence_matrix):
    """
    Analyze zero patterns using log-log scaling.
    
    Args:
        recurrence_matrix: NxN binary recurrence matrix
        
    Returns:
        Dictionary with log-log analysis results
    """
    zero_distributions = analyze_zero_distributions(recurrence_matrix)
    
    # Horizontal zero sequences - all lengths
    all_horizontal = [length for lengths in zero_distributions['horizontal'].values() 
                     for length in lengths]
    
    # Diagonal zero sequences - all lengths
    all_diagonal = [length for lengths in zero_distributions['diagonal'].values() 
                   for length in lengths]
    
    # Create log-log distributions
    horizontal_log_dist = {}
    if all_horizontal:
        # Count occurrences of each length
        lengths_count = np.bincount(all_horizontal)
        
        # Filter out zero counts and convert to log-log
        lengths = np.arange(len(lengths_count))
        mask = lengths_count > 0
        
        log_lengths = np.log10(lengths[mask])
        log_counts = np.log10(lengths_count[mask])
        
        horizontal_log_dist = {
            'log_lengths': log_lengths,
            'log_counts': log_counts,
            'lengths': lengths[mask],
            'counts': lengths_count[mask]
        }
        
        # Fit power law
        if len(log_lengths) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_lengths, log_counts)
            
            horizontal_log_dist['fit'] = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err
            }
    
    # Similar for diagonal
    diagonal_log_dist = {}
    if all_diagonal:
        # Count occurrences of each length
        lengths_count = np.bincount(all_diagonal)
        
        # Filter out zero counts and convert to log-log
        lengths = np.arange(len(lengths_count))
        mask = lengths_count > 0
        
        log_lengths = np.log10(lengths[mask])
        log_counts = np.log10(lengths_count[mask])
        
        diagonal_log_dist = {
            'log_lengths': log_lengths,
            'log_counts': log_counts,
            'lengths': lengths[mask],
            'counts': lengths_count[mask]
        }
        
        # Fit power law
        if len(log_lengths) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_lengths, log_counts)
            
            diagonal_log_dist['fit'] = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err
            }
    
    return {
        'horizontal': horizontal_log_dist,
        'diagonal': diagonal_log_dist
    }

def plot_zero_patterns_log_log(log_log_results, figsize=(12, 6)):
    """
    Plot zero patterns in log-log scale.
    
    Args:
        log_log_results: Results from analyze_zero_patterns_log_log
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot horizontal zero sequences
    horizontal = log_log_results['horizontal']
    if horizontal and 'log_lengths' in horizontal:
        ax1.scatter(horizontal['log_lengths'], horizontal['log_counts'], 
                   c='blue', alpha=0.7, label='Data')
        
        # Plot fit line if available
        if 'fit' in horizontal:
            fit = horizontal['fit']
            x_range = np.linspace(min(horizontal['log_lengths']), 
                                 max(horizontal['log_lengths']), 100)
            y_fit = fit['slope'] * x_range + fit['intercept']
            ax1.plot(x_range, y_fit, 'r-', 
                    label=f'Fit: y = {fit["slope"]:.2f}x + {fit["intercept"]:.2f}')
            
            # Add fit statistics
            ax1.text(0.05, 0.95, f"R² = {fit['r_value']**2:.3f}\np = {fit['p_value']:.3e}", 
                    transform=ax1.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        ax1.set_title('Horizontal Zero Sequences (Log-Log)')
        ax1.set_xlabel('Log₁₀(Length)')
        ax1.set_ylabel('Log₁₀(Frequency)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Insufficient data for log-log analysis', 
                horizontalalignment='center', verticalalignment='center')
    
    # Plot diagonal zero sequences
    diagonal = log_log_results['diagonal']
    if diagonal and 'log_lengths' in diagonal:
        ax2.scatter(diagonal['log_lengths'], diagonal['log_counts'], 
                   c='green', alpha=0.7, label='Data')
        
        # Plot fit line if available
        if 'fit' in diagonal:
            fit = diagonal['fit']
            x_range = np.linspace(min(diagonal['log_lengths']), 
                                 max(diagonal['log_lengths']), 100)
            y_fit = fit['slope'] * x_range + fit['intercept']
            ax2.plot(x_range, y_fit, 'r-', 
                    label=f'Fit: y = {fit["slope"]:.2f}x + {fit["intercept"]:.2f}')
            
            # Add fit statistics
            ax2.text(0.05, 0.95, f"R² = {fit['r_value']**2:.3f}\np = {fit['p_value']:.3e}", 
                    transform=ax2.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        ax2.set_title('Diagonal Zero Sequences (Log-Log)')
        ax2.set_xlabel('Log₁₀(Length)')
        ax2.set_ylabel('Log₁₀(Frequency)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for log-log analysis', 
                horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    return fig



def plot_zero_patterns_normal(log_log_results, figsize=(12, 6)):
    """
    Plot zero patterns in normal scale.
    
    Args:
        log_log_results: Results from analyze_zero_patterns_log_log
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot horizontal zero sequences
    horizontal = log_log_results['horizontal']
    if horizontal and 'lengths' in horizontal:
        ax1.scatter((horizontal['lengths']), horizontal['counts'], 
                   c='blue', alpha=0.7, label='Data')
        
        ax1.set_title('Horizontal Zero Sequences')
        ax1.set_xlabel('(Length)')
        ax1.set_ylabel('(Frequency)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Insufficient data for analysis', 
                horizontalalignment='center', verticalalignment='center')
    
    # Plot diagonal zero sequences
    diagonal = log_log_results['diagonal']
    if diagonal and 'lengths' in diagonal:
        ax2.scatter(diagonal['lengths'], diagonal['counts'], 
                   c='green', alpha=0.7, label='Data')
        
        ax2.set_title('Diagonal Zero Sequences')
        ax2.set_xlabel('(Length)')
        ax2.set_ylabel('(Frequency)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for analysis', 
                horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    return fig

def main(recurrence_matrix):
    """
    Main function to analyze distributions in recurrence matrix.
    
    Args:
        recurrence_matrix: NxN binary recurrence matrix
    """
    print("Analyzing zero distributions...")
    zero_distributions = analyze_zero_distributions(recurrence_matrix)
    
    print("Plotting zero distributions...")
    fig_zeros = plot_zero_distributions(zero_distributions)
    
    print("Analyzing temporal patterns...")
    temporal_results = analyze_temporal_patterns(recurrence_matrix)
    
    print("Plotting temporal patterns...")
    fig_temporal = plot_temporal_patterns(temporal_results)
    
    print("Analyzing zero patterns in log-log scale...")
    log_log_results = analyze_zero_patterns_log_log(recurrence_matrix)
    
    print("Plotting log-log patterns...")
    fig_log_log = plot_zero_patterns_log_log(log_log_results)

    print("Plotting normal patterns...")
    fig_log_log = plot_zero_patterns_normal(log_log_results)
    
    # Print summary statistics
    print("\n=== Zero Distribution Analysis ===")
    horizontal_zeros = [length for lengths in zero_distributions['horizontal'].values() 
                       for length in lengths]
    diagonal_zeros = [length for lengths in zero_distributions['diagonal'].values() 
                     for length in lengths]
    
    if horizontal_zeros:
        print("\nHorizontal zero sequences:")
        print(f"  Count: {len(horizontal_zeros)}")
        print(f"  Mean length: {np.mean(horizontal_zeros):.2f}")
        print(f"  Median length: {np.median(horizontal_zeros):.2f}")
        print(f"  Max length: {max(horizontal_zeros)}")
        print(f"  Min length: {min(horizontal_zeros)}")
    else:
        print("\nNo horizontal zero sequences found.")
    
    if diagonal_zeros:
        print("\nDiagonal zero sequences:")
        print(f"  Count: {len(diagonal_zeros)}")
        print(f"  Mean length: {np.mean(diagonal_zeros):.2f}")
        print(f"  Median length: {np.median(diagonal_zeros):.2f}")
        print(f"  Max length: {max(diagonal_zeros)}")
        print(f"  Min length: {min(diagonal_zeros)}")
    else:
        print("\nNo diagonal zero sequences found.")
    
    print("\n=== Temporal Pattern Analysis ===")
    print(f"Recurrence Rate: {np.mean(temporal_results['recurrence_rate']):.3f}")
    print(f"Determinism: {temporal_results['determinism']:.3f}")
    print(f"Laminarity: {temporal_results['laminarity']:.3f}")
    print(f"Avg. Diagonal Line Length: {temporal_results['avg_diag_length']:.2f}")
    print(f"Avg. Vertical Line Length: {temporal_results['avg_vert_length']:.2f}")
    print(f"Entropy: {temporal_results['entropy']:.3f}")
    
    print("\n=== Log-Log Analysis ===")
    if 'fit' in log_log_results['horizontal']:
        fit = log_log_results['horizontal']['fit']
        print("\nHorizontal zero sequences power law:")
        print(f"  Slope: {fit['slope']:.3f}")
        print(f"  R²: {fit['r_value']**2:.3f}")
        print(f"  p-value: {fit['p_value']:.3e}")
    else:
        print("\nInsufficient data for horizontal log-log analysis")
    
    if 'fit' in log_log_results['diagonal']:
        fit = log_log_results['diagonal']['fit']
        print("\nDiagonal zero sequences power law:")
        print(f"  Slope: {fit['slope']:.3f}")
        print(f"  R²: {fit['r_value']**2:.3f}")
        print(f"  p-value: {fit['p_value']:.3e}")
    else:
        print("\nInsufficient data for diagonal log-log analysis")
    
    # Show all figures
    plt.show()
    
    return {
        'zero_distributions': zero_distributions,
        'temporal_results': temporal_results,
        'log_log_results': log_log_results
    }

# Example usage
if __name__ == "__main__":

    recurrence_matrix = 1-cosine_sim_last_last.numpy()
    threshold = 0.3
    recurrence_matrix = recurrence_matrix < threshold

    
    # Run the analysis
    results = main(recurrence_matrix)