import numpy as np
import pdb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, and Pearson correlation between true and predicted values.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: {'rmse': float, 'mae': float, 'pearson_r': float, 'pearson_p': float}
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'rmse': rmse,
        'mae': mae,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'r2': r2
    }

def plot_parity(y_true, y_pred, scatter_size=20, alpha=0.6):
    """
    Creates a square parity plot (predicted vs true) with:
      - identity line y = x
      - 95% CI band
      - reduced scatter size
      - clean legend
    """
    # Compute RMSE for the 95% CI band
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ci = 1.96 * rmse

    # Create a square figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter true vs. predicted
    ax.scatter(y_true, y_pred, s=scatter_size, alpha=alpha, edgecolor='none', label='Model Predictions')

    # Identity line
    vmin = min(np.min(y_true), np.min(y_pred))
    vmax = max(np.max(y_true), np.max(y_pred))
    line = np.linspace(vmin, vmax, 100)
    ax.plot(line, line, 'k--', lw=2, label='Perfect Prediction')

    # 95% CI band around the identity line
    ax.fill_between(line, line - ci, line + ci,
                    color='gray', alpha=0.2, label='95% CI')

    # Labels and styling
    ax.set_xlabel('$\\Delta G_{ox}$ (eV) - Actual Values', fontsize=15)
    ax.set_ylabel('$\\Delta G_{ox}$ (eV) - Predicted Values',fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Legend
    ax.legend(fontsize=13)

    plt.tight_layout()
    return ax

def evaluate_and_plot(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      plot_path: str = None,
                      metrics_path: str = None) -> dict:
    """
    Compute metrics, generate parity plot, and optionally save outputs.

    Args:
        y_true (np.ndarray): True targets.
        y_pred (np.ndarray): Predicted targets.
        plot_path (str, optional): File path to save the parity plot (e.g., .png).
        metrics_path (str, optional): File path to save the metrics JSON.

    Returns:
        dict: Computed metrics.
    """
    # Compute metrics
    metrics = compute_regression_metrics(y_true, y_pred)

    # Plot
    ax = plot_parity(y_true, y_pred)
    if plot_path is not None:
        plt.savefig(plot_path)
        plt.close(ax.figure)
    else:
        plt.show()

    # Convert any numpy/torch types to plain Python floats/ints
    metrics_serializable = {}
    for k, v in metrics.items():
        # If it's a numpy or torch scalar, cast to Python
        try:
            metrics_serializable[k] = float(v)
        except (TypeError, ValueError):
            # e.g. if it's already a Python bool or int, or something else
            metrics_serializable[k] = v

    # Save metrics
    if metrics_path is not None:
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

    return metrics
