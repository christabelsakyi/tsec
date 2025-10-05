"""
Module for evaluating time series forecasting models.
Includes various metrics for comparing predicted vs actual values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate common evaluation metrics for time series forecasting.

    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values

    Returns:
    --------
    dict
        Dictionary containing various performance metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error (MAPE)
    # Handle potential division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    # More robust to outliers and zero values
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0  # Avoid division by zero
    smape = np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100

    # Theil's U statistic (U2)
    # Compares the forecast with a naive forecast
    # U2 < 1: better than naive, U2 = 1: same as naive, U2 > 1: worse than naive
    # For time series, naive forecast is often the previous value
    y_naive = np.roll(y_true, 1)
    y_naive[0] = y_true[0]  # Set first value

    numerator = np.sqrt(np.mean(np.square(y_true - y_pred)))
    denominator = np.sqrt(np.mean(np.square(y_true - y_naive)))

    if denominator != 0:
        theils_u = numerator / denominator
    else:
        theils_u = np.nan

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'SMAPE': smape,
        'Theils_U': theils_u
    }


def plot_forecast_vs_actual(y_true, y_pred, title='Forecast vs Actual', figsize=(12, 6)):
    """
    Plot the forecasted values against the actual values.

    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual values
    ax.plot(y_true, label='Actual', color='blue', marker='o', markersize=4, linestyle='-', alpha=0.7)

    # Plot predicted values
    ax.plot(y_pred, label='Forecast', color='red', marker='x', markersize=4, linestyle='--', alpha=0.7)

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)

    # Compute metrics and add as text annotation
    metrics = calculate_metrics(y_true, y_pred)

    metrics_text = (
        f"MAE: {metrics['MAE']:.2f}\n"
        f"RMSE: {metrics['RMSE']:.2f}\n"
        f"MAPE: {metrics['MAPE']:.2f}%\n"
        f"R²: {metrics['R2']:.3f}"
    )

    # Position text in the upper left
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.6))

    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, figsize=(12, 10)):
    """
    Create diagnostic plots for forecast residuals.

    Parameters:
    -----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    figsize : tuple, optional
        Figure size (width, height)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Residuals over time
    axes[0, 0].plot(residuals, color='blue', marker='o', markersize=3, linestyle='None', alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=20, color='blue', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--')
    axes[0, 1].set_title('Histogram of Residuals')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # QQ plot of residuals
    from scipy import stats
    stats.probplot(residuals, plot=axes[1, 0])
    axes[1, 0].set_title('QQ Plot of Residuals')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # Predicted vs Residuals
    axes[1, 1].scatter(y_pred, residuals, color='blue', alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_title('Predicted vs Residuals')
    axes[1, 1].set_xlabel('Predicted Value')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    # Calculate autocorrelation of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    fig_acf = plt.figure(figsize=(8, 4))
    plot_acf(residuals, lags=30, title='Autocorrelation of Residuals', alpha=0.05)

    plt.tight_layout()
    return fig, fig_acf


def cross_validation_time_series(model, data, target_col, window_size, horizon, step=1):
    """
    Perform time series cross-validation using a rolling window approach.

    Parameters:
    -----------
    model : object
        Model object with fit and predict methods
    data : pandas.DataFrame
        Time series data
    target_col : str
        Name of the target column
    window_size : int
        Size of the training window
    horizon : int
        Number of steps to forecast
    step : int, optional
        Step size between windows

    Returns:
    --------
    dict
        Dictionary containing actual values, predictions, and metrics for each fold
    """
    results = {
        'actual': [],
        'predicted': [],
        'metrics': [],
        'train_indices': [],
        'test_indices': []
    }

    # Ensure data is sorted by time
    data = data.sort_index()

    # Get total size
    total_size = len(data)

    # Need at least window_size + horizon data points
    if total_size < window_size + horizon:
        raise ValueError(f"Not enough data points. Need at least {window_size + horizon}")

    # Calculate number of folds
    n_folds = (total_size - window_size - horizon) // step + 1

    for fold in range(n_folds):
        # Calculate indices
        train_start = fold * step
        train_end = train_start + window_size
        test_start = train_end
        test_end = test_start + horizon

        # Ensure we don't go out of bounds
        if test_end > total_size:
            break

        # Get train and test data
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]

        # Store indices
        results['train_indices'].append((train_start, train_end))
        results['test_indices'].append((test_start, test_end))

        # Fit model
        model.fit(train_data)

        # Predict
        y_pred = model.predict(horizon)

        # Get actual values
        y_true = test_data[target_col].values

        # Store results
        results['actual'].append(y_true)
        results['predicted'].append(y_pred)

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        results['metrics'].append(metrics)

    # Aggregate metrics across folds
    aggregated_metrics = {}
    for metric in results['metrics'][0].keys():
        values = [fold_metrics[metric] for fold_metrics in results['metrics']]
        aggregated_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    results['aggregated_metrics'] = aggregated_metrics

    return results