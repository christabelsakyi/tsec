"""
Visualization utilities for time series forecasting.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import binned_statistic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Optional imports for interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set default styling
# Use compatible seaborn style for older and newer versions
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
sns.set_palette("deep")


def plot_time_series(data, column, title=None, figsize=(12, 6), date_format='%Y-%m-%d'):
    """
    Plot a time series variable.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the time series data.
    column : str
        The column name to plot.
    title : str, optional
        Plot title. If None, uses column name.
    figsize : tuple, optional
        Figure size (width, height).
    date_format : str, optional
        Format for the x-axis date labels.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the time series
    ax.plot(data.index, data[column], marker='.', markersize=2, linestyle='-', linewidth=1)

    # Set title and labels
    ax.set_title(title or f'Time Series: {column}')
    ax.set_xlabel('Date')
    ax.set_ylabel(column)

    # Format the date on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.xticks(rotation=45)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add a horizontal line for the mean
    mean_val = data[column].mean()
    ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7)
    ax.text(
        0.02, 0.95, f'Mean: {mean_val:.2f}',
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', alpha=0.1)
    )

    # Tight layout to avoid label cutoff
    plt.tight_layout()

    return fig


def plot_multiple_series(data, columns, title='Multiple Time Series', figsize=(12, 6), date_format='%Y-%m-%d'):
    """
    Plot multiple time series variables on the same graph.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the time series data.
    columns : list
        List of column names to plot.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size (width, height).
    date_format : str, optional
        Format for the x-axis date labels.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each time series
    for column in columns:
        ax.plot(data.index, data[column], marker='.', markersize=2, linewidth=1, label=column)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

    # Format the date on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.xticks(rotation=45)

    # Add legend and grid
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Tight layout to avoid label cutoff
    plt.tight_layout()

    return fig


def plot_seasonal_decomposition(decomposition, figsize=(12, 10)):
    """
    Plot the results of a seasonal decomposition.

    Parameters:
    -----------
    decomposition : statsmodels.tsa.seasonal.DecomposeResult
        Result from seasonal_decompose.
    figsize : tuple, optional
        Figure size (width, height).

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    return fig


def plot_acf_pacf(data, column, lags=30, figsize=(12, 6)):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the time series data.
    column : str
        The column name to analyze.
    lags : int, optional
        Number of lags to include.
    figsize : tuple, optional
        Figure size (width, height).

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot ACF and PACF
    plot_acf(data[column].dropna(), lags=lags, ax=axes[0])
    plot_pacf(data[column].dropna(), lags=lags, ax=axes[1])

    # Set titles
    axes[0].set_title(f'Autocorrelation Function (ACF): {column}')
    axes[1].set_title(f'Partial Autocorrelation Function (PACF): {column}')

    # Add grid
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    return fig


def plot_forecast(historical_data, forecast_data, column, prediction_col='yhat',
                  lower_col='yhat_lower', upper_col='yhat_upper',
                  title='Forecast', figsize=(12, 6), date_format='%Y-%m-%d'):
    """
    Plot historical data and forecast.

    Parameters:
    -----------
    historical_data : pandas.DataFrame
        DataFrame containing the historical time series data.
    forecast_data : pandas.DataFrame
        DataFrame containing the forecast data.
    column : str
        The column name in the historical data.
    prediction_col : str, optional
        The column name for predictions in the forecast data.
    lower_col : str, optional
        The column name for lower bound in the forecast data.
    upper_col : str, optional
        The column name for upper bound in the forecast data.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size (width, height).
    date_format : str, optional
        Format for the x-axis date labels.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot historical data
    ax.plot(historical_data.index, historical_data[column],
            label='Historical Data', color='blue', marker='.', markersize=2)

    # Plot forecast
    ax.plot(forecast_data.index, forecast_data[prediction_col],
            label='Forecast', color='red', linestyle='--')

    # Plot confidence interval if available
    if lower_col in forecast_data.columns and upper_col in forecast_data.columns:
        ax.fill_between(
            forecast_data.index,
            forecast_data[lower_col],
            forecast_data[upper_col],
            color='red',
            alpha=0.2,
            label='Confidence Interval'
        )

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

    # Format the date on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.xticks(rotation=45)

    # Add legend and grid
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Tight layout to avoid label cutoff
    plt.tight_layout()

    return fig


def plot_residuals_vs_fitted(y_true, y_pred, figsize=(10, 6)):
    """
    Plot residuals versus fitted values.

    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    figsize : tuple, optional
        Figure size (width, height).

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot residuals vs fitted
    ax.scatter(y_pred, residuals, color='blue', alpha=0.6)
    ax.axhline(y=0, color='red', linestyle='--')

    # Add a smoothed line
    bin_means, bin_edges, binnumber = binned_statistic(
        y_pred, residuals, statistic='mean', bins=20
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.plot(bin_centers, bin_means, color='red', linewidth=2)

    # Set title and labels
    ax.set_title('Residuals vs Fitted Values')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Tight layout to avoid label cutoff
    plt.tight_layout()

    return fig


def plot_hourly_patterns(data, column, title='Hourly Patterns', figsize=(12, 6)):
    """
    Plot average values by hour of day.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with datetime index.
    column : str
        The column name to analyze.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size (width, height).

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    hourly_avg = data.groupby(data.index.hour)[column].mean()
    hourly_std = data.groupby(data.index.hour)[column].std()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot average by hour
    ax.plot(hourly_avg.index, hourly_avg.values, marker='o', markersize=6, linewidth=2)

    # Add error bars for standard deviation
    ax.fill_between(
        hourly_avg.index,
        hourly_avg.values - hourly_std.values,
        hourly_avg.values + hourly_std.values,
        alpha=0.2
    )

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(f'Average {column}')

    # Set x-axis ticks to show every hour
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(range(0, 24))

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Tight layout to avoid label cutoff
    plt.tight_layout()

    return fig


def create_monthly_heatmap(data, column, title='Monthly Pattern Heatmap'):
    """
    Create a heatmap showing patterns by day of month and hour of day.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with datetime index.
    column : str
        The column name to analyze.
    title : str, optional
        Plot title.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    # Pivot data to create a matrix of day of month (rows) by hour (columns)
    pivot_data = data.pivot_table(
        index=data.index.day,
        columns=data.index.hour,
        values=column,
        aggfunc='mean'
    )

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create heatmap
    sns.heatmap(
        pivot_data,
        cmap='viridis',
        annot=False,
        fmt='.1f',
        cbar_kws={'label': f'Average {column}'},
        ax=ax
    )

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Month')

    # Adjust labels for better visibility
    plt.tight_layout()

    return fig


def plot_daily_and_weekly_patterns(data, column, figsize=(14, 10)):
    """
    Create plots showing daily and weekly patterns.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with datetime index.
    column : str
        The column name to analyze.
    figsize : tuple, optional
        Figure size (width, height).

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Daily pattern (by hour)
    hourly_avg = data.groupby(data.index.hour)[column].mean()
    axes[0].plot(hourly_avg.index, hourly_avg.values, marker='o', markersize=6, linewidth=2)
    axes[0].set_title(f'Average {column} by Hour of Day')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel(f'Average {column}')
    axes[0].set_xticks(range(0, 24))
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Weekly pattern (by day of week)
    # 0 = Monday, 6 = Sunday
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg = data.groupby(data.index.dayofweek)[column].mean()
    axes[1].bar(weekly_avg.index, weekly_avg.values)
    axes[1].set_title(f'Average {column} by Day of Week')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel(f'Average {column}')
    axes[1].set_xticks(range(0, 7))
    axes[1].set_xticklabels(day_names)
    axes[1].grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()

    return fig


def plot_model_evaluation(model_results, metric='RMSE', figsize=(10, 6)):
    """
    Plot evaluation results for multiple models.

    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and metric dictionaries as values.
    metric : str, optional
        Metric to plot.
    figsize : tuple, optional
        Figure size (width, height).

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    models = list(model_results.keys())
    values = [result[metric] for result in model_results.values()]

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    bars = ax.bar(models, values, color=sns.color_palette("deep", len(models)))

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.02 * max(values),
            f'{height:.3f}',
            ha='center', va='bottom',
            fontweight='bold'
        )

    # Set title and labels
    ax.set_title(f'Model Comparison by {metric}')
    ax.set_xlabel('Model')
    ax.set_ylabel(metric)

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    # Adjust layout
    plt.tight_layout()

    return fig


# Interactive plotting functions - only available if plotly is installed
def _check_plotly():
    """Check if plotly is available and raise an error if not."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Please install it with 'pip install plotly'."
        )


def create_interactive_forecast_plot(historical_data, forecast_data, column, prediction_col='yhat',
                                     lower_col='yhat_lower', upper_col='yhat_upper', title='Forecast'):
    """
    Create an interactive plotly plot for forecast visualization.

    Parameters:
    -----------
    historical_data : pandas.DataFrame
        DataFrame containing the historical time series data.
    forecast_data : pandas.DataFrame
        DataFrame containing the forecast data.
    column : str
        The column name in the historical data.
    prediction_col : str, optional
        The column name for predictions in the forecast data.
    lower_col : str, optional
        The column name for lower bound in the forecast data.
    upper_col : str, optional
        The column name for upper bound in the forecast data.
    title : str, optional
        Plot title.

    Returns:
    --------
    plotly.graph_objects.Figure
        The created interactive figure.
    """
    _check_plotly()

    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data[column],
            name='Historical',
            mode='lines',
            line=dict(color='blue')
        )
    )

    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data[prediction_col],
            name='Forecast',
            mode='lines',
            line=dict(color='red', dash='dash')
        )
    )

    # Add confidence interval if available
    if lower_col in forecast_data.columns and upper_col in forecast_data.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data[upper_col],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data[lower_col],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.2)',
                fill='tonexty',
                name='Confidence Interval'
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x'
    )

    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig


def create_interactive_dashboard(historical_data, forecast_data, column, prediction_col='yhat'):
    """
    Create an interactive dashboard with multiple visualizations.

    Parameters:
    -----------
    historical_data : pandas.DataFrame
        DataFrame containing the historical time series data.
    forecast_data : pandas.DataFrame
        DataFrame containing the forecast data.
    column : str
        The column name in the historical data.
    prediction_col : str, optional
        The column name for predictions in the forecast data.

    Returns:
    --------
    plotly.graph_objects.Figure
        The created interactive dashboard.
    """
    _check_plotly()

    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Historical and Forecast Data',
            'Hourly Pattern',
            'Weekly Pattern',
            'Forecast Distribution'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "histogram"}]
        ]
    )

    # Add historical and forecast data (top left)
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data[column],
            name='Historical',
            mode='lines',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_data.index,
            y=forecast_data[prediction_col],
            name='Forecast',
            mode='lines',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )

    # Add hourly pattern (top right)
    hourly_avg = historical_data.groupby(historical_data.index.hour)[column].mean().reset_index()
    hourly_avg.columns = ['hour', 'value']

    fig.add_trace(
        go.Bar(
            x=hourly_avg['hour'],
            y=hourly_avg['value'],
            name='Hourly Pattern',
            marker_color='green'
        ),
        row=1, col=2
    )

    # Add weekly pattern (bottom left)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_avg = historical_data.groupby(historical_data.index.dayofweek)[column].mean().reset_index()
    weekly_avg.columns = ['day', 'value']

    fig.add_trace(
        go.Bar(
            x=[day_names[int(i)] for i in weekly_avg['day']],
            y=weekly_avg['value'],
            name='Weekly Pattern',
            marker_color='purple'
        ),
        row=2, col=1
    )

    # Add forecast distribution (bottom right)
    fig.add_trace(
        go.Histogram(
            x=forecast_data[prediction_col],
            name='Forecast Distribution',
            marker_color='orange',
            nbinsx=20
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text='Energy Consumption Analysis Dashboard',
        height=800,
        showlegend=False,
        template='plotly_white'
    )

    # Update axes labels
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Value', row=1, col=1)

    fig.update_xaxes(title_text='Hour of Day', row=1, col=2)
    fig.update_yaxes(title_text='Average Value', row=1, col=2)

    fig.update_xaxes(title_text='Day of Week', row=2, col=1)
    fig.update_yaxes(title_text='Average Value', row=2, col=1)

    fig.update_xaxes(title_text='Value', row=2, col=2)
    fig.update_yaxes(title_text='Frequency', row=2, col=2)

    return fig


def plot_forecast_components(forecast_model, forecast_data, figsize=(12, 10)):
    """
    Plot the components of a Prophet forecast.

    Parameters:
    -----------
    forecast_model : prophet.Prophet
        The trained Prophet model.
    forecast_data : pandas.DataFrame
        DataFrame containing the forecast data.
    figsize : tuple, optional
        Figure size (width, height).

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure.
    """
    try:
        fig = forecast_model.plot_components(forecast_data, figsize=figsize)
        return fig
    except AttributeError:
        raise ValueError("This function is only compatible with Prophet models")