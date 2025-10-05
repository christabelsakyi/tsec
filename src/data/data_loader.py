"""
Data loader module for energy consumption forecasting.
Handles loading, validation, and initial inspection of datasets.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger('data_loader')


def load_csv_data(file_path, date_col=None, parse_dates=True, index_col=None):
    """
    Load data from a CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file.
    date_col : str or list, optional
        Column(s) to parse as dates.
    parse_dates : bool, optional
        Whether to parse dates.
    index_col : str or int, optional
        Column to set as index.

    Returns:
    --------
    pandas.DataFrame
        The loaded data.
    """
    logger.info(f"Loading CSV data from {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Determine file size for potential chunking
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB

        # For large files (>100MB), use chunking to avoid memory issues
        if file_size > 100:
            logger.info(f"File size is {file_size:.2f} MB. Using chunked loading.")
            chunks = []
            for chunk in pd.read_csv(
                    file_path,
                    parse_dates=date_col if parse_dates else False,
                    chunksize=100000  # Adjust based on available memory
            ):
                chunks.append(chunk)
            df = pd.concat(chunks)

            # Set index if specified
            if index_col is not None:
                df.set_index(index_col, inplace=True)
        else:
            # For smaller files, load all at once
            df = pd.read_csv(
                file_path,
                parse_dates=date_col if parse_dates else False,
                index_col=index_col
            )

        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        raise


def load_energy_data(file_path, config):
    """
    Load energy consumption data with specific handling for the dataset.

    Parameters:
    -----------
    file_path : str
        Path to the data file.
    config : dict
        Configuration dictionary with data loading parameters.

    Returns:
    --------
    pandas.DataFrame
        The loaded and initially processed energy data.
    """
    logger.info(f"Loading energy consumption data from {file_path}")

    date_col = config['data']['date_column']

    try:
        # Load the data
        df = load_csv_data(file_path, date_col=date_col, parse_dates=True)

        # Determine if we're working with the UCI dataset or AEP dataset
        if 'Global_active_power' in df.columns:
            logger.info("Detected UCI Household Electric Power Consumption dataset")
            return process_uci_dataset(df, config)
        elif 'AEP_MW' in df.columns:
            logger.info("Detected AEP hourly energy consumption dataset")
            return process_aep_dataset(df, config)
        else:
            # Generic processing for other datasets
            logger.info("Using generic data processing")
            return process_generic_dataset(df, config)

    except Exception as e:
        logger.error(f"Error loading energy data: {str(e)}")
        raise


def process_uci_dataset(df, config):
    """
    Process the UCI Household Electric Power Consumption dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        The raw dataset.
    config : dict
        Configuration dictionary.

    Returns:
    --------
    pandas.DataFrame
        The processed dataset.
    """
    logger.info("Processing UCI Household Electric Power Consumption dataset")

    # Combine Date and Time columns if they exist separately
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df.drop(['Date', 'Time'], axis=1, inplace=True)

    # Set the datetime column as index
    df.set_index('Datetime', inplace=True)

    # Convert numeric columns (some might be strings with '?' for missing values)
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    for col in numeric_cols:
        if col in df.columns:
            # Replace '?' with NaN
            if df[col].dtype == 'object':
                df[col] = df[col].replace('?', np.nan)

            # Convert to float
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate total power consumption if needed
    if 'Global_active_power' in df.columns and config['data']['target_column'] == 'power_consumption':
        # Global_active_power is in kilowatts, convert to power consumption
        df['power_consumption'] = df['Global_active_power'] * 1000  # Convert to watts

    # Handle missing values
    df = handle_missing_values(df, config)

    # Resample if needed
    if config['data']['resampling']['enabled']:
        freq = config['data']['resampling']['freq']
        df = df.resample(freq).mean()
        logger.info(f"Resampled data to {freq} frequency")

    return df


def process_aep_dataset(df, config):
    """
    Process the AEP hourly energy consumption dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        The raw dataset.
    config : dict
        Configuration dictionary.

    Returns:
    --------
    pandas.DataFrame
        The processed dataset.
    """
    logger.info("Processing AEP hourly energy consumption dataset")

    # Ensure the date column is datetime
    date_col = config['data']['date_column']
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    # Handle missing values
    df = handle_missing_values(df, config)

    # Resample if needed
    if config['data']['resampling']['enabled']:
        freq = config['data']['resampling']['freq']
        df = df.resample(freq).mean()
        logger.info(f"Resampled data to {freq} frequency")

    return df


def process_generic_dataset(df, config):
    """
    Generic processing for energy consumption datasets.

    Parameters:
    -----------
    df : pandas.DataFrame
        The raw dataset.
    config : dict
        Configuration dictionary.

    Returns:
    --------
    pandas.DataFrame
        The processed dataset.
    """
    logger.info("Applying generic dataset processing")

    date_col = config['data']['date_column']

    # Ensure the date column is datetime and set as index
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)

    # Handle missing values
    df = handle_missing_values(df, config)

    # Resample if needed
    if config['data']['resampling']['enabled']:
        freq = config['data']['resampling']['freq']
        df = df.resample(freq).mean()
        logger.info(f"Resampled data to {freq} frequency")

    return df


def handle_missing_values(df, config):
    """
    Handle missing values in the dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset with potential missing values.
    config : dict
        Configuration dictionary.

    Returns:
    --------
    pandas.DataFrame
        The dataset with handled missing values.
    """
    # Check for missing values
    missing_count = df.isna().sum()
    total_missing = missing_count.sum()

    if total_missing > 0:
        logger.info(f"Found {total_missing} missing values")
        logger.info(f"Missing values by column:\n{missing_count[missing_count > 0]}")

        # For time series, interpolation is often a good approach
        df = df.interpolate(method='time')

        # For any remaining NAs at the beginning or end, use forward/backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Check if we still have missing values
        remaining_missing = df.isna().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"Still have {remaining_missing} missing values after interpolation")
        else:
            logger.info("All missing values have been handled")
    else:
        logger.info("No missing values found in the dataset")

    return df


def analyze_dataset(df):
    """
    Perform initial analysis of the dataset and return summary statistics.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze.

    Returns:
    --------
    dict
        Dictionary with analysis results.
    """
    analysis = {}

    # Basic information
    analysis['shape'] = df.shape
    analysis['columns'] = list(df.columns)
    analysis['dtypes'] = df.dtypes.to_dict()
    analysis['memory_usage'] = df.memory_usage(deep=True).sum() / (1024 * 1024)  # in MB

    # Time range
    analysis['time_range'] = {
        'start': df.index.min(),
        'end': df.index.max(),
        'duration_days': (df.index.max() - df.index.min()).days
    }

    # Check for gaps in time series
    time_diff = df.index.to_series().diff().dropna()
    analysis['time_gaps'] = {
        'min_gap': time_diff.min(),
        'max_gap': time_diff.max(),
        'median_gap': time_diff.median(),
        'mean_gap': time_diff.mean()
    }

    # Summary statistics
    analysis['summary_stats'] = df.describe().to_dict()

    # Missing values
    analysis['missing_values'] = df.isna().sum().to_dict()

    return analysis


def split_data(df, target_col, test_size=0.2, validation_size=0.1):
    """
    Split the data into training, validation, and test sets.
    For time series, splitting is done chronologically.

    Parameters:
    -----------
    df : pandas.DataFrame
        The processed dataset with datetime index.
    target_col : str
        The target column name.
    test_size : float, optional
        Proportion of data to use for testing.
    validation_size : float, optional
        Proportion of data to use for validation.

    Returns:
    --------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("Splitting data into train, validation, and test sets")

    # Sort by time
    df = df.sort_index()

    # Calculate split points
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - validation_size))

    # Split data
    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx:test_idx]
    test_df = df.iloc[test_idx:]

    logger.info(f"Train set: {train_df.shape}, Validation set: {val_df.shape}, Test set: {test_df.shape}")

    # For time series, we often need all features for prediction
    # Here, X is the dataframe and y is the target series
    X_train, y_train = train_df, train_df[target_col]
    X_val, y_val = val_df, val_df[target_col]
    X_test, y_test = test_df, test_df[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_data_sample(df, n_samples=5):
    """
    Get a sample of the data for inspection.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset.
    n_samples : int, optional
        Number of samples to return.

    Returns:
    --------
    pandas.DataFrame
        A sample of the data.
    """
    # Get samples from beginning, middle, and end
    first = df.iloc[:n_samples]
    middle_idx = len(df) // 2
    middle = df.iloc[middle_idx:middle_idx + n_samples]
    last = df.iloc[-n_samples:]

    return pd.concat([first, middle, last])