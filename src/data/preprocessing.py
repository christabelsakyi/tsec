#!/usr/bin/env python3
"""
Data preprocessing script for Energy Consumption Forecasting.
Cleans data, handles missing values, and creates features.
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


def load_raw_data(config):
    """Load the raw dataset."""
    logger = logging.getLogger('preprocessing')

    raw_data_path = os.path.join(
        project_root,
        config['data']['raw_data_path']
    )

    if not os.path.exists(raw_data_path):
        logger.error(f"Raw data file not found at {raw_data_path}")
        sys.exit(1)

    logger.info(f"Loading raw data from {raw_data_path}")

    # Detect file type and load accordingly
    if raw_data_path.endswith('.csv'):
        df = pd.read_csv(raw_data_path)
    elif raw_data_path.endswith('.xlsx') or raw_data_path.endswith('.xls'):
        df = pd.read_excel(raw_data_path)
    else:
        logger.error(f"Unsupported file format: {raw_data_path}")
        sys.exit(1)

    logger.info(f"Raw data loaded successfully. Shape: {df.shape}")
    return df


def clean_data(df, config):
    """Clean the dataset by handling missing values and outliers."""
    logger = logging.getLogger('preprocessing')
    logger.info("Cleaning data")

    # Convert date column to datetime
    date_col = config['data']['date_column']
    target_col = config['data']['target_column']

    # Make a copy to avoid modifying the original
    df_clean = df.copy()

    # Convert date column to datetime
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])

    # Set the date column as index
    df_clean.set_index(date_col, inplace=True)

    # Sort by date
    df_clean.sort_index(inplace=True)

    # Handle missing values in the target column
    missing_values = df_clean[target_col].isna().sum()
    if missing_values > 0:
        logger.info(f"Found {missing_values} missing values in {target_col}")

        # Interpolate missing values
        df_clean[target_col] = df_clean[target_col].interpolate(method='time')

        # If there are still missing values at the beginning or end, forward/backward fill
        df_clean[target_col] = df_clean[target_col].fillna(method='ffill').fillna(method='bfill')

    # Resample data if enabled in config
    if config['data']['resampling']['enabled']:
        freq = config['data']['resampling']['freq']
        logger.info(f"Resampling data to {freq} frequency")
        df_clean = df_clean.resample(freq).mean()

    # Handle outliers (Z-score method)
    z_scores = np.abs((df_clean[target_col] - df_clean[target_col].mean()) / df_clean[target_col].std())
    outliers = z_scores > 3

    if outliers.sum() > 0:
        logger.info(f"Found {outliers.sum()} outliers in {target_col}")

        # Replace outliers with the median of their neighboring points
        outlier_indices = np.where(outliers)[0]
        for idx in outlier_indices:
            # Use a window of 5 points (2 before, 2 after, excluding the outlier)
            window_start = max(0, idx - 2)
            window_end = min(len(df_clean), idx + 3)
            neighbors = list(range(window_start, window_end))
            neighbors.remove(idx)

            # Replace outlier with median of neighbors
            neighbor_values = df_clean.iloc[neighbors][target_col]
            df_clean.iloc[idx, df_clean.columns.get_loc(target_col)] = neighbor_values.median()

    logger.info("Data cleaning completed")
    return df_clean


def create_features(df, config):
    """Create time-based features and lag features."""
    logger = logging.getLogger('preprocessing')
    logger.info("Creating features")

    df_features = df.copy()
    target_col = config['data']['target_column']

    # Create time-based features
    if config['features']['add_time_features']:
        time_features = config['features']['time_features']
        logger.info(f"Adding time features: {time_features}")

        if 'hour' in time_features:
            df_features['hour'] = df_features.index.hour

        if 'dayofweek' in time_features:
            df_features['dayofweek'] = df_features.index.dayofweek

        if 'month' in time_features:
            df_features['month'] = df_features.index.month

        if 'quarter' in time_features:
            df_features['quarter'] = df_features.index.quarter

        if 'is_weekend' in time_features:
            df_features['is_weekend'] = df_features.index.dayofweek.isin([5, 6]).astype(int)

    # Create lag features
    if config['features']['create_lag_features']:
        lag_hours = config['features']['lag_hours']
        logger.info(f"Creating lag features: {lag_hours}")

        for lag in lag_hours:
            df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)

    # Drop rows with NaN (from creating lag features)
    if config['features']['create_lag_features']:
        df_features = df_features.dropna()
        logger.info(f"Dropped {len(df) - len(df_features)} rows with NaN values after creating lag features")

    logger.info("Feature creation completed")
    return df_features


def save_processed_data(df, config):
    """Save the processed dataset."""
    logger = logging.getLogger('preprocessing')

    processed_data_path = os.path.join(
        project_root,
        config['data']['processed_data_path']
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    logger.info(f"Saving processed data to {processed_data_path}")
    df.to_csv(processed_data_path)
    logger.info("Processed data saved successfully")


def main():
    """Main function for data preprocessing."""
    # Set up logging
    log_config_path = os.path.join(project_root, 'config', 'logging_config.yaml')
    if os.path.exists(log_config_path):
        with open(log_config_path, 'r') as f:
            log_config = yaml.safe_load(f)
            logging.config.dictConfig(log_config)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger('preprocessing')
    logger.info("Starting data preprocessing")

    # Load configuration
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load raw data
    df_raw = load_raw_data(config)

    # Clean data
    df_clean = clean_data(df_raw, config)

    # Create features
    df_processed = create_features(df_clean, config)

    # Save processed data
    save_processed_data(df_processed, config)

    logger.info("Data preprocessing completed successfully")


if __name__ == "__main__":
    main()