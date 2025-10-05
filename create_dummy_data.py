#!/usr/bin/env python3
"""
Script to create dummy data for development and testing.
This is useful when you don't have the real dataset yet.
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure that the project root directory is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def setup_logging():
    """Configure basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config():
    """Load the application configuration."""
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logging.error(f"Configuration file not found at {config_path}")
        sys.exit(1)


def create_dummy_raw_data(config):
    """Create dummy raw data for development."""
    logger = logging.getLogger(__name__)

    raw_data_path = os.path.join(project_root, config['data']['raw_data_path'])

    # Create dummy time series data
    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=365)  # One year of data

    # Create date range
    hours = int((end_date - start_date).total_seconds() / 3600) + 1
    timestamps = [start_date + timedelta(hours=h) for h in range(hours)]

    # Create synthetic data with daily, weekly, and yearly patterns
    values = []
    for ts in timestamps:
        hour_of_day = ts.hour
        day_of_week = ts.weekday()
        day_of_year = ts.timetuple().tm_yday

        # Daily pattern (higher during day, lower at night)
        daily_pattern = 100 + 50 * np.sin((hour_of_day - 12) * np.pi / 12)

        # Weekly pattern (lower on weekends)
        weekly_pattern = 0 if day_of_week < 5 else -20

        # Yearly pattern (higher in summer and winter)
        yearly_pattern = 30 * np.sin((day_of_year - 80) * 2 * np.pi / 365)

        # Add some noise
        noise = np.random.normal(0, 10)

        # Combine patterns
        value = daily_pattern + weekly_pattern + yearly_pattern + noise
        values.append(max(0, value))  # Ensure no negative values

    # Create DataFrame
    df = pd.DataFrame({
        config['data']['date_column']: timestamps,
        config['data']['target_column']: values
    })

    # Ensure directory exists
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

    # Save to CSV
    df.to_csv(raw_data_path, index=False)
    logger.info(f"Created dummy raw data at {raw_data_path}")

    return df


def create_dummy_processed_data(config, raw_df=None):
    """Create dummy processed data with added features."""
    logger = logging.getLogger(__name__)

    processed_data_path = os.path.join(project_root, config['data']['processed_data_path'])

    if raw_df is None:
        # If no raw data is provided, check if raw data file exists
        raw_data_path = os.path.join(project_root, config['data']['raw_data_path'])
        if os.path.exists(raw_data_path):
            logger.info(f"Loading existing raw data from {raw_data_path}")
            raw_df = pd.read_csv(raw_data_path)
            raw_df[config['data']['date_column']] = pd.to_datetime(raw_df[config['data']['date_column']])
        else:
            logger.error("No raw data available. Run create_dummy_raw_data first.")
            return

    # Create a copy of the raw DataFrame
    df = raw_df.copy()

    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[config['data']['date_column']]):
        df[config['data']['date_column']] = pd.to_datetime(df[config['data']['date_column']])

    # Set the date column as index
    df.set_index(config['data']['date_column'], inplace=True)

    # Add time-based features if specified in config
    if config['features']['add_time_features']:
        logger.info("Adding time features")

        # Add selected time features
        for feature in config['features']['time_features']:
            if feature == 'hour':
                df['hour'] = df.index.hour
            elif feature == 'dayofweek':
                df['dayofweek'] = df.index.dayofweek
            elif feature == 'month':
                df['month'] = df.index.month
            elif feature == 'quarter':
                df['quarter'] = df.index.quarter
            elif feature == 'is_weekend':
                df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

    # Add lag features if specified in config
    if config['features']['create_lag_features']:
        logger.info("Adding lag features")

        target_col = config['data']['target_column']

        for lag in config['features']['lag_hours']:
            df[f'lag_{lag}'] = df[target_col].shift(lag)

    # Reset index to move datetime back to a column
    df.reset_index(inplace=True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # Save to CSV
    df.to_csv(processed_data_path, index=False)
    logger.info(f"Created dummy processed data at {processed_data_path}")


def main():
    """Main function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting dummy data creation")

    # Load configuration
    config = load_config()

    # Create dummy raw data
    raw_df = create_dummy_raw_data(config)

    # Create dummy processed data
    create_dummy_processed_data(config, raw_df)

    logger.info("Dummy data creation completed")


if __name__ == "__main__":
    main()