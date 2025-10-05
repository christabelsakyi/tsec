#!/usr/bin/env python3
"""
Model training script for Energy Consumption Forecasting.
Supports ARIMA, Prophet, and LSTM models.
"""

import os
import sys
import yaml
import argparse
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


def setup_logging():
    """Configure basic logging with fallback to simple configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    log_config_path = os.path.join(project_root, 'config', 'logging_config.yaml')

    try:
        if os.path.exists(log_config_path):
            with open(log_config_path, 'r') as f:
                import logging.config
                log_config = yaml.safe_load(f)

                # Make sure all log file directories exist
                for handler in log_config.get('handlers', {}).values():
                    if 'filename' in handler:
                        log_file = handler['filename']
                        # Handle relative paths
                        if not os.path.isabs(log_file):
                            log_file = os.path.join(project_root, log_file)
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)

                logging.config.dictConfig(log_config)
                return
    except Exception as e:
        print(f"Warning: Could not use the logging config file: {str(e)}")
        print("Falling back to basic logging configuration")

    # Fallback to basic logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, 'training.log')),
            logging.StreamHandler()
        ]
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


def load_data(config):
    """Load and preprocess the dataset."""
    logger = logging.getLogger('models')

    # Check if processed data exists
    processed_data_path = os.path.join(
        project_root,
        config['data']['processed_data_path']
    )

    if os.path.exists(processed_data_path):
        logger.info(f"Loading processed data from {processed_data_path}")
        df = pd.read_csv(processed_data_path)

        # Convert date column to datetime
        df[config['data']['date_column']] = pd.to_datetime(df[config['data']['date_column']])
        df.set_index(config['data']['date_column'], inplace=True)

        # Set explicit frequency if we have time series data
        if isinstance(df.index, pd.DatetimeIndex):
            # Infer frequency from data if possible
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                df = df.asfreq(inferred_freq)
            else:
                # Default to hourly if we can't infer (based on your model setup)
                df = df.asfreq('h')
                logger.info("Could not infer frequency from data, defaulting to hourly (h)")

        return df
    else:
        # Try to create dummy data
        logger.warning(f"Processed data file not found at {processed_data_path}")
        try:
            # Create dummy data
            logger.info("Creating dummy data for training")

            # Create dummy time series data
            end_date = pd.Timestamp.now().floor('H')
            start_date = end_date - pd.Timedelta(days=90)  # 90 days of hourly data
            date_range = pd.date_range(start=start_date, end=end_date, freq='h')

            # Create synthetic data with daily and weekly patterns
            df = pd.DataFrame({
                config['data']['date_column']: date_range,
                config['data']['target_column']: [
                    300 + 50 * np.sin(hour / 24 * 2 * np.pi) +
                    20 * np.sin(day / 7 * 2 * np.pi) +
                    np.random.normal(0, 10)
                    for day, hour in zip(
                        date_range.dayofyear,
                        date_range.hour
                    )
                ]
            })

            # Set date column as index
            df.set_index(config['data']['date_column'], inplace=True)
            # Frequency is already set by pd.date_range with freq='H'

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

            # Save dummy data
            df.reset_index().to_csv(processed_data_path, index=False)
            logger.info(f"Saved dummy data to {processed_data_path}")

            return df

        except Exception as e:
            logger.error(f"Failed to create dummy data: {str(e)}")
            logger.error(f"Processed data not found. Please run preprocessing script first, or create the data file.")
            sys.exit(1)


# Import required modules based on model type
def train_arima_model(df, config):
    """Train an ARIMA model."""
    try:
        import statsmodels.api as sm
        from pmdarima import auto_arima
    except ImportError:
        logging.error("Required packages for ARIMA model not installed. Run 'pip install statsmodels pmdarima'")
        sys.exit(1)

    logger = logging.getLogger('models.arima')
    logger.info("Training ARIMA model")

    # Get configuration parameters
    p = config['models']['arima']['p']
    d = config['models']['arima']['d']
    q = config['models']['arima']['q']
    seasonal_order = config['models']['arima']['seasonal_order']

    # Set explicit frequency for the time series data
    df_model = df.copy()
    # Make sure the index is datetime and set frequency to hourly
    if not isinstance(df_model.index, pd.DatetimeIndex):
        df_model.index = pd.to_datetime(df_model.index)
    df_model = df_model.asfreq('h')

    # Check if auto_arima should be used
    if p == 'auto' or d == 'auto' or q == 'auto':
        logger.info("Using auto_arima to determine optimal parameters")
        auto_model = auto_arima(
            df_model[config['data']['target_column']],
            seasonal=True,
            m=seasonal_order['s'],
            stepwise=True,
            trace=True
        )
        p = auto_model.order[0] if p == 'auto' else p
        d = auto_model.order[1] if d == 'auto' else d
        q = auto_model.order[2] if q == 'auto' else q

        # Get seasonal order if auto
        if any(param == 'auto' for param in [seasonal_order['p'], seasonal_order['d'], seasonal_order['q']]):
            s_p = auto_model.seasonal_order[0] if seasonal_order['p'] == 'auto' else seasonal_order['p']
            s_d = auto_model.seasonal_order[1] if seasonal_order['d'] == 'auto' else seasonal_order['d']
            s_q = auto_model.seasonal_order[2] if seasonal_order['q'] == 'auto' else seasonal_order['q']
            s_s = seasonal_order['s']
        else:
            s_p = seasonal_order['p']
            s_d = seasonal_order['d']
            s_q = seasonal_order['q']
            s_s = seasonal_order['s']
    else:
        s_p = seasonal_order['p']
        s_d = seasonal_order['d']
        s_q = seasonal_order['q']
        s_s = seasonal_order['s']

    logger.info(f"ARIMA order: ({p}, {d}, {q}) with seasonal order: ({s_p}, {s_d}, {s_q}, {s_s})")

    # Fit the ARIMA model
    model = sm.tsa.SARIMAX(
        df_model[config['data']['target_column']],
        order=(p, d, q),
        seasonal_order=(s_p, s_d, s_q, s_s),
        freq='h'  # Explicitly set frequency
    )

    results = model.fit(disp=False)
    logger.info("ARIMA model training completed")

    return results


def train_prophet_model(df, config):
    """Train a Facebook Prophet model."""
    try:
        from prophet import Prophet
    except ImportError:
        logging.error("Prophet package not installed. Run 'pip install prophet'")
        sys.exit(1)

    logger = logging.getLogger('models.prophet')
    logger.info("Training Prophet model")

    # Prepare data for Prophet (needs 'ds' and 'y' columns)
    prophet_df = df.reset_index().rename(
        columns={
            config['data']['date_column']: 'ds',
            config['data']['target_column']: 'y'
        }
    )

    # Create and fit model
    model = Prophet(
        seasonality_mode=config['models']['prophet']['seasonality_mode'],
        yearly_seasonality=config['models']['prophet']['yearly_seasonality'],
        weekly_seasonality=config['models']['prophet']['weekly_seasonality'],
        daily_seasonality=config['models']['prophet']['daily_seasonality']
    )

    # Add additional regressors if available in the dataframe
    for feature in config['features']['time_features']:
        if feature in prophet_df.columns:
            model.add_regressor(feature)

    model.fit(prophet_df)
    logger.info("Prophet model training completed")

    return model


def train_lstm_model(df, config):
    """Train an LSTM model with improved GPU compatibility."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler

        # Force TensorFlow to use CPU
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Limit TensorFlow memory growth to avoid OOM errors
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError as e:
                print(f"Unable to set memory growth: {e}")

    except ImportError:
        logging.error("Required packages for LSTM model not installed. Run 'pip install tensorflow sklearn'")
        sys.exit(1)

    logger = logging.getLogger('models.lstm')
    logger.info("Training LSTM model")

    # Extract the target column
    data = df[config['data']['target_column']].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM
    seq_length = config['models']['lstm']['sequence_length']
    X, y = [], []

    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length])

    X = np.array(X)
    y = np.array(y)

    # Split into training and testing sets
    train_size = int(len(X) * (1 - config['training']['test_size']))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build a simpler LSTM model that should work on CPU or older GPUs
    model = Sequential()
    model.add(LSTM(
        units=min(64, config['models']['lstm']['units']),  # Use fewer units
        input_shape=(X_train.shape[1], X_train.shape[2]),
        activation='relu',
        recurrent_activation='sigmoid'  # Simpler activation
    ))
    model.add(Dropout(config['models']['lstm']['dropout']))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        # Use float32 precision instead of default float16 for mixed precision
        run_eagerly=True  # Run eagerly for better compatibility
    )

    # Define early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Reduce batch size for better memory usage
    adjusted_batch_size = min(32, config['models']['lstm']['batch_size'])

    # Reduce epochs for testing
    adjusted_epochs = min(50, config['models']['lstm']['epochs'])

    # Train the model
    model.fit(
        X_train,
        y_train,
        epochs=adjusted_epochs,
        batch_size=adjusted_batch_size,
        validation_split=config['models']['lstm']['validation_split'],
        callbacks=[early_stop],
        verbose=1
    )

    logger.info("LSTM model training completed")

    # Keep the last sequence for future predictions
    latest_data = scaled_data[-seq_length:]

    # Return both the model and the scaler for later use
    return {
        'model': model,
        'scaler': scaler,
        'latest_data': latest_data
    }


def save_model(model, model_type, config):
    """Save the trained model to disk."""
    logger = logging.getLogger('models')

    # Create directory if it doesn't exist
    model_dir = os.path.join(project_root, 'models', 'saved')
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(
        model_dir,
        f"{model_type}_model.pkl"
    )

    logger.info(f"Saving {model_type} model to {model_path}")
    joblib.dump(model, model_path)
    logger.info("Model saved successfully")


def main():
    """Main function to train models."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger('models')

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train time series forecasting models')
    parser.add_argument(
        '--model',
        type=str,
        choices=['arima', 'prophet', 'lstm'],
        help='Model type to train'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Determine model type
    model_type = args.model if args.model else config['models']['selected_model']
    logger.info(f"Training {model_type} model")

    # Load data
    df = load_data(config)

    # Train model based on type
    if model_type == 'arima':
        model = train_arima_model(df, config)
    elif model_type == 'prophet':
        model = train_prophet_model(df, config)
    elif model_type == 'lstm':
        model = train_lstm_model(df, config)
    else:
        logger.error(f"Unsupported model type: {model_type}")
        sys.exit(1)

    # Save trained model
    save_model(model, model_type, config)

    logger.info(f"Model training pipeline completed successfully")


if __name__ == "__main__":
    main()