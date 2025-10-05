"""
Implementation of LSTM neural network model for energy consumption forecasting.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


class LSTMForecaster:
    """
    LSTM-based forecasting model for energy consumption.

    Parameters:
    -----------
    config : dict
        Configuration parameters for the model.
    """

    def __init__(self, config):
        """Initialize the LSTM forecasting model."""
        self.logger = logging.getLogger('models.lstm')
        self.config = config
        self.model = None
        self.scaler = None
        self.target_col = config['data']['target_column']
        self.date_col = config['data']['date_column']

        # LSTM parameters
        self.units = config['models']['lstm']['units']
        self.dropout = config['models']['lstm']['dropout']
        self.recurrent_dropout = config['models']['lstm']['recurrent_dropout']
        self.epochs = config['models']['lstm']['epochs']
        self.batch_size = config['models']['lstm']['batch_size']
        self.sequence_length = config['models']['lstm']['sequence_length']
        self.validation_split = config['models']['lstm']['validation_split']

        # Store latest sequence for continuing predictions
        self.latest_sequence = None

        self.logger.info("Initialized LSTM forecaster")

    def prepare_data(self, df):
        """
        Prepare the data for LSTM model by scaling and creating sequences.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame with datetime index.

        Returns:
        --------
        tuple
            (X, y) where X is the sequences and y is the target values.
        """
        self.logger.info("Preparing data for LSTM model")

        # Extract the target column
        if isinstance(df, pd.DataFrame):
            if self.target_col in df.columns:
                values = df[self.target_col].values
            else:
                raise ValueError(f"Target column '{self.target_col}' not found in data")
        else:
            # Assume it's already a series or array
            values = np.array(df)

        # Reshape for scaling if needed
        if len(values.shape) == 1:
            values = values.reshape(-1, 1)

        # Initialize and fit scaler
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_values = self.scaler.fit_transform(values)
        else:
            scaled_values = self.scaler.transform(values)

        # Create sequences
        X, y = self._create_sequences(scaled_values)

        # Store the latest sequence for future predictions
        self.latest_sequence = scaled_values[-self.sequence_length:].reshape(1, self.sequence_length, 1)

        self.logger.info(f"Prepared sequences with shape X: {X.shape}, y: {y.shape}")
        return X, y

    def _create_sequences(self, values):
        """
        Create sequences for LSTM training.

        Parameters:
        -----------
        values : numpy.ndarray
            Scaled values.

        Returns:
        --------
        tuple
            (X, y) where X is the sequences and y is the target values.
        """
        seq_length = self.sequence_length
        X, y = [], []

        for i in range(len(values) - seq_length):
            X.append(values[i:i + seq_length])
            y.append(values[i + seq_length])

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """
        Build the LSTM model architecture.

        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features).

        Returns:
        --------
        tensorflow.keras.models.Sequential
            The LSTM model.
        """
        self.logger.info("Building LSTM model architecture")

        model = Sequential()

        # First LSTM layer with return sequences for stacking
        model.add(LSTM(
            units=self.units,
            return_sequences=True,
            input_shape=input_shape,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout
        ))

        # Second LSTM layer
        model.add(LSTM(
            units=self.units // 2,  # Reduce units for deeper layers
            return_sequences=False,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout
        ))

        # Add a dense layer
        model.add(Dense(units=self.units // 4, activation='relu'))
        model.add(Dropout(self.dropout))

        # Output layer
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.summary(print_fn=self.logger.info)
        return model

    def fit(self, df, validation_data=None, callbacks=None):
        """
        Fit the LSTM model to the training data.

        Parameters:
        -----------
        df : pandas.DataFrame or pandas.Series
            The training data.
        validation_data : tuple, optional
            (X_val, y_val) for validation.
        callbacks : list, optional
            List of Keras callbacks.

        Returns:
        --------
        self
            The fitted model instance.
        """
        self.logger.info("Fitting LSTM model")

        # Prepare training data
        X_train, y_train = self.prepare_data(df)

        # Build the model if not already built
        if self.model is None:
            # Input shape: (sequence_length, features)
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape)

        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    filepath='./models/saved/lstm_best_model.h5',
                    monitor='val_loss',
                    save_best_only=True
                )
            ]

        # Fit the model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split if validation_data is None else 0,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        self.logger.info("LSTM model training completed")
        return history

    def predict(self, periods=None, horizon=None):
        """
        Generate forecast using the fitted model.

        Parameters:
        -----------
        periods : int, optional
            Number of periods to forecast.
        horizon : int, optional
            Alternative to periods, forecast horizon.

        Returns:
        --------
        numpy.ndarray
            The forecasted values.
        """
        if self.model is None:
            self.logger.error("Model has not been trained yet")
            raise ValueError("Model has not been trained yet. Call fit() first.")

        if self.latest_sequence is None:
            self.logger.error("No data available for prediction")
            raise ValueError("No data available for prediction")

        # Determine number of periods to forecast
        if horizon is not None:
            periods = horizon
        elif periods is None:
            periods = self.config['prediction']['forecast_horizon']

        self.logger.info(f"Generating forecast for {periods} periods")

        # Make predictions
        predictions = []
        current_sequence = self.latest_sequence.copy()

        for _ in range(periods):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])

            # Update sequence for next prediction (rolling window)
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]

        # Convert predictions back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()

        return predictions

    def evaluate(self, y_true):
        """
        Evaluate the model performance.

        Parameters:
        -----------
        y_true : array-like
            The true values to compare against the forecast.

        Returns:
        --------
        dict
            Dictionary of evaluation metrics.
        """
        # Get the forecasted values
        y_pred = self.predict(periods=len(y_true))

        # Ensure y_true is a numpy array
        y_true = np.array(y_true)

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # Mean Absolute Percentage Error (MAPE)
        mask = y_true != 0  # Avoid division by zero
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        # Symmetric Mean Absolute Percentage Error (SMAPE)
        denom = np.abs(y_true) + np.abs(y_pred)
        mask_smape = denom != 0  # Avoid division by zero
        smape = np.mean(2.0 * np.abs(y_pred[mask_smape] - y_true[mask_smape]) / denom[mask_smape]) * 100

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape,
            'R2': r2
        }

        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def plot_training_history(self, history, figsize=(12, 6)):
        """
        Plot the training history.

        Parameters:
        -----------
        history : tensorflow.keras.callbacks.History
            Training history.
        figsize : tuple, optional
            Figure size.

        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing loss plots.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot training & validation loss
        ax.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax.plot(history.history['val_loss'], label='Validation Loss')

        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        return fig

    def save_model(self, path):
        """
        Save the fitted model to disk.

        Parameters:
        -----------
        path : str
            Path to save the model.
        """
        if self.model is None:
            self.logger.error("No trained model to save")
            raise ValueError("Model has not been trained yet")

        self.logger.info(f"Saving model to {path}")

        # Save the LSTM model
        self.model.save(path)

        # Save the scaler and latest sequence separately
        import joblib
        scaler_path = path.replace('.h5', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        sequence_path = path.replace('.h5', '_sequence.npy')
        np.save(sequence_path, self.latest_sequence)

    @classmethod
    def load_model(cls, path, config):
        """
        Load a saved model from disk.

        Parameters:
        -----------
        path : str
            Path to the saved model.
        config : dict
            Configuration dictionary.

        Returns:
        --------
        LSTMForecaster
            Loaded model instance.
        """
        import joblib

        logger = logging.getLogger('models.lstm')
        logger.info(f"Loading model from {path}")

        # Create a new instance
        instance = cls(config)

        # Load the LSTM model
        instance.model = load_model(path)

        # Load the scaler and latest sequence
        scaler_path = path.replace('.h5', '_scaler.pkl')
        sequence_path = path.replace('.h5', '_sequence.npy')

        try:
            instance.scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler successfully")
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise

        try:
            instance.latest_sequence = np.load(sequence_path)
            logger.info("Loaded sequence successfully")
        except Exception as e:
            logger.error(f"Error loading sequence: {str(e)}")
            raise

        return instance