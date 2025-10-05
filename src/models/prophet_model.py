"""
Implementation of the Facebook Prophet model for energy consumption forecasting.
"""

import logging
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric


class ProphetForecaster:
    """
    Prophet-based forecasting model for energy consumption.

    Parameters:
    -----------
    config : dict
        Configuration parameters for the model.
    """

    def __init__(self, config):
        """Initialize the Prophet forecasting model."""
        self.logger = logging.getLogger('models.prophet')
        self.config = config
        self.model = None
        self.target_col = config['data']['target_column']
        self.date_col = config['data']['date_column']
        self.forecast = None

        # Configure Prophet parameters
        self.seasonality_mode = config['models']['prophet']['seasonality_mode']
        self.yearly_seasonality = config['models']['prophet']['yearly_seasonality']
        self.weekly_seasonality = config['models']['prophet']['weekly_seasonality']
        self.daily_seasonality = config['models']['prophet']['daily_seasonality']
        self.uncertainty_samples = config['models']['prophet']['uncertainty_samples']

        self.logger.info("Initialized Prophet forecaster")

    def prepare_data(self, df):
        """
        Prepare the data for Prophet model.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame with datetime index.

        Returns:
        --------
        pandas.DataFrame
            DataFrame formatted for Prophet with 'ds' and 'y' columns.
        """
        self.logger.info("Preparing data for Prophet model")

        # Prophet requires columns named 'ds' (date) and 'y' (target)
        prophet_df = df.reset_index().rename(
            columns={
                self.date_col if self.date_col in df.columns else df.index.name: 'ds',
                self.target_col: 'y'
            }
        )

        # Add additional features if specified in config
        if 'add_time_features' in self.config['features'] and self.config['features']['add_time_features']:
            self.logger.info("Adding time features")
            time_features = self.config['features']['time_features']

            # Add supported features
            for feature in time_features:
                if feature == 'hour':
                    prophet_df['hour'] = prophet_df['ds'].dt.hour
                elif feature == 'dayofweek':
                    prophet_df['dayofweek'] = prophet_df['ds'].dt.dayofweek
                elif feature == 'month':
                    prophet_df['month'] = prophet_df['ds'].dt.month
                elif feature == 'quarter':
                    prophet_df['quarter'] = prophet_df['ds'].dt.quarter
                elif feature == 'is_weekend':
                    prophet_df['is_weekend'] = prophet_df['ds'].dt.dayofweek.isin([5, 6]).astype(int)

        self.logger.info(f"Prepared data shape: {prophet_df.shape}")
        return prophet_df

    def fit(self, df):
        """
        Fit the Prophet model to the training data.

        Parameters:
        -----------
        df : pandas.DataFrame
            The training data.

        Returns:
        --------
        self
            The fitted model instance.
        """
        self.logger.info("Fitting Prophet model")

        # Prepare data
        prophet_df = self.prepare_data(df)

        # Initialize model
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            uncertainty_samples=self.uncertainty_samples
        )

        # Add regressor columns if available
        time_features = self.config['features'].get('time_features', [])
        for feature in time_features:
            if feature in prophet_df.columns and feature not in ['ds', 'y']:
                self.model.add_regressor(feature)
                self.logger.info(f"Added regressor: {feature}")

        # Fit the model
        self.model.fit(prophet_df)
        self.logger.info("Prophet model fitting completed")

        return self

    def predict(self, periods=None, horizon=None, freq='H', include_history=False):
        """
        Generate forecast using the fitted model.

        Parameters:
        -----------
        periods : int, optional
            Number of periods to forecast.
        horizon : int, optional
            Alternative to periods, forecast horizon in hours.
        freq : str, optional
            Frequency of forecast, default is hourly.
        include_history : bool, optional
            Whether to include the historical data in the forecast.

        Returns:
        --------
        pandas.DataFrame or numpy.ndarray
            The forecasted values. Returns DataFrame if include_history is True,
            otherwise returns only the forecasted values as an array.
        """
        if self.model is None:
            self.logger.error("Model has not been fitted yet")
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Determine number of periods to forecast
        if horizon is not None:
            periods = horizon
        elif periods is None:
            periods = self.config['prediction']['forecast_horizon']

        self.logger.info(f"Generating forecast for {periods} periods with frequency {freq}")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # Add regressor values if needed
        time_features = self.config['features'].get('time_features', [])
        for feature in time_features:
            if feature in ['hour', 'dayofweek', 'month', 'quarter',
                           'is_weekend'] and feature in self.model.extra_regressors:
                if feature == 'hour':
                    future['hour'] = future['ds'].dt.hour
                elif feature == 'dayofweek':
                    future['dayofweek'] = future['ds'].dt.dayofweek
                elif feature == 'month':
                    future['month'] = future['ds'].dt.month
                elif feature == 'quarter':
                    future['quarter'] = future['ds'].dt.quarter
                elif feature == 'is_weekend':
                    future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)

        # Generate forecast
        self.forecast = self.model.predict(future)

        if include_history:
            return self.forecast
        else:
            # Return only the forecasted part (not including history)
            forecast_values = self.forecast.tail(periods)['yhat'].values
            return forecast_values

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
        if self.forecast is None:
            self.logger.error("No forecast available for evaluation")
            raise ValueError("No forecast available. Call predict() first.")

        # Get the forecasted values for the evaluation period
        y_pred = self.forecast.tail(len(y_true))['yhat'].values

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

    def cross_validate(self, df, initial=None, period=None, horizon=None, parallel=None):
        """
        Perform cross-validation for the Prophet model.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame.
        initial : str, optional
            Initial training period.
        period : str, optional
            Spacing between cutoff dates.
        horizon : str, optional
            Forecast horizon.
        parallel : str or None, optional
            Parallelization method ('processes' or 'threads').

        Returns:
        --------
        tuple
            (cv_results, cv_metrics)
        """
        self.logger.info("Performing cross-validation")

        # Prepare data
        prophet_df = self.prepare_data(df)

        # Set default parameters if not provided
        if initial is None:
            initial = '180 days'
        if period is None:
            period = '30 days'
        if horizon is None:
            horizon = '30 days'

        # Perform cross-validation
        cv_results = cross_validation(
            model=self.model,
            df=prophet_df,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel=parallel
        )

        # Calculate performance metrics
        cv_metrics = performance_metrics(cv_results)

        self.logger.info(f"Cross-validation completed with {len(cv_results)} cutoffs")
        return cv_results, cv_metrics

    def plot_components(self, figsize=(16, 12)):
        """
        Plot the components of the Prophet forecast.

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height).

        Returns:
        --------
        matplotlib.figure.Figure
            The plot of components.
        """
        if self.model is None or self.forecast is None:
            self.logger.error("Model has not been fitted or no forecast available")
            raise ValueError("Model needs to be fitted and forecast generated first")

        self.logger.info("Plotting forecast components")
        fig = self.model.plot_components(self.forecast, figsize=figsize)
        return fig

    def save_model(self, path):
        """
        Save the fitted model to disk.

        Parameters:
        -----------
        path : str
            Path to save the model.
        """
        import joblib

        if self.model is None:
            self.logger.error("No fitted model to save")
            raise ValueError("Model has not been fitted yet")

        self.logger.info(f"Saving model to {path}")
        joblib.dump(self.model, path)

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
        ProphetForecaster
            Loaded model instance.
        """
        import joblib

        logger = logging.getLogger('models.prophet')
        logger.info(f"Loading model from {path}")

        # Create a new instance
        instance = cls(config)

        # Load the model
        instance.model = joblib.load(path)

        return instance