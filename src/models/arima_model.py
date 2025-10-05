"""
Implementation of ARIMA/SARIMAX models for energy consumption forecasting.
"""

import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima


class ARIMAForecaster:
    """
    ARIMA-based forecasting model for energy consumption.

    Parameters:
    -----------
    config : dict
        Configuration parameters for the model.
    """

    def __init__(self, config):
        """Initialize the ARIMA forecasting model."""
        self.logger = logging.getLogger('models.arima')
        self.config = config
        self.model = None
        self.results = None
        self.target_col = config['data']['target_column']
        self.date_col = config['data']['date_column']

        # Model parameters
        self.p = config['models']['arima']['p']
        self.d = config['models']['arima']['d']
        self.q = config['models']['arima']['q']
        self.seasonal_order = config['models']['arima']['seasonal_order']
        self.use_seasonal = any(x != 0 for x in self.seasonal_order.values())

        self.logger.info("Initialized ARIMA forecaster")

    def prepare_data(self, df):
        """
        Prepare the data for ARIMA model.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input DataFrame with datetime index.

        Returns:
        --------
        pandas.Series
            Series with the target variable.
        """
        self.logger.info("Preparing data for ARIMA model")

        # ARIMA models work with a single time series
        if isinstance(df, pd.DataFrame):
            # Extract the target column
            if self.target_col in df.columns:
                series = df[self.target_col]
            else:
                raise ValueError(f"Target column '{self.target_col}' not found in data")
        else:
            # Assume it's already a series
            series = df

        # Ensure the data is a pandas Series with datetime index
        if not isinstance(series, pd.Series):
            series = pd.Series(series)

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index")

        # Make sure the series is sorted by time
        series = series.sort_index()

        self.logger.info(f"Prepared time series with {len(series)} observations")
        return series

    def determine_order(self, series):
        """
        Determine the optimal ARIMA order using auto_arima.

        Parameters:
        -----------
        series : pandas.Series
            The time series data.

        Returns:
        --------
        tuple
            (p, d, q, seasonal_order)
        """
        self.logger.info("Determining optimal ARIMA order using auto_arima")

        # Get the seasonal period from config
        seasonal_period = self.seasonal_order['s']

        # Run auto_arima
        auto_model = auto_arima(
            series,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            d=None, max_d=2,
            D=None, max_D=1,
            seasonal=self.use_seasonal,
            m=seasonal_period if self.use_seasonal else 1,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic'
        )

        # Extract the optimal order
        p, d, q = auto_model.order

        if self.use_seasonal:
            seasonal_order = auto_model.seasonal_order
            P, D, Q, m = seasonal_order
            self.logger.info(f"Optimal ARIMA order: ({p},{d},{q})({P},{D},{Q},{m})")
            return p, d, q, (P, D, Q, m)
        else:
            self.logger.info(f"Optimal ARIMA order: ({p},{d},{q})")
            return p, d, q, None

    def fit(self, df, auto_order=False):
        """
        Fit the ARIMA model to the training data.

        Parameters:
        -----------
        df : pandas.DataFrame or pandas.Series
            The training data.
        auto_order : bool, optional
            Whether to use auto_arima to determine the optimal order.

        Returns:
        --------
        self
            The fitted model instance.
        """
        self.logger.info("Fitting ARIMA model")

        # Prepare data
        series = self.prepare_data(df)

        # Determine model order
        if auto_order or self.p == 'auto' or self.d == 'auto' or self.q == 'auto':
            p, d, q, seasonal_order = self.determine_order(series)
        else:
            p = self.p
            d = self.d
            q = self.q

            if self.use_seasonal:
                seasonal_order = (
                    self.seasonal_order['p'],
                    self.seasonal_order['d'],
                    self.seasonal_order['q'],
                    self.seasonal_order['s']
                )
            else:
                seasonal_order = None

        # Initialize and fit the model
        if self.use_seasonal:
            self.logger.info(
                f"Fitting SARIMAX({p},{d},{q})({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})")
            model = SARIMAX(
                series,
                order=(p, d, q),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            self.logger.info(f"Fitting ARIMA({p},{d},{q})")
            model = ARIMA(series, order=(p, d, q))

        # Fit the model
        self.results = model.fit()
        self.model = model

        self.logger.info("ARIMA model fitting completed")
        return self

    def predict(self, periods=None, horizon=None, return_conf_int=False, alpha=0.05):
        """
        Generate forecast using the fitted model.

        Parameters:
        -----------
        periods : int, optional
            Number of periods to forecast.
        horizon : int, optional
            Alternative to periods, forecast horizon.
        return_conf_int : bool, optional
            Whether to return confidence intervals.
        alpha : float, optional
            Significance level for confidence intervals.

        Returns:
        --------
        numpy.ndarray or tuple
            Forecasted values, or (forecast, lower, upper) if return_conf_int is True.
        """
        if self.results is None:
            self.logger.error("Model has not been fitted yet")
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Determine number of periods to forecast
        if horizon is not None:
            periods = horizon
        elif periods is None:
            periods = self.config['prediction']['forecast_horizon']

        self.logger.info(f"Generating forecast for {periods} periods")

        # Generate forecast
        if return_conf_int:
            forecast = self.results.get_forecast(steps=periods)
            mean_forecast = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=alpha)
            lower = conf_int.iloc[:, 0]
            upper = conf_int.iloc[:, 1]

            return mean_forecast.values, lower.values, upper.values
        else:
            forecast = self.results.forecast(steps=periods)
            return forecast.values

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

    def summary(self):
        """
        Return a summary of the fitted model.

        Returns:
        --------
        str
            Model summary.
        """
        if self.results is None:
            self.logger.error("Model has not been fitted yet")
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return self.results.summary()

    def plot_diagnostics(self, figsize=(16, 12)):
        """
        Plot diagnostic plots for the ARIMA model.

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size.

        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing diagnostic plots.
        """
        if self.results is None:
            self.logger.error("Model has not been fitted yet")
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        self.logger.info("Creating diagnostic plots")
        fig = self.results.plot_diagnostics(figsize=figsize)
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

        if self.results is None:
            self.logger.error("No fitted model to save")
            raise ValueError("Model has not been fitted yet")

        self.logger.info(f"Saving model to {path}")
        joblib.dump(self.results, path)

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
        ARIMAForecaster
            Loaded model instance.
        """
        import joblib

        logger = logging.getLogger('models.arima')
        logger.info(f"Loading model from {path}")

        # Create a new instance
        instance = cls(config)

        # Load the model results
        instance.results = joblib.load(path)

        return instance