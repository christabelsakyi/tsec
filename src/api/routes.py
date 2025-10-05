from flask import request, jsonify
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from .utils import load_model

logger = logging.getLogger('api')


def register_routes(app, config):
    """
    Register all application routes
    """
    # Load model
    model = load_model(config)

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        status = "ok" if model is not None else "warning"
        message = "Service is running" if model is not None else "Service is running but model is not loaded"

        return jsonify({
            "status": status,
            "message": message,
            "model_type": config['models']['selected_model'],
            "debug_mode": config['api']['debug']
        })

    @app.route('/api/forecast', methods=['POST', 'OPTIONS'])
    def forecast():
        """Endpoint for forecasting energy consumption"""
        # Handle OPTIONS request for CORS preflight
        if request.method == 'OPTIONS':
            response = app.make_default_options_response()
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Access-Control-Allow-Methods', 'POST')
            return response

        try:
            # Get request data
            data = request.get_json()
            horizon = data.get('horizon', config['prediction']['forecast_horizon'])

            # Validate input
            if not isinstance(horizon, int) or horizon <= 0:
                return jsonify({"status": "error", "message": "Horizon must be a positive integer"}), 400

            max_horizon = config['api']['max_forecast_days'] * 24  # Convert days to hours
            if horizon > max_horizon:
                return jsonify({
                    "status": "error",
                    "message": f"Forecast horizon cannot exceed {max_horizon} hours ({config['api']['max_forecast_days']} days)"
                }), 400

            # Check if model is loaded
            if model is None:
                return jsonify({
                    "status": "error",
                    "message": "Model not loaded. Please train the model first or enable debug mode."
                }), 500

            # Generate forecast
            logger.info(f"Generating forecast for next {horizon} hours")

            # Implementation based on model type
            try:
                if config['models']['selected_model'] == 'prophet':
                    # Prophet-specific forecast
                    future = model.make_future_dataframe(periods=horizon, freq='h')

                    # Add time features based on config
                    time_features = config['features'].get('time_features', [])

                    if 'hour' in time_features or 'hour' in getattr(model, 'extra_regressors_names', []):
                        future['hour'] = future['ds'].dt.hour

                    if 'dayofweek' in time_features or 'dayofweek' in getattr(model, 'extra_regressors_names', []):
                        future['dayofweek'] = future['ds'].dt.dayofweek

                    if 'month' in time_features or 'month' in getattr(model, 'extra_regressors_names', []):
                        future['month'] = future['ds'].dt.month

                    if 'quarter' in time_features or 'quarter' in getattr(model, 'extra_regressors_names', []):
                        future['quarter'] = future['ds'].dt.quarter

                    if 'is_weekend' in time_features or 'is_weekend' in getattr(model, 'extra_regressors_names', []):
                        future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)

                    # Always make sure 'hour' is added since it's commonly used
                    if 'hour' not in future.columns and not getattr(model, 'is_dummy', False):
                        future['hour'] = future['ds'].dt.hour

                    # Make sure we handle the result DataFrame correctly
                    forecast = model.predict(future)
                    if isinstance(forecast, pd.DataFrame):
                        forecast_result = forecast.tail(horizon)
                        # Extract relevant columns and convert to Python lists for JSON serialization
                        result = {
                            "timestamps": forecast_result['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                            "predictions": forecast_result['yhat'].tolist(),
                            "lower_bound": forecast_result[
                                'yhat_lower'].tolist() if 'yhat_lower' in forecast_result.columns else None,
                            "upper_bound": forecast_result[
                                'yhat_upper'].tolist() if 'yhat_upper' in forecast_result.columns else None
                        }
                    else:
                        # Handle the case where predict returns something other than a DataFrame
                        now = datetime.now()
                        timestamps = [(now + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(horizon)]

                        if hasattr(forecast, 'tolist'):
                            predictions = forecast.tolist()
                        elif isinstance(forecast, list):
                            predictions = forecast
                        else:
                            predictions = [float(forecast)] * horizon

                        result = {
                            "timestamps": timestamps,
                            "predictions": predictions,
                            "lower_bound": None,
                            "upper_bound": None
                        }

                elif config['models']['selected_model'] == 'arima':
                    # Get forecast from ARIMA model
                    try:
                        # First try with return_conf_int=True for confidence intervals
                        forecast_mean, forecast_lower, forecast_upper = model.predict(periods=horizon,
                                                                                      return_conf_int=True)
                        has_conf_intervals = True
                    except (ValueError, TypeError):
                        try:
                            # If that fails, try without confidence intervals
                            forecast_result = model.predict(periods=horizon)
                            has_conf_intervals = False

                            # Handle different potential return types
                            if isinstance(forecast_result, pd.DataFrame):
                                forecast_mean = forecast_result.values
                            elif isinstance(forecast_result, pd.Series):
                                forecast_mean = forecast_result.values
                            else:
                                forecast_mean = forecast_result
                        except (ValueError, TypeError, AttributeError):
                            # If model.predict fails, try model.forecast
                            try:
                                forecast_result = model.forecast(steps=horizon)
                                has_conf_intervals = False

                                if isinstance(forecast_result, pd.DataFrame):
                                    forecast_mean = forecast_result.values
                                elif isinstance(forecast_result, pd.Series):
                                    forecast_mean = forecast_result.values
                                else:
                                    forecast_mean = forecast_result
                            except Exception as e:
                                logger.error(f"All forecast methods failed: {str(e)}")
                                raise

                    # Generate timestamp list
                    now = datetime.now()
                    timestamps = [(now + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(horizon)]

                    # Convert numpy arrays to Python lists for JSON serialization
                    if hasattr(forecast_mean, 'tolist'):
                        predictions = forecast_mean.tolist()
                    else:
                        predictions = list(forecast_mean)

                    # Prepare result with or without confidence intervals
                    if has_conf_intervals:
                        lower_bound = forecast_lower.tolist() if hasattr(forecast_lower, 'tolist') else list(
                            forecast_lower)
                        upper_bound = forecast_upper.tolist() if hasattr(forecast_upper, 'tolist') else list(
                            forecast_upper)

                        result = {
                            "timestamps": timestamps,
                            "predictions": predictions,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound
                        }
                    else:
                        result = {
                            "timestamps": timestamps,
                            "predictions": predictions,
                            "lower_bound": None,
                            "upper_bound": None
                        }

                elif config['models']['selected_model'] == 'lstm':
                    # LSTM-specific forecast
                    if isinstance(model, dict) and 'model' in model:
                        # This is an LSTM model dict with the actual model inside
                        lstm_model = model['model']
                        scaler = model['scaler']
                        latest_data = model['latest_data']

                        # Generate predictions for the horizon
                        predictions = []
                        current_sequence = latest_data.copy()

                        for _ in range(horizon):
                            # Reshape for LSTM input
                            current_input = current_sequence.reshape(1, current_sequence.shape[0], 1)
                            # Get prediction
                            next_pred = lstm_model.predict(current_input, verbose=0)
                            # Add to predictions
                            predictions.append(float(scaler.inverse_transform(next_pred)[0, 0]))
                            # Update sequence
                            current_sequence = np.roll(current_sequence, -1)
                            current_sequence[-1] = next_pred
                    else:
                        # Simple case: just call predict
                        predictions = model.predict(horizon)
                        if isinstance(predictions, (pd.DataFrame, pd.Series)):
                            predictions = predictions.values

                        if hasattr(predictions, 'tolist'):
                            predictions = predictions.tolist()
                        elif not isinstance(predictions, list):
                            predictions = list(predictions)

                    # Generate timestamp list
                    now = datetime.now()
                    timestamps = [(now + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(horizon)]

                    result = {
                        "timestamps": timestamps,
                        "predictions": predictions,
                        "lower_bound": None,
                        "upper_bound": None
                    }

                else:
                    # Generic dummy forecast for testing
                    now = datetime.now()
                    timestamps = [(now + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(horizon)]
                    predictions = [300 + 10 * np.sin(i / 12 * np.pi) + np.random.normal(0, 5) for i in range(horizon)]

                    result = {
                        "timestamps": timestamps,
                        "predictions": predictions,
                        "lower_bound": [p - 20 for p in predictions],
                        "upper_bound": [p + 20 for p in predictions]
                    }

                return jsonify({
                    "status": "success",
                    "forecast": result,
                    "model_type": config['models']['selected_model'],
                    "is_dummy": getattr(model, "is_dummy", False)
                })

            except Exception as e:
                logger.error(f"Error generating forecast: {str(e)}", exc_info=True)

                # If in debug mode, return a dummy forecast
                if config['api']['debug']:
                    logger.info("Generating dummy forecast due to error")

                    # Generate dummy forecast data
                    now = datetime.now()
                    timestamps = [(now + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(horizon)]
                    predictions = [300 + 10 * np.sin(i / 12 * np.pi) + np.random.normal(0, 5) for i in range(horizon)]

                    result = {
                        "timestamps": timestamps,
                        "predictions": predictions,
                        "lower_bound": [p - 20 for p in predictions],
                        "upper_bound": [p + 20 for p in predictions]
                    }

                    return jsonify({
                        "status": "warning",
                        "message": f"Error in model prediction, generated dummy data: {str(e)}",
                        "forecast": result,
                        "model_type": config['models']['selected_model'],
                        "is_dummy": True
                    })
                else:
                    return jsonify({"status": "error", "message": f"Error generating forecast: {str(e)}"}), 500

        except Exception as e:
            logger.error(f"Error in forecast endpoint: {str(e)}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/data/summary', methods=['GET'])
    def data_summary():
        """Endpoint to get a summary of the data"""
        try:
            # Check if processed data exists
            project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
            processed_data_path = os.path.join(project_root, config['data']['processed_data_path'])

            if not os.path.exists(processed_data_path):
                return jsonify({
                    "status": "error",
                    "message": "Processed data not found. Run preprocessing first."
                }), 404

            # Load data
            df = pd.read_csv(processed_data_path)

            # Convert date column to datetime
            date_col = config['data']['date_column']
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])

            # Generate summary statistics
            target_col = config['data']['target_column']

            summary = {
                "data_shape": list(df.shape),  # Convert tuple to list for JSON serialization
                "time_range": {
                    "start": df[date_col].min().strftime('%Y-%m-%d %H:%M:%S') if date_col in df.columns else None,
                    "end": df[date_col].max().strftime('%Y-%m-%d %H:%M:%S') if date_col in df.columns else None
                },
                "statistics": {
                    "min": float(df[target_col].min()),
                    "max": float(df[target_col].max()),
                    "mean": float(df[target_col].mean()),
                    "std": float(df[target_col].std()),
                    "median": float(df[target_col].median())
                },
                "recent_values": df.tail(24)[target_col].tolist() if len(df) >= 24 else df[target_col].tolist()
            }

            return jsonify({
                "status": "success",
                "summary": summary
            })

        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500