import os
import logging
import numpy as np
import joblib
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger('api')

def create_dummy_model():
    """Create a simple dummy model that mimics Prophet's interface for testing"""
    try:
        from prophet import Prophet

        # Create a basic Prophet model
        dummy_model = Prophet()

        # Add a simple implementation of predict
        def dummy_predict(future):
            """Simple dummy predict function"""
            # Check if future is an integer (horizon value)
            if isinstance(future, int):
                # Create a DataFrame with future dates
                horizon = future
                now = datetime.now()
                dates = [now + timedelta(hours=i) for i in range(horizon)]
                future_df = pd.DataFrame({
                    'ds': dates
                })
                # Create predictions
                result = future_df.copy()
                result['yhat'] = np.random.normal(300, 20, len(future_df))
                result['yhat_lower'] = result['yhat'] - 30
                result['yhat_upper'] = result['yhat'] + 30
                return result
            else:
                # Handle the case when future is already a DataFrame (original behavior)
                result = future.copy()
                result['yhat'] = np.random.normal(300, 20, len(future))
                result['yhat_lower'] = result['yhat'] - 30
                result['yhat_upper'] = result['yhat'] + 30
                return result

        # Attach the dummy predict function to the model
        dummy_model.real_predict = dummy_model.predict
        dummy_model.predict = dummy_predict
        dummy_model.is_dummy = True

        logger.info("Created dummy Prophet model for development")
        return dummy_model
    except ImportError:
        logger.error("Could not import Prophet to create dummy model")
        return None

def load_model(config):
    """Load the forecasting model or create a dummy model if needed"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    model_path = os.path.join(project_root, 'models', 'saved', f"{config['models']['selected_model']}_model.pkl")
    logger.info(f"Loading model from {model_path}")

    # Check if model file exists
    if not os.path.exists(model_path):
        logger.warning(f"Model file does not exist at {model_path}")

        # Return dummy model for development/testing
        if config['api']['debug']:
            logger.info("Creating dummy model for development/testing")
            return create_dummy_model()
        else:
            logger.error("Model file not found and not in debug mode")
            return None

    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

        # Return dummy model in debug mode
        if config['api']['debug']:
            logger.info("Creating dummy model after loading error")
            return create_dummy_model()
        return None