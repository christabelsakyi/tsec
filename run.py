import os
import sys
import yaml
import logging
import logging.config

# Ensure that the project root directory is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def load_config():
    """Load the application configuration."""
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Configuration file not found at {config_path}")
        sys.exit(1)

def setup_logging(config):
    """Configure logging based on the logging configuration file."""
    logging_config_path = os.path.join(project_root, config['logging']['config_path'])
    if os.path.exists(logging_config_path):
        with open(logging_config_path, 'r') as f:
            log_config = yaml.safe_load(f)
            logging.config.dictConfig(log_config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging config file not found at {logging_config_path}. Using default configuration.")

def check_directories(config):
    """Ensure that all required directories exist."""
    directories = config['directories']['required']
    for directory in directories:
        path = os.path.join(project_root, directory)
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info(f"Created directory: {path}")

def check_data_files(config):
    logger = logging.getLogger(__name__)
    raw_data_path = os.path.join(project_root, config['data']['raw_data_path'])
    processed_data_path = os.path.join(project_root, config['data']['processed_data_path'])

    if not os.path.exists(raw_data_path):
        logger.warning(f"Raw data file not found at {raw_data_path}")
        logger.info(config['data']['raw_data_message'])

    if not os.path.exists(processed_data_path):
        logger.warning(f"Processed data file not found at {processed_data_path}")
        logger.info(config['data']['process_data_message'])

def check_model_files(config):
    logger = logging.getLogger(__name__)
    model_path = os.path.join(project_root, config['models']['model_path'], f"{config['models']['selected_model']}_model.pkl")
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        if config['api']['debug']:
            logger.info(config['models']['debug_message'])
        else:
            logger.warning(config['models']['train_instruction'])

def create_dummy_data(config):
    logger = logging.getLogger(__name__)
    if not config['api']['debug']:
        return

    raw_data_path = os.path.join(project_root, config['data']['raw_data_path'])
    processed_data_path = os.path.join(project_root, config['data']['processed_data_path'])

    if not os.path.exists(raw_data_path) and not os.path.exists(processed_data_path):
        try:
            import pandas as pd
            import numpy as np

            logger.info(config['logging']['dummy_data_creation_message'])

            forecast_days = config['api']['max_forecast_days']
            freq = config['data']['resampling']['freq']

            end_date = pd.Timestamp.now().floor(freq)
            start_date = end_date - pd.Timedelta(days=forecast_days)
            date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
            date_series = pd.Series(date_range)

            df = pd.DataFrame({
                config['data']['date_column']: date_range,
                config['data']['target_column']: [
                    config['dummy_data']['base'] +
                    config['dummy_data']['daily_amplitude'] * np.sin(hour / 24 * 2 * np.pi) +
                    config['dummy_data']['weekly_amplitude'] * np.sin(day / 7 * 2 * np.pi) +
                    np.random.normal(0, config['dummy_data']['noise'])
                    for day, hour in zip(date_series.dt.dayofyear, date_series.dt.hour)
                ]
            })

            os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
            df.to_csv(str(processed_data_path), index=False)
            logger.info(f"Saved dummy data to {processed_data_path}")

        except ImportError:
            logger.warning("Could not create dummy data: pandas or numpy not installed")
        except Exception as e:
            logger.error(f"Error creating dummy data: {str(e)}")

def check_production_dependencies(config):
    logger = logging.getLogger(__name__)
    if not config['api']['debug']:
        for server in config['api']['production_servers']:
            try:
                __import__(server)
                logger.info(f"Using {server.capitalize()} as the production WSGI server")
                return server
            except ImportError:
                continue
        logger.warning("No production WSGI server found. Using Flask dev server.")
    return None

def main():
    config = load_config()
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info(config['logging']['start_message'])

    check_directories(config)
    check_data_files(config)
    check_model_files(config)
    create_dummy_data(config)

    # Important: Fix the import to use the correct path
    from src.api.app import app as flask_app
    logger.info(config['logging']['flask_start_message'].format(**config['api']))

    if config['api']['debug']:
        flask_app.run(host=config['api']['host'], port=config['api']['port'], debug=True)
    else:
        server_type = check_production_dependencies(config)
        if server_type == "waitress":
            from waitress import serve
            serve(flask_app, host=config['api']['host'], port=config['api']['port'])
        elif server_type == "gunicorn":
            import multiprocessing
            from gunicorn.app.base import BaseApplication

            class StandaloneApplication(BaseApplication):
                def __init__(self, flask_app_instance, gunicorn_options=None):
                    self.options = gunicorn_options or {}
                    self.application = flask_app_instance
                    super().__init__()

                def load_config(self):
                    for key, value in self.options.items():
                        self.cfg.set(key.lower(), value)

                def load(self):
                    return self.application

            server_options = {
                'bind': f"{config['api']['host']}:{config['api']['port']}",
                'workers': (multiprocessing.cpu_count() * 2) + 1,
            }
            StandaloneApplication(flask_app, server_options).run()
        else:
            logger.warning(config['logging']['fallback_message'])
            flask_app.run(host=config['api']['host'], port=config['api']['port'], debug=False)

if __name__ == "__main__":
    main()