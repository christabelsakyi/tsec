import os
import yaml
import sys
import logging.config
from flask import Flask
from flask_cors import CORS

# Add the project root to Python path (this needs to come before any other imports)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

# Now import the routes after adjusting the path
from .routes import register_routes

# Load configuration
with open(os.path.join(project_root, 'config', 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

# Setup logging
with open(os.path.join(project_root, 'config', 'logging_config.yaml'), 'r') as file:
    log_config = yaml.safe_load(file)

    # Create log directories if they don't exist
    for handler in log_config.get('handlers', {}).values():
        if 'filename' in handler:
            log_dir = os.path.dirname(handler['filename'])
            # Check if the path is relative and convert it to absolute if needed
            if not os.path.isabs(log_dir):
                log_dir = os.path.join(current_dir, log_dir)
            os.makedirs(log_dir, exist_ok=True)

    logging.config.dictConfig(log_config)

logger = logging.getLogger('api')

# Initialize Flask app
app = Flask(__name__)

# Setup CORS - Allow frontend origin
CORS(app, resources={r"/api/*": {"origins": config.get('frontend', {}).get('url', '*')}})

# Register routes
register_routes(app, config)

if __name__ == "__main__":
    app.run(
        host=config['api']['host'],
        port=config['api']['port'],
        debug=config['api']['debug']
    )