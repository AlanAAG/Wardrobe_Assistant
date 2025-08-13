# render_server.py
import os
import sys
import logging
from pathlib import Path

# --- Setup Project Path ---
# This ensures all your project modules can be imported correctly
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# --- Application Entry Point ---
# We explicitly import the app object AFTER setting up the environment.
# This is the key to solving the startup issue.
try:
    from api.webhook_server import app
    
    # Configure logging for production
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logging.info("Server entry point initialized successfully.")

except ImportError as e:
    logging.critical(f"Failed to import Flask app: {e}")
    # If the app can't be imported, we create a dummy one to prevent Render from crashing on startup
    from flask import Flask, jsonify
    app = Flask(__name__)
    @app.route('/health')
    def health_check_error():
        return jsonify({"status": "error", "message": "Application failed to start. Check logs."}), 500