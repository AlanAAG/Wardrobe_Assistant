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
# Explicitly import the FastAPI app object
try:
    from services.webhook_server import app

    # Configure logging for production
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logging.info("FastAPI server entry point initialized successfully.")

except ImportError as e:
    logging.critical(f"Failed to import FastAPI app: {e}")
    # Fallback to a dummy FastAPI app to prevent Render from crashing
    try:
        from fastapi import FastAPI
        app = FastAPI()
        @app.get('/health')
        async def health_check_error():
            return {"status": "error", "message": "Application failed to start. Check logs."}
    except ImportError:
        pass