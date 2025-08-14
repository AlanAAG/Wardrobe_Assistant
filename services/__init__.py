
"""
API module with lazy loading to prevent circular imports.
This module provides access to API components with proper error handling.
"""

import logging

# Module-level variables for lazy loading
_webhook_server = None

def get_flask_app():
    """Get Flask app with lazy loading."""
    global _webhook_server
    if _webhook_server is None:
        try:
            from .webhook_server import app
            _webhook_server = app
        except ImportError as e:
            logging.error(f"Failed to import Flask app: {e}")
            _webhook_server = None
    return _webhook_server

def get_workflow_functions():
    """Get workflow functions with lazy loading."""
    try:
        from .webhook_server import (
            determine_workflow_type, 
            handle_outfit_workflow, 
            handle_travel_workflow
        )
        return {
            'determine_workflow_type': determine_workflow_type,
            'handle_outfit_workflow': handle_outfit_workflow,
            'handle_travel_workflow': handle_travel_workflow
        }
    except ImportError as e:
        logging.error(f"Failed to import workflow functions: {e}")
        return {}

# Export the getter functions
__all__ = [
    'get_flask_app',
    'get_workflow_functions'
]