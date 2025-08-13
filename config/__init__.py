"""
Configuration module with lazy loading to prevent circular imports.
This module provides access to configuration data without causing dependency issues.
"""

import logging

# Module-level variables for lazy loading
_travel_config = None

def get_destinations_config():
    """Get destinations configuration with lazy loading."""
    global _travel_config
    if _travel_config is None:
        try:
            from .travel_config import DESTINATIONS_CONFIG
            _travel_config = DESTINATIONS_CONFIG
        except ImportError as e:
            logging.error(f"Failed to import travel config: {e}")
            _travel_config = {}
    return _travel_config

def get_weight_constraints():
    """Get weight constraints with lazy loading."""
    try:
        from .travel_config import WEIGHT_CONSTRAINTS
        return WEIGHT_CONSTRAINTS
    except ImportError as e:
        logging.error(f"Failed to import weight constraints: {e}")
        return {}

# Export the getter functions
__all__ = [
    'get_destinations_config',
    'get_weight_constraints'
]
