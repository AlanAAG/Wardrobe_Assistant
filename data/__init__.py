"""
Data layer module with lazy loading to prevent circular imports.
This module provides access to data management components without 
causing circular dependency issues.
"""

import logging

# Module-level variables to hold lazy-loaded instances
_notion_client = None
_supabase_client = None
_wardrobe_data_manager = None

def get_notion():
    """Get notion client with lazy loading."""
    global _notion_client
    if _notion_client is None:
        try:
            from .notion_utils import notion
            _notion_client = notion
        except ImportError as e:
            logging.error(f"Failed to import notion client: {e}")
            _notion_client = None
    return _notion_client

def get_supabase_client():
    """Get supabase client with lazy loading and error handling."""
    global _supabase_client
    if _supabase_client is None:
        try:
            from .supabase_client import supabase_client
            _supabase_client = supabase_client
        except ImportError as e:
            logging.warning(f"Supabase not available: {e}")
            _supabase_client = None
    return _supabase_client

def get_wardrobe_data_manager():
    """Get wardrobe data manager with lazy loading."""
    global _wardrobe_data_manager
    if _wardrobe_data_manager is None:
        try:
            from .data_manager import wardrobe_data_manager
            _wardrobe_data_manager = wardrobe_data_manager
        except ImportError as e:
            logging.error(f"Failed to import wardrobe data manager: {e}")
            _wardrobe_data_manager = None
    return _wardrobe_data_manager

# Legacy compatibility - maintain old interface but with lazy loading
@property
def notion():
    return get_notion()

@property  
def supabase_client():
    return get_supabase_client()

@property
def wardrobe_data_manager():
    return get_wardrobe_data_manager()

# Export the getter functions
__all__ = [
    'get_notion',
    'get_supabase_client', 
    'get_wardrobe_data_manager'
]