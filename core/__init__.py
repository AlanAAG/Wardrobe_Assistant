
"""
Core module with lazy loading to prevent circular imports.
This module provides access to core functionality without 
causing circular dependency issues.
"""

import logging

# Module-level variables for lazy loading
_outfit_logic = None
_pipeline_orchestrator = None
_travel_pipeline_orchestrator = None
_llm_agents = None

def get_build_outfit():
    """Get build_outfit function with lazy loading."""
    global _outfit_logic
    if _outfit_logic is None:
        try:
            from .outfit_logic import build_outfit
            _outfit_logic = build_outfit
        except ImportError as e:
            logging.error(f"Failed to import outfit_logic: {e}")
            _outfit_logic = None
    return _outfit_logic

def get_pipeline_orchestrator():
    """Get pipeline orchestrator with lazy loading."""
    global _pipeline_orchestrator
    if _pipeline_orchestrator is None:
        try:
            from .pipeline_orchestrator import run_enhanced_outfit_pipeline
            _pipeline_orchestrator = run_enhanced_outfit_pipeline
        except ImportError as e:
            logging.error(f"Failed to import pipeline orchestrator: {e}")
            _pipeline_orchestrator = None
    return _pipeline_orchestrator

def get_travel_pipeline_orchestrator():
    """Get travel pipeline orchestrator with lazy loading."""
    global _travel_pipeline_orchestrator
    if _travel_pipeline_orchestrator is None:
        try:
            from .travel_pipeline_orchestrator import travel_pipeline_orchestrator
            _travel_pipeline_orchestrator = travel_pipeline_orchestrator
        except ImportError as e:
            logging.error(f"Failed to import travel pipeline orchestrator: {e}")
            _travel_pipeline_orchestrator = None
    return _travel_pipeline_orchestrator

def get_llm_agents():
    """Get LLM agents with lazy loading."""
    global _llm_agents
    if _llm_agents is None:
        try:
            from .llm_agents import outfit_llm_agents
            _llm_agents = outfit_llm_agents
        except ImportError as e:
            logging.error(f"Failed to import LLM agents: {e}")
            _llm_agents = None
    return _llm_agents

# Export the getter functions
__all__ = [
    'get_build_outfit',
    'get_pipeline_orchestrator',
    'get_travel_pipeline_orchestrator',
    'get_llm_agents'
]