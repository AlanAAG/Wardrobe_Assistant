import json
import os
import logging
from dotenv import load_dotenv
from logic.weather_utils import get_weather_forecast
from logic.outfit_logic import build_outfit
from logic.notion_utils import (
    post_outfit_to_notion_page,
    clear_page_content,
    clear_trigger_fields,
)
from logic.data_manager import wardrobe_data_manager

load_dotenv()

def run_outfit_pipeline(trigger_data):
    """
    Enhanced pipeline with hierarchical data access (Supabase -> Cache -> Notion -> Error).
    Ready for LLM integration.
    
    Args:
        trigger_data (dict): Contains 'page_id', 'aesthetics', and 'prompt'
        
    Returns:
        dict: Result with success status and relevant info
    """
    try:
        page_id = trigger_data["page_id"]
        aesthetics = trigger_data["aesthetics"]
        prompt = trigger_data["prompt"]
        
        # Use the first aesthetic if multiple are selected
        desired_aesthetic = aesthetics[0] if aesthetics else "Minimalist"
        
        logging.info(f"Starting enhanced outfit pipeline for page {page_id}")
        logging.info(f"Using aesthetic: {desired_aesthetic}, prompt: '{prompt}'")
        
        # Step 1: Get weather forecast
        logging.info("Fetching weather forecast...")
        forecast = get_weather_forecast()
        avg_temp = forecast["avg_temp"]
        condition = forecast["condition"]
        is_hot = forecast["weather_tag"] == "hot"
        weather_tag = "Hot" if is_hot else "Cold"
        logging.info(f"Weather: {avg_temp}°C, {condition}, Tag: {weather_tag}")
        
        # Step 2: Get wardrobe data using hierarchical data manager
        logging.info("Loading wardrobe data...")
        try:
            # Get optimized data for the current conditions
            wardrobe_items = wardrobe_data_manager.get_filtered_wardrobe_items(
                aesthetic=desired_aesthetic,
                weather_tag=weather_tag,
                washed_only=True
            )
            
            if not wardrobe_items:
                # Fallback to all clean items if filtered set is empty
                logging.warning("No items match filters, falling back to all clean items")
                wardrobe_items = wardrobe_data_manager.get_filtered_wardrobe_items(
                    washed_only=True
                )
            
            logging.info(f"Retrieved {len(wardrobe_items)} suitable wardrobe items")
            
        except Exception as e:
            logging.error(f"Failed to load wardrobe data: {e}")
            return {
                "success": False,
                "error": f"Data loading failed: {str(e)}",
                "page_id": page_id
            }
        
        # Step 3: Build the outfit using existing logic
        logging.info("Building outfit with logic engine...")
        outfit = build_outfit(wardrobe_items, is_hot, desired_aesthetic, washed_required="Done")
        
        if not outfit:
            logging.warning("Logic engine could not build an outfit")
            return {
                "success": False,
                "error": "No suitable outfit could be generated with available items",
                "page_id": page_id,
                "generation_method": "logic_failed"
            }
        
        logging.info(f"Built outfit with {len(outfit)} items: {[item['item'] for item in outfit]}")
        
        # Step 4: Clear previous content from the page
        logging.info("Clearing previous content from Notion page...")
        clear_page_content(page_id)
        
        # Step 5: Post new outfit to Notion
        logging.info("Posting outfit to Notion...")
        post_outfit_to_notion_page(page_id, outfit)
        
        # Step 6: Clear trigger fields to reset for next use
        logging.info("Clearing trigger fields...")
        clear_trigger_fields(page_id)
        
        logging.info("✅ Enhanced outfit pipeline completed successfully!")
        
        return {
            "success": True,
            "page_id": page_id,
            "outfit_items": len(outfit),
            "aesthetic": desired_aesthetic,
            "weather": {
                "temp": avg_temp,
                "condition": condition,
                "is_hot": is_hot,
                "tag": weather_tag
            },
            "generation_method": "logic",
            "data_sources_used": _get_data_source_info()
        }
        
    except Exception as e:
        logging.error(f"Error in enhanced outfit pipeline: {e}")
        return {
            "success": False,
            "error": str(e),
            "page_id": trigger_data.get("page_id"),
            "data_sources_used": _get_data_source_info()
        }

def prepare_llm_context(aesthetic: str, weather_tag: str) -> dict:
    """
    Prepare optimized context for LLM agents.
    This function will be used by future LLM integrations.
    
    Args:
        aesthetic: Desired aesthetic style
        weather_tag: Weather condition
        
    Returns:
        Dictionary with organized wardrobe data for LLM consumption
    """
    try:
        logging.info(f"Preparing LLM context for aesthetic: {aesthetic}, weather: {weather_tag}")
        
        context = wardrobe_data_manager.get_llm_optimized_context(aesthetic, weather_tag)
        
        # Add metadata for LLM prompt engineering
        context["metadata"] = {
            "total_available_items": sum(len(items) for items in context["available_items"].values()),
            "filtering_applied": True,
            "data_source": "hierarchical_fallback"
        }
        
        logging.info(f"LLM context prepared: {context['metadata']['total_available_items']} items")
        return context
        
    except Exception as e:
        logging.error(f"Failed to prepare LLM context: {e}")
        raise Exception(f"LLM context preparation failed: {str(e)}")

def _get_data_source_info() -> dict:
    """Get information about which data sources are available/used"""
    try:
        return wardrobe_data_manager.get_data_stats()
    except Exception:
        return {"error": "Could not retrieve data source stats"}

# Legacy function for backward compatibility
def load_wardrobe_cache():
    """
    Legacy function for backward compatibility.
    Now uses the hierarchical data manager.
    """
    logging.warning("Using legacy load_wardrobe_cache() - consider updating to use data_manager directly")
    return wardrobe_data_manager.get_all_wardrobe_items()