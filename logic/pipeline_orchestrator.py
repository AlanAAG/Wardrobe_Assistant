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

load_dotenv()

CACHE_FILE = "wardrobe_cache.json"

def load_wardrobe_cache():
    """Load wardrobe items from cache file."""
    cache_path = os.path.join(os.path.dirname(__file__), "..", CACHE_FILE)
    if not os.path.exists(cache_path):
        raise FileNotFoundError("Wardrobe cache not found. Please run sync_notion_to_cache.py first.")
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("wardrobe", [])

def run_outfit_pipeline(trigger_data):
    """
    Main pipeline function that generates and posts an outfit based on webhook trigger data.
    
    Args:
        trigger_data (dict): Contains 'page_id', 'aesthetics', and 'prompt'
        
    Returns:
        dict: Result with success status and any relevant info
    """
    try:
        page_id = trigger_data["page_id"]
        aesthetics = trigger_data["aesthetics"]
        prompt = trigger_data["prompt"]
        
        # Use the first aesthetic if multiple are selected
        desired_aesthetic = aesthetics[0] if aesthetics else "Minimalist"
        
        logging.info(f"Starting outfit pipeline for page {page_id}")
        logging.info(f"Using aesthetic: {desired_aesthetic}, prompt: '{prompt}'")
        
        # Step 1: Load wardrobe from cache
        logging.info("Loading wardrobe from cache...")
        wardrobe_items = load_wardrobe_cache()
        logging.info(f"Loaded {len(wardrobe_items)} wardrobe items from cache")
        
        # Step 2: Get weather forecast
        logging.info("Fetching weather forecast...")
        forecast = get_weather_forecast()
        avg_temp = forecast["avg_temp"]
        condition = forecast["condition"]
        is_hot = forecast["weather_tag"] == "hot"
        logging.info(f"Weather: {avg_temp}°C, {condition}, Hot: {is_hot}")
        
        # Step 3: Build the outfit
        logging.info("Building outfit...")
        outfit = build_outfit(wardrobe_items, is_hot, desired_aesthetic, washed_required="Done")
        
        if not outfit:
            logging.warning("Could not build an outfit with the available items")
            return {
                "success": False,
                "error": "No suitable outfit could be generated with available items",
                "page_id": page_id
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
        
        logging.info("✅ Outfit pipeline completed successfully!")
        
        return {
            "success": True,
            "page_id": page_id,
            "outfit_items": len(outfit),
            "aesthetic": desired_aesthetic,
            "weather": {
                "temp": avg_temp,
                "condition": condition,
                "is_hot": is_hot
            }
        }
        
    except FileNotFoundError as e:
        logging.error(f"Cache file not found: {e}")
        return {
            "success": False,
            "error": "Wardrobe cache not found - please sync from Notion first",
            "page_id": trigger_data.get("page_id")
        }
        
    except Exception as e:
        logging.error(f"Error in outfit pipeline: {e}")
        return {
            "success": False,
            "error": str(e),
            "page_id": trigger_data.get("page_id")
        }