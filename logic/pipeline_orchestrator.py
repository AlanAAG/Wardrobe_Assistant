import json
import os
import logging
import asyncio
from dotenv import load_dotenv
from logic.weather_utils import get_weather_forecast
from logic.outfit_logic import build_outfit
from logic.llm_agents import outfit_llm_agents
from logic.notion_utils import (
    post_outfit_to_notion_page,
    clear_page_content,
    clear_trigger_fields,
)
from logic.data_manager import wardrobe_data_manager

load_dotenv()

async def run_enhanced_outfit_pipeline(trigger_data):
    """
    Enhanced async pipeline with LLM agents and hierarchical fallback:
    Gemini API -> Groq API -> Logic Engine -> Error
    
    Args:
        trigger_data (dict): Contains 'page_id', 'aesthetics', and 'prompt'
        
    Returns:
        dict: Result with success status and generation method used
    """
    try:
        page_id = trigger_data["page_id"]
        aesthetics = trigger_data["aesthetics"]
        user_prompt = trigger_data["prompt"]
        
        # Use the first aesthetic if multiple are selected
        desired_aesthetic = aesthetics[0] if aesthetics else "Minimalist"
        
        logging.info(f"Starting enhanced async outfit pipeline for page {page_id}")
        logging.info(f"Aesthetic: {desired_aesthetic}, Prompt: '{user_prompt}'")
        
        # Step 1: Get weather forecast (async wrapped)
        logging.info("Fetching weather forecast...")
        forecast = await asyncio.to_thread(get_weather_forecast)
        avg_temp = forecast["avg_temp"]
        condition = forecast["condition"]
        is_hot = forecast["weather_tag"] == "hot"
        weather_tag = "Hot" if is_hot else "Cold"
        logging.info(f"Weather: {avg_temp}°C, {condition}, Tag: {weather_tag}")
        
        # Step 2: Prepare LLM-optimized context (async wrapped)
        logging.info("Preparing LLM context...")
        try:
            llm_context = await asyncio.to_thread(
                wardrobe_data_manager.get_llm_optimized_context,
                desired_aesthetic, weather_tag
            )
            
            # Add user prompt to context
            llm_context["user_prompt"] = user_prompt
            
            total_items = sum(len(items) for items in llm_context["available_items"].values())
            logging.info(f"LLM context prepared: {total_items} items available")
            
            if total_items == 0:
                logging.warning("No suitable items found for LLM context")
                return {
                    "success": False,
                    "error": "No suitable items available for current conditions",
                    "page_id": page_id,
                    "generation_method": "no_items"
                }
            
        except Exception as e:
            logging.error(f"Failed to prepare LLM context: {e}")
            return {
                "success": False,
                "error": f"Context preparation failed: {str(e)}",
                "page_id": page_id,
                "generation_method": "context_failed"
            }
        
        # Step 3: Try Gemini API (Primary Agent) with timeout
        logging.info("🤖 Attempting outfit generation with Gemini API...")
        success, outfit_items, error_msg = await outfit_llm_agents.generate_outfit_with_gemini(
            llm_context, timeout=25
        )
        
        if success and outfit_items:
            logging.info("✅ Gemini API generated outfit successfully")
            return await _finalize_outfit(page_id, outfit_items, "gemini", {
                "weather": {"temp": avg_temp, "condition": condition, "tag": weather_tag},
                "aesthetic": desired_aesthetic,
                "user_prompt": user_prompt,
                "items_count": len(outfit_items)
            })
        else:
            logging.warning(f"Gemini failed: {error_msg}")
        
        # Step 4: Try Groq API (Secondary Agent) with timeout
        logging.info("🤖 Attempting outfit generation with Groq API...")
        success, outfit_items, error_msg = await outfit_llm_agents.generate_outfit_with_groq(
            llm_context, timeout=20
        )
        
        if success and outfit_items:
            logging.info("✅ Groq API generated outfit successfully")
            return await _finalize_outfit(page_id, outfit_items, "groq", {
                "weather": {"temp": avg_temp, "condition": condition, "tag": weather_tag},
                "aesthetic": desired_aesthetic,
                "user_prompt": user_prompt,
                "items_count": len(outfit_items)
            })
        else:
            logging.warning(f"Groq failed: {error_msg}")
        
        # Step 5: Fallback to Logic Engine (async wrapped)
        logging.info("🔧 Falling back to logic engine...")
        try:
            # Get all suitable items for logic engine
            wardrobe_items = await asyncio.to_thread(
                wardrobe_data_manager.get_filtered_wardrobe_items,
                aesthetic=desired_aesthetic,
                weather_tag=weather_tag,
                washed_only=True
            )
            
            if not wardrobe_items:
                # Final fallback: get all clean items
                wardrobe_items = await asyncio.to_thread(
                    wardrobe_data_manager.get_filtered_wardrobe_items,
                    washed_only=True
                )
            
            outfit_items = await asyncio.to_thread(
                build_outfit, wardrobe_items, is_hot, desired_aesthetic, "Done"
            )
            
            if outfit_items:
                logging.info("✅ Logic engine generated outfit successfully")
                return await _finalize_outfit(page_id, outfit_items, "logic", {
                    "weather": {"temp": avg_temp, "condition": condition, "tag": weather_tag},
                    "aesthetic": desired_aesthetic,
                    "user_prompt": user_prompt,
                    "items_count": len(outfit_items)
                })
            else:
                logging.error("Logic engine could not generate outfit")
        
        except Exception as e:
            logging.error(f"Logic engine error: {e}")
        
        # Step 6: All methods failed
        logging.error("❌ All outfit generation methods failed")
        return {
            "success": False,
            "error": "All generation methods failed - no suitable outfit could be created",
            "page_id": page_id,
            "generation_method": "all_failed",
            "attempted_methods": ["gemini", "groq", "logic"]
        }
        
    except Exception as e:
        logging.error(f"Critical error in enhanced pipeline: {e}")
        return {
            "success": False,
            "error": f"Pipeline error: {str(e)}",
            "page_id": trigger_data.get("page_id"),
            "generation_method": "pipeline_error"
        }

async def _finalize_outfit(page_id: str, outfit_items: list, generation_method: str, context: dict):
    """
    Async finalize outfit by posting to Notion and cleaning up
    
    Args:
        page_id: Notion page ID
        outfit_items: List of selected outfit items
        generation_method: Which method generated the outfit
        context: Additional context information
        
    Returns:
        dict: Success result with details
    """
    try:
        # Step 1: Clear previous content from the page (async wrapped)
        logging.info("Clearing previous content from Notion page...")
        await asyncio.to_thread(clear_page_content, page_id)
        
        # Step 2: Post new outfit to Notion (async wrapped)
        logging.info("Posting outfit to Notion...")
        await asyncio.to_thread(post_outfit_to_notion_page, page_id, outfit_items)
        
        # Step 3: Clear trigger fields to reset for next use (async wrapped)
        logging.info("Clearing trigger fields...")
        await asyncio.to_thread(clear_trigger_fields, page_id)
        
        logging.info(f"✅ Enhanced outfit pipeline completed successfully using {generation_method}!")
        
        return {
            "success": True,
            "page_id": page_id,
            "generation_method": generation_method,
            "outfit_items": context["items_count"],
            "aesthetic": context["aesthetic"],
            "user_prompt": context["user_prompt"],
            "weather": context["weather"],
            "item_names": [item.get("item", "Unknown") for item in outfit_items],
            "data_sources_used": await _get_data_source_info()
        }
        
    except Exception as e:
        logging.error(f"Error finalizing outfit: {e}")
        return {
            "success": False,
            "error": f"Failed to finalize outfit: {str(e)}",
            "page_id": page_id,
            "generation_method": f"{generation_method}_finalization_failed"
        }

async def _get_data_source_info() -> dict:
    """Async get information about which data sources are available/used"""
    try:
        return await asyncio.to_thread(wardrobe_data_manager.get_data_stats)
    except Exception:
        return {"error": "Could not retrieve data source stats"}

async def test_llm_agents(aesthetic: str = "Casual", user_prompt: str = "something comfortable for work"):
    """
    Async test function to verify LLM agents are working
    
    Args:
        aesthetic: Desired aesthetic style
        user_prompt: User's outfit request
    """
    logging.info(f"Testing LLM agents with aesthetic: {aesthetic}, prompt: '{user_prompt}'")
    
    try:
        # Get weather and prepare context
        forecast = await asyncio.to_thread(get_weather_forecast)
        weather_tag = "Hot" if forecast["weather_tag"] == "hot" else "Cold"
        
        llm_context = await asyncio.to_thread(
            wardrobe_data_manager.get_llm_optimized_context, aesthetic, weather_tag
        )
        llm_context["user_prompt"] = user_prompt
        
        total_items = sum(len(items) for items in llm_context["available_items"].values())
        print(f"Test context: {total_items} items, {weather_tag} weather, {aesthetic} aesthetic")
        
        # Test Gemini
        print("\n🤖 Testing Gemini...")
        success, outfit, error = await outfit_llm_agents.generate_outfit_with_gemini(llm_context)
        if success:
            print(f"✅ Gemini success: {[item['item'] for item in outfit]}")
        else:
            print(f"❌ Gemini failed: {error}")
        
        # Test Groq
        print("\n🤖 Testing Groq...")
        success, outfit, error = await outfit_llm_agents.generate_outfit_with_groq(llm_context)
        if success:
            print(f"✅ Groq success: {[item['item'] for item in outfit]}")
        else:
            print(f"❌ Groq failed: {error}")
            
    except Exception as e:
        print(f"Test error: {e}")

# Convenience function to run async test from sync context
def run_test_llm_agents(aesthetic: str = "Casual", user_prompt: str = "something comfortable for work"):
    """Sync wrapper to run async test"""
    return asyncio.run(test_llm_agents(aesthetic, user_prompt))

# Legacy compatibility wrapper - maintains backward compatibility
def run_outfit_pipeline(trigger_data):
    """
    Legacy wrapper for backward compatibility - runs async pipeline in sync context
    
    DEPRECATED: Use run_enhanced_outfit_pipeline() directly with asyncio.run() or await
    """
    logging.warning("Using legacy run_outfit_pipeline wrapper - consider updating to async")
    return asyncio.run(run_enhanced_outfit_pipeline(trigger_data))