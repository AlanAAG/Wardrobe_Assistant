import asyncio
import logging
from typing import Dict, List, Tuple, Optional

from core.outfit_logic import build_outfit
from core.llm_agents import outfit_llm_agents

class OutfitPlannerAgent:
    """
    Generates an outfit using a hierarchical fallback: Gemini -> Groq -> Rule-based.
    """

    def __init__(self, api_retry_attempts: int = 3, api_retry_delay: float = 1.0):
        self._api_retry_attempts = api_retry_attempts
        self._api_retry_delay = api_retry_delay

    async def generate_outfit(
        self, wardrobe_items: List[Dict], weather_tag: str, desired_aesthetic: str
    ) -> Dict:
        """
        Generates an outfit using the hierarchical fallback chain.
        """
        context = {
            "available_items": self._categorize_items(wardrobe_items),
            "weather_condition": weather_tag,
            "desired_aesthetic": desired_aesthetic,
        }

        # 1. Try Gemini
        success, outfit, error_msg = await outfit_llm_agents.generate_outfit_with_gemini(context)
        if success and outfit:
            return {"success": True, "outfit": outfit, "generation_method": "gemini"}
        logging.warning(f"Gemini failed for outfit generation: {error_msg}")

        # 2. Try Groq
        success, outfit, error_msg = await outfit_llm_agents.generate_outfit_with_groq(context)
        if success and outfit:
            return {"success": True, "outfit": outfit, "generation_method": "groq"}
        logging.warning(f"Groq failed for outfit generation: {error_msg}")

        # 3. Fallback to rule-based logic
        logging.info("Falling back to rule-based outfit generation.")
        outfit = build_outfit(wardrobe_items, weather_tag == "hot", desired_aesthetic)
        if outfit:
            return {"success": True, "outfit": outfit, "generation_method": "rule_based_fallback"}

        return {"success": False, "error": "All methods failed to generate an outfit."}

    async def generate_example_outfits(
        self, selected_items: List[Dict], trip_config: Dict
    ) -> str:
        """
        Generates example outfits for a travel packing list.
        """
        # This will use an AI agent to generate the outfits.
        # For now, I'll just use a simple prompt with Gemini.
        context = {
            "available_items": self._categorize_items(selected_items),
            "trip_config": trip_config,
        }

        prompt = f"Given the following packed items, create 3 example outfits for different occasions during the trip. Items: {json.dumps(context['available_items'], indent=2)}"

        success, response, error_msg = await outfit_llm_agents.generate_outfit_with_gemini({"user_prompt": prompt, "available_items": context["available_items"], "weather_condition": "varied", "desired_aesthetic": "varied"})

        if success:
            return response
        else:
            logging.error(f"Failed to generate example outfits: {error_msg}")
            return "Could not generate example outfits."

    def _categorize_items(self, items: List[Dict]) -> Dict:
        """Categorizes a list of items by their 'category' property."""
        categorized = {}
        for item in items:
            category = item.get('category', 'Unknown')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)
        return categorized

outfit_planner_agent = OutfitPlannerAgent()
