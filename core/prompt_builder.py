import logging
from typing import Dict, List, Optional
import asyncio
import json
from datetime import datetime
import logging

from core.utils import categorize_items_by_category

class PromptBuilder:
    """Builds prompts for the TravelPackingAgent."""

    def __init__(self, average_weights: Dict, gemini_model):
        self.weights = average_weights
        self.gemini_model = gemini_model

    def build_dynamic_service_prompt(self, context: Dict) -> str:
        """
        Builds a fully dynamic, AI-driven prompt that instructs the model on how to analyze
        raw user input, including dynamic bag limits, to generate a packing list.
        """
        bags_string = ", ".join(context.get("raw_bags", ["Not specified"]))
        prompt = f"""You are an expert AI travel packing consultant. Your task is to analyze a user's travel plan and wardrobe to create the most weight-efficient packing list possible.

    **USER'S TRAVEL PLAN FROM NOTION**
    * **Destinations & Dates**: "{context['raw_destinations_and_dates']}"
    * **Purpose & Preferences**: "{context['raw_preferences_and_purpose']}"
    * **Luggage Allowance**: {bags_string}

    **YOUR ANALYSIS PROCESS (Follow these steps):**
    1.  **Calculate Clothing Weight Budget**: First, calculate the total luggage weight allowance from the user's input (e.g., "Checked Bag: 23kg, Cabin Bag: 10kg" = 33kg total). Then, intelligently estimate a realistic portion of this total weight that should be allocated for clothes, reserving the rest for essentials like electronics, toiletries, and shoes. This will be your **Clothing Weight Budget**.
    2.  **Analyze Itinerary**: Parse the user's input to identify the destinations, dates, and purpose of the trip.
    3.  **Determine Climate & Culture**: For each destination and its timeline, use your knowledge to determine the expected climate and cultural dress norms.
    4.  **Prioritize Business Requirements**: Since this is business school travel, ensure you include:
       - At least 1 suit for formal events
       - Formal shoes (dress shoes/loafers)
       - Business shirts or polos with formal aesthetic
       - Professional casual items for daily wear
    5.  **Focus on Versatility**: Select items that can be mixed and matched across multiple occasions and weather conditions.
    6.  **Synthesize a Plan**: Based on all of the above, formulate a packing strategy that respects the **Clothing Weight Budget** you calculated in step 1.

    **AVAILABLE WARDROBE (SELECT ONLY FROM THIS LIST)**
    {self._format_items_with_intelligence(context["available_items"], context)}

    **CRITICAL OUTPUT INSTRUCTIONS**
    Your entire response must be ONLY a list of the selected items under the heading "SELECTED_ITEMS:". Each item must be on a new line.

    **QUALITY CHECKS** (Ensure your selection includes):
    ✓ At least 1 suit (required for business events)
    ✓ At least 1 pair of formal shoes
    ✓ 3-5 business appropriate shirts/polos
    ✓ 5-8 casual tops for variety
    ✓ 3-5 bottoms that mix and match
    ✓ Items suitable for the expected climate
    ✓ Total weight under your calculated clothing budget

    **YOUR RESPONSE:**
    SELECTED_ITEMS:
    """
        return prompt

    def build_groq_service_prompt(self, context: Dict) -> str:
        """
        Builds a definitive, highly concise, Groq-optimized service prompt that
        leverages the model's speed and efficiency.
        """
        prompt = f"""**TASK**: Analyze the user's travel plan and create an optimized packing list.

    **USER INPUT**:
    * **Destinations & Dates**: "{context['raw_destinations_and_dates']}"
    * **Purpose & Preferences**: "{context['raw_preferences_and_purpose']}"
    * **Luggage**: {", ".join(context.get('bags', ["Not specified"]))}

    **ANALYSIS & SELECTION PROCESS**:
    1.  **Calculate Clothing Budget**: Estimate a realistic weight for clothes from the user's luggage allowance.
    2.  **Analyze Trip**: Parse the user's input to determine climate, cultural norms, and key activities.
    3.  **Select Items**: Choose the most versatile and weight-efficient items from the available wardrobe that meet all trip requirements and respect the calculated weight budget.

    **AVAILABLE WARDROBE (SELECT ONLY FROM THIS LIST):**
    {self._format_items_with_intelligence(context["available_items"], context)}

    **OUTPUT INSTRUCTIONS**:
    Your response must ONLY be a list of the exact item names under the heading "SELECTED_ITEMS:", with each item on a new line.

    **YOUR RESPONSE:**
    SELECTED_ITEMS:
    """
        return prompt

    def _format_items_with_intelligence(self, available_items: Dict, context: Dict) -> str:
        """
        Formats available items with a balance of essential detail and conciseness
        to ensure high-quality AI responses without timeouts.
        """
        formatted = ""
        for category, items in available_items.items():
            if not items:
                continue

            formatted += f"\n**{category.upper()} ({len(items)} items):**\n"

            item_lines = [f"- {item['item']} (Aesthetics: {', '.join(item.get('aesthetic', ['N/A']))})" for item in items]

            formatted += "\n".join(item_lines)
            formatted += "\n"

        return formatted

    def get_groq_system_prompt(self) -> str:
        """System prompt for Groq"""
        return """You are an expert travel packing consultant specializing in long-term business relocations. You optimize for weight efficiency, cultural appropriateness, climate adaptation, and professional requirements. You provide precise, actionable packing recommendations with detailed reasoning."""

    async def _get_web_search_context(self, city: str, trip_config: Dict) -> str:
        """
        Performs web searches to gather real-time context for a given destination.
        This method is a placeholder for where web search tools would be integrated.
        """
        # In a real implementation, this method would use tools like:
        # from your_tools import google_search, view_text_website

        logging.info(f"Web search for {city} is a placeholder. No live search performed.")
        # Returning an empty string to signify that no web context is available.
        # The prompt is designed to work even with an empty context string.
        return ""

    def build_destination_tip_prompt(self, city: str, trip_config: Dict, web_context: str) -> str:
        """
        Builds a detailed, dynamic prompt for Gemini to generate comprehensive travel tips.
        """
        # Extract trip context for personalization
        purpose = trip_config.get("raw_preferences_and_purpose", "a business school trip")

        # Get trip dates for seasonal context
        start_date_str = trip_config.get("dates", {}).get("start")
        end_date_str = trip_config.get("dates", {}).get("end")

        if start_date_str and end_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str[:10])
                end_date = datetime.fromisoformat(end_date_str[:10])
                duration = (end_date - start_date).days
                season = f"from {start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}"
            except (ValueError, TypeError):
                season = "an upcoming trip"
        else:
            season = "an upcoming trip"

        # Dynamically build the web context section only if context is available
        web_context_section = ""
        if web_context and web_context.strip():
            web_context_section = f"""
**WEB SEARCH CONTEXT:**
Here is some information from recent web searches about {city}. Use this to ensure your tips are current and accurate.
---
{web_context}
---
"""

        prompt = f"""
You are an expert travel content creator and destination specialist. Your task is to generate a rich, personalized, and genuinely helpful travel guide for a traveler visiting **{city}**.

**TRAVELER CONTEXT:**
* **Destination:** {city}
* **Purpose of Trip:** {purpose}
* **Travel Period:** {season}
{web_context_section}
**INSTRUCTIONS:**
Based on your extensive knowledge and the provided web search context (if any), create a comprehensive set of travel tips. The tips should be practical, encouraging, and culturally sensitive. Your entire response MUST be a valid JSON object.

**JSON OUTPUT STRUCTURE:**
Please structure your response in the following JSON format. Each category should contain a list of concise, actionable tips as strings.

{{
  "cultural_intelligence": [
    "Tip 1...",
    "Tip 2..."
  ],
  "climate_and_weather": [
    "Tip 1...",
    "Tip 2..."
  ],
  "transportation_and_navigation": [
    "Tip 1...",
    "Tip 2..."
  ],
  "food_and_dining": [
    "Tip 1...",
    "Tip 2..."
  ],
  "business_traveler_specifics": [
    "Tip 1...",
    "Tip 2..."
  ],
  "safety_and_health": [
    "Tip 1...",
    "Tip 2..."
  ],
  "local_life_and_hidden_gems": [
    "Tip 1...",
    "Tip 2..."
  ]
}}

**CONTENT REQUIREMENTS:**
1.  **Cultural Intelligence:** Cover local customs, greetings, tipping practices, and dress codes for religious or cultural sites.
2.  **Climate & Weather:** Provide practical advice on what to wear, best times for outdoor activities, and seasonal considerations for {season}.
3.  **Transportation:** Recommend the best apps and methods for getting around, comment on traffic, and explain public transport etiquette.
4.  **Food & Dining:** Suggest must-try local specialties, advise on street food safety, and explain meal timing customs.
5.  **Business Traveler Specifics:** Include tips on business etiquette, networking spots, co-working spaces, and professional dress norms.
6.  **Safety & Health:** Offer practical safety awareness, health precautions, and emergency contact info without being alarmist.
7.  **Local Life & Hidden Gems:** Share authentic experiences, unique shopping areas, and useful local phrases.

Ensure the tone is enthusiastic and the information is current and accurate.
"""
        return prompt

    async def generate_destination_tips(self, trip_config: Dict) -> Dict:
        """
        Generates rich, dynamic, and actionable destination-specific tips using AI and web search.
        This method replaces the static, template-based approach.
        """
        tips: Dict[str, Dict] = {}

        # Determine destination city list from trip_config
        cities: List[str] = []
        if isinstance(trip_config.get("destinations"), list) and trip_config["destinations"] and isinstance(trip_config["destinations"][0], dict):
            cities = [str(d.get("city", "")).title() for d in trip_config["destinations"] if d.get("city")]
        else:
            raw_dests = trip_config.get("raw_destinations_and_dates", [])
            if isinstance(raw_dests, list):
                cities = [str(x).title() for x in raw_dests]
            else:
                # Basic parsing for comma-separated city names
                cities = [c.strip().title() for c in str(raw_dests).split(',') if c.strip()]

        if not cities:
            logging.warning("No destination cities found in trip_config for tip generation.")
            return {}

        for city in cities:
            try:
                logging.info(f"Generating dynamic tips for {city}...")

                # Step 1: Gather real-time context from the web (placeholder)
                web_context = await self._get_web_search_context(city, trip_config)

                # Step 2: Build the prompt with the web context
                prompt = self.build_destination_tip_prompt(city, trip_config, web_context)

                if not self.gemini_model:
                    logging.error("Gemini model not available for tip generation.")
                    tips[city.lower()] = {"error": "AI model not configured."}
                    continue

                # Step 3: Generate content with the AI
                response = await asyncio.to_thread(
                    self.gemini_model.generate_content,
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )

                # Step 4: Sanitize and parse the JSON response
                response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                tip_data = json.loads(response_text)

                tips[city.lower()] = tip_data
                logging.info(f"Successfully generated and parsed tips for {city}.")

            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON from Gemini response for {city}. Response text: {response.text[:500]}")
                tips[city.lower()] = {"error": "Failed to parse AI response."}
            except Exception as e:
                logging.error(f"An unexpected error occurred while generating tips for {city}: {e}", exc_info=True)
                tips[city.lower()] = {"error": f"An unexpected error occurred: {e}"}

        return tips

    def build_example_outfits_prompt(self, selected_items: List[Dict], trip_config: Dict) -> str:
        """Builds a prompt to generate three example outfits from the selected items."""

        trip_overview = trip_config.get("trip_overview", {})
        destinations = ", ".join([d.get('city', '').title() for d in trip_config.get("destinations", [])])

        prompt = f"""You are a fashion stylist creating example outfits from a pre-selected travel wardrobe.

**CONTEXT**
* **Trip**: A {trip_overview.get('total_duration_months', 'long')} month business school trip to {destinations}.
* **Goal**: Create three distinct, stylish, and practical example outfits using ONLY the clothes provided below.

**AVAILABLE ITEMS FOR OUTFITS**
{self._format_items_with_intelligence(categorize_items_by_category(selected_items), {})}

**INSTRUCTIONS**
1.  Create exactly three outfits: one for a business formal event, one for a business casual school day, and one for a casual weekend outing.
2.  For each outfit, list the specific items used (top, bottom, footwear, and outerwear if appropriate).
3.  Provide a brief, one-sentence recommendation or style tip for each outfit.
4.  Your response must be ONLY the three outfits, formatted exactly like the example below.

**EXAMPLE FORMAT:**
OUTFIT 1: Business Formal
* Items: [Item Name], [Item Name], [Item Name]
* Recommendation: A classic and professional look perfect for networking events.

OUTFIT 2: Business Casual
* Items: [Item Name], [Item Name], [Item Name]
* Recommendation: This versatile outfit is comfortable for classes and stylish enough for after-school study groups.

OUTFIT 3: Weekend Exploration
* Items: [Item Name], [Item Name], [Item Name]
* Recommendation: A relaxed and cool outfit for exploring the city on a warm day.

**YOUR RESPONSE:**
"""
        return prompt

    async def generate_example_outfits(self, selected_items: List[Dict], trip_config: Dict, timeout: int = 45) -> Optional[str]:
        """Generates three example outfits using Gemini."""
        if not self.gemini_model:
            logging.warning("Gemini model not available for generating example outfits.")
            return None

        try:
            prompt = self.build_example_outfits_prompt(selected_items, trip_config)
            response = await asyncio.wait_for(
                asyncio.to_thread(self.gemini_model.generate_content, prompt),
                timeout=timeout
            )
            return response.text
        except Exception as e:
            logging.error(f"Failed to generate example outfits: {e}")
            return None
