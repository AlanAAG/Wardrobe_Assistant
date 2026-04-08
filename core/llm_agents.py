import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from core.outfit_logic import UPPER_BODY, LOWER_BODY_HOT, FOOTWEAR
import google.generativeai as genai
from groq import Groq

load_dotenv()

class OutfitLLMAgents:
    """
    Async LLM agents for outfit creation with hierarchical fallback:
    Gemini API -> Groq API -> Logic Engine -> Error
    Using Structured Outputs (JSON Schema).
    """
    
    def __init__(self):
        # Initialize Gemini
        self.gemini_api_key = os.getenv("GEMINI_AI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            logging.warning("GEMINI_AI_API_KEY not found")
            self.gemini_model = None
        
        # Initialize Groq
        self.groq_api_key = os.getenv("GROQ_AI_API_KEY")
        if self.groq_api_key:
            self.groq_client = Groq(api_key=self.groq_api_key)
        else:
            logging.warning("GROQ_AI_API_KEY not found")
            self.groq_client = None
    
    async def generate_outfit_with_gemini(self, context: Dict, timeout: int = 25) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """
        Primary agent: Generate outfit using Gemini API with timeout using JSON schema.
        """
        if not self.gemini_model:
            return False, None, "Gemini API not configured"
        
        try:
            service_prompt = self._build_service_prompt(context)
            
            # Using JSON response format for Gemini
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json",
            )
            
            response = await asyncio.wait_for(
                asyncio.to_thread(self.gemini_model.generate_content, service_prompt, generation_config=generation_config),
                timeout=timeout
            )
            
            if not response.text:
                return False, None, "Gemini returned empty response"
            
            outfit_items = self._parse_json_response(response.text, context["available_items"])
            
            if not outfit_items:
                return False, None, "Gemini could not generate valid outfit from JSON string"
            
            logging.info(f"Gemini successfully generated outfit with {len(outfit_items)} items")
            return True, outfit_items, None
            
        except asyncio.TimeoutError:
            error_msg = f"Gemini API timeout after {timeout} seconds"
            logging.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            logging.error(error_msg)
            return False, None, error_msg
    
    async def generate_outfit_with_groq(self, context: Dict, timeout: int = 20) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """
        Secondary agent: Generate outfit using Groq API with timeout using JSON schema.
        """
        if not self.groq_client:
            return False, None, "Groq API not configured"
        
        try:
            service_prompt = self._build_service_prompt(context)
            
            chat_completion = await asyncio.wait_for(
                asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": service_prompt}
                    ],
                    model="llama3-8b-8192",
                    temperature=0.3,
                    max_tokens=1000,
                    top_p=0.9,
                    response_format={"type": "json_object"}
                ),
                timeout=timeout
            )
            
            response_text = chat_completion.choices[0].message.content
            
            if not response_text:
                return False, None, "Groq returned empty response"
            
            outfit_items = self._parse_json_response(response_text, context["available_items"])
            
            if not outfit_items:
                return False, None, "Groq could not generate valid outfit from JSON string"
            
            logging.info(f"Groq successfully generated outfit with {len(outfit_items)} items")
            return True, outfit_items, None
            
        except asyncio.TimeoutError:
            error_msg = f"Groq API timeout after {timeout} seconds"
            logging.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Groq API error: {str(e)}"
            logging.error(error_msg)
            return False, None, error_msg

    def _get_system_prompt(self) -> str:
        """System prompt for Groq/Gemini chat completion"""
        return """You are a professional fashion stylist AI specialized in creating weather-appropriate, aesthetically coherent outfits. You analyze user requests, weather conditions, and available wardrobe items to create perfect outfit combinations.

Key principles:
- Weather appropriateness is non-negotiable
- Color coordination is essential
- Aesthetic consistency must be maintained
- All selections must come from provided available items only
- Provide clear, actionable recommendations
- You MUST return a pure JSON object mapping strictly to the item IDs provided."""
    
    def _build_service_prompt(self, context: Dict) -> str:
        """Build specialized service prompt expecting JSON"""
        weather_condition = context["weather_condition"]
        desired_aesthetic = context["desired_aesthetic"]
        user_prompt = context.get("user_prompt", "")
        available_items = context["available_items"]
        
        total_items = sum(len(items) for items in available_items.values())
        
        prompt = f"""You are a professional fashion stylist AI specialized in creating weather-appropriate, aesthetically coherent outfits.

**REQUIREMENTS:**
- Weather: {weather_condition}
- Style: {desired_aesthetic}
- Request: "{user_prompt}"
- Must be color coordinated and weather appropriate

**AVAILABLE ITEMS BY ID ({total_items} total):**

**TOPS:**
{self._format_items_for_prompt(available_items.get('tops', []))}

**BOTTOMS:**
{self._format_items_for_prompt(available_items.get('bottoms', []))}

**OUTERWEAR:**
{self._format_items_for_prompt(available_items.get('outerwear', []))}

**FOOTWEAR:**
{self._format_items_for_prompt(available_items.get('footwear', []))}

**OUTPUT FORMAT (CRITICAL - Follow exactly):**
You MUST return a raw JSON object string of the following schema, and NOTHING ELSE. Choose exactly one item ID from each required category (outerwear only if cold).

{{
  "selected_ids": [
    "exact-id-from-tops",
    "exact-id-from-bottoms",
    "exact-id-from-footwear",
    "exact-id-from-outerwear-if-needed"
  ],
  "reasoning": "Brief explanation of choices."
}}"""
        return prompt
    
    def _format_items_for_prompt(self, items: List[Dict]) -> str:
        """Format wardrobe items with their IDs for LLM prompt"""
        if not items:
            return "None available"
        
        formatted = []
        for item in items:
            # Assume 'id' exists. If not, use 'item' as ID as fallback, but typically DB items have an ID.
            item_id = item.get('id', item.get('item', 'unknown_id'))
            colors = ", ".join(item.get('color', []))
            aesthetics = ", ".join(item.get('aesthetic', []))
            weather_tags = ", ".join(item.get('weather', []))
            
            item_info = f"- ID: '{item_id}' | Name: {item['item']} ({item['category']})"
            if colors:
                item_info += f" | Colors: {colors}"
            if aesthetics:
                item_info += f" | Aesthetics: {aesthetics}"
            if weather_tags:
                item_info += f" | Weather: {weather_tags}"
            
            formatted.append(item_info)
        
        return "\n".join(formatted)

    def _parse_json_response(self, response_text: str, available_items: Dict) -> Optional[List[Dict]]:
        """
        Parse LLM JSON response to extract selected outfit items based entirely on IDs.
        No regex or fuzzy matching needed.
        """
        try:
            # Clean up potential markdown blocks if LLM still returned them
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()

            parsed = json.loads(response_text)
            selected_ids = parsed.get("selected_ids", [])
            
            if not selected_ids:
                logging.error("No selected_ids found in JSON response")
                return None
            
            # Map IDs back to full item objects
            all_items_lookup = {}
            for category, items in available_items.items():
                for item in items:
                    item_id = item.get('id', item.get('item', 'unknown_id'))
                    all_items_lookup[item_id] = item

            selected_items = []
            for item_id in selected_ids:
                if item_id in all_items_lookup:
                    selected_items.append(all_items_lookup[item_id])
                else:
                    logging.warning(f"Returned ID {item_id} not found in available items.")
            
            # Validate outfit completeness
            if not self._validate_outfit_completeness(selected_items):
                return None
            
            return selected_items if selected_items else None
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from LLM: {e}\nResponse: {response_text}")
            return None
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            return None

    def _validate_outfit_completeness(self, selected_items: List[Dict]) -> bool:
        """
        Validate that the outfit has the required pieces
        """
        categories = {item['category'] for item in selected_items}
        
        has_top = bool(categories.intersection(UPPER_BODY))
        has_bottom = bool(categories.intersection(LOWER_BODY_HOT))
        has_footwear = bool(categories.intersection(FOOTWEAR))
        
        if not (has_top and has_bottom and has_footwear):
            logging.warning(f"Incomplete outfit: top={has_top}, bottom={has_bottom}, footwear={has_footwear}")
            logging.warning(f"Selected categories: {categories}")
            return False
        
        return True

outfit_llm_agents = OutfitLLMAgents()