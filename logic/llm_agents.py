import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq

load_dotenv()

class OutfitLLMAgents:
    """
    Async LLM agents for outfit creation with hierarchical fallback:
    Gemini API -> Groq API -> Logic Engine -> Error
    """
    
    def __init__(self):
        # Initialize Gemini
        self.gemini_api_key = os.getenv("GEMINI_AI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
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
        Primary agent: Generate outfit using Gemini API with timeout
        
        Args:
            context: Dictionary with wardrobe items, weather, aesthetic, and user prompt
            timeout: Timeout in seconds for the API call
            
        Returns:
            Tuple of (success: bool, outfit_items: List[Dict], error_message: str)
        """
        if not self.gemini_model:
            return False, None, "Gemini API not configured"
        
        try:
            # Prepare the specialized service prompt
            service_prompt = self._build_gemini_service_prompt(context)
            
            # Run with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(self.gemini_model.generate_content, service_prompt),
                timeout=timeout
            )
            
            if not response.text:
                return False, None, "Gemini returned empty response"
            
            # Parse the response to extract outfit items
            outfit_items = self._parse_llm_response(response.text, context["available_items"])
            
            if not outfit_items:
                return False, None, "Gemini could not generate valid outfit"
            
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
        Secondary agent: Generate outfit using Groq API with timeout
        
        Args:
            context: Dictionary with wardrobe items, weather, aesthetic, and user prompt
            timeout: Timeout in seconds for the API call
            
        Returns:
            Tuple of (success: bool, outfit_items: List[Dict], error_message: str)
        """
        if not self.groq_client:
            return False, None, "Groq API not configured"
        
        try:
            # Prepare the specialized service prompt
            service_prompt = self._build_groq_service_prompt(context)
            
            # Generate response using Groq with timeout
            chat_completion = await asyncio.wait_for(
                asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    messages=[
                        {"role": "system", "content": self._get_groq_system_prompt()},
                        {"role": "user", "content": service_prompt}
                    ],
                    model="llama3-8b-8192",  # Fast, reliable model
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_tokens=1000,
                    top_p=0.9
                ),
                timeout=timeout
            )
            
            response_text = chat_completion.choices[0].message.content
            
            if not response_text:
                return False, None, "Groq returned empty response"
            
            # Parse the response to extract outfit items
            outfit_items = self._parse_llm_response(response_text, context["available_items"])
            
            if not outfit_items:
                return False, None, "Groq could not generate valid outfit"
            
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
    
    def _build_gemini_service_prompt(self, context: Dict) -> str:
        """Build specialized service prompt for Gemini"""
        weather_condition = context["weather_condition"]
        desired_aesthetic = context["desired_aesthetic"]
        user_prompt = context.get("user_prompt", "")
        available_items = context["available_items"]
        
        # Count total items for context
        total_items = sum(len(items) for items in available_items.values())
        
        prompt = f"""You are a professional fashion stylist AI specialized in creating weather-appropriate, aesthetically coherent outfits.

**CRITICAL CONSTRAINTS - MUST FOLLOW:**
1. Weather: {weather_condition} - Only select items suitable for this weather
2. Aesthetic: {desired_aesthetic} - All items must match this aesthetic style
3. Color Coordination: Ensure all selected items work together colorwise
4. User Request: "{user_prompt}" - Interpret and fulfill this specific request
5. Completeness: Must include TOP, BOTTOM, FOOTWEAR (and OUTERWEAR if cold weather)

**AVAILABLE WARDROBE ({total_items} items total):**

**TOPS ({len(available_items.get('tops', []))} available):**
{self._format_items_for_prompt(available_items.get('tops', []))}

**BOTTOMS ({len(available_items.get('bottoms', []))} available):**
{self._format_items_for_prompt(available_items.get('bottoms', []))}

**OUTERWEAR ({len(available_items.get('outerwear', []))} available):**
{self._format_items_for_prompt(available_items.get('outerwear', []))}

**FOOTWEAR ({len(available_items.get('footwear', []))} available):**
{self._format_items_for_prompt(available_items.get('footwear', []))}

**INSTRUCTIONS:**
1. Analyze the user prompt: "{user_prompt}"
2. Consider the {weather_condition} weather conditions
3. Select items that match the {desired_aesthetic} aesthetic
4. Ensure color harmony between all selected pieces
5. Provide exactly one item from each required category

**OUTPUT FORMAT (CRITICAL - Follow exactly):**
SELECTED_OUTFIT:
TOP: [exact item name from tops list]
BOTTOM: [exact item name from bottoms list]
FOOTWEAR: [exact item name from footwear list]
OUTERWEAR: [exact item name from outerwear list - only if cold weather]

REASONING:
[Brief explanation of your choices considering weather, aesthetic, colors, and user request]

Remember: You MUST select items only from the provided lists above. Use exact item names."""

        return prompt
    
    def _get_groq_system_prompt(self) -> str:
        """System prompt for Groq chat completion"""
        return """You are a professional fashion stylist AI specialized in creating weather-appropriate, aesthetically coherent outfits. You analyze user requests, weather conditions, and available wardrobe items to create perfect outfit combinations.

Key principles:
- Weather appropriateness is non-negotiable
- Color coordination is essential
- Aesthetic consistency must be maintained
- All selections must come from provided available items only
- Provide clear, actionable recommendations"""
    
    def _build_groq_service_prompt(self, context: Dict) -> str:
        """Build specialized service prompt for Groq"""
        weather_condition = context["weather_condition"]
        desired_aesthetic = context["desired_aesthetic"]
        user_prompt = context.get("user_prompt", "")
        available_items = context["available_items"]
        
        # Count total items for context
        total_items = sum(len(items) for items in available_items.values())
        
        prompt = f"""OUTFIT CREATION REQUEST

**REQUIREMENTS:**
- Weather: {weather_condition}
- Style: {desired_aesthetic}
- Request: "{user_prompt}"
- Must be color coordinated and weather appropriate

**AVAILABLE ITEMS ({total_items} total):**

**TOPS:**
{self._format_items_for_prompt(available_items.get('tops', []))}

**BOTTOMS:**
{self._format_items_for_prompt(available_items.get('bottoms', []))}

**OUTERWEAR:**
{self._format_items_for_prompt(available_items.get('outerwear', []))}

**FOOTWEAR:**
{self._format_items_for_prompt(available_items.get('footwear', []))}

Select one item from each category (outerwear only if cold). Use exact item names.

FORMAT YOUR RESPONSE AS:
SELECTED_OUTFIT:
TOP: [exact item name]
BOTTOM: [exact item name]
FOOTWEAR: [exact item name]
OUTERWEAR: [exact item name - if needed]

REASONING: [Brief explanation]"""

        return prompt
    
    def _format_items_for_prompt(self, items: List[Dict]) -> str:
        """Format wardrobe items for LLM prompt"""
        if not items:
            return "None available"
        
        formatted = []
        for item in items:
            colors = ", ".join(item.get('color', []))
            aesthetics = ", ".join(item.get('aesthetic', []))
            weather_tags = ", ".join(item.get('weather', []))
            
            item_info = f"- {item['item']} ({item['category']})"
            if colors:
                item_info += f" | Colors: {colors}"
            if aesthetics:
                item_info += f" | Aesthetics: {aesthetics}"
            if weather_tags:
                item_info += f" | Weather: {weather_tags}"
            
            formatted.append(item_info)
        
        return "\n".join(formatted)
    
    def _parse_llm_response(self, response_text: str, available_items: Dict) -> Optional[List[Dict]]:
        """
        Parse LLM response to extract selected outfit items
        
        Args:
            response_text: Raw LLM response
            available_items: Dictionary of available items by category
            
        Returns:
            List of selected item dictionaries or None if parsing fails
        """
        try:
            selected_items = []
            
            # Create a lookup dictionary for quick item finding
            all_items = {}
            for category, items in available_items.items():
                for item in items:
                    all_items[item['item'].lower().strip()] = item
            
            # Extract outfit section
            if "SELECTED_OUTFIT:" in response_text:
                outfit_section = response_text.split("SELECTED_OUTFIT:")[1]
                if "REASONING:" in outfit_section:
                    outfit_section = outfit_section.split("REASONING:")[0]
                
                lines = outfit_section.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        category, item_name = line.split(':', 1)
                        category = category.strip().upper()
                        item_name = item_name.strip()
                        
                        # Find the item in available items
                        item_key = item_name.lower().strip()
                        if item_key in all_items:
                            selected_items.append(all_items[item_key])
                            logging.info(f"Selected {category}: {item_name}")
                        else:
                            logging.warning(f"Could not find item: {item_name}")
            
            # Validate outfit completeness
            categories = {item['category'] for item in selected_items}
            required_categories = {'Polo', 'T-shirt', 'Sport T-shirt', 'Shirt'}  # At least one top
            has_top = bool(categories.intersection(required_categories))
            
            bottom_categories = {'Cargo Pants', 'Chinos', 'Jeans', 'Joggers', 'Pants', 'Shorts'}
            has_bottom = bool(categories.intersection(bottom_categories))
            
            footwear_categories = {'Shoes', 'Sneakers'}
            has_footwear = bool(categories.intersection(footwear_categories))
            
            if not (has_top and has_bottom and has_footwear):
                logging.warning(f"Incomplete outfit: top={has_top}, bottom={has_bottom}, footwear={has_footwear}")
                return None
            
            return selected_items if selected_items else None
            
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            return None

# Create global instance
outfit_llm_agents = OutfitLLMAgents()