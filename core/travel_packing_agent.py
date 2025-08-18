from multiprocessing import context
import os
import json
import logging
import asyncio
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from click import prompt
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq
from config.travel_config import (
    AVERAGE_WEIGHTS, 
    DESTINATIONS_CONFIG, 
    WEIGHT_CONSTRAINTS,
    BUSINESS_SCHOOL_REQUIREMENTS,
    OUTFIT_COMBINATIONS,
    VALIDATION_RULES
)

load_dotenv()

class TravelPackingAgent:
    """
    Advanced AI agent for multi-destination travel packing optimization.
    Specialized for long-term business school relocations with weight constraints.
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
        
        # Load configurations
        self.weights = AVERAGE_WEIGHTS
        self.destinations = DESTINATIONS_CONFIG
        self.constraints = WEIGHT_CONSTRAINTS
        self.business_reqs = BUSINESS_SCHOOL_REQUIREMENTS
        self.outfit_matrix = OUTFIT_COMBINATIONS
        self.validation = VALIDATION_RULES
        
        # Initialize analysis modules
        self.current_destinations = []
    
    def _prepare_travel_context(self, trip_config: Dict, available_items: Dict) -> Dict:
        """
        Prepare and normalize the raw context used by AI prompt builders.
        Ensures expected keys exist and are in the right format.
        """
        try:
            raw_destinations_and_dates = (
                trip_config.get("raw_destinations_and_dates")
                or trip_config.get("destinations")
                or ""
            )
            raw_preferences_and_purpose = (
                trip_config.get("raw_preferences_and_purpose")
                or trip_config.get("preferences")
                or ""
            )
            # Normalize bags to list[str]
            bags_value = trip_config.get("bags", [])
            if isinstance(bags_value, str):
                bags_list = [bags_value]
            elif isinstance(bags_value, (list, tuple)):
                bags_list = [str(b) for b in bags_value]
            else:
                bags_list = []

            context = {
                "page_id": trip_config.get("page_id"),
                "raw_destinations_and_dates": str(raw_destinations_and_dates),
                "raw_preferences_and_purpose": str(raw_preferences_and_purpose),
                # Both keys supported by different prompt builders
                "bags": bags_list,
                "raw_bags": bags_list,
                "dates": trip_config.get("dates", {}),
                "weight_constraints": trip_config.get("weight_constraints", self.constraints),
                "available_items": available_items or {},
            }

            return context
        except Exception as e:
            logging.error(f"Error preparing travel context: {e}", exc_info=True)
            # Return minimal context to avoid hard failure
            return {
                "raw_destinations_and_dates": str(trip_config.get("raw_destinations_and_dates", "")),
                "raw_preferences_and_purpose": str(trip_config.get("raw_preferences_and_purpose", "")),
                "bags": [str(b) for b in trip_config.get("bags", []) if isinstance(b, (str, int, float))],
                "raw_bags": [str(b) for b in trip_config.get("bags", []) if isinstance(b, (str, int, float))],
                "dates": trip_config.get("dates", {}),
                "weight_constraints": trip_config.get("weight_constraints", self.constraints),
                "available_items": available_items or {},
            }

    async def generate_multi_destination_packing_list(self, trip_config: Dict, available_items: Dict, timeout: int = 120) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Primary method: Generate a comprehensive packing list using the Gemini API
        by providing it with raw user input for dynamic analysis.
    
        Args:
            trip_config: Raw trip configuration data from Notion.
            available_items: Dictionary of available wardrobe items by category.
            timeout: API timeout in seconds.
        
        Returns:
            Tuple of (success: bool, packing_result: Dict, error_message: str)
        """
        if not self.gemini_model:
            return False, None, "Gemini API not configured"
    
        try:
            # The agent no longer pre-processes the trip config; it prepares it for the AI.
            context = self._prepare_travel_context(trip_config, available_items)
        
            # Build the new, fully dynamic service prompt
            service_prompt = self._build_dynamic_service_prompt(context)
        
            # Generate response with an extended timeout for complex analysis
            response = await asyncio.wait_for(
                asyncio.to_thread(self.gemini_model.generate_content, service_prompt),
                timeout=timeout
            )
        
            if not response.text:
                return False, None, "Gemini returned an empty response"
        
            # Parse the AI's response and finalize the packing list
            packing_result = await self._parse_and_optimize_packing_response(
                response.text, available_items, trip_config
            )
        
            if not packing_result:
                return False, None, "Failed to parse a valid packing list from the AI's response"
        
            logging.info(f"Gemini generated a packing list with {packing_result['total_items']} items")
            return True, packing_result, None
        
        except asyncio.TimeoutError:
            error_msg = f"Gemini API timeout after {timeout} seconds"
            logging.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"An unexpected Gemini API error occurred: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    async def generate_packing_list_with_groq(self, trip_config: Dict, available_items: Dict, timeout: int = 90) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Secondary method: Generate a packing list using the Groq API as a fast fallback,
        providing it with raw user input for dynamic analysis.
    
        Args:
            trip_config: Raw trip configuration data from Notion.
            available_items: Dictionary of available wardrobe items by category.
            timeout: API timeout in seconds.
        
        Returns:
            Tuple of (success: bool, packing_result: Dict, error_message: str)
        """
        if not self.groq_client:
            return False, None, "Groq API not configured"
    
        try:
            # The agent prepares the raw context for the AI prompt
            context = self._prepare_travel_context(trip_config, available_items)
        
            # Build the new, specialized, and highly concise Groq prompt
            service_prompt = self._build_groq_service_prompt(context)
        
            # Generate response using Groq with a suitable timeout
            chat_completion = await asyncio.wait_for(
                asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    messages=[
                        {"role": "system", "content": self._get_groq_system_prompt()},
                        {"role": "user", "content": service_prompt}
                    ],
                    model="llama3-8b-8192",  # A fast and capable model
                    temperature=0.2,
                    max_tokens=2000
                ),
                timeout=timeout
            )
        
            response_text = chat_completion.choices[0].message.content
            if not response_text:
                return False, None, "Groq returned an empty response"
        
            # Parse the AI's response and finalize the packing list
            packing_result = await self._parse_and_optimize_packing_response(
                response_text, available_items, trip_config
            )
        
            if not packing_result:
                return False, None, "Failed to parse a valid packing list from Groq's response"
        
            logging.info(f"Groq generated a packing list with {packing_result['total_items']} items")
            return True, packing_result, None
            
        except asyncio.TimeoutError:
            error_msg = f"Groq API timeout after {timeout} seconds"
            logging.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"An unexpected Groq API error occurred: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    def _calculate_weight_constraints(self) -> Dict:
        """Calculate realistic weight constraints"""
        return {
            "total_clothes_budget_kg": self.constraints["clothes_allocation"]["total_clothes_budget"],
            "checked_bag_clothes_kg": self.constraints["clothes_allocation"]["checked_bag_clothes_kg"],
            "cabin_bag_clothes_kg": self.constraints["clothes_allocation"]["cabin_bag_clothes_kg"],
            "target_efficiency": self.constraints["target_efficiency_ratio"],
            "heavy_item_limit": self.constraints["heavy_clothing_limit"],
            "weight_buffer_kg": self.constraints["essential_weight_reserve"]
        }
    
    def _analyze_business_requirements(self) -> Dict:
        """Analyze business school specific requirements"""
        return {
            "formal_events_monthly": self.business_reqs["formal_events"]["frequency_per_month"],
            "business_casual_weekly": self.business_reqs["business_casual_classes"]["frequency_per_week"],
            "presentations_monthly": self.business_reqs["presentations"]["frequency_per_month"],
            "required_outfit_types": list(self.outfit_matrix.keys()),
            "minimum_formal_outfits": 3,  # Based on frequency
            "minimum_business_casual_outfits": 10  # Based on weekly needs
        }
    
    def _define_optimization_strategy(self, trip_config: Dict) -> Dict:
        """Define optimization strategy based on trip characteristics"""
        return {
            "priority_order": [
                "weight_efficiency",
                "climate_coverage", 
                "business_appropriateness",
                "cultural_compliance",
                "versatility"
            ],
            "selection_criteria": {
                "must_have_categories": ["business_formal", "business_casual", "climate_essentials"],
                "optimization_focus": "multi_destination_versatility",
                "weight_distribution": "strategic_bag_allocation"
            }
        }
    
    def _analyze_available_items(self, available_items: Dict) -> Dict:
        """Analyze available items for optimization insights"""
        analysis = {
            "total_items": sum(len(items) for items in available_items.values()),
            "categories": list(available_items.keys()),
            "weight_analysis": {},
            "business_appropriate_count": 0,
            "climate_versatile_count": 0
        }
        
        # Weight analysis by category
        for category, items in available_items.items():
            weights = [self.weights.get(item['category'], 0.5) for item in items]
            analysis["weight_analysis"][category] = {
                "count": len(items),
                "avg_weight": sum(weights) / len(weights) if weights else 0,
                "total_weight": sum(weights)
            }
        
        # Business and climate analysis
        for category, items in available_items.items():
            for item in items:
                aesthetics = [a.lower() for a in item.get('aesthetic', [])]
                if any(ba in ' '.join(aesthetics) for ba in ['business', 'formal', 'minimalist']):
                    analysis["business_appropriate_count"] += 1
                
                weather_tags = [w.lower() for w in item.get('weather', [])]
                if len(weather_tags) >= 2 or not weather_tags:
                    analysis["climate_versatile_count"] += 1
        
        return analysis

    def _prepare_travel_context(self, trip_config: Dict, available_items: Dict) -> Dict:
        """Prepares a lean context for prompting using raw inputs.

        This method is resilient to both legacy and new trigger formats.
        """
        # Destinations can be a list (e.g., ["Dubai", "Gurgaon"]) or a raw string
        raw_dests = trip_config.get("raw_destinations_and_dates", "")
        if isinstance(raw_dests, list):
            raw_destinations_and_dates = ", ".join([str(x) for x in raw_dests])
        else:
            raw_destinations_and_dates = str(raw_dests)

        # Preferences as plain string
        raw_preferences_and_purpose = str(trip_config.get("raw_preferences_and_purpose", "")).strip()

        # Bags as provided by Notion, e.g., ["Checked Bag: 23kg", "Cabin Bag: 10kg"]
        bags_list = trip_config.get("bags", []) or []
        if not isinstance(bags_list, list):
            bags_list = [str(bags_list)]

        context = {
            "raw_destinations_and_dates": raw_destinations_and_dates,
            "raw_preferences_and_purpose": raw_preferences_and_purpose,
            # Provide both keys for prompts that reference either
            "raw_bags": bags_list,
            "bags": bags_list,
            "available_items": available_items,
        }
        return context
    
    def _build_dynamic_service_prompt(self, context: Dict) -> str:
        """
        Builds a fully dynamic, AI-driven prompt that instructs the model on how to analyze
        raw user input, including dynamic bag limits, to generate a packing list.
        """
        # Convert the list of bags into a readable string for the prompt
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
    
    def _build_groq_service_prompt(self, context: Dict) -> str:
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
        
            # Provides the name and the most important context (aesthetics)
            item_lines = [f"- {item['item']} (Aesthetics: {', '.join(item.get('aesthetic', ['N/A']))})" for item in items]
        
            formatted += "\n".join(item_lines)
            formatted += "\n"
    
        return formatted
    
    def _format_items_concise(self, available_items: Dict) -> str:
        """Format items concisely for Groq"""
        formatted = ""
        
        for category, items in available_items.items():
            if not items:
                continue
                
            formatted += f"\n{category}: "
            item_names = [f"{item['item']} ({self.weights.get(item['category'], 0.5)}kg)" 
                         for item in items[:5]]
            formatted += ", ".join(item_names)
            if len(items) > 5:
                formatted += f" + {len(items) - 5} more"
        
        return formatted
    
    def _get_groq_system_prompt(self) -> str:
        """System prompt for Groq"""
        return """You are an expert travel packing consultant specializing in long-term business relocations. You optimize for weight efficiency, cultural appropriateness, climate adaptation, and professional requirements. You provide precise, actionable packing recommendations with detailed reasoning."""
    
    async def _parse_and_optimize_packing_response(self, response_text: str, available_items: Dict, trip_config: Dict) -> Optional[Dict]:
        """Parse AI response and optimize the packing selection"""
        try:
            # Extract selected items from response
            selected_items = self._extract_selected_items(response_text, available_items)
            
            if not selected_items:
                logging.error("No valid items extracted from AI response")
                logging.debug(f"AI response text (first 500 chars): {response_text[:500]}")
                return None
            
            logging.info(f"Successfully extracted {len(selected_items)} items from AI response")
            
            # Optimize weight and completeness
            optimized_selection = self._optimize_selection(selected_items, trip_config)
            
            # Calculate comprehensive results
            packing_result = await self._calculate_packing_results(optimized_selection, trip_config)
            
            # Log key metrics before validation
            logging.info(f"Packing result: {packing_result['total_items']} items, {packing_result['total_weight_kg']}kg, business readiness: {packing_result['business_readiness']['readiness_score']}")
            
            # Validate completeness
            if not self._validate_packing_completeness(packing_result):
                logging.warning("Packing list failed completeness validation - but returning result anyway for debugging")
                # Return the result even if validation fails, for debugging purposes
                packing_result["validation_failed"] = True
                return packing_result
            
            packing_result["validation_failed"] = False
            return packing_result
            
        except Exception as e:
            logging.error(f"Error parsing packing response: {e}", exc_info=True)
            return None
    
    def _extract_selected_items(self, response_text: str, available_items: Dict) -> List[Dict]:
        """
        Extracts selected items from the AI response using a robust, regex-based approach
        that is resilient to minor formatting variations.
        """
        selected_items = []
    
        # Create a flattened list and a lookup dictionary of all available items
        all_items_flat = [item for category_items in available_items.values() for item in category_items]
        all_items_lookup = {item['item'].lower().strip(): item for item in all_items_flat}

        try:
            # Find the block of text after the final "SELECTED_ITEMS:" heading
            items_block = response_text.split("SELECTED_ITEMS:")[-1]
        
            # Use a flexible regex to find all non-empty lines, ignoring bullet points or numbering
            potential_matches = re.findall(r'^\s*[-•*]?\s*(.+?)\s*$', items_block, re.MULTILINE)
        
            for match in potential_matches:
                match_lower = match.lower().strip()
                # Direct match is fastest and most reliable
                if match_lower in all_items_lookup:
                    selected_items.append(all_items_lookup[match_lower])
                else:
                    # Use fuzzy matching as a fallback for slight AI variations
                    fuzzy_match = self._fuzzy_match_item(match_lower, all_items_lookup)
                    if fuzzy_match:
                        selected_items.append(fuzzy_match)

        except IndexError:
            logging.error("Could not find 'SELECTED_ITEMS:' heading in the AI response.")
            return []

        # Remove any duplicates before returning
        seen_ids = set()
        unique_items = []
        for item in selected_items:
            if item['id'] not in seen_ids:
                seen_ids.add(item['id'])
                unique_items.append(item)
    
        logging.info(f"Extracted {len(unique_items)} unique items from AI response.")
        return unique_items
    
    def _extract_item_name_from_line(self, line: str) -> Optional[str]:
        """Extract item name from various line formats"""
        # Remove common prefixes and formatting
        line = re.sub(r'^[-•*]\s*', '', line)  # Remove bullet points
        line = re.sub(r'^\d+\.\s*', '', line)  # Remove numbers
        
        # Try to extract name before weight, category, or reasoning
        patterns = [
            r'^([^(]+)\s*\([^)]*\)',  # "Item Name (Category)"
            r'^([^-]+)\s*-',          # "Item Name - description"
            r'^([^:]+):',             # "Item Name: description"  
            r'^([^|]+)\|',            # "Item Name | weight"
            r'^([^\n\r]+)'            # Just take the whole line if no patterns match
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up common artifacts
                name = re.sub(r'\s*\(.*?\)\s*$', '', name)  # Remove trailing category
                name = re.sub(r'\s*-.*$', '', name)         # Remove trailing descriptions
                if name and len(name) > 2:
                    return name
        
        return None
    
    def _fuzzy_match_item(self, target_name: str, items_dict: Dict) -> Optional[Dict]:
        """Fuzzy matching for item names"""
        # Word-based matching
        target_words = set(target_name.lower().split())
        
        best_match = None
        best_score = 0
        
        for item_name, item_data in items_dict.items():
            item_words = set(item_name.lower().split())
            
            # Calculate overlap score
            if target_words and item_words:
                overlap = len(target_words.intersection(item_words))
                total_words = len(target_words.union(item_words))
                score = overlap / total_words
                
                if score > best_score and score >= 0.4:  # Minimum 40% overlap
                    best_score = score
                    best_match = item_data
        
        return best_match
    
    def _optimize_selection(self, selected_items: List[Dict], trip_config: Dict) -> List[Dict]:
        """Optimize selection for weight and completeness"""
        
        # Calculate efficiency scores
        scored_items = []
        for item in selected_items:
            efficiency_score = self._calculate_comprehensive_efficiency(item, trip_config)
            weight = self.weights.get(item['category'], 0.5)
            scored_items.append((item, efficiency_score, weight))
        
        # Sort by efficiency (highest first)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Select items within weight budget
        optimized = []
        current_weight = 0
        weight_budget = self.constraints["clothes_allocation"]["total_clothes_budget"]
        
        for item, score, weight in scored_items:
            if current_weight + weight <= weight_budget:
                optimized.append(item)
                current_weight += weight
        
        logging.info(f"Optimization: {len(selected_items)} → {len(optimized)} items, weight: {current_weight:.2f}kg")
        return optimized
    
    def _calculate_comprehensive_efficiency(self, item: Dict, trip_config: Dict) -> float:
        """Calculate comprehensive efficiency score for an item"""
        
        # Base versatility score
        versatility = self._calculate_versatility_score(item)
        
        # Climate coverage score
        climate_score = self._calculate_climate_coverage_score(item, trip_config)
        
        # Business appropriateness score
        business_score = self._calculate_business_score(item)
        
        # Cultural compliance score
        cultural_score = self._calculate_cultural_score(item)
        
        # Weight efficiency
        weight = self.weights.get(item['category'], 0.5)
        weight_efficiency = 1 / weight if weight > 0 else 1
        
        # Combined score
        total_score = (versatility * 0.3 + 
                      climate_score * 0.25 + 
                      business_score * 0.25 + 
                      cultural_score * 0.1 + 
                      weight_efficiency * 0.1)
        
        return total_score
    
    def _calculate_versatility_score(self, item: Dict) -> float:
        """Calculate how versatile an item is across occasions"""
        category = item['category']
        aesthetics = item.get('aesthetic', [])
        
        # Base versatility by category
        versatility_map = {
            'Chinos': 0.9, 'Polo': 0.85, 'Shirt': 0.8, 'Sneakers': 0.75,
            'T-shirt': 0.7, 'Jeans': 0.65, 'Suit': 0.4, 'Shoes': 0.6
        }
        
        base_score = versatility_map.get(category, 0.5)
        
        # Bonus for multiple aesthetics
        aesthetic_bonus = len(aesthetics) * 0.1
        
        return min(base_score + aesthetic_bonus, 1.0)
    
    def _calculate_climate_coverage_score(self, item: Dict, trip_config: Dict) -> float:
        """Calculate how well item covers climate needs"""
        weather_tags = [w.lower() for w in item.get('weather', [])]
        
        if not weather_tags:
            return 0.6  # Neutral items get medium score
        
        # Check coverage for destination climates
        coverage = 0
        if 'hot' in weather_tags:
            coverage += 0.5  # Dubai coverage
        if 'cold' in weather_tags:
            coverage += 0.5  # Gurgaon winter coverage
        if len(weather_tags) >= 2:
            coverage += 0.2  # Versatile bonus
        
        return min(coverage, 1.0)
    
    def _calculate_business_score(self, item: Dict) -> float:
        """Calculate business appropriateness score"""
        aesthetics = [a.lower() for a in item.get('aesthetic', [])]
        category = item['category']
        
        business_aesthetics = ['business casual', 'minimalist', 'formal']
        business_categories = ['Suit', 'Shirt', 'Chinos', 'Shoes', 'Polo']
        
        score = 0
        if any(ba in ' '.join(aesthetics) for ba in business_aesthetics):
            score += 0.6
        if category in business_categories:
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_cultural_score(self, item: Dict) -> float:
        """Calculate cultural appropriateness score"""
        # For Dubai's high modesty requirements
        category = item['category']
        
        # High score for modest items
        modest_categories = ['Shirt', 'Chinos', 'Pants', 'Polo']
        if category in modest_categories:
            return 1.0
        
        # Medium score for neutral items
        return 0.7
    
    async def _calculate_packing_results(self, selected_items: List[Dict], trip_config: Dict) -> Dict:
        """Calculate comprehensive packing results"""
        
        # Basic calculations
        total_weight = sum(self.weights.get(item['category'], 0.5) for item in selected_items)
        total_items = len(selected_items)
        
        # Bag allocation
        bag_allocation = self._allocate_items_to_bags(selected_items)
        
        # Outfit analysis
        outfit_analysis = self._analyze_outfit_possibilities(selected_items)
        
        # Generate comprehensive results
        results = {
            "selected_items": selected_items,
            "total_items": total_items,
            "total_weight_kg": round(total_weight, 2),
            "weight_efficiency": round(total_items / total_weight, 1) if total_weight > 0 else 0,
            "bag_allocation": bag_allocation,
            "outfit_analysis": outfit_analysis,
            "business_readiness": self._assess_business_readiness(selected_items),
            "climate_coverage": self._assess_climate_coverage(selected_items, trip_config),
            "cultural_compliance": self._assess_cultural_compliance(selected_items),
            "packing_guide": self._generate_packing_guide(selected_items, bag_allocation),
            "trip_tips": await self._generate_destination_tips(trip_config)
        }
        
        return results
    
    def _allocate_items_to_bags(self, selected_items: List[Dict]) -> Dict:
        """Allocate items between checked and cabin bags with simplified logic."""
        
        checked_items = []
        cabin_items = []
        checked_weight = 0.0
        cabin_weight = 0.0
        
        # Get capacities from config
        checked_bag_capacity = self.constraints.get("clothes_allocation", {}).get("checked_bag_clothes_kg", 15)
        cabin_bag_capacity = self.constraints.get("clothes_allocation", {}).get("cabin_bag_clothes_kg", 3)

        for item in selected_items:
            weight = self.weights.get(item.get('category'), 0.5)
            
            # Prioritize cabin for essentials if space allows
            is_essential = item.get('category') in ['T-shirt', 'Polo', 'Underwear', 'Socks']
            cabin_has_space = cabin_weight + weight <= cabin_bag_capacity
            
            if is_essential and cabin_has_space and len(cabin_items) < 5:
                cabin_items.append(item)
                cabin_weight += weight
            # Then, try to put in checked bag
            elif checked_weight + weight <= checked_bag_capacity:
                checked_items.append(item)
                checked_weight += weight
            # Fallback to cabin bag if checked bag is full but cabin has space
            elif cabin_has_space:
                cabin_items.append(item)
                cabin_weight += weight
            else:
                logging.warning(f"Could not fit item {item.get('item', 'N/A')} in any bag (weight: {weight}kg).")

        return {
            "checked_bag": {
                "items": checked_items,
                "weight_kg": round(checked_weight, 2),
                "space_utilization": round(checked_weight / checked_bag_capacity * 100, 1) if checked_bag_capacity > 0 else 0,
            },
            "cabin_bag": {
                "items": cabin_items,
                "weight_kg": round(cabin_weight, 2),
                "space_utilization": round(cabin_weight / cabin_bag_capacity * 100, 1) if cabin_bag_capacity > 0 else 0,
            },
            "strategy_notes": [
                "Simplified allocation: essentials to cabin, then fill checked, then cabin for overflow."
            ]
        }
    
    def _analyze_outfit_possibilities(self, selected_items: List[Dict]) -> Dict:
        """Analyze outfit possibilities from selected items"""
        
        # Categorize items
        categories = {}
        for item in selected_items:
            cat = item['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        
        # Count outfit possibilities
        business_formal = self._count_business_formal_outfits(categories)
        business_casual = self._count_business_casual_outfits(categories)
        casual = self._count_casual_outfits(categories)
        
        return {
            "business_formal_outfits": business_formal,
            "business_casual_outfits": business_casual,
            "casual_outfits": casual,
            "total_outfit_combinations": business_formal + business_casual + casual,
            "category_breakdown": {cat: len(items) for cat, items in categories.items()}
        }
    
    def _count_business_formal_outfits(self, categories: Dict) -> int:
        """Count possible business formal outfits"""
        suits = len(categories.get('Suit', []))
        dress_shoes = len(categories.get('Shoes', []))
        dress_shirts = len(categories.get('Shirt', []))
        
        return min(suits, dress_shoes, dress_shirts)
    
    def _count_business_casual_outfits(self, categories: Dict) -> int:
        """Count possible business casual outfits"""
        bottoms = len(categories.get('Chinos', [])) + len(categories.get('Pants', []))
        tops = len(categories.get('Polo', [])) + len(categories.get('Shirt', []))
        shoes = len(categories.get('Shoes', [])) + len(categories.get('Sneakers', []))
        
        return min(bottoms, tops, shoes) * 2  # Multiple combinations possible
    
    def _count_casual_outfits(self, categories: Dict) -> int:
        """Count possible casual outfits"""
        bottoms = (len(categories.get('Jeans', [])) + 
                  len(categories.get('Shorts', [])) + 
                  len(categories.get('Chinos', [])))
        tops = (len(categories.get('T-shirt', [])) + 
               len(categories.get('Polo', [])))
        footwear = len(categories.get('Sneakers', []))
        
        return min(bottoms, tops, footwear) * 3  # High combination potential
    
    def _assess_business_readiness(self, selected_items: List[Dict]) -> Dict:
        """Assess business readiness of selection"""
        
        suits = [i for i in selected_items if i['category'] == 'Suit']
        dress_shoes = [i for i in selected_items if i['category'] == 'Shoes' and 
                      any('formal' in a.lower() for a in i.get('aesthetic', []))]
        business_shirts = [i for i in selected_items if i['category'] in ['Shirt', 'Polo'] 
                          and any('business' in a.lower() or 'formal' in a.lower() 
                                 for a in i.get('aesthetic', []))]
        
        # More flexible scoring: 1 suit is acceptable, focus on formal shoes
        readiness_score = min(len(suits) / 1, 1.0) * 0.5  # Need at least 1 suit
        readiness_score += min(len(dress_shoes) / 1, 1.0) * 0.3  # Need at least 1 formal shoe
        readiness_score += min(len(business_shirts) / 2, 1.0) * 0.2  # Need at least 2 business shirts
        
        return {
            "readiness_score": round(readiness_score, 2),
            "suits_count": len(suits),
            "dress_shoes_count": len(dress_shoes),
            "business_shirts_count": len(business_shirts),
            "meets_requirements": readiness_score >= 0.6  # Lower threshold from 0.8 to 0.6
        }
    
    def _assess_climate_coverage(self, selected_items: List[Dict], trip_config: Dict) -> Dict:
        """Assess climate coverage of selection with robust destination parsing."""
        hot_weather_items = [
            item for item in selected_items
            if any('hot' in w.lower() for w in item.get('weather', []))
        ]
        cold_weather_items = [
            item for item in selected_items
            if any('cold' in w.lower() for w in item.get('weather', []))
        ]
        versatile_items = [
            item for item in selected_items
            if len(item.get('weather', [])) == 0 or len(item.get('weather', [])) >= 2
        ]

        # Determine cities from either structured or raw input
        cities: List[str] = []
        if isinstance(trip_config.get("destinations"), list) and trip_config["destinations"] and isinstance(trip_config["destinations"][0], dict):
            cities = [str(d.get("city", "")).lower() for d in trip_config["destinations"] if d.get("city")]
        else:
            raw_dests = trip_config.get("raw_destinations_and_dates", [])
            if isinstance(raw_dests, list):
                cities = [str(x).lower() for x in raw_dests]
            else:
                # Extract simple city tokens from a freeform string (split by commas)
                cities = [c.strip().lower() for c in str(raw_dests).split(",") if c.strip()]

        temp_range = self._calculate_temperature_range(cities)

        return {
            "hot_weather_items": hot_weather_items,
            "cold_weather_items": cold_weather_items,
            "versatile_items_list": versatile_items,
            "hot_weather_coverage": len(hot_weather_items),
            "cold_weather_coverage": len(cold_weather_items),
            "versatile_items": len(versatile_items),
            "temperature_range_covered": f"{temp_range['min']}°C - {temp_range['max']}°C",
            "coverage_adequacy": "excellent" if len(versatile_items) > 10 else "good" if len(versatile_items) > 5 else "needs_improvement"
        }

    def _calculate_temperature_range(self, cities: List[str]) -> Dict:
        """Calculate min/max temperature across destination cities using config seasons.

        If cities is empty, consider all configured cities.
        """
        if not cities:
            cities = list(self.destinations.keys())
        # Normalize to config keys
        normalized = []
        for c in cities:
            key = str(c).strip().lower()
            if key in self.destinations:
                normalized.append(key)
        if not normalized:
            normalized = list(self.destinations.keys())

        min_temp = float("inf")
        max_temp = float("-inf")
        for city in normalized:
            seasons = self.destinations[city].get("seasons", {})
            for m, data in seasons.items():
                tr = data.get("temp_range") or []
                if isinstance(tr, (list, tuple)) and len(tr) == 2:
                    try:
                        min_temp = min(min_temp, float(tr[0]))
                        max_temp = max(max_temp, float(tr[1]))
                    except Exception:
                        continue
        if min_temp == float("inf") or max_temp == float("-inf"):
            return {"min": 0, "max": 0}
        return {"min": int(min_temp), "max": int(max_temp)}
    
    def _assess_cultural_compliance(self, selected_items: List[Dict]) -> Dict:
        """Assess cultural compliance of selection"""
        
        modest_items = [i for i in selected_items 
                       if i['category'] in ['Shirt', 'Chinos', 'Pants', 'Polo']]
        
        compliance_score = len(modest_items) / len(selected_items) if selected_items else 0
        
        return {
            "compliance_score": round(compliance_score, 2),
            "modest_items_count": len(modest_items),
            "total_items": len(selected_items),
            "dubai_ready": compliance_score >= 0.7,
            "recommendations": [
                "Long sleeves preferred for Dubai",
                "Full coverage pants recommended",
                "Conservative color choices advisable"
            ]
        }
    
    def _generate_packing_guide(self, selected_items: List[Dict], bag_allocation: Dict) -> Dict:
        """Generate comprehensive packing guide"""
        
        return {
            "packing_techniques": [
                "Roll casual items (T-shirts, underwear) to save space",
                "Fold formal items (suits, dress shirts) with tissue paper",
                "Use packing cubes for organization by category",
                "Place shoes in shoe bags to protect clothes",
                "Keep heavy items at bottom of suitcase"
            ],
            "organization_strategy": {
                "checked_bag_organization": [
                    "Bottom layer: Heavy items (shoes, suits)",
                    "Middle layer: Folded casual clothes", 
                    "Top layer: Delicate items and accessories"
                ],
                "cabin_bag_organization": [
                    "Main compartment: Essential clothes for 2-3 days",
                    "Quick access: Change of shirt and underwear",
                    "Exterior pocket: Travel documents and electronics"
                ]
            },
            "space_optimization": [
                "Stuff socks inside shoes",
                "Use every inch of space efficiently",
                "Consider vacuum-sealed bags for bulky items",
                "Wear heaviest items during travel"
            ],
            "travel_day_strategy": {
                "wear_during_travel": [
                    "Heaviest pair of shoes",
                    "Thickest jacket or coat",
                    "Business casual outfit (ready for arrival)"
                ],
                "cabin_essentials": [
                    "Complete change of clothes",
                    "Essential toiletries",
                    "Important medications"
                ]
            }
        }
    
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

    def _build_destination_tip_prompt(self, city: str, trip_config: Dict, web_context: str) -> str:
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

    async def _generate_destination_tips(self, trip_config: Dict) -> Dict:
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
                prompt = self._build_destination_tip_prompt(city, trip_config, web_context)

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

    def _categorize_items_for_travel(self, items: List[Dict]) -> Dict:
        """Categorize a flat list of items by their 'category' field."""
        categorized: Dict[str, List[Dict]] = {}
        for item in items or []:
            category = item.get("category", "Unknown")
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)
        return categorized
    
    def _validate_packing_completeness(self, packing_result: Dict) -> bool:
        """Validate that packing list meets minimum requirements"""
        
        # Check weight constraint
        if packing_result["total_weight_kg"] > self.constraints["clothes_allocation"]["total_clothes_budget"]:
            logging.warning(f"Packing exceeds weight budget: {packing_result['total_weight_kg']}kg > {self.constraints['clothes_allocation']['total_clothes_budget']}kg")
            # Don't fail completely, just warn
        
        # Check business readiness (make it a warning, not a failure)
        if not packing_result["business_readiness"]["meets_requirements"]:
            logging.warning(f"Business readiness score: {packing_result['business_readiness']['readiness_score']} (suits: {packing_result['business_readiness']['suits_count']}, formal shoes: {packing_result['business_readiness']['dress_shoes_count']}, business shirts: {packing_result['business_readiness']['business_shirts_count']})")
            # Don't fail for business readiness unless it's extremely low
            if packing_result["business_readiness"]["readiness_score"] < 0.3:
                logging.error("Extremely low business readiness - failing validation")
                return False
        
        # Check minimum items (more flexible)
        min_items_required = max(5, self.validation["minimum_items_per_category"]["casual_tops"])
        if packing_result["total_items"] < min_items_required:
            logging.warning(f"Low item count: {packing_result['total_items']} items (recommended: {min_items_required}+)")
            # Only fail if extremely low
            if packing_result["total_items"] < 3:
                logging.error("Extremely low item count - failing validation")
                return False
        
        return True
    
    def _build_example_outfits_prompt(self, selected_items: List[Dict], trip_config: Dict) -> str:
        """Builds a prompt to generate three example outfits from the selected items."""
        
        trip_overview = trip_config.get("trip_overview", {})
        destinations = ", ".join([d.get('city', '').title() for d in trip_config.get("destinations", [])])

        prompt = f"""You are a fashion stylist creating example outfits from a pre-selected travel wardrobe.

**CONTEXT**
* **Trip**: A {trip_overview.get('total_duration_months', 'long')} month business school trip to {destinations}.
* **Goal**: Create three distinct, stylish, and practical example outfits using ONLY the clothes provided below.

**AVAILABLE ITEMS FOR OUTFITS**
{self._format_items_with_intelligence(self._categorize_items_for_travel(selected_items), {})}

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
            prompt = self._build_example_outfits_prompt(selected_items, trip_config)
            response = await asyncio.wait_for(
                asyncio.to_thread(self.gemini_model.generate_content, prompt),
                timeout=timeout
            )
            return response.text
        except Exception as e:
            logging.error(f"Failed to generate example outfits: {e}")
            return None

# Create global instance
travel_packing_agent = TravelPackingAgent()