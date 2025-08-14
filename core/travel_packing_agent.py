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
    
    async def generate_multi_destination_packing_list(self, trip_config: Dict, available_items: Dict, timeout: int = 30) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Primary method: Generate comprehensive packing list using Gemini API
        
        Args:
            trip_config: Trip configuration with destinations, dates, constraints
            available_items: Dictionary of available wardrobe items by category
            timeout: API timeout in seconds
            
        Returns:
            Tuple of (success: bool, packing_result: Dict, error_message: str)
        """
        if not self.gemini_model:
            return False, None, "Gemini API not configured"
        
        try:
            self.current_destinations = [dest["city"] for dest in trip_config["destinations"]]
            
            # Prepare comprehensive context
            context = self._prepare_travel_context(trip_config, available_items)
            
            # Build specialized service prompt
            service_prompt = self._build_dynamic_service_prompt(context)
            
            # Generate response with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(self.gemini_model.generate_content, service_prompt),
                timeout=timeout
            )
            
            if not response.text:
                return False, None, "Gemini returned empty response"
            
            # Parse response and optimize selection
            packing_result = self._parse_and_optimize_packing_response(
                response.text, available_items, trip_config
            )
            
            if not packing_result:
                return False, None, "Failed to parse valid packing list"
            
            logging.info(f"Gemini generated packing list with {packing_result['total_items']} items")
            return True, packing_result, None
            
        except asyncio.TimeoutError:
            error_msg = f"Gemini API timeout after {timeout} seconds"
            logging.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Gemini API error: {str(e)}"
            logging.error(error_msg)
            return False, None, error_msg
    
    async def generate_packing_list_with_groq(self, trip_config: Dict, available_items: Dict, timeout: int = 25) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Secondary method: Generate packing list using Groq API as fallback
        
        Args:
            trip_config: Trip configuration
            available_items: Available wardrobe items
            timeout: API timeout in seconds
            
        Returns:
            Tuple of (success: bool, packing_result: Dict, error_message: str)
        """
        if not self.groq_client:
            return False, None, "Groq API not configured"
        
        try:
            self.current_destinations = [dest["city"] for dest in trip_config["destinations"]]
            
            # Prepare context
            context = self._prepare_travel_context(trip_config, available_items)
            
            # Build Groq-optimized prompt
            service_prompt = self._build_dynamic_service_prompt(context)
            
            # Generate response with timeout
            chat_completion = await asyncio.wait_for(
                asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    messages=[
                        {"role": "system", "content": self._get_groq_system_prompt()},
                        {"role": "user", "content": service_prompt}
                    ],
                    model="llama3-8b-8192",
                    temperature=0.2,  # Lower temperature for consistent packing decisions
                    max_tokens=2000,  # More tokens for comprehensive packing lists
                    top_p=0.85
                ),
                timeout=timeout
            )
            
            response_text = chat_completion.choices[0].message.content
            
            if not response_text:
                return False, None, "Groq returned empty response"
            
            # Parse and optimize
            packing_result = self._parse_and_optimize_packing_response(
                response_text, available_items, trip_config
            )
            
            if not packing_result:
                return False, None, "Failed to parse valid packing list"
            
            logging.info(f"Groq generated packing list with {packing_result['total_items']} items")
            return True, packing_result, None
            
        except asyncio.TimeoutError:
            error_msg = f"Groq API timeout after {timeout} seconds"
            logging.error(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Groq API error: {str(e)}"
            logging.error(error_msg)
            return False, None, error_msg
    
    def _prepare_travel_context(self, trip_config: Dict, available_items: Dict) -> Dict:
        """Prepare comprehensive context for AI processing"""
        
        context = {
            "trip_overview": self._analyze_trip_overview(trip_config),
            "destination_analysis": self._analyze_destinations(trip_config["destinations"]),
            "weight_constraints": self._calculate_weight_constraints(),
            "business_requirements": self._analyze_business_requirements(),
            "available_items": available_items,
            "optimization_strategy": self._define_optimization_strategy(trip_config)
        }
        
        # Add item analysis
        context["item_analysis"] = self._analyze_available_items(available_items)
        
        return context
    
    def _analyze_trip_overview(self, trip_config: Dict) -> Dict:
        """Analyze overall trip characteristics"""
        destinations = trip_config["destinations"]
        
        # Calculate total duration
        start_date = datetime.strptime(destinations[0]["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(destinations[-1]["end_date"], "%Y-%m-%d")
        total_duration = (end_date - start_date).days
        
        # Analyze temperature range across all destinations/seasons
        temp_range = self._calculate_temperature_range(destinations)
        
        # Identify critical challenges
        challenges = self._identify_packing_challenges(destinations)
        
        return {
            "total_duration_days": total_duration,
            "total_duration_months": round(total_duration / 30, 1),
            "destination_count": len(destinations),
            "temperature_range": temp_range,
            "climate_diversity": len(set(self.destinations[d["city"]]["climate_profile"] for d in destinations)),
            "critical_challenges": challenges,
            "trip_type": "long_term_business_school_relocation"
        }
    
    def _calculate_temperature_range(self, destinations: List[Dict]) -> Dict:
        """Calculate temperature range across all destinations and seasons"""
        min_temp = float('inf')
        max_temp = float('-inf')
        
        for dest in destinations:
            city_config = self.destinations[dest["city"]]
            months = self._get_months_in_destination(dest["start_date"], dest["end_date"])
            
            for month in months:
                if month in city_config["seasons"]:
                    temp_range = city_config["seasons"][month]["temp_range"]
                    min_temp = min(min_temp, temp_range[0])
                    max_temp = max(max_temp, temp_range[1])
        
        return {
            "min": min_temp,
            "max": max_temp,
            "span": max_temp - min_temp
        }
    
    def _get_months_in_destination(self, start_date: str, end_date: str) -> List[str]:
        """Get months covered in a destination stay"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        months = []
        current = start.replace(day=1)
        while current <= end:
            months.append(current.strftime("%B").lower())
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return months
    
    def _identify_packing_challenges(self, destinations: List[Dict]) -> List[str]:
        """Identify critical packing challenges"""
        challenges = []
        
        # Temperature span challenge
        temp_range = self._calculate_temperature_range(destinations)
        if temp_range["span"] > 30:
            challenges.append(f"extreme_temperature_range_{temp_range['span']}C")
        
        # Climate diversity
        climates = set(self.destinations[d["city"]]["climate_profile"] for d in destinations)
        if len(climates) > 1:
            challenges.append("multi_climate_adaptation")
        
        # Cultural requirements
        if any("high" in self.destinations[d["city"]]["cultural_context"]["modesty_level"] 
               for d in destinations):
            challenges.append("high_cultural_modesty_requirements")
        
        # Seasonal transitions
        challenges.append("seasonal_transitions")
        
        return challenges
    
    def _analyze_destinations(self, destinations: List[Dict]) -> List[Dict]:
        """Detailed analysis of each destination"""
        analysis = []
        
        for dest in destinations:
            city_config = self.destinations[dest["city"]]
            
            # Calculate seasonal progression
            months = self._get_months_in_destination(dest["start_date"], dest["end_date"])
            seasonal_analysis = self._analyze_seasonal_progression(city_config, months)
            
            dest_analysis = {
                "city": dest["city"],
                "duration_months": len(months),
                "months": months,
                "climate_profile": city_config["climate_profile"],
                "seasonal_analysis": seasonal_analysis,
                "cultural_requirements": city_config["cultural_context"],
                "weight_priorities": city_config["weight_priorities"],
                "activity_requirements": city_config["activity_types"],
                "climate_recommendations": city_config["climate_recommendations"]
            }
            
            analysis.append(dest_analysis)
        
        return analysis
    
    def _analyze_seasonal_progression(self, city_config: Dict, months: List[str]) -> Dict:
        """Analyze how seasons progress in a destination"""
        temp_progression = []
        challenges = []
        
        for month in months:
            if month in city_config["seasons"]:
                season_data = city_config["seasons"][month]
                temp_progression.append({
                    "month": month,
                    "temp_range": season_data["temp_range"],
                    "weather": season_data["weather"]
                })
                
                # Add specific challenges
                if "monsoon" in season_data.get("weather", ""):
                    challenges.append("monsoon_preparation")
                if season_data["temp_range"][1] > 35:
                    challenges.append("extreme_heat")
                if season_data["temp_range"][0] < 10:
                    challenges.append("cold_weather")
        
        return {
            "temp_progression": temp_progression,
            "challenges": challenges,
            "months_count": len(months)
        }
    
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
    
    def _build_dynamic_service_prompt(self, context: Dict) -> str:
        """
        Builds the definitive, AI-driven prompt using structured, dynamic user input
        from Notion for the highest quality analysis.
        """
        prompt = f"""You are an expert AI travel packing consultant. Your task is to analyze a user's travel plan, use your own world knowledge, and create the most weight-efficient and versatile packing list possible.

    **USER'S TRAVEL PLAN FROM NOTION**
    * **Destinations**: {", ".join(context['destinations'])}
    * **Overall Trip Dates**: From {context['dates']['start']} to {context['dates']['end']}
    * **Purpose, Itinerary & Preferences**: "{context['raw_preferences_and_purpose']}"
    * **Weight Limit**: Absolute maximum of {context['weight_constraints']['clothes_allocation']['total_clothes_budget']}kg for all clothing.

    **YOUR ANALYSIS PROCESS (Follow these steps):**
    1.  **Analyze Itinerary**: First, parse the user's "Purpose, Itinerary & Preferences" to understand the timeline for each destination (e.g., "Dubai from September to December").
    2.  **Determine Climate & Culture**: For each city and its specific timeline, use your knowledge to determine the expected seasons, temperature range (in Celsius), and cultural dress norms.
    3.  **Synthesize a Plan**: Based on your analysis, create a packing strategy that balances all the user's needs.

    **AVAILABLE WARDROBE (SELECT ONLY FROM THIS LIST)**
    {self._format_items_with_intelligence(context["available_items"], context)}

    **CRITICAL OUTPUT INSTRUCTIONS**
    Your entire response must be ONLY a list of the selected items under the heading "SELECTED_ITEMS:". Each item must be on a new line.

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
    
    def _parse_and_optimize_packing_response(self, response_text: str, available_items: Dict, trip_config: Dict) -> Optional[Dict]:
        """Parse AI response and optimize the packing selection"""
        try:
            # Extract selected items from response
            selected_items = self._extract_selected_items(response_text, available_items)
            
            if not selected_items:
                logging.error("No valid items extracted from AI response")
                return None
            
            # Optimize weight and completeness
            optimized_selection = self._optimize_selection(selected_items, trip_config)
            
            # Calculate comprehensive results
            packing_result = self._calculate_packing_results(optimized_selection, trip_config)
            
            # Validate completeness
            if not self._validate_packing_completeness(packing_result):
                logging.warning("Packing list failed completeness validation")
                return None
            
            return packing_result
            
        except Exception as e:
            logging.error(f"Error parsing packing response: {e}")
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
    
    def _calculate_packing_results(self, selected_items: List[Dict], trip_config: Dict) -> Dict:
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
            "trip_tips": self._generate_destination_tips(trip_config)
        }
        
        return results
    
    def _allocate_items_to_bags(self, selected_items: List[Dict]) -> Dict:
        """Allocate items between checked and cabin bags"""
        
        checked_items = []
        cabin_items = []
        checked_weight = 0
        cabin_weight = 0
        
        # Strategic allocation
        for item in selected_items:
            weight = self.weights.get(item['category'], 0.5)
            category = item['category']
            
            # Heavy formal items → checked
            if category in ['Suit', 'Shoes', 'Overcoat'] or weight > 0.8:
                if checked_weight + weight <= self.constraints["clothes_allocation"]["checked_bag_clothes_kg"]:
                    checked_items.append(item)
                    checked_weight += weight
                else:
                    cabin_items.append(item)
                    cabin_weight += weight
            
            # Light essentials → cabin (for delays)
            elif category in ['T-shirt', 'Polo'] and len(cabin_items) < 4:
                if cabin_weight + weight <= self.constraints["clothes_allocation"]["cabin_bag_clothes_kg"]:
                    cabin_items.append(item)
                    cabin_weight += weight
                else:
                    checked_items.append(item)
                    checked_weight += weight
            
            # Everything else → checked (unless cabin has space)
            else:
                if checked_weight + weight <= self.constraints["clothes_allocation"]["checked_bag_clothes_kg"]:
                    checked_items.append(item)
                    checked_weight += weight
                else:
                    cabin_items.append(item)
                    cabin_weight += weight
        
        return {
            "checked_bag": {
                "items": checked_items,
                "weight_kg": round(checked_weight, 2),
                "space_utilization": round(checked_weight / self.constraints["clothes_allocation"]["checked_bag_clothes_kg"] * 100, 1)
            },
            "cabin_bag": {
                "items": cabin_items,
                "weight_kg": round(cabin_weight, 2),
                "space_utilization": round(cabin_weight / self.constraints["clothes_allocation"]["cabin_bag_clothes_kg"] * 100, 1)
            },
            "strategy_notes": [
                "Heavy formal items in checked bag",
                "Essential backup items in cabin",
                "Strategic distribution for travel delays"
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
        dress_shoes = [i for i in selected_items if i['category'] == 'Shoes']
        business_shirts = [i for i in selected_items if i['category'] in ['Shirt', 'Polo'] 
                          and any('business' in a.lower() or 'formal' in a.lower() 
                                 for a in i.get('aesthetic', []))]
        
        readiness_score = min(len(suits) / 2, 1.0) * 0.4  # Need at least 2 suits
        readiness_score += min(len(dress_shoes) / 2, 1.0) * 0.3  # Need dress shoes
        readiness_score += min(len(business_shirts) / 5, 1.0) * 0.3  # Need business shirts
        
        return {
            "readiness_score": round(readiness_score, 2),
            "suits_count": len(suits),
            "dress_shoes_count": len(dress_shoes),
            "business_shirts_count": len(business_shirts),
            "meets_requirements": readiness_score >= 0.8
        }
    
    def _assess_climate_coverage(self, selected_items: List[Dict], trip_config: Dict) -> Dict:
        """Assess climate coverage of selection"""
        
        hot_weather_items = [i for i in selected_items 
                            if 'hot' in [w.lower() for w in i.get('weather', [])]]
        cold_weather_items = [i for i in selected_items 
                             if 'cold' in [w.lower() for w in i.get('weather', [])]]
        versatile_items = [i for i in selected_items 
                          if len(i.get('weather', [])) == 0 or len(i.get('weather', [])) >= 2]
        
        temp_range = self._calculate_temperature_range(trip_config["destinations"])
        
        return {
            "hot_weather_coverage": len(hot_weather_items),
            "cold_weather_coverage": len(cold_weather_items),
            "versatile_items": len(versatile_items),
            "temperature_range_covered": f"{temp_range['min']}°C - {temp_range['max']}°C",
            "coverage_adequacy": "excellent" if len(versatile_items) > 10 else "good" if len(versatile_items) > 5 else "needs_improvement"
        }
    
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
    
    def _generate_destination_tips(self, trip_config: Dict) -> Dict:
        """Generate destination-specific tips"""
        
        tips = {}
        for dest in trip_config["destinations"]:
            city = dest["city"]
            city_config = self.destinations[city]
            
            tips[city] = {
                "cultural_tips": [
                    f"Modesty level: {city_config['cultural_context']['modesty_level']}",
                    f"Business formality: {city_config['cultural_context']['business_formality']}",
                    "Respect local dress codes at all times"
                ],
                "climate_preparation": [
                    f"Climate type: {city_config['climate_profile']}",
                    f"Expected weather: {', '.join([city_config['seasons'][m]['weather'] for m in self._get_months_in_destination(dest['start_date'], dest['end_date']) if m in city_config['seasons']])}"
                ],
                "practical_advice": city_config.get("climate_recommendations", {}).get("essential_items", [])
            }
        
        return tips
    
    def _validate_packing_completeness(self, packing_result: Dict) -> bool:
        """Validate that packing list meets minimum requirements"""
        
        # Check weight constraint
        if packing_result["total_weight_kg"] > self.constraints["clothes_allocation"]["total_clothes_budget"]:
            logging.error(f"Packing exceeds weight budget: {packing_result['total_weight_kg']}kg > {self.constraints['clothes_allocation']['total_clothes_budget']}kg")
            return False
        
        # Check business readiness
        if not packing_result["business_readiness"]["meets_requirements"]:
            logging.error("Packing fails business readiness requirements")
            return False
        
        # Check minimum items
        if packing_result["total_items"] < self.validation["minimum_items_per_category"]["casual_tops"]:
            logging.error("Insufficient items for long-term trip")
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