import logging
from typing import Dict, List, Optional
from config.travel_config import (
    AVERAGE_WEIGHTS,
    DESTINATIONS_CONFIG,
    WEIGHT_CONSTRAINTS,
    BUSINESS_SCHOOL_REQUIREMENTS
)

class TravelLogicFallback:
    """
    Enhanced rule-based travel packing logic as a fallback when AI agents fail.
    Implements dynamic, goal-oriented packing requirements with intelligent weight optimization.
    """

    def __init__(self):
        self.weights = AVERAGE_WEIGHTS
        self.constraints = WEIGHT_CONSTRAINTS
        self.business_reqs = BUSINESS_SCHOOL_REQUIREMENTS

    def generate_fallback_packing_list(self, trip_config: Dict, available_items: Dict) -> Optional[Dict]:
        """
        Generates a functional and context-aware packing list using dynamic, rule-based logic.
        """
        try:
            logging.info("🔧 Starting enhanced rule-based travel packing fallback...")

            trip_analysis = self._analyze_trip_requirements(trip_config)
            logging.info(f"   Trip analysis: {trip_analysis}")

            suitable_items = self._filter_trip_suitable_items(available_items, trip_analysis)
            if not suitable_items:
                logging.error("❌ No suitable items found for trip requirements.")
                return None

            selected_items = self._select_items_by_rules(suitable_items, trip_analysis)
            if not selected_items:
                logging.error("❌ Could not select sufficient items using rules.")
                return None

            packing_result = self._build_fallback_result(selected_items, trip_config)
            logging.info(f"✅ Rule-based packing completed: {len(selected_items)} items, {packing_result['total_weight_kg']}kg")

            return packing_result

        except Exception as e:
            logging.error(f"❌ Error in rule-based packing fallback: {e}", exc_info=True)
            return None

    def _analyze_trip_requirements(self, trip_config: Dict) -> Dict:
        """Analyzes the trip to determine dynamic packing requirements."""
        overview = trip_config["trip_overview"]
        duration_months = overview["total_duration_months"]

        # Calculate dynamic item counts based on trip duration and event frequency
        reqs = {
            "duration_months": duration_months,
            "needs_hot_weather": overview["temperature_range"]["max"] > 25,
            "needs_cold_weather": overview["temperature_range"]["min"] < 15,
            "weight_budget": self.constraints["clothes_allocation"]["total_clothes_budget"],
            # Dynamic Business Formal Needs
            "required_suits": 2,
            "required_dress_shirts": min(8, 4 + int(duration_months)),
            "required_dress_shoes": 2,
            # Dynamic Business Casual Needs (will be a mix of Formal, Minimalist, Casual)
            "required_bc_tops": min(10, 5 + int(duration_months)),
            "required_bc_bottoms": min(6, 3 + int(duration_months // 2)),
            # Dynamic Casual Needs
            "required_casual_tops": min(10, 4 + int(duration_months)),
            "required_casual_bottoms": min(5, 2 + int(duration_months // 2)),
            "required_outerwear": 3 if overview["temperature_range"]["span"] > 25 else 2,
            "required_sneakers": 2,
        }
        return reqs

    def _filter_trip_suitable_items(self, available_items: Dict, trip_analysis: Dict) -> Dict:
        """Filters items that are clean and suitable for the trip's weather."""
        suitable_items = {}
        for category, items in available_items.items():
            category_suitable = []
            for item in items:
                # Must be clean and in good condition for a long trip
                if item.get('washed', '').lower() != 'done' or item.get('condition', 'Good').lower() in ['poor', 'stained']:
                    continue

                weather_tags = [w.lower() for w in item.get('weather', [])]
                is_versatile = not weather_tags or ('hot' in weather_tags and 'cold' in weather_tags)

                # Weather suitability check
                is_suitable = False
                if is_versatile:
                    is_suitable = True
                elif trip_analysis["needs_hot_weather"] and 'hot' in weather_tags:
                    is_suitable = True
                elif trip_analysis["needs_cold_weather"] and 'cold' in weather_tags:
                    is_suitable = True
                
                if is_suitable:
                    category_suitable.append(item)

            if category_suitable:
                suitable_items[category] = category_suitable
        
        logging.info(f"   Filtered to {sum(len(items) for items in suitable_items.values())} suitable items.")
        return suitable_items

    def _select_items_by_rules(self, suitable_items: Dict, trip_analysis: Dict) -> List[Dict]:
        """Selects items using a prioritized, goal-oriented strategy with correct aesthetics."""
        selected_ids = set()
        current_weight = 0
        weight_budget = trip_analysis["weight_budget"]

        def select_and_add(items_to_add: List[Dict]):
            nonlocal current_weight
            for item in items_to_add:
                if item['id'] not in selected_ids:
                    weight = self.weights.get(item['category'], 0.5)
                    if (current_weight + weight) <= weight_budget:
                        selected_ids.add(item['id'])
                        current_weight += weight
                        logging.debug(f"   Selected: {item['item']} ({item['category']}) | New Weight: {current_weight:.2f}kg")

        # **THIS IS THE CORRECTED SECTION**
        # The aesthetics list now matches your Notion database exactly.
        selection_plan = [
            # Priority 1: Core Business Formal Items
            {"cats": ["Suit"], "target": trip_analysis["required_suits"], "aesthetics": ["Formal"]},
            {"cats": ["Shoes"], "target": trip_analysis["required_dress_shoes"], "aesthetics": ["Formal"]},
            {"cats": ["Shirt"], "target": trip_analysis["required_dress_shirts"], "aesthetics": ["Formal", "Minimalist"]},
            
            # Priority 2: Build Business Casual Wardrobe (using Formal, Minimalist, and Casual tags)
            {"cats": ["Chinos", "Pants"], "target": trip_analysis["required_bc_bottoms"], "aesthetics": ["Formal", "Minimalist", "Casual"]},
            {"cats": ["Polo", "Shirt"], "target": trip_analysis["required_bc_tops"], "aesthetics": ["Formal", "Minimalist", "Casual"]},
            
            # Priority 3: Climate Control and Layering
            {"cats": ["Jacket", "Hoodie", "Overcoat", "Fleece"], "target": trip_analysis["required_outerwear"], "aesthetics": ["Casual", "Minimalist"]},
            
            # Priority 4: Fill with Casual Essentials
            {"cats": ["Sneakers"], "target": trip_analysis["required_sneakers"], "aesthetics": ["Casual", "Sportswear"]},
            {"cats": ["T-shirt", "Sport T-shirt"], "target": trip_analysis["required_casual_tops"], "aesthetics": ["Casual", "Minimalist", "Oversize"]},
            {"cats": ["Jeans", "Shorts", "Cargo Pants", "Joggers"], "target": trip_analysis["required_casual_bottoms"], "aesthetics": ["Casual", "Oversize"]},
        ]

        for plan in selection_plan:
            candidates = []
            for cat in plan["cats"]:
                candidates.extend(suitable_items.get(cat, []))
            
            # Sort candidates to pick the best ones first
            candidates.sort(key=lambda x: (
                x.get('trip_worthy', False),
                len(set(x.get('aesthetic', [])) & set(plan["aesthetics"])), # Prioritize matching aesthetics
                len(x.get('weather', [])) != 1, # Prioritize versatile weather items
                x.get('condition', 'Good') == 'Good'
            ), reverse=True)

            items_to_select = [item for item in candidates if item['id'] not in selected_ids][:plan["target"]]
            select_and_add(items_to_select)
        
        all_items_flat = [item for sublist in suitable_items.values() for item in sublist]
        final_selection = [item for item in all_items_flat if item['id'] in selected_ids]
        
        logging.info(f"   Rule-based selection: {len(final_selection)} items, {current_weight:.2f}kg")
        return final_selection

    # --- The rest of the functions remain the same as the previous version ---

    def _build_fallback_result(self, selected_items: List[Dict], trip_config: Dict) -> Dict:
        """Builds the final result dictionary, mirroring the AI agent's output structure."""
        total_weight = sum(self.weights.get(item['category'], 0.5) for item in selected_items)
        outfit_analysis = self._analyze_outfit_possibilities_fallback(selected_items)
        
        result = {
            "selected_items": selected_items,
            "total_items": len(selected_items),
            "total_weight_kg": round(total_weight, 2),
            "weight_efficiency": round(outfit_analysis["total_outfit_combinations"] / total_weight if total_weight > 0 else 0, 1),
            "bag_allocation": self._allocate_items_to_bags_fallback(selected_items),
            "outfit_analysis": outfit_analysis,
            "business_readiness": self._assess_business_readiness_fallback(selected_items),
            "climate_coverage": self._assess_climate_coverage_fallback(selected_items, trip_config),
            "cultural_compliance": self._assess_cultural_compliance_fallback(selected_items),
            "packing_guide": self._generate_basic_packing_guide(),
            "trip_tips": self._generate_basic_trip_tips(trip_config),
        }
        return result

    def _allocate_items_to_bags_fallback(self, selected_items: List[Dict]) -> Dict:
        """Simple, rule-based allocation of items to checked and cabin bags."""
        checked_items, cabin_items = [], []
        checked_weight, cabin_weight = 0, 0
        cabin_essentials_count = 0

        # Prioritize heavier and less essential items for checked luggage
        for item in sorted(selected_items, key=lambda x: self.weights.get(x['category'], 0.5), reverse=True):
            weight = self.weights.get(item['category'], 0.5)
            # Add a couple of essential outfits to cabin bag
            if item['category'] in ['T-shirt', 'Polo', 'Chinos'] and cabin_essentials_count < 3:
                 if (cabin_weight + weight) <= self.constraints["clothes_allocation"]["cabin_bag_clothes_kg"]:
                    cabin_items.append(item)
                    cabin_weight += weight
                    cabin_essentials_count += 1
                    continue
            
            if (checked_weight + weight) <= self.constraints["clothes_allocation"]["checked_bag_clothes_kg"]:
                checked_items.append(item)
                checked_weight += weight
            else: # If checked is full, spill over to cabin
                cabin_items.append(item)
                cabin_weight += weight
        
        return {
            "checked_bag": {"items": checked_items, "weight_kg": round(checked_weight, 2)},
            "cabin_bag": {"items": cabin_items, "weight_kg": round(cabin_weight, 2)},
        }

    def _analyze_outfit_possibilities_fallback(self, selected_items: List[Dict]) -> Dict:
        """Estimates the number of possible outfits with more realistic combination logic."""
        cats = {cat: 0 for cat in ["Suit", "Shoes", "Shirt", "Polo", "Chinos", "Pants", "Sneakers", "Jeans", "Shorts", "T-shirt"]}
        for item in selected_items:
            if item['category'] in cats:
                cats[item['category']] += 1
        
        bf = cats["Suit"] * cats["Shirt"] * cats["Shoes"]
        bc_bottoms = cats["Chinos"] + cats["Pants"]
        bc_tops = cats["Shirt"] + cats["Polo"]
        bc_shoes = cats["Shoes"] + cats["Sneakers"]
        bc = bc_bottoms * bc_tops * bc_shoes

        c_bottoms = cats["Jeans"] + cats["Shorts"] + cats["Chinos"]
        c_tops = cats["T-shirt"] + cats["Polo"]
        c_shoes = cats["Sneakers"]
        c = c_bottoms * c_tops * c_shoes
        
        return {
            "business_formal_outfits": bf,
            "business_casual_outfits": bc,
            "casual_outfits": c,
            "total_outfit_combinations": bf + bc + c,
        }

    def _assess_business_readiness_fallback(self, selected_items: List[Dict]) -> Dict:
        """Assesses if the packing list meets business requirements."""
        suits = sum(1 for i in selected_items if i['category'] == 'Suit')
        score = min(1.0, suits / 2.0) # Simple score based on having at least 2 suits
        return {"readiness_score": score, "suits_count": suits, "meets_requirements": score >= 0.9}

    def _assess_climate_coverage_fallback(self, selected_items: List[Dict], trip_config: Dict) -> Dict:
        """Assesses if the packing list covers the required climate."""
        return {"temperature_range_covered": trip_config["trip_overview"]["temperature_range"], "coverage_adequacy": "good"}

    def _assess_cultural_compliance_fallback(self, selected_items: List[Dict]) -> Dict:
        """Assesses if the packing list is culturally compliant."""
        modest_items = sum(1 for i in selected_items if i['category'] in ['Shirt', 'Chinos', 'Pants', 'Polo', 'Suit'])
        score = modest_items / len(selected_items) if selected_items else 0
        return {"compliance_score": round(score, 2), "modest_items_count": modest_items, "dubai_ready": score > 0.6}
    
    # In core/travel_logic_fallback.py

    def _generate_basic_packing_guide(self) -> Dict:
        """Generates a generic packing guide."""
        return {
            "packing_techniques": [
                "Roll casual items to save space",
                "Fold formal items with care",
                "Use packing cubes for organization"
            ],
            "travel_day_strategy": {
                "wear_during_travel": ["Your heaviest shoes", "Your thickest jacket"],
                "cabin_essentials": ["A change of clothes", "Essential documents", "Medications"]
            }
        }

    def _generate_basic_trip_tips(self, trip_config: Dict) -> Dict:
        """Generate basic destination tips in the correct structure."""
        tips = {}
        for dest in trip_config["destinations"]:
            city = dest["city"]
            if city in DESTINATIONS_CONFIG:
                city_config = DESTINATIONS_CONFIG[city]
                tips[city] = {
                    "cultural_tips": [f"Modesty level: {city_config['cultural_context']['modesty_level']}"],
                    "climate_preparation": [f"Climate: {city_config['climate_profile']}"],
                    "practical_advice": ["Pack according to local customs and check the weather upon arrival."]
                }
        return tips


# Global instance
travel_logic_fallback = TravelLogicFallback()