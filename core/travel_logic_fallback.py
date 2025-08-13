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
    Rule-based travel packing logic as fallback when AI agents fail.
    Implements essential business school packing requirements with weight optimization.
    """
    
    def __init__(self):
        self.weights = AVERAGE_WEIGHTS
        self.constraints = WEIGHT_CONSTRAINTS
        self.business_reqs = BUSINESS_SCHOOL_REQUIREMENTS
        
    def generate_fallback_packing_list(self, trip_config: Dict, available_items: Dict) -> Optional[Dict]:
        """
        Generate a basic but functional packing list using rule-based logic.
        
        Args:
            trip_config: Trip configuration with destinations and requirements
            available_items: Available wardrobe items by category
            
        Returns:
            Dict with packing results or None if impossible
        """
        try:
            logging.info("🔧 Starting rule-based travel packing fallback...")
            
            # Analyze trip requirements
            trip_analysis = self._analyze_trip_requirements(trip_config)
            logging.info(f"   Trip analysis: {trip_analysis}")
            
            # Apply filters for trip suitability
            suitable_items = self._filter_trip_suitable_items(available_items, trip_analysis)
            
            if not suitable_items:
                logging.error("❌ No suitable items found for trip requirements")
                return None
            
            # Select items using rule-based prioritization
            selected_items = self._select_items_by_rules(suitable_items, trip_analysis)
            
            if not selected_items:
                logging.error("❌ Could not select sufficient items using rules")
                return None
            
            # Build comprehensive result
            packing_result = self._build_fallback_result(selected_items, trip_config)
            
            logging.info(f"✅ Rule-based packing completed: {len(selected_items)} items, {packing_result['total_weight_kg']}kg")
            
            return packing_result
            
        except Exception as e:
            logging.error(f"❌ Error in rule-based packing fallback: {e}", exc_info=True)
            return None
    
    def _analyze_trip_requirements(self, trip_config: Dict) -> Dict:
        """Analyze trip to determine packing requirements"""
        
        # Basic trip characteristics
        duration_months = trip_config["trip_overview"]["total_duration_months"]
        temp_range = trip_config["trip_overview"]["temperature_range"]
        destinations = trip_config["destinations"]
        
        # Determine climate needs
        needs_hot_weather = temp_range["max"] > 25
        needs_cold_weather = temp_range["min"] < 15
        extreme_range = temp_range["span"] > 25
        
        # Determine cultural requirements (check Dubai)
        needs_high_modesty = any(
            DESTINATIONS_CONFIG[dest["city"]]["cultural_context"]["modesty_level"] == "high"
            for dest in destinations
        )
        
        # Business requirements based on duration
        business_events_total = duration_months * self.business_reqs["formal_events"]["frequency_per_month"]
        business_casual_total = duration_months * 4 * self.business_reqs["business_casual_classes"]["frequency_per_week"]
        
        return {
            "duration_months": duration_months,
            "needs_hot_weather": needs_hot_weather,
            "needs_cold_weather": needs_cold_weather,
            "extreme_temperature_range": extreme_range,
            "needs_high_modesty": needs_high_modesty,
            "business_events_total": business_events_total,
            "business_casual_total": business_casual_total,
            "weight_budget": self.constraints["clothes_allocation"]["total_clothes_budget"]
        }
    
    def _filter_trip_suitable_items(self, available_items: Dict, trip_analysis: Dict) -> Dict:
        """Filter items suitable for this specific trip"""
        
        suitable_items = {}
        
        for category, items in available_items.items():
            category_suitable = []
            
            for item in items:
                if self._is_item_suitable_for_trip(item, trip_analysis):
                    category_suitable.append(item)
            
            if category_suitable:
                suitable_items[category] = category_suitable
        
        logging.info(f"   Filtered to {sum(len(items) for items in suitable_items.values())} suitable items")
        return suitable_items
    
    def _is_item_suitable_for_trip(self, item: Dict, trip_analysis: Dict) -> bool:
        """Check if individual item is suitable for trip"""
        
        # Must be clean
        if item.get('washed', '').lower() != 'done':
            return False
        
        # Check weather suitability
        weather_tags = [w.lower() for w in item.get('weather', [])]
        
        if trip_analysis["needs_hot_weather"] and trip_analysis["needs_cold_weather"]:
            # Need versatile items for extreme range
            if weather_tags and len(weather_tags) == 1:
                # Single weather items less useful for extreme range
                if not (weather_tags[0] in ['hot'] and trip_analysis["extreme_temperature_range"]):
                    pass  # Still include hot items for extreme heat destinations
        
        # Check cultural requirements (for Dubai)
        if trip_analysis["needs_high_modesty"]:
            category = item.get('category', '')
            # Prefer modest categories
            if category in ['Shirt', 'Chinos', 'Pants', 'Polo']:
                return True
            elif category in ['Shorts', 'Tank Top']:
                return False  # Exclude non-modest items for Dubai
        
        # Prefer trip-worthy items
        trip_worthy_bonus = item.get('trip_worthy', False)
        
        return True
    
    def _select_items_by_rules(self, suitable_items: Dict, trip_analysis: Dict) -> List[Dict]:
        """Select items using rule-based prioritization"""
        
        selected = []
        current_weight = 0
        weight_budget = trip_analysis["weight_budget"]
        
        # Priority 1: Business Essentials (non-negotiable)
        business_essentials = self._select_business_essentials(suitable_items, trip_analysis)
        for item in business_essentials:
            weight = self.weights.get(item['category'], 0.5)
            if current_weight + weight <= weight_budget:
                selected.append(item)
                current_weight += weight
                logging.debug(f"   ✅ Business essential: {item['item']} ({weight}kg)")
        
        # Priority 2: Climate Essentials 
        climate_essentials = self._select_climate_essentials(suitable_items, trip_analysis, selected)
        for item in climate_essentials:
            weight = self.weights.get(item['category'], 0.5)
            if current_weight + weight <= weight_budget:
                selected.append(item)
                current_weight += weight
                logging.debug(f"   ✅ Climate essential: {item['item']} ({weight}kg)")
        
        # Priority 3: Versatile Basics (fill remaining weight)
        versatile_items = self._select_versatile_basics(suitable_items, trip_analysis, selected)
        for item in versatile_items:
            weight = self.weights.get(item['category'], 0.5)
            if current_weight + weight <= weight_budget:
                selected.append(item)
                current_weight += weight
                logging.debug(f"   ✅ Versatile basic: {item['item']} ({weight}kg)")
            else:
                break  # Weight budget exceeded
        
        logging.info(f"   Rule-based selection: {len(selected)} items, {current_weight:.2f}kg")
        return selected
    
    def _select_business_essentials(self, suitable_items: Dict, trip_analysis: Dict) -> List[Dict]:
        """Select non-negotiable business items"""
        essentials = []
        
        # Must have: 2 business suits
        suits = suitable_items.get('Suit', [])
        essentials.extend(suits[:2])
        
        # Must have: 2 pairs dress shoes
        dress_shoes = suitable_items.get('Shoes', [])
        essentials.extend(dress_shoes[:2])
        
        # Must have: 5+ dress shirts
        dress_shirts = suitable_items.get('Shirt', [])
        # Prioritize business/formal aesthetic shirts
        business_shirts = [s for s in dress_shirts if any('business' in a.lower() or 'formal' in a.lower() or 'minimalist' in a.lower() for a in s.get('aesthetic', []))]
        other_shirts = [s for s in dress_shirts if s not in business_shirts]
        essentials.extend(business_shirts[:3])
        essentials.extend(other_shirts[:2])
        
        logging.info(f"   Business essentials: {len(essentials)} items")
        return essentials
    
    def _select_climate_essentials(self, suitable_items: Dict, trip_analysis: Dict, already_selected: List[Dict]) -> List[Dict]:
        """Select items essential for climate coverage"""
        climate_items = []
        selected_ids = {item['id'] for item in already_selected}
        
        # Hot weather essentials
        if trip_analysis["needs_hot_weather"]:
            # Light t-shirts
            t_shirts = [item for item in suitable_items.get('T-shirt', []) if item['id'] not in selected_ids]
            climate_items.extend(t_shirts[:3])
            
            # Lightweight chinos/pants
            chinos = [item for item in suitable_items.get('Chinos', []) if item['id'] not in selected_ids]
            climate_items.extend(chinos[:2])
        
        # Cold weather essentials
        if trip_analysis["needs_cold_weather"]:
            # Warm layers
            hoodies = [item for item in suitable_items.get('Hoodie', []) if item['id'] not in selected_ids]
            climate_items.extend(hoodies[:1])
            
            jackets = [item for item in suitable_items.get('Jacket', []) if item['id'] not in selected_ids]
            climate_items.extend(jackets[:1])
            
            # Long pants
            jeans = [item for item in suitable_items.get('Jeans', []) if item['id'] not in selected_ids]
            climate_items.extend(jeans[:2])
        
        logging.info(f"   Climate essentials: {len(climate_items)} items")
        return climate_items
    
    def _select_versatile_basics(self, suitable_items: Dict, trip_analysis: Dict, already_selected: List[Dict]) -> List[Dict]:
        """Select versatile items to fill remaining weight budget"""
        versatile_items = []
        selected_ids = {item['id'] for item in already_selected}
        
        # Sort categories by versatility
        versatile_categories = [
            ('Polo', 3),      # Very versatile for business casual
            ('Sneakers', 2),  # Essential footwear
            ('Chinos', 2),    # Versatile bottoms
            ('T-shirt', 4),   # Casual basics
            ('Jeans', 1),     # Casual bottoms
        ]
        
        for category, max_count in versatile_categories:
            if category in suitable_items:
                available = [item for item in suitable_items[category] if item['id'] not in selected_ids]
                # Sort by trip-worthy status and aesthetic count
                available.sort(key=lambda x: (x.get('trip_worthy', False), len(x.get('aesthetic', []))), reverse=True)
                versatile_items.extend(available[:max_count])
        
        logging.info(f"   Versatile basics: {len(versatile_items)} items")
        return versatile_items
    
    def _build_fallback_result(self, selected_items: List[Dict], trip_config: Dict) -> Dict:
        """Build comprehensive result structure matching AI agent output"""
        
        # Calculate basic metrics
        total_weight = sum(self.weights.get(item['category'], 0.5) for item in selected_items)
        total_items = len(selected_items)
        weight_efficiency = round(total_items / total_weight, 1) if total_weight > 0 else 0
        
        # Bag allocation using same logic as AI agent
        bag_allocation = self._allocate_items_to_bags_fallback(selected_items)
        
        # Outfit analysis
        outfit_analysis = self._analyze_outfit_possibilities_fallback(selected_items)
        
        # Assessments
        business_readiness = self._assess_business_readiness_fallback(selected_items)
        climate_coverage = self._assess_climate_coverage_fallback(selected_items, trip_config)
        cultural_compliance = self._assess_cultural_compliance_fallback(selected_items)
        
        # Build result structure
        result = {
            "selected_items": selected_items,
            "total_items": total_items,
            "total_weight_kg": round(total_weight, 2),
            "weight_efficiency": weight_efficiency,
            "bag_allocation": bag_allocation,
            "outfit_analysis": outfit_analysis,
            "business_readiness": business_readiness,
            "climate_coverage": climate_coverage,
            "cultural_compliance": cultural_compliance,
            "packing_guide": self._generate_basic_packing_guide(),
            "trip_tips": self._generate_basic_trip_tips(trip_config),
            "generation_method": "rule_based_fallback"
        }
        
        return result
    
    def _allocate_items_to_bags_fallback(self, selected_items: List[Dict]) -> Dict:
        """Simple bag allocation logic"""
        checked_items = []
        cabin_items = []
        checked_weight = 0
        cabin_weight = 0
        
        for item in selected_items:
            weight = self.weights.get(item['category'], 0.5)
            category = item['category']
            
            # Heavy items go to checked bag
            if category in ['Suit', 'Shoes'] or weight > 0.8:
                checked_items.append(item)
                checked_weight += weight
            # Light essentials go to cabin
            elif category in ['T-shirt', 'Polo'] and len(cabin_items) < 4:
                cabin_items.append(item)
                cabin_weight += weight
            # Everything else to checked
            else:
                checked_items.append(item)
                checked_weight += weight
        
        return {
            "checked_bag": {
                "items": checked_items,
                "weight_kg": round(checked_weight, 2),
                "space_utilization": round(checked_weight / 15 * 100, 1)
            },
            "cabin_bag": {
                "items": cabin_items,
                "weight_kg": round(cabin_weight, 2),
                "space_utilization": round(cabin_weight / 3 * 100, 1)
            },
            "strategy_notes": [
                "Rule-based allocation: Heavy items in checked bag",
                "Essential backups in cabin bag",
                "Simple weight distribution strategy"
            ]
        }
    
    def _analyze_outfit_possibilities_fallback(self, selected_items: List[Dict]) -> Dict:
        """Basic outfit analysis"""
        categories = {}
        for item in selected_items:
            cat = item['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        # Simple counting
        suits = categories.get('Suit', 0)
        dress_shoes = categories.get('Shoes', 0)
        shirts = categories.get('Shirt', 0)
        business_formal = min(suits, dress_shoes, shirts)
        
        chinos = categories.get('Chinos', 0)
        polos = categories.get('Polo', 0)
        sneakers = categories.get('Sneakers', 0)
        business_casual = min(chinos + categories.get('Pants', 0), shirts + polos, dress_shoes + sneakers) * 2
        
        casual = min(
            categories.get('Jeans', 0) + chinos,
            categories.get('T-shirt', 0) + polos,
            sneakers
        ) * 3
        
        return {
            "business_formal_outfits": business_formal,
            "business_casual_outfits": business_casual,
            "casual_outfits": casual,
            "total_outfit_combinations": business_formal + business_casual + casual,
            "category_breakdown": categories
        }
    
    def _assess_business_readiness_fallback(self, selected_items: List[Dict]) -> Dict:
        """Simple business readiness assessment"""
        suits = len([i for i in selected_items if i['category'] == 'Suit'])
        dress_shoes = len([i for i in selected_items if i['category'] == 'Shoes'])
        business_shirts = len([i for i in selected_items if i['category'] == 'Shirt'])
        
        readiness_score = min(suits / 2, 1.0) * 0.4 + min(dress_shoes / 2, 1.0) * 0.3 + min(business_shirts / 5, 1.0) * 0.3
        
        return {
            "readiness_score": round(readiness_score, 2),
            "suits_count": suits,
            "dress_shoes_count": dress_shoes,
            "business_shirts_count": business_shirts,
            "meets_requirements": readiness_score >= 0.7  # Slightly lower threshold for fallback
        }
    
    def _assess_climate_coverage_fallback(self, selected_items: List[Dict], trip_config: Dict) -> Dict:
        """Simple climate coverage assessment"""
        temp_range = trip_config["trip_overview"]["temperature_range"]
        
        hot_items = len([i for i in selected_items if 'hot' in [w.lower() for w in i.get('weather', [])]])
        cold_items = len([i for i in selected_items if 'cold' in [w.lower() for w in i.get('weather', [])]])
        versatile_items = len([i for i in selected_items if len(i.get('weather', [])) == 0 or len(i.get('weather', [])) >= 2])
        
        return {
            "hot_weather_coverage": hot_items,
            "cold_weather_coverage": cold_items,
            "versatile_items": versatile_items,
            "temperature_range_covered": f"{temp_range['min']}°C - {temp_range['max']}°C",
            "coverage_adequacy": "good" if versatile_items > 8 else "basic"
        }
    
    def _assess_cultural_compliance_fallback(self, selected_items: List[Dict]) -> Dict:
        """Simple cultural compliance assessment"""
        modest_items = len([i for i in selected_items if i['category'] in ['Shirt', 'Chinos', 'Pants', 'Polo']])
        total_items = len(selected_items)
        compliance_score = modest_items / total_items if total_items > 0 else 0
        
        return {
            "compliance_score": round(compliance_score, 2),
            "modest_items_count": modest_items,
            "total_items": total_items,
            "dubai_ready": compliance_score >= 0.6,  # Lower threshold for fallback
            "recommendations": [
                "Rule-based selection prioritized modest items",
                "Manual review recommended for Dubai requirements"
            ]
        }
    
    def _generate_basic_packing_guide(self) -> Dict:
        """Generate basic packing guide"""
        return {
            "packing_techniques": [
                "Roll casual items to save space",
                "Fold formal items with care",
                "Use packing cubes for organization",
                "Place heavy items at bottom of bag"
            ],
            "travel_day_strategy": {
                "wear_during_travel": ["Heavy shoes", "Thickest jacket", "Business casual outfit"],
                "cabin_essentials": ["Change of clothes", "Essential documents", "Medications"]
            }
        }
    
    def _generate_basic_trip_tips(self, trip_config: Dict) -> Dict:
        """Generate basic destination tips"""
        tips = {}
        for dest in trip_config["destinations"]:
            city = dest["city"]
            if city in DESTINATIONS_CONFIG:
                city_config = DESTINATIONS_CONFIG[city]
                tips[city] = {
                    "cultural_tips": [f"Modesty level: {city_config['cultural_context']['modesty_level']}"],
                    "climate_preparation": [f"Climate: {city_config['climate_profile']}"],
                    "practical_advice": ["Pack according to local customs"]
                }
        return tips

# Create global instance
travel_logic_fallback = TravelLogicFallback()