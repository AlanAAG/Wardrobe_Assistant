# travel_config.py
"""
Complete configuration for travel packing system.
Scalable and destination-agnostic configuration for multi-city business travel packing optimization.
"""

# Weight constants for all clothing categories (in kg)
AVERAGE_WEIGHTS = {
    "T-shirt": 0.18,        # Light cotton T-shirt
    "Jeans": 0.60,          # Denim is heavy
    "Shoes": 1.20,          # Standard leather or dress shoes
    "Hoodie": 0.70,         # Cotton blend, without heavy fleece
    "Shirt": 0.25,          # Button-up shirt
    "Suit": 1.50,           # Jacket + pants
    "Chinos": 0.50,         # Lighter than jeans
    "Polo": 0.30,           # Thicker than T-shirt
    "Joggers": 0.45,        # Cotton/polyester
    "Jacket": 0.80,         # Lightweight jacket
    "Crewneck": 0.55,       # Midweight fleece sweatshirt
    "Fleece": 0.60,         # Full-zip fleece jacket
    "Sport T-shirt": 0.20,  # Dry-fit type
    "Overcoat": 1.20,       # Heavy wool or padded
    "Overshirt": 0.40,      # Thick cotton flannel
    "Cargo Pants": 0.65,    # Extra pockets add weight
    "Pants": 0.55,          # General category, e.g., slacks
    "Shorts": 0.25,         # Cotton or athletic
    "Sneakers": 1.00        # Running shoes or casual sneakers
}

# Realistic weight constraints for airline travel
WEIGHT_CONSTRAINTS = {
    # Total luggage capacity
    "checked_bag_limit_kg": 23,
    "cabin_bag_limit_kg": 10,
    "total_luggage_capacity": 33,  # 23 + 10
    
    # Realistic allocation for clothes vs other items
    "clothes_allocation": {
        "checked_bag_clothes_kg": 15,    # 8kg reserved for electronics, shoes, toiletries, books, etc.
        "cabin_bag_clothes_kg": 3,       # 7kg reserved for laptop, documents, essentials, chargers
        "total_clothes_budget": 18       # 15 + 3 = realistic clothes weight limit
    },
    
    # Non-clothes weight estimates (for reference and prompt context)
    "non_clothes_estimates": {
        "electronics": 4,        # Laptop, chargers, adapters, phone accessories
        "toiletries": 2,         # Shampoo, skincare, medications, etc.
        "books_documents": 2,    # Business school materials, documents
        "shoes_bags": 3,         # Extra shoes, bags, accessories
        "miscellaneous": 2       # Gifts, souvenirs buffer, etc.
    },
    
    # Strategic packing rules
    "heavy_items_placement": {
        "shoes": "checked",           # Heavy shoes in checked bag
        "formal_suits": "checked",    # Business suits in checked bag  
        "electronics": "cabin",       # Electronics always in cabin
        "daily_essentials": "cabin"   # 2-3 day clothes backup in cabin
    },
    
    # Optimization targets
    "target_efficiency_ratio": 12,   # 12 complete outfits per kg of clothes
    "heavy_clothing_limit": 2,       # Max 2 clothing items over 1kg (suits, heavy shoes)
    "essential_weight_reserve": 1,    # 1kg buffer for unexpected items
    "minimum_outfit_count": 15        # Minimum viable outfits for 8-month trip
}

# Scalable destination configurations - easily extensible for new cities
DESTINATIONS_CONFIG = {
    "dubai": {
        "climate_profile": "hot_arid",
        "months": ["september", "october", "november", "december"],
        
        # Seasonal weather patterns
        "seasons": {
            "september": {
                "temp_range": [28, 42], 
                "weather": "extremely_hot_dry", 
                "rain_prob": 0.05,
                "humidity": "moderate",
                "cultural_events": "high_season"
            },
            "october": {
                "temp_range": [25, 38], 
                "weather": "hot_dry", 
                "rain_prob": 0.1,
                "humidity": "moderate",
                "cultural_events": "peak_season"
            },
            "november": {
                "temp_range": [22, 32], 
                "weather": "warm_dry", 
                "rain_prob": 0.1,
                "humidity": "low",
                "cultural_events": "optimal_weather"
            },
            "december": {
                "temp_range": [18, 28], 
                "weather": "mild_dry", 
                "rain_prob": 0.1,
                "humidity": "low",
                "cultural_events": "tourist_season"
            }
        },
        
        # Cultural and business requirements
        "cultural_context": {
            "modesty_level": "high",
            "business_formality": "very_high",
            "color_preferences": ["neutral", "conservative", "navy", "white", "beige"],
            "color_avoid": ["bright_colors", "revealing_cuts"],
            "fabric_requirements": ["breathable", "modest_coverage", "wrinkle_resistant"],
            "sleeve_requirements": "long_sleeve_preferred",
            "length_requirements": "full_coverage"
        },
        
        # Weight optimization priorities
        "weight_priorities": {
            "ultra_lightweight_bonus": 0.3,  # Major bonus for items under 0.2kg
            "breathable_essential": True,
            "formal_weight_budget": 4,        # Max 4kg for business formal needs (suits, dress shoes)
            "casual_weight_budget": 8,        # 8kg for versatile casual items
            "cultural_compliance_weight": 1   # 1kg buffer for cultural appropriateness
        },
        
        # Activity requirements
        "activity_types": [
            "business_formal_events",
            "business_casual_networking", 
            "cultural_site_visits",
            "beach_activities",
            "shopping_districts",
            "fine_dining",
            "academic_presentations"
        ],
        
        # Climate-specific recommendations
        "climate_recommendations": {
            "essential_items": ["lightweight_blazer", "breathable_dress_shirts", "cotton_chinos"],
            "avoid_items": ["heavy_wool", "dark_colors", "thick_fabrics"],
            "fabric_priorities": ["cotton", "linen", "bamboo", "moisture_wicking"],
            "color_strategy": "light_colors_heat_reflection"
        }
    },
    
    "gurgaon": {
        "climate_profile": "subtropical_continental",
        "months": ["january", "february", "march", "april", "may"],
        
        # Seasonal weather patterns (wide temperature variation)
        "seasons": {
            "january": {
                "temp_range": [6, 20], 
                "weather": "cool_dry", 
                "rain_prob": 0.1,
                "humidity": "low",
                "air_quality": "poor_winter"
            },
            "february": {
                "temp_range": [9, 24], 
                "weather": "mild_dry", 
                "rain_prob": 0.1,
                "humidity": "low",
                "air_quality": "moderate"
            },
            "march": {
                "temp_range": [14, 29], 
                "weather": "warm_dry", 
                "rain_prob": 0.15,
                "humidity": "moderate",
                "air_quality": "moderate"
            },
            "april": {
                "temp_range": [20, 36], 
                "weather": "hot_dry", 
                "rain_prob": 0.2,
                "humidity": "increasing",
                "air_quality": "poor_heat"
            },
            "may": {
                "temp_range": [25, 40], 
                "weather": "extremely_hot_humid", 
                "rain_prob": 0.4,
                "humidity": "high",
                "monsoon_prep": True,
                "air_quality": "poor_pre_monsoon"
            }
        },
        
        # Cultural and business requirements
        "cultural_context": {
            "modesty_level": "medium_high",
            "business_formality": "high",
            "color_preferences": ["any", "vibrant_acceptable"],
            "fabric_requirements": ["versatile", "monsoon_ready", "layerable"],
            "seasonal_adaptation": "essential"
        },
        
        # Weight optimization priorities
        "weight_priorities": {
            "versatile_weight_priority": 0.2,   # Bonus for items that work across seasons
            "weather_protection_budget": 2.5,   # 2.5kg for rain/cold protection
            "formal_weight_budget": 4,          # 4kg for business requirements
            "casual_weight_budget": 8,          # 8kg for daily wear
            "seasonal_transition_bonus": 0.15   # Bonus for items that work in multiple seasons
        },
        
        # Activity requirements
        "activity_types": [
            "business_formal_presentations",
            "business_casual_classes",
            "academic_conferences",
            "networking_events",
            "cultural_exploration",
            "monsoon_weather_activities",
            "air_quality_considerations"
        ],
        
        # Climate-specific recommendations
        "climate_recommendations": {
            "essential_items": ["layerable_pieces", "rain_jacket", "versatile_blazer", "breathable_base_layers"],
            "seasonal_musts": ["warm_layer_jan_feb", "rain_protection_may", "heat_management_apr_may"],
            "fabric_priorities": ["quick_dry", "versatile_cotton", "wrinkle_resistant", "breathable_synthetic"],
            "layering_strategy": "essential_for_temperature_range"
        }
    }
}

# Business school specific requirements
BUSINESS_SCHOOL_REQUIREMENTS = {
    "formal_events": {
        "frequency_per_month": 4,
        "required_items": ["business_suit", "dress_shoes", "formal_shirts", "ties_optional"],
        "cultural_adaptation": "destination_specific"
    },
    
    "business_casual_classes": {
        "frequency_per_week": 15,  # 3 per day, 5 days
        "required_items": ["chinos", "polo_or_shirt", "casual_blazer", "loafers_or_dress_sneakers"],
        "versatility_priority": "high"
    },
    
    "casual_social": {
        "frequency_per_week": 10,
        "required_items": ["t_shirts", "jeans_or_shorts", "sneakers", "casual_layers"],
        "climate_adaptation": "essential"
    },
    
    "presentations": {
        "frequency_per_month": 8,
        "required_items": ["business_formal", "confident_appearance", "wrinkle_free"],
        "backup_requirements": "essential"
    }
}

# Outfit multiplication matrix - for calculating versatility scores
OUTFIT_COMBINATIONS = {
    "business_formal": {
        "required": ["suit_jacket", "dress_pants", "dress_shirt", "dress_shoes"],
        "optional": ["tie", "formal_belt", "cufflinks"],
        "combinations_per_set": 1
    },
    
    "business_casual": {
        "required": ["chinos_or_dress_pants", "polo_or_casual_shirt", "dress_shoes_or_loafers"],
        "optional": ["blazer", "casual_belt"],
        "combinations_per_set": 3  # Multiple mixing options
    },
    
    "smart_casual": {
        "required": ["jeans_or_chinos", "shirt_or_polo", "sneakers_or_casual_shoes"],
        "optional": ["casual_jacket", "accessories"],
        "combinations_per_set": 6  # High mixing potential
    },
    
    "casual": {
        "required": ["casual_bottom", "casual_top", "casual_footwear"],
        "optional": ["layer", "accessories"],
        "combinations_per_set": 12  # Maximum versatility
    },
    
    "climate_specific": {
        "hot_weather": ["shorts", "t_shirt", "sneakers"],
        "cold_weather": ["long_pants", "long_sleeve", "closed_shoes", "layer"],
        "rain_ready": ["rain_resistant", "quick_dry", "appropriate_footwear"]
    }
}

# Future destination template (for easy expansion)
DESTINATION_TEMPLATE = {
    "city_name": {
        "climate_profile": "climate_type",
        "months": ["list", "of", "months"],
        "seasons": {
            "month": {
                "temp_range": [min, max],
                "weather": "description",
                "rain_prob": 0.0,
                "humidity": "level",
                "special_considerations": "notes"
            }
        },
        "cultural_context": {
            "modesty_level": "level",
            "business_formality": "level",
            "color_preferences": ["list"],
            "fabric_requirements": ["list"]
        },
        "weight_priorities": {
            "specific_bonus": 0.0,
            "formal_weight_budget": 0,
            "casual_weight_budget": 0
        },
        "activity_types": ["list", "of", "activities"],
        "climate_recommendations": {
            "essential_items": ["list"],
            "avoid_items": ["list"],
            "fabric_priorities": ["list"]
        }
    }
}

# Error handling and validation
VALIDATION_RULES = {
    "minimum_items_per_category": {
        "business_formal": 2,  # At least 2 business outfits
        "casual_tops": 8,      # At least 8 casual tops for variety
        "bottoms": 6,          # At least 6 bottoms for mixing
        "footwear": 3          # At least 3 pairs of shoes
    },
    
    "maximum_items_per_category": {
        "heavy_items": 3,      # Max 3 items over 1kg
        "single_occasion": 2   # Max 2 items for single occasion only
    },
    
    "required_coverage": {
        "temperature_range": 34,    # Must cover 34°C range (6° to 40°)
        "cultural_compliance": True,
        "business_readiness": True,
        "climate_adaptation": True
    }
}