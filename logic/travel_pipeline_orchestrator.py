import json
import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from logic.travel_packing_agent import travel_packing_agent
from logic.data_manager import wardrobe_data_manager
from logic.notion_utils import (
    notion,
    clear_page_content,
    clear_trigger_fields,
    retrieve_page_blocks
)
from logic.travel_config import DESTINATIONS_CONFIG, WEIGHT_CONSTRAINTS

load_dotenv()

class TravelPipelineOrchestrator:
    """
    Orchestrates the complete travel packing pipeline using hierarchical AI agents.
    Gemini API -> Groq API -> Error (no logic fallback for travel - too complex)
    """
    
    def __init__(self):
        self.packing_guide_page_id = os.getenv("NOTION_PACKING_GUIDE_ID")
        self.wardrobe_db_id = os.getenv("NOTION_WARDROBE_DB_ID")
        
        if not self.packing_guide_page_id:
            raise EnvironmentError("NOTION_PACKING_GUIDE_ID not set in environment variables")
        if not self.wardrobe_db_id:
            raise EnvironmentError("NOTION_WARDROBE_DB_ID not set in environment variables")
    
    async def run_travel_packing_pipeline(self, trigger_data: Dict) -> Dict:
        """
        Enhanced async pipeline for travel packing optimization.
        Gemini API -> Groq API -> Error (no logic fallback - too complex for rule-based)
        
        Args:
            trigger_data (dict): Contains trip configuration and user preferences
            
        Returns:
            dict: Result with success status and generation method used
        """
        try:
            # Extract and validate trip configuration
            trip_config = await self._prepare_trip_configuration(trigger_data)
            if not trip_config:
                return {
                    "success": False,
                    "error": "Invalid trip configuration",
                    "page_id": trigger_data.get("page_id"),
                    "generation_method": "config_validation_failed"
                }
            
            logging.info(f"Starting travel packing pipeline for {len(trip_config['destinations'])} destinations")
            logging.info(f"Trip duration: {trip_config['trip_overview']['total_duration_months']} months")
            
            # Step 1: Get wardrobe data using hierarchical fallback
            logging.info("Fetching wardrobe data...")
            try:
                available_items = await asyncio.to_thread(
                    self._get_travel_optimized_wardrobe_data
                )
                
                if not available_items or sum(len(items) for items in available_items.values()) == 0:
                    return {
                        "success": False,
                        "error": "No wardrobe items available for packing optimization",
                        "page_id": trigger_data.get("page_id"),
                        "generation_method": "no_wardrobe_data"
                    }
                
                total_items = sum(len(items) for items in available_items.values())
                logging.info(f"Wardrobe data prepared: {total_items} items across {len(available_items)} categories")
                
            except Exception as e:
                logging.error(f"Failed to fetch wardrobe data: {e}")
                return {
                    "success": False,
                    "error": f"Wardrobe data fetch failed: {str(e)}",
                    "page_id": trigger_data.get("page_id"),
                    "generation_method": "data_fetch_failed"
                }
            
            # Step 2: Try Gemini API (Primary Agent) with timeout
            logging.info("🤖 Attempting packing optimization with Gemini API...")
            success, packing_result, error_msg = await travel_packing_agent.generate_multi_destination_packing_list(
                trip_config, available_items, timeout=35
            )
            
            if success and packing_result:
                logging.info("✅ Gemini API generated packing list successfully")
                return await self._finalize_packing_results(
                    trigger_data.get("page_id"), packing_result, "gemini", trip_config
                )
            else:
                logging.warning(f"Gemini failed: {error_msg}")
            
            # Step 3: Try Groq API (Secondary Agent) with timeout
            logging.info("🤖 Attempting packing optimization with Groq API...")
            success, packing_result, error_msg = await travel_packing_agent.generate_packing_list_with_groq(
                trip_config, available_items, timeout=30
            )
            
            if success and packing_result:
                logging.info("✅ Groq API generated packing list successfully")
                return await self._finalize_packing_results(
                    trigger_data.get("page_id"), packing_result, "groq", trip_config
                )
            else:
                logging.warning(f"Groq failed: {error_msg}")
            
            # Step 4: All AI methods failed - no logic fallback for travel (too complex)
            logging.error("❌ All AI packing agents failed - travel optimization requires AI intelligence")
            return {
                "success": False,
                "error": "All AI agents failed - travel packing requires advanced AI reasoning",
                "page_id": trigger_data.get("page_id"),
                "generation_method": "all_ai_failed",
                "attempted_methods": ["gemini", "groq"],
                "recommendation": "Check AI API connectivity and try again"
            }
            
        except Exception as e:
            logging.error(f"Critical error in travel packing pipeline: {e}")
            return {
                "success": False,
                "error": f"Pipeline error: {str(e)}",
                "page_id": trigger_data.get("page_id"),
                "generation_method": "pipeline_error"
            }
    
    async def _prepare_trip_configuration(self, trigger_data: Dict) -> Optional[Dict]:
        """
        Prepare comprehensive trip configuration from trigger data.
        Scalable for any destinations defined in travel_config.py
        """
        try:
            # Extract basic trip info
            destinations_list = trigger_data.get("destinations", [])
            user_preferences = trigger_data.get("preferences", {})
            
            if not destinations_list:
                logging.error("No destinations specified in trigger data")
                return None
            
            # Validate all destinations are configured
            for dest in destinations_list:
                if dest["city"] not in DESTINATIONS_CONFIG:
                    logging.error(f"Destination '{dest['city']}' not configured in travel_config.py")
                    return None
            
            # Build comprehensive trip configuration
            trip_config = {
                "destinations": destinations_list,
                "user_preferences": user_preferences,
                "trip_overview": self._calculate_trip_overview(destinations_list),
                "weight_constraints": WEIGHT_CONSTRAINTS,
                "optimization_goals": user_preferences.get("optimization_goals", [
                    "weight_efficiency", 
                    "business_readiness", 
                    "climate_coverage", 
                    "cultural_compliance"
                ])
            }
            
            # Add destination-specific analysis
            trip_config["destination_analysis"] = []
            for dest in destinations_list:
                dest_analysis = self._analyze_destination_requirements(dest)
                trip_config["destination_analysis"].append(dest_analysis)
            
            return trip_config
            
        except Exception as e:
            logging.error(f"Error preparing trip configuration: {e}")
            return None
    
    def _calculate_trip_overview(self, destinations: List[Dict]) -> Dict:
        """Calculate overall trip characteristics"""
        # Calculate total duration
        start_date = datetime.strptime(destinations[0]["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(destinations[-1]["end_date"], "%Y-%m-%d")
        total_duration = (end_date - start_date).days
        
        # Calculate temperature range
        min_temp = float('inf')
        max_temp = float('-inf')
        
        for dest in destinations:
            city_config = DESTINATIONS_CONFIG[dest["city"]]
            for month_data in city_config["seasons"].values():
                temp_range = month_data["temp_range"]
                min_temp = min(min_temp, temp_range[0])
                max_temp = max(max_temp, temp_range[1])
        
        # Identify climate types
        climate_types = set(DESTINATIONS_CONFIG[d["city"]]["climate_profile"] for d in destinations)
        
        return {
            "total_duration_days": total_duration,
            "total_duration_months": round(total_duration / 30, 1),
            "destination_count": len(destinations),
            "climate_diversity": len(climate_types),
            "temperature_range": {
                "min": min_temp,
                "max": max_temp,
                "span": max_temp - min_temp
            },
            "climate_types": list(climate_types)
        }
    
    def _analyze_destination_requirements(self, destination: Dict) -> Dict:
        """Analyze requirements for a specific destination"""
        city = destination["city"]
        city_config = DESTINATIONS_CONFIG[city]
        
        # Calculate months in destination
        start_date = datetime.strptime(destination["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(destination["end_date"], "%Y-%m-%d")
        
        months = []
        current = start_date.replace(day=1)
        while current <= end_date:
            months.append(current.strftime("%B").lower())
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        # Get seasonal requirements
        seasonal_requirements = []
        for month in months:
            if month in city_config["seasons"]:
                seasonal_requirements.append(city_config["seasons"][month])
        
        return {
            "city": city,
            "duration_months": len(months),
            "months": months,
            "seasonal_requirements": seasonal_requirements,
            "cultural_context": city_config["cultural_context"],
            "climate_profile": city_config["climate_profile"],
            "weight_priorities": city_config["weight_priorities"],
            "activity_types": city_config["activity_types"],
            "climate_recommendations": city_config["climate_recommendations"]
        }
    
    async def _get_travel_optimized_wardrobe_data(self) -> Dict:
        """
        Get wardrobe data optimized for travel packing analysis.
        Uses hierarchical fallback: Supabase -> Cache -> Notion -> Error
        """
        try:
            # Get all wardrobe items (let the data manager handle hierarchical fallback)
            all_items = wardrobe_data_manager.get_all_wardrobe_items()
            
            if not all_items:
                logging.error("No wardrobe items available from any data source")
                return {}
            
            # Organize items by category for AI optimization
            categorized_items = {}
            
            for item in all_items:
                category = item.get('category', 'Unknown')
                if category not in categorized_items:
                    categorized_items[category] = []
                categorized_items[category].append(item)
            
            # Filter and prioritize for travel context
            travel_optimized = {}
            
            # Prioritize categories important for long-term travel
            priority_categories = [
                'Suit', 'Shirt', 'Chinos', 'Shoes',  # Business essentials
                'Polo', 'T-shirt', 'Jeans',          # Versatile basics
                'Sneakers', 'Hoodie', 'Jacket'       # Comfort & weather
            ]
            
            for category in priority_categories:
                if category in categorized_items:
                    # Sort by trip-worthy status and aesthetic versatility
                    sorted_items = sorted(
                        categorized_items[category],
                        key=lambda x: (
                            x.get('trip_worthy', False),  # Trip-worthy first
                            len(x.get('aesthetic', [])),  # More aesthetics = more versatile
                            x.get('washed', '').lower() == 'done'  # Clean items preferred
                        ),
                        reverse=True
                    )
                    travel_optimized[category] = sorted_items
            
            # Add remaining categories
            for category, items in categorized_items.items():
                if category not in travel_optimized:
                    travel_optimized[category] = items
            
            logging.info(f"Travel-optimized wardrobe data: {len(travel_optimized)} categories")
            return travel_optimized
            
        except Exception as e:
            logging.error(f"Error getting travel-optimized wardrobe data: {e}")
            return {}
    
    async def _finalize_packing_results(self, page_id: str, packing_result: Dict, 
                                      generation_method: str, trip_config: Dict) -> Dict:
        """
        Async finalize packing results by updating Notion and generating comprehensive guide
        
        Args:
            page_id: Notion page ID for packing guide
            packing_result: AI-generated packing results
            generation_method: Which AI method generated the results
            trip_config: Trip configuration
            
        Returns:
            dict: Success result with details
        """
        try:
            # Step 1: Update trip-worthy checkboxes in wardrobe database (async)
            logging.info("Updating trip-worthy selections in wardrobe database...")
            await asyncio.to_thread(
                self._update_trip_worthy_selections, 
                packing_result["selected_items"]
            )
            
            # Step 2: Clear previous content from packing guide page (async)
            logging.info("Clearing previous packing guide content...")
            await asyncio.to_thread(clear_page_content, page_id)
            
            # Step 3: Generate and post comprehensive packing guide (async)
            logging.info("Generating comprehensive packing guide...")
            await asyncio.to_thread(
                self._post_comprehensive_packing_guide,
                page_id, packing_result, trip_config, generation_method
            )
            
            # Step 4: Clear trigger fields if they exist (async)
            logging.info("Clearing trigger fields...")
            await asyncio.to_thread(
                self._clear_travel_trigger_fields,
                page_id
            )
            
            logging.info(f"✅ Travel packing pipeline completed successfully using {generation_method}!")
            
            return {
                "success": True,
                "page_id": page_id,
                "generation_method": generation_method,
                "total_items_selected": packing_result["total_items"],
                "total_weight_kg": packing_result["total_weight_kg"],
                "weight_efficiency": packing_result["weight_efficiency"],
                "business_readiness": packing_result["business_readiness"]["readiness_score"],
                "destinations": [dest["city"] for dest in trip_config["destinations"]],
                "trip_duration_months": trip_config["trip_overview"]["total_duration_months"],
                "outfit_possibilities": packing_result["outfit_analysis"]["total_outfit_combinations"],
                "bag_allocation": {
                    "checked_weight": packing_result["bag_allocation"]["checked_bag"]["weight_kg"],
                    "cabin_weight": packing_result["bag_allocation"]["cabin_bag"]["weight_kg"]
                },
                "data_sources_used": await self._get_data_source_info()
            }
            
        except Exception as e:
            logging.error(f"Error finalizing packing results: {e}")
            return {
                "success": False,
                "error": f"Failed to finalize packing results: {str(e)}",
                "page_id": page_id,
                "generation_method": f"{generation_method}_finalization_failed"
            }
    
    def _update_trip_worthy_selections(self, selected_items: List[Dict]) -> None:
        """
        Update trip-worthy checkboxes in the wardrobe database for selected items
        """
        try:
            selected_ids = set(item['id'] for item in selected_items)
            
            # Get all wardrobe items to update both selected and unselected
            all_items = wardrobe_data_manager.get_all_wardrobe_items()
            
            for item in all_items:
                item_id = item['id']
                should_be_selected = item_id in selected_ids
                
                # Update the trip-worthy checkbox
                try:
                    notion.pages.update(
                        page_id=item_id,
                        properties={
                            "Trip-worthy": {
                                "checkbox": should_be_selected
                            }
                        }
                    )
                except Exception as e:
                    logging.warning(f"Failed to update trip-worthy for item {item_id}: {e}")
            
            logging.info(f"Updated trip-worthy selections: {len(selected_ids)} items selected")
            
        except Exception as e:
            logging.error(f"Error updating trip-worthy selections: {e}")
            raise
    
    def _post_comprehensive_packing_guide(self, page_id: str, packing_result: Dict, 
                                        trip_config: Dict, generation_method: str) -> None:
        """
        Post comprehensive packing guide to Notion page
        """
        try:
            # Build comprehensive guide content
            guide_blocks = self._build_packing_guide_blocks(
                packing_result, trip_config, generation_method
            )
            
            # Post to Notion page
            notion.blocks.children.append(
                block_id=page_id,
                children=guide_blocks
            )
            
            logging.info(f"Posted comprehensive packing guide with {len(guide_blocks)} sections")
            
        except Exception as e:
            logging.error(f"Error posting packing guide: {e}")
            raise
    
    def _build_packing_guide_blocks(self, packing_result: Dict, trip_config: Dict, 
                                  generation_method: str) -> List[Dict]:
        """
        Build comprehensive packing guide as Notion blocks
        """
        blocks = []
        
        # Title and Overview
        blocks.extend([
            {
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": "🧳 AI-Generated Travel Packing Guide"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "text": {"content": f"Generated using {generation_method.upper()} AI • "}},
                        {"type": "text", "text": {"content": f"{datetime.now().strftime('%B %d, %Y')}"}}
                    ]
                }
            }
        ])
        
        # Trip Overview
        blocks.extend(self._create_trip_overview_blocks(trip_config))
        
        # Packing Summary
        blocks.extend(self._create_packing_summary_blocks(packing_result))
        
        # Selected Items by Category
        blocks.extend(self._create_selected_items_blocks(packing_result))
        
        # Bag Allocation Strategy
        blocks.extend(self._create_bag_allocation_blocks(packing_result))
        
        # Outfit Analysis
        blocks.extend(self._create_outfit_analysis_blocks(packing_result))
        
        # Packing Organization Guide
        blocks.extend(self._create_packing_guide_blocks_section(packing_result))
        
        # Destination-Specific Tips
        blocks.extend(self._create_destination_tips_blocks(packing_result, trip_config))
        
        # Assessment & Recommendations
        blocks.extend(self._create_assessment_blocks(packing_result))
        
        return blocks
    
    def _create_trip_overview_blocks(self, trip_config: Dict) -> List[Dict]:
        """Create trip overview section"""
        overview = trip_config["trip_overview"]
        destinations = trip_config["destinations"]
        
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "🌍 Trip Overview"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Duration: {overview['total_duration_months']} months ({overview['total_duration_days']} days)"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Destinations: {overview['destination_count']} cities"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Temperature Range: {overview['temperature_range']['min']}°C to {overview['temperature_range']['max']}°C ({overview['temperature_range']['span']}°C span)"}}]
                }
            }
        ]
        
        # Add destination details
        for dest in destinations:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"{dest['city'].title()}: {dest['start_date']} to {dest['end_date']}"}}]
                }
            })
        
        return blocks
    
    def _create_packing_summary_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create packing summary section"""
        return [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "📊 Packing Summary"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Total Items: {packing_result['total_items']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Total Weight: {packing_result['total_weight_kg']}kg"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Weight Efficiency: {packing_result['weight_efficiency']} outfits per kg"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Business Readiness: {packing_result['business_readiness']['readiness_score']} (Need ≥0.8)"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Total Outfit Combinations: {packing_result['outfit_analysis']['total_outfit_combinations']}"}}]
                }
            }
        ]
    
    def _create_selected_items_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create selected items section organized by category"""
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "👕 Selected Items by Category"}}]
                }
            }
        ]
        
        # Group items by category
        items_by_category = {}
        for item in packing_result["selected_items"]:
            category = item['category']
            if category not in items_by_category:
                items_by_category[category] = []
            items_by_category[category].append(item)
        
        # Add category sections
        for category, items in items_by_category.items():
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": f"{category} ({len(items)} items)"}}]
                }
            })
            
            for item in items:
                aesthetics = ', '.join(item.get('aesthetic', []))
                weather = ', '.join(item.get('weather', []))
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": f"{item['item']} - {aesthetics} - {weather}"}}]
                    }
                })
        
        return blocks
    
    def _create_bag_allocation_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create bag allocation section"""
        allocation = packing_result["bag_allocation"]
        
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "🎒 Bag Allocation Strategy"}}]
                }
            },
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": f"Checked Bag ({allocation['checked_bag']['weight_kg']}kg)"}}]
                }
            }
        ]
        
        for item in allocation['checked_bag']['items']:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"{item['item']} ({item['category']})"}}]
                }
            })
        
        blocks.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {
                "rich_text": [{"type": "text", "text": {"content": f"Cabin Bag ({allocation['cabin_bag']['weight_kg']}kg)"}}]
            }
        })
        
        for item in allocation['cabin_bag']['items']:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"{item['item']} ({item['category']})"}}]
                }
            })
        
        return blocks
    
    def _create_outfit_analysis_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create outfit analysis section"""
        analysis = packing_result["outfit_analysis"]
        
        return [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "👔 Outfit Analysis"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Business Formal Outfits: {analysis['business_formal_outfits']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Business Casual Outfits: {analysis['business_casual_outfits']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Casual Outfits: {analysis['casual_outfits']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Total Combinations: {analysis['total_outfit_combinations']}"}}]
                }
            }
        ]
    
    def _create_packing_guide_blocks_section(self, packing_result: Dict) -> List[Dict]:
        """Create packing organization guide section"""
        guide = packing_result["packing_guide"]
        
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "📦 Packing Organization Guide"}}]
                }
            },
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "Packing Techniques"}}]
                }
            }
        ]
        
        for technique in guide["packing_techniques"]:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": technique}}]
                }
            })
        
        blocks.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {
                "rich_text": [{"type": "text", "text": {"content": "Travel Day Strategy"}}]
            }
        })
        
        wear_items = ', '.join(guide["travel_day_strategy"]["wear_during_travel"])
        cabin_essentials = ', '.join(guide["travel_day_strategy"]["cabin_essentials"])
        
        blocks.extend([
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Wear during travel: {wear_items}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Cabin essentials: {cabin_essentials}"}}]
                }
            }
        ])
        
        return blocks
    
    def _create_destination_tips_blocks(self, packing_result: Dict, trip_config: Dict) -> List[Dict]:
        """Create destination-specific tips section"""
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "🗺️ Destination-Specific Tips"}}]
                }
            }
        ]
        
        trip_tips = packing_result.get("trip_tips", {})
        
        for destination in trip_config["destinations"]:
            city = destination["city"]
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": f"{city.title()}"}}]
                }
            })
            
            if city in trip_tips:
                city_tips = trip_tips[city]
                
                # Cultural tips
                if "cultural_tips" in city_tips:
                    for tip in city_tips["cultural_tips"]:
                        blocks.append({
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"Cultural: {tip}"}}]
                            }
                        })
                
                # Climate preparation
                if "climate_preparation" in city_tips:
                    for tip in city_tips["climate_preparation"]:
                        blocks.append({
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"Climate: {tip}"}}]
                            }
                        })
                
                # Practical advice
                if "practical_advice" in city_tips:
                    for tip in city_tips["practical_advice"]:
                        blocks.append({
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"Practical: {tip}"}}]
                            }
                        })
        
        return blocks
    
    def _create_assessment_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create assessment and recommendations section"""
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "✅ Assessment & Recommendations"}}]
                }
            }
        ]
        
        # Business readiness assessment
        business = packing_result["business_readiness"]
        status = "✅ Excellent" if business["meets_requirements"] else "⚠️ Needs Improvement"
        
        blocks.extend([
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "Business Readiness"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Status: {status}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Suits: {business['suits_count']} (minimum 2 recommended)"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Dress shoes: {business['dress_shoes_count']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Business shirts: {business['business_shirts_count']}"}}]
                }
            }
        ])
        
        # Climate coverage assessment
        climate = packing_result["climate_coverage"]
        blocks.extend([
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "Climate Coverage"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Temperature range: {climate['temperature_range_covered']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Hot weather items: {climate['hot_weather_coverage']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Cold weather items: {climate['cold_weather_coverage']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Versatile items: {climate['versatile_items']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Coverage adequacy: {climate['coverage_adequacy'].replace('_', ' ').title()}"}}]
                }
            }
        ])
        
        # Cultural compliance assessment
        cultural = packing_result["cultural_compliance"]
        cultural_status = "✅ Dubai Ready" if cultural["dubai_ready"] else "⚠️ Review Needed"
        
        blocks.extend([
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "Cultural Compliance"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Status: {cultural_status}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Compliance score: {cultural['compliance_score']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Modest items: {cultural['modest_items_count']}/{cultural['total_items']}"}}]
                }
            }
        ])
        
        # Add cultural recommendations
        for recommendation in cultural.get("recommendations", []):
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Tip: {recommendation}"}}]
                }
            })
        
        return blocks
    
    def _clear_travel_trigger_fields(self, page_id: str) -> None:
        """
        Clear travel trigger fields to reset for next use.
        Safely handles case where trigger fields don't exist.
        """
        try:
            # Check if page has trigger properties by retrieving it first
            page = notion.pages.retrieve(page_id=page_id)
            properties = page.get("properties", {})
            
            # Only clear fields that exist
            update_properties = {}
            
            if "Generate Travel Packing" in properties:
                update_properties["Generate Travel Packing"] = {"checkbox": False}
            
            if "Destinations" in properties:
                update_properties["Destinations"] = {"multi_select": []}
            
            if "Travel Preferences" in properties:
                update_properties["Travel Preferences"] = {"rich_text": []}
            
            # Only update if there are properties to update
            if update_properties:
                notion.pages.update(page_id=page_id, properties=update_properties)
                logging.info(f"Cleared travel trigger fields for page {page_id}")
            else:
                logging.info(f"No travel trigger fields found on page {page_id}")
                
        except Exception as e:
            # Don't fail the entire pipeline if trigger clearing fails
            logging.warning(f"Could not clear travel trigger fields for page {page_id}: {e}")
    
    async def _get_data_source_info(self) -> Dict:
        """Async get information about which data sources are available/used"""
        try:
            return await asyncio.to_thread(wardrobe_data_manager.get_data_stats)
        except Exception:
            return {"error": "Could not retrieve data source stats"}

# Create global instance
travel_pipeline_orchestrator = TravelPipelineOrchestrator()

# Async test function for development
async def test_travel_packing_pipeline(destinations: List[Dict] = None, preferences: Dict = None):
    """
    Async test function to verify travel packing pipeline
    
    Args:
        destinations: List of destination dictionaries with city, start_date, end_date
        preferences: User preferences dictionary
    """
    # Default test configuration
    if not destinations:
        destinations = [
            {
                "city": "dubai",
                "start_date": "2024-09-01",
                "end_date": "2024-12-31"
            },
            {
                "city": "gurgaon",
                "start_date": "2025-01-01", 
                "end_date": "2025-05-31"
            }
        ]
    
    if not preferences:
        preferences = {
            "optimization_goals": ["weight_efficiency", "business_readiness", "climate_coverage"],
            "packing_style": "minimalist_professional",
            "trip_type": "business_school_relocation"
        }
    
    trigger_data = {
        "page_id": travel_pipeline_orchestrator.packing_guide_page_id,
        "destinations": destinations,
        "preferences": preferences
    }
    
    logging.info(f"Testing travel packing pipeline...")
    logging.info(f"Destinations: {[d['city'] for d in destinations]}")
    logging.info(f"Duration: {len(destinations)} destinations")
    
    try:
        result = await travel_pipeline_orchestrator.run_travel_packing_pipeline(trigger_data)
        
        if result["success"]:
            print(f"✅ Travel packing pipeline test successful!")
            print(f"   Method: {result['generation_method']}")
            print(f"   Items selected: {result['total_items_selected']}")
            print(f"   Total weight: {result['total_weight_kg']}kg")
            print(f"   Weight efficiency: {result['weight_efficiency']} outfits/kg")
            print(f"   Business readiness: {result['business_readiness']}")
            print(f"   Destinations: {result['destinations']}")
            print(f"   Trip duration: {result['trip_duration_months']} months")
            print(f"   Outfit possibilities: {result['outfit_possibilities']}")
        else:
            print(f"❌ Travel packing pipeline test failed: {result['error']}")
            print(f"   Generation method: {result['generation_method']}")
            
    except Exception as e:
        print(f"💥 Test error: {e}")

# Convenience function to run async test from sync context
def run_test_travel_packing_pipeline(destinations: List[Dict] = None, preferences: Dict = None):
    """Sync wrapper to run async test"""
    return asyncio.run(test_travel_packing_pipeline(destinations, preferences))

# Legacy compatibility wrapper
def run_travel_packing_pipeline(trigger_data):
    """
    Legacy wrapper for backward compatibility - runs async pipeline in sync context
    
    DEPRECATED: Use run_travel_packing_pipeline() directly with asyncio.run() or await
    """
    logging.warning("Using legacy run_travel_packing_pipeline wrapper - consider updating to async")
    return asyncio.run(travel_pipeline_orchestrator.run_travel_packing_pipeline(trigger_data))