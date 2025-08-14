import json
import os
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Internal imports
from core.travel_logic_fallback import travel_logic_fallback
from core.travel_packing_agent import travel_packing_agent
from data.data_manager import wardrobe_data_manager
from data.notion_utils import (
    notion,
    clear_page_content,
    clear_trigger_fields,
    retrieve_page_blocks
)
from config.travel_config import DESTINATIONS_CONFIG, WEIGHT_CONSTRAINTS

# Import for monitoring integration
try:
    from monitoring.system_monitor import system_monitor
    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False
    logging.warning("Monitoring not available - running without system monitor")

load_dotenv()

class PipelineStage(Enum):
    """Pipeline execution stages for tracking"""
    INIT = "initialization"
    CONFIG_PREP = "config_preparation"
    DATA_FETCH = "data_fetch"
    GEMINI_ATTEMPT = "gemini_attempt"
    GROQ_ATTEMPT = "groq_attempt"
    FALLBACK_ATTEMPT = "fallback_attempt"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PipelineMetrics:
    """Metrics tracking for pipeline execution"""
    total_items_processed: int = 0
    data_source_used: str = ""
    ai_provider_used: str = ""
    execution_time_ms: float = 0
    memory_usage_mb: float = 0
    stage_timings: Dict[str, float] = None
    
    def __post_init__(self):
        if self.stage_timings is None:
            self.stage_timings = {}

class TravelPipelineOrchestrator:
    """
    Enhanced travel packing pipeline orchestrator with improved error handling,
    monitoring integration, and maintainable code structure.
    
    Features:
    - Hierarchical AI fallback: Gemini -> Groq -> Rule-based
    - Comprehensive error handling and retry logic
    - Performance monitoring and metrics
    - Structured logging and debugging
    - Rate limiting and batch operations
    - Memory management and cleanup
    """
    
    def __init__(self):
        """Initialize the orchestrator with comprehensive validation."""
        self._start_time = time.time()
        
        logging.info("🔧 Initializing TravelPipelineOrchestrator...")
            
    async def ensure_ready(self):
        """Validate env and confirm Notion connectivity RIGHT BEFORE running."""
        self._validate_environment()
    
        # Configuration setup
        self.packing_guide_page_id = os.getenv("NOTION_PACKING_GUIDE_ID")
        self.wardrobe_db_id = os.getenv("NOTION_WARDROBE_DB_ID")
    
        await self._test_notion_connectivity()
    
        # Rate limiting configuration
        self.batch_size = 20
        self.batch_delay = 0.2  # seconds between batches
        self.api_retry_attempts = 3
        self.api_retry_delay = 1.0  # seconds
    
        # Performance tracking
        self.metrics = PipelineMetrics()
        self.current_stage = PipelineStage.INIT
    
        logging.info("✅ TravelPipelineOrchestrator initialized successfully")
        logging.info(f"   Initialization time: {(time.time() - self._start_time) * 1000:.1f}ms")
    
    def _validate_environment(self) -> None:
        """Validate all required environment variables."""
        required_vars = {
            'NOTION_TOKEN': os.getenv('NOTION_TOKEN'),
            'NOTION_PACKING_GUIDE_ID': os.getenv('NOTION_PACKING_GUIDE_ID'),
            'NOTION_WARDROBE_DB_ID': os.getenv('NOTION_WARDROBE_DB_ID')
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {missing_vars}"
            logging.error(f"❌ {error_msg}")
            raise EnvironmentError(error_msg)
        
        # Log configuration (masked for security)
        for var, value in required_vars.items():
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            logging.info(f"   {var}: {masked_value}")
    
    async def _test_notion_connectivity(self) -> None:
        """Test Notion API connectivity during initialization."""
        try:
            # FIX: Run the synchronous Notion call in a separate thread
            await asyncio.to_thread(
                notion.pages.retrieve,
                page_id=self.packing_guide_page_id
            )
            logging.info("✅ Notion connectivity test successful")
        except Exception as e:
            error_msg = f"Notion connectivity test failed: {e}"
            logging.error(f"❌ {error_msg}")
            raise ConnectionError(error_msg)
    
    async def run_travel_packing_pipeline(self, trigger_data: Dict) -> Dict:
        """
        Main pipeline execution with comprehensive monitoring and error handling.
    
        Args:
            trigger_data: Trip configuration and user preferences
        
        Returns:
            Detailed result dictionary with success status and metrics
        """
        # This is the fix: Ensure the orchestrator is ready before executing.
        await self.ensure_ready()
    
        pipeline_start = time.time()
    
        # Monitoring integration
        if MONITORING_ENABLED:
            return await system_monitor.track_operation(
                "travel_packing_pipeline_full",
                self._execute_pipeline_with_monitoring,
                trigger_data, pipeline_start
            )
        else:
            return await self._execute_pipeline_with_monitoring(trigger_data, pipeline_start)
    
    async def _execute_pipeline_with_monitoring(self, trigger_data: Dict, pipeline_start: float) -> Dict:
        """Execute the pipeline with internal monitoring."""
        try:
            logging.info("🧳 Starting enhanced travel packing pipeline...")
            self._log_trigger_data(trigger_data)
            
            # Stage 1: Trip Configuration Preparation
            self.current_stage = PipelineStage.CONFIG_PREP
            stage_start = time.time()
            
            trip_config = await self._prepare_trip_configuration_enhanced(trigger_data)
            if not trip_config:
                return self._create_error_result(
                    trigger_data, "config_validation_failed",
                    "Invalid trip configuration", pipeline_start
                )
            
            self.metrics.stage_timings[PipelineStage.CONFIG_PREP.value] = (time.time() - stage_start) * 1000
            logging.info(f"✅ Trip configuration prepared ({self.metrics.stage_timings[PipelineStage.CONFIG_PREP.value]:.1f}ms)")
            
            # Stage 2: Wardrobe Data Acquisition
            self.current_stage = PipelineStage.DATA_FETCH
            stage_start = time.time()
            
            available_items = await self._get_travel_optimized_wardrobe_data_enhanced()
            if not available_items or sum(len(items) for items in available_items.values()) == 0:
                return self._create_error_result(
                    trigger_data, "no_wardrobe_data",
                    "No wardrobe items available for packing optimization", pipeline_start
                )
            
            self.metrics.stage_timings[PipelineStage.DATA_FETCH.value] = (time.time() - stage_start) * 1000
            self.metrics.total_items_processed = sum(len(items) for items in available_items.values())
            logging.info(f"✅ Wardrobe data acquired ({self.metrics.stage_timings[PipelineStage.DATA_FETCH.value]:.1f}ms)")
            
            # Stage 3: AI-Powered Packing Optimization (with fallback chain)
            packing_result = await self._execute_packing_optimization_chain(
                trip_config, available_items, pipeline_start
            )
            
            if not packing_result["success"]:
                return packing_result
            
            # Stage 4: Results Finalization
            self.current_stage = PipelineStage.FINALIZATION
            stage_start = time.time()
            
            final_result = await self._finalize_packing_results_enhanced(
                trigger_data.get("page_id"), packing_result["data"], 
                packing_result["generation_method"], trip_config, pipeline_start
            )
            
            self.metrics.stage_timings[PipelineStage.FINALIZATION.value] = (time.time() - stage_start) * 1000
            
            if final_result["success"]:
                self.current_stage = PipelineStage.COMPLETED
                total_time = (time.time() - pipeline_start) * 1000
                logging.info(f"🎉 Travel packing pipeline completed successfully in {total_time:.1f}ms")
                
                # Add performance metrics to result
                final_result["performance_metrics"] = {
                    "total_execution_time_ms": total_time,
                    "stage_timings": self.metrics.stage_timings,
                    "items_processed": self.metrics.total_items_processed,
                    "data_source": self.metrics.data_source_used,
                    "ai_provider": self.metrics.ai_provider_used
                }
            
            return final_result
            
        except Exception as e:
            self.current_stage = PipelineStage.FAILED
            error_msg = f"Critical pipeline error: {str(e)}"
            logging.error(f"❌ {error_msg}", exc_info=True)
            
            return self._create_error_result(
                trigger_data, "pipeline_error", error_msg, pipeline_start
            )
    
    def _log_trigger_data(self, trigger_data: Dict) -> None:
        """Log trigger data with proper formatting."""
        destinations = trigger_data.get("destinations", [])
        preferences = trigger_data.get("preferences", {})
        
        logging.info(f"🧳 Pipeline trigger data:")
        logging.info(f"   Page ID: {trigger_data.get('page_id', 'unknown')}")
        logging.info(f"   Destinations: {len(destinations)} cities")
        for dest in destinations:
            logging.info(f"     {dest.get('city', 'unknown')}: {dest.get('start_date', 'N/A')} to {dest.get('end_date', 'N/A')}")
        logging.info(f"   Preferences: {preferences}")
    
    async def _prepare_trip_configuration_enhanced(self, trigger_data: Dict) -> Optional[Dict]:
        """Enhanced trip configuration preparation with validation."""
        try:
            logging.info("🧳 Preparing enhanced trip configuration...")
        
            destinations_list = trigger_data.get("destinations", [])
            user_preferences = trigger_data.get("preferences", []) # Correctly handle as a list
        
            if not destinations_list:
                logging.error("❌ No destinations specified in trigger data")
                return None
        
            # Validate destinations against configuration
            for dest in destinations_list:
                city = dest.get("city")
                if not city or city not in DESTINATIONS_CONFIG:
                    logging.error(f"❌ Invalid destination: '{city}' not in {list(DESTINATIONS_CONFIG.keys())}")
                    return None
        
            logging.info("✅ All destinations validated")
        
            # Build comprehensive configuration
            trip_config = {
                "destinations": destinations_list,
                "user_preferences": user_preferences,
                "trip_overview": self._calculate_trip_overview_enhanced(destinations_list),
                "weight_constraints": WEIGHT_CONSTRAINTS,
                "optimization_goals": user_preferences if isinstance(user_preferences, list) else [
                    "weight_efficiency", "business_readiness", 
                    "climate_coverage", "cultural_compliance"
                ]
            }
        
            # Add destination analysis
            trip_config["destination_analysis"] = []
            for dest in destinations_list:
                analysis = self._analyze_destination_requirements_enhanced(dest)
                trip_config["destination_analysis"].append(analysis)
                logging.info(f"   {dest['city']}: {analysis['duration_months']} months")
        
            return trip_config
        
        except Exception as e:
            logging.error(f"❌ Error in trip configuration preparation: {e}", exc_info=True)
            return None
    
    def _calculate_trip_overview_enhanced(self, destinations: List[Dict]) -> Dict:
        """Calculate enhanced trip overview with detailed metrics."""
        try:
            # Duration calculations
            start_date = datetime.strptime(destinations[0]["start_date"], "%Y-%m-%d")
            end_date = datetime.strptime(destinations[-1]["end_date"], "%Y-%m-%d")
            total_duration = (end_date - start_date).days
            
            # Temperature analysis
            min_temp, max_temp = float('inf'), float('-inf')
            climate_types = set()
            cultural_requirements = set()
            
            for dest in destinations:
                city_config = DESTINATIONS_CONFIG[dest["city"]]
                climate_types.add(city_config["climate_profile"])
                cultural_requirements.add(city_config["cultural_context"]["modesty_level"])
                
                # Analyze temperature across seasons
                for season_data in city_config["seasons"].values():
                    temp_range = season_data["temp_range"]
                    min_temp = min(min_temp, temp_range[0])
                    max_temp = max(max_temp, temp_range[1])
            
            overview = {
                "total_duration_days": total_duration,
                "total_duration_months": round(total_duration / 30, 1),
                "destination_count": len(destinations),
                "climate_diversity": len(climate_types),
                "temperature_range": {
                    "min": min_temp,
                    "max": max_temp,
                    "span": max_temp - min_temp
                },
                "climate_types": list(climate_types),
                "cultural_requirements": list(cultural_requirements),
                "complexity_score": self._calculate_trip_complexity(destinations)
            }
            
            logging.info(f"   Trip overview: {overview['total_duration_months']}mo, {overview['temperature_range']['span']}°C span")
            return overview
            
        except Exception as e:
            logging.error(f"Error calculating trip overview: {e}")
            return {"error": str(e)}
    
    def _calculate_trip_complexity(self, destinations: List[Dict]) -> float:
        """Calculate trip complexity score for optimization strategy."""
        complexity = 0.0
        
        # Duration complexity
        total_months = len(destinations) * 4  # Rough estimate
        complexity += min(total_months / 12, 1.0) * 0.3  # Normalize to 1 year
        
        # Climate diversity
        climates = set(DESTINATIONS_CONFIG[d["city"]]["climate_profile"] for d in destinations)
        complexity += len(climates) / 3 * 0.3  # Max 3 climate types
        
        # Cultural requirements
        modesty_levels = [DESTINATIONS_CONFIG[d["city"]]["cultural_context"]["modesty_level"] for d in destinations]
        if "high" in modesty_levels:
            complexity += 0.4
        
        return min(complexity, 1.0)
    
    def _analyze_destination_requirements_enhanced(self, destination: Dict) -> Dict:
        """Enhanced destination requirement analysis."""
        city = destination["city"]
        city_config = DESTINATIONS_CONFIG[city]
        
        # Calculate months with better date handling
        try:
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
        
        except ValueError as e:
            logging.error(f"Date parsing error for {city}: {e}")
            months = []
        
        # Enhanced seasonal analysis
        seasonal_requirements = []
        temperature_progression = []
        
        for month in months:
            if month in city_config["seasons"]:
                season_data = city_config["seasons"][month]
                seasonal_requirements.append(season_data)
                temperature_progression.append({
                    "month": month,
                    "temp_range": season_data["temp_range"],
                    "weather": season_data["weather"]
                })
        
        return {
            "city": city,
            "duration_months": len(months),
            "months": months,
            "seasonal_requirements": seasonal_requirements,
            "temperature_progression": temperature_progression,
            "cultural_context": city_config["cultural_context"],
            "climate_profile": city_config["climate_profile"],
            "weight_priorities": city_config["weight_priorities"],
            "activity_types": city_config["activity_types"],
            "climate_recommendations": city_config["climate_recommendations"]
        }
    
    async def _get_travel_optimized_wardrobe_data_enhanced(self) -> Dict:
        """Enhanced wardrobe data retrieval with performance optimization."""
        try:
            logging.info("🧳 Acquiring travel-optimized wardrobe data...")
            
            # Use data manager's hierarchical fallback
            all_items = await asyncio.to_thread(
                wardrobe_data_manager.get_all_wardrobe_items
            )
            
            if not all_items:
                logging.error("❌ No wardrobe items available from any data source")
                self.metrics.data_source_used = "none"
                return {}
            
            self.metrics.data_source_used = "hierarchical_fallback"
            logging.info(f"✅ Retrieved {len(all_items)} wardrobe items")
            
            # Optimize categorization for travel context
            categorized_items = self._categorize_items_for_travel(all_items)
            
            return categorized_items
            
        except Exception as e:
            logging.error(f"❌ Error in wardrobe data acquisition: {e}", exc_info=True)
            return {}
    
    def _categorize_items_for_travel(self, all_items: List[Dict]) -> Dict:
        """Categorize and prioritize items for travel optimization."""
        categorized = {}
        
        # Priority categories for long-term travel
        priority_categories = [
            'Suit', 'Shirt', 'Chinos', 'Shoes',      # Business essentials
            'Polo', 'T-shirt', 'Jeans',              # Versatile basics
            'Sneakers', 'Hoodie', 'Jacket',          # Comfort & weather
            'Cargo Pants', 'Pants', 'Shorts',        # Additional bottoms
            'Crewneck', 'Fleece', 'Overcoat'         # Weather layers
        ]
        
        # Categorize all items
        for item in all_items:
            category = item.get('category', 'Unknown')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)
        
        # Sort each category by travel suitability
        travel_optimized = {}
        
        for category in priority_categories:
            if category in categorized:
                # Sort by multiple criteria for travel optimization
                sorted_items = sorted(
                    categorized[category],
                    key=lambda x: (
                        x.get('trip_worthy', False),           # Trip-worthy first
                        x.get('washed', '').lower() == 'done', # Clean items
                        len(x.get('aesthetic', [])),           # Versatility
                        len(x.get('weather', [])) != 1         # Multi-weather items
                    ),
                    reverse=True
                )
                travel_optimized[category] = sorted_items
                logging.info(f"   {category}: {len(sorted_items)} items (prioritized)")
        
        # Add remaining categories
        for category, items in categorized.items():
            if category not in travel_optimized:
                travel_optimized[category] = items
        
        return travel_optimized
    
    async def _execute_packing_optimization_chain(self, trip_config: Dict, 
                                                 available_items: Dict, 
                                                 pipeline_start: float) -> Dict:
        """Execute the hierarchical AI optimization chain."""
        
        # Stage 3a: Gemini API Attempt
        self.current_stage = PipelineStage.GEMINI_ATTEMPT
        stage_start = time.time()
        
        logging.info("🧳 Attempting Gemini API optimization...")
        success, packing_result, error_msg = await self._try_gemini_with_retry(
            trip_config, available_items
        )
        
        self.metrics.stage_timings[PipelineStage.GEMINI_ATTEMPT.value] = (time.time() - stage_start) * 1000
        
        if success and packing_result:
            self.metrics.ai_provider_used = "gemini"
            logging.info(f"✅ Gemini optimization successful ({self.metrics.stage_timings[PipelineStage.GEMINI_ATTEMPT.value]:.1f}ms)")
            return {
                "success": True,
                "data": packing_result,
                "generation_method": "gemini"
            }
        else:
            logging.warning(f"Gemini failed: {error_msg}")
        
        # Stage 3b: Groq API Attempt
        self.current_stage = PipelineStage.GROQ_ATTEMPT
        stage_start = time.time()
        
        logging.info("🧳 Attempting Groq API optimization...")
        success, packing_result, error_msg = await self._try_groq_with_retry(
            trip_config, available_items
        )
        
        self.metrics.stage_timings[PipelineStage.GROQ_ATTEMPT.value] = (time.time() - stage_start) * 1000
        
        if success and packing_result:
            self.metrics.ai_provider_used = "groq"
            logging.info(f"✅ Groq optimization successful ({self.metrics.stage_timings[PipelineStage.GROQ_ATTEMPT.value]:.1f}ms)")
            return {
                "success": True,
                "data": packing_result,
                "generation_method": "groq"
            }
        else:
            logging.warning(f"Groq failed: {error_msg}")
        
        # Stage 3c: Rule-Based Fallback
        self.current_stage = PipelineStage.FALLBACK_ATTEMPT
        stage_start = time.time()
        
        logging.info("🧳 Attempting rule-based fallback...")
        try:
            packing_result = await asyncio.to_thread(
                travel_logic_fallback.generate_fallback_packing_list,
                trip_config, available_items
            )
            
            self.metrics.stage_timings[PipelineStage.FALLBACK_ATTEMPT.value] = (time.time() - stage_start) * 1000
            
            if packing_result:
                self.metrics.ai_provider_used = "rule_based_fallback"
                logging.info(f"✅ Rule-based fallback successful ({self.metrics.stage_timings[PipelineStage.FALLBACK_ATTEMPT.value]:.1f}ms)")
                return {
                    "success": True,
                    "data": packing_result,
                    "generation_method": "rule_based_fallback"
                }
            else:
                logging.error("Rule-based fallback returned empty result")
        
        except Exception as e:
            logging.error(f"Rule-based fallback error: {e}", exc_info=True)
        
        # All methods failed
        return {
            "success": False,
            "error": "All optimization methods failed",
            "attempted_methods": ["gemini", "groq", "rule_based_fallback"]
        }
    
    async def _try_gemini_with_retry(self, trip_config: Dict, available_items: Dict) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Try Gemini API with retry logic and enhanced error logging."""
        for attempt in range(self.api_retry_attempts):
            try:
                if attempt > 0:
                    await asyncio.sleep(self.api_retry_delay * attempt)
                    logging.info(f"   Gemini retry attempt {attempt + 1}/{self.api_retry_attempts}")
            
                # Use a generous timeout for the AI
                return await travel_packing_agent.generate_multi_destination_packing_list(
                    trip_config, available_items, timeout=120
                )
            
            except asyncio.TimeoutError:
                logging.warning(f"   Gemini timeout on attempt {attempt + 1}")
                if attempt == self.api_retry_attempts - 1:
                    return False, None, f"Gemini timeout after {self.api_retry_attempts} attempts"
                
            except Exception as e:
                # THIS IS THE CRITICAL ADDITION: Log the specific error
                logging.error(f"   Gemini error on attempt {attempt + 1}: {str(e)}", exc_info=True)
                if attempt == self.api_retry_attempts - 1:
                    return False, None, f"Gemini error: {str(e)}"
    
        return False, None, "Gemini retry attempts exhausted"


    async def _try_groq_with_retry(self, trip_config: Dict, available_items: Dict) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Try Groq API with retry logic and enhanced error logging."""
        for attempt in range(self.api_retry_attempts):
            try:
                if attempt > 0:
                    await asyncio.sleep(self.api_retry_delay * attempt)
                    logging.info(f"   Groq retry attempt {attempt + 1}/{self.api_retry_attempts}")
            
                return await travel_packing_agent.generate_packing_list_with_groq(
                    trip_config, available_items, timeout=90
                )
            
            except asyncio.TimeoutError:
                logging.warning(f"   Groq timeout on attempt {attempt + 1}")
                if attempt == self.api_retry_attempts - 1:
                    return False, None, f"Groq timeout after {self.api_retry_attempts} attempts"
                
            except Exception as e:
                # THIS IS THE CRITICAL ADDITION: Log the specific error
                logging.error(f"   Groq error on attempt {attempt + 1}: {str(e)}", exc_info=True)
                if attempt == self.api_retry_attempts - 1:
                    return False, None, f"Groq error: {str(e)}"
    
        return False, None, "Groq retry attempts exhausted"
    
    async def _finalize_packing_results_enhanced(self, page_id: str, packing_result: Dict, 
                                               generation_method: str, trip_config: Dict,
                                               pipeline_start: float) -> Dict:
        """Enhanced results finalization with better error handling."""
        try:
            logging.info(f"🧳 Finalizing packing results using {generation_method}...")
            self._log_packing_summary(packing_result)
            
            # Step 1: Update trip-worthy selections with enhanced error handling
            await self._update_trip_worthy_selections_enhanced(packing_result["selected_items"])
            
            # Step 2: Clear and update Notion page
            await asyncio.to_thread(clear_page_content, page_id)
            await asyncio.to_thread(
                self._post_comprehensive_packing_guide_enhanced,
                page_id, packing_result, trip_config, generation_method
            )
            
            # Step 3: Clear trigger fields
            await asyncio.to_thread(self._clear_travel_trigger_fields_safe, page_id)
            
            # Calculate final metrics
            total_time = (time.time() - pipeline_start) * 1000
            
            # Build comprehensive result
            final_result = {
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
                "execution_time_ms": total_time,
                "data_sources_used": await self._get_data_source_info_safe()
            }
            
            logging.info(f"✅ Finalization completed successfully")
            return final_result
            
        except Exception as e:
            logging.error(f"❌ Error in results finalization: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Finalization failed: {str(e)}",
                "page_id": page_id,
                "generation_method": f"{generation_method}_finalization_failed"
            }
    
    def _log_packing_summary(self, packing_result: Dict) -> None:
        """Log packing result summary."""
        logging.info(f"🧳 Packing optimization results:")
        logging.info(f"   Items selected: {packing_result.get('total_items', 'unknown')}")
        logging.info(f"   Total weight: {packing_result.get('total_weight_kg', 'unknown')}kg")
        logging.info(f"   Weight efficiency: {packing_result.get('weight_efficiency', 'unknown')} outfits/kg")
        logging.info(f"   Business readiness: {packing_result.get('business_readiness', {}).get('readiness_score', 'unknown')}")
        logging.info(f"   Outfit combinations: {packing_result.get('outfit_analysis', {}).get('total_outfit_combinations', 'unknown')}")
    
    async def _update_trip_worthy_selections_enhanced(self, selected_items: List[Dict]) -> None:
        """Enhanced trip-worthy updates with better rate limiting and error handling."""
        try:
            logging.info(f"🧳 Updating trip-worthy selections for {len(selected_items)} items...")
            
            selected_ids = set(item['id'] for item in selected_items)
            all_items = await asyncio.to_thread(wardrobe_data_manager.get_all_wardrobe_items)
            
            if not all_items:
                logging.warning("No items available for trip-worthy updates")
                return
            
            # Enhanced batch processing with exponential backoff
            updated_count = 0
            failed_count = 0
            rate_limit_count = 0
            
            for i in range(0, len(all_items), self.batch_size):
                batch = all_items[i:i + self.batch_size]
                batch_start = time.time()
                
                for item in batch:
                    item_id = item['id']
                    should_be_selected = item_id in selected_ids
                    
                    try:
                        await asyncio.to_thread(
                            notion.pages.update,
                            page_id=item_id,
                            properties={"Trip-worthy": {"checkbox": should_be_selected}}
                        )
                        updated_count += 1
                        
                        if should_be_selected:
                            logging.debug(f"   ✅ Marked as trip-worthy: {item.get('item', item_id)}")
                    
                    except Exception as e:
                        # Handle rate limiting specifically
                        if "rate" in str(e).lower() or "429" in str(e):
                            rate_limit_count += 1
                            wait_time = min(2 ** rate_limit_count, 10)  # Exponential backoff, max 10s
                            logging.warning(f"Rate limit hit, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            
                            # Retry the failed item
                            try:
                                await asyncio.to_thread(
                                    notion.pages.update,
                                    page_id=item_id,
                                    properties={"Trip-worthy": {"checkbox": should_be_selected}}
                                )
                                updated_count += 1
                            except Exception as retry_e:
                                failed_count += 1
                                logging.warning(f"Retry failed for item {item_id}: {retry_e}")
                        else:
                            failed_count += 1
                            logging.warning(f"Failed to update item {item_id}: {e}")
                
                # Adaptive delay between batches
                batch_time = time.time() - batch_start
                if batch_time < 1.0:  # If batch completed quickly, add delay
                    await asyncio.sleep(self.batch_delay)
                
                if i % (self.batch_size * 5) == 0:  # Progress logging every 5 batches
                    logging.info(f"   Progress: {min(i + self.batch_size, len(all_items))}/{len(all_items)} items processed")
            
            logging.info(f"✅ Trip-worthy update completed:")
            logging.info(f"   Selected items: {len(selected_ids)}")
            logging.info(f"   Updated successfully: {updated_count}")
            logging.info(f"   Failed updates: {failed_count}")
            logging.info(f"   Rate limit encounters: {rate_limit_count}")
            
        except Exception as e:
            logging.error(f"❌ Error in trip-worthy updates: {e}", exc_info=True)
            # Don't raise - this shouldn't fail the entire pipeline
    
    def _post_comprehensive_packing_guide_enhanced(self, page_id: str, packing_result: Dict, 
                                                 trip_config: Dict, generation_method: str) -> None:
        """Enhanced packing guide generation with improved content structure."""
        try:
            logging.info("🧳 Building comprehensive packing guide...")
            
            # Build all guide sections
            guide_blocks = []
            
            # 1. Executive Summary
            guide_blocks.extend(self._create_executive_summary_blocks(packing_result, trip_config))
            
            # 2. Trip Overview
            guide_blocks.extend(self._create_trip_overview_blocks(trip_config))
            
            # 3. Packing Summary
            guide_blocks.extend(self._create_packing_summary_blocks(packing_result))
            
            # 4. Selected Items by Category
            guide_blocks.extend(self._create_selected_items_blocks(packing_result))
            
            # 5. Bag Allocation Strategy
            guide_blocks.extend(self._create_bag_allocation_blocks(packing_result))
            
            # 6. Outfit Analysis
            guide_blocks.extend(self._create_outfit_analysis_blocks(packing_result))
            
            # 7. Assessment & Recommendations
            guide_blocks.extend(self._create_assessment_blocks(packing_result))
            
            # 8. Packing Organization Guide
            guide_blocks.extend(self._create_packing_guide_blocks_section(packing_result))
            
            # 9. Destination-Specific Tips
            guide_blocks.extend(self._create_destination_tips_blocks(packing_result, trip_config))
            
            # 10. Generation Method Info
            guide_blocks.extend(self._create_generation_info_blocks(generation_method))
            
            logging.info(f"   Generated {len(guide_blocks)} content blocks")
            
            # Post to Notion in optimized chunks
            self._post_blocks_in_chunks(page_id, guide_blocks)
            
            logging.info("✅ Comprehensive packing guide posted successfully")
            
        except Exception as e:
            logging.error(f"❌ Error posting packing guide: {e}", exc_info=True)
            raise
    
    def _post_blocks_in_chunks(self, page_id: str, blocks: List[Dict]) -> None:
        """Post blocks to Notion in optimal chunks to avoid API limits."""
        chunk_size = 100  # Notion's API limit
        
        for i in range(0, len(blocks), chunk_size):
            chunk = blocks[i:i + chunk_size]
            try:
                notion.blocks.children.append(block_id=page_id, children=chunk)
                logging.debug(f"   Posted chunk {i//chunk_size + 1}/{(len(blocks)-1)//chunk_size + 1}")
                
                # Small delay between chunks to be API-friendly
                if i + chunk_size < len(blocks):
                    time.sleep(0.1)
                    
            except Exception as e:
                logging.error(f"Failed to post chunk {i//chunk_size + 1}: {e}")
                raise
    
    def _create_executive_summary_blocks(self, packing_result: Dict, trip_config: Dict) -> List[Dict]:
        """Create executive summary section."""
        overview = trip_config["trip_overview"]
        
        return [
            {
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": "🧳 AI Travel Packing Guide"}}]
                }
            },
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": f"Optimized packing for {overview['destination_count']} destinations over {overview['total_duration_months']} months. Total weight: {packing_result['total_weight_kg']}kg with {packing_result['outfit_analysis']['total_outfit_combinations']} outfit combinations."}}],
                    "icon": {"emoji": "✈️"}
                }
            }
        ]
    
    def _create_trip_overview_blocks(self, trip_config: Dict) -> List[Dict]:
        """Create trip overview section."""
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
                    "rich_text": [{"type": "text", "text": {"content": f"Temperature Range: {overview['temperature_range']['min']}°C to {overview['temperature_range']['max']}°C ({overview['temperature_range']['span']}°C span)"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Climate Diversity: {overview['climate_diversity']} different climate types"}}]
                }
            }
        ]
        
        # Add destination details
        for dest in destinations:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"📍 {dest['city'].title()}: {dest['start_date']} to {dest['end_date']}"}}]
                }
            })
        
        return blocks
    
    def _create_packing_summary_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create packing summary section."""
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
                    "rich_text": [{"type": "text", "text": {"content": f"Total Weight: {packing_result['total_weight_kg']}kg (Budget: 18kg)"}}]
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
                    "rich_text": [{"type": "text", "text": {"content": f"Business Readiness Score: {packing_result['business_readiness']['readiness_score']}"}}]
                }
            }
        ]
    
    def _create_selected_items_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create selected items section organized by category."""
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
        
        # Add category sections with improved formatting
        for category, items in sorted(items_by_category.items()):
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
                item_details = f"{item['item']}"
                if aesthetics:
                    item_details += f" | {aesthetics}"
                if weather:
                    item_details += f" | Weather: {weather}"
                
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": item_details}}]
                    }
                })
        
        return blocks
    
    def _create_bag_allocation_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create bag allocation section."""
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
                    "rich_text": [{"type": "text", "text": {"content": f"✈️ Checked Bag ({allocation['checked_bag']['weight_kg']}kg / 15kg)"}}]
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
                "rich_text": [{"type": "text", "text": {"content": f"🎒 Cabin Bag ({allocation['cabin_bag']['weight_kg']}kg / 3kg)"}}]
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
        
        # Add strategy notes
        if allocation.get('strategy_notes'):
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": "Strategy: " + '; '.join(allocation['strategy_notes'])}}]
                }
            })
        
        return blocks
    
    def _create_outfit_analysis_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create outfit analysis section."""
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
                    "rich_text": [{"type": "text", "text": {"content": f"🤵 Business Formal Outfits: {analysis['business_formal_outfits']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"👔 Business Casual Outfits: {analysis['business_casual_outfits']}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"👕 Casual Outfits: {analysis['casual_outfits']}"}}]
                }
            },
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": f"Total Outfit Combinations: {analysis['total_outfit_combinations']}"}}],
                    "icon": {"emoji": "🎯"}
                }
            }
        ]
    # In core/travel_pipeline_orchestrator.py

    def _create_assessment_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create comprehensive assessment section."""
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "✅ Comprehensive Assessment"}}]
                }
            }
        ]

        # Business readiness
        business = packing_result["business_readiness"]
        status_emoji = "✅" if business["meets_requirements"] else "⚠️"
    
        blocks.extend([
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": f"{status_emoji} Business Readiness"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Readiness Score: {business.get('readiness_score', 0)}/1.0"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Business Suits: {business.get('suits_count', 0)} (minimum 2 recommended)"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    # THIS IS THE CORRECTED LINE
                    "rich_text": [{"type": "text", "text": {"content": f"Dress Shoes: {business.get('dress_shoes_count', 0)}"}}]
                }
            }
        ])
    
        # Climate coverage
        climate = packing_result["climate_coverage"]
        blocks.extend([
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "🌡️ Climate Coverage"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Temperature Range: {climate.get('temperature_range_covered', 'N/A')}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Coverage Quality: {str(climate.get('coverage_adequacy', 'Unknown')).replace('_', ' ').title()}"}}]
                }
            }
        ])
    
        # Cultural compliance
        cultural = packing_result["cultural_compliance"]
        cultural_emoji = "✅" if cultural.get("dubai_ready") else "⚠️"
    
        blocks.extend([
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": f"{cultural_emoji} Cultural Compliance"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Dubai Readiness: {'Ready' if cultural.get('dubai_ready') else 'Needs Review'}"}}]
                }
            },
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": f"Modest Items: {cultural.get('modest_items_count', 0)}/{cultural.get('total_items', 0)}"}}]
                }
            }
        ])
    
        return blocks
    
   # In core/travel_pipeline_orchestrator.py

    def _create_packing_guide_blocks_section(self, packing_result: Dict) -> List[Dict]:
        """Create packing organization guide section."""
        guide = packing_result.get("packing_guide", {})
    
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
                    "rich_text": [{"type": "text", "text": {"content": "🎯 Packing Techniques"}}]
                }
            }
        ]
    
        # Safely get packing techniques
        techniques = guide.get("packing_techniques", ["No techniques provided."])
        for technique in techniques:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [{"type": "text", "text": {"content": technique}}]
                }
            })
    
        # Safely get travel day strategy
        travel_strategy = guide.get("travel_day_strategy", {})
        if travel_strategy:
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "✈️ Travel Day Strategy"}}]
                }
            })
            wear_during_travel = travel_strategy.get("wear_during_travel", ["Not specified."])
            cabin_essentials = travel_strategy.get("cabin_essentials", ["Not specified."])
        
            blocks.extend([
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": f"Wear during travel: {', '.join(wear_during_travel)}"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": f"Cabin essentials: {', '.join(cabin_essentials)}"}}]
                    }   
                }
            ])
    
        return blocks
    
    # In core/travel_pipeline_orchestrator.py
    def _create_destination_tips_blocks(self, packing_result: Dict, trip_config: Dict) -> List[Dict]:
        """Create destination-specific tips section."""
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
                    "rich_text": [{"type": "text", "text": {"content": f"📍 {city.title()}"}}]
                }
            })
        
            city_tips = trip_tips.get(city, {})
            # Safely iterate over items, ensuring city_tips is a dictionary
            if isinstance(city_tips, dict):
                for tip_category, tips in city_tips.items():
                    if tips and isinstance(tips, list):
                        category_name = tip_category.replace('_', ' ').title()
                        for tip in tips:
                            blocks.append({
                                "object": "block",
                                "type": "bulleted_list_item",
                                "bulleted_list_item": {
                                    "rich_text": [{"type": "text", "text": {"content": f"{tip}"}}]
                                }
                            })
        return blocks
    
    def _create_generation_info_blocks(self, generation_method: str) -> List[Dict]:
        """Create generation method information section."""
        method_descriptions = {
            "gemini": "Generated using Google's Gemini AI with advanced travel optimization algorithms",
            "groq": "Generated using Groq AI with specialized packing optimization prompts",
            "rule_based_fallback": "Generated using rule-based algorithms with business travel optimization"
        }
        
        return [
            {
                "object": "block",
                "type": "divider",
                "divider": {}
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": f"🤖 {method_descriptions.get(generation_method, f'Generated using {generation_method}')}"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M UTC')}"}}]
                }
            }
        ]
    
    def _clear_travel_trigger_fields_safe(self, page_id: str) -> None:
        """Safely clear travel trigger fields with comprehensive error handling."""
        try:
            logging.info(f"🧳 Safely clearing travel trigger fields for page {page_id}")
            
            # First, retrieve the page to check available properties
            page = notion.pages.retrieve(page_id=page_id)
            properties = page.get("properties", {})
            
            # Build update payload only for existing properties
            update_properties = {}
            
            # Possible trigger field names to check and clear
            trigger_fields = {
                "Generate": {"checkbox": False},
                "Generate Travel Packing": {"checkbox": False},
                "Generate Packing": {"checkbox": False},
                "Travel Generate": {"checkbox": False}
            }
            
            text_fields = {
                "Travel Preferences": {"rich_text": []},
                "Preferences": {"rich_text": []},
                "Notes": {"rich_text": []}
            }
            
            multi_select_fields = {
                "Destinations": {"multi_select": []}
            }
            
            # Check and add existing fields to update payload
            for field_name, field_value in {**trigger_fields, **text_fields, **multi_select_fields}.items():
                if field_name in properties:
                    update_properties[field_name] = field_value
                    logging.debug(f"   Will clear: {field_name}")
            
            # Only update if there are properties to clear
            if update_properties:
                notion.pages.update(page_id=page_id, properties=update_properties)
                logging.info(f"✅ Cleared {len(update_properties)} trigger fields")
            else:
                logging.info("ℹ️  No trigger fields found to clear")
                
        except Exception as e:
            # Don't fail the pipeline for trigger field clearing
            logging.warning(f"⚠️  Could not clear trigger fields (non-critical): {e}")
    
    async def _get_data_source_info_safe(self) -> Dict:
        """Safely get data source information."""
        try:
            return await asyncio.to_thread(wardrobe_data_manager.get_data_stats)
        except Exception as e:
            logging.warning(f"Could not retrieve data source stats: {e}")
            return {"error": "Stats unavailable"}
    
    def _create_error_result(self, trigger_data: Dict, generation_method: str, 
                           error_message: str, pipeline_start: float) -> Dict:
        """Create standardized error result."""
        return {
            "success": False,
            "error": error_message,
            "page_id": trigger_data.get("page_id"),
            "generation_method": generation_method,
            "execution_time_ms": (time.time() - pipeline_start) * 1000,
            "current_stage": self.current_stage.value,
            "stage_timings": self.metrics.stage_timings
        }


# Global instance creation with error handling
try:
    travel_pipeline_orchestrator = TravelPipelineOrchestrator()
    logging.info("✅ Global travel_pipeline_orchestrator created successfully")
except Exception as e:
    logging.error(f"❌ Failed to create global travel_pipeline_orchestrator: {e}")
    travel_pipeline_orchestrator = None


# Testing and Development Functions
async def test_travel_packing_pipeline(destinations: List[Dict] = None, 
                                     preferences: Dict = None) -> None:
    """
    Comprehensive test function for travel packing pipeline.
    
    Args:
        destinations: List of destination dictionaries
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
            "trip_type": "business_school_relocation",
            "user_notes": "Test run for AI travel packing system"
        }
    
    # Build trigger data
    trigger_data = {
        "page_id": travel_pipeline_orchestrator.packing_guide_page_id if travel_pipeline_orchestrator else "test_page_id",
        "destinations": destinations,
        "preferences": preferences
    }
    
    print(f"🧳 Testing travel packing pipeline...")
    print(f"   Destinations: {[d['city'].title() for d in destinations]}")
    print(f"   Duration: {sum((datetime.strptime(d['end_date'], '%Y-%m-%d') - datetime.strptime(d['start_date'], '%Y-%m-%d')).days for d in destinations)} days")
    print(f"   Orchestrator available: {travel_pipeline_orchestrator is not None}")
    
    if not travel_pipeline_orchestrator:
        print("❌ Travel pipeline orchestrator not initialized!")
        return
    
    try:
        # Run the pipeline
        start_time = time.time()
        result = await travel_pipeline_orchestrator.run_travel_packing_pipeline(trigger_data)
        execution_time = (time.time() - start_time) * 1000
        
        # Display results
        if result["success"]:
            print(f"\n✅ Travel packing pipeline test SUCCESSFUL!")
            print(f"   Execution time: {execution_time:.1f}ms")
            print(f"   Generation method: {result['generation_method']}")
            print(f"   Items selected: {result['total_items_selected']}")
            print(f"   Total weight: {result['total_weight_kg']}kg / 18kg budget")
            print(f"   Weight efficiency: {result['weight_efficiency']} outfits/kg")
            print(f"   Business readiness: {result['business_readiness']}")
            print(f"   Destinations: {result['destinations']}")
            print(f"   Trip duration: {result['trip_duration_months']} months")
            print(f"   Outfit possibilities: {result['outfit_possibilities']}")
            
            # Bag allocation details
            bag_alloc = result['bag_allocation']
            print(f"   Bag allocation:")
            print(f"     Checked: {bag_alloc['checked_weight']}kg")
            print(f"     Cabin: {bag_alloc['cabin_weight']}kg")
            
            # Performance metrics if available
            if 'performance_metrics' in result:
                perf = result['performance_metrics']
                print(f"   Performance metrics:")
                print(f"     Items processed: {perf['items_processed']}")
                print(f"     Data source: {perf['data_source']}")
                print(f"     AI provider: {perf['ai_provider']}")
                
                if 'stage_timings' in perf:
                    print(f"     Stage timings:")
                    for stage, timing in perf['stage_timings'].items():
                        print(f"       {stage}: {timing:.1f}ms")
        else:
            print(f"\n❌ Travel packing pipeline test FAILED!")
            print(f"   Execution time: {execution_time:.1f}ms")
            print(f"   Error: {result['error']}")
            print(f"   Generation method: {result['generation_method']}")
            
            if 'attempted_methods' in result:
                print(f"   Attempted methods: {result['attempted_methods']}")
            
            if 'current_stage' in result:
                print(f"   Failed at stage: {result['current_stage']}")
                
    except Exception as e:
        print(f"\n💥 Test crashed with exception: {e}")
        logging.error(f"❌ Test error: {e}", exc_info=True)


def run_test_travel_packing_pipeline(destinations: List[Dict] = None, 
                                   preferences: Dict = None) -> None:
    """Synchronous wrapper for async test function."""
    return asyncio.run(test_travel_packing_pipeline(destinations, preferences))


# Health Check Functions
async def health_check_travel_orchestrator() -> Dict[str, Any]:
    """Comprehensive health check for travel orchestrator."""
    health_status = {
        "orchestrator_initialized": travel_pipeline_orchestrator is not None,
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    if not travel_pipeline_orchestrator:
        health_status["status"] = "unhealthy"
        health_status["error"] = "Orchestrator not initialized"
        return health_status
    
    try:
        # Check environment variables
        env_check = {
            "notion_token": bool(os.getenv("NOTION_TOKEN")),
            "packing_guide_id": bool(os.getenv("NOTION_PACKING_GUIDE_ID")),
            "wardrobe_db_id": bool(os.getenv("NOTION_WARDROBE_DB_ID")),
            "ai_keys": {
                "gemini": bool(os.getenv("GEMINI_AI_API_KEY")),
                "groq": bool(os.getenv("GROQ_AI_API_KEY"))
            }
        }
        health_status["checks"]["environment"] = env_check
        
        # Check Notion connectivity
        try:
            await asyncio.to_thread(
                notion.pages.retrieve,
                page_id=travel_pipeline_orchestrator.packing_guide_page_id
            )
            health_status["checks"]["notion_connectivity"] = {"status": "healthy"}
        except Exception as e:
            health_status["checks"]["notion_connectivity"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
        
        # Check data manager
        try:
            data_stats = await asyncio.to_thread(wardrobe_data_manager.get_data_stats)
            health_status["checks"]["data_manager"] = {
                "status": "healthy",
                "stats": data_stats
            }
        except Exception as e:
            health_status["checks"]["data_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check AI agents
        try:
            # Test if agents are importable and have required methods
            agent_status = {
                "travel_packing_agent": hasattr(travel_packing_agent, 'generate_multi_destination_packing_list'),
                "travel_logic_fallback": hasattr(travel_logic_fallback, 'generate_fallback_packing_list')
            }
            health_status["checks"]["ai_agents"] = {
                "status": "healthy" if all(agent_status.values()) else "degraded",
                "agents": agent_status
            }
        except Exception as e:
            health_status["checks"]["ai_agents"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall health determination
        check_results = [check.get("status", "unknown") for check in health_status["checks"].values()]
        if all(status == "healthy" for status in check_results):
            health_status["status"] = "healthy"
        elif any(status == "unhealthy" for status in check_results):
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "unknown"
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = f"Health check failed: {str(e)}"
    
    return health_status


def run_health_check_travel_orchestrator() -> Dict[str, Any]:
    """Synchronous wrapper for health check."""
    return asyncio.run(health_check_travel_orchestrator())


# Legacy compatibility functions (DEPRECATED but maintained for backward compatibility)
def run_travel_packing_pipeline(trigger_data: Dict) -> Dict:
    """
    Legacy synchronous wrapper for backward compatibility.
    
    DEPRECATED: Use async run_travel_packing_pipeline() directly with asyncio.run() or await
    
    Args:
        trigger_data: Trip configuration and preferences
        
    Returns:
        Pipeline execution result
    """
    logging.warning("Using legacy run_travel_packing_pipeline wrapper - consider updating to async")
    
    if not travel_pipeline_orchestrator:
        raise RuntimeError("Travel pipeline orchestrator not initialized")
    
    return asyncio.run(travel_pipeline_orchestrator.run_travel_packing_pipeline(trigger_data))


# Configuration and utility functions
def get_orchestrator_config() -> Dict[str, Any]:
    """Get current orchestrator configuration for debugging."""
    if not travel_pipeline_orchestrator:
        return {"error": "Orchestrator not initialized"}
    
    return {
        "packing_guide_page_id": travel_pipeline_orchestrator.packing_guide_page_id,
        "wardrobe_db_id": travel_pipeline_orchestrator.wardrobe_db_id,
        "batch_size": travel_pipeline_orchestrator.batch_size,
        "batch_delay": travel_pipeline_orchestrator.batch_delay,
        "api_retry_attempts": travel_pipeline_orchestrator.api_retry_attempts,
        "api_retry_delay": travel_pipeline_orchestrator.api_retry_delay,
        "current_stage": travel_pipeline_orchestrator.current_stage.value if travel_pipeline_orchestrator.current_stage else None,
        "monitoring_enabled": MONITORING_ENABLED
    }


def update_orchestrator_config(**kwargs) -> bool:
    """Update orchestrator configuration parameters."""
    if not travel_pipeline_orchestrator:
        return False
    
    try:
        for key, value in kwargs.items():
            if hasattr(travel_pipeline_orchestrator, key):
                setattr(travel_pipeline_orchestrator, key, value)
                logging.info(f"Updated {key} to {value}")
            else:
                logging.warning(f"Unknown configuration parameter: {key}")
        
        return True
    except Exception as e:
        logging.error(f"Failed to update configuration: {e}")
        return False


# Module information
__version__ = "2.0.0"
__author__ = "AI Wardrobe Management System"
__description__ = "Enhanced travel pipeline orchestrator with comprehensive monitoring and error handling"

# Export key functions and classes for external use
__all__ = [
    "TravelPipelineOrchestrator",
    "travel_pipeline_orchestrator",
    "test_travel_packing_pipeline",
    "run_test_travel_packing_pipeline",
    "health_check_travel_orchestrator",
    "run_health_check_travel_orchestrator",
    "run_travel_packing_pipeline",  # Legacy compatibility
    "get_orchestrator_config",
    "update_orchestrator_config",
    "PipelineStage",
    "PipelineMetrics"
]