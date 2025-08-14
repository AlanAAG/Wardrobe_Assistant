import logging
import asyncio
import time
import os
from typing import Dict

from dotenv import load_dotenv

from core.trip_configurator import TripConfigurator
from core.ai_packing_optimizer import AIPackingOptimizer
from core.notion_result_publisher import NotionResultPublisher
from data.data_manager import wardrobe_data_manager
from data.notion_utils import notion

load_dotenv()

class TravelPipelineOrchestrator:
    """
    Orchestrates the travel packing pipeline by coordinating various components.
    """

    def __init__(self):
        self._start_time = time.time()
        self.packing_guide_page_id = None
        self.wardrobe_db_id = None
        logging.info("🔧 Initializing TravelPipelineOrchestrator...")

    async def ensure_ready(self):
        """Validates environment and confirms Notion connectivity."""
        self._validate_environment()
        self.packing_guide_page_id = os.getenv("NOTION_PACKING_GUIDE_ID")
        self.wardrobe_db_id = os.getenv("NOTION_WARDROBE_DB_ID")
        await self._test_notion_connectivity()
        logging.info("✅ TravelPipelineOrchestrator initialized successfully")

    def _validate_environment(self) -> None:
        """Validates all required environment variables."""
        required_vars = [
            'NOTION_TOKEN', 'NOTION_PACKING_GUIDE_ID', 'NOTION_WARDROBE_DB_ID'
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            error_msg = f"Missing required environment variables: {missing_vars}"
            logging.error(f"❌ {error_msg}")
            raise EnvironmentError(error_msg)

    async def _test_notion_connectivity(self) -> None:
        """Tests Notion API connectivity."""
        try:
            await asyncio.to_thread(
                notion.pages.retrieve, page_id=self.packing_guide_page_id
            )
            logging.info("✅ Notion connectivity test successful")
        except Exception as e:
            error_msg = f"Notion connectivity test failed: {e}"
            logging.error(f"❌ {error_msg}")
            raise ConnectionError(error_msg)

    async def run_travel_packing_pipeline(self, trigger_data: Dict) -> Dict:
        """Main pipeline execution."""
        await self.ensure_ready()
        pipeline_start = time.time()

        try:
            # 1. Prepare trip configuration
            trip_configurator = TripConfigurator(trigger_data)
            trip_config = trip_configurator.prepare_trip_configuration()
            if not trip_config:
                return self._create_error_result("Invalid trip configuration", pipeline_start)

            # 2. Get wardrobe data
            available_items = await self._get_travel_optimized_wardrobe_data()
            if not available_items:
                return self._create_error_result("No wardrobe items available", pipeline_start)

            # 3. Run AI packing optimization
            optimizer = AIPackingOptimizer()
            packing_result = await optimizer.execute_packing_optimization_chain(
                trip_config, available_items
            )
            if not packing_result["success"]:
                return self._create_error_result(packing_result["error"], pipeline_start)

            # 4. Finalize and publish results
            publisher = NotionResultPublisher()
            final_result = await publisher.finalize_packing_results(
                trigger_data.get("page_id"),
                packing_result["data"],
                packing_result["generation_method"],
                trip_config,
            )
            if not final_result["success"]:
                return self._create_error_result(final_result["error"], pipeline_start)

            total_time = (time.time() - pipeline_start) * 1000
            logging.info(f"🎉 Travel packing pipeline completed successfully in {total_time:.1f}ms")
            return {"success": True, "execution_time_ms": total_time}

        except Exception as e:
            logging.error(f"❌ Critical pipeline error: {str(e)}", exc_info=True)
            return self._create_error_result(f"Critical pipeline error: {str(e)}", pipeline_start)

    async def _get_travel_optimized_wardrobe_data(self) -> Dict:
        """Retrieves and categorizes wardrobe data for travel."""
        try:
            logging.info("🧳 Acquiring travel-optimized wardrobe data...")
            all_items = await asyncio.to_thread(wardrobe_data_manager.get_all_wardrobe_items)
            if not all_items:
                logging.error("❌ No wardrobe items available from any data source")
                return {}
            logging.info(f"✅ Retrieved {len(all_items)} wardrobe items")
            return self._categorize_items_for_travel(all_items)
        except Exception as e:
            logging.error(f"❌ Error in wardrobe data acquisition: {e}", exc_info=True)
            return {}

    def _categorize_items_for_travel(self, all_items: list) -> Dict:
        """Categorizes items for travel optimization."""
        categorized = {}
        for item in all_items:
            category = item.get('category', 'Unknown')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)
        return categorized

    def _create_error_result(self, error_message: str, pipeline_start: float) -> Dict:
        """Creates a standardized error result."""
        return {
            "success": False,
            "error": error_message,
            "execution_time_ms": (time.time() - pipeline_start) * 1000,
        }

travel_pipeline_orchestrator = TravelPipelineOrchestrator()