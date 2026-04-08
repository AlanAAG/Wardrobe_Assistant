import logging
from typing import Dict

from data.data_manager import wardrobe_data_manager
from data.weather_utils import get_weather_forecast
from core.outfit_planner_agent import outfit_planner_agent
from data.notion_utils import (
    get_selected_aesthetic_for_page,
    get_selected_color_for_page,
    post_outfit_to_notion_page,
    clear_page_content,
    clear_trigger_fields,
    get_checked_items_from_page,
    create_page_in_dirty_clothes_db,
    update_wardrobe_item_status,
    update_page_status,
)


class OutfitPipelineOrchestrator:
    """
    Orchestrates the daily outfit generation pipeline.
    """

    def __init__(self):
        pass

    async def _add_outfit_to_dirty_clothes(self, outfit_items: list, outfit_log_id: str):
        """
        Adds worn items to the "Dirty Clothes" database.
        """
        logging.info("🧺 Adding worn items to dirty clothes database...")
        for item in outfit_items:
            # Add to dirty clothes DB
            create_page_in_dirty_clothes_db(
                item_name=item["item"],
                clothing_item_id=item["id"],
                outfit_log_id=outfit_log_id
            )
            # Mark as "Not Done" in main wardrobe
            update_wardrobe_item_status(item["id"], "Not Done")
        logging.info(f"Added {len(outfit_items)} items to dirty clothes database.")

    async def run_daily_outfit_pipeline(self, page_id: str) -> Dict:
        """
        Main pipeline execution for generating a daily outfit.
        """
        if not page_id:
            return {"success": False, "error": "No page_id provided."}
        try:
            logging.info(f"👕 Starting daily outfit pipeline for page {page_id}...")
            update_page_status(page_id, "In Progress")

            # 1. Get user preferences from Notion
            selected_aesthetics = get_selected_aesthetic_for_page(page_id)
            desired_aesthetic = selected_aesthetics[0] if selected_aesthetics else "Minimalist"
            desired_color = get_selected_color_for_page(page_id)

            # 2. Get weather forecast
            forecast = get_weather_forecast()
            weather_tag = forecast["weather_tag"]

            # 3. Get LLM-optimized context from data manager
            context = wardrobe_data_manager.get_llm_optimized_context(
                aesthetic=desired_aesthetic,
                weather_tag=weather_tag,
                color_tag=desired_color
            )

            # 4. Generate outfit
            result = await outfit_planner_agent.generate_outfit(context)

            if not result["success"]:
                update_page_status(page_id, "Failed")
                return result

            # 5. Post outfit to Notion
            clear_page_content(page_id)
            post_outfit_to_notion_page(page_id, result["outfit"])

            # 6. Add worn items to dirty clothes database
            await self._add_outfit_to_dirty_clothes(
                outfit_items=result["outfit"],
                outfit_log_id=page_id
            )

            update_page_status(page_id, "Complete")

            logging.info("✅ Daily outfit pipeline completed successfully.")
            return {"success": True}

        except Exception as e:
            logging.error(f"❌ Critical pipeline error in daily outfit pipeline: {str(e)}", exc_info=True)
            if page_id:
                update_page_status(page_id, "Failed")
            return {"success": False, "error": f"Critical pipeline error: {str(e)}"}

outfit_pipeline_orchestrator = OutfitPipelineOrchestrator()
