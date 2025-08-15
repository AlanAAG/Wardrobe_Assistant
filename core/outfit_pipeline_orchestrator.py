import logging
from typing import Dict

from data.data_manager import wardrobe_data_manager
from data.weather_utils import get_weather_forecast
from core.outfit_planner_agent import outfit_planner_agent
from data.notion_utils import (
    get_selected_aesthetic_from_output_db,
    get_output_page_id,
    post_outfit_to_notion_page,
    clear_page_content,
    clear_trigger_fields,
    OUTPUT_DB_ID,
    get_checked_todo_items_from_page,
    create_page_in_dirty_clothes_db,
    update_items_washed_status
)


class OutfitPipelineOrchestrator:
    """
    Orchestrates the daily outfit generation pipeline.
    """

    def __init__(self):
        pass

    async def _add_outfit_to_dirty_clothes(self, page_id: str):
        """
        Adds worn items to the "Dirty Clothes" database.
        """
        try:
            logging.info("🧺 Adding worn items to dirty clothes database...")
            checked_items = get_checked_todo_items_from_page(page_id)
            for item in checked_items:
                # Add to dirty clothes DB
                create_page_in_dirty_clothes_db(
                    item_name=item["name"],
                    clothing_item_id=item["id"],
                    outfit_log_id=page_id
                )
                # Mark as "Not Done" in main wardrobe
                update_items_washed_status(item["id"], "Not Done")
            logging.info(f"Added {len(checked_items)} items to dirty clothes database.")
        except Exception as e:
            logging.error(f"Error adding items to dirty clothes database: {e}", exc_info=True)

    async def run_daily_outfit_pipeline(self) -> Dict:
        """
        Main pipeline execution for generating a daily outfit.
        """
        try:
            logging.info("👕 Starting daily outfit pipeline...")

            # 1. Get wardrobe data
            wardrobe_items = wardrobe_data_manager.get_all_wardrobe_items()
            if not wardrobe_items:
                return {"success": False, "error": "No wardrobe items available."}

            # 2. Get weather forecast
            forecast = get_weather_forecast()
            weather_tag = forecast["weather_tag"]

            # 3. Get user preferences
            selected_aesthetics = get_selected_aesthetic_from_output_db(OUTPUT_DB_ID)
            desired_aesthetic = selected_aesthetics[0] if selected_aesthetics else "Minimalist"

            # 4. Generate outfit
            result = await outfit_planner_agent.generate_outfit(
                wardrobe_items, weather_tag, desired_aesthetic
            )

            if not result["success"]:
                return result

            # 5. Post outfit to Notion
            output_page_id = get_output_page_id(OUTPUT_DB_ID)
            if not output_page_id:
                return {"success": False, "error": "Could not find the output page in Notion."}

            clear_page_content(output_page_id)
            post_outfit_to_notion_page(output_page_id, result["outfit"])

            # 6. Add worn items to dirty clothes database
            await self._add_outfit_to_dirty_clothes(output_page_id)

            clear_trigger_fields(output_page_id)

            logging.info("✅ Daily outfit pipeline completed successfully.")
            return {"success": True}

        except Exception as e:
            logging.error(f"❌ Critical pipeline error in daily outfit pipeline: {str(e)}", exc_info=True)
            return {"success": False, "error": f"Critical pipeline error: {str(e)}"}

outfit_pipeline_orchestrator = OutfitPipelineOrchestrator()
