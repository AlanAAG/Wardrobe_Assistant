import logging
from typing import Dict

from data.data_manager import wardrobe_data_manager
from data.weather_utils import get_weather_forecast
from core.outfit_planner_agent import outfit_planner_agent
from data.notion_utils import (
    get_selected_aesthetic_from_output_db,
    get_selected_color_from_output_db,
    get_output_page_id,
    post_outfit_to_notion_page,
    clear_page_content,
    clear_trigger_fields,
    OUTPUT_DB_ID,
    get_checked_items_from_page,
    create_page_in_dirty_clothes_db,
    update_wardrobe_item_status
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

    async def run_daily_outfit_pipeline(self) -> Dict:
        """
        Main pipeline execution for generating a daily outfit.
        """
        try:
            logging.info("👕 Starting daily outfit pipeline...")

            # 1. Get user preferences from Notion
            selected_aesthetics = get_selected_aesthetic_from_output_db(OUTPUT_DB_ID)
            desired_aesthetic = selected_aesthetics[0] if selected_aesthetics else "Minimalist"
            desired_color = get_selected_color_from_output_db(OUTPUT_DB_ID)

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
                return result

            # 5. Post outfit to Notion
            output_page_id = get_output_page_id(OUTPUT_DB_ID)
            if not output_page_id:
                return {"success": False, "error": "Could not find the output page in Notion."}

            clear_page_content(output_page_id)
            post_outfit_to_notion_page(output_page_id, result["outfit"])

            # 6. Add worn items to dirty clothes database
            await self._add_outfit_to_dirty_clothes(
                outfit_items=result["outfit"],
                outfit_log_id=output_page_id
            )

            clear_trigger_fields(output_page_id)

            logging.info("✅ Daily outfit pipeline completed successfully.")
            return {"success": True}

        except Exception as e:
            logging.error(f"❌ Critical pipeline error in daily outfit pipeline: {str(e)}", exc_info=True)
            return {"success": False, "error": f"Critical pipeline error: {str(e)}"}

outfit_pipeline_orchestrator = OutfitPipelineOrchestrator()
