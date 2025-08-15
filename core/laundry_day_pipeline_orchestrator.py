import logging
from data.notion_utils import (
    get_related_wardrobe_item_id,
    update_wardrobe_item_status,
    delete_page
)

class LaundryDayPipelineOrchestrator:
    """
    Orchestrates the "Laundry Day" workflow.
    """

    def __init__(self):
        pass

    async def run_laundry_day_pipeline(self, page_id: str):
        """
        Main pipeline for the "Laundry Day" workflow.
        """
        try:
            logging.info(f"🧺 Starting 'Laundry Day' pipeline for page {page_id}...")

            # 1. Get the related wardrobe item ID
            wardrobe_item_id = get_related_wardrobe_item_id(page_id)
            if not wardrobe_item_id:
                logging.error(f"Could not find related wardrobe item for page {page_id}")
                return {"success": False, "error": "Could not find related wardrobe item."}

            # 2. Update the wardrobe item's status to "Done"
            update_wardrobe_item_status(wardrobe_item_id, "Done")

            # 3. Delete the page from the "Dirty Clothes" database
            delete_page(page_id)

            logging.info(f"✅ 'Laundry Day' pipeline completed successfully for page {page_id}.")
            return {"success": True}

        except Exception as e:
            logging.error(f"❌ Critical pipeline error in 'Laundry Day' pipeline: {str(e)}", exc_info=True)
            return {"success": False, "error": f"Critical pipeline error: {str(e)}"}

laundry_day_pipeline_orchestrator = LaundryDayPipelineOrchestrator()
