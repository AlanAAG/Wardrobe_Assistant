import logging
from data.notion_utils import (
    get_checked_items_from_page,
    add_items_to_dirty_clothes_db,
    uncheck_hamper_trigger,
    update_clothing_washed_status
)

class HamperPipelineOrchestrator:
    """
    Orchestrates the "Send to Hamper" workflow.
    """

    def __init__(self):
        pass

    async def run_hamper_pipeline(self, page_id: str):
        """
        Main pipeline for the "Send to Hamper" workflow.
        """
        try:
            logging.info(f"🧺 Starting 'Send to Hamper' pipeline for page {page_id}...")

            # 1. Get checked items from the page
            checked_items = get_checked_items_from_page(page_id)
            if not checked_items:
                logging.warning("No checked items found to send to hamper.")
                # Still need to uncheck the trigger
                uncheck_hamper_trigger(page_id)
                return {"success": True, "message": "No items to send to hamper."}

            # 2. Add items to the "Dirty Clothes" database
            add_items_to_dirty_clothes_db(checked_items, page_id)

            # 3. Uncheck the "Send to Hamper" trigger
            uncheck_hamper_trigger(page_id)

            logging.info(f"✅ 'Send to Hamper' pipeline completed successfully for page {page_id}.")
            return {"success": True}

        except Exception as e:
            logging.error(f"❌ Critical pipeline error in 'Send to Hamper' pipeline: {str(e)}", exc_info=True)
            # It's important to still try and uncheck the trigger to prevent re-triggering.
            try:
                uncheck_hamper_trigger(page_id)
                logging.info(f"Unchecked hamper trigger for page {page_id} after error.")
            except Exception as uncheck_e:
                logging.error(f"Failed to uncheck hamper trigger for page {page_id} after error: {uncheck_e}", exc_info=True)

            return {"success": False, "error": f"Critical pipeline error: {str(e)}"}

hamper_pipeline_orchestrator = HamperPipelineOrchestrator()
