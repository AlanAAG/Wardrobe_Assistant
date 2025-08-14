import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List

from data.notion_utils import notion, clear_page_content
from data.data_manager import wardrobe_data_manager
from core.outfit_planner_agent import outfit_planner_agent


class NotionResultPublisher:
    """Handles the formatting and publishing of results to Notion."""

    def __init__(self, batch_size: int = 20, batch_delay: float = 0.2):
        self._batch_size = batch_size
        self._batch_delay = batch_delay

    async def finalize_packing_results(
        self, page_id: str, packing_result: Dict, generation_method: str, trip_config: Dict
    ) -> Dict:
        """
        Finalizes the packing results by updating Notion.
        """
        try:
            logging.info(f"🧳 Finalizing packing results using {generation_method}...")
            self._log_packing_summary(packing_result)

            await self._update_trip_worthy_selections(packing_result["selected_items"])
            await asyncio.to_thread(clear_page_content, page_id)
            await asyncio.to_thread(
                self._post_comprehensive_packing_guide,
                page_id,
                packing_result,
                trip_config,
                generation_method,
            )
            await self._generate_and_post_example_outfits(page_id, packing_result, trip_config)
            await asyncio.to_thread(self._clear_travel_trigger_fields_safe, page_id)

            logging.info("✅ Finalization completed successfully")
            return {"success": True}

        except Exception as e:
            logging.error(f"❌ Error in results finalization: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to finalize outfit: {str(e)}",
            }

    def _log_packing_summary(self, packing_result: Dict) -> None:
        """Log packing result summary."""
        logging.info("🧳 Packing optimization results:")
        logging.info(f"   Items selected: {packing_result.get('total_items', 'unknown')}")
        logging.info(f"   Total weight: {packing_result.get('total_weight_kg', 'unknown')}kg")

    async def _update_trip_worthy_selections(self, selected_items: List[Dict]) -> None:
        """Updates the 'Trip-worthy' status of items in Notion."""
        try:
            logging.info(f"🧳 Updating trip-worthy selections for {len(selected_items)} items...")
            selected_ids = set(item['id'] for item in selected_items)
            all_items = await asyncio.to_thread(wardrobe_data_manager.get_all_wardrobe_items)

            if not all_items:
                logging.warning("No items available for trip-worthy updates")
                return

            for i in range(0, len(all_items), self._batch_size):
                batch = all_items[i:i + self._batch_size]
                for item in batch:
                    item_id = item['id']
                    should_be_selected = item_id in selected_ids
                    try:
                        await asyncio.to_thread(
                            notion.pages.update,
                            page_id=item_id,
                            properties={"Trip-worthy": {"checkbox": should_be_selected}},
                        )
                    except Exception as e:
                        logging.warning(f"Failed to update item {item_id}: {e}")

                await asyncio.sleep(self._batch_delay)

            logging.info("✅ Trip-worthy update completed.")

        except Exception as e:
            logging.error(f"❌ Error in trip-worthy updates: {e}", exc_info=True)

    def _post_comprehensive_packing_guide(
        self, page_id: str, packing_result: Dict, trip_config: Dict, generation_method: str
    ) -> None:
        """Posts the comprehensive packing guide to Notion."""
        guide_blocks = self._create_guide_blocks(packing_result, trip_config, generation_method)
        logging.info(f"   Generated {len(guide_blocks)} content blocks")
        self._post_blocks_in_chunks(page_id, guide_blocks)
        logging.info("✅ Comprehensive packing guide posted successfully")

    def _create_guide_blocks(self, packing_result: Dict, trip_config: Dict, generation_method: str) -> List[Dict]:
        """Creates all blocks for the packing guide."""
        blocks = []
        blocks.extend(self._create_executive_summary_blocks(packing_result, trip_config))
        blocks.extend(self._create_selected_items_blocks(packing_result))
        blocks.extend(self._create_generation_info_blocks(generation_method))
        return blocks

    def _create_executive_summary_blocks(self, packing_result: Dict, trip_config: Dict) -> List[Dict]:
        """Create executive summary section."""
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
                    "rich_text": [{"type": "text", "text": {"content": f"Total weight: {packing_result['total_weight_kg']}kg"}}],
                    "icon": {"emoji": "✈️"}
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

        items_by_category = {}
        for item in packing_result["selected_items"]:
            category = item['category']
            if category not in items_by_category:
                items_by_category[category] = []
            items_by_category[category].append(item)

        for category, items in sorted(items_by_category.items()):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": f"{category} ({len(items)} items)"}}]
                }
            })
            for item in items:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": item['item']}}]
                    }
                })
        return blocks

    def _create_generation_info_blocks(self, generation_method: str) -> List[Dict]:
        """Create generation method information section."""
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
                    "rich_text": [{"type": "text", "text": {"content": f"🤖 Generated using {generation_method}"}}]
                }
            }
        ]

    def _post_blocks_in_chunks(self, page_id: str, blocks: List[Dict]) -> None:
        """Post blocks to Notion in optimal chunks to avoid API limits."""
        chunk_size = 100
        for i in range(0, len(blocks), chunk_size):
            chunk = blocks[i:i + chunk_size]
            try:
                notion.blocks.children.append(block_id=page_id, children=chunk)
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Failed to post chunk {i//chunk_size + 1}: {e}")
                raise

    async def _generate_and_post_example_outfits(self, page_id: str, packing_result: Dict, trip_config: Dict):
        """Generates and posts example outfits to the Notion page."""
        logging.info("👗 Generating example outfits from the selected wardrobe...")
        example_outfits_text = await outfit_planner_agent.generate_example_outfits(
            packing_result["selected_items"], trip_config
        )

        if example_outfits_text:
            outfit_blocks = self._create_example_outfits_blocks(example_outfits_text)
            await asyncio.to_thread(
                self._post_blocks_in_chunks, page_id, outfit_blocks
            )
            logging.info("✅ Example outfits posted to Notion.")
        else:
            logging.warning("⚠️ Could not generate example outfits.")

    def _create_example_outfits_blocks(self, outfit_text: str) -> List[Dict]:
        """Creates Notion blocks for the example outfits section."""
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": "💡 Example Outfit Ideas"}}]}
            }
        ]
        outfits = outfit_text.split("OUTFIT")[1:]
        for outfit in outfits:
            lines = outfit.strip().split('\n')
            if not lines:
                continue
            title = lines[0].strip().replace(":", "")
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": [{"type": "text", "text": {"content": title}}]}
            })
            for line in lines[1:]:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line.replace("*","").strip()}}]}
                })
        return blocks

    def _clear_travel_trigger_fields_safe(self, page_id: str) -> None:
        """Safely clear travel trigger fields."""
        try:
            logging.info(f"🧳 Safely clearing travel trigger fields for page {page_id}")
            page = notion.pages.retrieve(page_id=page_id)
            properties = page.get("properties", {})
            update_properties = {}
            trigger_fields = {
                "Generate": {"checkbox": False},
                "Generate Travel Packing": {"checkbox": False},
            }
            for field_name, field_value in trigger_fields.items():
                if field_name in properties:
                    update_properties[field_name] = field_value
            if update_properties:
                notion.pages.update(page_id=page_id, properties=update_properties)
                logging.info(f"✅ Cleared {len(update_properties)} trigger fields")
        except Exception as e:
            logging.warning(f"⚠️  Could not clear trigger fields (non-critical): {e}")
