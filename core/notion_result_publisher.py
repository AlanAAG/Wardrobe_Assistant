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

            # Immediately clear trigger fields to prevent re-triggering
            await asyncio.to_thread(self._clear_travel_trigger_fields_safe, page_id)

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
        blocks.extend(self._create_analysis_blocks(packing_result))
        blocks.extend(self._create_packing_guide_blocks(packing_result))
        blocks.extend(self._create_trip_tips_blocks(packing_result))
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

    def _validate_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Validate and sanitize blocks before sending to Notion API."""
        validated_blocks = []
        for i, block in enumerate(blocks):
            try:
                # Check for common formatting issues
                if block.get("type") == "paragraph":
                    paragraph = block.get("paragraph", {})
                    rich_text = paragraph.get("rich_text", [])
                    for rt_item in rich_text:
                        # Ensure annotations are at the top level, not nested in text
                        if "text" in rt_item and "annotations" in rt_item["text"]:
                            # Move annotations to top level
                            rt_item["annotations"] = rt_item["text"]["annotations"]
                            del rt_item["text"]["annotations"]
                
                validated_blocks.append(block)
            except Exception as e:
                logging.warning(f"⚠️  Block {i} validation issue: {e}, skipping block")
                continue
        
        return validated_blocks

    def _post_blocks_in_chunks(self, page_id: str, blocks: List[Dict]) -> None:
        """Post blocks to Notion in optimal chunks with retry logic for conflicts."""
        # Validate blocks first
        validated_blocks = self._validate_blocks(blocks)
        logging.info(f"📝 Validated {len(validated_blocks)} blocks (original: {len(blocks)})")
        
        chunk_size = 100
        max_retries = 3
        
        for i in range(0, len(validated_blocks), chunk_size):
            chunk = validated_blocks[i:i + chunk_size]
            chunk_num = i//chunk_size + 1
            
            for attempt in range(max_retries):
                try:
                    notion.blocks.children.append(block_id=page_id, children=chunk)
                    time.sleep(0.2)  # Slightly longer delay between chunks
                    logging.debug(f"✅ Posted chunk {chunk_num} successfully")
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if ("409" in error_str or "conflict" in error_str) and attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 1.0  # Exponential backoff: 1s, 2s, 3s
                        logging.warning(f"⚠️  Conflict on chunk {chunk_num}, retry {attempt + 1}/{max_retries} in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    elif "body failed validation" in error_str:
                        logging.error(f"❌ Block validation error in chunk {chunk_num}: {e}")
                        # Log the problematic chunk for debugging
                        logging.error(f"Problematic chunk content: {chunk}")
                        raise
                    else:
                        logging.error(f"❌ Failed to post chunk {chunk_num} after {attempt + 1} attempts: {e}")
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

    def _create_rich_text(self, content: str, bold: bool = False) -> List[Dict]:
        """Helper to create properly formatted rich text for Notion API."""
        rich_text_item = {
            "type": "text",
            "text": {"content": content}
        }
        if bold:
            rich_text_item["annotations"] = {"bold": True}
        return [rich_text_item]

    def _create_analysis_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create analysis section with business readiness, climate coverage, etc."""
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "📊 Travel Analysis"}}]
                }
            }
        ]
        
        # Business readiness
        business = packing_result.get("business_readiness", {})
        if business:
            blocks.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": f"Business Readiness Score: {business.get('readiness_score', 0):.1f}/1.0\nSuits: {business.get('suits_count', 0)} | Formal shoes: {business.get('dress_shoes_count', 0)} | Business shirts: {business.get('business_shirts_count', 0)}"}}],
                    "icon": {"emoji": "💼"}
                }
            })
        
        # Climate coverage
        climate = packing_result.get("climate_coverage", {})
        if climate:
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "Climate Coverage Analysis"}}]
                }
            })

            climate_adequacy = climate.get('coverage_adequacy', 'unknown')
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "text": {"content": "Overall Adequacy: "}, "annotations": {"bold": True}},
                        {"type": "text", "text": {"content": climate_adequacy.replace('_', ' ').title()}}
                    ]
                }
            })

            hot_items = climate.get('hot_weather_items', [])
            if hot_items:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"type": "text", "text": {"content": f"Hot Weather Items ({len(hot_items)}): "}, "annotations": {"bold": True}},
                            {"type": "text", "text": {"content": ", ".join(item['item'] for item in hot_items)}}
                        ]
                    }
                })

            cold_items = climate.get('cold_weather_items', [])
            if cold_items:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"type": "text", "text": {"content": f"Cold Weather Items ({len(cold_items)}): "}, "annotations": {"bold": True}},
                            {"type": "text", "text": {"content": ", ".join(item['item'] for item in cold_items)}}
                        ]
                    }
                })

            versatile_items = climate.get('versatile_items_list', [])
            if versatile_items:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {"type": "text", "text": {"content": f"Versatile Items ({len(versatile_items)}): "}, "annotations": {"bold": True}},
                            {"type": "text", "text": {"content": ", ".join(item['item'] for item in versatile_items)}}
                        ]
                    }
                })
        
        # Weight efficiency and bag allocation
        blocks.append({
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": f"Weight Efficiency: {packing_result.get('weight_efficiency', 0)} items/kg | Total Weight: {packing_result.get('total_weight_kg', 0)}kg"}}],
                "icon": {"emoji": "⚖️"}
            }
        })
        
        return blocks
    
    def _create_packing_guide_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create packing guide section."""
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "🎒 Packing Guide"}}]
                }
            }
        ]
        
        # Bag allocation
        bag_allocation = packing_result.get("bag_allocation", {})
        if bag_allocation:
            checked_bag = bag_allocation.get("checked_bag", {})
            cabin_bag = bag_allocation.get("cabin_bag", {})
            
            checked_items = checked_bag.get('items', [])
            cabin_items = cabin_bag.get('items', [])

            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "text": {"content": "Checked Bag: "}, "annotations": {"bold": True}},
                        {"type": "text", "text": {"content": f"{len(checked_items)} items, {checked_bag.get('weight_kg', 0):.1f}kg ({checked_bag.get('space_utilization', 0):.1f}% full)"}}
                    ]
                }
            })
            
            for item in checked_items:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": item['item']}}]
                    }
                })

            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {"type": "text", "text": {"content": "Cabin Bag: "}, "annotations": {"bold": True}},
                        {"type": "text", "text": {"content": f"{len(cabin_items)} items, {cabin_bag.get('weight_kg', 0):.1f}kg ({cabin_bag.get('space_utilization', 0):.1f}% full)"}}
                    ]
                }
            })

            for item in cabin_items:
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": item['item']}}]
                    }
                })
        
        # Packing guide content
        packing_guide = packing_result.get("packing_guide", {})
        if isinstance(packing_guide, dict):
            for guide_section, content in packing_guide.items():
                if content and isinstance(content, str):
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": f"{guide_section.replace('_', ' ').title()}: "}, "annotations": {"bold": True}},
                                {"type": "text", "text": {"content": content}}
                            ]
                        }
                    })
        
        return blocks
    
    def _create_trip_tips_blocks(self, packing_result: Dict) -> List[Dict]:
        """Create trip tips section."""
        blocks = [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": "💡 Trip Tips"}}]
                }
            }
        ]

        trip_tips = packing_result.get("trip_tips", {})
        if not isinstance(trip_tips, dict):
            if isinstance(trip_tips, str) and trip_tips:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": trip_tips}}]}
                })
            return blocks

        for destination, tips_by_category in trip_tips.items():
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": self._create_rich_text(destination, bold=True)}
            })

            if not isinstance(tips_by_category, dict):
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": str(tips_by_category)}}]}
                })
                continue

            for category, tips in tips_by_category.items():
                if not tips:
                    continue

                # Add category title
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": self._create_rich_text(f"{category.replace('_', ' ').title()}:", bold=True)}
                })

                # Add tips as a bulleted list
                if isinstance(tips, list):
                    for tip in tips:
                        if tip:
                            blocks.append({
                                "object": "block",
                                "type": "bulleted_list_item",
                                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": str(tip)}}]}
                            })
                else: # Fallback for non-list tips
                     blocks.append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": str(tips)}}]}
                    })

        return blocks
