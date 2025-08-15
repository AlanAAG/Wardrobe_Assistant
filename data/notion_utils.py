import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ── LAZY NOTION CLIENT ──────────────────────────────────────────────
# We do NOT create a Notion client at import time anymore.
# This prevents the entire app from crashing if NOTION_TOKEN is missing
# or Notion is temporarily unreachable when modules import this file.

NOTION_TOKEN_ENV_NAME = "NOTION_TOKEN"

try:
    from notion_client import Client  # keep import at module scope for type hints
except Exception:
    Client = None  # handled at runtime

class _LazyNotion:
    def __init__(self):
        self._client = None

    def _ensure(self):
        if self._client is None:
            token = os.getenv(NOTION_TOKEN_ENV_NAME, "").strip()
            if not token:
                raise EnvironmentError(
                    f"{NOTION_TOKEN_ENV_NAME} not set. "
                    "Travel/Outfit features that need Notion will fail until you configure it."
                )
            if Client is None:
                raise RuntimeError("notion_client package is not installed.")
            self._client = Client(auth=token)

    def __getattr__(self, name):
        # When first attribute is accessed, instantiate the real client
        self._ensure()
        return getattr(self._client, name)

# Exported symbol used across the app:
notion = _LazyNotion()

# Add the new environment variable for the output database ID
OUTPUT_DB_ID = os.getenv("NOTION_OUTFIT_LOG_DB_ID")
# ────────────────────────────────────────────────────────────────────
def query_database(database_id):
    """
    Query entire database, handling pagination.
    Returns a list of all results.
    """
    all_results = []
    next_cursor = None

    while True:
        try:
            response = notion.databases.query(
                database_id=database_id,
                start_cursor=next_cursor
            )
        except Exception as e:
            logging.error(f"Error querying database {database_id}: {e}")
            break

        all_results.extend(response.get("results", []))
        if not response.get("has_more"):
            break
        next_cursor = response.get("next_cursor")

    logging.info(f"Queried {len(all_results)} items from database {database_id}")
    return all_results

def get_wardrobe_items(wardrobe_db_id):
    """
    Fetch all wardrobe items from Notion database.
    Returns a list of dicts with all relevant properties and image URLs.
    """
    results = query_database(wardrobe_db_id)
    items = []

    for result in results:
        props = result.get("properties", {})
        page_id = result.get("id")

        # The 'Item' or 'Name' field is a special 'title' type
        item_rich_text = props.get("Item", {}).get("title", [])
        item_name = "".join([t.get("plain_text", "") for t in item_rich_text]) if item_rich_text else "Unnamed"

        # Reading "Category" as a single-select property, not multi-select.
        category = props.get("Category", {}).get("select", {}).get("name")

        # The rest of the properties are read as multi-select or select as appropriate
        print_select = props.get("Print", {}).get("select", {})
        print_str = print_select.get("name") if print_select else "No"

        brand_select = props.get("Brand", {}).get("select", {})
        brand_name = brand_select.get("name") if brand_select else None

        washed_select = props.get("Washed", {}).get("select", {})
        washed_value = washed_select.get("name") if washed_select else "Not Done"

        color_multi = props.get("Color", {}).get("multi_select", [])
        color_names = [c.get("name") for c in color_multi]

        aesthetic_multi = props.get("Aesthetic", {}).get("multi_select", [])
        aesthetic_names = [a.get("name") for a in aesthetic_multi]

        weather_multi = props.get("Weather", {}).get("multi_select", [])
        weather_tags = [w.get("name") for w in weather_multi]

        condition_select = props.get("Condition", {}).get("select", {})
        condition_value = condition_select.get("name") if condition_select else None

        # Trip-worthy checkbox
        trip_worthy_checkbox = props.get("Trip-worthy", {}).get("checkbox", False)

        # Image fetching logic remains the same
        image_url = get_page_cover_image_url(page_id)
        if not image_url:
            blocks = retrieve_page_blocks(page_id)
            image_url = extract_image_url_from_blocks(blocks)

        items.append({
            "id": page_id,
            "category": category,
            "item": item_name,
            "color": color_names,
            "aesthetic": aesthetic_names,
            "print": print_str,
            "weather": weather_tags,
            "brand": brand_name,
            "washed": washed_value,
            "condition": condition_value,
            "trip_worthy": trip_worthy_checkbox,
            "image_url": image_url,
        })

    logging.info(f"Fetched {len(items)} wardrobe items successfully.")
    return items

def get_latest_wardrobe_pages_with_edit_times(wardrobe_db_id):
    """
    Fetches all wardrobe pages and returns a list of dicts with
    page 'id' and 'last_edited_time'.
    This helps to check if the wardrobe cache is outdated.
    """
    pages = query_database(wardrobe_db_id)
    simplified = []
    for page in pages:
        simplified.append({
            "id": page.get("id"),
            "last_edited_time": page.get("last_edited_time"),
        })
    return simplified

# Remaining existing functions unchanged below...

def get_page_cover_image_url(page_id):
    """
    Get image URL from page cover (external or file).
    Returns URL string or None if not found.
    """
    try:
        page = notion.pages.retrieve(page_id=page_id)
        cover = page.get("cover")
        if cover:
            cover_type = cover.get("type")
            if cover_type == "external":
                return cover["external"].get("url")
            elif cover_type == "file":
                return cover["file"].get("url")
    except Exception as e:
        logging.error(f"Error fetching cover for page {page_id}: {e}")
    return None

def retrieve_page_blocks(page_id):
    """
    Retrieve children blocks of a page with pagination.
    Returns list of blocks or empty list.
    """
    blocks = []
    next_cursor = None
    try:
        while True:
            response = notion.blocks.children.list(block_id=page_id, start_cursor=next_cursor)
            blocks.extend(response.get("results", []))
            if not response.get("has_more"):
                break
            next_cursor = response.get("next_cursor")
    except Exception as e:
        logging.error(f"Error fetching blocks for page {page_id}: {e}")
    return blocks

def extract_image_url_from_blocks(blocks):
    """
    Extract first image URL found in the blocks.
    Returns URL string or None.
    """
    for block in blocks:
        if block.get("type") == "image":
            image = block.get("image", {})
            img_type = image.get("type")
            if img_type == "external":
                return image["external"].get("url")
            elif img_type == "file":
                return image["file"].get("url")
    return None

def post_outfit_to_notion_page(page_id, outfit_items):
    """
    Appends a to_do block for each outfit item to the given Notion page.
    outfit_items: list of outfit item dictionaries.
    """
    if not outfit_items:
        logging.warning("No outfit items to post.")
        return

    children_to_append = []
    for item in outfit_items:
        # Create a to_do block that contains a "mention" of the item's page
        block = {
            "object": "block",
            "type": "to_do",
            "to_do": {
                "rich_text": [
                    {
                        "type": "mention",
                        "mention": {
                            "type": "page",
                            "page": {"id": item["id"]}
                        }
                    }
                ],
                "checked": False
            }
        }
        children_to_append.append(block)

    # Add the final "Send to Hamper" to-do block
    children_to_append.append({
        "object": "block",
        "type": "to_do",
        "to_do": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {
                        "content": "Send to Hamper 🧺"
                    }
                }
            ],
            "checked": False
        }
    })

    try:
        notion.blocks.children.append(block_id=page_id, children=children_to_append)
        logging.info(f"Posted to-do list for {len(outfit_items)} items to Notion page {page_id}")
    except Exception as e:
        logging.error(f"Failed to post outfit to Notion page: {e}")

def get_selected_aesthetic_from_output_db(output_db_id):
    """
    Query the output database (should contain one page)
    and return the list of selected aesthetics (multi-select).
    Returns a list of strings or empty list.
    """
    try:
        results = query_database(output_db_id)
        if not results:
            logging.warning("Output database has no pages.")
            return []
        page = results[0]
        props = page.get("properties", {})
        aesthetic_prop = props.get("Desired Aesthetic", {})
        multi_select = aesthetic_prop.get("multi_select", [])
        selected = [tag.get("name") for tag in multi_select]
        return selected
    except Exception as e:
        logging.error(f"Failed to get selected aesthetic from output DB {output_db_id}: {e}")
        return []

def get_output_page_id(output_db_id):
    """
    Query output database for the single page ID inside.
    Returns the page ID string or None if not found.
    """
    try:
        results = query_database(output_db_id)
        if not results:
            logging.warning("Output database has no pages.")
            return None
        page = results[0]
        return page.get("id")
    except Exception as e:
        logging.error(f"Failed to get output page ID from database {output_db_id}: {e}")
        return None

def clear_page_content(page_id):
    """
    Retrieves and deletes all content blocks from a given Notion page.
    """
    try:
        # First, get a list of all blocks on the page
        all_blocks = retrieve_page_blocks(page_id)

        if not all_blocks:
            logging.info(f"Page {page_id} is already empty. No blocks to delete.")
            return

        logging.info(f"Found {len(all_blocks)} blocks to delete on page {page_id}.")

        # Loop through and delete each block by its ID
        for block in all_blocks:
            notion.blocks.delete(block_id=block["id"])

        logging.info(f"Successfully cleared all content from page {page_id}.")

    except Exception as e:
        logging.error(f"Failed to clear content from page {page_id}: {e}")

def get_outfit_db_pages(output_db_id):
    """
    Get all pages from the outfit database with their properties.
    Returns a list of page data including prompts and aesthetics.
    """
    try:
        results = query_database(output_db_id)
        pages = []

        for result in results:
            props = result.get("properties", {})

            # Get Desired Aesthetic
            aesthetic_prop = props.get("Desired Aesthetic", {})
            multi_select = aesthetic_prop.get("multi_select", [])
            aesthetics = [tag.get("name") for tag in multi_select]

            # Get Prompt text
            prompt_prop = props.get("Prompt", {})
            rich_text = prompt_prop.get("rich_text", [])
            prompt_text = "".join([t.get("plain_text", "") for t in rich_text]) if rich_text else ""

            # Get Outfit Date
            date_prop = props.get("Outfit Date", {})
            outfit_date = date_prop.get("date", {}).get("start") if date_prop.get("date") else None

            pages.append({
                "id": result.get("id"),
                "aesthetics": aesthetics,
                "prompt": prompt_text,
                "date": outfit_date,
                "last_edited_time": result.get("last_edited_time")
            })

        return pages
    except Exception as e:
        logging.error(f"Failed to get outfit DB pages: {e}")
        return []
    
def get_outfit_db_pages(output_db_id):
    """
    Get all pages from the outfit database with their properties.
    Returns a list of page data including prompts and aesthetics.
    """
    try:
        results = query_database(output_db_id)
        pages = []

        for result in results:
            props = result.get("properties", {})

            # Get Desired Aesthetic
            aesthetic_prop = props.get("Desired Aesthetic", {})
            multi_select = aesthetic_prop.get("multi_select", [])
            aesthetics = [tag.get("name") for tag in multi_select]

            # Get Prompt text
            prompt_prop = props.get("Prompt", {})
            rich_text = prompt_prop.get("rich_text", [])
            prompt_text = "".join([t.get("plain_text", "") for t in rich_text]) if rich_text else ""

            # Get Outfit Date
            date_prop = props.get("Outfit Date", {})
            outfit_date = date_prop.get("date", {}).get("start") if date_prop.get("date") else None

            pages.append({
                "id": result.get("id"),
                "aesthetics": aesthetics,
                "prompt": prompt_text,
                "date": outfit_date,
                "last_edited_time": result.get("last_edited_time")
            })

        return pages
    except Exception as e:
        logging.error(f"Failed to get outfit DB pages: {e}")
        return []

def clear_trigger_fields(page_id):
    """
    Clears the 'Desired Aesthetic' and 'Prompt' fields in a Notion page
    to reset the trigger conditions for the next outfit request.
    """
    try:
        properties = {
            "Desired Aesthetic": {
                "multi_select": []
            },
            "Prompt": {
                "rich_text": []
            }
        }
        
        notion.pages.update(page_id=page_id, properties=properties)
        logging.info(f"Successfully cleared trigger fields for page {page_id}")
        
    except Exception as e:
        logging.error(f"Failed to clear trigger fields for page {page_id}: {e}")
        raise

def update_items_washed_status(page_id: str, status: str = "Done"):
    """
    Updates the 'Washed' status of a given clothing item page.
    """
    try:
        properties = {
            "Washed": {
                "select": {
                    "name": status
                }
            }
        }
        notion.pages.update(page_id=page_id, properties=properties)
        logging.info(f"Updated 'Washed' status to '{status}' for page {page_id}")
    except Exception as e:
        logging.error(f"Failed to update 'Washed' status for page {page_id}: {e}")
        raise


def archive_page(page_id: str):
    """
    Archives a given Notion page.
    """
    try:
        notion.pages.update(page_id=page_id, archived=True)
        logging.info(f"Successfully archived page {page_id}")
    except Exception as e:
        logging.error(f"Failed to archive page {page_id}: {e}")
        raise

def create_page_in_dirty_clothes_db(item_name: str, clothing_item_id: str, outfit_log_id: str):
    """
    Creates a new page in the 'Dirty Clothes' database.
    """
    dirty_clothes_db_id = os.getenv("NOTION_DIRTY_CLOTHES_DB_ID")
    if not dirty_clothes_db_id:
        logging.error("NOTION_DIRTY_CLOTHES_DB_ID not set.")
        return

    try:
        properties = {
            "Item Name": {
                "title": [
                    {
                        "text": {
                            "content": item_name
                        }
                    }
                ]
            },
            "Ready for Laundry": {
                "checkbox": True
            },
            "Clothing Item": {
                "relation": [
                    {
                        "id": clothing_item_id
                    }
                ]
            },
            "Outfit Log": {
                "relation": [
                    {
                        "id": outfit_log_id
                    }
                ]
            }
        }
        notion.pages.create(parent={"database_id": dirty_clothes_db_id}, properties=properties)
        logging.info(f"Added '{item_name}' to Dirty Clothes database.")
    except Exception as e:
        logging.error(f"Failed to create page in Dirty Clothes database: {e}")
        raise

def get_checked_items_from_page(page_id: str) -> list:
    """
    Retrieves all checked 'to_do' blocks from a page that mention a clothing item,
    ignoring the "Send to Hamper" block.
    Returns a list of dicts, each containing the clothing item's page ID and name.
    """
    try:
        blocks = retrieve_page_blocks(page_id)
        checked_items = []
        for block in blocks:
            if block.get("type") == "to_do" and block.get("to_do", {}).get("checked") is True:
                rich_text = block.get("to_do", {}).get("rich_text", [])
                # Ignore the "Send to Hamper" block
                if any("Send to Hamper" in t.get("text", {}).get("content", "") for t in rich_text):
                    continue
                for text_item in rich_text:
                    if text_item.get("type") == "mention" and text_item.get("mention", {}).get("type") == "page":
                        clothing_item_id = text_item["mention"]["page"]["id"]
                        item_name = text_item["plain_text"]
                        checked_items.append({"id": clothing_item_id, "name": item_name})
        logging.info(f"Found {len(checked_items)} checked items on page {page_id}")
        return checked_items
    except Exception as e:
        logging.error(f"Failed to get checked items from page {page_id}: {e}")
        return []

def add_items_to_dirty_clothes_db(items: list, outfit_log_id: str):
    """
    Adds a list of clothing items to the 'Dirty Clothes' database.
    items: A list of dicts, each with "id" and "name" of the item.
    """
    for item in items:
        try:
            create_page_in_dirty_clothes_db(
                item_name=item["name"],
                clothing_item_id=item["id"],
                outfit_log_id=outfit_log_id
            )
        except Exception as e:
            logging.error(f"Failed to add item {item['id']} to dirty clothes DB: {e}")

def uncheck_hamper_trigger(page_id: str):
    """
    Unchecks the 'Send to Hamper' checkbox on a given page.
    """
    try:
        properties = {
            "Send to Hamper": {
                "checkbox": False
            }
        }
        notion.pages.update(page_id=page_id, properties=properties)
        logging.info(f"Unchecked 'Send to Hamper' for page {page_id}")
    except Exception as e:
        logging.error(f"Failed to uncheck 'Send to Hamper' for page {page_id}: {e}")