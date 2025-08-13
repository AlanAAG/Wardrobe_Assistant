import json
import logging
import os
from dotenv import load_dotenv
from data.notion_utils import get_wardrobe_items, get_outfit_db_pages

load_dotenv()

WARDROBE_DB_ID = os.getenv("NOTION_WARDROBE_DB_ID")
CACHE_FILE = "wardrobe_cache.json"

def get_latest_edit_time(pages):
    if not pages:
        return None
    return max(page.get("last_edited_time", "") for page in pages)

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return None, None
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        wardrobe = data.get("wardrobe")
        last_edit = data.get("last_edit")
        return wardrobe, last_edit

def save_cache(wardrobe, last_edit):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"wardrobe": wardrobe, "last_edit": last_edit}, f, indent=2)

def main():
    print("Checking for wardrobe updates in Notion...")
    pages = get_outfit_db_pages(WARDROBE_DB_ID)  # Or another method to get all wardrobe pages with last_edited_time

    latest_edit = get_latest_edit_time(pages)

    _, cached_edit = load_cache()

    if latest_edit and cached_edit and latest_edit <= cached_edit:
        print("No changes detected since last cache update. Skipping fetch.")
        return

    print("Fetching wardrobe items from Notion...")
    wardrobe = get_wardrobe_items(WARDROBE_DB_ID)
    save_cache(wardrobe, latest_edit)
    print(f"Cache updated with {len(wardrobe)} items.")

if __name__ == "__main__":
    main()
