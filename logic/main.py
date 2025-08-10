import json
import os
from dotenv import load_dotenv
from weather_utils import get_weather_forecast
from outfit_logic import build_outfit
from notion_utils import (
    get_selected_aesthetic_from_output_db,
    get_output_page_id,
    post_outfit_to_notion_page,
    clear_page_content,
)

load_dotenv()

OUTPUT_DB_ID = os.getenv("NOTION_OUTFIT_LOG_DB_ID")
CACHE_FILE = "wardrobe_cache.json"

def load_wardrobe_cache():
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError("Wardrobe cache not found. Please run sync_notion_to_cache.py first.")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("wardrobe", [])

def main():
    print("Loading wardrobe from cache...")
    wardrobe_items = load_wardrobe_cache()

    print("Fetching weather forecast...")
    forecast = get_weather_forecast()
    avg_temp = forecast["avg_temp"]
    condition = forecast["condition"]
    is_hot = forecast["weather_tag"] == "hot"
    print(f"Avg temp: {avg_temp}°C, Condition: {condition}, Hot: {is_hot}")

    print("Getting selected aesthetic from Notion...")
    selected_aesthetics = get_selected_aesthetic_from_output_db(OUTPUT_DB_ID)
    desired_aesthetic = selected_aesthetics[0] if selected_aesthetics else "Minimalist"
    print(f"Using aesthetic: {desired_aesthetic}")

    print("Building the outfit...")
    outfit = build_outfit(wardrobe_items, is_hot, desired_aesthetic, washed_required="Done")

    if not outfit:
        print("❌ Could not build an outfit with the available items.")
        return

    print("Getting output Notion page ID...")
    output_page_id = get_output_page_id(OUTPUT_DB_ID)
    if not output_page_id:
        print("❌ Could not find the output page in Notion.")
        return

    print("Clearing previous outfit from Notion page...")
    clear_page_content(output_page_id)

    print("Posting outfit to Notion...")
    post_outfit_to_notion_page(output_page_id, outfit)

    print("✅ Done! Your new outfit has been posted to Notion.")

if __name__ == "__main__":
    main()