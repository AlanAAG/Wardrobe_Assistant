import os
import logging
from supabase import create_client, Client
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class SupabaseClient:
    """
    Supabase client wrapper for wardrobe data operations.
    Handles connection, queries, and data formatting for LLM consumption.
    """

    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_PROJECT_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        self.client: Optional[Client] = None

        if self.supabase_url and self.supabase_key:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                logging.info("✅ Supabase client initialized successfully")
            except Exception as e:
                logging.error(f"❌ Failed to initialize Supabase client: {e}")
                self.client = None
        else:
            logging.warning("⚠️ Supabase credentials not found in environment")

    def is_connected(self) -> bool:
        """Check if Supabase client is properly initialized"""
        return self.client is not None

    def get_all_wardrobe_items(self) -> List[Dict]:
        """Fetch all wardrobe items"""
        if not self.client:
            return []
        try:
            response = self.client.table("wardrobe_items").select("*").execute()
            items = [record.get("item_data", {}) for record in response.data if record.get("item_data")]
            logging.info(f"Retrieved {len(items)} items from Supabase")
            return items
        except Exception as e:
            logging.error(f"Failed to fetch wardrobe items: {e}")
            return []

    def get_filtered_wardrobe_items(
        self,
        aesthetic: Optional[str] = None,
        weather_tag: Optional[str] = None,
        color_tag: Optional[str] = None,
        categories: Optional[List[str]] = None,
        washed_only: bool = True,
    ) -> List[Dict]:
        """
        Filter wardrobe items for LLM context.

        - aesthetic: multi-select
        - color_tag: multi-select
        - category: single-select
        - washed: single-select
        """
        if not self.client:
            return []

        try:
            query = self.client.table("wardrobe_items").select("item_data")

            # Washed (single select)
            if washed_only:
                query = query.eq("item_data->>washed", "Done")

            # Aesthetic (multi-select array contains)
            if aesthetic:
                query = query.filter("item_data->aesthetic", "cs", f'["{aesthetic}"]')

            # Weather (multi-select array contains)
            if weather_tag:
                query = query.filter("item_data->weather", "cs", f'["{weather_tag}"]')

            # Category (single select string)
            if categories:
                if len(categories) == 1:
                    query = query.eq("item_data->>category", categories[0])
                else:
                    query = query.in_("item_data->>category", categories)

            # Color (multi-select array contains)
            if color_tag:
                query = query.filter("item_data->color", "cs", f'["{color_tag}"]')

            response = query.limit(50).execute()
            items = [record["item_data"] for record in response.data if record.get("item_data")]

            logging.info(f"Retrieved {len(items)} filtered items from Supabase")
            return items

        except Exception as e:
            logging.error(f"Filtered query failed ({e}) → using client-side filtering")
            return self._fallback_client_side_filtering(aesthetic, weather_tag, color_tag, categories, washed_only)

    def _fallback_client_side_filtering(
        self,
        aesthetic: Optional[str],
        weather_tag: Optional[str],
        color_tag: Optional[str],
        categories: Optional[List[str]],
        washed_only: bool,
    ) -> List[Dict]:
        """Filter in Python if Supabase query fails"""
        try:
            all_items = self.get_all_wardrobe_items()
            filtered = []

            for item in all_items:
                # Washed (single select)
                if washed_only and item.get("washed", "").lower() != "done":
                    continue

                # Aesthetic (multi-select)
                if aesthetic:
                    aesthetics = [a.lower() for a in item.get("aesthetic", [])]
                    if aesthetic.lower() not in aesthetics:
                        continue

                # Weather (multi-select + versatile logic)
                if weather_tag:
                    weathers = [w.lower() for w in item.get("weather", [])]
                    if not (
                        weather_tag.lower() in weathers
                        or ("hot" in weathers and "cold" in weathers)
                        or len(weathers) == 0
                    ):
                        continue

                # Category (single select string)
                if categories and item.get("category") not in categories:
                    continue

                # Color (multi-select)
                if color_tag:
                    colors = [c.lower() for c in item.get("color", [])]
                    if color_tag.lower() not in colors:
                        continue

                filtered.append(item)

            logging.info(f"Client-side filtering returned {len(filtered)} items")
            return filtered[:50]

        except Exception as e:
            logging.error(f"Client-side filtering failed: {e}")
            return []


# Global instance
supabase_client = SupabaseClient()
