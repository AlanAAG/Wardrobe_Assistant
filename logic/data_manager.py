import json
import os
import logging
from typing import List, Dict, Optional
from logic.supabase_client import supabase_client
from logic.notion_utils import get_wardrobe_items

class WardrobeDataManager:
    """
    Hierarchical data manager: Supabase -> Local Cache -> Notion -> Error
    Provides consistent interface for all data consumers (logic engine + LLM agents)
    """
    
    def __init__(self):
        self.cache_file = "wardrobe_cache.json"
        self.wardrobe_db_id = os.getenv("NOTION_WARDROBE_DB_ID")
    
    def get_all_wardrobe_items(self) -> List[Dict]:
        """
        Get all wardrobe items using hierarchical fallback.
        
        Returns:
            List of all wardrobe items
        """
        # 1st: Try Supabase
        try:
            if supabase_client.is_connected():
                items = supabase_client.get_all_wardrobe_items()
                if items:
                    logging.info(f"Retrieved {len(items)} items from Supabase")
                    return items
        except Exception as e:
            logging.warning(f"Supabase fetch failed: {e}")
        
        # 2nd: Try local cache
        try:
            items = self._load_from_cache()
            if items:
                logging.info(f"Retrieved {len(items)} items from local cache")
                return items
        except Exception as e:
            logging.warning(f"Cache load failed: {e}")
        
        # 3rd: Try fresh Notion sync
        try:
            items = self._fetch_from_notion()
            if items:
                logging.info(f"Retrieved {len(items)} items from fresh Notion sync")
                # Update cache for future use
                self._save_to_cache(items)
                return items
        except Exception as e:
            logging.error(f"Notion fetch failed: {e}")
        
        # 4th: Error state
        logging.error("All data sources failed")
        raise Exception("Unable to retrieve wardrobe data from any source")
    
    def get_filtered_wardrobe_items(self,
                                   aesthetic: str = None,
                                   weather_tag: str = None,
                                   categories: List[str] = None,
                                   washed_only: bool = True) -> List[Dict]:
        """
        Get filtered wardrobe items optimized for LLM context.
        Uses efficient Supabase filtering when available, falls back to local filtering.
        
        Args:
            aesthetic: Filter by aesthetic (e.g., "Minimalist", "Casual")
            weather_tag: Filter by weather ("Hot", "Cold")
            categories: List of categories to include
            washed_only: Only include washed items
            
        Returns:
            List of filtered wardrobe items
        """
        # 1st: Try Supabase with native filtering (most efficient)
        try:
            if supabase_client.is_connected():
                items = supabase_client.get_filtered_wardrobe_items(
                    aesthetic=aesthetic,
                    weather_tag=weather_tag,
                    categories=categories,
                    washed_only=washed_only
                )
                if items:
                    logging.info(f"Retrieved {len(items)} filtered items from Supabase")
                    return items
        except Exception as e:
            logging.warning(f"Supabase filtered query failed: {e}")
        
        # 2nd: Fallback to local filtering
        try:
            all_items = self.get_all_wardrobe_items()  # This will use the hierarchy
            filtered_items = self._apply_local_filters(
                all_items, aesthetic, weather_tag, categories, washed_only
            )
            logging.info(f"Applied local filtering: {len(filtered_items)} items match criteria")
            return filtered_items
        except Exception as e:
            logging.error(f"Local filtering failed: {e}")
            raise Exception("Unable to retrieve filtered wardrobe data")
    
    def get_llm_optimized_context(self, aesthetic: str, weather_tag: str) -> Dict:
        """
        Get wardrobe data organized by category for optimal LLM consumption.
        Minimizes tokens while providing complete context.
        
        Args:
            aesthetic: Desired aesthetic style
            weather_tag: Weather condition ("Hot" or "Cold")
            
        Returns:
            Dictionary organized by clothing categories
        """
        try:
            context = {
                "weather_condition": weather_tag,
                "desired_aesthetic": aesthetic,
                "available_items": {
                    "tops": self.get_filtered_wardrobe_items(
                        aesthetic=aesthetic,
                        weather_tag=weather_tag,
                        categories=["Polo", "T-shirt", "Sport T-shirt", "Shirt"]
                    ),
                    "bottoms": self.get_filtered_wardrobe_items(
                        aesthetic=aesthetic,
                        weather_tag=weather_tag,
                        categories=["Cargo Pants", "Chinos", "Jeans", "Joggers", "Pants", "Shorts"]
                    ),
                    "outerwear": self.get_filtered_wardrobe_items(
                        aesthetic=aesthetic,
                        weather_tag=weather_tag,
                        categories=["Crewneck", "Hoodie", "Fleece", "Jacket", "Overcoat", "Overshirt", "Suit"]
                    ),
                    "footwear": self.get_filtered_wardrobe_items(
                        aesthetic=aesthetic,
                        weather_tag=weather_tag,
                        categories=["Shoes", "Sneakers"]
                    )
                }
            }
            
            # Log context size for token optimization
            total_items = sum(len(items) for items in context["available_items"].values())
            logging.info(f"LLM context prepared: {total_items} items across categories")
            
            return context
            
        except Exception as e:
            logging.error(f"Failed to prepare LLM context: {e}")
            raise Exception("Unable to prepare LLM-optimized wardrobe context")
    
    def sync_notion_to_supabase(self) -> bool:
        """
        Sync fresh data from Notion to Supabase.
        
        Returns:
            True if sync successful, False otherwise
        """
        if not supabase_client.is_connected():
            logging.warning("Cannot sync to Supabase - client not connected")
            return False
        
        try:
            # Get fresh data from Notion
            fresh_items = self._fetch_from_notion()
            if not fresh_items:
                logging.warning("No items retrieved from Notion for sync")
                return False
            
            # Sync to Supabase
            success = supabase_client.sync_wardrobe_items(fresh_items)
            
            if success:
                # Update local cache as well
                self._save_to_cache(fresh_items)
                logging.info("Successfully synced Notion -> Supabase -> Cache")
            
            return success
            
        except Exception as e:
            logging.error(f"Sync failed: {e}")
            return False
    
    def _load_from_cache(self) -> List[Dict]:
        """Load wardrobe items from local JSON cache"""
        if not os.path.exists(self.cache_file):
            return []
        
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('wardrobe', [])
    
    def _save_to_cache(self, items: List[Dict]) -> None:
        """Save wardrobe items to local JSON cache"""
        cache_data = {
            'wardrobe': items,
            'last_updated': 'auto_update'
        }
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
    
    def _fetch_from_notion(self) -> List[Dict]:
        """Fetch fresh data from Notion"""
        if not self.wardrobe_db_id:
            raise Exception("NOTION_WARDROBE_DB_ID not set")
        
        return get_wardrobe_items(self.wardrobe_db_id)
    
    def _apply_local_filters(self, items: List[Dict], aesthetic: str, weather_tag: str, 
                           categories: List[str], washed_only: bool) -> List[Dict]:
        """Apply filtering logic locally (fallback when Supabase filtering fails)"""
        filtered = []
        
        for item in items:
            # Washed filter
            if washed_only and item.get('washed', '').lower() != 'done':
                continue
            
            # Aesthetic filter
            if aesthetic:
                item_aesthetics = [a.lower() for a in item.get('aesthetic', [])]
                if aesthetic.lower() not in item_aesthetics:
                    continue
            
            # Weather filter
            if weather_tag:
                item_weather = [w.lower() for w in item.get('weather', [])]
                weather_match = (
                    weather_tag.lower() in item_weather or
                    ('hot' in item_weather and 'cold' in item_weather) or
                    len(item_weather) == 0
                )
                if not weather_match:
                    continue
            
            # Category filter
            if categories and item.get('category') not in categories:
                continue
            
            filtered.append(item)
        
        return filtered[:50]  # Limit for performance
    
    def get_data_stats(self) -> Dict:
        """Get statistics about data availability across all sources"""
        stats = {
            'supabase_connected': supabase_client.is_connected(),
            'cache_available': os.path.exists(self.cache_file),
            'notion_configured': bool(self.wardrobe_db_id)
        }
        
        # Add Supabase stats if available
        if stats['supabase_connected']:
            stats['supabase_stats'] = supabase_client.get_wardrobe_stats()
        
        # Add cache stats if available
        if stats['cache_available']:
            try:
                cache_items = self._load_from_cache()
                stats['cache_items'] = len(cache_items)
            except Exception:
                stats['cache_items'] = 0
        
        return stats

# Create global instance
wardrobe_data_manager = WardrobeDataManager()