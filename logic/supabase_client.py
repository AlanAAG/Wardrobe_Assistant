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
        self.client = None
        
        if self.supabase_url and self.supabase_key:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                logging.info("Supabase client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Supabase client: {e}")
                self.client = None
        else:
            logging.warning("Supabase credentials not found in environment")
    
    def is_connected(self) -> bool:
        """Check if Supabase client is properly initialized"""
        return self.client is not None
    
    def get_all_wardrobe_items(self) -> List[Dict]:
        """
        Get all wardrobe items from Supabase.
        Returns list of wardrobe items or empty list if failed.
        """
        if not self.client:
            return []
        
        try:
            response = self.client.table('wardrobe_items').select('*').execute()
            
            # Extract item_data from each record
            items = []
            for record in response.data:
                item_data = record.get('item_data', {})
                if item_data:
                    items.append(item_data)
            
            logging.info(f"Retrieved {len(items)} items from Supabase")
            return items
            
        except Exception as e:
            logging.error(f"Failed to fetch wardrobe items from Supabase: {e}")
            return []
    
    def get_filtered_wardrobe_items(self, 
                                   aesthetic: str = None, 
                                   weather_tag: str = None,
                                   categories: List[str] = None,
                                   washed_only: bool = True) -> List[Dict]:
        """
        Get filtered wardrobe items optimized for LLM context.
        Uses PostgreSQL JSONB querying for efficient filtering.
        
        Args:
            aesthetic: Filter by aesthetic (e.g., "Minimalist", "Casual")
            weather_tag: Filter by weather ("Hot", "Cold")
            categories: List of categories to include
            washed_only: Only include washed items
            
        Returns:
            List of filtered wardrobe items
        """
        if not self.client:
            return []
        
        try:
            # Start with base query
            query = self.client.table('wardrobe_items').select('item_data')
            
            # Apply washed filter
            if washed_only:
                query = query.eq('item_data->>washed', 'Done')
            
            # Apply aesthetic filter
            if aesthetic:
                # Use @> operator to check if JSONB array contains the value
                query = query.filter('item_data->aesthetic', 'cs', f'["{aesthetic}"]')
            
            # Apply weather filter - this is the tricky part
            if weather_tag:
                # Option 1: Simple containment check for the specific weather tag
                query = query.filter('item_data->weather', 'cs', f'["{weather_tag}"]')
                
                # Note: For "versatile" items with both Hot and Cold, we'd need a separate query
                # or we can use a more complex filter, but let's start simple
            
            # Apply category filter
            if categories:
                # Use 'in' operator for multiple categories
                if len(categories) == 1:
                    query = query.eq('item_data->>category', categories[0])
                else:
                    # For multiple categories, we need to use 'in' operator
                    query = query.in_('item_data->>category', categories)
            
            # Execute query with reasonable limit for LLM context
            response = query.limit(50).execute()
            
            # Extract and return item_data
            items = [record['item_data'] for record in response.data if record.get('item_data')]
            
            logging.info(f"Retrieved {len(items)} filtered items from Supabase")
            return items
            
        except Exception as e:
            logging.error(f"Failed to fetch filtered items from Supabase: {e}")
            return []
    
    def get_filtered_wardrobe_items_advanced(self, 
                                           aesthetic: str = None, 
                                           weather_tag: str = None,
                                           categories: List[str] = None,
                                           washed_only: bool = True) -> List[Dict]:
        """
        Advanced filtering with better weather logic.
        Handles "versatile" items that work in both hot and cold weather.
        """
        if not self.client:
            return []
        
        try:
            # Build the base query
            conditions = []
            
            # Washed condition
            if washed_only:
                conditions.append("item_data->>'washed' = 'Done'")
            
            # Aesthetic condition
            if aesthetic:
                conditions.append(f"item_data->'aesthetic' @> '[\"{aesthetic}\"]'")
            
            # Weather condition - more sophisticated
            if weather_tag:
                weather_conditions = [
                    f"item_data->'weather' @> '[\"{weather_tag}\"]'",  # Specific weather
                    f"item_data->'weather' @> '[\"Hot\", \"Cold\"]'",  # Versatile items
                    f"jsonb_array_length(item_data->'weather') = 0"     # No weather tags (assume versatile)
                ]
                conditions.append(f"({' OR '.join(weather_conditions)})")
            
            # Category condition
            if categories:
                if len(categories) == 1:
                    conditions.append(f"item_data->>'category' = '{categories[0]}'")
                else:
                    category_list = "', '".join(categories)
                    conditions.append(f"item_data->>'category' IN ('{category_list}')")
            
            # Combine all conditions
            where_clause = " AND ".join(conditions)
            
            # Use RPC call for complex query
            if where_clause:
                # For now, let's use a simpler approach and filter client-side
                # This is less efficient but more reliable
                return self._fallback_client_side_filtering(aesthetic, weather_tag, categories, washed_only)
            else:
                response = self.client.table('wardrobe_items').select('item_data').limit(50).execute()
                items = [record['item_data'] for record in response.data if record.get('item_data')]
                return items
                
        except Exception as e:
            logging.error(f"Advanced filtering failed: {e}")
            return self._fallback_client_side_filtering(aesthetic, weather_tag, categories, washed_only)
    
    def _fallback_client_side_filtering(self, aesthetic: str, weather_tag: str, 
                                       categories: List[str], washed_only: bool) -> List[Dict]:
        """
        Fallback to client-side filtering when Supabase queries are too complex.
        """
        try:
            # Get all items and filter client-side
            all_items = self.get_all_wardrobe_items()
            filtered = []
            
            for item in all_items:
                # Washed filter
                if washed_only and item.get('washed', '').lower() != 'done':
                    continue
                
                # Aesthetic filter
                if aesthetic:
                    item_aesthetics = [a.lower() for a in item.get('aesthetic', [])]
                    if aesthetic.lower() not in item_aesthetics:
                        continue
                
                # Weather filter with versatile item logic
                if weather_tag:
                    item_weather = [w.lower() for w in item.get('weather', [])]
                    weather_match = (
                        weather_tag.lower() in item_weather or
                        ('hot' in item_weather and 'cold' in item_weather) or
                        len(item_weather) == 0  # No weather tags = versatile
                    )
                    if not weather_match:
                        continue
                
                # Category filter
                if categories and item.get('category') not in categories:
                    continue
                
                filtered.append(item)
            
            logging.info(f"Client-side filtering returned {len(filtered)} items")
            return filtered[:50]  # Limit for performance
            
        except Exception as e:
            logging.error(f"Client-side filtering failed: {e}")
            return []
    
    def sync_wardrobe_items(self, wardrobe_items: List[Dict]) -> bool:
        """
        Sync wardrobe items to Supabase from Notion.
        
        Args:
            wardrobe_items: List of wardrobe item dictionaries
            
        Returns:
            True if sync successful, False otherwise
        """
        if not self.client:
            logging.error("Cannot sync - Supabase client not initialized")
            return False
        
        try:
            # Clear existing data
            self.client.table('wardrobe_items').delete().neq('id', -1).execute()
            logging.info("Cleared existing wardrobe data from Supabase")
            
            # Prepare data for insertion
            records_to_insert = []
            for item in wardrobe_items:
                record = {
                    'item_data': item  # Store entire item as JSONB
                }
                records_to_insert.append(record)
            
            # Insert in batches to avoid size limits
            batch_size = 100
            total_inserted = 0
            
            for i in range(0, len(records_to_insert), batch_size):
                batch = records_to_insert[i:i + batch_size]
                response = self.client.table('wardrobe_items').insert(batch).execute()
                total_inserted += len(batch)
                logging.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} items")
            
            logging.info(f"Successfully synced {total_inserted} wardrobe items to Supabase")
            return True
            
        except Exception as e:
            logging.error(f"Failed to sync wardrobe items to Supabase: {e}")
            return False
    
    def get_wardrobe_stats(self) -> Dict:
        """
        Get basic statistics about wardrobe data for monitoring.
        """
        if not self.client:
            return {}
        
        try:
            # Get total count
            total_response = self.client.table('wardrobe_items').select('id', count='exact').execute()
            total_count = total_response.count
            
            # Get washed count using client-side filtering for now
            all_items = self.get_all_wardrobe_items()
            washed_count = sum(1 for item in all_items if item.get('washed', '').lower() == 'done')
            
            return {
                'total_items': total_count,
                'washed_items': washed_count,
                'unwashed_items': total_count - washed_count
            }
            
        except Exception as e:
            logging.error(f"Failed to get wardrobe stats: {e}")
            return {}

# Create global instance
supabase_client = SupabaseClient()