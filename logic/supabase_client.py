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
            
            # Apply filters using JSONB operations
            if washed_only:
                query = query.eq('item_data->>washed', 'Done')
            
            if aesthetic:
                # Check if aesthetic array contains the desired aesthetic
                query = query.contains('item_data->aesthetic', [aesthetic])
            
            if weather_tag:
                # Check if weather array contains the weather tag OR both Hot and Cold
                weather_filter = f'item_data->weather.cs.{{{weather_tag}}} or item_data->weather.cs.{{Hot,Cold}}'
                query = query.or_(weather_filter)
            
            if categories:
                # Filter by category
                category_conditions = ' or '.join([f'item_data->>category.eq.{cat}' for cat in categories])
                query = query.or_(category_conditions)
            
            # Execute query with reasonable limit for LLM context
            response = query.limit(50).execute()
            
            # Extract and return item_data
            items = [record['item_data'] for record in response.data if record.get('item_data')]
            
            logging.info(f"Retrieved {len(items)} filtered items from Supabase")
            return items
            
        except Exception as e:
            logging.error(f"Failed to fetch filtered items from Supabase: {e}")
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
            
            # Get washed count
            washed_response = self.client.table('wardrobe_items').select('id', count='exact').eq('item_data->>washed', 'Done').execute()
            washed_count = washed_response.count
            
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