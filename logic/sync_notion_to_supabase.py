#!/usr/bin/env python3
"""
Sync script to transfer wardrobe data from Notion to Supabase.
Can be run manually or scheduled as a cron job.
"""

import logging
import sys
from dotenv import load_dotenv
from logic.data_manager import wardrobe_data_manager
from logic.supabase_client import supabase_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main sync function"""
    print("🔄 Starting Notion to Supabase sync...")
    
    # Load environment variables
    load_dotenv()
    
    # Check Supabase connection
    if not supabase_client.is_connected():
        print("❌ Supabase client not connected. Check your environment variables:")
        print("   - SUPABASE_PROJECT_URL")
        print("   - SUPABASE_ANON_KEY")
        sys.exit(1)
    
    print("✅ Supabase connection verified")
    
    # Perform sync
    try:
        print("📥 Fetching data from Notion...")
        success = wardrobe_data_manager.sync_notion_to_supabase()
        
        if success:
            print("✅ Sync completed successfully!")
            
            # Show stats
            stats = supabase_client.get_wardrobe_stats()
            print(f"📊 Supabase now contains:")
            print(f"   - Total items: {stats.get('total_items', 0)}")
            print(f"   - Washed items: {stats.get('washed_items', 0)}")
            print(f"   - Unwashed items: {stats.get('unwashed_items', 0)}")
            
        else:
            print("❌ Sync failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Sync error: {e}")
        logging.error(f"Sync failed with exception: {e}")
        sys.exit(1)

def test_data_hierarchy():
    """Test the data hierarchy fallback system"""
    print("\n🧪 Testing data hierarchy...")
    
    try:
        # Test getting all items
        print("Testing get_all_wardrobe_items()...")
        all_items = wardrobe_data_manager.get_all_wardrobe_items()
        print(f"✅ Retrieved {len(all_items)} total items")
        
        # Test filtered query
        print("Testing filtered query...")
        filtered_items = wardrobe_data_manager.get_filtered_wardrobe_items(
            aesthetic="Casual",
            weather_tag="Hot",
            washed_only=True
        )
        print(f"✅ Retrieved {len(filtered_items)} filtered items")
        
        # Test LLM context
        print("Testing LLM context preparation...")
        llm_context = wardrobe_data_manager.get_llm_optimized_context("Casual", "Hot")
        total_items = sum(len(items) for items in llm_context["available_items"].values())
        print(f"✅ Prepared LLM context with {total_items} items")
        
        # Show data source stats
        print("\n📊 Data source availability:")
        stats = wardrobe_data_manager.get_data_stats()
        print(f"   - Supabase connected: {stats['supabase_connected']}")
        print(f"   - Cache available: {stats['cache_available']}")
        print(f"   - Notion configured: {stats['notion_configured']}")
        
        if stats.get('supabase_stats'):
            sb_stats = stats['supabase_stats']
            print(f"   - Supabase items: {sb_stats.get('total_items', 0)}")
        
        if stats.get('cache_items'):
            print(f"   - Cache items: {stats['cache_items']}")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logging.error(f"Test failed with exception: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_data_hierarchy()
    else:
        main()