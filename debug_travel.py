#!/usr/bin/env python3
"""
Enhanced debug script for the AI-driven travel packing pipeline.
Tests the new dynamic workflow where raw user input, including luggage,
is processed directly by the AI.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Set up logging for the debug script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_environment():
    """Check all required environment variables."""
    print("🔧 Testing Environment Setup...")
    required_vars = [
        'NOTION_TOKEN', 'NOTION_PACKING_GUIDE_ID', 'NOTION_WARDROBE_DB_ID',
        'GEMINI_AI_API_KEY', 'GROQ_AI_API_KEY'
    ]
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"   ✅ {var}: {masked}")
        else:
            missing.append(var)
            print(f"   ❌ {var}: Not set")
    return not missing

def test_imports():
    """Test importing all required modules."""
    print("\n🔧 Testing Imports...")
    try:
        from data.notion_utils import notion
        print("   ✅ notion_utils import successful")
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        print("   ✅ travel_pipeline_orchestrator import successful")
        from core.travel_packing_agent import travel_packing_agent
        print("   ✅ travel_packing_agent import successful")
        from api.webhook_server import get_travel_trigger_data
        print("   ✅ webhook_server functions import successful")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

async def test_notion_connection():
    """Test Notion API connection."""
    print("\n🔧 Testing Notion Connection...")
    try:
        from data.notion_utils import notion
        packing_guide_id = os.getenv('NOTION_PACKING_GUIDE_ID')
        page = await asyncio.to_thread(notion.pages.retrieve, page_id=packing_guide_id)
        print("   ✅ Notion connection successful")
        title_property = page.get('properties', {}).get('AI Generated Luggage Explanation', {})
        page_title = title_property.get('title', [{}])[0].get('plain_text', 'N/A')
        print(f"   📋 Page Title: '{page_title}'")
        return True
    except Exception as e:
        print(f"   ❌ Notion connection failed: {e}")
        return False

async def test_data_extraction():
    """Tests the new data extraction logic in the webhook server."""
    print("\n🔧 Testing Raw Data Extraction...")
    try:
        from api.webhook_server import get_travel_trigger_data
        page_id = os.getenv('NOTION_PACKING_GUIDE_ID')
        
        trigger_data = await asyncio.to_thread(get_travel_trigger_data, page_id)
        
        if not trigger_data:
            print("   ❌ Failed to extract any trigger data from Notion.")
            return False
        
        print("   ✅ Successfully extracted trigger data from Notion:")
        print(f"      -> Destinations: {trigger_data.get('destinations')}")
        print(f"      -> Preferences: \"{trigger_data.get('preferences')}\"")
        print(f"      -> Bags: {trigger_data.get('bags')}")
        
        return True
    except Exception as e:
        print(f"   ❌ Data extraction test failed: {e}")
        return False

async def run_full_pipeline_test():
    """Tests the complete, AI-driven travel pipeline end-to-end."""
    print("\n🚀 **RUNNING FULL END-TO-END PIPELINE TEST** 🚀")
    try:
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        from api.webhook_server import get_travel_trigger_data

        page_id = os.getenv('NOTION_PACKING_GUIDE_ID')
        print(f"   📡 Fetching real-time trigger data from Notion page: {page_id}")
        trigger_data = await asyncio.to_thread(get_travel_trigger_data, page_id)
        
        if not trigger_data or not trigger_data.get("destinations"):
             print("   ❌ Could not fetch valid trigger data from Notion. Please check your page properties.")
             return False

        print("\n   🧠 Running complete AI pipeline (this may take a few minutes)...")
        result = await travel_pipeline_orchestrator.run_travel_packing_pipeline(trigger_data)
        
        if result and result.get('success'):
            print("\n   🎉 **Full pipeline test SUCCESSFUL!** 🎉")
            print(f"      AI Generation Method: {result.get('generation_method', 'unknown')}")
            print(f"      Items Selected: {result.get('total_items_selected', 'unknown')}")
            print(f"      Total Weight: {result.get('total_weight_kg', 'unknown')}kg")
            print("\n   ✅ Your Notion page should now be updated with the AI-generated packing list and example outfits.")
            return True
        else:
            error_message = result.get('error', 'Unknown error') if result else 'No result returned'
            print(f"\n   ❌ **Full pipeline test FAILED!**")
            print(f"      Error: {error_message}")
            return False
        
    except Exception as e:
        print(f"   💥 Full pipeline test crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main debug function."""
    print("="*50)
    print("🚀 AI Wardrobe Assistant - Debug & Validation Suite 🚀")
    print("="*50)
    
    results = {}
    
    results['environment'] = check_environment()
    results['imports'] = test_imports()
    results['notion_connection'] = await test_notion_connection()
    results['data_extraction'] = await test_data_extraction()
    
    all_basic_passed = all(results.values())
    
    if all_basic_passed:
        print("\n✅ All basic checks passed. Proceeding to full end-to-end test...")
        results['full_pipeline'] = await run_full_pipeline_test()
    else:
        print("\n⚠️  Skipping full pipeline test because basic checks failed.")
        results['full_pipeline'] = False
        
    print("\n" + "="*50)
    print("📊 **FINAL TEST SUMMARY** 📊")
    print("="*50)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status.ljust(8)} | {test_name.replace('_', ' ').title()}")
    
    if not all(results.values()):
        print("\n" + "="*50)
        print("⚠️  **Action Required**: Please fix the failing tests above before deploying.")
    else:
        print("\n" + "="*50)
        print("🎉 **Congratulations! All systems are nominal. Your application is ready.** 🎉")

if __name__ == "__main__":
    asyncio.run(main())