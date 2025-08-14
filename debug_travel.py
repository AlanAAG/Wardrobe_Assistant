#!/usr/bin/env python3
"""
Enhanced debug script for the AI-driven travel packing pipeline.
Tests the new dynamic workflow where raw user input is processed by the AI.
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
        print(f"   📋 Page Title: '{page.get('properties', {}).get('AI Generated Luggage Explanation', {}).get('title', [{}])[0].get('plain_text', 'N/A')}'")
        return True
    except Exception as e:
        print(f"   ❌ Notion connection failed: {e}")
        return False

async def test_workflow_detection():
    """Test workflow detection logic."""
    print("\n🔧 Testing Workflow Detection...")
    try:
        from api.webhook_server import determine_workflow_type
        packing_guide_id = os.getenv('NOTION_PACKING_GUIDE_ID')
        print(f"   🔍 Testing workflow for page: {packing_guide_id}")
        workflow = await asyncio.to_thread(determine_workflow_type, packing_guide_id)
        print(f"   📋 Detected workflow: {workflow}")
        if workflow == 'travel':
            print("   ✅ Travel workflow detected successfully")
            return True
        else:
            print(f"   ⚠️  Expected 'travel' workflow, got '{workflow}'")
            return False
    except Exception as e:
        print(f"   ❌ Workflow detection failed: {e}")
        return False

async def test_travel_pipeline_structure():
    """Tests the new structure where raw data is passed to the agent."""
    print("\n🔧 Testing Travel Pipeline Structure...")
    try:
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        
        # This now simulates the raw text extracted from Notion
        mock_trigger_data = {
            "destinations": "Dubai (Sept-Dec), Gurgaon (Jan-April)",
            "preferences": "Business school trip, focus on business casual and minimalist styles.",
            "dates": {"start": "2025-09-01", "end": "2026-04-30", "days": 242}
        }
        
        print("   📦 Testing with raw mock data...")
        # We test the orchestrator's ability to prepare this raw data for the AI
        trip_config = await travel_pipeline_orchestrator._prepare_trip_configuration_enhanced(mock_trigger_data)
        
        if trip_config and "raw_destinations_and_dates" in trip_config:
            print("   ✅ Orchestrator correctly prepared raw data for the AI.")
            return True
        else:
            print("   ❌ Orchestrator failed to prepare raw data.")
            return False
    except Exception as e:
        print(f"   ❌ Pipeline structure test failed: {e}")
        return False

async def run_full_pipeline_test():
    """Tests the complete, AI-driven travel pipeline end-to-end."""
    print("\n🚀 **RUNNING FULL END-TO-END PIPELINE TEST** 🚀")
    try:
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        from api.webhook_server import get_travel_trigger_data

        # Fetch the real, raw data from your Notion page
        page_id = os.getenv('NOTION_PACKING_GUIDE_ID')
        print(f"   📡 Fetching real trigger data from Notion page: {page_id}")
        trigger_data = await asyncio.to_thread(get_travel_trigger_data, page_id)
        
        if not trigger_data or not trigger_data.get("destinations"):
             print("   ❌ Could not fetch valid trigger data from Notion. Please check your page.")
             return False

        print("   ✅ Successfully fetched trigger data from Notion.")
        print(f"      -> Destinations: \"{trigger_data['destinations']}\"")
        print(f"      -> Preferences: \"{trigger_data['preferences']}\"")
        
        print("\n   🧠 Running complete AI pipeline (this may take a few minutes)...")
        result = await travel_pipeline_orchestrator.run_travel_packing_pipeline(trigger_data)
        
        if result and result.get('success'):
            print("\n   🎉 **Full pipeline test SUCCESSFUL!** 🎉")
            print(f"      AI Generation Method: {result.get('generation_method', 'unknown')}")
            print(f"      Items Selected: {result.get('total_items_selected', 'unknown')}")
            print(f"      Total Weight: {result.get('total_weight_kg', 'unknown')}kg")
            print("\n   ✅ Your Notion page should now be updated with the AI-generated packing list.")
            return True
        else:
            print(f"\n   ❌ **Full pipeline test FAILED!**")
            print(f"      Error: {result.get('error', 'Unknown error')}")
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
    results['notion'] = await test_notion_connection()
    results['workflow'] = await test_workflow_detection()
    results['pipeline_structure'] = await test_travel_pipeline_structure()
    
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
