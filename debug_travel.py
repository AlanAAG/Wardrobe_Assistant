#!/usr/bin/env python3
"""
Debug script for travel pipeline
"""
import sys
import os
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_environment():
    """Test environment setup"""
    print("🔧 Testing Environment Setup...")
    
    required_vars = [
        'NOTION_TOKEN',
        'NOTION_PACKING_GUIDE_ID', 
        'NOTION_WARDROBE_DB_ID',
        'GEMINI_AI_API_KEY',
        'GROQ_AI_API_KEY'
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        status = "✅" if value else "❌"
        masked_value = f"{value[:8]}..." if value and len(value) > 8 else "NOT SET"
        print(f"   {status} {var}: {masked_value}")
    
    return all(os.getenv(var) for var in required_vars)

def test_imports():
    """Test all imports"""
    print("\n🔧 Testing Imports...")
    
    try:
        from data.notion_utils import notion
        print("   ✅ notion_utils import successful")
    except Exception as e:
        print(f"   ❌ notion_utils import failed: {e}")
        return False
    
    try:
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        print("   ✅ travel_pipeline_orchestrator import successful")
        print(f"   📋 Orchestrator methods: {[m for m in dir(travel_pipeline_orchestrator) if not m.startswith('_')]}")
    except Exception as e:
        print(f"   ❌ travel_pipeline_orchestrator import failed: {e}")
        import traceback
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        return False
    
    try:
        from core.travel_packing_agent import travel_packing_agent
        print("   ✅ travel_packing_agent import successful")
    except Exception as e:
        print(f"   ❌ travel_packing_agent import failed: {e}")
        return False
    
    return True

def test_notion_connection():
    """Test Notion connection"""
    print("\n🔧 Testing Notion Connection...")
    
    try:
        from data.notion_utils import notion
        
        # Test basic connection
        packing_guide_id = os.getenv("NOTION_PACKING_GUIDE_ID")
        if packing_guide_id:
            page = notion.pages.retrieve(page_id=packing_guide_id)
            print(f"   ✅ Notion connection successful")
            
            # Show page properties
            props = page.get('properties', {})
            print(f"   📋 Page properties:")
            for prop_name, prop_data in props.items():
                prop_type = prop_data.get('type', 'unknown')
                if prop_type == 'checkbox':
                    value = prop_data.get('checkbox', False)
                    print(f"      {prop_name} (checkbox): {value}")
                elif prop_type == 'rich_text':
                    text = "".join([t.get("plain_text", "") for t in prop_data.get("rich_text", [])])
                    print(f"      {prop_name} (rich_text): '{text[:50]}...'")
                else:
                    print(f"      {prop_name} ({prop_type})")
            
            return True
        else:
            print("   ❌ NOTION_PACKING_GUIDE_ID not set")
            return False
            
    except Exception as e:
        print(f"   ❌ Notion connection failed: {e}")
        import traceback
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        return False

def test_travel_pipeline():
    """Test travel pipeline with mock data"""
    print("\n🔧 Testing Travel Pipeline...")
    
    try:
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        
        if not travel_pipeline_orchestrator:
            print("   ❌ Travel orchestrator is None!")
            return False
        
        # Create mock trigger data
        mock_trigger_data = {
            "page_id": os.getenv("NOTION_PACKING_GUIDE_ID"),
            "destinations": [
                {
                    "city": "dubai",
                    "start_date": "2024-09-01",
                    "end_date": "2024-12-31"
                },
                {
                    "city": "gurgaon",
                    "start_date": "2025-01-01", 
                    "end_date": "2025-05-31"
                }
            ],
            "preferences": {
                "optimization_goals": ["weight_efficiency", "business_readiness"],
                "packing_style": "minimalist_professional"
            }
        }
        
        print(f"   📦 Testing with mock data:")
        print(f"      Destinations: {[d['city'] for d in mock_trigger_data['destinations']]}")
        print(f"      Page ID: {mock_trigger_data['page_id']}")
        
        # Test trip configuration preparation
        print(f"   🔧 Testing trip configuration preparation...")
        trip_config = travel_pipeline_orchestrator._prepare_trip_configuration(mock_trigger_data)
        if trip_config:
            print("   ✅ Trip configuration preparation successful")
            print(f"      Trip overview: {trip_config['trip_overview']}")
            print(f"      Destinations: {len(trip_config['destinations'])}")
            print(f"      Weight constraints: {trip_config['weight_constraints']['clothes_allocation']['total_clothes_budget']}kg")
        else:
            print("   ❌ Trip configuration preparation failed")
            return False
        
        # Test wardrobe data retrieval
        print(f"   🔧 Testing wardrobe data retrieval...")
        try:
            import asyncio
            wardrobe_data = asyncio.run(
                asyncio.to_thread(travel_pipeline_orchestrator._get_travel_optimized_wardrobe_data)
            )
            
            if wardrobe_data:
                total_items = sum(len(items) for items in wardrobe_data.values())
                print(f"   ✅ Wardrobe data retrieval successful: {total_items} items")
                print(f"      Categories: {list(wardrobe_data.keys())}")
                for cat, items in wardrobe_data.items():
                    print(f"         {cat}: {len(items)} items")
            else:
                print("   ⚠️  No wardrobe data retrieved (might be normal if no items in DB)")
        except Exception as e:
            print(f"   ⚠️  Wardrobe data test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Travel pipeline test failed: {e}")
        import traceback
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        return False

def test_workflow_detection():
    """Test workflow detection with actual page"""
    print("\n🔧 Testing Workflow Detection...")
    
    try:
        from api.webhook_server import determine_workflow_type, get_travel_trigger_data
        
        page_id = os.getenv("NOTION_PACKING_GUIDE_ID")
        if not page_id:
            print("   ❌ NOTION_PACKING_GUIDE_ID not set")
            return False
        
        print(f"   🔍 Testing workflow detection for page: {page_id}")
        
        # Test workflow detection
        workflow_type = determine_workflow_type(page_id)
        print(f"   📋 Detected workflow: {workflow_type}")
        
        if workflow_type == "travel":
            print("   ✅ Travel workflow detected!")
            
            # Test trigger data extraction
            trigger_data = get_travel_trigger_data(page_id)
            if trigger_data:
                print("   ✅ Travel trigger data extraction successful")
                print(f"      Destinations: {len(trigger_data['destinations'])}")
                print(f"      Preferences: {trigger_data['preferences']}")
                return True
            else:
                print("   ❌ Travel trigger data extraction failed")
                return False
        else:
            print(f"   ⚠️  Expected 'travel' workflow, got '{workflow_type}'")
            print("   💡 Make sure the 'Generate' checkbox is checked in your Notion page")
            return False
        
    except Exception as e:
        print(f"   ❌ Workflow detection test failed: {e}")
        import traceback
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        return False

def test_ai_agents():
    """Test AI agents initialization"""
    print("\n🔧 Testing AI Agents...")
    
    try:
        from core.travel_packing_agent import travel_packing_agent
        
        # Test Gemini initialization
        gemini_key = os.getenv("GEMINI_AI_API_KEY")
        if gemini_key and travel_packing_agent.gemini_model:
            print("   ✅ Gemini AI initialized")
        else:
            print("   ⚠️  Gemini AI not initialized")
        
        # Test Groq initialization
        groq_key = os.getenv("GROQ_AI_API_KEY")
        if groq_key and travel_packing_agent.groq_client:
            print("   ✅ Groq AI initialized")
        else:
            print("   ⚠️  Groq AI not initialized")
        
        if not (gemini_key or groq_key):
            print("   ❌ No AI providers configured!")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ AI agents test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Travel Pipeline Debug Script\n")
    
    tests = [
        ("Environment Variables", test_environment),
        ("Imports", test_imports), 
        ("Notion Connection", test_notion_connection),
        ("Workflow Detection", test_workflow_detection),
        ("AI Agents", test_ai_agents),
        ("Travel Pipeline", test_travel_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print(f"\n📊 Test Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print(f"\n🎉 All tests passed! Travel pipeline should work.")
        print(f"\n💡 To trigger the travel pipeline:")
        print(f"   1. Go to your Notion travel planning page")
        print(f"   2. Check the 'Generate' checkbox")
        print(f"   3. The webhook should automatically trigger the travel pipeline")
    else:
        print(f"\n⚠️  Some tests failed. Fix these issues before running the travel pipeline.")
        
        # Provide specific guidance
        failed_tests = [test_name for test_name, result in results if not result]
        if "Environment Variables" in failed_tests:
            print(f"\n🔧 Environment Variables Fix:")
            print(f"   Make sure your .env file contains all required variables")
            
        if "Notion Connection" in failed_tests:
            print(f"\n🔧 Notion Connection Fix:")
            print(f"   Check your NOTION_TOKEN and NOTION_PACKING_GUIDE_ID")
            
        if "Workflow Detection" in failed_tests:
            print(f"\n🔧 Workflow Detection Fix:")
            print(f"   Make sure your Notion page has a 'Generate' checkbox property")
            print(f"   And that the checkbox is actually checked")

if __name__ == "__main__":
    main()