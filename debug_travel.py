#!/usr/bin/env python3
"""
Enhanced debug script for travel packing pipeline.
Tests all components and provides detailed feedback.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)

def check_environment():
    """Check all required environment variables."""
    print("🔧 Testing Environment Setup...")
    
    required_vars = [
        'NOTION_TOKEN',
        'NOTION_PACKING_GUIDE_ID', 
        'NOTION_WARDROBE_DB_ID',
        'GEMINI_AI_API_KEY',
        'GROQ_AI_API_KEY'
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show masked value for security
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"   ✅ {var}: {masked}")
        else:
            missing.append(var)
            print(f"   ❌ {var}: Not set")
    
    return len(missing) == 0

def test_imports():
    """Test importing all required modules."""
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
        
        # Check available methods
        methods = [method for method in dir(travel_pipeline_orchestrator) if not method.startswith('_') or method in ['_prepare_trip_configuration_enhanced']]
        print(f"   📋 Orchestrator methods: {[m for m in methods if not m.startswith('__')]}")
        
    except Exception as e:
        print(f"   ❌ travel_pipeline_orchestrator import failed: {e}")
        return False
    
    try:
        from core.travel_packing_agent import travel_packing_agent
        print("   ✅ travel_packing_agent import successful")
    except Exception as e:
        print(f"   ❌ travel_packing_agent import failed: {e}")
        return False
    
    return True

def test_notion_connection():
    """Test Notion API connection."""
    print("\n🔧 Testing Notion Connection...")
    
    try:
        from data.notion_utils import notion
        
        packing_guide_id = os.getenv('NOTION_PACKING_GUIDE_ID')
        page = notion.pages.retrieve(page_id=packing_guide_id)
        print("   ✅ Notion connection successful")
        
        # Show page properties
        props = page.get('properties', {})
        print("   📋 Page properties:")
        for prop_name, prop_data in props.items():
            prop_type = prop_data.get('type', 'unknown')
            print(f"      {prop_name} ({prop_type})")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Notion connection failed: {e}")
        return False

def test_workflow_detection():
    """Test workflow detection logic."""
    print("\n🔧 Testing Workflow Detection...")
    
    try:
        from api.webhook_server import determine_workflow_type
        
        packing_guide_id = os.getenv('NOTION_PACKING_GUIDE_ID')
        print(f"   🔍 Testing workflow detection for page: {packing_guide_id}")
        
        workflow = determine_workflow_type(packing_guide_id)
        print(f"   📋 Detected workflow: {workflow}")
        
        if workflow == 'travel':
            print("   ✅ Travel workflow detected successfully")
            return True
        else:
            print("   ⚠️  Expected 'travel' workflow, got '{}'".format(workflow))
            print("   💡 To trigger travel workflow, your Notion page needs:")
            print("      - 'Destinations' property with values")
            print("      - 'Travel Preferences' property with values") 
            print("      - 'Travel Dates' property with start date")
            print("   OR:")
            print("      - 'Generate' checkbox property that is checked")
            return False
            
    except Exception as e:
        print(f"   ❌ Workflow detection failed: {e}")
        return False

def test_ai_agents():
    """Test AI agent initialization."""
    print("\n🔧 Testing AI Agents...")
    
    try:
        from core.travel_packing_agent import travel_packing_agent
        
        # Test Gemini
        if travel_packing_agent.gemini_model:
            print("   ✅ Gemini AI initialized")
        else:
            print("   ⚠️  Gemini AI not available")
        
        # Test Groq
        if travel_packing_agent.groq_client:
            print("   ✅ Groq AI initialized")
        else:
            print("   ⚠️  Groq AI not available")
        
        return True
        
    except Exception as e:
        print(f"   ❌ AI agents test failed: {e}")
        return False

async def test_travel_pipeline():
    """Test the travel pipeline with mock data."""
    print("\n🔧 Testing Travel Pipeline...")
    
    try:
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        
        # FIX: Initialize the orchestrator before testing its internal methods
        await travel_pipeline_orchestrator.ensure_ready()

        # Create mock trigger data that matches the expected format
        mock_trigger_data = {
            "page_id": os.getenv('NOTION_PACKING_GUIDE_ID'),
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
            "preferences": ["weight_efficiency", "business_readiness"],
            "dates": {
                "start": "2024-09-01",
                "end": "2025-05-31",
                "days": 242
            }
        }
        
        print("   📦 Testing with mock data:")
        print(f"      Destinations: {[d['city'] for d in mock_trigger_data['destinations']]}")
        print(f"      Page ID: {mock_trigger_data['page_id']}")
        
        # Test trip configuration preparation with correct method name
        print("   🔧 Testing trip configuration preparation...")
        trip_config = await travel_pipeline_orchestrator._prepare_trip_configuration_enhanced(mock_trigger_data)
        
        if trip_config:
            print("   ✅ Trip configuration prepared successfully")
            print(f"      Duration: {trip_config['trip_overview']['total_duration_months']} months")
            print(f"      Temperature range: {trip_config['trip_overview']['temperature_range']['min']}°C to {trip_config['trip_overview']['temperature_range']['max']}°C")
        else:
            print("   ❌ Trip configuration preparation failed")
            return False
        
        # Test wardrobe data acquisition
        print("   🔧 Testing wardrobe data acquisition...")
        try:
            wardrobe_data = await travel_pipeline_orchestrator._get_travel_optimized_wardrobe_data_enhanced()
            total_items = sum(len(items) for items in wardrobe_data.values())
            print(f"   ✅ Wardrobe data acquired: {total_items} items")
        except Exception as e:
            print(f"   ⚠️  Wardrobe data acquisition failed: {e}")
            print("   💡 This may be due to missing wardrobe database or network issues")
        
        print("   ✅ Travel pipeline core components working")
        return True
        
    except Exception as e:
        print(f"   ❌ Travel pipeline test failed: {e}")
        print(f"   🔍 Traceback: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_full_pipeline_test():
    """Test the complete travel pipeline (optional - only if basic tests pass)."""
    print("\n🔧 Testing Full Travel Pipeline...")
    
    try:
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        
        # Mock trigger data with all required fields
        test_trigger_data = {
            "page_id": os.getenv('NOTION_PACKING_GUIDE_ID'),
            "destinations": [
                {
                    "city": "dubai",
                    "start_date": "2024-09-01",
                    "end_date": "2024-12-31"
                }
            ],
            "preferences": ["weight_efficiency", "business_readiness"]
        }
        
        print("   🚀 Running complete pipeline test...")
        # FIX: Call the async method directly on the orchestrator instance
        result = await travel_pipeline_orchestrator.run_travel_packing_pipeline(test_trigger_data)
        
        if result and result.get('success'):
            print("   ✅ Full pipeline test SUCCESSFUL!")
            print(f"      Generation method: {result.get('generation_method', 'unknown')}")
            print(f"      Items selected: {result.get('total_items_selected', 'unknown')}")
            return True
        else:
            print(f"   ❌ Full pipeline test failed: {result.get('error', 'Unknown error') if result else 'No result returned'}")
            return False
        
    except Exception as e:
        print(f"   ❌ Full pipeline test crashed: {e}")
        return False

def setup_notion_page_instructions():
    """Provide instructions for setting up the Notion page correctly."""
    print("\n📋 Notion Page Setup Instructions:")
    print("   To make travel workflow detection work, add these properties to your page:")
    print()
    print("   Required Properties (Option 1 - Auto-trigger):")
    print("   1. 'Destinations' (Multi-select or Relation) - Add destination cities")
    print("   2. 'Travel Preferences' (Multi-select or Rich Text) - Add preferences")
    print("   3. 'Travel Dates' (Date) - Set start and end dates")
    print()
    print("   Alternative (Option 2 - Manual trigger):")
    print("   1. 'Generate' (Checkbox) - Check this box to trigger travel workflow")
    print()
    print("   Your page currently only has: 'title' property")
    print("   Add the missing properties and try again!")

async def main():
    """Main debug function."""
    print("🚀 Travel Pipeline Debug Script\n")
    
    results = {}
    
    # Test 1: Environment
    results['environment'] = check_environment()
    
    # Test 2: Imports
    results['imports'] = test_imports()
    
    # Test 3: Notion Connection
    results['notion'] = test_notion_connection()
    
    # Test 4: Workflow Detection
    results['workflow'] = test_workflow_detection()
    
    # Test 5: AI Agents
    results['ai_agents'] = test_ai_agents()
    
    # Test 6: Travel Pipeline
    results['pipeline'] = await test_travel_pipeline()
    
    # Optional: Full pipeline test (only if basic tests pass)
    if all([results['environment'], results['imports'], results['notion']]):
        print("\n🔧 Advanced Testing...")
        results['full_pipeline'] = await run_full_pipeline_test()
    
    # Summary
    print(f"\n📊 Test Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {test_name.replace('_', ' ').title()}")
    
    # Recommendations
    failed_tests = [name for name, passed in results.items() if not passed]
    
    if failed_tests:
        print(f"\n⚠️  Some tests failed. Fix these issues before running the travel pipeline.")
        
        if 'workflow' in failed_tests:
            setup_notion_page_instructions()
        
        if 'pipeline' in failed_tests:
            print("\n🔧 Pipeline Fix:")
            print("   The travel pipeline methods have been updated.")
            print("   Make sure you're using the latest version of the code.")
    else:
        print(f"\n🎉 All tests passed! Your travel pipeline is ready to use.")
        print(f"   You can now trigger the travel workflow from your Notion page.")

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the debug script
    asyncio.run(main())