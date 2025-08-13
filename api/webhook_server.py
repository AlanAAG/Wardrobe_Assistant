from flask import Flask, request, jsonify
import logging
import os
import asyncio
import threading
import sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from data.notion_utils import notion
from core.pipeline_orchestrator import run_enhanced_outfit_pipeline
from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator

# [START monitoring_imports]
from monitoring.system_monitor import system_monitor
from caching.advanced_cache import advanced_cache
# [END monitoring_imports]

load_dotenv()

app = Flask(__name__)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('travel_debug.log')
    ]
)

# Set specific loggers
logging.getLogger('notion_client').setLevel(logging.DEBUG)
logging.getLogger('core.travel_pipeline_orchestrator').setLevel(logging.DEBUG)
logging.getLogger('core.travel_packing_agent').setLevel(logging.DEBUG)

# Create thread pool for async operations
executor = ThreadPoolExecutor(max_workers=5)

def check_environment_variables():
    """Check all required environment variables"""
    required_vars = {
        'NOTION_TOKEN': os.getenv('NOTION_TOKEN'),
        'NOTION_PACKING_GUIDE_ID': os.getenv('NOTION_PACKING_GUIDE_ID'),
        'NOTION_WARDROBE_DB_ID': os.getenv('NOTION_WARDROBE_DB_ID'),
        'GEMINI_AI_API_KEY': os.getenv('GEMINI_AI_API_KEY'),
        'GROQ_AI_API_KEY': os.getenv('GROQ_AI_API_KEY')
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        logging.error(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    logging.info("✅ All environment variables present")
    for var, value in required_vars.items():
        masked_value = value[:8] + "..." if len(value) > 8 else "***"
        logging.info(f"   {var}: {masked_value}")
    
    return True

@app.route('/webhook/notion', methods=['POST'])
def handle_unified_notion_webhook():
    """
    UNIFIED webhook handler for both outfit generation and travel packing.
    Routes to appropriate pipeline based on page properties.
    """
    # [START webhook_monitor]
    async def _handle_webhook_async():
        webhook_data = request.get_json()
        
        if not webhook_data:
            logging.warning("Received webhook with no JSON data")
            return jsonify({"error": "No JSON data received"}), 400
        
        logging.info(f"🔄 Received unified webhook: {webhook_data}")
        
        if "challenge" in webhook_data:
            challenge = webhook_data["challenge"]
            logging.info(f"Responding to Notion verification challenge: {challenge}")
            return jsonify({"challenge": challenge}), 200
        
        if "verification_token" in webhook_data:
            verification_token = webhook_data["verification_token"]
            logging.info(f"Responding to Notion verification token: {verification_token}")
            return jsonify({"message": "Verification token received"}), 200
        
        entity = webhook_data.get("entity", {})
        page_id = entity.get("id")
        
        if not page_id:
            logging.warning("Webhook missing entity.id (page_id)")
            return jsonify({"error": "Missing entity.id in webhook"}), 400
        
        logging.info(f"🔍 Processing webhook for page: {page_id}")
        
        workflow_type = determine_workflow_type(page_id)
        logging.info(f"🎯 Detected workflow type: {workflow_type}")
        
        if workflow_type == "outfit":
            return handle_outfit_workflow(page_id)
        elif workflow_type == "travel":
            return handle_travel_workflow(page_id)
        else:
            logging.info(f"No workflow triggered for page {page_id}")
            return jsonify({"message": "No workflow conditions met"}), 200
    
    try:
        return asyncio.run(system_monitor.track_operation(
            "notion_webhook_handler",
            _handle_webhook_async
        ))
    except Exception as e:
        # The monitor.track_operation will handle logging the error
        logging.error(f"❌ Webhook handler crashed: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
    # [END webhook_monitor]


def determine_workflow_type(page_id):
    """
    🎯 SMART ROUTING: Analyze page properties to determine workflow type
    Enhanced with comprehensive debugging
    
    Returns:
        - "outfit": if outfit generation should be triggered
        - "travel": if travel packing should be triggered  
        - None: if no workflow should be triggered
    """
    try:
        logging.info(f"🔍 Determining workflow type for page: {page_id}")
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        
        # LOG ALL PROPERTIES FOR DEBUGGING
        logging.info(f"🔍 DEBUG: All properties for page {page_id}:")
        for prop_name, prop_data in props.items():
            prop_type = prop_data.get("type", "unknown")
            if prop_type == "checkbox":
                value = prop_data.get("checkbox", False)
                logging.info(f"   📋 {prop_name} (checkbox): {value}")
            elif prop_type == "multi_select":
                values = [tag.get("name") for tag in prop_data.get("multi_select", [])]
                logging.info(f"   🏷️  {prop_name} (multi_select): {values}")
            elif prop_type == "rich_text":
                text = "".join([t.get("plain_text", "") for t in prop_data.get("rich_text", [])])
                logging.info(f"   📝 {prop_name} (rich_text): '{text[:50]}...' (length: {len(text)})")
            else:
                logging.info(f"   🔧 {prop_name} ({prop_type}): {prop_data}")
        
        # Check for travel workflow triggers (multiple possible property names)
        travel_triggers = ["Generate", "Generate Travel Packing", "Generate Packing", "Travel Generate"]
        travel_generate = False
        
        for trigger_name in travel_triggers:
            if trigger_name in props:
                travel_generate = props.get(trigger_name, {}).get("checkbox", False)
                logging.info(f"🔍 Found travel trigger '{trigger_name}': {travel_generate}")
                if travel_generate:
                    break
        
        if travel_generate:
            logging.info(f"🧳 Travel packing workflow detected for page {page_id}")
            return "travel"
        
        # Check for outfit workflow triggers
        aesthetic_prop = props.get("Desired Aesthetic", {})
        prompt_prop = props.get("Prompt", {})
        
        has_aesthetic = len(aesthetic_prop.get("multi_select", [])) > 0
        has_prompt = len(prompt_prop.get("rich_text", [])) > 0 and any(
            t.get("plain_text", "").strip() for t in prompt_prop.get("rich_text", [])
        )
        
        logging.info(f"🔍 Outfit triggers - aesthetic: {has_aesthetic}, prompt: {has_prompt}")
        
        if has_aesthetic and has_prompt:
            logging.info(f"👕 Outfit generation workflow detected for page {page_id}")
            return "outfit"
        
        logging.info(f"❌ No workflow triggers found for page {page_id}")
        return None
        
    except Exception as e:
        logging.error(f"❌ Error determining workflow type for page {page_id}: {e}", exc_info=True)
        return None

def handle_outfit_workflow(page_id):
    """Handle outfit generation workflow (existing logic)"""
    try:
        logging.info(f"👕 ENTERING handle_outfit_workflow for page {page_id}")
        
        if not validate_outfit_trigger_conditions(page_id):
            logging.info("Outfit trigger conditions not met")
            return jsonify({"message": "Outfit trigger conditions not met"}), 200
        
        trigger_data = get_outfit_trigger_data(page_id)
        if not trigger_data:
            return jsonify({"error": "Failed to extract outfit trigger data"}), 400
        
        logging.info(f"Outfit trigger: aesthetic={trigger_data['aesthetics']}, prompt='{trigger_data['prompt']}'")
        
        future = executor.submit(run_async_outfit_pipeline, trigger_data)
        
        return jsonify({
            "message": "Outfit generation started",
            "page_id": page_id,
            "workflow": "outfit",
            "status": "processing",
            "aesthetic": trigger_data['aesthetics'],
            "prompt_preview": trigger_data['prompt'][:50] + "..." if len(trigger_data['prompt']) > 50 else trigger_data['prompt']
        }), 200
        
    except Exception as e:
        logging.error(f"❌ Outfit workflow error: {e}", exc_info=True)
        return jsonify({"error": "Outfit workflow failed"}), 500

def handle_travel_workflow(page_id):
    """Handle travel packing workflow (enhanced with debugging)"""
    try:
        logging.info(f"🧳 ENTERING handle_travel_workflow for page {page_id}")
        
        # Extract travel trigger data with debugging
        logging.info(f"🧳 About to call get_travel_trigger_data")
        travel_trigger_data = get_travel_trigger_data(page_id)
        
        if not travel_trigger_data:
            logging.error("❌ Failed to extract travel trigger data")
            return jsonify({"error": "Failed to extract travel trigger data"}), 400
        
        logging.info(f"✅ Travel trigger data extracted: {travel_trigger_data}")
        
        # Test orchestrator access
        try:
            logging.info(f"🧳 Testing travel_pipeline_orchestrator access...")
            test_result = hasattr(travel_pipeline_orchestrator, 'run_travel_packing_pipeline')
            logging.info(f"🧳 Orchestrator has run_travel_packing_pipeline method: {test_result}")
        except Exception as e:
            logging.error(f"❌ Orchestrator access test failed: {e}")
            return jsonify({"error": f"Orchestrator access failed: {str(e)}"}), 500
        
        logging.info(f"🧳 Starting async travel pipeline...")
        future = executor.submit(run_async_travel_pipeline, travel_trigger_data)
        
        return jsonify({
            "message": "Travel packing started", 
            "page_id": page_id,
            "workflow": "travel",
            "status": "processing",
            "trip_type": "business_school_relocation",
            "destinations": ["Dubai (Sep-Dec)", "Gurgaon (Jan-May)"],
            "debug_info": {
                "trigger_data_extracted": True,
                "orchestrator_accessible": True,
                "async_pipeline_started": True
            }
        }), 200
        
    except Exception as e:
        logging.error(f"❌ Travel workflow error: {e}", exc_info=True)
        return jsonify({
            "error": "Travel workflow failed", 
            "details": str(e),
            "page_id": page_id
        }), 500

def validate_outfit_trigger_conditions(page_id):
    """Validate outfit generation trigger conditions (existing logic)"""
    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        
        aesthetic_prop = props.get("Desired Aesthetic", {})
        multi_select = aesthetic_prop.get("multi_select", [])
        has_aesthetic = len(multi_select) > 0
        
        prompt_prop = props.get("Prompt", {})
        rich_text = prompt_prop.get("rich_text", [])
        has_prompt = len(rich_text) > 0 and any(t.get("plain_text", "").strip() for t in rich_text)
        
        logging.info(f"Outfit validation: aesthetic={has_aesthetic}, prompt={has_prompt}")
        return has_aesthetic and has_prompt
        
    except Exception as e:
        logging.error(f"Error validating outfit trigger: {e}")
        return False

def get_outfit_trigger_data(page_id):
    """Extract outfit trigger data (existing logic)"""
    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        
        aesthetic_prop = props.get("Desired Aesthetic", {})
        multi_select = aesthetic_prop.get("multi_select", [])
        aesthetics = [tag.get("name") for tag in multi_select]
        
        prompt_prop = props.get("Prompt", {})
        rich_text = prompt_prop.get("rich_text", [])
        prompt_text = "".join([t.get("plain_text", "") for t in rich_text]) if rich_text else ""
        
        return {
            "page_id": page_id,
            "aesthetics": aesthetics,
            "prompt": prompt_text.strip()
        }
        
    except Exception as e:
        logging.error(f"Error extracting outfit trigger data: {e}")
        return None

def get_travel_trigger_data(page_id):
    """
    Extract travel packing configuration from your AI Travel Planner page
    Enhanced with debugging and flexible property detection
    """
    try:
        logging.info(f"🧳 Extracting travel trigger data for page {page_id}")
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        
        # Log all properties for debugging
        logging.info(f"🧳 Available properties: {list(props.keys())}")
        
        # Try multiple possible property names for travel preferences
        preferences_prop_names = ["Travel Preferences", "Preferences", "Notes", "Description"]
        preferences_text = ""
        
        for prop_name in preferences_prop_names:
            if prop_name in props:
                prop_data = props.get(prop_name, {})
                rich_text = prop_data.get("rich_text", [])
                preferences_text = "".join([t.get("plain_text", "") for t in rich_text]) if rich_text else ""
                if preferences_text.strip():
                    logging.info(f"🧳 Found preferences in '{prop_name}': '{preferences_text[:100]}...'")
                    break
        
        # Try multiple possible property names for travel dates
        dates_prop_names = ["Travel Dates", "Dates", "Date", "Trip Dates"]
        start_date = "2024-09-01"
        end_date = "2025-05-31"
        
        for prop_name in dates_prop_names:
            if prop_name in props:
                dates_prop = props.get(prop_name, {})
                date_range = dates_prop.get("date", {})
                if date_range:
                    start_date = date_range.get("start", start_date)
                    end_date = date_range.get("end", end_date)
                    logging.info(f"🧳 Found dates in '{prop_name}': {start_date} to {end_date}")
                    break
        
        # Default destination configuration (can be made configurable later)
        destinations_config = [
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
        ]
        
        trigger_data = {
            "page_id": page_id,
            "destinations": destinations_config,
            "preferences": {
                "optimization_goals": ["weight_efficiency", "business_readiness", "climate_coverage", "cultural_compliance"],
                "trip_type": "business_school_relocation",
                "user_notes": preferences_text,
                "packing_style": "minimalist_professional"
            }
        }
        
        logging.info(f"✅ Travel trigger data constructed successfully")
        logging.info(f"   Destinations: {len(destinations_config)} cities")
        logging.info(f"   Preferences length: {len(preferences_text)} chars")
        logging.info(f"   Date range: {start_date} to {end_date}")
        
        return trigger_data
        
    except Exception as e:
        logging.error(f"❌ Error extracting travel trigger data: {e}", exc_info=True)
        return None

def run_async_outfit_pipeline(trigger_data):
    """Run outfit pipeline in background thread (existing logic)"""
    # [START outfit_pipeline_monitor]
    async def _run_pipeline_task():
        return await run_enhanced_outfit_pipeline(trigger_data)

    try:
        logging.info("Starting async outfit pipeline...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(system_monitor.track_operation(
                f"outfit_generation_{trigger_data['page_id']}",
                _run_pipeline_task
            ))
            
            if result["success"]:
                method = result["generation_method"]
                items = result.get("outfit_items", "N/A")
                logging.info(f"✅ Outfit pipeline completed successfully!")
                logging.info(f"   Method: {method}")
                logging.info(f"   Items: {items}")
            else:
                logging.error(f"❌ Outfit pipeline failed: {result['error']}")
                
        finally:
            loop.close()
            
    except Exception as e:
        logging.error(f"Error in async outfit pipeline: {e}")
    # [END outfit_pipeline_monitor]

def run_async_travel_pipeline(trigger_data):
    """Run travel packing pipeline in background thread (enhanced with debugging)"""
    # [START travel_pipeline_monitor]
    async def _run_travel_task():
        logging.info(f"🧳 Calling travel_pipeline_orchestrator.run_travel_packing_pipeline")
        logging.info(f"🧳 Trigger data: {trigger_data}")
        return await travel_pipeline_orchestrator.run_travel_packing_pipeline(trigger_data)

    try:
        logging.info("🧳 Starting async travel packing pipeline...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(system_monitor.track_operation(
                f"travel_packing_{trigger_data['page_id']}",
                _run_travel_task
            ))
            
            if result["success"]:
                logging.info(f"✅ Travel packing completed successfully!")
                logging.info(f"   Method: {result['generation_method']}")
                logging.info(f"   Items selected: {result.get('total_items_selected', 'N/A')}")
                logging.info(f"   Weight: {result.get('total_weight_kg', 'N/A')}kg")
                logging.info(f"   Destinations: {result.get('destinations', [])}")
                logging.info(f"   Duration: {result.get('trip_duration_months', 'N/A')} months")
            else:
                logging.error(f"❌ Travel packing failed: {result['error']}")
                logging.error(f"   Method attempted: {result.get('generation_method', 'unknown')}")
                
        finally:
            loop.close()
            
    except Exception as e:
        logging.error(f"❌ Error in travel packing pipeline: {e}", exc_info=True)
    # [END travel_pipeline_monitor]


@app.route('/debug/page/<page_id>', methods=['GET'])
def debug_page_properties(page_id):
    """Debug endpoint to inspect page properties for routing"""
    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        
        property_analysis = {}
        for prop_name, prop_data in props.items():
            prop_type = prop_data.get("type")
            if prop_type == "checkbox":
                property_analysis[prop_name] = {
                    "type": "checkbox",
                    "value": prop_data.get("checkbox", False)
                }
            elif prop_type == "multi_select":
                property_analysis[prop_name] = {
                    "type": "multi_select",
                    "values": [tag.get("name") for tag in prop_data.get("multi_select", [])]
                }
            elif prop_type == "rich_text":
                property_analysis[prop_name] = {
                    "type": "rich_text",
                    "text": "".join([t.get("plain_text", "") for t in prop_data.get("rich_text", [])])
                }
            elif prop_type == "date":
                property_analysis[prop_name] = {
                    "type": "date",
                    "date_range": prop_data.get("date", {})
                }
        
        workflow = determine_workflow_type(page_id)
        
        return jsonify({
            "page_id": page_id,
            "properties": property_analysis,
            "detected_workflow": workflow,
            "page_title": page.get("properties", {}).get("title", {})
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug/travel/test/<page_id>', methods=['GET'])
def debug_travel_workflow(page_id):
    """Debug endpoint to test travel workflow detection"""
    try:
        logging.info(f"🔍 DEBUG: Testing travel workflow for page {page_id}")
        
        # Test workflow detection
        workflow_type = determine_workflow_type(page_id)
        logging.info(f"🔍 DEBUG: Detected workflow: {workflow_type}")
        
        if workflow_type == "travel":
            # Test travel trigger data extraction
            travel_data = get_travel_trigger_data(page_id)
            logging.info(f"🔍 DEBUG: Travel trigger data: {travel_data}")
            
            # Test orchestrator initialization
            try:
                orchestrator_status = hasattr(travel_pipeline_orchestrator, 'run_travel_packing_pipeline')
                logging.info("✅ Travel orchestrator access successful")
                
                return jsonify({
                    "status": "success",
                    "workflow_detected": workflow_type,
                    "travel_data": travel_data,
                    "orchestrator_status": "accessible" if orchestrator_status else "not_accessible",
                    "orchestrator_methods": [method for method in dir(travel_pipeline_orchestrator) if not method.startswith('_')]
                })
                
            except Exception as e:
                logging.error(f"❌ Orchestrator error: {e}")
                return jsonify({
                    "status": "error",
                    "error": f"Orchestrator error: {str(e)}"
                }), 500
        else:
            return jsonify({
                "status": "no_travel_workflow",
                "workflow_detected": workflow_type
            })
            
    except Exception as e:
        logging.error(f"❌ Debug endpoint error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    # [START health_check_update]
    try:
        # Get detailed performance data from the monitor
        performance_dashboard = asyncio.run(system_monitor.get_performance_dashboard())
        cache_stats = asyncio.run(advanced_cache.get_stats())
        
        return jsonify({
            "status": "healthy",
            "workflows": ["outfit", "travel"],
            "performance": performance_dashboard,
            "cache": cache_stats,
            "environment_check": check_environment_variables()
        }), 200
    except Exception as e:
        logging.error(f"Error generating health check response: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": "Failed to retrieve system metrics"
        }), 500
    # [END health_check_update]

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "message": "Unified AI Wardrobe Assistant", 
        "workflows": ["outfit_generation", "travel_packing"],
        "debug_endpoints": [
            "/debug/page/<page_id>",
            "/debug/travel/test/<page_id>",
            "/health"
        ]
    }), 200

if __name__ == '__main__':
    logging.info("🚀 Starting AI Wardrobe Assistant...")
    
    if not check_environment_variables():
        logging.error("❌ Environment check failed. Exiting.")
        sys.exit(1)
    
    # Test imports
    try:
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        logging.info("✅ Travel pipeline orchestrator imported successfully")
    except Exception as e:
        logging.error(f"❌ Failed to import travel orchestrator: {e}")
        sys.exit(1)
    
    port = int(os.environ.get('PORT', 5000))
    logging.info(f"🌐 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)