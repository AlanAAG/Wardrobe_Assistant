from flask import Flask, request, jsonify
import logging
import os
import asyncio
import threading
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
    Decide which workflow to run for the given Notion page.

    Travel (NEW default): run when ALL are true
      - 'Destinations'   not empty
      - 'Travel Preferences' not empty
      - 'Travel Dates'   has start (and ideally end)

    The old manual travel checkbox still works as an override:
      - 'Generate' / 'Generate Travel Packing' / 'Generate Packing' / 'Travel Generate' == True

    Outfit: run when BOTH are true:
      - 'Desired Aesthetic' not empty
      - 'Prompt' rich_text not empty
    """
    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})

        # manual travel override (still supported)
        for name in ["Generate", "Generate Travel Packing", "Generate Packing", "Travel Generate"]:
            if name in props and props[name].get("type") == "checkbox":
                if props[name].get("checkbox", False):
                    logging.info("🧳 Travel override checkbox detected.")
                    return "travel"

        # travel auto-trigger (per your spec)
        dest_ok = _prop_nonempty(props, ["Destinations", "Locations", "Cities"])
        prefs_ok = _prop_nonempty(props, ["Travel Preferences", "Trip Preferences", "Preferences"])
        dates_ok = _date_present(props, ["Travel Dates", "Trip Dates", "Dates"])

        logging.info(f"🔍 Travel triggers - destinations:{dest_ok} prefs:{prefs_ok} dates:{dates_ok}")
        if dest_ok and prefs_ok and dates_ok:
            return "travel"

        # outfit trigger (unchanged)
        aesthetic_prop = props.get("Desired Aesthetic", {})
        prompt_prop = props.get("Prompt", {})

        has_aesthetic = len(aesthetic_prop.get("multi_select", [])) > 0
        has_prompt = len(prompt_prop.get("rich_text", [])) > 0 and any(
            t.get("plain_text", "").strip() for t in prompt_prop.get("rich_text", [])
        )

        logging.info(f"🔍 Outfit triggers - aesthetic:{has_aesthetic} prompt:{has_prompt}")
        if has_aesthetic and has_prompt:
            return "outfit"

        return None
    except Exception as e:
        logging.error(f"❌ Error determining workflow type for {page_id}: {e}", exc_info=True)
        return None


def _prop_nonempty(props, candidates):
    """True if any candidate property is present with non-empty user content."""
    for name in candidates:
        p = props.get(name)
        if not p:
            continue
        typ = p.get("type")
        if typ == "multi_select" and p.get("multi_select"):
            return True
        if typ == "rich_text" and any(t.get("plain_text", "").strip() for t in p.get("rich_text", [])):
            return True
        if typ == "relation" and p.get("relation"):
            return True
        if typ == "title" and any(t.get("plain_text", "").strip() for t in p.get("title", [])):
            return True
    return False


def _date_present(props, candidates):
    """True if any candidate date property has a start date."""
    for name in candidates:
        p = props.get(name)
        if p and p.get("type") == "date":
            d = p.get("date") or {}
            if (d.get("start") or "").strip():
                return True
    return False


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
    Build the trigger_data for the travel pipeline using Notion page fields:
      - Destinations (multi_select / relation / title/rich_text as CSV)
      - Travel Preferences (multi_select or rich_text)
      - Travel Dates (date range)
    """
    page = notion.pages.retrieve(page_id=page_id)
    props = page.get("properties", {})

    # --- Destinations ---
    destinations = _read_destinations(props, page_id)

    # --- Preferences ---
    preferences = _read_preferences(props)

    # --- Dates ---
    date_info = _read_dates(props)
    total_days = date_info["days"]

    # distribute total days evenly across destinations if per-city durations are not provided
    if destinations and total_days > 0:
        base = total_days // len(destinations)
        extra = total_days % len(destinations)
        for i, d in enumerate(destinations):
            d.setdefault("city", d.get("name", ""))
            d["start_date"] = date_info["start"]
            d["end_date"] = date_info["end"]
            d["days"] = base + (1 if i < extra else 0)

    return {
        "page_id": page_id,
        "destinations": destinations,
        "preferences": preferences,
        "dates": date_info,
    }


def _read_destinations(props, page_id):
    # try a few likely property names
    for name in ["Destinations", "Locations", "Cities"]:
        p = props.get(name)
        if not p:
            continue
        typ = p.get("type")
        try:
            if typ == "multi_select":
                return [{"city": tag["name"]} for tag in p.get("multi_select", [])]
            if typ == "relation":
                out = []
                for rel in p.get("relation", []):
                    rel_page = notion.pages.retrieve(page_id=rel["id"])
                    title = _page_title(rel_page)
                    if title:
                        out.append({"city": title})
                return out
            if typ in ("title", "rich_text"):
                txt = ",".join(
                    t.get("plain_text", "") for t in p.get(typ, [])
                )
                names = [s.strip() for s in txt.split(",") if s.strip()]
                return [{"city": n} for n in names]
        except Exception as e:
            logging.warning(f"⚠️ Failed to parse '{name}' for page {page_id}: {e}", exc_info=True)
            continue
    # if nothing found, return empty list (pipeline will error out nicely)
    return []


def _read_preferences(props):
    for name in ["Travel Preferences", "Trip Preferences", "Preferences"]:
        p = props.get(name)
        if not p:
            continue
        typ = p.get("type")
        if typ == "multi_select":
            return [tag["name"] for tag in p.get("multi_select", [])]
        if typ == "rich_text":
            return [
                t.get("plain_text", "").strip()
                for t in p.get("rich_text", [])
                if t.get("plain_text", "").strip()
            ]
    return []


def _read_dates(props):
    for name in ["Travel Dates", "Trip Dates", "Dates"]:
        p = props.get(name)
        if p and p.get("type") == "date":
            d = p.get("date") or {}
            start = (d.get("start") or "").strip()
            end = (d.get("end") or "").strip() or start  # if missing end, assume a single day
            try:
                ds = datetime.fromisoformat(start[:19])
                de = datetime.fromisoformat(end[:19])
            except Exception:
                # leave as raw strings; orchestrator will still carry them
                days = 0
            else:
                days = max((de - ds).days + 1, 1)
            return {"start": start, "end": end, "days": days}
    return {"start": "", "end": "", "days": 0}


def run_async_travel_pipeline(pipeline, trigger_data):
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(pipeline.run_travel_packing_pipeline(trigger_data))
    finally:
        try:
            loop.run_until_complete(asyncio.sleep(0))
        except Exception:
            pass
        loop.close()

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