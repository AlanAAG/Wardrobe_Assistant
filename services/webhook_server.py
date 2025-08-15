import logging
import os
import asyncio
import threading
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv

# Only import Flask if available
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.error("Flask not available - install with: pip install flask")

load_dotenv()

# Create Flask app only if Flask is available
if FLASK_AVAILABLE:
    app = Flask(__name__)
else:
    app = None

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
        'NOTION_OUTFIT_LOG_DB_ID': os.getenv('NOTION_OUTFIT_LOG_DB_ID'),
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

def _get_notion_client():
    """Lazy load Notion client to avoid circular imports."""
    try:
        from data.notion_utils import notion
        return notion
    except ImportError as e:
        logging.error(f"Failed to import Notion client: {e}")
        return None

def _get_core_functions():
    """Lazy load core functions to avoid circular imports."""
    functions = {}
    
    try:
        from core.outfit_pipeline_orchestrator import outfit_pipeline_orchestrator
        functions['outfit_pipeline_orchestrator'] = outfit_pipeline_orchestrator
    except ImportError as e:
        logging.error(f"Outfit pipeline orchestrator import failed: {e}")
        functions['outfit_pipeline_orchestrator'] = None
    
    try:
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        functions['travel_pipeline_orchestrator'] = travel_pipeline_orchestrator
    except ImportError as e:
        logging.error(f"Travel pipeline orchestrator import failed: {e}")
        functions['travel_pipeline_orchestrator'] = None
    
    return functions

def _get_monitoring():
    """Lazy load monitoring components to avoid circular imports."""
    try:
        from monitoring import get_system_monitor
        return get_system_monitor()
    except ImportError as e:
        logging.warning(f"System monitor not available: {e}")
        return None

def _get_advanced_cache():
    """Lazy load advanced cache to avoid circular imports."""
    try:
        from caching.advanced_cache import advanced_cache
        return advanced_cache
    except ImportError as e:
        logging.warning(f"Advanced cache not available: {e}")
        return None

def _page_title(page):
    """Extract the plain text title from a Notion page object."""
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            title_data = prop.get("title", [])
            if title_data:
                return "".join(t.get("plain_text", "") for t in title_data).strip()
    return None

    # Startup validation and initialization
def initialize_server():
    """Initialize server with comprehensive validation"""
    logging.info("🚀 Starting AI Wardrobe Assistant...")
    
    if not FLASK_AVAILABLE:
        logging.error("❌ Flask not available. Server cannot start.")
        return False
    
    if not check_environment_variables():
        logging.error("❌ Environment check failed. Exiting.")
        return False
    
    # Test core imports
    try:
        core_functions = _get_core_functions()
        travel_orchestrator = core_functions.get('travel_pipeline_orchestrator')
        outfit_orchestrator = core_functions.get('outfit_pipeline_orchestrator')
        
        if travel_orchestrator:
            logging.info("✅ Travel pipeline orchestrator loaded successfully")
        else:
            logging.warning("⚠️  Travel pipeline orchestrator not available")
        
        if outfit_orchestrator:
            logging.info("✅ Outfit pipeline orchestrator loaded successfully")
        else:
            logging.warning("⚠️  Outfit pipeline orchestrator not available")
            
    except Exception as e:
        logging.error(f"❌ Failed to load core components: {e}")
        return False
    
    return True

# Only define routes if Flask is available
if FLASK_AVAILABLE:
    @app.route('/webhook/notion', methods=['POST'])
    def handle_unified_notion_webhook():
        """
        UNIFIED webhook handler for both outfit generation and travel packing.
        Routes to appropriate pipeline based on page properties.
        """
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
        elif workflow_type == "laundry_day":
            return handle_laundry_day_workflow(page_id)
        else:
            logging.info(f"No workflow triggered for page {page_id}")
            return jsonify({"message": "No workflow conditions met"}), 200

def determine_workflow_type(page_id):
    """
    Decide which workflow to run for the given Notion page.
    """
    notion = _get_notion_client()
    if not notion:
        logging.error("Notion client not available for workflow detection")
        return None

    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        parent_db_id = page.get("parent", {}).get("database_id", "").replace("-", "")

        # Laundry Day workflow
        dirty_clothes_db_id = os.getenv("NOTION_DIRTY_CLOTHES_DB_ID", "").replace("-", "")
        if parent_db_id and parent_db_id == dirty_clothes_db_id:
            ready_for_laundry_prop = props.get("Ready for Laundry", {})
            if ready_for_laundry_prop.get("checkbox") is False:
                logging.info("🧺 Laundry day trigger detected.")
                return "laundry_day"

        # Manual travel override
        for name in ["Generate", "Generate Travel Packing", "Generate Packing", "Travel Generate"]:
            if name in props and props[name].get("type") == "checkbox" and props[name].get("checkbox"):
                logging.info("🧳 Travel override checkbox detected.")
                return "travel"

        # Travel auto-trigger
        dest_ok = _prop_nonempty(props, ["Destinations", "Locations", "Cities"])
        prefs_ok = _prop_nonempty(props, ["Travel Preferences", "Trip Preferences", "Preferences"])
        dates_ok = _date_present(props, ["Travel Dates", "Trip Dates", "Dates"])
        if dest_ok and prefs_ok and dates_ok:
            logging.info("✈️ Travel auto-trigger detected.")
            return "travel"

        # Outfit trigger
        aesthetic_prop = props.get("Desired Aesthetic", {})
        prompt_prop = props.get("Prompt", {})
        has_aesthetic = len(aesthetic_prop.get("multi_select", [])) > 0
        has_prompt = len(prompt_prop.get("rich_text", [])) > 0 and any(
            t.get("plain_text", "").strip() for t in prompt_prop.get("rich_text", [])
        )
        if has_aesthetic and has_prompt:
            logging.info("👕 Outfit trigger detected.")
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
    """Handle outfit generation workflow"""
    try:
        logging.info(f"👕 ENTERING handle_outfit_workflow for page {page_id}")
        
        # Get the pipeline function
        core_functions = _get_core_functions()
        outfit_orchestrator = core_functions.get('outfit_pipeline_orchestrator')
        
        if not outfit_orchestrator:
            return jsonify({"error": "Outfit pipeline not available"}), 500
        
        future = executor.submit(run_async_outfit_pipeline, outfit_orchestrator)

        return jsonify({
            "message": "Outfit generation started",
            "page_id": page_id,
            "workflow": "outfit",
            "status": "processing",
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
        
        # Get the travel orchestrator
        core_functions = _get_core_functions()
        travel_orchestrator = core_functions.get('travel_pipeline_orchestrator')
        
        if not travel_orchestrator:
            return jsonify({"error": "Travel pipeline not available"}), 500
        
        # Test orchestrator access
        try:
            logging.info(f"🧳 Testing travel_pipeline_orchestrator access...")
            test_result = hasattr(travel_orchestrator, 'run_travel_packing_pipeline')
            logging.info(f"🧳 Orchestrator has run_travel_packing_pipeline method: {test_result}")
        except Exception as e:
            logging.error(f"❌ Orchestrator access test failed: {e}")
            return jsonify({"error": f"Orchestrator access failed: {str(e)}"}), 500
        
        logging.info(f"🧳 Starting async travel pipeline...")
        future = executor.submit(run_async_travel_pipeline, travel_orchestrator, travel_trigger_data)
        
        return jsonify({
            "message": "Travel packing started", 
            "page_id": page_id,
            "workflow": "travel",
            "status": "processing",
            "trip_type": "business_school_relocation",
            "destinations": [d.get("city", "Unknown") for d in travel_trigger_data.get("destinations", [])],
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


def handle_laundry_day_workflow(page_id):
    """
    Handles the laundry day workflow.
    """
    try:
        logging.info(f"🧺 ENTERING handle_laundry_day_workflow for page {page_id}")
        from data.notion_utils import update_items_washed_status, archive_page

        notion = _get_notion_client()
        if not notion:
            return jsonify({"error": "Notion client not available"}), 500

        # Retrieve the page from the "Dirty Clothes" database
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})

        # Get the related clothing item
        clothing_item_relation = props.get("Clothing Item", {}).get("relation", [])
        if not clothing_item_relation:
            logging.error(f"No clothing item relation found for page {page_id}")
            return jsonify({"error": "No clothing item relation found"}), 400

        clothing_item_id = clothing_item_relation[0]["id"]

        # Update the washed status and archive the page
        update_items_washed_status(clothing_item_id, "Done")
        archive_page(page_id)

        logging.info(f"✅ Laundry day workflow completed for page {page_id}")
        return jsonify({
            "message": "Laundry day workflow completed",
            "page_id": page_id,
            "workflow": "laundry_day",
            "status": "completed",
        }), 200

    except Exception as e:
        logging.error(f"❌ Laundry day workflow error: {e}", exc_info=True)
        return jsonify({"error": "Laundry day workflow failed"}), 500


def validate_outfit_trigger_conditions(page_id):
    """Validate outfit generation trigger conditions (existing logic)"""
    notion = _get_notion_client()
    if not notion:
        return False
        
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
    notion = _get_notion_client()
    if not notion:
        return None
        
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
    Builds the trigger_data for the travel pipeline by extracting the raw,
    unprocessed data directly from the user's Notion page.
    """
    notion = _get_notion_client()
    if not notion:
        return None
        
    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})

        # The function now extracts raw text for the AI to analyze,
        # including the new dynamic bag information.
        return {
            "page_id": page_id,
            "destinations": _read_destinations(props, page_id),
            "preferences": _read_preferences(props),
            "dates": _read_dates(props),
            "bags": _read_bags(props)
        }
    
    except Exception as e:
        logging.error(f"Error extracting travel trigger data: {e}")
        return None

def _read_destinations(props, page_id):
    """Reads a list of destination city names from the multi-select property."""
    for name in ["Destinations", "Locations", "Cities"]:
        p = props.get(name)
        if p and p.get("type") == "multi_select":
            # Return a simple list of city names
            return [tag.get("name", "") for tag in p.get("multi_select", [])]
    return []

def _read_preferences(props):
    """Reads the raw text from the Travel Preferences property."""
    for name in ["Travel Preferences", "Trip Preferences", "Preferences"]:
        p = props.get(name)
        if p and p.get("type") == "rich_text":
            return "".join(t.get("plain_text", "") for t in p.get("rich_text", [])).strip()
    return ""

def _read_dates(props):
    """Read travel dates from date properties"""
    for name in ["Travel Dates", "Trip Dates", "Dates"]:
        p = props.get(name)
        if p and p.get("type") == "date":
            d = p.get("date") or {}
            start = (d.get("start") or "").strip()
            end = (d.get("end") or "").strip() or start  # if missing end, assume a single day
            try:
                from datetime import datetime
                ds = datetime.fromisoformat(start[:19])
                de = datetime.fromisoformat(end[:19])
                days = max((de - ds).days + 1, 1)
            except Exception:
                # leave as raw strings; orchestrator will still carry them
                days = 0
            return {"start": start, "end": end, "days": days}
    return {"start": "", "end": "", "days": 0}

def _read_bags(props):
    """Reads bag types and their weight limits from the multi-select property."""
    for name in ["Bags & Weight Limits", "Luggage"]:
        p = props.get(name)
        if p and p.get("type") == "multi_select":
            # Returns a list of strings, e.g., ["Checked Bag: 23kg", "Cabin Bag: 10kg"]
            return [tag.get("name", "") for tag in p.get("multi_select", [])]
    return []


def run_async_outfit_pipeline(outfit_orchestrator):
    """Run outfit pipeline in async context"""
    try:
        import asyncio
        
        # Check if we're in an async context already
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, outfit_orchestrator.run_daily_outfit_pipeline())
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(outfit_orchestrator.run_daily_outfit_pipeline())
    
    except Exception as e:
        logging.error(f"Error in async outfit pipeline: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def run_async_travel_pipeline(travel_orchestrator, trigger_data):
    """Run travel pipeline in async context"""
    try:
        import asyncio
        
        # Check if we're in an async context already
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    travel_orchestrator.run_travel_packing_pipeline(trigger_data)
                )
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(travel_orchestrator.run_travel_packing_pipeline(trigger_data))
    
    except Exception as e:
        logging.error(f"Error in async travel pipeline: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# Only define debug routes if Flask is available
if FLASK_AVAILABLE:
    @app.route('/debug/page/<page_id>', methods=['GET'])
    def debug_page_properties(page_id):
        """Debug endpoint to inspect page properties for routing"""
        try:
            notion = _get_notion_client()
            if not notion:
                return jsonify({"error": "Notion client not available"}), 500
                
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
                    core_functions = _get_core_functions()
                    travel_orchestrator = core_functions.get('travel_pipeline_orchestrator')
                    orchestrator_status = bool(travel_orchestrator and hasattr(travel_orchestrator, 'run_travel_packing_pipeline'))
                    logging.info("✅ Travel orchestrator access successful")
                    
                    return jsonify({
                        "status": "success",
                        "workflow_detected": workflow_type,
                        "travel_data": travel_data,
                        "orchestrator_status": "accessible" if orchestrator_status else "not_accessible",
                        "orchestrator_methods": [method for method in dir(travel_orchestrator) if not method.startswith('_')] if travel_orchestrator else []
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
        try:
            # Get monitoring components if available
            system_monitor = _get_monitoring()
            advanced_cache = _get_advanced_cache()
            
            health_data = {
                "status": "healthy",
                "workflows": ["outfit", "travel"],
                "environment_check": check_environment_variables(),
                "components": {
                    "flask": FLASK_AVAILABLE,
                    "notion_client": bool(_get_notion_client()),
                    "core_functions": bool(_get_core_functions()),
                    "monitoring": bool(system_monitor),
                    "cache": bool(advanced_cache)
                }
            }
            
            # Add performance data if monitoring is available
            if system_monitor:
                try:
                    import asyncio
                    performance_dashboard = asyncio.run(system_monitor.get_performance_dashboard())
                    health_data["performance"] = performance_dashboard
                except Exception as e:
                    health_data["performance_error"] = str(e)
            
            # Add cache stats if available
            if advanced_cache:
                try:
                    import asyncio
                    cache_stats = asyncio.run(advanced_cache.get_stats())
                    health_data["cache"] = cache_stats
                except Exception as e:
                    health_data["cache_error"] = str(e)
            
            return jsonify(health_data), 200
            
        except Exception as e:
            logging.error(f"Error generating health check response: {e}")
            return jsonify({
                "status": "unhealthy",
                "error": "Failed to retrieve system metrics"
            }), 500

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
            ],
            "flask_available": FLASK_AVAILABLE,
            "components_loaded": {
                "notion_client": bool(_get_notion_client()),
                "core_functions": len([f for f in _get_core_functions().values() if f is not None]),
                "monitoring": bool(_get_monitoring()),
                "cache": bool(_get_advanced_cache())
            }
        }), 200
    
    initialize_server()


if __name__ == '__main__':
    if app:
        port = int(os.environ.get('PORT', 5000))
        logging.info(f"🌐 Starting server for LOCAL DEVELOPMENT on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        logging.critical("Flask application could not be created. Exiting.")
        sys.exit(1)