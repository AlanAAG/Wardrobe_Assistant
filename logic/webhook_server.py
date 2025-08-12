from flask import Flask, request, jsonify
import logging
import os
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from logic.notion_utils import notion
from logic.pipeline_orchestrator import run_enhanced_outfit_pipeline
from logic.travel_pipeline_orchestrator import travel_pipeline_orchestrator

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create thread pool for async operations
executor = ThreadPoolExecutor(max_workers=5)  # Increased for both workflows

@app.route('/webhook/notion', methods=['POST'])
def handle_unified_notion_webhook():
    """
    UNIFIED webhook handler for both outfit generation and travel packing.
    Routes to appropriate pipeline based on page properties.
    """
    try:
        # Get webhook data
        webhook_data = request.get_json()
        
        if not webhook_data:
            logging.warning("Received webhook with no JSON data")
            return jsonify({"error": "No JSON data received"}), 400
        
        logging.info(f"Received unified webhook: {webhook_data}")
        
        # Handle Notion webhook verification (both challenge and verification_token)
        if "challenge" in webhook_data:
            challenge = webhook_data["challenge"]
            logging.info(f"Responding to Notion verification challenge: {challenge}")
            return jsonify({"challenge": challenge}), 200
        
        if "verification_token" in webhook_data:
            verification_token = webhook_data["verification_token"]
            logging.info(f"Responding to Notion verification token: {verification_token}")
            return jsonify({"message": "Verification token received"}), 200
        
        # Extract page info from webhook
        entity = webhook_data.get("entity", {})
        page_id = entity.get("id")
        
        if not page_id:
            logging.warning("Webhook missing entity.id (page_id)")
            return jsonify({"error": "Missing entity.id in webhook"}), 400
        
        # 🚀 INTELLIGENT ROUTING: Determine which workflow to trigger
        workflow_type = determine_workflow_type(page_id)
        
        if workflow_type == "outfit":
            return handle_outfit_workflow(page_id)
        elif workflow_type == "travel":
            return handle_travel_workflow(page_id)
        else:
            logging.info(f"No workflow triggered for page {page_id}")
            return jsonify({"message": "No workflow conditions met"}), 200
        
    except Exception as e:
        logging.error(f"Unified webhook error: {e}")
        return jsonify({"error": "Internal server error"}), 500

def determine_workflow_type(page_id):
    """
    🎯 SMART ROUTING: Analyze page properties to determine workflow type
    
    Returns:
        - "outfit": if outfit generation should be triggered
        - "travel": if travel packing should be triggered  
        - None: if no workflow should be triggered
    """
    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        
        # Check for TRAVEL PACKING triggers
        # Looking for "Generate" checkbox in travel planner
        travel_generate = props.get("Generate", {}).get("checkbox", False)
        
        if travel_generate:
            logging.info(f"🧳 Travel packing workflow detected for page {page_id}")
            return "travel"
        
        # Check for OUTFIT GENERATION triggers
        # Looking for both "Desired Aesthetic" and "Prompt" fields (existing outfit logic)
        aesthetic_prop = props.get("Desired Aesthetic", {})
        prompt_prop = props.get("Prompt", {})
        
        has_aesthetic = len(aesthetic_prop.get("multi_select", [])) > 0
        has_prompt = len(prompt_prop.get("rich_text", [])) > 0 and any(
            t.get("plain_text", "").strip() for t in prompt_prop.get("rich_text", [])
        )
        
        if has_aesthetic and has_prompt:
            logging.info(f"👕 Outfit generation workflow detected for page {page_id}")
            return "outfit"
        
        # No workflow triggers detected
        logging.info(f"No workflow triggers found for page {page_id}")
        return None
        
    except Exception as e:
        logging.error(f"Error determining workflow type: {e}")
        return None

def handle_outfit_workflow(page_id):
    """Handle outfit generation workflow (existing logic)"""
    try:
        # Validate outfit trigger conditions
        if not validate_outfit_trigger_conditions(page_id):
            logging.info("Outfit trigger conditions not met")
            return jsonify({"message": "Outfit trigger conditions not met"}), 200
        
        # Get outfit trigger data
        trigger_data = get_outfit_trigger_data(page_id)
        if not trigger_data:
            return jsonify({"error": "Failed to extract outfit trigger data"}), 400
        
        logging.info(f"Outfit trigger: aesthetic={trigger_data['aesthetics']}, prompt='{trigger_data['prompt']}'")
        
        # Start async processing
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
        logging.error(f"Outfit workflow error: {e}")
        return jsonify({"error": "Outfit workflow failed"}), 500

def handle_travel_workflow(page_id):
    """Handle travel packing workflow (new logic)"""
    try:
        # Get travel trigger data
        travel_trigger_data = get_travel_trigger_data(page_id)
        if not travel_trigger_data:
            return jsonify({"error": "Failed to extract travel trigger data"}), 400
        
        logging.info(f"Travel trigger: preferences='{travel_trigger_data.get('preferences', '')}'")
        
        # Start async processing
        future = executor.submit(run_async_travel_pipeline, travel_trigger_data)
        
        return jsonify({
            "message": "Travel packing started", 
            "page_id": page_id,
            "workflow": "travel",
            "status": "processing",
            "trip_type": "business_school_relocation",
            "destinations": ["Dubai (Sep-Dec)", "Gurgaon (Jan-May)"]
        }), 200
        
    except Exception as e:
        logging.error(f"Travel workflow error: {e}")
        return jsonify({"error": "Travel workflow failed"}), 500

def validate_outfit_trigger_conditions(page_id):
    """Validate outfit generation trigger conditions (existing logic)"""
    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        
        # Check Desired Aesthetic (multi-select)
        aesthetic_prop = props.get("Desired Aesthetic", {})
        multi_select = aesthetic_prop.get("multi_select", [])
        has_aesthetic = len(multi_select) > 0
        
        # Check Prompt (rich text)
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
        
        # Get Desired Aesthetic
        aesthetic_prop = props.get("Desired Aesthetic", {})
        multi_select = aesthetic_prop.get("multi_select", [])
        aesthetics = [tag.get("name") for tag in multi_select]
        
        # Get Prompt text
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
    """
    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        
        # Extract Travel Preferences (text field)
        preferences_prop = props.get("Travel Preferences", {})
        rich_text = preferences_prop.get("rich_text", [])
        preferences_text = "".join([t.get("plain_text", "") for t in rich_text]) if rich_text else ""
        
        # Extract Travel Dates (date field) 
        dates_prop = props.get("Travel Dates", {})
        date_range = dates_prop.get("date", {})
        start_date = date_range.get("start") if date_range else "2024-09-01"
        end_date = date_range.get("end") if date_range else "2025-05-31"
        
        # 🎯 FIXED DESTINATIONS: Dubai + Gurgaon (as per your use case)
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
        
        return {
            "page_id": page_id,
            "destinations": destinations_config,
            "preferences": {
                "optimization_goals": ["weight_efficiency", "business_readiness", "climate_coverage", "cultural_compliance"],
                "trip_type": "business_school_relocation",
                "user_notes": preferences_text,
                "packing_style": "minimalist_professional"
            }
        }
        
    except Exception as e:
        logging.error(f"Error extracting travel trigger data: {e}")
        return None

def run_async_outfit_pipeline(trigger_data):
    """Run outfit pipeline in background thread (existing logic)"""
    try:
        logging.info("Starting async outfit pipeline...")
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async pipeline
            result = loop.run_until_complete(run_enhanced_outfit_pipeline(trigger_data))
            
            # Log the result
            if result["success"]:
                method = result["generation_method"]
                items = result["outfit_items"]
                logging.info(f"✅ Outfit pipeline completed successfully!")
                logging.info(f"   Method: {method}")
                logging.info(f"   Items: {items}")
            else:
                logging.error(f"❌ Outfit pipeline failed: {result['error']}")
                
        finally:
            loop.close()
            
    except Exception as e:
        logging.error(f"Error in async outfit pipeline: {e}")

def run_async_travel_pipeline(trigger_data):
    """Run travel packing pipeline in background thread (NEW)"""
    try:
        logging.info("Starting async travel packing pipeline...")
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the travel packing pipeline
            result = loop.run_until_complete(
                travel_pipeline_orchestrator.run_travel_packing_pipeline(trigger_data)
            )
            
            # Log results
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
        logging.error(f"Error in travel packing pipeline: {e}")

@app.route('/debug/page/<page_id>', methods=['GET'])
def debug_page_properties(page_id):
    """Debug endpoint to inspect page properties for routing"""
    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        
        # Analyze properties for debugging
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
        
        # Determine what workflow would be triggered
        workflow = determine_workflow_type(page_id)
        
        return jsonify({
            "page_id": page_id,
            "properties": property_analysis,
            "detected_workflow": workflow,
            "page_title": page.get("properties", {}).get("title", {})
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({"status": "healthy", "workflows": ["outfit", "travel"]}), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        "message": "Unified AI Wardrobe Assistant", 
        "workflows": ["outfit_generation", "travel_packing"]
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)