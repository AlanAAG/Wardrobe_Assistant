from flask import Flask, request, jsonify
import logging
import os
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from logic.notion_utils import notion
from logic.pipeline_orchestrator import run_enhanced_outfit_pipeline

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create thread pool for async operations
executor = ThreadPoolExecutor(max_workers=3)

@app.route('/webhook/notion', methods=['POST'])
def handle_notion_webhook():
    """
    Handles incoming Notion webhooks and triggers outfit generation pipeline.
    Responds quickly to webhook, processes outfit generation in background.
    """
    try:
        # Get webhook data
        webhook_data = request.get_json()
        
        if not webhook_data:
            logging.warning("Received webhook with no JSON data")
            return jsonify({"error": "No JSON data received"}), 400
        
        logging.info(f"Received webhook: {webhook_data}")
        
        # Handle Notion webhook verification (both challenge and verification_token)
        if "challenge" in webhook_data:
            challenge = webhook_data["challenge"]
            logging.info(f"Responding to Notion verification challenge: {challenge}")
            return jsonify({"challenge": challenge}), 200
        
        # Handle verification_token (ongoing verification checks)
        if "verification_token" in webhook_data:
            verification_token = webhook_data["verification_token"]
            logging.info(f"Responding to Notion verification token: {verification_token}")
            return jsonify({"message": "Verification token received"}), 200
        
        # Extract page info from webhook (actual page update)
        entity = webhook_data.get("entity", {})
        page_id = entity.get("id")
        
        if not page_id:
            logging.warning("Webhook missing entity.id (page_id)")
            return jsonify({"error": "Missing entity.id in webhook"}), 400
        
        # Validate trigger conditions (both fields not empty)
        if not validate_trigger_conditions(page_id):
            logging.info("Trigger conditions not met - ignoring webhook")
            return jsonify({"message": "Trigger conditions not met"}), 200
        
        # Get the trigger values
        trigger_data = get_trigger_data(page_id)
        if not trigger_data:
            logging.warning("Could not extract trigger data")
            return jsonify({"error": "Failed to extract trigger data"}), 400
        
        logging.info(f"Trigger data: aesthetic={trigger_data['aesthetics']}, prompt='{trigger_data['prompt']}'")
        
        # **KEY CHANGE: Start async processing in background thread**
        # Respond immediately to webhook, process outfit generation asynchronously
        future = executor.submit(run_async_outfit_pipeline, trigger_data)
        
        # Quick response to Notion webhook (prevents timeout)
        return jsonify({
            "message": "Webhook received - outfit generation started",
            "page_id": page_id,
            "status": "processing",
            "aesthetic": trigger_data['aesthetics'],
            "prompt_preview": trigger_data['prompt'][:50] + "..." if len(trigger_data['prompt']) > 50 else trigger_data['prompt']
        }), 200
        
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return jsonify({"error": "Internal server error"}), 500

def run_async_outfit_pipeline(trigger_data):
    """
    Wrapper to run async pipeline in a separate thread.
    This prevents webhook timeouts while allowing async processing.
    """
    try:
        logging.info("Starting async outfit pipeline in background...")
        
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
                logging.info(f"✅ Async pipeline completed successfully!")
                logging.info(f"   Method: {method}")
                logging.info(f"   Items: {items}")
                logging.info(f"   Page: {result['page_id']}")
            else:
                logging.error(f"❌ Async pipeline failed: {result['error']}")
                
        finally:
            loop.close()
            
    except Exception as e:
        logging.error(f"Error in async outfit pipeline: {e}")

def validate_trigger_conditions(page_id):
    """
    Validates that both Desired Aesthetic and Prompt fields are not empty.
    Returns True if both conditions are met.
    """
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
        
        logging.info(f"Trigger validation: aesthetic={has_aesthetic}, prompt={has_prompt}")
        return has_aesthetic and has_prompt
        
    except Exception as e:
        logging.error(f"Error validating trigger conditions: {e}")
        return False

def get_trigger_data(page_id):
    """
    Extracts the trigger data (aesthetics and prompt) from the Notion page.
    Returns dict with 'aesthetics' and 'prompt' or None if failed.
    """
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
        logging.error(f"Error extracting trigger data: {e}")
        return None

@app.route('/debug/env', methods=['GET'])
def debug_webhook_environment():
    """Debug endpoint to check environment in webhook context"""
    try:
        from logic.supabase_client import supabase_client
        
        # Check environment
        supabase_url = os.getenv("SUPABASE_PROJECT_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        debug_info = {
            "supabase_url": supabase_url,
            "key_preview": f"{supabase_key[:20]}...{supabase_key[-8:]}" if supabase_key else "None",
            "key_length": len(supabase_key) if supabase_key else 0,
            "key_type": "service_role" if supabase_key and supabase_key.startswith('eyJ') else "anon_or_other",
            "client_connected": supabase_client.is_connected(),
            "cwd": os.getcwd(),
            "env_file_exists": os.path.exists('.env')
        }
        
        # Test Supabase connection
        if supabase_client.is_connected():
            try:
                response = supabase_client.client.table('wardrobe_items').select('id').limit(1).execute()
                debug_info["supabase_test"] = {
                    "success": True,
                    "data_length": len(response.data),
                    "error": getattr(response, 'error', None),
                    "raw_response": str(response)[:200]  # First 200 chars
                }
            except Exception as e:
                debug_info["supabase_test"] = {
                    "success": False,
                    "error": str(e),
                    "error_type": str(type(e))
                }
        
        return jsonify(debug_info), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({"status": "healthy"}), 200

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({"message": "Outfit Generator Webhook Server"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)