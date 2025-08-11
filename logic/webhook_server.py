from flask import Flask, request, jsonify
import logging
import os
from dotenv import load_dotenv
from logic.notion_utils import notion
from logic.pipeline_orchestrator import run_outfit_pipeline

load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.route('/webhook/notion', methods=['POST'])
def handle_notion_webhook():
    """
    Handles incoming Notion webhooks and triggers outfit generation pipeline.
    Also handles Notion's webhook verification process.
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
            # Just return success - the token was already validated during webhook setup
            return jsonify({"message": "Verification token received"}), 200
        
        # Extract page info from webhook (actual page update)
        # For page.properties_updated events, page ID is in entity.id
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
        
        # Run the outfit pipeline (will create this function next)
        # result = run_outfit_pipeline(trigger_data)
        
        # For now, just return success
        return jsonify({
            "message": "Webhook received successfully",
            "page_id": page_id,
            "trigger_data": trigger_data
        }), 200
        
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return jsonify({"error": "Internal server error"}), 500

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