import logging
import os
import sys
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv

try:
    from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.error("FastAPI not available - install with: pip install fastapi uvicorn")

load_dotenv()

if FASTAPI_AVAILABLE:
    app = FastAPI(title="Unified AI Wardrobe Assistant")
else:
    app = None

# Enhanced logging configuration using standard logging since system_monitor returns structlog config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("webhook_server")

def check_environment_variables():
    """Check all required environment variables"""
    required_vars = {
        'NOTION_TOKEN': os.getenv('NOTION_TOKEN'),
        'NOTION_PACKING_GUIDE_ID': os.getenv('NOTION_PACKING_GUIDE_ID'),
        'NOTION_WARDROBE_DB_ID': os.getenv('NOTION_WARDROBE_DB_ID'),
        'NOTION_OUTFIT_LOG_DB_ID': os.getenv('NOTION_OUTFIT_LOG_DB_ID'),
        'GEMINI_AI_API_KEY': os.getenv('GEMINI_AI_API_KEY'),
        'GROQ_AI_API_KEY': os.getenv('GROQ_AI_API_KEY'),
        'NOTION_BOT_ID': os.getenv('NOTION_BOT_ID')
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        return False
    
    logger.info("All environment variables present")
    return True

def _get_notion_client():
    from data.notion_utils import notion
    return notion

def _get_webhook_cache():
    from caching.webhook_cache import webhook_cache
    return webhook_cache

def _get_core_functions():
    functions = {}
    try:
        from core.outfit_pipeline_orchestrator import outfit_pipeline_orchestrator
        functions['outfit_pipeline_orchestrator'] = outfit_pipeline_orchestrator
    except ImportError as e:
        logger.error(f"Outfit pipeline orchestrator import failed: {e}")
        functions['outfit_pipeline_orchestrator'] = None
    
    try:
        from core.travel_pipeline_orchestrator import travel_pipeline_orchestrator
        functions['travel_pipeline_orchestrator'] = travel_pipeline_orchestrator
    except ImportError as e:
        logger.error(f"Travel pipeline orchestrator import failed: {e}")
        functions['travel_pipeline_orchestrator'] = None

    try:
        from core.hamper_pipeline_orchestrator import hamper_pipeline_orchestrator
        functions['hamper_pipeline_orchestrator'] = hamper_pipeline_orchestrator
    except ImportError as e:
        logger.error(f"Hamper pipeline orchestrator import failed: {e}")
        functions['hamper_pipeline_orchestrator'] = None

    try:
        from core.laundry_day_pipeline_orchestrator import laundry_day_pipeline_orchestrator
        functions['laundry_day_pipeline_orchestrator'] = laundry_day_pipeline_orchestrator
    except ImportError as e:
        logger.error(f"Laundry day pipeline orchestrator import failed: {e}")
        functions['laundry_day_pipeline_orchestrator'] = None
    
    return functions

def initialize_server():
    logger.info("Starting AI Wardrobe Assistant...")
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI not available. Server cannot start.")
        return False
    if not check_environment_variables():
        logger.error("Environment check failed.")
        return False
    return True

if FASTAPI_AVAILABLE:
    @app.post("/webhook/notion")
    async def handle_unified_notion_webhook(request: Request, background_tasks: BackgroundTasks):
        webhook_data = await request.json()
        
        if not webhook_data:
            return JSONResponse({"error": "No JSON data received"}, status_code=400)
        
        logger.info(f"Received unified webhook", webhook_data=webhook_data)
        
        if "challenge" in webhook_data:
            return {"challenge": webhook_data["challenge"]}
        
        if "verification_token" in webhook_data:
            return {"message": "Verification token received"}
        
        entity = webhook_data.get("entity", {})
        page_id = entity.get("id")
        
        if not page_id:
            return JSONResponse({"error": "Missing entity.id in webhook"}, status_code=400)
        
        webhook_cache = _get_webhook_cache()
        if await webhook_cache.is_recently_processed(page_id):
            return {"message": "Event recently processed, ignoring"}

        notion = _get_notion_client()
        if not notion:
            return JSONResponse({"error": "Notion client not available"}, status_code=500)

        try:
            # We are calling sync retrieve in async execution temporarily or via sync wrapper, 
            # ideally notion-client-async exists but for now it's okay unless it blocks too hard.
            # Notion-client by default is sync.
            page = await asyncio.to_thread(notion.pages.retrieve, page_id=page_id)
            last_edited_by_id = page.get("last_edited_by", {}).get("id")

            notion_bot_id = os.getenv("NOTION_BOT_ID")
            if last_edited_by_id and last_edited_by_id == notion_bot_id:
                return {"message": "Event from bot, ignoring"}

        except Exception as e:
            logger.error(f"Failed to validate page for {page_id}", error=str(e))
            return JSONResponse({"error": "Failed to validate page"}, status_code=500)
        
        await webhook_cache.add(page_id)

        workflow_type = await asyncio.to_thread(determine_workflow_type, page_id, page)
        logger.info(f"Detected workflow type: {workflow_type}")
        
        if workflow_type == "outfit":
            background_tasks.add_task(handle_outfit_workflow, page_id)
            return {"message": "Outfit workflow triggered", "workflow": "outfit"}
        elif workflow_type == "travel":
            background_tasks.add_task(handle_travel_workflow, page_id)
            return {"message": "Travel workflow triggered", "workflow": "travel"}
        elif workflow_type == "laundry_day":
            background_tasks.add_task(handle_laundry_day_workflow, page_id)
            return {"message": "Laundry day workflow triggered", "workflow": "laundry_day"}
        elif workflow_type == "hamper":
            background_tasks.add_task(handle_hamper_workflow, page_id)
            return {"message": "Hamper workflow triggered", "workflow": "hamper"}
        elif workflow_type == "dirty_unchecked":
            background_tasks.add_task(handle_dirty_unchecked_workflow, page_id)
            return {"message": "Dirty unchecked workflow triggered", "workflow": "dirty_unchecked"}
        else:
            return {"message": "No workflow conditions met"}

def determine_workflow_type(page_id, page=None):
    notion = _get_notion_client()
    if not notion:
        return None

    try:
        if not page:
            page = notion.pages.retrieve(page_id=page_id)

        props = page.get("properties", {})
        parent_db_id = page.get("parent", {}).get("database_id", "").replace("-", "")

        dirty_clothes_db_id = os.getenv("NOTION_DIRTY_CLOTHES_DB_ID", "").replace("-", "")
        if parent_db_id and parent_db_id == dirty_clothes_db_id:
            dirty_prop = props.get("Dirty", {})
            if dirty_prop.get("type") == "checkbox" and not dirty_prop.get("checkbox"):
                return "dirty_unchecked"
            washed_prop = props.get("Washed", {})
            if washed_prop.get("type") == "checkbox" and washed_prop.get("checkbox"):
                return "laundry_day"

        for name in ["Generate", "Generate Travel Packing", "Generate Packing", "Travel Generate"]:
            if name in props and props[name].get("type") == "checkbox" and props[name].get("checkbox"):
                status_prop = props.get("Status", {})
                current_status = status_prop.get("select", {}).get("name") if isinstance(status_prop.get("select"), dict) else None
                if current_status == "In Progress":
                    return None
                return "travel"

        dest_ok = _prop_nonempty(props, ["Destinations", "Locations", "Cities"])
        prefs_ok = _prop_nonempty(props, ["Travel Preferences", "Trip Preferences", "Preferences"])
        dates_ok = _date_present(props, ["Travel Dates", "Trip Dates", "Dates"])
        if dest_ok and prefs_ok and dates_ok:
            status_prop = props.get("Status", {})
            current_status = status_prop.get("select", {}).get("name")
            if current_status in ["In Progress", "Complete"]:
                return None
            return "travel"

        aesthetic_prop = props.get("Desired Aesthetic", {})
        prompt_prop = props.get("Prompt", {})
        has_aesthetic = len(aesthetic_prop.get("multi_select", [])) > 0
        has_prompt = len(prompt_prop.get("rich_text", [])) > 0 and any(
            t.get("plain_text", "").strip() for t in prompt_prop.get("rich_text", [])
        )
        if has_aesthetic and has_prompt:
            status_prop = props.get("Status", {})
            current_status = status_prop.get("select", {}).get("name")
            if current_status in ["In Progress", "Complete"]:
                return None
            return "outfit"

        outfit_log_db_id = os.getenv("NOTION_OUTFIT_LOG_DB_ID", "").replace("-", "")
        hamper_prop = props.get("Send to Hamper", {})
        if parent_db_id and parent_db_id == outfit_log_db_id:
            prop_checked = (hamper_prop.get("type") == "checkbox" and hamper_prop.get("checkbox"))
            has_checked_block = False
            try:
                from data.notion_utils import has_checked_hamper_todo_block
                has_checked_block = has_checked_hamper_todo_block(page_id)
            except Exception:
                pass
            if prop_checked or has_checked_block:
                status_prop = props.get("Status", {})
                current_status = status_prop.get("select", {}).get("name")
                if current_status in ["In Progress", "Complete"]:
                    return None
                return "hamper"

        return None
    except Exception as e:
        logger.error(f"Error determining workflow type: {e}")
        return None

def _prop_nonempty(props, candidates):
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
    for name in candidates:
        p = props.get(name)
        if p and p.get("type") == "date":
            d = p.get("date") or {}
            if (d.get("start") or "").strip():
                return True
    return False

async def handle_outfit_workflow(page_id):
    try:
        core_functions = _get_core_functions()
        outfit_orchestrator = core_functions.get('outfit_pipeline_orchestrator')
        if outfit_orchestrator:
            await outfit_orchestrator.run_daily_outfit_pipeline(page_id)
    except Exception as e:
        logger.error(f"Outfit workflow logic failed: {e}")

async def handle_travel_workflow(page_id):
    try:
        travel_trigger_data = await asyncio.to_thread(get_travel_trigger_data, page_id)
        if travel_trigger_data:
            core_functions = _get_core_functions()
            travel_orchestrator = core_functions.get('travel_pipeline_orchestrator')
            if travel_orchestrator:
                await travel_orchestrator.run_travel_packing_pipeline(travel_trigger_data)
    except Exception as e:
        logger.error(f"Travel workflow logic failed: {e}")

async def handle_laundry_day_workflow(page_id):
    try:
        core_functions = _get_core_functions()
        laundry_day_orchestrator = core_functions.get('laundry_day_pipeline_orchestrator')
        if laundry_day_orchestrator:
            await laundry_day_orchestrator.run_laundry_day_pipeline(page_id)
    except Exception as e:
        logger.error(f"Laundry day workflow logic failed: {e}")

async def handle_hamper_workflow(page_id):
    try:
        core_functions = _get_core_functions()
        hamper_orchestrator = core_functions.get('hamper_pipeline_orchestrator')
        if hamper_orchestrator:
            await hamper_orchestrator.run_hamper_pipeline(page_id)
    except Exception as e:
        logger.error(f"Hamper workflow logic failed: {e}")

async def handle_dirty_unchecked_workflow(page_id):
    try:
        from data.notion_utils import remove_from_dirty_clothes_and_mark_washed
        await asyncio.to_thread(remove_from_dirty_clothes_and_mark_washed, page_id)
    except Exception as e:
        logger.error(f"Dirty unchecked workflow logic failed: {e}")

def get_travel_trigger_data(page_id):
    notion = _get_notion_client()
    if not notion:
        return None
    try:
        page = notion.pages.retrieve(page_id=page_id)
        props = page.get("properties", {})
        return {
            "page_id": page_id,
            "destinations": _read_destinations(props, page_id),
            "preferences": _read_preferences(props),
            "dates": _read_dates(props),
            "bags": _read_bags(props)
        }
    except Exception as e:
        logger.error(f"Error extracting travel trigger data: {e}")
        return None

def _read_destinations(props, page_id):
    for name in ["Destinations", "Locations", "Cities"]:
        p = props.get(name)
        if p and p.get("type") == "multi_select":
            return [tag.get("name", "") for tag in p.get("multi_select", [])]
    return []

def _read_preferences(props):
    for name in ["Travel Preferences", "Trip Preferences", "Preferences"]:
        p = props.get(name)
        if p and p.get("type") == "rich_text":
            return "".join(t.get("plain_text", "") for t in p.get("rich_text", [])).strip()
    return ""

def _read_dates(props):
    for name in ["Travel Dates", "Trip Dates", "Dates"]:
        p = props.get(name)
        if p and p.get("type") == "date":
            d = p.get("date") or {}
            start = (d.get("start") or "").strip()
            end = (d.get("end") or "").strip() or start
            try:
                ds = datetime.fromisoformat(start[:19])
                de = datetime.fromisoformat(end[:19])
                days = max((de - ds).days + 1, 1)
            except Exception:
                days = 0
            return {"start": start, "end": end, "days": days}
    return {"start": "", "end": "", "days": 0}

def _read_bags(props):
    for name in ["Bags & Weight Limits", "Luggage"]:
        p = props.get(name)
        if p and p.get("type") == "multi_select":
            return [tag.get("name", "") for tag in p.get("multi_select", [])]
    return []

if FASTAPI_AVAILABLE:
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "environment_check": check_environment_variables(),
            "fastapi": FASTAPI_AVAILABLE,
        }

    @app.get("/")
    async def root():
        return {
            "message": "Unified AI Wardrobe Assistant API",
            "workflows": ["outfit_generation", "travel_packing"]
        }

initialize_server()