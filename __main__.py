#!/usr/bin/env python3
"""
Safe startup script for AI Wardrobe Assistant.
Handles import order and initialization properly.
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('wardrobe_assistant.log')
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        ('flask', 'flask'),
        ('python-dotenv', 'dotenv'),
        ('requests', 'requests'),
        ('notion-client', 'notion_client'),
        ('google-generativeai', 'google.generativeai'),
        ('groq', 'groq'),
        ('supabase', 'supabase'),
        ('structlog', 'structlog'),
        ('aiofiles', 'aiofiles'),
        ('psutil', 'psutil'),
        ('aioredis', 'aioredis')
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            logger.debug(f"✅ {package_name} available")
        except ImportError:
            missing.append(package_name)
            logger.warning(f"❌ {package_name} missing")
    
    if missing:
        logger.error(f"Missing required packages: {missing}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
    
    logger.info("✅ All dependencies available")
    return True

def check_environment():
    """Check if required environment variables are set."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = [
            'NOTION_TOKEN',
            'NOTION_PACKING_GUIDE_ID',
            'NOTION_WARDROBE_DB_ID'
        ]
        
        optional_vars = [
            'GEMINI_AI_API_KEY',
            'GROQ_AI_API_KEY',
            'OPENWEATHER_API_KEY',
            'DEFAULT_CITY'
        ]
        
        missing = []
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            logger.error(f"❌ Missing required environment variables: {missing}")
            logger.error("Please set these in your .env file")
            return False
        
        # Log optional vars status
        for var in optional_vars:
            if os.getenv(var):
                logger.info(f"✅ {var} configured")
            else:
                logger.warning(f"⚠️ {var} not set (optional, some features may be limited)")
        
        logger.info("✅ Environment configuration valid")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment check failed: {e}")
        return False

def start_server():
    """Start the Flask server safely."""
    try:
        logger.info("Initializing Flask server with lazy imports...")
        
        # Import the webhook server with lazy loading
        from services.webhook_server import app, initialize_server
        
        # Initialize server components
        logger.info("Running server initialization checks...")
        if not initialize_server():
            logger.error("❌ Server initialization failed")
            return False
        
        if not app:
            logger.error("❌ Flask app not available")
            return False
        
        # Get port configuration
        port = int(os.environ.get('PORT', 5000))
        debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"🌐 Starting AI Wardrobe Assistant server...")
        logger.info(f"   Host: 0.0.0.0")
        logger.info(f"   Port: {port}")
        logger.info(f"   Debug: {debug_mode}")
        logger.info(f"   URL: http://localhost:{port}")
        
        # Start the server
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug_mode
        )
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure all modules are properly installed and __init__.py files use lazy loading")
        return False
    except Exception as e:
        logger.error(f"❌ Server startup error: {e}")
        return False

def show_help():
    """Show usage help."""
    print("""
AI Wardrobe Assistant - Usage

Starting the server:
    python -m your_project_name
    # or
    python __main__.py

Environment setup:
    1. Copy .env.example to .env
    2. Fill in your API keys and Notion IDs
    3. Install dependencies: pip install -r requirements.txt

Available endpoints:
    GET  /                     - Root endpoint with system info
    POST /webhook/notion       - Notion webhook handler
    GET  /health              - System health check
    GET  /debug/page/<id>     - Debug page properties

For more information, visit: https://github.com/your-repo
""")

def main():
    """Main entry point."""
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h', 'help']:
            show_help()
            return
        elif sys.argv[1] in ['--version', '-v']:
            print("AI Wardrobe Assistant v2.0.0")
            return
    
    logger.info("🚀 Starting AI Wardrobe Assistant...")
    
    try:
        # Step 1: Check dependencies
        logger.info("Step 1: Checking dependencies...")
        if not check_dependencies():
            logger.error("❌ Dependency check failed")
            sys.exit(1)
        
        # Step 2: Check environment
        logger.info("Step 2: Checking environment configuration...")
        if not check_environment():
            logger.error("❌ Environment check failed")
            sys.exit(1)
        
        # Step 3: Start server
        logger.info("Step 3: Starting server...")
        if not start_server():
            logger.error("❌ Server startup failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("👋 Received shutdown signal, stopping server...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()