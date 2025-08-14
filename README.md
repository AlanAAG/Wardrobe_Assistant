# Wardrobe Assistant

AI-powered wardrobe automation using Notion & weather. This project provides two main functionalities, both triggered by webhooks from Notion:

1.  **AI-Powered Daily Outfit Planner**: Generates a daily outfit based on weather, user preferences, and available wardrobe items.
2.  **AI-Powered Travel Packing Planner**: Generates a comprehensive packing list for trips based on destination, duration, and travel purpose.

## Pipelines

### Daily Outfit Planner

*   **Trigger**: A webhook is sent from Notion when a page in the Outfit Log database has the "Desired Aesthetic" and "Prompt" fields filled out.
*   **Data Fetching**: The pipeline fetches weather data from the OpenWeather API and wardrobe data from Supabase (with a fallback to a local cache and then to Notion).
*   **AI Logic**: An AI chain is used to generate the outfit:
    1.  **Gemini**: Attempts to generate the outfit first.
    2.  **Groq**: If Gemini fails, Groq is used as a fallback.
    3.  **Rule-based Fallback**: If both AI providers fail, a local rule-based logic (`core/outfit_logic.py`) is used.
*   **Output**: The generated outfit is posted back to the Notion page.

### Travel Packing Planner

*   **Trigger**: A webhook is sent from Notion when a page in the Travel Planner database has the "Destinations", "Travel Preferences", and "Travel Dates" fields filled out.
*   **Data Fetching**: The pipeline fetches wardrobe data from Supabase (with a fallback to a local cache and then to Notion).
*   **AI Logic**: An AI chain is used to generate the packing list:
    1.  **Gemini**: Attempts to generate the packing list first.
    2.  **Groq**: If Gemini fails, Groq is used as a fallback.
*   **Output**:
    *   The "Trip-worthy" checkbox is marked as true for all selected clothes in the Notion wardrobe database.
    *   A comprehensive recommendations page is constructed in the Travel Planner database.
    *   3 example outfits are generated and added to the recommendations page.

## Getting Started

### Prerequisites

*   Python 3.11+
*   Pip
*   Git

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd wardrobe-assistant
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Set up the environment variables (see below).

## Environment Variables

Create a `.env` file in the root of the project and add the following environment variables:

```
# Notion
NOTION_TOKEN="your_notion_integration_token"
NOTION_WARDROBE_DB_ID="your_notion_wardrobe_database_id"
NOTION_OUTFIT_LOG_DB_ID="your_notion_outfit_log_database_id"
NOTION_PACKING_GUIDE_ID="your_notion_packing_guide_page_id"

# AI Providers
GEMINI_AI_API_KEY="your_gemini_api_key"
GROQ_AI_API_KEY="your_groq_api_key"

# Other Services
OPENWEATHER_API_KEY="your_openweathermap_api_key"
SUPABASE_URL="your_supabase_url"
SUPABASE_KEY="your_supabase_key"
REDIS_URL="your_redis_url"
```

## Running the Application

The main entry point for the application is `main.py`. This script starts the webhook server, which listens for triggers from Notion.

To run the server:
```bash
python main.py
```

## Running Tests

This project uses `pytest` for testing. To run the test suite:
```bash
pytest
```

## Project Structure
```
.
├── core/                # Core business logic
│   ├── ai_packing_optimizer.py
│   ├── daily_outfit_pipeline.py
│   ├── llm_agents.py
│   ├── notion_result_publisher.py
│   ├── outfit_logic.py
│   ├── outfit_planner_agent.py
│   ├── outfit_pipeline_orchestrator.py
│   ├── travel_logic_fallback.py
│   ├── travel_packing_agent.py
│   └── travel_pipeline_orchestrator.py
├── data/                # Data management (Notion, Supabase, etc.)
├── services/            # Entry points (webhook server)
├── tests/               # Test suite
├── config/              # Configuration files
├── main.py              # Main application entry point
├── README.md
└── requirements.txt
```
