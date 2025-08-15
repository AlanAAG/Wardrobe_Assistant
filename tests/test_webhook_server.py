import pytest
from unittest.mock import MagicMock, patch
from services.webhook_server import determine_workflow_type

# Mock Notion page objects
def get_mock_page(parent_db_id, properties):
    return {
        "parent": {"database_id": parent_db_id},
        "properties": properties
    }

@pytest.fixture
def mock_notion_client():
    with patch('services.webhook_server._get_notion_client') as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_env_vars():
    with patch('os.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda key, default="": {
            "NOTION_OUTFIT_LOG_DB_ID": "outfit_log_db_id",
            "NOTION_DIRTY_CLOTHES_DB_ID": "dirty_clothes_db_id"
        }.get(key, default)
        yield mock_getenv

# --- Test Cases ---

def test_determine_workflow_hamper_success(mock_notion_client, mock_env_vars):
    """
    Tests that the hamper workflow is triggered correctly.
    """
    page_id = "test_page"
    properties = {
        "Send to Hamper": {"type": "checkbox", "checkbox": True}
    }
    mock_page = get_mock_page("outfit_log_db_id", properties)
    mock_notion_client.pages.retrieve.return_value = mock_page

    workflow = determine_workflow_type(page_id)
    assert workflow == "hamper"

def test_determine_workflow_hamper_wrong_db(mock_notion_client, mock_env_vars):
    """
    Tests that the hamper workflow is NOT triggered for the wrong database.
    """
    page_id = "test_page"
    properties = {
        "Send to Hamper": {"type": "checkbox", "checkbox": True}
    }
    mock_page = get_mock_page("wrong_db_id", properties)
    mock_notion_client.pages.retrieve.return_value = mock_page

    workflow = determine_workflow_type(page_id)
    assert workflow is None

def test_determine_workflow_hamper_not_checked(mock_notion_client, mock_env_vars):
    """
    Tests that the hamper workflow is NOT triggered if the checkbox is not checked.
    """
    page_id = "test_page"
    properties = {
        "Send to Hamper": {"type": "checkbox", "checkbox": False}
    }
    mock_page = get_mock_page("outfit_log_db_id", properties)
    mock_notion_client.pages.retrieve.return_value = mock_page

    workflow = determine_workflow_type(page_id)
    assert workflow is None

def test_determine_workflow_laundry_day(mock_notion_client, mock_env_vars):
    """
    Tests that the laundry day workflow is still triggered correctly.
    """
    page_id = "test_page"
    properties = {
        "Washed": {"type": "checkbox", "checkbox": True}
    }
    mock_page = get_mock_page("dirty_clothes_db_id", properties)
    mock_notion_client.pages.retrieve.return_value = mock_page

    workflow = determine_workflow_type(page_id)
    assert workflow == "laundry_day"

def test_determine_workflow_no_trigger(mock_notion_client, mock_env_vars):
    """
    Tests that no workflow is triggered when no conditions are met.
    """
    page_id = "test_page"
    properties = {}
    mock_page = get_mock_page("some_other_db", properties)
    mock_notion_client.pages.retrieve.return_value = mock_page

    workflow = determine_workflow_type(page_id)
    assert workflow is None
