import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.notion_result_publisher import NotionResultPublisher

@pytest.fixture
def notion_publisher(mocker):
    """Fixture to create a NotionResultPublisher with mocked dependencies."""
    mocker.patch('core.notion_result_publisher.notion', MagicMock())
    mocker.patch('core.notion_result_publisher.wardrobe_data_manager', MagicMock())
    mock_agent = mocker.patch('core.notion_result_publisher.outfit_planner_agent', MagicMock())
    mock_agent.generate_example_outfits = AsyncMock(return_value="outfit text")
    return NotionResultPublisher()

@pytest.mark.asyncio
async def test_finalize_packing_results_success(notion_publisher, mocker):
    """
    Tests that the finalization process runs successfully.
    """
    page_id = "test_page_id"
    packing_result = {
        "selected_items": [{"id": "item1"}, {"id": "item2"}],
        "total_items": 2,
        "total_weight_kg": 5.5,
    }
    trip_config = {}

    with patch('core.notion_result_publisher.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        result = await notion_publisher.finalize_packing_results(
            page_id, packing_result, "gemini", trip_config
        )

        assert result["success"] is True
        mock_to_thread.assert_called()

@pytest.mark.asyncio
async def test_finalize_packing_results_failure(notion_publisher, mocker):
    """
    Tests that the finalization process handles errors gracefully.
    """
    page_id = "test_page_id"
    packing_result = {
        "selected_items": [{"id": "item1"}],
        "total_items": 1,
        "total_weight_kg": 2.0,
    }
    trip_config = {}

    with patch('core.notion_result_publisher.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.side_effect = Exception("Notion API failed")
        result = await notion_publisher.finalize_packing_results(
            page_id, packing_result, "gemini", trip_config
        )

        assert result["success"] is False
        assert "Failed to finalize outfit" in result["error"]
