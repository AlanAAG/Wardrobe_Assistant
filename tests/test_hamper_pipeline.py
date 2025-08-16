import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from core.hamper_pipeline_orchestrator import HamperPipelineOrchestrator

@pytest.fixture
def mock_get_checked_items(mocker):
    return mocker.patch('core.hamper_pipeline_orchestrator.get_checked_items_from_page', return_value=[{"id": "item1", "name": "T-shirt"}])

@pytest.fixture
def mock_add_to_db(mocker):
    return mocker.patch('core.hamper_pipeline_orchestrator.add_items_to_dirty_clothes_db')

@pytest.fixture
def mock_uncheck_trigger(mocker):
    return mocker.patch('core.hamper_pipeline_orchestrator.uncheck_hamper_trigger')

@pytest.fixture
def mock_update_washed_status(mocker):
    return mocker.patch('core.hamper_pipeline_orchestrator.update_clothing_washed_status')

@pytest.fixture
def hamper_orchestrator():
    return HamperPipelineOrchestrator()

@pytest.mark.asyncio
async def test_run_hamper_pipeline_success(hamper_orchestrator, mock_get_checked_items, mock_add_to_db, mock_uncheck_trigger):
    """
    Tests that the hamper pipeline runs successfully.
    """
    page_id = "test_page_id"

    result = await hamper_orchestrator.run_hamper_pipeline(page_id)

    assert result["success"] is True
    mock_get_checked_items.assert_called_once_with(page_id)
    mock_add_to_db.assert_called_once_with([{"id": "item1", "name": "T-shirt"}], page_id)
    mock_uncheck_trigger.assert_called_once_with(page_id)

@pytest.mark.asyncio
async def test_run_hamper_pipeline_no_items(hamper_orchestrator, mock_get_checked_items, mock_add_to_db, mock_uncheck_trigger):
    """
    Tests that the hamper pipeline handles the case where there are no items to process.
    """
    page_id = "test_page_id"
    mock_get_checked_items.return_value = []

    result = await hamper_orchestrator.run_hamper_pipeline(page_id)

    assert result["success"] is True
    assert result["message"] == "No items to send to hamper."
    mock_get_checked_items.assert_called_once_with(page_id)
    mock_add_to_db.assert_not_called()
    mock_uncheck_trigger.assert_called_once_with(page_id)

@pytest.mark.asyncio
async def test_run_hamper_pipeline_error(hamper_orchestrator, mock_get_checked_items, mock_add_to_db, mock_uncheck_trigger):
    """
    Tests that the hamper pipeline handles errors gracefully.
    """
    page_id = "test_page_id"
    mock_get_checked_items.side_effect = Exception("Test error")

    result = await hamper_orchestrator.run_hamper_pipeline(page_id)

    assert result["success"] is False
    assert "Test error" in result["error"]
    mock_get_checked_items.assert_called_once_with(page_id)
    mock_add_to_db.assert_not_called()
    mock_uncheck_trigger.assert_called_once_with(page_id)

@pytest.mark.asyncio 
async def test_dirty_clothes_workflow_integration():
    """
    Test the complete dirty clothes workflow integration.
    """
    with patch('data.notion_utils.remove_from_dirty_clothes_and_mark_washed') as mock_remove:
        with patch('services.webhook_server.jsonify') as mock_jsonify:
            mock_remove.return_value = True
            mock_jsonify.return_value = ("mocked_response", 200)
            
            # Test the workflow handler
            from services.webhook_server import handle_dirty_unchecked_workflow
            
            result = handle_dirty_unchecked_workflow("dirty_page_id")
            
            # Should return success response
            assert result[1] == 200  # HTTP status code
            mock_remove.assert_called_once_with("dirty_page_id")
            mock_jsonify.assert_called()

def test_washed_field_updates():
    """
    Test that washed field updates work correctly.
    """
    with patch('data.notion_utils.update_clothing_washed_status') as mock_update:
        with patch('data.notion_utils.get_related_wardrobe_item_id') as mock_get_id:
            with patch('data.notion_utils.delete_page') as mock_delete:
                mock_get_id.return_value = "clothing_item_id"
                
                from data.notion_utils import remove_from_dirty_clothes_and_mark_washed
                
                result = remove_from_dirty_clothes_and_mark_washed("dirty_page_id")
                
                assert result is True
                mock_get_id.assert_called_once_with("dirty_page_id")
                mock_delete.assert_called_once_with("dirty_page_id")
                mock_update.assert_called_once_with("clothing_item_id", "washed")
