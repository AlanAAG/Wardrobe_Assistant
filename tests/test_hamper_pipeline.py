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
