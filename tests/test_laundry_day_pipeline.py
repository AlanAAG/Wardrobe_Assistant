import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from core.laundry_day_pipeline_orchestrator import LaundryDayPipelineOrchestrator

@pytest.fixture
def mock_get_related_item(mocker):
    return mocker.patch('core.laundry_day_pipeline_orchestrator.get_related_wardrobe_item_id', return_value="wardrobe_item_id")

@pytest.fixture
def mock_update_status(mocker):
    return mocker.patch('core.laundry_day_pipeline_orchestrator.update_wardrobe_item_status')

@pytest.fixture
def mock_delete_page(mocker):
    return mocker.patch('core.laundry_day_pipeline_orchestrator.delete_page')

@pytest.fixture
def laundry_day_orchestrator():
    return LaundryDayPipelineOrchestrator()

@pytest.mark.asyncio
async def test_run_laundry_day_pipeline_success(laundry_day_orchestrator, mock_get_related_item, mock_update_status, mock_delete_page):
    """
    Tests that the laundry day pipeline runs successfully.
    """
    page_id = "test_page_id"

    result = await laundry_day_orchestrator.run_laundry_day_pipeline(page_id)

    assert result["success"] is True
    mock_get_related_item.assert_called_once_with(page_id)
    mock_update_status.assert_called_once_with("wardrobe_item_id", "Done")
    mock_delete_page.assert_called_once_with(page_id)

@pytest.mark.asyncio
async def test_run_laundry_day_pipeline_no_item(laundry_day_orchestrator, mock_get_related_item, mock_update_status, mock_delete_page):
    """
    Tests that the laundry day pipeline handles the case where there is no related item.
    """
    page_id = "test_page_id"
    mock_get_related_item.return_value = None

    result = await laundry_day_orchestrator.run_laundry_day_pipeline(page_id)

    assert result["success"] is False
    assert "Could not find related wardrobe item." in result["error"]
    mock_get_related_item.assert_called_once_with(page_id)
    mock_update_status.assert_not_called()
    mock_delete_page.assert_not_called()

@pytest.mark.asyncio
async def test_run_laundry_day_pipeline_error(laundry_day_orchestrator, mock_get_related_item, mock_update_status, mock_delete_page):
    """
    Tests that the laundry day pipeline handles errors gracefully.
    """
    page_id = "test_page_id"
    mock_get_related_item.side_effect = Exception("Test error")

    result = await laundry_day_orchestrator.run_laundry_day_pipeline(page_id)

    assert result["success"] is False
    assert "Test error" in result["error"]
    mock_get_related_item.assert_called_once_with(page_id)
    mock_update_status.assert_not_called()
    mock_delete_page.assert_not_called()
