import pytest
from unittest.mock import AsyncMock, MagicMock
from core.travel_pipeline_orchestrator import TravelPipelineOrchestrator

@pytest.fixture
def orchestrator(mocker):
    """Fixture to create a TravelPipelineOrchestrator with base dependencies mocked."""
    mocker.patch('core.travel_pipeline_orchestrator.wardrobe_data_manager', MagicMock())
    mocker.patch('core.travel_pipeline_orchestrator.notion', MagicMock())
    orchestrator_instance = TravelPipelineOrchestrator()
    orchestrator_instance.ensure_ready = AsyncMock()
    orchestrator_instance._get_travel_optimized_wardrobe_data = AsyncMock(return_value={"some": "items"})
    return orchestrator_instance

@pytest.mark.asyncio
async def test_run_travel_packing_pipeline_success(orchestrator, mocker):
    """
    Tests the full pipeline orchestration succeeds.
    """
    trigger_data = {"page_id": "test_page"}

    mock_configurator_class = mocker.patch('core.travel_pipeline_orchestrator.TripConfigurator')
    mock_optimizer_class = mocker.patch('core.travel_pipeline_orchestrator.AIPackingOptimizer')
    mock_publisher_class = mocker.patch('core.travel_pipeline_orchestrator.NotionResultPublisher')

    mock_configurator_instance = mock_configurator_class.return_value
    mock_optimizer_instance = mock_optimizer_class.return_value
    mock_publisher_instance = mock_publisher_class.return_value

    mock_configurator_instance.prepare_trip_configuration.return_value = {"some": "config"}
    mock_optimizer_instance.execute_packing_optimization_chain = AsyncMock(return_value={"success": True, "data": {}, "generation_method": "gemini"})
    mock_publisher_instance.finalize_packing_results = AsyncMock(return_value={"success": True})

    result = await orchestrator.run_travel_packing_pipeline(trigger_data)

    assert result["success"] is True
    mock_configurator_class.assert_called_once_with(trigger_data)
    mock_optimizer_class.assert_called_once()
    mock_publisher_class.assert_called_once()

@pytest.mark.asyncio
async def test_run_travel_packing_pipeline_config_fails(orchestrator, mocker):
    """
    Tests the pipeline handles failure in trip configuration.
    """
    trigger_data = {"page_id": "test_page"}
    mock_configurator_class = mocker.patch('core.travel_pipeline_orchestrator.TripConfigurator')
    mock_configurator_instance = mock_configurator_class.return_value
    mock_configurator_instance.prepare_trip_configuration.return_value = None

    result = await orchestrator.run_travel_packing_pipeline(trigger_data)

    assert result["success"] is False
    assert "Invalid trip configuration" in result["error"]

@pytest.mark.asyncio
async def test_run_travel_packing_pipeline_data_fails(orchestrator, mocker):
    """
    Tests the pipeline handles failure in data acquisition.
    """
    trigger_data = {"page_id": "test_page"}
    mocker.patch('core.travel_pipeline_orchestrator.TripConfigurator').return_value.prepare_trip_configuration.return_value = {"some": "config"}
    orchestrator._get_travel_optimized_wardrobe_data = AsyncMock(return_value={})

    result = await orchestrator.run_travel_packing_pipeline(trigger_data)

    assert result["success"] is False
    assert "No wardrobe items available" in result["error"]

@pytest.mark.asyncio
async def test_run_travel_packing_pipeline_optimizer_fails(orchestrator, mocker):
    """
    Tests the pipeline handles failure in the AI optimizer.
    """
    trigger_data = {"page_id": "test_page"}
    mocker.patch('core.travel_pipeline_orchestrator.TripConfigurator').return_value.prepare_trip_configuration.return_value = {"some": "config"}
    mock_optimizer_class = mocker.patch('core.travel_pipeline_orchestrator.AIPackingOptimizer')
    mock_optimizer_instance = mock_optimizer_class.return_value
    mock_optimizer_instance.execute_packing_optimization_chain = AsyncMock(return_value={"success": False, "error": "AI failed"})

    result = await orchestrator.run_travel_packing_pipeline(trigger_data)

    assert result["success"] is False
    assert "AI failed" in result["error"]

@pytest.mark.asyncio
async def test_run_travel_packing_pipeline_publisher_fails(orchestrator, mocker):
    """
    Tests the pipeline handles failure in the Notion publisher.
    """
    trigger_data = {"page_id": "test_page"}
    mocker.patch('core.travel_pipeline_orchestrator.TripConfigurator').return_value.prepare_trip_configuration.return_value = {"some": "config"}
    mocker.patch('core.travel_pipeline_orchestrator.AIPackingOptimizer').return_value.execute_packing_optimization_chain = AsyncMock(return_value={"success": True, "data": {}, "generation_method": "gemini"})
    mock_publisher_class = mocker.patch('core.travel_pipeline_orchestrator.NotionResultPublisher')
    mock_publisher_instance = mock_publisher_class.return_value
    mock_publisher_instance.finalize_packing_results = AsyncMock(return_value={"success": False, "error": "Publisher failed"})

    result = await orchestrator.run_travel_packing_pipeline(trigger_data)

    assert result["success"] is False
    assert "Publisher failed" in result["error"]
