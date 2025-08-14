import pytest
from unittest.mock import AsyncMock, MagicMock
from core.ai_packing_optimizer import AIPackingOptimizer

@pytest.fixture
def trip_config():
    """Provides a minimal trip_config for testing."""
    return {
        "trip_overview": {
            "total_duration_days": 10,
        }
    }

@pytest.mark.asyncio
async def test_execute_packing_optimization_chain_gemini_success(mocker, trip_config):
    """
    Tests that the optimization chain returns success when Gemini succeeds.
    """
    mock_gemini = mocker.patch('core.ai_packing_optimizer.travel_packing_agent.generate_multi_destination_packing_list', new_callable=AsyncMock)
    mock_gemini.return_value = (True, {"items": ["shirt"]}, None)

    optimizer = AIPackingOptimizer()
    result = await optimizer.execute_packing_optimization_chain(trip_config, {})

    assert result["success"] is True
    assert result["generation_method"] == "gemini"
    assert result["data"] == {"items": ["shirt"]}

@pytest.mark.asyncio
async def test_execute_packing_optimization_chain_groq_success(mocker, trip_config):
    """
    Tests that the optimization chain falls back to Groq and returns success when Groq succeeds.
    """
    mock_gemini = mocker.patch('core.ai_packing_optimizer.travel_packing_agent.generate_multi_destination_packing_list', new_callable=AsyncMock)
    mock_gemini.return_value = (False, None, "Gemini failed")

    mock_groq = mocker.patch('core.ai_packing_optimizer.travel_packing_agent.generate_packing_list_with_groq', new_callable=AsyncMock)
    mock_groq.return_value = (True, {"items": ["pants"]}, None)

    optimizer = AIPackingOptimizer()
    result = await optimizer.execute_packing_optimization_chain(trip_config, {})

    assert result["success"] is True
    assert result["generation_method"] == "groq"
    assert result["data"] == {"items": ["pants"]}

@pytest.mark.asyncio
async def test_execute_packing_optimization_chain_all_fail(mocker, trip_config):
    """
    Tests that the optimization chain returns failure when all methods fail.
    """
    mock_gemini = mocker.patch('core.ai_packing_optimizer.travel_packing_agent.generate_multi_destination_packing_list', new_callable=AsyncMock)
    mock_gemini.return_value = (False, None, "Gemini failed")

    mock_groq = mocker.patch('core.ai_packing_optimizer.travel_packing_agent.generate_packing_list_with_groq', new_callable=AsyncMock)
    mock_groq.return_value = (False, None, "Groq failed")

    optimizer = AIPackingOptimizer()
    result = await optimizer.execute_packing_optimization_chain(trip_config, {})

    assert result["success"] is False
    assert result["error"] == "Both Gemini and Groq failed to generate a packing list."
