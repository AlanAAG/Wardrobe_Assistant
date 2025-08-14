import pytest
from unittest.mock import MagicMock
from core.trip_configurator import TripConfigurator

@pytest.fixture
def mock_logging(mocker):
    """Fixture to mock the logging module."""
    mocker.patch('logging.info')
    mocker.patch('logging.error')

def test_prepare_trip_configuration_success(mock_logging):
    """
    Tests that the trip configuration is prepared successfully with valid trigger data.
    """
    trigger_data = {
        "page_id": "test_page_id",
        "destinations": "Dubai, 3 months",
        "preferences": "business, minimalist",
        "dates": {"start": "2024-09-01", "end": "2024-12-01"},
        "bags": ["checked", "cabin"],
    }
    configurator = TripConfigurator(trigger_data)
    trip_config = configurator.prepare_trip_configuration()

    assert trip_config is not None
    assert trip_config["page_id"] == "test_page_id"
    assert trip_config["raw_destinations_and_dates"] == "Dubai, 3 months"
    assert trip_config["raw_preferences_and_purpose"] == "business, minimalist"
    assert trip_config["dates"] == {"start": "2024-09-01", "end": "2024-12-01"}
    assert trip_config["bags"] == ["checked", "cabin"]
    assert "weight_constraints" in trip_config

def test_prepare_trip_configuration_missing_destinations(mock_logging):
    """
    Tests that the trip configuration preparation fails when destinations are missing.
    """
    trigger_data = {
        "page_id": "test_page_id",
        "preferences": "business, minimalist",
        "dates": {"start": "2024-09-01", "end": "2024-12-01"},
        "bags": ["checked", "cabin"],
    }
    configurator = TripConfigurator(trigger_data)
    trip_config = configurator.prepare_trip_configuration()

    assert trip_config is None

def test_prepare_trip_configuration_missing_dates(mock_logging):
    """
    Tests that the trip configuration preparation fails when dates are missing.
    """
    trigger_data = {
        "page_id": "test_page_id",
        "destinations": "Dubai, 3 months",
        "preferences": "business, minimalist",
        "bags": ["checked", "cabin"],
    }
    configurator = TripConfigurator(trigger_data)
    trip_config = configurator.prepare_trip_configuration()

    assert trip_config is None
