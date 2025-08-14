import logging
from typing import Dict, Optional

from config.travel_config import WEIGHT_CONSTRAINTS


class TripConfigurator:
    """Handles the preparation and validation of trip configuration data."""

    def __init__(self, trigger_data: Dict):
        self._trigger_data = trigger_data
        self._trip_config: Optional[Dict] = None

    def prepare_trip_configuration(self) -> Optional[Dict]:
        """
        Prepares a lean configuration by passing raw user input directly to the AI agent
        for dynamic analysis and interpretation.
        """
        logging.info("🧳 Preparing raw trip configuration for AI analysis...")
        self._log_trigger_data()

        self._trip_config = {
            "page_id": self._trigger_data.get("page_id"),
            "raw_destinations_and_dates": self._trigger_data.get("destinations", ""),
            "raw_preferences_and_purpose": self._trigger_data.get("preferences", ""),
            "dates": self._trigger_data.get("dates", {}),
            "bags": self._trigger_data.get("bags", []),
            "weight_constraints": WEIGHT_CONSTRAINTS,
        }

        if not self._trip_config["raw_destinations_and_dates"] or not self._trip_config["dates"]:
            logging.error("❌ Critical trip information (destinations or dates) is missing.")
            return None

        logging.info("✅ Raw trip configuration prepared for AI agent.")
        return self._trip_config

    def _log_trigger_data(self) -> None:
        """Logs the raw trigger data received from the Notion page."""
        logging.info("🧳 Pipeline trigger data:")
        logging.info(f"   Page ID: {self._trigger_data.get('page_id', 'unknown')}")
        logging.info(f"   Destinations Input: \"{self._trigger_data.get('raw_destinations_and_dates', 'N/A')}\"")
        logging.info(f"   Preferences Input: \"{self._trigger_data.get('raw_preferences_and_purpose', 'N/A')}\"")
        logging.info(f"   Bags Input: {', '.join(self._trigger_data.get('bags', ['N/A']))}")
