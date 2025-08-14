import asyncio
import logging
from typing import Dict, Tuple, Optional

from core.travel_logic_fallback import travel_logic_fallback
from core.travel_packing_agent import travel_packing_agent


class AIPackingOptimizer:
    """Manages the AI provider fallback chain for packing optimization."""

    def __init__(self, api_retry_attempts: int = 3, api_retry_delay: float = 1.0):
        self._api_retry_attempts = api_retry_attempts
        self._api_retry_delay = api_retry_delay

    async def execute_packing_optimization_chain(
        self, trip_config: Dict, available_items: Dict
    ) -> Dict:
        """Executes the hierarchical AI optimization chain."""
        # Try Gemini
        success, packing_result, error_msg = await self._try_gemini_with_retry(
            trip_config, available_items
        )
        if success and packing_result:
            logging.info("✅ Gemini optimization successful")
            return {
                "success": True,
                "data": packing_result,
                "generation_method": "gemini",
            }
        logging.warning(f"Gemini failed: {error_msg}")

        # Try Groq
        success, packing_result, error_msg = await self._try_groq_with_retry(
            trip_config, available_items
        )
        if success and packing_result:
            logging.info("✅ Groq optimization successful")
            return {
                "success": True,
                "data": packing_result,
                "generation_method": "groq",
            }
        logging.warning(f"Groq failed: {error_msg}")

        return {
            "success": False,
            "error": "Both Gemini and Groq failed to generate a packing list.",
            "attempted_methods": ["gemini", "groq"],
        }

    async def _try_gemini_with_retry(
        self, trip_config: Dict, available_items: Dict
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Try Gemini API with retry logic and enhanced error logging."""
        for attempt in range(self._api_retry_attempts):
            try:
                if attempt > 0:
                    await asyncio.sleep(self._api_retry_delay * attempt)
                    logging.info(f"   Gemini retry attempt {attempt + 1}/{self._api_retry_attempts}")

                return await travel_packing_agent.generate_multi_destination_packing_list(
                    trip_config, available_items, timeout=120
                )

            except asyncio.TimeoutError:
                logging.warning(f"   Gemini timeout on attempt {attempt + 1}")
                if attempt == self._api_retry_attempts - 1:
                    return False, None, f"Gemini timeout after {self._api_retry_attempts} attempts"

            except Exception as e:
                logging.error(f"   Gemini error on attempt {attempt + 1}: {str(e)}", exc_info=True)
                if attempt == self._api_retry_attempts - 1:
                    return False, None, f"Gemini error: {str(e)}"

        return False, None, "Gemini retry attempts exhausted"

    async def _try_groq_with_retry(
        self, trip_config: Dict, available_items: Dict
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Try Groq API with retry logic and enhanced error logging."""
        for attempt in range(self._api_retry_attempts):
            try:
                if attempt > 0:
                    await asyncio.sleep(self._api_retry_delay * attempt)
                    logging.info(f"   Groq retry attempt {attempt + 1}/{self._api_retry_attempts}")

                return await travel_packing_agent.generate_packing_list_with_groq(
                    trip_config, available_items, timeout=90
                )

            except asyncio.TimeoutError:
                logging.warning(f"   Groq timeout on attempt {attempt + 1}")
                if attempt == self._api_retry_attempts - 1:
                    return False, None, f"Groq timeout after {self._api_retry_attempts} attempts"

            except Exception as e:
                logging.error(f"   Groq error on attempt {attempt + 1}: {str(e)}", exc_info=True)
                if attempt == self._api_retry_attempts - 1:
                    return False, None, f"Groq error: {str(e)}"

        return False, None, "Groq retry attempts exhausted"
