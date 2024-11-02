from random import random
import time
import requests
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def run_flow(
    BASE_API_URL: str,
        message: str,
        endpoint: str,
        output_type: str = "chat",
        input_type: str = "chat",
        tweaks: Optional[dict] = None,
        api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()


class LangFlowClient:

    def __init__(
        self,
        BASE_API_URL: str = "http://localhost:7860",
        FLOW_ID: str = "",
        TWEAKS: dict = {},
    ) -> None:
        self.BASE_API_URL = BASE_API_URL
        self.FLOW_ID = FLOW_ID
        self.TWEAKS = TWEAKS

    def _api_call(self, message: dict) -> dict:
        response = run_flow(
            BASE_API_URL=self.BASE_API_URL,
            message=json.dumps(message),
            endpoint=self.FLOW_ID,
            output_type="text",
            input_type="text",
            tweaks=self.TWEAKS,
        )

        # logger.info(f"Response: {response}")

        res = json.loads(response["outputs"][0]["outputs"][0]["outputs"]["text"]["message"])

        return {
            "population": res["population"],
            "set_variable": res["variables"],
        }
    
    def run(self, message: str) -> str:
        """
        Generate n responses
        """
        time.sleep(random())
        for attempt in range(1000):
            try:
                response_cur = self._api_call(message)
            except Exception as e:
                logger.exception(e)
                logger.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
            else:
                break

        if response_cur is None:
            logger.info("Code terminated due to too many failed attempts!")
            exit()
        
        return response_cur
