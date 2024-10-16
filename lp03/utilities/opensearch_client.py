import logging
from typing import Optional, Dict, Any

from utilities.rest_client import RESTClient

logger = logging.getLogger(__name__)

class OpenSearchClient():
    rest_client: RESTClient

    def __init__(self, rest_client: RESTClient) -> None:
        self.rest_client = rest_client

    def create_index(self, index_name: str, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"Creating index: {index_name} with settings: {settings}")
        endpoint = f"{index_name}"
        return self.rest_client.put(endpoint, data=settings)

    def describe_index(self, index_name: str) -> Dict[str, Any]:
        logger.info(f"Describing index: {index_name}")
        endpoint = f"{index_name}"
        return self.rest_client.get(endpoint)

    def update_index(self, index_name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Updating index: {index_name} with settings: {settings}")
        endpoint = f"{index_name}/_settings"
        return self.rest_client.put(endpoint, data=settings)

    def delete_index(self, index_name: str) -> Dict[str, Any]:
        logger.info(f"Deleting index: {index_name}")
        endpoint = f"{index_name}"
        return self.rest_client.delete(endpoint)