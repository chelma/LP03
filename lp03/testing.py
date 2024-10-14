import logging

from langchain_core.messages import HumanMessage

from python_expert.prompting import get_transform_index_prompt
from python_expert.graph import PythonState, PYTHON_GRAPH_RUNNER, python_state_to_json
from utilities.logging import configure_logging
from utilities.rest_client import ConnectionDetails

configure_logging("./debug.log", "./info.log")

logger = logging.getLogger(__name__)

transform_input = {
    "indexName": "test_index",
    "indexJson": {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "type1": {
                "properties": {
                    "title": { "type": "text" }
                }
            },
            "type2": {
                "properties": {
                    "contents": { "type": "text" }
                }
            }
        }
    }
}


system_message = get_transform_index_prompt(
    "Elasticsearch 6.8",
    "OpenSearch 2.14",
    transform_input
)


connection_details = ConnectionDetails(
    base_url="http://localhost:29200"
)

python_state = PythonState(
    input = transform_input,
    python_turns = [
        system_message
    ],
    connection_details = connection_details,
    transform_files_dir="/tmp/transforms"
)
python_state["python_turns"].append(
    HumanMessage(content="Please make the transform")
)
final_state = PYTHON_GRAPH_RUNNER(python_state, 42)
logger.info(f"Final state: {python_state_to_json(final_state)}")