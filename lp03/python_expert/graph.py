from functools import wraps
import importlib.util
import json
import logging
from typing import Annotated, Any, Callable, Dict, List, Literal
from typing_extensions import TypedDict
import uuid

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledGraph

from python_expert.tools import TOOLS_ALL, TOOLS_NORMAL, Transform, make_transform_tool
from utilities.transforms import get_transform_file_path, get_transform_input_file_path, get_transform_output_file_path, load_transform_from_file

logger = logging.getLogger(__name__)

# Define our LLM
llm = ChatBedrockConverse(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0,
    max_tokens=4096,
    region_name="us-west-2"
)
llm_with_tools = llm.bind_tools(TOOLS_ALL)

# Define the state our graph will be operating on
class PythonState(TypedDict):
    # Store a copy of the original input and current transform output
    input: Dict[str, Any]
    output: List[Dict[str, Any]]

    # Hold the internal conversation of the Python expert
    python_turns: Annotated[List[BaseMessage], add_messages]

    # Retain details on the current Transform we're working on
    transform: Transform
    transform_id: str
    transform_files_dir: str

def python_state_to_json(state: PythonState) -> Dict[str, Any]:
    return {
        "python_turns": [turn.to_json() for turn in state.get("python_turns", [])],
        "transform": state.get("transform").to_json() if state.get("transform") else None,
        "transform_id": state.get("transform_id", None),
        "transform_files_dir": state.get("transform_files_dir", None)
    }
    
def trace_python_node(func: Callable[[PythonState], Dict[str, Any]]) -> Callable[[PythonState], Dict[str, Any]]:
    @wraps(func)
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        logging.info(f"Entering node: {func.__name__}")
        state_json = python_state_to_json(state)
        logging.debug(f"Starting state: {str(state_json)}")
        
        result = func(state)
        
        logging.debug(f"Output of {func.__name__}: {result}")
        
        return result
    
    return wrapper

# Set up our tools
tools_normal_by_name = {tool.name: tool for tool in TOOLS_NORMAL}

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Define our graph
python_graph = StateGraph(PythonState)

# Set up our graph nodes
class MissingStateError(Exception):
    pass

@trace_python_node
def node_validate_starting_state(state: PythonState):
    if not state.get("input", None):
        raise MissingStateError("State 'input' is missing.  You must provide the original input to be transformed.")

    if not state.get("transform_files_dir", None):
        raise MissingStateError("State 'transform_files_dir' is missing.  You must provide an absolute path to a directory to store transform files.")

    return {"python_turns": []}

@trace_python_node
def node_invoke_llm_python(state: PythonState):
    python_turns = state["python_turns"]
    response = llm_with_tools.invoke(python_turns)
    return {"python_turns": [response]}

@trace_python_node
def node_make_transform(state: PythonState) -> Dict[str, any]:
    """
    Node to create or update our transform
    """
    # Generate the transform
    tool_call = state["python_turns"][-1].tool_calls[-1]
    transform = make_transform_tool.invoke(tool_call["args"])
    transform_id = state.get("transform_id", str(uuid.uuid4())) # Pick an ID if we don't have one
    logger.info(f"Generated transform with ID: {transform_id}")

    # Store the transform in a file
    raw_file_contents = f"{transform.imports}\n\n\"\"\"\n{transform.description}\n\"\"\"\n\n{transform.code}"
    file_path = get_transform_file_path(state["transform_files_dir"], transform_id)
    with open(file_path, "w") as f:
        f.write(raw_file_contents)
    logger.info(f"Transform written to file: {file_path}")

    # Store the input in a file as well
    input_file_path = get_transform_input_file_path(state["transform_files_dir"], transform_id)
    with open(input_file_path, "w") as f:
        f.write(json.dumps(state.get("input", {}), indent=4))
    
    # Update our State and exit the node.  We create a tool message to capture our work creating the transform,
    # and an AIMessage message to return to the original caller.
    result = []
    tool_message = ToolMessage(name="MakeTransform", content=transform.to_json(), tool_call_id=tool_call["id"])
    ai_message = AIMessage(content=f"Transform created at path: {file_path}")
    result.append(tool_message)
    result.append(ai_message)

    return {"python_turns": result, "transform": transform, "transform_id": transform_id}

@trace_python_node
def node_test_transform(state: PythonState) -> Dict[str, Any]:
    """
    Node to test the transform.
    """
    # Load the input data from the file
    input_file_path = get_transform_input_file_path(state["transform_files_dir"], state["transform_id"])
    with open(input_file_path, "r") as f:
        input_data = json.load(f)

    # Load the transform function to be tested
    transform_file_path = get_transform_file_path(state["transform_files_dir"], state["transform_id"])
    transform_function = load_transform_from_file(transform_file_path)

    # Execute the transform using the input data
    output_data = transform_function(input_data)

    # Store the output in a file
    output_file_path = get_transform_output_file_path(state["transform_files_dir"], state["transform_id"])
    with open(output_file_path, "w") as f:
        f.write(json.dumps(output_data, indent=4))

    # Update our State and exit the node.  We create a tool message to capture our work testing the transform,
    # and an AIMessage message to return to the original caller.
    result = []
    transform_test_result_message = f"Result of executing the transform on the input:\n{json.dumps(output_data)}"
    tool_message = ToolMessage(name="TestTransform", content=transform_test_result_message, tool_call_id=uuid.uuid4())
    ai_message = AIMessage(content=f"Transform tested successfully.  Output written to: {output_file_path}")
    result.append(tool_message)
    result.append(ai_message)

    return {"python_turns": result}

python_graph.add_node("node_validate_starting_state", node_validate_starting_state)
python_graph.add_node("node_invoke_llm_python", node_invoke_llm_python)
python_graph.add_node("node_make_transform", node_make_transform)
python_graph.add_node("node_test_transform", node_test_transform)

# Define our graph edges
# def next_node(state: PythonState) -> Literal["node_make_transform", END]:
#     python_turns = state["python_turns"]
#     last_message = python_turns[-1]

#     if last_message.tool_calls and last_message.tool_calls[-1]["name"] == "MakeTransform":
#         return "node_make_transform"
    
#     return END

python_graph.add_edge(START, "node_validate_starting_state")
python_graph.add_edge("node_validate_starting_state", "node_invoke_llm_python")
python_graph.add_edge("node_invoke_llm_python", "node_make_transform")
# python_graph.add_conditional_edges("node_invoke_llm_python", next_node)
python_graph.add_edge("node_make_transform", "node_test_transform")
python_graph.add_edge("node_test_transform", END)

# Finally, compile the graph into a LangChain Runnable
PYTHON_GRAPH = python_graph.compile(checkpointer=checkpointer)

def _create_runner(workflow: CompiledGraph):
    def run_workflow(cw_state: PythonState, thread: int) -> PythonState:
        states = workflow.stream(
            cw_state,
            config={"configurable": {"thread_id": thread}},
            stream_mode="values"
        )

        final_state = None
        for state in states:
            if "python_turns" in state:
                state["python_turns"][-1].pretty_print()
                logger.info(state["python_turns"][-1].to_json())
            final_state = state

        return final_state

    return run_workflow

PYTHON_GRAPH_RUNNER = _create_runner(PYTHON_GRAPH)