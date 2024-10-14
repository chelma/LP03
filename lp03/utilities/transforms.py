import importlib
import os
from typing import Any, Callable, Dict, List

def get_transform_file_path(transform_files_dir: str, transform_id: str) -> str:
    return os.path.join(transform_files_dir, f"{transform_id}.json")

def get_transform_input_file_path(transform_files_dir: str, transform_id: str) -> str:
    return os.path.join(transform_files_dir, f"{transform_id}_input.json")

def get_transform_output_file_path(transform_files_dir: str, transform_id: str) -> str:
    return os.path.join(transform_files_dir, f"{transform_id}_output.json")

def load_transform_from_file(transform_file_path: str) -> Callable[[Dict[str, Any]], List[Dict[str, Any]]]:
    module_spec = importlib.util.spec_from_file_location("transform", transform_file_path)
    if module_spec is None:
        raise ImportError(f"Cannot load the transform module from {transform_file_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.transform