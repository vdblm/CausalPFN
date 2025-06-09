"""
Helper functions used for saving and loading causal effects and qini results in the notebooks.
"""

import os
import re
from contextlib import contextmanager

import numpy as np


def check_match(name, pattern):
    # Escape any special characters in the pattern except '*'
    pattern = re.escape(pattern)

    # Replace '*' with '.*' (the regex pattern for matching any characters)
    pattern = pattern.replace(r"\*", ".*")

    # Match the pattern with the name using full-string match
    return bool(re.fullmatch(pattern, name))


def save_results(result: dict, dataset_name: str, method_name: str, idx: int, artifact_dir: str):
    save_path = os.path.join(artifact_dir, f"{dataset_name}_{method_name}[{idx}].npz")
    np.savez(save_path, **result)


@contextmanager
def result_saver(
    dataset_name: str,
    method_name: str,
    all_method_patterns: list,
    all_datasets_patterns: list,
    idx: int,
    artifact_dir: str,
    replace: bool = False,
):
    save_path = os.path.join(artifact_dir, f"{dataset_name}_{method_name}[{idx}].npz")
    dset_match = False
    for dset_pattern in all_datasets_patterns:
        if check_match(dataset_name, dset_pattern):
            dset_match = True
            break
    method_match = False
    for method_pattern in all_method_patterns:
        if check_match(method_name, method_pattern):
            method_match = True
            break
    if not method_match or not dset_match:
        yield None  # Skip code execution if method is not in run_methods or dataset is not in run_datasets
    elif os.path.exists(save_path) and not replace:
        yield None  # Skip code execution
    else:
        result = {}
        yield result  # Let user fill this with results inside the context
        save_results(result, dataset_name, method_name, idx, artifact_dir)


def load_results(dataset_name: str, method_name: str, idx: int, artifact_dir: str) -> dict:
    load_path = os.path.join(artifact_dir, f"{dataset_name}_{method_name}[{idx}].npz")
    if os.path.exists(load_path):
        data = np.load(load_path)
        return data
    else:
        return None


def load_all_results(dataset_name: str, artifact_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    results = {}
    # go through all the files in the path
    for file in os.listdir(artifact_dir):
        if file.startswith(dataset_name) and file.endswith(".npz"):
            method_name = file.split("_")[1].split("[")[0]
            idx = int(file.split("[")[1].split("]")[0]) if "[" in file else 0
            loaded_dict = load_results(dataset_name, method_name, idx, artifact_dir)
            if method_name not in results:
                results[method_name] = {}
            for key, value in loaded_dict.items():
                if key not in results[method_name]:
                    results[method_name][key] = [value]
                else:
                    results[method_name][key].append(value)

    return results
