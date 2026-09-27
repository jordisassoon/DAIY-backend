"This module takes care of saving results from all processes."

import json
import yaml
import numpy as np
from typing import Any, Dict


def save_json(dictionary: Dict[str, Any], path: str):
    """Saves a dictionary as json file in a given path.

    Args:
        dictionary (Dict[str, Any]): the dictionary to be saved.
        path (str): the path to which the dictionary will be saved.
    """
    # serializing json
    json_object = json.dumps(dictionary, indent=4)
    # writing to sample.json
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def save_numpy(np_array: np.ndarray, path: str):
    """Saves a Numpy array in a given path.

    Args:
        np_array (np.ndarray): the numpy array to save.
        path (str): path to which the array will be saved.
    """
    np.save(path, np_array)


def save_yaml(dictionary: Dict[str, Any], path: str):
    if path is None:
        return
    with open(path, "w") as file:
        yaml.dump(dictionary, file)
