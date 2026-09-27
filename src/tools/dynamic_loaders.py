"This module contains a simple function that returns the shape of a numpy array given a file path"

import math
import numpy as np
import glob, os
import yaml
import json


def find_size(path: str):
    """Finds the shape of a .npy saved array.

    Args:
        path (str): path of the .npy file

    Returns:
        _type_: shape of the numpy array
    """
    features = np.load(path)

    return features.shape


def input_dim(path: str):
    return find_size(path=path)[1]


def load_class_num(annotations_file: str):
    data = load_json(annotations_file)

    unique_labels = set()
    for video_data in data["database"].values():
        for annotation in video_data["annotations"]:
            unique_labels.add(annotation["label"])

    return len(unique_labels)


def create_class_map(annotations_file):
    data = load_json(annotations_file)

    unique_labels = {}
    idx = 0
    for video_data in data["database"].values():
        for annotation in video_data["annotations"]:
            if annotation["label"] not in unique_labels:
                unique_labels[annotation["label"]] = idx
                idx += 1
            annotation["label_id"] = unique_labels[annotation["label"]]

    return unique_labels


def compute_video_duration(features, feature_stride, fps):
    length = math.ceil(features.shape[1] * feature_stride * 10 / fps) / 10
    return length


def load_extension(feature_path: str):
    return ".npy"


def load_yaml(yaml_path):
    if yaml_path is None:
        return
    """
    Loads a YAML file and returns its contents as a dictionary.

    :param yaml_path: Path to the YAML file.
    :return: Dictionary containing the YAML data.
    """
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def load_json(json_path):
    if json_path is None:
        return

    with open(json_path, "r") as file:
        data = json.load(file)
    return data
