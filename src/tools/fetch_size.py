"This module contains a simple function that returns the shape of a numpy array given a file path"

import math
import numpy as np


def find_size(path: str):
    """Finds the shape of a .npy saved array.

    Args:
        path (str): path of the .npy file

    Returns:
        _type_: shape of the numpy array
    """
    features = np.load(path)

    return features.shape


def compute_video_duration(features, feature_stride, fps):
    length = math.ceil(features.shape[1] * feature_stride * 10 / fps) / 10
    return length
