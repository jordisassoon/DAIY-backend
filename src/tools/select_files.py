"This module is responsible for selecting which files to process in a directory."

import json
import os
import random
from typing import Union, List
from tools.assertions import (
    assert_source_path_is_found,
    assert_target_path_already_exists,
)


def select_files(
    data_dir: str,
    save_dir: str,
    source_format: str,
    target_format: str,
    subset_path: Union[str, None] = None,
) -> List[str]:
    """
    Creates a list of names of files to process.
    Takes a directory containing the source files, and the directory to which they will be saved.
    To avoid processing twice, target files already contained in the save directory will be ignored.
    A json containing a list of filenames can be passed to only take a subset
    of the files in the source directory.

    Args:
        data_dir (str): directory containing the source files
        save_dir (str): directory where the target files are saved
        subset_path (Union[str, None], optional): path to the json file containing the names
            of the source files to process. If none, then it processes the whole source directory.
        source_format(str): the format of the source files
        target_format(str): the format of the target files

    Returns:
        List[str]: a list of filenames to process.
    """
    if subset_path is not None:
        with open(subset_path, encoding="utf-8") as f:
            dir_list = json.load(f)
    else:
        dir_list = [
            filename.split(".")[0]
            for filename in os.listdir(data_dir)
            if filename.endswith(source_format)
        ]

    file_list = []

    for file_id in dir_list:
        source_url = os.path.join(data_dir, file_id + source_format)
        target_url = os.path.join(save_dir, file_id + target_format)

        if assert_source_path_is_found(
            source_url
        ) and not assert_target_path_already_exists(target_url):
            file_list.append(file_id)

    random.shuffle(file_list)

    return file_list


def compute_urls(
    data_dir: str,
    save_dir: str,
    source_format: str,
    target_format: str,
    subset_path: Union[str, None] = None,
):
    file_list = select_files(
        data_dir=data_dir,
        save_dir=save_dir,
        source_format=source_format,
        target_format=target_format,
        subset_path=subset_path,
    )
    save_urls = [os.path.join(save_dir, vid_id + target_format) for vid_id in file_list]
    return file_list, save_urls
