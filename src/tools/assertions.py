"This module contains all the non-fatal warnings and error checks."

import os
from typing import List
import torch
from tools import bcolors


def assert_target_path_already_exists(url: str) -> bool:
    """Asserts that a target path already exists.

    Args:
        url (str): the path of the target file

    Returns:
        bool: True if the path already exists.
    """
    if os.path.exists(url):
        print(
            f"{bcolors.WARNING}WARNING: Source file {url} has already been "
            + f"processed and saved in the target directory.{bcolors.ENDC}"
        )
        return True
    return False


def assert_checkpoints_already_exist(save_dir: str, target_format: str) -> bool:
    """Asserts that a directory already contains checkpoints.

    Args:
        save_dir (str): the directory in which checkpoints are trying to be saved.
        target_format (str): format of the checkpoint files.

    Returns:
        bool: True if there are already checkpoints present in the directory.
    """
    dir_list = [
        filename.split(".")[0]
        for filename in os.listdir(save_dir)
        if filename.endswith(target_format)
    ]
    if len(dir_list) != 0:
        print(
            f"{bcolors.FAIL}ERROR: Target directory {save_dir} already contains checkpoints."
            + f" They will not be overwritten by default.{bcolors.ENDC}"
        )
        return True
    return False


def assert_source_path_is_found(url: str) -> bool:
    """Asserts that the source path exists.

    Args:
        url (str): path to the source directory or file.

    Returns:
        bool: True if the path exists.
    """
    if not os.path.exists(url):
        print(f"{bcolors.FAIL}ERROR: Path {url} was not found.{bcolors.ENDC}")
        return False
    return True


def assert_files_left_to_process(file_list: List[str]) -> bool:
    """Asserts that the array of files to process is not empty.

    Args:
        video_list (str]): array of filenames left to be processed.

    Returns:
        bool: True if the file list is not empty.
    """
    if len(file_list) == 0:
        print(
            f"{bcolors.WARNING}WARNING: There are no files left to process!{bcolors.ENDC}"
        )
        return False
    return True


def assert_cuda_device_found(device: str) -> bool:
    try:
        torch.device(device)
        return True
    except:
        print(
            f"{bcolors.FAIL}ERROR: Torch device not found! Falling back to current GPU...{bcolors.ENDC}"
        )
        return False
