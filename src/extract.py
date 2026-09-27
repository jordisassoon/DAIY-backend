"""This module serves as an endpoint to run the extraction class from the cli."""

import argparse
from processes.extraction.videomae_extractor import (
    VideoMAEExtractor,
)


def main(args):
    """Extraction endpoint runner.

    Args:
        args (_type_): arguments needed to run the extraction class.
        For more information, see the README.md, or the argparser below.
    """
    process = VideoMAEExtractor(**vars(args))
    process.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract TAD features using a pre-trained model"
    )
    parser.add_argument(
        "--arch_config_path",
        type=str,
        default="./models/architectures/videomae/vit_s_k710_dl_from_giant_config.yaml",
        help="Path to the config file for the architecture.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        metavar="DIR",
        default="./ckpt/videomae/vit_s_k710_dl_from_giant.pth",
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        metavar="DIR",
        help="Path of the directory where the videos are stored in.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        metavar="DIR",
        help="Path of the directory where the features will be saved in.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Cuda device to run the training on.",
    )
    parser.add_argument(
        "--subset_path",
        type=str,
        metavar="DIR",
        help="Path to the json file containing a list of video ids.",
    )
    _args = parser.parse_args()
    main(_args)
