"""This module serves as an endpoint to run the deployment class from the cli."""

import argparse
from processes.deployment.tridet_deployer import (
    TridetDeployer,
)


def main(args):
    """Deployment endpoint runner.

    Args:
        args (_type_): arguments needed to run the deployment class.
        For more information, see the README.md, or the argparser below.
    """
    process = TridetDeployer(**vars(args))
    process.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy a model for inference on pre-extracted video features."
    )
    parser.add_argument(
        "--training_config",
        type=str,
        metavar="DIR",
        help="Path to the file containing the model information.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        metavar="DIR",
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        metavar="DIR",
        help="Path of the directory where the features are stored in.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        metavar="DIR",
        help="Path of the directory where the annotations will be saved in.",
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
