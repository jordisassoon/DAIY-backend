"""This module serves as an endpoint to run the evaluation class from the cli."""

import argparse
from processes.evaluation.tridet_evaluator import (
    TridetEvaluator,
)


def main(args):
    """Evaluation endpoint runner.

    Args:
        args (_type_): arguments needed to run the evaluation class.
        For more information, see the README.md, or the argparser below.
    """
    process = TridetEvaluator(**vars(args))
    process.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-trained model on pre-extracted video features."
    )
    parser.add_argument(
        "--training_config",
        type=str,
        metavar="DIR",
        help="Path to the file containing the model information.",
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
        "--ckpt_path",
        type=str,
        metavar="DIR",
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Cuda device to run the training on.",
    )
    _args = parser.parse_args()
    main(_args)
