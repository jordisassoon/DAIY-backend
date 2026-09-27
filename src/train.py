"""This module serves as an endpoint to run the training class from the cli."""

import argparse
from processes.training.tridet_trainer import TridetTrainer


def main(args):
    """Training endpoint runner.

    Args:
        args (_type_): arguments needed to run the training class.
        For more information, see the README.md, or the argparser below.
    """
    process = TridetTrainer(**vars(args))
    process.run()


if __name__ == "__main__":
    # the arg parser
    parser = argparse.ArgumentParser(
        description="Train a model for TAD on pre-extracted features"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        metavar="DIR",
        help="Path to the file containing the model information.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        metavar="DIR",
        help="Path to the file containing the dataset information.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        metavar="DIR",
        help="Path of the directory where the features are stored in.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        metavar="DIR",
        help="Path of the directory where the checkpoints will be saved in.",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate of the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--validation_step",
        type=int,
        help="Number of epochs between model validations.",
    )
    parser.add_argument(
        "--ckpt_freq",
        type=int,
        help="Number of epochs between model checkpointing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Cuda device to run the training on.",
    )
    _args = parser.parse_args()
    main(_args)
