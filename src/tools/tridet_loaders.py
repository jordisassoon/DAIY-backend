"This module loads the configuration files according to the TriDet repo."

from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import numpy as np

from libs.core import load_config
from libs.datasets import make_data_loader, make_dataset
from libs.modeling import make_meta_arch
from libs.utils import ANETdetection
from tools.print_helper import process_output
from tools.dynamic_loaders import (
    load_yaml,
    load_extension,
    load_class_num,
    load_json,
    create_class_map,
)
from tools.save import save_json


def load_configs(
    model_config: str,
    dataset_config: str,
    annotations_file: str,
    data_dir: str,
    backbone_type: str,
    batch_size: int,
    artefact_file: str,
    learning_rate: float,
    epochs: int,
    device: str,
) -> Dict[str, Any]:
    """Loads the configuration dictionary for model and dataset characteristics.

    Args:
        model_config (str): path to the model config file
        dataset_config (str): path to the dataset config file
        device (str): device on which to run the model on

    Returns:
        Dict[str, Any]: a full configuration dictionary containing both model and dataset.
    """
    class_map = create_class_map(annotations_file=annotations_file)
    extra_dataset_configs = load_dataset_config(
        annotations_file=annotations_file,
        data_dir=data_dir,
        backbone_type=backbone_type,
        batch_size=batch_size,
        class_map=class_map,
    )
    fe_params = load_feature_extraction_params(artefact_file=artefact_file)

    cfg = load_config(model_config)
    cfg = load_config(dataset_config, defaults=cfg)
    cfg = load_config(extra_dataset_configs, defaults=cfg, load_yaml_flag=False)
    cfg = load_config(fe_params, defaults=cfg, load_yaml_flag=False)
    cfg["devices"] = [device]
    cfg["opt"]["learning_rate"] = learning_rate
    cfg["opt"]["epochs"] = epochs
    return cfg


def load_for_inference(
    training_config: str, data_dir: str, batch_size: int, device: str
):
    extra_dataset_configs = {
        "dataset": {
            "feat_folder": data_dir,
        },
        "loader": {"batch_size": batch_size},
    }
    cfg = load_config(training_config)
    cfg = load_config(extra_dataset_configs, defaults=cfg, load_yaml_flag=False)
    cfg["devices"] = device

    return cfg


def load_dataset_config(
    annotations_file: str,
    backbone_type: str,
    batch_size: int,
    data_dir: str,
    class_map: dict,
):
    cfg = {
        "dataset": {
            "json_file": annotations_file,
            "feat_folder": data_dir,
            "file_ext": load_extension(data_dir),
            "num_classes": load_class_num(annotations_file),
            "backbone_type": backbone_type,
            "class_map": class_map,
        },
        "loader": {"batch_size": batch_size},
    }
    return cfg


def load_feature_extraction_params(artefact_file):
    feature_extraction_params = load_yaml(artefact_file)
    cfg = {
        "dataset": {
            "input_dim": feature_extraction_params["feature_size"],
            "feat_stride": feature_extraction_params["stride"],
            "num_frames": feature_extraction_params["frame_window"],
        }
    }
    return cfg


def load_val_data(cfg) -> Tuple[torch.utils.data.DataLoader, ANETdetection]:
    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
    )
    # data loaders
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg["loader"]["num_workers"]
    )

    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds=val_db_vars["tiou_thresholds"],
    )

    return val_loader, det_eval


def load_train_data(cfg, rng_generator) -> torch.utils.data.DataLoader:
    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    # data loaders
    train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg["loader"])

    return train_loader


def load_model(cfg, device):
    model = make_meta_arch(cfg["model_name"], **cfg["model"])
    model = nn.DataParallel(model, device_ids=device)
    return model


def load_model_from_ckpt(cfg, ckpt_path, device):
    model = load_model(cfg=cfg, device=device)
    checkpoint = load_ckpt(ckpt_path=ckpt_path, device=device[0])
    model.load_state_dict(checkpoint["state_dict_ema"])
    del checkpoint
    return model


def load_ckpt(ckpt_path, device):
    process_output("Loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    return checkpoint


def load_features(features_file):
    numpy_features = np.load(features_file)
    torch_features = torch.from_numpy(numpy_features).T
    return torch_features
