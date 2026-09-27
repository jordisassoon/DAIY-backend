"""Extract features for temporal action detection datasets"""

import os

import numpy as np
import torch
import torchvision
from timm.models import create_model
from tools.print_helper import process_output, process_output_builder
from torchvision import transforms
from tqdm import tqdm
import time
from typing import Callable

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from tools.save import save_numpy
from dataset.loader import get_video_loader
from processes.extraction.extractor import Extract
from tools.assertions import assert_target_path_already_exists


def resize(vid, size, interpolation="bilinear"):
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False,
    )


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


def thumos14_range(num_frames, frame_window, stride):
    return range(0, num_frames - frame_window - 1, stride)


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


class VideoMAEExtractor(Extract):
    def __init__(
        self,
        data_dir: str,
        subset_path: str,
        ckpt_path: str,
        arch_config_path: str,
        save_dir: str,
        device: str,
    ):
        self.name = "VideoMAE Extractor"
        super().__init__(
            arch_config_path=arch_config_path,
            ckpt_path=ckpt_path,
            data_dir=data_dir,
            subset_path=subset_path,
            device=device,
            save_dir=save_dir,
        )

    def load_model(self) -> torch._dynamo.eval_frame.OptimizedModule:
        # get base_model & load ckpt
        model = create_model(
            self.arch_configs["base_model"],
            img_size=self.arch_configs["img_size"],
            pretrained=self.arch_configs["pretrained"],
            num_classes=self.arch_configs["num_classes"],
            all_frames=self.arch_configs["frame_window"],
            tubelet_size=self.arch_configs["tubelet_size"],
            drop_path_rate=self.arch_configs["drop_path_rate"],
            use_mean_pooling=self.arch_configs["use_mean_pooling"],
        )

        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)

        for model_key in ["model", "module"]:
            if model_key in ckpt:
                ckpt = ckpt[model_key]
                break

        model.load_state_dict(ckpt)
        model = torch.compile(model, backend="inductor")

        model.eval()
        # model.half()
        model.to(self.device)

        return model

    def extract_features(
        self,
        vr: Callable,
        transform: torchvision.transforms.transforms.Compose,
        model: torch._dynamo.eval_frame.OptimizedModule,
        vid_name: str,
    ) -> np.ndarray:
        video_path = os.path.join(self.data_dir, vid_name + ".mp4")

        video_loader = vr(video_path)

        feature_list = []

        for start_idx in tqdm(
            thumos14_range(
                len(video_loader),
                self.arch_configs["frame_window"],
                self.arch_configs["stride"],
            ),
            desc=process_output_builder(
                f"Extracting features from frame batches of {vid_name}"
            ),
        ):
            data = video_loader.get_batch(
                np.arange(start_idx, start_idx + self.arch_configs["frame_window"])
            ).asnumpy()

            frame = torch.from_numpy(data)  # torch.Size([16, 566, 320, 3])
            frame_q = transform(frame).to(self.device)  # torch.Size([3, 16, 224, 224])
            input_data = frame_q.unsqueeze(0)

            with torch.no_grad():
                feature = model.forward_features(input_data)

            feature_list.append(feature)

        features = torch.cat(feature_list, dim=0)

        return features.cpu().numpy()

    def batch_extract(self) -> None:
        video_loader = get_video_loader()
        transform = transforms.Compose(
            [
                ToFloatTensorInZeroOne(),
                Resize((self.arch_configs["img_size"], self.arch_configs["img_size"])),
            ]
        )

        process_output(f"Loading model from checkpoint {self.ckpt_path}")
        model = self.load_model()

        with torch.autocast(device_type=self.device, dtype=torch.float16):
            for idx, vid_name in tqdm(
                enumerate(self.vid_list),
                desc=process_output_builder(
                    f"Extracting features from videos in {self.data_dir}"
                ),
                total=len(self.vid_list),
            ):
                if assert_target_path_already_exists(self.save_urls[idx]):
                    continue

                save_numpy(np.zeros(1), self.save_urls[idx])

                feature_list = self.extract_features(
                    vr=video_loader,
                    transform=transform,
                    model=model,
                    vid_name=vid_name,
                )

                save_numpy(feature_list, self.save_urls[idx])

        return
