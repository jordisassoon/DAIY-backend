# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.utils.data
from torch.nn import DataParallel

from tools.print_helper import process_output_builder
from tools.tridet_loaders import load_model_from_ckpt, load_features, load_for_inference
from tools.save import save_json
from tools.fetch_size import compute_video_duration
from tools.assertions import assert_target_path_already_exists
from processes.deployment.deployer import Deploy


class TridetDeployer(Deploy):
    def __init__(
        self,
        training_config: str,
        ckpt_path: str,
        device: str,
        save_dir: str,
        data_dir: str,
        subset_path: str,
        batch_size: int,
    ) -> None:
        self.name = "Tridet Deployer"
        super().__init__(
            training_config=training_config,
            ckpt_path=ckpt_path,
            device=device,
            save_dir=save_dir,
            data_dir=data_dir,
            subset_path=subset_path,
            batch_size=batch_size,
        )
        self.cfg = load_for_inference(
            self.training_config, self.data_dir, self.batch_size, self.device
        )
        self.fps = 29.97002997002997

    def annotate(self, model: DataParallel, video_id: str) -> dict:
        source_url = os.path.join(self.data_dir, video_id + self.source_format)
        features = load_features(features_file=source_url)

        video_duration = compute_video_duration(
            features=features,
            feature_stride=self.cfg["dataset"]["feat_stride"],
            fps=self.fps,
        )

        dataloader = [
            {
                "video_id": video_id,
                "feats": features,
                "segments": [],
                "labels": torch.tensor([]),
                "fps": self.fps,
                "duration": video_duration,
                "feat_stride": self.cfg["dataset"]["feat_stride"],
                "feat_num_frames": self.cfg["dataset"]["num_frames"],
                "additional_feats": None,
            }
        ]

        output = model(dataloader)[0]

        results = {
            "video_id": video_id,
            "t-start": output["segments"][:, 0].tolist(),
            "t-end": output["segments"][:, 1].tolist(),
            "label": [
                list(self.cfg["dataset"]["class_map"].keys())[
                    list(self.cfg["dataset"]["class_map"].values()).index(label)
                ]
                for label in output["labels"].tolist()
            ],
            "score": output["scores"].tolist(),
        }

        return results

    def batch_annotate(self) -> None:
        model = load_model_from_ckpt(self.cfg, self.ckpt_path, self.device)
        model.eval()

        for video_id, save_url in tqdm(
            zip(self.vid_list, self.save_urls),
            desc=process_output_builder(
                f'Annotating videos from feature folder {self.cfg["dataset"]["feat_folder"]}'
            ),
            total=len(self.vid_list),
        ):
            if not assert_target_path_already_exists(save_url):
                save_json({}, save_url)

                results = self.annotate(model=model, video_id=video_id)

                save_json(results, save_url)
