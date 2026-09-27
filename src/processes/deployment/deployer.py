from torch.nn import DataParallel

from tools.select_files import compute_urls
from tools.assertions import (
    assert_cuda_device_found,
    assert_files_left_to_process,
    assert_source_path_is_found,
)
from tools.print_helper import run_process
from processes.process import Process


class Deploy(Process):
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
        super().__init__()
        self.training_config = training_config
        self.ckpt_path = ckpt_path
        self.device = device
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.subset_path = subset_path
        self.batch_size = batch_size
        self.source_format = ".npy"
        self.target_format = ".json"
        self.vid_list, self.save_urls = compute_urls(
            data_dir=self.data_dir,
            save_dir=self.save_dir,
            source_format=self.source_format,
            target_format=self.target_format,
            subset_path=self.subset_path,
        )

        self.sanity_check()
        self.print_class()

    def annotate(self, model: DataParallel, video_id: str) -> dict:
        raise NotImplementedError

    def batch_annotate(self) -> None:
        raise NotImplementedError

    def sanity_check(self) -> None:
        for path in [
            self.training_config,
            self.ckpt_path,
            self.data_dir,
            self.save_dir,
        ]:
            assert_source_path_is_found(url=path)

        if self.subset_path is not None:
            assert_source_path_is_found(self.subset_path)

        if assert_cuda_device_found(device=self.device):
            self.device = [self.device]
        else:
            self.device = ["cuda"]

    def run(self) -> None:
        run_process(f"Deploying the model on features in {self.data_dir}...")
        if assert_files_left_to_process(self.vid_list):
            self.batch_annotate()
