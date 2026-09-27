import os
import yaml

from tools.print_helper import run_process
from tools.select_files import compute_urls
from tools.dynamic_loaders import load_yaml
from tools.save import save_yaml
from tools.assertions import *
import tools.bcolors as bcolors
from processes.process import Process


class Extract(Process):
    def __init__(
        self, arch_config_path, ckpt_path, data_dir, subset_path, device, save_dir
    ):
        super().__init__()
        self.arch_config_path = arch_config_path
        self.ckpt_path = ckpt_path
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.device = device
        self.source_format = ".mp4"
        self.target_format = ".npy"
        self.subset_path = subset_path
        self.artefact_file = os.path.join(self.save_dir, "extraction_artefact.yaml")

        self.vid_list, self.save_urls = compute_urls(
            data_dir=self.data_dir,
            save_dir=self.save_dir,
            source_format=self.source_format,
            target_format=self.target_format,
            subset_path=self.subset_path,
        )
        self.arch_configs = load_yaml(arch_config_path)
        self.print_class()

    def sanity_check(self) -> None:
        for path in [
            self.ckpt_path,
            self.data_dir,
            self.save_dir,
        ]:
            assert_source_path_is_found(url=path)

        if assert_cuda_device_found(device=self.device):
            self.device = [self.device]
        else:
            self.device = ["cuda"]

    def extract_features(self):
        raise NotImplementedError

    def batch_extract(self):
        raise NotImplementedError

    def run(self):
        run_process(f"Extracting features from videos in {self.data_dir}...")
        if assert_files_left_to_process(self.vid_list):
            self.batch_extract()
            save_yaml(self.arch_configs, self.artefact_file)
