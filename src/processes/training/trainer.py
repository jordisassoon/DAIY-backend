from tools.assertions import *
from processes.process import Process
from tools.print_helper import run_process
from tools.save import save_yaml


class Train(Process):
    def __init__(
        self,
        model_config: str,
        dataset_config: str,
        data_dir: str,
        save_dir: str,
        annotations_file: str,
        batch_size: int,
        validation_step: int,
        ckpt_freq: int,
        device: str,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.cfg = None
        self.save_dir = save_dir
        self.data_dir = data_dir
        self.validation_step = validation_step
        self.ckpt_freq = ckpt_freq
        self.device = device
        self.source_format = ".npy"
        self.target_format = ".pth.tar"
        self.artefact_file = os.path.join(self.data_dir, "extraction_artefact.yaml")
        self.annotations_file = annotations_file
        self.batch_size = batch_size

        self.sanity_check()
        self.print_class()

    def train(self) -> None:
        raise NotImplementedError

    def sanity_check(self) -> None:
        for path in [
            self.model_config,
            self.dataset_config,
            self.data_dir,
            self.save_dir,
        ]:
            assert_source_path_is_found(url=path)

        if assert_cuda_device_found(device=self.device):
            self.device = [self.device]
        else:
            self.device = ["cuda"]

    def run(self) -> None:
        run_process(f"Training the model on features contained in {self.data_dir}...")
        if not assert_checkpoints_already_exist(self.save_dir, self.target_format):
            self.train()
            save_yaml(self.cfg, os.path.join(self.save_dir, "config.yaml"))
