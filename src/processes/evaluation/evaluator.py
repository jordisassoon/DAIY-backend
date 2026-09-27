from tools.print_helper import run_process
from tools.assertions import *
from processes.process import Process


class Evaluate(Process):
    def __init__(
        self,
        training_config: str,
        data_dir: str,
        ckpt_path: str,
        device: str,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.training_config = training_config
        self.data_dir = data_dir
        self.ckpt_path = ckpt_path
        self.device = device
        self.batch_size = batch_size

        self.sanity_check()
        self.print_class()

    def eval(self) -> None:
        raise NotImplementedError

    def sanity_check(self) -> None:
        for path in [
            self.training_config,
            self.ckpt_path,
            self.data_dir,
        ]:
            assert_source_path_is_found(url=path)

        if assert_cuda_device_found(device=self.device):
            self.device = [self.device]
        else:
            self.device = ["cuda"]

    def run(self) -> None:
        run_process(f"Evaluating the model on features contained in {self.data_dir}...")
        self.eval()
