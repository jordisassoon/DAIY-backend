import time

from libs.utils import valid_one_epoch
from tools.print_helper import process_output
from tools.tridet_loaders import load_for_inference, load_model_from_ckpt, load_val_data
from tools.dynamic_loaders import load_yaml
from processes.evaluation.evaluator import Evaluate


class TridetEvaluator(Evaluate):
    def __init__(
        self,
        training_config: str,
        data_dir: str,
        ckpt_path: str,
        device: str,
        batch_size: int,
    ) -> None:
        self.name = "Tridet Evaluator"
        super().__init__(
            training_config=training_config,
            data_dir=data_dir,
            ckpt_path=ckpt_path,
            device=device,
            batch_size=batch_size,
        )
        self.cfg = load_for_inference(
            self.training_config, self.data_dir, self.batch_size, self.device
        )

    def eval(self) -> None:
        val_loader, det_eval = load_val_data(self.cfg)
        model = load_model_from_ckpt(self.cfg, self.ckpt_path, self.device)
        model.eval()

        process_output("\nStart testing model {:s} ...".format(self.cfg["model_name"]))
        start = time.time()
        _ = valid_one_epoch(
            val_loader,
            model,
            -1,
            evaluator=det_eval,
            output_file=None,
            ext_score_file=self.cfg["test_cfg"]["ext_score_file"],
            tb_writer=None,
            print_freq=30,
        )
        end = time.time()
        process_output("All done! Total time: {:0.2f} sec".format(end - start))
        return
