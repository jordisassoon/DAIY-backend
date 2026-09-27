# python imports
import os
from pprint import pprint

from tqdm import tqdm
from tools.print_helper import process_output, process_output_builder
from tools.save import save_yaml
import wandb

# our code
from libs.utils import valid_one_epoch
from tools.tridet_loaders import (
    load_configs,
    load_model,
    load_train_data,
    load_val_data,
)
from libs.utils import (
    ModelEma,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    train_one_epoch,
    valid_one_epoch,
)

from processes.training.trainer import Train


class TridetTrainer(Train):
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
        learning_rate: float,
        epochs: int,
        device: str,
    ) -> None:
        self.name = "Tridet Trainer"
        super().__init__(
            model_config=model_config,
            dataset_config=dataset_config,
            validation_step=validation_step,
            ckpt_freq=ckpt_freq,
            data_dir=data_dir,
            save_dir=save_dir,
            annotations_file=annotations_file,
            batch_size=batch_size,
            device=device,
        )
        self.cfg = load_configs(
            model_config=self.model_config,
            dataset_config=self.dataset_config,
            annotations_file=self.annotations_file,
            data_dir=self.data_dir,
            backbone_type="videomae2",
            batch_size=self.batch_size,
            artefact_file=self.artefact_file,
            learning_rate=learning_rate,
            epochs=epochs,
            device=self.device,
        )

    def train(self) -> None:
        train_loader = load_train_data(cfg=self.cfg, rng_generator=self.rng_generator)
        val_loader, det_eval = load_val_data(cfg=self.cfg)
        model = load_model(cfg=self.cfg, device=self.device)

        optimizer = make_optimizer(model=model, optimizer_config=self.cfg["opt"])
        scheduler = make_scheduler(
            optimizer=optimizer,
            optimizer_config=self.cfg["opt"],
            num_iters_per_epoch=len(train_loader),
        )
        model_ema = ModelEma(model)

        # wandb.init(
        #     project="tal-backend",
        #     config=self.cfg,
        # )

        model_name = self.cfg["model_name"]
        process_output(f"\nStart training {model_name} model...")

        max_epochs = self.cfg["opt"].get(
            "early_stop_epochs",
            self.cfg["opt"]["epochs"] + self.cfg["opt"]["warmup_epochs"],
        )

        mean_ap = 0.0

        for epoch in tqdm(
            iterable=range(max_epochs),
            desc=process_output_builder(f"Training {model_name}"),
        ):
            train_one_epoch(
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                curr_epoch=epoch,
                model_ema=model_ema,
                clip_grad_l2norm=self.cfg["train_cfg"]["clip_grad_l2norm"],
                print_freq=10,
            )

            if (
                (self.validation_step > 0)
                and (epoch % self.validation_step == 0)
                and (epoch > 0)
            ):
                mean_ap = valid_one_epoch(
                    val_loader,
                    model,
                    -1,
                    evaluator=det_eval,
                    output_file=None,
                    ext_score_file=self.cfg["test_cfg"]["ext_score_file"],
                    tb_writer=None,
                    print_freq=10,
                )
                # wandb.log({"epoch": epoch, "mAP": mean_ap})

            if (epoch == max_epochs - 1) or (
                (self.ckpt_freq > 0) and (epoch % self.ckpt_freq == 0) and (epoch > 0)
            ):
                save_states = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "state_dict_ema": model_ema.module.state_dict(),
                }
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=self.save_dir,
                    file_name="epoch_{:03d}.pth.tar".format(epoch),
                )

        process_output(f"Starting validating {model_name} model...")
        mean_ap = valid_one_epoch(
            val_loader,
            model,
            -1,
            evaluator=det_eval,
            output_file=None,
            ext_score_file=self.cfg["test_cfg"]["ext_score_file"],
            tb_writer=None,
            print_freq=10,
        )

        wandb.finish()

        process_output("All done!")
        process_output(f"Final mAP score:{mean_ap}")
