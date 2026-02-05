import sys
from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig

# FIXME: messing with sys.path is a bad idea. Factor this out.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    from yolo.utils.logging_utils import setup
    callbacks, loggers, save_path = setup(cfg)

    from lightning import Trainer
    from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel

    # Add ModelCheckpoint callback to save best model
    if getattr(cfg, "save_best", True) and cfg.task.task == "train":
        from lightning.pytorch.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_path / "checkpoints",
            filename="best-{epoch:04d}-{map:.4f}",
            monitor="map",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Parse timeout for max_time parameter
    timeout = getattr(cfg, "timeout", None)
    max_time = None
    if timeout is not None and timeout != "null" and timeout != "":
        try:
            timeout_seconds = float(timeout)
            # Convert seconds to timedelta dict for Lightning
            hours = int(timeout_seconds // 3600)
            minutes = int((timeout_seconds % 3600) // 60)
            seconds = int(timeout_seconds % 60)
            max_time = {"hours": hours, "minutes": minutes, "seconds": seconds}
        except (ValueError, TypeError):
            pass

    trainer = Trainer(
        accelerator=cfg.accelerator,
        max_epochs=getattr(cfg.task, "epoch", None),
        max_time=max_time,
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="value",
        # deterministic=True,
        enable_progress_bar=not getattr(cfg, "quite", False),
        default_root_dir=save_path,
        num_sanity_val_steps=0,
    )

    if cfg.task.task == "train":
        model = TrainModel(cfg)
        trainer.fit(model)
    if cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == "inference":
        model = InferenceModel(cfg)
        trainer.predict(model)


if __name__ == "__main__":
    main()
