"""
RIST Training Script with PyTorch Lightning and Hydra.

Run with:
    python src/train.py experiment=keypointnet
    python src/train.py experiment=keypointnet data.rotate=true

Resume training:
    python src/train.py resume=results/my_experiment
"""

import os
import warnings
from pathlib import Path

# Disable DDP subprocess log files - must be set before importing lightning
os.environ["PL_SUBPROCESS_STDOUT"] = "/dev/null"
os.environ["PL_SUBPROCESS_STDERR"] = "/dev/null"

# Suppress LeafSpec deprecation warning globally (Lightning internal issue)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

import rootutils

# Setup root directory and add to pythonpath
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Suppress harmless warnings
warnings.filterwarnings("ignore", message=".*AccumulateGrad node's stream.*")
warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*is deprecated.*")

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)

CONFIG_FILENAME = "config.yaml"
WANDB_ID_FILENAME = "wandb_id.txt"


def setup_experiment(cfg: DictConfig) -> DictConfig:
    """Setup experiment directory and handle resume logic.

    Args:
        cfg: Configuration composed by Hydra.

    Returns:
        Updated configuration with experiment paths set.
    """
    # Apply run_name_postfix if provided
    if cfg.get("run_name_postfix"):
        new_run_name = f"{cfg.run_name}-{cfg.run_name_postfix}"
        OmegaConf.update(cfg, "run_name", new_run_name)

    # Handle resume: load config from existing experiment
    if cfg.get("resume"):
        resume_dir = Path(cfg.resume)
        if not resume_dir.exists():
            raise ValueError(f"Resume directory does not exist: {resume_dir}")

        config_path = resume_dir / CONFIG_FILENAME
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        # Load saved config and use its run_name
        saved_cfg = OmegaConf.load(config_path)
        OmegaConf.update(cfg, "run_name", saved_cfg.run_name)

        # Set checkpoint path to last checkpoint
        ckpt_path = resume_dir / "checkpoints" / "last.ckpt"
        if ckpt_path.exists():
            OmegaConf.update(cfg, "ckpt_path", str(ckpt_path))
            log.info(f"Resuming from checkpoint: {ckpt_path}")
        else:
            log.warning(f"No last.ckpt found in {resume_dir}/checkpoints")

        # Load wandb run ID for resuming
        wandb_id_path = resume_dir / WANDB_ID_FILENAME
        if wandb_id_path.exists():
            wandb_id = wandb_id_path.read_text().strip()
            OmegaConf.update(cfg, "logger.wandb.id", wandb_id)
            OmegaConf.update(cfg, "logger.wandb.resume", "must")
            log.info(f"Resuming wandb run: {wandb_id}")


    # Create experiment directory
    experiment_dir = Path(cfg.paths.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save config (only on new runs)
    if not cfg.get("resume"):
        config_path = experiment_dir / CONFIG_FILENAME
        OmegaConf.save(cfg, config_path)
        log.info(f"Saved config to: {config_path}")

    return cfg


def save_wandb_id(cfg: DictConfig) -> None:
    """Save wandb run ID to experiment directory for future resume."""
    import wandb

    if wandb.run is not None:
        experiment_dir = Path(cfg.paths.experiment_dir)
        wandb_id_path = experiment_dir / WANDB_ID_FILENAME
        wandb_id_path.write_text(wandb.run.id)
        log.info(f"Saved wandb run ID to: {wandb_id_path}")


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train the model.

    Args:
        cfg: Configuration composed by Hydra.

    Returns:
        Tuple containing metrics dict and dict of all instantiated objects.
    """
    # Setup experiment directory and handle resume
    cfg = setup_experiment(cfg)

    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
        # Save wandb run ID for future resume
        save_wandb_id(cfg)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best checkpoint not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    test_metrics = trainer.callback_metrics

    # Merge metrics
    metrics = {**train_metrics, **test_metrics}

    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    Args:
        cfg: Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric value for hyperparameter optimization.
    """
    # Print config
    if cfg.get("print_config"):
        print(OmegaConf.to_yaml(cfg))

    # Train the model
    metrics, _ = train(cfg)

    # Return metric value for hyperparameter optimization
    metric_value = metrics.get(cfg.get("optimized_metric"))
    return metric_value


if __name__ == "__main__":
    main()
