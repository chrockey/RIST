"""
RIST Evaluation Script with PyTorch Lightning and Hydra.

Run with:
    python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
    python src/eval.py experiment=keypointnet_airplane ckpt_path=/path/to/checkpoint.ckpt
"""

import rootutils

# Setup root directory and add to pythonpath
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.train import WANDB_ID_FILENAME
from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
)

log = RankedLogger(__name__, rank_zero_only=True)

EXPERIMENT_CONFIG_FILENAME = "config.yaml"


def load_experiment_config(cfg: DictConfig) -> DictConfig:
    """Load and merge experiment config from checkpoint directory.

    Loads the saved config.yaml from the experiment directory and merges it
    with the eval config. This ensures model/data settings match training.

    Args:
        cfg: Configuration composed by Hydra.

    Returns:
        Updated configuration with experiment settings merged.
    """
    if not cfg.get("ckpt_path"):
        return cfg

    # Find experiment directory from checkpoint path
    # Expected structure: results/<experiment>/checkpoints/<ckpt>.ckpt
    ckpt_path = Path(cfg.ckpt_path)
    experiment_dir = ckpt_path.parent.parent

    # Load saved experiment config
    experiment_config_path = experiment_dir / EXPERIMENT_CONFIG_FILENAME
    if experiment_config_path.exists():
        log.info(f"Loading experiment config from: {experiment_config_path}")
        experiment_cfg = OmegaConf.load(experiment_config_path)

        # Merge model and data configs from experiment (training config takes priority)
        # Disable struct mode to allow new keys from saved config
        OmegaConf.set_struct(cfg, False)
        if "model" in experiment_cfg:
            cfg.model = experiment_cfg.model
        if "data" in experiment_cfg:
            # Keep eval-specific overrides (like rotate: true)
            cfg.data = OmegaConf.merge(experiment_cfg.data, cfg.data)
        OmegaConf.set_struct(cfg, True)
    else:
        log.warning(f"Experiment config not found at {experiment_config_path}")

    # Setup wandb resume
    wandb_id_path = experiment_dir / WANDB_ID_FILENAME
    if wandb_id_path.exists():
        wandb_id = wandb_id_path.read_text().strip()
        OmegaConf.update(cfg, "logger.wandb.id", wandb_id)
        OmegaConf.update(cfg, "logger.wandb.resume", "allow")
        # Preserve original run name from training
        if "run_name" in experiment_cfg:
            OmegaConf.update(cfg, "logger.wandb.name", experiment_cfg.run_name)
        log.info(f"Resuming wandb run: {wandb_id}")
    else:
        log.warning(f"wandb_id.txt not found in {experiment_dir}, creating new run")

    return cfg


def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate the model.

    Args:
        cfg: Configuration composed by Hydra.

    Returns:
        Tuple containing metrics dict and dict of all instantiated objects.
    """
    assert cfg.ckpt_path, "Checkpoint path is required for evaluation!"

    # Load experiment config and setup wandb resume
    cfg = load_experiment_config(cfg)

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

    log.info("Starting testing!")
    log.info(f"Loading checkpoint: {cfg.ckpt_path}")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, weights_only=False)

    metrics = trainer.callback_metrics

    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg: Configuration composed by Hydra.
    """
    metrics, _ = evaluate(cfg)

    # Print results
    log.info("=" * 50)
    log.info("Evaluation Results:")
    log.info("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            log.info(f"{key}: {value:.4f}")
        else:
            log.info(f"{key}: {value}")


if __name__ == "__main__":
    main()
