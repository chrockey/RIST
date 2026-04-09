from typing import List, Optional

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils.logging_utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: Optional[DictConfig]) -> List[Callback]:
    """Instantiate callbacks from config.

    Args:
        callbacks_cfg: A DictConfig object containing callback configurations.

    Returns:
        A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: Optional[DictConfig]) -> List[Logger]:
    """Instantiate loggers from config.

    Args:
        logger_cfg: A DictConfig object containing logger configurations.

    Returns:
        A list of instantiated loggers.
    """
    loggers: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers
