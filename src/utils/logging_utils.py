import logging
from typing import Any, Dict, Optional, Mapping

from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly Python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra or {})
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, *args, **kwargs) -> None:
        if self.rank_zero_only:
            _rank_zero_log(self.logger, level, msg, *args, **kwargs)
        else:
            self.logger.log(level, msg, *args, **kwargs)


@rank_zero_only
def _rank_zero_log(logger: logging.Logger, level: int, msg: str, *args, **kwargs) -> None:
    logger.log(level, msg, *args, **kwargs)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Log hyperparameters to all loggers.

    This method also controls which parameters from the config will be saved by Lightning loggers.

    Args:
        object_dict: A dictionary containing the following objects:
            - "cfg": A DictConfig object containing the main config.
            - "model": The Lightning model.
            - "trainer": The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    # Save config to hparams
    hparams["cfg"] = cfg

    # Save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # Send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
