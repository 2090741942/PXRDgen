from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from typing import List
from omegaconf import DictConfig
from lightning.pytorch.loggers import WandbLogger
import wandb


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "model_checkpoints" in cfg.logging:
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.logging.model_checkpoints.dirpath,
                monitor=cfg.logging.model_checkpoints.monitor,
                mode=cfg.logging.model_checkpoints.mode,
                save_top_k=cfg.logging.model_checkpoints.save_top_k,
                verbose=cfg.logging.model_checkpoints.verbose,
                save_last=cfg.logging.model_checkpoints.save_last,
            )
        )

        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.logging.model_checkpoints.dirpath,
                filename='last_one'
            )
        )

    if "lr_monitor" in cfg.logging:
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    # if "early_stopping" in cfg.logging:
    #     callbacks.append(
    #         EarlyStopping(
    #             monitor=cfg.logging.early_stopping.monitor,
    #             mode=cfg.logging.early_stopping.mode,
    #             patience=cfg.logging.early_stopping.patience,
    #             verbose=cfg.logging.early_stopping.verbose,
    #         )
    #     )

    return callbacks


def Wandb_logger(cfg: DictConfig):
    wandb_logger = None
    if "wandb" in cfg.logging:
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
    
    return wandb_logger









