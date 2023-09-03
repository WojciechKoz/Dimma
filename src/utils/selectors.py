import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from enum import Enum


class Loss(str, Enum):
    L1 = "l1"
    MSE = "mse"
    NLL = "nll"


class Optimizer(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class Scheduler(str, Enum):
    CONSTANT = "constant"
    COSINE = "cosine"
    ONE_CYCLE = "one_cycle"


def get_optimizers(model: pl.LightningModule, config):
    if config.name == Optimizer.ADAMW:
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.name == Optimizer.ADAM:
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif config.name == Optimizer.SGD:
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.name} is not implemented.")

    if config.scheduler is None or config.scheduler.name == Scheduler.CONSTANT:
        return optimizer

    if config.scheduler.name == Scheduler.ONE_CYCLE:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config.lr,
            total_steps=model.trainer.estimated_stepping_batches,
        )
    elif config.scheduler.name == Scheduler.COSINE:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=model.trainer.estimated_stepping_batches,
        )
    else:
        raise NotImplementedError(
            f"Scheduler {config.scheduler.name} is not implemented."
        )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": config.scheduler.frequency,
        },
    }


def get_loss(config, **kwargs):
    if config.name == Loss.L1:
        return nn.L1Loss(reduction=config.reduction)
    if config.name == Loss.MSE:
        return torchmetrics.MeanSquaredError()
    else:
        raise NotImplementedError(f"Loss {config.name} is not implemented.")
