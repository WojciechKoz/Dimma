import argparse
import json
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.datasets import MixHQDataModule  # noqa: I900
from src.models import LitDimma # noqa: I900


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/FS-Dark/stage1/6shot-fsd.yaml")
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    print(OmegaConf.to_yaml(conf))

    pl.seed_everything(conf.seed)

    dm = MixHQDataModule(config=conf.dataset)

    model = LitDimma(config=conf)

    callbacks = [
        pl.callbacks.progress.TQDMProgressBar(),
        ModelCheckpoint(
            monitor="val/psnr",
            mode="max",
            save_top_k=conf.logger.save_top_k,
            save_last=False,
            auto_insert_metric_name=False,
            filename=conf.name,
            dirpath=f'{conf.logger.checkpoint_dir}/{conf.name}',
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # uncomment to use wandb
    """
    logger = WandbLogger(
        entity="your-entity", 
        project="dimma", 
        name=conf.name,
        save_dir="logs",
    )
    """

    trainer = pl.Trainer(
        accelerator=conf.device,
        devices=1,
        callbacks=callbacks,
        # logger=logger, # uncomment to use wandb
        max_steps=conf.iter,
        val_check_interval=conf.eval_freq,
    )
    trainer.fit(model, dm)

    # load best model
    model = LitDimma.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path, config=conf
    )

    # test
    output = trainer.test(model, datamodule=dm)

    # save results as json
    with open(f'{conf.logger.checkpoint_dir}/{conf.name}/results.json', 'w') as f:
        json.dump(output[0], f, indent=4)