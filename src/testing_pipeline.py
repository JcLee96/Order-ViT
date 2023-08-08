import os
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
    Callback,
)
from pytorch_lightning.loggers import LightningLoggerBase
import yaml
from src import utils


log = utils.get_logger(__name__)


def test(config: DictConfig) -> None:
    """Contains minimal example of the testing pipeline.
    Evaluates given checkpoint on a testset.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        None
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(
            hydra.utils.get_original_cwd(), config.ckpt_path
        )

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    if len(config["logger"]) > 0 and list(config["logger"].keys())[0] == "wandb":
        config.logger.wandb.name = (
            (
                (
                    (
                            (
                                    (
                                        (
                                            f"{config.model.name}_{config.model.key}{str(config.model.threshold)}_threshold_"
                                            + config.model.sampling
                                        )
                                        + "_Sampling_"
                                    )
                                    + str(config.model.num_sample)
                                    + "_weighted_sum_"
                            )
                            + str(config.model.vote_score_way)
                            + "_vote_score_way_"
                    )
                    + str(config.model.weighted_sum)
                    + "_decide_total_probs_"
                )
                + str(config.model.decide_by_total_probs)
                + "_sampling_"
            )
            + str(config.datamodule.data_ratio)
            + "_seed_"
        ) + str(config.seed)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, logger=logger, callbacks=callbacks
    )

    # Log hyperparameters
    if logger:
        trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})

    log.info("Starting testing!")
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.ckpt_path,
    )
    import wandb

    wandb.finish()