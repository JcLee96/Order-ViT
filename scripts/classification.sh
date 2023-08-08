#!/bin/bash

# colon test1 & gastric dataset classification
python ../train.py model.name='vit_base_r50_s16_384' seed=42 model=classify4.yaml model.loss_weight=1.0 datamodule.data_ratio=1.0 datamodule=colon.yaml model.scheduler='CosineAnnealingWarmRestarts' logger.wandb.project='colon_Analysis' logger.wandb.tags=['classify'] datamodule.num_workers=8 trainer.devices=\'0,1\' datamodule.batch_size=16
python ../train.py model.name='vit_base_r50_s16_384' seed=42 model=classify4.yaml model.loss_weight=1.0 datamodule.data_ratio=1.0 datamodule=gastric.yaml model.scheduler='CosineAnnealingWarmRestarts' logger.wandb.project='colon_Analysis' logger.wandb.tags=['classify'] datamodule.num_workers=8 trainer.devices=\'0,1\' datamodule.batch_size=16

# colon test2 dataset classification
python /path/to/LJC/colon_compare/train_test.py model.name='vit_base_r50_s16_384' seed=42 model=classify4.yaml datamodule.data_ratio=1.0 datamodule=colon_test2.yaml model.scheduler='CosineAnnealingWarmRestarts' logger.wandb.project='colon_test2_Analysis' logger.wandb.tags=['classification','colon_test2'] ckpt_path="../scripts/scripts/logs/experiments/runs/colon/classifycompare/vit_base_r50_s16_384/2023-05-09_00-45-13/checkpoints/epoch_*.ckpt" datamodule.num_workers=8 trainer.devices=\'0,1\' datamodule.batch_size=16

