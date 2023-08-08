#!/bin/bash

python ../test.py seed=42 model.seed=42 model.name='vit_base_r50_s16_384' model=savefeatures.yaml datamodule=colon_feature.yaml logger=none datamodule.num_workers=8 trainer.devices=[0] ckpt_path="../scripts/scripts/logs/experiments/runs/gastric/classifycompare/vit_base_r50_s16_384/2023-05-09_00-45-13/checkpoints/epoch_*.ckpt"
python ../test.py seed=42 model.seed=42 model.name='vit_base_r50_s16_384' model=savefeatures.yaml datamodule=gastric_feature.yaml logger=none datamodule.num_workers=8 trainer.devices=[0] ckpt_path="../scripts/scripts/logs/experiments/runs/gastric/classifycompare/vit_base_r50_s16_384/2023-05-09_00-45-13/checkpoints/epoch_*.ckpt"
