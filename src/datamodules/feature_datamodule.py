from src.utils import (
    bring_colon_dataset_csv,
    bring_dataset_colontest2_csv,
    bring_gastric_dataset_csv,
)
import pandas as pd
from src.datamodules.colon_datamodule import ColonDataset, ColonDataModule

from src.datamodules.colon_datamodule_aug2 import ColonDataset as ColonDataset2, ColonDataModule as ColonDataModule2
from src.datamodules.colon_test2_datamodule import ColonTestDataset as Test2, ColonTestDataModule, prepare_colon_wsi_patch
from src.datamodules.colon_all_datamodule import ColonAllDataModule
from src.datamodules.harvard_datamodule import (
    HarvardDataModule,
    prepare_prostate_harvard_data,
    HarvardDataset,
)
from src.datamodules.ubc_datamodule_classification import UbcDataModule, UbcDataset, make_ubc_dataset
from src.datamodules.prostate_datamodule import ProstateDataModule, ProstateDataset
from src.datamodules.gastric_datamodule_classification import GastricDataModule, GastricDataset

from src.datamodules.K16_datamodule import K16DataModule, K16Dataset
from src.datamodules.K19_datamodule import K19DataModule, K19Dataset, prepare_colon_kather_data
from src.datamodules.agcc_datamodule_classification import AGCCDataset, AGCCDataModule, bring_agcc_dataset_csv, prepare_AGGC_V30_data
from src.datamodules.panda_datamodule import PANDADataset, PANDADataModule, prepare_panda_512_data
from src.datamodules.panda_datamodule2 import PANDADataset_Opposite, PANDADataModule_Opposite

# from src.datamodules.ubc_datamodule_classification import UbcDataset, UbcDataModule, make_ubc_dataset

"""
This module is for saving trained features
so self.test_dataset should be 'train dataset'

data ratio: Variables for how much percentage you want to get from the train dataset
"""


class ColonDataModule(ColonDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        img_size: int = 256,
        num_workers: int = 0,
        batch_size: int = 16,
        pin_memory=False,
        drop_last=False,
        data_name="colon",
        data_ratio=1.0,
    ):
        super().__init__()

    def setup(self, stage=None):
        # test for the trained dataset
        if stage != "fit" and stage is not None:
            train_df, validf_df = bring_colon_dataset_csv(
                datatype="COLON_MANUAL_512", stage=None
            )
            if self.hparams.data_ratio < 1.0:
                train_df = (
                    train_df.groupby("class")
                    .apply(lambda x: x.sample(frac=self.hparams.data_ratio))
                    .reset_index(drop=True)
                )
            self.test_dataset = ColonDataset(pd.concat([train_df, validf_df]), self.test_transform)

class ColonTestDataModule(ColonTestDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        img_size: int = 256,
        num_workers: int = 0,
        batch_size: int = 16,
        pin_memory=False,
        drop_last=False,
        data_name="colon",
        data_ratio=1.0,
    ):
        super().__init__()

    def setup(self, stage=None):
        # test for the trained dataset
        if stage != "fit" and stage is not None:
            train_df, _ = bring_colon_dataset_csv(
                datatype="COLON_MANUAL_512", stage=None
            )
            if self.hparams.data_ratio < 1.0:
                train_df = (
                    train_df.groupby("class")
                    .apply(lambda x: x.sample(frac=self.hparams.data_ratio))
                    .reset_index(drop=True)
                )

            test_set = prepare_colon_wsi_patch()
            self.test_dataset = Test2(test_set, self.test_transform)


class GastricDataModule(GastricDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        img_size: int = 256,
        num_workers: int = 8,
        batch_size: int = 16,
        pin_memory=False,
        drop_last=False,
        data_name="gastric",
        data_ratio=1.0,
    ):
        super().__init__()

    def setup(self, stage=None):
        # test for the trained dataset
        if stage != "fit" and stage is not None:
            train_df, _ = bring_gastric_dataset_csv(stage=None)
            if self.hparams.data_ratio < 1.0:
                train_df = (
                    train_df.groupby("class")
                    .apply(
                        lambda x: x.sample(
                            frac=self.hparams.data_ratio, random_state=42
                        )
                    )
                    .reset_index(drop=True)
                )

            self.test_dataset = GastricDataset(pd.concat([train_df, _]), self.test_transform)

