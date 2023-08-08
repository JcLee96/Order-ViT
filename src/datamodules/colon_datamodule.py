import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2

from imgaug import augmenters as iaa
import imgaug as ia
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.utils import bring_colon_dataset_csv, bring_dataset_colontest2_csv
from cv2 import cv2
import glob


class ColonDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_id = df["path"].values
        self.labels = df["class"].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_id[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)

        # If Using albumentation Augmentation Argorithm image["image"] else torch.from_numpy(image.copy()).permute(2,0,1)
        return torch.from_numpy(image.copy()).permute(2,0,1), label


class ColonDataModule(LightningDataModule):
    """
    There is 1 dataset of colon

    - (A) is the datasets that has Training Validation Testing I
    https://www.sciencedirect.com/science/article/pii/S1361841521002516

    This datamodule use (A) for train, validation, test
    """

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
        self.save_hyperparameters(logger=False)
        resize_value = 256 if self.hparams.img_size == 224 else 456

        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        self.train_transform = iaa.Sequential(
                [
                    # apply the following augmenters to most images

                    iaa.Resize({"height": self.hparams.img_size, "width": self.hparams.img_size},
                               interpolation='nearest'),

                    iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                    iaa.Flipud(0.5),  # vertically flip 50% of all images
                    sometimes(iaa.Affine(
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                        mode='symmetric'
                        # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                    # execute 0 to 5 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((0, 5),
                               [
                                   iaa.OneOf([
                                       iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                       iaa.AverageBlur(k=(2, 7)),
                                       # blur image using local means with kernel sizes between 2 and 7
                                       iaa.MedianBlur(k=(3, 11)),
                                       # blur image using local medians with kernel sizes between 2 and 7
                                   ]),
                                   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                                   # add gaussian noise to images
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # change brightness of images (by -10 to 10 of original value)
                                   iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                                   iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               ],
                               random_order=True
                               )
                ],
                random_order=True
            )
        self.test_transform = iaa.Resize({"height": self.hparams.img_size, "width": self.hparams.img_size},
                               interpolation='nearest')

    @property
    def num_classes(self) -> int:
        return 4

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_df, valid_df = bring_colon_dataset_csv(
                datatype="COLON_MANUAL_512", stage=None
            )
            if self.hparams.data_ratio < 1.0:
                train_df = (
                    train_df.groupby("class")
                    .apply(lambda x: x.sample(frac=self.hparams.data_ratio))
                    .reset_index(drop=True)
                )

            self.train_dataset = ColonDataset(train_df, self.train_transform)
            self.valid_dataset = ColonDataset(valid_df, self.test_transform)

        else:
            train_df, valid_df = bring_colon_dataset_csv(stage=None)
            # self.test_dataset = ColonDataset(valid_df, self.test_transform)
            test_df = bring_colon_dataset_csv(stage="test")
            self.test_dataset = ColonDataset(test_df, self.test_transform)
            # test_df2 = bring_dataset_colontest2_csv(stage="test")
            # self.test_dataset = ColonDataset(
            #     test_df2, self.test_transform
            # )


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False, # True
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
            shuffle=False,
        )
