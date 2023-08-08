from torch.utils.data import Dataset
from src.utils import bring_colon_dataset_csv
from cv2 import cv2
import glob
from src.datamodules.colon_datamodule import ColonDataset, ColonDataModule
from imgaug import augmenters as iaa
import imgaug as ia
import torch

class ColonTestDataset(Dataset):
    def __init__(self, pair_list, transform=None):
        self.pair_list = pair_list
        self.transform = transform

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        image = cv2.imread(pair[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = pair[1]

        if self.transform:
            image = self.transform(image=image)

        return torch.from_numpy(image.copy()).permute(2,0,1), label

class ColonTestDataModule(ColonDataModule):
    """

    - (B) is the "Testing II" that were used for test dataset refered to this paper
    https://www.sciencedirect.com/science/article/pii/S1361841521002516

    This datamodule use (B) for test
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

        self.test_transform = iaa.Resize({"height": self.hparams.img_size, "width": self.hparams.img_size},
                                         interpolation='nearest')


    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_df, valid_df = bring_colon_dataset_csv(
                datatype="COLON_MANUAL_512", stage=None
            )
            self.train_dataset = ColonDataset(train_df, self.train_transform)

        else:
            test_set = prepare_colon_wsi_patch()
            self.test_dataset = ColonTestDataset(test_set, self.test_transform)


def prepare_colon_wsi_patch():
    def load_data_info_from_list(data_dir):

        file_list = glob.glob(f"{data_dir}/*/*/*.png")
        label_list = [
            int(file_path.split("_")[-1].split(".")[0]) - 1 for file_path in file_list
        ]

        return list(zip(file_list, label_list))

    data_dir = "/data2/lju/colon/test_2/colon_45WSIs_1144_08_step05_05"

    return load_data_info_from_list(data_dir)
