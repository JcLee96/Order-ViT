"""Colon cancer grading datasets, augmentation and data loaders."""
import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = ["BN", "WD", "MD", "PD"]


def resize(image, size):
    """Nearest-neighbour resize to a square `size` x `size` image."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)


def random_affine(image):
    """Random rotation (+-45 deg) and shear (+-16 deg) with symmetric border."""
    h, w = image.shape[:2]
    angle = random.uniform(-45, 45)
    shear = np.tan(np.deg2rad(random.uniform(-16, 16)))
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    shear_mat = np.array([[1, shear, -shear * h / 2], [0, 1, 0]], dtype=np.float64)
    mat = np.vstack([rot, [0, 0, 1]]) @ np.vstack([shear_mat, [0, 0, 1]])
    interp = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR])
    return cv2.warpAffine(image, mat[:2], (w, h), flags=interp, borderMode=cv2.BORDER_REFLECT)


def random_blur(image):
    """One of Gaussian, average or median blur."""
    choice = random.randint(0, 2)
    if choice == 0:
        return cv2.GaussianBlur(image, (0, 0), sigmaX=random.uniform(1e-3, 3.0))
    if choice == 1:
        k = random.randint(2, 7)
        return cv2.blur(image, (k, k))
    return cv2.medianBlur(image, random.choice([3, 5, 7, 9, 11]))


def gaussian_noise(image):
    """Additive Gaussian noise with std up to 0.05 * 255."""
    noise = np.random.normal(0, random.uniform(0, 0.05 * 255), image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def pixel_dropout(image):
    """Zero out 1-10% of the pixels."""
    mask = np.random.rand(*image.shape[:2]) < random.uniform(0.01, 0.1)
    out = image.copy()
    out[mask] = 0
    return out


def hue_saturation(image):
    """Shift hue and saturation by up to +-20."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-10, 10)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-20, 20), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def linear_contrast(image):
    """Scale contrast around 128 by a factor in [0.5, 2.0]."""
    alpha = random.uniform(0.5, 2.0)
    return np.clip(128 + alpha * (image.astype(np.float32) - 128), 0, 255).astype(np.uint8)


def train_augment(image, size):
    """Training augmentation: resize, flips, 20% affine, then 0-5 photometric ops."""
    image = resize(image, size)
    if random.random() < 0.5:
        image = image[:, ::-1]
    if random.random() < 0.5:
        image = image[::-1]
    image = np.ascontiguousarray(image)
    if random.random() < 0.2:
        image = random_affine(image)
    ops = [random_blur, gaussian_noise, pixel_dropout, hue_saturation, linear_contrast]
    for op in random.sample(ops, random.randint(0, len(ops))):
        image = op(image)
    return image


class PatchDataset(Dataset):
    """Image patches given as (path, label) pairs; returns uint8 CHW tensors in [0, 255]."""

    def __init__(self, items, img_size, train=False):
        self.items = list(items)
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = train_augment(image, self.img_size) if self.train else resize(image, self.img_size)
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1), int(label)


def read_split_csv(csv_path, image_root):
    """Read a split CSV with columns `path,class`; paths are relative to `image_root`."""
    df = pd.read_csv(csv_path)
    return [(os.path.join(image_root, p), int(c)) for p, c in zip(df["path"], df["class"])]


def read_label_folder(image_root, pattern="*/*/*.png"):
    """Collect images whose file name ends with `_<label+1>.png` (colon Test II layout)."""
    files = sorted(glob.glob(os.path.join(image_root, pattern)))
    return [(f, int(os.path.splitext(f)[0].split("_")[-1]) - 1) for f in files]


def load_split(data_cfg, split):
    """Return (path, label) pairs for `train`, `valid`, `test` or `test2`."""
    if split == "test2":
        return read_label_folder(data_cfg["test2_root"])
    return read_split_csv(os.path.join(data_cfg["split_dir"], f"{split}.csv"), data_cfg["image_root"])


def make_loader(items, data_cfg, train=False, shuffle=None):
    """Build a DataLoader; training loaders shuffle and drop the last incomplete batch."""
    dataset = PatchDataset(items, data_cfg["img_size"], train=train)
    return DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=train if shuffle is None else shuffle,
        drop_last=train,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
