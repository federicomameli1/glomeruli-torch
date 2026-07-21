"""Glomerulus segmentation dataset — reads the patch PNGs produced by the existing
preprocessing (data/dataset/{split}/{img,mask}/*.png). We reuse that OUTPUT, so no
openslide rewrite is needed."""
import glob
import os

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_tf(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),           # together: the D4 group
        A.HueSaturationValue(10, 15, 10, p=0.5),  # cheap stain jitter stand-in
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # ponytail: swap for Lunit stats if they differ
        ToTensorV2(),
    ])


def eval_tf(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ])


class GlomDataset(Dataset):
    def __init__(self, split_dir, size=512, train=False):
        self.imgs = sorted(glob.glob(os.path.join(split_dir, "img", "*.png")))
        if not self.imgs:
            raise FileNotFoundError(f"no img/*.png in {split_dir}")
        self.mask_dir = os.path.join(split_dir, "mask")
        self.tf = train_tf(size) if train else eval_tf(size)

    def __len__(self):
        return len(self.imgs)

    def _mask_path(self, name):
        # preprocess writes mask/<same-name>.png; fall back to <stem>_mask.png just in case
        p = os.path.join(self.mask_dir, name)
        if os.path.exists(p):
            return p
        return os.path.join(self.mask_dir, os.path.splitext(name)[0] + "_mask.png")

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.imgs[i]), cv2.COLOR_BGR2RGB)
        name = os.path.basename(self.imgs[i])
        m = cv2.imread(self._mask_path(name), cv2.IMREAD_GRAYSCALE)
        mask = (m > 0).astype("float32")          # glomerulus = 1
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].unsqueeze(0)   # mask -> (1, H, W)
