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
    """Either glob a split_dir (img/ + mask/) or take an explicit image_paths list
    (used by LOSO, where patches from several split dirs are pooled per slide)."""
    def __init__(self, split_dir=None, size=512, train=False, image_paths=None):
        if image_paths is not None:
            self.imgs = sorted(image_paths)
        else:
            self.imgs = sorted(glob.glob(os.path.join(split_dir, "img", "*.png")))
        if not self.imgs:
            raise FileNotFoundError(f"no images ({split_dir or 'image_paths'})")
        self.tf = train_tf(size) if train else eval_tf(size)

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def _mask_path(img_path):
        # mask mirrors the image: .../img/NAME.png -> .../mask/NAME.png
        p = img_path.replace("/img/", "/mask/").replace(os.sep + "img" + os.sep,
                                                        os.sep + "mask" + os.sep)
        if os.path.exists(p):
            return p
        base, ext = os.path.splitext(p)                # fall back to <stem>_mask.png
        return base + "_mask" + ext

    def __getitem__(self, i):
        img_path = self.imgs[i]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        m = cv2.imread(self._mask_path(img_path), cv2.IMREAD_GRAYSCALE)
        mask = (m > 0).astype("float32")          # glomerulus = 1
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].unsqueeze(0)   # mask -> (1, H, W)
