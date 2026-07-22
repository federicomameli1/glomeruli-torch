"""Reinhard stain-normalise a folder of glomerulus crops toward a common target,
to kill the staining/colour axis before embedding. If the clustering was separating
by stain (high slide_nmi in analyze_clusters), re-embedding these normalised crops
should drop that confound and let morphology drive the clusters.

Target = the dataset's own average LAB mean/std (a neutral consensus), so every crop
is pulled toward the same colour centre. Run in an env with scikit-image + Pillow
(the TF `glomeruli` env has both).

  python stain_normalize_crops.py --crops-dir <crops> --out <crops_norm>
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb


def lab_of(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return rgb2lab(rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.crops_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"no *.png in {args.crops_dir}")

    # pass 1: consensus target = mean over crops of each crop's LAB mean/std
    means, stds = [], []
    for p in paths:
        lab = lab_of(p).reshape(-1, 3)
        means.append(lab.mean(0))
        stds.append(lab.std(0))
    t_mean, t_std = np.mean(means, 0), np.mean(stds, 0)
    print(f"target LAB mean={t_mean.round(2)} std={t_std.round(2)}")

    # pass 2: map each crop's LAB stats onto the target, write out
    os.makedirs(args.out, exist_ok=True)
    for p in paths:
        lab = lab_of(p)
        m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)
        s = np.where(s < 1e-6, 1e-6, s)
        norm = (lab - m) / s * t_std + t_mean
        rgb = np.clip(lab2rgb(norm), 0, 1)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(args.out, os.path.basename(p)))
    print(f"normalised {len(paths)} crops -> {args.out}")


if __name__ == "__main__":
    main()
