"""Adapter: HuBMAP 'Hacking the Kidney' -> our {split}/{img,mask}/ PNG layout.

HuBMAP gives huge PAS kidney WSIs (TIFF) + train.csv with one run-length-encoded
glomerulus mask per WSI. This tiles each WSI, cuts the matching mask tile from the
decoded RLE, keeps tiles that contain glomeruli (plus a few tissue negatives),
resizes, and writes PNG img/mask pairs, split BY WSI id (leave-slides-out — no leakage).

Only the ~8 train WSIs have public masks; the test WSIs do not, so we only use train.csv.

You download the data yourself (Kaggle rules + auth — see README):
  kaggle competitions download -c hubmap-kidney-segmentation

RLE ORIENTATION: HuBMAP's flatten order is easy to get wrong (mask ends up transposed).
This writes a few overlay PNGs to <out>/_sanity/ — LOOK at them: the mask must sit on
the glomeruli. If it's scrambled/transposed, rerun with the other --rle-order.
"""
import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd
import tifffile
import zarr
from tqdm import tqdm


def rle_decode(rle, shape, order):
    s = np.asarray(str(rle).split(), dtype=np.int64)
    starts, lengths = s[0::2] - 1, s[1::2]
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, ln in zip(starts, lengths):
        flat[lo:lo + ln] = 1
    return flat.reshape(shape, order=order)


def open_wsi(path):
    """Return (H, W, get) where get(y0,y1,x0,x1) reads an RGB window lazily."""
    z = zarr.open(tifffile.imread(path, aszarr=True), mode="r")
    arr = z if hasattr(z, "shape") else z[0]        # level 0 of a pyramid
    if arr.ndim == 3 and arr.shape[0] in (3, 4):    # CHW
        h, w = arr.shape[1], arr.shape[2]
        get = lambda y0, y1, x0, x1: np.transpose(np.asarray(arr[:3, y0:y1, x0:x1]), (1, 2, 0))
    else:                                           # HWC
        h, w = arr.shape[0], arr.shape[1]
        get = lambda y0, y1, x0, x1: np.asarray(arr[y0:y1, x0:x1, :3])
    return h, w, get


def is_tissue(tile, white=220, frac=0.4):
    return (tile.mean(axis=2) < white).mean() > frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubmap-dir", required=True, help="dir with train/ and train.csv")
    ap.add_argument("--out", required=True, help="output root -> {train,validation,test}/{img,mask}")
    ap.add_argument("--tile", type=int, default=1024, help="tile size read at level 0")
    ap.add_argument("--out-size", type=int, default=512, help="patch size written to disk")
    ap.add_argument("--stride", type=int, default=None, help="default = tile (no overlap)")
    ap.add_argument("--rle-order", default="F", choices=["F", "C"])
    ap.add_argument("--neg-ratio", type=float, default=0.15, help="keep this fraction of glom-free tissue tiles")
    ap.add_argument("--val-ids", default="", help="comma-separated WSI ids for validation")
    ap.add_argument("--test-ids", default="", help="comma-separated WSI ids for test")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    stride = args.stride or args.tile
    df = pd.read_csv(os.path.join(args.hubmap_dir, "train.csv"))
    val_ids = set(filter(None, args.val_ids.split(",")))
    test_ids = set(filter(None, args.test_ids.split(",")))
    sanity_dir = os.path.join(args.out, "_sanity")
    os.makedirs(sanity_dir, exist_ok=True)

    for _, row in df.iterrows():
        wid, enc = str(row["id"]), row["encoding"]
        split = "test" if wid in test_ids else "validation" if wid in val_ids else "train"
        img_dir = os.path.join(args.out, split, "img")
        msk_dir = os.path.join(args.out, split, "mask")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        path = os.path.join(args.hubmap_dir, "train", wid + ".tiff")
        h, w, get = open_wsi(path)
        mask = rle_decode(enc, (h, w), args.rle_order)   # full-res mask (heavy: h*w bytes)
        print(f"{wid} [{split}] {w}x{h}  glom pixels={int(mask.sum())}")

        kept = sanity_done = 0
        for y in tqdm(range(0, h - 1, stride), desc=wid, leave=False):
            for x in range(0, w - 1, stride):
                y1, x1 = min(y + args.tile, h), min(x + args.tile, w)
                mtile = mask[y:y1, x:x1]
                has_glom = mtile.any()
                if not has_glom and random.random() > args.neg_ratio:
                    continue
                itile = get(y, y1, x, x1)
                if not has_glom and not is_tissue(itile):
                    continue
                im = cv2.resize(itile, (args.out_size, args.out_size), interpolation=cv2.INTER_AREA)
                mk = cv2.resize(mtile, (args.out_size, args.out_size), interpolation=cv2.INTER_NEAREST)
                name = f"{wid}_{y}_{x}.png"
                cv2.imwrite(os.path.join(img_dir, name), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(msk_dir, name), (mk * 255).astype(np.uint8))
                kept += 1
                # a couple of overlays per WSI so RLE orientation is visually verifiable
                if has_glom and sanity_done < 2:
                    ov = im.copy()
                    ov[mk > 0] = (0.5 * ov[mk > 0] + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
                    cv2.imwrite(os.path.join(sanity_dir, name), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
                    sanity_done += 1
        print(f"  kept {kept} tiles")

    print(f"done. CHECK the overlays in {sanity_dir} before training "
          f"(mask must sit on glomeruli; if not, rerun with --rle-order "
          f"{'C' if args.rle_order == 'F' else 'F'}).")


def _selfcheck():
    # tiny roundtrip: a 2x2 mask -> RLE -> decode must match (order-consistent)
    m = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    flat = m.reshape(-1, order="F")
    rle, i = [], 0
    while i < len(flat):
        if flat[i]:
            j = i
            while j < len(flat) and flat[j]:
                j += 1
            rle += [str(i + 1), str(j - i)]
            i = j
        else:
            i += 1
    back = rle_decode(" ".join(rle), (2, 2), "F")
    assert (back == m).all(), back
    print("selfcheck OK")


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
