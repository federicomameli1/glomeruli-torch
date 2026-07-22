"""Leave-one-slide-out CV for the PyTorch pipeline — the honest generalization
number (vs a single fixed split), mirroring the TF loso_cv.py design:

  * pool the BASE patches (drop *_reinhard_* variants) across train/validation/test,
    group by slide of origin (RECHERCHE-NNN from the filename);
  * for each of the 9 slides: train on the other 8, evaluate on the held-out one;
  * NO validation set / NO early stopping — fixed budget (phase1 frozen + phase2
    fine-tune), evaluate the FINAL model (a held-in val set would reintroduce the
    small-val unreliability LOSO exists to avoid; the test slide would be leakage).

  python train_loso.py --data-dir ../Glomeruli-FP03-2026/data/dataset \
      --encoder-weights imagenet --out out/loso_imagenet --tta
  python train_loso.py --data-dir ... --encoder-weights none \
      --lunit-weights weights/bt_rn50.torch --out out/loso_lunit --tta
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import GlomDataset
from model import build_model, load_lunit_weights
from train import make_loss, evaluate   # reuse the loss + confusion-matrix eval

SLIDE_RE = re.compile(r"(RECHERCHE-\d+)")


def gather_by_slide(data_dir):
    by = defaultdict(list)
    for split in ("train", "validation", "test"):
        for img in sorted(glob.glob(os.path.join(data_dir, split, "img", "*.png"))):
            name = os.path.basename(img)
            if "_reinhard_" in name:          # static Reinhard variants would leak
                continue
            m = SLIDE_RE.match(name)
            if m:
                by[m.group(1)].append(img)
    return dict(by)


def train_fixed(model, loss_fn, loader, device, epochs, lr, freeze):
    for p in model.encoder.parameters():
        p.requires_grad = not freeze
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=lr, momentum=0.9, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="resnet50")
    ap.add_argument("--decoder", default="deeplabv3plus")
    ap.add_argument("--encoder-weights", default="imagenet")
    ap.add_argument("--lunit-weights", default=None)
    ap.add_argument("--loss", default="bcedice", choices=["bcedice", "focaltversky"])
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--phase1-epochs", type=int, default=10)
    ap.add_argument("--phase2-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default="out/loso")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = make_loss(args.loss)
    by = gather_by_slide(args.data_dir)
    slides = sorted(by)
    print(f"{len(slides)} slides, {sum(len(v) for v in by.values())} base patches:")
    for s in slides:
        print(f"  {s}: {len(by[s])}")

    results = {}
    for i, held in enumerate(slides, 1):
        train_paths = [p for s in slides if s != held for p in by[s]]
        test_paths = by[held]
        print(f"\n===== fold {i}/{len(slides)} — hold out {held} "
              f"(train {len(train_paths)} / test {len(test_paths)}) =====")
        tr = DataLoader(GlomDataset(image_paths=train_paths, size=args.img_size, train=True),
                        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                        drop_last=True)   # avoid a size-1 last batch breaking ASPP BatchNorm
        te = DataLoader(GlomDataset(image_paths=test_paths, size=args.img_size, train=False),
                        batch_size=args.batch_size, num_workers=4, pin_memory=True)

        model = build_model(args.encoder, args.decoder, args.encoder_weights).to(device)
        if args.lunit_weights:
            load_lunit_weights(model, args.lunit_weights)
        train_fixed(model, loss_fn, tr, device, args.phase1_epochs, args.lr, freeze=True)
        train_fixed(model, loss_fn, tr, device, args.phase2_epochs, args.lr / 10, freeze=False)

        m = evaluate(model, te, device, tta=args.tta)
        results[held] = {"n_test": len(test_paths), **m}
        print(f"[{held}] glom IoU {m['iou']:.4f} | dice {m['dice']:.4f}")
        del model
        torch.cuda.empty_cache()

    iou = np.array([r["iou"] for r in results.values()])
    dice = np.array([r["dice"] for r in results.values()])
    summary = {"per_slide": results,
               "glom_iou_mean": float(iou.mean()), "glom_iou_std": float(iou.std()),
               "dice_mean": float(dice.mean()), "dice_std": float(dice.std())}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + "_loso.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ LOSO summary ================")
    print(f"{'slide':<16}{'n_test':>8}{'glomIoU':>10}{'dice':>9}")
    for s in slides:
        r = results[s]
        print(f"{s:<16}{r['n_test']:>8}{r['iou']:>10.4f}{r['dice']:>9.4f}")
    print("-" * 43)
    print(f"{'MEAN±STD':<16}{'':>8}{iou.mean():>7.4f}±{iou.std():.3f}  {dice.mean():.4f}±{dice.std():.3f}")
    print(f"saved: {args.out}_loso.json")


if __name__ == "__main__":
    main()
