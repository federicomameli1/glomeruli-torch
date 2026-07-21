"""Two-phase segmentation training in PyTorch.

Examples:
  # ImageNet baseline (deeplabv3+, 512px, TTA):
  python train.py --data-dir ../Glomeruli-FP03-2026/data/dataset --encoder-weights imagenet --out out/imagenet --tta
  # Lunit pathology encoder, matched setup:
  python train.py --data-dir ../Glomeruli-FP03-2026/data/dataset --encoder-weights none \
      --lunit-weights weights/bt_rn50.torch --out out/lunit_bt --tta
  # Focal Tversky loss (penalises false positives -> curbs boundary over-segmentation):
  python train.py ... --loss focaltversky
  # sanity check without data:
  python train.py --smoke
"""
import argparse
import os

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GlomDataset
from model import build_model, load_lunit_weights


def make_loss(name):
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = smp.losses.SoftBCEWithLogitsLoss()
    if name == "focaltversky":
        # alpha (FP) > beta (FN): penalise false positives harder — targets the
        # boundary over-segmentation seen in the TF runs (precision 0.82 < recall 0.88).
        ft = smp.losses.TverskyLoss(mode="binary", from_logits=True,
                                    alpha=0.7, beta=0.3, gamma=1.333)
        return lambda z, y: ft(z, y)
    return lambda z, y: bce(z, y) + dice(z, y)


def tta_predict(model, x):
    # D4 subset: identity + 3 flips (each is its own inverse). Cheap, +IoU at eval.
    views = (lambda t: t, lambda t: torch.flip(t, [3]),
             lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2, 3]))
    return torch.stack([f(model(f(x))) for f in views]).mean(0)


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = tta_predict(model, x) if tta else model(x)
        p = (logits.sigmoid() > 0.5).float()
        tp += int(((p == 1) & (y == 1)).sum())
        fp += int(((p == 1) & (y == 0)).sum())
        fn += int(((p == 0) & (y == 1)).sum())
    eps = 1e-9
    return dict(iou=tp / (tp + fp + fn + eps), dice=2 * tp / (2 * tp + fp + fn + eps),
                recall=tp / (tp + fn + eps), precision=tp / (tp + fp + eps))


def run_phase(model, loss_fn, tr, va, device, epochs, lr, freeze, out_path, best):
    for p in model.encoder.parameters():
        p.requires_grad = not freeze
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    for e in range(epochs):
        model.train()
        for x, y in tqdm(tr, desc=f"  epoch {e + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = evaluate(model, va, device)
        print(f"  epoch {e + 1}: val glomIoU={m['iou']:.4f} dice={m['dice']:.4f} "
              f"rec={m['recall']:.3f} prec={m['precision']:.3f}")
        if m["iou"] > best["iou"]:
            best["iou"] = m["iou"]
            torch.save(model.state_dict(), out_path)
            print(f"    saved best -> {out_path}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="dir with train/ validation/ test/ subdirs")
    ap.add_argument("--encoder", default="resnet50")
    ap.add_argument("--decoder", default="deeplabv3plus",
                    choices=["unet", "unetpp", "deeplabv3plus"])
    ap.add_argument("--encoder-weights", default="imagenet", help="'imagenet' | 'none'")
    ap.add_argument("--lunit-weights", default=None, help="path to a Lunit .torch checkpoint")
    ap.add_argument("--loss", default="bcedice", choices=["bcedice", "focaltversky"])
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--phase1-epochs", type=int, default=10)
    ap.add_argument("--phase2-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default="out/run")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="build + 1 forward pass, no data")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.encoder, args.decoder, args.encoder_weights).to(device)
    if args.lunit_weights:
        load_lunit_weights(model, args.lunit_weights)

    if args.smoke:
        x = torch.randn(2, 3, args.img_size, args.img_size, device=device)
        out = model(x)
        assert out.shape == (2, 1, args.img_size, args.img_size), out.shape
        print(f"smoke OK: {tuple(out.shape)} on {device}")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    loss_fn = make_loss(args.loss)
    ld = lambda split, tr: DataLoader(
        GlomDataset(os.path.join(args.data_dir, split), args.img_size, train=tr),
        batch_size=args.batch_size, shuffle=tr, num_workers=4, pin_memory=True)
    tr, va, te = ld("train", True), ld("validation", False), ld("test", False)

    best = {"iou": -1.0}
    out_path = args.out + "_best.pt"
    print(f"=== Phase 1: frozen encoder ({args.decoder}/{args.encoder}, loss={args.loss}) ===")
    run_phase(model, loss_fn, tr, va, device, args.phase1_epochs, args.lr, True, out_path, best)
    print("=== Phase 2: fine-tune all ===")
    run_phase(model, loss_fn, tr, va, device, args.phase2_epochs, args.lr / 10, False, out_path, best)

    model.load_state_dict(torch.load(out_path, map_location=device))
    m = evaluate(model, te, device, tta=args.tta)
    print(f"TEST{' +TTA' if args.tta else ''}: glomIoU={m['iou']:.4f} dice={m['dice']:.4f} "
          f"rec={m['recall']:.3f} prec={m['precision']:.3f}")


if __name__ == "__main__":
    main()
