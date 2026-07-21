"""Embed glomerulus crops with a frozen backbone, then cluster — the unsupervised
stage. Point of the experiment: does a pathology-domain backbone (Lunit DINO ViT)
separate the glomeruli better than a generic ImageNet one? Run both, compare the
internal validity scores (there are no A-F labels, so internal metrics only).

Examples:
  python embed_and_cluster.py --backbone lunit_dino --lunit-weights weights/dino_vits16.torch \
      --crops-dir ../Glomeruli-FP03-2026/data/glomeruli/crops --out out/dino
  python embed_and_cluster.py --backbone imagenet_r50 \
      --crops-dir ../Glomeruli-FP03-2026/data/glomeruli/crops --out out/imagenet_r50
  python embed_and_cluster.py --backbone imagenet_r50 --smoke
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropDataset(Dataset):
    def __init__(self, crops_dir, size=224):
        self.paths = sorted(glob.glob(os.path.join(crops_dir, "*.png")))
        if not self.paths:
            raise FileNotFoundError(f"no *.png crops in {crops_dir}")
        self.tf = A.Compose([A.Resize(size, size),
                             A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        return self.tf(image=img)["image"]


def build_backbone(name, lunit_weights, device):
    import timm
    if name == "lunit_dino":
        assert lunit_weights, "lunit_dino needs --lunit-weights"
        m = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
        sd = torch.load(lunit_weights, map_location="cpu")
        for k in ("state_dict", "teacher", "student", "model"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
        clean = {}
        for k, v in sd.items():
            for p in ("module.", "backbone."):
                if k.startswith(p):
                    k = k[len(p):]
            clean[k] = v
        missing, unexpected = m.load_state_dict(clean, strict=False)
        loaded = len(clean) - len(unexpected)
        print(f"DINO weights: ~{loaded} loaded | {len(missing)} missing | {len(unexpected)} unexpected")
        assert loaded > 50, "almost nothing loaded — ViT/checkpoint key mismatch; inspect it"
    elif name == "imagenet_r50":
        m = timm.create_model("resnet50", pretrained=True, num_classes=0, global_pool="avg")
    else:
        raise ValueError(name)
    return m.eval().to(device)


@torch.no_grad()
def embed(model, loader, device):
    out = []
    for x in loader:
        out.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(out, 0)


def cluster_and_score(emb):
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import umap
    import hdbscan

    n = len(emb)
    x = normalize(emb)
    x = PCA(n_components=0.9).fit_transform(x)     # their preprocess: L2 -> PCA(0.9) -> L2
    x = normalize(x)
    z = umap.UMAP(n_neighbors=max(5, int(0.05 * n)), min_dist=0.0,
                  random_state=42).fit_transform(x)
    labels = hdbscan.HDBSCAN(min_cluster_size=max(5, int(0.03 * n))).fit_predict(z)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    scores = {"n_samples": n, "n_clusters": n_clusters,
              "noise_frac": round(float((labels == -1).mean()), 3)}
    mask = labels != -1
    if n_clusters >= 2 and mask.sum() > n_clusters:
        scores["silhouette"] = round(float(silhouette_score(z[mask], labels[mask])), 4)
    try:
        scores["dbcv"] = round(float(hdbscan.validity.validity_index(
            z.astype("float64"), labels)), 4)
    except Exception:
        pass
    return z, labels, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir")
    ap.add_argument("--backbone", default="imagenet_r50", choices=["lunit_dino", "imagenet_r50"])
    ap.add_argument("--lunit-weights", default=None)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="out/embed")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_backbone(args.backbone, args.lunit_weights, device)

    if args.smoke:
        x = torch.randn(2, 3, args.img_size, args.img_size, device=device)
        with torch.no_grad():
            print(f"smoke OK: {args.backbone} -> {tuple(model(x).shape)} on {device}")
        return

    loader = DataLoader(CropDataset(args.crops_dir, args.img_size),
                        batch_size=args.batch_size, num_workers=4)
    emb = embed(model, loader, device)
    print(f"embedded {emb.shape[0]} crops -> dim {emb.shape[1]}")

    _, labels, scores = cluster_and_score(emb)
    print(f"CLUSTER [{args.backbone}]: " + json.dumps(scores))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out + "_emb.npy", emb)
    np.save(args.out + "_labels.npy", labels)
    with open(args.out + "_metrics.json", "w") as f:
        json.dump({"backbone": args.backbone, **scores}, f, indent=2)


if __name__ == "__main__":
    main()
