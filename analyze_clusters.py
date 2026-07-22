"""Analyze clustering results self-contained.

Two outputs, no broken links:
  1. backbone_comparison.csv — ranks every backbone's embeddings by internal validity
     (silhouette / DBCV / Davies-Bouldin / Calinski-Harabasz / n_clusters / noise).
  2. montage_<backbone>/cluster<k>.png — a grid of the ACTUAL glomerulus crops in each
     cluster, so you can judge morphological coherence by eye.

The pipeline's own index.html reports embed absolute scratch paths to the crops
(/mnt/beegfs/.../crops/...) which don't resolve off-cluster — this reads the crops
directly and bakes them into PNGs you can scp and open anywhere.

Run in the glomeruli-torch env. Crops and embeddings must be the SAME set, in the
same sorted order (they are, if you regenerate crops with the same extractor).

  python analyze_clusters.py \
      --embeddings-dir ~/Glomeruli-FP03-2026/models/pipeline-1843596/glomeruli/embeddings \
      --crops-dir      ~/Glomeruli-FP03-2026/data/glomeruli/crops \
      --out cluster_analysis
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, normalized_mutual_info_score)
import umap
import hdbscan


def slide_of(path):
    # crop names look like RECHERCHE-003_0000.png -> slide id is everything before the last _
    return os.path.basename(path).rsplit("_", 1)[0]


def cluster(emb):
    n = len(emb)
    x = normalize(emb)
    x = PCA(n_components=0.9).fit_transform(x)     # same preprocess as cluster_hdbscan
    x = normalize(x)
    z = umap.UMAP(n_neighbors=max(5, int(0.05 * n)), min_dist=0.0,
                  random_state=42).fit_transform(x)
    labels = hdbscan.HDBSCAN(min_cluster_size=max(5, int(0.03 * n))).fit_predict(z)
    return z, labels


def score(z, labels):
    n_c = len(set(labels)) - (1 if -1 in labels else 0)
    out = {"n_clusters": n_c, "noise_frac": round(float((labels == -1).mean()), 3)}
    m = labels != -1
    if n_c >= 2 and m.sum() > n_c:
        out["silhouette"] = round(float(silhouette_score(z[m], labels[m])), 4)
        out["davies_bouldin"] = round(float(davies_bouldin_score(z[m], labels[m])), 4)
        out["calinski_harabasz"] = round(float(calinski_harabasz_score(z[m], labels[m])), 1)
    try:
        out["dbcv"] = round(float(hdbscan.validity.validity_index(z.astype("float64"), labels)), 4)
    except Exception:
        pass
    return out


def montage(crop_paths, labels, out_dir, tag, per=25, thumb=128, cols=5):
    os.makedirs(out_dir, exist_ok=True)
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0][:per]
        rows = max(1, int(np.ceil(len(idx) / cols)))
        canvas = np.full((rows * thumb, cols * thumb, 3), 255, np.uint8)
        for k, i in enumerate(idx):
            im = cv2.imread(crop_paths[i])
            if im is None:
                continue
            r, cc = divmod(k, cols)
            canvas[r * thumb:(r + 1) * thumb, cc * thumb:(cc + 1) * thumb] = cv2.resize(im, (thumb, thumb))
        label = "noise" if c == -1 else f"cluster{c}"
        cv2.imwrite(os.path.join(out_dir, f"{tag}_{label}.png"), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--crops-dir", required=True)
    ap.add_argument("--out", default="cluster_analysis")
    ap.add_argument("--montage-backbone", default=None,
                    help="substring to pick which backbone to montage; default = best silhouette")
    args = ap.parse_args()

    crops = sorted(glob.glob(os.path.join(args.crops_dir, "*.png")))
    if not crops:
        raise FileNotFoundError(f"no crops in {args.crops_dir}")

    slides = np.array([slide_of(p) for p in crops])
    rows, cache = [], {}
    for npy in sorted(glob.glob(os.path.join(args.embeddings_dir, "*.npy"))):
        name = os.path.basename(npy).replace("_crops_embeddings.npy", "").replace("_embeddings.npy", "").replace(".npy", "")
        emb = np.load(npy)
        z, labels = cluster(emb)
        s = score(z, labels)
        # slide_nmi: how much cluster membership is explained by the slide of origin.
        # ~1 = clusters ARE the slides (stain/batch confound); ~0 = independent (morphology).
        if len(emb) == len(crops):
            m = labels != -1
            if m.sum() > 1 and len(set(labels[m])) > 1:
                s["slide_nmi"] = round(float(normalized_mutual_info_score(labels[m], slides[m])), 4)
        rows.append({"backbone": name, "n_samples": len(emb), **s})
        cache[name] = (labels, len(emb))
        print(f"{name}: {json.dumps(s)}")

    df = pd.DataFrame(rows).sort_values("silhouette", ascending=False, na_position="last")
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "backbone_comparison.csv"), index=False)
    print("\n=== backbone ranking (by silhouette) ===")
    print(df.to_string(index=False))

    pick = args.montage_backbone or df.iloc[0]["backbone"]
    for name, (labels, n) in cache.items():
        if pick in name:
            if n != len(crops):
                print(f"\nWARN: {name} has {n} embeddings but {len(crops)} crops found — "
                      f"order can't be trusted, montage skipped. Regenerate crops with the same extractor.")
            else:
                montage(crops, labels, os.path.join(args.out, "montage_" + name), name)
                print(f"\nmontages -> {args.out}/montage_{name}/  (scp these and open them)")
                ct = pd.crosstab(labels, slides)
                print(f"\n=== cluster x slide of origin ({name}) ===")
                print(ct.to_string())
                print("(if each cluster is dominated by a few slides -> stain/batch "
                      "confound, not morphology)")
            break


if __name__ == "__main__":
    main()
