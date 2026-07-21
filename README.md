# glomeruli-torch

PyTorch experiment: how far can the 9-slide glomerulus dataset be pushed by starting
segmentation from **Lunit pathology-domain pretrained weights** (same domain as our data),
plus modern decoder / loss / TTA levers. Separated from the main `Glomeruli-FP03-2026`
repo for logical isolation. Uses PyTorch so Lunit checkpoints load **natively — no
weight conversion** (the TF path needed a whole conversion + verification script).

## What lives where
- **This repo**: segmentation (`train.py`, `model.py`, `dataset.py`) + the clustering
  embedding stage (next: `embed_and_cluster.py`, Lunit DINO ViT).
- **Main repo** (`Glomeruli-FP03-2026`): preprocessing (produces the patch PNGs) and the
  clustering algorithms. We consume the preprocessing **output** (`data/dataset/...`),
  not the code — no openslide rewrite.

## Applied levers (vs the plain SegNet-VGG19 recipe)
- `deeplabv3plus` decoder (default) — usually beats vanilla U-Net at equal encoder.
- `--img-size 512` — glomerulus boundaries are fine; 400px loses detail.
- `--loss focaltversky` (optional) — alpha>beta penalises false positives, targeting the
  boundary over-segmentation from the TF runs (precision 0.82 < recall 0.88).
- `--tta` — D4-flip test-time augmentation.
- Matched baseline vs pathology comparison via `run_experiments.sh`.

## Run
```bash
pip install -r requirements.txt
python train.py --smoke                              # imports + shapes, no data
bash run_experiments.sh                              # baseline vs Lunit, matched setup
```

## Needed inputs
- Preprocessed patches at `<data-dir>/{train,validation,test}/{img,mask}/*.png`.
- Lunit weights in `weights/` (ResNet50 Barlow Twins for segmentation; DINO ViT for the
  clustering stage) from `lunit-io/benchmark-ssl-pathology`.

## External data — HuBMAP "Hacking the Kidney" (cross-domain test)
Human PAS kidney WSIs from a different source. Only the ~8 train WSIs have public
glomerulus masks (RLE in `train.csv`), so this is a **generalisation test**, not "more data".

```bash
# 1. download (needs a Kaggle account + accepted competition rules + ~/.kaggle/kaggle.json)
kaggle competitions download -c hubmap-kidney-segmentation -p hubmap && unzip -q hubmap/*.zip -d hubmap
# 2. self-check the RLE decoder
python prep_hubmap.py --selfcheck
# 3. convert -> our layout (split by WSI id to avoid leakage; ids are the .tiff filenames)
python prep_hubmap.py --hubmap-dir hubmap --out data/hubmap \
    --val-ids <one-wsi-id> --test-ids <another-wsi-id>
# 4. LOOK at data/hubmap/_sanity/*.png — the red mask must sit on the glomeruli.
#    If it's transposed/scrambled, rerun step 3 with --rle-order C.
# 5. train on it, same pipeline:
python train.py --data-dir data/hubmap --encoder-weights imagenet --out out/hubmap --tta
```
`prep_hubmap.py` decodes each WSI's RLE, tiles it, keeps glomerulus tiles (+ some tissue
negatives), resizes, and writes PNG img/mask pairs — exactly what `GlomDataset` reads.

## Honest prior
FP03's TF runs showed pathology encoders did **not** beat ImageNet *for segmentation* —
this repo re-tests that cleanly (matched setup) and, more promisingly, applies Lunit
**DINO** to the *clustering* embeddings, where domain features have a better shot.
