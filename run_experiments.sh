#!/bin/bash
# Matched comparison: identical decoder / input size / schedule / loss — the ONLY
# thing that changes between runs is the encoder initialisation. That is what makes
# the pathology-domain effect actually isolable (unlike the earlier TF runs).
set -euo pipefail

DATA=${1:-../Glomeruli-FP03-2026/data/dataset}
COMMON="--data-dir $DATA --decoder deeplabv3plus --img-size 512 --tta"

echo "### baseline: ImageNet encoder ###"
python train.py $COMMON --encoder resnet50 --encoder-weights imagenet --out out/imagenet

echo "### experiment: Lunit pathology (Barlow Twins) encoder ###"
python train.py $COMMON --encoder resnet50 --encoder-weights none \
    --lunit-weights weights/bt_rn50.torch --out out/lunit_bt

echo "### optional: Focal Tversky loss on the winning encoder ###"
# python train.py $COMMON --encoder resnet50 --encoder-weights imagenet --loss focaltversky --out out/imagenet_ft
