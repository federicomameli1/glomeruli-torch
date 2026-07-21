#!/bin/bash
# Download Lunit pathology-domain SSL weights (native PyTorch checkpoints).
#   - bt_rn50.torch     : ResNet50 Barlow Twins  -> segmentation encoder
#   - dino_vits16.torch : ViT-S/16 DINO          -> clustering embeddings (later stage)
# Run on the login node (has internet). Verify the URLs against the release page if a
# download 404s: https://github.com/lunit-io/benchmark-ssl-pathology/releases
set -euo pipefail

DEST=${1:-weights}
mkdir -p "$DEST"
BASE=https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights

# local-name -> remote-file  (edit the remote name here if the release renames them)
download() {
    local out="$DEST/$1" url="$BASE/$2"
    echo "-> $url"
    curl -fL --retry 3 -o "$out" "$url"
    local sz; sz=$(stat -c%s "$out")
    # real weights are tens of MB; a 404 HTML page is a few KB -> fail loud
    [ "$sz" -gt 1000000 ] || { echo "ERROR: $1 is only $sz bytes (bad URL?). Check the release page."; exit 1; }
    echo "   ok: $out ($sz bytes)"
}

download bt_rn50.torch     bt_rn50_ep200.torch
download dino_vits16.torch dino_vit_small_patch16_ep200.torch

echo "done -> $DEST/"
