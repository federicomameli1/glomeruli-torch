#!/bin/bash
#SBATCH --job-name=glom_torch
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s348829@studenti.polito.it
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# =============================================================================
# glomeruli-torch — matched baseline (ImageNet) vs Lunit pathology encoder.
# Self-contained: regenerates the patch dataset if missing (reusing the TF env
# for that one step), then trains both models in a dedicated torch env.
#
# Prereqs:
#   * ~/glomeruli-torch          (this repo) with weights/bt_rn50.torch present
#     -> run ./download_weights.sh on the login node first.
#   * ~/Glomeruli-FP03-2026      (main repo, for the preprocessing script)
#   * ~/slides/*.svs             (raw slides + xml)
# =============================================================================
set -euo pipefail

REPO=$HOME/glomeruli-torch
MAIN_REPO=$HOME/Glomeruli-FP03-2026
SLIDES_DIR=${SLIDES_DIR:-$HOME/slides}
DATA=${DATA:-$MAIN_REPO/data/dataset}
RESULTS=$REPO/results-$SLURM_JOB_ID

cd "$REPO"
nvidia-smi || true
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# --- 1. preprocessing (reuse the TF env; only if the patches are missing) -----
if [ ! -d "$DATA/train/img" ] || [ -z "$(ls -A "$DATA/train/img" 2>/dev/null)" ]; then
    echo "===== preprocessing patches (env: glomeruli) ====="
    conda activate glomeruli
    ( cd "$MAIN_REPO" && python scripts/preprocessing/preprocess_data.py "$SLIDES_DIR" )
    conda deactivate
else
    echo "patches already present at $DATA — skipping preprocessing"
fi

# --- 2. torch env (create once; idempotent) ----------------------------------
if ! conda env list | grep -q "glomeruli-torch"; then
    echo "===== creating conda env glomeruli-torch (first run only) ====="
    conda create -n glomeruli-torch python=3.11 -y
    conda activate glomeruli-torch
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install --quiet -r requirements.txt
else
    conda activate glomeruli-torch
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- 3. weights present? ------------------------------------------------------
[ -f weights/bt_rn50.torch ] || { echo "ERROR: run ./download_weights.sh first"; exit 1; }

# --- 4. matched experiments (only the encoder init differs) -------------------
mkdir -p "$RESULTS"
COMMON="--data-dir $DATA --decoder deeplabv3plus --img-size 512 --tta"

echo "===== baseline: ImageNet encoder ====="
python train.py $COMMON --encoder resnet50 --encoder-weights imagenet \
    --out "$RESULTS/imagenet" 2>&1 | tee "$RESULTS/imagenet.log" || echo "baseline FAILED"

echo "===== experiment: Lunit Barlow Twins encoder ====="
python train.py $COMMON --encoder resnet50 --encoder-weights none \
    --lunit-weights weights/bt_rn50.torch \
    --out "$RESULTS/lunit_bt" 2>&1 | tee "$RESULTS/lunit_bt.log" || echo "lunit FAILED"

# --- 5. glomerulus crops (reuse TF env; only if missing) ----------------------
CROPS=$MAIN_REPO/data/glomeruli/crops
if [ ! -d "$CROPS" ] || [ -z "$(ls -A "$CROPS" 2>/dev/null)" ]; then
    echo "===== extracting glomerulus crops (env: glomeruli) ====="
    conda activate glomeruli
    ( cd "$MAIN_REPO" && python scripts/extract_glomeruli_from_annotations.py "$SLIDES_DIR" )
    conda activate glomeruli-torch
fi

# --- 6. clustering: Lunit DINO vs ImageNet, matched ---------------------------
if [ -f weights/dino_vits16.torch ]; then
    echo "===== clustering: Lunit DINO ViT ====="
    python embed_and_cluster.py --backbone lunit_dino --lunit-weights weights/dino_vits16.torch \
        --crops-dir "$CROPS" --out "$RESULTS/clust_dino" 2>&1 | tee "$RESULTS/clust_dino.log" || echo "dino cluster FAILED"
    echo "===== clustering: ImageNet ResNet50 ====="
    python embed_and_cluster.py --backbone imagenet_r50 \
        --crops-dir "$CROPS" --out "$RESULTS/clust_imagenet" 2>&1 | tee "$RESULTS/clust_imagenet.log" || echo "imagenet cluster FAILED"
else
    echo "weights/dino_vits16.torch absent — skipping clustering stage"
fi

# --- 7. summary ---------------------------------------------------------------
echo ""; echo "===== SUMMARY (glomeruli-torch job $SLURM_JOB_ID) ====="
echo "-- segmentation (glomerulus IoU/Dice) --"
grep -h "^TEST" "$RESULTS"/*.log 2>/dev/null | sed 's/^/  /' || echo "  (none)"
echo "-- clustering (internal validity) --"
grep -h "^CLUSTER" "$RESULTS"/*.log 2>/dev/null | sed 's/^/  /' || echo "  (none)"
echo "results in $RESULTS"
