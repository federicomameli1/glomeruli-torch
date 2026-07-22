#!/bin/bash
#SBATCH --job-name=glom_loso
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s348829@studenti.polito.it
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# =============================================================================
# Leave-one-slide-out CV for the PyTorch pipeline — 9 trainings, the honest
# generalization number to compare against the TF SegNet LOSO (0.702 ± 0.094).
#
# One encoder per job (9 folds ~ many hours). Pick the encoder with env vars:
#   ImageNet baseline (default):  sbatch loso.sbatch
#   Lunit pathology:              sbatch --export=ALL,LUNIT_WEIGHTS=weights/bt_rn50.torch loso.sbatch
# =============================================================================
set -euo pipefail

REPO=$HOME/glomeruli-torch
MAIN_REPO=$HOME/Glomeruli-FP03-2026
SLIDES_DIR=${SLIDES_DIR:-$HOME/slides}
DATA=${DATA:-$MAIN_REPO/data/dataset}
LUNIT_WEIGHTS=${LUNIT_WEIGHTS:-}          # empty => ImageNet baseline
RESULTS=$REPO/loso-$SLURM_JOB_ID

cd "$REPO"
nvidia-smi || true
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# --- preprocessing (TF env; only if patches missing) --------------------------
if [ ! -d "$DATA/train/img" ] || [ -z "$(ls -A "$DATA/train/img" 2>/dev/null)" ]; then
    echo "===== preprocessing patches (env: glomeruli) ====="
    conda activate glomeruli
    ( cd "$MAIN_REPO" && python scripts/preprocessing/preprocess_data.py "$SLIDES_DIR" )
fi

# --- torch env (create-or-repair; verified by a real import) ------------------
need_setup=0
if ! conda env list | awk '{print $1}' | grep -qx glomeruli-torch; then
    need_setup=1
elif ! conda run -n glomeruli-torch python -c "import torch, cv2, segmentation_models_pytorch" 2>/dev/null; then
    echo "glomeruli-torch env broken -> recreating"; conda env remove -n glomeruli-torch -y; need_setup=1
fi
if [ "$need_setup" = "1" ]; then
    conda create -n glomeruli-torch -c conda-forge --override-channels python=3.11 pip -y
    conda activate glomeruli-torch
    python -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
    python -m pip install --quiet -r requirements.txt
else
    conda activate glomeruli-torch
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- LOSO ---------------------------------------------------------------------
mkdir -p "$RESULTS"
if [ -n "$LUNIT_WEIGHTS" ]; then
    echo "===== LOSO with Lunit encoder ($LUNIT_WEIGHTS) ====="
    ENC=(--encoder-weights none --lunit-weights "$LUNIT_WEIGHTS")
    tag=lunit
else
    echo "===== LOSO with ImageNet encoder ====="
    ENC=(--encoder-weights imagenet)
    tag=imagenet
fi
python train_loso.py --data-dir "$DATA" --decoder deeplabv3plus --img-size 512 --tta \
    "${ENC[@]}" --out "$RESULTS/$tag" 2>&1 | tee "$RESULTS/$tag.log"

echo ""; echo "===== LOSO result ====="
grep -A20 "LOSO summary" "$RESULTS/$tag.log" | tail -n +1
echo "json: $RESULTS/${tag}_loso.json"
