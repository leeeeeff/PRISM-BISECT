#!/bin/bash
# deepfri_setup.sh — DeepFRI env + model download
# Run: bash deepfri_setup.sh

set -e
DEEPFRI_DIR="/home/welcome1/sw1686/DIFFUSE/hMuscle/model/DeepFRI"
MODEL_DIR="${DEEPFRI_DIR}/trained_models"
LOG="../../logs_isoform/deepfri_setup_$(date +%Y%m%d_%H%M).log"

mkdir -p "$(dirname "$LOG")"

echo "=== DeepFRI Setup ===" | tee -a "$LOG"
echo "Time: $(date)" | tee -a "$LOG"

# ── 1. Clone DeepFRI ───────────────────────────────────────────────
if [ ! -d "$DEEPFRI_DIR" ]; then
    echo "[1] Cloning DeepFRI..." | tee -a "$LOG"
    git clone https://github.com/flatironinstitute/DeepFRI.git "$DEEPFRI_DIR" 2>&1 | tee -a "$LOG"
else
    echo "[1] DeepFRI already cloned: $DEEPFRI_DIR" | tee -a "$LOG"
fi

# ── 2. Create conda env ────────────────────────────────────────────
if ! conda env list | grep -q "deepfri_env"; then
    echo "[2] Creating deepfri_env (python=3.7)..." | tee -a "$LOG"
    conda create -n deepfri_env python=3.7 -y 2>&1 | tee -a "$LOG"
else
    echo "[2] deepfri_env already exists" | tee -a "$LOG"
fi

# ── 3. Install TF + dependencies ──────────────────────────────────
echo "[3] Installing tensorflow + dependencies..." | tee -a "$LOG"
# Try TF 2.3.1 CPU first (no GPU dependency on CUDA 10.1)
/home/welcome1/miniconda3/envs/deepfri_env/bin/pip install \
    tensorflow==2.3.1 \
    networkx==2.4 \
    scikit-learn==0.23.1 \
    biopython==1.76 \
    numpy==1.18.5 \
    h5py==2.10.0 \
    2>&1 | tee -a "$LOG" || {
    echo "  [WARN] TF 2.3.1 failed. Trying TF 2.4.0..." | tee -a "$LOG"
    /home/welcome1/miniconda3/envs/deepfri_env/bin/pip install \
        tensorflow==2.4.0 \
        networkx==2.4 \
        scikit-learn==0.24.1 \
        biopython==1.78 \
        numpy==1.19.5 \
        h5py==2.10.0 \
        2>&1 | tee -a "$LOG"
}

echo "[3] TF install done" | tee -a "$LOG"
/home/welcome1/miniconda3/envs/deepfri_env/bin/python -c \
    "import tensorflow as tf; print('TF version:', tf.__version__)" 2>&1 | tee -a "$LOG"

# ── 4. Download pre-trained models (903MB) ─────────────────────────
mkdir -p "$MODEL_DIR"
MODELS_TGZ="${MODEL_DIR}/trained_models.tar.gz"

if [ ! -f "${MODEL_DIR}/GCN-LM_MF0.05.hdf5" ]; then
    echo "[4] Downloading DeepFRI models (903MB)..." | tee -a "$LOG"
    MODEL_URL="https://users.flatironinstitute.org/~renfrew/DeepFRI_data/trained_models.tar.gz"
    wget -q --show-progress -O "$MODELS_TGZ" "$MODEL_URL" 2>&1 | tee -a "$LOG"
    echo "  Extracting..." | tee -a "$LOG"
    tar -xzf "$MODELS_TGZ" -C "$MODEL_DIR" --strip-components=1 2>&1 | tee -a "$LOG"
    echo "  Models extracted:" | tee -a "$LOG"
    ls "$MODEL_DIR"/*.hdf5 2>/dev/null | tee -a "$LOG"
else
    echo "[4] Models already downloaded: $MODEL_DIR" | tee -a "$LOG"
    ls "$MODEL_DIR"/*.hdf5 | head -5 | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "=== Setup complete ===" | tee -a "$LOG"
echo "Log: $LOG"
