#!/usr/bin/env bash
# reproduce_results.sh — Reproduce all main figures and tables from the paper.
#
# Prerequisites:
#   1. conda activate diffuse_env  (or equivalent env with requirements.txt)
#   2. BISECT config.yaml configured with correct paths (see config.yaml.example)
#   3. ESM-2 embeddings available (muscle: hMuscle/data/, brain: set in config.yaml)
#   4. Pre-trained model: hMuscle/saved_models/v15d_bp_clean.pt
#
# Estimated runtime: ~4 hours on 1x A100 GPU (training), ~2 hours CPU (BISECT deep mode)
# For evaluation only (no retraining): ~30 minutes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs_reproduce"
mkdir -p "$LOG_DIR"

echo "=== DIFFUSE + BISECT Reproduction Script ==="
echo "Date: $(date)"
echo "Working dir: $SCRIPT_DIR"
echo ""

# -----------------------------------------------------------------------
# Step 1: DIFFUSE evaluation (Table 1, Figure 1b)
# -----------------------------------------------------------------------
echo "[Step 1/5] DIFFUSE muscle evaluation (Table 1)..."
cd "$SCRIPT_DIR/hMuscle"

python results_isoform/evaluation.py \
    --model saved_models/v15d_bp_clean.pt \
    --output results_isoform/eval_output/ \
    2>&1 | tee "$LOG_DIR/step1_diffuse_eval.log"

echo "  Expected: Macro AUPRC (Type-B) = 0.685 ± 0.017, LR baseline = 0.363"
echo ""

# -----------------------------------------------------------------------
# Step 2: Brain zero-shot evaluation (Table S3)
# -----------------------------------------------------------------------
echo "[Step 2/5] Brain zero-shot evaluation (Supplementary Table S3)..."

python model/v15d_brain_eval.py \
    --model saved_models/v15d_bp_clean.pt \
    --output results_isoform/brain_eval_output/ \
    2>&1 | tee "$LOG_DIR/step2_brain_eval.log"

echo "  Expected: Macro AUPRC (brain, all 18 terms) = 0.5998"
echo ""

# -----------------------------------------------------------------------
# Step 3: pos_bias bootstrap CI (Table 2)
# -----------------------------------------------------------------------
echo "[Step 3/5] pos_bias bootstrap confidence intervals (Table 2)..."

python model/v10_bootstrap_ci.py \
    --model saved_models/v15d_bp_clean.pt \
    --n-bootstrap 1000 \
    --output results_isoform/posbiasci_output/ \
    2>&1 | tee "$LOG_DIR/step3_posbias.log"

echo ""

# -----------------------------------------------------------------------
# Step 4: BISECT pipeline (Figures 2–4, Tables S2, S5)
# -----------------------------------------------------------------------
echo "[Step 4/5] BISECT full pipeline (deep mode, 4 workers)..."
cd "$SCRIPT_DIR/Final_analysis/pipeline_bioanalysis"

if [ ! -f config.yaml ]; then
    echo "ERROR: config.yaml not found."
    echo "  Copy config.yaml.example to config.yaml and set your paths."
    exit 1
fi

python orchestrate.py --mode deep --workers 4 \
    2>&1 | tee "$LOG_DIR/step4_bisect.log"

echo "  Expected: 26 Stage 2 PASS, 13 SUPPORTED"
echo ""

# -----------------------------------------------------------------------
# Step 5: Generate figures
# -----------------------------------------------------------------------
echo "[Step 5/5] Generating figures..."

Rscript figA_pipeline.R 2>&1 | tee "$LOG_DIR/step5_figA.log"
echo "  Figure 1: outputs/figA_pipeline.pdf"

Rscript figB_cases.R 2>&1 | tee "$LOG_DIR/step5_figB.log"
echo "  Figure 2: outputs/figB_cases.pdf"

Rscript figC_heatmap.R 2>&1 | tee "$LOG_DIR/step5_figC.log"
echo "  Figure 3: outputs/figC_heatmap.pdf"

Rscript figD_network.R 2>&1 | tee "$LOG_DIR/step5_figD.log"
echo "  Figure 4: outputs/figD_network.pdf"

echo ""
echo "=== Reproduction complete ==="
echo "All logs saved to: $LOG_DIR"
echo ""
echo "Key output files:"
echo "  DIFFUSE eval:  hMuscle/results_isoform/eval_output/"
echo "  Brain eval:    hMuscle/results_isoform/brain_eval_output/"
echo "  BISECT output: Final_analysis/pipeline_bioanalysis/outputs/"
echo "  Figures:       Final_analysis/pipeline_bioanalysis/outputs/fig*.pdf"
