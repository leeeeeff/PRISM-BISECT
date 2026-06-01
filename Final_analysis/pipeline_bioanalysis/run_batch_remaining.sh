#!/usr/bin/env bash
# =============================================================================
# run_batch_remaining.sh
# BISECT M10-M12 batch runner for the 44 remaining cases
# =============================================================================
# Usage:
#   bash run_batch_remaining.sh
#   bash run_batch_remaining.sh --dry-run
#
# The script skips the 9 already-completed cases (M10/M11/M12 present),
# runs orchestrate.py --force on each remaining case sequentially,
# sleeps 5s between cases to respect API rate limits, and logs failures.
# =============================================================================

set -euo pipefail

PIPELINE_DIR="/home/welcome1/sw1686/DIFFUSE/Final_analysis/pipeline_bioanalysis"
OUTPUTS_DIR="${PIPELINE_DIR}/outputs"
LOGS_DIR="${PIPELINE_DIR}/logs"
CONDA_ENV="isoform_env"
DATE_TAG=$(date +%Y%m%d)
LOG_FILE="${LOGS_DIR}/batch_M10M12_${DATE_TAG}.log"
FAIL_LOG="${LOGS_DIR}/batch_M10M12_${DATE_TAG}_failures.log"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] No cases will actually be executed."
fi

# ---------------------------------------------------------------------------
# Ensure logs directory exists
# ---------------------------------------------------------------------------
mkdir -p "${LOGS_DIR}"

# ---------------------------------------------------------------------------
# Already-completed genes (M10/M11/M12 verified present)
# These 9 cases are skipped unconditionally.
# ---------------------------------------------------------------------------
COMPLETED_GENES=(
    NDUFS4
    DLG1
    KIF21B
    PTPRF
    IFT122
    FANCA
    SYNE1
    RGS3
    ADGRB2
)

is_completed() {
    local gene="$1"
    for done in "${COMPLETED_GENES[@]}"; do
        if [[ "${done}" == "${gene}" ]]; then
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Full 53-case list (gene_name,cell_type) in CSV order
# ---------------------------------------------------------------------------
declare -a ALL_CASES=(
    "NDUFS4:Excitatory"
    "DLG1:OPC"
    "KIF21B:Excitatory"
    "SGIP1:Inhibitory"
    "EXD2:Oligodendrocyte"
    "HDAC5:Excitatory"
    "ZCCHC17:Oligodendrocyte"
    "IFT122:Excitatory"
    "FANCA:Excitatory"
    "DMD:Inhibitory"
    "PLD5:Oligodendrocyte"
    "MTMR3:Excitatory"
    "AMPD2:Inhibitory"
    "PML:Excitatory"
    "CCAR1:Inhibitory"
    "CDH18:Inhibitory"
    "ITGAV:Inhibitory"
    "SYNE1:Inhibitory"
    "HSPA12A:Excitatory"
    "CTNNA2:Excitatory"
    "MTHFD1:OPC"
    "RGS3:Astrocyte"
    "PPFIA4:Inhibitory"
    "ADGRB2:Inhibitory"
    "BSG:Oligodendrocyte"
    "ZNF268:Microglia"
    "FOXP1:Excitatory"
    "PTPRS:Astrocyte"
    "FRMD4A:Excitatory"
    "ZNF623:Oligodendrocyte"
    "ATP1B1:Inhibitory"
    "CEP78:Excitatory"
    "STXBP5L:Excitatory"
    "KTN1:Excitatory"
    "LRPPRC:Oligodendrocyte"
    "IFI16:Oligodendrocyte"
    "ZNF43:Inhibitory"
    "MAN1B1:Oligodendrocyte"
    "ATP6V0E2:Astrocyte"
    "ASXL3:Excitatory"
    "GOLGB1:Astrocyte"
    "ARPP19:Microglia"
    "PTPRF:Inhibitory"
    "RIF1:Oligodendrocyte"
    "DOCK11:Inhibitory"
    "ADAR:Oligodendrocyte"
    "ANKRD44:Oligodendrocyte"
    "ADD2:OPC"
    "SNTG1:Inhibitory"
    "ZNF397:Oligodendrocyte"
    "SLC8A1:Excitatory"
    "TMEM130:Excitatory"
    "DNM1:Excitatory"
)

# ---------------------------------------------------------------------------
# Build the work queue (skip completed genes)
# ---------------------------------------------------------------------------
declare -a WORK_QUEUE=()
for entry in "${ALL_CASES[@]}"; do
    gene="${entry%%:*}"
    cell="${entry##*:}"
    if is_completed "${gene}"; then
        continue
    fi
    WORK_QUEUE+=("${gene}:${cell}")
done

TOTAL=${#WORK_QUEUE[@]}
echo "=================================================================="
echo "  BISECT M10-M12 Batch Runner"
echo "  Date      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Cases     : ${TOTAL} remaining (of 53 total)"
echo "  Conda env : ${CONDA_ENV}"
echo "  Log       : ${LOG_FILE}"
echo "  Fail log  : ${FAIL_LOG}"
if $DRY_RUN; then
    echo "  Mode      : DRY-RUN"
else
    echo "  Mode      : LIVE (--force, sequential, 5s sleep between cases)"
fi
echo "=================================================================="
echo ""

# Initialize log files
{
    echo "=================================================================="
    echo "BISECT M10-M12 Batch — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Cases to run: ${TOTAL}"
    echo "=================================================================="
} | tee -a "${LOG_FILE}"

> "${FAIL_LOG}"

# ---------------------------------------------------------------------------
# Run each case
# ---------------------------------------------------------------------------
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
CASE_NUM=0

for entry in "${WORK_QUEUE[@]}"; do
    gene="${entry%%:*}"
    cell="${entry##*:}"
    CASE_NUM=$(( CASE_NUM + 1 ))

    case_dir="${OUTPUTS_DIR}/${gene}_${cell}"

    echo ""
    echo "Case ${CASE_NUM}/${TOTAL}: ${gene} [${cell}]"
    echo "[$(date '+%H:%M:%S')] Case ${CASE_NUM}/${TOTAL}: ${gene} [${cell}]" | tee -a "${LOG_FILE}"

    # Check if analysis.json exists and already has M10 data
    # (catches cases where a previous partial run left analysis.json without M10)
    if [[ -f "${case_dir}/analysis.json" ]]; then
        has_m10=$(python3 -c "
import json, sys
try:
    with open('${case_dir}/analysis.json') as f:
        d = json.load(f)
    m10 = d.get('m10_alphafold', {})
    if isinstance(m10, dict) and (m10.get('ct') or m10.get('ad')):
        sys.stdout.write('yes')
    else:
        sys.stdout.write('no')
except Exception:
    sys.stdout.write('no')
" 2>/dev/null)

        if [[ "${has_m10}" == "yes" ]]; then
            echo "  SKIP: ${gene}_${cell} already has M10/M11/M12 data."
            echo "  SKIP: ${gene}_${cell} already has M10/M11/M12 data." >> "${LOG_FILE}"
            SKIP_COUNT=$(( SKIP_COUNT + 1 ))
            continue
        fi
    fi

    if $DRY_RUN; then
        echo "  [DRY-RUN] would run: conda run -n ${CONDA_ENV} python orchestrate.py --case ${gene} --force --mode deep"
        echo "  [DRY-RUN] ${gene} [${cell}]" >> "${LOG_FILE}"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
        continue
    fi

    # Run the pipeline for this case
    set +e
    conda run -n "${CONDA_ENV}" \
        python "${PIPELINE_DIR}/orchestrate.py" \
            --case "${gene}" \
            --force \
            --mode deep \
            --workers 1 \
            --output-dir "${OUTPUTS_DIR}" \
        2>&1 | tee -a "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "  PASS: ${gene} [${cell}] completed (exit 0)" | tee -a "${LOG_FILE}"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
    else
        echo "  FAIL: ${gene} [${cell}] exited with code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
        echo "${gene} [${cell}] — exit code ${EXIT_CODE} — $(date '+%Y-%m-%d %H:%M:%S')" >> "${FAIL_LOG}"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    fi

    # Sleep to respect API rate limits (STRING=1s, UCSC=0.5s, AlphaFold=2s)
    # 5s total gives comfortable headroom for all three
    if [[ ${CASE_NUM} -lt ${TOTAL} ]]; then
        echo "  Sleeping 5s before next case..."
        sleep 5
    fi
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  Batch complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Ran    : $(( PASS_COUNT + FAIL_COUNT )) cases"
echo "  PASS   : ${PASS_COUNT}"
echo "  FAIL   : ${FAIL_COUNT}"
echo "  SKIP   : ${SKIP_COUNT} (already had M10/M11/M12)"
if [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo "  Failures logged to: ${FAIL_LOG}"
fi
echo "=================================================================="

{
    echo ""
    echo "=================================================================="
    echo "Batch complete — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "PASS: ${PASS_COUNT}  FAIL: ${FAIL_COUNT}  SKIP: ${SKIP_COUNT}"
    echo "=================================================================="
} >> "${LOG_FILE}"

if [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo ""
    echo "Failed cases:"
    cat "${FAIL_LOG}"
    exit 1
fi

exit 0
