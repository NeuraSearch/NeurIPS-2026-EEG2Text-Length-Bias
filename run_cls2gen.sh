#!/bin/bash
# =============================================================================
# Run All Classification-to-Generation Experiments
# =============================================================================
# Runs classification_to_generation.py for every model × dataset × condition.
# Results saved to ./checkpoints/classification_to_generation/
#
# Usage:
#   bash run_cls2gen.sh                            # all datasets, all models
#   bash run_cls2gen.sh --dataset brennan
#   bash run_cls2gen.sh --dataset brennan --model eegnet
#   nohup bash run_cls2gen.sh > ./logs/cls2gen.log 2>&1 &
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION — update paths to match your checkpoint locations
# =============================================================================

# Dataset paths
declare -A DATASET_PATHS
DATASET_PATHS["brennan"]="./datasets/Brennan/split_by_subject/"
DATASET_PATHS["hbn"]="./datasets/HBN/split_by_subject/"

# EEG channels per dataset
declare -A IN_CHANNELS
IN_CHANNELS["brennan"]="62"
IN_CHANNELS["hbn"]="129"

# Mel lengths per dataset
declare -A MEL_LEN
MEL_LEN["brennan"]="1200"
MEL_LEN["hbn"]="1200"

# Classification checkpoint root
CLS_CKPT_ROOT="./checkpoints/classification"

# Stage 2 alignment checkpoints for frozen encoder models
declare -A BRAINECHO_CKPT
BRAINECHO_CKPT["brennan"]="./checkpoints/brennan/b16_40_0.0001_alignment.pt"
BRAINECHO_CKPT["hbn"]="./checkpoints/hbn/b16_40_0.0001_alignment.pt"

declare -A BRAINECHO_BIOT_CKPT
BRAINECHO_BIOT_CKPT["brennan"]="./checkpoints/brennan/brainecho_biot/b16_40_0.0001_alignment.pt"
BRAINECHO_BIOT_CKPT["hbn"]="./checkpoints/hbn/brainecho_biot/b16_40_0.0001_alignment.pt"

declare -A BRAINECHO_STFT_CKPT
BRAINECHO_STFT_CKPT["brennan"]="./checkpoints/brennan/brainecho_stft/b16_40_0.0001_alignment.pt"
BRAINECHO_STFT_CKPT["hbn"]="./checkpoints/hbn/brainecho_stft/b16_40_0.0001_alignment.pt"

# Output and log directories
OUTPUT_ROOT="./checkpoints/classification_to_generation"
LOG_DIR="./logs/cls2gen"

# Model sets
BRAINDECODE_MODELS="eegnet shallowconvnet deepconvnet eegconformer tidnet atcnet biot"
FROZEN_MODELS="brainecho brainecho_biot brainecho_stft"
ALL_DATASETS="brennan hbn"
ALL_CONDITIONS="normal input_noise length_noise"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

DATASETS_TO_RUN="$ALL_DATASETS"
MODELS_TO_RUN=""   # empty = run all

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASETS_TO_RUN="$2"; shift 2 ;;
        --model)   MODELS_TO_RUN="$2";   shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: bash run_cls2gen.sh [--dataset DATASET] [--model MODEL]"
            exit 1 ;;
    esac
done

# =============================================================================
# HELPERS
# =============================================================================

get_condition_flag() {
    case $1 in
        input_noise)  echo "--input_noise" ;;
        length_noise) echo "--length_noise" ;;
        normal)       echo "" ;;
    esac
}

get_log_suffix() {
    case $1 in
        input_noise)  echo "_noise" ;;
        length_noise) echo "_length_noise" ;;
        normal)       echo "" ;;
    esac
}

run_cls2gen() {
    local dataset=$1
    local model=$2
    local condition=$3
    local cls_ckpt=$4
    local encoder_ckpt=$5   # empty for braindecode models

    local save_path="${OUTPUT_ROOT}/${dataset}/${model}"
    local log_suffix=$(get_log_suffix "$condition")
    local condition_flag=$(get_condition_flag "$condition")
    local log_file="${LOG_DIR}/${dataset}_${model}${log_suffix}.log"

    mkdir -p "$save_path"
    mkdir -p "$LOG_DIR"

    if [[ ! -f "$cls_ckpt" ]]; then
        echo "  [SKIP] cls checkpoint missing: $cls_ckpt"
        return 0
    fi

    if [[ -n "$encoder_ckpt" && ! -f "$encoder_ckpt" ]]; then
        echo "  [SKIP] encoder checkpoint missing: $encoder_ckpt"
        return 0
    fi

    echo "  --> ${dataset} | ${model} | ${condition}"

    CMD="python classification_to_generation.py \
        --dataset_name ${dataset} \
        --dataset_path ${DATASET_PATHS[$dataset]} \
        --model_name ${model} \
        --cls_checkpoint ${cls_ckpt} \
        --in_chan ${IN_CHANNELS[$dataset]} \
        --mel_len ${MEL_LEN[$dataset]} \
        --save_path ${save_path} \
        ${condition_flag}"

    if [[ -n "$encoder_ckpt" ]]; then
        CMD="${CMD} --encoder_checkpoint ${encoder_ckpt}"
    fi

    eval $CMD > "$log_file" 2>&1
    echo "      Done. Log: ${log_file}"
}

should_run_model() {
    [[ -z "$MODELS_TO_RUN" || "$MODELS_TO_RUN" == "$1" ]]
}

# =============================================================================
# MAIN
# =============================================================================

echo "================================================================"
echo "CLASSIFICATION-TO-GENERATION EXPERIMENTS"
echo "Datasets : ${DATASETS_TO_RUN}"
echo "Models   : ${MODELS_TO_RUN:-all}"
echo "================================================================"

for dataset in $DATASETS_TO_RUN; do

    echo ""
    echo "================================================================"
    echo "DATASET: ${dataset^^}"
    echo "================================================================"

    # ── Braindecode models ──────────────────────────────────────────────
    for model in $BRAINDECODE_MODELS; do
        should_run_model "$model" || continue
        cls_ckpt="${CLS_CKPT_ROOT}/${dataset}/${model}/${dataset}_${model}_best.pt"
        for condition in $ALL_CONDITIONS; do
            run_cls2gen "$dataset" "$model" "$condition" "$cls_ckpt" ""
        done
    done

    # ── BrainECHO ──────────────────────────────────────────────────────
    if should_run_model "brainecho"; then
        cls_ckpt="${CLS_CKPT_ROOT}/${dataset}/brainecho/${dataset}_brainecho_best.pt"
        enc_ckpt="${BRAINECHO_CKPT[$dataset]}"
        for condition in $ALL_CONDITIONS; do
            run_cls2gen "$dataset" "brainecho" "$condition" "$cls_ckpt" "$enc_ckpt"
        done
    fi

    # ── BrainECHO-BIOT ─────────────────────────────────────────────────
    if should_run_model "brainecho_biot"; then
        cls_ckpt="${CLS_CKPT_ROOT}/${dataset}/brainecho_biot/${dataset}_brainecho_biot_best.pt"
        enc_ckpt="${BRAINECHO_BIOT_CKPT[$dataset]}"
        for condition in $ALL_CONDITIONS; do
            run_cls2gen "$dataset" "brainecho_biot" "$condition" "$cls_ckpt" "$enc_ckpt"
        done
    fi

    # ── BrainECHO-STFT ─────────────────────────────────────────────────
    if should_run_model "brainecho_stft"; then
        cls_ckpt="${CLS_CKPT_ROOT}/${dataset}/brainecho_stft/${dataset}_brainecho_stft_best.pt"
        enc_ckpt="${BRAINECHO_STFT_CKPT[$dataset]}"
        for condition in $ALL_CONDITIONS; do
            run_cls2gen "$dataset" "brainecho_stft" "$condition" "$cls_ckpt" "$enc_ckpt"
        done
    fi

done

echo ""
echo "================================================================"
echo "ALL DONE"
echo "Results in : ${OUTPUT_ROOT}"
echo "Logs in    : ${LOG_DIR}"
echo "================================================================"
