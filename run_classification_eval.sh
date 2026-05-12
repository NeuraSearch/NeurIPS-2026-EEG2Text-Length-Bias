#!/bin/bash
# ===========================================================================
# Classification Evaluation Script
# ===========================================================================
# Runs all classification model evaluations for all datasets.
#
# Braindecode models (end-to-end):
#   eegnet, shallowconvnet, deepconvnet, eegconformer, tidnet, atcnet, biot
#   Must be pre-trained before running noise conditions. The script will
#   error if the checkpoint is missing.
#
# Frozen encoder models:
#   brainecho, brainecho_biot, brainecho_stft
#   The classification head is trained once (normal condition). If the
#   checkpoint already exists, training is skipped and only evaluation runs.
#
# Conditions evaluated: normal, input_noise, length_noise
#
# Usage:
#   bash run_classification_eval.sh                              # all datasets
#   bash run_classification_eval.sh --dataset brennan
#   bash run_classification_eval.sh --dataset brennan --condition length_noise
#   bash run_classification_eval.sh --dataset brennan --model eegnet
# ===========================================================================

set -e

# ===========================================================================
# CONFIGURATION
# ===========================================================================

BRAINDECODE_MODELS="eegnet shallowconvnet deepconvnet eegconformer tidnet atcnet biot"
ALL_DATASETS="brennan hbn"
ALL_CONDITIONS="normal input_noise length_noise"

# Dataset paths
declare -A DATASET_PATHS
DATASET_PATHS["brennan"]="./datasets/Brennan/split_by_subject/"
DATASET_PATHS["hbn"]="./datasets/HBN/split_by_subject/"

# EEG channels per dataset
declare -A IN_CHANNELS
IN_CHANNELS["brennan"]="62"
IN_CHANNELS["hbn"]="129"

declare -A MEL_LEN
MEL_LEN["brennan"]="1200"
MEL_LEN["hbn"]="1200"

# num_classes (inferred from data for both datasets)
declare -A NUM_CLASSES_ARG
NUM_CLASSES_ARG["brennan"]=""
NUM_CLASSES_ARG["hbn"]=""

# Stage 2 alignment checkpoint paths for frozen encoder models
declare -A BRAINECHO_ALIGN_CKPT
BRAINECHO_ALIGN_CKPT["brennan"]="./checkpoints/brennan/b16_40_0.0001_alignment.pt"
BRAINECHO_ALIGN_CKPT["hbn"]="./checkpoints/hbn/b16_40_0.0001_alignment.pt"

declare -A BRAINECHO_BIOT_ALIGN_CKPT
BRAINECHO_BIOT_ALIGN_CKPT["brennan"]="./checkpoints/brennan/brainecho_biot/b16_40_0.0001_alignment.pt"
BRAINECHO_BIOT_ALIGN_CKPT["hbn"]="./checkpoints/hbn/brainecho_biot/b16_40_0.0001_alignment.pt"

declare -A BRAINECHO_STFT_ALIGN_CKPT
BRAINECHO_STFT_ALIGN_CKPT["brennan"]="./checkpoints/brennan/brainecho_stft/b16_40_0.0001_alignment.pt"
BRAINECHO_STFT_ALIGN_CKPT["hbn"]="./checkpoints/hbn/brainecho_stft/b16_40_0.0001_alignment.pt"

# Classification checkpoint base
CLASS_CKPT_BASE="./checkpoints/classification"

# Log base
LOG_BASE="./logs/classification_eval"

# ===========================================================================
# ARGUMENT PARSING
# ===========================================================================

DATASETS_TO_RUN="$ALL_DATASETS"
CONDITIONS_TO_RUN="$ALL_CONDITIONS"
MODELS_TO_RUN=""  # empty = run all

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASETS_TO_RUN="$2"; shift 2 ;;
        --condition)
            CONDITIONS_TO_RUN="$2"; shift 2 ;;
        --model)
            MODELS_TO_RUN="$2"; shift 2 ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: bash run_classification_eval.sh [--dataset DATASET] [--condition CONDITION] [--model MODEL]"
            exit 1 ;;
    esac
done

# ===========================================================================
# HELPERS
# ===========================================================================

get_condition_flag() {
    case $1 in
        input_noise)   echo "--input_noise" ;;
        length_noise)  echo "--length_noise" ;;
        normal)        echo "" ;;
        *)  echo "[ERROR] Unknown condition: $1"; exit 1 ;;
    esac
}

get_log_suffix() {
    case $1 in
        input_noise)   echo "_noise" ;;
        length_noise)  echo "_length_noise" ;;
        normal)        echo "" ;;
    esac
}

# ---------------------------------------------------------------------------
# run_braindecode_eval DATASET MODEL CONDITION
#
# Braindecode models must be pre-trained before calling this script
# (run train_eval_classification.py manually for the normal condition first).
# Noise-condition runs retrain from scratch but apply noise only at test time,
# so training is always on clean EEG.
# ---------------------------------------------------------------------------
run_braindecode_eval() {
    local dataset=$1
    local model=$2
    local condition=$3

    local class_ckpt="${CLASS_CKPT_BASE}/${dataset}/${model}/${dataset}_${model}_best.pt"
    local save_path="${CLASS_CKPT_BASE}/${dataset}/${model}"
    local log_dir="${LOG_BASE}/${dataset}"
    local condition_flag=$(get_condition_flag "$condition")
    local log_suffix=$(get_log_suffix "$condition")
    local log_file="${log_dir}/${model}${log_suffix}.log"

    mkdir -p "$log_dir"

    if [[ ! -f "$class_ckpt" ]]; then
        echo "[ERROR] Classification checkpoint missing for ${dataset}/${model}: ${class_ckpt}"
        echo "        Train this model first: python train_eval_classification.py --dataset_name ${dataset} --model_name ${model} ..."
        exit 1
    fi

    echo "  --> [braindecode] ${dataset} | ${model} | ${condition}"
    echo "      Log: ${log_file}"

    python train_eval_classification.py \
        --dataset_name "$dataset" \
        --dataset_path "${DATASET_PATHS[$dataset]}" \
        --model_name "$model" \
        --save_path "$save_path" \
        ${NUM_CLASSES_ARG[$dataset]} \
        $condition_flag \
        > "$log_file" 2>&1

    echo "      Done."
}

# ---------------------------------------------------------------------------
# run_frozen_encoder_eval DATASET MODEL CONDITION ALIGN_CKPT
#
# Trains the linear classification head once (normal condition), then runs
# evaluation under the requested condition by loading the saved head.
# ---------------------------------------------------------------------------
run_frozen_encoder_eval() {
    local dataset=$1
    local model=$2
    local condition=$3
    local align_ckpt=$4

    local class_ckpt="${CLASS_CKPT_BASE}/${dataset}/${model}/${dataset}_${model}_best.pt"
    local save_path="${CLASS_CKPT_BASE}/${dataset}/${model}"
    local log_dir="${LOG_BASE}/${dataset}"
    local condition_flag=$(get_condition_flag "$condition")
    local log_suffix=$(get_log_suffix "$condition")
    local log_file="${log_dir}/${model}${log_suffix}.log"

    mkdir -p "$log_dir"
    mkdir -p "$save_path"

    if [[ ! -f "$align_ckpt" ]]; then
        echo "  [SKIP] Alignment checkpoint missing: ${align_ckpt}"
        return 0
    fi

    # Train classification head if not already done
    if [[ ! -f "$class_ckpt" ]]; then
        echo "  --> [frozen encoder] ${dataset} | ${model} | training classification head..."
        echo "      Alignment ckpt : ${align_ckpt}"
        echo "      Log            : ${log_dir}/${model}_train.log"

        python train_eval_classification.py \
            --dataset_name "$dataset" \
            --dataset_path "${DATASET_PATHS[$dataset]}" \
            --model_name "$model" \
            --checkpoint "$align_ckpt" \
            --in_chan "${IN_CHANNELS[$dataset]}" \
            --d_model 256 \
            --mel_len "${MEL_LEN[$dataset]}" \
            --save_path "$save_path" \
            ${NUM_CLASSES_ARG[$dataset]} \
            > "${log_dir}/${model}_train.log" 2>&1

        echo "      Training done."
    else
        echo "  --> [frozen encoder] ${dataset} | ${model} | checkpoint exists, skipping training."
    fi

    # Evaluate under requested condition
    echo "  --> [frozen encoder] ${dataset} | ${model} | ${condition}"
    echo "      Log: ${log_file}"

    python train_eval_classification.py \
        --dataset_name "$dataset" \
        --dataset_path "${DATASET_PATHS[$dataset]}" \
        --model_name "$model" \
        --checkpoint "$align_ckpt" \
        --in_chan "${IN_CHANNELS[$dataset]}" \
        --d_model 256 \
        --mel_len "${MEL_LEN[$dataset]}" \
        --save_path "$save_path" \
        ${NUM_CLASSES_ARG[$dataset]} \
        $condition_flag \
        > "$log_file" 2>&1

    echo "      Done."
}

# ===========================================================================
# MAIN
# ===========================================================================

echo "================================================================"
echo "CLASSIFICATION EVALUATION"
echo "Datasets  : ${DATASETS_TO_RUN}"
echo "Conditions: ${CONDITIONS_TO_RUN}"
echo "Models    : ${MODELS_TO_RUN:-all}"
echo "================================================================"

for dataset in $DATASETS_TO_RUN; do

    if [[ -z "${DATASET_PATHS[$dataset]}" ]]; then
        echo "[ERROR] Unknown dataset: ${dataset}"
        exit 1
    fi

    echo ""
    echo "================================================================"
    echo "DATASET: ${dataset^^}"
    echo "================================================================"

    # ----------------------------------------------------------------
    # braindecode models
    # ----------------------------------------------------------------
    for model in $BRAINDECODE_MODELS; do
        if [[ -n "$MODELS_TO_RUN" && "$MODELS_TO_RUN" != "$model" ]]; then
            continue
        fi
        for condition in $CONDITIONS_TO_RUN; do
            run_braindecode_eval "$dataset" "$model" "$condition"
        done
    done

    # ----------------------------------------------------------------
    # frozen encoder: brainecho
    # ----------------------------------------------------------------
    if [[ -z "$MODELS_TO_RUN" || "$MODELS_TO_RUN" == "brainecho" ]]; then
        for condition in $CONDITIONS_TO_RUN; do
            run_frozen_encoder_eval \
                "$dataset" "brainecho" "$condition" \
                "${BRAINECHO_ALIGN_CKPT[$dataset]}"
        done
    fi

    # ----------------------------------------------------------------
    # frozen encoder: brainecho_biot
    # ----------------------------------------------------------------
    if [[ -z "$MODELS_TO_RUN" || "$MODELS_TO_RUN" == "brainecho_biot" ]]; then
        for condition in $CONDITIONS_TO_RUN; do
            run_frozen_encoder_eval \
                "$dataset" "brainecho_biot" "$condition" \
                "${BRAINECHO_BIOT_ALIGN_CKPT[$dataset]}"
        done
    fi

    # ----------------------------------------------------------------
    # frozen encoder: brainecho_stft
    # ----------------------------------------------------------------
    if [[ -z "$MODELS_TO_RUN" || "$MODELS_TO_RUN" == "brainecho_stft" ]]; then
        for condition in $CONDITIONS_TO_RUN; do
            run_frozen_encoder_eval \
                "$dataset" "brainecho_stft" "$condition" \
                "${BRAINECHO_STFT_ALIGN_CKPT[$dataset]}"
        done
    fi

done

echo ""
echo "================================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "Results in: ${CLASS_CKPT_BASE}"
echo "Logs in   : ${LOG_BASE}"
echo "================================================================"
