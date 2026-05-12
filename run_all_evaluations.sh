#!/bin/bash
# =============================================================================
# FULL GENERATION EVALUATION SCRIPT
# Datasets : Brennan, HBN
# Models   : BrainECHO, BrainECHO-BIOT, BrainECHO-STFT
# Baselines: Gaussian noise, Length-preserving noise, Random sentence selection
# =============================================================================
# USAGE:
#   Run all datasets:
#       bash run_all_evaluations.sh
#
#   Run specific datasets:
#       bash run_all_evaluations.sh brennan
#       bash run_all_evaluations.sh hbn
#       bash run_all_evaluations.sh brennan hbn
#
# NOTE: All model checkpoints must exist before running.
# =============================================================================

GPU=0

# =============================================================================
# PARSE DATASET ARGUMENTS
# =============================================================================
if [ $# -eq 0 ]; then
    RUN_BRENNAN=true
    RUN_HBN=true
else
    RUN_BRENNAN=false
    RUN_HBN=false
    for arg in "$@"; do
        case $arg in
            brennan) RUN_BRENNAN=true ;;
            hbn)     RUN_HBN=true ;;
            *)       echo "Unknown dataset: $arg. Choose from: brennan hbn"; exit 1 ;;
        esac
    done
fi

echo "===== EVALUATION PLAN ====="
echo "  Brennan: $RUN_BRENNAN"
echo "  HBN:     $RUN_HBN"
echo "  GPU:     $GPU"
echo "==========================="

# =============================================================================
# HELPER FUNCTION
# Args: DATASET MODEL DATASET_PATH SAVE_PATH LOG_DIR CHECKPOINT RESULTS_NAME EXTRA_FLAGS
# =============================================================================
run_eval() {
    local DATASET=$1
    local MODEL=$2
    local DATASET_PATH=$3
    local SAVE_PATH=$4
    local LOG_DIR=$5
    local CHECKPOINT=$6
    local RESULTS_NAME=$7
    local EXTRA_FLAGS=$8

    echo ""
    echo "----- $DATASET / $MODEL / $RESULTS_NAME -----"

    python main_run.py \
        model_type=$MODEL \
        dataset=$DATASET \
        vqvae_dataset=$DATASET \
        dataset_name=$DATASET \
        dataset_path=$DATASET_PATH \
        save_path=$SAVE_PATH \
        log_dir=$LOG_DIR \
        cuda=$GPU \
        stages_to_run=[] \
        run_evaluation=true \
        eval_checkpoint=$CHECKPOINT \
        eval_results_name=$RESULTS_NAME \
        eval_output_dir=./result \
        save_results=true \
        $EXTRA_FLAGS \
        2>&1 | tee ${LOG_DIR}/${MODEL}_${RESULTS_NAME}.log

    echo "----- DONE: $DATASET / $MODEL / $RESULTS_NAME -----"
}

# =============================================================================
# BRENNAN
# =============================================================================
if [ "$RUN_BRENNAN" = true ]; then
    echo ""
    echo "============================================="
    echo "BRENNAN"
    echo "============================================="

    mkdir -p ./logs/brennan/brainecho_biot ./logs/brennan/brainecho_stft

    BPATH=./datasets/Brennan/split_by_subject/
    BSAVE=./checkpoints/brennan
    BLOG=./logs/brennan
    BE_CKPT=./checkpoints/brennan/b16_40_0.0001_finetune.pt
    BB_CKPT=./checkpoints/brennan/brainecho_biot/b16_40_0.0001_finetune.pt
    BS_CKPT=./checkpoints/brennan/brainecho_stft/b16_40_0.0001_finetune.pt

    # BrainECHO
    run_eval brennan brainecho $BPATH $BSAVE $BLOG $BE_CKPT brennan_brainecho_results ""
    run_eval brennan brainecho $BPATH $BSAVE $BLOG $BE_CKPT brennan_brainecho_noise_results "eval_input_noise=true"
    run_eval brennan brainecho $BPATH $BSAVE $BLOG $BE_CKPT brennan_brainecho_length_noise_results "eval_length_noise=true"
    run_eval brennan brainecho $BPATH $BSAVE $BLOG $BE_CKPT brennan_brainecho_random_results "eval_random_select=true"

    # BrainECHO-BIOT
    run_eval brennan brainecho_biot $BPATH $BSAVE/brainecho_biot $BLOG/brainecho_biot $BB_CKPT brennan_brainecho_biot_results ""
    run_eval brennan brainecho_biot $BPATH $BSAVE/brainecho_biot $BLOG/brainecho_biot $BB_CKPT brennan_brainecho_biot_noise_results "eval_input_noise=true"
    run_eval brennan brainecho_biot $BPATH $BSAVE/brainecho_biot $BLOG/brainecho_biot $BB_CKPT brennan_brainecho_biot_length_noise_results "eval_length_noise=true"
    run_eval brennan brainecho_biot $BPATH $BSAVE/brainecho_biot $BLOG/brainecho_biot $BB_CKPT brennan_brainecho_biot_random_results "eval_random_select=true"

    # BrainECHO-STFT
    run_eval brennan brainecho_stft $BPATH $BSAVE/brainecho_stft $BLOG/brainecho_stft $BS_CKPT brennan_brainecho_stft_results ""
    run_eval brennan brainecho_stft $BPATH $BSAVE/brainecho_stft $BLOG/brainecho_stft $BS_CKPT brennan_brainecho_stft_noise_results "eval_input_noise=true"
    run_eval brennan brainecho_stft $BPATH $BSAVE/brainecho_stft $BLOG/brainecho_stft $BS_CKPT brennan_brainecho_stft_length_noise_results "eval_length_noise=true"
    run_eval brennan brainecho_stft $BPATH $BSAVE/brainecho_stft $BLOG/brainecho_stft $BS_CKPT brennan_brainecho_stft_random_results "eval_random_select=true"

    echo "===== BRENNAN COMPLETE ====="
fi

# =============================================================================
# HBN
# =============================================================================
if [ "$RUN_HBN" = true ]; then
    echo ""
    echo "============================================="
    echo "HBN"
    echo "============================================="

    mkdir -p ./logs/hbn/brainecho_biot ./logs/hbn/brainecho_stft

    HPATH=./datasets/HBN/split_by_subject/
    HSAVE=./checkpoints/hbn
    HLOG=./logs/hbn
    HE_CKPT=./checkpoints/hbn/b16_40_0.0001_finetune.pt
    HB_CKPT=./checkpoints/hbn/brainecho_biot/b16_40_0.0001_finetune.pt
    HS_CKPT=./checkpoints/hbn/brainecho_stft/b16_40_0.0001_finetune.pt
    HBN_EXTRA="exclude_the_present_eval=true"

    # BrainECHO
    run_eval hbn brainecho $HPATH $HSAVE $HLOG $HE_CKPT hbn_brainecho_results "$HBN_EXTRA"
    run_eval hbn brainecho $HPATH $HSAVE $HLOG $HE_CKPT hbn_brainecho_noise_results "eval_input_noise=true $HBN_EXTRA"
    run_eval hbn brainecho $HPATH $HSAVE $HLOG $HE_CKPT hbn_brainecho_length_noise_results "eval_length_noise=true $HBN_EXTRA"
    run_eval hbn brainecho $HPATH $HSAVE $HLOG $HE_CKPT hbn_brainecho_random_results "eval_random_select=true $HBN_EXTRA"

    # BrainECHO-BIOT
    run_eval hbn brainecho_biot $HPATH $HSAVE/brainecho_biot $HLOG/brainecho_biot $HB_CKPT hbn_brainecho_biot_results "$HBN_EXTRA"
    run_eval hbn brainecho_biot $HPATH $HSAVE/brainecho_biot $HLOG/brainecho_biot $HB_CKPT hbn_brainecho_biot_noise_results "eval_input_noise=true $HBN_EXTRA"
    run_eval hbn brainecho_biot $HPATH $HSAVE/brainecho_biot $HLOG/brainecho_biot $HB_CKPT hbn_brainecho_biot_length_noise_results "eval_length_noise=true $HBN_EXTRA"
    run_eval hbn brainecho_biot $HPATH $HSAVE/brainecho_biot $HLOG/brainecho_biot $HB_CKPT hbn_brainecho_biot_random_results "eval_random_select=true $HBN_EXTRA"

    # BrainECHO-STFT
    run_eval hbn brainecho_stft $HPATH $HSAVE/brainecho_stft $HLOG/brainecho_stft $HS_CKPT hbn_brainecho_stft_results "$HBN_EXTRA"
    run_eval hbn brainecho_stft $HPATH $HSAVE/brainecho_stft $HLOG/brainecho_stft $HS_CKPT hbn_brainecho_stft_noise_results "eval_input_noise=true $HBN_EXTRA"
    run_eval hbn brainecho_stft $HPATH $HSAVE/brainecho_stft $HLOG/brainecho_stft $HS_CKPT hbn_brainecho_stft_length_noise_results "eval_length_noise=true $HBN_EXTRA"
    run_eval hbn brainecho_stft $HPATH $HSAVE/brainecho_stft $HLOG/brainecho_stft $HS_CKPT hbn_brainecho_stft_random_results "eval_random_select=true $HBN_EXTRA"

    echo "===== HBN COMPLETE ====="
fi

echo ""
echo "===== ALL SELECTED EVALUATIONS COMPLETE ====="
