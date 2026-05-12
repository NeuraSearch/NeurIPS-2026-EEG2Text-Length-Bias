import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
import pickle
import math
from einops import rearrange
from omegaconf import DictConfig, ListConfig

from collections import defaultdict
import numpy as np

from dataset_configs import get_dataset_config
from dataset import EEG_dataset_whisper
from model import BrainECHO, BrainECHOBIOT, BrainECHOSTFT
from metrics import compute_metrics
from utils.utils import *


# ============================================================================
# PER-SUBJECT METRICS HELPER
# ============================================================================

def _compute_per_subject_metrics(pred_string_list, target_string_list, subject_id_list):
    """
    Compute metrics per subject and return mean ± SD across subjects.

    Returns dict with:
        per_subject_scores: {subject_id: {metric: value}}
        {metric}_subject_mean: mean across subjects
        {metric}_subject_std: std across subjects
    """
    subject_preds = defaultdict(list)
    subject_targets = defaultdict(list)

    for pred, target, subj in zip(pred_string_list, target_string_list, subject_id_list):
        subject_preds[str(subj)].append(pred)
        subject_targets[str(subj)].append(target)

    per_subject_scores = {}
    for subj in subject_preds:
        try:
            per_subject_scores[subj] = compute_metrics(
                subject_preds[subj], subject_targets[subj]
            )
        except Exception as e:
            print(f"[WARN] Could not compute metrics for subject {subj}: {e}")
            continue

    subject_stats = {}
    if per_subject_scores:
        metrics_keys = ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4',
                        'wer', 'rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure']
        for metric in metrics_keys:
            vals = [per_subject_scores[s][metric]
                    for s in per_subject_scores
                    if metric in per_subject_scores[s]]
            if vals:
                subject_stats[f'{metric}_subject_mean'] = float(np.mean(vals))
                subject_stats[f'{metric}_subject_std'] = float(np.std(vals))

    subject_stats['per_subject_scores'] = per_subject_scores
    subject_stats['n_subjects'] = len(per_subject_scores)
    return subject_stats


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _setup_eval_environment(cfg, dataset_config):
    """Common setup for all evaluation functions"""

    seed_val = cfg.get('seed', 312)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if torch.cuda.is_available():
        dev = cfg.get('cuda', 'cpu')
        if isinstance(dev, int):
            dev = f"cuda:{dev}"
    else:
        dev = "cpu"
    device = torch.device(dev)

    batch_size = cfg.get('batch_size', 16)
    if isinstance(batch_size, (list, ListConfig)):
        batch_size = int(batch_size[-1])
    else:
        batch_size = int(batch_size)

    return device, batch_size


def _load_test_dataset(cfg, dataset_config):
    """Load test dataset with dataset-specific kwargs"""
    dataset_path = cfg.dataset_path
    subject = cfg.get('subject', 'all')
    exclude_the_present = cfg.get('exclude_the_present_eval', False)
    max_samples = cfg.get('eval_max_samples', None)

    dataset_kwargs = {"path": dataset_path + "test", "subject": subject}
    if dataset_config.name == "hbn":
        dataset_kwargs["exclude_the_present"] = exclude_the_present
    if max_samples is not None:
        dataset_kwargs["max_samples"] = max_samples

    return EEG_dataset_whisper(**dataset_kwargs)


def _load_all_sents(dataset_path):
    """Load all sentences pickle"""
    all_sents_path = dataset_path + "../all_sents.pickle"
    with open(all_sents_path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# BRAINECHO EVALUATION (unchanged from original)
# ============================================================================

def evaluate_brainecho(cfg: DictConfig, logger=None):
    """Evaluate BrainECHO model"""

    print("\n" + "="*80)
    print("BRAINECHO EVALUATION")
    print("="*80 + "\n")

    dataset_name = cfg.dataset_name
    dataset_path = cfg.dataset_path
    checkpoint_path = cfg.eval_checkpoint
    subject = cfg.get('subject', 'all')
    split_mel = cfg.get('split_mel', False)
    split_len = cfg.get('split_len', 4) * 100
    codebook_size = cfg.get('cb_size', 2048)
    r = cfg.get('ratio', 4)
    is_teacher_forcing = cfg.get('eval_teacher_forcing', False)
    input_noise = cfg.get('eval_input_noise', False)
    length_noise = cfg.get('eval_length_noise', False)
    random_select = cfg.get('eval_random_select', False)
    wo_ft = cfg.get('eval_without_finetuning', False)
    results_name = cfg.get('eval_results_name', 'evaluation_results')
    output_dir = cfg.get('eval_output_dir', './result')
    os.makedirs(output_dir, exist_ok=True)
    output_all_results_path = os.path.join(output_dir, f"{results_name}.txt")

    dataset_config = get_dataset_config(dataset_name)
    print(f"[EVAL] Dataset: {dataset_config.name.upper()}")

    mel_len = dataset_config.get_mel_length(dataset_path, split_mel, split_len)
    print(f"[EVAL] Mel length: {mel_len}")

    device, batch_size = _setup_eval_environment(cfg, dataset_config)
    print(f"[EVAL] Device: {device}, Batch size: {batch_size}")

    # Load mel features
    mel_features_path = dataset_path + "../log_mel_features.pickle"
    with open(mel_features_path, "rb") as f:
        log_mel_features = pickle.load(f)
    if isinstance(log_mel_features, np.ndarray):
        log_mel_features = torch.from_numpy(log_mel_features).float()
    log_mel_features = log_mel_features.to(device)
    actual_mel_len = log_mel_features.shape[2]
    print(f"[EVAL] Mel features: {tuple(log_mel_features.shape)}  dtype={log_mel_features.dtype}")

    all_sents = _load_all_sents(dataset_path)
    print(f"[EVAL] Loaded {len(all_sents)} sentences")

    test_set = _load_test_dataset(cfg, dataset_config)
    print(f"[EVAL] Test set size: {len(test_set)}")

    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    tokenizer = WhisperTokenizer.from_pretrained(dataset_config.whisper_model)
    pretrained = WhisperForConditionalGeneration.from_pretrained(dataset_config.whisper_model)

    model_mel_len = mel_len

    model_type = cfg.get('model_type', 'brainecho')
    if model_type == 'brainecho_biot':
        model = BrainECHOBIOT(
            None,
            in_chan=dataset_config.in_channels,
            d_model=256,
            ffn_dim=1024,
            num_layers=4,
            mel_len=mel_len,
            codebook_size=cfg.get('cb_size', 2048),
            r=cfg.get('ratio', 4),
            split_mel=cfg.get('split_mel', False),
        )
    elif model_type == 'brainecho_stft':
        model = BrainECHOSTFT(
            None,
            in_chan=dataset_config.in_channels,
            d_model=256,
            ffn_dim=1024,
            num_layers=4,
            mel_len=mel_len,
            codebook_size=cfg.get('cb_size', 2048),
            r=cfg.get('ratio', 4),
            split_mel=cfg.get('split_mel', False),
        )
    else:
        model = BrainECHO(
            None,
            in_chan=dataset_config.in_channels,
            d_model=256,
            ffn_dim=1024,
            num_layers=4,
            mel_len=mel_len,
            codebook_size=cfg.get('cb_size', 2048),
            r=cfg.get('ratio', 4),
            split_mel=cfg.get('split_mel', False),
        )

    if not wo_ft:
        model.pretrained = pretrained

    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)

    if wo_ft:
        model.pretrained = pretrained.to(device)

    model.task = "task2"

    # Force decoder language for non-English datasets
    forced_decoder_ids = None
    if dataset_config.language and dataset_config.language != "en":
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained(dataset_config.whisper_model)
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=dataset_config.language, task="transcribe"
        )
        print(f"[EVAL] Forcing decoder language: {dataset_config.language}")

    results = _eval_model_brainecho(
        test_dataloader, device, tokenizer, model, dataset_config,
        is_teacher_forcing=is_teacher_forcing,
        split_mel=split_mel, mel_len=mel_len,
        input_noise=input_noise, length_noise=length_noise, random_select=random_select,
        codebook_size=codebook_size, all_sents=all_sents,
        forced_decoder_ids=forced_decoder_ids,
        output_all_results_path=output_all_results_path, logger=logger,
    )

    print(f"\n[EVAL] Results saved to: {output_all_results_path}")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return results




# ============================================================================
# ORIGINAL INTERNAL EVAL (kept for BrainECHO backward compatibility)
# ============================================================================

def _eval_model_brainecho(
    test_dataloader, device, tokenizer, model, dataset_config,
    is_teacher_forcing=False, split_mel=False, mel_len=1200,
    input_noise=False, length_noise=False, random_select=False, codebook_size=2048,
    all_sents=None, forced_decoder_ids=None,
    output_all_results_path="./results/temp.txt", logger=None,
):
    """BrainECHO evaluation — teacher forcing or beam search"""

    if is_teacher_forcing:
        model.eval()
        target_string_list = []
        pred_string_list = []
        subject_id_list = []
        start_time = time.time()
        codebook_usage = torch.zeros(codebook_size).to(device)
        compute_cb_usage = not random_select

        with open(output_all_results_path, "w") as f:
            for rawEEG, input_masks, target_ids, target_mask, sent_ids, target_string, subject_ids \
                    in tqdm(test_dataloader, ncols=80):

                if dataset_config.eeg_length_mode == "dynamic":
                    max_eeg_length = rawEEG.shape[-1]
                else:
                    max_eeg_length = dataset_config.fixed_eeg_length

                rawEEG_batch = rawEEG[:, :, :max_eeg_length].to(device).float()

                if input_noise:
                    rawEEG_batch = torch.randn_like(rawEEG_batch)

                if length_noise:
                    real_lens = input_masks.sum(dim=1).long()  # (batch,)
                    for i in range(rawEEG_batch.shape[0]):
                        real_len = real_lens[i].item()
                        rawEEG_batch[i, :, :real_len] = torch.randn(
                            rawEEG_batch.shape[1], real_len, device=device, dtype=rawEEG_batch.dtype
                        )

                if split_mel:
                    rawEEG_batch = rearrange(rawEEG_batch, "b c (p t)->(b p) c t", t=800)

                target_ids_batch = target_ids.to(device)
                target_mask_batch = target_mask.to(device)
                target_string = tokenizer.batch_decode(target_ids_batch, skip_special_tokens=True)
                target_string_list.extend(target_string)
                subject_id_list.extend(
                    subject_ids.tolist() if hasattr(subject_ids, 'tolist') else list(subject_ids)
                )
                target_ids_batch[target_mask_batch == 0] = -100

                with torch.no_grad():
                    output = model(rawEEG_batch, target_ids_batch, None)
                logits = output.logits
                probs = logits.softmax(dim=-1)
                _, predictions = probs.topk(1)
                predictions = torch.squeeze(predictions)

                if compute_cb_usage:
                    codebook_usage[output.indices.unique()] = 1

                predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                if random_select:
                    random_ids = random.choices(list(range(len(all_sents))), k=rawEEG_batch.shape[0])
                    predicted_string = [all_sents[i] for i in random_ids]

                for i in range(len(target_string)):
                    f.write(f"target string: {target_string[i]}\n")
                    f.write(f"predicted string: {predicted_string[i]}\n")
                    f.write(f"################################################\n\n\n")

                pred_string_list.extend(predicted_string)

        inference_time = time.time() - start_time
        result = compute_metrics(pred_string_list, target_string_list)
        if compute_cb_usage:
            result["codebook_usage"] = codebook_usage.sum().item() / codebook_size
        result["inference_time_seconds"] = inference_time
        result["num_samples"] = len(target_string_list)
        subject_stats = _compute_per_subject_metrics(
            pred_string_list, target_string_list, subject_id_list
        )
        result.update(subject_stats)
        if logger:
            logger.info(result)
        return result

    else:
        model.eval()
        target_string_list = []
        pred_string_list = []
        subject_id_list = []
        start_time = time.time()
        codebook_usage = torch.zeros(codebook_size).to(device)
        compute_cb_usage = not random_select

        with open(output_all_results_path, "w") as f:
            for rawEEG, input_masks, target_ids, target_mask, sent_ids, target_string, subject_ids \
                    in tqdm(test_dataloader, ncols=80):

                if dataset_config.eeg_length_mode == "dynamic":
                    max_eeg_length = rawEEG.shape[-1]
                else:
                    max_eeg_length = dataset_config.fixed_eeg_length

                rawEEG_batch = rawEEG[:, :, :max_eeg_length].to(device).float()

                if input_noise:
                    rawEEG_batch = torch.randn_like(rawEEG_batch)

                if length_noise:
                    real_lens = input_masks.sum(dim=1).long()  # (batch,)
                    for i in range(rawEEG_batch.shape[0]):
                        real_len = real_lens[i].item()
                        rawEEG_batch[i, :, :real_len] = torch.randn(
                            rawEEG_batch.shape[1], real_len, device=device, dtype=rawEEG_batch.dtype
                        )
                if split_mel:
                    rawEEG_batch = rearrange(rawEEG_batch, "b c (p t)->(b p) c t", t=mel_len*2)

                target_ids_batch = target_ids.to(device)
                target_mask_batch = target_mask.to(device)
                target_string = tokenizer.batch_decode(target_ids_batch, skip_special_tokens=True)
                target_string_list.extend(target_string)
                subject_id_list.extend(
                    subject_ids.tolist() if hasattr(subject_ids, 'tolist') else list(subject_ids)
                )
                target_ids_batch[target_mask_batch == 0] = -100

                output = model.generate(
                    rawEEG_batch, target_ids_batch,
                    max_length=256, num_beams=5,
                    do_sample=False, repetition_penalty=5.0,
                    no_repeat_ngram_size=2,
                    forced_decoder_ids=forced_decoder_ids,
                )
                predictions = output.predictions

                if compute_cb_usage:
                    codebook_usage[output.indices.unique()] = 1

                predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                if random_select:
                    random_ids = random.choices(list(range(len(all_sents))), k=rawEEG_batch.shape[0])
                    predicted_string = [all_sents[i] for i in random_ids]

                for i in range(len(target_string)):
                    f.write(f"target string: {target_string[i]}\n")
                    f.write(f"predicted string: {predicted_string[i]}\n")
                    f.write(f"################################################\n\n\n")

                pred_string_list.extend(predicted_string)

        run_time = math.ceil(time.time() - start_time)
        result = compute_metrics(pred_string_list, target_string_list)
        if compute_cb_usage:
            result["codebook_usage"] = codebook_usage.sum().item() / codebook_size
        result["inference_time_seconds"] = run_time
        result["num_samples"] = len(target_string_list)
        subject_stats = _compute_per_subject_metrics(
            pred_string_list, target_string_list, subject_id_list
        )
        result.update(subject_stats)
        if logger:
            logger.info(result)
        return result