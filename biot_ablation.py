#!/usr/bin/env python3
"""
BIOT Ablation Study — Length Bias Investigation
================================================
Trains and evaluates ablated variants of BIOT to identify which architectural
component is responsible for immunity to sentence-length bias in EEG classification.

Three components are ablated independently and in combination:
  - STFT tokenisation   (use_stft)
  - Channel tokens      (use_channel_tokens)
  - Positional encoding (use_positional_encoding)

When use_stft=False, a Conv1d patcher replaces the STFT + PatchFrequencyEmbedding
pipeline, producing the same (batch, ts, emb_size) sequence with the same
hop_length but operating directly on raw time-domain signal.

Datasets supported: brennan, hbn, neurasearch (same pickle format and
ClassificationDataset logic as train_eval_classification.py).

Usage
-----
# Full model (all components on) — baseline
python biot_ablation.py \\
    --dataset_name brennan \\
    --dataset_path ./datasets/Brennan/split_by_subject/ \\
    --save_path ./checkpoints/ablation/brennan/ \\
    --ablation full

# No positional encoding
python biot_ablation.py \\
    --dataset_name brennan \\
    --dataset_path ./datasets/Brennan/split_by_subject/ \\
    --save_path ./checkpoints/ablation/brennan/ \\
    --ablation no_pos_enc

# Run all 7 ablations across all 3 datasets
python biot_ablation.py --run_all

Output
------
Per run:  {save_path}/{dataset}_{ablation_name}_results.json
Summary:  {save_path}/ablation_summary.json  (when --run_all)
"""

import os
import sys
import math
import json
import time
import glob
import random
import pickle
import argparse
import collections
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from linear_attention_transformer import LinearAttentionTransformer
    _LINEAR_ATTN_OK = True
except ImportError:
    _LINEAR_ATTN_OK = False
    print("[WARN] linear_attention_transformer not installed. "
          "Install with: pip install linear-attention-transformer")


# ===========================================================================
# Ablation configurations
# Seven ablation variants covering all combinations of the three components
# plus the full model baseline.
# ===========================================================================

ABLATION_CONFIGS = {
    # name              : (use_stft, use_channel_tokens, use_pos_enc)
    "full"              : (True,  True,  True),   # baseline — all components on
    "no_pos_enc"        : (True,  True,  False),  # remove positional encoding only
    "no_chan_tok"       : (True,  False, True),   # remove channel tokens only
    "no_stft"           : (False, True,  True),   # replace STFT with conv patcher
    "no_pos_no_chan"    : (True,  False, False),  # remove pos enc + channel tokens
    "no_pos_no_stft"    : (False, True,  False),  # remove pos enc + replace STFT
    "no_chan_no_stft"   : (False, False, True),   # remove channel tokens + replace STFT
    "no_all"            : (False, False, False),  # remove all three components
}

# ===========================================================================
# Dataset configurations (identical to train_eval_classification.py)
# ===========================================================================

DATASET_CONFIGS = {
    "brennan":     {"in_channels": 62,  "eeg_length": 2400},
    "hbn":         {"in_channels": 129, "eeg_length": 2400},
    "neurasearch": {"in_channels": 62,  "eeg_length": None},
}

DATASET_PATHS = {
    "brennan":     "./datasets/Brennan/split_by_subject/",
    "hbn":         "./datasets/HBN/split_by_subject/",
    "neurasearch": "./datasets/NeuraSearch/split_by_subject/",
}

HBN_TASK_OFFSETS = {
    "DespicableMe":     0,
    "DiaryOfAWimpyKid": 76,
    "FunwithFractals":  133,
}


# ===========================================================================
# Dataset (copied from train_eval_classification.py for self-containment)
# ===========================================================================

class ClassificationDataset(Dataset):
    """
    Reads the same BrainECHO pickle format used in the main classification
    benchmark. Handles NeuraSearch global_sent_id, HBN per-task offsets, and
    ThePresent exclusion identically to train_eval_classification.py.
    """

    def __init__(self, path: str, eeg_length: Optional[int] = 2400,
                 subject: str = "all"):
        self.path = path
        self.eeg_length = eeg_length
        self.subject = subject
        self.samples: List[dict] = []
        self._load(path)

        if len(self.samples) == 0:
            raise RuntimeError(f"ClassificationDataset: no samples found in {path}")

        self.num_classes = max(s["sent_id"] for s in self.samples) + 1
        print(f"[INFO] ClassificationDataset: {len(self.samples)} samples, "
              f"{self.num_classes} classes, path={path}")
        self.subjects = sorted(set(s["subject"] for s in self.samples))

    def _load(self, path: str):
        pkl_files = sorted(
            glob.glob(os.path.join(path, "*.pkl")) +
            glob.glob(os.path.join(path, "*.pickle"))
        )
        if not pkl_files:
            raise RuntimeError(f"No pickle files found in {path}")

        for fpath in pkl_files:
            with open(fpath, "rb") as f:
                data = pickle.load(f)

            subj = data.get("subject", data.get("subject_id", "unknown"))
            if self.subject != "all" and str(subj) != str(self.subject):
                continue

            if "global_sent_id" in data:
                label = int(data["global_sent_id"])
            elif "task" in data:
                task = data["task"]
                if task not in HBN_TASK_OFFSETS:
                    continue
                label = HBN_TASK_OFFSETS[task] + int(data["sent_id"])
            else:
                label = int(data["sent_id"])

            self.samples.append({
                "rawEEG":          data["rawEEG"],
                "input_attn_mask": data.get("input_attn_mask", None),
                "sent_id":         label,
                "subject":         str(subj),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        eeg = s["rawEEG"]

        if self.eeg_length is not None:
            T = eeg.shape[-1]
            if isinstance(eeg, torch.Tensor):
                if T >= self.eeg_length:
                    eeg = eeg[..., :self.eeg_length]
                else:
                    pad = torch.zeros(*eeg.shape[:-1], self.eeg_length - T)
                    eeg = torch.cat([eeg, pad], dim=-1)
            else:
                if T >= self.eeg_length:
                    eeg = eeg[..., :self.eeg_length]
                else:
                    pad = np.zeros((*eeg.shape[:-1], self.eeg_length - T), dtype=eeg.dtype)
                    eeg = np.concatenate([eeg, pad], axis=-1)

        eeg_tensor = eeg.float() if isinstance(eeg, torch.Tensor) \
                     else torch.from_numpy(eeg).float()
        label = torch.tensor(s["sent_id"], dtype=torch.long)

        mask = s["input_attn_mask"]
        if mask is None:
            mask = torch.ones(eeg_tensor.shape[-1], dtype=torch.int32)
        elif not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.int32)

        return eeg_tensor, label, s["subject"], mask


def collate_fn(batch):
    eegs, labels, subjects, masks = zip(*batch)
    return torch.stack(eegs), torch.stack(labels), list(subjects), torch.stack(masks)


# ===========================================================================
# Ablatable BIOT components
# ===========================================================================

class _PatchFrequencyEmbedding(nn.Module):
    """Linear projection over frequency axis: (B, T, n_freq) → (B, T, emb_size)."""

    def __init__(self, emb_size: int = 256, n_freq: int = 101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, n_freq, T) → (B, T, n_freq)
        return self.projection(x) # (B, T, emb_size)


class _ConvPatcher(nn.Module):
    """
    Raw-time-domain replacement for STFT + PatchFrequencyEmbedding.

    Splits each channel's time series into overlapping windows of size
    `n_fft` with stride `hop_length` using a Conv1d layer, projecting each
    window directly to emb_size. Produces the same (B, ts, emb_size) shape
    as the STFT path so the transformer receives identically shaped input.
    """

    def __init__(self, emb_size: int = 256, n_fft: int = 200,
                 hop_length: int = 100):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=emb_size,
            kernel_size=n_fft,
            stride=hop_length,
            padding=0,
        )
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        """
        x : (B, n_times)  — single channel time series
        returns : (B, ts, emb_size)
        """
        x = x.unsqueeze(1)       # (B, 1, n_times)
        x = self.conv(x)          # (B, emb_size, ts)
        x = x.permute(0, 2, 1)   # (B, ts, emb_size)
        return self.norm(x)


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, emb_size: int, drop_prob: float = 0.1,
                 max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ===========================================================================
# Ablatable BIOT encoder
# ===========================================================================

class AblatedBIOTEncoder(nn.Module):
    """
    BIOT encoder with three independently switchable components:

      use_stft            : use STFT + PatchFrequencyEmbedding (True)
                            or Conv1d patcher (False)
      use_channel_tokens  : add learned per-channel identity embedding
      use_positional_enc  : add sinusoidal position embeddings

    All other architectural details (LinearAttentionTransformer, mean pooling,
    embedding dimension) are identical to the original BIOTEncoder.
    """

    def __init__(
        self,
        n_chans: int,
        emb_size: int = 256,
        num_heads: int = 8,
        n_layers: int = 4,
        n_fft: int = 200,
        hop_length: int = 100,
        drop_prob: float = 0.1,
        max_seq_len: int = 1024,
        attn_dropout: float = 0.2,
        attn_layer_dropout: float = 0.2,
        # ablation flags
        use_stft: bool = True,
        use_channel_tokens: bool = True,
        use_positional_enc: bool = True,
    ):
        super().__init__()
        if not _LINEAR_ATTN_OK:
            raise RuntimeError(
                "linear_attention_transformer is required. "
                "Install with: pip install linear-attention-transformer"
            )

        self.n_fft = int(n_fft)
        self.hop_length = hop_length
        self.use_stft = use_stft
        self.use_channel_tokens = use_channel_tokens
        self.use_positional_enc = use_positional_enc

        # --- Tokenisation pathway ---
        if use_stft:
            self.patch_embedding = _PatchFrequencyEmbedding(
                emb_size=emb_size,
                n_freq=int(self.n_fft // 2 + 1),
            )
        else:
            self.conv_patcher = _ConvPatcher(
                emb_size=emb_size,
                n_fft=n_fft,
                hop_length=hop_length,
            )

        # --- Channel identity tokens ---
        if use_channel_tokens:
            self.channel_tokens = nn.Embedding(
                num_embeddings=n_chans,
                embedding_dim=emb_size,
            )
            self.register_buffer(
                "index",
                torch.arange(n_chans, dtype=torch.long),
                persistent=False,
            )

        # --- Positional encoding ---
        if use_positional_enc:
            self.positional_encoding = _PositionalEncoding(
                emb_size=emb_size,
                drop_prob=drop_prob,
                max_len=max_seq_len,
            )

        # --- Transformer (unchanged) ---
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=num_heads,
            depth=n_layers,
            max_seq_len=max_seq_len,
            attn_layer_dropout=attn_layer_dropout,
            attn_dropout=attn_dropout,
        )

    def _stft(self, sample: torch.Tensor) -> torch.Tensor:
        """sample: (B, 1, n_times) → (B, n_fft//2+1, ts)"""
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def _tokenise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shared tokenisation logic for forward() and encode_sequence().
        x : (B, C, T)
        returns : (B, C*ts, emb_size)
        """
        emb_seq = []
        for i in range(x.shape[1]):
            chan = x[:, i:i + 1, :]
            if self.use_stft:
                spec = self._stft(chan)
                tok = self.patch_embedding(spec)
            else:
                tok = self.conv_patcher(chan.squeeze(1))
            B, ts, E = tok.shape
            if self.use_channel_tokens:
                chan_tok = (
                    self.channel_tokens(self.index[i])
                    .unsqueeze(0).unsqueeze(0)
                    .repeat(B, ts, 1)
                )
                tok = tok + chan_tok
            if self.use_positional_enc:
                tok = self.positional_encoding(tok)
            emb_seq.append(tok)
        return torch.cat(emb_seq, dim=1)  # (B, C*ts, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)
        returns : (B, emb_size)  — mean-pooled, for classification
        """
        emb = self._tokenise(x)
        return self.transformer(emb).mean(dim=1)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)
        returns : (B, C*ts, emb_size)  — full sequence, for BrainECHOBIOT
        """
        emb = self._tokenise(x)
        return self.transformer(emb)  # (B, C*ts, emb_size)


class AblatedBIOT(nn.Module):
    """
    Full BIOT model with ablatable encoder.
    Wraps AblatedBIOTEncoder with the standard BIOT classification head
    (ELU activation + linear layer).
    """

    def __init__(
        self,
        n_chans: int,
        n_classes: int,
        n_times: int,
        emb_size: int = 256,
        num_heads: int = 8,
        n_layers: int = 4,
        sfreq: int = 200,
        hop_length: int = 100,
        drop_prob: float = 0.5,
        max_seq_len: int = 1024,
        attn_dropout: float = 0.2,
        attn_layer_dropout: float = 0.2,
        use_stft: bool = True,
        use_channel_tokens: bool = True,
        use_positional_enc: bool = True,
    ):
        super().__init__()

        # Determine n_fft from sfreq (same logic as braindecode BIOT)
        n_fft = int(sfreq) if (n_times / sfreq) >= 1.0 else n_times

        self.encoder = AblatedBIOTEncoder(
            n_chans=n_chans,
            emb_size=emb_size,
            num_heads=num_heads,
            n_layers=n_layers,
            n_fft=n_fft,
            hop_length=hop_length,
            drop_prob=drop_prob,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            attn_layer_dropout=attn_layer_dropout,
            use_stft=use_stft,
            use_channel_tokens=use_channel_tokens,
            use_positional_enc=use_positional_enc,
        )

        self.head = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ===========================================================================
# Metrics
# ===========================================================================

def _subject_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    acc = float((y_pred == y_true).mean())
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": float(p),
            "recall": float(r), "f1": float(f)}


# ===========================================================================
# Training and evaluation
# ===========================================================================

def train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for eeg, labels, _, _masks in tqdm(loader, ncols=80, leave=False):
        eeg, labels = eeg.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(eeg)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss    += loss.item() * eeg.size(0)
        total_n       += eeg.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(
    model, loader, criterion, device,
    per_subject: bool = False,
    input_noise: bool = False,
    length_noise: bool = False,
) -> Dict:
    model.eval()
    total_loss = 0.0
    all_preds:  List[int] = []
    all_labels: List[int] = []
    subj_preds:  Dict[str, List[int]] = collections.defaultdict(list)
    subj_labels: Dict[str, List[int]] = collections.defaultdict(list)

    for eeg, labels, subjects, masks in loader:
        eeg, labels, masks = eeg.to(device), labels.to(device), masks.to(device)

        if input_noise:
            eeg = torch.randn_like(eeg)
        elif length_noise:
            real_lens = masks.sum(dim=1).long()
            for i in range(eeg.shape[0]):
                rl = real_lens[i].item()
                eeg[i, :, :rl] = torch.randn(
                    eeg.shape[1], rl, device=device, dtype=eeg.dtype
                )

        logits = model(eeg)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=-1)

        total_loss += loss.item() * eeg.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        if per_subject:
            for i, subj in enumerate(subjects):
                subj_preds[subj].append(int(preds[i].item()))
                subj_labels[subj].append(int(labels[i].item()))

    all_preds_np  = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    overall_acc   = float((all_preds_np == all_labels_np).mean())
    p, r, f, _    = precision_recall_fscore_support(
        all_labels_np, all_preds_np, average="macro", zero_division=0
    )

    result: Dict = {
        "loss":      total_loss / len(all_labels),
        "accuracy":  overall_acc,
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f),
    }

    if per_subject:
        result["per_subject"] = {
            subj: _subject_metrics(
                np.array(subj_labels[subj]),
                np.array(subj_preds[subj]),
            )
            for subj in sorted(subj_preds)
        }

    return result


def train_and_eval(
    model, train_loader, valid_loader, test_loader,
    num_classes, args, device, save_path, run_name,
) -> Dict:
    os.makedirs(save_path, exist_ok=True)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler    = CosineAnnealingLR(optimizer, T_max=args.epochs)
    ckpt_path    = os.path.join(save_path, f"{run_name}_best.pt")
    best_valid   = -1.0
    patience_ctr = 0

    if os.path.exists(ckpt_path):
        print(f"[INFO] Checkpoint found — skipping training: {ckpt_path}")
    else:
        for epoch in range(args.epochs):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                              criterion, device)
            val_m = evaluate(model, valid_loader, criterion, device)
            scheduler.step()

            print(f"Epoch {epoch:3d}/{args.epochs-1} | "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
                  f"valid_loss={val_m['loss']:.4f}  valid_acc={val_m['accuracy']:.4f} | "
                  f"{time.time()-t0:.1f}s")

            if val_m["accuracy"] > best_valid:
                best_valid = val_m["accuracy"]
                patience_ctr = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    print(f"\nRestoring best checkpoint ({ckpt_path})")
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )

    # Evaluate under all three conditions
    results = {}
    for condition, kwargs in [
        ("normal",       {}),
        ("input_noise",  {"input_noise":  True}),
        ("length_noise", {"length_noise": True}),
    ]:
        m = evaluate(model, test_loader, criterion, device,
                     per_subject=True, **kwargs)
        per_subj = m["per_subject"]

        def ms(key):
            vals = [v[key] for v in per_subj.values()]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)

        mean_acc,  std_acc  = ms("accuracy")
        mean_f1,   std_f1   = ms("f1")

        results[condition] = {
            "overall_accuracy":  m["accuracy"],
            "overall_f1":        m["f1"],
            "mean_accuracy":     mean_acc,
            "std_accuracy":      std_acc,
            "mean_f1":           mean_f1,
            "std_f1":            std_f1,
            "chance_level":      1.0 / num_classes,
            "num_classes":       num_classes,
            "per_subject":       per_subj,
        }

        print(f"\n  [{condition}] "
              f"Acc={m['accuracy']:.4f}  "
              f"F1={m['f1']:.4f}  "
              f"(subj mean: {mean_acc:.4f}±{std_acc:.4f})")

    # Compute length-bias sensitivity score:
    # drop in accuracy from normal to length_noise condition.
    # A large drop means the model is sensitive to the padding boundary.
    normal_acc = results["normal"]["mean_accuracy"]
    lnoise_acc = results["length_noise"]["mean_accuracy"]
    results["length_bias_sensitivity"] = float(normal_acc - lnoise_acc)

    print(f"\n  Length-bias sensitivity (normal - length_noise): "
          f"{results['length_bias_sensitivity']:.4f}  "
          f"(lower = more length-invariant)")

    return results


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="BIOT ablation study for length bias investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # dataset
    p.add_argument("--dataset_name", default="brennan",
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--dataset_path", default=None,
                   help="Override default dataset path")

    # ablation
    p.add_argument("--ablation", default="full",
                   choices=list(ABLATION_CONFIGS.keys()),
                   help="Which ablation variant to run")

    # model hyperparameters
    p.add_argument("--emb_size",   type=int, default=256)
    p.add_argument("--num_heads",  type=int, default=8)
    p.add_argument("--n_layers",   type=int, default=4)
    p.add_argument("--hop_length", type=int, default=100,
                   help="STFT / Conv1d stride (default 100 @ 200Hz = 0.5s)")
    p.add_argument("--drop_prob",  type=float, default=0.5)
    p.add_argument("--sfreq",      type=int, default=200)

    # training
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=312)

    # output
    p.add_argument("--save_path", default="./checkpoints/ablation/",
                   help="Root directory for checkpoints and results")

    # convenience: run all ablations × all datasets
    p.add_argument("--run_all", action="store_true",
                   help="Run all 8 ablation variants across all 3 datasets")
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASET_CONFIGS.keys()),
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Datasets to include when using --run_all")
    p.add_argument("--ablations", nargs="+",
                   default=list(ABLATION_CONFIGS.keys()),
                   choices=list(ABLATION_CONFIGS.keys()),
                   help="Ablations to include when using --run_all")

    return p.parse_args()


# ===========================================================================
# Single run helper
# ===========================================================================

def run_single(dataset_name: str, ablation_name: str, args,
               device: torch.device) -> Dict:
    """Run one dataset × one ablation configuration."""

    use_stft, use_chan_tok, use_pos_enc = ABLATION_CONFIGS[ablation_name]

    print("\n" + "="*70)
    print(f"Dataset  : {dataset_name}")
    print(f"Ablation : {ablation_name}  "
          f"(stft={use_stft}, chan_tok={use_chan_tok}, pos_enc={use_pos_enc})")
    print("="*70)

    dcfg = DATASET_CONFIGS[dataset_name]
    in_channels = dcfg["in_channels"]
    eeg_length  = dcfg["eeg_length"]

    dataset_path = args.dataset_path or DATASET_PATHS[dataset_name]
    if not dataset_path.endswith("/"):
        dataset_path += "/"

    train_set = ClassificationDataset(dataset_path + "train", eeg_length)
    valid_set = ClassificationDataset(dataset_path + "valid", eeg_length)
    test_set  = ClassificationDataset(dataset_path + "test",  eeg_length)

    num_classes = train_set.num_classes
    print(f"[INFO] num_classes: {num_classes}  |  chance: {1/num_classes:.4f}")

    if eeg_length is None:
        eeg_length = train_set[0][0].shape[-1]
        print(f"[INFO] Inferred eeg_length: {eeg_length}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn)

    model = AblatedBIOT(
        n_chans=in_channels,
        n_classes=num_classes,
        n_times=eeg_length,
        emb_size=args.emb_size,
        num_heads=args.num_heads,
        n_layers=args.n_layers,
        sfreq=args.sfreq,
        hop_length=args.hop_length,
        drop_prob=args.drop_prob,
        use_stft=use_stft,
        use_channel_tokens=use_chan_tok,
        use_positional_enc=use_pos_enc,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params: {n_params:,}")

    save_path = os.path.join(args.save_path, dataset_name)
    run_name  = f"{dataset_name}_{ablation_name}"

    results = train_and_eval(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        args=args,
        device=device,
        save_path=save_path,
        run_name=run_name,
    )

    results["ablation"]  = ablation_name
    results["dataset"]   = dataset_name
    results["config"] = {
        "use_stft":            use_stft,
        "use_channel_tokens":  use_chan_tok,
        "use_positional_enc":  use_pos_enc,
    }

    # Save per-run results
    def _ser(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        elif isinstance(v, dict):
            return {kk: _ser(vv) for kk, vv in v.items()}
        elif isinstance(v, (bool, float, int, str)):
            return v
        return v

    results_path = os.path.join(save_path, f"{run_name}_ablation_results.json")
    with open(results_path, "w") as f:
        json.dump({k: _ser(v) for k, v in results.items()}, f, indent=2)
    print(f"[INFO] Results saved to: {results_path}")

    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    if args.run_all:
        # Run all specified ablations × all specified datasets
        all_results = {}

        for dataset_name in args.datasets:
            all_results[dataset_name] = {}
            for ablation_name in args.ablations:
                r = run_single(dataset_name, ablation_name, args, device)
                all_results[dataset_name][ablation_name] = {
                    "normal_acc":           r["normal"]["mean_accuracy"],
                    "normal_f1":            r["normal"]["mean_f1"],
                    "length_noise_acc":     r["length_noise"]["mean_accuracy"],
                    "length_noise_f1":      r["length_noise"]["mean_f1"],
                    "input_noise_acc":      r["input_noise"]["mean_accuracy"],
                    "length_bias_sensitivity": r["length_bias_sensitivity"],
                }

        # Print summary table
        print("\n" + "="*80)
        print("ABLATION SUMMARY — Length Bias Sensitivity")
        print("(sensitivity = normal_acc - length_noise_acc; lower = more invariant)")
        print("="*80)
        for dataset_name in args.datasets:
            print(f"\n  {dataset_name.upper()}")
            print(f"  {'Ablation':<22}  {'Normal Acc':>10}  "
                  f"{'LNoise Acc':>10}  {'Sensitivity':>12}")
            print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*12}")
            for ablation_name, r in all_results[dataset_name].items():
                print(f"  {ablation_name:<22}  "
                      f"{r['normal_acc']:>10.4f}  "
                      f"{r['length_noise_acc']:>10.4f}  "
                      f"{r['length_bias_sensitivity']:>12.4f}")

        # Save summary
        summary_path = os.path.join(args.save_path, "ablation_summary.json")
        os.makedirs(args.save_path, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[INFO] Summary saved to: {summary_path}")

    else:
        run_single(args.dataset_name, args.ablation, args, device)


if __name__ == "__main__":
    main()
