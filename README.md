# On Length Bias in EEG-to-Text Decoding

This repository contains the code and benchmark infrastructure for the paper **"On Length Bias in EEG-to-Text Decoding"**, submitted to the NeurIPS 2026 Datasets and Benchmarks track. The paper introduces the **SeLis** (Sentence Listening) dataset, establishes a multi-dataset benchmark for listening EEG-to-text decoding, and systematically quantifies the role of sequence length cues in reported model performance.

The generation pipeline and model architectures are built on [BrainECHO](https://arxiv.org/abs/2410.14971) (Findings of ACL 2025). We thank the original authors for releasing their code.

---

## Datasets

The benchmark is built on three EEG datasets recorded during passive listening.

**Brennan** — 49 subjects listening to a continuous reading of *Alice in Wonderland*. EEG recorded at 62 channels. Available at: https://deepblue.lib.umich.edu/data/concern/data_sets/bg257f92t

**HBN (Healthy Brain Network)** — up to 136 subjects watching four short films (*Despicable Me*, *Diary of a Wimpy Kid*, *Fun with Fractals*, and *The Present*). EEG recorded at 129 channels (BIDS format). Released as part of the Child Mind Institute HBN project. Available at: https://nemar.org/dataexplorer/detail?dataset_id=ds005505 (Release 1, 136 subjects). The S3 path for direct download is `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1`.

**SeLis (Sentence Listening)** — 62-channel EEG recorded during listening, loud recall, imagined recall, and reading of 48 short sentences. This dataset is not yet publicly available. Preprocessing scripts will be released upon dataset publication.

---

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate BrainEchoRep3.10
```

Install the engine core package that provides the base runner and trainer interfaces:

```bash
pip install engineCoreNew-0.1.0-py3-none-any.whl
```

Install braindecode separately (v1.12 or later is required for the renamed model APIs):

```bash
pip install "braindecode>=1.12"
```

---

## Data Preprocessing

The expected directory structure after preprocessing is the same for all datasets:

```
datasets/
  Brennan/
    split_by_subject/
      train/   *.pickle
      valid/   *.pickle
      test/    *.pickle
    all_sents.pickle
    log_mel_features.pickle
  HBN/
    split_by_subject/
      train/   *.pickle
      valid/   *.pickle
      test/    *.pickle
    all_sents.pickle
    log_mel_features.pickle
  SeLis/
    split_by_subject/
      train/   *.pickle
      valid/   *.pickle
      test/    *.pickle
    all_sents.pickle
    log_mel_features.pickle
```

Each pickle file contains a single EEG trial with the following fields: `rawEEG` (channels × timepoints), `sent_id`, `subject`, `target_string`, and `input_attn_mask` (1 for real signal, 0 for zero-padding). HBN trials additionally include a `task` field (movie name). SeLis trials include a `global_sent_id` field (0–47, shared across experimental conditions).

The train/valid/test split is subject-held-out: entire subjects are withheld for validation and test, so no subject appears in more than one split. Splits are seeded with `random.seed(42)`.

### Brennan

Download the raw `.mat` subject files from the link above and place them in a local directory (referred to below as `ALICE_RAW`). All preprocessing scripts are in `preprocess/Brennan/`.

**Step 1** — Transcribe audio stimuli using WhisperX:

```bash
python preprocess/Brennan/asr_pipeline.py \
    --input_dir $ALICE_RAW/audio \
    --output_dir $ALICE_RAW/processed \
    --backend whisperx
```

**Step 2** — Segment EEG into per-trial pickles and create an intermediate flat split. Edit `base_path` in the script to point to `ALICE_RAW` before running:

```bash
python preprocess/Brennan/data_processing_split_by_shuffle.py
```

**Step 3** — Re-partition into subject-held-out train/valid/test:

```bash
python preprocess/Brennan/data_processing_split_by_subject.py
```

This writes the final per-trial pickles to `datasets/Brennan/split_by_subject/`.

**Step 4** — Extract mel spectrograms from the audio stimuli and save as `log_mel_features.pickle` at `datasets/Brennan/log_mel_features.pickle` (one level above `split_by_subject/`).

### HBN

All preprocessing scripts are in `preprocess/HBN/`.

**Step 1** — Download the movie audio stimuli:

```bash
python preprocess/HBN/download_hbn_audio_stimuli.py
```

**Step 2** — Segment and resample audio to 16 kHz:

```bash
python preprocess/HBN/hbn_audio_preprocessing.py
```

**Step 3** — Transcribe audio with WhisperX:

```bash
python preprocess/Brennan/asr_pipeline.py \
    --input_dir ./datasets/HBN/audio \
    --output_dir ./datasets/HBN/processed \
    --backend whisperx
```

**Step 4** — Convert raw BIDS EDF/BDF files to per-trial pickles, aligned to transcript timestamps:

```bash
python preprocess/HBN/initial_hbn_preprocessing.py
```

**Step 5** — Create the subject-held-out split:

```bash
python preprocess/HBN/data_splitting_hbn.py
```

This writes the final pickles to `datasets/HBN/split_by_subject/`. The *ThePresent* movie is excluded from training and validation by default; it can optionally be excluded from evaluation as well (see the `exclude_the_present_eval` flag below).

### SeLis

SeLis is not yet publicly available. Preprocessing scripts will be released alongside the dataset.

---

## Generation Experiments

The generation pipeline runs in three stages. Stage 1 trains a Mel-VQVAE autoencoder on audio spectrograms. Stages 2 and 3 train the EEG encoder for brain-audio alignment and then fine-tune Whisper for text decoding.

All experiments are launched through `main_run.py` using Hydra configuration. Dataset-specific parameters (channel count, EEG length, mel length, AdaLoRA schedule) are defined in `dataset_configs.py` and selected automatically via `dataset_name`. Valid values for `dataset_name` are `brennan` and `hbn`.

### Stage 1 — Mel-VQVAE Pretraining

```bash
python main_run.py \
    dataset=brennan dataset_name=brennan \
    dataset_path=./datasets/Brennan/split_by_subject/ \
    save_path=./checkpoints/brennan/ \
    log_dir=./logs/brennan/ \
    vqvae_dataset=brennan \
    model_type=brainecho \
    stages_to_run=[1] \
    checkpoint=null
```

The checkpoint is saved inside `save_path` with the filename pattern `MelVQVAE_b{batch_size}_{epochs}_{lr}_autoencode.pt`. With default training settings this produces `MelVQVAE_b16_400_0.0002_autoencode.pt`. Verify the exact filename in your checkpoints directory before proceeding to Stages 2 and 3.

When running HBN, set `vqvae_dataset=hbn` to ensure the VQVAE trains on the correct dataset config. Run Stage 1 for each dataset before proceeding.

### Stages 2 and 3 — Brain-Audio Alignment and Whisper Fine-Tuning

Three model variants are available: `brainecho` (convolutional EEG encoder), `brainecho_biot` (BIOT patch encoder), and `brainecho_stft` (STFT patch encoder).

```bash
python main_run.py \
    dataset=brennan dataset_name=brennan \
    dataset_path=./datasets/Brennan/split_by_subject/ \
    save_path=./checkpoints/brennan/ \
    log_dir=./logs/brennan/ \
    model_type=brainecho \
    checkpoint=./checkpoints/brennan/MelVQVAE_b16_400_0.0002_autoencode.pt \
    stages_to_run=[2,3]
```

Stage 2 produces an alignment checkpoint and Stage 3 produces the final fine-tuned checkpoint. Both follow the pattern `b{batch_size}_{epochs}_{lr}_{stage}.pt` — verify the exact filenames in your checkpoints directory. Both are needed downstream.

### Evaluation

To evaluate a trained checkpoint without rerunning training. The `eval_checkpoint` should point to the Stage 3 finetune checkpoint (verify the exact filename in your checkpoints directory — it follows the pattern `b{batch_size}_{epochs}_{lr}_finetune.pt`):

```bash
python main_run.py \
    dataset=brennan dataset_name=brennan \
    dataset_path=./datasets/Brennan/split_by_subject/ \
    save_path=./checkpoints/brennan/ \
    log_dir=./logs/brennan/ \
    model_type=brainecho \
    checkpoint=null \
    stages_to_run=[] \
    run_evaluation=true \
    eval_checkpoint=./checkpoints/brennan/b16_40_0.0001_finetune.pt \
    eval_results_name=brennan_brainecho_results \
    eval_output_dir=./result \
    save_results=true
```

Metrics reported are BLEU-1/2/3/4, WER, ROUGE-1/2/L — both overall and as mean ± SD across held-out test subjects.

### Noise and Baseline Conditions

Three baseline conditions can be appended to any evaluation call:

`eval_input_noise=true` replaces the entire EEG tensor with Gaussian noise before inference.

`eval_length_noise=true` replaces only the real-signal region of each trial with Gaussian noise while preserving the zero-padding boundary. This isolates the contribution of sequence length cues to model performance.

`eval_random_select=true` ignores the model's output entirely and returns a randomly drawn sentence from the stimulus set.

### Running All Generation Evaluations

Once all checkpoints are trained, the full evaluation matrix (all models × all datasets × all conditions) can be run with:

```bash
bash run_all_evaluations.sh               # all datasets
bash run_all_evaluations.sh brennan       # single dataset
bash run_all_evaluations.sh brennan hbn   # subset
```

Update the checkpoint path variables at the top of the script to match your directory layout before running. Results are written to `./result/` and logs to `./logs/`.

**HBN note:** all HBN evaluations automatically pass `exclude_the_present_eval=true`, which excludes *ThePresent* trials from the test set to ensure a consistent evaluation corpus across runs.

---

## Classification Experiments

The classification benchmark trains models to predict sentence identity (which of the N stimulus sentences was heard) from a single EEG trial. This is a closed-set task and serves as a complementary diagnostic alongside the open-vocabulary generation experiments.

Two families of models are evaluated.

**Braindecode baselines** — seven end-to-end EEG classifiers trained directly on raw EEG: EEGNet, ShallowConvNet, DeepConvNet, EEGConformer, TIDNet, ATCNet, and BIOT.

**Frozen encoder classifiers** — the EEG encoder from each BrainECHO variant (standard, BIOT, STFT) is loaded from its Stage 2 alignment checkpoint, frozen, and a linear classification head is trained on top.

### Training and Evaluation

```bash
# Braindecode model (end-to-end)
python train_eval_classification.py \
    --dataset_name brennan \
    --dataset_path ./datasets/Brennan/split_by_subject/ \
    --model_name eegnet \
    --save_path ./checkpoints/classification/brennan/eegnet/

# Frozen BrainECHO encoder
python train_eval_classification.py \
    --dataset_name brennan \
    --dataset_path ./datasets/Brennan/split_by_subject/ \
    --model_name brainecho \
    --checkpoint ./checkpoints/brennan/b16_40_0.0001_alignment.pt \
    --in_chan 62 --d_model 256 --mel_len 1200 \
    --save_path ./checkpoints/classification/brennan/brainecho/
```

Noise conditions are applied at test time via `--input_noise` (full Gaussian noise) or `--length_noise` (length-preserving noise). These flags do not affect training.

### Running All Classification Evaluations

```bash
bash run_classification_eval.sh                              # all
bash run_classification_eval.sh --dataset brennan
bash run_classification_eval.sh --dataset brennan --model eegnet
bash run_classification_eval.sh --dataset brennan --condition length_noise
```

For frozen encoder models, the script trains the classification head once (normal condition) and caches the checkpoint. Subsequent condition evaluations load the cached head. For braindecode models, each condition run retrains from scratch — since noise is applied only at test time, training is always on clean EEG regardless of the condition flag.

Results are written as JSON to `./checkpoints/classification/` and logs to `./logs/classification_eval/`.

---

## Classification-to-Generation Bridge

To translate classification predictions into generation metrics (BLEU, WER, ROUGE), the predicted sentence class is used to retrieve the corresponding sentence text, which is then compared against the ground truth. This bridges the two benchmarks and answers the question: *what generation performance would a model achieve if it perfectly identified which sentence was heard?*

```bash
python classification_to_generation.py \
    --dataset_name brennan \
    --dataset_path ./datasets/Brennan/split_by_subject/ \
    --model_name eegnet \
    --cls_checkpoint ./checkpoints/classification/brennan/eegnet/brennan_eegnet_best.pt \
    --save_path ./checkpoints/classification_to_generation/brennan/eegnet/
```

For frozen encoder models, additionally pass `--encoder_checkpoint` pointing to the Stage 2 alignment checkpoint. The same `--input_noise` and `--length_noise` flags are available.

To run all model × dataset × condition combinations:

```bash
bash run_cls2gen.sh
bash run_cls2gen.sh --dataset brennan
bash run_cls2gen.sh --dataset brennan --model eegnet
```

Results are written to `./checkpoints/classification_to_generation/` and logs to `./logs/cls2gen/`.

---

## Checkpoint Naming Conventions

Checkpoints are named automatically based on training hyperparameters following the pattern `b{batch_size}_{epochs}_{lr}_{stage}.pt`. With default settings the files produced are:

| Stage | Filename | Used by |
|---|---|---|
| Stage 1 | `MelVQVAE_b16_400_0.0002_autoencode.pt` | Input to Stages 2/3 |
| Stage 2 | `b16_40_0.0001_alignment.pt` | Frozen encoder classification; Cls2Gen |
| Stage 3 | `b16_40_0.0001_finetune.pt` | Generation evaluation |
| Classification | `{dataset}_{model}_best.pt` | Cls2Gen |

Always verify exact filenames in your checkpoints directory before running downstream steps.

---

## Citation

If you use the BrainECHO model architecture, please also cite the original work:

```bibtex
@inproceedings{brainecho2025,
  title     = {{BrainECHO}: Semantic Brain Signal Decoding through Vector-Quantized
               Spectrogram Reconstruction for Whisper-Enhanced Text Generation},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  year      = {2025},
}
```

If you use the Brennan dataset:

```bibtex
@misc{brennan2019,
  author    = {Brennan, Jonathan R. and Hale, John T.},
  title     = {Hierarchical structure guides rapid linguistic predictions during naturalistic listening},
  year      = {2019},
  publisher = {University of Michigan},
  note      = {\url{https://deepblue.lib.umich.edu/data/concern/data_sets/bg257f92t}},
}
```

If you use the HBN dataset:

```bibtex
@article{alexander2017,
  author    = {Alexander, Lindsay M. and others},
  title     = {An open resource for transdiagnostic research in pediatric mental health and learning disorders},
  journal   = {Scientific Data},
  year      = {2017},
  volume    = {4},
  pages     = {170181},
}
```
