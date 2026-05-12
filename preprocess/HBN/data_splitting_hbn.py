import librosa
import mne
import math
import torch
import numpy as np
import codecs
import pickle
import os
import json
import random
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import robust_scale
import pandas as pd
from transformers import WhisperProcessor, WhisperTokenizer
import gc

# ============================================================================
# CONFIGURATION FLAGS
# ============================================================================

# EEG and Audio Parameters
EEG_SAMPLING_RATE = 200  # Hz (from preprocessing)
AUDIO_SAMPLING_RATE = 16000  # Hz (Whisper standard)
MAX_SEGMENT_DURATION = 12.0  # seconds (max segment length)

# Derived parameters
MAX_EEG_SAMPLES = int(MAX_SEGMENT_DURATION * EEG_SAMPLING_RATE)  # 2400 samples
# For Whisper: hop_length=160, so 100 frames per second
MAX_MEL_FRAMES = int(MAX_SEGMENT_DURATION * (AUDIO_SAMPLING_RATE / 160))  # 1200 frames

# Text parameters
MAX_TEXT_LENGTH = 45  # tokens

# How to handle "The Present" movie
THE_PRESENT_CONFIG = {
    'use_fixed_segments': True,  # True = fixed intervals, False = WhisperX
    'segment_duration': 12.0,  # Duration in seconds for fixed segments
    'include_in_stage1': True,  # Include in mel-spectrogram training
    'include_in_stage2': True,  # Include in EEG-audio alignment
    'include_in_stage3': False,  # Include in Whisper fine-tuning (needs text)
    'dummy_text': True  # If True, use "segment_N" as text; if False, empty string
}

print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"EEG sampling rate: {EEG_SAMPLING_RATE} Hz")
print(f"Audio sampling rate: {AUDIO_SAMPLING_RATE} Hz")
print(f"Max segment duration: {MAX_SEGMENT_DURATION} seconds")
print(f"Max EEG samples: {MAX_EEG_SAMPLES} samples")
print(f"Max mel frames: {MAX_MEL_FRAMES} frames")
print(f"Max text length: {MAX_TEXT_LENGTH} tokens")
print("=" * 60 + "\n")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_preprocessed_eeg(subject_path, task_name):
    """Load preprocessed EEG data for a specific subject and task."""
    subject_id = subject_path.name

    eeg_file = subject_path / f"{subject_id}_task-{task_name}_eeg.set"
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)

    events_file = subject_path / f"{subject_id}_task-{task_name}_events.tsv"
    events_df = pd.read_csv(events_file, sep='\t')

    return raw, events_df


def get_video_timing(events_df):
    """Extract video start and stop sample indices from events."""
    video_start = events_df[events_df['value'] == 'video_start']
    video_stop = events_df[events_df['value'] == 'video_stop']

    if len(video_start) == 0 or len(video_stop) == 0:
        raise ValueError("Video start/stop events not found")

    return int(video_start['sample'].values[0]), int(video_stop['sample'].values[0])


def create_fixed_segments(audio_duration, segment_duration=10.0):
    """
    Create fixed-duration segments for movies without proper transcription.

    Parameters:
    -----------
    audio_duration : float
        Total duration of audio in seconds
    segment_duration : float
        Duration of each segment in seconds

    Returns:
    --------
    list: List of segment dictionaries with 'start', 'end', 'text'
    """
    segments = []
    num_segments = int(np.ceil(audio_duration / segment_duration))

    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, audio_duration)

        # Create dummy text if configured
        if THE_PRESENT_CONFIG['dummy_text']:
            text = f" segment_{i}"  # Space prefix like WhisperX
        else:
            text = " "  # Minimal text

        segments.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })

    return segments


def generate_all_mel_features(movies_audio_dir, transcription_dir, tasks, processor):
    """
    Generate mel-spectrograms for all audio segments across all movies.
    Returns: (all_mels list, all_sents list, segment_mapping dict)
    """
    all_mels = []
    all_sents = []
    segment_mapping = {}  # Maps (task, sent_id) -> index in all_mels

    global_idx = 0
    for task in tasks:
        # Load audio
        audio_file = Path(movies_audio_dir) / f"{task}_16kHz.wav"
        audio, sr = librosa.load(audio_file, sr=AUDIO_SAMPLING_RATE)

        # Get segments
        if task == 'ThePresent' and THE_PRESENT_CONFIG['use_fixed_segments']:
            audio_duration = len(audio) / sr
            segments = create_fixed_segments(audio_duration, THE_PRESENT_CONFIG['segment_duration'])
        else:
            transcription_file = Path(transcription_dir) / f"{task}_16kHz.json"
            with open(transcription_file, 'r') as f:
                transcription = json.load(f)
            segments = transcription['segments']

        # Generate mel for each segment
        for seg_idx, segment in enumerate(segments):
            text = segment.get('text', '').strip()
            if not text:
                continue
            if not text.startswith(' '):
                text = f' {text}'

            audio_segment = process_audio_segment(audio, segment)
            mel_features = processor(audio_segment, sampling_rate=AUDIO_SAMPLING_RATE,
                                     return_tensors='pt').input_features

            # Adjust to MAX_MEL_FRAMES
            if mel_features.shape[2] > MAX_MEL_FRAMES:
                mel_features = mel_features[:, :, :MAX_MEL_FRAMES]
            elif mel_features.shape[2] < MAX_MEL_FRAMES:
                padding = torch.zeros((1, 80, MAX_MEL_FRAMES - mel_features.shape[2]))
                mel_features = torch.cat([mel_features, padding], dim=2)

            all_mels.append(mel_features[0])  # (80, 1200)
            all_sents.append(text)
            segment_mapping[(task, seg_idx)] = global_idx
            global_idx += 1

    return all_mels, all_sents, segment_mapping

def process_eeg_segment(raw_data, segment, video_start_sample,
                        eeg_sr=EEG_SAMPLING_RATE, max_len=MAX_EEG_SAMPLES):
    """Extract and pad EEG segment based on audio timing."""
    segment_start_eeg = video_start_sample + int(segment['start'] * eeg_sr)
    segment_end_eeg = video_start_sample + int(segment['end'] * eeg_sr)
    eeg_len = segment_end_eeg - segment_start_eeg

    eeg_segment = raw_data[:, segment_start_eeg:segment_end_eeg]

    if eeg_len < max_len:
        padding = np.zeros((raw_data.shape[0], max_len - eeg_len))
        eeg_padded = np.concatenate([eeg_segment, padding], axis=1)
    else:
        eeg_padded = eeg_segment[:, :max_len]
        eeg_len = max_len

    return eeg_padded, eeg_len


def process_audio_segment(audio, segment, audio_sr=AUDIO_SAMPLING_RATE):
    """Extract audio segment."""
    audio_start = int(segment['start'] * audio_sr)
    audio_end = int(segment['end'] * audio_sr)
    return audio[audio_start:audio_end]


def create_input_sample(eeg_padded, eeg_len, text, subject_id,
                        task_name, sent_id, tokenizer,
                        max_seq_len=MAX_TEXT_LENGTH, max_eeg_len=MAX_EEG_SAMPLES):
    """Create a single training sample in BrainECHO format."""
    input_sample = {
        'subject': subject_id,
        'task': task_name,
        'sent_id': sent_id,
        'rawEEG': eeg_padded,
        'target_string': text,
    }

    target_tokenized = tokenizer(
        text, padding='max_length', max_length=max_seq_len,
        truncation=True, return_tensors='pt', return_attention_mask=True
    )

    input_sample['target_ids'] = target_tokenized['input_ids'][0].numpy()
    input_sample['target_mask'] = target_tokenized['attention_mask'][0].numpy()

    # Create attention mask for EEG (0 = masked, 1 = not masked)
    input_sample['input_attn_mask'] = np.zeros(max_eeg_len, dtype=np.int32)
    input_sample['input_attn_mask'][:eeg_len] = 1

    return input_sample


def process_subject(subject_path, movies_audio_dir, transcription_dir,
                    tasks, tokenizer, verbose=True):
    """Process all tasks for a single subject."""
    subject_id = subject_path.name
    all_samples = []

    if verbose:
        print(f"Processing {subject_id}...")

    for task in tasks:
        try:
            raw, events_df = load_preprocessed_eeg(subject_path, task)
            video_start_sample, video_stop_sample = get_video_timing(events_df)

            raw_data = raw.get_data()
            raw_data = robust_scale(raw_data.T).T
            raw_data = np.clip(raw_data, -10, 10)
            raw_data = raw_data / np.max(np.abs(raw_data))

            if raw_data.shape[0] < 129:
                padding = np.zeros((129 - raw_data.shape[0], raw_data.shape[1]))
                raw_data = np.vstack([raw_data, padding])

            # Get segments - either from WhisperX or fixed intervals
            if task == 'ThePresent' and THE_PRESENT_CONFIG['use_fixed_segments']:
                # Use fixed segments for ThePresent
                audio_file = Path(movies_audio_dir) / f"{task}_16kHz.wav"
                audio, sr = librosa.load(audio_file, sr=AUDIO_SAMPLING_RATE)
                audio_duration = len(audio) / sr

                segments = create_fixed_segments(
                    audio_duration,
                    THE_PRESENT_CONFIG['segment_duration']
                )
                if verbose:
                    print(f"  {task}: Using fixed {THE_PRESENT_CONFIG['segment_duration']}s segments")
            else:
                # Use WhisperX transcription
                transcription_file = Path(transcription_dir) / f"{task}_16kHz.json"
                with open(transcription_file, 'r') as f:
                    transcription = json.load(f)
                segments = transcription['segments']

            # Process each segment
            for seg_idx, segment in enumerate(segments):
                text = segment.get('text', '').strip()
                if not text:
                    continue
                if not text.startswith(' '):
                    text = f' {text}'

                # Extract EEG only
                eeg_padded, eeg_len = process_eeg_segment(raw_data, segment, video_start_sample)

                sample = create_input_sample(
                    eeg_padded, eeg_len, text,
                    subject_id, task, seg_idx, tokenizer
                )

                all_samples.append(sample)

            if verbose:
                print(f"  {task}: {len([s for s in all_samples if s['task'] == task])} segments")

        except Exception as e:
            print(f"  Error processing {task}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_samples


# ============================================================================
# SPLITTING FUNCTIONS
# ============================================================================

def split_by_instance(samples, train_ratio=0.8, valid_ratio=0.1, seed=42):
    """Split by instance: shuffle all samples randomly."""
    random.seed(seed)
    samples_copy = samples.copy()
    random.shuffle(samples_copy)

    n = len(samples_copy)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    return {
        'train': samples_copy[:train_end],
        'valid': samples_copy[train_end:valid_end],
        'test': samples_copy[valid_end:]
    }


def split_by_sentence_within_subject(samples, train_ratio=0.8, valid_ratio=0.1, seed=42):
    """Split by sentence within subject (Brennan style)."""
    random.seed(seed)

    subjects_dict = {}
    for sample in samples:
        subj = sample['subject']
        if subj not in subjects_dict:
            subjects_dict[subj] = []
        subjects_dict[subj].append(sample)

    splits = {'train': [], 'valid': [], 'test': []}

    for subj, subj_samples in subjects_dict.items():
        random.shuffle(subj_samples)
        n = len(subj_samples)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))

        splits['train'].extend(subj_samples[:train_end])
        splits['valid'].extend(subj_samples[train_end:valid_end])
        splits['test'].extend(subj_samples[valid_end:])

    return splits


def save_sample(sample, output_file):
    """Save a single sample to pickle file (numpy format for efficiency)."""
    with codecs.open(output_file, 'wb') as handle:
        pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)


# ============================================================================
# BATCH PROCESSING FOR split_by_instance AND split_by_sentence_within_subject
# ============================================================================

def process_batch_simple_splits(
        subject_dirs,
        movies_audio_dir,
        transcription_dir,
        output_dir,
        tasks,
        tokenizer,
        split_types=['split_by_instance', 'split_by_sentence_within_subject'],
        train_ratio=0.8,
        valid_ratio=0.1,
        seed=42
):
    """
    Process a batch of subjects and save to splits that can be done in batches.
    """
    batch_samples = []

    for subject_dir in tqdm(subject_dirs, desc="Processing batch"):
        try:
            samples = process_subject(
                subject_dir, movies_audio_dir, transcription_dir,
                tasks, tokenizer, verbose=False
            )
            batch_samples.extend(samples)
        except Exception as e:
            print(f"Error processing {subject_dir.name}: {e}")
            continue

    if len(batch_samples) == 0:
        print("No samples in batch, skipping...")
        return []

    # Save to each split type
    for split_type in split_types:
        if split_type == 'split_by_instance':
            splits = split_by_instance(batch_samples, train_ratio, valid_ratio, seed)
        elif split_type == 'split_by_sentence_within_subject':
            splits = split_by_sentence_within_subject(batch_samples, train_ratio, valid_ratio, seed)
        else:
            continue

        for split_name in ['train', 'valid', 'test']:
            split_dir = output_dir / split_type / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for sample in splits[split_name]:
                filename = f"{sample['subject']}-task-{sample['task']}-sent-{sample['sent_id']}.pickle"
                save_sample(sample, split_dir / filename)

    # Return samples for mel collection
    return batch_samples


# ============================================================================
# MEMORY-EFFICIENT split_by_subject
# ============================================================================

def split_by_subject_efficient(
        all_subject_dirs,
        movies_audio_dir,
        transcription_dir,
        output_dir,
        tasks,
        tokenizer,
        train_ratio=0.7,
        valid_ratio=0.15,
        batch_size=10,
        seed=42
):
    """
    Memory-efficient split by subject.
    Splits subject names first, then processes each split separately in batches.
    """
    random.seed(seed)

    # 1. Get subject names and shuffle
    subject_names = [d.name for d in all_subject_dirs]
    random.shuffle(subject_names)

    # 2. Split subject names into train/valid/test
    n_subj = len(subject_names)
    train_end = int(n_subj * train_ratio)
    valid_end = int(n_subj * (train_ratio + valid_ratio))

    splits_subjects = {
        'train': subject_names[:train_end],
        'valid': subject_names[train_end:valid_end],
        'test': subject_names[valid_end:]
    }

    print(f"\nSplit assignment:")
    print(f"  Train subjects: {len(splits_subjects['train'])}")
    print(f"  Valid subjects: {len(splits_subjects['valid'])}")
    print(f"  Test subjects: {len(splits_subjects['test'])}")

    all_samples_collected = []

    # 3. Process each split separately in batches
    for split_name, split_subjects in splits_subjects.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {split_name} split ({len(split_subjects)} subjects)")
        print(f"{'=' * 60}")

        split_dir = output_dir / 'split_by_subject' / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        num_batches = (len(split_subjects) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(split_subjects))
            batch_subject_names = split_subjects[start_idx:end_idx]

            print(f"\n  Batch {batch_idx + 1}/{num_batches} ({len(batch_subject_names)} subjects)")

            batch_dirs = [d for d in all_subject_dirs if d.name in batch_subject_names]

            batch_samples = []
            for subject_dir in tqdm(batch_dirs, desc=f"  Processing {split_name}"):
                try:
                    samples = process_subject(
                        subject_dir, movies_audio_dir, transcription_dir,
                        tasks, tokenizer, verbose=False
                    )
                    batch_samples.extend(samples)
                except Exception as e:
                    print(f"    Error processing {subject_dir.name}: {e}")
                    continue

            if len(batch_samples) == 0:
                print(f"    No samples in batch, skipping...")
                continue

            print(f"    Saving {len(batch_samples)} samples to {split_name}/...")
            for sample in batch_samples:
                filename = f"{sample['subject']}-task-{sample['task']}-sent-{sample['sent_id']}.pickle"
                save_sample(sample, split_dir / filename)

            # Collect for mel features
            all_samples_collected.extend(batch_samples)

            del batch_samples
            gc.collect()

        print(f"\n✓ {split_name} split complete")

    print(f"\n{'=' * 60}")
    print("✓ split_by_subject complete")
    print(f"{'=' * 60}")

    return all_samples_collected


# ============================================================================
# MEMORY-EFFICIENT split_by_sentence_holdout
# ============================================================================

def split_by_sentence_holdout_efficient(
        all_subject_dirs,
        movies_audio_dir,
        transcription_dir,
        output_dir,
        tasks,
        tokenizer,
        train_ratio=0.8,
        valid_ratio=0.1,
        batch_size=10,
        seed=42
):
    """
    Memory-efficient split by sentence holdout.

    Strategy:
    1. Process one subject to get all unique sentences
    2. Split sentences into train/valid/test
    3. Process all subjects in batches, assigning samples based on sentence split
    """
    random.seed(seed)

    print(f"\n{'=' * 60}")
    print("SPLIT BY SENTENCE HOLDOUT")
    print(f"{'=' * 60}")

    # 1. Get all unique sentences from one subject
    print("\nStep 1: Extracting unique sentences from first subject...")
    first_subject = all_subject_dirs[0]

    # Load transcriptions to get all unique sentences
    unique_sentences = set()

    for task in tasks:
        try:
            if task == 'ThePresent' and THE_PRESENT_CONFIG['use_fixed_segments']:
                # For ThePresent with fixed segments
                audio_file = Path(movies_audio_dir) / f"{task}_16kHz.wav"
                audio, sr = librosa.load(audio_file, sr=AUDIO_SAMPLING_RATE)
                audio_duration = len(audio) / sr
                segments = create_fixed_segments(audio_duration, THE_PRESENT_CONFIG['segment_duration'])
            else:
                # Load WhisperX transcription
                transcription_file = Path(transcription_dir) / f"{task}_16kHz.json"
                with open(transcription_file, 'r') as f:
                    transcription = json.load(f)
                segments = transcription['segments']

            for segment in segments:
                text = segment.get('text', '').strip()
                if text:
                    if not text.startswith(' '):
                        text = f' {text}'
                    unique_sentences.add(text)
        except Exception as e:
            print(f"  Error loading {task}: {e}")
            continue

    print(f"  Found {len(unique_sentences)} unique sentences")

    # 2. Split sentences into train/valid/test
    sentences_list = list(unique_sentences)
    random.shuffle(sentences_list)

    n_sent = len(sentences_list)
    train_end = int(n_sent * train_ratio)
    valid_end = int(n_sent * (train_ratio + valid_ratio))

    sentence_splits = {
        'train': set(sentences_list[:train_end]),
        'valid': set(sentences_list[train_end:valid_end]),
        'test': set(sentences_list[valid_end:])
    }

    print(f"\nSentence split:")
    print(f"  Train sentences: {len(sentence_splits['train'])}")
    print(f"  Valid sentences: {len(sentence_splits['valid'])}")
    print(f"  Test sentences: {len(sentence_splits['test'])}")

    # 3. Create output directories
    for split_name in ['train', 'valid', 'test']:
        split_dir = output_dir / 'split_by_sentence_holdout' / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

    all_samples_collected = []

    # 4. Process all subjects in batches, assigning to splits based on sentence
    print("\nStep 2: Processing all subjects...")
    num_batches = (len(all_subject_dirs) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_subject_dirs))
        batch_dirs = all_subject_dirs[start_idx:end_idx]

        print(f"\n  Batch {batch_idx + 1}/{num_batches} ({len(batch_dirs)} subjects)")

        batch_samples = []
        for subject_dir in tqdm(batch_dirs, desc="  Processing"):
            try:
                samples = process_subject(
                    subject_dir, movies_audio_dir, transcription_dir,
                    tasks, tokenizer, verbose=False
                )
                batch_samples.extend(samples)
            except Exception as e:
                print(f"    Error processing {subject_dir.name}: {e}")
                continue

        if len(batch_samples) == 0:
            print(f"    No samples in batch, skipping...")
            continue

        # Assign samples to splits based on their sentence
        print(f"    Assigning samples to splits...")
        for sample in batch_samples:
            text = sample['target_string']

            # Determine which split this sentence belongs to
            if text in sentence_splits['train']:
                split_name = 'train'
            elif text in sentence_splits['valid']:
                split_name = 'valid'
            elif text in sentence_splits['test']:
                split_name = 'test'
            else:
                print(f"    Warning: Sentence not found in splits: '{text[:50]}...'")
                continue

                # Save to appropriate split
                split_dir = output_dir / 'split_by_sentence_holdout' / split_name
                filename = f"{sample['subject']}-task-{sample['task']}-sent-{sample['sent_id']}.pickle"
                save_sample(sample, split_dir / filename)

        # Collect for mel features
        all_samples_collected.extend(batch_samples)

        del batch_samples
        gc.collect()

    print(f"\n{'=' * 60}")
    print("✓ split_by_sentence_holdout complete")
    print(f"{'=' * 60}")

    return all_samples_collected


# ============================================================================
# MEL FEATURES COLLECTION
# ============================================================================

def save_mel_features(all_mels, all_sents, output_dir):
    """Save pre-generated mel features."""
    log_mel_features = torch.stack(all_mels)

    with open(output_dir / "log_mel_features.pickle", "wb") as f:
        pickle.dump(log_mel_features, f)

    with open(output_dir / "all_sents.pickle", "wb") as f:
        pickle.dump(all_sents, f)

    print(f"✓ Saved log_mel_features.pickle: {log_mel_features.shape}")
    print(f"✓ Saved all_sents.pickle: {len(all_sents)} sentences")


# ============================================================================
# TEST MODE VERIFICATION
# ============================================================================

def verify_test_mode_output(output_dir, test_subject):
    """Verify that test mode produced correct output."""
    print("\n" + "=" * 60)
    print("TEST MODE VERIFICATION")
    print("=" * 60)

    # Check split directories exist
    splits = ['split_by_subject']
    for split_type in splits:
        split_dir = output_dir / split_type
        if not split_dir.exists():
            print(f"✗ {split_type} directory not found!")
            continue

        train_files = list((split_dir / 'train').glob('*.pickle'))
        valid_files = list((split_dir / 'valid').glob('*.pickle'))
        test_files = list((split_dir / 'test').glob('*.pickle'))

        print(f"\n{split_type}:")
        print(f"  Train: {len(train_files)} files")
        print(f"  Valid: {len(valid_files)} files")
        print(f"  Test: {len(test_files)} files")
        print(f"  Total: {len(train_files) + len(valid_files) + len(test_files)} files")

    # Check one pickle file
    print("\nSample pickle file structure:")
    sample_files = list((output_dir / 'split_by_subject' / 'train').glob('*.pickle'))
    if sample_files:
        with open(sample_files[0], 'rb') as f:
            sample = pickle.load(f)

        print(f"  Keys: {list(sample.keys())}")
        print(f"  Subject: {sample['subject']}")
        print(f"  Task: {sample['task']}")
        print(f"  Sentence ID: {sample['sent_id']}")
        print(f"  Text: '{sample['target_string']}'")
        print(f"  rawEEG shape: {sample['rawEEG'].shape}")
        print(f"  Has mel_spectrogram: {'mel_spectrogram' in sample}")
        if 'mel_spectrogram' in sample:
            print(f"  mel_spectrogram shape: {sample['mel_spectrogram'].shape}")

    # Check mel features files
    print("\nMel features files:")
    if (output_dir / "log_mel_features.pickle").exists():
        with open(output_dir / "log_mel_features.pickle", "rb") as f:
            mels = pickle.load(f)
        print(f"  ✓ log_mel_features.pickle: {mels.shape}")
    else:
        print(f"  ✗ log_mel_features.pickle not found")

    if (output_dir / "all_sents.pickle").exists():
        with open(output_dir / "all_sents.pickle", "rb") as f:
            sents = pickle.load(f)
        print(f"  ✓ all_sents.pickle: {len(sents)} sentences")
        print(f"  First 3 sentences:")
        for i in range(min(3, len(sents))):
            print(f"    [{i}]: {sents[i]}")
    else:
        print(f"  ✗ all_sents.pickle not found")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # ============ CONFIGURATION ============
    base_dir = Path("/vol/fastdata/ehb24169/BrainEchoReplication/HealthyBrainNetwork")

    preprocessed_eeg_dir = base_dir / "Release1Processed"
    movies_audio_dir = base_dir / "wav16"
    transcription_dir = base_dir / "transcription"
    output_dir = Path("/vol/fastdata/ehb24169/Repositories/Projects/BrainEchoReplica/BrainRecallExtended/datasets/HBN")

    tasks = ['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'ThePresent']

    # TEST MODE
    TEST_MODE = False
    TEST_SUBJECT = 'sub-NDARAC904DMU'

    # BATCH PROCESSING
    BATCH_SIZE = 10  # Subjects per batch

    # WHICH SPLITS TO CREATE
    SPLIT_CONFIGS = {
        'split_by_instance': False,  # Disabled - has batch logic issues
        'split_by_sentence_within_subject': False,
        'split_by_subject': True,
        'split_by_sentence_holdout': False
    }

    # ============ SETUP ============
    print("=" * 60)
    print("HBN DATA PROCESSING PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Test mode: {TEST_MODE}")
    if TEST_MODE:
        print(f"  Test subject: {TEST_SUBJECT}")
    print(f"  Batch size: {BATCH_SIZE} subjects")
    print(f"  Tasks: {tasks}")
    print(f"  ThePresent handling:")
    print(f"    Use fixed segments: {THE_PRESENT_CONFIG['use_fixed_segments']}")
    if THE_PRESENT_CONFIG['use_fixed_segments']:
        print(f"    Segment duration: {THE_PRESENT_CONFIG['segment_duration']}s")
        print(f"    Dummy text: {THE_PRESENT_CONFIG['dummy_text']}")
    print(f"\nSplits to create:")
    for split_name, enabled in SPLIT_CONFIGS.items():
        print(f"    {split_name}: {'✓' if enabled else '✗'}")

    print("\nLoading Whisper processor and tokenizer...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base.en")

    # Generate mel features ONCE from audio
    print("\nGenerating mel-spectrograms from audio...")
    all_mels, all_sents, segment_mapping = generate_all_mel_features(
        movies_audio_dir, transcription_dir, tasks, processor
    )
    print(f"Generated {len(all_mels)} mel-spectrograms")

    # Save immediately
    save_mel_features(all_mels, all_sents, output_dir)

    # Get subjects to process
    if TEST_MODE:
        print(f"\n{'=' * 60}")
        print(f"TEST MODE: Processing only {TEST_SUBJECT}")
        print(f"{'=' * 60}")
        subjects_to_process = [TEST_SUBJECT]
        output_dir = output_dir / "TEST"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        subjects_file = preprocessed_eeg_dir.parent / "Release1" / "subjects_with_all_5_tasks.txt"
        if subjects_file.exists():
            with open(subjects_file, 'r') as f:
                subjects_to_process = [line.split()[0] for line in f
                                       if line.strip() and not line.startswith('#')]
        else:
            subjects_to_process = [d.name for d in preprocessed_eeg_dir.iterdir()
                                   if d.is_dir() and d.name.startswith('sub-')]

    all_subject_dirs = [preprocessed_eeg_dir / subj for subj in subjects_to_process]
    all_subject_dirs = [d for d in all_subject_dirs if d.exists()]
    all_subject_dirs = sorted(all_subject_dirs)

    print(f"\nFound {len(all_subject_dirs)} subjects to process")

    # ============ PROCESSING ============

    # Process split_by_instance and split_by_sentence_within_subject (batch-friendly)
    if SPLIT_CONFIGS.get('split_by_instance', False) or SPLIT_CONFIGS.get('split_by_sentence_within_subject', False):
        print(f"\n{'#' * 60}")
        print("# PROCESSING BATCH-FRIENDLY SPLITS")
        print(f"{'#' * 60}")

        splits_to_process = []
        if SPLIT_CONFIGS.get('split_by_instance', False):
            splits_to_process.append('split_by_instance')
        if SPLIT_CONFIGS.get('split_by_sentence_within_subject', False):
            splits_to_process.append('split_by_sentence_within_subject')

        num_batches = (len(all_subject_dirs) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(all_subject_dirs))
            batch_dirs = all_subject_dirs[start_idx:end_idx]

            print(f"\nBatch {batch_idx + 1}/{num_batches} ({len(batch_dirs)} subjects)")

            batch_samples = process_batch_simple_splits(
                batch_dirs, movies_audio_dir, transcription_dir,
                output_dir, tasks, tokenizer,
                split_types=splits_to_process
            )

    # Process split_by_subject (needs subject-level organization)
    if SPLIT_CONFIGS.get('split_by_subject', False):
        print(f"\n{'#' * 60}")
        print("# PROCESSING SPLIT_BY_SUBJECT")
        print(f"{'#' * 60}")

        samples = split_by_subject_efficient(
            all_subject_dirs, movies_audio_dir, transcription_dir,
            output_dir, tasks, tokenizer,
            batch_size=BATCH_SIZE
        )

    # Process split_by_sentence_holdout (needs sentence-level organization)
    if SPLIT_CONFIGS.get('split_by_sentence_holdout', False):
        print(f"\n{'#' * 60}")
        print("# PROCESSING SPLIT_BY_SENTENCE_HOLDOUT")
        print(f"{'#' * 60}")

        samples = split_by_sentence_holdout_efficient(
            all_subject_dirs, movies_audio_dir, transcription_dir,
            output_dir, tasks, tokenizer,
            batch_size=BATCH_SIZE
        )

    # ============ FINAL SUMMARY ============
    print("\n" + "=" * 60)
    print("✓ ALL PROCESSING COMPLETE")
    print("=" * 60)

    if TEST_MODE:
        verify_test_mode_output(output_dir, TEST_SUBJECT)
    else:
        # Count samples in each split
        print("\ndataset statistics:")
        for split_type in ['split_by_instance', 'split_by_sentence_within_subject',
                           'split_by_subject', 'split_by_sentence_holdout']:
            if not SPLIT_CONFIGS.get(split_type, False):
                continue

            split_dir = output_dir / split_type
            if split_dir.exists():
                train_count = len(list((split_dir / 'train').glob('*.pickle')))
                valid_count = len(list((split_dir / 'valid').glob('*.pickle')))
                test_count = len(list((split_dir / 'test').glob('*.pickle')))
                total = train_count + valid_count + test_count

                print(f"\n{split_type}:")
                print(f"  Train: {train_count}")
                print(f"  Valid: {valid_count}")
                print(f"  Test: {test_count}")
                print(f"  Total: {total}")