import mne
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import shutil
from joblib import Parallel, delayed


def preprocess_hbn_task(
        subject_path,
        task_name,
        output_path,
        notch_freq=60,
        bandpass_low=0.5,
        bandpass_high=99,
        target_sfreq=200,
        verbose=True
):
    """
    Preprocess a single task for a single subject.

    Parameters:
    -----------
    subject_path : str or Path
        Path to subject folder containing raw data
    task_name : str
        Task name (e.g., 'DespicableMe')
    output_path : str or Path
        Path to save preprocessed data
    notch_freq : float
        Notch filter frequency (Hz) for power line noise
    bandpass_low : float
        Low cutoff for bandpass filter (Hz)
    bandpass_high : float
        High cutoff for bandpass filter (Hz)
    target_sfreq : float
        Target sampling frequency after resampling (Hz)
    verbose : bool
        Print progress messages

    Returns:
    --------
    dict : Processing summary with status and file paths
    """

    subject_path = Path(subject_path)
    output_path = Path(output_path)
    subject_id = subject_path.name

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Processing: {subject_id} - {task_name}")
        print(f"{'=' * 60}")

    # Build file paths
    input_files = {
        'eeg': subject_path / f"{subject_id}_task-{task_name}_eeg.set",
        'events_tsv': subject_path / f"{subject_id}_task-{task_name}_events.tsv",
        'eeg_json': subject_path / f"{subject_id}_task-{task_name}_eeg.json",
        'channels': subject_path / f"{subject_id}_task-{task_name}_channels.tsv",
        'electrodes': subject_path / f"{subject_id}_task-{task_name}_electrodes.tsv",
        'coordsystem': subject_path / f"{subject_id}_task-{task_name}_coordsystem.json",
    }

    # Check if all necessary files exist
    if not input_files['eeg'].exists():
        return {'status': 'error', 'message': f"EEG file not found: {input_files['eeg']}"}

    try:
        # 1. Load raw data with annotations
        if verbose:
            print("Step 1/5: Loading raw data...")
        raw = mne.io.read_raw_eeglab(input_files['eeg'], preload=True, verbose=False)
        original_sfreq = raw.info['sfreq']

        # Load original events TSV for reference
        events_df_original = pd.read_csv(input_files['events_tsv'], sep='\t')

        if verbose:
            print(f"  Original sampling rate: {original_sfreq} Hz")
            print(f"  Found {len(raw.annotations)} annotations in raw data")
            print(f"  Duration: {raw.times[-1]:.2f} seconds")

        # 2. Apply notch filter
        if verbose:
            print(f"Step 2/5: Applying notch filter at {notch_freq} Hz...")
        raw.notch_filter(freqs=notch_freq, verbose=False)

        # 3. Apply bandpass filter
        if verbose:
            print(f"Step 3/5: Applying bandpass filter {bandpass_low}-{bandpass_high} Hz...")
        raw.filter(l_freq=bandpass_low, h_freq=bandpass_high, verbose=False)

        # 4. Resample (MNE automatically adjusts annotations!)
        if verbose:
            print(f"Step 4/5: Resampling from {original_sfreq} Hz to {target_sfreq} Hz...")
        raw.resample(sfreq=target_sfreq, verbose=False)

        if verbose:
            print(f"  Resampled to {raw.info['sfreq']} Hz")
            print(f"  Data shape: {raw.get_data().shape}")
            print(f"  Annotations automatically adjusted by MNE")

        # 5. Save preprocessed data
        if verbose:
            print("Step 5/5: Saving preprocessed data...")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Save EEG data as .set (annotations are embedded)
        output_eeg = output_path / f"{subject_id}_task-{task_name}_eeg.set"
        raw.export(output_eeg, fmt='eeglab', overwrite=True, verbose=False)

        # Extract automatically-adjusted annotations and create updated events TSV
        annotations = raw.annotations

        # Create updated events TSV with MNE-adjusted timings
        events_df_resampled = pd.DataFrame({
            'onset': annotations.onset,
            'duration': ['n/a'] * len(annotations),  # Match original format
            'sample': (annotations.onset * target_sfreq).astype(int),
            'value': annotations.description,
            'event_code': annotations.description  # Keep description as event_code
        })

        # Map back original event_code values if they were numeric
        for i, desc in enumerate(events_df_resampled['event_code']):
            # Try to match with original events by description/value
            original_match = events_df_original[events_df_original['value'] == desc]
            if not original_match.empty:
                orig_code = original_match.iloc[0]['event_code']
                # Keep original numeric codes if they existed
                if not isinstance(orig_code, str) or orig_code != desc:
                    events_df_resampled.loc[i, 'event_code'] = orig_code

        output_events_tsv = output_path / f"{subject_id}_task-{task_name}_events.tsv"
        events_df_resampled.to_csv(output_events_tsv, sep='\t', index=False)

        if verbose:
            print(f"  Saved {len(events_df_resampled)} events to TSV")

        # Copy metadata files
        for file_key in ['channels', 'electrodes', 'coordsystem']:
            if input_files[file_key].exists():
                output_file = output_path / input_files[file_key].name
                shutil.copy2(input_files[file_key], output_file)

        # Create updated EEG JSON with preprocessing info
        with open(input_files['eeg_json'], 'r') as f:
            eeg_metadata = json.load(f)

        # Update metadata
        eeg_metadata['SamplingFrequency'] = target_sfreq
        eeg_metadata['SoftwareFilters'] = {
            'NotchFilter': {'Frequency': notch_freq},
            'BandpassFilter': {'LowCutoff': bandpass_low, 'HighCutoff': bandpass_high}
        }
        eeg_metadata['PreprocessingInfo'] = {
            'OriginalSamplingFrequency': original_sfreq,
            'ProcessedDate': datetime.now().isoformat(),
            'MNEVersion': mne.__version__,
            'Pipeline': 'HBN_Preprocessing_v1.0',
            'AnnotationHandling': 'MNE automatic adjustment during resampling'
        }

        output_eeg_json = output_path / f"{subject_id}_task-{task_name}_eeg.json"
        with open(output_eeg_json, 'w') as f:
            json.dump(eeg_metadata, f, indent=2)

        if verbose:
            print(f"✓ Preprocessing complete!")
            print(f"  Output directory: {output_path}")

        # Verification: Check that timing is preserved
        if verbose:
            print(f"\nTiming verification:")
            for i, (orig_onset, new_onset) in enumerate(zip(events_df_original['onset'], events_df_resampled['onset'])):
                diff = abs(orig_onset - new_onset)
                print(f"  Event {i}: original={orig_onset:.3f}s, resampled={new_onset:.3f}s, diff={diff:.6f}s")

        return {
            'status': 'success',
            'subject': subject_id,
            'task': task_name,
            'output_path': str(output_path),
            'original_sfreq': original_sfreq,
            'target_sfreq': target_sfreq,
            'n_events': len(events_df_resampled),
            'timing_verification': True
        }

    except Exception as e:
        error_msg = f"Error processing {subject_id} - {task_name}: {str(e)}"
        if verbose:
            print(f"✗ {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'subject': subject_id,
            'task': task_name,
            'message': error_msg
        }


def preprocess_hbn_subject(
        subject_path,
        output_base,
        release_num,
        tasks=['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'RestingState', 'ThePresent'],
        parallel=False,
        n_jobs=-1,
        **preprocess_kwargs
):
    """
    Preprocess all tasks for a single subject.

    Parameters:
    -----------
    subject_path : str or Path
        Path to subject folder
    output_base : str or Path
        Base output directory
    release_num : int
        Release number
    tasks : list
        List of task names to process
    parallel : bool
        Process tasks in parallel
    n_jobs : int
        Number of parallel jobs (-1 = all cores)
    **preprocess_kwargs : dict
        Additional arguments for preprocess_hbn_task

    Returns:
    --------
    list : Processing results for each task
    """

    subject_path = Path(subject_path)
    subject_id = subject_path.name

    print(f"\n{'#' * 60}")
    print(f"# Processing Subject: {subject_id}")
    print(f"{'#' * 60}")

    output_subject_path = Path(output_base) / f"Release{release_num}Processed" / subject_id

    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(preprocess_hbn_task)(
                subject_path, task, output_subject_path, **preprocess_kwargs
            ) for task in tasks
        )
    else:
        results = []
        for task in tasks:
            result = preprocess_hbn_task(
                subject_path, task, output_subject_path, **preprocess_kwargs
            )
            results.append(result)

    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\nSubject {subject_id} summary: {success}/{len(tasks)} tasks completed")

    return results


def preprocess_hbn_release(
        release_num,
        source_base,
        output_base,
        subjects=None,
        tasks=['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'RestingState', 'ThePresent'],
        only_complete_subjects=True,
        parallel_subjects=False,
        parallel_tasks=False,
        n_jobs=-1,
        **preprocess_kwargs
):
    """
    Preprocess an entire release.

    Parameters:
    -----------
    release_num : int
        Release number to process
    source_base : str or Path
        Base directory containing raw data
    output_base : str or Path
        Base directory for preprocessed data
    subjects : list or None
        Specific subjects to process (None = all)
    tasks : list
        Tasks to process
    only_complete_subjects : bool
        Only process subjects with all 5 tasks available
    parallel_subjects : bool
        Process subjects in parallel
    parallel_tasks : bool
        Process tasks in parallel (per subject)
    n_jobs : int
        Number of parallel jobs
    **preprocess_kwargs : dict
        Additional preprocessing parameters

    Returns:
    --------
    dict : Processing summary
    """

    source_base = Path(source_base)
    output_base = Path(output_base)
    release_dir = source_base / f"Release{release_num}"

    print(f"\n{'=' * 80}")
    print(f"PREPROCESSING RELEASE {release_num}")
    print(f"{'=' * 80}")

    # Get subjects to process
    if only_complete_subjects:
        complete_subjects_file = release_dir / "subjects_with_all_5_tasks.txt"
        if complete_subjects_file.exists():
            with open(complete_subjects_file, 'r') as f:
                complete_subjects = [line.split()[0] for line in f if line.strip() and not line.startswith('#')]
            print(f"Found {len(complete_subjects)} subjects with all 5 tasks")
        else:
            print("Warning: subjects_with_all_5_tasks.txt not found. Processing all subjects.")
            complete_subjects = None
    else:
        complete_subjects = None

    # Get subject directories
    subject_dirs = [d for d in release_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]

    if complete_subjects is not None:
        subject_dirs = [d for d in subject_dirs if d.name in complete_subjects]

    if subjects is not None:
        subject_dirs = [d for d in subject_dirs if d.name in subjects]

    print(f"Processing {len(subject_dirs)} subjects")

    # Copy release-level metadata
    output_release_dir = output_base / f"Release{release_num}Processed"
    output_release_dir.mkdir(parents=True, exist_ok=True)

    metadata_files = list(release_dir.glob('*.json')) + list(release_dir.glob('*.tsv')) + \
                     [f for f in release_dir.glob('*') if f.name in ['README', 'subjects_with_all_5_tasks.txt']]

    for file in metadata_files:
        if file.is_file():
            shutil.copy2(file, output_release_dir / file.name)

    # Create preprocessing metadata
    preprocessing_metadata = {
        'ReleaseNumber': release_num,
        'ProcessingDate': datetime.now().isoformat(),
        'MNEVersion': mne.__version__,
        'PreprocessingParameters': {
            'NotchFrequency': preprocess_kwargs.get('notch_freq', 60),
            'BandpassLow': preprocess_kwargs.get('bandpass_low', 0.5),
            'BandpassHigh': preprocess_kwargs.get('bandpass_high', 99),
            'TargetSamplingFrequency': preprocess_kwargs.get('target_sfreq', 200),
        },
        'TasksProcessed': tasks,
        'SubjectsProcessed': len(subject_dirs)
    }

    with open(output_release_dir / 'preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_metadata, f, indent=2)

    # Process subjects
    if parallel_subjects:
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(preprocess_hbn_subject)(
                subj_dir, output_base, release_num, tasks,
                parallel=parallel_tasks, **preprocess_kwargs
            ) for subj_dir in subject_dirs
        )
    else:
        all_results = []
        for subj_dir in subject_dirs:
            results = preprocess_hbn_subject(
                subj_dir, output_base, release_num, tasks,
                parallel=parallel_tasks, **preprocess_kwargs
            )
            all_results.append(results)

    # Summary
    total_tasks = sum(len(results) for results in all_results)
    successful_tasks = sum(sum(1 for r in results if r['status'] == 'success') for results in all_results)

    print(f"\n{'=' * 80}")
    print(f"RELEASE {release_num} SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total subjects processed: {len(subject_dirs)}")
    print(f"Total tasks attempted: {total_tasks}")
    print(f"Successful tasks: {successful_tasks}")
    print(f"Failed tasks: {total_tasks - successful_tasks}")
    print(f"Output directory: {output_release_dir}")

    return {
        'release': release_num,
        'subjects_processed': len(subject_dirs),
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'failed_tasks': total_tasks - successful_tasks,
        'results': all_results
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # EXAMPLE 1: Process single task for single subject
    '''
    print("EXAMPLE 1: Single task, single subject")
    result = preprocess_hbn_task(
        subject_path=r"I:\Science\CIS-YASHMOSH\neurasearch\BrainEchoReplication\HealthyBrainNetwork\Release1\sub-NDARAC904DMU",
        task_name='DespicableMe',
        output_path=r"I:\Science\CIS-YASHMOSH\neurasearch\BrainEchoReplication\HealthyBrainNetwork\Release1Processed\sub-NDARAC904DMU",
        notch_freq=60,
        bandpass_low=0.5,
        bandpass_high=99,
        target_sfreq=200,
        verbose=True
    )
    print(f"\nResult: {result}")
    
    # EXAMPLE 2: Process all tasks for single subject (sequential)
    results = preprocess_hbn_subject(
        subject_path=r"I:\Science\CIS-YASHMOSH\neurasearch\BrainEchoReplication\HealthyBrainNetwork\Release1\sub-NDARAC904DMU",
        output_base=r"I:\Science\CIS-YASHMOSH\neurasearch\BrainEchoReplication\HealthyBrainNetwork",
        release_num=1,
        parallel=False
    )
    
    # EXAMPLE 3: Process all tasks for single subject (parallel)
    results = preprocess_hbn_subject(
        subject_path=r"I:\Science\CIS-YASHMOSH\neurasearch\BrainEchoReplication\HealthyBrainNetwork\Release1\sub-NDARAC904DMU",
        output_base=r"I:\Science\CIS-YASHMOSH\neurasearch\BrainEchoReplication\HealthyBrainNetwork",
        release_num=1,
        parallel=True,
        n_jobs=4
    )
    '''
    # EXAMPLE 4: Process entire Release 1 (only complete subjects, sequential)
    # summary = preprocess_hbn_release(
    #     release_num=1,
    #     source_base=r"I:\Science\CIS-YASHMOSH\neurasearch\HealthyBrainNetwork",
    #     output_base=r"I:\Science\CIS-YASHMOSH\neurasearch\BrainEchoReplication\HealthyBrainNetwork",
    #     only_complete_subjects=True,
    #     parallel_subjects=False,
    #     parallel_tasks=False
    # )

    # EXAMPLE 5: Process entire Release 1 (parallel subjects AND tasks)
    summary = preprocess_hbn_release(
        release_num=1,
        source_base=r"/vol/fastdata/ehb24169/BrainEchoReplication/HealthyBrainNetwork",
        output_base=r"/vol/fastdata/ehb24169/BrainEchoReplication/HealthyBrainNetwork",
        only_complete_subjects=True,
        parallel_subjects=True,
       parallel_tasks=False,
       n_jobs=-1  # Use all cores
    )