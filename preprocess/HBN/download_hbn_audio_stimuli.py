import subprocess
import os
from pathlib import Path
import shutil


def download_stimulus_audio(urls, output_dir="stimuli_audio"):
    """
    Download audio from video URLs for HBN stimulus files.

    Parameters:
    -----------
    urls : dict
        Dictionary mapping task names to URLs
    output_dir : str
        Directory to save audio files
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading audio files to: {output_path.absolute()}\n")

    results = {}

    for task_name, url in urls.items():
        print(f"{'=' * 60}")
        print(f"Downloading: {task_name}")
        print(f"URL: {url}")
        print(f"{'=' * 60}")

        output_file = output_path / f"{task_name}.wav"

        try:
            # Download audio using yt-dlp
            cmd = [
                'yt-dlp',
                '--extractor-args', 'youtube:player_client=android',  # Use Android client (bypasses JS)
                '-x',
                '--audio-format', 'wav',
                '--audio-quality', '0',
                '-o', str(output_file),
                url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Check if file was created
            if output_file.exists():
                file_size = output_file.stat().st_size / (1024 ** 2)  # MB
                print(f"✓ Downloaded successfully: {output_file.name} ({file_size:.2f} MB)\n")
                results[task_name] = {'status': 'success', 'file': str(output_file)}
            else:
                print(f"✗ File not created: {task_name}\n")
                results[task_name] = {'status': 'error', 'message': 'File not created'}

        except subprocess.CalledProcessError as e:
            print(f"✗ Error downloading {task_name}:")
            print(f"  {e.stderr}\n")
            results[task_name] = {'status': 'error', 'message': str(e)}
        except Exception as e:
            print(f"✗ Unexpected error: {e}\n")
            results[task_name] = {'status': 'error', 'message': str(e)}

    # Summary
    print(f"\n{'=' * 60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 60}")
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"Successful: {successful}/{len(urls)}")
    print(f"Failed: {len(urls) - successful}/{len(urls)}")

    return results


def verify_audio_durations(audio_dir="stimuli_audio", show_mismatches_only=False):
    """
    Verify audio durations match expected EEG event timings.
    """
    import librosa

    print(f"\n{'=' * 60}")
    print("VERIFYING AUDIO DURATIONS")
    print(f"{'=' * 60}")

    audio_path = Path(audio_dir)

    # Expected durations from HBN
    expected_durations = {
        'DespicableMe': 170.5,
        'DiaryOfAWimpyKid': 117.4,
        'FunwithFractals': 163,
        'ThePresent': 203.1
    }

    mismatches = []

    for task_name, expected_dur in expected_durations.items():
        audio_file = audio_path / f"{task_name}.wav"

        if audio_file.exists():
            try:
                audio, sr = librosa.load(audio_file, sr=None)
                duration = len(audio) / sr
                diff = abs(duration - expected_dur)
                matches = diff < 1.0

                if not show_mismatches_only or not matches:
                    print(f"\n{task_name}:")
                    print(f"  Duration: {duration:.2f} seconds")
                    print(f"  Sample rate: {sr} Hz")

                    if matches:
                        print(f"  ✓ Matches expected duration ({expected_dur}s)")
                    else:
                        print(f"  ⚠ Duration mismatch! Expected ~{expected_dur}s, got {duration:.2f}s")
                        print(f"    Difference: {diff:.2f}s")
                        mismatches.append(task_name)

            except Exception as e:
                print(f"\n{task_name}: Error loading - {e}")
                mismatches.append(task_name)
        else:
            print(f"\n{task_name}: File not found")
            mismatches.append(task_name)

    return mismatches


def trim_stimulus_audio(input_dir="stimuli_audio", output_dir="stimuli_audio_trimmed"):
    """
    Trim audio files to match HBN experimental timings.
    """

    try:
        from pydub import AudioSegment
    except ImportError:
        print("Installing pydub...")
        subprocess.run(['pip', 'install', 'pydub'], check=True)
        from pydub import AudioSegment

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("TRIMMING AUDIO FILES")
    print(f"{'=' * 60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}\n")

    # Files that don't need trimming
    no_trim = ['DespicableMe', 'DiaryOfAWimpyKid']

    for task in no_trim:
        src = input_path / f"{task}.wav"
        dst = output_path / f"{task}.wav"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✓ {task}: Copied (no trimming needed)")
        else:
            print(f"✗ {task}: Source file not found")

    # Trim Fun with Fractals: 0:08 to 2:51
    print(f"\nTrimming FunwithFractals (0:08 to 2:51)...")
    fractals_src = input_path / "FunwithFractals.wav"
    if fractals_src.exists():
        fractals_audio = AudioSegment.from_wav(fractals_src)

        start_ms = 8 * 1000  # 8 seconds
        end_ms = (2 * 60 + 51) * 1000  # 2 minutes 51 seconds = 171 seconds

        fractals_trimmed = fractals_audio[start_ms:end_ms]
        fractals_trimmed.export(output_path / "FunwithFractals.wav", format="wav")

        duration = len(fractals_trimmed) / 1000.0
        print(f"✓ FunwithFractals: Trimmed to {duration:.2f}s")
        print(f"  Expected: ~163s, Got: {duration:.2f}s")
    else:
        print(f"✗ FunwithFractals: Source file not found")

    # Trim The Present: up to credits (~203s based on EEG timing)
    print(f"\nTrimming ThePresent (removing credits)...")
    present_src = input_path / "ThePresent.wav"
    if present_src.exists():
        present_audio = AudioSegment.from_wav(present_src)

        # Trim to match EEG timing
        credits_start_ms = 203 * 1000  # ~203 seconds based on EEG events and where credits start

        present_trimmed = present_audio[:credits_start_ms]
        present_trimmed.export(output_path / "ThePresent.wav", format="wav")

        duration = len(present_trimmed) / 1000.0
        print(f"✓ ThePresent: Trimmed to {duration:.2f}s")
        print(f"  Expected: ~203s, Got: {duration:.2f}s")
    else:
        print(f"✗ ThePresent: Source file not found")

    print(f"\n✓ All trimmed files saved to: {output_path}")


def process_hbn_stimuli(urls, base_dir, download=True, trim=True, verify_final=True):
    """
    Complete pipeline: download, trim, and verify HBN stimulus audio.

    Parameters:
    -----------
    urls : dict
        Dictionary mapping task names to URLs
    base_dir : str or Path
        Base directory for all audio files
    download : bool
        Download audio files
    trim : bool
        Trim audio files to match experimental timings
    verify_final : bool
        Verify final trimmed audio
    """

    base_path = Path(base_dir)
    raw_dir = base_path / "stimuli_audio_raw"
    trimmed_dir = base_path / "stimuli_audio_trimmed"

    # Step 1: Download
    if download:
        print(f"\n{'#' * 60}")
        print("# STEP 1: DOWNLOADING AUDIO")
        print(f"{'#' * 60}")
        download_results = download_stimulus_audio(urls, output_dir=raw_dir)

        # Verify downloads
        print(f"\n{'#' * 60}")
        print("# VERIFYING DOWNLOADS")
        print(f"{'#' * 60}")
        mismatches = verify_audio_durations(raw_dir, show_mismatches_only=False)
    else:
        print("Skipping download step...")
        mismatches = verify_audio_durations(raw_dir, show_mismatches_only=True)

    # Step 2: Trim
    if trim and mismatches:
        print(f"\n{'#' * 60}")
        print("# STEP 2: TRIMMING AUDIO")
        print(f"{'#' * 60}")
        trim_stimulus_audio(input_dir=raw_dir, output_dir=trimmed_dir)
    elif trim and not mismatches:
        print(f"\nNo trimming needed - all files match expected durations!")
        # Still copy to trimmed directory
        shutil.copytree(raw_dir, trimmed_dir, dirs_exist_ok=True)
    else:
        print("Skipping trim step...")

    # Step 3: Final verification
    if verify_final and trim:
        print(f"\n{'#' * 60}")
        print("# STEP 3: FINAL VERIFICATION")
        print(f"{'#' * 60}")
        final_mismatches = verify_audio_durations(trimmed_dir, show_mismatches_only=False)

        if not final_mismatches:
            print(f"\n{'=' * 60}")
            print("✓✓✓ SUCCESS! All audio files are ready for processing ✓✓✓")
            print(f"{'=' * 60}")
            print(f"Final audio files located at: {trimmed_dir}")
        else:
            print(f"\n{'=' * 60}")
            print("⚠ Some files still have duration mismatches")
            print(f"{'=' * 60}")
            print(f"Files with issues: {', '.join(final_mismatches)}")


if __name__ == "__main__":

    # CHECK IF yt-dlp IS INSTALLED
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: yt-dlp not found!")
        print("Install with: pip install yt-dlp")
        exit(1)

    # DEFINE YOUR URLs
    stimulus_urls = {
        'DespicableMe': 'https://www.youtube.com/watch?v=HNXxJIhVALI',
        'DiaryOfAWimpyKid': 'https://www.youtube.com/watch?v=7ZVEIgPeDCE',
        'FunwithFractals': 'https://www.youtube.com/watch?v=XwWyTts06tU',
        'ThePresent': 'https://vimeo.com/152985022'
    }

    # Base directory for all stimulus audio
    base_dir = r"I:\Science\CIS-YASHMOSH\neurasearch\BrainEchoReplication\HealthyBrainNetwork"

    # Run complete pipeline
    process_hbn_stimuli(
        urls=stimulus_urls,
        base_dir=base_dir,
        download=True,  # Set to False to skip downloading
        trim=True,  # Set to False to skip trimming
        verify_final=True
    )