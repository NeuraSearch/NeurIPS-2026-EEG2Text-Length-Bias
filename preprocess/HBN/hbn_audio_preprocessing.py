import soundfile as sf
import os
import json
import tqdm
import librosa
import numpy as np
import whisperx
import argparse
from pathlib import Path


def makedirs(output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    return output_dir


def process_hbn_movie_audio(
        audio_path,
        movie_name,
        output_base_dir,
        device='cuda',
        batch_size=16,
        compute_type='float16',
        language='en',
        max_segment_length=12.0  # Match Brennan: 12 seconds max
):
    """
    Process a single HBN movie audio file using WhisperX segmentation.

    Parameters:
    -----------
    audio_path : str
        Path to trimmed movie audio file
    movie_name : str
        Name of movie (e.g., 'DespicableMe')
    output_base_dir : str
        Base directory for outputs
    device : str
        'cuda' or 'cpu'
    batch_size : int
        Batch size for WhisperX
    compute_type : str
        'float16' or 'float32'
    language : str
        Language code (default 'en')
    max_segment_length : float
        Maximum segment length in seconds
    """

    print(f"\n{'=' * 60}")
    print(f"Processing: {movie_name}")
    print(f"{'=' * 60}")

    # Setup output directories
    wav16_dir = Path(output_base_dir) / 'wav16'
    transcription_dir = Path(output_base_dir) / 'transcription'
    wav16_dir.mkdir(parents=True, exist_ok=True)
    transcription_dir.mkdir(parents=True, exist_ok=True)

    # Load and resample audio to 16kHz
    print("Step 1: Loading and resampling audio to 16kHz...")
    wav, orig_sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav[:, 0]  # Take first channel if stereo

    target_sr = 16000
    if orig_sr != target_sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)

    # Save 16kHz audio
    wav16_path = wav16_dir / f"{movie_name}_16kHz.wav"
    sf.write(wav16_path, wav, samplerate=target_sr)
    print(f"  Saved 16kHz audio: {wav16_path}")
    print(f"  Duration: {len(wav) / target_sr:.2f} seconds")

    # Load WhisperX model
    print("\nStep 2: Loading WhisperX model...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    # Transcribe with WhisperX
    print("\nStep 3: Transcribing with WhisperX...")
    audio = whisperx.load_audio(str(wav16_path))
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        chunk_size=30,  # WhisperX parameter
        language=language
    )

    # Align segments
    print("\nStep 4: Aligning segments...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result['language'],
        device=device
    )
    segments_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    # Update result with aligned segments
    for k in ['segments', 'word_segments']:
        if k in segments_result:
            result[k] = segments_result[k]

    # Filter/adjust segments to max length (like Brennan's 12 seconds)
    print(f"\nStep 5: Adjusting segments to max {max_segment_length}s...")
    adjusted_segments = []

    for seg in result['segments']:
        seg_duration = seg['end'] - seg['start']

        if seg_duration <= max_segment_length:
            # Segment is already short enough
            adjusted_segments.append(seg)
        else:
            # Split long segment into chunks
            print(f"  Warning: Segment too long ({seg_duration:.2f}s), splitting...")
            # Simple split by time (could be improved with word boundaries)
            num_chunks = int(np.ceil(seg_duration / max_segment_length))
            chunk_duration = seg_duration / num_chunks

            for i in range(num_chunks):
                chunk_start = seg['start'] + i * chunk_duration
                chunk_end = min(seg['start'] + (i + 1) * chunk_duration, seg['end'])

                adjusted_segments.append({
                    'start': chunk_start,
                    'end': chunk_end,
                    'text': seg['text'],  # Keep full text (not ideal but simple)
                })

    result['segments'] = adjusted_segments

    # Statistics
    print(f"\n  Total segments: {len(result['segments'])}")
    segment_durations = [s['end'] - s['start'] for s in result['segments']]
    print(f"  Segment duration range: {min(segment_durations):.2f}s - {max(segment_durations):.2f}s")
    print(f"  Mean segment duration: {np.mean(segment_durations):.2f}s")

    # Count segments with/without text
    segments_with_text = sum(1 for s in result['segments'] if s.get('text', '').strip())
    print(f"  Segments with text: {segments_with_text}/{len(result['segments'])}")

    # Save transcription
    print("\nStep 6: Saving transcription...")
    transcribe_path = transcription_dir / f"{movie_name}_16kHz.json"
    with open(transcribe_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"  Saved transcription: {transcribe_path}")

    # Clean up escaped characters (from original Brennan script)
    with open(transcribe_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        line = line.replace(r'--\"', ' ')
        line = line.replace(r"\"'", '')
        line = line.replace(r'\"', '')
        cleaned_lines.append(line)

    with open(transcribe_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    print(f"\n✓ Processing complete for {movie_name}")

    return {
        'movie': movie_name,
        'wav16_path': str(wav16_path),
        'transcription_path': str(transcribe_path),
        'n_segments': len(result['segments']),
        'segments_with_text': segments_with_text,
        'duration': len(wav) / target_sr
    }


def process_all_hbn_movies(
        movies_dir,
        output_base_dir,
        movies=['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'ThePresent'],
        **kwargs
):
    """
    Process all HBN movie audio files.
    """

    movies_dir = Path(movies_dir)

    print(f"\n{'#' * 60}")
    print("# PROCESSING HBN MOVIE AUDIO WITH WHISPERX")
    print(f"{'#' * 60}")
    print(f"Input directory: {movies_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Movies to process: {movies}")

    results = {}

    for movie in movies:
        audio_path = movies_dir / f"{movie}.wav"

        if not audio_path.exists():
            print(f"\n⚠ Warning: {movie}.wav not found, skipping...")
            continue

        try:
            result = process_hbn_movie_audio(
                audio_path=audio_path,
                movie_name=movie,
                output_base_dir=output_base_dir,
                **kwargs
            )
            results[movie] = result
        except Exception as e:
            print(f"\n✗ Error processing {movie}: {e}")
            import traceback
            traceback.print_exc()
            results[movie] = {'status': 'error', 'message': str(e)}

    # Summary
    print(f"\n{'#' * 60}")
    print("# PROCESSING SUMMARY")
    print(f"{'#' * 60}")

    for movie, result in results.items():
        if 'n_segments' in result:
            print(f"\n{movie}:")
            print(f"  Segments: {result['n_segments']}")
            print(f"  With text: {result['segments_with_text']}")
            print(f"  Duration: {result['duration']:.2f}s")
        else:
            print(f"\n{movie}: {result.get('status', 'unknown')}")

    total_segments = sum(r.get('n_segments', 0) for r in results.values())
    total_with_text = sum(r.get('segments_with_text', 0) for r in results.values())

    print(f"\nTotal segments across all movies: {total_segments}")
    print(f"Total segments with text: {total_with_text}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process HBN movie audio with WhisperX")
    parser.add_argument(
        "--movies_dir",
        type=str,
        required=True,
        help="Directory containing trimmed movie audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed audio and transcriptions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for WhisperX"
    )
    parser.add_argument(
        "--max_segment_length",
        type=float,
        default=12.0,
        help="Maximum segment length in seconds (default: 12 to match Brennan)"
    )
    parser.add_argument(
        "--movies",
        nargs='+',
        default=['DespicableMe', 'DiaryOfAWimpyKid', 'FunwithFractals', 'ThePresent'],
        help="List of movies to process"
    )

    args = parser.parse_args()

    # Check WhisperX is installed
    try:
        import whisperx
    except ImportError:
        print("ERROR: WhisperX not installed!")
        print("Install with: pip install whisperx")
        exit(1)

    # Process movies
    results = process_all_hbn_movies(
        movies_dir=args.movies_dir,
        output_base_dir=args.output_dir,
        movies=args.movies,
        device=args.device,
        batch_size=args.batch_size,
        max_segment_length=args.max_segment_length
    )

    print("\n✓ All processing complete!")