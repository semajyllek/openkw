
import argparse
import logging
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_args():
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a negative audio dataset from source files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing source audio files (e.g., .wav, .flac).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the generated negative clips.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for all output clips.",
    )
    parser.add_argument(
        "--clip-duration-ms",
        type=int,
        default=1000,
        help="Duration of each output clip in milliseconds.",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=0.01,
        help="RMS energy threshold to discard silent clips. Tune this based on your data.",
    )
    parser.add_argument(
        "--max-clips-per-file",
        type=int,
        default=20,
        help="Maximum number of clips to extract from a single source file to ensure diversity.",
    )
    parser.add_argument(
        "--file-ext",
        type=str,
        default="wav",
        help="The file extension for the output audio clips.",
    )
    return parser.parse_args()


def process_audio_file(
    file_path: Path,
    output_dir: Path,
    sample_rate: int,
    clip_samples: int,
    rms_threshold: float,
    max_clips: int,
    file_ext: str,
) -> int:
    """
    Loads an audio file, extracts non-silent clips, and saves them.

    Args:
        file_path (Path): Path to the source audio file.
        output_dir (Path): Directory to save extracted clips.
        sample_rate (int): Target audio sample rate.
        clip_samples (int): Number of samples per clip.
        rms_threshold (float): Minimum RMS energy to be considered non-silent.
        max_clips (int): Maximum number of clips to generate from this file.
        file_ext (str): Output file extension.

    Returns:
        int: The number of valid clips generated from the file.
    """
    try:
        # Load audio, ensuring it's mono and at the target sample rate
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)

        # Skip if audio is shorter than the desired clip duration
        if len(audio) < clip_samples:
            return 0

        # Calculate number of possible clips with 50% overlap
        hop_length = clip_samples // 2
        num_frames = 1 + (len(audio) - clip_samples) // hop_length
        
        possible_starts = [i * hop_length for i in range(num_frames)]
        random.shuffle(possible_starts) # Shuffle to get a random sample
        
        clips_generated = 0
        for start_sample in possible_starts:
            if clips_generated >= max_clips:
                break
                
            end_sample = start_sample + clip_samples
            clip = audio[start_sample:end_sample]

            # Calculate Root Mean Square (RMS) energy to filter out silence
            rms = np.sqrt(np.mean(np.square(clip)))

            if rms >= rms_threshold:
                # Construct a unique filename for the clip
                output_filename = f"{file_path.stem}_neg_{start_sample}.{file_ext}"
                output_path = output_dir / output_filename
                
                # Save the clip
                sf.write(output_path, clip, sample_rate)
                clips_generated += 1
                
        return clips_generated

    except Exception as e:
        logging.error(f"Failed to process {file_path.name}: {e}")
        return 0


def main():
    """Main function to run the dataset creation process."""
    args = get_args()

    # --- Validation and Setup ---
    if not args.input_dir.is_dir():
        logging.error(f"Input directory not found: {args.input_dir}")
        return

    # Create the output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output clips will be saved to: {args.output_dir}")

    # --- Find Audio Files ---
    supported_formats = ["*.wav", "*.flac", "*.mp3", "*.ogg"]
    audio_files = []
    for fmt in supported_formats:
        audio_files.extend(list(args.input_dir.rglob(fmt)))

    if not audio_files:
        logging.warning("No audio files found in the input directory. Exiting.")
        return

    logging.info(f"Found {len(audio_files)} audio files to process.")

    # --- Processing ---
    clip_duration_s = args.clip_duration_ms / 1000.0
    clip_samples = int(args.sample_rate * clip_duration_s)
    total_clips_created = 0

    # Use tqdm for a progress bar
    progress_bar = tqdm(audio_files, desc="Processing files", unit="file")
    for file_path in progress_bar:
        count = process_audio_file(
            file_path,
            args.output_dir,
            args.sample_rate,
            clip_samples,
            args.rms_threshold,
            args.max_clips_per_file,
            args.file_ext,
        )
        if count > 0:
            total_clips_created += count
            progress_bar.set_postfix({"total_clips": total_clips_created})

    logging.info("-" * 30)
    logging.info("Dataset creation complete.")
    logging.info(f"Total negative clips created: {total_clips_created}")
    logging.info("-" * 30)


if __name__ == "__main__":
    main()
