
import torch
import torchaudio
import random
from tqdm import tqdm
from pathlib import Path
import sys
import os
import shutil

# Add the project's root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration from a centralized file
from config.params import (
    SAMPLE_RATE, NEGATIVE_DATA_PATH_REL, EXTERNAL_DATA_ROOT_REL, LIBRISPEECH_SUBSET,
    NUM_NEGATIVES_TO_COLLECT, NUM_SLICES_PER_UTTERANCE
)

# Constants derived from config
TARGET_LENGTH = SAMPLE_RATE # 1 second (16000 samples)

# --- (The _segment_and_save_words function remains the same) ---

def _segment_and_save_words(waveform, unique_id, total_saved_ref, negative_data_path):
    """
    Samples short, word-centric segments, pads them to 1s, and saves as WAV files.
    This is an internal helper function.
    """
    if waveform.ndim != 1:
        waveform = waveform.mean(dim=0)
    
    num_samples = waveform.size(0)
    samples_saved = 0
    
    for j in range(NUM_SLICES_PER_UTTERANCE):
        if total_saved_ref[0] >= NUM_NEGATIVES_TO_COLLECT:
            break

        slice_ms = random.randint(200, 800)
        slice_samples = int(SAMPLE_RATE * (slice_ms / 1000.0))
        
        max_start = num_samples - slice_samples
        if max_start <= 0:
             continue
        
        start = random.randint(0, max_start)
        end = start + slice_samples
        segment = waveform[start:end]

        padding_needed = TARGET_LENGTH - segment.size(0)
        
        if padding_needed > 0:
            padding_start = torch.zeros(padding_needed // 2)
            padding_end = torch.zeros(padding_needed - padding_needed // 2)
            segment = torch.cat([padding_start, segment, padding_end], dim=0)
        else:
            segment = segment[:TARGET_LENGTH]
            
        if segment.size(0) != TARGET_LENGTH:
            continue

        filename = f"neg_{unique_id}_{j}.wav"
        save_path = negative_data_path / filename
        
        torchaudio.save(
            uri=str(save_path),
            src=segment.unsqueeze(0),
            sample_rate=SAMPLE_RATE,
            encoding="PCM_S",
            bits_per_sample=16
        )
        samples_saved += 1
        total_saved_ref[0] += 1
            
    return samples_saved

# --- (The main function is updated below) ---

def generate_word_unit_negatives(data_root: Path):
    """
    Main function to generate and save negative word units.
    It downloads LibriSpeech and saves samples to the specified data_root.
    
    Args:
        data_root (Path): The root directory where all data will be stored.
    """
    
    # Resolve absolute paths from the relative paths in config
    negative_data_path = data_root / NEGATIVE_DATA_PATH_REL
    external_data_root = data_root / EXTERNAL_DATA_ROOT_REL

    # Ensure the destination directories exist
    negative_data_path.mkdir(parents=True, exist_ok=True)
    external_data_root.mkdir(parents=True, exist_ok=True)

    # Use a list as a mutable reference for counting
    total_saved_ref = [0]
    
    print(f"Downloading/loading LibriSpeech subset: {LIBRISPEECH_SUBSET}. This may take a while...")

    # Define the expected path for the extracted LibriSpeech dataset
    extracted_dataset_path = external_data_root / "LibriSpeech" / LIBRISPEECH_SUBSET

    try:
        # Check if the extracted dataset already exists
        if os.path.exists(extracted_dataset_path) and len(os.listdir(extracted_dataset_path)) > 0:
            print(f"✅ LibriSpeech dataset already exists at {extracted_dataset_path}. Skipping download.")
            # Load the dataset from the existing directory
            dataset = torchaudio.datasets.LIBRISPEECH(
                root=str(external_data_root), 
                url=LIBRISPEECH_SUBSET, 
                download=False
            )
        else:
            print(f"Downloading LibriSpeech to {external_data_root}...")
            # Download and extract the dataset
            dataset = torchaudio.datasets.LIBRISPEECH(
                root=str(external_data_root), 
                url=LIBRISPEECH_SUBSET, 
                download=True
            )
    except Exception as e:
        print(f"🛑 CRITICAL ERROR during LibriSpeech download. Check disk space/permissions for {external_data_root}: {e}")
        return # Halt execution on failure

    # Process and save
    for i, (waveform, sr, transcript, speaker_id, chapter_id, utterance_id) in enumerate(tqdm(dataset, desc="Processing Utterances")):
        
        if total_saved_ref[0] >= NUM_NEGATIVES_TO_COLLECT:
            break

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        unique_id = f"{speaker_id}_{chapter_id}_{utterance_id}"
        
        _segment_and_save_words(waveform.squeeze(0), unique_id, total_saved_ref, negative_data_path)

    print(f"\n✅ Finished generating negatives. Total samples saved: {total_saved_ref[0]} in {negative_data_path}")


if __name__ == '__main__':
    # This block allows the script to be run locally for testing.
    LOCAL_DATA_ROOT = Path("./local_run_data")
    os.makedirs(LOCAL_DATA_ROOT, exist_ok=True)
    generate_word_unit_negatives(LOCAL_DATA_ROOT)
