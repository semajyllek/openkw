
import torch
import torchaudio
import random
from tqdm import tqdm
from pathlib import Path
import sys

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent)) 
from config.params import (
    SAMPLE_RATE, NEGATIVE_DATA_PATH, LIBRISPEECH_SUBSET, 
    NUM_NEGATIVES_TO_COLLECT, NUM_SLICES_PER_UTTERANCE, KNOWN_KEYWORDS
)

# Constants derived from config
TARGET_LENGTH = SAMPLE_RATE # 1 second

def segment_and_save_words(waveform, transcript, unique_id, total_saved_ref):
    """
    Samples short, word-centric segments, pads them to 1s, and saves as WAV files.
    Relies on global config parameters.
    """
    if waveform.ndim != 1:
        waveform = waveform.mean(dim=0)
        
    num_samples = waveform.size(0)
    
    words = transcript.lower().split()
    
    # Filter: Skip the whole utterance if it contains any known keyword
    if any(word in KNOWN_KEYWORDS for word in words):
        return 0 

    samples_saved = 0
    
    for j in range(NUM_SLICES_PER_UTTERANCE):
        if total_saved_ref[0] >= NUM_NEGATIVES_TO_COLLECT:
            break

        # 1. Sample a random slice length (200ms to 800ms)
        slice_ms = random.randint(200, 800)
        slice_samples = int(SAMPLE_RATE * (slice_ms / 1000.0))
        
        max_start = num_samples - slice_samples
        if max_start <= 0:
             continue
        
        start = random.randint(0, max_start)
        end = start + slice_samples
        segment = waveform[start:end]

        # 2. Padding to 1 second
        padding_needed = TARGET_LENGTH - segment.size(0)
        
        if padding_needed > 0:
            padding_start = torch.zeros(padding_needed // 2)
            padding_end = torch.zeros(padding_needed - padding_needed // 2)
            segment = torch.cat([padding_start, segment, padding_end], dim=0)
        else:
            segment = segment[:TARGET_LENGTH]
            
        if segment.size(0) != TARGET_LENGTH:
            continue

        # 3. Save the padded 1-second segment
        filename = f"neg_{unique_id}_{j}.wav"
        save_path = NEGATIVE_DATA_PATH / filename
        
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

def generate_word_unit_negatives():
    """Main execution function for generating and saving negatives."""
    
    NEGATIVE_DATA_PATH.mkdir(parents=True, exist_ok=True)
    total_saved_ref = [0]
    
    print(f"Downloading/loading LibriSpeech subset: {LIBRISPEECH_SUBSET}. This may take a while...")
    try:
        dataset = torchaudio.datasets.LIBRISPEECH(
            root="./data_external", 
            url=LIBRISPEECH_SUBSET, 
            download=True
        )
    except Exception as e:
        print(f"Error downloading LibriSpeech. Ensure torchaudio is installed: {e}")
        return

    for i, (waveform, sr, transcript, speaker_id, chapter_id, utterance_id) in enumerate(tqdm(dataset, desc="Processing Utterances")):
        
        if total_saved_ref[0] >= NUM_NEGATIVES_TO_COLLECT:
            break

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        unique_id = f"{speaker_id}_{chapter_id}_{utterance_id}"
        segment_and_save_words(waveform.squeeze(0), transcript, unique_id, total_saved_ref)

    print(f"\n✅ Finished generating negatives. Total samples saved: {total_saved_ref[0]} in {NEGATIVE_DATA_PATH}")


if __name__ == '__main__':
    generate_word_unit_negatives()
