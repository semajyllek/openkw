
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob
from tqdm import tqdm
import os
import sys
from typing import Dict

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.params import SAMPLE_RATE, AUDIO_LENGTH_SAMPLES, NON_WAKE_WORD_LABEL, WAKE_WORD_LABEL

# --- Helper Functions (used by both dataset classes) ---

# In data/kw_dataset.py


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """
    Pads audio tensors and stacks labels in a batch.
    This version is more robust and handles both dictionary and tuple outputs,
    and also ensures labels are tensors before stacking.
    """
    first_item = batch[0]
    
    if isinstance(first_item, dict):
        audio_tensors = [item["audio"] for item in batch]
        labels = [item["labels"] for item in batch]
    elif isinstance(first_item, tuple):
        # Assumes the tuple is in the format (audio, labels)
        audio_tensors = [item[0] for item in batch]
        labels = [item[1] for item in batch]
    else:
        raise TypeError(f"Batch items must be a dict or a tuple, got {type(first_item)}")

    # Pad the audio tensors to the longest length
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True, padding_value=0)
    
    # Ensure all labels are tensors before stacking
    labels_tensors = [torch.tensor(label) if not isinstance(label, torch.Tensor) else label for label in labels]
    
    return {
        "audio": audio_padded,
        "labels": torch.stack(labels_tensors)
    }


# --- Dataset Classes ---

class NegativeWordUnitDataset(Dataset):
    """
    Loads all pre-generated 1-second negative word unit audio files 
    from the configured directory.
    """
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.sample_rate = SAMPLE_RATE
        
        # Recursively find all .wav files in the directory
        self.file_paths = sorted(glob(str(self.data_path / "*.wav")))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No WAV files found in the negative dataset directory: {self.data_path}. Did you run generate_negatives.py?")
            
        print(f"Loaded {len(self.file_paths)} pre-generated negative samples.")

    def __len__(self):
        """Returns the total number of negative samples."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single negative sample by its index.
        """
        file_path = self.file_paths[idx]
        
        waveform, sr = torchaudio.load(file_path)
        
        # Ensure consistent sample rate and single channel
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        return {"audio": waveform.squeeze(0), "labels": torch.tensor(NON_WAKE_WORD_LABEL)}

class CustomWakeWordDataset(Dataset):
    """
    Loads custom wake word audio files and prepares them as positive samples.
    """
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.sample_rate = SAMPLE_RATE
        
        # Recursively find all .wav files
        self.file_paths = sorted(glob(str(self.data_path / "*.wav")))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No WAV files found in the custom wake word directory: {self.data_path}. Please record and upload some samples.")
            
        print(f"Loaded {len(self.file_paths)} custom wake word samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)

        # Pad or trim the audio to a consistent length (1s)
        if waveform.size(1) < AUDIO_LENGTH_SAMPLES:
            padding = torch.zeros(1, AUDIO_LENGTH_SAMPLES - waveform.size(1))
            waveform = torch.cat([waveform, padding], 1)
        elif waveform.size(1) > AUDIO_LENGTH_SAMPLES:
            waveform = waveform[:, :AUDIO_LENGTH_SAMPLES]

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Ensure single channel
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        return {"audio": waveform.squeeze(0), "labels": torch.tensor(WAKE_WORD_LABEL)}
