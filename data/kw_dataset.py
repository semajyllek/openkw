
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob
from tqdm import tqdm
import os
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.params import SAMPLE_RATE, AUDIO_LENGTH_SAMPLES, NON_WAKE_WORD_LABEL, WAKE_WORD_LABEL

# --- Helper Functions (used by both dataset classes) ---

def collate_fn(batch):
    """
    Pads audio tensors in a batch to the longest sequence length.
    """
    audio_tensors = [item["audio"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad the audio tensors to the longest length
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True, padding_value=0)
    
    # Return as a dictionary for clarity
    return {
        "audio": audio_padded,
        "labels": torch.stack(labels)
    }

# --- Dataset Classes ---

class NegativeWordUnitDataset(Dataset):
    """
    Loads all pre-generated 1-second negative word unit audio files 
    from the configured directory. This dataset's purpose is to provide
    'hard negative' samples for the model to learn to reject.
    """
    def __init__(self, data_path: Path):
        """
        Initializes the dataset by finding all audio files.
        
        Args:
            data_path (Path): The path to the directory containing the
                              pre-generated negative audio files.
        """
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

    def __getitem__(self, idx):
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
        """
        Initializes the dataset by finding all audio files.
        
        Args:
            data_path (Path): The path to the directory containing your custom 
                              wake word audio files.
        """
        self.data_path = data_path
        self.sample_rate = SAMPLE_RATE
        
        # Recursively find all .wav files
        self.file_paths = sorted(glob(str(self.data_path / "*.wav")))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No WAV files found in the custom wake word directory: {self.data_path}. Please record and upload some samples.")
            
        print(f"Loaded {len(self.file_paths)} custom wake word samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
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
