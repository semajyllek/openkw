
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
    from the configured directory.
    """
    def __init__(self, data_path: Path, label: int):
        """
        Initializes the dataset by finding all audio files.
        
        Args:
            data_path (Path): The path to the directory containing the
                              pre-generated negative audio files.
            label (int): The label to assign to these samples.
        """
        self.data_path = data
