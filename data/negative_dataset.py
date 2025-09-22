
import torch
from torch.utils.data import Dataset
import torchaudio
from glob import glob
from pathlib import Path
import sys

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent)) 
from config.params import NEGATIVE_DATA_PATH, SAMPLE_RATE

class NegativeWordUnitDataset(Dataset):
    """
    Loads all pre-generated 1-second negative word unit audio files 
    from the configured directory.
    """
    def __init__(self):
        self.data_path = NEGATIVE_DATA_PATH
        self.sample_rate = SAMPLE_RATE
        
        self.file_paths = sorted(glob(str(self.data_path / "*.wav")))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No WAV files found in the negative dataset directory: {self.data_path}. Did you run 01_generate_negatives.py?")
            
        self.negative_label = 0 
        
        print(f"Loaded {len(self.file_paths)} pre-generated negative samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        waveform, sr = torchaudio.load(file_path)
        
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # We return the waveform and a placeholder label/ids for compatibility
        return waveform, self.sample_rate, self.negative_label, 0, 0
