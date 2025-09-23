
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
from glob import glob
import sys

# Add the project's root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.params import SAMPLE_RATE

class NegativeWordUnitDataset(Dataset):
    """
    Loads all pre-generated 1-second negative word unit audio files 
    from the configured directory. This dataset's purpose is to provide
    "hard negative" samples for the model to learn to reject.
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
            
        # The label for all samples is 0 (Negative/Unknown)
        self.negative_label = 0 
        
        print(f"Loaded {len(self.file_paths)} pre-generated negative samples.")

    def __len__(self):
        """Returns the total number of negative samples."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Retrieves a single negative sample by its index.
        
        Args:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            tuple: A tuple containing the waveform, sample rate, and a placeholder label.
        """
        file_path = self.file_paths[idx]
        
        # Load the audio file (already at 16kHz and 1 second)
        waveform, sr = torchaudio.load(file_path)
        
        # Ensure it's 1-channel mono and consistent sample rate
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        if waveform.size(0) > 1:
            # Convert stereo to mono by averaging channels
            waveform = waveform.mean(dim=0, keepdim=True)

        # The output of this dataset must match the expected format of the collate_fn.
        # Format: (waveform, sample_rate, label, speaker_id, utterance_id) 
        # We simplify the last two for this custom dataset.
        
        return waveform, self.sample_rate, self.negative_label, 0, 0
