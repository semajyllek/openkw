
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
from typing import Dict

# Import your global parameters
from config.params import SAMPLE_RATE, AUDIO_LENGTH_SAMPLES, NON_WAKE_WORD_LABEL, WAKE_WORD_LABEL

# --- Dataset Classes ---

class NegativeWordUnitDataset(Dataset):
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.sample_rate = SAMPLE_RATE
        self.file_paths = sorted(glob(str(self.data_path / "*.wav")))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No WAV files found in the negative dataset directory: {self.data_path}")
            
        print(f"Loaded {len(self.file_paths)} pre-generated negative samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        
        # Ensure consistent sample rate and single channel
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Squeeze to a 1D tensor
        waveform = waveform.squeeze()

        # The audio should already be 1 second, but we will trim/pad just in case
        if waveform.size(0) > AUDIO_LENGTH_SAMPLES:
            waveform = waveform[:AUDIO_LENGTH_SAMPLES]
        elif waveform.size(0) < AUDIO_LENGTH_SAMPLES:
            padding = torch.zeros(AUDIO_LENGTH_SAMPLES - waveform.size(0))
            waveform = torch.cat([waveform, padding], dim=0)

        return {"audio": waveform, "labels": torch.tensor(NON_WAKE_WORD_LABEL)}


class CustomWakeWordDataset(Dataset):
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.sample_rate = SAMPLE_RATE
        self.file_paths = sorted(glob(str(self.data_path / "*.wav")))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No WAV files found in the custom wake word directory: {self.data_path}")
            
        print(f"Loaded {len(self.file_paths)} custom wake word samples.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)

        # Ensure consistent sample rate and single channel
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Squeeze to a 1D tensor before processing
        waveform = waveform.squeeze()

        # Pad or trim the audio to a consistent length (1s)
        if waveform.size(0) > AUDIO_LENGTH_SAMPLES:
            waveform = waveform[:AUDIO_LENGTH_SAMPLES]
        elif waveform.size(0) < AUDIO_LENGTH_SAMPLES:
            padding = torch.zeros(AUDIO_LENGTH_SAMPLES - waveform.size(0))
            waveform = torch.cat([waveform, padding], dim=0)
            
        return {"audio": waveform, "labels": torch.tensor(WAKE_WORD_LABEL)}

        
def collate_fn(batch):
    audio_tensors = [item["audio"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad the 1D audio tensors to the longest length
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True, padding_value=0)
    
    return {
        "audio": audio_padded,
        "labels": torch.stack(labels)
    }

