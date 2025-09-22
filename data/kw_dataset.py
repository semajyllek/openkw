
import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram

# Imports from your repository structure
from config.params import SAMPLE_RATE, AUDIO_LENGTH_SAMPLES, N_MELS 
# --- End Configuration ---


def get_mel_spectrogram_transform(device):
    """
    returns the Mel Spectrogram transformation pipeline, configured for 
    KW standards (25ms window, 10ms hop, 40 Mel bins).
    """
    mel_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_mels=N_MELS, 
        n_fft=int(SAMPLE_RATE * 0.025), # 25ms window
        hop_length=int(SAMPLE_RATE * 0.010) # 10ms hop
    ).to(device)
    return mel_transform


def collate_fn(batch):
    """
    pads or truncates raw audio waveforms in the batch to a fixed size 
    (AUDIO_LENGTH_SAMPLES) and returns the audio tensor and label indices.
    """
    processed_waveforms = []
    labels = []
    
    for waveform, sr, label, _, _ in batch:
        # ensure waveform is (1, Samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # truncate longer audio
        if waveform.shape[1] > AUDIO_LENGTH_SAMPLES:
            waveform = waveform[:, :AUDIO_LENGTH_SAMPLES]
        # zero-pad shorter audio
        elif waveform.shape[1] < AUDIO_LENGTH_SAMPLES:
            padding = torch.zeros(1, AUDIO_LENGTH_SAMPLES - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        
        processed_waveforms.append(waveform)
        labels.append(label)
        
    audio_tensors = torch.cat(processed_waveforms, dim=0).unsqueeze(1) 
    
    # Convert string labels to numerical indices
    label_indices = torch.tensor([
        SPEECHCOMMANDS.LABELS.index(l) for l in labels
    ])
    
    return audio_tensors, label_indices

def get_data_loaders(batch_size, num_workers=4):
    """
    Loads the SPEECHCOMMANDS dataset and returns the training and testing DataLoaders.
    """
    # Load the datasets, which handles downloading if needed
    train_dataset = SPEECHCOMMANDS(root="./data", download=True, subset="training")
    test_dataset = SPEECHCOMMANDS(root="./data", download=True, subset="testing")
    
    # Get the number of classes for model setup
    NUM_CLASSES = len(SPEECHCOMMANDS.LABELS)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, test_loader, NUM_CLASSES
