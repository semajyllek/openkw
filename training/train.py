
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram

# Import the miner and loss function from the metric learning library
from pytorch_metric_learning import losses, miners 

# Imports from your repository structure
from model_arch.kws_model import KWTModel 
from config.params import MODEL_PATH, SAMPLE_RATE, AUDIO_LENGTH_SAMPLES

# --- Training Configuration ---
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EMBEDDING_DIM = 128
MARGIN = 0.5  
N_MELS = 40   
# --- End Configuration ---


def collate_fn(batch):
    """
    Collates a batch of raw audio tensors, ensuring they are all the same length.
    (Omitted for brevity, but same as previous version)
    """
    # ... (Keep the exact collate_fn from the previous step)
    
    processed_waveforms = []
    labels = []
    
    for waveform, sr, label, _, _ in batch:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        if waveform.shape[1] > AUDIO_LENGTH_SAMPLES:
            waveform = waveform[:, :AUDIO_LENGTH_SAMPLES]
        elif waveform.shape[1] < AUDIO_LENGTH_SAMPLES:
            padding = torch.zeros(1, AUDIO_LENGTH_SAMPLES - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        
        processed_waveforms.append(waveform)
        labels.append(label)
        
    audio_tensors = torch.cat(processed_waveforms, dim=0).unsqueeze(1) 
    label_indices = torch.tensor([
        SPEECHCOMMANDS.LABELS.index(l) for l in labels
    ])
    
    return audio_tensors, label_indices


def train_kws_model():
    """
    Trains the KWT model using combined Cross-Entropy and Hard-Mined Triplet Loss.
    """
    # 0. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data Loading
    train_dataset = SPEECHCOMMANDS(root="./data", download=True, subset="training")
    NUM_CLASSES = len(SPEECHCOMMANDS.LABELS)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4
    )

    # 2. Model, Transform, Optimizer, Miner, and Loss
    mel_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_mels=N_MELS, 
        n_fft=int(SAMPLE_RATE * 0.025), 
        hop_length=int(SAMPLE_RATE * 0.010)
    ).to(device)
    time_size = (AUDIO_LENGTH_SAMPLES - int(SAMPLE_RATE * 0.025)) // int(SAMPLE_RATE * 0.010) + 1
    
    model = KWTModel(
        freq_size=N_MELS, 
        time_size=time_size,
        num_classes=NUM_CLASSES, 
        embedding_dim=EMBEDDING_DIM
    ).to(device)
    
    # Loss functions: 
    ce_loss = nn.CrossEntropyLoss()
    
    # *** UPGRADED: Add the Triplet Miner for Hard Mining ***
    triplet_miner = miners.TripletMarginMiner(
        margin=MARGIN,
        type_of_triplets="hard" # Selects the hardest (most informative) triplets in the batch
    )
    # The loss function takes the triplets provided by the miner
    triplet_loss_func = losses.TripletMarginLoss(margin=MARGIN)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print(f"Starting KWT training with Hard Triplet Mining for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_ce_loss = 0.0
        running_triplet_loss = 0.0
        
        for i, (audio_data, labels) in enumerate(train_loader):
            audio_data, labels = audio_data.to(device), labels.to(device)
            mel_spec = mel_transform(audio_data)
            
            # Forward pass
            logits, embeddings = model(mel_spec)
            
            # 3.1 Classification Loss (standard)
            loss_ce = ce_loss(logits, labels)
            
            # 3.2 Metric Learning Loss (with Mining)
            # Find the hard triplets (A, P, N indices)
            mined_triplets = triplet_miner(embeddings, labels)
            
            # Only calculate loss if hard triplets were found in the batch
            if mined_triplets[0].shape[0] > 0:
                loss_triplet = triplet_loss_func(embeddings, labels, mined_triplets)
            else:
                loss_triplet = torch.tensor(0.0, device=device) # No informative triplets found
            
            # Combined Loss (Use a weighting factor if necessary, e.g., loss_ce + 0.1 * loss_triplet)
            loss = loss_ce + loss_triplet
            
            # 3.3 Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_ce_loss += loss_ce.item()
            running_triplet_loss += loss_triplet.item()
            
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Avg CE Loss: {running_ce_loss/len(train_loader):.4f} | Avg Triplet Loss (Mined): {running_triplet_loss/len(train_loader):.4f}")

    # 4. Save the Final Model Weights
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nTraining complete. Model saved to: {MODEL_PATH}")


if __name__ == '__main__':
    # Ensure config/params.py is set up before running
    train_kws_model()
