
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners 

# --- Imports from your repository structure ---
from model_arch.kws_model import KWTModel 
from data.kws_dataset import get_data_loaders, get_mel_spectrogram_transform 
from config.params import MODEL_PATH, AUDIO_LENGTH_SAMPLES
# ---

# --- Training Configuration (same as before) ---
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EMBEDDING_DIM = 128
MARGIN = 0.5  
# --- End Configuration ---


def train_kws_model():
    """Trains the KWT model using combined Cross-Entropy and Hard-Mined Triplet Loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data Loading: Get DataLoaders and number of classes from the data module
    train_loader, _, NUM_CLASSES = get_data_loaders(BATCH_SIZE)
    
    # 2. Model, Transform, Optimizer, Miner, and Loss Setup
    mel_transform = get_mel_spectrogram_transform(device)
    
    # Calculate time steps (T) needed for KWT initialization
    # Note: KWT expects this for positional encoding setup
    N_FFT = int(SAMPLE_RATE * 0.025)
    HOP_LENGTH = int(SAMPLE_RATE * 0.010)
    time_size = (AUDIO_LENGTH_SAMPLES - N_FFT) // HOP_LENGTH + 1
    
    model = KWTModel(
        freq_size=40, # N_MELS is 40
        time_size=time_size,
        num_classes=NUM_CLASSES, 
        embedding_dim=EMBEDDING_DIM
    ).to(device)
    
    # Loss functions and Miner (same Hard Mining strategy)
    ce_loss = nn.CrossEntropyLoss()
    triplet_miner = miners.TripletMarginMiner(margin=MARGIN, type_of_triplets="hard")
    triplet_loss_func = losses.TripletMarginLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop (remains identical)
    print(f"Starting KWT training with Hard Triplet Mining for {NUM_EPOCHS} epochs...")
    # ... (Keep the exact training loop logic from the previous train.py) ...
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_ce_loss = 0.0
        running_triplet_loss = 0.0
        
        for i, (audio_data, labels) in enumerate(train_loader):
            audio_data, labels = audio_data.to(device), labels.to(device)
            mel_spec = mel_transform(audio_data)
            
            logits, embeddings = model(mel_spec)
            
            loss_ce = ce_loss(logits, labels)
            
            mined_triplets = triplet_miner(embeddings, labels)
            
            if mined_triplets[0].shape[0] > 0:
                loss_triplet = triplet_loss_func(embeddings, labels, mined_triplets)
            else:
                loss_triplet = torch.tensor(0.0, device=device)
            
            loss = loss_ce + loss_triplet
            
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
    train_kws_model()
