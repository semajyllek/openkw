
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent)) 

# Imports from your repository structure
from model_arch.kwt_model import KWTModel 
from data.kw_dataset import get_data_loaders, get_mel_spectrogram_transform
from data.02_negative_dataset import NegativeWordUnitDataset 
from data.weighted_sampler import HardNegativeWeightedSampler
from config.params import (
    N_MELS, EMBEDDING_DIM, SAMPLE_RATE, AUDIO_LENGTH_SAMPLES,
    NUM_EPOCHS, BATCH_SIZE, CE_LOSS_WEIGHT, TRIPLET_LOSS_WEIGHT,
    TRIPLET_MARGIN, MODEL_PATH_REL, NEGATIVE_DATA_PATH_REL
)
from data.kw_dataset import collate_fn # Assuming this is the correct import

# --- Configuration Flag (for easy switching) ---
USE_HARD_NEGATIVE_SAMPLER = True
# --- End Configuration ---


def _extract_positive_anchors(model, mel_transform, positive_loader, device):
    """
    Generates mean embeddings for all true positive GSC classes (>0) 
    to serve as anchor points for the Hard Negative Sampler.
    """
    model.eval()
    embeddings_by_class = {}
    
    print("Generating positive anchor embeddings...")
    dataset = positive_loader.dataset 
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    with torch.no_grad():
        for audio_data, _, labels, _, _ in tqdm(loader, desc="Extracting Anchors"):
            audio_data, labels = audio_data.to(device), labels.to(device)
            mel_spec = mel_transform(audio_data)
            
            _, embeddings = model(mel_spec) 
            
            for embed, label in zip(embeddings, labels):
                label_id = label.item()
                if label_id not in embeddings_by_class:
                    embeddings_by_class[label_id] = []
                embeddings_by_class[label_id].append(embed.cpu().numpy())

    mean_anchors = []
    for label_id in sorted(embeddings_by_class.keys()):
        if label_id > 0 and embeddings_by_class[label_id]:
            mean_anchor = np.mean(embeddings_by_class[label_id], axis=0)
            mean_anchors.append(mean_anchor)
    
    model.train()
    if not mean_anchors:
        raise ValueError("No positive anchors generated. Check GSC labels (>0).")
        
    print(f"Extracted {len(mean_anchors)} positive anchor embeddings.")
    return np.array(mean_anchors)


# ------------------------------------------------------------------
## Setup Functions
# ------------------------------------------------------------------

def _setup_training(device, num_classes):
    """Initializes model, transform, loss functions, and optimizer."""
    
    time_size = int(np.floor(AUDIO_LENGTH_SAMPLES / (SAMPLE_RATE * 0.01))) - 1 # Recalculate T
    
    model = KWTModel(
        freq_size=N_MELS, time_size=time_size,
        num_classes=num_classes, embedding_dim=EMBEDDING_DIM
    ).to(device)
    
    mel_transform = get_mel_spectrogram_transform(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    loss_funcs = {
        'ce': nn.CrossEntropyLoss(),
        'miner': HardNegativeWeightedSampler(margin=TRIPLET_MARGIN, type_of_triplets="hardest"),
        'triplet': nn.TripletMarginLoss(margin=TRIPLET_MARGIN)
    }
    
    return model, mel_transform, optimizer, loss_funcs

def _setup_data_loaders(model, mel_transform, device, data_root):
    """Sets up data loaders based on the hard negative sampling flag."""
    
    positive_loader, _, num_classes = get_data_loaders(BATCH_SIZE)
    
    if USE_HARD_NEGATIVE_SAMPLER:
        print("Initializing Hard Negative Weighted Sampling strategy...")
        
        negative_data_path = data_root / NEGATIVE_DATA_PATH_REL
        
        try:
            negative_dataset = NegativeWordUnitDataset(negative_data_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"FATAL: Did you run 01_generate_negatives.py? {e}")
            
        positive_anchors = _extract_positive_anchors(model, mel_transform, positive_loader, device)

        negative_sampler = HardNegativeWeightedSampler(
            negative_dataset=negative_dataset,
            positive_anchors=positive_anchors,
            model=model,
            device=device,
            batch_size=BATCH_SIZE
        )
        
        negative_loader = DataLoader(
            negative_dataset, 
            batch_size=BATCH_SIZE,
            sampler=negative_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=True
        )

        combined_loader = zip(positive_loader, negative_loader)
        num_batches = min(len(positive_loader), len(negative_loader))
        
        return combined_loader, num_batches, num_classes, negative_sampler
    
    else:
        print("Using standard random sampling (GSC data only).")
        return positive_loader, len(positive_loader), num_classes, None

# ------------------------------------------------------------------
## Training and Orchestration
# ------------------------------------------------------------------

def _train_one_epoch(model, combined_loader, loss_funcs, optimizer, mel_transform, device, num_batches):
    """Executes the training logic for a single epoch."""
    
    model.train()
    total_loss = 0.0
    
    for _, batch_data in enumerate(tqdm(combined_loader, total=num_batches, desc=f"Training Batch")):
        optimizer.zero_grad()
        
        if USE_HARD_NEGATIVE_SAMPLER:
            (audio_A, _, labels_A, _, _), (audio_B, _, labels_B, _, _) = batch_data
            
            audio_data = torch.cat([audio_A, audio_B], dim=0)
            labels = torch.cat([labels_A, labels_B], dim=0)
        else:
            audio_data, _, labels, _, _ = batch_data
            
        audio_data, labels = audio_data.to(device).squeeze(1), labels.to(device)
        
        mel_spec = mel_transform(audio_data)
        logits, embeddings = model(mel_spec)
        
        loss_ce = loss_funcs['ce'](logits, labels)
        
        mined_triplets = loss_funcs['miner'](embeddings, labels)
        if mined_triplets:
            loss_triplet = loss_funcs['triplet'](embeddings, labels, mined_triplets)
        else:
            loss_triplet = torch.tensor(0.0).to(device)

        loss = CE_LOSS_WEIGHT * loss_ce + TRIPLET_LOSS_WEIGHT * loss_triplet 
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / num_batches

def train_kws_model(data_root: Path):
    """Main function to orchestrate the KWT training process."""
    
    model_path = data_root / MODEL_PATH_REL
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training KWT model on device: {device}")

    # Set up model and data
    # (Note: NUM_CLASSES will be updated by the data loader)
    model, mel_transform, optimizer, loss_funcs = _setup_training(device, num_classes=37)
    combined_loader, num_batches, num_classes, negative_sampler = _setup_data_loaders(model, mel_transform, device, data_root)
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        if USE_HARD_NEGATIVE_SAMPLER and negative_sampler and (epoch % negative_sampler.reweight_frequency == 0):
            print(f"Epoch {epoch+1}: Recalculating hard negative weights...")
            negative_sampler.recalculate_weights()
            
        avg_loss = _train_one_epoch(
            model, combined_loader, loss_funcs, optimizer, mel_transform, device, num_batches
        )
        
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(model_path.parent, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} with improved loss: {best_loss:.4f} 🔥")

    print("\nTraining complete.")

if __name__ == '__main__':
    # Default local path for standalone execution outside of notebook
    LOCAL_DATA_ROOT = Path("./local_run_data")
    os.makedirs(LOCAL_DATA_ROOT, exist_ok=True)
    train_kws_model(LOCAL_DATA_ROOT)
