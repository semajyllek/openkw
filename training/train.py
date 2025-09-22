
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from pytorch_metric_learning import losses, miners
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import itertools # Used for zipping loaders

# Imports from your repository structure
from model_arch.kw_model import KWTModel 
from data.kw_dataset import get_data_loaders, get_mel_spectrogram_transform, collate_fn
from data.02_negative_dataset import NegativeWordUnitDataset 
from data.weighted_sampler import HardNegativeWeightedSampler
from config.params import (
    MODEL_PATH, NUM_EPOCHS, BATCH_SIZE, EMBEDDING_DIM, 
    SAMPLE_RATE, AUDIO_LENGTH_SAMPLES
)

# --- Configuration Flags ---
USE_HARD_NEGATIVE_SAMPLER = True 
NEGATIVE_DATA_PATH = Path("data/negative_word_units") 
# --- End Configuration ---


def _extract_positive_anchors(model, mel_transform, positive_loader, device):
    """
    Generates mean embeddings for all true positive GSC classes (>0) 
    to serve as anchor points for the Hard Negative Sampler.
    """
    model.eval()
    embeddings_by_class = {}
    
    print("Generating positive anchor embeddings...")
    # Use the raw dataset to ensure we get all classes
    dataset = positive_loader.dataset 
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    with torch.no_grad():
        for audio_data, _, labels, _, _ in tqdm(loader, desc="Extracting Anchors"):
            audio_data, labels = audio_data.to(device), labels.to(device)
            mel_spec = mel_transform(audio_data)
            
            # Forward pass: model returns (logits, embeddings)
            _, embeddings = model(mel_spec) 
            
            for embed, label in zip(embeddings, labels):
                label_id = label.item()
                if label_id not in embeddings_by_class:
                    embeddings_by_class[label_id] = []
                embeddings_by_class[label_id].append(embed.cpu().numpy())

    # Calculate the mean anchor embedding for each class (skipping class 0)
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
# --- STEP 1: SETUP ---
# ------------------------------------------------------------------

def _setup_training(device, num_classes):
    """Initializes model, transform, loss functions, and optimizer."""
    
    # Calculate time steps (T) needed for KWT initialization
    N_FFT = int(SAMPLE_RATE * 0.025)
    HOP_LENGTH = int(SAMPLE_RATE * 0.010)
    time_size = (AUDIO_LENGTH_SAMPLES - N_FFT) // HOP_LENGTH + 1
    
    model = KWTModel(
        freq_size=40, time_size=time_size,
        num_classes=num_classes, embedding_dim=EMBEDDING_DIM
    ).to(device)
    
    mel_transform = get_mel_spectrogram_transform(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Loss functions
    loss_funcs = {
        'ce': nn.CrossEntropyLoss(),
        'miner': miners.TripletMarginMiner(margin=0.2, type_of_triplets="hardest"),
        'triplet': losses.TripletMarginLoss(margin=0.2)
    }
    
    return model, mel_transform, optimizer, loss_funcs


# ------------------------------------------------------------------
# --- STEP 2: DATA LOADING ---
# ------------------------------------------------------------------

def _setup_data_loaders(model, mel_transform, device):
    """Sets up data loaders based on the hard negative sampling flag."""
    
    # 1. Load GSC data (always needed for positives and labels)
    positive_loader, _, num_classes = get_data_loaders(BATCH_SIZE)
    
    if USE_HARD_NEGATIVE_SAMPLER:
        print("Initializing Hard Negative Weighted Sampling strategy...")
        
        # Load the custom negative dataset (LibriSpeech word units)
        try:
            negative_dataset = NegativeWordUnitDataset(NEGATIVE_DATA_PATH)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"FATAL: {e}. Please run 'python data/01_generate_negatives.py' first.")
            
        # Extract positive anchor embeddings
        positive_anchors = _extract_positive_anchors(model, mel_transform, positive_loader, device)

        # Instantiate the Weighted Sampler 
        negative_sampler = HardNegativeWeightedSampler(
            negative_dataset=negative_dataset,
            positive_anchors=positive_anchors,
            model=model,
            device=device,
            batch_size=BATCH_SIZE
        )
        
        # Create a NEGATIVE DataLoader using the weighted sampler
        negative_loader = DataLoader(
            negative_dataset, 
            batch_size=BATCH_SIZE,
            sampler=negative_sampler, # Use the custom sampler
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=True # Essential for consistent batch sizes
        )

        # Combine Loaders: Use itertools.zip_longest to handle potentially unequal lengths
        combined_loader = zip(positive_loader, negative_loader)
        num_batches = min(len(positive_loader), len(negative_loader))
        
        return combined_loader, num_batches, num_classes, negative_sampler
    
    else:
        print("Using standard random sampling (GSC data only).")
        return positive_loader, len(positive_loader), num_classes, None


# ------------------------------------------------------------------
# --- STEP 3: EPOCH TRAINING ---
# ------------------------------------------------------------------

def _train_one_epoch(model, combined_loader, loss_funcs, optimizer, mel_transform, device, num_batches):
    """Executes the training logic for a single epoch."""
    
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch_data in enumerate(tqdm(combined_loader, total=num_batches, desc=f"Training Batch")):
        optimizer.zero_grad()
        
        if USE_HARD_NEGATIVE_SAMPLER:
            # Unpack the interleaved batch: (GSC batch), (Hard Negatives batch)
            (audio_A, _, labels_A, _, _), (audio_B, _, labels_B, _, _) = batch_data
            
            # Combine Positive and Negative Data for a single forward pass
            audio_data = torch.cat([audio_A, audio_B], dim=0)
            labels = torch.cat([labels_A, labels_B], dim=0)
        else:
            # Standard GSC batch
            audio_data, _, labels, _, _ = batch_data
            
        audio_data, labels = audio_data.to(device).squeeze(1), labels.to(device)
        
        # 1. Feature Extraction and Forward Pass
        mel_spec = mel_transform(audio_data)
        logits, embeddings = model(mel_spec)
        
        # 2. Loss Calculation
        loss_ce = loss_funcs['ce'](logits, labels)
        
        mined_triplets = loss_funcs['miner'](embeddings, labels)
        if mined_triplets:
            loss_triplet = loss_funcs['triplet'](embeddings, labels, mined_triplets)
        else:
            loss_triplet = torch.tensor(0.0).to(device)

        # Combine losses (Weighting is a hyperparameter)
        loss = 0.8 * loss_ce + 0.2 * loss_triplet 
        
        # 3. Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / num_batches


# ------------------------------------------------------------------
# --- STEP 4: ORCHESTRATION ---
# ------------------------------------------------------------------

def train_kw_model():
    """Main function to orchestrate the KWT training process."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training KWT model on device: {device}")

    # 1. Initial Setup (Model, Loss) - Placeholder NUM_CLASSES=37 for initial setup
    model, mel_transform, optimizer, loss_funcs = _setup_training(device, num_classes=37)
    
    # 2. Data Setup (Loaders, Samplers) - Updates model in place if hard sampling is used
    combined_loader, num_batches, num_classes, negative_sampler = _setup_data_loaders(model, mel_transform, device)
    
    # Update model's final classification layer if num_classes was determined during data loading
    # (Although get_data_loaders returns a fixed number, this is good practice)
    # model.set_num_classes(num_classes) 

    # 3. Training Loop
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # Recalculate weights periodically (only for hard negative sampler)
        if USE_HARD_NEGATIVE_SAMPLER and negative_sampler is not None and (epoch % negative_sampler.reweight_frequency == 0):
            print(f"Epoch {epoch+1}: Recalculating hard negative weights...")
            negative_sampler.recalculate_weights()
            
        # Execute one epoch
        avg_loss = _train_one_epoch(
            model, combined_loader, loss_funcs, optimizer, mel_transform, device, num_batches
        )
        
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        # 4. Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH} with improved loss: {best_loss:.4f} 🔥")

    print("\nTraining complete.")

if __name__ == '__main__':
    # Ensure the model directory exists
    if not os.path.exists(Path(MODEL_PATH).parent):
        os.makedirs(Path(MODEL_PATH).parent)
    
    train_kw_model()
