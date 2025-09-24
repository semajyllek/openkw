
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.params import (
    PRETRAINED_MODEL, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE,
    MODEL_PATH_REL, WAKE_WORD_LABEL, NON_WAKE_WORD_LABEL
)
from data.negative_dataset import NegativeWordUnitDataset
from data.kw_dataset import CustomWakeWordDataset
from data.hard_negative_miner import HardNegativeMiner, HardNegativeDataset

def _train_one_stage(model, processor, data_loader, device, epochs):
    """Helper function to run a training stage."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch+1}"):
            inputs = processor(
                audio=[b.numpy() for b in batch['audio']], 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = batch['labels'].to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

def fine_tune_model(data_root: Path, wake_word_samples: Path):
    """
    Fine-tunes a pre-trained Wav2Vec2 model in a two-stage process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Pre-trained Model and Processor
    print("Loading pre-trained Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained(PRETRAINED_MODEL)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=NUM_CLASSES)
    model.to(device)

    # 2. Prepare DataLoaders
    negative_data_path = data_root / "negative_word_units"
    negative_dataset = NegativeWordUnitDataset(negative_data_path)
    
    # Check if custom wake word samples exist before creating the dataset
    if not os.path.exists(wake_word_samples):
        raise FileNotFoundError(f"Custom wake word samples not found at: {wake_word_samples}. Please upload your samples first.")
        
    positive_dataset = CustomWakeWordDataset(wake_word_samples)
    
    # --- Stage 1: Fine-tuning with Easy Negatives ---
    print("\n--- Stage 1: Initial Fine-tuning with Easy Negatives ---")
    combined_dataset_easy = torch.utils.data.ConcatDataset([positive_dataset, negative_dataset])
    combined_loader_easy = DataLoader(combined_dataset_easy, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
    
    _train_one_stage(model, processor, combined_loader_easy, device, epochs=NUM_EPOCHS)

    # --- Stage 2: Mining and Retraining with Hard Negatives ---
    print("\n--- Stage 2: Hard Negative Mining and Retraining ---")
    
    # Initialize the miner
    miner = HardNegativeMiner(model, processor, device)
    
    # Mine a small number of hard negatives (e.g., 50)
    num_hard_negatives = len(positive_dataset) * 5 
    hard_negatives_paths = miner.mine_hard_negatives(negative_dataset, num_to_mine=num_hard_negatives, strategy="ranking")
    
    # Create a new dataset with the identified hard negatives
    hard_negative_dataset = HardNegativeDataset(hard_negatives_paths, label=NON_WAKE_WORD_LABEL)
    
    # Combine the small positive dataset with the new, small, and difficult negative dataset
    combined_dataset_hard = torch.utils.data.ConcatDataset([positive_dataset, hard_negative_dataset])
    combined_loader_hard = DataLoader(combined_dataset_hard, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
    
    # Retrain for a few more epochs on the difficult examples
    _train_one_stage(model, processor, combined_loader_hard, device, epochs=NUM_EPOCHS)
    
    # 3. Save the Final Model
    model_path = data_root / MODEL_PATH_REL
    os.makedirs(model_path.parent, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Fine-tuned model (with hard negatives) saved to {model_path}.")
    
if __name__ == "__main__":
    # Example usage:
    # First, generate negatives using generate_negatives.py
    # Then, provide a path to your custom wake word samples
    # fine_tune_model(data_root=Path("./local_data"), wake_word_samples=Path("./my_wake_word_audio"))
    pass
