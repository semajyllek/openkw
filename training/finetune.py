import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm.autonotebook import tqdm
import os
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from data.kw_dataset import NegativeWordUnitDataset, CustomWakeWordDataset, collate_fn 
from data.hard_negative_miner import HardNegativeMiner, HardNegativeDataset

# --- Constants from your params.py file ---
PRETRAINED_MODEL = "anton-l/wav2vec2-finetuned-keyword-spotting"
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 10
# ----------------------------------------


def _train_one_stage(model, processor, data_loader, device, epochs):
    """Helper function to run a training stage."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch+1}"):
            inputs = processor(
                audio=batch['audio'].squeeze().tolist(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = batch['labels'].to(device).long() # Ensure labels are long type

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

# The complete fine_tune_model function with all fixes
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
    negative_dataset = NegativeWordUnitDataset(data_path=negative_data_path)
    
    if not os.path.exists(wake_word_samples):
        raise FileNotFoundError(f"Custom wake word samples not found at: {wake_word_samples}. Please upload your samples first.")
        
    positive_dataset = CustomWakeWordDataset(data_path=wake_word_samples)
    
    # Combine datasets for the initial training stage
    combined_dataset_easy = torch.utils.data.ConcatDataset([positive_dataset, negative_dataset])
    
    # 🛑 --- NEW DATA INTEGRITY CHECK --- 🛑
    print("\n--- Performing Data Integrity Check on the Dataset ---")
    for i, item in enumerate(tqdm(combined_dataset_easy, desc="Checking Dataset Integrity")):
        try:
            label = item["labels"]
            if not isinstance(label.item(), int):
                raise TypeError(f"Label is not an integer. Found type: {type(label.item())} at sample index {i}")
            if label.item() not in [0, 1]:
                raise ValueError(f"Label out of range. Expected 0 or 1, but got {label.item()} at sample index {i}")
        except Exception as e:
            print(f"\n🛑 Data Integrity Check Failed at sample {i}. Error: {e}")
            print(f"File causing the error is likely related to dataset index {i}.")
            print("To fix, you may need to manually inspect and remove the problematic audio file.")
            return # Halt execution to allow user to fix the issue

    print("✅ Dataset integrity check passed.")

    # --- Stage 1: Fine-tuning with Easy Negatives ---
    print("\n--- Stage 1: Initial Fine-tuning with Easy Negatives ---")
    combined_loader_easy = DataLoader(
        combined_dataset_easy,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    _train_one_stage(model, processor, combined_loader_easy, device, epochs=NUM_EPOCHS)

    # --- Stage 2: Mining and Retraining with Hard Negatives ---
    print("\n--- Stage 2: Hard Negative Mining and Retraining ---")
    hard_negative_miner = HardNegativeMiner(model, processor, negative_dataset, device, top_k=500)
    hard_negative_samples = hard_negative_miner.mine_hard_negatives()
    
    hard_negative_dataset = HardNegativeDataset(hard_negative_samples)
    
    combined_dataset_hard = torch.utils.data.ConcatDataset([positive_dataset, hard_negative_dataset])
    combined_loader_hard = DataLoader(
        combined_dataset_hard,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    _train_one_stage(model, processor, combined_loader_hard, device, epochs=NUM_EPOCHS)
    
    # 3. Save the fine-tuned model
    print("\nSaving the fine-tuned model...")
    model.save_pretrained("./finetuned_model")
    processor.save_pretrained("./finetuned_model")
    print("Model and processor saved to './finetuned_model/'")
