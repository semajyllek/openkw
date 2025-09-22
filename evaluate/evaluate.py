
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

# Imports from your repository structure
from model_arch.kws_model import KWTModel 
from data.kws_dataset import get_data_loaders, get_mel_spectrogram_transform 
from config.params import MODEL_PATH, SAMPLE_RATE, AUDIO_LENGTH_SAMPLES

# --- Configuration ---
BATCH_SIZE = 128
EMBEDDING_DIM = 128
# N_MELS is defined in data/kws_dataset.py, but we use 40 here for clarity
N_MELS = 40   
# --- End Configuration ---


def _setup_model_and_data(device):
    """Initializes the model, loads weights, and sets up the test DataLoader/Transform."""
    
    # 1. Data Setup: Use the utility function to get the test loader
    _, test_loader, NUM_CLASSES = get_data_loaders(BATCH_SIZE)
    
    # 2. Model & Transform Setup
    mel_transform = get_mel_spectrogram_transform(device)

    # Calculate time steps (T) needed for KWT initialization
    N_FFT = int(SAMPLE_RATE * 0.025)
    HOP_LENGTH = int(SAMPLE_RATE * 0.010)
    time_size = (AUDIO_LENGTH_SAMPLES - N_FFT) // HOP_LENGTH + 1
    
    model = KWTModel(
        freq_size=N_MELS, 
        time_size=time_size,
        num_classes=NUM_CLASSES, 
        embedding_dim=EMBEDDING_DIM
    ).to(device)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}. Please run training/train.py first.")
        
    model.eval() # Set model to evaluation mode
    
    return model, mel_transform, test_loader

def _run_inference(model, mel_transform, test_loader, device):
    """Runs the inference loop and collects all predictions, labels, and embeddings."""
    all_predictions = []
    all_true_labels = []
    all_embeddings = []

    print("Running inference on test dataset...")
    with torch.no_grad():
        for audio_data, labels in tqdm(test_loader, desc="Evaluating"):
            audio_data, labels = audio_data.to(device), labels.to(device)
            
            mel_spec = mel_transform(audio_data)
            # Forward pass returns logits and embeddings
            logits, embeddings = model(mel_spec) 
            
            # Classification predictions
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())
            
    return np.array(all_predictions), np.array(all_true_labels), np.array(all_embeddings)

def _calculate_classification_metrics(all_true_labels, all_predictions):
    """Calculates and reports the standard classification accuracy."""
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print("\n" + "="*50)
    print(f"1. Classification Accuracy: {accuracy*100:.2f}%")
    print("="*50)
    return accuracy

def _calculate_embedding_metrics(all_embeddings, all_true_labels):
    """Calculates and reports metric learning quality metrics."""
    print("2. Embedding Separation Metrics (Quality for User Enrollment):")
    
    # Use the AccuracyCalculator for common metric learning metrics
    calculator = AccuracyCalculator(k="max_bin_count") 
    metrics = calculator.get_all_metrics(all_embeddings, all_true_labels, all_embeddings, all_true_labels)
    
    mAP = metrics['mean_average_precision_at_r']
    P_at_1 = metrics['precision_at_1']
    
    print(f"   Mean Average Precision (mAP): {mAP:.4f}")
    print(f"   Precision @ 1 (P@1): {P_at_1:.4f}")
    print("="*50)
    
    if P_at_1 > 0.95:
        print("✅ Metric Learning performance is high, indicating robust user enrollment embeddings!")
    else:
        print("⚠️ Metric Learning performance needs improvement (consider more epochs or a different mining strategy).")
        
    return metrics

def evaluate_kws_model():
    """Main evaluation function: sets up, runs inference, and reports metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading KWT model for evaluation on device: {device}")

    try:
        # Step 1: Setup
        model, mel_transform, test_loader = _setup_model_and_data(device)
        
        # Step 2: Run Inference
        all_predictions, all_true_labels, all_embeddings = _run_inference(
            model, mel_transform, test_loader, device
        )
        
        # Step 3: Calculate Metrics
        _calculate_classification_metrics(all_true_labels, all_predictions)
        _calculate_embedding_metrics(all_embeddings, all_true_labels)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")


if __name__ == '__main__':
    evaluate_kws_model()
