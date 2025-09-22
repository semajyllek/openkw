
from pathlib import Path
import os
import torch
from typing import Set

# --- GLOBAL DEVICE SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL ARCHITECTURE PARAMETERS ---
# Number of mel-frequency bands (audio features) - Standard for KWS/ASR
N_MELS = 40 
EMBEDDING_DIM = 256  # Size of the final feature embedding layer

# --- AUDIO AND DATA PARAMETERS ---
SAMPLE_RATE = 16000     # Standard sample rate for Speech Commands
AUDIO_LENGTH_SAMPLES = SAMPLE_RATE  # 1 second of audio at 16000 samples
NUM_CLASSES = 37        # 35 GSC classes + '_silence_' + '_unknown_' (fixed for GSC)
DATA_PATH = Path("./data") # Root directory for data operations

# --- TRAINING PARAMETERS ---
NUM_EPOCHS = 30
BATCH_SIZE = 64
# Path where the best model weights will be saved
MODEL_PATH = Path("models/best_kwt_model.pth") 

# --- TRIPLET LOSS PARAMETERS ---
TRIPLET_MARGIN = 0.2
CE_LOSS_WEIGHT = 0.8    # Weight for Cross-Entropy Loss
TRIPLET_LOSS_WEIGHT = 0.2 # Weight for Triplet Loss (CE_LOSS_WEIGHT + TRIPLET_LOSS_WEIGHT = 1.0)

# --- HARD NEGATIVE MINING PARAMETERS ---
# Directory where generated LibriSpeech word unit files are saved
NEGATIVE_DATA_PATH = DATA_PATH / "negative_word_units" 
# The LibriSpeech subset to download (e.g., 'train-clean-100' is 100 hours of clean speech)
LIBRISPEECH_SUBSET = "train-clean-100" 
NUM_NEGATIVES_TO_COLLECT = 50000 
# Number of random short slices to attempt to extract from each long utterance
NUM_SLICES_PER_UTTERANCE = 20 

# List of all keywords (GSC + your custom ones) to ensure they are filtered out
KNOWN_KEYWORDS: Set[str] = set([
    "up", "down", "left", "right", "go", "stop", "yes", "no", 
    "on", "off", "one", "two", "three", "four", "five", "six", 
    "seven", "eight", "nine", "zero", "bed", "bird", "cat", 
    "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
    "backward", "forward", "follow", "learn", "visual"
    # IMPORTANT: ADD ANY CUSTOM KEYWORDS HERE
])

# --- DIRECTORY CHECK ---
# Ensure directories exist upon module load for safety
os.makedirs(MODEL_PATH.parent, exist_ok=True)
os.makedirs(NEGATIVE_DATA_PATH, exist_ok=True)
