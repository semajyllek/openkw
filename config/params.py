
from pathlib import Path
import torch
from typing import Set

# --- GLOBAL DEVICE SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL ARCHITECTURE PARAMETERS ---
N_MELS = 40  # Number of mel-frequency bands
EMBEDDING_DIM = 256  # Size of the final feature embedding layer
NUM_CLASSES = 37  # 35 GSC classes + '_silence_' + '_unknown_'

# --- AUDIO AND DATA PARAMETERS ---
SAMPLE_RATE = 16000  # Standard sample rate for speech
AUDIO_LENGTH_SAMPLES = SAMPLE_RATE  # 1 second of audio at 16000 samples

# --- TRAINING PARAMETERS ---
NUM_EPOCHS = 30
BATCH_SIZE = 64
TRIPLET_MARGIN = 0.2
CE_LOSS_WEIGHT = 0.8  # Weight for Cross-Entropy Loss
TRIPLET_LOSS_WEIGHT = 0.2  # Weight for Triplet Loss

# --- PATHS (Defined relative to the project's data root) ---
# The data root is passed in at runtime by the main training script.
MODEL_PATH_REL = Path("models/best_kwt_model.pth")
NEGATIVE_DATA_PATH_REL = Path("negative_word_units")
EXTERNAL_DATA_ROOT_REL = Path("external_datasets")

# --- HARD NEGATIVE MINING PARAMETERS ---
LIBRISPEECH_SUBSET = "train-clean-100"  # LibriSpeech subset for negatives
NUM_NEGATIVES_TO_COLLECT = 50000  # Target number of negative samples to generate
NUM_SLICES_PER_UTTERANCE = 20  # Number of segments to slice from each LibriSpeech utterance

# --- KEYWORDS (GSC and Custom) ---
GSC_KEYWORDS = [
    "bed", "bird", "cat", "dog", "down", "eight", "five", "four",
    "go", "happy", "house", "left", "marvin", "nine", "no", "off",
    "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "wow", "yes", "zero",
    "backward", "forward", "follow", "learn", "visual"
]
CUSTOM_KEYWORDS = ["mykeyword", "trigger", "arm"]
KNOWN_KEYWORDS: Set[str] = set(GSC_KEYWORDS + CUSTOM_KEYWORDS)
