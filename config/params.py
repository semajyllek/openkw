
from pathlib import Path
import torch
from typing import Set

# --- GLOBAL DEVICE SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL AND TRAINING PARAMETERS ---
# Model from Hugging Face for transfer learning
PRETRAINED_MODEL = "facebook/wav2vec2-base-960h" 
# Output classes for fine-tuning (wake word vs. non-wake word)
NUM_CLASSES = 2  
NUM_EPOCHS = 5  # Fewer epochs are needed for fine-tuning
BATCH_SIZE = 32

# --- PATHS (Relative to the data root) ---
MODEL_PATH_REL = Path("models/finetuned_kws.pth")
EXTERNAL_DATA_ROOT_REL = Path("external_datasets")
POSITIVE_DATA_PATH_REL = Path("positive_keywords")
NEGATIVE_DATA_PATH_REL = Path("negative_word_units")

# --- AUDIO PARAMETERS ---
SAMPLE_RATE = 16000
AUDIO_LENGTH_SAMPLES = SAMPLE_RATE

# --- KEYWORDS (Now empty for a "blank slate" system) ---
KNOWN_KEYWORDS: Set[str] = set()

# --- LABELING ---
WAKE_WORD_LABEL = 1
NON_WAKE_WORD_LABEL = 0
