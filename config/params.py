
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
# This path is for the negative examples we generate
NEGATIVE_DATA_PATH_REL = Path("negative_word_units")
# This is the expected path for the custom wake word you record
POSITIVE_DATA_PATH_REL = Path("my_wake_word")

# --- AUDIO PARAMETERS ---
SAMPLE_RATE = 16000
AUDIO_LENGTH_SAMPLES = SAMPLE_RATE
N_MELS = 40

# --- HARD NEGATIVE MINING PARAMETERS ---
LIBRISPEECH_SUBSET = "train-clean-100"  # LibriSpeech subset for negatives
NUM_NEGATIVES_TO_COLLECT = 50000  # Target number of negative samples to generate
NUM_SLICES_PER_UTTERANCE = 20  # Number of segments to slice from each LibriSpeech utterance

# --- KEYWORDS ---
# This set is now empty as the system is a "blank slate" and learns a custom wake word
KNOWN_KEYWORDS: Set[str] = set()

# --- LABELING ---
WAKE_WORD_LABEL = 1
NON_WAKE_WORD_LABEL = 0
