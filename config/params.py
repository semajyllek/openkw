
import os
from pathlib import Path

# --- Runtime Paths (Standardized and Persistent) ---
# Use platform-independent way to determine the home directory (~/.config/voice-cli-dispatcher)
CONFIG_DIR = Path.home() / ".config" / "openkw"
COMMANDS_JSON_PATH = CONFIG_DIR / "commands.json"
KEYWORDS_DATA_PATH = CONFIG_DIR / "keywords.dat" 
DEFAULT_COMMANDS_PATH = Path(os.getcwd()) / "commands.json" # Used for initial copy

# Audio Parameters
SAMPLE_RATE = 16000
AUDIO_LENGTH_SECONDS = 1.0
AUDIO_LENGTH_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH_SECONDS)

# Model Parameters
MODEL_FILENAME = "kw_tcresnet_embedder.pth"
MODEL_PATH = Path("models") / MODEL_FILENAME
DETECTION_THRESHOLD = 0.82 
MAX_RUNTIME_SECONDS = 3600 * 24
