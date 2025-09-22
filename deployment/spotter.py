
import argparse
import torch
import os
import sys
import shutil
import pickle
import subprocess

# Import core logic and paths
from kw_dispatcher.commands import load_commands_from_file
from kw_dispatcher.runtime import run_command_dispatcher
from config.params import (
    MODEL_PATH, DETECTION_THRESHOLD, CONFIG_DIR, KEYWORDS_DATA_PATH, COMMANDS_JSON_PATH, 
    DEFAULT_COMMANDS_PATH
)


# Global keyword dictionary - loaded at runtime
ENROLLED_KEYWORDS = {} 


# --- PERSISTENCE FUNCTIONS ---

def load_enrolled_keywords():
    """Loads keyword reference vectors from persistent storage."""
    global ENROLLED_KEYWORDS
    if KEYWORDS_DATA_PATH.exists():
        try:
            # We use torch.load for PyTorch Tensor persistence
            ENROLLED_KEYWORDS = torch.load(KEYWORDS_DATA_PATH)
            print(f"Loaded {len(ENROLLED_KEYWORDS)} enrolled keywords from persistent storage.")
        except Exception as e:
            print(f"[ERROR] Failed to load keywords from {KEYWORDS_DATA_PATH}: {e}")
            ENROLLED_KEYWORDS = {} # Reset to empty map
    else:
        print("[INFO] No previous keyword data found.")
    return ENROLLED_KEYWORDS


def save_enrolled_keywords():
    """Saves the current keyword reference vectors to persistent storage."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True) # Ensure config directory exists
        torch.save(ENROLLED_KEYWORDS, KEYWORDS_DATA_PATH)
        print(f"Saved {len(ENROLLED_KEYWORDS)} keywords to {KEYWORDS_DATA_PATH}.")
    except Exception as e:
        print(f"[ERROR] Failed to save keywords: {e}")


# --- SETUP AND CHECK FUNCTIONS ---

def check_dependencies():
    """Checks for required system dependencies (like SoX/rec)."""
    print("Checking system dependencies...")
    try:
        # Check for 'rec' (the recording binary usually bundled with SoX)
        subprocess.run(["rec", "--version"], capture_output=True, check=True, timeout=5)
        print("[SUCCESS] 'rec' (SoX) found.")
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError):
        print("\n" + "="*80)
        print("[FATAL ERROR] 'SoX' (required for audio recording) not found or not working.")
        print("Please install it via your system package manager:")
        print("  macOS (Homebrew):   $ brew install sox")
        print("  Linux (Debian/Ubuntu): $ sudo apt install sox libsox-fmt-all")
        print("="*80 + "\n")
        sys.exit(1)


def run_setup_wizard():
    """Initializes the configuration directory and copies default files."""
    print("--- Running Voice Command Dispatcher Setup ---")
    
    # 1. Create the configuration directory
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created configuration directory: {CONFIG_DIR}")
    
    # 2. Copy the default commands.json
    try:
        shutil.copy(DEFAULT_COMMANDS_PATH, COMMANDS_JSON_PATH)
        print(f"[OK] Copied default commands to: {COMMANDS_JSON_PATH}")
        print("\nTO DO: Edit this file to customize your commands.")
        print("TO DO: Create a 'commands/' directory and your executable shell scripts.")
    except FileNotFoundError:
        print(f"[WARNING] Could not find default commands.json at {DEFAULT_COMMANDS_PATH}.")
        print("Please ensure you run 'voice-cli setup' from the root of the project.")
        
    print("\nSetup complete. You can now use 'voice-cli enroll' and 'voice-cli listen'.")


# --- CORE LOGIC PLACEHOLDERS (Simplified) ---

def load_optimized_model(model_path):
    """Loads the model, now using the Path object."""
    if not model_path.exists():
        print(f"[ERROR] Model weights file not found at {model_path}. Cannot proceed.")
        sys.exit(1)
    # Placeholder for actual PyTorch loading
    return {"status": "Loaded", "path": str(model_path)}

def enroll_keyword(keyword: str, audio_files: list, inference_model):
    """Processes audio and saves the resulting reference vector."""
    print(f"Enrolling keyword '{keyword}' with {len(audio_files)} samples...")
    # --- Actual ML logic goes here (Get embeddings, average, normalize) ---
    
    # Mock Action: Create a dummy vector for persistence test
    dummy_vector = torch.randn(128) 
    
    # Update global map
    ENROLLED_KEYWORDS[keyword] = dummy_vector
    
    # Save to disk (Persistence step)
    save_enrolled_keywords()
    
    print(f"Enrollment complete. Keyword '{keyword}' ready for use.")


# --- CLI Main Function ---

def cli_main():
    parser = argparse.ArgumentParser(description='Voice Command Dispatcher CLI.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 1. SETUP command
    subparsers.add_parser('setup', help='Initializes the configuration directory and copies default files.')

    # 2. ENROLL command
    parser_enroll = subparsers.add_parser('enroll', help='Enroll a new keyword from audio samples.')
    parser_enroll.add_argument('keyword', type=str, help='The command word to enroll (e.g., "empty").')
    parser_enroll.add_argument('audio_files', nargs='+', help='List of recorded WAV files (e.g., audio_samples/word_*.wav).')

    # 3. LISTEN command
    parser_listen = subparsers.add_parser('listen', help='Starts the continuous voice command dispatcher.')
    parser_listen.add_argument('--threshold', type=float, default=DETECTION_THRESHOLD, help=f'Detection threshold (default: {DETECTION_THRESHOLD}).')

    args = parser.parse_args()
    
    if args.command == 'setup':
        check_dependencies()
        run_setup_wizard()
        return

    # For 'enroll' and 'listen' commands, we need to load dependencies and the model
    check_dependencies()
    inference_model = load_optimized_model(MODEL_PATH)
    load_enrolled_keywords() # Load saved keywords for spotting/updating

    if args.command == 'enroll':
        enroll_keyword(args.keyword, args.audio_files, inference_model)
        
    elif args.command == 'listen':
        # Load user-defined commands from the external JSON file
        load_commands_from_file()
        
        # Start the continuous listening and dispatching process
        run_command_dispatcher(inference_model, args.threshold)


if __name__ == "__main__":
    cli_main()
