
import subprocess
import json
import os
from typing import Dict, Callable
from config.params import COMMANDS_JSON_PATH # Import standardized path

COMMAND_MAP: Dict[str, Callable] = {}

def run_command(command: str):
    """Utility to run the shell script specified by the command string."""
    print(f"[DISPATCH] Executing command: {command}")
    
    # Check if the command is a path to a file
    if os.path.exists(command) and not os.access(command, os.X_OK):
        print(f"[ERROR] Command script '{command}' is not executable. Run: chmod +x {command}")
        return

    try:
        # Popen runs the command asynchronously so the KWS thread is not blocked
        subprocess.Popen(command, shell=True)
    except Exception as e:
        print(f"[ERROR] Failed to execute command '{command}': {e}")


def load_commands_from_file(file_path=COMMANDS_JSON_PATH):
    """Loads keyword-to-action mappings from the external JSON configuration."""
    global COMMAND_MAP
    COMMAND_MAP.clear()
    
    if not file_path.exists():
        print(f"[WARNING] Configuration file '{file_path}' not found. Run 'voice-cli setup'.")
        return
        
    try:
        with open(file_path, 'r') as f:
            user_commands = json.load(f)
        
        for keyword, command_string in user_commands.items():
            # Bind the keyword to a lambda that executes the corresponding shell command
            COMMAND_MAP[keyword] = lambda cmd=command_string: run_command(cmd)
            
        print(f"Loaded {len(COMMAND_MAP)} user-defined commands from {file_path}.")
        
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON format in {file_path}. Please check file syntax.")
    except Exception as e:
        print(f"[ERROR] Error loading commands: {e}")
