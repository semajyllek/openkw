
import time
import torch
import sounddevice as sd
import numpy as np
from .commands import COMMAND_MAP
from config.params import SAMPLE_RATE, AUDIO_LENGTH_SAMPLES, MAX_RUNTIME_SECONDS, DETECTION_THRESHOLD

# Mock/Placeholder functions (Assumed to be defined in deployment/spotter.py or imported)
def spot_keyword(audio_segment, inference_model, threshold) -> tuple:
    # This function would take the audio, process it with the KWS model,
    # and return (detected, score, keyword)
    # For a real implementation, it would use the ENROLLED_KEYWORDS reference vectors
    # to check against the audio embedding.
    
    # Placeholder: Return a mock detection after 10 seconds for testing
    if time.time() % 10 < 0.1:
        return True, 0.9, "test_keyword"
    return False, 0.0, ""

def dispatch_command(keyword: str):
    """Dispatches the command by looking up the keyword in the loaded map."""
    action_function = COMMAND_MAP.get(keyword)
    
    if action_function:
        action_function()
    else:
        print(f"[ERROR] Keyword '{keyword}' detected, but no shell script binding found in COMMAND_MAP.")


def run_command_dispatcher(inference_model, threshold=DETECTION_THRESHOLD):
    """The main continuous audio processing loop."""
    
    # Check if commands are loaded, otherwise, there's nothing to do
    if not COMMAND_MAP:
        print("\n[STOPPED] Cannot start dispatcher. COMMAND_MAP is empty. Please check commands.json.")
        return

    BLOCKSIZE = AUDIO_LENGTH_SAMPLES # 1 second audio chunk
    
    print("\n--- VOICE COMMAND DISPATCHER STARTED ---")
    print(f"Sample Rate: {SAMPLE_RATE} Hz | Listening for: {list(COMMAND_MAP.keys())}")
    
    try:
        # Start a non-blocking audio stream
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=BLOCKSIZE) as stream:
            start_time = time.time()
            
            while (time.time() - start_time) < MAX_RUNTIME_SECONDS:
                # 1. Capture 1-second audio chunk
                audio_data, overflowed = stream.read(BLOCKSIZE)

                if overflowed:
                    print("Warning: Audio buffer overflowed!", end='\r')

                # 2. Convert to PyTorch tensor for model processing
                live_audio_segment = torch.from_numpy(audio_data.flatten()).float()
                
                # 3. Spotting Check
                # Iterate over all ENROLLED_KEYWORDS and check similarity (logic assumed inside spot_keyword)
                detected, score, keyword = spot_keyword(live_audio_segment, inference_model, threshold)
                
                if detected and keyword in COMMAND_MAP:
                    print(f"\n[DETECTED] Keyword: '{keyword}' (Score: {score:.4f})")
                    dispatch_command(keyword)
                    time.sleep(1.0) # Pause briefly after command to prevent re-triggering
    
    except KeyboardInterrupt:
        print("\nDispatcher manually stopped.")
    except Exception as e:
        print(f"\nAn audio stream error occurred: {e}")
