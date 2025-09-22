
# use a slim Linux base image with Python
FROM python:3.10-slim

# 1. install System Dependencies:
# - build-essential: For compiling Python packages.
# - sox, libsox-fmt-all: For the 'rec' command.
# - libportaudio2: For the 'sounddevice' Python package.
# - libpulse-dev: CRITICAL: Provides the client libraries to connect to the host's PulseAudio server.
# - alsa-utils: Optional, but useful for debugging audio within the container (e.g., 'aplay').
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        sox \
        libsox-fmt-all \
        libportaudio2 \
        libpulse-dev \
        alsa-utils && \
    rm -rf /var/lib/apt/lists/*

# 2. set the working directory inside the container
WORKDIR /app

# 3. copy Python dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. copy the entire application code
COPY . .

# 5. set environment variable for command execution path
ENV PATH="/app/commands:${PATH}"

# 6. set the default entry point to the CLI
ENTRYPOINT ["python", "deployment/spotter.py"]
