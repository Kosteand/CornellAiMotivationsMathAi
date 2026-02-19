FROM nvidia/cuda:12.1.0-base-ubuntu22.04


# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and GUI-related system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    # For Gymnasium rendering
    libgl1-mesa-dev \
    libosmesa6-dev \
    freeglut3-dev \
    # The virtual display engine
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
# Ensure pyvirtualdisplay and gymnasium[all] are in your requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

CMD ["bash"]