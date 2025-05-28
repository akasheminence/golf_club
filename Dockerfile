# Start from NVIDIA’s CUDA 12.1 base image with Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch (CUDA 12.1) and TorchVision
#   Ensure that the versions match your local environment and code requirements.
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install additional Python dependencies
RUN pip install \
    opencv-python \
    supervision \
    tqdm \
    mediapipe \
    firebase-admin \
    runpod \
    timm \
    transformers \
    addict \
    yapf \
    pycocotools \
    scipy 


RUN python3 -c "\
from transformers import AutoModel, AutoTokenizer;\
AutoModel.from_pretrained('bert-base-uncased');\
AutoTokenizer.from_pretrained('bert-base-uncased')\
"
# Copy the current directory's content into /app
COPY . .
RUN bash checkpoints/download_ckpts.sh
RUN bash gdino_checkpoints/download_ckpts.sh

# Install GroundingDINO as an editable package
RUN pip install --no-build-isolation -e grounding_dino

# Install Segment Anything 2 (SAM2) as an editable package
RUN pip install -e ".[notebooks]"

# Expose any port you might want to use in your container if you plan to serve via HTTP
EXPOSE 80

ENV FIREBASE_CRED=""\
    FIREBASE_BUCKET=""

# By default, run the RunPod handler. Adjust if you prefer to run a different entrypoint.
CMD ["python3","-u", "main.py"]
