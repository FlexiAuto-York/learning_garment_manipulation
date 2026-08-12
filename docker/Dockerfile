# Prefer `docker compose` from this directory -- see docker/README.md.
#   cd docker && docker compose build && docker compose run --rm robot
#
# ============================================================
# Base image: CUDA 12.1 + Ubuntu 22.04
# ============================================================

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

LABEL maintainer="halidkadi.robot@gmail.com"
LABEL description="Garment Folding dual-arm manipulation environment"

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 1. Install system dependencies and Python 3.10
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl python3 python3-venv python3-pip build-essential cmake \
    libusb-1.0-0-dev libgl1-mesa-glx udev \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb1 \
    libsm6 libxext6 libxrender1 libfontconfig1 libfreetype6 && \
    rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

# ------------------------------------------------------------
# 2. Agent/arena framework
# ------------------------------------------------------------
WORKDIR /workspace
RUN git clone --branch develop https://github.com/halid1020/actoris_harena.git

# ------------------------------------------------------------
# 3. Virtual environment + framework install
# ------------------------------------------------------------
RUN pip install --upgrade pip && pip install uv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace/actoris_harena
RUN uv pip install -e ".[torch]"

RUN pip install pycurl
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# ------------------------------------------------------------
# 4. SAM weights for real-world perception
# ------------------------------------------------------------
RUN mkdir -p /workspace/learning_garment_manipulation/real_robot/models && \
    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O /workspace/learning_garment_manipulation/real_robot/models/sam_vit_h_4b8939.pth

# This repository and SoftGym are bind-mounted at run time (see docker-compose.yml),
# so the image stays valid while you edit code on the host.
WORKDIR /workspace/learning_garment_manipulation

# ------------------------------------------------------------
# 5. Environment variables
# ------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ------------------------------------------------------------
# 6. Default command
# ------------------------------------------------------------
CMD ["/bin/bash"]
