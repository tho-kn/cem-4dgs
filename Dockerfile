# Use CUDA 12.8 to support Blackwell (sm_100/sm_120) and newer GPUs
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    build-essential \
    cmake \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install PyTorch Nightly with CUDA 12.8 support (cu128)
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install PyTorch3D build-time dependencies
RUN pip install fvcore iopath

# Install PyTorch3D from source with Blackwell support
# Using 12.0+PTX as suggested by working example
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0+PTX"
ENV FORCE_CUDA="1"
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

# Install FAISS
RUN pip install faiss-gpu-cu12

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
# Filter out conflicting packages from requirements if needed, but for now install all
# Removing torch/numpy pins if any, letting pip build resolve or using installed versions
RUN grep -v "submodules/" requirements.txt | \
    grep -v "pytorch3d" | \
    grep -v "torch" > requirements_pypi.txt && \
    pip install -r requirements_pypi.txt "numpy<2.0"

# Copy application
COPY . /app

# IMPORTANT: Set CUDA architecture list
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0+PTX"
ENV FORCE_CUDA="1"

# Install submodules with Blackwell support
RUN pip install submodules/diff_gaussian_rasterization_df --no-build-isolation
RUN pip install submodules/simple-knn --no-build-isolation

CMD ["/bin/bash"]