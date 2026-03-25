# 1. Base Image (PyTorch 2.5.1 + CUDA 12.4 for RTX 5090)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set PYTHONPATH to ensure Python can find the mounted models and layers
ENV PYTHONPATH="/app"

# 3. Set the working directory
WORKDIR /app

# 4. Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python packages (using requirements.txt)
# Copy requirements.txt into the container first
COPY requirements.txt .
# Then install dependencies to leverage Docker cache mechanism
RUN pip install --no-cache-dir -r requirements.txt

# We will mount the directory via -v during docker run
# 6. Default to entering the Bash shell instead of running the script directly
CMD ["/bin/bash"]