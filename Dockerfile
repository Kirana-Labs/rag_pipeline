# RunPod Serverless Dockerfile for RAG Pipeline with GPU support
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml uv.lock* ./

# Install uv for faster dependency management
RUN pip install uv

# Install project dependencies
RUN uv pip install --system -e .

# Install RunPod SDK
RUN pip install runpod

# Copy application code
COPY . .

# Create a requirements.txt file for RunPod compatibility
RUN uv export --format requirements-txt > requirements.txt

# Expose port (for local testing)
EXPOSE 8000

# Run the RunPod handler
CMD ["python", "-u", "rp_handler.py"]