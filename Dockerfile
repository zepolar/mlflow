# Lightweight Python base image
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (build tools for scientific libs) and cleanup
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code and relevant project files
COPY src/ ./src/
COPY MLproject conda.yaml README.md ./

# Default command runs the training script; parameters can be overridden at runtime
ENTRYPOINT ["python", "src/train.py"]
CMD ["--C", "1.0", "--max_iter", "200", "--test_size", "0.2", "--random_state", "42"]
