FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src ./src
COPY utils ./utils

# Make required directories
RUN mkdir -p /app/data/raw/clemson_faculty_docs && \
    mkdir -p /app/data/processed/clemson_faculty_docs && \
    logs

# The command to run your backend (adjust as needed)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]