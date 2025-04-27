FROM python:3.10-slim-buster

WORKDIR /app

# Install dependencies
COPY requirements.txt .

# Copy application code
COPY src ./src
COPY utils ./utils
COPY config.yaml .

# Make required directories
RUN mkdir -p /app/data/raw/clemson_faculty_docs && \
    mkdir -p /app/data/processed/clemson_faculty_docs && \
    mkdir logs

RUN pip install --no-cache-dir -r requirements.txt


# The command to run your backend (adjust as needed)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]