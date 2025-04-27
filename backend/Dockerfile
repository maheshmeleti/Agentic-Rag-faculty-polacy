FROM python:3.10-slim-buster

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY utils ./utils
COPY config.yaml .

# Make required directories
# RUN mkdir -p data/raw/clemson_faculty_docs
# RUN mkdir -p data/processed/clemson_faculty_docs
RUN mkdir logs

# The command to run your backend (adjust as needed)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]