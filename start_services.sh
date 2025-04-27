#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready
while ! curl -s http://localhost:11434 >/dev/null; do
  echo "Waiting for Ollama to start..."
  sleep 1
done

# Start FastAPI backend
uvicorn src.app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend
streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0