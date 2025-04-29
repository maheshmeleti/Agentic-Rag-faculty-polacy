# Use the official Ollama image as the base
FROM docker.io/ollama/ollama:latest

# Set environment variables
ENV OLLAMA_KEEP_ALIVE=24h
ENV OLLAMA_HOST=0.0.0.0

# Expose the port for Ollama
EXPOSE 11434

# Start the Ollama service, pull models, and keep serving
ENTRYPOINT ["/bin/sh", "-c", "\
    ollama serve & \
    sleep 5 && \
    ollama pull nomic-embed-text && \
    ollama pull llama3.2:3b && \
    wait"]
