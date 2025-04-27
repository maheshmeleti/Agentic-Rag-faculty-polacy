FROM python:3.10-slim-buster

WORKDIR /app

# Copy application code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# The command to run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
