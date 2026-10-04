# Lightweight Python base image (same Python minor version as training)
FROM python:3.11-slim

# Hugging Face Spaces run containers as user 1000 -> create it so the HF cache / streamlit config are writable
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR $HOME/app

# Install dependencies first to leverage Docker layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user . .

# HF Spaces routes traffic to port 7860
EXPOSE 7860

# Launch the Streamlit app, listening on all interfaces
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
