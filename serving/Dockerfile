# Use lightweight Python 3.10 base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODEL_DIR=/app/serving_model

# Set working directory
WORKDIR /app

# Copy serving requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py /app/

# Expose default port
EXPOSE 8000

# Launch FastAPI app with Uvicorn dynamically reading PORT for Railway
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
