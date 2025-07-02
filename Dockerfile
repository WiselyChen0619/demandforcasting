# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-cloudrun.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-cloudrun.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY data/*.csv ./data/

# Create directories for runtime
RUN mkdir -p /app/output /app/logs

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run the API server with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "300", "src.app:app"]