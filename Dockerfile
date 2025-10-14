FROM python:3.10-slim

# Disable tokenizer parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Copy your code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command (can be overridden in docker-compose)