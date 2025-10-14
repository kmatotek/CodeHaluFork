FROM python:3.10-slim

# Disable tokenizer parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Set working directory inside the container
WORKDIR /app

# Copy everything from the build context (the project root)
COPY . /app

# Add src to PYTHONPATH so imports like "from src.utils import X" work
ENV PYTHONPATH=/app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
