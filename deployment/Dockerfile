FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Rust for CDP SDK
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port (Cloud Run sets PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get(f'http://localhost:{os.environ.get(\"PORT\", \"8080\")}/health')"

# Run the application
CMD ["python", "run_fixed.py"]