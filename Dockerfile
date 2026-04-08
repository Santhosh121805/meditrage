FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate patient notes on image build
RUN python -c "from data_generator import generate; generate()" 2>/dev/null || true

# Default command: run inference using MODEL_NAME from .env
CMD ["python", "inference.py", "--task", "all", "--output", "results.json"]
