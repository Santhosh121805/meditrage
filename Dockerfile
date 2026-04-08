FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate patient notes on image build
RUN python -c "from data_generator import generate; generate()" 2>/dev/null || true

# Set environment variable to indicate running on HF Space
ENV RUNNING_ON_SPACES=true
ENV PYTHONUNBUFFERED=1

# Run Flask server on port 7860 (HF Spaces standard)
EXPOSE 7860
CMD ["python", "-u", "inference.py", "--server"]
