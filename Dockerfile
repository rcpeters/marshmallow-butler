# Lean ARM64 Python 3.12 runtime image for Arduino VENTUNO Q (~500 MB vs 11+ GB)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Minimal runtime system packages for OpenCV, USB webcams, and speaker playback
RUN apt-get update && apt-get install -y --no-install-recommends \
    v4l-utils \
    alsa-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "-m", "src.main", "--mode", "cli"]
