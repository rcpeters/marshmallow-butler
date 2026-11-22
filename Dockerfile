# ARM64 (Ubuntu 24.04 / Python 3.12) runtime image for Arduino VENTUNO Q
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# System packages required for LeRobot/PyAV, OpenCV, v4l2, and building optional ARM64 wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    ffmpeg \
    build-essential \
    cmake \
    git \
    curl \
    v4l-utils \
    alsa-utils \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Optional: install placo if an aarch64 wheel exists (falls back cleanly to built-in NumPy IK)
RUN python3 -m pip install --no-cache-dir placo || true

COPY . .

CMD ["python3", "-m", "src.main", "--mode", "cli"]
