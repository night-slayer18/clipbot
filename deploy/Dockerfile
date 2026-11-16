FROM python:3.12-slim

# Install system dependencies (FFmpeg with libass, git, libGL for OpenCV, and Unicode fonts for captions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    fonts-noto-core \
    fonts-indic \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

COPY config/ config/
COPY src/ src/

ENV PYTHONUNBUFFERED=1

CMD ["clipbot", "daemon", "--profile", "gta_chaos"]
