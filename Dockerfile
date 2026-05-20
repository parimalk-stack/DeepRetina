# Standalone Dockerfile for OCT Analysis Webapp

FROM python:3.11-slim

# =============================================================================
# Environment Variables
# =============================================================================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASK_ENV=production \
    FLASK_DEBUG=0 \
    PYTHONPATH=/app

# =============================================================================
# System Dependencies Installation
# =============================================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    wget \
    curl \
    git \
    nano \
    htop \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff5-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# Python Dependencies Installation
# =============================================================================
WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# =============================================================================
# Application Setup
# =============================================================================

RUN mkdir -p /app/static/uploads \
             /app/static/css \
             /app/static/js \
             /app/static/image \
             /app/templates \
             /app/utils \
             /app/models \
             /app/logs

COPY app.py test_setup.py ./
COPY utils/ ./utils/
COPY templates/ ./templates/
COPY static/ ./static/
COPY models/ ./models/

# =============================================================================
# Security and User Setup
# =============================================================================

RUN groupadd -r octuser && \
    useradd -r -g octuser -d /app -s /bin/bash octuser && \
    chown -R octuser:octuser /app && \
    chmod -R 755 /app && \
    chmod 777 /app/static/uploads

# =============================================================================
# Container Configuration
# =============================================================================

USER octuser

WORKDIR /app

EXPOSE 5000

# =============================================================================
# Labels and Metadata
# =============================================================================

LABEL maintainer="OCT Analysis Team" \
      version="1.0" \
      description="Complete OCT Image Analysis Webapp with CNN/ViT/Swin models" \
      org.opencontainers.image.title="OCT Analysis Webapp" \
      org.opencontainers.image.description="Medical imaging analysis with deep learning" \
      org.opencontainers.image.version="1.0"

# =============================================================================
# Startup Command (Runs the Flask App Directly)
# =============================================================================

CMD ["python", "app.py"]
