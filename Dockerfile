# ─────────────────────────────────────────────
# FROM: Start with official Python 3.11 image
# "slim" = minimal Ubuntu + Python only
# No extra tools = smaller image = faster deploy
# ─────────────────────────────────────────────
FROM python:3.11-slim

# ─────────────────────────────────────────────
# RUN: Install system packages we need
# curl = needed for health check
# && chains commands (one RUN layer = smaller image)
# rm -rf cleans apt cache after install
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# WORKDIR: All commands run from /app inside container
# If /app doesn't exist Docker creates it
# ─────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────
# COPY requirements BEFORE code
# WHY: Docker builds in layers and caches each one
# If requirements.txt hasn't changed → skip pip install
# If you copy code first → any code change = reinstall packages
# This saves 2-3 minutes on every rebuild
# ─────────────────────────────────────────────
COPY requirements.txt .

# ─────────────────────────────────────────────
# RUN: Install Python packages
# --no-cache-dir = don't cache pip downloads = smaller image
# setuptools==69.5.1 = fixes pkg_resources issue we saw earlier
# ─────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools==69.5.1 && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# COPY: Copy your application code
# . means "current directory on your machine"
# /app means "paste into /app inside container"
# ─────────────────────────────────────────────
COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py .

# ─────────────────────────────────────────────
# COPY: Copy model files
# mlruns = MLflow registry (has your Staging model)
# artifacts = threshold.txt + backup joblib
# ─────────────────────────────────────────────
# Create empty directories — populated at runtime via docker volumes
RUN mkdir -p mlruns/ artifacts/ logs/
# ─────────────────────────────────────────────
# COPY: Environment config
# We copy .env.example as .env
# In real production: inject secrets via environment variables
# Never copy real .env with secrets into Docker image
# ─────────────────────────────────────────────
COPY .env.example .env

# ─────────────────────────────────────────────
# RUN: Create logs directory
# Container needs this folder to write logs
# ─────────────────────────────────────────────
RUN mkdir -p logs

# ─────────────────────────────────────────────
# EXPOSE: Document that app listens on 8000
# This is documentation only
# Actual port publishing happens at docker run -p
# ─────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────
# HEALTHCHECK: Docker pings /health every 30s
# --interval=30s  check every 30 seconds
# --timeout=10s   wait max 10s for response
# --start-period=15s  give app 15s to start up first
# --retries=3     fail 3 times before marking unhealthy
# ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ─────────────────────────────────────────────
# CMD: Command that runs when container starts
# This is what starts your FastAPI server
# Only ONE CMD allowed per Dockerfile
# ─────────────────────────────────────────────
CMD ["python", "main.py"]
