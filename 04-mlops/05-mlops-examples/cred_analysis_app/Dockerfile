# Dockerfile — Application Container Image Definition
# =====================================================
# Role in the System Architecture:
#   This file defines HOW the FastAPI application (api.py) is packaged
#   into a Docker image called "app-service".
#
#   Steps performed at build time:
#   1. Starts from a minimal Python 3.9 base image.
#   2. Installs OS-level dependencies required by Evidently AI (tkinter/tk).
#   3. Installs all Python dependencies listed in requirements.txt.
#   4. Copies the full application source code into the image.
#   5. Creates the /app/data directory that will hold production_logs.csv
#      and drift_report.html (mounted as a volume at runtime).
#
#   The resulting image is used by docker-compose (app-service) and is
#   also a suitable base for running monitoring.py inside the container.
#
#   Where it fits:
#       Dockerfile ──► docker build ──► app-service image
#                                            └──► docker-compose up ──► running API

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for Evidently (tkinter)
RUN apt-get update && apt-get install -y \
    tk \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure data directory exists
RUN mkdir -p data

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
