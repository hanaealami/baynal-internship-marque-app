# CPU-only image with brand detection (YOLO) + damage classification.
# Heavier than the deployed single-model version — intended for local use.
FROM python:3.11-slim

# System libs OpenCV/Ultralytics need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir Flask==3.0.3 gunicorn==22.0.0 \
        ultralytics==8.4.127 Werkzeug==3.0.3 transformers==4.44.2

# App code + model.
COPY . .

# Create DB schema at build time.
RUN python -c "from app import init_db; init_db()"

# Pre-download the damage-detection model weights so the running
# container never needs network access (also lets us load it offline
# per-request instead of keeping it resident in memory).
RUN python -c "\
from transformers import AutoImageProcessor, AutoModelForImageClassification; \
n = 'beingamit99/car_damage_detection'; \
AutoImageProcessor.from_pretrained(n); \
AutoModelForImageClassification.from_pretrained(n)"

EXPOSE 5000

# 1 worker keeps memory low; the YOLO model loads once per worker,
# the damage model loads/frees per request.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", \
     "--timeout", "120", "app:app"]
