FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app ./app
COPY models ./models

# Health + non-root
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=models/pet_classifier.pkl

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
