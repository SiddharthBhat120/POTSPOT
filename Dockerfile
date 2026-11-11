FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml/ ml/
COPY backend/ backend/
COPY models/ models/
COPY demo/ demo/
COPY --from=web /web/dist frontend/dist
ENV POTSPOT_DATA=/data YOLO_CONFIG_DIR=/tmp/ultralytics
EXPOSE 8002
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8002"]
