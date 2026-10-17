FROM python:3.14.8-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.lock /app/
COPY separation_service/requirements.txt separation_service/requirements.lock /app/separation_service/
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu --require-hashes -r separation_service/requirements.lock
COPY backend /app/backend
COPY scripts /app/scripts
COPY separation_service /app/separation_service
RUN python -m scripts.download_separation_model --output /app/separation-model
RUN python -c "import numpy as np; from separation_service.model import Separator; m=Separator('/app/separation-model'); assert m.separate(np.zeros(8000, dtype=np.float32)).shape == (8000, 2)"
RUN useradd --create-home --uid 10001 inference
ENV SEPARATION_MODEL_DIR=/app/separation-model HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
USER inference
CMD ["sh", "-c", "exec python -m uvicorn separation_service.app:app --host 0.0.0.0 --port ${PORT:-8101} --workers 1 --no-proxy-headers --no-access-log"]
