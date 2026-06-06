FROM python:3.11-slim

# System deps for faiss-cpu + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding + rerank models so container starts offline
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
EOF

COPY . .

EXPOSE 8000

# Healthcheck so compose / Render know when the app is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--timeout-keep-alive", "30"]
