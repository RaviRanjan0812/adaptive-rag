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

# Make startup script executable
RUN chmod +x scripts/startup.sh

EXPOSE 8000

# Healthcheck — longer start period to allow for first-boot index build (~15 min)
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/healthz')"

# Run startup script which auto-builds index if missing, then starts uvicorn
CMD ["scripts/startup.sh"]
