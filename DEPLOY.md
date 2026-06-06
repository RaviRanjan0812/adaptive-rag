# Deployment guide

## Local (full stack)

```bash
# 1. Copy and fill env vars
cp .env.example .env
#    GEMINI_API_KEY=AIza...
#    SEC_USER_AGENT=Your Name your@email.com
#    LANGFUSE_PUBLIC_KEY=pk-lf-...   (from http://localhost:3000 after first run)
#    LANGFUSE_SECRET_KEY=sk-lf-...

# 2. Build indexes (one-time, ~10 min)
python -m ingest.fetch_filings
python -m ingest.build_index
python -m ingest.build_graph

# 3. Start everything
docker compose up --build

# API  : http://localhost:8000/docs
# Langfuse traces: http://localhost:3000
```

## Render (API only)

1. Push to GitHub.
2. New Web Service → "From existing repo" → select this repo.
3. Render detects `render.yaml` automatically.
4. Add secrets in the Render dashboard (GEMINI_API_KEY, SEC_USER_AGENT, etc.).
5. Mount a persistent disk at `/app/data` (2 GB) and upload your pre-built indexes:
   ```bash
   # Upload indexes via Render shell or scp
   rsync -av data/index/ <render-ssh-host>:/app/data/index/
   rsync -av data/raw/   <render-ssh-host>:/app/data/raw/
   ```
6. Deploy. Health check: `GET /healthz`.

## Railway (API only)

1. `railway login && railway init`
2. `railway up` — Railway reads `railway.json`.
3. Set env vars: `railway variables set GEMINI_API_KEY=...`
4. Mount a volume at `/app/data` in the Railway dashboard and upload indexes.

## Hugging Face Spaces (Streamlit demo)

1. Create a new Space: **Streamlit**, Python 3.11.
2. Upload `demo_app/streamlit_app.py` as `app.py`.
3. Upload `demo_app/requirements.txt` as `requirements.txt`.
4. Add a Space secret: `API_BASE_URL` = your Render/Railway API URL.
5. The demo calls your hosted API — it never runs heavy inference locally.

## Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | yes | — | Google AI Studio / Vertex AI key |
| `GEMINI_MODEL` | no | `gemini-2.0-flash` | Override to `gemini-1.5-pro` for quality |
| `SEC_USER_AGENT` | yes (ingest only) | — | `"Name email"` per SEC policy |
| `LANGFUSE_PUBLIC_KEY` | no | — | Enables Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | no | — | Enables Langfuse tracing |
| `LANGFUSE_HOST` | no | `http://localhost:3000` | Self-hosted or cloud |
| `DEMO_MODE` | no | `false` | Cache answers; protect public cost |
| `RATE_LIMIT_RPM` | no | `30` | Per-IP request cap |
| `FAITH_GATE` | no | `0.60` | CI faithfulness regression threshold |

## Protecting public cost

Set `DEMO_MODE=true` before any public URL goes live.  This caches every
answer the first time it's computed so repeated identical questions never
charge the LLM again.  Combined with `RATE_LIMIT_RPM=20` this makes the
public demo essentially free to run after the first warm-up pass.

Run a warm-up pass to pre-populate the cache:

```bash
python scripts/warmup_cache.py --api http://your-api-url --queries eval/queries.jsonl
```
