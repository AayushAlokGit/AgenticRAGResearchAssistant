# Backend image for Hugging Face Spaces (Docker SDK).
#
# The corpus vectors live in Neon (pgvector), so nothing data-heavy is baked here EXCEPT the local
# cross-encoder reranker, which must be present so it can load fully offline at runtime.
FROM python:3.12-slim

# HF Spaces runs the container as uid 1000. Create that user so the app dir + HF model cache are
# writable at runtime (the model is cached under $HOME during build).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Install deps + the package EDITABLE (in place). config.project_root() walks up from the source file
# to find pyproject.toml, so the package must stay next to config/ and prompts/ — a normal
# site-packages install would break that lookup. Copy only what the runtime needs: NOT corpus/ or the
# local Chroma store (the store is pgvector on Neon), NOT the frontend.
COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user -e .
COPY --chown=user config ./config
COPY --chown=user prompts ./prompts

# Bake the cross-encoder reranker into the image cache (network is available here at build), then
# force offline for the running container so a blip on huggingface.co can never crash a request
# (rag/rerank.py loads with local_files_only). STORE_PROVIDER=pgvector is the deployed store.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    STORE_PROVIDER=pgvector \
    PYTHONUNBUFFERED=1


# HF Spaces routes traffic to the port declared as app_port in the Space README (7860 by default).
EXPOSE 7860
CMD ["uvicorn", "agentic_rag.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
