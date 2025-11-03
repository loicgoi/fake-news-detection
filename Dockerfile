# ------ NE TOUCHE À RIEN D’AUTRE ------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Variables uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Copier les deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copier le code
COPY . .

# Installer le projet
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Créer dossiers persistants
RUN mkdir -p /app/chroma_db /app/data

# VARIABLE ESSENTIELLE
ENV OLLAMA_API_BASE=http://ollama:11434
ENV PYTHONPATH=/app
EXPOSE 8501

CMD ["/bin/bash", "-c", "source /app/.venv/bin/activate && streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0"]